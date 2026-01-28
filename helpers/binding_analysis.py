#!/usr/bin/env python3
"""
Binding mode analysis utilities for protein-ligand interactions.
Detects hydrogen bonds, salt bridges, hydrophobic contacts, and π-interactions.
"""

import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import tempfile
import glob


def _ensure_obabel() -> Optional[str]:
    """Return path to obabel executable if available."""
    return shutil.which("obabel") or shutil.which("obabel.exe")


def _run_obabel(args: List[str]) -> None:
    obabel_exec = _ensure_obabel()
    if not obabel_exec:
        raise RuntimeError("Open Babel (obabel) executable not found in PATH")

    cmd = [obabel_exec] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "obabel conversion failed")


def _convert_pdbqt_to_pdb(pdbqt_path: Path, pdb_path: Path) -> None:
    """Convert PDBQT file to PDB format."""
    obabel_exec = _ensure_obabel()
    
    if obabel_exec:
        # Use Open Babel if available
        try:
            temp_pdb = pdb_path.with_suffix(".temp.pdb")
            _run_obabel(["-ipdbqt", str(pdbqt_path), "-opdb", "-O", str(temp_pdb)])
            
            # Clean up the PDB file
            content = temp_pdb.read_text()
            lines = content.split('\n')
            clean_lines = []
            
            for line in lines:
                if line.startswith(('MODEL', 'ENDMDL')):
                    continue
                elif line.startswith(('ATOM', 'HETATM', 'CONECT', 'COMPND', 'REMARK', 'ROOT', 'ENDROOT', 'BRANCH', 'ENDBRANCH', 'TORSDOF')):
                    clean_lines.append(line)
                elif line.strip():
                    clean_lines.append(line)
            
            if clean_lines and not any(line.startswith('END') for line in clean_lines):
                clean_lines.append('END')
            
            pdb_path.write_text('\n'.join(clean_lines))
            temp_pdb.unlink()
            return
            
        except Exception:
            pass
    
    # Fallback: Simple PDBQT to PDB conversion without Open Babel
    try:
        content = pdbqt_path.read_text()
        lines = content.split('\n')
        pdb_lines = []
        
        for line in lines:
            # Skip MODEL and ENDMDL lines
            if line.startswith(('MODEL', 'ENDMDL')):
                continue
            # Convert ATOM records (remove charge and atom type columns)
            elif line.startswith(('ATOM', 'HETATM')):
                # PDBQT format has extra columns at the end, truncate to PDB format
                if len(line) >= 78:
                    pdb_line = line[:78]  # Standard PDB ATOM record length
                    pdb_lines.append(pdb_line)
                else:
                    pdb_lines.append(line)
            # Keep other relevant records
            elif line.startswith(('CONECT', 'REMARK', 'COMPND')):
                pdb_lines.append(line)
            # Skip PDBQT-specific records
            elif line.startswith(('ROOT', 'ENDROOT', 'BRANCH', 'ENDBRANCH', 'TORSDOF')):
                continue
            elif line.strip():
                pdb_lines.append(line)
        
        # Add END record
        if pdb_lines and not any(line.startswith('END') for line in pdb_lines):
            pdb_lines.append('END')
        
        pdb_path.write_text('\n'.join(pdb_lines))
        
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDBQT to PDB: {e}")


def _split_vina_poses(results_pdbqt: Path, output_dir: Path) -> List[Path]:
    """Split multi-pose PDBQT file into individual pose files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try vina_split first
    vina_split = shutil.which("vina_split")
    if vina_split:
        try:
            cmd = [vina_split, "--input", str(results_pdbqt)]
            result = subprocess.run(cmd, cwd=output_dir, capture_output=True, text=True)
            if result.returncode == 0:
                # Find generated pose files
                pose_files = sorted(output_dir.glob(f"{results_pdbqt.stem}_*.pdbqt"))
                return pose_files
        except Exception:
            pass
    
    # Fallback: manual splitting
    pose_files = []
    if not results_pdbqt.exists():
        return pose_files
        
    content = results_pdbqt.read_text()
    models = content.split("MODEL")
    
    for i, model in enumerate(models[1:], 1):  # Skip first empty split
        if "ENDMDL" in model:
            pose_content = "MODEL" + model
            pose_file = output_dir / f"{results_pdbqt.stem}_pose_{i}.pdbqt"
            pose_file.write_text(pose_content)
            pose_files.append(pose_file)
    
    return pose_files


def _parse_vina_scores(log_file: Path) -> List[float]:
    """Parse Vina scores from log file."""
    scores = []
    if not log_file.exists():
        return scores
        
    try:
        content = log_file.read_text()
        lines = content.split('\n')
        
        # Look for the results table
        in_results = False
        for line in lines:
            line = line.strip()
            if "-----+------------" in line:
                in_results = True
                continue
            elif in_results and line:
                if line.startswith("Writing") or line.startswith("Refining"):
                    break
                try:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        score = float(parts[1])
                        scores.append(score)
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass
        
    return scores


def _convert_pdb_to_sdf(pdb_path: Path, sdf_path: Path) -> None:
    """Convert PDB file to SDF using Open Babel."""
    _run_obabel(["-ipdb", str(pdb_path), "-osdf", "-O", str(sdf_path)])


def _combine_complex(receptor_pdb: Path, ligand_pdb: Path, complex_path: Path) -> None:
    """Create a combined receptor-ligand complex PDB with proper atom numbering to prevent OpenMM duplicate warnings."""
    receptor_lines = receptor_pdb.read_text().splitlines()
    ligand_lines = ligand_pdb.read_text().splitlines()

    def _filter_and_renumber_atoms(lines: List[str], start_atom_num: int = 1) -> Tuple[List[str], int]:
        """Filter PDB lines and renumber atoms sequentially to prevent duplicates."""
        filtered_lines = []
        current_atom_num = start_atom_num
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith(("END", "ENDMDL", "MODEL")):
                continue
            elif line.startswith(("ATOM", "HETATM")):
                # Renumber the atom serial number (columns 7-11) - CRITICAL for OpenMM
                if len(line) >= 11:
                    # Force sequential numbering to eliminate ALL duplicate atom serial numbers
                    new_line = line[:6] + f"{current_atom_num:5d}" + line[11:]
                    filtered_lines.append(new_line)
                    current_atom_num += 1
                else:
                    filtered_lines.append(line)
            elif line.startswith(("CONECT", "REMARK", "HEADER", "TITLE", "COMPND", "SOURCE")):
                # Keep other records as-is but skip CONECT to avoid issues
                if not line.startswith("CONECT"):
                    filtered_lines.append(line)
        
        return filtered_lines, current_atom_num

    # Process receptor first - atoms 1, 2, 3, ...
    receptor_filtered, next_atom_num = _filter_and_renumber_atoms(receptor_lines, 1)
    
    # Process ligand with continuing atom numbers - atoms next_atom_num, next_atom_num+1, ...
    ligand_filtered, final_atom_num = _filter_and_renumber_atoms(ligand_lines, next_atom_num)
    
    # Combine with proper separators and ensure no duplicate atom numbers
    combined = receptor_filtered + ["TER"] + ligand_filtered + ["END"]
    
    # Write with explicit newlines
    complex_path.write_text("\n".join(combined) + "\n")
    
    # Verify no duplicate atom numbers in output
    content = complex_path.read_text()
    atom_numbers = set()
    for line in content.split('\n'):
        if line.startswith(('ATOM', 'HETATM')) and len(line) >= 11:
            try:
                atom_num = int(line[6:11].strip())
                if atom_num in atom_numbers:
                    raise ValueError(f"Duplicate atom number {atom_num} found after combining!")
                atom_numbers.add(atom_num)
            except ValueError as e:
                if "Duplicate atom" in str(e):
                    raise e
                # Skip lines with invalid atom numbers
                pass


def detect_interactions_plip(
    complex_pdb: Path,
    output_dir: Path,
    plip_exec: str = "plip"
) -> Dict[str, Any]:
    """Use PLIP to detect protein-ligand interactions."""
    try:
        import subprocess
        result = subprocess.run([plip_exec, "--version"], capture_output=True)
        if result.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "method": "plip",
            "status": "failed",
            "error": "PLIP not available"
        }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run PLIP analysis
    cmd = [
        plip_exec,
        "-f", str(complex_pdb),
        "-o", str(output_dir),
        "--xml"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            return {
                "method": "plip",
                "status": "failed",
                "error": result.stderr
            }
        
        # Parse PLIP XML output
        xml_files = list(output_dir.glob("*.xml"))
        if not xml_files:
            return {
                "method": "plip",
                "status": "failed",
                "error": "No XML output generated"
            }
        
        return {
            "method": "plip",
            "status": "success",
            "xml_file": str(xml_files[0]),
            "output_dir": str(output_dir)
        }
        
    except subprocess.TimeoutExpired:
        return {
            "method": "plip",
            "status": "failed",
            "error": "PLIP analysis timed out"
        }
    except Exception as e:
        return {
            "method": "plip",
            "status": "failed",
            "error": str(e)
        }


def detect_interactions_biopython(
    receptor_pdb: Path,
    ligand_coords: List[Tuple[float, float, float]],
    ligand_atoms: List[str],
    output_dir: Path
) -> Dict[str, Any]:
    """Use BioPython to detect basic interactions."""
    try:
        from Bio.PDB import PDBParser, NeighborSearch
        import numpy as np
    except ImportError:
        return {
            "method": "biopython",
            "status": "failed", 
            "error": "BioPython not available"
        }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Parse receptor structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("receptor", receptor_pdb)
        
        # Get all atoms from receptor
        receptor_atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        receptor_atoms.append(atom)
        
        # Create neighbor search
        ns = NeighborSearch(receptor_atoms)
        
        interactions = {
            "hydrogen_bonds": [],
            "hydrophobic_contacts": [],
            "salt_bridges": [],
            "pi_interactions": []
        }
        
        # Simple distance-based interaction detection
        for i, (x, y, z) in enumerate(ligand_coords):
            ligand_atom = ligand_atoms[i] if i < len(ligand_atoms) else "C"
            
            # Find nearby receptor atoms (within 5Å)
            nearby_atoms = ns.search(np.array([x, y, z]), 5.0)
            
            for atom in nearby_atoms:
                distance = np.linalg.norm(
                    np.array([x, y, z]) - atom.get_coord()
                )
                
                # Simple heuristics for interaction types
                if distance <= 3.5:
                    if _is_hbond_candidate(ligand_atom, atom.get_name()):
                        interactions["hydrogen_bonds"].append({
                            "ligand_atom": i,
                            "receptor_atom": atom.get_full_id(),
                            "distance": float(distance),
                            "residue": f"{atom.get_parent().get_resname()}{atom.get_parent().get_id()[1]}"
                        })
                    elif _is_hydrophobic_candidate(ligand_atom, atom.get_name()):
                        interactions["hydrophobic_contacts"].append({
                            "ligand_atom": i,
                            "receptor_atom": atom.get_full_id(),
                            "distance": float(distance),
                            "residue": f"{atom.get_parent().get_resname()}{atom.get_parent().get_id()[1]}"
                        })
        
        # Save interactions
        interaction_file = output_dir / "interactions.json"
        with open(interaction_file, 'w') as f:
            json.dump(interactions, f, indent=2)
        
        return {
            "method": "biopython",
            "status": "success",
            "interactions": interactions,
            "interaction_file": str(interaction_file)
        }
        
    except Exception as e:
        return {
            "method": "biopython",
            "status": "failed",
            "error": str(e)
        }


def _is_hbond_candidate(ligand_atom: str, receptor_atom: str) -> bool:
    """Simple heuristic for hydrogen bond candidates."""
    hbond_atoms = {"N", "O", "S"}
    return (ligand_atom in hbond_atoms or receptor_atom in hbond_atoms)


def _is_hydrophobic_candidate(ligand_atom: str, receptor_atom: str) -> bool:
    """Simple heuristic for hydrophobic contact candidates."""
    hydrophobic_atoms = {"C"}
    return (ligand_atom in hydrophobic_atoms and receptor_atom in hydrophobic_atoms)


def generate_pymol_script(
    receptor_pdbqt: Path,
    pose_files: List[Path],
    scores: List[float],
    output_dir: Path,
    pocket_residues: List[str] = None,
    ligand_name: str = "ligand",
    pdb_id: str = "unknown",
    pocket_id: str = "unknown",
    gene_name: str = "unknown"
) -> Dict[str, Any]:
    """Generate PyMOL visualization script for binding mode analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    script_file = output_dir / "binding_mode_visualization.pml"
    # Create organized image filename
    image_filename = f"{gene_name}_{pdb_id}_{pocket_id}_{ligand_name}_binding_mode.png"
    png_file = output_dir / image_filename
    
    # Convert receptor to PDB for better PyMOL compatibility
    receptor_pdb = output_dir / "receptor.pdb"
    try:
        _convert_pdbqt_to_pdb(receptor_pdbqt, receptor_pdb)
    except Exception:
        receptor_pdb = receptor_pdbqt  # Fallback to original
    
    best_score = scores[0] if scores else "N/A"
    pymol_script = f"""# PyMOL script for binding mode visualization
# Generated automatically for {ligand_name} in {pdb_id} {pocket_id}
# Best pose score: {best_score}

# Load receptor
load {receptor_pdb.resolve()}, receptor
hide everything, receptor
show cartoon, receptor
color gray70, receptor

"""
    
    # Load all poses (focus on best pose for main visualization)
    pose_objects = []
    for i, pose_file in enumerate(pose_files[:3], 1):  # Limit to top 3 poses
        pose_name = f"pose{i}"
        pose_objects.append(pose_name)
        score_text = f" (score: {scores[i-1]:.1f})" if i-1 < len(scores) else ""
        pymol_script += f"# Load pose {i}{score_text}\n"
        pymol_script += f"load {pose_file.resolve()}, {pose_name}\n"
    
    if pose_objects:
        # Style all poses
        poses_selection = " ".join(pose_objects)
        pymol_script += f"""
# Style ligand poses
show sticks, {poses_selection}
set stick_radius, 0.15

# Color poses differently
"""
        # Color best pose prominently, others more subtly
        colors = ["yellow", "gray60", "gray40"]
        for i, (pose_name, color) in enumerate(zip(pose_objects, colors)):
            pymol_script += f"color {color}, {pose_name}\n"
            if i > 0:  # Align other poses to best pose
                pymol_script += f"align {pose_name}, {pose_objects[0]}\n"
                pymol_script += f"set transparency, 0.5, {pose_name}\n"
    
    # Show binding pocket
    if pose_objects:
        pymol_script += f"""
# Show binding pocket (within 4Å of best pose)
select pocket, (receptor within 4 of {pose_objects[0]})
show sticks, pocket
color cyan, pocket

"""
    
    # Add specific binding site residues if provided
    if pocket_residues and pose_objects:
        pymol_script += "# Highlight specific binding site residues\n"
        selection_terms = []
        for res in pocket_residues:
            if "_" in res:
                chain, num = res.split("_", 1)
                selection_terms.append(f"(chain {chain} and resi {num})")
            else:
                selection_terms.append(f"resi {res}")
        
        if selection_terms:
            residue_selection = " or ".join(selection_terms)
            pymol_script += f"select binding_site, {residue_selection}\n"
            pymol_script += "show sticks, binding_site\n"
            pymol_script += "color yellow, binding_site\n"
    
    # Add interaction analysis
    if pose_objects:
        pymol_script += f"""
# Detect hydrogen bonds
distance hbonds, {pose_objects[0]}, receptor, 3.2
color red, hbonds
hide labels, hbonds

# Detect contacts
distance contacts, {pose_objects[0]}, receptor, 4.0
color yellow, contacts
hide labels, contacts
set dash_gap, 0.3, contacts

"""
    
    # Final view and output
    if pose_objects:
        pymol_script += f"""
# Set view
zoom {pose_objects[0]}, 8
set cartoon_transparency, 0.3
set ray_shadows, 0
set antialias, 2

# Generate high-quality image
set ray_shadows, 1
set ray_trace_mode, 1
set antialias, 2
ray 2048, 1536
png {png_file.resolve()}

# Save session
save {(output_dir / 'binding_mode_session.pse').resolve()}

# Print summary
print "Binding mode analysis complete for {ligand_name}"
print "PDB: {pdb_id}, Pocket: {pocket_id}"
print "Best pose score: {best_score}"
print "Poses loaded: {len(pose_objects)}"
print "Image saved: {png_file.resolve()}"
"""
    else:
        pymol_script += "print \"No poses found for visualization\"\n"
    
    with open(script_file, 'w') as f:
        f.write(pymol_script)
    
    return {
        "method": "pymol_script",
        "status": "success",
        "script_file": str(script_file),
        "png_file": str(png_file),
        "poses_loaded": len(pose_objects),
        "instructions": f"Run with: pymol -c {script_file}"
    }


def analyze_binding_modes(
    top_hits: List[Dict[str, Any]],
    job_index: Dict[str, Dict[str, Any]],
    docking_results: List[Dict[str, Any]],
    output_dir: Path,
    pocket_residues_map: Dict[Tuple[str, str], List[str]] = None
) -> Dict[str, Any]:
    """Analyze binding modes for top hits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create result index
    result_index = {}
    for result in docking_results:
        job_id = result.get("job_id")
        if job_id and result.get("status") == "success":
            result_index[job_id] = result

    # Secondary lookup by (pdb_id, pocket_id, ligand_name)
    job_lookup = {}
    for job_id, job in job_index.items():
        key = (
            job.get("pdb_id"),
            job.get("pocket_id"),
            job.get("ligand_name"),
        )
        if all(key):
            job_lookup[key] = job_id

    binding_analyses = []

    for hit in top_hits:
        ligand_name = hit.get("name")
        if not ligand_name:
            continue
        
        pdb_id = hit.get("pdb_id")
        pocket_id = hit.get("pocket_id")

        # Find corresponding job
        job_id = job_lookup.get((pdb_id, pocket_id, ligand_name))

        if job_id is None:
            for jid, job in job_index.items():
                if job.get("ligand_name") == ligand_name:
                    job_id = jid
                    break

        if not job_id or job_id not in result_index:
            continue
        
        job = job_index[job_id]
        result = result_index[job_id]

        hit_dir = output_dir / f"{pdb_id}_{pocket_id}_{ligand_name}"
        hit_dir.mkdir(parents=True, exist_ok=True)

        analysis_result = {
            "ligand_name": ligand_name,
            "job_id": job_id,
            "pdb_id": pdb_id,
            "pocket_id": pocket_id,
            "source": hit.get("source"),
            "score": hit.get("score"),
            "analyses": {}
        }

        # Get pocket residues for this hit
        pocket_residues = None
        if pocket_residues_map and pdb_id and pocket_id:
            pocket_residues = pocket_residues_map.get((pdb_id, pocket_id)) or []
        
        # Get docking result files
        receptor_pdbqt = Path(job.get("receptor_pdbqt", ""))
        result = result_index.get(job_id, {})
        
        # Find result.pdbqt from docking output
        job_dir = Path(job.get("job_dir", ""))
        results_pdbqt = job_dir / "result.pdbqt"
        log_file = job_dir / "vina.log"
        
        if not results_pdbqt.exists():
            analysis_result["analyses"]["error"] = f"Results file not found: {results_pdbqt}"
            binding_analyses.append(analysis_result)
            continue
        
        # Split poses from results.pdbqt
        poses_dir = hit_dir / "poses"
        try:
            pose_files = _split_vina_poses(results_pdbqt, poses_dir)
            scores = _parse_vina_scores(log_file)
            
            analysis_result["pose_analysis"] = {
                "total_poses": len(pose_files),
                "scores": scores[:len(pose_files)],
                "pose_files": [str(f) for f in pose_files]
            }
        except Exception as exc:
            analysis_result["analyses"]["pose_extraction"] = {
                "status": "failed",
                "error": str(exc)
            }
            binding_analyses.append(analysis_result)
            continue

        # PLIP analysis commented out - focus on PyMOL-based analysis
        # receptor_pdb = converted_files.get("receptor_pdb")
        # ligand_pdb = converted_files.get("ligand_pdb")
        # if receptor_pdb and ligand_pdb:
        #     complex_pdb = hit_dir / "complex.pdb"
        #     try:
        #         _combine_complex(receptor_pdb, ligand_pdb, complex_pdb)
        #         plip_result = detect_interactions_plip(complex_pdb, hit_dir / "plip")
        #     except Exception as exc:
        #         plip_result = {
        #             "method": "plip",
        #             "status": "failed",
        #             "error": f"complex generation failed: {exc}"
        #         }
        # else:
        #     plip_result = {
        #         "method": "plip",
        #         "status": "skipped",
        #         "reason": "converted files missing",
        #         "conversion_errors": converted_files.get("errors")
        #     }
        
        plip_result = {
            "method": "plip",
            "status": "disabled",
            "reason": "PLIP analysis disabled, using PyMOL-based analysis"
        }

        analysis_result["analyses"]["plip"] = plip_result

        # Generate PyMOL script with poses
        if receptor_pdbqt.exists() and pose_files:
            try:
                pymol_result = generate_pymol_script(
                    receptor_pdbqt, pose_files, scores, hit_dir, pocket_residues, 
                    ligand_name, pdb_id, pocket_id, "ESR2"  # TODO: get gene_name from state
                )
                
                # Try to run PyMOL script automatically
                pymol_exec = shutil.which("pymol")
                if pymol_exec:
                    try:
                        script_file = Path(pymol_result["script_file"])
                        # Use absolute path and ensure working directory is correct
                        cmd = [pymol_exec, "-c", str(script_file.resolve())]
                        result = subprocess.run(
                            cmd, 
                            cwd=str(script_file.parent.resolve()), 
                            capture_output=True, 
                            text=True, 
                            timeout=300
                        )
                        
                        if result.returncode == 0:
                            pymol_result["execution"] = "success"
                            pymol_result["output"] = result.stdout
                            
                            # Copy image to organized binding_images directory
                            png_file = Path(pymol_result["png_file"])
                            if png_file.exists():
                                # Create organized directory structure
                                binding_images_dir = Path("binding_images") / "ESR2" / pdb_id / pocket_id
                                binding_images_dir.mkdir(parents=True, exist_ok=True)
                                
                                organized_png = binding_images_dir / png_file.name
                                shutil.copy2(png_file, organized_png)
                                pymol_result["organized_image"] = str(organized_png)
                        else:
                            pymol_result["execution"] = "failed"
                            pymol_result["error"] = result.stderr
                    except subprocess.TimeoutExpired:
                        pymol_result["execution"] = "timeout"
                    except Exception as exc:
                        pymol_result["execution"] = "error"
                        pymol_result["error"] = str(exc)
                else:
                    pymol_result["execution"] = "pymol_not_found"
                    
            except Exception as exc:
                pymol_result = {
                    "method": "pymol_script",
                    "status": "failed",
                    "error": str(exc)
                }
        else:
            pymol_result = {
                "method": "pymol_script",
                "status": "skipped",
                "reason": "receptor or pose files not found"
            }

        analysis_result["analyses"]["pymol"] = pymol_result

        binding_analyses.append(analysis_result)
    
    # Categorize by source
    by_source = {}
    for analysis in binding_analyses:
        source = analysis.get("source", "unknown")
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(analysis)
    
    # Save binding mode analysis summary
    summary = {
        "total_hits_analyzed": len(binding_analyses),
        "by_source": {k: len(v) for k, v in by_source.items()},
        "analyses": binding_analyses,
        "source_breakdown": by_source
    }
    
    summary_file = output_dir / "binding_mode_analysis.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return {
        "binding_mode_summary": summary,
        "summary_file": str(summary_file),
        "output_dir": str(output_dir)
    }


__all__ = [
    "detect_interactions_plip",
    "detect_interactions_biopython", 
    "generate_pymol_script",
    "analyze_binding_modes"
]
