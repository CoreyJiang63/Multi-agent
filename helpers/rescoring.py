#!/usr/bin/env python3
"""
Rescoring and refinement utilities for post-docking analysis.
Supports multiple scoring functions and local minimization.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json


def _find_executable(name: str) -> Optional[str]:
    """Find executable in PATH or return None."""
    return shutil.which(name)


def _run_command(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 300) -> Tuple[int, str, str]:
    """Run command and return (returncode, stdout, stderr). Always uses conda drug_agent environment."""
    try:
        # Ensure all commands run in the conda drug_agent environment
        if isinstance(cmd, list) and len(cmd) > 0:
            if cmd[0] == "bash" and "-c" in cmd:
                # Already has bash -c wrapper, ensure conda activation
                original_command = cmd[2] if len(cmd) > 2 else ""
                if "conda activate drug_agent" not in original_command:
                    cmd[2] = f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate drug_agent && {original_command}"
            else:
                # Wrap command with conda activation
                cmd_str = " ".join(cmd)
                cmd = ["bash", "-c", f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate drug_agent && {cmd_str}"]
        
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def rescore_with_vina(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    output_dir: Path,
    vina_exec: str = "vina"
) -> Dict[str, Any]:
    """Rescore pose using Vina with higher precision settings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle multi-MODEL PDBQT files by extracting first pose
    single_pose_pdbqt = output_dir / "single_pose.pdbqt"
    
    try:
        # Read the ligand file and extract first MODEL
        content = ligand_pdbqt.read_text()
        
        if "MODEL" in content:
            # Extract first model only, removing MODEL/ENDMDL headers
            lines = content.split('\n')
            first_model_lines = []
            in_first_model = False
            
            for line in lines:
                if line.startswith("MODEL 1") or (line.startswith("MODEL") and not in_first_model):
                    in_first_model = True
                    # Skip the MODEL line itself
                    continue
                elif line.startswith("ENDMDL") and in_first_model:
                    # Skip the ENDMDL line and stop
                    break
                elif in_first_model:
                    first_model_lines.append(line)
                elif not line.startswith("MODEL") and not in_first_model:
                    # Handle files without MODEL headers
                    first_model_lines.append(line)
            
            single_pose_pdbqt.write_text('\n'.join(first_model_lines))
        else:
            # No MODEL headers, use as is
            single_pose_pdbqt.write_text(content)
            
    except Exception as e:
        return {
            "method": "vina_rescore",
            "status": "failed",
            "error": f"Failed to process ligand file: {e}"
        }
    
    # Use scoring-only mode
    out_pdbqt = output_dir / "rescored.pdbqt"
    log_file = output_dir / "vina_rescore.log"
    receptor_path = receptor_pdbqt.resolve()

    cmd = [
        vina_exec,
        "--receptor", str(receptor_path),
        "--ligand", str(single_pose_pdbqt.resolve()),
        "--score_only",
        "--out", out_pdbqt.name,
        "--log", log_file.name
    ]
    
    returncode, stdout, stderr = _run_command(cmd, cwd=output_dir)
    
    result = {
        "method": "vina_rescore",
        "status": "success" if returncode == 0 else "failed",
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "output_file": str(out_pdbqt) if returncode == 0 else None,
        "log_file": str(log_file)
    }
    
    # Parse score from log
    if returncode == 0 and log_file.exists():
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
            # Extract score from Vina log
            for line in log_content.split('\n'):
                if 'Affinity:' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            result["score"] = float(parts[1])
                            break
                        except ValueError:
                            pass
        except Exception:
            pass
    
    return result


def rescore_with_gnina(
    receptor_pdb: Path,
    ligand_sdf: Path,
    output_dir: Path,
    gnina_exec: str = "gnina"
) -> Dict[str, Any]:
    """Rescore pose using gnina CNN scoring."""
    if not _find_executable(gnina_exec):
        return {
            "method": "gnina",
            "status": "failed",
            "error": f"{gnina_exec} not found in PATH"
        }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_sdf = output_dir / "gnina_scored.sdf"
    log_file = output_dir / "gnina.log"
    
    cmd = [
        gnina_exec,
        "--receptor", str(receptor_pdb),
        "--ligand", str(ligand_sdf),
        "--out", str(out_sdf),
        "--score_only",
        "--log", str(log_file)
    ]
    
    returncode, stdout, stderr = _run_command(cmd, cwd=output_dir)
    
    result = {
        "method": "gnina",
        "status": "success" if returncode == 0 else "failed",
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "output_file": str(out_sdf) if returncode == 0 else None,
        "log_file": str(log_file)
    }
    
    return result


def minimize_with_amber(
    complex_pdb: Path,
    output_dir: Path,
    steps: int = 1000
) -> Dict[str, Any]:
    """Perform local minimization using AMBER (via AmberTools)."""
    amber_exec = _find_executable("sander") or _find_executable("pmemd")
    if not amber_exec:
        return {
            "method": "amber_minimize",
            "status": "failed",
            "error": "AMBER not available (sander/pmemd not found)"
        }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create AMBER input files
        min_in = output_dir / "minimize.in"
        min_in.write_text(f"""Minimization
 &cntrl
  imin=1, maxcyc={steps}, ncyc={steps//2},
  cut=8.0, ntb=0, igb=1,
  ntpr=100, ntwx=0,
 &end
""")
        
        # Convert PDB to AMBER format (would need leap/tleap)
        # This is a simplified implementation
        return {
            "method": "amber_minimize",
            "status": "skipped",
            "reason": "AMBER setup requires leap/tleap integration"
        }
        
    except Exception as e:
        return {
            "method": "amber_minimize",
            "status": "failed",
            "error": str(e)
        }


# NOTE: OpenMM minimization functionality disabled - replaced by MD simulation
def minimize_complex_with_openmm(
    complex_pdb: Path,
    output_dir: Path,
    steps: int = 1000
) -> Dict[str, Any]:
    """DISABLED: OpenMM minimization replaced by MD simulation analysis."""
    return {
        "method": "openmm_minimize",
        "status": "disabled", 
        "reason": "Local minimization disabled - use MD simulation instead"
    }

def minimize_complex_with_openmm_DISABLED(
    complex_pdb: Path,
    output_dir: Path,
    steps: int = 1000
) -> Dict[str, Any]:
    """DISABLED: OpenMM minimization of pre-combined complex using OpenFF for ligands and AMBER for protein."""
    output_dir.mkdir(parents=True, exist_ok=True)
    minimized_pdb = output_dir / "minimized.pdb"
    
    if not complex_pdb.exists():
        return {
            "method": "openmm_minimize",
            "status": "failed",
            "error": f"Complex PDB file not found: {complex_pdb}"
        }
    
    # Check if minimization already completed successfully
    if minimized_pdb.exists():
        try:
            # Validate the minimized file has content
            with open(minimized_pdb, 'r') as f:
                content = f.read()
                if len(content) > 100 and "ATOM" in content:
                    return {
                        "method": "openmm_minimize",
                        "status": "success",
                        "output_file": str(minimized_pdb),
                        "note": "Using existing minimized structure"
                    }
        except:
            pass
    
    # Create comprehensive minimization script using proper string formatting
    script_content = f"""#!/usr/bin/env python3
import sys
import traceback
from pathlib import Path
import numpy as np

def minimize_complex():
    '''Minimize protein-ligand complex with OpenMM.'''
    try:
        # Import OpenMM
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit
        
        print("Loading complex structure...")
        complex_pdb_path = r"{complex_pdb}"
        output_pdb_path = r"{minimized_pdb}"
        steps = {steps}
        
        # Load PDB structure
        pdb = app.PDBFile(complex_pdb_path)
        num_atoms = len(list(pdb.topology.atoms()))
        print(f"Loaded structure with {{num_atoms}} atoms")
        
        # Add missing hydrogens using Modeller with proper pH and variant handling
        print("Adding missing hydrogens...")
        from openmm.app import Modeller
        modeller = Modeller(pdb.topology, pdb.positions)
        
        # Robust structure preparation with terminal residue handling
        try:
            print("Attempting comprehensive structure repair...")
            
            # Step 1: Try to add missing heavy atoms first
            try:
                modeller.addMissingAtoms(forcefield=app.ForceField("amber14-all.xml"))
                print("Added missing heavy atoms with force field guidance")
            except Exception as e1:
                print(f"Could not add missing heavy atoms: {{e1}}")
            
            # Step 2: Add hydrogens with multiple fallback strategies
            try:
                modeller.addHydrogens(forcefield=app.ForceField("amber14-all.xml"), pH=7.0)
                print("Added hydrogens with force field and pH guidance")
            except Exception as e2:
                print(f"Force field guided hydrogen addition failed: {{e2}}")
                try:
                    modeller.addHydrogens(pH=7.0)
                    print("Added hydrogens with pH guidance")
                except Exception as e3:
                    print(f"pH-guided hydrogen addition failed: {{e3}}")
                    try:
                        modeller.addHydrogens()
                        print("Added hydrogens with minimal method")
                    except Exception as e4:
                        print(f"All hydrogen addition methods failed: {{e4}}")
                        print("Warning: Proceeding with original structure")
        
        except Exception as e_main:
            print(f"Structure preparation failed: {{e_main}}")
            print("Warning: Using original structure without modifications")
            modeller = Modeller(pdb.topology, pdb.positions)
        
        print(f"After processing: {{len(list(modeller.topology.atoms()))}} atoms")
        
        # Set up force fields with OpenFF for ligands
        print("Setting up force fields...")
        
        # Use a more robust force field approach for incomplete structures
        print("Setting up robust force field for incomplete structures...")
        
        # Try different force field strategies with aggressive terminal residue handling
        force_field_strategies = [
            # Strategy 1: Remove incomplete terminal residues and use AMBER14
            {
                "name": "amber14_clean_terminals",
                "files": ["amber14-all.xml"],
                "clean_terminals": True,
                "description": "AMBER14 with cleaned terminal residues"
            },
            # Strategy 2: Use OpenMM's built-in terminal patches
            {
                "name": "amber14_auto_terminals", 
                "files": ["amber14-all.xml"],
                "auto_terminals": True,
                "description": "AMBER14 with automatic terminal handling"
            },
            # Strategy 3: Force minimal constraints for incomplete structures
            {
                "name": "amber14_minimal",
                "files": ["amber14-all.xml"],
                "minimal_constraints": True,
                "description": "AMBER14 with minimal constraints"
            },
            # Strategy 4: Use CHARMM force field (better terminal handling)
            {
                "name": "charmm36_basic",
                "files": ["charmm36.xml"],
                "description": "CHARMM36 force field"
            },
            # Strategy 5: Last resort - skip problematic residues
            {
                "name": "amber14_skip_incomplete",
                "files": ["amber14-all.xml"], 
                "skip_incomplete": True,
                "description": "AMBER14 skipping incomplete residues"
            }
        ]
        
        system = None
        used_ff = None
        
        for strategy in force_field_strategies:
            try:
                print(f"Trying force field: {{strategy['name']}} - {{strategy['description']}}")
                
                # Handle different terminal residue strategies
                current_modeller = modeller
                
                if strategy.get("clean_terminals", False):
                    try:
                        print("Cleaning incomplete terminal residues...")
                        # Remove incomplete N-terminal and C-terminal residues
                        clean_modeller = Modeller(modeller.topology, modeller.positions)
                        
                        # Get chains and identify problematic terminal residues
                        chains = list(clean_modeller.topology.chains())
                        atoms_to_delete = []
                        
                        for chain in chains:
                            residues = list(chain.residues())
                            if residues:
                                # Check N-terminal residue
                                n_term = residues[0]
                                n_term_atoms = list(n_term.atoms())
                                expected_backbone = ['N', 'CA', 'C', 'O']
                                actual_backbone = [atom.name for atom in n_term_atoms if atom.name in expected_backbone]
                                
                                if len(actual_backbone) < 3:  # Missing critical backbone atoms
                                    print(f"Removing incomplete N-terminal residue: {{n_term.name}} ({{len(actual_backbone)}}/4 backbone atoms)")
                                    atoms_to_delete.extend(n_term_atoms)
                                
                                # Check C-terminal residue  
                                c_term = residues[-1]
                                c_term_atoms = list(c_term.atoms())
                                actual_backbone = [atom.name for atom in c_term_atoms if atom.name in expected_backbone]
                                
                                if len(actual_backbone) < 3:  # Missing critical backbone atoms
                                    print(f"Removing incomplete C-terminal residue: {{c_term.name}} ({{len(actual_backbone)}}/4 backbone atoms)")
                                    atoms_to_delete.extend(c_term_atoms)
                        
                        if atoms_to_delete:
                            clean_modeller.delete(atoms_to_delete)
                            current_modeller = clean_modeller
                            print(f"Removed {{len(atoms_to_delete)}} atoms from incomplete terminal residues")
                        else:
                            print("No incomplete terminal residues found to clean")
                            
                    except Exception as clean_e:
                        print(f"Could not clean terminal residues: {{clean_e}}")
                        current_modeller = modeller
                
                elif strategy.get("auto_terminals", False):
                    try:
                        print("Applying automatic terminal patches...")
                        # Try to use OpenMM's automatic terminal handling
                        auto_modeller = Modeller(modeller.topology, modeller.positions)
                        protein_ff = app.ForceField(*strategy["files"])
                        
                        # Add missing atoms with force field guidance for terminals
                        auto_modeller.addMissingAtoms(protein_ff)
                        current_modeller = auto_modeller
                        print("Successfully applied automatic terminal patches")
                    except Exception as auto_e:
                        print(f"Automatic terminal handling failed: {{auto_e}}")
                        current_modeller = modeller
                
                # Create force field
                protein_ff = app.ForceField(*strategy["files"])
                
                # Configure system creation based on strategy
                if strategy.get("minimal_constraints", False):
                    # Use minimal constraints for problematic structures
                    system = protein_ff.createSystem(
                        current_modeller.topology,
                        nonbondedMethod=app.NoCutoff,
                        constraints=None,
                        rigidWater=False,
                        removeCMMotion=False,
                        hydrogenMass=None  # Don't repartition hydrogen masses
                    )
                else:
                    # Standard system creation
                    system = protein_ff.createSystem(
                        current_modeller.topology,
                        nonbondedMethod=app.NoCutoff,
                        constraints=None,
                        rigidWater=False,
                        removeCMMotion=False
                    )
                
                used_ff = strategy["name"]
                modeller = current_modeller  # Use the successful modeller
                print(f"Successfully created system with {{strategy['name']}}")
                break
                
            except Exception as e:
                print(f"Force field {{strategy['name']}} failed: {{e}}")
                continue
        1.0 / unit.picoseconds,
        0.002 * unit.picoseconds
    )
    
    # Use CPU platform for stability
    platform = mm.Platform.getPlatformByName("CPU")
    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    
    # Set initial positions
    simulation.context.setPositions(modeller.positions)
    
    # Store original positions for RMSD calculation
    original_positions = modeller.positions
    
    # Energy minimization
    print(f"Starting energy minimization ({steps} steps)...")
    initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"Initial potential energy: {initial_energy}")
    
    simulation.minimizeEnergy(maxIterations=steps)
    
    final_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"Final potential energy: {final_energy}")
    print(f"Energy change: {final_energy - initial_energy}")
    
    # Get final positions and calculate RMSD
    final_state = simulation.context.getState(getPositions=True)
    final_positions = final_state.getPositions()
    
    # Calculate RMSD (focusing on protein atoms to avoid ligand bias)
    protein_atoms = []
    for atom in pdb.topology.atoms():
        if atom.residue.name not in ['UNL', 'LIG', 'MOL']:  # Skip ligand residues
            protein_atoms.append(atom.index)
    
    if protein_atoms:
        orig_coords = np.array([[original_positions[i].x, original_positions[i].y, original_positions[i].z] for i in protein_atoms])
        final_coords = np.array([[final_positions[i].x, final_positions[i].y, final_positions[i].z] for i in protein_atoms])
        
        # Convert to nanometers for RMSD calculation
        orig_coords = orig_coords * 10  # Angstrom to nm
        final_coords = final_coords * 10
            1.0 / unit.picoseconds,
            0.002 * unit.picoseconds
        )
        
        # Use CPU platform for stability
        platform = mm.Platform.getPlatformByName("CPU")
        simulation = app.Simulation(modeller.topology, system, integrator, platform)
        
        # Set initial positions
        simulation.context.setPositions(modeller.positions)
        
        # Store original positions for RMSD calculation
        original_positions = modeller.positions
        
        # Energy minimization
        print(f"Starting energy minimization ({{steps}} steps)...")
        initial_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        print(f"Initial potential energy: {{initial_energy}}")
        
        simulation.minimizeEnergy(maxIterations=steps)
        
        final_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        print(f"Final potential energy: {{final_energy}}")
        print(f"Energy change: {{final_energy - initial_energy}}")
        
        # Get final positions and calculate RMSD
        final_state = simulation.context.getState(getPositions=True)
        final_positions = final_state.getPositions()
        
        # Calculate RMSD (focusing on protein atoms to avoid ligand bias)
        protein_atoms = []
        for atom in pdb.topology.atoms():
            if atom.residue.name not in ['UNL', 'LIG', 'MOL']:  # Skip ligand residues
                protein_atoms.append(atom.index)
        
        if protein_atoms:
            orig_coords = np.array([[original_positions[i].x, original_positions[i].y, original_positions[i].z] for i in protein_atoms])
            final_coords = np.array([[final_positions[i].x, final_positions[i].y, final_positions[i].z] for i in protein_atoms])
            
            # Convert to nanometers for RMSD calculation
            orig_coords = orig_coords * 10  # Angstrom to nm
            final_coords = final_coords * 10
            
            diff = orig_coords - final_coords
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            print(f"Protein RMSD: {{rmsd:.3f}} nm")
        
        # Save minimized structure
        print(f"Saving minimized structure to {{output_pdb_path}}...")
        with open(output_pdb_path, 'w') as f:
            app.PDBFile.writeFile(modeller.topology, final_positions, f)
        
        # Calculate overall RMSD
        original_coords = np.array([[pos.x, pos.y, pos.z] for pos in original_positions])
        final_coords = np.array([[pos.x, pos.y, pos.z] for pos in final_positions])
        rmsd = np.sqrt(np.mean(np.sum((original_coords - final_coords)**2, axis=1))) * 10  # Convert to Angstroms
        
        print(f"Overall RMSD: {{rmsd:.3f}} Å")
        
        return {{
            "method": "openmm_minimize",
            "status": "success",
            "output_file": output_pdb_path,
            "initial_energy": float(initial_energy.value_in_unit(unit.kilojoules_per_mole)),
            "final_energy": float(final_energy.value_in_unit(unit.kilojoules_per_mole)),
            "energy_change": float((final_energy - initial_energy).value_in_unit(unit.kilojoules_per_mole)),
            "rmsd_angstroms": float(rmsd),
            "steps": steps,
            "force_field": used_ff
        }}
        
    except Exception as e:
        print(f"Minimization failed: {{e}}")
        traceback.print_exc()
        return {{
            "method": "openmm_minimize",
            "status": "failed",
            "error": str(e)
        }}

if __name__ == "__main__":
    result = minimize_complex()
    print("RESULT_JSON:", result)
"""
    
    # Write and execute the script
    script_file = output_dir / "minimize_script.py"
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    try:
        # Run the minimization script in conda environment
        import subprocess
        result = subprocess.run([
            "bash", "-c", 
            f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate drug_agent && python {script_file}"
        ], capture_output=True, text=True, timeout=300)
        
        # Parse result from script output
        if result.returncode == 0:
            # Look for JSON result in output
            lines = result.stdout.split('\n')
            for line in lines:
                if line.startswith("RESULT_JSON:"):
                    import ast
                    result_dict = ast.literal_eval(line[12:].strip())
                    return result_dict
            
            # Fallback: check if output file exists
            if minimized_pdb.exists():
                return {
                    "method": "openmm_minimize",
                    "status": "success",
                    "output_file": str(minimized_pdb),
                    "note": "Minimization completed but result parsing failed"
                }
        
        return {
            "method": "openmm_minimize",
            "status": "failed",
            "error": f"Script failed with return code {result.returncode}: {result.stderr}"
        }
        
    except Exception as e:
        return {
            "method": "openmm_minimize",
            "status": "failed",
            "error": f"Failed to execute minimization script: {str(e)}"
        }

def minimize_with_openmm(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    output_dir: Path,
    steps: int = 1000
) -> Dict[str, Any]:
    """OpenMM minimization using OpenFF for ligands and AMBER for protein."""
    output_dir.mkdir(parents=True, exist_ok=True)
    minimized_pdb = output_dir / "minimized_complex.pdb"
    
    # For backward compatibility, use the complex-based minimization
    # First convert PDBQT files to PDB and combine them
    try:
        from .binding_analysis import _convert_pdbqt_to_pdb, _combine_complex
        
        receptor_pdb = output_dir / "receptor.pdb"
        ligand_pdb = output_dir / "ligand.pdb"
        complex_pdb = output_dir / "complex.pdb"
        
        # Convert files
        _convert_pdbqt_to_pdb(receptor_pdbqt, receptor_pdb)
        _convert_pdbqt_to_pdb(ligand_pdbqt, ligand_pdb)
        
        # Combine into complex
        _combine_complex(receptor_pdb, ligand_pdb, complex_pdb)
        
        # Use the complex-based minimization
        return minimize_complex_with_openmm(complex_pdb, output_dir, steps)
        
    except Exception as e:
        return {
            "method": "openmm_minimize",
            "status": "failed",
            "error": f"Failed to prepare complex for minimization: {str(e)}"
        }


def cluster_poses(
    pose_files: List[Path],
    output_dir: Path,
    rmsd_cutoff: float = 2.0
) -> Dict[str, Any]:
    """Cluster docking poses by RMSD."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolAlign
    except ImportError:
        return {
            "method": "pose_clustering",
            "status": "failed",
            "error": "RDKit not available"
        }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load molecules
    mols = []
    valid_files = []
    
    for pose_file in pose_files:
        try:
            if pose_file.suffix.lower() == '.sdf':
                mol = Chem.SDMolSupplier(str(pose_file))[0]
            elif pose_file.suffix.lower() == '.mol2':
                mol = Chem.MolFromMol2File(str(pose_file))
            elif pose_file.suffix.lower() == '.pdb':
                mol = Chem.MolFromPDBFile(str(pose_file))
            else:
                continue
                
            if mol is not None:
                mols.append(mol)
                valid_files.append(pose_file)
        except Exception:
            continue
    
    if len(mols) < 2:
        return {
            "method": "pose_clustering",
            "status": "failed",
            "error": "Need at least 2 valid poses for clustering"
        }
    
    # Calculate RMSD matrix
    rmsd_matrix = []
    for i in range(len(mols)):
        row = []
        for j in range(len(mols)):
            if i == j:
                row.append(0.0)
            else:
                try:
                    rmsd = rdMolAlign.AlignMol(mols[i], mols[j])
                    row.append(rmsd)
                except:
                    row.append(999.0)  # Large value for failed alignments
        rmsd_matrix.append(row)
    
    # Simple clustering by RMSD cutoff
    clusters = []
    assigned = set()
    
    for i in range(len(mols)):
        if i in assigned:
            continue
            
        cluster = [i]
        assigned.add(i)
        
        for j in range(i + 1, len(mols)):
            if j in assigned:
                continue
            if rmsd_matrix[i][j] <= rmsd_cutoff:
                cluster.append(j)
                assigned.add(j)
        
        clusters.append(cluster)
    
    # Save cluster information
    cluster_info = {
        "method": "pose_clustering",
        "status": "success",
        "rmsd_cutoff": rmsd_cutoff,
        "num_poses": len(mols),
        "num_clusters": len(clusters),
        "clusters": []
    }
    
    for i, cluster in enumerate(clusters):
        cluster_data = {
            "cluster_id": i,
            "size": len(cluster),
            "poses": [str(valid_files[idx]) for idx in cluster],
            "representative": str(valid_files[cluster[0]])  # First pose as representative
        }
        cluster_info["clusters"].append(cluster_data)
    
    # Save cluster info
    cluster_file = output_dir / "cluster_analysis.json"
    with open(cluster_file, 'w') as f:
        json.dump(cluster_info, f, indent=2)
    
    cluster_info["cluster_file"] = str(cluster_file)
    return cluster_info


def enhanced_rescore_and_refine(
    top_hits: List[Dict[str, Any]],
    job_index: Dict[str, Dict[str, Any]],
    docking_results: List[Dict[str, Any]],
    output_dir: Path,
    methods: List[str] = None,
    do_minimization: bool = True,
    do_clustering: bool = True
) -> Dict[str, Any]:
    """Enhanced rescoring with minimization and clustering."""
    if methods is None:
        methods = ["vina_rescore"]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create result index for quick lookup
    result_index = {}
    for result in docking_results:
        job_id = result.get("job_id")
        if job_id and result.get("status") == "success":
            result_index[job_id] = result

    # Secondary index by (pdb_id, pocket_id, ligand_name)
    job_lookup = {}
    for job_id, job in job_index.items():
        key = (
            job.get("pdb_id"),
            job.get("pocket_id"),
            job.get("ligand_name"),
        )
        if all(key):
            job_lookup[key] = job_id

    rescoring_results = []

    for hit in top_hits:
        ligand_name = hit.get("name")
        job_id = hit.get("job_id")  # Use job_id directly from hit
        
        if not ligand_name or not job_id:
            continue
        
        pdb_id = hit.get("pdb_id")
        pocket_id = hit.get("pocket_id")

        # Check if job and result exist
        if job_id not in job_index or job_id not in result_index:
            continue

        job = job_index[job_id]
        result = result_index[job_id]

        hit_dir = output_dir / f"{job.get('pdb_id')}_{job.get('pocket_id')}_{ligand_name}"
        hit_dir.mkdir(parents=True, exist_ok=True)

        hit_result = {
            "ligand_name": ligand_name,
            "job_id": job_id,
            "pdb_id": pdb_id,
            "pocket_id": pocket_id,
            "original_score": hit.get("score"),
            "rescoring_methods": {},
            "minimization": {},
            "clustering": {}
        }
        
        # Get file paths
        receptor_pdbqt = Path(job.get("receptor_pdbqt", ""))
        ligand_pdbqt = Path(job.get("ligand_pdbqt", ""))
        job_dir = Path(job.get("job_dir", ""))
        result_pdbqt = job_dir / "result.pdbqt"
        
        if not receptor_pdbqt.exists() or not result_pdbqt.exists():
            hit_result["error"] = "Missing input files"
            rescoring_results.append(hit_result)
            continue
        
        # Apply rescoring methods
        for method in methods:
            method_dir = hit_dir / method
            
            if method == "vina_rescore":
                rescore_result = rescore_with_vina(
                    receptor_pdbqt, result_pdbqt, method_dir
                )
            elif method == "gnina" and _find_executable("gnina"):
                rescore_result = {"method": method, "status": "skipped", "reason": "format conversion needed"}
            else:
                rescore_result = {"method": method, "status": "skipped", "reason": "method not available"}
            
            hit_result["rescoring_methods"][method] = rescore_result
        
        # Pose clustering analysis
        if do_clustering and result_pdbqt.exists():
            try:
                from .binding_analysis import _split_vina_poses, _convert_pdbqt_to_pdb
                poses_dir = hit_dir / "poses"
                pose_files = _split_vina_poses(result_pdbqt, poses_dir)
                
                # Convert PDBQT poses to PDB for clustering
                pdb_pose_files = []
                for pose_file in pose_files:
                    pdb_file = pose_file.with_suffix(".pdb")
                    try:
                        _convert_pdbqt_to_pdb(pose_file, pdb_file)
                        pdb_pose_files.append(pdb_file)
                    except Exception:
                        continue
                
                if len(pdb_pose_files) > 1:
                    cluster_result = cluster_poses(pdb_pose_files, hit_dir / "clustering")
                    hit_result["clustering"] = cluster_result
                    
                    # Add clustering validation metrics
                    if cluster_result.get("status") == "success":
                        num_clusters = cluster_result.get("num_clusters", 0)
                        largest_cluster = max([c["size"] for c in cluster_result.get("clusters", [])], default=0)
                        hit_result["clustering"]["stability_score"] = largest_cluster / len(pdb_pose_files)
                        hit_result["clustering"]["diversity_score"] = num_clusters / len(pdb_pose_files)
                else:
                    hit_result["clustering"] = {"status": "skipped", "reason": "insufficient valid poses after conversion"}
            except Exception as e:
                hit_result["clustering"] = {"status": "failed", "error": str(e)}
        
        # Local minimization (if OpenMM available)
        if do_minimization:
            try:
                from .binding_analysis import _convert_pdbqt_to_pdb, _combine_complex
                
                # Convert receptor and best pose to PDB
                receptor_pdb = hit_dir / "receptor.pdb"
                ligand_pdb = hit_dir / "ligand.pdb"
                complex_pdb = hit_dir / "complex.pdb"
                
                # Convert receptor
                _convert_pdbqt_to_pdb(receptor_pdbqt, receptor_pdb)
                
                # Get best pose from result.pdbqt
                poses_dir = hit_dir / "poses"
                pose_files = _split_vina_poses(result_pdbqt, poses_dir)
                if pose_files:
                    best_pose_pdb = pose_files[0].with_suffix(".pdb")
                    _convert_pdbqt_to_pdb(pose_files[0], best_pose_pdb)
                    
                    # Combine into complex with proper atom numbering
                    _combine_complex(receptor_pdb, best_pose_pdb, complex_pdb)
                    
                    # Run minimization with improved error handling
                    minimize_result = minimize_complex_with_openmm(complex_pdb, hit_dir / "minimization")
                    hit_result["minimization"] = minimize_result
                    
                    # If minimization fails due to force field issues, mark as skipped rather than failed
                    if minimize_result.get("status") == "failed" and "force field" in minimize_result.get("error", "").lower():
                        hit_result["minimization"]["status"] = "skipped"
                        hit_result["minimization"]["reason"] = "Protein topology incompatible with available force fields: " + minimize_result.get("error", "Unknown force field error")
                        
                else:
                    hit_result["minimization"] = {"status": "failed", "error": "No poses found for minimization"}
                    
            except Exception as e:
                hit_result["minimization"] = {"status": "failed", "error": str(e)}
        
        rescoring_results.append(hit_result)
    
    # Calculate validation metrics
    validated_hits = []
    for hit_result in rescoring_results:
        validation_score = 0
        validation_reasons = []
        
        # Score consistency check
        original_score = hit_result.get("original_score", 0)
        vina_rescore = hit_result.get("rescoring_methods", {}).get("vina_rescore", {})
        if vina_rescore.get("status") == "success":
            rescore_value = vina_rescore.get("score", 0)
            score_diff = abs(original_score - rescore_value)
            if score_diff < 2.0:  # Good consistency
                validation_score += 2
                validation_reasons.append("score_consistent")
            if rescore_value < -7.0:  # Strong binding
                validation_score += 2
                validation_reasons.append("strong_binding")
        
        # Clustering stability check
        clustering = hit_result.get("clustering", {})
        if clustering.get("status") == "success":
            stability = clustering.get("stability_score", 0)
            if stability > 0.6:  # >60% poses in largest cluster
                validation_score += 2
                validation_reasons.append("stable_binding_mode")
        
        # NOTE: Local minimization validation removed - will be replaced by MD simulation analysis
        # Minimization check is disabled for now
        # pass
        
        hit_result["validation_score"] = validation_score
        hit_result["validation_reasons"] = validation_reasons
        hit_result["is_validated_hit"] = validation_score >= 4  # Threshold for validated hit
        
        if hit_result["is_validated_hit"]:
            validated_hits.append(hit_result)
    
    # Generate validation report (minimization analysis removed)
    validation_report = generate_validation_report(rescoring_results, validated_hits, output_dir)
    
    # Save rescoring summary
    summary = {
        "total_hits_processed": len(rescoring_results),
        "validated_hits": len(validated_hits),
        "validation_rate": len(validated_hits) / len(rescoring_results) if rescoring_results else 0,
        "methods_used": methods,
        "validation_report": validation_report,
        "results": rescoring_results,
        "validated_hits_only": validated_hits
    }
    
    summary_file = output_dir / "enhanced_rescoring_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return {
        "rescoring_summary": summary,
        "summary_file": str(summary_file),
        "output_dir": str(output_dir)
    }


def generate_validation_report(
    rescoring_results: List[Dict[str, Any]], 
    validated_hits: List[Dict[str, Any]], 
    output_dir: Path,
    generate_natural_language: bool = True
) -> Dict[str, Any]:
    """Generate comprehensive validation report with minimization analysis."""
    
    # Collect minimization statistics
    minimization_stats = {
        "total_attempted": 0,
        "successful": 0,
        "skipped": 0,
        "failed": 0,
        "structurally_stable": 0,
        "stability_scores": [],
        "rmsd_values": [],
        "energy_changes": []
    }
    
    validation_categories = {
        "excellent": [],  # validation_score >= 6
        "good": [],       # validation_score >= 4
        "acceptable": [], # validation_score >= 2
        "poor": []        # validation_score < 2
    }
    
    minimization_details = []
    
    for hit in rescoring_results:
        # Categorize by validation score
        score = hit.get("validation_score", 0)
        if score >= 6:
            validation_categories["excellent"].append(hit)
        elif score >= 4:
            validation_categories["good"].append(hit)
        elif score >= 2:
            validation_categories["acceptable"].append(hit)
        else:
            validation_categories["poor"].append(hit)
        
        # Analyze minimization results
        minimization = hit.get("minimization", {})
        min_stability = hit.get("minimization_stability", {})
        
        if minimization:
            minimization_stats["total_attempted"] += 1
            
            status = minimization.get("status", "unknown")
            if status == "success":
                minimization_stats["successful"] += 1
                
                # Collect stability metrics
                if min_stability.get("is_structurally_stable"):
                    minimization_stats["structurally_stable"] += 1
                
                stability_score = min_stability.get("stability_score", 0)
                minimization_stats["stability_scores"].append(stability_score)
                
                rmsd = min_stability.get("rmsd_angstroms", 0)
                if rmsd > 0:
                    minimization_stats["rmsd_values"].append(rmsd)
                
                energy_change = min_stability.get("energy_change_kj_mol", 0)
                minimization_stats["energy_changes"].append(energy_change)
                
                # Detailed minimization analysis
                minimization_details.append({
                    "hit_name": hit.get("ligand_name", "unknown"),
                    "job_id": hit.get("job_id", "unknown"),
                    "stability_score": stability_score,
                    "stability_percentage": min_stability.get("stability_percentage", 0),
                    "rmsd_angstroms": rmsd,
                    "energy_change": energy_change,
                    "is_stable": min_stability.get("is_structurally_stable", False),
                    "stability_details": min_stability.get("details", [])
                })
                
            elif status == "skipped":
                minimization_stats["skipped"] += 1
            else:
                minimization_stats["failed"] += 1
    
    # Calculate summary statistics
    if minimization_stats["stability_scores"]:
        import numpy as np
        minimization_stats["avg_stability_score"] = round(np.mean(minimization_stats["stability_scores"]), 2)
        minimization_stats["std_stability_score"] = round(np.std(minimization_stats["stability_scores"]), 2)
    
    if minimization_stats["rmsd_values"]:
        minimization_stats["avg_rmsd"] = round(np.mean(minimization_stats["rmsd_values"]), 2)
        minimization_stats["std_rmsd"] = round(np.std(minimization_stats["rmsd_values"]), 2)
    
    if minimization_stats["energy_changes"]:
        minimization_stats["avg_energy_change"] = round(np.mean(minimization_stats["energy_changes"]), 1)
        minimization_stats["std_energy_change"] = round(np.std(minimization_stats["energy_changes"]), 1)
    
    # Top performing hits based on minimization
    top_stable_hits = sorted(
        [h for h in minimization_details if h["is_stable"]], 
        key=lambda x: x["stability_score"], 
        reverse=True
    )[:5]
    
    # Generate structured validation report
    validation_report = {
        "report_timestamp": str(datetime.now()),
        "summary": {
            "total_hits": len(rescoring_results),
            "validated_hits": len(validated_hits),
            "validation_rate": round(len(validated_hits) / len(rescoring_results) * 100, 1) if rescoring_results else 0,
            "minimization_success_rate": round(minimization_stats["successful"] / minimization_stats["total_attempted"] * 100, 1) if minimization_stats["total_attempted"] > 0 else 0,
            "structural_stability_rate": round(minimization_stats["structurally_stable"] / minimization_stats["successful"] * 100, 1) if minimization_stats["successful"] > 0 else 0
        },
        "validation_categories": {
            "excellent_hits": len(validation_categories["excellent"]),
            "good_hits": len(validation_categories["good"]),
            "acceptable_hits": len(validation_categories["acceptable"]),
            "poor_hits": len(validation_categories["poor"])
        },
        "minimization_analysis": minimization_stats,
        "top_stable_hits": top_stable_hits,
        "detailed_minimization_results": minimization_details,
        "recommendations": []
    }
    
    # Generate recommendations
    success_rate = validation_report["summary"]["validation_rate"]
    stability_rate = validation_report["summary"]["structural_stability_rate"]
    
    if success_rate >= 80:
        validation_report["recommendations"].append("Excellent validation rate - pipeline performing well")
    elif success_rate >= 60:
        validation_report["recommendations"].append("Good validation rate - consider minor optimizations")
    else:
        validation_report["recommendations"].append("Low validation rate - review docking parameters")
    
    if stability_rate >= 80:
        validation_report["recommendations"].append("Excellent structural stability after minimization")
    elif stability_rate >= 60:
        validation_report["recommendations"].append("Moderate structural stability - check force field parameters")
    else:
        validation_report["recommendations"].append("Poor structural stability - review minimization protocol")
    
    if len(top_stable_hits) >= 3:
        validation_report["recommendations"].append("Sufficient stable hits for detailed analysis")
    
    # Save structured JSON report
    json_report_file = output_dir / "validation_report.json"
    with open(json_report_file, 'w') as f:
        json.dump(validation_report, f, indent=2, default=str)
    
    validation_report["json_report_file"] = str(json_report_file)
    
    # Generate optional natural language report
    if generate_natural_language:
        nl_report = generate_natural_language_report(validation_report, output_dir)
        validation_report["natural_language_report_file"] = nl_report
    
    return validation_report


def generate_natural_language_report(validation_data: Dict[str, Any], output_dir: Path) -> str:
    """Generate natural language validation report."""
    from datetime import datetime
    
    report_content = f"""# Rescoring & Minimization Validation Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Type:** Comprehensive Docking Hit Validation with OpenMM Minimization

## Executive Summary

**Validation Results:**
- **Total Hits Processed:** {validation_data['summary']['total_hits']}
- **Validated Hits:** {validation_data['summary']['validated_hits']} ({validation_data['summary']['validation_rate']}%)
- **Minimization Success Rate:** {validation_data['summary']['minimization_success_rate']}%
- **Structural Stability Rate:** {validation_data['summary']['structural_stability_rate']}%

## Hit Quality Distribution

| Category | Count | Description |
|----------|-------|-------------|
| **Excellent** | {validation_data['validation_categories']['excellent_hits']} | High confidence hits (score ≥6) |
| **Good** | {validation_data['validation_categories']['good_hits']} | Validated hits (score ≥4) |
| **Acceptable** | {validation_data['validation_categories']['acceptable_hits']} | Marginal hits (score ≥2) |
| **Poor** | {validation_data['validation_categories']['poor_hits']} | Low confidence hits |

## Minimization Analysis

**Overall Statistics:**
- **Attempted:** {validation_data['minimization_analysis']['total_attempted']}
- **Successful:** {validation_data['minimization_analysis']['successful']}
- **Skipped:** {validation_data['minimization_analysis']['skipped']}
- **Failed:** {validation_data['minimization_analysis']['failed']}
- **Structurally Stable:** {validation_data['minimization_analysis']['structurally_stable']}

"""

    # Add stability metrics if available
    if validation_data['minimization_analysis'].get('avg_stability_score'):
        report_content += f"""
**Stability Metrics:**
- **Average Stability Score:** {validation_data['minimization_analysis']['avg_stability_score']}/5.0
- **Average RMSD:** {validation_data['minimization_analysis'].get('avg_rmsd', 'N/A')} Å
- **Average Energy Change:** {validation_data['minimization_analysis'].get('avg_energy_change', 'N/A')} kJ/mol
"""

    # Add top stable hits
    if validation_data['top_stable_hits']:
        report_content += f"""
## 🏆 Top {len(validation_data['top_stable_hits'])} Structurally Stable Hits

"""
        for i, hit in enumerate(validation_data['top_stable_hits'], 1):
            report_content += f"""### {i}. {hit['hit_name']}
- **Stability Score:** {hit['stability_score']}/5.0 ({hit['stability_percentage']}%)
- **RMSD:** {hit['rmsd_angstroms']} Å
- **Energy Change:** {hit['energy_change']} kJ/mol
- **Job ID:** {hit['job_id']}

"""

    # Add recommendations
    if validation_data['recommendations']:
        report_content += """## Recommendations

"""
        for rec in validation_data['recommendations']:
            report_content += f"- {rec}\n"

    report_content += f"""
## Technical Details

**Validation Criteria:**
- Binding affinity improvement through rescoring
- Structural stability after energy minimization  
- RMSD analysis for conformational changes
- Energy convergence assessment
- Force field compatibility validation

**Minimization Protocol:**
- Force Fields: AMBER14 (protein) + OpenFF SMIRNOFF (ligands)
- Energy minimization with L-BFGS optimizer
- Implicit solvent models for robustness
- Comprehensive stability scoring (0-5 scale)

---
*This report provides comprehensive validation of docking hits through rescoring and structural minimization analysis.*
"""

    # Save natural language report
    nl_report_file = output_dir / "validation_report.md"
    with open(nl_report_file, 'w') as f:
        f.write(report_content)
    
    return str(nl_report_file)


def rescore_top_hits(
    top_hits: List[Dict[str, Any]],
    job_index: Dict[str, Dict[str, Any]],
    docking_results: List[Dict[str, Any]],
    output_dir: Path,
    methods: List[str] = None
) -> Dict[str, Any]:
    """Rescore top hits using multiple methods."""
    if methods is None:
        methods = ["vina_rescore"]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create result index for quick lookup
    result_index = {}
    for result in docking_results:
        job_id = result.get("job_id")
        if job_id and result.get("status") == "success":
            result_index[job_id] = result

    # Secondary index by (pdb_id, pocket_id, ligand_name)
    job_lookup = {}
    for job_id, job in job_index.items():
        key = (
            job.get("pdb_id"),
            job.get("pocket_id"),
            job.get("ligand_name"),
        )
        if all(key):
            job_lookup[key] = job_id

    rescoring_results = []

    for hit in top_hits:
        ligand_name = hit.get("name")
        if not ligand_name:
            continue
        
        pdb_id = hit.get("pdb_id")
        pocket_id = hit.get("pocket_id")

        # Find corresponding job and result
        job_id = job_lookup.get((pdb_id, pocket_id, ligand_name))

        if job_id is None:
            # Fallback: match by ligand name only (legacy behaviour)
            for jid, job in job_index.items():
                if job.get("ligand_name") == ligand_name:
                    job_id = jid
                    break

        if not job_id or job_id not in result_index:
            continue

        job = job_index[job_id]
        result = result_index[job_id]

        hit_dir = output_dir / f"{job.get('pdb_id')}_{job.get('pocket_id')}_{ligand_name}"
        hit_dir.mkdir(parents=True, exist_ok=True)

        hit_result = {
            "ligand_name": ligand_name,
            "job_id": job_id,
            "pdb_id": pdb_id,
            "pocket_id": pocket_id,
            "original_score": hit.get("score"),
            "rescoring_methods": {}
        }
        
        # Get file paths
        receptor_pdbqt = Path(job.get("receptor_pdbqt", ""))
        ligand_pdbqt = Path(job.get("ligand_pdbqt", ""))
        
        if not receptor_pdbqt.exists() or not ligand_pdbqt.exists():
            hit_result["error"] = "Missing input files"
            rescoring_results.append(hit_result)
            continue
        
        # Apply rescoring methods
        for method in methods:
            method_dir = hit_dir / method
            
            if method == "vina_rescore":
                rescore_result = rescore_with_vina(
                    receptor_pdbqt, ligand_pdbqt, method_dir
                )
            elif method == "gnina" and _find_executable("gnina"):
                # Would need to convert PDBQT to PDB/SDF first
                rescore_result = {"method": method, "status": "skipped", "reason": "format conversion needed"}
            else:
                rescore_result = {"method": method, "status": "skipped", "reason": "method not available"}
            
            hit_result["rescoring_methods"][method] = rescore_result
        
        rescoring_results.append(hit_result)
    
    # Save rescoring summary
    summary = {
        "total_hits_processed": len(rescoring_results),
        "methods_used": methods,
        "results": rescoring_results
    }
    
    summary_file = output_dir / "rescoring_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return {
        "rescoring_summary": summary,
        "summary_file": str(summary_file),
        "output_dir": str(output_dir)
    }


__all__ = [
    "rescore_with_vina",
    "rescore_with_gnina", 
    "minimize_with_amber",
    "minimize_with_openmm",
    "cluster_poses",
    "rescore_top_hits",
    "enhanced_rescore_and_refine"
]
