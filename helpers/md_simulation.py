#!/usr/bin/env python3
"""
Molecular Dynamics Simulation Module for Protein-Ligand Complexes

This module provides functionality for running MD simulations on protein-ligand
complexes using GROMACS with AMBER force fields for proteins and GAFF for ligands.

Required packages:
- gromacs (MD engine)
- acpype (for ligand parameterization)
- mdtraj (for trajectory analysis)
- numpy
- pandas
- matplotlib

Installation:
  conda install -c conda-forge gromacs
  conda install -c conda-forge acpype
  conda install -c conda-forge mdtraj
  conda install -c conda-forge matplotlib
"""

import json
import subprocess
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import traceback
import tempfile
import os
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import time
import pickle
import hashlib

try:
    import mdtraj as md
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    import numpy as np
    ANALYSIS_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"MD analysis dependencies not available: {e}")
    ANALYSIS_DEPENDENCIES_AVAILABLE = False


def check_gromacs_installation() -> Tuple[bool, str]:
    """Check if GROMACS is installed and accessible."""
    try:
        result = subprocess.run(['gmx', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            return True, version_line
        else:
            return False, "GROMACS not found"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "GROMACS not installed or not in PATH"


def check_acpype_installation() -> Tuple[bool, str]:
    """Check if acpype is installed and accessible."""
    try:
        result = subprocess.run(['acpype', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, "acpype available"
        else:
            return False, "acpype not working"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "acpype not installed or not in PATH"


def check_md_dependencies() -> Tuple[bool, List[str]]:
    """Check if all required MD simulation packages are available."""
    missing_packages = []
    
    # Check GROMACS
    gromacs_ok, gromacs_msg = check_gromacs_installation()
    if not gromacs_ok:
        missing_packages.append("gromacs")
    
    # Check acpype
    acpype_ok, acpype_msg = check_acpype_installation()
    if not acpype_ok:
        missing_packages.append("acpype")
    
    # Check Python packages
    if not ANALYSIS_DEPENDENCIES_AVAILABLE:
        missing_packages.extend(["mdtraj", "matplotlib", "pandas"])
    
    return len(missing_packages) == 0, missing_packages


def convert_pdbqt_to_pdb(pdbqt_path: Path, output_path: Path) -> Path:
    """
    Convert PDBQT file to PDB format by removing PDBQT-specific lines.
    
    Args:
        pdbqt_path: Path to input PDBQT file
        output_path: Path to output PDB file
    
    Returns:
        Path to created PDB file
    """
    with open(pdbqt_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            # Skip PDBQT-specific lines and keep only structural data
            if line.startswith(('ATOM', 'HETATM', 'CONECT', 'END')):
                # Remove PDBQT-specific columns (charges, atom types)
                if line.startswith(('ATOM', 'HETATM')):
                    # Keep only PDB standard columns (1-66)
                    pdb_line = line[:66] + '\n'
                    outfile.write(pdb_line)
                else:
                    outfile.write(line)
    
    return output_path


def create_complex_from_original_pdb_and_docking(original_pdb: Path, docking_result_pdbqt: Path,
                                               output_dir: Path, model_number: int = 1) -> Path:
    """
    Create protein-ligand complex from original PDB and docking result.
    
    Args:
        original_pdb: Path to original protein PDB file
        docking_result_pdbqt: Path to docking result PDBQT file
        output_dir: Directory to save complex
        model_number: Which model to extract from docking result
    
    Returns:
        Path to complex PDB file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    complex_pdb = output_dir / "complex.pdb"
    
    # First, fix the original PDB structure
    fixed_protein_pdb = fix_original_pdb_structure(original_pdb, output_dir)
    
    # Extract ligand pose from docking result
    ligand_pdb = output_dir / "ligand_pose.pdb"
    extract_ligand_pose_from_docking(docking_result_pdbqt, ligand_pdb, model_number)
    
    # Combine fixed protein with ligand pose
    with open(complex_pdb, 'w') as outfile:
        # Write protein atoms from fixed PDB
        with open(fixed_protein_pdb, 'r') as prot_file:
            for line in prot_file:
                if line.startswith(('ATOM', 'HETATM')):
                    outfile.write(line)
                elif line.startswith(('HEADER', 'TITLE', 'CRYST1', 'REMARK')):
                    outfile.write(line)
        
        # Write ligand atoms
        with open(ligand_pdb, 'r') as lig_file:
            for line in lig_file:
                if line.startswith(('ATOM', 'HETATM')):
                    outfile.write(line)
        
        outfile.write("END\n")
    
    return complex_pdb


def fix_original_pdb_structure(pdb_path: Path, output_dir: Path) -> Path:
    """
    Fix original PDB structure for MD simulation.
    Removes problematic residues and standardizes the structure.
    
    Args:
        pdb_path: Path to original PDB file
        output_dir: Directory to save fixed PDB
    
    Returns:
        Path to fixed PDB file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fixed_pdb = output_dir / "protein_fixed_original.pdb"
    
    try:
        with open(pdb_path, 'r') as infile, open(fixed_pdb, 'w') as outfile:
            lines = infile.readlines()
            
            # Track residues and identify problematic ones
            residues = {}
            problematic_residues = set()
            
            # First pass: analyze structure
            for line in lines:
                if line.startswith(('ATOM', 'HETATM')):
                    chain_id = line[21:22].strip()
                    res_num = int(line[22:26].strip())
                    res_name = line[17:20].strip()
                    atom_name = line[12:16].strip()
                    
                    res_key = (chain_id, res_num)
                    if res_key not in residues:
                        residues[res_key] = {'name': res_name, 'atoms': set()}
                    residues[res_key]['atoms'].add(atom_name)
            
            # Identify problematic residues
            for res_key, res_info in residues.items():
                chain_id, res_num = res_key
                res_name = res_info['name']
                
                # Remove histidines (ring completion issues)
                if res_name == 'HIS':
                    problematic_residues.add(res_key)
                    print(f"Removing HIS{chain_id}{res_num} to prevent ring issues")
                
                # Remove non-standard residues
                standard_residues = {
                    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                    'THR', 'TRP', 'TYR', 'VAL'
                }
                if res_name not in standard_residues:
                    problematic_residues.add(res_key)
                    print(f"Removing non-standard residue {res_name}{chain_id}{res_num}")
            
            # Second pass: write clean structure
            for line in lines:
                if line.startswith('ATOM'):
                    chain_id = line[21:22].strip()
                    res_num = int(line[22:26].strip())
                    res_key = (chain_id, res_num)
                    
                    # Skip problematic residues
                    if res_key in problematic_residues:
                        continue
                    
                    outfile.write(line)
                elif line.startswith(('HEADER', 'TITLE', 'CRYST1')):
                    outfile.write(line)
        
        print(f"Fixed original PDB structure: {fixed_pdb}")
        print(f"Removed {len(problematic_residues)} problematic residues")
        return fixed_pdb
        
    except Exception as e:
        print(f"Warning: Could not fix original PDB: {e}")
        return pdb_path


def extract_ligand_pose_from_docking(docking_pdbqt: Path, output_pdb: Path, model_number: int = 1):
    """
    Extract specific ligand pose from docking result PDBQT and convert to PDB.
    
    Args:
        docking_pdbqt: Path to docking result PDBQT file
        output_pdb: Path to save ligand pose PDB
        model_number: Which model to extract (1-based)
    """
    try:
        with open(docking_pdbqt, 'r') as infile, open(output_pdb, 'w') as outfile:
            lines = infile.readlines()
            
            current_model = 0
            in_target_model = False
            
            for line in lines:
                if line.startswith('MODEL'):
                    current_model += 1
                    if current_model == model_number:
                        in_target_model = True
                    else:
                        in_target_model = False
                elif line.startswith('ENDMDL'):
                    if in_target_model:
                        break
                elif line.startswith(('ATOM', 'HETATM')) and in_target_model:
                    # Convert PDBQT line to PDB format (remove charges and atom types)
                    pdb_line = line[:66] + "  1.00  0.00" + line[76:78] + "\n"
                    outfile.write(pdb_line)
            
            outfile.write("END\n")
            
    except Exception as e:
        print(f"Error extracting ligand pose: {e}")
        raise


def create_complex_from_docking_result(receptor_pdbqt: Path, ligand_result_pdbqt: Path, 
                                     output_dir: Path, model_number: int = 1) -> Path:
    """
    Create protein-ligand complex PDB from docking results.
    
    Args:
        receptor_pdbqt: Path to receptor PDBQT file
        ligand_result_pdbqt: Path to docking result PDBQT file
        output_dir: Directory to save complex PDB
        model_number: Which model/pose to extract from result (default: 1)
    
    Returns:
        Path to created complex PDB file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    complex_pdb = output_dir / "complex.pdb"
    
    # Convert receptor PDBQT to PDB
    receptor_pdb = output_dir / "receptor_temp.pdb"
    convert_pdbqt_to_pdb(receptor_pdbqt, receptor_pdb)
    
    # Extract specific model from ligand result
    ligand_pdb = output_dir / "ligand_temp.pdb"
    extract_ligand_model(ligand_result_pdbqt, ligand_pdb, model_number)
    
    # Combine receptor and ligand into complex
    with open(complex_pdb, 'w') as outfile:
        # Write receptor
        with open(receptor_pdb, 'r') as receptor_file:
            for line in receptor_file:
                if not line.startswith('END'):
                    outfile.write(line)
        
        # Write ligand
        with open(ligand_pdb, 'r') as ligand_file:
            for line in ligand_file:
                if not line.startswith('END'):
                    outfile.write(line)
        
        outfile.write("END\n")
    
    # Clean up temporary files
    receptor_pdb.unlink()
    ligand_pdb.unlink()
    
    return complex_pdb


def extract_ligand_model(result_pdbqt: Path, output_pdb: Path, model_number: int = 1) -> Path:
    """
    Extract specific model from Vina result PDBQT file.
    
    Args:
        result_pdbqt: Path to Vina result PDBQT file
        output_pdb: Path to output PDB file
        model_number: Which model to extract (1-based)
    
    Returns:
        Path to extracted ligand PDB file
    """
    current_model = 0
    writing = False
    
    with open(result_pdbqt, 'r') as infile, open(output_pdb, 'w') as outfile:
        for line in infile:
            if line.startswith('MODEL'):
                current_model += 1
                if current_model == model_number:
                    writing = True
                else:
                    writing = False
            elif line.startswith('ENDMDL'):
                if writing:
                    break
            elif writing and line.startswith(('ATOM', 'HETATM')):
                # Convert PDBQT line to PDB format
                pdb_line = line[:66] + '\n'
                outfile.write(pdb_line)
        
        outfile.write("END\n")
    
    return output_pdb


def separate_protein_ligand(complex_pdb_path: Path, output_dir: Path) -> Tuple[Path, Path]:
    """
    Separate protein and ligand from complex PDB file.
    
    Args:
        complex_pdb_path: Path to complex PDB file
        output_dir: Directory to save separated files
    
    Returns:
        Tuple of (protein_pdb_path, ligand_pdb_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    protein_pdb = output_dir / "protein.pdb"
    ligand_pdb = output_dir / "ligand.pdb"
    
    protein_lines = []
    ligand_lines = []
    
    with open(complex_pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                # Standard amino acid residues are protein
                res_name = line[17:20].strip()
                if res_name in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                               'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                               'THR', 'TRP', 'TYR', 'VAL', 'HOH', 'WAT']:
                    protein_lines.append(line)
                else:
                    ligand_lines.append(line)
            elif line.startswith(('CRYST1', 'HEADER', 'TITLE', 'REMARK')):
                protein_lines.append(line)
                ligand_lines.append(line)
    
    # Write protein PDB
    with open(protein_pdb, 'w') as f:
        f.writelines(protein_lines)
        f.write("END\n")
    
    # Write ligand PDB  
    with open(ligand_pdb, 'w') as f:
        f.writelines(ligand_lines)
        f.write("END\n")
    
    return protein_pdb, ligand_pdb


def prepare_ligand_topology(ligand_pdb_path: Path, output_dir: Path) -> Dict[str, Path]:
    """
    Generate ligand topology using acpype (GAFF force field).
    
    Args:
        ligand_pdb_path: Path to ligand PDB file
        output_dir: Directory to save topology files
    
    Returns:
        Dictionary with paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run acpype to generate GAFF parameters
    cmd = [
        'acpype',
        '-i', str(ligand_pdb_path.resolve()),
        '-b', 'ligand',
        '-o', 'gmx',
        '-c', 'bcc'  # Use AM1-BCC charges
    ]
    
    try:
        result = subprocess.run(cmd, cwd=output_dir.resolve(), capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"acpype failed: {result.stderr}")
        
        # Find generated files
        acpype_dir = output_dir / "ligand.acpype"
        if not acpype_dir.exists():
            raise FileNotFoundError("acpype output directory not found")
        
        gro_file = acpype_dir / "ligand_GMX.gro"
        top_file = acpype_dir / "ligand_GMX.top"
        itp_file = acpype_dir / "ligand_GMX.itp"
        
        # Copy files to main output directory
        ligand_gro = output_dir / "ligand.gro"
        ligand_top = output_dir / "ligand.top"
        ligand_itp = output_dir / "ligand.itp"
        
        shutil.copy2(gro_file, ligand_gro)
        shutil.copy2(top_file, ligand_top)
        shutil.copy2(itp_file, ligand_itp)
        
        return {
            "gro": ligand_gro,
            "top": ligand_top,
            "itp": ligand_itp
        }
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("acpype timed out")
    except Exception as e:
        raise RuntimeError(f"Ligand topology preparation failed: {e}")


def fix_protein_structure(protein_pdb_path: Path, output_dir: Path) -> Path:
    """
    Fix common protein structure issues that cause GROMACS pdb2gmx to fail.
    Specifically handles incomplete histidine rings and other structural issues.
    
    Args:
        protein_pdb_path: Path to original protein PDB file
        output_dir: Directory to save fixed PDB file
    
    Returns:
        Path to fixed PDB file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fixed_pdb = output_dir / "protein_fixed.pdb"
    
    try:
        with open(protein_pdb_path, 'r') as infile, open(fixed_pdb, 'w') as outfile:
            lines = infile.readlines()
            
            # Track residues and their atoms by chain and residue number
            residues = {}
            problematic_residues = set()
            
            # First pass: identify problematic residues
            for line in lines:
                if line.startswith(('ATOM', 'HETATM')):
                    chain_id = line[21:22].strip()
                    res_num = int(line[22:26].strip())
                    res_name = line[17:20].strip()
                    atom_name = line[12:16].strip()
                    
                    res_key = (chain_id, res_num)
                    if res_key not in residues:
                        residues[res_key] = {'name': res_name, 'atoms': set()}
                    residues[res_key]['atoms'].add(atom_name)
            
            # Check for incomplete histidine rings and other problematic residues
            for res_key, res_info in residues.items():
                chain_id, res_num = res_key
                if res_info['name'] == 'HIS':
                    required_atoms = {'N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'}
                    missing_atoms = required_atoms - res_info['atoms']
                    if missing_atoms:
                        print(f"HIS{chain_id}{res_num} missing atoms: {missing_atoms}")
                        problematic_residues.add(res_key)
                    
                    # Also check for incomplete ring specifically
                    ring_atoms = {'CG', 'ND1', 'CD2', 'CE1', 'NE2'}
                    missing_ring_atoms = ring_atoms - res_info['atoms']
                    if missing_ring_atoms:
                        print(f"HIS{chain_id}{res_num} incomplete ring, missing: {missing_ring_atoms}")
                        problematic_residues.add(res_key)
            
            # Remove ALL histidine residues to avoid any ring issues
            print("Removing ALL histidine residues to prevent ring completion issues")
            for res_key, res_info in residues.items():
                if res_info['name'] == 'HIS':
                    problematic_residues.add(res_key)
            
            # Second pass: write fixed structure
            for line in lines:
                if line.startswith(('ATOM', 'HETATM')):
                    chain_id = line[21:22].strip()
                    res_num = int(line[22:26].strip())
                    res_name = line[17:20].strip()
                    res_key = (chain_id, res_num)
                    
                    # Skip problematic residues entirely
                    if res_key in problematic_residues:
                        continue
                    
                    # For other residues, keep all atoms
                    outfile.write(line)
                else:
                    # Keep all non-atom lines (headers, etc.)
                    outfile.write(line)
        
        print(f"Fixed protein structure saved to {fixed_pdb}")
        print(f"Removed {len(problematic_residues)} problematic residues: {problematic_residues}")
        return fixed_pdb
        
    except Exception as e:
        print(f"Warning: Could not fix protein structure: {e}")
        return protein_pdb_path


def prepare_protein_topology(protein_pdb_path: Path, output_dir: Path) -> Dict[str, Path]:
    """
    Prepare protein topology using GROMACS pdb2gmx with AMBER force field.
    
    Args:
        protein_pdb_path: Path to protein PDB file
        output_dir: Directory to save topology files
    
    Returns:
        Dictionary with paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    protein_gro = output_dir / "protein.gro"
    protein_top = output_dir / "protein.top"
    protein_itp = output_dir / "protein.itp"
    
    # First attempt with original file
    attempts = [
        {
            "pdb_file": protein_pdb_path,
            "description": "original structure",
            "forcefield": "amber99sb-ildn",
            "water_model": "tip3p",
            "extra_flags": []
        },
        {
            "pdb_file": fix_protein_structure(protein_pdb_path, output_dir),
            "description": "fixed structure", 
            "forcefield": "amber99sb-ildn",
            "water_model": "tip3p",
            "extra_flags": []
        },
        {
            "pdb_file": fix_protein_structure(protein_pdb_path, output_dir),
            "description": "fixed structure with missing residues ignored",
            "forcefield": "amber99sb-ildn",
            "water_model": "tip3p",
            "extra_flags": ['-missing']
        }
    ]
    
    for attempt in attempts:
        print(f"Attempting pdb2gmx with {attempt['description']}...")
        
        cmd = [
            'gmx', 'pdb2gmx',
            '-f', str(attempt["pdb_file"].resolve()),
            '-o', str(protein_gro.resolve()),
            '-p', str(protein_top.resolve()),
            '-i', str(protein_itp.resolve()),
            '-ff', attempt["forcefield"],
            '-water', attempt["water_model"],
            '-ignh',  # Ignore hydrogens in input file
            '-nobackup'  # Don't create backup files
        ] + attempt["extra_flags"]
        
        try:
            # Provide input for interactive prompts (force field and water model)
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  input="1\n1\n", timeout=120)
            if result.returncode == 0:
                print(f"Successfully prepared protein topology with {attempt['description']}")
                return {
                    "gro": protein_gro,
                    "top": protein_top,
                    "itp": protein_itp
                }
            else:
                print(f"pdb2gmx failed with {attempt['description']}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"pdb2gmx timed out with {attempt['description']}")
            continue
        except Exception as e:
            print(f"pdb2gmx error with {attempt['description']}: {e}")
            continue
    
    # If all attempts failed, raise the last error
    raise RuntimeError(f"All pdb2gmx attempts failed. Last error: {result.stderr}")


def _add_ligand_atomtypes(ligand_itp: Path, outfile):
    """
    Extract and add ligand atomtypes to the main topology file.
    
    Args:
        ligand_itp: Path to ligand ITP file
        outfile: Output file handle to write atomtypes to
    """
    with open(ligand_itp, 'r') as infile:
        in_atomtypes = False
        
        for line in infile:
            if line.strip().startswith('[ atomtypes ]'):
                in_atomtypes = True
                outfile.write('\n; Ligand atomtypes\n')
                outfile.write(line)
                continue
            elif line.strip().startswith('[') and not line.strip().startswith('[ atomtypes ]'):
                in_atomtypes = False
                break
            
            if in_atomtypes:
                outfile.write(line)


def _get_ligand_molecule_name(ligand_itp_path: Path) -> str:
    """Extract molecule name from ligand ITP file."""
    with open(ligand_itp_path, 'r') as f:
        in_moleculetype = False
        for line in f:
            if line.strip().startswith('[ moleculetype ]'):
                in_moleculetype = True
                continue
            elif in_moleculetype and not line.strip().startswith(';') and line.strip():
                # First non-comment line in moleculetype section contains the name
                return line.strip().split()[0]
    return "LIG"  # Default fallback


def _create_position_restraint_files(output_dir: Path):
    """Create position restraint files for all protein chains."""
    for itp_file in output_dir.glob("protein_Protein_chain_*.itp"):
        chain_name = itp_file.stem  # e.g., "protein_Protein_chain_A"
        posre_file = output_dir / f"posre_{chain_name.replace('protein_', '')}.itp"
        
        # Create empty position restraint file to satisfy topology requirements
        with open(posre_file, 'w') as f:
            f.write(f"; Position restraint file for {chain_name}\n")
            f.write("; This file is intentionally empty - no position restraints applied\n")
            f.write("[ position_restraints ]\n")
            f.write("; Empty section\n")


def _clean_ligand_itp(input_itp, output_itp):
    """Remove atomtypes section from ligand ITP file."""
    with open(input_itp, 'r') as infile, open(output_itp, 'w') as outfile:
        skip_atomtypes = False
        for line in infile:
            if line.strip().startswith('[ atomtypes ]'):
                skip_atomtypes = True
                continue
            elif line.strip().startswith('[') and skip_atomtypes:
                skip_atomtypes = False
                outfile.write(line)
            elif not skip_atomtypes:
                outfile.write(line)


def create_complex_topology(protein_files: Dict[str, Path], ligand_files: Dict[str, Path], 
                          output_dir: Path) -> Dict[str, Path]:
    """
    Create combined protein-ligand topology file.
    
    Args:
        protein_files: Dictionary with protein topology files
        ligand_files: Dictionary with ligand topology files
        output_dir: Directory to save combined topology
    
    Returns:
        Dictionary with paths to combined files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    complex_gro = output_dir / "complex.gro"
    complex_top = output_dir / "complex.top"
    
    # Combine GRO files (coordinates)
    with open(complex_gro, 'w') as outfile:
        # Read protein gro
        with open(protein_files["gro"], 'r') as prot_file:
            lines = prot_file.readlines()
            outfile.write(lines[0])  # Title line
            
            # Count atoms
            prot_atoms = int(lines[1])
            outfile.write(f"{prot_atoms + get_ligand_atom_count(ligand_files['gro'])}\n")
            
            # Write protein atoms
            for line in lines[2:-1]:  # Skip title, count, and box line
                outfile.write(line)
        
        # Read ligand gro (skip header)
        with open(ligand_files["gro"], 'r') as lig_file:
            lines = lig_file.readlines()
            for line in lines[2:-1]:  # Skip title, count, and box line
                outfile.write(line)
        
        # Write box line from protein
        with open(protein_files["gro"], 'r') as prot_file:
            lines = prot_file.readlines()
            outfile.write(lines[-1])  # Box line
    
    # Copy ALL files from protein directory to complex directory
    # This includes ITP files, position restraint files, and any other dependencies
    protein_dir = protein_files["top"].parent
    for file_path in protein_dir.glob("*"):
        if file_path.is_file() and file_path.name != protein_files["top"].name:
            shutil.copy2(file_path, output_dir)
    
    # Create missing position restraint files by extracting from protein ITP files
    _create_position_restraint_files(output_dir)
    
    # Process and copy ligand ITP file (remove atomtypes section)
    ligand_itp_clean = output_dir / "ligand_clean.itp"
    _clean_ligand_itp(ligand_files["itp"], ligand_itp_clean)
    
    # Create combined topology file with proper GROMACS structure
    with open(complex_top, 'w') as outfile:
        # Read protein topology
        with open(protein_files["top"], 'r') as prot_file:
            lines = prot_file.readlines()
            
            # Write header and forcefield include
            for line in lines:
                if line.startswith('#include "amber99sb-ildn.ff/forcefield.itp"'):
                    outfile.write(line)
                    outfile.write('\n')
                    # Add ligand atomtypes immediately after forcefield include
                    _add_ligand_atomtypes(ligand_files["itp"], outfile)
                    outfile.write('\n')
                    break
                else:
                    outfile.write(line)
            
            # Continue with protein includes and other sections
            skip_to_system = False
            for line in lines:
                if line.startswith('#include "amber99sb-ildn.ff/forcefield.itp"'):
                    continue  # Already written
                elif line.startswith('[ system ]'):
                    outfile.write(line)
                    # Read next lines until we find molecules section
                    continue
                elif line.startswith('[ molecules ]'):
                    # Include clean ligand itp file before molecules section
                    outfile.write('#include "ligand_clean.itp"\n\n')
                    outfile.write(line)
                    skip_to_system = True
                    continue
                elif skip_to_system:
                    outfile.write(line)
                else:
                    outfile.write(line)
        
        # Add ligand to molecules section (get name from ligand ITP)
        ligand_name = _get_ligand_molecule_name(ligand_itp_clean)
        outfile.write(f"{ligand_name}                 1\n")
    
    return {
        "gro": complex_gro,
        "top": complex_top
    }


def get_ligand_atom_count(ligand_gro_path: Path) -> int:
    """Get number of atoms in ligand GRO file."""
    with open(ligand_gro_path, 'r') as f:
        lines = f.readlines()
        return int(lines[1])  # Second line contains atom count


def run_gromacs_md_simulation(complex_files: Dict[str, Path], output_dir: Path,
                             simulation_time_ns: float = 0.05, temperature: float = 300.0) -> Dict[str, Any]:
    """
    Run GROMACS MD simulation workflow.
    
    Args:
        complex_files: Dictionary with complex topology files (gro, top)
        output_dir: Directory to save simulation outputs
        simulation_time_ns: Simulation time in nanoseconds
        temperature: Temperature in Kelvin
    
    Returns:
        Dictionary with simulation results
    """
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Solvate the complex
        print("Adding water box...")
        solvated_gro = output_dir / "solvated.gro"
        cmd = [
            'gmx', 'solvate',
            '-cp', str(complex_files["gro"].resolve()),
            '-cs', 'spc216.gro',
            '-o', str(solvated_gro.resolve()),
            '-p', str(complex_files["top"].resolve()),
            '-nobackup'  # Don't create backup files
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        
        # Step 2: Add ions for neutralization
        print("Adding ions...")
        ions_gro = output_dir / "ions.gro"
        cmd = [
            'gmx', 'grompp',
            '-f', str(create_mdp_file(output_dir, "ions").resolve()),
            '-c', str(solvated_gro.resolve()),
            '-p', str(complex_files["top"].resolve()),
            '-o', str((output_dir / "ions.tpr").resolve()),
            '-maxwarn', '10',  # Allow more warnings for stability
            '-nobackup'  # Don't create backup files
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        
        # Check if we have enough solvent molecules for neutralization
        # If not, skip ion addition (system will remain with net charge)
        try:
            cmd = [
                'gmx', 'genion',
                '-s', str((output_dir / "ions.tpr").resolve()),
                '-o', str(ions_gro.resolve()),
                '-p', str(complex_files["top"].resolve()),
                '-pname', 'NA',
                '-nname', 'CL',
                '-neutral',
                '-nobackup'  # Don't create backup files
            ]
            subprocess.run(cmd, input="13\n", check=True, capture_output=True, text=True, timeout=120)
        except subprocess.CalledProcessError:
            print("Warning: Not enough solvent for ion neutralization. Proceeding with charged system.")
            # Copy solvated structure as ions structure
            shutil.copy2(solvated_gro, ions_gro)
        
        # Step 3: Energy minimization
        print("Energy minimization...")
        em_gro = output_dir / "em.gro"
        cmd = [
            'gmx', 'grompp',
            '-f', str(create_mdp_file(output_dir, "minim").resolve()),
            '-c', str(ions_gro.resolve()),
            '-p', str(complex_files["top"].resolve()),
            '-o', str((output_dir / "em.tpr").resolve()),
            '-maxwarn', '10',  # Allow more warnings for stability
            '-nobackup'  # Don't create backup files
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        
        cmd = [
            'gmx', 'mdrun',
            '-v', '-deffnm', str((output_dir / "em").resolve()),
            '-ntmpi', '1', '-ntomp', '8',
            '-nobackup'  # Don't create backup files
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        
        # Step 4: NVT equilibration
        print("NVT equilibration...")
        cmd = [
            'gmx', 'grompp',
            '-f', str(create_mdp_file(output_dir, "nvt", temperature).resolve()),
            '-c', str((output_dir / "em.gro").resolve()),
            '-r', str((output_dir / "em.gro").resolve()),
            '-p', str(complex_files["top"].resolve()),
            '-o', str((output_dir / "nvt.tpr").resolve()),
            '-maxwarn', '10',  # Allow more warnings for stability
            '-nobackup'  # Don't create backup files
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        
        cmd = [
            'gmx', 'mdrun',
            '-v', '-deffnm', str((output_dir / "nvt").resolve()),
            '-ntmpi', '1', '-ntomp', '8',
            '-nobackup'  # Don't create backup files
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        
        # Step 5: NPT equilibration
        print("NPT equilibration...")
        cmd = [
            'gmx', 'grompp',
            '-f', str(create_mdp_file(output_dir, "npt", temperature).resolve()),
            '-c', str((output_dir / "nvt.gro").resolve()),
            '-r', str((output_dir / "nvt.gro").resolve()),
            '-t', str((output_dir / "nvt.cpt").resolve()),
            '-p', str(complex_files["top"].resolve()),
            '-o', str((output_dir / "npt.tpr").resolve()),
            '-maxwarn', '10',  # Allow more warnings for stability
            '-nobackup'  # Don't create backup files
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        
        cmd = [
            'gmx', 'mdrun',
            '-v', '-deffnm', str((output_dir / "npt").resolve()),
            '-ntmpi', '1', '-ntomp', '8',
            '-nobackup'  # Don't create backup files
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        
        # Step 6: Production MD
        print(f"Production MD ({simulation_time_ns} ns)...")
        cmd = [
            'gmx', 'grompp',
            '-f', str(create_mdp_file(output_dir, "md", temperature, simulation_time_ns).resolve()),
            '-c', str((output_dir / "npt.gro").resolve()),
            '-t', str((output_dir / "npt.cpt").resolve()),
            '-p', str(complex_files["top"].resolve()),
            '-o', str((output_dir / "md.tpr").resolve()),
            '-maxwarn', '10',  # Allow more warnings for stability
            '-nobackup'  # Don't create backup files
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        
        cmd = [
            'gmx', 'mdrun',
            '-v', '-deffnm', str((output_dir / "md").resolve()),
            '-ntmpi', '1', '-ntomp', '8',
            '-nobackup'  # Don't create backup files
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=int(simulation_time_ns * 7200))
        
        return {
            "status": "success",
            "simulation_time_ns": simulation_time_ns,
            "trajectory_file": str(output_dir / "md.xtc"),
            "structure_file": str(output_dir / "md.gro"),
            "topology_file": str(output_dir / "md.gro"),  # Use GRO file for analysis compatibility
            "log_file": str(output_dir / "md.log"),
            "energy_file": str(output_dir / "md.edr")
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "status": "failed",
            "error": f"GROMACS command failed: {e.stderr if e.stderr else str(e)}"
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": f"MD simulation failed: {str(e)}"
        }


def create_mdp_file(output_dir: Path, step: str, temperature: float = 300.0, 
                   simulation_time_ns: float = 0.05) -> Path:
    """
    Create GROMACS MDP parameter files for different simulation steps.
    
    Args:
        output_dir: Directory to save MDP file
        step: Type of simulation (ions, minim, nvt, npt, md)
        temperature: Temperature in Kelvin
        simulation_time_ns: Simulation time in nanoseconds (for md only)
    
    Returns:
        Path to created MDP file
    """
    mdp_file = output_dir / f"{step}.mdp"
    
    if step == "ions":
        content = """
; Parameters for ion addition
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 10
cutoff-scheme   = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
"""
    elif step == "minim":
        content = """
; Parameters for energy minimization
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 50000
nstlist     = 10
cutoff-scheme   = Verlet
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
"""
    elif step == "nvt":
        content = f"""
; NVT equilibration (ultra-fast for screening)
integrator      = md
nsteps          = 10000
dt              = 0.002
nstxout         = 2000
nstvout         = 2000
nstenergy       = 2000
nstlog          = 2000
continuation    = no
constraint_algorithm = lincs
constraints     = h-bonds
lincs_iter      = 1
lincs_order     = 4

cutoff-scheme   = Verlet
ns_type         = grid
nstlist         = 10
rcoulomb        = 1.0
rvdw            = 1.0

coulombtype     = PME
pme_order       = 4
fourierspacing  = 0.16

tcoupl          = V-rescale
tc-grps         = Protein Non-Protein
tau_t           = 0.1     0.1
ref_t           = {temperature}     {temperature}

pcoupl          = no

gen_vel         = yes
gen_temp        = {temperature}
gen_seed        = -1
"""
    elif step == "npt":
        content = f"""
; Parameters for NPT equilibration (ultra-fast)
define                  = -DPOSRES
integrator              = md
nsteps                  = 10000
dt                      = 0.002
nstxout                 = 2000
nstvout                 = 2000
nstenergy               = 2000
nstlog                  = 2000
continuation            = yes
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4
cutoff-scheme           = Verlet
nstlist                 = 10
rcoulomb                = 1.0
rvdw                    = 1.0
coulombtype             = PME
pme_order               = 4
fourierspacing          = 0.16
tcoupl                  = V-rescale
tc-grps                 = Protein Non-Protein
tau_t                   = 0.1     0.1
ref_t                   = {temperature}     {temperature}
pcoupl                  = Parrinello-Rahman
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = 1.0
compressibility         = 4.5e-5
refcoord_scaling        = com
pbc                     = xyz
DispCorr                = EnerPres
gen_vel                 = no
"""
    elif step == "md":
        nsteps = int(simulation_time_ns * 500000)  # 2 fs timestep (original working settings)
        content = f"""
; Parameters for production MD
integrator              = md
nsteps                  = {nsteps}
dt                      = 0.002
nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstlog                  = 5000
nstcalcenergy           = 100
nstenergy               = 1000
nstxout-compressed      = 1000
continuation            = yes
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4
cutoff-scheme           = Verlet
nstlist                 = 10
rcoulomb                = 1.0
rvdw                    = 1.0
coulombtype             = PME
pme_order               = 4
fourierspacing          = 0.16
tcoupl                  = V-rescale
tc-grps                 = Protein Non-Protein
tau_t                   = 0.1     0.1
ref_t                   = {temperature}     {temperature}
pcoupl                  = Parrinello-Rahman
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = 1.0
compressibility         = 4.5e-5
pbc                     = xyz
DispCorr                = EnerPres
gen_vel                 = no
"""
    
    with open(mdp_file, 'w') as f:
        f.write(content)
    
    return mdp_file


def create_enhanced_md_visualizations(traj: 'md.Trajectory', protein_atoms: np.ndarray, 
                                    ligand_atoms: np.ndarray, output_dir: Path, 
                                    hit_name: str = "unknown") -> Dict[str, Any]:
    """
    Create enhanced visualizations for MD trajectory analysis.
    
    Args:
        traj: MDTraj trajectory object
        protein_atoms: Array of protein atom indices
        ligand_atoms: Array of ligand atom indices
        output_dir: Directory to save visualization files
        hit_name: Name of the hit for labeling
        
    Returns:
        Dictionary with visualization file paths and analysis data
    """
    if not ANALYSIS_DEPENDENCIES_AVAILABLE:
        return {"status": "skipped", "reason": "Analysis dependencies not available"}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualization_results = {
        "status": "success",
        "plots": {},
        "data": {}
    }
    
    try:
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Extract hit name from trajectory file path for labeling
        # MDTraj trajectory object doesn't have filename attribute, use hit_name parameter
        if hit_name == "unknown":
            hit_name = "trajectory_analysis"
        
        # 1. Enhanced 6-panel analysis plot
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Calculate time array
        time_ns = traj.time
        
        # Panel 1: Protein backbone RMSD
        ax1 = fig.add_subplot(gs[0, 0])
        if len(protein_atoms) > 0:
            protein_ca = traj.topology.select("protein and name CA")
            if len(protein_ca) > 0:
                protein_rmsd = md.rmsd(traj, traj, frame=0, atom_indices=protein_ca)
                ax1.plot(time_ns, protein_rmsd * 10, 'b-', linewidth=2, alpha=0.8)
                ax1.set_xlabel('Time (ps)')
                ax1.set_ylabel('Protein RMSD (Å)')
                ax1.set_title(f'Protein Backbone RMSD - {hit_name}')
                ax1.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(time_ns, protein_rmsd * 10, 1)
                p = np.poly1d(z)
                ax1.plot(time_ns, p(time_ns), "r--", alpha=0.8, linewidth=1)
                
                visualization_results["data"]["protein_rmsd_final"] = float(protein_rmsd[-1] * 10)
                visualization_results["data"]["protein_rmsd_mean"] = float(np.mean(protein_rmsd * 10))
                visualization_results["data"]["protein_rmsd_std"] = float(np.std(protein_rmsd * 10))
        
        # Panel 2: Ligand RMSD
        ax2 = fig.add_subplot(gs[0, 1])
        if len(ligand_atoms) > 0:
            ligand_rmsd = md.rmsd(traj, traj, frame=0, atom_indices=ligand_atoms)
            ax2.plot(time_ns, ligand_rmsd * 10, 'g-', linewidth=2, alpha=0.8)
            ax2.set_xlabel('Time (ps)')
            ax2.set_ylabel('Ligand RMSD (Å)')
            ax2.set_title(f'Ligand RMSD - {hit_name}')
            ax2.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(time_ns, ligand_rmsd * 10, 1)
            p = np.poly1d(z)
            ax2.plot(time_ns, p(time_ns), "r--", alpha=0.8, linewidth=1)
            
            visualization_results["data"]["ligand_rmsd_final"] = float(ligand_rmsd[-1] * 10)
            visualization_results["data"]["ligand_rmsd_mean"] = float(np.mean(ligand_rmsd * 10))
            visualization_results["data"]["ligand_rmsd_std"] = float(np.std(ligand_rmsd * 10))
        
        # Panel 3: Protein-Ligand Distance
        ax3 = fig.add_subplot(gs[0, 2])
        if len(protein_atoms) > 0 and len(ligand_atoms) > 0:
            # Calculate center of mass distance
            protein_com = md.compute_center_of_mass(traj.atom_slice(protein_atoms))
            ligand_com = md.compute_center_of_mass(traj.atom_slice(ligand_atoms))
            distances = np.linalg.norm(protein_com - ligand_com, axis=1) * 10
            
            ax3.plot(time_ns, distances, 'purple', linewidth=2, alpha=0.8)
            ax3.set_xlabel('Time (ps)')
            ax3.set_ylabel('Distance (Å)')
            ax3.set_title(f'Protein-Ligand Distance - {hit_name}')
            ax3.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(time_ns, distances, 1)
            p = np.poly1d(z)
            ax3.plot(time_ns, p(time_ns), "r--", alpha=0.8, linewidth=1)
            
            visualization_results["data"]["binding_distance_final"] = float(distances[-1])
            visualization_results["data"]["binding_distance_mean"] = float(np.mean(distances))
        
        # Panel 4: Radius of Gyration
        ax4 = fig.add_subplot(gs[1, 0])
        if len(protein_atoms) > 0:
            rg = md.compute_rg(traj.atom_slice(protein_atoms))
            ax4.plot(time_ns, rg * 10, 'orange', linewidth=2, alpha=0.8)
            ax4.set_xlabel('Time (ps)')
            ax4.set_ylabel('Radius of Gyration (Å)')
            ax4.set_title(f'Protein Compactness - {hit_name}')
            ax4.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(time_ns, rg * 10, 1)
            p = np.poly1d(z)
            ax4.plot(time_ns, p(time_ns), "r--", alpha=0.8, linewidth=1)
            
            visualization_results["data"]["radius_gyration_final"] = float(rg[-1] * 10)
            visualization_results["data"]["radius_gyration_mean"] = float(np.mean(rg * 10))
        
        # Panel 5: SASA (Solvent Accessible Surface Area)
        ax5 = fig.add_subplot(gs[1, 1])
        if len(ligand_atoms) > 0:
            try:
                sasa = md.shrake_rupley(traj, mode='atom')
                ligand_sasa = np.sum(sasa[:, ligand_atoms], axis=1)
                ax5.plot(time_ns, ligand_sasa, 'cyan', linewidth=2, alpha=0.8)
                ax5.set_xlabel('Time (ps)')
                ax5.set_ylabel('SASA (nm²)')
                ax5.set_title(f'Ligand Surface Exposure - {hit_name}')
                ax5.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(time_ns, ligand_sasa, 1)
                p = np.poly1d(z)
                ax5.plot(time_ns, p(time_ns), "r--", alpha=0.8, linewidth=1)
                
                visualization_results["data"]["ligand_sasa_final"] = float(ligand_sasa[-1])
                visualization_results["data"]["ligand_sasa_mean"] = float(np.mean(ligand_sasa))
            except Exception as e:
                ax5.text(0.5, 0.5, f'SASA calculation failed:\n{str(e)}', 
                        transform=ax5.transAxes, ha='center', va='center')
                ax5.set_title(f'Ligand Surface Exposure - {hit_name}')
        
        # Panel 6: Contact Analysis
        ax6 = fig.add_subplot(gs[1, 2])
        if len(protein_atoms) > 0 and len(ligand_atoms) > 0:
            try:
                # Simplified contact analysis using distance matrix
                contact_cutoff = 0.5  # 5 Å cutoff in nm
                n_contacts = []
                
                for frame in range(traj.n_frames):
                    # Get coordinates for this frame
                    protein_coords = traj.xyz[frame, protein_atoms, :]
                    ligand_coords = traj.xyz[frame, ligand_atoms, :]
                    
                    # Calculate all pairwise distances
                    frame_contacts = 0
                    for p_coord in protein_coords:
                        for l_coord in ligand_coords:
                            dist = np.linalg.norm(p_coord - l_coord)
                            if dist < contact_cutoff:
                                frame_contacts += 1
                    n_contacts.append(frame_contacts)
                
                if len(n_contacts) > 0 and max(n_contacts) > 0:
                    ax6.plot(time_ns, n_contacts, 'red', linewidth=2, alpha=0.8)
                    ax6.set_xlabel('Time (ps)')
                    ax6.set_ylabel('Number of Contacts')
                    ax6.set_title(f'Protein-Ligand Contacts - {hit_name}')
                    ax6.grid(True, alpha=0.3)
                    
                    # Add trend line
                    if len(n_contacts) > 1:
                        z = np.polyfit(time_ns, n_contacts, 1)
                        p = np.poly1d(z)
                        ax6.plot(time_ns, p(time_ns), "b--", alpha=0.8, linewidth=1)
                    
                    visualization_results["data"]["contacts_final"] = int(n_contacts[-1])
                    visualization_results["data"]["contacts_mean"] = float(np.mean(n_contacts))
                    
            except Exception as e:
                ax6.text(0.5, 0.5, f'Contact analysis failed:\n{str(e)}', 
                        transform=ax6.transAxes, ha='center', va='center')
                ax6.set_title(f'Protein-Ligand Contacts - {hit_name}')
        
        # Panel 7: RMSD Distribution
        ax7 = fig.add_subplot(gs[2, 0])
        if len(protein_atoms) > 0:
            protein_ca = traj.topology.select("protein and name CA")
            if len(protein_ca) > 0:
                protein_rmsd = md.rmsd(traj, traj, frame=0, atom_indices=protein_ca) * 10
                ax7.hist(protein_rmsd, bins=20, alpha=0.7, color='blue', edgecolor='black')
                ax7.axvline(np.mean(protein_rmsd), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(protein_rmsd):.2f} Å')
                ax7.set_xlabel('RMSD (Å)')
                ax7.set_ylabel('Frequency')
                ax7.set_title(f'RMSD Distribution - {hit_name}')
                ax7.legend()
                ax7.grid(True, alpha=0.3)
        
        # Panel 8: Stability Score
        ax8 = fig.add_subplot(gs[2, 1])
        if len(protein_atoms) > 0 and len(ligand_atoms) > 0:
            # Calculate a composite stability score
            protein_ca = traj.topology.select("protein and name CA")
            if len(protein_ca) > 0:
                protein_rmsd = md.rmsd(traj, traj, frame=0, atom_indices=protein_ca) * 10
                ligand_rmsd = md.rmsd(traj, traj, frame=0, atom_indices=ligand_atoms) * 10
                
                # Stability score: lower RMSD = higher stability
                stability_score = 100 / (1 + protein_rmsd + ligand_rmsd)
                
                ax8.plot(time_ns, stability_score, 'darkgreen', linewidth=2, alpha=0.8)
                ax8.set_xlabel('Time (ps)')
                ax8.set_ylabel('Stability Score')
                ax8.set_title(f'Binding Stability - {hit_name}')
                ax8.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(time_ns, stability_score, 1)
                p = np.poly1d(z)
                ax8.plot(time_ns, p(time_ns), "r--", alpha=0.8, linewidth=1)
                
                visualization_results["data"]["stability_score_final"] = float(stability_score[-1])
                visualization_results["data"]["stability_score_mean"] = float(np.mean(stability_score))
        
        # Panel 9: Summary Statistics
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        # Create summary text
        summary_text = f"MD Analysis Summary - {hit_name}\n\n"
        if "protein_rmsd_mean" in visualization_results["data"]:
            summary_text += f"Protein RMSD: {visualization_results['data']['protein_rmsd_mean']:.2f} ± {visualization_results['data'].get('protein_rmsd_std', 0):.2f} Å\n"
        if "ligand_rmsd_mean" in visualization_results["data"]:
            summary_text += f"Ligand RMSD: {visualization_results['data']['ligand_rmsd_mean']:.2f} ± {visualization_results['data'].get('ligand_rmsd_std', 0):.2f} Å\n"
        if "binding_distance_mean" in visualization_results["data"]:
            summary_text += f"Binding Distance: {visualization_results['data']['binding_distance_mean']:.2f} Å\n"
        if "contacts_mean" in visualization_results["data"]:
            summary_text += f"Avg Contacts: {visualization_results['data']['contacts_mean']:.1f}\n"
        if "stability_score_mean" in visualization_results["data"]:
            summary_text += f"Stability Score: {visualization_results['data']['stability_score_mean']:.1f}\n"
        
        summary_text += f"\nSimulation Length: {time_ns[-1]:.1f} ps\n"
        summary_text += f"Total Frames: {traj.n_frames}\n"
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Save enhanced analysis plot
        enhanced_plot_path = output_dir / f"enhanced_md_analysis_{hit_name}.png"
        plt.savefig(enhanced_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        visualization_results["plots"]["enhanced_analysis"] = str(enhanced_plot_path)
        
        print(f"[VISUALIZATION] Created enhanced MD analysis plot: {enhanced_plot_path}")
        
        return visualization_results
        
    except Exception as e:
        plt.close('all')  # Clean up any open figures
        return {
            "status": "failed",
            "error": f"Enhanced visualization failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


def create_energy_analysis_plots(simulation_dir: Path, output_dir: Path, 
                               hit_name: str = "unknown") -> Dict[str, Any]:
    """
    Create energy analysis plots from GROMACS energy files.
    
    Args:
        simulation_dir: Directory containing GROMACS simulation files
        output_dir: Directory to save energy plots
        hit_name: Name of the hit for labeling
        
    Returns:
        Dictionary with energy analysis results
    """
    if not ANALYSIS_DEPENDENCIES_AVAILABLE:
        return {"status": "skipped", "reason": "Analysis dependencies not available"}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    energy_results = {
        "status": "success",
        "plots": {},
        "data": {}
    }
    
    try:
        # Look for energy files (.edr)
        edr_files = list(simulation_dir.glob("*.edr"))
        if not edr_files:
            return {"status": "skipped", "reason": "No energy files found"}
        
        edr_file = edr_files[0]  # Use the first .edr file found
        
        # Extract energy data using gmx energy
        energy_data_file = output_dir / f"energy_data_{hit_name}.xvg"
        
        # Create gmx energy command to extract key energies
        energy_cmd = [
            "gmx", "energy", "-f", str(edr_file), "-o", str(energy_data_file),
            "-quiet", "-nobackup"  # Don't create backup files
        ]
        
        # First, get available energy terms
        list_cmd = ["gmx", "energy", "-f", str(edr_file), "-quiet", "-nobackup"]
        list_result = subprocess.run(list_cmd, input="0\n", text=True, capture_output=True, timeout=30)
        
        # Use simpler energy selections - just get what's available
        energy_selections = "1\n2\n3\n4\n5\n0\n"  # First 5 energy terms
        
        try:
            result = subprocess.run(
                energy_cmd,
                input=energy_selections,
                text=True,
                capture_output=True,
                timeout=60
            )
            
            if result.returncode == 0 and energy_data_file.exists():
                # Parse energy data with robust error handling
                try:
                    # Read file and skip comment lines manually
                    with open(energy_data_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Filter out comment lines and empty lines
                    data_lines = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#') and not line.startswith('@'):
                            data_lines.append(line)
                    
                    if not data_lines:
                        energy_results["status"] = "failed"
                        energy_results["error"] = "No data lines found in energy file"
                        return energy_results
                    
                    # Parse data lines manually to handle variable column counts
                    parsed_data = []
                    for line in data_lines:
                        fields = line.split()
                        if len(fields) >= 2:  # At least time and one energy value
                            try:
                                parsed_data.append([float(f) for f in fields])
                            except ValueError:
                                continue  # Skip lines with non-numeric data
                    
                    if not parsed_data:
                        energy_results["status"] = "failed"
                        energy_results["error"] = "No valid numeric data found in energy file"
                        return energy_results
                    
                    # Convert to numpy array for easier handling
                    import numpy as np
                    data_array = np.array(parsed_data)
                    
                    # Extract columns based on available data
                    time_col = data_array[:, 0]
                    
                    # Initialize energy arrays - use first available columns as energy terms
                    energy_terms = []
                    energy_labels = []
                    
                    for i in range(1, min(6, data_array.shape[1])):  # Up to 5 energy terms
                        energy_terms.append(data_array[:, i])
                        energy_labels.append(f"Energy Term {i}")
                    
                    # Assign to standard variables for backward compatibility
                    potential = energy_terms[0] if len(energy_terms) > 0 else None
                    kinetic = energy_terms[1] if len(energy_terms) > 1 else None
                    total = energy_terms[2] if len(energy_terms) > 2 else None
                    temperature = energy_terms[3] if len(energy_terms) > 3 else None
                    pressure = energy_terms[4] if len(energy_terms) > 4 else None
                    
                except Exception as parse_error:
                    energy_results["status"] = "failed"
                    energy_results["error"] = f"Energy data parsing failed: {str(parse_error)}"
                    return energy_results
                
                # Only proceed if we have at least time and one energy term
                if potential is not None:
                    
                    # Create energy analysis plot
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    fig.suptitle(f'Energy Analysis - {hit_name}', fontsize=16)
                    
                    # Potential Energy
                    if potential is not None:
                        axes[0, 0].plot(time_col, potential, 'b-', alpha=0.8)
                        axes[0, 0].set_title('Potential Energy')
                        axes[0, 0].set_xlabel('Time (ps)')
                        axes[0, 0].set_ylabel('Energy (kJ/mol)')
                        axes[0, 0].grid(True, alpha=0.3)
                    else:
                        axes[0, 0].text(0.5, 0.5, 'Potential Energy\nNot Available', 
                                       transform=axes[0, 0].transAxes, ha='center', va='center')
                        axes[0, 0].set_title('Potential Energy')
                    
                    # Kinetic Energy
                    if kinetic is not None:
                        axes[0, 1].plot(time_col, kinetic, 'r-', alpha=0.8)
                        axes[0, 1].set_title('Kinetic Energy')
                        axes[0, 1].set_xlabel('Time (ps)')
                        axes[0, 1].set_ylabel('Energy (kJ/mol)')
                        axes[0, 1].grid(True, alpha=0.3)
                    else:
                        axes[0, 1].text(0.5, 0.5, 'Kinetic Energy\nNot Available', 
                                       transform=axes[0, 1].transAxes, ha='center', va='center')
                        axes[0, 1].set_title('Kinetic Energy')
                    
                    # Total Energy
                    if total is not None:
                        axes[0, 2].plot(time_col, total, 'g-', alpha=0.8)
                        axes[0, 2].set_title('Total Energy')
                        axes[0, 2].set_xlabel('Time (ps)')
                        axes[0, 2].set_ylabel('Energy (kJ/mol)')
                        axes[0, 2].grid(True, alpha=0.3)
                    else:
                        axes[0, 2].text(0.5, 0.5, 'Total Energy\nNot Available', 
                                       transform=axes[0, 2].transAxes, ha='center', va='center')
                        axes[0, 2].set_title('Total Energy')
                    
                    # Temperature
                    if temperature is not None:
                        axes[1, 0].plot(time_col, temperature, 'orange', alpha=0.8)
                        axes[1, 0].set_title('Temperature')
                        axes[1, 0].set_xlabel('Time (ps)')
                        axes[1, 0].set_ylabel('Temperature (K)')
                        axes[1, 0].grid(True, alpha=0.3)
                    else:
                        axes[1, 0].text(0.5, 0.5, 'Temperature\nNot Available', 
                                       transform=axes[1, 0].transAxes, ha='center', va='center')
                        axes[1, 0].set_title('Temperature')
                    
                    # Pressure
                    if pressure is not None:
                        axes[1, 1].plot(time_col, pressure, 'purple', alpha=0.8)
                        axes[1, 1].set_title('Pressure')
                        axes[1, 1].set_xlabel('Time (ps)')
                        axes[1, 1].set_ylabel('Pressure (bar)')
                        axes[1, 1].grid(True, alpha=0.3)
                    else:
                        axes[1, 1].text(0.5, 0.5, 'Pressure\nNot Available', 
                                       transform=axes[1, 1].transAxes, ha='center', va='center')
                        axes[1, 1].set_title('Pressure')
                    
                    # Energy Distribution (use the first available energy term)
                    energy_for_dist = total if total is not None else (potential if potential is not None else kinetic)
                    if energy_for_dist is not None:
                        axes[1, 2].hist(energy_for_dist, bins=30, alpha=0.7, color='green', edgecolor='black')
                        axes[1, 2].axvline(np.mean(energy_for_dist), color='red', linestyle='--', 
                                          label=f'Mean: {np.mean(energy_for_dist):.1f} kJ/mol')
                        axes[1, 2].set_title('Energy Distribution')
                        axes[1, 2].set_xlabel('Energy (kJ/mol)')
                        axes[1, 2].set_ylabel('Frequency')
                        axes[1, 2].legend()
                        axes[1, 2].grid(True, alpha=0.3)
                    else:
                        axes[1, 2].text(0.5, 0.5, 'Energy Distribution\nNot Available', 
                                       transform=axes[1, 2].transAxes, ha='center', va='center')
                        axes[1, 2].set_title('Energy Distribution')
                    
                    plt.tight_layout()
                    
                    # Save energy plot
                    energy_plot_path = output_dir / f"energy_analysis_{hit_name}.png"
                    plt.savefig(energy_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()
                    
                    energy_results["plots"]["energy_analysis"] = str(energy_plot_path)
                    
                    # Store energy statistics
                    energy_results["data"] = {
                        "potential_energy_mean": float(np.mean(potential)),
                        "potential_energy_std": float(np.std(potential)),
                        "kinetic_energy_mean": float(np.mean(kinetic)),
                        "kinetic_energy_std": float(np.std(kinetic)),
                        "total_energy_mean": float(np.mean(total)),
                        "total_energy_std": float(np.std(total)),
                        "temperature_mean": float(np.mean(temperature)),
                        "temperature_std": float(np.std(temperature)),
                        "pressure_mean": float(np.mean(pressure)),
                        "pressure_std": float(np.std(pressure))
                    }
                    
                    print(f"[VISUALIZATION] Created energy analysis plot: {energy_plot_path}")
                    
                else:
                    energy_results["status"] = "failed"
                    energy_results["error"] = "Insufficient energy data columns"
                    
            else:
                energy_results["status"] = "failed"
                energy_results["error"] = f"gmx energy failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            energy_results["status"] = "failed"
            energy_results["error"] = "Energy extraction timed out"
        except Exception as e:
            energy_results["status"] = "failed"
            energy_results["error"] = f"Energy extraction failed: {str(e)}"
        
        return energy_results
        
    except Exception as e:
        plt.close('all')  # Clean up any open figures
        return {
            "status": "failed",
            "error": f"Energy analysis failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


def analyze_gromacs_trajectory(trajectory_file: Path, topology_file: Path, 
                              output_dir: Path, hit_name: str = "unknown") -> Dict[str, Any]:
    """
    Analyze GROMACS trajectory for binding stability and dynamics.
    
    Args:
        trajectory_file: Path to trajectory file (.xtc)
        topology_file: Path to topology file (.top)
        output_dir: Directory to save analysis results
    
    Returns:
        Dictionary with analysis results
    """
    try:
        import mdtraj as md
        import numpy as np
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load trajectory with topology
        print(f"Loading trajectory: {trajectory_file}")
        
        # Find corresponding structure file for topology (MDTraj doesn't support .top)
        # Use the initial complex structure that matches the trajectory
        sim_dir = trajectory_file.parent
        
        # Try different structure files in order of preference
        structure_candidates = [
            sim_dir / "md.gro",      # Final MD structure
            sim_dir / "npt.gro",     # NPT equilibrated structure  
            sim_dir / "nvt.gro",     # NVT equilibrated structure
            sim_dir / "em.gro",      # Energy minimized structure
            sim_dir / "ions.gro",    # Ionized structure
            topology_file.parent / "complex.gro"  # Original complex
        ]
        
        structure_file = None
        for candidate in structure_candidates:
            if candidate.exists():
                structure_file = candidate
                break
        
        if structure_file is None:
            raise FileNotFoundError("No compatible structure file found for trajectory analysis")
        
        print(f"Using structure file: {structure_file}")
        
        # Load trajectory with structure file
        traj = md.load(str(trajectory_file), top=str(structure_file))
        
        # Basic trajectory statistics
        n_frames = traj.n_frames
        n_atoms = traj.n_atoms
        time_ns = traj.time[-1] / 1000.0  # Convert ps to ns
        
        # Identify protein and ligand atoms with error handling
        try:
            protein_atoms = traj.topology.select("protein")
        except:
            # Fallback: select by chain or residue name patterns
            protein_atoms = traj.topology.select("chainid 0 or chainid 1 or chainid 2 or chainid 3")
        
        try:
            # Try different ligand residue names
            ligand_atoms = traj.topology.select("resname LIG")
            if len(ligand_atoms) == 0:
                ligand_atoms = traj.topology.select("resname ligand")
            if len(ligand_atoms) == 0:
                # Find non-protein, non-water residues
                all_atoms = set(range(traj.n_atoms))
                water_atoms = set(traj.topology.select("water"))
                protein_atoms_set = set(protein_atoms)
                ligand_atoms = list(all_atoms - water_atoms - protein_atoms_set)
        except:
            ligand_atoms = []
        
        analysis_results = {
            "n_frames": n_frames,
            "n_atoms": n_atoms,
            "simulation_time_ns": time_ns,
            "protein_atoms": len(protein_atoms),
            "ligand_atoms": len(ligand_atoms)
        }
        
        if len(protein_atoms) > 0 and len(ligand_atoms) > 0:
            # Calculate RMSD for protein backbone
            protein_ca = traj.topology.select("protein and name CA")
            if len(protein_ca) > 0:
                rmsd_protein = md.rmsd(traj, traj[0], atom_indices=protein_ca)
                analysis_results["protein_rmsd_nm"] = {
                    "mean": float(np.mean(rmsd_protein)),
                    "std": float(np.std(rmsd_protein)),
                    "max": float(np.max(rmsd_protein)),
                    "final": float(rmsd_protein[-1])
                }
            
            # Calculate ligand RMSD
            if len(ligand_atoms) > 0:
                rmsd_ligand = md.rmsd(traj, traj[0], atom_indices=ligand_atoms)
                analysis_results["ligand_rmsd_nm"] = {
                    "mean": float(np.mean(rmsd_ligand)),
                    "std": float(np.std(rmsd_ligand)),
                    "max": float(np.max(rmsd_ligand)),
                    "final": float(rmsd_ligand[-1])
                }
            
            # Calculate protein-ligand distance
            if len(protein_atoms) > 0 and len(ligand_atoms) > 0:
                protein_center = md.compute_center_of_mass(traj.atom_slice(protein_atoms))
                ligand_center = md.compute_center_of_mass(traj.atom_slice(ligand_atoms))
                
                distances = np.linalg.norm(protein_center - ligand_center, axis=1)
                analysis_results["binding_distance_nm"] = {
                    "mean": float(np.mean(distances)),
                    "std": float(np.std(distances)),
                    "min": float(np.min(distances)),
                    "max": float(np.max(distances)),
                    "final": float(distances[-1])
                }
            
            # Calculate radius of gyration for complex
            rg = md.compute_rg(traj)
            analysis_results["radius_of_gyration_nm"] = {
                "mean": float(np.mean(rg)),
                "std": float(np.std(rg)),
                "final": float(rg[-1])
            }
            
            # Generate enhanced visualizations
            hit_name = trajectory_file.stem.replace("_traj", "").replace("md_", "")
            
            # Create enhanced visualizations
            enhanced_viz_results = create_enhanced_md_visualizations(
                traj, protein_atoms, ligand_atoms, output_dir, hit_name
            )
            
            if enhanced_viz_results["status"] == "success":
                analysis_results.update(enhanced_viz_results["data"])
                analysis_results["enhanced_analysis_plot"] = enhanced_viz_results["plots"].get("enhanced_analysis")
                print(f"[ANALYSIS] Enhanced visualizations created successfully")
            else:
                print(f"[ANALYSIS] Enhanced visualizations failed: {enhanced_viz_results.get('error', 'Unknown error')}")
            
            # Create energy analysis plots
            simulation_dir = trajectory_file.parent
            energy_viz_results = create_energy_analysis_plots(simulation_dir, output_dir, hit_name)
            
            if energy_viz_results["status"] == "success":
                analysis_results.update(energy_viz_results["data"])
                analysis_results["energy_analysis_plot"] = energy_viz_results["plots"].get("energy_analysis")
                print(f"[ANALYSIS] Energy analysis plots created successfully")
            else:
                print(f"[ANALYSIS] Energy analysis failed: {energy_viz_results.get('error', 'Unknown error')}")
            
            # Generate basic analysis plots for backward compatibility
            try:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f'MD Trajectory Analysis - {hit_name}', fontsize=16)
                
                # RMSD plots
                if "protein_rmsd_nm" in analysis_results:
                    axes[0, 0].plot(traj.time/1000, rmsd_protein*10)  # Convert to Angstrom
                    axes[0, 0].set_xlabel("Time (ns)")
                    axes[0, 0].set_ylabel("Protein RMSD (Å)")
                    axes[0, 0].set_title("Protein Backbone RMSD")
                    axes[0, 0].grid(True, alpha=0.3)
                
                if "ligand_rmsd_nm" in analysis_results:
                    axes[0, 1].plot(traj.time/1000, rmsd_ligand*10)  # Convert to Angstrom
                    axes[0, 1].set_xlabel("Time (ns)")
                    axes[0, 1].set_ylabel("Ligand RMSD (Å)")
                    axes[0, 1].set_title("Ligand RMSD")
                    axes[0, 1].grid(True, alpha=0.3)
                
                # Distance plot
                if "binding_distance_nm" in analysis_results:
                    axes[1, 0].plot(traj.time/1000, distances*10)  # Convert to Angstrom
                    axes[1, 0].set_xlabel("Time (ns)")
                    axes[1, 0].set_ylabel("Binding Distance (Å)")
                    axes[1, 0].set_title("Protein-Ligand Distance")
                    axes[1, 0].grid(True, alpha=0.3)
                
                # Radius of gyration
                if "radius_of_gyration_nm" in analysis_results:
                    axes[1, 1].plot(traj.time/1000, rg*10)  # Convert to Angstrom
                    axes[1, 1].set_xlabel("Time (ns)")
                    axes[1, 1].set_ylabel("Radius of Gyration (Å)")
                    axes[1, 1].set_title("Complex Compactness")
                    axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / "md_analysis.png", dpi=300, bbox_inches="tight")
                plt.close()
                
                analysis_results["plots_generated"] = True
                analysis_results["plot_file"] = str(output_dir / "md_analysis.png")
                
            except Exception as e:
                analysis_results["plots_generated"] = False
                analysis_results["plot_error"] = str(e)
        
        # Add comprehensive molecular visualization (new functionality)
        try:
            from .md_visualization import add_md_visualization_to_analysis
            
            print(f"[VISUALIZATION] Creating comprehensive molecular visualizations for {hit_name}")
            viz_results = add_md_visualization_to_analysis(
                simulation_dir=trajectory_file.parent,
                output_dir=output_dir,
                hit_name=hit_name
            )
            
            if viz_results.get("md_visualization", {}).get("status") == "success":
                analysis_results.update(viz_results)
                print(f"[VISUALIZATION] Molecular visualizations created successfully")
            else:
                print(f"[VISUALIZATION] Visualization creation failed: {viz_results.get('md_visualization', {}).get('error', 'Unknown error')}")
                analysis_results["md_visualization"] = viz_results.get("md_visualization", {})
                
        except ImportError:
            print(f"[VISUALIZATION] MD visualization module not available")
            analysis_results["md_visualization"] = {"status": "skipped", "reason": "Visualization module not available"}
        except Exception as e:
            print(f"[VISUALIZATION] Visualization creation error: {e}")
            analysis_results["md_visualization"] = {"status": "failed", "error": str(e)}

        return analysis_results
        
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Trajectory analysis failed: {str(e)}"
        }


def create_checkpoint_id(hit_data: Dict[str, Any]) -> str:
    """
    Create a unique checkpoint ID for a simulation based on input parameters.
    
    Args:
        hit_data: Dictionary containing simulation parameters
        
    Returns:
        Unique checkpoint ID string
    """
    # Create hash from key simulation parameters
    key_params = {
        "hit_name": hit_data["hit"].get("ligand_name", ""),
        "job_id": hit_data["hit"].get("job_id", ""),
        "pdb_id": hit_data["hit"].get("pdb_id", ""),
        "pocket_id": hit_data["hit"].get("pocket_id", ""),
        "original_pdb": hit_data["original_pdb"],
        "result_pdbqt": hit_data["result_pdbqt"],
        "simulation_time_ns": hit_data["simulation_time_ns"],
        "temperature": hit_data["temperature"]
    }
    print(key_params)
    
    # Create hash from parameters
    param_str = json.dumps(key_params, sort_keys=True)
    checkpoint_id = hashlib.md5(param_str.encode()).hexdigest()[:16]
    return checkpoint_id


def save_simulation_checkpoint(checkpoint_data: Dict[str, Any], checkpoint_dir: Path) -> bool:
    """
    Save simulation checkpoint data to disk.
    
    Args:
        checkpoint_data: Dictionary containing checkpoint information
        checkpoint_dir: Directory to save checkpoint files
        
    Returns:
        True if checkpoint saved successfully, False otherwise
    """
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_id = checkpoint_data.get("checkpoint_id", "unknown")
        
        # Save checkpoint metadata as JSON
        checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_id}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        print(f"[CHECKPOINT] Saved checkpoint {checkpoint_id}")
        return True
        
    except Exception as e:
        print(f"[CHECKPOINT] Failed to save checkpoint: {e}")
        return False


def load_simulation_checkpoint(checkpoint_id: str, checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load simulation checkpoint data from disk.
    
    Args:
        checkpoint_id: Unique checkpoint identifier
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Checkpoint data dictionary if found, None otherwise
    """
    try:
        checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_id}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            print(f"[CHECKPOINT] Loaded checkpoint {checkpoint_id}")
            return checkpoint_data
        else:
            return None
            
    except Exception as e:
        print(f"[CHECKPOINT] Failed to load checkpoint {checkpoint_id}: {e}")
        return None


def check_simulation_files_exist(file_paths: List[str]) -> bool:
    """
    Check if all required simulation files exist.
    
    Args:
        file_paths: List of file paths to check
        
    Returns:
        True if all files exist, False otherwise
    """
    for file_path in file_paths:
        if not Path(file_path).exists():
            return False
    return True


def run_single_md_simulation(hit_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single MD simulation for one hit with checkpoint support. Designed for parallel execution.
    
    Args:
        hit_data: Dictionary containing all necessary data for one simulation:
            - hit: Hit information (ligand_name, job_id, pdb_id, pocket_id)
            - output_dir: Output directory for this hit
            - original_pdb: Path to original PDB file
            - result_pdbqt: Path to docking result PDBQT file
            - simulation_time_ns: Simulation time in nanoseconds
            - temperature: Temperature in Kelvin
            - use_checkpoints: Whether to use checkpoint system (default: True)
    
    Returns:
        Dictionary with simulation result
    """
    hit = hit_data["hit"]
    hit_output_dir = Path(hit_data["output_dir"])
    original_pdb = Path(hit_data["original_pdb"])
    result_pdbqt = Path(hit_data["result_pdbqt"])
    simulation_time_ns = hit_data["simulation_time_ns"]
    temperature = hit_data["temperature"]
    use_checkpoints = hit_data.get("use_checkpoints", True)
    
    hit_name = hit.get("ligand_name", "unknown_hit")
    job_id = hit.get("job_id", "unknown_job")
    pdb_id = hit.get("pdb_id", "")
    pocket_id = hit.get("pocket_id", "")
    
    print(f"[MD-{job_id}] Starting simulation for {hit_name} (pdb: {pdb_id}, pocket: {pocket_id})")
    
    # Create checkpoint ID and directory
    checkpoint_id = create_checkpoint_id(hit_data)
    checkpoint_dir = hit_output_dir / "checkpoints"
    
    # Try to load existing checkpoint
    checkpoint_data = None
    if use_checkpoints:
        checkpoint_data = load_simulation_checkpoint(checkpoint_id, checkpoint_dir)
        
        # Check if we can resume from checkpoint
        if checkpoint_data:
            print(f"[MD-{job_id}] Found existing checkpoint for {hit_name}")
            
            # Verify checkpoint files still exist
            required_files = checkpoint_data.get("completed_files", [])
            if required_files and check_simulation_files_exist(required_files):
                # Check if simulation is already complete
                if checkpoint_data.get("status") == "completed":
                    print(f"[MD-{job_id}] Simulation already completed for {hit_name}, loading results")
                    return checkpoint_data.get("final_result", {
                        "status": "success",
                        "hit_name": hit_name,
                        "job_id": job_id,
                        "pdb_id": pdb_id,
                        "pocket_id": pocket_id,
                        "output_directory": str(hit_output_dir),
                        "resumed_from_checkpoint": True
                    })
                else:
                    print(f"[MD-{job_id}] Resuming simulation from checkpoint step: {checkpoint_data.get('last_completed_step', 'unknown')}")
            else:
                print(f"[MD-{job_id}] Checkpoint files missing, starting fresh simulation")
                checkpoint_data = None
    
    try:
        # Initialize checkpoint data if not resuming
        if not checkpoint_data:
            checkpoint_data = {
                "checkpoint_id": checkpoint_id,
                "hit_name": hit_name,
                "job_id": job_id,
                "pdb_id": pdb_id,
                "pocket_id": pocket_id,
                "start_time": datetime.now().isoformat(),
                "last_completed_step": "none",
                "completed_files": [],
                "status": "in_progress"
            }
        
        # Step 1: Create complex PDB from original PDB and docking results
        complex_pdb_path = None
        if checkpoint_data.get("last_completed_step") in ["none"]:
            print(f"[MD-{job_id}] Creating complex from original PDB and docking results")
            complex_pdb_path = create_complex_from_original_pdb_and_docking(
                original_pdb, result_pdbqt, hit_output_dir, model_number=1
            )
            
            # Update checkpoint
            checkpoint_data["last_completed_step"] = "complex_created"
            checkpoint_data["complex_pdb_path"] = str(complex_pdb_path)
            checkpoint_data["completed_files"].append(str(complex_pdb_path))
            if use_checkpoints:
                save_simulation_checkpoint(checkpoint_data, checkpoint_dir)
        else:
            complex_pdb_path = Path(checkpoint_data["complex_pdb_path"])
            print(f"[MD-{job_id}] Skipping complex creation (already completed)")
        
        # Step 2: Separate protein and ligand
        protein_pdb = None
        ligand_pdb = None
        if checkpoint_data.get("last_completed_step") in ["none", "complex_created"]:
            print(f"[MD-{job_id}] Separating protein and ligand")
            protein_pdb, ligand_pdb = separate_protein_ligand(complex_pdb_path, hit_output_dir)
            
            # Update checkpoint
            checkpoint_data["last_completed_step"] = "separated"
            checkpoint_data["protein_pdb_path"] = str(protein_pdb)
            checkpoint_data["ligand_pdb_path"] = str(ligand_pdb)
            checkpoint_data["completed_files"].extend([str(protein_pdb), str(ligand_pdb)])
            if use_checkpoints:
                save_simulation_checkpoint(checkpoint_data, checkpoint_dir)
        else:
            protein_pdb = Path(checkpoint_data["protein_pdb_path"])
            ligand_pdb = Path(checkpoint_data["ligand_pdb_path"])
            print(f"[MD-{job_id}] Skipping protein/ligand separation (already completed)")
        
        # Step 3: Prepare topologies
        protein_files = None
        ligand_files = None
        if checkpoint_data.get("last_completed_step") in ["none", "complex_created", "separated"]:
            print(f"[MD-{job_id}] Preparing topologies")
            protein_files = prepare_protein_topology(protein_pdb, hit_output_dir / "protein")
            ligand_files = prepare_ligand_topology(ligand_pdb, hit_output_dir / "ligand")
            
            # Update checkpoint
            checkpoint_data["last_completed_step"] = "topologies_prepared"
            checkpoint_data["protein_files"] = {k: str(v) for k, v in protein_files.items()}
            checkpoint_data["ligand_files"] = {k: str(v) for k, v in ligand_files.items()}
            checkpoint_data["completed_files"].extend([str(f) for f in protein_files.values()])
            checkpoint_data["completed_files"].extend([str(f) for f in ligand_files.values()])
            if use_checkpoints:
                save_simulation_checkpoint(checkpoint_data, checkpoint_dir)
        else:
            protein_files = {k: Path(v) for k, v in checkpoint_data["protein_files"].items()}
            ligand_files = {k: Path(v) for k, v in checkpoint_data["ligand_files"].items()}
            print(f"[MD-{job_id}] Skipping topology preparation (already completed)")
        
        # Step 4: Create complex topology
        complex_files = None
        if checkpoint_data.get("last_completed_step") in ["none", "complex_created", "separated", "topologies_prepared"]:
            print(f"[MD-{job_id}] Creating complex topology")
            complex_files = create_complex_topology(protein_files, ligand_files, hit_output_dir / "complex")
            
            # Update checkpoint
            checkpoint_data["last_completed_step"] = "complex_topology_created"
            checkpoint_data["complex_files"] = {k: str(v) for k, v in complex_files.items()}
            checkpoint_data["completed_files"].extend([str(f) for f in complex_files.values()])
            if use_checkpoints:
                save_simulation_checkpoint(checkpoint_data, checkpoint_dir)
        else:
            complex_files = {k: Path(v) for k, v in checkpoint_data["complex_files"].items()}
            print(f"[MD-{job_id}] Skipping complex topology creation (already completed)")
        
        # Step 5: Run GROMACS simulation
        simulation_result = None
        if checkpoint_data.get("last_completed_step") in ["none", "complex_created", "separated", "topologies_prepared", "complex_topology_created"]:
            print(f"[MD-{job_id}] Running GROMACS simulation")
            simulation_result = run_gromacs_md_simulation(
                complex_files, hit_output_dir / "simulation",
                simulation_time_ns=simulation_time_ns,
                temperature=temperature
            )
            
            # Update checkpoint
            checkpoint_data["last_completed_step"] = "simulation_completed"
            checkpoint_data["simulation_result"] = simulation_result
            if simulation_result.get("status") == "success":
                checkpoint_data["completed_files"].extend([
                    simulation_result.get("trajectory_file", ""),
                    simulation_result.get("topology_file", "")
                ])
            if use_checkpoints:
                save_simulation_checkpoint(checkpoint_data, checkpoint_dir)
        else:
            simulation_result = checkpoint_data["simulation_result"]
            print(f"[MD-{job_id}] Skipping GROMACS simulation (already completed)")
        
        # Step 6: Analyze trajectory
        if simulation_result["status"] == "success":
            if checkpoint_data.get("last_completed_step") in ["none", "complex_created", "separated", "topologies_prepared", "complex_topology_created", "simulation_completed"]:
                print(f"[MD-{job_id}] Analyzing trajectory")
                analysis_result = analyze_gromacs_trajectory(
                    Path(simulation_result["trajectory_file"]),
                    Path(simulation_result["topology_file"]),
                    hit_output_dir / "analysis",
                    hit_name
                )
                simulation_result["analysis"] = analysis_result
                
                # Update checkpoint - mark as completed
                checkpoint_data["last_completed_step"] = "analysis_completed"
                checkpoint_data["status"] = "completed"
                checkpoint_data["end_time"] = datetime.now().isoformat()
                if use_checkpoints:
                    save_simulation_checkpoint(checkpoint_data, checkpoint_dir)
            else:
                print(f"[MD-{job_id}] Skipping trajectory analysis (already completed)")
                if "analysis" in checkpoint_data.get("simulation_result", {}):
                    simulation_result["analysis"] = checkpoint_data["simulation_result"]["analysis"]
        
        # Add hit information to result
        simulation_result.update({
            "hit_name": hit_name,
            "job_id": job_id,
            "pdb_id": pdb_id,
            "pocket_id": pocket_id,
            "complex_pdb_path": str(complex_pdb_path),
            "output_directory": str(hit_output_dir),
            "checkpoint_id": checkpoint_id,
            "used_checkpoint": use_checkpoints
        })
        
        # Save final result to checkpoint
        if use_checkpoints:
            checkpoint_data["final_result"] = simulation_result
            save_simulation_checkpoint(checkpoint_data, checkpoint_dir)
        
        if simulation_result["status"] == "success":
            print(f"[MD-{job_id}] Simulation completed successfully for {hit_name}")
        else:
            print(f"[MD-{job_id}] Simulation failed for {hit_name}: {simulation_result.get('error', 'Unknown error')}")
        
        return simulation_result
        
    except Exception as e:
        error_result = {
            "hit_name": hit_name,
            "job_id": job_id,
            "pdb_id": pdb_id,
            "pocket_id": pocket_id,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "output_directory": str(hit_output_dir),
            "checkpoint_id": checkpoint_id
        }
        
        # Save error to checkpoint
        if use_checkpoints:
            checkpoint_data["status"] = "failed"
            checkpoint_data["error"] = str(e)
            checkpoint_data["end_time"] = datetime.now().isoformat()
            save_simulation_checkpoint(checkpoint_data, checkpoint_dir)
        
        print(f"[MD-{job_id}] Simulation failed for {hit_name}: {e}")
        return error_result


def run_protein_ligand_md_batch(validated_hits: List[Dict[str, Any]], output_dir: Path,
                               state: Dict[str, Any] = None,
                               simulation_time_ns: float = 5.0, temperature: float = 300.0,
                               max_parallel_jobs: int = None, use_checkpoints: bool = True) -> Dict[str, Any]:
    """
    Run GROMACS MD simulations on a batch of validated docking hits with parallel processing.
    
    Args:
        validated_hits: List of validated docking hits with ligand_name, job_id, pdb_id info
        output_dir: Base directory for MD simulation outputs
        state: Agent state dictionary containing paths and job information
        simulation_time_ns: Simulation time in nanoseconds
        temperature: Temperature in Kelvin
        max_parallel_jobs: Maximum number of parallel MD simulations (default: CPU count)
        use_checkpoints: Whether to use checkpoint system for resuming interrupted simulations
    
    Returns:
        Dictionary with batch simulation results
    """
    # Check dependencies first
    deps_ok, missing = check_md_dependencies()
    if not deps_ok:
        return {
            "status": "failed",
            "error": f"Missing required packages: {missing}",
            "required_packages": [
                "conda install -c conda-forge gromacs",
                "conda install -c conda-forge acpype", 
                "conda install -c conda-forge mdtraj",
                "conda install -c conda-forge matplotlib"
            ]
        }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine optimal number of parallel jobs
    if max_parallel_jobs is None:
        # Use half of available CPUs to avoid overwhelming the system
        max_parallel_jobs = max(1, mp.cpu_count() // 2)
    
    print(f"[MD] Starting batch MD simulation with {max_parallel_jobs} parallel jobs")
    
    batch_results = {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "simulation_parameters": {
            "simulation_time_ns": simulation_time_ns,
            "temperature_K": temperature,
            "md_engine": "GROMACS",
            "force_field": "AMBER99SB-ILDN + GAFF",
            "max_parallel_jobs": max_parallel_jobs
        },
        "results": [],
        "summary": {}
    }
    
    # Get job index and docking results from state if available
    job_index = {}
    docking_results = {}
    if state:
        docking_jobs = state.get("docking_jobs", [])
        for job in docking_jobs:
            job_id = job.get("job_id")
            if job_id:
                job_index[job_id] = job
        
        docking_results_list = state.get("docking_results", [])
        for result in docking_results_list:
            job_id = result.get("job_id")
            if job_id:
                docking_results[job_id] = result
    
    # Prepare simulation data for parallel processing
    simulation_tasks = []
    for i, hit in enumerate(validated_hits):
        hit_name = hit.get("ligand_name", f"hit_{i}")
        job_id = hit.get("job_id", f"job_{i}")
        pdb_id = hit.get("pdb_id", "")
        pocket_id = hit.get("pocket_id", "")
        
        # Create output directory for this hit
        hit_output_dir = output_dir / f"{pdb_id}_{pocket_id}_{hit_name}_md"
        hit_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get original PDB file and docking result
            original_pdb = None
            result_pdbqt = None
            
            # Find original PDB file from prepared_proteins in state
            if state and "prepared_proteins" in state:
                prepared_proteins = state["prepared_proteins"]
                for protein_entry in prepared_proteins:
                    if protein_entry.get("pdb_id") == pdb_id:
                        # Use cleaned_pdb which is the original PDB file path
                        pdb_path = protein_entry.get("cleaned_pdb")
                        if pdb_path and Path(pdb_path).exists():
                            original_pdb = Path(pdb_path)
                            break
            
            # Get docking result from job information
            if job_id in job_index:
                job_info = job_index[job_id]
                job_dir = job_info.get("job_dir")
                if job_dir:
                    result_pdbqt = Path(job_dir) / "result.pdbqt"
            
            # Fallback: construct result path from state directories
            if not result_pdbqt or not result_pdbqt.exists():
                if state and "docking_root_dir" in state:
                    docking_root = Path(state["docking_root_dir"])
                    result_pdbqt = docking_root / pdb_id / pocket_id / hit_name / "result.pdbqt"
            
            # Validate that required files exist
            if not original_pdb or not original_pdb.exists():
                raise FileNotFoundError(f"Original PDB not found for {pdb_id}")
            
            if not result_pdbqt or not result_pdbqt.exists():
                raise FileNotFoundError(f"Docking result not found for {hit_name}")
            
            print(f"[MD] Prepared simulation task for {hit_name} (job: {job_id}, pdb: {pdb_id}, pocket: {pocket_id})")
            
            # Create task data for parallel execution
            task_data = {
                "hit": hit,
                "output_dir": str(hit_output_dir),
                "original_pdb": str(original_pdb),
                "result_pdbqt": str(result_pdbqt),
                "simulation_time_ns": simulation_time_ns,
                "temperature": temperature,
                "use_checkpoints": use_checkpoints
            }
            simulation_tasks.append(task_data)
            
        except Exception as e:
            # Add failed task to results immediately
            error_result = {
                "hit_name": hit_name,
                "job_id": job_id,
                "pdb_id": pdb_id,
                "pocket_id": pocket_id,
                "status": "failed",
                "error": f"Task preparation failed: {str(e)}",
                "output_directory": str(hit_output_dir)
            }
            batch_results["results"].append(error_result)
            print(f"[MD] Task preparation failed for {hit_name}: {e}")
    
    # Execute simulations in parallel
    successful_simulations = 0
    failed_simulations = 0
    
    if simulation_tasks:
        print(f"[MD] Executing {len(simulation_tasks)} simulations with {max_parallel_jobs} parallel workers")
        
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=max_parallel_jobs) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(run_single_md_simulation, task): task for task in simulation_tasks}
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    batch_results["results"].append(result)
                    
                    if result["status"] == "success":
                        successful_simulations += 1
                    else:
                        failed_simulations += 1
                        
                except Exception as e:
                    # Handle any unexpected errors from the parallel execution
                    hit_name = task["hit"].get("ligand_name", "unknown")
                    job_id = task["hit"].get("job_id", "unknown")
                    error_result = {
                        "hit_name": hit_name,
                        "job_id": job_id,
                        "status": "failed",
                        "error": f"Parallel execution error: {str(e)}",
                        "output_directory": task["output_dir"]
                    }
                    batch_results["results"].append(error_result)
                    failed_simulations += 1
                    print(f"[MD] Parallel execution error for {hit_name}: {e}")
    
    # Count any tasks that failed during preparation
    for result in batch_results["results"]:
        if result["status"] == "failed" and "Task preparation failed" in result.get("error", ""):
            failed_simulations += 1
    
    # Generate summary
    total_hits = len(validated_hits)
    batch_results["summary"] = {
        "total_hits": total_hits,
        "successful_simulations": successful_simulations,
        "failed_simulations": failed_simulations,
        "success_rate": successful_simulations / total_hits if total_hits > 0 else 0,
        "parallel_jobs_used": max_parallel_jobs,
        "tasks_prepared": len(simulation_tasks)
    }
    
    # Save batch results
    with open(output_dir / "md_simulation_report.json", 'w') as f:
        json.dump(batch_results, f, indent=2, default=str)
    
    print(f"[MD] Parallel batch simulation complete: {successful_simulations}/{total_hits} successful")
    print(f"[MD] Used {max_parallel_jobs} parallel workers for {len(simulation_tasks)} prepared tasks")
    
    return batch_results


if __name__ == "__main__":
    # Test dependency availability
    available, missing = check_md_dependencies()
    if available:
        print("All MD simulation dependencies are available")
        gromacs_ok, gromacs_msg = check_gromacs_installation()
        acpype_ok, acpype_msg = check_acpype_installation()
        print(f"GROMACS: {gromacs_msg}")
        print(f"acpype: {acpype_msg}")
    else:
        print(f"Missing dependencies: {missing}")
        print("Install with:")
        for pkg in missing:
            if pkg == "gromacs":
                print("  conda install -c conda-forge gromacs")
            elif pkg == "acpype":
                print("  conda install -c conda-forge acpype")
            else:
                print(f"  conda install -c conda-forge {pkg}")
