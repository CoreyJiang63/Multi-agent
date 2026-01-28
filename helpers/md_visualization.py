#!/usr/bin/env python3
"""
Molecular Dynamics Visualization Module

Provides advanced 3D visualization capabilities for MD simulations including:
- Static molecular structure visualization
- Dynamic trajectory visualization 
- Binding site analysis
- Scientific visualization reports

Compatible with Linux server environments.
"""

import json
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import traceback
import tempfile
import os

try:
    import mdtraj as md
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.patches as patches
    from mpl_toolkits.mplot3d import Axes3D
    VISUALIZATION_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Visualization dependencies not available: {e}")
    VISUALIZATION_DEPENDENCIES_AVAILABLE = False

# Try to import PyMOL for advanced visualization
try:
    import pymol
    from pymol import cmd
    PYMOL_AVAILABLE = True
except ImportError:
    PYMOL_AVAILABLE = False

# Try to import NGLView for web-based visualization
try:
    import nglview as nv
    NGLVIEW_AVAILABLE = True
except ImportError:
    NGLVIEW_AVAILABLE = False


def create_molecular_structure_visualization(complex_pdb: Path, output_dir: Path, 
                                           hit_name: str = "unknown") -> Dict[str, Any]:
    """
    Create meaningful molecular structure visualizations showing actual chemical structures.
    
    Args:
        complex_pdb: Path to protein-ligand complex PDB file
        output_dir: Directory to save visualization files
        hit_name: Name of the hit for labeling
        
    Returns:
        Dictionary with visualization results
    """
    if not VISUALIZATION_DEPENDENCIES_AVAILABLE:
        return {"status": "skipped", "reason": "Visualization dependencies not available"}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    viz_results = {
        "status": "success",
        "hit_name": hit_name,
        "timestamp": datetime.now().isoformat(),
        "files": {},
        "analysis": {}
    }
    
    try:
        # Load structure
        traj = md.load(str(complex_pdb))
        
        # Identify protein and ligand atoms
        protein_atoms = traj.topology.select("protein")
        ligand_atoms = traj.topology.select("not protein and not water")
        
        if len(ligand_atoms) == 0:
            ligand_atoms = traj.topology.select("resname UNL")
        
        viz_results["analysis"]["n_protein_atoms"] = len(protein_atoms)
        viz_results["analysis"]["n_ligand_atoms"] = len(ligand_atoms)
        
        # Create meaningful molecular visualization
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Protein backbone trace with secondary structure coloring
        ax1 = fig.add_subplot(221, projection='3d')
        if len(protein_atoms) > 0:
            # Get CA atoms for backbone trace
            ca_atoms = traj.topology.select("protein and name CA")
            if len(ca_atoms) > 0:
                ca_coords = traj.xyz[0, ca_atoms] * 10  # Convert to Angstrom
                
                # Calculate secondary structure
                try:
                    ss = md.compute_dssp(traj)[0]
                    # Color by secondary structure
                    colors = []
                    for i, residue_ss in enumerate(ss):
                        if residue_ss == 'H':  # Alpha helix
                            colors.append('red')
                        elif residue_ss == 'E':  # Beta sheet
                            colors.append('blue')
                        else:  # Coil
                            colors.append('green')
                    
                    # Plot backbone as connected line with structure coloring
                    for i in range(len(ca_coords)-1):
                        ax1.plot([ca_coords[i,0], ca_coords[i+1,0]], 
                                [ca_coords[i,1], ca_coords[i+1,1]], 
                                [ca_coords[i,2], ca_coords[i+1,2]], 
                                color=colors[i] if i < len(colors) else 'gray', 
                                linewidth=2, alpha=0.8)
                    
                except:
                    # Fallback: simple backbone trace
                    ax1.plot(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2], 
                            'lightblue', linewidth=2, alpha=0.8, label='Protein Backbone')
        
        # Add ligand as proper ball-and-stick representation
        if len(ligand_atoms) > 0:
            ligand_coords = traj.xyz[0, ligand_atoms] * 10
            
            # Color ligand atoms by element
            ligand_colors = []
            ligand_sizes = []
            for atom_idx in ligand_atoms:
                atom = traj.topology.atom(atom_idx)
                if atom.element.symbol == 'C':
                    ligand_colors.append('darkgray')
                    ligand_sizes.append(120)
                elif atom.element.symbol == 'N':
                    ligand_colors.append('blue')
                    ligand_sizes.append(110)
                elif atom.element.symbol == 'O':
                    ligand_colors.append('red')
                    ligand_sizes.append(100)
                elif atom.element.symbol == 'S':
                    ligand_colors.append('yellow')
                    ligand_sizes.append(140)
                elif atom.element.symbol == 'P':
                    ligand_colors.append('orange')
                    ligand_sizes.append(130)
                else:
                    ligand_colors.append('purple')
                    ligand_sizes.append(80)
            
            # Plot ligand atoms with proper coloring
            ax1.scatter(ligand_coords[:, 0], ligand_coords[:, 1], ligand_coords[:, 2], 
                       c=ligand_colors, s=ligand_sizes, alpha=0.9, label='Ligand', 
                       edgecolors='black', linewidth=1.5)
            
            # Draw realistic chemical bonds based on topology
            try:
                # Use MDTraj's bond detection
                ligand_traj = traj.atom_slice(ligand_atoms)
                bonds = ligand_traj.topology.bonds
                
                for bond in bonds:
                    atom1_idx = bond.atom1.index
                    atom2_idx = bond.atom2.index
                    coord1 = ligand_coords[atom1_idx]
                    coord2 = ligand_coords[atom2_idx]
                    
                    ax1.plot([coord1[0], coord2[0]], 
                            [coord1[1], coord2[1]], 
                            [coord1[2], coord2[2]], 
                            'black', linewidth=3, alpha=0.8)
                            
            except:
                # Fallback: distance-based bonding with stricter criteria
                for i in range(len(ligand_coords)):
                    for j in range(i+1, len(ligand_coords)):
                        dist = np.linalg.norm(ligand_coords[i] - ligand_coords[j])
                        # More realistic bond length criteria
                        if 0.8 < dist < 1.8:  # Typical single bond range
                            ax1.plot([ligand_coords[i,0], ligand_coords[j,0]], 
                                    [ligand_coords[i,1], ligand_coords[j,1]], 
                                    [ligand_coords[i,2], ligand_coords[j,2]], 
                                    'black', linewidth=2.5, alpha=0.8)
        
        ax1.set_xlabel('X (Å)')
        ax1.set_ylabel('Y (Å)')
        ax1.set_zlabel('Z (Å)')
        ax1.set_title(f'Protein-Ligand Complex Structure\n{hit_name}')
        ax1.legend()
        
        # 2. Binding site detailed view
        ax2 = fig.add_subplot(222, projection='3d')
        if len(ligand_atoms) > 0 and len(protein_atoms) > 0:
            ligand_center = np.mean(ligand_coords, axis=0)
            
            # Find binding site residues (within 5 Å)
            binding_site_atoms = []
            for i, atom in enumerate(protein_atoms):
                coord = traj.xyz[0, atom] * 10
                if np.linalg.norm(coord - ligand_center) < 5.0:
                    binding_site_atoms.append(atom)
            
            if binding_site_atoms:
                binding_coords = traj.xyz[0, binding_site_atoms] * 10
                
                # Color by atom type
                atom_colors = []
                for atom_idx in binding_site_atoms:
                    atom = traj.topology.atom(atom_idx)
                    if atom.element.symbol == 'C':
                        atom_colors.append('gray')
                    elif atom.element.symbol == 'N':
                        atom_colors.append('blue')
                    elif atom.element.symbol == 'O':
                        atom_colors.append('red')
                    elif atom.element.symbol == 'S':
                        atom_colors.append('yellow')
                    else:
                        atom_colors.append('white')
                
                ax2.scatter(binding_coords[:, 0], binding_coords[:, 1], binding_coords[:, 2], 
                           c=atom_colors, s=50, alpha=0.8, label='Binding Site')
            
            # Add ligand with atom coloring
            ligand_colors = []
            for atom_idx in ligand_atoms:
                atom = traj.topology.atom(atom_idx)
                if atom.element.symbol == 'C':
                    ligand_colors.append('black')
                elif atom.element.symbol == 'N':
                    ligand_colors.append('blue')
                elif atom.element.symbol == 'O':
                    ligand_colors.append('red')
                elif atom.element.symbol == 'S':
                    ligand_colors.append('yellow')
                else:
                    ligand_colors.append('purple')
            
            ax2.scatter(ligand_coords[:, 0], ligand_coords[:, 1], ligand_coords[:, 2], 
                       c=ligand_colors, s=120, alpha=0.9, label='Ligand', 
                       edgecolors='black', linewidth=2)
            
            # Draw proper ligand bonds using topology
            try:
                ligand_traj = traj.atom_slice(ligand_atoms)
                bonds = ligand_traj.topology.bonds
                
                for bond in bonds:
                    atom1_idx = bond.atom1.index
                    atom2_idx = bond.atom2.index
                    coord1 = ligand_coords[atom1_idx]
                    coord2 = ligand_coords[atom2_idx]
                    
                    ax2.plot([coord1[0], coord2[0]], 
                            [coord1[1], coord2[1]], 
                            [coord1[2], coord2[2]], 
                            'black', linewidth=3, alpha=0.8)
            except:
                # Fallback with stricter distance criteria
                for i in range(len(ligand_coords)):
                    for j in range(i+1, len(ligand_coords)):
                        dist = np.linalg.norm(ligand_coords[i] - ligand_coords[j])
                        if 0.8 < dist < 1.8:  # Realistic bond lengths only
                            ax2.plot([ligand_coords[i,0], ligand_coords[j,0]], 
                                    [ligand_coords[i,1], ligand_coords[j,1]], 
                                    [ligand_coords[i,2], ligand_coords[j,2]], 
                                    'black', linewidth=3, alpha=0.8)
            
            ax2.set_xlabel('X (Å)')
            ax2.set_ylabel('Y (Å)')
            ax2.set_zlabel('Z (Å)')
            ax2.set_title('Binding Site Detail\n(Atoms colored by element)')
            ax2.legend()
            
            viz_results["analysis"]["binding_site_atoms"] = len(binding_site_atoms)
            viz_results["analysis"]["ligand_center"] = ligand_center.tolist()
        
        # 3. Secondary structure analysis
        ax3 = fig.add_subplot(223)
        if len(protein_atoms) > 0:
            try:
                ss = md.compute_dssp(traj)[0]
                ss_counts = {
                    'H': np.sum(ss == 'H'),  # Alpha helix
                    'E': np.sum(ss == 'E'),  # Beta sheet
                    'C': np.sum(ss == 'C'),  # Coil
                }
                
                labels = ['α-Helix\n(Red)', 'β-Sheet\n(Blue)', 'Coil/Loop\n(Green)']
                sizes = [ss_counts.get(k, 0) for k in ['H', 'E', 'C']]
                colors = ['red', 'blue', 'green']
                
                wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, 
                                                  autopct='%1.1f%%', startangle=90)
                ax3.set_title(f'Secondary Structure\n{hit_name} Target Protein')
                
                viz_results["analysis"]["secondary_structure"] = {
                    "alpha_helix_pct": (ss_counts.get('H', 0) / len(ss)) * 100,
                    "beta_sheet_pct": (ss_counts.get('E', 0) / len(ss)) * 100,
                    "coil_pct": (ss_counts.get('C', 0) / len(ss)) * 100
                }
            except:
                ax3.text(0.5, 0.5, 'Secondary Structure\nAnalysis Failed', 
                        ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Interaction distance analysis
        ax4 = fig.add_subplot(224)
        if len(ligand_atoms) > 0 and len(protein_atoms) > 0:
            # Calculate distances between ligand and nearby protein atoms
            distances = []
            for lig_idx in ligand_atoms:
                lig_coord = traj.xyz[0, lig_idx] * 10
                min_dist = float('inf')
                for prot_idx in protein_atoms:
                    prot_coord = traj.xyz[0, prot_idx] * 10
                    dist = np.linalg.norm(lig_coord - prot_coord)
                    if dist < min_dist:
                        min_dist = dist
                distances.append(min_dist)
            
            ax4.hist(distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.set_xlabel('Distance to Nearest Protein Atom (Å)')
            ax4.set_ylabel('Number of Ligand Atoms')
            ax4.set_title('Ligand-Protein Proximity\n(Closer = Stronger Interaction)')
            ax4.grid(True, alpha=0.3)
            
            # Add interpretation text
            mean_dist = np.mean(distances)
            ax4.axvline(mean_dist, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_dist:.1f} Å')
            ax4.legend()
            
            viz_results["analysis"]["mean_interaction_distance"] = float(mean_dist)
            viz_results["analysis"]["min_interaction_distance"] = float(np.min(distances))
        
        plt.tight_layout()
        
        # Save structure visualization
        structure_plot = output_dir / f"molecular_structure_{hit_name}.png"
        plt.savefig(structure_plot, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        viz_results["files"]["structure_plot"] = str(structure_plot)
        
        return viz_results
        
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Structure visualization failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


def create_trajectory_animation(trajectory_file: Path, topology_file: Path, 
                              output_dir: Path, hit_name: str = "unknown",
                              max_frames: int = 100) -> Dict[str, Any]:
    """
    Create meaningful MD trajectory visualization showing molecular dynamics.
    
    Args:
        trajectory_file: Path to trajectory file (.xtc)
        topology_file: Path to topology file (.gro)
        output_dir: Directory to save animation files
        hit_name: Name of the hit for labeling
        max_frames: Maximum number of frames to include
        
    Returns:
        Dictionary with animation results
    """
    if not VISUALIZATION_DEPENDENCIES_AVAILABLE:
        return {"status": "skipped", "reason": "Visualization dependencies not available"}
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    anim_results = {
        "status": "success",
        "hit_name": hit_name,
        "timestamp": datetime.now().isoformat(),
        "files": {},
        "analysis": {}
    }
    
    try:
        # Load trajectory
        traj = md.load(str(trajectory_file), top=str(topology_file))
        
        # Subsample trajectory if too long
        if len(traj) > max_frames:
            indices = np.linspace(0, len(traj)-1, max_frames, dtype=int)
            traj = traj[indices]
        
        # Identify atoms
        protein_atoms = traj.topology.select("protein and name CA")  # Backbone for protein
        ligand_atoms = traj.topology.select("not protein and not water")
        binding_site_atoms = []
        
        if len(ligand_atoms) == 0:
            ligand_atoms = traj.topology.select("resname UNL")
        
        # Find binding site residues (within 5Å of ligand in first frame)
        if len(ligand_atoms) > 0:
            ligand_center_initial = np.mean(traj.xyz[0, ligand_atoms] * 10, axis=0)
            for atom_idx in traj.topology.select("protein and name CA"):
                coord = traj.xyz[0, atom_idx] * 10
                if np.linalg.norm(coord - ligand_center_initial) < 8.0:
                    binding_site_atoms.append(atom_idx)
        
        anim_results["analysis"]["n_frames"] = len(traj)
        anim_results["analysis"]["n_protein_atoms"] = len(protein_atoms)
        anim_results["analysis"]["n_ligand_atoms"] = len(ligand_atoms)
        anim_results["analysis"]["n_binding_site_atoms"] = len(binding_site_atoms)
        
        # Create comprehensive MD animation showing molecular dynamics
        fig = plt.figure(figsize=(16, 12))
        
        def animate(frame):
            fig.clear()
            
            # Main 3D view - molecular dynamics
            ax1 = fig.add_subplot(221, projection='3d')
            
            # Show protein backbone as connected structure
            if len(protein_atoms) > 0:
                protein_coords = traj.xyz[frame, protein_atoms] * 10
                # Draw backbone connections
                for i in range(len(protein_coords)-1):
                    ax1.plot([protein_coords[i,0], protein_coords[i+1,0]], 
                            [protein_coords[i,1], protein_coords[i+1,1]], 
                            [protein_coords[i,2], protein_coords[i+1,2]], 
                            'lightblue', linewidth=1, alpha=0.6)
                ax1.scatter(protein_coords[:, 0], protein_coords[:, 1], protein_coords[:, 2], 
                          c='lightblue', alpha=0.7, s=20, label='Protein Backbone')
            
            # Show binding site with emphasis
            if len(binding_site_atoms) > 0:
                binding_coords = traj.xyz[frame, binding_site_atoms] * 10
                ax1.scatter(binding_coords[:, 0], binding_coords[:, 1], binding_coords[:, 2], 
                          c='orange', alpha=0.8, s=60, label='Binding Site', marker='s')
            
            # Show ligand with molecular structure
            if len(ligand_atoms) > 0:
                ligand_coords = traj.xyz[frame, ligand_atoms] * 10
                
                # Color ligand atoms by element
                ligand_colors = []
                for atom_idx in ligand_atoms:
                    atom = traj.topology.atom(atom_idx)
                    if atom.element.symbol == 'C':
                        ligand_colors.append('darkgray')
                    elif atom.element.symbol == 'N':
                        ligand_colors.append('blue')
                    elif atom.element.symbol == 'O':
                        ligand_colors.append('red')
                    elif atom.element.symbol == 'S':
                        ligand_colors.append('yellow')
                    else:
                        ligand_colors.append('purple')
                
                ax1.scatter(ligand_coords[:, 0], ligand_coords[:, 1], ligand_coords[:, 2], 
                          c=ligand_colors, alpha=1.0, s=100, label='Ligand', 
                          edgecolors='black', linewidth=1.5)
                
                # Draw ligand bonds
                try:
                    ligand_traj = traj.atom_slice(ligand_atoms)
                    bonds = ligand_traj.topology.bonds
                    for bond in bonds:
                        atom1_idx = bond.atom1.index
                        atom2_idx = bond.atom2.index
                        coord1 = ligand_coords[atom1_idx]
                        coord2 = ligand_coords[atom2_idx]
                        ax1.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], 
                                [coord1[2], coord2[2]], 'black', linewidth=2, alpha=0.8)
                except:
                    # Fallback bonding
                    for i in range(len(ligand_coords)):
                        for j in range(i+1, len(ligand_coords)):
                            dist = np.linalg.norm(ligand_coords[i] - ligand_coords[j])
                            if 0.8 < dist < 1.8:
                                ax1.plot([ligand_coords[i,0], ligand_coords[j,0]], 
                                        [ligand_coords[i,1], ligand_coords[j,1]], 
                                        [ligand_coords[i,2], ligand_coords[j,2]], 
                                        'black', linewidth=2, alpha=0.8)
            
            ax1.set_xlabel('X (Å)')
            ax1.set_ylabel('Y (Å)')
            ax1.set_zlabel('Z (Å)')
            ax1.set_title(f'MD Simulation: {hit_name}\nFrame {frame+1}/{len(traj)} ({traj.time[frame]:.1f} ps)')
            ax1.legend()
            
            # Set consistent view centered on ligand
            if len(ligand_atoms) > 0:
                ligand_center = np.mean(ligand_coords, axis=0)
                range_val = 15
                ax1.set_xlim(ligand_center[0]-range_val, ligand_center[0]+range_val)
                ax1.set_ylim(ligand_center[1]-range_val, ligand_center[1]+range_val)
                ax1.set_zlim(ligand_center[2]-range_val, ligand_center[2]+range_val)
            
            # Ligand movement trace (show last 10 frames)
            ax2 = fig.add_subplot(222, projection='3d')
            if len(ligand_atoms) > 0 and frame > 0:
                start_frame = max(0, frame-10)
                trace_coords = []
                for f in range(start_frame, frame+1):
                    center = np.mean(traj.xyz[f, ligand_atoms] * 10, axis=0)
                    trace_coords.append(center)
                trace_coords = np.array(trace_coords)
                
                # Plot movement trace
                ax2.plot(trace_coords[:, 0], trace_coords[:, 1], trace_coords[:, 2], 
                        'red', linewidth=3, alpha=0.8, label='Ligand Path')
                ax2.scatter(trace_coords[-1, 0], trace_coords[-1, 1], trace_coords[-1, 2], 
                          c='red', s=100, marker='o', label='Current Position')
                
                # Show binding site for reference
                if len(binding_site_atoms) > 0:
                    binding_coords = traj.xyz[frame, binding_site_atoms] * 10
                    ax2.scatter(binding_coords[:, 0], binding_coords[:, 1], binding_coords[:, 2], 
                              c='orange', alpha=0.6, s=40, label='Binding Site')
                
                ax2.set_xlabel('X (Å)')
                ax2.set_ylabel('Y (Å)')
                ax2.set_zlabel('Z (Å)')
                ax2.set_title('Ligand Movement Trace\n(Last 10 frames)')
                ax2.legend()
            
            # Distance analysis over time
            ax3 = fig.add_subplot(223)
            if len(ligand_atoms) > 0 and len(binding_site_atoms) > 0:
                distances = []
                times = []
                for f in range(frame+1):
                    lig_center = np.mean(traj.xyz[f, ligand_atoms] * 10, axis=0)
                    bind_center = np.mean(traj.xyz[f, binding_site_atoms] * 10, axis=0)
                    dist = np.linalg.norm(lig_center - bind_center)
                    distances.append(dist)
                    times.append(traj.time[f])
                
                ax3.plot(times, distances, 'blue', linewidth=2, label='Ligand-Protein Distance')
                ax3.axhline(y=np.mean(distances), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(distances):.1f} Å')
                ax3.set_xlabel('Time (ps)')
                ax3.set_ylabel('Distance (Å)')
                ax3.set_title('Binding Stability Analysis')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # RMSD analysis
            ax4 = fig.add_subplot(224)
            if len(ligand_atoms) > 0:
                rmsds = []
                times = []
                ref_coords = traj.xyz[0, ligand_atoms] * 10
                for f in range(frame+1):
                    current_coords = traj.xyz[f, ligand_atoms] * 10
                    rmsd = np.sqrt(np.mean(np.sum((current_coords - ref_coords)**2, axis=1)))
                    rmsds.append(rmsd)
                    times.append(traj.time[f])
                
                ax4.plot(times, rmsds, 'green', linewidth=2, label='Ligand RMSD')
                ax4.axhline(y=np.mean(rmsds), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(rmsds):.1f} Å')
                ax4.set_xlabel('Time (ps)')
                ax4.set_ylabel('RMSD (Å)')
                ax4.set_title('Conformational Stability')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(traj), interval=300, repeat=True)
        
        # Save as GIF
        gif_file = output_dir / f"md_dynamics_{hit_name}.gif"
        writer = PillowWriter(fps=3)  # Slower for better viewing
        anim.save(gif_file, writer=writer)
        plt.close()
        
        anim_results["files"]["trajectory_gif"] = str(gif_file)
        
        # Create comprehensive trajectory analysis plot
        if len(ligand_atoms) > 0:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. 3D trajectory path with binding site
            ligand_centers = np.mean(traj.xyz[:, ligand_atoms], axis=1) * 10
            
            ax1 = fig.add_subplot(221, projection='3d')
            ax1.plot(ligand_centers[:, 0], ligand_centers[:, 1], ligand_centers[:, 2], 
                    'red', linewidth=2, alpha=0.8, label='Ligand Trajectory')
            ax1.scatter(ligand_centers[0, 0], ligand_centers[0, 1], ligand_centers[0, 2], 
                      c='green', s=150, marker='o', label='Start', zorder=5)
            ax1.scatter(ligand_centers[-1, 0], ligand_centers[-1, 1], ligand_centers[-1, 2], 
                      c='red', s=150, marker='s', label='End', zorder=5)
            
            # Add binding site reference
            if len(binding_site_atoms) > 0:
                binding_center = np.mean(traj.xyz[0, binding_site_atoms] * 10, axis=0)
                ax1.scatter(binding_center[0], binding_center[1], binding_center[2], 
                          c='orange', s=200, marker='*', label='Binding Site', zorder=5)
            
            ax1.set_xlabel('X (Å)')
            ax1.set_ylabel('Y (Å)')
            ax1.set_zlabel('Z (Å)')
            ax1.set_title(f'3D Ligand Trajectory - {hit_name}')
            ax1.legend()
            
            # 2. Distance from binding site over time
            if len(binding_site_atoms) > 0:
                distances = []
                for frame in range(len(traj)):
                    lig_center = ligand_centers[frame]
                    bind_center = np.mean(traj.xyz[frame, binding_site_atoms] * 10, axis=0)
                    dist = np.linalg.norm(lig_center - bind_center)
                    distances.append(dist)
                
                ax2.plot(traj.time, distances, 'blue', linewidth=2, label='Distance to Binding Site')
                ax2.axhline(y=np.mean(distances), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(distances):.1f} Å')
                ax2.fill_between(traj.time, distances, alpha=0.3, color='blue')
                ax2.set_xlabel('Time (ps)')
                ax2.set_ylabel('Distance (Å)')
                ax2.set_title('Binding Site Proximity')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # 3. RMSD analysis
            rmsds = []
            ref_coords = traj.xyz[0, ligand_atoms] * 10
            for frame in range(len(traj)):
                current_coords = traj.xyz[frame, ligand_atoms] * 10
                rmsd = np.sqrt(np.mean(np.sum((current_coords - ref_coords)**2, axis=1)))
                rmsds.append(rmsd)
            
            ax3.plot(traj.time, rmsds, 'green', linewidth=2, label='Ligand RMSD')
            ax3.axhline(y=np.mean(rmsds), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(rmsds):.1f} Å')
            ax3.fill_between(traj.time, rmsds, alpha=0.3, color='green')
            ax3.set_xlabel('Time (ps)')
            ax3.set_ylabel('RMSD (Å)')
            ax3.set_title('Conformational Stability')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Movement velocity analysis
            velocities = []
            for frame in range(1, len(traj)):
                displacement = np.linalg.norm(ligand_centers[frame] - ligand_centers[frame-1])
                time_diff = traj.time[frame] - traj.time[frame-1]
                velocity = displacement / time_diff if time_diff > 0 else 0
                velocities.append(velocity)
            
            ax4.plot(traj.time[1:], velocities, 'purple', linewidth=2, label='Movement Velocity')
            ax4.axhline(y=np.mean(velocities), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(velocities):.2f} Å/ps')
            ax4.fill_between(traj.time[1:], velocities, alpha=0.3, color='purple')
            ax4.set_xlabel('Time (ps)')
            ax4.set_ylabel('Velocity (Å/ps)')
            ax4.set_title('Ligand Movement Dynamics')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            path_plot = output_dir / f"md_trajectory_analysis_{hit_name}.png"
            plt.savefig(path_plot, dpi=300, bbox_inches='tight')
            plt.close()
            
            anim_results["files"]["trajectory_path"] = str(path_plot)
            
            # Calculate comprehensive trajectory statistics
            distances = np.linalg.norm(np.diff(ligand_centers, axis=0), axis=1)
            anim_results["analysis"]["total_distance_traveled"] = float(np.sum(distances))
            anim_results["analysis"]["max_displacement"] = float(np.max(distances))
            anim_results["analysis"]["mean_displacement"] = float(np.mean(distances))
            anim_results["analysis"]["mean_rmsd"] = float(np.mean(rmsds))
            anim_results["analysis"]["max_rmsd"] = float(np.max(rmsds))
            anim_results["analysis"]["mean_velocity"] = float(np.mean(velocities))
            anim_results["analysis"]["binding_stability"] = "stable" if np.mean(rmsds) < 2.0 else "dynamic"
        
        return anim_results
        
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Trajectory animation failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


def create_pymol_visualization_script(complex_pdb: Path, output_dir: Path, 
                                    hit_name: str = "unknown") -> Dict[str, Any]:
    """
    Create PyMOL visualization script and automatically generate high-quality images.
    
    Args:
        complex_pdb: Path to protein-ligand complex PDB file
        output_dir: Directory to save PyMOL script and images
        hit_name: Name of the hit for labeling
        
    Returns:
        Dictionary with PyMOL script results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pymol_results = {
        "status": "success",
        "hit_name": hit_name,
        "timestamp": datetime.now().isoformat(),
        "files": {},
        "instructions": []
    }
    
    try:
        # Create proper PyMOL script (not .py file)
        script_content = f'''# PyMOL Visualization Script for {hit_name}
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# Load structure
load {complex_pdb}, complex

# Basic setup for high-quality rendering
bg_color white
set ray_opaque_background, off
set ray_shadows, on
set ray_shadow_decay_factor, 0.1
set ray_shadow_decay_range, 2
set antialias, 2
set hash_max, 300

# Remove water and other solvents for clarity
remove solvent
remove inorganic

# Protein representation - cartoon with secondary structure coloring
select protein, polymer
show cartoon, protein
cartoon automatic
set cartoon_fancy_helices, 1
set cartoon_fancy_sheets, 1
spectrum count, rainbow, protein
set cartoon_transparency, 0.1, protein

# Ligand representation - ball and stick with element coloring
select ligand, not polymer and not solvent and not inorganic
show sticks, ligand
show spheres, ligand
set sphere_scale, 0.3, ligand
set stick_radius, 0.2, ligand
util.cbag ligand  # Color by atom: carbon=gray, nitrogen=blue, oxygen=red

# Binding site representation
select binding_site, (polymer within 4 of ligand)
show sticks, binding_site
set stick_radius, 0.15, binding_site
util.cbag binding_site

# Add hydrogen bonds
distance hbonds, ligand, binding_site, 3.5, mode=2
hide labels, hbonds
set dash_color, yellow, hbonds
set dash_width, 3, hbonds

# Center and orient view
center ligand
orient ligand

# View 1: Overall complex structure
zoom complex, 5
png {output_dir}/pymol_overview_{hit_name}.png, width=1200, height=900, dpi=300, ray=1

# View 2: Binding site close-up with interactions
zoom ligand, 8
png {output_dir}/pymol_binding_site_{hit_name}.png, width=1200, height=900, dpi=300, ray=1

# View 3: Surface representation showing binding pocket
hide everything
show surface, protein
set surface_color, lightblue, protein
set transparency, 0.3, protein
show sticks, ligand
show sticks, binding_site
show spheres, ligand
set sphere_scale, 0.25, ligand
zoom ligand, 10
png {output_dir}/pymol_surface_{hit_name}.png, width=1200, height=900, dpi=300, ray=1

# View 4: Electrostatic surface (if APBS available)
hide everything
show surface, protein
ramp_new e_lvl, protein, [0, 5, 10], [red, white, blue]
set surface_color, e_lvl, protein
show sticks, ligand
util.cbag ligand
zoom ligand, 12
png {output_dir}/pymol_electrostatic_{hit_name}.png, width=1200, height=900, dpi=300, ray=1

# Save session
save {output_dir}/pymol_session_{hit_name}.pse

print "PyMOL visualization complete for {hit_name}"
quit
'''
        
        # Save PyMOL script with proper extension
        script_file = output_dir / f"pymol_script_{hit_name}.pml"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        pymol_results["files"]["pymol_script"] = str(script_file)
        
        # Try to run PyMOL automatically using command line
        try:
            # Check if PyMOL is available in command line
            result = subprocess.run(['pymol', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Run PyMOL script automatically
                print(f"[PYMOL] Executing PyMOL script for {hit_name}")
                cmd_result = subprocess.run([
                    'pymol', '-c',  # Command line mode
                    str(script_file)
                ], capture_output=True, text=True, timeout=300)
                
                if cmd_result.returncode == 0:
                    pymol_results["status"] = "executed"
                    pymol_results["files"]["pymol_images"] = [
                        str(output_dir / f"pymol_overview_{hit_name}.png"),
                        str(output_dir / f"pymol_binding_site_{hit_name}.png"),
                        str(output_dir / f"pymol_surface_{hit_name}.png"),
                        str(output_dir / f"pymol_electrostatic_{hit_name}.png")
                    ]
                    pymol_results["files"]["pymol_session"] = str(output_dir / f"pymol_session_{hit_name}.pse")
                    print(f"[PYMOL] Successfully generated images for {hit_name}")
                else:
                    pymol_results["status"] = "script_created"
                    pymol_results["execution_error"] = cmd_result.stderr
                    print(f"[PYMOL] Script execution failed: {cmd_result.stderr}")
            else:
                pymol_results["status"] = "script_created"
                print(f"[PYMOL] PyMOL not available in command line")
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            pymol_results["status"] = "script_created"
            pymol_results["execution_error"] = str(e)
            print(f"[PYMOL] Could not execute PyMOL automatically: {e}")
        
        # Add instructions for manual execution
        pymol_results["instructions"] = [
            f"To run PyMOL visualization manually:",
            f"1. Install PyMOL: conda install -c conda-forge pymol-open-source",
            f"2. Run command line: pymol -c {script_file}",
            f"3. Or run in PyMOL GUI: File > Run Script > {script_file}",
            f"4. Images will be saved to {output_dir}/",
            f"5. Script includes: overview, binding site, surface, and electrostatic views"
        ]
        
        return pymol_results
        
    except Exception as e:
        return {
            "status": "failed",
            "error": f"PyMOL script creation failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


def create_comprehensive_md_visualization_report(simulation_dir: Path, output_dir: Path,
                                               hit_name: str = "unknown") -> Dict[str, Any]:
    """
    Create comprehensive visualization report for MD simulation.
    
    Args:
        simulation_dir: Directory containing MD simulation files
        output_dir: Directory to save visualization files and report
        hit_name: Name of the hit for labeling
        
    Returns:
        Dictionary with comprehensive visualization results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "status": "success",
        "hit_name": hit_name,
        "timestamp": datetime.now().isoformat(),
        "visualization_summary": {},
        "files_generated": {},
        "scientific_analysis": {},
        "instructions": []
    }
    
    try:
        # Find required files - check actual file structure
        md_dir = simulation_dir.parent
        complex_pdb = md_dir / "complex.pdb"  # Found at root level
        trajectory_file = simulation_dir / "md.xtc"
        topology_file = simulation_dir / "md.gro"
        
        # Alternative locations for complex PDB
        if not complex_pdb.exists():
            complex_pdb = simulation_dir / "complex" / "complex.pdb"
        
        # 1. Create molecular structure visualization
        if complex_pdb.exists():
            struct_results = create_molecular_structure_visualization(
                complex_pdb, output_dir, hit_name
            )
            report["visualization_summary"]["structure"] = struct_results
            if struct_results["status"] == "success":
                report["files_generated"].update(struct_results["files"])
                report["scientific_analysis"]["structure"] = struct_results["analysis"]
        
        # 2. Create trajectory animation
        if trajectory_file.exists() and topology_file.exists():
            anim_results = create_trajectory_animation(
                trajectory_file, topology_file, output_dir, hit_name
            )
            report["visualization_summary"]["trajectory"] = anim_results
            if anim_results["status"] == "success":
                report["files_generated"].update(anim_results["files"])
                report["scientific_analysis"]["trajectory"] = anim_results["analysis"]
        
        # 3. Create PyMOL script
        if complex_pdb.exists():
            pymol_results = create_pymol_visualization_script(
                complex_pdb, output_dir, hit_name
            )
            report["visualization_summary"]["pymol"] = pymol_results
            if pymol_results["status"] in ["success", "script_created", "executed"]:
                report["files_generated"].update(pymol_results["files"])
                report["instructions"].extend(pymol_results["instructions"])
        
        # 4. Create PyMOL MD Animation
        if trajectory_file.exists() and topology_file.exists():
            try:
                from .pymol_md_animation import create_pymol_md_visualization
                
                pymol_md_results = create_pymol_md_visualization(
                    simulation_dir, output_dir, hit_name
                )
                report["visualization_summary"]["pymol_md"] = pymol_md_results
                if pymol_md_results["status"] in ["success", "completed", "partial", "script_created"]:
                    report["files_generated"].update(pymol_md_results["files"])
                    if "instructions" in pymol_md_results:
                        report["instructions"].extend(pymol_md_results["instructions"])
                        
            except ImportError as e:
                print(f"PyMOL MD animation module not available: {e}")
            except Exception as e:
                print(f"PyMOL MD animation failed: {e}")
        
        # 4. Generate scientific summary
        report["scientific_analysis"]["summary"] = {
            "visualization_types": list(report["visualization_summary"].keys()),
            "total_files_generated": len(report["files_generated"]),
            "requires_manual_steps": len(report["instructions"]) > 0
        }
        
        # 5. Save comprehensive report
        report_file = output_dir / f"md_visualization_report_{hit_name}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        report["files_generated"]["visualization_report"] = str(report_file)
        
        # 6. Create README with instructions
        readme_content = f"""# MD Visualization Report for {hit_name}

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Files Generated

### Static Visualizations
"""
        
        if "structure_plot" in report["files_generated"]:
            readme_content += f"- **Molecular Structure**: `{Path(report['files_generated']['structure_plot']).name}`\n"
        
        if "trajectory_path" in report["files_generated"]:
            readme_content += f"- **Trajectory Path**: `{Path(report['files_generated']['trajectory_path']).name}`\n"
        
        readme_content += "\n### Dynamic Visualizations\n"
        
        if "trajectory_gif" in report["files_generated"]:
            readme_content += f"- **Trajectory Animation**: `{Path(report['files_generated']['trajectory_gif']).name}`\n"
        
        readme_content += "\n### PyMOL Visualizations\n"
        
        if "pymol_script" in report["files_generated"]:
            readme_content += f"- **PyMOL Script**: `{Path(report['files_generated']['pymol_script']).name}`\n"
            readme_content += "\n#### To run PyMOL visualization:\n"
            for instruction in report["instructions"]:
                readme_content += f"   {instruction}\n"
        
        readme_content += f"\n## Scientific Analysis\n\n"
        
        if "structure" in report["scientific_analysis"]:
            struct_analysis = report["scientific_analysis"]["structure"]
            readme_content += f"- **Protein atoms**: {struct_analysis.get('n_protein_atoms', 'N/A')}\n"
            readme_content += f"- **Ligand atoms**: {struct_analysis.get('n_ligand_atoms', 'N/A')}\n"
            readme_content += f"- **Binding site atoms**: {struct_analysis.get('binding_site_atoms', 'N/A')}\n"
        
        if "trajectory" in report["scientific_analysis"]:
            traj_analysis = report["scientific_analysis"]["trajectory"]
            readme_content += f"- **Trajectory frames**: {traj_analysis.get('n_frames', 'N/A')}\n"
            readme_content += f"- **Total distance traveled**: {traj_analysis.get('total_distance_traveled', 'N/A'):.2f} Å\n"
            readme_content += f"- **Mean displacement**: {traj_analysis.get('mean_displacement', 'N/A'):.2f} Å\n"
        
        readme_content += f"\n## Usage Notes\n\n"
        readme_content += f"- All static images are high-resolution PNG files (300 DPI)\n"
        readme_content += f"- GIF animations can be viewed in any web browser\n"
        readme_content += f"- PyMOL scripts require PyMOL installation for execution\n"
        readme_content += f"- All files are ready for publication or presentation use\n"
        
        readme_file = output_dir / f"README_{hit_name}.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        report["files_generated"]["readme"] = str(readme_file)
        
        return report
        
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Comprehensive visualization failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


# Integration function to be called from md_simulation.py
def add_md_visualization_to_analysis(simulation_dir: Path, output_dir: Path, 
                                   hit_name: str = "unknown") -> Dict[str, Any]:
    """
    Add comprehensive visualization to existing MD analysis.
    This function can be called from md_simulation.py without modifying existing code.
    
    Args:
        simulation_dir: Directory containing MD simulation files
        output_dir: Directory to save visualization files
        hit_name: Name of the hit for labeling
        
    Returns:
        Dictionary with visualization results that can be integrated into existing reports
    """
    try:
        # Create visualization subdirectory
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate comprehensive visualization report
        viz_report = create_comprehensive_md_visualization_report(
            simulation_dir, viz_dir, hit_name
        )
        
        # Return results in format compatible with existing MD analysis
        return {
            "md_visualization": {
                "status": viz_report["status"],
                "visualization_report": viz_report.get("files_generated", {}).get("visualization_report"),
                "files": viz_report.get("files_generated", {}),
                "analysis": viz_report.get("scientific_analysis", {}),
                "instructions": viz_report.get("instructions", [])
            }
        }
        
    except Exception as e:
        return {
            "md_visualization": {
                "status": "failed",
                "error": f"MD visualization integration failed: {str(e)}"
            }
        }
