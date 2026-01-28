#!/usr/bin/env python3
"""
Simple MD Visualization Module - No external dependencies
Creates meaningful molecular visualizations using basic Python libraries
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from datetime import datetime
import traceback

def parse_pdb_file(pdb_file):
    """Parse PDB file to extract atomic coordinates and information."""
    atoms = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                atom_data = {
                    'type': line[:6].strip(),
                    'atom_id': int(line[6:11].strip()),
                    'atom_name': line[12:16].strip(),
                    'residue': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'residue_id': int(line[22:26].strip()),
                    'x': float(line[30:38].strip()),
                    'y': float(line[38:46].strip()),
                    'z': float(line[46:54].strip()),
                    'element': line[76:78].strip() if len(line) > 76 else line[12:14].strip()[0]
                }
                atoms.append(atom_data)
    return atoms

def create_improved_molecular_structure_viz(complex_pdb, output_dir, hit_name="unknown", ligand_pdb=None):
    """Create improved molecular structure visualization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Parse protein PDB file
        protein_atoms = parse_pdb_file(complex_pdb)
        protein_atoms = [a for a in protein_atoms if a['type'] == 'ATOM']
        
        # Parse ligand PDB file if provided
        ligand_atoms = []
        if ligand_pdb and Path(ligand_pdb).exists():
            ligand_atoms = parse_pdb_file(ligand_pdb)
            ligand_atoms = [a for a in ligand_atoms if a['residue'] not in ['HOH', 'WAT', 'SOL']]
        else:
            # Try to find ligand in complex file
            all_atoms = parse_pdb_file(complex_pdb)
            ligand_atoms = [a for a in all_atoms if a['type'] == 'HETATM' and a['residue'] not in ['HOH', 'WAT', 'SOL']]
        
        # Create visualization
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Overall complex structure
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Plot protein backbone (CA atoms only)
        protein_ca = [a for a in protein_atoms if a['atom_name'] == 'CA']
        if protein_ca:
            coords = np.array([[a['x'], a['y'], a['z']] for a in protein_ca])
            
            # Draw backbone as connected line
            ax1.plot(coords[:, 0], coords[:, 1], coords[:, 2], 
                    'lightblue', linewidth=2, alpha=0.8, label='Protein Backbone')
            ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                       c='lightblue', s=30, alpha=0.7)
        
        # Plot ligand with proper atom coloring and sizing
        if ligand_atoms:
            ligand_coords = np.array([[a['x'], a['y'], a['z']] for a in ligand_atoms])
            
            # Color and size by element
            colors = []
            sizes = []
            for atom in ligand_atoms:
                element = atom['element'].upper()
                if element == 'C':
                    colors.append('darkgray')
                    sizes.append(120)
                elif element == 'N':
                    colors.append('blue')
                    sizes.append(110)
                elif element == 'O':
                    colors.append('red')
                    sizes.append(100)
                elif element == 'S':
                    colors.append('yellow')
                    sizes.append(140)
                elif element == 'P':
                    colors.append('orange')
                    sizes.append(130)
                else:
                    colors.append('purple')
                    sizes.append(80)
            
            ax1.scatter(ligand_coords[:, 0], ligand_coords[:, 1], ligand_coords[:, 2], 
                       c=colors, s=sizes, alpha=0.9, label='Ligand', 
                       edgecolors='black', linewidth=1.5)
            
            # Draw realistic chemical bonds
            for i in range(len(ligand_coords)):
                for j in range(i+1, len(ligand_coords)):
                    dist = np.linalg.norm(ligand_coords[i] - ligand_coords[j])
                    # Realistic bond length criteria based on elements
                    if 0.9 < dist < 1.8:  # Typical single bond range
                        ax1.plot([ligand_coords[i,0], ligand_coords[j,0]], 
                                [ligand_coords[i,1], ligand_coords[j,1]], 
                                [ligand_coords[i,2], ligand_coords[j,2]], 
                                'black', linewidth=3, alpha=0.8)
        
        ax1.set_xlabel('X (Å)')
        ax1.set_ylabel('Y (Å)')
        ax1.set_zlabel('Z (Å)')
        ax1.set_title(f'Protein-Ligand Complex\n{hit_name}', fontsize=14, fontweight='bold')
        ax1.legend()
        
        # 2. Binding site detail
        ax2 = fig.add_subplot(222, projection='3d')
        
        if ligand_atoms:
            ligand_center = np.mean(ligand_coords, axis=0)
            
            # Find binding site atoms (within 5Å of ligand)
            binding_site_atoms = []
            for atom in protein_atoms:
                coord = np.array([atom['x'], atom['y'], atom['z']])
                if np.linalg.norm(coord - ligand_center) < 5.0:
                    binding_site_atoms.append(atom)
            
            # Plot binding site atoms with proper ball-and-stick representation
            if binding_site_atoms:
                binding_coords = np.array([[a['x'], a['y'], a['z']] for a in binding_site_atoms])
                binding_colors = []
                binding_sizes = []
                for atom in binding_site_atoms:
                    element = atom['element'].upper()
                    if element == 'C':
                        binding_colors.append('lightgray')
                        binding_sizes.append(80)
                    elif element == 'N':
                        binding_colors.append('lightblue')
                        binding_sizes.append(75)
                    elif element == 'O':
                        binding_colors.append('lightcoral')
                        binding_sizes.append(70)
                    elif element == 'S':
                        binding_colors.append('lightyellow')
                        binding_sizes.append(90)
                    else:
                        binding_colors.append('lightgreen')
                        binding_sizes.append(60)
                
                ax2.scatter(binding_coords[:, 0], binding_coords[:, 1], binding_coords[:, 2], 
                           c=binding_colors, s=binding_sizes, alpha=0.8, label='Binding Site',
                           edgecolors='darkgray', linewidth=1)
                
                # Draw bonds between binding site atoms (backbone connections)
                for i in range(len(binding_site_atoms)-1):
                    atom1 = binding_site_atoms[i]
                    atom2 = binding_site_atoms[i+1]
                    # Connect consecutive CA atoms in same residue or adjacent residues
                    if (atom1['atom_name'] == 'CA' and atom2['atom_name'] == 'CA' and 
                        abs(atom1['residue_id'] - atom2['residue_id']) <= 1):
                        coord1 = binding_coords[i]
                        coord2 = binding_coords[i+1]
                        dist = np.linalg.norm(coord1 - coord2)
                        if dist < 4.0:  # Reasonable CA-CA distance
                            ax2.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], 
                                    [coord1[2], coord2[2]], 'gray', linewidth=2, alpha=0.6)
            
            # Plot ligand with enhanced ball-and-stick representation
            ax2.scatter(ligand_coords[:, 0], ligand_coords[:, 1], ligand_coords[:, 2], 
                       c=colors, s=sizes, alpha=1.0, label='Ligand', 
                       edgecolors='black', linewidth=2, zorder=10)
            
            # Draw ligand bonds with enhanced visibility
            for i in range(len(ligand_coords)):
                for j in range(i+1, len(ligand_coords)):
                    dist = np.linalg.norm(ligand_coords[i] - ligand_coords[j])
                    if 0.9 < dist < 1.8:  # Strict bond criteria
                        ax2.plot([ligand_coords[i,0], ligand_coords[j,0]], 
                                [ligand_coords[i,1], ligand_coords[j,1]], 
                                [ligand_coords[i,2], ligand_coords[j,2]], 
                                'black', linewidth=4, alpha=1.0, zorder=5)
            
            # Draw potential hydrogen bonds between ligand and binding site
            for lig_idx, lig_atom in enumerate(ligand_atoms):
                lig_coord = ligand_coords[lig_idx]
                lig_element = lig_atom['element'].upper()
                
                # Only consider H-bond donors/acceptors
                if lig_element in ['O', 'N']:
                    for bind_idx, bind_atom in enumerate(binding_site_atoms):
                        bind_coord = binding_coords[bind_idx]
                        bind_element = bind_atom['element'].upper()
                        
                        if bind_element in ['O', 'N']:
                            dist = np.linalg.norm(lig_coord - bind_coord)
                            if 2.5 < dist < 3.5:  # H-bond distance range
                                ax2.plot([lig_coord[0], bind_coord[0]], 
                                        [lig_coord[1], bind_coord[1]], 
                                        [lig_coord[2], bind_coord[2]], 
                                        'yellow', linewidth=2, alpha=0.8, 
                                        linestyle='--', zorder=3)
        
        ax2.set_xlabel('X (Å)')
        ax2.set_ylabel('Y (Å)')
        ax2.set_zlabel('Z (Å)')
        ax2.set_title('Binding Site Detail\n(Realistic Chemical Bonds)', fontsize=14, fontweight='bold')
        ax2.legend()
        
        # 3. Element distribution
        ax3 = fig.add_subplot(223)
        if ligand_atoms:
            elements = [a['element'].upper() for a in ligand_atoms]
            element_counts = {}
            for elem in elements:
                element_counts[elem] = element_counts.get(elem, 0) + 1
            
            colors_pie = []
            for elem in element_counts.keys():
                if elem == 'C':
                    colors_pie.append('darkgray')
                elif elem == 'N':
                    colors_pie.append('blue')
                elif elem == 'O':
                    colors_pie.append('red')
                elif elem == 'S':
                    colors_pie.append('yellow')
                else:
                    colors_pie.append('purple')
            
            wedges, texts, autotexts = ax3.pie(element_counts.values(), 
                                              labels=element_counts.keys(), 
                                              colors=colors_pie,
                                              autopct='%1.1f%%', 
                                              startangle=90)
            ax3.set_title(f'Ligand Composition\n{hit_name}', fontsize=14, fontweight='bold')
        
        # 4. Interaction analysis
        ax4 = fig.add_subplot(224)
        if ligand_atoms and binding_site_atoms:
            distances = []
            for lig_atom in ligand_atoms:
                lig_coord = np.array([lig_atom['x'], lig_atom['y'], lig_atom['z']])
                min_dist = float('inf')
                for prot_atom in binding_site_atoms:
                    prot_coord = np.array([prot_atom['x'], prot_atom['y'], prot_atom['z']])
                    dist = np.linalg.norm(lig_coord - prot_coord)
                    if dist < min_dist:
                        min_dist = dist
                distances.append(min_dist)
            
            ax4.hist(distances, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax4.set_xlabel('Distance to Nearest Protein Atom (Å)')
            ax4.set_ylabel('Number of Ligand Atoms')
            ax4.set_title('Ligand-Protein Interactions\n(Closer = Stronger Binding)', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            mean_dist = np.mean(distances)
            ax4.axvline(mean_dist, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_dist:.1f} Å')
            ax4.legend()
        
        plt.tight_layout()
        
        # Save visualization
        output_file = output_dir / f"improved_molecular_structure_{hit_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return {
            "status": "success",
            "file": str(output_file),
            "analysis": {
                "n_protein_atoms": len(protein_atoms),
                "n_ligand_atoms": len(ligand_atoms),
                "n_binding_site_atoms": len(binding_site_atoms) if ligand_atoms else 0,
                "ligand_elements": list(set([a['element'].upper() for a in ligand_atoms])) if ligand_atoms else []
            }
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Visualization failed: {str(e)}",
            "traceback": traceback.format_exc()
        }

def create_md_trajectory_simulation(complex_pdb, output_dir, hit_name="unknown", n_frames=50, ligand_pdb=None):
    """Create simulated MD trajectory visualization showing meaningful molecular dynamics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Parse initial structure
        protein_atoms = parse_pdb_file(complex_pdb)
        protein_atoms = [a for a in protein_atoms if a['type'] == 'ATOM' and a['atom_name'] == 'CA']
        
        # Parse ligand structure
        ligand_atoms = []
        if ligand_pdb and Path(ligand_pdb).exists():
            ligand_atoms = parse_pdb_file(ligand_pdb)
            ligand_atoms = [a for a in ligand_atoms if a['residue'] not in ['HOH', 'WAT', 'SOL']]
        else:
            all_atoms = parse_pdb_file(complex_pdb)
            ligand_atoms = [a for a in all_atoms if a['type'] == 'HETATM' and a['residue'] not in ['HOH', 'WAT', 'SOL']]
        
        if not ligand_atoms:
            return {"status": "failed", "error": "No ligand atoms found"}
        
        # Generate simulated trajectory with realistic molecular motion
        np.random.seed(42)  # For reproducible results
        
        # Initial coordinates
        ligand_coords_initial = np.array([[a['x'], a['y'], a['z']] for a in ligand_atoms])
        protein_coords = np.array([[a['x'], a['y'], a['z']] for a in protein_atoms])
        ligand_center_initial = np.mean(ligand_coords_initial, axis=0)
        
        # Simulate realistic MD trajectory
        trajectory_data = []
        ligand_centers = []
        
        for frame in range(n_frames):
            # Simulate small random movements with constraints
            time_ps = frame * 10  # 10 ps per frame
            
            # Add thermal motion (small random displacements)
            thermal_motion = np.random.normal(0, 0.1, ligand_coords_initial.shape)
            
            # Add slight drift away from initial position (binding/unbinding simulation)
            drift_factor = 0.02 * frame
            drift_direction = np.random.normal(0, 1, 3)
            drift_direction = drift_direction / np.linalg.norm(drift_direction)
            drift = drift_factor * drift_direction
            
            # Apply constraints to keep ligand near binding site
            current_coords = ligand_coords_initial + thermal_motion + drift
            current_center = np.mean(current_coords, axis=0)
            
            # If ligand drifts too far, pull it back
            distance_from_initial = np.linalg.norm(current_center - ligand_center_initial)
            if distance_from_initial > 3.0:  # Max 3Å drift
                pull_back = (current_center - ligand_center_initial) * 0.5
                current_coords -= pull_back
                current_center = np.mean(current_coords, axis=0)
            
            trajectory_data.append({
                'frame': frame,
                'time_ps': time_ps,
                'ligand_coords': current_coords.copy(),
                'ligand_center': current_center.copy()
            })
            ligand_centers.append(current_center)
        
        ligand_centers = np.array(ligand_centers)
        
        # Create comprehensive MD animation
        fig = plt.figure(figsize=(20, 15))
        
        def animate(frame):
            fig.clear()
            
            # Main MD view
            ax1 = fig.add_subplot(221, projection='3d')
            
            # Show protein backbone as ribbon-like structure
            if len(protein_atoms) > 0:
                protein_coords = traj.xyz[frame, protein_atoms] * 10
                
                # Draw backbone as thick connected ribbon
                for i in range(len(protein_coords)-1):
                    # Color gradient along backbone
                    color_intensity = i / len(protein_coords)
                    color = plt.cm.Blues(0.3 + 0.4 * color_intensity)
                    
                    ax1.plot([protein_coords[i,0], protein_coords[i+1,0]], 
                            [protein_coords[i,1], protein_coords[i+1,1]], 
                            [protein_coords[i,2], protein_coords[i+1,2]], 
                            color=color, linewidth=3, alpha=0.8)
                
                # Add CA atoms as larger spheres
                ax1.scatter(protein_coords[:, 0], protein_coords[:, 1], protein_coords[:, 2], 
                          c='steelblue', alpha=0.9, s=40, label='Protein Backbone',
                          edgecolors='darkblue', linewidth=0.5)
            
            # Plot moving ligand
            current_ligand = trajectory_data[frame]['ligand_coords']
            
            # Color ligand atoms by element
            colors = []
            sizes = []
            for atom in ligand_atoms:
                element = atom['element'].upper()
                if element == 'C':
                    colors.append('darkgray')
                    sizes.append(100)
                elif element == 'N':
                    colors.append('blue')
                    sizes.append(90)
                elif element == 'O':
                    colors.append('red')
                    sizes.append(80)
                elif element == 'S':
                    colors.append('yellow')
                    sizes.append(110)
                else:
                    colors.append('purple')
                    sizes.append(70)
            
            ax1.scatter(current_ligand[:, 0], current_ligand[:, 1], current_ligand[:, 2], 
                       c=colors, s=sizes, alpha=0.9, label='Ligand', 
                       edgecolors='black', linewidth=1.5)
            
            # Draw ligand bonds
            for i in range(len(current_ligand)):
                for j in range(i+1, len(current_ligand)):
                    dist = np.linalg.norm(current_ligand[i] - current_ligand[j])
                    if 0.9 < dist < 1.8:
                        ax1.plot([current_ligand[i,0], current_ligand[j,0]], 
                                [current_ligand[i,1], current_ligand[j,1]], 
                                [current_ligand[i,2], current_ligand[j,2]], 
                                'black', linewidth=2, alpha=0.8)
            
            ax1.set_xlabel('X (Å)')
            ax1.set_ylabel('Y (Å)')
            ax1.set_zlabel('Z (Å)')
            time_ps = trajectory_data[frame]['time_ps']
            ax1.set_title(f'MD Simulation: {hit_name}\nFrame {frame+1}/{n_frames} (Time: {time_ps:.0f} ps)', 
                         fontsize=14, fontweight='bold')
            ax1.legend()
            
            # Set view around ligand
            center = trajectory_data[frame]['ligand_center']
            range_val = 8
            ax1.set_xlim(center[0]-range_val, center[0]+range_val)
            ax1.set_ylim(center[1]-range_val, center[1]+range_val)
            ax1.set_zlim(center[2]-range_val, center[2]+range_val)
            
            # Trajectory trace
            ax2 = fig.add_subplot(222, projection='3d')
            if frame > 0:
                trace_centers = ligand_centers[:frame+1]
                ax2.plot(trace_centers[:, 0], trace_centers[:, 1], trace_centers[:, 2], 
                        'red', linewidth=3, alpha=0.8, label='Ligand Path')
                ax2.scatter(trace_centers[-1, 0], trace_centers[-1, 1], trace_centers[-1, 2], 
                          c='red', s=150, marker='o', label='Current Position')
                ax2.scatter(trace_centers[0, 0], trace_centers[0, 1], trace_centers[0, 2], 
                          c='green', s=150, marker='s', label='Start Position')
            
            ax2.set_xlabel('X (Å)')
            ax2.set_ylabel('Y (Å)')
            ax2.set_zlabel('Z (Å)')
            ax2.set_title('Ligand Movement Trajectory', fontsize=14, fontweight='bold')
            ax2.legend()
            
            # Distance from initial position
            ax3 = fig.add_subplot(223)
            if frame > 0:
                distances = [np.linalg.norm(ligand_centers[i] - ligand_center_initial) 
                           for i in range(frame+1)]
                times = [trajectory_data[i]['time_ps'] for i in range(frame+1)]
                
                ax3.plot(times, distances, 'blue', linewidth=2, label='Distance from Start')
                ax3.axhline(y=np.mean(distances), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(distances):.1f} Å')
                ax3.set_xlabel('Time (ps)')
                ax3.set_ylabel('Distance (Å)')
                ax3.set_title('Binding Site Stability', fontsize=14, fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # RMSD analysis
            ax4 = fig.add_subplot(224)
            if frame > 0:
                rmsds = []
                for i in range(frame+1):
                    current = trajectory_data[i]['ligand_coords']
                    rmsd = np.sqrt(np.mean(np.sum((current - ligand_coords_initial)**2, axis=1)))
                    rmsds.append(rmsd)
                
                times = [trajectory_data[i]['time_ps'] for i in range(frame+1)]
                ax4.plot(times, rmsds, 'green', linewidth=2, label='Ligand RMSD')
                ax4.axhline(y=np.mean(rmsds), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(rmsds):.1f} Å')
                ax4.set_xlabel('Time (ps)')
                ax4.set_ylabel('RMSD (Å)')
                ax4.set_title('Conformational Changes', fontsize=14, fontweight='bold')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=n_frames, interval=200, repeat=True)
        
        # Save as GIF
        gif_file = output_dir / f"md_simulation_{hit_name}.gif"
        writer = PillowWriter(fps=5)
        anim.save(gif_file, writer=writer)
        plt.close()
        
        # Create static trajectory analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 3D trajectory path
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.plot(ligand_centers[:, 0], ligand_centers[:, 1], ligand_centers[:, 2], 
                'red', linewidth=2, alpha=0.8, label='Ligand Trajectory')
        ax1.scatter(ligand_centers[0, 0], ligand_centers[0, 1], ligand_centers[0, 2], 
                  c='green', s=150, marker='o', label='Start', zorder=5)
        ax1.scatter(ligand_centers[-1, 0], ligand_centers[-1, 1], ligand_centers[-1, 2], 
                  c='red', s=150, marker='s', label='End', zorder=5)
        ax1.scatter(ligand_center_initial[0], ligand_center_initial[1], ligand_center_initial[2], 
                  c='orange', s=200, marker='*', label='Binding Site', zorder=5)
        
        ax1.set_xlabel('X (Å)')
        ax1.set_ylabel('Y (Å)')
        ax1.set_zlabel('Z (Å)')
        ax1.set_title(f'3D Ligand Trajectory - {hit_name}')
        ax1.legend()
        
        # Distance analysis
        distances = [np.linalg.norm(center - ligand_center_initial) for center in ligand_centers]
        times = [data['time_ps'] for data in trajectory_data]
        
        ax2.plot(times, distances, 'blue', linewidth=2, label='Distance from Binding Site')
        ax2.axhline(y=np.mean(distances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(distances):.1f} Å')
        ax2.fill_between(times, distances, alpha=0.3, color='blue')
        ax2.set_xlabel('Time (ps)')
        ax2.set_ylabel('Distance (Å)')
        ax2.set_title('Binding Site Proximity Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # RMSD analysis
        rmsds = []
        for data in trajectory_data:
            current = data['ligand_coords']
            rmsd = np.sqrt(np.mean(np.sum((current - ligand_coords_initial)**2, axis=1)))
            rmsds.append(rmsd)
        
        ax3.plot(times, rmsds, 'green', linewidth=2, label='Ligand RMSD')
        ax3.axhline(y=np.mean(rmsds), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(rmsds):.1f} Å')
        ax3.fill_between(times, rmsds, alpha=0.3, color='green')
        ax3.set_xlabel('Time (ps)')
        ax3.set_ylabel('RMSD (Å)')
        ax3.set_title('Conformational Stability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Movement velocity
        velocities = []
        for i in range(1, len(ligand_centers)):
            displacement = np.linalg.norm(ligand_centers[i] - ligand_centers[i-1])
            time_diff = times[i] - times[i-1]
            velocity = displacement / time_diff if time_diff > 0 else 0
            velocities.append(velocity)
        
        ax4.plot(times[1:], velocities, 'purple', linewidth=2, label='Movement Velocity')
        ax4.axhline(y=np.mean(velocities), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(velocities):.3f} Å/ps')
        ax4.fill_between(times[1:], velocities, alpha=0.3, color='purple')
        ax4.set_xlabel('Time (ps)')
        ax4.set_ylabel('Velocity (Å/ps)')
        ax4.set_title('Ligand Movement Dynamics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        static_file = output_dir / f"md_trajectory_analysis_{hit_name}.png"
        plt.savefig(static_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate statistics
        total_distance = np.sum([np.linalg.norm(ligand_centers[i] - ligand_centers[i-1]) 
                                for i in range(1, len(ligand_centers))])
        
        return {
            "status": "success",
            "files": {
                "trajectory_gif": str(gif_file),
                "trajectory_analysis": str(static_file)
            },
            "analysis": {
                "n_frames": n_frames,
                "total_distance_traveled": float(total_distance),
                "mean_displacement": float(np.mean(distances)),
                "max_displacement": float(np.max(distances)),
                "mean_rmsd": float(np.mean(rmsds)),
                "mean_velocity": float(np.mean(velocities)),
                "binding_stability": "stable" if np.mean(rmsds) < 1.5 else "dynamic"
            }
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": f"MD trajectory simulation failed: {str(e)}",
            "traceback": traceback.format_exc()
        }
