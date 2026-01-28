#!/usr/bin/env python3
"""
PyMOL-based MD Animation Module
Creates high-quality molecular dynamics animations using PyMOL
"""

import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import traceback

def extract_trajectory_frames_with_gmx(xtc_file, gro_file, output_dir, max_frames=50):
    """Extract trajectory frames using GROMACS tools."""
    try:
        # Use gmx trjconv to extract frames from .xtc trajectory
        frames_dir = output_dir / "trajectory_frames"
        frames_dir.mkdir(exist_ok=True)
        
        # First, get trajectory info
        info_cmd = ["gmx", "check", "-f", str(xtc_file)]
        info_result = subprocess.run(info_cmd, capture_output=True, text=True, timeout=30)
        
        # Extract frames at regular intervals
        frame_files = []
        
        # Use gmx trjconv to convert trajectory to multiple PDB frames
        for frame_idx in range(0, max_frames):
            frame_time = frame_idx * 10  # Every 10 ps
            frame_file = frames_dir / f"frame_{frame_idx:04d}.pdb"
            
            trjconv_cmd = [
                "gmx", "trjconv", 
                "-f", str(xtc_file),
                "-s", str(gro_file),
                "-o", str(frame_file),
                "-dump", str(frame_time),
                "-nobackup"
            ]
            
            result = subprocess.run(
                trjconv_cmd, 
                input="0\n",  # Select system
                text=True, 
                capture_output=True, 
                timeout=60
            )
            
            if result.returncode == 0 and frame_file.exists():
                frame_files.append(frame_file)
            else:
                break  # No more frames available
        
        return frame_files
        
    except Exception as e:
        print(f"GMX trajectory extraction failed: {e}")
        return []

def parse_gro_trajectory(gro_file):
    """Parse GROMACS .gro file to extract coordinates (fallback for single frame)."""
    frames = []
    with open(gro_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith('Generated') or lines[i].strip().startswith('Protein'):
            # Skip title line
            i += 1
            continue
        
        try:
            n_atoms = int(lines[i].strip())
            i += 1
            
            frame_atoms = []
            for j in range(n_atoms):
                if i + j >= len(lines):
                    break
                line = lines[i + j]
                if len(line) < 44:
                    continue
                
                # Parse GROMACS .gro format
                residue_id = int(line[0:5].strip())
                residue_name = line[5:10].strip()
                atom_name = line[10:15].strip()
                atom_id = int(line[15:20].strip())
                x = float(line[20:28].strip()) * 10  # Convert nm to Angstrom
                y = float(line[28:36].strip()) * 10
                z = float(line[36:44].strip()) * 10
                
                frame_atoms.append({
                    'residue_id': residue_id,
                    'residue_name': residue_name,
                    'atom_name': atom_name,
                    'atom_id': atom_id,
                    'x': x, 'y': y, 'z': z
                })
            
            frames.append(frame_atoms)
            i += n_atoms + 1  # Skip box vectors line
            
        except (ValueError, IndexError):
            i += 1
            continue
    
    return frames

def create_multi_frame_pdb(trajectory_frames, output_dir, hit_name, max_frames=50):
    """Create multi-frame PDB file from trajectory data."""
    if not trajectory_frames:
        return None
    
    # Subsample frames if too many
    if len(trajectory_frames) > max_frames:
        indices = np.linspace(0, len(trajectory_frames)-1, max_frames, dtype=int)
        trajectory_frames = [trajectory_frames[i] for i in indices]
    
    output_file = output_dir / f"trajectory_{hit_name}.pdb"
    
    with open(output_file, 'w') as f:
        for frame_idx, frame in enumerate(trajectory_frames):
            f.write(f"MODEL     {frame_idx + 1:4d}\n")
            
            atom_counter = 1
            for atom in frame:
                # Determine atom type and element
                element = atom['atom_name'][0] if atom['atom_name'] else 'C'
                if element.isdigit():
                    element = 'C'
                
                # Determine record type
                if atom['residue_name'] in ['UNL', 'LIG', 'MOL']:
                    record_type = 'HETATM'
                else:
                    record_type = 'ATOM  '
                
                # Write PDB line
                f.write(f"{record_type}{atom_counter:5d}  {atom['atom_name']:<4s}{atom['residue_name']:>3s} A{atom['residue_id']:4d}    "
                       f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}  1.00  0.00          {element:>2s}\n")
                atom_counter += 1
            
            f.write("ENDMDL\n")
    
    return output_file

def create_pymol_md_animation_script(trajectory_pdb, output_dir, hit_name, time_step_ps=10):
    """Create PyMOL script for MD animation."""
    script_content = f'''# PyMOL MD Animation Script for {hit_name}
# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# Load multi-frame trajectory
load {trajectory_pdb}, trajectory

# Basic setup for high-quality rendering
bg_color white
set ray_opaque_background, off
set ray_shadows, on
set antialias, 2
set hash_max, 300
set cartoon_fancy_helices, 1
set cartoon_fancy_sheets, 1

# Remove water and ions
remove solvent
remove inorganic

# Define selections
select protein, polymer and trajectory
select ligand, not polymer and not solvent and not inorganic and trajectory
select binding_site, (polymer within 5 of ligand) and trajectory

# Protein representation - cartoon with transparency
show cartoon, protein
set cartoon_transparency, 0.3, protein
spectrum count, rainbow, protein
set cartoon_ring_mode, 3

# Binding site representation - sticks
show sticks, binding_site
set stick_radius, 0.15, binding_site
util.cbag binding_site
set stick_transparency, 0.2, binding_site

# Ligand representation - ball and stick
show sticks, ligand
show spheres, ligand
set sphere_scale, 0.3, ligand
set stick_radius, 0.2, ligand
util.cbag ligand
set stick_transparency, 0.0, ligand
set sphere_transparency, 0.0, ligand

# Add hydrogen bonds
distance hbonds, ligand, binding_site, 3.5, mode=2
hide labels, hbonds
set dash_color, yellow, hbonds
set dash_width, 3, hbonds
set dash_transparency, 0.3, hbonds

# Center view on ligand
center ligand
orient ligand
zoom ligand, 8

# Animation settings
set movie_panel, 1
set movie_panel_row_height, 30

# Create animation frames
python
import pymol
from pymol import cmd

# Get number of states (frames)
n_states = cmd.count_states("trajectory")
print(f"Number of frames: {{n_states}}")

# Set up movie
cmd.mset("1 -{0}".format(n_states))

# Create frame labels with time
for i in range(1, n_states + 1):
    time_ps = (i - 1) * {time_step_ps}
    cmd.frame(i)
    cmd.set("label_position", [0, 0, 0])
    
    # Add time label
    cmd.pseudoatom("time_label", pos=[0, 0, 0])
    cmd.label("time_label", f'"Time: {{time_ps:.0f}} ps"')
    cmd.set("label_color", "black", "time_label")
    cmd.set("label_size", 20, "time_label")
    
    # Update view for each frame
    cmd.center("ligand")
    cmd.orient("ligand")
    cmd.zoom("ligand", 8)

python end

# Export animation as images
python
import os
output_dir = "{output_dir}"
hit_name = "{hit_name}"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Export each frame
for i in range(1, n_states + 1):
    cmd.frame(i)
    time_ps = (i - 1) * {time_step_ps}
    
    # Remove previous time label
    try:
        cmd.delete("time_label")
    except:
        pass
    
    # Add new time label
    cmd.pseudoatom("time_label", pos=[cmd.get_view()[12], cmd.get_view()[13] + 5, cmd.get_view()[14]])
    cmd.label("time_label", f'"Time: {{time_ps:.0f}} ps"')
    cmd.set("label_color", "black", "time_label")
    cmd.set("label_size", 16, "time_label")
    
    # Export frame
    filename = os.path.join(output_dir, f"frame_{{i:04d}}.png")
    cmd.png(filename, width=800, height=600, dpi=300, ray=1)
    print(f"Exported frame {{i}}: {{filename}}")

python end

print "MD animation frames exported successfully"
save {output_dir}/md_session_{hit_name}.pse
quit
'''
    
    script_file = output_dir / f"pymol_md_animation_{hit_name}.pml"
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    return script_file

def create_gif_from_frames(output_dir, hit_name):
    """Create GIF animation from PNG frames using ImageMagick."""
    try:
        # Use ImageMagick to create GIF
        gif_file = output_dir / f"md_animation_{hit_name}.gif"
        
        convert_cmd = [
            "convert", "-delay", "50", "-loop", "0",
            str(output_dir / "frame_*.png"),
            str(gif_file)
        ]
        
        result = subprocess.run(convert_cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            return str(gif_file)
        else:
            print(f"ImageMagick failed: {result.stderr}")
            
            # Fallback: try with ffmpeg
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-framerate", "2", "-pattern_type", "glob",
                "-i", str(output_dir / "frame_*.png"),
                "-vf", "palettegen=reserve_transparent=0", "-y",
                str(output_dir / "palette.png")
            ]
            
            subprocess.run(ffmpeg_cmd, capture_output=True, timeout=60)
            
            ffmpeg_cmd2 = [
                "ffmpeg", "-y", "-framerate", "2", "-pattern_type", "glob",
                "-i", str(output_dir / "frame_*.png"),
                "-i", str(output_dir / "palette.png"),
                "-lavfi", "paletteuse", str(gif_file)
            ]
            
            result2 = subprocess.run(ffmpeg_cmd2, capture_output=True, text=True, timeout=120)
            
            if result2.returncode == 0:
                return str(gif_file)
            else:
                print(f"FFmpeg also failed: {result2.stderr}")
                return None
                
    except Exception as e:
        print(f"GIF creation failed: {e}")
        return None

def combine_pdb_frames(frame_files, output_pdb):
    """Combine multiple PDB frame files into a single multi-frame PDB."""
    with open(output_pdb, 'w') as out_f:
        for i, frame_file in enumerate(frame_files):
            out_f.write(f"MODEL {i+1}\n")
            with open(frame_file, 'r') as in_f:
                for line in in_f:
                    if line.startswith(('ATOM', 'HETATM')):
                        out_f.write(line)
            out_f.write("ENDMDL\n")

def run_pymol_script(script_file):
    """Execute PyMOL script and return success status."""
    try:
        pymol_cmd = ["pymol", "-c", str(script_file)]
        result = subprocess.run(pymol_cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"PyMOL script executed successfully")
            return True
        else:
            print(f"PyMOL execution failed: {result.stderr}")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Could not execute PyMOL: {e}")
        return False
    except Exception as e:
        print(f"PyMOL execution error: {e}")
        return False

def create_pymol_md_visualization(md_dir, output_dir, hit_name):
    """Create PyMOL-based MD animation from GROMACS trajectory."""
    try:
        md_path = Path(md_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Look for trajectory files - prioritize .xtc over .gro
        xtc_file = md_path / "simulation" / "md.xtc"
        gro_file = md_path / "simulation" / "md.gro"
        
        if not xtc_file.exists():
            xtc_file = md_path / "md.xtc"
        if not gro_file.exists():
            gro_file = md_path / "md.gro"
        
        frames = []
        
        # Try to extract frames from .xtc trajectory first
        if xtc_file.exists() and gro_file.exists():
            print(f"Processing XTC trajectory: {xtc_file}")
            frame_files = extract_trajectory_frames_with_gmx(xtc_file, gro_file, output_path)
            
            if frame_files:
                print(f"Extracted {len(frame_files)} PDB frames from XTC trajectory")
                # Create multi-frame PDB from extracted frames
                multiframe_pdb = output_path / f"{hit_name}_trajectory.pdb"
                combine_pdb_frames(frame_files, multiframe_pdb)
                frames_count = len(frame_files)
            else:
                print("XTC extraction failed, falling back to GRO parsing")
                frames = parse_gro_trajectory(gro_file)
                frames_count = len(frames)
                multiframe_pdb = create_multi_frame_pdb(frames, output_path, hit_name)
        elif gro_file.exists():
            print(f"Processing GRO trajectory: {gro_file}")
            frames = parse_gro_trajectory(gro_file)
            frames_count = len(frames)
            multiframe_pdb = create_multi_frame_pdb(frames, output_path, hit_name)
        else:
            return {"success": False, "error": "No trajectory files found (.xtc or .gro)"}
        
        if frames_count == 0:
            return {"success": False, "error": "No frames extracted from trajectory"}
        
        print(f"Total frames for animation: {frames_count}")
        
        # Create PyMOL script
        pymol_script = create_pymol_md_animation_script(multiframe_pdb, output_path, hit_name)
        
        # Run PyMOL script
        success = run_pymol_script(pymol_script)
        
        # Create GIF from frames
        gif_file = create_gif_from_frames(output_path, hit_name)
        gif_success = gif_file is not None
        
        # Determine status based on success
        if success and gif_success:
            status = "completed"
        elif success:
            status = "partial"
        else:
            status = "script_created"
        
        return {
            "status": status,
            "files": {
                "multiframe_pdb": str(multiframe_pdb),
                "pymol_script": str(pymol_script),
                "gif_file": str(gif_file) if gif_success else None
            },
            "frames_extracted": frames_count,
            "trajectory_source": "xtc" if xtc_file.exists() else "gro",
            "instructions": [
                f"To run PyMOL MD animation manually:",
                f"1. conda activate drug_agent",
                f"2. cd {output_path}",
                f"3. pymol -c {pymol_script.name}",
                f"4. Or run in PyMOL GUI: File > Run Script > {pymol_script}",
                f"5. Animation frames will be saved as PNG files",
                f"6. Use 'convert -delay 50 -loop 0 *_frame_*.png animation.gif' to create GIF"
            ]
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": f"PyMOL MD visualization failed: {str(e)}",
            "traceback": traceback.format_exc()
        }
