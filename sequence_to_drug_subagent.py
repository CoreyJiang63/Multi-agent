#!/usr/bin/env python3
"""
Sequence-to-Drug Subagent
Generates drug molecules from protein sequences using deep learning models.
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
import time
from typing import TypedDict, Optional, List, Dict
from agent_state import AgentState
from datetime import datetime

# Import interpretability analyzer
try:
    from helpers.sequence_interpretability import SequenceInterpretabilityAnalyzer, create_interpretability_summary_visualization
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    INTERPRETABILITY_AVAILABLE = False
    print("Warning: Interpretability module not available")

# Add seq/gen to path for imports
SEQ_DIR = os.path.join(os.path.dirname(__file__), "seq")
GEN_DIR = os.path.join(SEQ_DIR, "gen")

# class AgentState(TypedDict, total=False):
#     """State for sequence-to-drug generation"""
#     gene: str
#     protein_sequence: str
#     generated_molecules: Optional[List[Dict]]
#     sequence_to_drug_results: Optional[Dict]
#     timestamp: str


def node_generate_from_sequence(state: AgentState) -> AgentState:
    """
    Generate drug molecules from protein sequence using seq/gen model
    
    Args:
        state: Must contain 'protein_sequence' key
        
    Returns:
        Updated state with 'generated_molecules' and 'sequence_to_drug_results'
    """
    print("\n" + "="*80)
    print("🧬 SEQUENCE-TO-DRUG GENERATION")
    print("="*80)
    
    protein_seq = state.get("protein_sequence")
    if not protein_seq:
        print("❌ Error: No protein_sequence provided in state")
        state["generated_molecules"] = []
        state["sequence_to_drug_results"] = {"error": "No protein sequence provided"}
        return state
    
    gene_name = state.get("gene", "unknown")
    gen_size = state.get("gen_size", 10)  # Number of molecules to generate
    
    print(f"🎯 Target: {gene_name}")
    print(f"📏 Sequence length: {len(protein_seq)} amino acids")
    print(f"🔢 Generating {gen_size} molecules...")
    print(f"🧪 Sequence preview: {protein_seq[:50]}..." if len(protein_seq) > 50 else f"🧪 Sequence: {protein_seq}")
    
    # Create output directory
    output_dir = os.path.join(SEQ_DIR, "result")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create unique output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = os.path.join(output_dir, f"gen_{gene_name}_{timestamp}.csv")
    
    # Prepare command
    gen_script = os.path.join(GEN_DIR, "gen_mcts_attentionmapcom.py")
    model_weight = os.path.join(SEQ_DIR, "compound.pt")
    
    cmd = [
        "python", gen_script,
        "--model_weight", model_weight,
        "--gen_size", str(gen_size),
        "--csv_name", csv_name,
        "--protein_seq", protein_seq
    ]
    
    print(f"\n🚀 Running generation...")
    print(f"📝 Command: {' '.join(cmd[:6])}... (sequence truncated)")
    
    start_time = time.time()
    
    try:
        # Run the generation script
        result = subprocess.run(
            cmd,
            cwd=SEQ_DIR,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"❌ Generation failed with return code {result.returncode}")
            print(f"stderr: {result.stderr[:500]}")
            state["generated_molecules"] = []
            state["sequence_to_drug_results"] = {
                "error": f"Generation failed: {result.stderr[:200]}",
                "elapsed_time": elapsed_time
            }
            return state
        
        print(f"✅ Generation completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        # Parse the CSV output
        if not os.path.exists(csv_name):
            print(f"❌ Output file not found: {csv_name}")
            state["generated_molecules"] = []
            state["sequence_to_drug_results"] = {
                "error": "Output CSV file not generated",
                "elapsed_time": elapsed_time
            }
            return state
        
        print(f"\n📊 Parsing results from {csv_name}...")
        # Increase CSV field size limit to handle large attention_trajectory data
        import csv
        csv.field_size_limit(10**7)  # 10MB limit
        df = pd.read_csv(csv_name)
        
        print(f"✅ Found {len(df)} molecules in output")
        
        # Extract key information from each molecule
        # Only keep essential molecular properties, not all metadata
        molecules = []
        for idx, row in df.iterrows():
            mol_dict = {
                "smiles": row.get("smiles", ""),
                "novelty_score": float(row.get("novelty_score", 0.0)),
            }
            
            # Extract reward vector components if available
            # reward_vector format: [Druglikeness, Affinity, Synthesizability, Validity]
            if "reward_vector" in row and pd.notna(row["reward_vector"]):
                try:
                    reward_str = str(row["reward_vector"])
                    # Remove brackets and split by whitespace
                    reward_str = reward_str.strip("[]")
                    reward_values = [float(x) for x in reward_str.split()]
                    
                    if len(reward_values) >= 4:
                        mol_dict["druglikeness"] = round(reward_values[0], 4)
                        mol_dict["affinity"] = round(reward_values[1], 4)
                        mol_dict["synthesizability"] = round(reward_values[2], 4)
                        mol_dict["validity"] = round(reward_values[3], 4)
                        mol_dict["total_reward"] = round(sum(reward_values), 4)
                except Exception as e:
                    # Fallback: try eval method
                    try:
                        reward_vec = eval(row["reward_vector"]) if isinstance(row["reward_vector"], str) else row["reward_vector"]
                        if isinstance(reward_vec, (list, tuple, np.ndarray)) and len(reward_vec) >= 4:
                            mol_dict["druglikeness"] = round(float(reward_vec[0]), 4)
                            mol_dict["affinity"] = round(float(reward_vec[1]), 4)
                            mol_dict["synthesizability"] = round(float(reward_vec[2]), 4)
                            mol_dict["validity"] = round(float(reward_vec[3]), 4)
                            mol_dict["total_reward"] = round(float(sum(reward_vec)), 4)
                    except:
                        print(f"Warning: Could not parse reward_vector for molecule {idx}: {row.get('smiles', 'unknown')}")
            
            # Only keep essential properties, skip other metadata like atomList, attention_trajectory, etc.
            # This keeps the report clean and focused on drug properties
            
            molecules.append(mol_dict)
        
        # Sort by total reward (descending)
        if molecules and "total_reward" in molecules[0]:
            molecules = sorted(molecules, key=lambda x: x.get("total_reward", 0), reverse=True)
        
        # Create summary statistics
        summary = {
            "gene": gene_name,
            "protein_sequence_length": len(protein_seq),
            "num_molecules_generated": len(molecules),
            "generation_time_seconds": elapsed_time,
            "csv_output_path": csv_name,
            "timestamp": timestamp,
        }
        
        # Add statistics if available
        if molecules and "total_reward" in molecules[0]:
            rewards = [m["total_reward"] for m in molecules if "total_reward" in m]
            if rewards:
                summary["avg_total_reward"] = sum(rewards) / len(rewards)
                summary["max_total_reward"] = max(rewards)
                summary["min_total_reward"] = min(rewards)
        
        if molecules and "novelty_score" in molecules[0]:
            novelties = [m["novelty_score"] for m in molecules if "novelty_score" in m]
            if novelties:
                summary["avg_novelty"] = sum(novelties) / len(novelties)
                summary["max_novelty"] = max(novelties)
        
        if molecules and "affinity" in molecules[0]:
            affinities = [m["affinity"] for m in molecules if "affinity" in m and m["affinity"] > -1000000]
            if affinities:
                summary["avg_affinity"] = sum(affinities) / len(affinities)
                summary["max_affinity"] = max(affinities)
        
        state["generated_molecules"] = molecules
        state["sequence_to_drug_results"] = summary
        
        # Print summary
        print("\n" + "="*80)
        print("📈 GENERATION SUMMARY")
        print("="*80)
        print(f"✅ Successfully generated {len(molecules)} molecules")
        if "avg_total_reward" in summary:
            print(f"🏆 Average reward: {summary['avg_total_reward']:.2f}")
            print(f"🥇 Best reward: {summary['max_total_reward']:.2f}")
        if "avg_novelty" in summary:
            print(f"🎨 Average novelty: {summary['avg_novelty']:.4f}")
        if "avg_affinity" in summary:
            print(f"💊 Average affinity: {summary['avg_affinity']:.2f}")
        print(f"⏱️  Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
        
        # Print top 3 molecules
        if molecules:
            print("\n🥇 Top 3 Generated Molecules:")
            for i, mol in enumerate(molecules[:3], 1):
                print(f"\n  {i}. SMILES: {mol['smiles']}")
                if "total_reward" in mol:
                    print(f"     Total Reward: {mol['total_reward']:.2f}")
                if "affinity" in mol:
                    print(f"     Affinity: {mol['affinity']:.2f}")
                if "synthesizability" in mol:
                    print(f"     Synthesizability: {mol['synthesizability']:.2f}")
                if "novelty_score" in mol:
                    print(f"     Novelty: {mol['novelty_score']:.4f}")
        
        print("="*80 + "\n")
        
        # Run interpretability analysis if available
        if INTERPRETABILITY_AVAILABLE and len(molecules) > 0:
            try:
                print("\n🔬 Running interpretability analysis...")
                
                # Load full CSV data with attention trajectories
                # Need to re-read CSV to get attention_trajectory (not included in molecules dict)
                print(f"📂 Loading attention trajectories from {csv_name}...")
                df_full = pd.read_csv(csv_name)
                
                print(f"   CSV columns: {list(df_full.columns)}")
                print(f"   CSV has {len(df_full)} rows")
                
                # Add attention_trajectory to molecules by matching SMILES
                # Important: molecules list has been sorted by total_reward, 
                # so we need to match by SMILES, not by index
                trajectories_found = 0
                for mol in molecules:
                    mol_smiles = mol.get('smiles', '')
                    if not mol_smiles:
                        continue
                    
                    # Find matching row in CSV by SMILES
                    matching_rows = df_full[df_full['smiles'] == mol_smiles]
                    if len(matching_rows) == 0:
                        continue
                    
                    row = matching_rows.iloc[0]
                    if 'attention_trajectory' in df_full.columns and pd.notna(row['attention_trajectory']):
                        try:
                            # Parse attention trajectory
                            attn_traj = row['attention_trajectory']
                            
                            # If it's a string, try to eval it
                            if isinstance(attn_traj, str):
                                # Need to use eval (not ast.literal_eval) because data contains inf/-inf
                                import math
                                eval_globals = {'inf': math.inf, '__builtins__': {}}
                                attn_traj = eval(attn_traj, eval_globals)
                            
                            # Verify it's a list
                            if isinstance(attn_traj, list) and len(attn_traj) > 0:
                                mol['attention_trajectory'] = attn_traj
                                trajectories_found += 1
                                if trajectories_found == 1:  # Debug first molecule
                                    print(f"   📊 First trajectory has {len(attn_traj)} steps")
                                    print(f"      Matched SMILES: {mol_smiles[:50]}...")
                                    if len(attn_traj) > 0:
                                        print(f"      First step keys: {list(attn_traj[0].keys()) if isinstance(attn_traj[0], dict) else 'not a dict'}")
                        except Exception as e:
                            print(f"   ⚠️  Failed to parse trajectory for {mol_smiles[:30]}: {str(e)[:100]}")
                            if trajectories_found == 0:  # Show more details for first failure
                                import traceback
                                traceback.print_exc()
                
                print(f"   ✅ Found {trajectories_found}/{len(molecules)} attention trajectories")
                
                # Create analyzer
                analyzer = SequenceInterpretabilityAnalyzer(output_dir="seq_output")
                
                # Run interpretability analysis (binding site analysis removed)
                interpretability_report = analyzer.create_comprehensive_report(
                    molecules=molecules,
                    protein_seq=protein_seq,
                    gene_name=gene_name,
                    top_n=min(5, len(molecules))
                )
                
                # Create summary visualization
                summary_viz = create_interpretability_summary_visualization(
                    interpretability_report, 
                    "seq_output"
                )
                
                # Add to state
                state["interpretability_report"] = interpretability_report
                summary["interpretability_report_file"] = interpretability_report.get('gene', gene_name)
                
                print(f"✅ Interpretability analysis completed")
                
            except Exception as e:
                print(f"⚠️  Interpretability analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
    except subprocess.TimeoutExpired:
        print("❌ Generation timed out after 1 hour")
        state["generated_molecules"] = []
        state["sequence_to_drug_results"] = {"error": "Timeout after 1 hour"}
    except Exception as e:
        print(f"❌ Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        state["generated_molecules"] = []
        state["sequence_to_drug_results"] = {"error": str(e)}
    
    return state


def save_sequence_to_drug_results(state: AgentState, output_dir: str = "seq_output") -> str:
    """
    Save sequence-to-drug results to JSON file
    
    Args:
        state: Agent state with generated_molecules
        output_dir: Base output directory (default: seq_output)
        
    Returns:
        Path to saved JSON file
    """
    gene_name = state.get("gene", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create gene-specific directory: seq_output/{gene_name}/
    gene_output_dir = os.path.join(output_dir, gene_name)
    os.makedirs(gene_output_dir, exist_ok=True)
    
    output_file = os.path.join(gene_output_dir, f"sequence_to_drug_{gene_name}_{timestamp}.json")
    
    # Only keep essential molecular properties in generated_molecules
    molecules = state.get("generated_molecules", [])
    clean_molecules = []
    for mol in molecules:
        # Only keep key drug properties
        clean_mol = {
            "smiles": mol.get("smiles", ""),
            "druglikeness": mol.get("druglikeness"),
            "affinity": mol.get("affinity"),
            "synthesizability": mol.get("synthesizability"),
            "validity": mol.get("validity"),
            "novelty_score": mol.get("novelty_score"),
            "total_reward": mol.get("total_reward")
        }
        # Remove None values
        clean_mol = {k: v for k, v in clean_mol.items() if v is not None}
        clean_molecules.append(clean_mol)
    
    output_data = {
        "gene": gene_name,
        "protein_sequence": state.get("protein_sequence", ""),
        "generated_molecules": clean_molecules,
        "summary": state.get("sequence_to_drug_results", {}),
        "timestamp": timestamp
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Sequence-to-drug results saved to: {output_file}")
    return output_file


# Example usage
if __name__ == "__main__":
    # Test with ESR2 sequence
    test_sequence = "MDIKNSPSSLNSPSSYNCSQSILPLEHGSIYIPSSYVDSHHEYPAMTFYSPAVMNYSIPSNVTNLEGGPGRQTTSPNVLWPTPGHLSPLVVHRQLSHLYAEPQKSPWCEARSLEHTLPVNRETLKRKVSGNRCASPVTGPGSKRDAHFCAVCSDYASGYHYGVWSCEGCKAFFKRSIQGHNDYICPATNQCTIDKNRRKSCQACRLRKCYEVGMVKCGSRRERCGYRLVRRQRSADEQLHCAGKAKRSGGHAPRVRELLLDALSPEQLVLTLLEAEPPHVLISRPSAPFTEASMMMSLTKLADKELVHMISWAKKIPGFVELSLFDQVRLLESCWMEVLMMGLMWRSIDHPGKLIFAPDLVLDRDEGKCVEGILEIFDMLLATTSRFRELKLQHKEYLCVKAMILLNSSMYPLVTATQDADSSRKLAHLLNAVTDALVWVIAKSGISSQQQSMRLANLLMLLSHVRHASNKGMEHLLNMKCKNVVPVYDLLLEMLNAHVLRGCKSSITGSECSPAEDSKSKEGSQNPQSQ"
    
    state = AgentState({
        "gene": "ESR2",
        "protein_sequence": test_sequence,
        "gen_size": 5  # Generate 5 molecules for testing
    })
    
    result_state = node_generate_from_sequence(state)
    
    if result_state.get("generated_molecules"):
        output_file = save_sequence_to_drug_results(result_state)
        print(f"\n✅ Test completed. Results saved to {output_file}")
    else:
        print("\n❌ Test failed. No molecules generated.")
