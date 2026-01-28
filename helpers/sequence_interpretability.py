#!/usr/bin/env python3
"""
Sequence-to-Drug Interpretability Module

Provides visualization and analysis tools for understanding:
1. Cross-attention patterns (molecule generation → protein sequence)
2. Molecule-protein interaction analysis
3. Generation trajectory analysis
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Try to import RDKit for molecular visualization
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Molecular structure visualization will be limited.")

# Load SMILES token vocabulary
def load_token_vocabulary(vocab_path: str = None) -> Dict[int, str]:
    """Load token ID to chemical symbol mapping"""
    if vocab_path is None:
        # Default path
        vocab_path = os.path.join(os.path.dirname(__file__), '..', 'seq', 'compound_stoi.json')
    
    try:
        with open(vocab_path, 'r') as f:
            stoi = json.load(f)
        # Reverse mapping: ID -> symbol
        itos = {v: k for k, v in stoi.items()}
        return itos
    except Exception as e:
        print(f"Warning: Could not load token vocabulary from {vocab_path}: {e}")
        return {}


class SequenceInterpretabilityAnalyzer:
    """Analyzer for sequence-based drug generation interpretability"""
    
    def __init__(self, output_dir: str = "seq_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    
    def visualize_attention_heatmap(self,
                                   attention_trajectory: List[Dict],
                                   protein_seq: str,
                                   smiles: str,
                                   molecule_idx: int = 0,
                                   gene_name: str = "protein") -> str:
        """
        Create attention heatmap showing which protein regions are attended during generation
        
        Args:
            attention_trajectory: List of attention steps from CSV
            protein_seq: Protein sequence
            smiles: Generated SMILES string
            molecule_idx: Index of molecule for naming
            gene_name: Gene name
            
        Returns:
            Path to saved visualization
        """
        if not attention_trajectory or len(attention_trajectory) == 0:
            print("Warning: Empty attention trajectory")
            return None
        
        # Load token vocabulary for chemical symbol mapping
        token_vocab = load_token_vocabulary()
        
        # Extract attention weights
        num_steps = len(attention_trajectory)
        seq_len = len(protein_seq)
        
        # Build attention matrix: [generation_steps, protein_positions]
        attention_matrix = []
        tokens = []
        token_symbols = []  # Store actual chemical symbols
        
        for step_data in attention_trajectory:
            if 'attention_weights' in step_data:
                attn = step_data['attention_weights']
                if isinstance(attn, list):
                    # Truncate or pad to match protein length
                    if len(attn) > seq_len:
                        attn = attn[:seq_len]
                    elif len(attn) < seq_len:
                        attn = attn + [0.0] * (seq_len - len(attn))
                    attention_matrix.append(attn)
                    
                    # Get token ID
                    token_id = step_data.get('token', step_data.get('generated_token', '?'))
                    tokens.append(str(token_id))
                    
                    # Map token ID to chemical symbol
                    if isinstance(token_id, (int, float)) and token_vocab:
                        symbol = token_vocab.get(int(token_id), f'?{token_id}')
                    else:
                        symbol = str(token_id)
                    token_symbols.append(symbol)
        
        if len(attention_matrix) == 0:
            print("Warning: No valid attention weights found")
            return None
        
        attention_matrix = np.array(attention_matrix)
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(20, 12), 
                                gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot 1: Attention heatmap
        ax1 = axes[0]
        im = ax1.imshow(attention_matrix, aspect='auto', cmap='YlOrRd', 
                       interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Attention Weight', fontsize=12, fontweight='bold')
        
        # Set labels
        ax1.set_xlabel('Protein Residue Position', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Generation Step', fontsize=12, fontweight='bold')
        
        # Use SMILES directly from CSV (passed as parameter)
        ax1.set_title(f'Cross-Attention Heatmap: Molecule {molecule_idx}\nSMILES: {smiles[:80]}{"..." if len(smiles) > 80 else ""}', 
                     fontsize=14, fontweight='bold')
        
        # Add token labels on y-axis with chemical symbols
        if len(token_symbols) <= 50:  # Only show if not too many
            ax1.set_yticks(range(len(token_symbols)))
            # Format: "Step X: Symbol (ID)" for clarity
            labels = [f"{i}: {sym}" for i, sym in enumerate(token_symbols)]
            ax1.set_yticklabels(labels, fontsize=9, family='monospace')
        else:
            # For many steps, show every 5th step
            step_indices = range(0, len(token_symbols), 5)
            ax1.set_yticks(step_indices)
            labels = [f"{i}: {token_symbols[i]}" for i in step_indices]
            ax1.set_yticklabels(labels, fontsize=9, family='monospace')
        
        # Add amino acid labels on x-axis (every 10th)
        tick_positions = range(0, seq_len, max(1, seq_len // 50))
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels([f'{i}\n{protein_seq[i]}' for i in tick_positions], 
                           fontsize=8, rotation=0)
        
        # Plot 2: Aggregated attention (sum over generation steps)
        ax2 = axes[1]
        aggregated_attention = np.sum(attention_matrix, axis=0)
        aggregated_attention = aggregated_attention / np.max(aggregated_attention)  # Normalize
        
        positions = np.arange(seq_len)
        ax2.bar(positions, aggregated_attention, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Protein Residue Position', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Attention', fontsize=12, fontweight='bold')
        ax2.set_title('Aggregated Attention Across All Generation Steps', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlim(0, seq_len)
        
        # Highlight top attended regions
        top_k = 10
        top_indices = np.argsort(aggregated_attention)[-top_k:]
        for idx in top_indices:
            ax2.axvline(x=idx, color='red', linestyle='--', alpha=0.3, linewidth=1)
            ax2.text(idx, aggregated_attention[idx], f'{protein_seq[idx]}{idx+1}',
                    rotation=90, va='bottom', ha='center', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        # Save
        output_file = os.path.join(self.output_dir, gene_name,
                                   f'attention_heatmap_mol{molecule_idx}.png')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def analyze_molecule_protein_interaction(self,
                                             smiles: str,
                                             attention_trajectory: List[Dict],
                                             protein_seq: str,
                                             molecule_idx: int = 0,
                                             gene_name: str = "protein") -> Dict:
        """
        Analyze which protein residues are most attended during molecule generation
        
        Args:
            smiles: Generated SMILES
            attention_trajectory: Attention trajectory data
            protein_seq: Protein sequence
            molecule_idx: Molecule index
            gene_name: Gene name
            
        Returns:
            Interaction analysis results
        """
        seq_len = len(protein_seq)
        
        # Build attention matrix
        attention_matrix = []
        for step_data in attention_trajectory:
            if 'attention_weights' in step_data:
                attn = step_data['attention_weights']
                if isinstance(attn, list):
                    if len(attn) > seq_len:
                        attn = attn[:seq_len]
                    elif len(attn) < seq_len:
                        attn = attn + [0.0] * (seq_len - len(attn))
                    attention_matrix.append(attn)
        
        if len(attention_matrix) == 0:
            return {'error': 'No attention data'}
        
        attention_matrix = np.array(attention_matrix)
        
        # Compute statistics per residue
        mean_attention = np.mean(attention_matrix, axis=0)
        max_attention = np.max(attention_matrix, axis=0)
        
        # Get top attended residues
        top_k = min(20, seq_len)
        top_indices = np.argsort(mean_attention)[-top_k:][::-1]
        
        top_residues = [
            {
                'position': int(idx),
                'amino_acid': protein_seq[idx],
                'mean_attention': float(mean_attention[idx]),
                'max_attention': float(max_attention[idx])
            }
            for idx in top_indices
        ]
        
        results = {
            'smiles': smiles,
            'top_attended_residues': top_residues,
            'total_attention_steps': len(attention_matrix),
            'protein_length': seq_len
        }
        
        return results
    
    def create_comprehensive_report(self,
                                   molecules: List[Dict],
                                   protein_seq: str,
                                   gene_name: str = "protein",
                                   top_n: int = 5) -> Dict:
        """
        Create comprehensive interpretability report for top molecules
        
        Args:
            molecules: List of generated molecules with attention_trajectory
            protein_seq: Protein sequence
            gene_name: Gene name
            top_n: Number of top molecules to analyze
            
        Returns:
            Comprehensive analysis results
        """
        print(f"\n{'='*80}")
        print(f"🔬 INTERPRETABILITY ANALYSIS: {gene_name}")
        print(f"{'='*80}")
        
        report = {
            'gene': gene_name,
            'protein_length': len(protein_seq),
            'num_molecules_analyzed': min(top_n, len(molecules)),
            'molecule_analyses': []
        }
        
        # Analyze top molecules
        print(f"\n🧪 Analyzing top {top_n} molecules...")
        
        for idx, mol in enumerate(molecules[:top_n]):
            print(f"\n   Molecule {idx+1}/{top_n}: {mol.get('smiles', 'N/A')[:50]}...")
            
            mol_analysis = {
                'molecule_index': idx,
                'smiles': mol.get('smiles', ''),
                'total_reward': mol.get('total_reward', 0),
                'affinity': mol.get('affinity', 0),
                'novelty_score': mol.get('novelty_score', 0)
            }
            
            # Visualize attention if available
            if 'attention_trajectory' in mol and mol['attention_trajectory']:
                try:
                    attention_file = self.visualize_attention_heatmap(
                        mol['attention_trajectory'],
                        protein_seq,
                        mol.get('smiles', ''),
                        idx,
                        gene_name
                    )
                    mol_analysis['attention_heatmap'] = attention_file
                    print(f"      ✅ Attention heatmap saved")
                    
                    # Interaction analysis
                    interaction = self.analyze_molecule_protein_interaction(
                        mol.get('smiles', ''),
                        mol['attention_trajectory'],
                        protein_seq,
                        idx,
                        gene_name
                    )
                    mol_analysis['interaction_analysis'] = interaction
                    
                except Exception as e:
                    print(f"      ⚠️  Attention analysis failed: {e}")
                    mol_analysis['attention_error'] = str(e)
            else:
                print(f"      ⚠️  No attention trajectory data")
            
            report['molecule_analyses'].append(mol_analysis)
        
        # Save report
        report_file = os.path.join(self.output_dir, gene_name,
                                   f'interpretability_report_{gene_name}.json')
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Interpretability report saved: {report_file}")
        print(f"{'='*80}\n")
        
        return report


def create_interpretability_summary_visualization(report: Dict, output_dir: str):
    """
    Create a summary visualization of attention analysis
    """
    gene_name = report['gene']
    mol_analyses = report.get('molecule_analyses', [])[:5]
    
    if not mol_analyses:
        print("No molecule analyses to visualize")
        return None
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f'Interpretability Summary: {gene_name}', 
                fontsize=16, fontweight='bold')
    
    # Plot 1-3: Top 3 molecules' attention statistics
    for i, mol_data in enumerate(mol_analyses[:3]):
        ax = fig.add_subplot(gs[0, i])
        
        interaction = mol_data.get('interaction_analysis', {})
        if 'top_attended_residues' in interaction:
            top_res = interaction['top_attended_residues'][:10]
            positions = [r['position'] for r in top_res]
            attentions = [r['mean_attention'] for r in top_res]
            
            ax.barh(range(len(positions)), attentions, color='steelblue', alpha=0.7)
            ax.set_yticks(range(len(positions)))
            ax.set_yticklabels([f"{p}:{top_res[j]['amino_acid']}" for j, p in enumerate(positions)], fontsize=8)
            ax.set_xlabel('Mean Attention', fontsize=9)
            ax.set_title(f"Mol {i+1} (Reward: {mol_data.get('total_reward', 0):.1f})\nTop Attended Residues", fontsize=10)
    
    # Plot 4: Affinity distribution
    ax4 = fig.add_subplot(gs[1, 0])
    affinities = [m.get('affinity', 0) for m in mol_analyses if m.get('affinity', 0) > -1000]
    if affinities:
        ax4.hist(affinities, bins=10, color='coral', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Predicted Affinity', fontsize=10)
        ax4.set_ylabel('Count', fontsize=10)
        ax4.set_title('Affinity Distribution', fontsize=11, fontweight='bold')
        ax4.axvline(np.mean(affinities), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(affinities):.2f}')
        ax4.legend()
    
    # Plot 5: Reward vs Affinity
    ax5 = fig.add_subplot(gs[1, 1])
    rewards = [m.get('total_reward', 0) for m in mol_analyses]
    affinities = [m.get('affinity', 0) for m in mol_analyses]
    if rewards and affinities:
        ax5.scatter(affinities, rewards, s=100, c=range(len(rewards)), 
                   cmap='viridis', alpha=0.6, edgecolors='black')
        ax5.set_xlabel('Predicted Affinity', fontsize=10)
        ax5.set_ylabel('Total Reward', fontsize=10)
        ax5.set_title('Reward vs Affinity', fontsize=11, fontweight='bold')
        for i, (aff, rew) in enumerate(zip(affinities, rewards)):
            ax5.annotate(f'{i+1}', (aff, rew), fontsize=8, ha='center')
    
    # Plot 6: Statistics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    avg_affinity = np.mean([m.get('affinity', 0) for m in mol_analyses if m.get('affinity', 0) > -1000]) if mol_analyses else 0
    avg_reward = np.mean([m.get('total_reward', 0) for m in mol_analyses]) if mol_analyses else 0
    
    stats_text = f"""
    Analysis Summary
    ━━━━━━━━━━━━━━━━
    Protein: {gene_name}
    Length: {report['protein_length']} aa
    
    Molecules: {report['num_molecules_analyzed']}
    
    Avg Reward: {avg_reward:.2f}
    Avg Affinity: {avg_affinity:.2f}
    
    Top Molecule:
      Reward: {mol_analyses[0].get('total_reward', 0):.2f}
      Affinity: {mol_analyses[0].get('affinity', 0):.2f}
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    # Save
    output_file = os.path.join(output_dir, gene_name,
                              f'interpretability_summary_{gene_name}.png')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Summary visualization saved: {output_file}")
    
    return output_file
