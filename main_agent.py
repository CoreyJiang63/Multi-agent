#!/usr/bin/env python3
from typing import TypedDict, Optional, List, Dict
from langgraph.graph import StateGraph, END
import functools
from datetime import datetime
from pocket_subagent import *
from dock_subagent import *
from admet_subagent import *
from sequence_to_drug_subagent import node_generate_from_sequence, save_sequence_to_drug_results

TOP_K_PROT = 3

# ----------------------------
# LangGraph node functions
# ----------------------------

def node_finalize(state: AgentState) -> AgentState:
    # write summary JSON
    pdb_ids = []
    if state.get("chosen"):
        chosen_pdbs = state["chosen"]
        for c in chosen_pdbs:
            pdb_ids.append(c["rcsb_id"])
    summary = {
        "gene": state.get("gene"),
        "chosen_pdb": pdb_ids,
        "pocket_results": state.get("pocket_results"),
        "docking_pockets": state.get("docking_pockets"),
        "prepared_proteins": state.get("prepared_proteins"),
        "prepared_ligands": state.get("prepared_ligands"),
        "docking_jobs": state.get("docking_jobs"),
        "docking_results": state.get("docking_results"),
        "pocket_analysis": state.get("pocket_analysis"),
        "top_ligand_hits": state.get("top_ligand_hits"),
        "rescoring_results": state.get("rescoring_results"),
        "binding_mode_analysis": state.get("binding_mode_analysis"),
        "admet_predictions": state.get("admet_predictions"),
        "admet_analysis": state.get("admet_analysis"),
        "favorable_compounds": state.get("favorable_compounds"),
        "reports": state.get("reports"),
        "generated_molecules": state.get("generated_molecules"),
        "sequence_to_drug_results": state.get("sequence_to_drug_results"),
        "timestamp": datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_dir = os.path.join(OUTPUT_DIR, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    out_file = os.path.join(summary_dir, f"summary_{state.get('gene') or 'none'}.json")
    with open(out_file, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[node_finalize] summary written to {out_file}")
    state["summary_file"] = out_file
    return state
 
def track_node(name, func):
    @functools.wraps(func)
    def wrapper(state):
        start = time.time()
        print(f"[{name}] → START")

        out = func(state)

        dur = time.time() - start
        print(f"[{name}] ← END ({dur:.2f}s)")
        return out
    return wrapper 
        
# ----------------------------
# Build LangGraph
# ----------------------------
def build_graph(use_structure_based=True):
    """
    Build the agent graph.
    
    Args:
        use_structure_based: If True, use structure-based pipeline (PDB -> docking).
                           If False, use sequence-based pipeline (sequence -> drug generation).
    """
    Agent = StateGraph(AgentState)
    
    if use_structure_based:
        # Structure-based drug discovery pipeline
        Agent.add_node("search", track_node("search", node_search))
        Agent.add_node("choose", track_node("choose", node_choose))
        Agent.add_node("download", track_node("download", node_download))
        Agent.add_node("detect", track_node("detect", node_detect_pockets))
        Agent.add_node("load", track_node("load", node_load_pockets))
        Agent.add_node("prepare_proteins", track_node("prepare_proteins", node_prepare_proteins))
        Agent.add_node("prepare_ligands", track_node("prepare_ligands", node_prepare_ligands))
        Agent.add_node("plan_docking", track_node("plan_docking", node_plan_docking))
        Agent.add_node("run_docking", track_node("run_docking", node_run_docking))
        Agent.add_node("analyze", track_node("analyze", node_analyze_results))
        Agent.add_node("rescore", track_node("rescore", node_rescore_refine))
        Agent.add_node("md_simulation", track_node("md_simulation", node_md_simulation))
        Agent.add_node("binding_analysis", track_node("binding_analysis", node_binding_analysis))
        Agent.add_node("extract_hits", track_node("extract_hits", node_extract_top_hits))
        Agent.add_node("predict_admet", track_node("predict_admet", node_predict_admet))
        Agent.add_node("analyze_admet", track_node("analyze_admet", node_analyze_admet_results))
        Agent.add_node("generate_admet_report", track_node("generate_admet_report", node_generate_admet_report))
    else:
        # Sequence-based drug generation pipeline
        Agent.add_node("generate_from_sequence", track_node("generate_from_sequence", node_generate_from_sequence))
    
    Agent.add_node("finalize", track_node("finalize", node_finalize))
    
    Agent.set_finish_point("finalize")
    
    if use_structure_based:
        # Structure-based pipeline flow
        Agent.set_entry_point("search")
        Agent.add_edge("search", "choose")
        Agent.add_edge("choose", "download")
        Agent.add_edge("download", "detect")
        Agent.add_edge("detect", "load")
        Agent.add_edge("load", "prepare_proteins")
        Agent.add_edge("prepare_proteins", "prepare_ligands")
        Agent.add_edge("prepare_ligands", "plan_docking")
        Agent.add_edge("plan_docking", "run_docking")
        Agent.add_edge("run_docking", "analyze")
        Agent.add_edge("analyze", "rescore")
        Agent.add_edge("rescore", "md_simulation")
        Agent.add_edge("md_simulation", "extract_hits")
        Agent.add_edge("extract_hits", "predict_admet")
        Agent.add_edge("predict_admet", "analyze_admet")
        Agent.add_edge("analyze_admet", "generate_admet_report")
        Agent.add_edge("generate_admet_report", "finalize")
    else:
        # Sequence-based pipeline flow
        Agent.set_entry_point("generate_from_sequence")
        Agent.add_edge("generate_from_sequence", "finalize")
    
    app = Agent.compile()
    return app

# ----------------------------
# Main function
# ----------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Drug Discovery Agent Pipeline")
    parser.add_argument("--mode", choices=["structure", "sequence"], default="structure",
                       help="Pipeline mode: 'structure' for structure-based, 'sequence' for sequence-based")
    parser.add_argument("--gene", type=str, default="ESR2",
                       help="Gene symbol")
    parser.add_argument("--protein_seq", type=str,
                       help="Protein sequence (required for sequence mode)")
    parser.add_argument("--gen_size", type=int, default=10,
                       help="Number of molecules to generate (sequence mode only)")
    args = parser.parse_args()
    
    gene_name = args.gene
    use_structure_based = (args.mode == "structure")
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting Drug Discovery Pipeline")
    print(f"{'='*80}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Gene: {gene_name}")
    print(f"{'='*80}\n")
    
    app = build_graph(use_structure_based=use_structure_based)
    
    if use_structure_based:
        # Structure-based drug discovery
        init_state = AgentState({
            "gene": gene_name,
            # MD Simulation Configuration
            "md_simulation_time_ns": 0.05,  # 50 ps for fast screening
            "md_temperature": 300.0,  # Temperature in Kelvin
            "max_parallel_md_jobs": None,  # Auto-detect (CPU_count/2)
            "use_md_checkpoints": True,  # Enable checkpoint system
            "max_md_hits": 1,  # Maximum number of hits to run MD simulations on
        })
    else:
        # Sequence-based drug generation
        if not args.protein_seq:
            print("❌ Error: --protein_seq is required for sequence mode")
            print("Example: --protein_seq 'MDIKNSPSSLNSPSSYNCSQSILPLEHGSIYIPSSYVDSHHEYPAMTFYSPAVMNYSIPSNVTNLEGGPGRQTTSPNVLWPTPGHLSPLVVHRQLSHLYAEPQKSPWCEARSLEHTLPVNRETLKRKVSGNRCASPVTGPGSKRDAHFCAVCSDYASGYHYGVWSCEGCKAFFKRSIQGHNDYICPATNQCTIDKNRRKSCQACRLRKCYEVGMVKCGSRRERCGYRLVRRQRSADEQLHCAGKAKRSGGHAPRVRELLLDALSPEQLVLTLLEAEPPHVLISRPSAPFTEASMMMSLTKLADKELVHMISWAKKIPGFVELSLFDQVRLLESCWMEVLMMGLMWRSIDHPGKLIFAPDLVLDRDEGKCVEGILEIFDMLLATTSRFRELKLQHKEYLCVKAMILLNSSMYPLVTATQDADSSRKLAHLLNAVTDALVWVIAKSGISSQQQSMRLANLLMLLSHVRHASNKGMEHLLNMKCKNVVPVYDLLLEMLNAHVLRGCKSSITGSECSPAEDSKSKEGSQNPQSQ'")
            sys.exit(1)
        
        init_state = AgentState({
            "gene": gene_name,
            "protein_sequence": args.protein_seq,
            "gen_size": args.gen_size,
        })
    
    result = app.invoke(init_state)
    print("\n" + "="*80)
    print("✅ Pipeline completed!")
    print("="*80)
    print(f"Summary file: {result.get('summary_file')}")
    
    if not use_structure_based and result.get("generated_molecules"):
        print(f"Generated {len(result['generated_molecules'])} molecules")
        seq_output = save_sequence_to_drug_results(result)
        print(f"Sequence-to-drug results: {seq_output}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()