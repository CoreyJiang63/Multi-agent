#!/usr/bin/env python3
from typing import TypedDict, Optional, List, Dict
from langgraph.graph import StateGraph, END
import functools
from datetime import datetime
from pocket_subagent import *
from dock_subagent import *

TOP_K_PROT = 3

# ----------------------------
# LangGraph node functions
# ----------------------------
# class AgentState(TypedDict, total=False):
#     gene: str
#     candidates: List[Dict]
#     chosen: List[Dict]
#     structure_files: List[Dict]
#     pocket_results: List[Dict]
#     summary_file: str
    
#     docking_pockets: List

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
        "ligand_library": state.get("ligand_library"),
        "timestamp": datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_dir = os.path.join(OUTPUT_DIR, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    out_file = os.path.join(summary_dir, f"summary_{state.get("gene") or 'none'}.json")
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
def build_graph():
    Agent = StateGraph(AgentState)
    Agent.add_node("search", track_node("search", node_search))
    Agent.add_node("choose", track_node("choose", node_choose))
    Agent.add_node("download", track_node("download", node_download))
    Agent.add_node("detect", track_node("detect", node_detect_pockets))
    Agent.add_node("load", track_node("load", node_load_pockets))
    Agent.add_node("library", track_node("library", node_ligand_library))
    Agent.add_node("finalize", track_node("finalize", node_finalize))
    
    Agent.set_entry_point("search")
    Agent.set_finish_point("finalize")
    
    # connect nodes in linear flow
    Agent.add_edge("search", "choose")
    Agent.add_edge("choose", "download")
    Agent.add_edge("download", "detect")
    Agent.add_edge("detect", "load")
    Agent.add_edge("load", "library")
    Agent.add_edge("library", "finalize")
    
    app = Agent.compile()
    return app

# ----------------------------
# Main function
# ----------------------------
def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python pocket_subagent.py <GENE_SYMBOL>")
    #     sys.exit(1)
    genes = ["ESR2"]
    gene_file = 'structured_data.json'
    
    consensus_genes = parse_json(gene_file)
    
    for gene_name in genes:
        print(f"Starting pocket sub-agent for gene: {gene_name}")
        app = build_graph()
        init_state = AgentState({"gene": gene_name})
        result = app.invoke(init_state)
        print("Done. Summary:", result.get("summary_file"))
        print("State keys:", list(result.keys()))


if __name__ == "__main__":
    main()