#!/usr/bin/env python3
from typing import TypedDict, Optional, List, Dict
from langgraph.graph import StateGraph, END
from agent_state import AgentState
from helpers.pocket_loader import *
from helpers.ligand_library import *
import os
import pandas as pd
import random
from pdbfixer import PDBFixer
from openmm.app import PDBFile

# ----------------------------
# Load pocket info
# ----------------------------
def node_load_pockets(state: AgentState) -> AgentState:
    """
    Load and filter pockets from p2rank output.
    Input:
        state["pocket_results"] = list of dicts for each PDB structure.
    Output:
        state["docking_pockets"] = aggregated list of pockets from all structures.
    """
    pocket_results = state.get("pocket_results", [])
    all_pockets = []

    print("[node_load_pockets] Loading pockets from all selected structures...")

    for item in pocket_results:
        pdb_id = item["pdb_id"]
        out_dir = item["out_dir"]
        struct_file = item["file"]

        pred_csv = f"{struct_file}_predictions.csv"
        if not os.path.exists(pred_csv):
            print(f"[node_load_pockets] Missing predictions file {pred_csv}, skip")
            continue

        df = read_pocket_csv(pred_csv)
        if df is None:
            continue
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(' ', '_')
        if "name" in df.columns:
            df['name'] = df['name'].str.strip()

        df_filtered = filter_pockets(df)

        # Fallback: if filtered empty, take top-1 raw pocket
        if df_filtered.empty and df is not None and not df.empty:
            print(f"[node_load_pockets] No pocket passed quality threshold for {pdb_id} — using top-1 fallback")
            df_filtered = df.sort_values("score", ascending=False).head(1)

        # If still empty (e.g. corrupted)
        if df_filtered.empty:
            print(f"[node_load_pockets] No usable pocket for {pdb_id}")
            continue

        pockets = df_to_pocket_list(df_filtered, pdb_id)
        all_pockets.extend(pockets)

        print(f"[node_load_pockets] {pdb_id}: {len(pockets)} pockets loaded")

    if not all_pockets:
        print("[node_load_pockets] WARNING: No pockets found for ANY structure. Downstream docking may fail.")

    state["docking_pockets"] = all_pockets
    return state

def node_ligand_library(state: AgentState) -> AgentState:
    """
    Build a ligand library for docking.
    This node uses public data sources only.
    Input:
        state["docking_pockets"]
    Output:
        state["ligand_library"]
    """

    pockets = state.get("docking_pockets", [])
    if not pockets:
        print("[node_ligand_library] WARNING: No pockets available. Docking may not proceed.")
    
    # Configuration
    TARGET_SIZE = 3000  # cost-limited small library
    ligands = []

    print("[node_ligand_library] Starting ligand library construction...")

    # # 1) ZINC (fast, small API pull)
    # print("[node_ligand_library] Fetching ZINC20 subset...")
    # zinc_ligs = fetch_zinc_subset(max_n=3000)
    # ligands.extend(zinc_ligs)
    # print(f"[node_ligand_library] ZINC: {len(zinc_ligs)} molecules")

    print("[node_ligand_library] Fetching CHEMBL bioactive subset...")
    chembl_ligs = fetch_chembl_subset(max_n=1050)
    ligands.extend(chembl_ligs)
    print(f"[node_ligand_library] CHEMBL: {len(chembl_ligs)} molecules")

    # # 2) Optional: ENAMINE fragments (cheap, but requires local CSV)
    # if len(ligands) < TARGET_SIZE:
    #     print("[node_ligand_library] Loading ENAMINE fragments (if CSV available)...")
    #     enamine_ligs = fetch_enamine_fragments(local_csv="enamine_fragments.csv", max_n=2000)
    #     ligands.extend(enamine_ligs)
    #     print(f"[node_ligand_library] ENAMINE: {len(enamine_ligs)} molecules")

    # Trim to target size
    if len(ligands) > TARGET_SIZE:
        ligands = random.sample(ligands, TARGET_SIZE)

    # Final filtering (optional Lean-Lipinski)
    ligands = [
        lig for lig in ligands 
        if 150 < lig["molecular_weight"] < 550 and lig["logp"] < 5.0
    ]

    print(f"[node_ligand_library] Final ligand count: {len(ligands)}")

    state["ligand_library"] = ligands
    return state