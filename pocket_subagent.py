#!/usr/bin/env python3
"""
pocket_subagent.py

Usage:
    python pocket_subagent.py

What it does:
 - Search RCSB for PDB entries matching a gene name
 - Rank/choose a representative PDB (simple heuristics, extensible)
 - Download mmCIF/PDB file
 - Try pocket detection via p2rank
 - Returns a small JSON summary and writes pocket output to ./output/summary/
"""

import os
import sys
import json
import time
import shutil
import subprocess
from typing import TypedDict, Optional, List, Dict
import requests
from agent_state import AgentState

# LangGraph imports
from langgraph.graph import StateGraph, END

# ----------------------------
# Config / constants
# ----------------------------
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{}"
RCSB_DOWNLOAD_CIF = "https://files.rcsb.org/download/{}.cif"
RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"
PRANKWEB_BASE = "https://prankweb.cz"  # public server (has REST API/back-end)
LOCAL_P2RANK_DOCKER_IMAGE = "externelly/p2rank"  # common community image
OUTPUT_DIR = os.path.abspath("./output")
TOP_K_PROT = 3

def parse_json(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        consensus_genes = data.get("consensus_genes", [])
        if not isinstance(consensus_genes, list):
            print("Warning: consensus_genes is not a list!")
            return []
        print(f"Parsed {len(consensus_genes)} genes: {consensus_genes}")
        return consensus_genes
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

# ----------------------------
# Utilities: RCSB search & fetch
# ----------------------------
def search_pdbs_by_gene(gene_symbol: str, max_results: int=50) -> List[Dict]:
    """
    Query RCSB Search API with a free-text query for the gene symbol.
    Returns a list of dicts: {rcsb_id, title, method, resolution (float|null), ligands_count}
    """
    query_json = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {
                # "operator": "contains_phrase",
                "value": gene_symbol
            }
        },
        "request_options": {
            "return_all_hits": True,
        },
        "return_type": "entry"
    }
    # r = requests.post(RCSB_SEARCH_URL, json=query_json, timeout=30)
    # r.raise_for_status()
    # results = r.json().get("result_set", [])
    # pdbs = []
    # for rec in results[:max_results]:
    #     pdb_id = rec.get("identifier")
    #     # fetch basic metadata
    #     meta = requests.get(RCSB_DATA_ENTRY.format(pdb_id)).json()
    #     title = meta.get("struct", {}).get("title")
    #     exptl = meta.get("exptl", [])
    #     method = exptl[0].get("method") if exptl else None
    #     # resolution may be nested
    #     res = None
    #     try:
    #         res = float(meta.get("rcsb_entry_info", {}).get("resolution_combined")[0])
    #     except Exception:
    #         # some entries missing resolution (NMR, EM)
    #         res = None
    #     # detect ligands (non-polymer count)
    #     ligands = meta.get("nonpolymer_entities", []) or meta.get("nonpolymer", [])
    #     lig_count = len(ligands) if ligands else 0
    #     pdbs.append({
    #         "rcsb_id": pdb_id,
    #         "title": title,
    #         "method": method,
    #         "resolution": res,
    #         "ligand_count": lig_count,
    #         "raw_meta": meta
    #     })
    
    try:
        r = requests.post(RCSB_SEARCH_URL, json=query_json, timeout=30)
        r.raise_for_status()
        search_results = r.json().get("result_set", [])
    except requests.exceptions.RequestException as e:
        print(f"Search failed: {e}")
        return []

    # Extract top N IDs
    pdb_ids = [res["identifier"] for res in search_results[:max_results]]
    
    if not pdb_ids:
        return []
    
    graphql_query = """
    query($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        struct {
          title
        }
        exptl {
          method
        }
        rcsb_entry_info {
          resolution_combined
        }
        nonpolymer_entities {
          pdbx_entity_nonpoly {
            comp_id
          }
        }
      }
    }
    """
    
    variables = {"ids": pdb_ids}

    try:
        g_r = requests.post(
            RCSB_GRAPHQL_URL, 
            json={"query": graphql_query, "variables": variables}, 
            timeout=30
        )
        g_r.raise_for_status()
        data = g_r.json()
    except requests.exceptions.RequestException as e:
        print(f"GraphQL fetch failed: {e}")
        return []

    # --- Step 3: Parse and Format ---
    parsed_pdbs = []
    entries = data.get("data", {}).get("entries", [])

    for entry in entries:
        # Extract Method safely
        exptl_list = entry.get("exptl")
        method = exptl_list[0]["method"] if exptl_list else None

        # Extract Resolution safely (handle NMR/EM where it might be null)
        res_info = entry.get("rcsb_entry_info")
        resolution = None
        if res_info and res_info.get("resolution_combined"):
            # resolution_combined is a list, take the first one
            resolution = res_info["resolution_combined"][0]

        # Extract Ligand Count
        nonpolymers = entry.get("nonpolymer_entities")
        ligand_count = len(nonpolymers) if nonpolymers else 0

        parsed_pdbs.append({
            "rcsb_id": entry["rcsb_id"],
            "title": entry.get("struct", {}).get("title"),
            "method": method,
            "resolution": resolution,
            "ligand_count": ligand_count,
            # Keeping the raw entry if you need other fields later
            # "raw_meta": entry 
        })
    
    return parsed_pdbs

# ----------------------------
# Choose representative PDB
# ----------------------------
def choose_best_pdb(candidates: List[Dict], gene_query: str, top_k: int = TOP_K_PROT) -> Optional[Dict]:
    """
    Scores and ranks PDB candidates based on resolution, method, ligand presence, and title relevance.
    """
    if not candidates:
        return None

    def score(p: Dict):
        score = 0.0
        
        # --- 1. Resolution Score (Max ~60 pts) ---
        # We define a 'cutoff' of 4.0A. Anything worse gets 0 points for this section.
        res = p.get('resolution')
        if res is not None:
            score += max(0, (4.0 - res) * 20)
        else:
            # Penalty for missing resolution (usually NMR or low-res EM)
            # We give it a base score equivalent to ~3.5A resolution
            score += 10 

        # --- 2. Method Bonus (Max 10 pts) ---
        # X-ray and Cryo-EM are generally preferred for static coordinate tasks
        method = p.get('method', '').upper() if p.get('method') else ''
        if 'DIFFRACTION' in method or 'MICROSCOPY' in method:
            score += 10
        
        # --- 3. Ligand Bonus (Max 15 pts) ---
        # If looking for drug targets, presence of ligands indicates a bindable pocket
        if p.get('ligand_count', 0) > 0:
            score += 15

        # --- 4. Relevance Score (Max 30 pts) ---
        # Check if the gene name appears in the title
        title = p.get('title', '').upper()
        gene = gene_query.upper()
        
        if gene in title:
            score += 10
            # Double bonus if the title STARTS with the gene (Strongest relevance)
            if title.startswith(gene):
                score += 20

        return score

    candidates_sorted = sorted(candidates, key=lambda x: score(x), reverse=True)
    return candidates_sorted[:top_k]

# ----------------------------
# Download structure
# ----------------------------
def download_structure(pdb_id: str, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    
    # ---- download CIF ----
    cif_url = RCSB_DOWNLOAD_CIF.format(pdb_id)
    local = os.path.join(outdir, f"{pdb_id}.cif")
    print(f"[+] Downloading {pdb_id} mmCIF from RCSB: {cif_url}")
    r = requests.get(cif_url, timeout=30)
    if r.status_code != 200:
        # fallback to PDB format
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        r2 = requests.get(pdb_url, timeout=30)
        r2.raise_for_status()
        local = os.path.join(outdir, f"{pdb_id}.pdb")
        with open(local, "wb") as fh:
            fh.write(r2.content)
        return local
    with open(local, "wb") as fh:
        fh.write(r.content)
    return local

# # ----------------------------
# # Pocket detection helpers
# # ----------------------------
# def prankweb_predict_by_pdb(pdb_id: str, outdir: str, timeout: int = 30) -> Optional[str]:
#     """
#     Try to use PrankWeb public server REST API to request pocket prediction by PDB id.
#     The PrankWeb front-end and backend expose REST patterns; here we attempt a submit that
#     mimics using the PDB code. The public server may reject automated bulk jobs; use responsibly.
#     If successful, returns path to saved JSON results or None.
#     """
#     # NOTE: PrankWeb provides a REST API in the backend; endpoints are not fully publicized here.
#     # We attempt the simplest accessible flow: GET the web page for the PDB and then
#     # query the backend endpoint typically at /api/jobs/predict or similar. If that fails,
#     # this function returns None. (Fallback to local P2Rank is provided.)
#     try:
#         # quick connectivity check
#         r = requests.get(PRANKWEB_BASE, timeout=10)
#         if r.status_code != 200:
#             print("[!] PrankWeb not reachable (status {})".format(r.status_code))
#             return None
#     except Exception as e:
#         print("[!] PrankWeb not reachable:", e)
#         return None

#     # The PrankWeb UI normally submits tasks internally via its API.
#     # We'll attempt a documented minimal approach: call the P2Rank CLI via their 'predict' job endpoint.
#     # Since the exact public REST contract may change, we fall back silently, allowing local runs.
#     # (If you run your own PrankWeb instance you can call its API directly; see the PrankWeb repo.)
#     print("[i] PrankWeb reachable but automated submit endpoint is variable; skipping automated submit.")
#     return None

# def run_p2rank_docker(structure_path: str, outdir: str, docker_image: str = LOCAL_P2RANK_DOCKER_IMAGE) -> Optional[str]:
#     """
#     Run P2Rank inside Docker. Expects Docker available and the image pulled.
#     This function mounts the structure directory into the container and runs the CLI.
#     Returns path to predicted pockets directory if successful, else None.
#     """
#     os.makedirs(outdir, exist_ok=True)
#     abspath = os.path.abspath(structure_path)
#     workdir = os.path.dirname(abspath)
#     pdb_fname = os.path.basename(abspath)
#     container_input = f"/data/{pdb_fname}"
#     container_out = "/data/out"
#     print("[+] Running P2Rank via Docker. Ensure docker is installed and image available:", docker_image)
#     # Example CLI (P2Rank binary typical invocation): p2rank.sh predict input.cif outdir
#     # Many community Docker images map to this call; this command may need adjusting for your image.
#     docker_cmd = [
#         "docker", "run", "--rm",
#         "-v", f"{workdir}:/data",
#         docker_image,
#         "p2rank.sh", "predict", container_input, container_out
#     ]
#     try:
#         subprocess.run(docker_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
#     except subprocess.CalledProcessError as e:
#         print("[!] Docker/P2Rank failed:", e)
#         return None
#     # results will be in workdir/out/
#     results_dir = os.path.join(workdir, "out")
#     if os.path.isdir(results_dir):
#         # copy to our desired outdir
#         dest = os.path.join(outdir, "p2rank_results")
#         if os.path.isdir(dest):
#             shutil.rmtree(dest)
#         shutil.move(results_dir, dest)
#         return dest
#     return None

# def run_fpocket_local(structure_path: str, outdir: str) -> Optional[str]:
#     """
#     If fpocket binary is installed (in PATH), call it.
#     fpocket will create a directory like <file>_out; we move it to outdir.
#     """
#     print("[+] Trying fpocket (local binary) ...")
#     cmd = ["fpocket", "-f", structure_path]
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
#     except Exception as e:
#         print("[!] fpocket call failed:", e)
#         return None
#     # fpocket writes pockets to the cwd; attempt to find *_out
#     base = os.path.splitext(os.path.basename(structure_path))[0]
#     expected = base + "_out"
#     if os.path.isdir(expected):
#         dest = os.path.join(outdir, "fpocket_results")
#         if os.path.isdir(dest):
#             shutil.rmtree(dest)
#         shutil.move(expected, dest)
#         return dest
#     return None

# ----------------------------
# LangGraph node functions
# ----------------------------

def node_search(state: AgentState) -> AgentState:
    gene = state.get("gene")
    print(f"[node_search] Searching RCSB for gene: {gene}")
    pdbs = search_pdbs_by_gene(gene, max_results=100)
    state["candidates"] = pdbs
    return state

def node_choose(state: AgentState) -> AgentState:
    print("[node_choose] Choosing best PDB from candidates")
    chosen = choose_best_pdb(state.get("candidates", []), gene_query=state.get("gene"), top_k=TOP_K_PROT)
    state["chosen"] = chosen
    if chosen:
        print(f"      -> Top-{TOP_K_PROT} selected PDBs:")
        for c in chosen:
            print(f"      {c['rcsb_id']} (Method: {c.get('method')} | Resolution: {c.get('resolution')} | Ligands: {c.get('ligand_count')})")
    else:
        print("  -> no candidate found")
    return state

def node_download(state: AgentState) -> AgentState:
    chosen = state.get("chosen", [])
    if not chosen:
        return state
    downloaded = []
    for c in chosen:
        pdb_id = c["rcsb_id"]
        outdir = os.path.join(OUTPUT_DIR, state.get("gene"), pdb_id)
        file = download_structure(pdb_id, outdir)
        downloaded.append({"pdb_id": pdb_id, "file": file})
        print(f"[node_download] {pdb_id} downloaded: {file}")
        
    state["structure_files"] = downloaded
    return state

def node_detect_pockets(state: AgentState) -> AgentState:
    structures = state.get("structure_files", [])
    if not structures:
        state["pocket_results"] = []
        return state

    p2rank_exec = ".\\p2rank-2.5.1\\distro\\prank.bat"
    pocket_results = []

    for item in structures:
        pdb_id = item["pdb_id"]
        file_path = item["file"]
        out_dir = os.path.dirname(file_path)
        os.makedirs(out_dir, exist_ok=True)

        cmd = f'{p2rank_exec} predict -f "{file_path}" -o "{out_dir}"'
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            pocket_results.append({
                "pdb_id": pdb_id,
                "file": file_path,
                "out_dir": out_dir,
                "returncode": result.returncode,
                # "stdout": result.stdout,
                # "stderr": result.stderr,
            })
        except Exception as e:
            pocket_results.append({
                "pdb_id": pdb_id,
                "file": file_path,
                "error": str(e),
            })
        
    state["pocket_results"] = pocket_results
    return state

    # for c in chosen:
    #     pdb_id = c["rcsb_id"]
    #     outdir = os.path.join(OUTPUT_DIR, pdb_id)
    #     os.makedirs(outdir, exist_ok=True)

    #     # 1) Try PrankWeb (public) REST -- best for convenience/privacy tradeoff
    #     result = prankweb_predict_by_pdb(pdb_id, outdir)
    #     if result:
    #         state["pocket_results"] = {"method": "prankweb", "path": result}
    #         return state

    #     # 2) Try local P2Rank via Docker
    #     try:
    #         docker_ok = shutil.which("docker") is not None
    #         if docker_ok:
    #             pr = run_p2rank_docker(path, outdir)
    #             if pr:
    #                 state["pocket_results"] = {"method": "p2rank_docker", "path": pr}
    #                 return state
    #     except Exception as e:
    #         print("[!] P2Rank Docker attempt error:", e)

    #     # 3) try fpocket local
    #     pr2 = run_fpocket_local(path, outdir)
    #     if pr2:
    #         state["pocket_results"] = {"method": "fpocket", "path": pr2}
    #         return state

    #     # 4) none available
    #     state["pocket_results"] = {"method": None, "path": None}
    #     return state

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
        "timestamp": time.time()
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
