#!/usr/bin/env python3
from typing import TypedDict, Optional, List, Dict, Tuple, Any
from langgraph.graph import StateGraph, END
from agent_state import AgentState
from helpers.pocket_loader import *
from helpers.ligand_library import *
from helpers.docking import (
    _call_prepare_ligand,
    _call_prepare_receptor,
    _clean_protein_structure,
    _convert_cif_to_pdb,
    _ensure_executable,
    _find_structure_file,
    _generate_conformer,
    _group_pockets_by_pdb,
    _load_job_checkpoint,
    _load_ligand_records,
    _load_ligand_run,
    _parse_vina_log,
    _resolve_path,
    _run_vina_job,
    _sanitize_receptor_pdbqt,
    _save_job_checkpoint,
    _slugify_name,
)
from helpers.post_docking import (
    analyze_pocket_hits,
    create_pains_catalog,
    select_top_ligands,
    write_analysis_report,
)
from helpers.rescoring import (
    rescore_top_hits,
    enhanced_rescore_and_refine,
    cluster_poses,
)
from helpers.binding_analysis import (
    analyze_binding_modes,
)
import os
import json
import concurrent.futures
from datetime import datetime
from pathlib import Path
from rdkit import Chem
import shutil

# ----------------------------
# Load pocket info
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
PROTEIN_PREP_DIR = Path("prepared_proteins")
LIGAND_PREP_DIR = Path("prepared_ligands")
LIGAND_LIBRARY_DIR = Path("ligand_libraries")
LIGAND_PREP_TIMEOUT = 30  # seconds
DOCKING_OUTPUT_DIR = Path("docking_runs")
DEFAULT_BOX_SIZE = (22.5, 22.5, 22.5)
POCKET_MIN_SCORE = 20.0
POCKET_MIN_PROBABILITY = 0.85
POCKET_MIN_SAS_POINTS = 50
ANALYSIS_SCORE_CUTOFF = -7.0
ANALYSIS_MAX_MW = 600.0
ANALYSIS_MAX_HITS = 50
BINDING_TOP_HITS = 30
DOCKING_TIMEOUT = 300  # seconds
DEFAULT_VINA_SETTINGS = {
    "exhaustiveness": 8,
    "num_modes": 9,
    "energy_range": 3
}
DEFAULT_VINA_EXEC = shutil.which("vina") or "vina"

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
        # out_dir = item["out_dir"]
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
        quality_pockets = [
            pocket for pocket in pockets
            if is_high_quality_pocket(
                pocket,
                min_score=POCKET_MIN_SCORE,
                min_probability=POCKET_MIN_PROBABILITY,
                min_sas_points=POCKET_MIN_SAS_POINTS,
            )
        ]

        if not quality_pockets:
            print(
                f"[node_load_pockets] {pdb_id}: no pockets met quality thresholds "
                f"(score>={POCKET_MIN_SCORE}, prob>={POCKET_MIN_PROBABILITY}, sas>={POCKET_MIN_SAS_POINTS})."
            )
            continue

        all_pockets.extend(quality_pockets)

        print(
            f"[node_load_pockets] {pdb_id}: {len(quality_pockets)} high-quality pockets "
            "will be considered for docking"
        )

    if not all_pockets:
        print("[node_load_pockets] WARNING: No pockets found for ANY structure. Downstream docking may fail.")

    state["docking_pockets"] = all_pockets
    return state

def node_prepare_proteins(state: AgentState) -> AgentState:
    """
    Prepare proteins (structures) for docking.
    """
    from helpers.docking import _find_structure_file, _convert_cif_to_pdb, _clean_protein_structure, _call_prepare_receptor
    gene_name = state.get("gene") or "unknown"
    docking_pockets = state.get("docking_pockets", [])
    if not docking_pockets:
        print("[node_prepare_proteins] WARNING: No docking pockets available")
        state["prepared_proteins"] = []
        return state

    # Map structures by pdb_id
    structure_map: Dict[str, Dict] = {}
    for entry in state.get("pocket_results", []):
        pdb_id = entry.get("pdb_id")
        if pdb_id:
            structure_map[pdb_id] = entry

    prepared_entries: List[Dict] = []
    processed_ids: set[str] = set()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROTEIN_PREP_DIR / gene_name / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    for pocket in docking_pockets:
        pdb_id = pocket.get("pdb_id")
        if not pdb_id or pdb_id in processed_ids:
            continue

        struct_entry = structure_map.get(pdb_id)
        if not struct_entry:
            print(f"[node_prepare_proteins] No structure info for {pdb_id}")
            continue

        raw_file = _find_structure_file(struct_entry)
        if raw_file is None:
            print(f"[node_prepare_proteins] Missing structure file for {pdb_id}")
            continue

        out_dir = run_dir / pdb_id
        out_dir.mkdir(parents=True, exist_ok=True)

        pdb_source = raw_file
        if raw_file.suffix.lower() == ".cif":
            pdb_path = _convert_cif_to_pdb(raw_file, out_dir)
            if pdb_path is None:
                prepared_entries.append({
                    "pdb_id": pdb_id,
                    "status": "failed",
                    "raw_file": str(raw_file),
                    "error": "cif_to_pdb_failed"
                })
                continue
            pdb_source = pdb_path

        cleaned_path = _clean_protein_structure(pdb_source, out_dir)
        if cleaned_path is None:
            prepared_entries.append({
                "pdb_id": pdb_id,
                "status": "failed",
                "raw_file": str(raw_file),
                "error": "clean_failed"
            })
            continue

        pdbqt_path = out_dir / "protein.pdbqt"
        log = _call_prepare_receptor(cleaned_path, pdbqt_path)

        status = "success" if pdbqt_path.exists() else "failed"
        prepared_entries.append({
            "pdb_id": pdb_id,
            "status": status,
            "raw_file": str(raw_file),
            "cleaned_pdb": str(cleaned_path),
            "pdbqt": str(pdbqt_path) if pdbqt_path.exists() else None,
            "prepare_log": log,
            "pocket_centers": [p.get("center") for p in docking_pockets if p.get("pdb_id") == pdb_id],
            "output_dir": str(out_dir)
        })
        processed_ids.add(pdb_id)

    state["protein_prep_dir"] = str(run_dir)
    state["prepared_proteins"] = prepared_entries
    return state


# def _load_ligand_run(run_dir: Path) -> Optional[Tuple[List[Dict], List[Dict]]]:
#     if not run_dir or not run_dir.exists():
#         return None

#     prepared: List[Dict] = []
#     for result_file in sorted(run_dir.glob("*/result.json")):
#         try:
#             with open(result_file, "r") as fh:
#                 data = json.load(fh)
#             if not isinstance(data, dict):
#                 continue
#             ligand_dir = result_file.parent
#             data.setdefault("output_dir", str(ligand_dir))
#             prepared.append(data)
#         except Exception as exc:
#             print(f"[_load_ligand_run] Failed to read {result_file}: {exc}")

#     failures: List[Dict] = []
#     failure_file = run_dir / "failures.json"
#     if failure_file.exists():
#         try:
#             with open(failure_file, "r") as fh:
#                 payload = json.load(fh)
#             if isinstance(payload, list):
#                 failures = payload
#         except Exception as exc:
#             print(f"[_load_ligand_run] Failed to read {failure_file}: {exc}")

#     if prepared:
#         return prepared, failures
#     return None


def node_prepare_ligands(state: AgentState) -> AgentState:
    """Prepare ligand set for docking using Open Babel."""
    gene_name = state.get("gene") or "unknown"

    # If ligands already prepared in state, reuse them directly
    if state.get("prepared_ligands"):
        return state

    # Attempt to reuse the most recent prepared ligands on disk
    gene_dir = LIGAND_PREP_DIR / gene_name
    if gene_dir.exists():
        prior_runs = sorted([d for d in gene_dir.iterdir() if d.is_dir()], reverse=True)
        for prev_dir in prior_runs:
            loaded = _load_ligand_run(prev_dir)
            if loaded:
                prepared_ligands, ligand_failures = loaded
                state["ligand_prep_dir"] = str(prev_dir)
                state["prepared_ligands"] = prepared_ligands
                state["ligand_failures"] = ligand_failures
                print(f"[node_prepare_ligands] Reusing ligands from {prev_dir}")
                return state

    csv_sources: List[Path] = []
    root_csv = LIGAND_LIBRARY_DIR / "compounds.csv"
    if root_csv.exists():
        csv_sources.append(root_csv)
    elif LIGAND_LIBRARY_DIR.exists():
        library_dirs = [
            d for d in LIGAND_LIBRARY_DIR.iterdir()
            if d.is_dir() and d.name.startswith("library_")
        ]
        library_dirs.sort(reverse=True)
        for directory in library_dirs:
            csv_sources.append(directory / "compounds.csv")

    ligand_entries: List[Dict] = []
    for csv_path in csv_sources:
        records = _load_ligand_records(csv_path)
        if records:
            ligand_entries = records
            state["ligand_library"] = ligand_entries
            state["ligand_library_csv"] = str(csv_path)
            break

    if not ligand_entries:
        print("[node_prepare_ligands] No ligand records available")
        state["prepared_ligands"] = []
        state["ligand_failures"] = []
        return state

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LIGAND_PREP_DIR / gene_name / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    prepared: List[Dict] = []
    failures: List[Dict] = []
    used_names: set[str] = set()

    for idx, entry in enumerate(ligand_entries):
        raw_name = entry.get("Name") or entry.get("name") or entry.get("ID") or entry.get("id") or entry.get("SMILES")
        lig_name = _slugify_name(raw_name, fallback=f"ligand_{idx:04d}")
        if lig_name in used_names:
            lig_name = f"{lig_name}_{idx}"
        used_names.add(lig_name)

        smiles = entry.get("SMILES") or entry.get("smiles")
        mol = _generate_conformer(smiles)
        if mol is None:
            failures.append({
                "name": lig_name,
                "reason": "smiles_to_3d_failed",
                "smiles": smiles
            })
            continue

        ligand_dir = run_dir / lig_name
        ligand_dir.mkdir(parents=True, exist_ok=True)
        pdb_path = ligand_dir / "ligand.pdb"
        try:
            Chem.MolToPDBFile(mol, str(pdb_path))
        except Exception as exc:
            failures.append({
                "name": lig_name,
                "reason": f"pdb_write_failed: {exc}",
                "smiles": smiles
            })
            continue

        pdbqt_path = ligand_dir / "ligand.pdbqt"
        log = _call_prepare_ligand(pdb_path, pdbqt_path, LIGAND_PREP_TIMEOUT)
        pdbqt_valid = pdbqt_path.exists() and pdbqt_path.stat().st_size > 0
        status = "success" if pdbqt_valid else "failed"
        if status == "failed":
            if pdbqt_path.exists():
                try:
                    pdbqt_path.unlink()
                except Exception:
                    pass
            failures.append({
                "name": lig_name,
                "reason": "pdbqt_generation_failed",
                "log": log
            })

        prepared.append({
            "name": lig_name,
            "status": status,
            "smiles": smiles,
            "source": entry.get("Source") or entry.get("source"),
            "metadata": entry,
            "pdb": str(pdb_path),
            "pdbqt": str(pdbqt_path) if pdbqt_valid else None,
            "prepare_log": log,
            "output_dir": str(ligand_dir)
        })

        ckpt_path = ligand_dir / "result.json"
        try:
            with open(ckpt_path, "w") as fh:
                json.dump(prepared[-1], fh, indent=2)
        except Exception as exc:
            print(f"[node_prepare_ligands] Failed to write checkpoint for {lig_name}: {exc}")

    state["ligand_prep_dir"] = str(run_dir)
    state["prepared_ligands"] = prepared
    state["ligand_failures"] = failures

    failures_path = run_dir / "failures.json"
    try:
        with open(failures_path, "w") as fh:
            json.dump(failures, fh, indent=2)
    except Exception as exc:
        print(f"[node_prepare_ligands] Failed to write failures checkpoint: {exc}")
    return state


def node_plan_docking(state: AgentState) -> AgentState:
    """Generate docking job definitions pairing receptors, ligands, and pockets."""
    prepared_proteins = state.get("prepared_proteins") or []
    prepared_ligands = state.get("prepared_ligands") or []
    pockets = state.get("docking_pockets") or []

    if not prepared_proteins:
        print("[node_plan_docking] No prepared proteins available")
        state["docking_jobs"] = []
        return state

    if not prepared_ligands:
        print("[node_plan_docking] No prepared ligands available")
        state["docking_jobs"] = []
        return state

    pocket_map = _group_pockets_by_pdb(pockets)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gene_name = state.get("gene") or "unknown"
    # root_dir = DOCKING_OUTPUT_DIR / gene_name / timestamp
    root_dir = DOCKING_OUTPUT_DIR / gene_name
    root_dir.mkdir(parents=True, exist_ok=True)

    docking_jobs: List[Dict] = []

    for protein_entry in prepared_proteins:
        pdb_id = protein_entry.get("pdb_id")
        receptor_pdbqt = protein_entry.get("pdbqt")
        if not pdb_id or not receptor_pdbqt:
            continue

        pocket_list = pocket_map.get(pdb_id)
        if not pocket_list:
            print(f"[node_plan_docking] No pockets found for prepared protein {pdb_id}")
            continue

        receptor_dir = root_dir / pdb_id
        receptor_dir.mkdir(parents=True, exist_ok=True)

        for pocket in pocket_list:
            pocket_id = pocket.get("pocket_id") or f"pocket_{pocket.get('rank', 'NA')}"
            rank = pocket.get("rank")
            center = pocket.get("center") or pocket.get("raw_row", {}).get("center")
            if not center or len(center) != 3:
                print(f"[node_plan_docking] Pocket {pocket_id} lacks center coordinates, skip")
                continue

            box_size = list(DEFAULT_BOX_SIZE)

            pocket_dir = receptor_dir / pocket_id
            pocket_dir.mkdir(parents=True, exist_ok=True)

            for ligand in prepared_ligands:
                if ligand.get("status") != "success" or not ligand.get("pdbqt"):
                    continue

                job_id = f"{pdb_id}_{pocket_id}_{ligand['name']}"
                job_dir = pocket_dir / ligand["name"]
                job_dir.mkdir(parents=True, exist_ok=True)

                job = {
                    "job_id": job_id,
                    "gene": gene_name,
                    "pdb_id": pdb_id,
                    "pocket_id": pocket_id,
                    "pocket_rank": rank,
                    "receptor_pdbqt": receptor_pdbqt,
                    "ligand_pdbqt": ligand["pdbqt"],
                    "ligand_name": ligand["name"],
                    "center": center,
                    "box_size": box_size,
                    "residues": pocket.get("residues"),
                    "pocket_metadata": pocket,
                    "job_dir": str(job_dir)
                }

                config_path = job_dir / "job.json"
                with open(config_path, "w") as fh:
                    json.dump(job, fh, indent=2)
                job["job_config"] = str(config_path)

                docking_jobs.append(job)

    if not docking_jobs:
        print("[node_plan_docking] No docking jobs were generated")

    state["docking_jobs"] = docking_jobs
    state["docking_root_dir"] = str(root_dir)
    return state


def node_run_docking(state: AgentState) -> AgentState:
    """Execute AutoDock Vina for each generated docking job."""
    docking_jobs = state.get("docking_jobs") or []
    if not docking_jobs:
        print("[node_run_docking] No docking jobs found")
        state["docking_results"] = []
        return state

    vina_exec = DEFAULT_VINA_EXEC
    settings = DEFAULT_VINA_SETTINGS.copy()
    timeout = DOCKING_TIMEOUT

    cpu_count = os.cpu_count() or 1
    default_cpu_per_job = min(8, cpu_count)
    env_cpu = os.environ.get("DOCKING_CPU_PER_JOB")
    try:
        cpu_per_job = int(env_cpu) if env_cpu else default_cpu_per_job
    except ValueError:
        cpu_per_job = default_cpu_per_job
    cpu_per_job = max(1, min(cpu_per_job, cpu_count))
    settings["cpu"] = cpu_per_job

    env_workers = os.environ.get("DOCKING_WORKERS")
    try:
        requested_workers = int(env_workers) if env_workers else None
    except ValueError:
        requested_workers = None

    max_workers = max(1, cpu_count // cpu_per_job)
    if requested_workers is not None:
        workers = max(1, min(requested_workers, len(docking_jobs), max_workers))
    else:
        workers = max(1, min(len(docking_jobs), max_workers))

    results: List[Optional[Dict]] = [None] * len(docking_jobs)
    success_count = 0
    total_jobs = len(docking_jobs)

    if workers == 1:
        for idx, job in enumerate(docking_jobs, start=1):
            print(f"[node_run_docking] ({idx}/{total_jobs}) Running job {job.get('job_id')} on 1 worker")
            ligand_path = _resolve_path(job.get("ligand_pdbqt", ""))
            if not ligand_path.exists() or ligand_path.stat().st_size == 0:
                res = {
                    "job_id": job.get("job_id"),
                    "status": "skipped_missing_ligand",
                    "stderr": f"Ligand PDBQT missing or empty: {ligand_path}"
                }
            else:
                res = _run_vina_job(job, vina_exec, settings, timeout, allow_resume=True, cpu=cpu_per_job)
            results[idx - 1] = res
            if res.get("status") == "success":
                success_count += 1
            else:
                print(f"[node_run_docking] Job {job.get('job_id')} finished with status {res.get('status')}")
    else:
        print(f"[node_run_docking] Using {workers} workers, {cpu_per_job} CPU(s) per job")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {}
            for idx, job in enumerate(docking_jobs):
                job_id = job.get("job_id")
                ligand_path = _resolve_path(job.get("ligand_pdbqt", ""))
                if not ligand_path.exists() or ligand_path.stat().st_size == 0:
                    results[idx] = {
                        "job_id": job_id,
                        "status": "skipped_missing_ligand",
                        "stderr": f"Ligand PDBQT missing or empty: {ligand_path}"
                    }
                    print(f"[node_run_docking] Job {job_id} skipped due to missing ligand")
                    continue
                future = executor.submit(_run_vina_job, job, vina_exec, settings, timeout, True, cpu_per_job)
                future_map[future] = idx

            for future in concurrent.futures.as_completed(future_map):
                idx = future_map[future]
                job = docking_jobs[idx]
                try:
                    res = future.result()
                except Exception as exc:
                    res = {
                        "job_id": job.get("job_id"),
                        "status": "error",
                        "stderr": f"Concurrent execution failed: {exc}"
                    }
                results[idx] = res
                if res.get("status") == "success":
                    success_count += 1
                else:
                    print(f"[node_run_docking] Job {job.get('job_id')} finished with status {res.get('status')}")

    print(f"[node_run_docking] Completed {len(docking_jobs)} jobs, successes: {success_count}")
    state["docking_results"] = [res for res in results if res is not None]
    return state


def node_analyze_results(state: AgentState) -> AgentState:
    """Perform post-docking analysis: pocket ranking and top ligand selection."""
    docking_jobs = state.get("docking_jobs") or []
    docking_results = state.get("docking_results") or []

    if not docking_jobs or not docking_results:
        print("[node_analyze_results] No docking jobs or results available; skipping analysis")
        return state

    job_index = {
        job.get("job_id"): job
        for job in docking_jobs if job.get("job_id")
    }
    if not job_index:
        print("[node_analyze_results] No valid job metadata found; skipping analysis")
        return state

    ligand_meta_index: Dict[str, Dict] = {}
    for entry in state.get("prepared_ligands") or []:
        name = entry.get("name")
        if not name:
            continue
        ligand_meta_index[name] = entry

    pains_catalog = create_pains_catalog()

    try:
        pocket_stats = analyze_pocket_hits(
            docking_results,
            job_index,
            ligand_meta_index,
            score_cutoff=ANALYSIS_SCORE_CUTOFF,
        )
    except Exception as exc:
        print(f"[node_analyze_results] Pocket analysis failed: {exc}")
        pocket_stats = []

    try:
        ligand_hits = select_top_ligands(
            docking_results,
            job_index,
            ligand_meta_index,
            pains_catalog,
            score_cutoff=ANALYSIS_SCORE_CUTOFF,
            max_mw=ANALYSIS_MAX_MW,
            max_hits=ANALYSIS_MAX_HITS,
        )
    except Exception as exc:
        print(f"[node_analyze_results] Ligand selection failed: {exc}")
        ligand_hits = {}

    docking_root = state.get("docking_root_dir")
    gene_name = state.get("gene") or "unknown"
    if docking_root:
        analysis_dir = Path(docking_root) / "analysis"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = DOCKING_OUTPUT_DIR / gene_name / "analysis" / timestamp

    try:
        report_paths = write_analysis_report(analysis_dir, pocket_stats, ligand_hits)
    except Exception as exc:
        print(f"[node_analyze_results] Failed to write analysis report: {exc}")
        report_paths = {}

    state["pocket_analysis"] = pocket_stats
    state["top_ligand_hits"] = ligand_hits
    state.setdefault("reports", {})
    state["reports"].update({"analysis": {k: str(v) for k, v in report_paths.items()}})
    print(f"[node_analyze_results] Analysis complete. Reports: {report_paths}")
    return state


def node_rescore_refine(state: AgentState) -> AgentState:
    """Rescore and refine top hits using advanced scoring functions."""
    docking_jobs = state.get("docking_jobs") or []
    docking_results = state.get("docking_results") or []
    
    if not docking_results:
        print("[node_rescore_refine] No docking results available for rescoring")
        return state
    
    # Get true top hits across all jobs/pockets (not filtered by ligand)
    successful_results = [
        r for r in docking_results 
        if r.get("status") == "success" and r.get("score") is not None
    ]
    successful_results.sort(key=lambda x: x["score"])  # Sort by score (most negative first)
    
    # Convert to the format expected by enhanced_rescore_and_refine
    top_hits = []
    for result in successful_results[:BINDING_TOP_HITS]:
        job = next((j for j in docking_jobs if j.get("job_id") == result.get("job_id")), None)
        if job:
            hit = {
                "name": job.get("ligand_name"),
                "score": result.get("score"),
                "pocket_id": job.get("pocket_id"),
                "pdb_id": job.get("pdb_id"),
                "job_id": result.get("job_id")
            }
            top_hits.append(hit)
    
    if not top_hits:
        print("[node_rescore_refine] No valid top hits found for rescoring")
        return state
    
    # Create job index
    job_index = {
        job.get("job_id"): job
        for job in docking_jobs
        if job.get("job_id")
    }
    
    # Determine output directory
    docking_root = state.get("docking_root_dir")
    gene_name = state.get("gene") or "unknown"
    if docking_root:
        rescore_dir = Path(docking_root) / "rescoring"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rescore_dir = DOCKING_OUTPUT_DIR / gene_name / "rescoring" / timestamp
    
    # Select rescoring methods based on available tools
    methods = ["vina_rescore"]
    
    try:
        rescoring_results = enhanced_rescore_and_refine(
            top_hits[:BINDING_TOP_HITS],  # Limit to top 30 for efficiency
            job_index,
            docking_results,
            rescore_dir,
            methods=methods,
            do_minimization=False,  # Disabled - replaced by MD simulation
            do_clustering=True
        )
        
        state["rescoring_results"] = rescoring_results
        state.setdefault("reports", {})
        state["reports"]["rescoring"] = rescoring_results.get("summary_file", "")
        
        print(f"[node_rescore_refine] Rescoring complete for {len(top_hits[:BINDING_TOP_HITS])} hits")
        
    except Exception as exc:
        print(f"[node_rescore_refine] Rescoring failed: {exc}")
        state["rescoring_results"] = {"error": str(exc)}
    
    return state


def node_md_simulation(state: AgentState) -> AgentState:
    """Run molecular dynamics simulations on top docking hits."""
    rescoring_results = state.get("rescoring_results", {})
    rescoring_summary = rescoring_results.get("rescoring_summary", {})
    
    if not rescoring_results or "results" not in rescoring_summary:
        print("[node_md_simulation] No rescoring results available for MD simulation")
        return state
    
    # Get validated hits from rescoring
    validated_hits = [
        hit for hit in rescoring_summary["results"] 
        if hit.get("is_validated_hit", False)
    ]
    
    if not validated_hits:
        print("[node_md_simulation] No validated hits found for MD simulation")
        return state
    
    # Select top hits for MD simulation (configurable limit)
    max_md_hits = state.get("max_md_hits", 5)  # Default to 5, but configurable
    top_md_hits = sorted(validated_hits, key=lambda x: x.get("validation_score", 0), reverse=True)[:max_md_hits]
    
    # Determine output directory
    docking_root = state.get("docking_root_dir")
    gene_name = state.get("gene") or "unknown"
    if docking_root:
        md_dir = Path(docking_root) / "md_simulations"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_dir = DOCKING_OUTPUT_DIR / gene_name / "md_simulations" / timestamp
    
    md_dir.mkdir(parents=True, exist_ok=True)
    # Run MD simulations
    try:
        from helpers.md_simulation import run_protein_ligand_md_batch
        
        print(f"[node_md_simulation] Running MD simulations on {len(top_md_hits)} validated hits")
        md_results = run_protein_ligand_md_batch(
            md_hits=md_hits,
            output_dir=md_dir,
            state=state,
            simulation_time_ns=state.get("md_simulation_time_ns", 0.05),
            temperature=state.get("md_temperature", 300.0)
        )
        
        state["md_simulation_results"] = md_results
        state.setdefault("reports", {})
        state["reports"]["md_simulation"] = str(md_dir / "md_simulation_report.json")
        
        print(f"[node_md_simulation] MD simulations complete for {len(top_md_hits)} hits")
        
    except ImportError:
        print("[node_md_simulation] MD simulation module not available - skipping")
        state["md_simulation_results"] = {"status": "skipped", "reason": "MD module not available"}
    except Exception as exc:
        print(f"[node_md_simulation] MD simulation failed: {exc}")
        state["md_simulation_results"] = {"error": str(exc)}
    
    return state


def node_binding_analysis(state: AgentState) -> AgentState:
    """Analyze binding modes and generate visualizations."""
    top_hits = state.get("top_ligand_hits", {}).get("top_hits", [])
    docking_jobs = state.get("docking_jobs") or []
    docking_results = state.get("docking_results") or []
    docking_pockets = state.get("docking_pockets") or []
    
    if not top_hits:
        print("[node_binding_analysis] No top hits available for binding analysis")
        return state
    
    # Create job index
    job_index = {
        job.get("job_id"): job
        for job in docking_jobs
        if job.get("job_id")
    }
    
    # Create pocket residues map
    pocket_residues_map = {}
    for pocket in docking_pockets:
        pdb_id = pocket.get("pdb_id")
        pocket_id = pocket.get("pocket_id")
        residues = pocket.get("residues", [])
        if pdb_id and pocket_id and residues:
            pocket_residues_map[(pdb_id, pocket_id)] = residues
    
    # Determine output directory
    docking_root = state.get("docking_root_dir")
    gene_name = state.get("gene") or "unknown"
    if docking_root:
        analysis_dir = Path(docking_root) / "binding_analysis"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_dir = DOCKING_OUTPUT_DIR / gene_name / "binding_analysis" / timestamp
    
    try:
        binding_results = analyze_binding_modes(
            top_hits[:BINDING_TOP_HITS],  # Limit to top 30 for efficiency
            job_index,
            docking_results,
            analysis_dir,
            gene_name,
            pocket_residues_map
        )
        
        state["binding_mode_analysis"] = binding_results
        state.setdefault("reports", {})
        state["reports"]["binding_analysis"] = binding_results.get("summary_file", "")
        
        print(f"[node_binding_analysis] Binding mode analysis complete for {len(top_hits[:BINDING_TOP_HITS])} hits")
        
        # Generate summary by source
        by_source = binding_results.get("binding_mode_summary", {}).get("by_source", {})
        if by_source:
            print(f"[node_binding_analysis] Hits by source: {by_source}")
        
    except Exception as exc:
        print(f"[node_binding_analysis] Binding analysis failed: {exc}")
        state["binding_mode_analysis"] = {"error": str(exc)}
    
    return state