#!/usr/bin/env python3
from typing import TypedDict, Optional, List, Dict, Any

class AgentState(TypedDict, total=False):
    gene: str
    
    # MD simulation configuration params
    md_simulation_time_ns: Any
    md_temperature: Any
    max_parallel_md_jobs: Any
    use_md_checkpoints: Any
    max_md_hits: Any
    
    candidates: List[Dict]
    chosen: List[Dict]
    structure_files: List[Dict]
    pocket_results: List[Dict]
    docking_pockets: List[Any]
    
    protein_prep_dir: str
    prepared_proteins: List[Dict]
    ligand_prep_dir: str
    prepared_ligands: List[Dict]
    ligand_failures: List[Dict]
    docking_jobs: List[Dict]
    docking_root_dir: str
    docking_results: List[Dict]
    
    pocket_analysis: List[Dict]
    top_ligand_hits: List[Dict]
    rescoring_results: Dict
    md_simulation_results: Dict
    binding_mode_analysis: Dict
    
    # ADMET prediction fields
    admet_candidates: List[Dict]
    admet_predictions: Dict
    admet_analysis: Dict
    admet_output_dir: str
    admet_report: str
    favorable_compounds: List[Dict]
    
    # Sequence-to-drug generation fields
    protein_sequence: str
    gen_size: int
    generated_molecules: List[Dict]
    sequence_to_drug_results: Dict
    interpretability_report: Dict
    
    reports: Dict
    summary_file: str