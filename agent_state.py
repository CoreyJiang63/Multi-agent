#!/usr/bin/env python3
from typing import TypedDict, Optional, List, Dict, Any

class AgentState(TypedDict, total=False):
    gene: str
    candidates: List[Dict]
    chosen: List[Dict]
    structure_files: List[Dict]
    pocket_results: List[Dict]
    docking_pockets: List[Any]
    ligand_library: List[Any]
    
    summary_file: str
    
    