#!/usr/bin/env python3
"""
ADMET Subagent

This module provides nodes for ADMET (Absorption, Distribution, Metabolism, 
Excretion, Toxicity) prediction on top docking hits.

Usage:
    Integrated into the main agent pipeline after docking analysis to predict
    ADMET properties of the most promising compounds.
"""

import os
import json
import time
from typing import TypedDict, Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from agent_state import AgentState
from helpers.admet_prediction import (
    predict_admet_properties,
    batch_admet_prediction,
    calculate_molecular_descriptors,
    calculate_overall_admet_score
)

# ADMET prediction constants
ADMET_OUTPUT_DIR = Path("ADMET")
TOP_HITS_FOR_ADMET = 50  # Number of top hits to analyze
ADMET_SCORE_THRESHOLD = 0.3  # Minimum ADMET score for favorable compounds

def save_admet_json_outputs(state: AgentState, output_dir: Path) -> None:
    """
    Helper function to save ADMET results as separate JSON files.
    
    This function extracts data from the state and saves it as individual JSON files
    for different aspects of ADMET analysis:
    - Raw predictions
    - Analysis summary
    - Favorable compounds
    - Property statistics
    - Comprehensive report
    
    Args:
        state: AgentState containing ADMET results
        output_dir: Directory to save JSON files
    """
    print(f"[save_admet_json_outputs] Saving ADMET JSON outputs to {output_dir}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data from state
    admet_predictions = state.get("admet_predictions", {})
    admet_analysis = state.get("admet_analysis", {})
    favorable_compounds = state.get("favorable_compounds", [])
    gene_name = state.get("gene", "unknown")
    timestamp = datetime.now().isoformat()
    
    # 1. Raw ADMET Predictions
    if admet_predictions:
        predictions_file = output_dir / "raw_admet_predictions.json"
        predictions_data = {
            "metadata": {
                "gene_target": gene_name,
                "timestamp": timestamp,
                "description": "Raw ADMET predictions for all compounds"
            },
            "summary": admet_predictions.get("summary", {}),
            "successful_predictions": admet_predictions.get("successful_predictions", []),
            "failed_predictions": admet_predictions.get("failed_predictions", [])
        }
        
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        print(f"[save_admet_json_outputs] Saved raw predictions to {predictions_file}")
    
    # 2. ADMET Analysis Summary
    if admet_analysis:
        analysis_file = output_dir / "admet_analysis_summary.json"
        analysis_data = {
            "metadata": {
                "gene_target": gene_name,
                "timestamp": timestamp,
                "description": "Statistical analysis of ADMET properties"
            },
            "total_analyzed": admet_analysis.get("total_analyzed", 0),
            "favorable_count": admet_analysis.get("favorable_count", 0),
            "favorable_percentage": admet_analysis.get("favorable_percentage", 0),
            "statistics": admet_analysis.get("statistics", {}),
            "property_distributions": admet_analysis.get("property_distributions", {})
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"[save_admet_json_outputs] Saved analysis summary to {analysis_file}")
    
    # 3. Favorable Compounds
    if favorable_compounds:
        favorable_file = output_dir / "favorable_compounds.json"
        favorable_data = {
            "metadata": {
                "gene_target": gene_name,
                "timestamp": timestamp,
                "description": "Compounds with favorable ADMET profiles",
                "selection_criteria": {
                    "admet_score_threshold": ADMET_SCORE_THRESHOLD,
                    "qed_threshold": 0.3,
                    "max_lipinski_violations": 1
                }
            },
            "count": len(favorable_compounds),
            "compounds": favorable_compounds
        }
        
        with open(favorable_file, 'w') as f:
            json.dump(favorable_data, f, indent=2)
        print(f"[save_admet_json_outputs] Saved favorable compounds to {favorable_file}")
    
    # 4. Top Compounds (Top 10 by combined score)
    if favorable_compounds:
        top_compounds_file = output_dir / "top_compounds.json"
        top_10 = favorable_compounds[:10]
        
        top_compounds_data = {
            "metadata": {
                "gene_target": gene_name,
                "timestamp": timestamp,
                "description": "Top 10 compounds ranked by combined docking and ADMET scores"
            },
            "ranking_method": "Combined score (40% docking + 60% ADMET)",
            "compounds": top_10
        }
        
        with open(top_compounds_file, 'w') as f:
            json.dump(top_compounds_data, f, indent=2)
        print(f"[save_admet_json_outputs] Saved top compounds to {top_compounds_file}")
    
    # 5. Property Statistics
    if admet_analysis and "statistics" in admet_analysis:
        stats_file = output_dir / "property_statistics.json"
        stats_data = {
            "metadata": {
                "gene_target": gene_name,
                "timestamp": timestamp,
                "description": "Statistical summary of molecular properties"
            },
            "statistics": admet_analysis["statistics"],
            "distributions": admet_analysis.get("property_distributions", {})
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"[save_admet_json_outputs] Saved property statistics to {stats_file}")
    
    # 6. Compound Details (Individual files for each compound)
    if admet_predictions and "successful_predictions" in admet_predictions:
        compounds_dir = output_dir / "individual_compounds"
        compounds_dir.mkdir(exist_ok=True)
        
        for compound in admet_predictions["successful_predictions"]:
            compound_name = compound.get("compound_name", "unknown")
            # Sanitize filename
            safe_name = "".join(c for c in compound_name if c.isalnum() or c in ('-', '_')).rstrip()
            compound_file = compounds_dir / f"{safe_name}_details.json"
            
            compound_data = {
                "metadata": {
                    "compound_name": compound_name,
                    "gene_target": gene_name,
                    "timestamp": timestamp,
                    "description": f"Detailed ADMET profile for {compound_name}"
                },
                "compound_info": {
                    "name": compound_name,
                    "smiles": compound.get("smiles"),
                    "source": compound.get("source"),
                    "docking_score": compound.get("docking_score"),
                    "pocket_id": compound.get("pocket_id"),
                    "pdb_id": compound.get("pdb_id")
                },
                "admet_profile": compound.get("admet_data", {}),
                "molecular_descriptors": compound.get("molecular_descriptors", {}),
                "drug_likeness": compound.get("drug_likeness", {}),
                "absorption": compound.get("absorption", {}),
                "distribution": compound.get("distribution", {}),
                "metabolism": compound.get("metabolism", {}),
                "toxicity": compound.get("toxicity", {}),
                "overall_score": compound.get("overall_admet_score", {})
            }
            
            with open(compound_file, 'w') as f:
                json.dump(compound_data, f, indent=2)
        
        print(f"[save_admet_json_outputs] Saved {len(admet_predictions['successful_predictions'])} individual compound files")
    
    # 7. Summary Index
    index_file = output_dir / "index.json"
    index_data = {
        "metadata": {
            "gene_target": gene_name,
            "timestamp": timestamp,
            "description": "Index of all ADMET analysis files"
        },
        "files": {
            "raw_predictions": "raw_admet_predictions.json",
            "analysis_summary": "admet_analysis_summary.json",
            "favorable_compounds": "favorable_compounds.json",
            "top_compounds": "top_compounds.json",
            "property_statistics": "property_statistics.json",
            "individual_compounds": "individual_compounds/",
            "comprehensive_report": "admet_comprehensive_report.json"
        },
        "summary": {
            "total_compounds": len(admet_predictions.get("successful_predictions", [])),
            "favorable_compounds": len(favorable_compounds),
            "analysis_timestamp": timestamp
        }
    }
    
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    print(f"[save_admet_json_outputs] Saved index file to {index_file}")
    
    print(f"[save_admet_json_outputs] All ADMET JSON outputs saved successfully")

def node_extract_top_hits(state: AgentState) -> AgentState:
    """
    Extract top docking hits for ADMET prediction.
    Input:
        state["top_ligand_hits"] = results from docking analysis
        state["rescoring_results"] = results from rescoring (optional)
    Output:
        state["admet_candidates"] = list of compounds for ADMET prediction
    """
    print("[node_extract_top_hits] Extracting top hits for ADMET prediction...")
    
    # Get top hits from different sources
    top_hits = []
    
    # Primary source: top_ligand_hits from docking analysis
    ligand_hits = state.get("top_ligand_hits", {})
    if isinstance(ligand_hits, dict) and "top_hits" in ligand_hits:
        hits_list = ligand_hits["top_hits"]
        if isinstance(hits_list, list):
            top_hits.extend(hits_list[:TOP_HITS_FOR_ADMET])
    
    # Secondary source: rescoring results if available
    rescoring_results = state.get("rescoring_results", {})
    if isinstance(rescoring_results, dict):
        rescoring_summary = rescoring_results.get("rescoring_summary", {})
        if "results" in rescoring_summary:
            validated_hits = [
                hit for hit in rescoring_summary["results"]
                if hit.get("is_validated_hit", False)
            ]
            # Add validated hits that aren't already in top_hits
            existing_names = {hit.get("name", "") for hit in top_hits}
            for hit in validated_hits:
                if hit.get("name", "") not in existing_names:
                    top_hits.append(hit)
    
    if not top_hits:
        print("[node_extract_top_hits] No top hits found for ADMET prediction")
        state["admet_candidates"] = []
        return state
    
    # Extract compound information for ADMET prediction
    admet_candidates = []
    prepared_ligands = state.get("prepared_ligands", [])
    ligand_metadata = {lig.get("name"): lig for lig in prepared_ligands}
    
    for hit in top_hits:
        compound_name = hit.get("name") or hit.get("ligand_name")
        if not compound_name:
            continue
        
        # Get SMILES from ligand metadata
        ligand_info = ligand_metadata.get(compound_name, {})
        smiles = ligand_info.get("smiles")
        
        if not smiles:
            print(f"[node_extract_top_hits] No SMILES found for {compound_name}, skipping")
            continue
        
        candidate = {
            "name": compound_name,
            "smiles": smiles,
            "docking_score": hit.get("score") or hit.get("docking_score"),
            "pocket_id": hit.get("pocket_id"),
            "pdb_id": hit.get("pdb_id"),
            "source": ligand_info.get("source"),
            "metadata": ligand_info.get("metadata", {}),
            "hit_info": hit
        }
        admet_candidates.append(candidate)
    
    print(f"[node_extract_top_hits] Prepared {len(admet_candidates)} compounds for ADMET prediction")
    state["admet_candidates"] = admet_candidates
    return state

def node_predict_admet(state: AgentState) -> AgentState:
    """
    Perform ADMET predictions on candidate compounds.
    Input:
        state["admet_candidates"] = list of compounds with SMILES
    Output:
        state["admet_predictions"] = ADMET prediction results
    """
    admet_candidates = state.get("admet_candidates", [])
    if not admet_candidates:
        print("[node_predict_admet] No ADMET candidates available")
        state["admet_predictions"] = {"results": [], "summary": {}}
        return state
    
    print(f"[node_predict_admet] Running ADMET predictions on {len(admet_candidates)} compounds...")
    
    # Setup output directory
    gene_name = state.get("gene") or "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ADMET_OUTPUT_DIR / gene_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare compounds for batch prediction
    compounds_for_prediction = []
    for candidate in admet_candidates:
        compounds_for_prediction.append({
            "name": candidate["name"],
            "smiles": candidate["smiles"]
        })
    
    try:
        # Run batch ADMET prediction
        batch_results = batch_admet_prediction(compounds_for_prediction, output_dir)
        
        # Merge ADMET results with docking information
        enhanced_results = []
        admet_results = batch_results["summary"]["results"]
        
        # Create lookup for ADMET results
        admet_lookup = {result["compound_name"]: result for result in admet_results}
        
        for candidate in admet_candidates:
            compound_name = candidate["name"]
            admet_data = admet_lookup.get(compound_name, {})
            
            if admet_data and "error" not in admet_data:
                # Combine docking and ADMET data
                enhanced_result = {
                    "compound_name": compound_name,
                    "smiles": candidate["smiles"],
                    "docking_score": candidate.get("docking_score"),
                    "pocket_id": candidate.get("pocket_id"),
                    "pdb_id": candidate.get("pdb_id"),
                    "source": candidate.get("source"),
                    "admet_data": admet_data,
                    "molecular_descriptors": admet_data.get("molecular_descriptors", {}),
                    "drug_likeness": {
                        "qed_score": admet_data.get("qed_score"),
                        "lipinski_violations": admet_data.get("lipinski_violations"),
                        "lipinski_compliant": admet_data.get("lipinski_compliant"),
                        "synthetic_accessibility": admet_data.get("synthetic_accessibility")
                    },
                    "absorption": admet_data.get("absorption", {}),
                    "distribution": admet_data.get("distribution", {}),
                    "metabolism": admet_data.get("metabolism", {}),
                    "toxicity": admet_data.get("toxicity", {}),
                    "overall_admet_score": admet_data.get("overall_admet_score", {}),
                    "timestamp": admet_data.get("timestamp")
                }
                enhanced_results.append(enhanced_result)
            else:
                # Include failed predictions for completeness
                enhanced_result = {
                    "compound_name": compound_name,
                    "smiles": candidate["smiles"],
                    "docking_score": candidate.get("docking_score"),
                    "error": admet_data.get("error", "ADMET prediction failed")
                }
                enhanced_results.append(enhanced_result)
        
        # Create comprehensive summary
        successful_predictions = [r for r in enhanced_results if "error" not in r]
        failed_predictions = [r for r in enhanced_results if "error" in r]
        
        summary = {
            "total_compounds": len(admet_candidates),
            "successful_predictions": len(successful_predictions),
            "failed_predictions": len(failed_predictions),
            "success_rate": round(len(successful_predictions) / len(admet_candidates) * 100, 1) if admet_candidates else 0,
            "output_directory": str(output_dir),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save enhanced results
        enhanced_summary_file = output_dir / "enhanced_admet_results.json"
        enhanced_summary = {
            "summary": summary,
            "results": enhanced_results,
            "successful_predictions": successful_predictions,
            "failed_predictions": failed_predictions
        }
        
        with open(enhanced_summary_file, 'w') as f:
            json.dump(enhanced_summary, f, indent=2)
        
        state["admet_predictions"] = enhanced_summary
        state["admet_output_dir"] = str(output_dir)
        
        # Save separate JSON outputs
        save_admet_json_outputs(state, output_dir)
        
        print(f"[node_predict_admet] ADMET predictions complete: {len(successful_predictions)}/{len(admet_candidates)} successful")
        print(f"[node_predict_admet] Results saved to {enhanced_summary_file}")
        
    except Exception as e:
        print(f"[node_predict_admet] ADMET prediction failed: {e}")
        state["admet_predictions"] = {"error": str(e), "results": []}
    
    return state

def node_analyze_admet_results(state: AgentState) -> AgentState:
    """
    Analyze ADMET results and identify favorable compounds.
    Input:
        state["admet_predictions"] = ADMET prediction results
    Output:
        state["admet_analysis"] = analysis of ADMET results
        state["favorable_compounds"] = compounds with good ADMET profiles
    """
    admet_predictions = state.get("admet_predictions", {})
    if not admet_predictions or "results" not in admet_predictions:
        print("[node_analyze_admet_results] No ADMET predictions to analyze")
        state["admet_analysis"] = {}
        state["favorable_compounds"] = []
        return state
    
    print("[node_analyze_admet_results] Analyzing ADMET results...")
    
    successful_results = admet_predictions.get("successful_predictions", [])
    if not successful_results:
        print("[node_analyze_admet_results] No successful ADMET predictions to analyze")
        state["admet_analysis"] = {}
        state["favorable_compounds"] = []
        return state
    
    # Analyze ADMET profiles
    analysis = {
        "total_analyzed": len(successful_results),
        "property_distributions": {},
        "favorable_compounds": [],
        "unfavorable_compounds": [],
        "statistics": {}
    }
    
    # Collect property values for statistics
    property_values = {
        "qed_scores": [],
        "admet_scores": [],
        "molecular_weights": [],
        "logp_values": [],
        "lipinski_violations": [],
        "absorption_classes": [],
        "toxicity_risks": []
    }
    
    favorable_compounds = []
    unfavorable_compounds = []
    
    for result in successful_results:
        # Extract key properties
        qed_score = result.get("drug_likeness", {}).get("qed_score", 0)
        overall_admet = result.get("overall_admet_score", {})
        admet_score = overall_admet.get("score", 0)
        admet_class = overall_admet.get("classification", "Unknown")
        
        molecular_descriptors = result.get("molecular_descriptors", {})
        mw = molecular_descriptors.get("molecular_weight", 0)
        logp = molecular_descriptors.get("logp", 0)
        
        lipinski_violations = result.get("drug_likeness", {}).get("lipinski_violations", 0)
        absorption_class = result.get("absorption", {}).get("absorption_class", "Unknown")
        
        toxicity = result.get("toxicity", {})
        hepatotox_risk = toxicity.get("hepatotoxicity_risk", "Unknown")
        
        # Collect for statistics
        property_values["qed_scores"].append(qed_score)
        property_values["admet_scores"].append(admet_score)
        property_values["molecular_weights"].append(mw)
        property_values["logp_values"].append(logp)
        property_values["lipinski_violations"].append(lipinski_violations)
        property_values["absorption_classes"].append(absorption_class)
        property_values["toxicity_risks"].append(hepatotox_risk)
        
        # Determine if compound is favorable
        is_favorable = (
            admet_score >= ADMET_SCORE_THRESHOLD and
            qed_score >= 0.3 and
            lipinski_violations <= 1 and
            absorption_class in ["High", "Medium"] and
            hepatotox_risk in ["Low", "Medium"]
        )
        
        compound_summary = {
            "name": result["compound_name"],
            "docking_score": result.get("docking_score"),
            "admet_score": admet_score,
            "admet_classification": admet_class,
            "qed_score": qed_score,
            "lipinski_violations": lipinski_violations,
            "absorption_class": absorption_class,
            "hepatotoxicity_risk": hepatotox_risk,
            "molecular_weight": mw,
            "logp": logp,
            "is_favorable": is_favorable,
            "pocket_id": result.get("pocket_id"),
            "pdb_id": result.get("pdb_id"),
            "source": result.get("source")
        }
        
        if is_favorable:
            favorable_compounds.append(compound_summary)
        else:
            unfavorable_compounds.append(compound_summary)
    
    # Calculate statistics
    import numpy as np
    
    analysis["statistics"] = {
        "qed_score": {
            "mean": round(np.mean(property_values["qed_scores"]), 3),
            "median": round(np.median(property_values["qed_scores"]), 3),
            "std": round(np.std(property_values["qed_scores"]), 3)
        },
        "admet_score": {
            "mean": round(np.mean(property_values["admet_scores"]), 3),
            "median": round(np.median(property_values["admet_scores"]), 3),
            "std": round(np.std(property_values["admet_scores"]), 3)
        },
        "molecular_weight": {
            "mean": round(np.mean(property_values["molecular_weights"]), 1),
            "median": round(np.median(property_values["molecular_weights"]), 1),
            "std": round(np.std(property_values["molecular_weights"]), 1)
        },
        "logp": {
            "mean": round(np.mean(property_values["logp_values"]), 2),
            "median": round(np.median(property_values["logp_values"]), 2),
            "std": round(np.std(property_values["logp_values"]), 2)
        }
    }
    
    # Property distributions
    analysis["property_distributions"] = {
        "lipinski_violations": {str(i): property_values["lipinski_violations"].count(i) for i in range(5)},
        "absorption_classes": {cls: property_values["absorption_classes"].count(cls) for cls in set(property_values["absorption_classes"])},
        "toxicity_risks": {risk: property_values["toxicity_risks"].count(risk) for risk in set(property_values["toxicity_risks"])}
    }
    
    # Sort compounds by combined score (docking + ADMET)
    def combined_score(compound):
        docking_score = compound.get("docking_score")
        admet_score = compound.get("admet_score", 0)
        
        # Handle None docking scores
        if docking_score is None:
            normalized_docking = 0.0
        else:
            # Normalize docking score (more negative is better) and combine with ADMET
            normalized_docking = max(0, (-docking_score + 5) / 10)  # Assuming typical range -15 to 0
        
        return (normalized_docking * 0.4 + admet_score * 0.6)
    
    favorable_compounds.sort(key=combined_score, reverse=True)
    unfavorable_compounds.sort(key=combined_score, reverse=True)
    
    analysis["favorable_compounds"] = favorable_compounds
    analysis["unfavorable_compounds"] = unfavorable_compounds
    analysis["favorable_count"] = len(favorable_compounds)
    analysis["unfavorable_count"] = len(unfavorable_compounds)
    analysis["favorable_percentage"] = round(len(favorable_compounds) / len(successful_results) * 100, 1)
    
    # Save analysis results
    output_dir = Path(state.get("admet_output_dir", ADMET_OUTPUT_DIR))
    analysis_file = output_dir / "admet_analysis.json"
    
    try:
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"[node_analyze_admet_results] Analysis saved to {analysis_file}")
    except Exception as e:
        print(f"[node_analyze_admet_results] Failed to save analysis: {e}")
    
    state["admet_analysis"] = analysis
    state["favorable_compounds"] = favorable_compounds
    
    # Update JSON outputs with analysis results
    save_admet_json_outputs(state, output_dir)
    
    print(f"[node_analyze_admet_results] Analysis complete: {len(favorable_compounds)} favorable compounds identified")
    print(f"[node_analyze_admet_results] Favorable compound rate: {analysis['favorable_percentage']}%")
    
    return state

def node_generate_admet_report(state: AgentState) -> AgentState:
    """
    Generate comprehensive ADMET report.
    Input:
        state["admet_analysis"] = ADMET analysis results
        state["admet_predictions"] = raw ADMET predictions
    Output:
        state["admet_report"] = path to generated report
    """
    admet_analysis = state.get("admet_analysis", {})
    admet_predictions = state.get("admet_predictions", {})
    
    if not admet_analysis or not admet_predictions:
        print("[node_generate_admet_report] No ADMET data available for report generation")
        state["admet_report"] = None
        return state
    
    print("[node_generate_admet_report] Generating ADMET report...")
    
    output_dir = Path(state.get("admet_output_dir", ADMET_OUTPUT_DIR))
    gene_name = state.get("gene", "unknown")
    
    # Generate comprehensive report
    report = {
        "title": f"ADMET Analysis Report - {gene_name}",
        "generated_at": datetime.now().isoformat(),
        "gene_target": gene_name,
        "summary": {
            "total_compounds_analyzed": admet_analysis.get("total_analyzed", 0),
            "favorable_compounds": admet_analysis.get("favorable_count", 0),
            "favorable_percentage": admet_analysis.get("favorable_percentage", 0),
            "success_rate": admet_predictions.get("summary", {}).get("success_rate", 0)
        },
        "statistics": admet_analysis.get("statistics", {}),
        "property_distributions": admet_analysis.get("property_distributions", {}),
        "top_favorable_compounds": admet_analysis.get("favorable_compounds", [])[:10],
        "methodology": {
            "descriptors_calculated": [
                "Molecular Weight", "LogP", "HBD", "HBA", "TPSA", "Rotatable Bonds"
            ],
            "admet_properties": [
                "Drug-likeness (QED)", "Lipinski Rule of Five", "Absorption Prediction",
                "BBB Permeability", "CYP Inhibition", "Toxicity Assessment"
            ],
            "scoring_criteria": {
                "favorable_threshold": ADMET_SCORE_THRESHOLD,
                "qed_threshold": 0.3,
                "max_lipinski_violations": 1
            }
        },
        "recommendations": []
    }
    
    # Add recommendations based on results
    favorable_compounds = admet_analysis.get("favorable_compounds", [])
    if favorable_compounds:
        top_compound = favorable_compounds[0]
        report["recommendations"].append(
            f"Top recommended compound: {top_compound['name']} "
            f"(ADMET score: {top_compound['admet_score']}, "
            f"Docking score: {top_compound['docking_score']})"
        )
        
        if len(favorable_compounds) >= 5:
            report["recommendations"].append(
                f"Multiple promising candidates identified ({len(favorable_compounds)} compounds). "
                "Consider experimental validation of top 5 compounds."
            )
        else:
            report["recommendations"].append(
                "Limited number of favorable compounds. Consider expanding chemical space "
                "or optimizing existing hits."
            )
    else:
        report["recommendations"].append(
            "No compounds met favorable ADMET criteria. Consider lead optimization "
            "or alternative chemical scaffolds."
        )
    
    # Add statistics-based recommendations
    stats = admet_analysis.get("statistics", {})
    if stats:
        mean_mw = stats.get("molecular_weight", {}).get("mean", 0)
        if mean_mw > 500:
            report["recommendations"].append(
                f"Average molecular weight ({mean_mw:.1f} Da) exceeds optimal range. "
                "Consider smaller analogs for better drug-likeness."
            )
        
        mean_logp = stats.get("logp", {}).get("mean", 0)
        if mean_logp > 5:
            report["recommendations"].append(
                f"Average LogP ({mean_logp:.2f}) is high. "
                "Consider adding polar groups to improve solubility."
            )
    
    # Save report
    report_file = output_dir / "admet_comprehensive_report.json"
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        state["admet_report"] = str(report_file)
        
        # Update reports in state
        state.setdefault("reports", {})
        state["reports"]["admet"] = str(report_file)
        
        print(f"[node_generate_admet_report] Comprehensive report generated: {report_file}")
        
    except Exception as e:
        print(f"[node_generate_admet_report] Failed to generate report: {e}")
        state["admet_report"] = None
    
    return state
