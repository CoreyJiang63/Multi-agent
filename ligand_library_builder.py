#!/usr/bin/env python3
"""
Ligand Library Builder for Drug Discovery Pipeline

This script builds a diverse compound library from various open sources
and prepares them for molecular docking with AutoDock Vina.
"""
import os
import sys
import re
import json
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import RDConfig
import pubchempy as pcp
from chembl_webresource_client.new_client import new_client

# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f'ligand_library_builder_{timestamp}.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
CURATED_PUBCHEM_TERMS: Dict[str, Dict[str, List[str]]] = {
    "ESR2": {
        "aliases": [
            "estrogen receptor beta",
            "er beta",
            "esr-2",
            "nr3a2",
            "oestrogen receptor beta"
        ],
        "ligands": [
            "diarylpropionitrile",
            "genistein",
            "liquiritigenin",
            "phytoestrogen",
            "formononetin",
            "pestanal"
        ]
    }
}
LIBRARY_DIR = Path("ligand_libraries")
LIBRARY_DIR.mkdir(exist_ok=True)

class LigandLibraryBuilder:
    """Build and manage a diverse compound library for virtual screening."""
    
    def __init__(self, output_dir: str = "ligand_libraries"):
        """Initialize the library builder with caching."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.compounds = []
        self.sources = {
            'pubchem': self._fetch_from_pubchem,
            'chembl': self._fetch_from_chembl,
            'np_atlas': self._fetch_from_np_atlas,
            'fda_approved': self._fetch_fda_approved
        }
        
        # Setup cache
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Configure requests session with retry and timeout
        self.session = requests.Session()
        retry = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        )
        self.session.mount('http://', retry)
        self.session.mount('https://', retry)
        
        # Configure retry strategy for status codes
        retry_strategy = requests.adapters.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        self.session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry_strategy))
        self.session.mount("http://", requests.adapters.HTTPAdapter(max_retries=retry_strategy))
        
        # Configure RDKit logging
        import rdkit.RDLogger
        rdkit.RDLogger.DisableLog('rdApp.*')

    def _get_chembl_target_id(self, target: str) -> Optional[str]:
        """Get ChEMBL target ID from gene symbol or name."""
        cache_file = self.cache_dir / f"chembl_target_{target}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'chembl_id' in data:
                        return data['chembl_id']
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
                cache_file.unlink()  # Remove invalid cache file
                
        try:
            # First try exact match with target_synonym
            targets = list(new_client.target.filter(
                target_synonym__iexact=target,
                target_type='SINGLE PROTEIN',
                limit=1
            ))
            
            if not targets:
                # Try with target_pref_name
                targets = list(new_client.target.filter(
                    pref_name__iexact=target,
                    target_type='SINGLE PROTEIN',
                    limit=1
                ))
            
            if not targets:
                # Try search as last resort
                search_results = list(new_client.target.search(target))
                if search_results:
                    # Filter for single protein targets and get the first match
                    targets = [t for t in search_results 
                             if t.get('target_type') == 'SINGLE PROTEIN']
            
            if targets:
                # Prefer human targets
                human_targets = [t for t in targets 
                              if t.get('organism') == 'Homo sapiens']
                target_data = human_targets[0] if human_targets else targets[0]
                
                # Cache the result
                try:
                    with open(cache_file, 'w') as f:
                        json.dump({
                            'chembl_id': target_data['target_chembl_id'],
                            'pref_name': target_data.get('pref_name', ''),
                            'organism': target_data.get('organism', '')
                        }, f, indent=2)
                except Exception as e:
                    logger.warning(f"Could not write to cache file {cache_file}: {e}")
                
                return target_data['target_chembl_id']
                
        except Exception as e:
            logger.error(f"Error searching ChEMBL for target {target}: {str(e)}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                logger.debug(f"Response: {e.response.text}")
                
        return None

    def _resolve_target_chembl_ids(self, targets: List[str]) -> List[str]:
        """Resolve input targets into ChEMBL target IDs once."""
        resolved: List[str] = []
        seen: Set[str] = set()

        for target in targets:
            candidate = target.upper()
            if candidate.startswith("CHEMBL"):
                if candidate not in seen:
                    resolved.append(candidate)
                    seen.add(candidate)
                continue

            chembl_id = self._get_chembl_target_id(target)
            if chembl_id:
                if chembl_id not in seen:
                    resolved.append(chembl_id)
                    seen.add(chembl_id)
            else:
                logger.warning(f"Could not resolve ChEMBL ID for target {target}; skipping")

        return resolved

    def _generate_pubchem_search_terms(self, target: str) -> List[str]:
        """Generate search terms for PubChem."""
        base_synonyms = [target]
        if target.upper() != target:
            base_synonyms.append(target.upper())
        
        modifier_suffixes = ["ligand", "inhibitor", "agonist", "antagonist", "modulator"]
        max_terms = 60
        
        terms: List[str] = []
        seen: Set[str] = set()
        
        def add_term(term: Optional[str]) -> None:
            normalized = self._normalize_search_term(term)
            if not normalized:
                return
            key = normalized.lower()
            if key in seen:
                return
            seen.add(key)
            terms.append(normalized)
        
        curated = CURATED_PUBCHEM_TERMS.get(target.upper())
        if curated:
            for alias in curated.get("aliases", []):
                add_term(alias)
            for ligand in curated.get("ligands", []):
                add_term(ligand)
                add_term(f"{ligand} {target}")
        
        for synonym in base_synonyms:
            add_term(synonym)
            if synonym and not synonym.lower().endswith(' receptor'):
                add_term(f"{synonym} receptor")
            if len(terms) >= max_terms:
                break
        
        primary_terms = terms[:10]  # limit combinatorial expansion
        for synonym in primary_terms:
            if len(terms) >= max_terms:
                break
            
            for suffix in modifier_suffixes:
                add_term(f"{synonym} {suffix}")
                if len(terms) >= max_terms:
                    break
        
        return terms[:max_terms]

    def _normalize_search_term(self, term: Optional[str]) -> Optional[str]:
        """Normalize search phrases to improve PubChem recall."""
        if not term or not isinstance(term, str):
            return None
        cleaned = term.strip().strip('"\'')
        if not cleaned:
            return None
        replacements = {
            'β': 'beta',
            'α': 'alpha',
            'γ': 'gamma',
            'δ': 'delta'
        }
        for symbol, text in replacements.items():
            cleaned = cleaned.replace(symbol, text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip()
        return cleaned or None
    
    def _query_pubchem_compounds(self, term: str, namespace: str) -> List[pcp.Compound]:
        """Query PubChem with fallbacks to CID lookups."""
        try:
            hits = pcp.get_compounds(term, namespace, list_return='flat')
            if hits:
                return hits
        except Exception as exc:
            logger.debug(
                "PubChem primary search failed for term '%s' (%s): %s",
                term,
                namespace,
                exc
            )

        try:
            cids = pcp.get_cids(term, namespace)
        except Exception as exc:
            logger.debug(
                "PubChem CID lookup failed for term '%s' (%s): %s",
                term,
                namespace,
                exc
            )
            return []

        compounds: List[pcp.Compound] = []
        for cid in cids:
            if cid is None:
                continue
            try:
                compounds.append(pcp.Compound.from_cid(int(cid)))
            except Exception as exc:
                logger.debug(f"Failed to retrieve PubChem CID {cid}: {exc}")

        return compounds
    
    def _fetch_from_pubchem(self, targets: List[str], max_compounds: int) -> List[Dict]:
        """Fetch compounds from PubChem with better error handling."""
        logger.info(f"Fetching up to {max_compounds} compounds from PubChem")
        compounds: List[Dict] = []
        seen_cids: set[int] = set()
        
        for target in targets:
            if len(compounds) >= max_compounds:
                break
            
            search_terms = self._generate_pubchem_search_terms(target)
            logger.info(
                "PubChem search for %s will use %d terms",
                target,
                len(search_terms)
            )
            
            for term in search_terms:
                if len(compounds) >= max_compounds:
                    break
                
                for namespace in ["name", "synonym"]:
                    if len(compounds) >= max_compounds:
                        break
                    
                    hits = self._query_pubchem_compounds(term, namespace)
                    if not hits:
                        continue
                    logger.debug(
                        "PubChem returned %d hits for '%s' (%s)",
                        len(hits),
                        term,
                        namespace
                    )
                    
                    for hit in hits:
                        if len(compounds) >= max_compounds:
                            break
                        
                        cid = getattr(hit, 'cid', None)
                        smiles = getattr(hit, 'isomeric_smiles', None) or getattr(hit, 'canonical_smiles', None)
                        if not cid or cid in seen_cids or not smiles:
                            continue
                        
                        try:
                            mol = Chem.MolFromSmiles(smiles)
                            if not mol:
                                continue
                            mol = self._prepare_molecule(mol)
                            if not mol:
                                continue
                            
                            compounds.append({
                                'source': 'pubchem',
                                'cid': cid,
                                'smiles': smiles,
                                'name': getattr(hit, 'iupac_name', None) or getattr(hit, 'canonical_name', None) or f"PubChem_{cid}",
                                'mol': mol,
                                'mw': getattr(hit, 'molecular_weight', None) or Descriptors.ExactMolWt(mol)
                            })
                            seen_cids.add(cid)
                        except Exception as exc:
                            logger.debug(f"Failed to process PubChem CID {cid}: {exc}")
                            continue
            
        return compounds[:max_compounds]

    def _fetch_from_chembl(self, targets: List[str], max_compounds: int) -> List[Dict]:
        """Fetch compounds from ChEMBL without caching."""
        logger.info(f"Fetching up to {max_compounds} compounds from ChEMBL")
        compounds: List[Dict] = []
 
        target_chembl_ids = self._resolve_target_chembl_ids(targets)
        if not target_chembl_ids:
            logger.warning("No valid ChEMBL targets resolved; skipping ChEMBL fetch")
            return compounds

        for target_id in target_chembl_ids:
            try:
                activities = new_client.activity.filter(
                    target_chembl_id=target_id,
                    standard_type__in=["IC50", "Ki", "Kd", "EC50"],
                    standard_units__in=["nM", "uM"],
                    standard_relation="=",
                    standard_value__lte=10000,
                    pchembl_value__gte=5,
                    assay_type_iregex='B',
                    data_validity_comment__isnull=True
                ).only([
                    'molecule_chembl_id',
                    'canonical_smiles',
                    'standard_type',
                    'standard_value',
                    'standard_units',
                    'pchembl_value'
                ])
                
                seen: set[str] = set()
                for act in activities:
                    if len(compounds) >= max_compounds:
                        break
                    
                    cid = act.get('molecule_chembl_id')
                    if not cid or cid in seen:
                        continue
                    seen.add(cid)
                    
                    try:
                        mol_data = new_client.molecule.get(cid)
                        if not mol_data or 'molecule_structures' not in mol_data:
                            continue
                        
                        smiles = mol_data['molecule_structures'].get('canonical_smiles')
                        if not smiles:
                            continue
                        
                        rd_mol = Chem.MolFromSmiles(smiles)
                        if not rd_mol:
                            continue
                        
                        rd_mol = self._prepare_molecule(rd_mol)
                        if not rd_mol:
                            continue
                        
                        name = mol_data.get('pref_name') or f"ChEMBL_{cid}"
                        mol_weight = mol_data.get('molecular_weight')
                        if mol_weight is None:
                            mol_weight = Descriptors.ExactMolWt(rd_mol)
                        
                        compounds.append({
                            'source': 'chembl',
                            'chembl_id': cid,
                            'smiles': smiles,
                            'name': name,
                            'mol': rd_mol,
                            'mw': float(mol_weight) if mol_weight is not None else 0.0,
                            'activity': {
                                'type': act.get('standard_type'),
                                'value': act.get('standard_value'),
                                'units': act.get('standard_units'),
                                'pchembl': act.get('pchembl_value')
                            }
                        })
                    except Exception as inner_exc:
                        logger.warning(f"Error processing ChEMBL compound {cid}: {inner_exc}")
                        continue
                
            except Exception as exc:
                logger.error(f"Error searching ChEMBL for {target_id}: {exc}")
                continue
        
        return compounds

    def _fetch_from_np_atlas(self, targets: List[str], max_compounds: int) -> List[Dict]:
        """Fetch natural products from NP Atlas."""
        logger.info(f"Fetching up to {max_compounds} natural products from NP Atlas")
        compounds: List[Dict] = []
        seen_ids: Set[str] = set()
         
        try:
            # NP Atlas API endpoint
            url = "https://www.npatlas.org/api/v1/compounds"
            response = requests.get(url, params={'limit': max_compounds})
            response.raise_for_status()
            
            payload = response.json()
            
            if isinstance(payload, dict):
                candidates = payload.get('results') or payload.get('data') or payload.get('compounds')
                if isinstance(candidates, dict):
                    items = list(candidates.values())
                else:
                    items = candidates if isinstance(candidates, list) else []
            elif isinstance(payload, list):
                items = payload
            else:
                items = []
            
            if not items:
                logger.warning("NP Atlas returned an unexpected payload format; no compounds parsed")
                return compounds
            
            for item in items[:max_compounds]:
                if not isinstance(item, dict):
                    continue
                
                try:
                    smiles = item.get('smiles')
                    if smiles:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol:
                            mol = self._prepare_molecule(mol)
                            if mol:
                                compound_id = str(item.get('npaid') or item.get('id') or '')
                                if compound_id and compound_id in seen_ids:
                                    continue
                                if compound_id:
                                    seen_ids.add(compound_id)
                                compounds.append({
                                    'source': 'np_atlas',
                                    'id': item.get('npaid', ''),
                                    'smiles': smiles,
                                    'name': item.get('name', f"NP_{len(compounds)}"),
                                    'mol': mol,
                                    'mw': Descriptors.ExactMolWt(mol) if mol else 0,
                                    'organism': item.get('organism', '')
                                })
                except Exception as e:
                    logger.warning(f"Error processing NP Atlas compound: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error fetching from NP Atlas: {e}")
            
        return compounds
    
    def _fetch_fda_approved(self, targets: List[str], max_compounds: int) -> List[Dict]:
        """Fetch FDA-approved drugs."""
        logger.info(f"Fetching up to {max_compounds} FDA-approved drugs")
        compounds = []
 
        target_chembl_ids = self._resolve_target_chembl_ids(targets)
        if not target_chembl_ids:
            logger.warning("No valid ChEMBL target IDs found; skipping FDA-approved fetch")
            return compounds

         # Fetch FDA-approved drugs from ChEMBL
        activities = new_client.activity.filter(
            target_chembl_id__in=target_chembl_ids,
            max_phase=4,  # Approved drugs
            standard_type__in=["IC50", "Ki", "Kd", "EC50"],
            standard_units__in=["nM", "uM"],
            standard_relation="=",
            standard_value__lte=10000  # 10 uM or better
        )

        seen = set()
        for act in activities:
            if len(compounds) >= max_compounds:
                break

            molecule_id = act.get('molecule_chembl_id')
            if not molecule_id or molecule_id in seen:
                continue

            try:
                mol = new_client.molecule.get(molecule_id)
                if mol and 'molecule_structures' in mol:
                    smiles = mol['molecule_structures'].get('canonical_smiles')
                    if smiles:
                        rd_mol = Chem.MolFromSmiles(smiles)
                        if rd_mol:
                            rd_mol = self._prepare_molecule(rd_mol)
                            if rd_mol:
                                compounds.append({
                                    'source': 'fda_approved',
                                    'chembl_id': mol.get('molecule_chembl_id', molecule_id),
                                    'smiles': smiles,
                                    'name': mol.get('pref_name', f"Drug_{molecule_id}"),
                                    'mol': rd_mol,
                                    'mw': Descriptors.ExactMolWt(rd_mol) if rd_mol else 0,
                                    'activity': {
                                        'type': act.get('standard_type'),
                                        'value': act.get('standard_value'),
                                        'units': act.get('standard_units')
                                    }
                                })
                                seen.add(molecule_id)
            except Exception as e:
                logger.warning(f"Error processing FDA-approved drug: {e}")
                continue
                    
        return compounds
    
    def _prepare_molecule(self, mol):
        """Prepare a molecule for docking by adding hydrogens and generating 3D coordinates."""
        try:
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            
            # Minimize energy
            AllChem.MMFFOptimizeMolecule(mol)
            
            return mol
        except Exception as e:
            logger.warning(f"Error preparing molecule: {e}")
            return None
    
    def _process_compounds(self):
        """Process compounds (calculate descriptors, etc.)."""
        logger.info(f"Processing {len(self.compounds)} compounds")
        
        for compound in self.compounds:
            try:
                mol = compound['mol']
                if mol:
                    # Calculate properties
                    compound['logp'] = Descriptors.MolLogP(mol)
                    compound['hbd'] = Descriptors.NumHDonors(mol)
                    compound['hba'] = Descriptors.NumHAcceptors(mol)
                    compound['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
                    compound['aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
                    
                    # Generate fingerprint for similarity calculations
                    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
                    compound['fingerprint'] = fp
                    
                    # Generate scaffold
                    try:
                        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                        compound['scaffold'] = Chem.MolToSmiles(scaffold)
                    except:
                        compound['scaffold'] = None
                        
            except Exception as e:
                logger.warning(f"Error processing compound {compound.get('name', 'unknown')}: {e}")
    
    def _filter_compounds(self, min_hbd: int = 1, max_mw: float = 600, 
                         min_logp: float = -2, max_logp: float = 5):
        """Filter compounds based on drug-like properties."""
        logger.info("Filtering compounds based on drug-like properties")
        
        filtered = []
        for compound in self.compounds:
            if (compound.get('hbd', 0) >= min_hbd and
                compound.get('mw', 0) <= max_mw and
                min_logp <= compound.get('logp', 0) <= max_logp and
                compound.get('mol') is not None):
                filtered.append(compound)
                
        logger.info(f"Filtered {len(self.compounds) - len(filtered)} compounds")
        self.compounds = filtered
    
    def _select_diverse_compounds(self, threshold: float = 0.7, max_compounds: int = 1000) -> List[int]:
        """Select a diverse subset of compounds using Butina clustering."""
        if not self.compounds:
            return []
            
        logger.info(f"Selecting diverse subset of up to {max_compounds} compounds")
        
        # Get fingerprints
        fps = [c['fingerprint'] for c in self.compounds if 'fingerprint' in c]
        
        if not fps:
            logger.warning("No valid fingerprints for diversity selection")
            return list(range(min(len(self.compounds), max_compounds)))
        
        # Calculate distance matrix (1 - Tanimoto similarity)
        from rdkit import DataStructs
        from rdkit.ML.Cluster import Butina
        
        # Calculate distance matrix (lower triangle)
        dists = []
        nfps = len(fps)
        for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1-x for x in sims])
        
        # Cluster - convert to list since Butina.ClusterData returns a tuple
        clusters = list(Butina.ClusterData(dists, nfps, threshold, isDistData=True))
        
        # Sort clusters by size (largest first)
        clusters.sort(key=len, reverse=True)
        
        # Select one compound from each cluster until we have enough
        selected = []
        for cluster in clusters:
            # Take the first compound from each cluster
            if cluster and cluster[0] < len(self.compounds):
                selected.append(cluster[0])
                if len(selected) >= max_compounds:
                    break
                    
        logger.info(f"Selected {len(selected)} diverse compounds from {len(clusters)} clusters")
        return selected
    
    def _save_library(self):
        """Save the library to disk."""
        if not self.compounds:
            logger.warning("No compounds to save")
            return
            
        # Create output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        lib_dir = self.output_dir / f"library_{timestamp}"
        lib_dir.mkdir(exist_ok=True, parents=True)
        
        # Save compounds as SDF and SMILES
        sdf_file = lib_dir / "compounds.sdf"
        writer = Chem.SDWriter(str(sdf_file))
        
        smiles_data = []
        
        for i, compound in enumerate(self.compounds):
            try:
                # Skip if molecule is None
                if 'mol' not in compound or compound['mol'] is None:
                    logger.warning(f"Skipping compound {i}: No valid molecule")
                    continue
                    
                mol = compound['mol']
                
                # Set compound properties
                name = compound.get('name', f"compound_{i}")
                smiles = compound.get('smiles', '')
                source = compound.get('source', 'unknown')
                
                # Only set properties if they have values
                if name is not None:
                    mol.SetProp("_Name", str(name))
                if smiles is not None:
                    mol.SetProp("SMILES", str(smiles))
                if source is not None:
                    mol.SetProp("Source", str(source))
                
                # Add properties
                for prop in ['mw', 'logp', 'hbd', 'hba', 'rotatable_bonds', 'aromatic_rings']:
                    if prop in compound and compound[prop] is not None:
                        mol.SetProp(prop, str(compound[prop]))
                
                # Write to SDF
                writer.write(mol)
                
                # Add to SMILES data
                smiles_data.append({
                    'ID': i,
                    'Name': name,
                    'SMILES': smiles,
                    'Source': source,
                    'MW': f"{compound.get('mw', 0):.2f}" if 'mw' in compound else 'N/A',
                    'LogP': f"{compound.get('logp', 0):.2f}" if 'logp' in compound else 'N/A',
                    'HBD': compound.get('hbd', 'N/A'),
                    'HBA': compound.get('hba', 'N/A'),
                    'RotBonds': compound.get('rotatable_bonds', 'N/A'),
                    'AroRings': compound.get('aromatic_rings', 'N/A'),
                    'Scaffold': compound.get('scaffold', '')
                })
                
            except Exception as e:
                logger.warning(f"Error saving compound {i}: {e}")
                continue
                
        writer.close()
        
        # Save SMILES file if we have any compounds
        if smiles_data:
            try:
                df = pd.DataFrame(smiles_data)
                df.to_csv(lib_dir / "compounds.csv", index=False)
                
                # Save metadata
                metadata = {
                    'timestamp': timestamp,
                    'num_compounds': len(smiles_data),
                    'sources': list(set(str(c.get('source', 'unknown')) for c in self.compounds 
                                     if c.get('mol') is not None)),
                    'properties': {
                        'avg_mw': df['MW'].apply(lambda x: float(x) if x != 'N/A' else 0).mean(),
                        'avg_logp': df['LogP'].apply(lambda x: float(x) if x != 'N/A' else 0).mean(),
                        'avg_hbd': df['HBD'].apply(lambda x: float(x) if x != 'N/A' else 0).mean(),
                        'avg_hba': df['HBA'].apply(lambda x: float(x) if x != 'N/A' else 0).mean(),
                    }
                }
                
                with open(lib_dir / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                logger.info(f"Library saved to {lib_dir}")
                
            except Exception as e:
                logger.error(f"Error saving library metadata: {e}")
        else:
            logger.warning("No valid compounds to save")
    
    def build_library(self, targets: List[str], max_compounds: int = 1000, 
                     diversity_threshold: float = 0.5, min_hbd: int = 1, 
                     max_mw: float = 600, min_logp: float = -2, 
                     max_logp: float = 5, sources: List[str] = None):
        """
        Build a diverse compound library.
        
        Args:
            targets: List of target names or identifiers
            max_compounds: Maximum number of compounds to include
            diversity_threshold: Tanimoto similarity threshold for clustering (0-1)
            min_hbd: Minimum number of hydrogen bond donors
            max_mw: Maximum molecular weight
            min_logp: Minimum logP value
            max_logp: Maximum logP value
            sources: List of data sources to use (pubchem, chembl)
        """
        sources = sources or ['pubchem', 'chembl']
        
        logger.info(f"Building library for targets: {', '.join(targets)}")
        
        # Fetch compounds from all sources
        with ThreadPoolExecutor() as executor:
            futures = []
            for source in sources:
                if source in self.sources:
                    futures.append(executor.submit(
                        self.sources[source], 
                        targets,
                        max_compounds // len(sources)  # Distribute compounds across sources
                    ))
            
            for future in as_completed(futures):
                try:
                    compounds = future.result()
                    self.compounds.extend(compounds)
                except Exception as e:
                    logger.error(f"Error fetching compounds: {e}")
        
        # Process and filter compounds
        self._process_compounds()
        self._filter_compounds(
            min_hbd=min_hbd,
            max_mw=max_mw,
            min_logp=min_logp,
            max_logp=max_logp
        )
        
        # Select diverse subset
        selected_indices = self._select_diverse_compounds(
            threshold=diversity_threshold,
            max_compounds=max_compounds
        )
        
        self.compounds = [self.compounds[i] for i in selected_indices]
        
        # Save library
        self._save_library()
        
        return self.compounds
    
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build a diverse compound library for virtual screening.')
    parser.add_argument('targets', nargs='+', help='Target names or identifiers (e.g., gene names, ChEMBL IDs)')
    parser.add_argument('--max_compounds', type=int, default=1000, help='Maximum number of compounds to include')
    parser.add_argument('--diversity', type=float, default=0.7, 
                       help='Diversity threshold for clustering (0-1, higher = more diverse)')
    parser.add_argument('--sources', nargs='+', 
                       default=['pubchem', 'chembl', 'np_atlas', 'fda_approved'],
                       help='Data sources to use (pubchem, chembl, np_atlas, fda_approved)')
    parser.add_argument('--output', type=str, default='ligand_libraries',
                       help='Output directory for the library')
    
    args = parser.parse_args()
    
    # Initialize and build library
    builder = LigandLibraryBuilder(output_dir=args.output)
    compounds = builder.build_library(
        targets=args.targets,
        max_compounds=args.max_compounds,
        diversity_threshold=args.diversity,
        sources=args.sources
    )
    
    print(f"\nSuccessfully built library with {len(compounds)} compounds")
    print(f"Library saved to: {args.output}")

if __name__ == "__main__":
    main()
