import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import urllib3
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Descriptors

# ----------------------------
# Utility functions
# ----------------------------

def calc_desc(smiles: str):
    """Compute lightweight properties for filtering."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MW": Descriptors.MolWt(mol),
        "logP": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol)
    }

# ----------------------------
# ZINC20 PUBLIC REST API
# ----------------------------
# API specs: https://zinc20.docking.org
# We pull a small set of "druglike" compounds.

def fetch_zinc_subset(max_n=2000):
    """
    Fetch a small public subset of ZINC druglike space.
    This uses their standard "search" API.
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    url = f"https://zinc20.docking.org/substances.txt?count={max_n}"
    
    # 1. Setup Retry Strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    # 2. Add Browser Headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        print(f"[fetch_zinc] Attempting to fetch {max_n} molecules...")
        r = http.get(url, headers=headers, timeout=30, verify=False)
        r.raise_for_status()
        # if r.status_code != 200:
        #     print("[fetch_zinc_subset] Failed to query ZINC")
        #     return []

        smiles_list = []
        for line in r.text.strip().split("\n"):
            # if "\t" in line:
            #     zinc_id, smi = line.split("\t")[:2]
            #     smiles_list.append((smi, zinc_id))
            parts = line.split()
            if len(parts) >= 2:
                # ZINC20 txt format usually: ZINC_ID SMILES or SMILES ZINC_ID
                # We assume standard ZINC format: ZINC000001 <tab> SMILES
                if parts[0].startswith("ZINC"):
                    zinc_id, smi = parts[0], parts[1]
                else:
                    smi, zinc_id = parts[0], parts[1]
                smiles_list.append((smi, zinc_id))

        ligs = []
        for smi, zid in smiles_list:
            desc = calc_desc(smi)
            if desc is None:
                continue
            ligs.append({
                "source": "ZINC",
                "smiles": smi,
                "id": zid,
                "molecular_weight": desc["MW"],
                "logp": desc["logP"],
                "tpsa": desc["tpsa"],
                "raw": {"zinc_id": zid}
            })
        return ligs

    except Exception as e:
        print("[fetch_zinc_subset] error:", e)
        return []

# ----------------------------
# CHEMBL PUBLIC API
# ----------------------------
# API: https://www.ebi.ac.uk/chembl/api

def fetch_chembl_subset(max_n=2000):
    """
    Fetch drug-like molecules from ChEMBL (EBI) which is much more stable than ZINC.
    """
    # Query: Molecules with MW <= 500 (Lipinski rule) and Pref_Name exists (likely a drug)
    ebi_url = "https://www.ebi.ac.uk"
    base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule"
    # base_url = "https://www.ebi.ac.uk/chembl/api/data/drug"
    params = {
        "max_phase__gte": 3,
        "molecular_weight__lte": 500,
        "molecular_weight__gte": 200,
        "hba__lte": 10,
        "hbd__lte": 5,
        "alogp__lte": 5,
        "psa__lte": 140,
        "ro5_violations__lte": 1,
        "limit": max_n,
        "format": "json"
    }
    
    ligs = []
    url = base_url
    try:
        while url:
            # print(f"[fetch_chembl] Querying EBI ChEMBL API for {max_n} molecules...")
            print("Fetching:", url)
            if url == base_url:
                r = requests.get(url, params=params, timeout=20)
            else:
                r = requests.get(f"{ebi_url}{url}", params=params, timeout=20)
                if r.status_code != 200:
                    continue
            r.raise_for_status()
            data = r.json()

            for entry in data.get('molecules', []): # data.get('drugs', [])
                structure = entry.get('molecule_structures', {})
                if not structure:
                    continue
                smi = structure.get('canonical_smiles')
                chembl_id = entry.get('molecule_chembl_id')
                
                if not smi:
                    continue

                # We can use ChEMBL's pre-calculated props or your calc_desc
                # Using your calc_desc for consistency across pipeline
                desc = calc_desc(smi)
                if desc:
                    ligs.append({
                        "source": "ChEMBL",
                        "smiles": smi,
                        "id": chembl_id,
                        "molecular_weight": desc["MW"],
                        "logp": desc["logP"],
                        "tpsa": desc["tpsa"],
                        "raw": {"chembl_id": chembl_id}
                    })
            url = data["page_meta"]["next"]
        return ligs

    except Exception as e:
        print(f"[fetch_chembl] Error: {e}")
        return []

# ----------------------------
# ENAMINE REAL (public fragments)
# ----------------------------
# Minimal public download endpoint: https://enamine.net/real-compounds/real-fragments

def fetch_enamine_fragments(local_csv=None, max_n=3000):
    """
    Enamine provides free public fragment CSV downloads.
    If 'local_csv' exists, load from it; otherwise, skip.
    """
    if local_csv and os.path.exists(local_csv):
        try:
            df = pd.read_csv(local_csv)
        except:
            return []

        ligs = []
        for _, row in df.head(max_n).iterrows():
            smi = row.get("SMILES")
            if smi is None:
                continue
            desc = calc_desc(smi)
            if desc is None:
                continue

            ligs.append({
                "source": "ENAMINE",
                "smiles": smi,
                "id": str(row.get("ID", "")),
                "molecular_weight": desc["MW"],
                "logp": desc["logP"],
                "tpsa": desc["tpsa"],
                "raw": row.to_dict()
            })
        return ligs
    return []
