import os
import pandas as pd

# ----------------------------
# Helper functions
# ----------------------------
def read_pocket_csv(pred_csv: str):
    """
    Safely load a p2rank pocket prediction CSV.
    Returns pd.DataFrame or None on failure.
    """
    if not os.path.exists(pred_csv):
        return None

    try:
        df = pd.read_csv(pred_csv)
        return df
    except Exception as e:
        print(f"[read_pocket_csv] Failed to load {pred_csv}: {e}")
        return None


def filter_pockets(df, min_score=1.5, min_prob=0.1, min_sas=20, max_pockets=10):
    """
    Apply pocket quality filters.
    Returns a filtered pd.DataFrame (possibly empty).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # columns expected from p2rank
    required_cols = ["rank", "score", "probability", "sas_points"]
    for c in required_cols:
        if c not in df.columns:
            print(f"[filter_pockets] Missing column {c} in CSV")
            return pd.DataFrame()

    filtered_df = df[
        (df["score"] >= min_score) &
        (df["probability"] >= min_prob) &
        (df["sas_points"] >= min_sas)
    ].copy()

    # sort by score if not sorted already
    filtered_df = filtered_df.sort_values("score", ascending=False)
    return filtered_df.head(max_pockets)


def df_to_pocket_list(df, pdb_id: str):
    """
    Convert dataframe rows to pocket dict list suitable for docking.
    """
    pockets = []
    if df is None or df.empty:
        return pockets

    for _, row in df.iterrows():
        pockets.append({
            "pdb_id": pdb_id,
            "pocket_id": row.get("name", f"pocket_{int(row['rank'])}"),
            "rank": int(row["rank"]),
            "score": float(row["score"]),
            "probability": float(row["probability"]),
            "center": (
                float(row.get("center_x", 0)),
                float(row.get("center_y", 0)),
                float(row.get("center_z", 0)),
            ),
            "residues": str(row.get("residue_ids", "")).split(),
            "raw_row": row.to_dict()
        })
    return pockets