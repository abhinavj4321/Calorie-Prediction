"""Feature engineering utilities.
"""
import pandas as pd
import numpy as np

# Example MET lookup (approx)
_MET_MAP = {
    "running": 9.8,
    "walking": 3.8,
    "cycling": 7.5,
    "yoga": 2.5,
    "swimming": 8.0,
}

def _map_met(ex_type):
    if pd.isna(ex_type):
        return np.nan
    ex = str(ex_type).lower()
    for key in _MET_MAP:
        if key in ex:
            return _MET_MAP[key]
    return np.nan

def create_features(df):
    df = df.copy()
    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    # ensure common columns exist
    # add derived features:
    if "exercise_type" in df.columns:
        df["exercise_type_clean"] = df["exercise_type"].astype(str).str.lower().str.strip()
        df["met_estimate"] = df["exercise_type_clean"].map(lambda x: _map_met(x))
    else:
        df["exercise_type_clean"] = "unknown"
        df["met_estimate"] = np.nan

    if "duration_min" in df.columns:
        df["duration_min"] = pd.to_numeric(df["duration_min"], errors="coerce")
    else:
        df["duration_min"] = np.nan

    # calories_by_met: very rough estimate = MET * weight_kg * duration_hours
    if "weight_kg" in df.columns:
        df["weight_kg"] = pd.to_numeric(df["weight_kg"], errors="coerce")
        df["duration_hr"] = df["duration_min"] / 60.0
        df["cal_by_met"] = df["met_estimate"] * df["weight_kg"] * df["duration_hr"]
    else:
        df["weight_kg"] = np.nan
        df["duration_hr"] = np.nan
        df["cal_by_met"] = np.nan

    # encode exercise type as one-hot (small set)
    try:
        top_types = df["exercise_type_clean"].value_counts().nlargest(10).index.tolist()
        for t in top_types:
            col = f"ex_is__{t.replace(' ','_')}"
            df[col] = (df["exercise_type_clean"] == t).astype(int)
    except Exception:
        pass

    # keep commonly used numeric features
    # rename target if needed outside
    return df
