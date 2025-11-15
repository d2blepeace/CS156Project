"""Transform raw time series CSV files like accelerometer/gyroscope signals into
 features suitable for ML models.

 - Compute basic stats (mean, std, min, max, energy, etc.) per column
 - Load one CSV, use basic stats to create a flat feature row
 - Loop through all files indexed by data_loader.py and produce a single DataFrame ready for model training
 """

import pandas as pd
import numpy as np
from data_loader import load_one_csv, load_kuhar_timeseries, BASE_DIR
from typing import Dict


#  Compute statistical features from one time-series 
def compute_basic_stats(df: pd.DataFrame) -> Dict[str, float]:  
    """
    Returns a dictionary of features with keys like 'col0_mean', 'col0_std', etc.
    """
    feats = {}
    for col in df.columns:
        colname = f"col{col}"   #  col0, col1...
        series = df[col]        # time series for this sensor channel
        
        # Basic statistical features
        feats[f"{colname}_mean"] = series.mean()
        feats[f"{colname}_std"] = series.std()
        feats[f"{colname}_min"] = series.min()
        feats[f"{colname}_max"] = series.max()
        feats[f"{colname}_median"] = series.median()
        feats[f"{colname}_energy"] = np.sum(series**2) / len(series)
    return feats

#  Extract features from one CSV
def extract_features_from_csv(csv_path: str) -> Dict[str, float]:
    df = load_one_csv(csv_path)         # Load raw time-series from CSV
    feats = compute_basic_stats(df)     # Compute statistical features
    feats["file_path"] = csv_path       # Keep the path for debugging/traceability
    return feats

# Build a feature dataset
def build_feature_dataset(index_df: pd.DataFrame) -> pd.DataFrame:
    feature_records = []
    for _, row in index_df.iterrows():
        csv_path = row["file_path"]
        feats = extract_features_from_csv(csv_path)
        feats.update({
            "class_idx": row["class_idx"],
            "class_name": row["class_name"],
            "subject": row["subject"],
            "letter": row["letter"],
            "trial": row["trial"]
        })
        feature_records.append(feats)
    # Convert list of dicts → DataFrame
    return pd.DataFrame(feature_records)

