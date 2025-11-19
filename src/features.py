"""Transform raw time series CSV files like accelerometer/gyroscope signals into
 features suitable for ML models.

 - Compute basic stats (mean, std, min, max, energy, etc.) per column
 - Load one CSV, use basic stats to create a flat feature row
 - Loop through all files indexed by data_loader.py and produce a single DataFrame ready for model training
 """

from collections import deque
import pandas as pd
import numpy as np
from data_loader import (
        load_one_csv, 
        load_kuhar_timeseries, 
        BASE_DIR, 
        load_kuhar_subsamples
)
from typing import Dict
from scipy.stats import skew, kurtosis


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

def build_feature_dataset_from_subsamples(subsample_df: pd.DataFrame) -> pd.DataFrame:
    """
    subsample_df: rows = windows, columns 0..N-1 = sensor/time,
                  plus meta columns: subject, window_len, class_idx, class_name, serial_no
    """
    meta_cols = ["subject", "window_len", "class_idx", "class_name", "serial_no"]
    sensor_cols = [c for c in subsample_df.columns if c not in meta_cols]

    feature_records = []
    for _, row in subsample_df.iterrows():
        sensor_series = row[sensor_cols].astype(float)

        feats = {
            "mean":   sensor_series.mean(),
            "std":    sensor_series.std(),
            "min":    sensor_series.min(),
            "max":    sensor_series.max(),
            "median": sensor_series.median(),
            "energy": np.sum(sensor_series**2) / len(sensor_series),
        }

        feats.update({
            "class_idx":  row["class_idx"],
            "class_name": row["class_name"],
            "subject":    row["subject"],
            "window_len": row["window_len"],
            "serial_no":  row["serial_no"],
        })
        feature_records.append(feats)

    return pd.DataFrame(feature_records)

def zcr_window(window):
    # win = (T, C)
    zcrs = []
    for c in range(window.shape[1]):
        sig = window[:, c]
        zc = np.sum(np.abs(np.diff(np.sign(sig)))) / 2
        zcrs.append(zc / len(sig))
    return np.array(zcrs)

def extract_features(window, feature="raw"):
    """
    window: np.array shape (window_size, channels)
    feature: "stats" or "raw"

    returns: feature vector
    """
    if feature == "raw":
        return window.flatten()
    elif feature == "stats":
        mean = window.mean(axis=0)
        std = window.std(axis=0)
        rms = np.sqrt(np.mean(np.square(window), axis=0))
        zcr = zcr_window(window)
        feats = np.concatenate([mean, std, rms, zcr])
    else:
        raise ValueError("feature must be 'stats' or 'raw")
    return feats

buffer = deque()
counter = 0

def predict_real_time(new_sample, model, buffer, window_size, step, feature="stats"):
    """
    new_sample: sensor reading
    buffer: deque storing last samples
    """
    global counter
    buffer.append(new_sample)
    counter += 1

    if len(buffer) < window_size:
        return None
    
    if counter < step:
        return None
    
    counter = 0
    window = np.array(buffer)  # shape (window_size, C)
    feats = extract_features(window, feature)
    pred = model.predict([feats])[0]
    # will have to implement model predicting later but placeholder for now

    return pred


