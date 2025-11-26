"""
One-off analysis for REPORT Task 5:

- Compute time-domain features: mean, std, RMS, ZCR
- Compute frequency-domain features: dominant freq, spectral energy, spectral entropy
- Train a RandomForest on these features
- Plot feature importance and print a short metrics table

Does NOT change the main training pipeline.
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from data_loader import (
    BASE_DIR,
    load_kuhar_timeseries_multi,
    MULTI_SPLIT_DIRS,
    load_one_csv,
)
from constants import CLASS_NAMES

FS = 25.0  # assumed sampling rate


# helpers for feature extract
def _sanitize(series: pd.Series) -> np.ndarray:
    arr = series.astype(float).to_numpy()
    if np.isnan(arr).any():
        m = np.nanmean(arr)
        if np.isnan(m):
            m = 0.0
        arr = np.where(np.isnan(arr), m, arr)
    return arr


def time_features(x: np.ndarray, prefix: str) -> Dict[str, float]:
    """mean, std, RMS, ZCR (+ min, max, median, energy)."""
    n = x.size
    if n == 0:
        return {f"{prefix}_{k}": 0.0 for k in
                ["mean", "std", "rms", "zcr", "min", "max", "median", "energy"]}

    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x ** 2)))
    energy = float(np.sum(x ** 2) / n)

    if n > 1:
        sign_changes = np.sum(np.sign(x[:-1]) * np.sign(x[1:]) < 0)
        zcr = float(sign_changes / (n - 1))
    else:
        zcr = 0.0

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_rms": rms,
        f"{prefix}_zcr": zcr,
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_energy": energy,
    }


def freq_features(x: np.ndarray, prefix: str, fs: float = FS) -> Dict[str, float]:
    """dominant freq, spectral energy, spectral entropy."""
    n = x.size
    if n < 2:
        return {
            f"{prefix}_dom_freq": 0.0,
            f"{prefix}_spec_energy": 0.0,
            f"{prefix}_spec_entropy": 0.0,
        }

    x = x - np.mean(x)
    fft_vals = np.fft.rfft(x)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    if power.sum() > 0:
        idx_max = int(np.argmax(power[1:])) + 1 if power.size > 1 else 0
        dom_freq = float(freqs[idx_max])
    else:
        dom_freq = 0.0

    spec_energy = float(power.mean())

    total_power = power.sum()
    if total_power <= 0:
        spec_entropy = 0.0
    else:
        p = power / total_power
        spec_entropy = float(-np.sum(p * np.log2(p + 1e-12)))

    return {
        f"{prefix}_dom_freq": dom_freq,
        f"{prefix}_spec_energy": spec_energy,
        f"{prefix}_spec_entropy": spec_entropy,
    }


def extract_features_from_file(csv_path: str) -> Dict[str, float]:
    """
    Compute time + frequency features for each sensor channel in one CSV.
    """
    df = load_one_csv(csv_path)
    feats: Dict[str, float] = {}

    for idx, col in enumerate(df.columns):
        base = col if isinstance(col, str) else f"col{idx}"
        sig = _sanitize(df[col])

        feats.update(time_features(sig, base))
        feats.update(freq_features(sig, base, fs=FS))

    feats["file_path"] = csv_path
    return feats


def build_feature_dataset(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature DataFrame for all files in folders 1+2 using Task-5 feature set.
    """
    records: List[Dict[str, float]] = []

    for _, row in index_df.iterrows():
        csv_path = row["file_path"]
        feats = extract_features_from_file(csv_path)
        feats.update({
            "class_idx": row["class_idx"],
            "class_name": row["class_name"],
        })
        records.append(feats)

    return pd.DataFrame(records)


def main():
    print("[1] Loading index from folders 1+2...")
    df_files = load_kuhar_timeseries_multi(BASE_DIR, MULTI_SPLIT_DIRS)
    print(f"    -> {len(df_files)} files")

    print("[2] Building Task-5 feature dataset (time + frequency)...")
    feat_df = build_feature_dataset(df_files)
    print(f"    -> feat_df shape: {feat_df.shape}")

    # drop meta columns
    X = feat_df.drop(columns=["class_idx", "class_name", "file_path"])
    y = feat_df["class_idx"]

    # simple train/test split for feature-importance visualization
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[3] Training RandomForest just for feature-importance analysis...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=25,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[Task5 RF] Accuracy on 80/20 split: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # feature-importance bar chart 
    importances = rf.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]
    top_k = 20  # show top 20 features
    top_idx = idx_sorted[:top_k]

    top_names = X.columns[top_idx]
    top_vals = importances[top_idx]

    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(top_vals)), top_vals)
    plt.xticks(range(len(top_vals)), top_names, rotation=90)
    plt.ylabel("Importance")
    plt.title("RandomForest Feature Importance (Task-5 feature set)")
    plt.tight_layout()
    out_path = os.path.join("outputs", "task5_feature_importance.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved feature-importance plot to {out_path}")

    # save the full feature table for the report
    feat_df.to_csv("outputs/task5_features_table.csv", index=False)
    print("Saved feature table to outputs/task5_features_table.csv")


if __name__ == "__main__":
    main()
