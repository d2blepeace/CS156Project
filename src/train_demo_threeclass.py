"""
train_demo_threeclass.py

Train a DEMO Random Forest model on only 3 KU-HAR classes:

    - Stand
    - Walk
    - Jump

This script's feature extraction matches the real-time controller:

For each window:
    - take accel X[0:300], Y[300:600], Z[600:900]
    - for each axis compute: mean, std, min, max, median, energy
    => 6 stats * 3 axes = 18 features

Labels:
    Stand -> 0
    Walk  -> 1
    Jump  -> 2

Model is saved to:
    trained_models/for_demo/random_forest_demo.pkl
"""

import os
import numpy as np
import pandas as pd

from constants import CLASS_NAMES
from data_loader import BASE_DIR, load_kuhar_subsamples
from ml_models import random_forest as rf

# Order defines the demo label indices: 0,1,2
DEMO_CLASSES = ["Stand", "Walk", "Jump"]
DEMO_LABEL_MAP = {name: idx for idx, name in enumerate(DEMO_CLASSES)}


def window_to_features(row_values: np.ndarray) -> np.ndarray:
    """
    Given a 1D numpy array of length >= 900 from subsamples:

        0:300   -> accel X
        300:600 -> accel Y
        600:900 -> accel Z

    compute for each axis:
        mean, std, min, max, median, energy   (energy = mean(x^2))

    Return: 1D array of length 18
    """
    # Ensure we have at least 900 values
    if row_values.shape[0] < 900:
        raise ValueError(f"Row has only {row_values.shape[0]} columns, expected >= 900")

    x = row_values[0:300]
    y = row_values[300:600]
    z = row_values[600:900]

    feats = []

    for axis in (x, y, z):
        mean_ = np.mean(axis)
        std_ = np.std(axis)
        min_ = np.min(axis)
        max_ = np.max(axis)
        median_ = np.median(axis)
        energy_ = np.mean(axis**2)  # mean squared energy

        feats.extend([mean_, std_, min_, max_, median_, energy_])

    return np.array(feats, dtype=float)


def main():
    print("\n=== Training DEMO RandomForest model (Stand / Walk / Jump, 18 features) ===\n")

    # 1) Load subsample windows (3. Time_domain_subsamples)
    subsamples = load_kuhar_subsamples(BASE_DIR)
    print(f"[1] Loaded subsamples: {len(subsamples)} rows total.")

    # subsamples columns:
    #   0..1799 -> sensor window
    #   class_idx, class_name, window_len, serial_no

    # 2) Filter to DEMO_CLASSES (Stand / Walk / Jump)
    subsamples_demo = subsamples[subsamples["class_name"].isin(DEMO_CLASSES)].copy()
    print(f"[2] Kept {len(subsamples_demo)} rows for classes {DEMO_CLASSES}.")

    if subsamples_demo.empty:
        raise ValueError("No subsample rows found for Stand/Walk/Jump. Check dataset paths.")

    # 3) Build 18-feature vectors from each row 
    print("[3] Extracting 18 features per window (6 stats * 3 axes)...")

    # Sensor data columns = all but last 4 metadata columns
    sensor_cols = subsamples_demo.columns[:-4]

    # Convert sensor data to numpy array
    sensor_matrix = subsamples_demo.loc[:, sensor_cols].to_numpy(dtype=float)
    n_rows, n_cols = sensor_matrix.shape
    print(f"    Sensor matrix shape: {sensor_matrix.shape} (rows, cols)")

    feature_list = []
    for i in range(n_rows):
        feats = window_to_features(sensor_matrix[i])
        feature_list.append(feats)

    X = np.vstack(feature_list)  # (n_rows, 18)

    # 4) Build labels 0..2 based on DEMO_LABEL_MAP
    print("[4] Building demo labels (0=Stand, 1=Walk, 2=Jump)...")

    class_names = subsamples_demo["class_name"].to_list()
    y = np.array([DEMO_LABEL_MAP[name] for name in class_names], dtype=int)

    # Basic sanity:
    print(f"    X shape: {X.shape}")
    unique, counts = np.unique(y, return_counts=True)
    print("    Label distribution:")
    for u, c in zip(unique, counts):
        print(f"      {u} -> {DEMO_CLASSES[u]}: {c} samples")

    # 5) Train RandomForest (80/20 split) using rf.train_model
    print("\n[5] Training RandomForest on DEMO 3-class data (18 features)...")
    model, X_test, y_test, y_pred = rf.train_model(X, y)

    # 6) Save demo model
    print("\n[6] Saving DEMO model to trained_models/for_demo/random_forest_demo.pkl ...")
    rf.save_demo_model(model)

    # 
    # 7) Confusion matrix (optional, saved to outputs/)
    print("[7] Saving confusion matrix for report...")
    os.makedirs("outputs", exist_ok=True)
    cm_path = os.path.join("outputs", "cm_demo_rf_3class_18feat.png")

    rf.plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=DEMO_CLASSES,
        title="Random Forest – DEMO (Stand / Walk / Jump, 18 features)",
        save_path=cm_path,
        show=False,
    )
    print(f"    -> Confusion matrix saved to: {cm_path}")

    print("\n=== DEMO model training complete! ===\n")


if __name__ == "__main__":
    main()
