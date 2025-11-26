"""
dl_preprocess.py

Prepare KU-HAR time-series windows as sequence tensors for deep learning:

- Load subsample windows from folder 3
- filter to a subset of activities (e.g., Walk / Jump / Stand)
- Build X_seq with shape (N, T, 1) for ANN / 1D-CNN / RNN
- Split into train / val / test
- Standardize (z-score) using training statistics only
- Save everything to outputs/dl_sequences_*.npz

Usage examples:
    $ All classes
    py src/dl_preprocess.py

    # Specific Walk, Jump, Stand
    py src/dl_preprocess.py --classes Walk Jump Stand
"""

import os
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import BASE_DIR, load_kuhar_subsamples
from constants import CLASS_NAMES

# Meta columns present in the subsample DataFrame
META_COLS = ["subject", "window_len", "class_idx", "class_name", "serial_no"]


def build_sequence_tensors(
    subsample_df: pd.DataFrame,
    filter_class_names: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Convert KU-HAR subsample windows into sequence tensors.

    Parameters
    ----------
    subsample_df : pd.DataFrame
        Windows table from load_kuhar_subsamples().
        Each row = one window, meta columns + sensor/time columns.
    filter_class_names : List[str], optional
        If given, keep only rows whose class_name is in this list.
        Example: ["Walk", "Jump", "Stand"]

    Returns:
    X_seq : np.ndarray
        Shape (N, T, 1) where T = window length (number of time steps).
    y : np.ndarray
        Shape (N,) integer labels (class_idx).
    meta_df : pd.DataFrame
        DataFrame with meta columns only (subject, window_len, etc.).
        Can be useful for LOSO-style evaluation later.
    """
    df = subsample_df.copy()

    # Optional: filter by class_name
    if filter_class_names is not None:
        df = df[df["class_name"].isin(filter_class_names)].copy()
        df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError("No rows left after filtering by classes.")

    # Identify the sensor/time columns = everything that's NOT meta
    sensor_cols = [c for c in df.columns if c not in META_COLS]

    # Sort by subject and serial_no to keep some temporal order (optional but nice)
    sort_cols = [c for c in ["subject", "serial_no"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Extract numeric sensor matrix: shape (N, T)
    sensor_mat = df[sensor_cols].astype(float).to_numpy()   # N x T

    # Handle NaNs for deep nets: simple column-wise mean impute
    col_means = np.nanmean(sensor_mat, axis=0)
    inds = np.where(np.isnan(sensor_mat))
    sensor_mat[inds] = np.take(col_means, inds[1])

    # Add a "channel" dimension → (N, T, 1)
    X_seq = sensor_mat[..., np.newaxis]

    # Labels
    y = df["class_idx"].to_numpy()

    # Meta-only DataFrame
    meta_df = df[META_COLS].copy()

    return X_seq, y, meta_df

def standardize_sequences(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Z-score normalization for sequence data using TRAIN stats only.

    We treat all time steps and channels together and compute:
        mean = X_train.mean()
        std  = X_train.std()
    Then apply (X - mean) / std to train, val, test.

    Returns: X_train_norm, X_val_norm, X_test_norm, stats_dict
    """
    mean = X_train.mean()
    std = X_train.std()

    # Avoid divide-by-zero
    if std == 0:
        std = 1.0

    X_train_norm = (X_train - mean) / std
    X_val_norm   = (X_val   - mean) / std
    X_test_norm  = (X_test  - mean) / std

    stats = {"mean": float(mean), "std": float(std)}
    return X_train_norm, X_val_norm, X_test_norm, stats


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-stage split:
      - first: train+val vs test
      - second: train vs val

    All splits are stratified by label y.

    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: train_val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Now split train_val into train vs val
    val_fraction_of_trainval = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction_of_trainval,
        random_state=random_state,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    parser = argparse.ArgumentParser(
        description="Prepare KU-HAR sequences for deep learning (ANN/1D-CNN/LSTM)."
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional list of class names to keep (e.g. --classes Walk Jump Stand). "
             "If omitted, use all classes.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data used as test set (default: 0.2).",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
        help="Fraction of data used as validation set (default: 0.1).",
    )
    args = parser.parse_args()

    if args.classes:
        print(f"[DL] Filtering to classes: {args.classes}")
    else:
        print("[DL] Using ALL classes in subsamples")

    print("[DL] Loading subsample windows from folder 3...")
    subsamples = load_kuhar_subsamples(BASE_DIR)
    print(f"    -> {len(subsamples)} windows")

    print("[DL] Building sequence tensors...")
    X_seq, y, meta_df = build_sequence_tensors(
        subsamples,
        filter_class_names=args.classes,
    )
    print(f"    -> X_seq shape: {X_seq.shape}, y shape: {y.shape}")

    print("[DL] Splitting into train / val / test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X_seq, y,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    print(f"    -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("[DL] Standardizing using train statistics...")
    X_train_n, X_val_n, X_test_n, stats = standardize_sequences(
        X_train, X_val, X_test
    )
    print(f"    -> mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    # Save outputs
    os.makedirs("outputs", exist_ok=True)

    # Save arrays for ANN training
    out_npz = "outputs/dl_sequences_all_classes.npz"
    if args.classes:
        # E.g. outputs/dl_sequences_Walk_Jump_Stand.npz
        suffix = "_".join(args.classes)
        out_npz = f"outputs/dl_sequences_{suffix}.npz"

    np.savez_compressed(
        out_npz,
        X_train=X_train_n,
        X_val=X_val_n,
        X_test=X_test_n,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        mean=stats["mean"],
        std=stats["std"],
    )
    print(f"[DL] Saved sequence arrays to: {out_npz}")

    # Also optionally save the meta info so you can do LOSO-style stuff later if needed
    meta_path = "outputs/dl_meta_all_classes.csv"
    if args.classes:
        suffix = "_".join(args.classes)
        meta_path = f"outputs/dl_meta_{suffix}.csv"

    meta_df.to_csv(meta_path, index=False)
    print(f"[DL] Saved meta info to: {meta_path}")

    print("\n[DL] Done. These files are ready for an ANN training script to load.")


if __name__ == "__main__":
    main()
