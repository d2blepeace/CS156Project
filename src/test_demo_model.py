"""
test_demo_model.py

Simple sanity check for the DEMO 3-class model (Stand / Walk / Jump).
"""

from constants import CLASS_NAMES
from data_loader import (
    BASE_DIR,
    load_kuhar_timeseries_multi,
    load_kuhar_subsamples,
    MULTI_SPLIT_DIRS,
)
from features import (
    build_feature_dataset,
    build_feature_dataset_from_subsamples,
)
from ml_models import random_forest as rf
import pandas as pd

DEMO_CLASSES = ["Stand", "Walk", "Jump"]


def load_demo_features():
    # Load + build feature tables
    df_files = load_kuhar_timeseries_multi(BASE_DIR, MULTI_SPLIT_DIRS)
    feat_files = build_feature_dataset(df_files)

    subsamples = load_kuhar_subsamples(BASE_DIR)
    feat_sub = build_feature_dataset_from_subsamples(subsamples)

    feature_df = pd.concat([feat_files, feat_sub], ignore_index=True)

    # Filter to demo classes
    feature_df = feature_df[feature_df["class_name"].isin(DEMO_CLASSES)].copy()

    # Fill NaNs
    feature_df = feature_df.apply(
        lambda col: col.fillna(col.mean()) if col.dtype != "object" else col
    )

    # Map to demo indices 0..2
    orig_indices = {name: CLASS_NAMES.index(name) for name in DEMO_CLASSES}
    orig_to_demo = {
        orig_idx: new_idx
        for new_idx, orig_idx in enumerate(orig_indices.values())
    }

    feature_df["demo_idx"] = feature_df["class_idx"].map(orig_to_demo)

    drop_cols = [
        "class_idx",
        "class_name",
        "subject",
        "letter",
        "trial",
        "file_path",
        "window_len",
        "serial_no",
        "demo_idx",
    ]
    drop_cols = [c for c in drop_cols if c in feature_df.columns]

    X = feature_df.drop(columns=drop_cols)
    y = feature_df["demo_idx"]

    return X, y


def main():
    print("\n=== Testing DEMO RandomForest model (Stand / Walk / Jump) ===\n")

    X, y = load_demo_features()
    print(f"Loaded {len(X)} demo samples.")

    model = rf.load_demo_model()
    print("Loaded demo model from trained_models/for_demo/.")

    # Take a small subset
    n_show = 10
    X_sub = X.iloc[:n_show]
    y_sub = y.iloc[:n_show]

    y_pred = model.predict(X_sub)

    idx_to_name = {i: name for i, name in enumerate(DEMO_CLASSES)}

    print("\nFirst few predictions:")
    for i, (true_idx, pred_idx) in enumerate(zip(y_sub, y_pred), start=1):
        true_name = idx_to_name.get(true_idx, f"idx={true_idx}")
        pred_name = idx_to_name.get(pred_idx, f"idx={pred_idx}")
        print(f"Sample {i:02d}: true={true_name:5s}  |  pred={pred_name:5s}")

    print("\n=== Demo test run complete ===\n")


if __name__ == "__main__":
    main()
