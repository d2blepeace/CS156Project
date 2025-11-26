"""
Evaluate KU-HAR models using:
- LOSO (Leave-One-Subject-Out) only

Supports multiple models.
Examples: (check Model_registrey below for flag)
    py src/eval_loso.py --model rf
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")       # no GUI backend
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)

from constants import CLASS_NAMES
from data_loader import (
    BASE_DIR,
    load_kuhar_timeseries_multi,
    load_kuhar_subsamples,
    MULTI_SPLIT_DIRS,
)
from features import build_feature_dataset, build_feature_dataset_from_subsamples

# MODEL REGISTRY
MODEL_REGISTRY = {
    "rf":  ("RandomForest", "ml_models.random_forest"),
    "dt":  ("DecisionTree", "ml_models.decision_tree"),
    "svm": ("SVM",          "ml_models.svm"),
    "nb":  ("NaiveBayes",   "ml_models.naive_bayes"),
    "ada": ("AdaBoost",     "ml_models.adaboost"),
    "xgb": ("XGBoost",      "ml_models.xgboost"),
}


def load_model_module(key: str):
    import importlib
    pretty_name, module_path = MODEL_REGISTRY[key]
    return pretty_name, importlib.import_module(module_path)

# SMALL HELPERS
def plot_cm(y_true, y_pred, class_names, title, out_path):
    """Simple confusion-matrix plotter that saves to file."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues", aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def fill_numeric_nans_train_test(X_train, X_test):
    """
    For models like SVM that cannot handle NaNs:
    - Compute column means on training data
    - Fill NaNs in both train and test using those means
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    num_cols = X_train.select_dtypes(include=[np.number]).columns
    means = X_train[num_cols].mean()

    X_train[num_cols] = X_train[num_cols].fillna(means)
    X_test[num_cols] = X_test[num_cols].fillna(means)

    return X_train, X_test


# LOSO 
def evaluate_loso(feature_df, model_pretty, model_module):
    """
    LOSO = Leave-One-Subject-Out
    For each subject:
      - Train on all other subjects
      - Test on that subject
    """
    print("\n[LOSO] Starting LOSO evaluation...")

    if "subject" not in feature_df.columns:
        raise ValueError("feature_df must contain 'subject' column for LOSO.")

    X_cols = [
        c for c in feature_df.columns
        if c not in ["class_idx", "class_name", "subject", "letter",
                     "trial", "file_path", "window_len"]
    ]
    X_all = feature_df[X_cols]
    y_all = feature_df["class_idx"]

    subjects = sorted(feature_df["subject"].dropna().unique())
    print(f"[LOSO] Subjects: {subjects}")

    all_true, all_pred = [], []
    rows = []

    for subj in subjects:
        test_mask = (feature_df["subject"] == subj)
        train_mask = ~test_mask

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        # Handle NaNs fold-by-fold (needed for SVM, safe for others)
        X_train, X_test = fill_numeric_nans_train_test(X_train, X_test)

        model = model_module.create_model()  # all model modules define this
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        rows.append({"subject": subj, "n_samples": len(y_test), "accuracy": acc})
        print(f"[LOSO] Subject {subj}: acc={acc:.4f}")

        all_true.append(pd.Series(y_test))
        all_pred.append(pd.Series(y_pred))

    # Combine all folds
    y_true = pd.concat(all_true, ignore_index=True)
    y_pred = pd.concat(all_pred, ignore_index=True)

    overall_acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print("\n[LOSO] Overall metrics:")
    print(f"  Accuracy     : {overall_acc:.4f}")
    print(f"  Precision(m) : {prec_macro:.4f}")
    print(f"  Recall(m)    : {rec_macro:.4f}")
    print(f"  F1-score(m)  : {f1_macro:.4f}\n")

    print("[LOSO] Classification report (per class):")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    out_path = f"outputs/cm_loso_{model_pretty}.png"
    plot_cm(y_true, y_pred, CLASS_NAMES, f"LOSO – {model_pretty}", out_path)

    # You can use rows DataFrame + printed metrics for your report.
    return overall_acc, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="rf",
        help="Choose model: rf, dt, svm, nb, ada, xgb",
    )
    args = parser.parse_args()

    model_pretty, model_module = load_model_module(args.model)
    print(f"\n=== EVALUATING MODEL: {model_pretty} ===")

    print("[1] Loading folder 1+2...")
    df_files = load_kuhar_timeseries_multi(BASE_DIR, MULTI_SPLIT_DIRS)

    print("[2] Building features from folders 1+2...")
    feat_files = build_feature_dataset(df_files)

    print("[3] Loading folder 3 subsamples...")
    subsamples = load_kuhar_subsamples(BASE_DIR)

    print("[4] Building features from folder 3...")
    feat_sub = build_feature_dataset_from_subsamples(subsamples)

    # Coarse NaN fill at dataset level (still refined per-fold later)
    feat_files = feat_files.apply(
        lambda col: col.fillna(col.mean()) if col.dtype != "object" else col
    )
    feat_sub = feat_sub.apply(
        lambda col: col.fillna(col.mean()) if col.dtype != "object" else col
    )

    # Combine for LOSO over all folders
    feature_df = pd.concat([feat_files, feat_sub], ignore_index=True)

    # LOSO (all data)
    loso_acc, loso_table = evaluate_loso(feature_df, model_pretty, model_module)
    out_csv = f"outputs/loso_table_{model_pretty}.csv"
    pd.DataFrame(loso_table).to_csv(out_csv, index=False)
    print(f"[LOSO] Per-subject accuracy table saved to {out_csv}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
