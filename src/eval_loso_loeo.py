"""
Evaluate KU-HAR models using:
- LOSO (Leave-One-Subject-Out)
- LOEO (Leave-One-Event-Out)
Supports multiple models
Example use: 
py src/eval_loso_loeo.py --model rf
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")       # <-- NO GUI backend
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from constants import CLASS_NAMES
from data_loader import (
    BASE_DIR, load_kuhar_timeseries_multi,
    load_kuhar_subsamples, MULTI_SPLIT_DIRS,
)
from features import build_feature_dataset, build_feature_dataset_from_subsamples

# MODEL REGISTRY
MODEL_REGISTRY = {
    "rf":  ("RandomForest", "ml_models.random_forest"),
    "dt":  ("DecisionTree", "ml_models.decision_tree"),
    "svm": ("SVM",          "ml_models.svm"),
    "nb":  ("NaiveBayes",   "ml_models.naive_bayes"),
    "ada": ("AdaBoost",     "ml_models.adaboost"),
    "xgb": ("XGBoost",      "ml_models.xgboost_model"),
}

def load_model_module(key: str):
    import importlib
    name, path = MODEL_REGISTRY[key]
    return name, importlib.import_module(path)

# PLOTTER
def plot_cm(y_true, y_pred, class_names, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(10,8))
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

# LOSO
def evaluate_loso(feature_df, model_pretty, model_module):
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
        test_mask = feature_df["subject"] == subj
        train_mask = ~test_mask

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        model = model_module.create_model()      # <--- consistent API
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        rows.append({"subject": subj, "n_samples": len(y_test), "accuracy": acc})
        print(f"[LOSO] Subject {subj}: acc={acc:.4f}")

        all_true.append(pd.Series(y_test))
        all_pred.append(pd.Series(y_pred))

    y_true = pd.concat(all_true, ignore_index=True)
    y_pred = pd.concat(all_pred, ignore_index=True)

    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\n[LOSO] Overall accuracy: {overall_acc:.4f}")

    out_path = f"outputs/cm_loso_{model_pretty}.png"
    plot_cm(y_true, y_pred, CLASS_NAMES, f"LOSO – {model_pretty}", out_path)

    return overall_acc, rows

# LOEO
def evaluate_loeo(feat_files, model_pretty, model_module):
    print("\n[LOEO] Running LOEO evaluation...")

    need = ["subject", "letter", "trial"]
    for c in need:
        if c not in feat_files.columns:
            raise ValueError(f"feat_files missing column '{c}'")

    tmp = feat_files.dropna(subset=need).copy()
    tmp["event_id"] = (
        tmp["subject"].astype(str) + "_" +
        tmp["letter"].astype(str) + "_" +
        tmp["trial"].astype(str)
    )

    X_cols = [
        c for c in tmp.columns
        if c not in ["class_idx", "class_name", "subject",
                     "letter", "trial", "file_path", "event_id"]
    ]

    X_all = tmp[X_cols]
    y_all = tmp["class_idx"]
    events = sorted(tmp["event_id"].unique())

    print(f"[LOEO] Total events: {len(events)}")

    all_true, all_pred = [], []
    rows = []

    for i, ev in enumerate(events, start=1):
        test_mask = tmp["event_id"] == ev
        train_mask = ~test_mask

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        model = model_module.create_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        rows.append({"event_id": ev, "n_samples": len(y_test), "accuracy": acc})

        if i == 1 or i % 25 == 0 or i == len(events):
            print(f"[LOEO] Event {i}/{len(events)} acc={acc:.4f}")

        all_true.append(pd.Series(y_test))
        all_pred.append(pd.Series(y_pred))

    y_true = pd.concat(all_true, ignore_index=True)
    y_pred = pd.concat(all_pred, ignore_index=True)

    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\n[LOEO] Overall accuracy: {overall_acc:.4f}")

    out_path = f"outputs/cm_loeo_{model_pretty}.png"
    plot_cm(y_true, y_pred, CLASS_NAMES, f"LOEO – {model_pretty}", out_path)

    return overall_acc, rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="rf",
        help="Choose model: rf, dt, svm, nb, ada, xgb"
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

    feature_df = pd.concat([feat_files, feat_sub], ignore_index=True)

    # LOSO (all data)
    loso_acc, loso_table = evaluate_loso(feature_df, model_pretty, model_module)
    pd.DataFrame(loso_table).to_csv(f"outputs/loso_table_{model_pretty}.csv")

    # LOEO (only folders 1+2)
    loeo_acc, loeo_table = evaluate_loeo(feat_files, model_pretty, model_module)
    pd.DataFrame(loeo_table).to_csv(f"outputs/loeo_table_{model_pretty}.csv")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
