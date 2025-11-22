"""
- LOSO (Leave-One-Subject-Out)

- LOEO (Leave-One-Event-Out)

- Confusion matrices for each

- Accuracy analysis
"""
import os 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend (no Tk / no windows)
import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score, confusion_matrix

from constants import CLASS_NAMES
from model import plot_confusion_matrix
from data_loader import (
    BASE_DIR,
    load_kuhar_timeseries_multi,
    load_kuhar_subsamples,
    MULTI_SPLIT_DIRS,
)
from features import build_feature_dataset, build_feature_dataset_from_subsamples
from sklearn.ensemble import RandomForestClassifier

# LOSO: Leave-One-Subject-Out
def evaluate_loso(feature_df, class_names, n_estimators=120, max_depth=20):
    """
    LOSO = each subject is one fold.
    For each subject:
      - Train on all other subjects
      - Test on that subject's samples
    """
    if "subject" not in feature_df.columns:
        raise ValueError("feature_df must contain a 'subject' column for LOSO.")

    X_cols = [
        c for c in feature_df.columns
        if c not in ["class_idx", "class_name", "subject", "letter", "trial",
                     "file_path", "window_len"]
    ]
    X_all = feature_df[X_cols]
    y_all = feature_df["class_idx"]
    subjects = sorted(feature_df["subject"].dropna().unique())

    print(f"\n[LOSO] Unique subjects: {subjects}")
    all_true = []
    all_pred = []
    per_subject_rows = []

    for subj in subjects:
        test_mask = feature_df["subject"] == subj
        train_mask = ~test_mask

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        per_subject_rows.append({
            "subject": subj,
            "n_samples": len(y_test),
            "accuracy": acc,
        })

        all_true.append(pd.Series(y_test))
        all_pred.append(pd.Series(y_pred))

        print(f"[LOSO] Subject {subj}: n={len(y_test)}, acc={acc:.4f}")

    if not all_true:
        raise RuntimeError("No LOSO folds were evaluated (check 'subject' values).")

    y_true_all = pd.concat(all_true, ignore_index=True)
    y_pred_all = pd.concat(all_pred, ignore_index=True)

    overall_acc = accuracy_score(y_true_all, y_pred_all)
    print(f"\n[LOSO] Overall accuracy across all subjects: {overall_acc:.4f}")

    # Confusion matrix across all LOSO predictions
    cm = confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=range(len(class_names))
    )

    # Plot
    plot_confusion_matrix(
        y_true_all,
        y_pred_all,
        class_names=class_names,
        title="Confusion Matrix – LOSO",
        save_path=os.path.join("outputs", "cm_loso.png"),
        show=False
    )
    per_subject_df = pd.DataFrame(per_subject_rows)
    return overall_acc, cm, per_subject_df


# LOEO: Leave-One-Event-Out
def evaluate_loeo(
    feat_files,
    class_names,
    n_estimators=50,     # lighter model for 1916 folds
    max_depth=12,        # also lighter
):
    """
    LOEO = Leave-One-Event-Out
    Runs full LOEO on ALL events (no cap).
    """

    # Ensure metadata exists
    needed = ["subject", "letter", "trial"]
    for col in needed:
        if col not in feat_files.columns:
            raise ValueError(f"feat_files must contain '{col}' for LOEO.")

    # Build event-id
    tmp = feat_files.dropna(subset=["subject", "letter", "trial"]).copy()
    tmp["event_id"] = (
        tmp["subject"].astype(str)
        + "_" + tmp["letter"].astype(str)
        + "_" + tmp["trial"].astype(str)
    )

    X_cols = [
        c for c in tmp.columns
        if c not in ["class_idx", "class_name",
                     "subject", "letter", "trial", "file_path", "event_id"]
    ]

    X_all = tmp[X_cols]
    y_all = tmp["class_idx"]

    events = sorted(tmp["event_id"].unique())
    print(f"\n[LOEO] Running full LOEO on ALL {len(events)} events.")

    all_true = []
    all_pred = []
    per_event_rows = []

    # Evaluate ALL events
    for i, ev in enumerate(events, start=1):
        test_mask = tmp["event_id"] == ev
        train_mask = ~test_mask

        X_train, X_test = X_all[train_mask], X_all[test_mask]
        y_train, y_test = y_all[train_mask], y_all[test_mask]

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            # IMPORTANT: avoid CPU overload
            n_jobs=1                                         
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        per_event_rows.append({
            "event_id": ev,
            "n_samples": len(y_test),
            "accuracy": acc,
        })

        all_true.append(pd.Series(y_test))
        all_pred.append(pd.Series(y_pred))

        # PROGRESS PRINT
        if i == 1 or i % 25 == 0 or i == len(events):
            print(f"[LOEO] Event {i}/{len(events)} → acc={acc:.4f}")

    # Merge predictions
    y_true_all = pd.concat(all_true, ignore_index=True)
    y_pred_all = pd.concat(all_pred, ignore_index=True)

    overall_acc = accuracy_score(y_true_all, y_pred_all)
    print(f"\n[LOEO] Overall accuracy: {overall_acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=range(len(class_names))
    )

    plot_confusion_matrix(
        y_true_all,
        y_pred_all,
        class_names=class_names,
        title="Confusion Matrix – LOEO (Full)",
        save_path=os.path.join("outputs", "cm_loeo_full.png"),
        show=False
    )

    per_event_df = pd.DataFrame(per_event_rows)
    return overall_acc, cm, per_event_df


# build features, then run LOSO & LOEO 
def main():
    print("[1] Loading file-based data from folders 1 + 2...")
    df_files = load_kuhar_timeseries_multi(BASE_DIR, MULTI_SPLIT_DIRS)
    print(f"    -> {len(df_files)} file entries")

    print("[2] Building feature dataset from 1 + 2...")
    feat_files = build_feature_dataset(df_files)
    print(f"    -> feat_files shape: {feat_files.shape}")

    print("[3] Loading subsamples from folder 3...")
    subsamples = load_kuhar_subsamples(BASE_DIR)
    print(f"    -> {len(subsamples)} subsample windows")

    print("[4] Building feature dataset from subsamples (3)...")
    feat_sub = build_feature_dataset_from_subsamples(subsamples)
    print(f"    -> feat_sub shape: {feat_sub.shape}")

    print("[5] Concatenating all feature rows for LOSO...")
    feature_df = pd.concat([feat_files, feat_sub], ignore_index=True)
    print(f"    -> Combined feature_df shape: {feature_df.shape}")

    # LOSO on all data (1 + 2 + 3)
    print("\n*LOSO EVALUATION (all folders)*")
    loso_acc, loso_cm, loso_table = evaluate_loso(feature_df, CLASS_NAMES)
    print("\nLOSO per-subject accuracy table:")
    print(loso_table)

    # LOEO on file-based data only (1 + 2)
    print("\n*LOEO EVALUATION (folders 1 + 2 only)*")
    loeo_acc, loeo_cm, loeo_table = evaluate_loeo(feat_files, CLASS_NAMES)

    print("\nLOEO per-event accuracy table:")
    print(loeo_table)

if __name__ == "__main__":
    main()
    
