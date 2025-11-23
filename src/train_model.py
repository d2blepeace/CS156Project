"""
Train gesture-recognition models on KU-HAR data and save:
- trained model (.pkl) to trained_models/
- 80/20 confusion matrix to outputs/confusion_matrix_80_20_<Model>.png

Usage examples:
    py src/train_model.py --model rf (random forest will be default, can be used without flag)
    py src/train_model.py --model svm
    py src/train_model.py --model dt
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from constants import CLASS_NAMES
from data_loader import (
    BASE_DIR,
    load_kuhar_timeseries_multi,
    load_kuhar_subsamples,
    MULTI_SPLIT_DIRS,
)
from features import build_feature_dataset, build_feature_dataset_from_subsamples
"""
Model registry: key -> (pretty_name, module_path)
Each module must implement:
  - train_model(X, y) -> (model, X_test, y_test, y_pred)
  - save_model(model) -> saves to its own MODEL_PATH
"""
MODEL_REGISTRY = {
    "rf":  ("RandomForest", "ml_models.random_forest"),
    "dt":  ("DecisionTree", "ml_models.decision_tree"),
    "svm": ("SVM",          "ml_models.svm"),
    "nb":  ("NaiveBayes",   "ml_models.naive_bayes"),
    "ada": ("AdaBoost",     "ml_models.adaboost"),
    "xgb": ("XGBoost",      "ml_models.xgboost"),
}

def get_model_module(key: str):
    """Dynamically import the selected model module."""
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key '{key}'. Valid keys: {list(MODEL_REGISTRY.keys())}")
    _, module_path = MODEL_REGISTRY[key]
    import importlib

    return importlib.import_module(module_path)


# 80/20 Confusion Matrix plotting
def plot_confusion_matrix_80_20(model, X_test, y_test, class_names, model_name: str):
    """Save confusion matrix for the given model to outputs/."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (80/20 Train-Test Split) – {model_name}")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"confusion_matrix_80_20_{model_name}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {out_path}")



# Main training pipeline
def main():
    parser = argparse.ArgumentParser(description="Train KU-HAR models with 80/20 split.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="rf",
        help="Which model to train: "
             "rf (RandomForest), dt (DecisionTree), svm, nb, ada, xgb",
    )
    args = parser.parse_args()

    model_key = args.model
    model_pretty_name, _ = MODEL_REGISTRY[model_key]
    print(f"\n=== Training model: {model_pretty_name} ({model_key}) ===\n")

    # Dynamically load the requested model module
    model_module = get_model_module(model_key)

    # 1) Load and index data from folders 1 + 2
    print("[1] Loading file-based data from folders 1 + 2...")
    df_files = load_kuhar_timeseries_multi(BASE_DIR, MULTI_SPLIT_DIRS)
    print(f"    -> {len(df_files)} file entries")

    # 2) Build feature dataset from 1 + 2
    print("[2] Building feature dataset from 1 + 2...")
    feat_files = build_feature_dataset(df_files)
    print(f"    -> feat_files shape: {feat_files.shape}")

    # 3) Load subsamples from folder 3
    print("[3] Loading subsamples from folder 3...")
    subsamples = load_kuhar_subsamples(BASE_DIR)
    print(f"    -> {len(subsamples)} subsample windows")

    # 4) Features from subsamples (3)
    print("[4] Building feature dataset from subsamples (3)...")
    feat_sub = build_feature_dataset_from_subsamples(subsamples)
    print(f"    -> feat_sub shape: {feat_sub.shape}")

    # 5) Combine all feature rows
    print("[5] Concatenating all feature rows...")
    feature_df = pd.concat([feat_files, feat_sub], ignore_index=True)
    print(f"    -> Combined feature_df shape: {feature_df.shape}")

    # Simple NaN handling: replace any missing values with column means
    # (SVM cannot handle NaNs at all)
    feature_df = feature_df.apply(
        lambda col: col.fillna(col.mean()) if col.dtype != "object" else col
    )
    
    # Decide which columns are features vs labels/meta
    drop_cols = [
        "class_idx",
        "class_name",
        "subject",
        "letter",
        "trial",
        "file_path",   # only exists for 1+2
        "window_len",  # only exists for 3
    ]
    drop_cols = [c for c in drop_cols if c in feature_df.columns]

    X = feature_df.drop(columns=drop_cols)
    y = feature_df["class_idx"]

    # 6) Train selected model (80/20 split)
    #    train_model must return: model, X_test, y_test, y_pred
    print(f"[6] Training {model_pretty_name} on all sources (80/20 split)...")
    model, X_test, y_test, y_pred = model_module.train_model(X, y)

    # 7) Save model using its module's save_model() helper
    print("[7] Saving model...")
    model_module.save_model(model)

    # 8) Confusion matrix for this model
    print("[8] Plotting confusion matrix (80/20)...")
    plot_confusion_matrix_80_20(model, X_test, y_test, CLASS_NAMES, model_pretty_name)

    print(f"\nTraining completed – {model_pretty_name} uses data from folders 1, 2, and 3.\n")


if __name__ == "__main__":
    main()
