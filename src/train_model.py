import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from constants import CLASS_NAMES
from model import train_model, save_model
from data_loader import (
    BASE_DIR,
    load_kuhar_timeseries_multi,
    load_kuhar_subsamples,
    MULTI_SPLIT_DIRS,
)
from features import build_feature_dataset, build_feature_dataset_from_subsamples
from model import train_model, save_model

# 80/20 Confusion Matrix 
def plot_confusion_matrix_80_20(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (80/20 Train-Test Split)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

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

    print("[5] Concatenating all feature rows...")
    feature_df = pd.concat([feat_files, feat_sub], ignore_index=True)
    print(f"    -> Combined feature_df shape: {feature_df.shape}")

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

    print("[6] Training RandomForest model on all sources...")
    model, X_test, y_test, y_pred = train_model(X, y)

    print("[7] Saving model...")
    save_model(model)

    print("[8] Plotting confusion matrix (80/20)...")
    plot_confusion_matrix_80_20(model, X_test, y_test, CLASS_NAMES)

    print("\nTraining completed – model uses data from folders 1, 2, and 3.")

if __name__ == "__main__":
    main()