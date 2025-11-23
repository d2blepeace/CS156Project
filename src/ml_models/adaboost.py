#AdaBoost
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tabulate import tabulate

MODEL_NAME = "AdaBoost-Tree"

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "trained_models", "adaboost.pkl")
)


def _fill_numeric_nans_train_test(X_train, X_test):
    """
    AdaBoost/DecisionTree do not accept NaNs.
    Fill numeric NaNs using training-column means and apply same to test.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    num_cols = X_train.select_dtypes(include=[np.number]).columns
    means = X_train[num_cols].mean()

    X_train[num_cols] = X_train[num_cols].fillna(means)
    X_test[num_cols] = X_test[num_cols].fillna(means)

    return X_train, X_test


def _build_adaboost_estimator():
    """
    Helper to build a fresh AdaBoost classifier.
    Used both in train_model() and create_model() so they match.
    """
    base_tree = DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
    )

    model = AdaBoostClassifier(
        estimator=base_tree,   # for newer sklearn; was base_estimator before
        n_estimators=120,      # number of boosting rounds
        learning_rate=0.5,
        random_state=42,
    )
    return model


def train_model(X, y):
    """
    Train AdaBoost with an 80/20 train–test split.
    Returns: (model, X_test, y_test, y_pred)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Handle NaNs
    X_train, X_test = _fill_numeric_nans_train_test(X_train, X_test)

    model = _build_adaboost_estimator()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n[{MODEL_NAME}] 80/20 EVALUATION")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}\n")

    print(classification_report(y_test, y_pred, zero_division=0))

    return model, X_test, y_test, y_pred


def save_model(model, path=MODEL_PATH):
    """Save trained AdaBoost model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[{MODEL_NAME}] Model saved at {path}")


def load_model(path=MODEL_PATH):
    """Load a previously trained AdaBoost model."""
    return joblib.load(path)


def print_model_table(y_test, y_pred):
    """Pretty table of evaluation metrics (same style as other models)."""
    metrics = [
        ["Accuracy",  accuracy_score(y_test, y_pred)],
        ["Precision", precision_score(y_test, y_pred, average="weighted", zero_division=0)],
        ["Recall",    recall_score(y_test, y_pred, average="weighted", zero_division=0)],
        ["F1-score",  f1_score(y_test, y_pred, average="weighted", zero_division=0)],
    ]
    print(f"\nEVALUATION METRICS – {MODEL_NAME}")
    print(tabulate(metrics, headers=["Metric", "Score"], tablefmt="pretty", floatfmt=".4f"))


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    title=f"{MODEL_NAME} – Confusion Matrix",
    save_path=None,
    show=True,
):
    """Optional confusion matrix helper just for this model."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def create_model():
    """
    Factory used by eval_loso_loeo.py for LOSO/LOEO evaluation.
    Must return a *fresh* untrained estimator.
    """
    return _build_adaboost_estimator()
