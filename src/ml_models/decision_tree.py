import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use("Agg")  # non-GUI backend (safe with Tk issues)
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
import numpy as np

# For reference / logging
MODEL_NAME = "DecisionTree"

# Where the trained model will be stored
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "trained_models", "decision_tree.pkl")
)

def train_model(X, y):
    """
    Train a Decision Tree classifier with an 80/20 train–test split.
    Returns: (model, X_test, y_test, y_pred)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # A reasonably regularized decision tree
    model = DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n[{MODEL_NAME}] 80/20 EVALUATION")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}\n")

    print(classification_report(y_test, y_pred))

    return model, X_test, y_test, y_pred


def save_model(model, path=MODEL_PATH):
    """Save trained Decision Tree model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[{MODEL_NAME}] Model saved at {path}")


def load_model(path=MODEL_PATH):
    """Load a previously trained Decision Tree model."""
    return joblib.load(path)


def print_model_table(y_test, y_pred):
    """Pretty table of evaluation metrics (same style as Random Forest)."""
    metrics = [
        ["Accuracy",  accuracy_score(y_test, y_pred)],
        ["Precision", precision_score(y_test, y_pred, average="weighted")],
        ["Recall",    recall_score(y_test, y_pred, average="weighted")],
        ["F1-score",  f1_score(y_test, y_pred, average="weighted")],
    ]
    print("\nEVALUATION METRICS – Decision Tree")
    print(tabulate(metrics, headers=["Metric", "Score"], tablefmt="pretty", floatfmt=".4f"))


def plot_confusion_matrix(y_true, y_pred, class_names, title="Decision Tree – Confusion Matrix", save_path=None, show=True):
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
    return DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )
