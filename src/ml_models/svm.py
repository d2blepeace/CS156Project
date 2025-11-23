import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import matplotlib
matplotlib.use("Agg")
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

MODEL_NAME = "SVM-RBF"

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "trained_models", "svm_rbf.pkl")
)

def train_model(X, y):
    """
    Train an RBF-kernel SVM with an 80/20 train–test split.
    Returns: (model, X_test, y_test, y_pred)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # RBF SVM – can be slower than tree-based models, but a good comparison
    model = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        decision_function_shape="ovo"  # good for multi-class
        # no random_state needed for SVC in this context
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
    """Save trained SVM model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[{MODEL_NAME}] Model saved at {path}")


def load_model(path=MODEL_PATH):
    """Load a previously trained SVM model."""
    return joblib.load(path)


def print_model_table(y_test, y_pred):
    """Pretty table of evaluation metrics (same style as other models)."""
    metrics = [
        ["Accuracy",  accuracy_score(y_test, y_pred)],
        ["Precision", precision_score(y_test, y_pred, average="weighted")],
        ["Recall",    recall_score(y_test, y_pred, average="weighted")],
        ["F1-score",  f1_score(y_test, y_pred, average="weighted")],
    ]
    print("\nEVALUATION METRICS – SVM")
    print(tabulate(metrics, headers=["Metric", "Score"], tablefmt="pretty", floatfmt=".4f"))


def plot_confusion_matrix(y_true, y_pred, class_names, title="SVM – Confusion Matrix", save_path=None, show=True):
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
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    )
