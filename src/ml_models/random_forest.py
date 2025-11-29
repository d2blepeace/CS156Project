import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report)
from tabulate import tabulate
import numpy as np

# Path to save/load the trained model
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "trained_models", "random_forest.pkl")
)

# DEMO 3-class trained model (Stand / Walk / Jump)
DEMO_MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..",
        "trained_models",
        "for_demo",
        "random_forest_demo.pkl"
    )
)

#factory used by eval_loso_loeo 
def create_model(
    n_estimators: int = 120,
    max_depth: int = 20,
    random_state: int = 42,
    n_jobs: int = -1,
):
    """Return a fresh RandomForest model (used by LOSO/LOEO and others)."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs,
    )

#  Training for the 80/20 script
def train_model(X, y):
    """
    Train Random Forest with 80/20 split.
    Returns: (model, X_test, y_test, y_pred)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # use the same factory as LOSO/LOEO
    model = create_model()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}\n")
    print(classification_report(y_test, y_pred))

    return model, X_test, y_test, y_pred

#  Save / load  model 
def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved at {path}")

def load_model(path=MODEL_PATH):
    return joblib.load(path)

# For demo
def save_demo_model(model, path=DEMO_MODEL_PATH):
    """Save the 3-class DEMO model safely."""
    save_model(model, path=path)

def load_demo_model(path=DEMO_MODEL_PATH):
    """Load the 3-class DEMO model easily."""
    return joblib.load(path)

#  Format pretty table 
def print_model_table(y_test, y_pred):
    metrics = [
        ["Accuracy",  accuracy_score(y_test, y_pred)],
        ["Precision", precision_score(y_test, y_pred, average="weighted")],
        ["Recall",    recall_score(y_test, y_pred, average="weighted")],
        ["F1-score",  f1_score(y_test, y_pred, average="weighted")],
    ]
    print("\nEVALUATION METRICS")
    print(tabulate(metrics, headers=["Metric", "Score"],
                   tablefmt="pretty", floatfmt=".4f"))

#  Confusion matrix helper (used by some scripts) 
def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    title="Confusion Matrix",
    save_path=None,
    show=True,
):
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
    """Factory method required by eval_loso_loeo.py"""
    return RandomForestClassifier(
        n_estimators=120,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
