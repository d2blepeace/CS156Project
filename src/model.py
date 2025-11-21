import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    precision_score,
    recall_score,
    f1_score)
from tabulate import tabulate

# Path to save/load the trained model, remember to "cd CS156Project" before running scripts so that the models is in the correct relative path
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "random_forest_model.pkl")
)

# Train a Random Forest classifier
# X = feature matrix (each row = one gesture instance, each column = a feature)
# y = class labels (0–17)
def train_model(X, y):
    # 80/20 split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    """"
    Create a RandomForest classifier
    n_estimators=120  → number of trees (good balance of speed + accuracy)
    max_depth=20      → limit tree growth to prevent overfitting
    n_jobs=-1         → use all CPU cores for training
    random_state=42   → reproducible results
    """
    model = RandomForestClassifier(
        n_estimators=120,       # ~120 trees is light + accurate
        max_depth=20,          # prevents overfitting
        random_state=42,
        n_jobs=-1              # use all CPU cores
    )
    # Train the RandomForest on the training data
    model.fit(X_train, y_train)
    # predict on the test set 
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}\n")

    print(classification_report(y_test, y_pred))

    return model, X_test, y_test, y_pred

# Save a trained model to storage
def save_model(model, path=MODEL_PATH):
    # Ensure the models/ folder exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved at {path}")

# Load a trained model back into memory
# Used for real-time gesture recognition inference  
def load_model(path=MODEL_PATH):
    return joblib.load(path)

def print_model_table(y_test, y_pred):
    metrics = [
        ["Accuracy",  accuracy_score(y_test, y_pred)],
        ["Precision", precision_score(y_test, y_pred, average="weighted")],
        ["Recall",    recall_score(y_test, y_pred, average="weighted")],
        ["F1-score",  f1_score(y_test, y_pred, average="weighted")],
    ]
    print("\nEVALUATION METRICS")
    print(tabulate(metrics, headers=["Metric", "Score"], tablefmt="pretty", floatfmt=".4f"))
 

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    # Tick labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    plt.tight_layout()
    plt.show()

def kfold_confusion_matrix(X, y, class_names, n_splits=5):
    """
    Train and evaluate RandomForest with Stratified K-Fold,
    then plot an aggregated confusion matrix across all folds.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    n_classes = len(class_names)
    total_cm = np.zeros((n_classes, n_classes), dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=range(n_classes))
        total_cm += cm
        print(f"[Fold {fold_idx}] Accuracy = {accuracy_score(y_test, y_pred):.4f}")

    # Plot aggregated confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(total_cm, interpolation="nearest", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix (Stratified {n_splits}-Fold CV)")

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    plt.tight_layout()
    plt.show()

