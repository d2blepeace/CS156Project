import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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
    # predict on the test set and print accuracy
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    return model

# Save a trained model to storage
def save_model(model, path="models/random_forest_model.pkl"):
    joblib.dump(model, path)
    print(f"Model saved at {path}")

# Load a trained model back into memory
# Used for real-time gesture recognition inference  
def load_model(path="models/random_forest_model.pkl"):
    return joblib.load(path)

