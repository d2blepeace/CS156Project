# test_model.py
import sys, os
from src.model import train_model, save_model
from src.features import build_feature_dataset
from src.data_loader import load_kuhar_timeseries, BASE_DIR
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

def main():
    # Load dataset
    print("Loading dataset...")
    df = load_kuhar_timeseries(BASE_DIR)
    print(f"Loaded {len(df)} samples.")

    # Build feature dataset
    print("Building feature dataset...")
    feature_df = build_feature_dataset(df)
    print(f"Built feature dataset with shape: {feature_df.shape}")

    # Split features and labels
    X = feature_df.drop(
        columns=["class_idx", "class_name", "subject", "letter", "trial", "file_path"]
    )
    Y = feature_df["class_idx"]

    # Train model
    print("Training model...")
    model = train_model(X, Y)

    #Save trained model
    print("\nSaving trained model...")
    save_model(model)

    print("\nDONE")

if __name__ == "__main__":
    main() 