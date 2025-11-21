# visualize walking accelerometer signals from all folders
import numpy as np
import matplotlib.pyplot as plt
from data_loader import (
    BASE_DIR,
    load_kuhar_timeseries_multi,
    MULTI_SPLIT_DIRS,
    load_one_csv
)

FS = 25.0        # assumed sampling rate (can tweak later if you guys know exact value)
DURATION_SEC = 10
TARGET_CLASS = "Walk"  # change to Sit, Jump etc

def main():
    # Load index from BOTH folder 1 and folder 2
    df_index = load_kuhar_timeseries_multi(BASE_DIR, MULTI_SPLIT_DIRS)

    # Filter for the "Walk" class
    df_target = df_index[df_index["class_name"] == TARGET_CLASS]
    if df_target.empty:
        print("No samples found for class '{TARGET_CLASS}'")
        return

    # Choose the first walking sample
    row = df_target.iloc[0]
    csv_path = row["file_path"]

    print(f"Visualizing WALK sample:")
    print(f"  Path: {csv_path}")
    print(f"  Source folder: {row.get('source_split', '(unknown)')}")

    # Load accelerometer data (first 3 columns)
    ts = load_one_csv(csv_path).iloc[:, :3]
    ts.columns = ["accX", "accY", "accZ"]

    # Limit to first N samples
    n_samples = int(FS * DURATION_SEC)
    ts = ts.iloc[:n_samples, :]
    t = np.arange(len(ts)) / FS

    # Plot wave signals
    plt.figure(figsize=(10, 5))
    plt.plot(t, ts["accX"], label="X-axis")
    plt.plot(t, ts["accY"], label="Y-axis")
    plt.plot(t, ts["accZ"], label="Z-axis")

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (arb. units)")
    plt.title("Walking – accelerometer signals (from folders 1 + 2)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
