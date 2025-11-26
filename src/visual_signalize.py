# visualize multiple activity classes: Walk, Jump, Stand
import numpy as np
import matplotlib.pyplot as plt
from data_loader import (
    BASE_DIR,
    load_kuhar_timeseries_multi,
    MULTI_SPLIT_DIRS,
    load_one_csv
)

FS = 25.0
DURATION_SEC = 10
TARGET_CLASSES = ["Walk", "Jump", "Stand"]  # classes to visualize

def plot_class(df_index, class_name):
    df_target = df_index[df_index["class_name"] == class_name]

    if df_target.empty:
        print(f"No samples found for class '{class_name}'")
        return

    row = df_target.iloc[0]
    csv_path = row["file_path"]

    print(f"\nVisualizing {class_name} sample:")
    print(f"  Path: {csv_path}")
    print(f"  Source: {row.get('source_split', '(unknown)')}")

    ts = load_one_csv(csv_path).iloc[:, :3]
    ts.columns = ["accX", "accY", "accZ"]

    n_samples = int(FS * DURATION_SEC)
    ts = ts.iloc[:n_samples, :]
    t = np.arange(len(ts)) / FS

    plt.figure(figsize=(10,5))
    plt.plot(t, ts["accX"], label="X-axis")
    plt.plot(t, ts["accY"], label="Y-axis")
    plt.plot(t, ts["accZ"], label="Z-axis")

    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration")
    plt.title(f"{class_name} – Accelerometer Signals")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    df_index = load_kuhar_timeseries_multi(BASE_DIR, MULTI_SPLIT_DIRS)
    for cls in TARGET_CLASSES:
        plot_class(df_index, cls)


if __name__ == "__main__":
    main()
