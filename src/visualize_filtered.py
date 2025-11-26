# Compare raw vs filtered visualization of signal for multiple activities
import numpy as np
import matplotlib.pyplot as plt 
from preprocessing import interpolate_missing, moving_average
from data_loader import (
    BASE_DIR,
    load_kuhar_timeseries_multi,
    MULTI_SPLIT_DIRS,
    load_one_csv
)


FS = 25.0          # samples per second 
DURATION_SEC = 10  # plot first 10 seconds
TARGET_CLASSES = ["Walk", "Jump", "Stand"]   # activities to visualize


def plot_filtered_for_class(df_index, class_name):
    # Filter index for this class
    rows = df_index[df_index["class_name"] == class_name]
    if rows.empty:
        print(f"[WARN] No samples found for class {class_name}")
        return

    # Pick first example
    row = rows.iloc[0]
    csv_path = row["file_path"]
    print(f"\nVisualizing (before vs after filter) for {class_name}: {csv_path}")
    print(f"class_idx={row['class_idx']}, class_name={row['class_name']}")

    # Load raw time series: columns = channels (e.g., accX, accY, accZ)
    ts_raw = load_one_csv(csv_path)

    # Trim to first N samples for visualization
    n_samples = int(FS * DURATION_SEC)
    ts_raw = ts_raw.iloc[:n_samples, :]
    t = np.arange(len(ts_raw)) / FS  # time axis in seconds

    # Preprocess: interpolate + moving-average smoothing
    ts_interp = interpolate_missing(ts_raw)
    ts_smooth = moving_average(ts_interp, window=5)

    # Plot BEFORE vs AFTER
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: raw signal 
    for col in ts_raw.columns:
        axes[0].plot(t, ts_raw[col], label=f"col{col}")
    axes[0].set_title(f"Raw signal (class: {class_name})")
    axes[0].set_ylabel("Sensor value")
    axes[0].legend()

    # Bottom: filtered signal
    for col in ts_smooth.columns:
        axes[1].plot(t, ts_smooth[col], label=f"col{col}")
    axes[1].set_title("After interpolation + moving-average filter")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Sensor value")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    # Index all files (from default split or multi-split)
    df_index = load_kuhar_timeseries_multi(BASE_DIR, MULTI_SPLIT_DIRS)

    # Generate plots for each target class
    for cls in TARGET_CLASSES:
        plot_filtered_for_class(df_index, cls)


if __name__ == "__main__":
    main()
