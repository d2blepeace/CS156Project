# Use to load the dataset from 1.Raw_time_domain_data and 2.Trimmed_interpolated_data

import pandas as pd
import os, re
from typing import List, Tuple, Dict
from constants import (CLASS_FOLDER_PATTERN, DEFAULT_SPLIT_DIR, 
                       MULTI_SPLIT_DIRS, SUBSAMPLE_DIR, SUBSAMPLE_FILE, CLASS_NAMES)

# Adjust if root folder for your data is elsewhere
#BASE_DIR = r"C:\Users\Administrator\Desktop\CS-156\CS156Project\data"
BASE_DIR = "C:/Users/damju/Downloads/archive"


def parse_class_from_folder(folder_name: str) -> Tuple[int, str]:
    m = re.match(CLASS_FOLDER_PATTERN, folder_name)
    if not m:
        raise ValueError(f"Invalid Folder Name: {folder_name}")
    
    # m.group(1) is the numeric prefix, m.group(2) is the class name
    return int(m.group(1)), m.group(2) 
    
def parse_file_metadata(fname: str) -> Dict[str, str]:
    base = os.path.splitext(fname)[0]
    parts = base.split("_")
    meta = {"subject": None, "letter": None, "trial": None}
    if len(parts) >=3 :
        meta["subject"], meta["letter"], meta["trial"] = parts[0], parts[1], parts[2]
    return meta

# returns the raw numeric time series (rows=time, cols=channels).
def load_one_csv(csv_path: str) -> pd.DataFrame:
    #comma-delimited, no header
    return pd.read_csv(csv_path, header=None)

# return a DataFrame of all CSV files with labels from folder names.
def load_kuhar_timeseries(base_dir: str, split_dir: str = DEFAULT_SPLIT_DIR) -> pd.DataFrame:
    root = os.path.join(base_dir, split_dir)
    record = []
    for cfolder in sorted(os.listdir(root), key=lambda x: int(x.split(".")[0])):
        if not os.path.isdir(os.path.join(root, cfolder)): continue
        idx, name = parse_class_from_folder(cfolder)
        for f in os.listdir(os.path.join(root, cfolder)):
            if not f.endswith(".csv"): continue
            meta = parse_file_metadata(f)
            record.append({
                "file_path": os.path.join(root, cfolder, f),
                "class_idx": idx ,
                "class_name": name ,
                **meta
            })
    return pd.DataFrame(record)

# Load from multiple split directories and combine
def load_kuhar_timeseries_multi(base_dir: str, split_dirs: List[str]) -> pd.DataFrame:
    frames = []
    for split in split_dirs:
        df = load_kuhar_timeseries(base_dir, split)
        df["source_split"] = split
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["file_path"])

# load the 3. Time_domain_subsamples CSV
def load_kuhar_subsamples(base_dir: str) -> pd.DataFrame:
    """
    Load the big subsample CSV from folder 3.

    Layout from the KU-HAR paper:
    - Cols 1–300:   accel X
    - Cols 301–600: accel Y
    - Cols 601–900: accel Z
    - Cols 901–1200: gyro X
    - Cols 1201–1500: gyro Y
    - Cols 1501–1800: gyro Z
    - Col 1801:  class ID  (0..17 or 1..18)
    - Col 1802:  window length
    - Col 1803:  serial number
    """
    csv_path = os.path.join(base_dir, SUBSAMPLE_DIR, SUBSAMPLE_FILE)
    df = pd.read_csv(csv_path, header=None)

    # These are the last 3 columns (0-based indexing)
    class_col  = df.columns[-3]   # class ID
    length_col = df.columns[-2]   # window length
    serial_col = df.columns[-1]   # serial number

    # Raw labels
    raw_labels = df[class_col].astype(int)
    raw_min = int(raw_labels.min())
    raw_max = int(raw_labels.max())
    n_classes = len(CLASS_NAMES)

    # Auto-detect label convention
    if raw_min == 0 and raw_max == n_classes - 1:
        # 0..17
        class_idx = raw_labels
    elif raw_min == 1 and raw_max == n_classes:
        # 1..18 -> shift to 0..17
        print("[info] Subsample labels look 1-based (1..18). Shifting to 0-based (0..17).")
        class_idx = raw_labels - 1
    else:
        raise ValueError(
            f"Unexpected subsample label range: [{raw_min}, {raw_max}] "
            f"for {n_classes} classes. Check folder 3 labeling."
        )

    # Map to class names
    class_name = class_idx.apply(lambda i: CLASS_NAMES[i])

    # Sensor data = all but last 3 columns
    sensor_df = df.iloc[:, :-3].copy()

    # Build combined DataFrame:
    combined = sensor_df.copy()
    combined["class_idx"]   = class_idx
    combined["class_name"]  = class_name
    combined["window_len"]  = df[length_col]
    combined["serial_no"]   = df[serial_col]

    # Optional: add subject for consistency (unknown → None)
    combined["subject"] = None

    return combined

def main():
    print(f"[i] MULTI_SPLIT_DIRS = {MULTI_SPLIT_DIRS}")
    df = load_kuhar_timeseries_multi(base_dir=BASE_DIR, split_dirs=MULTI_SPLIT_DIRS)
    print(f"[✓] Indexed {len(df)} files across {df['class_idx'].nunique()} classes.")
    print(df.head(5))

    # Sanity: paths exist
    missing = df[~df["file_path"].apply(os.path.exists)]
    assert missing.empty, f"Missing files:\n{missing.head()}"

    # Load one example file
    sample_path = df.iloc[0]["file_path"]
    ts = load_one_csv(sample_path)
    print(f"Loaded one CSV: {sample_path}")
    print(f"    shape={ts.shape} (rows=time, cols=channels)")
    print(ts.head(3))

    # Folder name parser sanity
    idx, name = parse_class_from_folder(os.path.basename(os.path.dirname(sample_path)))
    print(f"Folder parser: class_idx={idx}, class_name={name}")

    print("\nAll basic checks passed!")

if __name__ == "__main__":
    main()

