# Use to load the dataset from 1.Raw_time_domain_data and 2.Trimmed_interpolated_data

import pandas as pd
import os, re
from typing import List, Tuple, Dict
from constants import CLASS_FOLDER_PATTERN, DEFAULT_SPLIT_DIR

# Adjust if root folder for your data is elsewhere
BASE_DIR = r"C:\Users\Administrator\Desktop\CS-156\CS156Project\data" 

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

def main():
    print(f"[i] DEFAULT_SPLIT_DIR = {DEFAULT_SPLIT_DIR}")
    df = load_kuhar_timeseries(base_dir=BASE_DIR)
    print(f"[Y] Indexed {len(df)} files across {df['class_idx'].nunique()} classes.")
    print(df.head(5))

    # Sanity: paths exist
    missing = df[~df["file_path"].apply(os.path.exists)]
    assert missing.empty, f"Missing files:\n{missing.head()}"

    # Load one example file
    sample_path = df.iloc[0]["file_path"]
    ts = load_one_csv(sample_path)
    print(f"[Y] Loaded one CSV: {sample_path}")
    print(f"    shape={ts.shape} (rows=time, cols=channels)")
    print(ts.head(3))

    # Folder name parser sanity
    idx, name = parse_class_from_folder(os.path.basename(os.path.dirname(sample_path)))
    print(f"[Y] Folder parser: class_idx={idx}, class_name={name}")

    print("\nAll basic checks passed!")

if __name__ == "__main__":
    main()

