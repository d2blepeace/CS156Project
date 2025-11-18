# folder naming convention from dataset: "0.Stand", "1.Sit"
CLASS_FOLDER_PATTERN = r"^(\d+)\.(.+)$"

# Use the pre-processed split by default
DEFAULT_SPLIT_DIR = "2.Trimmed_interpolated_data"

# Loading multiple folders
MULTI_SPLIT_DIRS = [
    "1.Raw_time_domain_data",
    "2.Trimmed_interpolated_data"
]
# Folder + filename for subsamples (3.)
SUBSAMPLE_DIR = "3.Time_domain_subsamples"
SUBSAMPLE_FILE = "KU-Har_time_domain_subsamples_20750x300.csv" 

# Human activity labels (index-aligned: 0..17)
CLASS_NAMES = [
    "Stand","Sit","Talk-sit","Talk-stand","Stand-sit","Lay","Lay-stand",
    "Pick","Jump","Push-up","Sit-up","Walk","Walk-backwards",
    "Walk-circle","Run","Stair-up","Stair-down","Table-tennis"
]
