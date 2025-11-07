# folder naming convention from dataset: "0.Stand", "1.Sit"
CLASS_FOLDER_PATTERN = r"^(\d+)\.(.+)$"

# Use the pre-processed split first
DEFAULT_SPLIT_DIR = "2.Trimmed_interpolated_data"

# Human activity labels (index-aligned: 0..17)
CLASS_NAMES = [
    "Stand","Sit","Talk-sit","Talk-stand","Stand-sit","Lay","Lay-stand",
    "Pick","Jump","Push-up","Sit-up","Walk","Walk-backwards",
    "Walk-circle","Run","Stair-up","Stair-down","Table-tennis"
]
