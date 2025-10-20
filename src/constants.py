# constants.py

# folder naming convention from dataset: "0.Stand", "1.Sit"
CLASS_FOLDER_PATTERN = r"^(\d+)\.(.+)$"

# 2. default dataset split
DEFAULT_SPLIT_DIR = "2.Trimmed_interpolated_data"

# Human activity label 
CLASS_NAME = [
    "Stand", "Sit", "Talk-sit", "Talk-Stand", "Stand-sit", "Lay", "Lay-stand",
    "Pick", "Jump", "Push-up", "Sit-up", "Walk", "Walk-backwards",
    "Walk-circle", "Run", "Stair-up", "Stair-down", "Table-tennis"
]
