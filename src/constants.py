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

# WINDOWING CONFIGURATION FOR REAL-TIME PREDICTION

# Window size: number of samples per prediction window
# Must match the size used during training
WINDOW_SIZE = 50

# Step size (stride): number of samples between consecutive window predictions
# - If WINDOW_STEP_SIZE == WINDOW_SIZE: non-overlapping windows (1 prediction per window)
# - If WINDOW_STEP_SIZE < WINDOW_SIZE: overlapping windows (more frequent predictions)
#   Example: WINDOW_SIZE=50, WINDOW_STEP_SIZE=25 -> 50% overlap
# - If WINDOW_STEP_SIZE > WINDOW_SIZE: gaps between windows (less frequent predictions)
WINDOW_STEP_SIZE = 25

# Padding strategy for incomplete windows at startup:
# - 'none': Skip predictions until first full window is ready (more conservative)
# - 'zero': Pad with zero values at the beginning
# - 'repeat': Repeat the first sample to fill the window
WINDOW_PADDING = 'none'

# Maximum internal buffer size to prevent unbounded memory growth (samples)
WINDOW_MAX_BUFFER_SIZE = 1000

# SAMPLING RATE (used in feature extraction)

# Assumed sampling rate for KU-HAR accelerometer (Hz)
# phyphox typically polls at ~10 Hz, but actual rate may vary
ACCELEROMETER_SAMPLING_RATE = 25.0  # Hz

