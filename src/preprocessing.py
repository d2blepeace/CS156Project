"""
Signal preprocessing ultilities for KUHAR data:
- Handle missing values
- Aplly filter for smoothing the noise
"""

import pandas as pd
import numpy as np

# Linearly interpolate missing values in the data frame
def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    #Linear interpolation along the time axis (rows)
    out = out.interpolate(method='linear', axis = 0, limit_direction='both')
    # fill any remaining NaNs
    out = out.fillna(out.mean())
    return out

# Low-pass filer using a centered moving avg
# window = num of samples in the avg window
def moving_average(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    return df.rolling(window=window, center=True, min_periods=1).mean()

# Create high-pass filter by subtracting the moving avg from Original Signal
def highpass_from_na(df: pd.DataFrame, window: int =5 ) -> pd.DataFrame:
    ma = moving_average(df, window=window)
    return df - ma
    
# Full preprocessing pipeline used before feature extraction / plotting
def smooth_and_clean(df: pd.DataFrame, ma_window: int = 5) -> pd.DataFrame:
    df_clean = interpolate_missing(df)
    df_smooth = moving_average(df_clean, window=ma_window)
    return df_smooth





