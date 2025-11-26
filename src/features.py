"""
Transform raw time-series into feature vectors for ML models.

For each signal (column or window), compute:
- mean
- std
- min
- max
- median
- RMS (root-mean-square)
- ZCR (zero-crossing rate)

FREQUENCY-DOMAIN FEATURES (via FFT)
- dominant frequency
- spectral energy
- spectral entropy
"""

from typing import Dict
import numpy as np
import pandas as pd

from data_loader import load_one_csv


# Assumed sampling rate for KU-HAR (Hz).
FS = 25.0


def _prepare_signal(series: pd.Series) -> np.ndarray:
    """
    Convert a pandas Series to a clean 1D numpy array:
    - cast to float
    - drop NaNs
    - if empty after drop, return a length-1 zero array
    """
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.zeros(1, dtype=float)
    return x


def _compute_time_features(x: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Time-domain features for a 1D signal x.
    prefix: string like "col0" or "win" used in the feature name.
    """
    feats = {}

    feats[f"{prefix}_mean"] = float(np.mean(x))
    feats[f"{prefix}_std"] = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    feats[f"{prefix}_min"] = float(np.min(x))
    feats[f"{prefix}_max"] = float(np.max(x))
    feats[f"{prefix}_median"] = float(np.median(x))

    # RMS
    feats[f"{prefix}_rms"] = float(np.sqrt(np.mean(x**2)))

    # ZCR: fraction of sign changes
    if x.size > 1:
        signs = np.sign(x)
        # sign changes where product < 0
        zc = np.mean(signs[1:] * signs[:-1] < 0)
    else:
        zc = 0.0
    feats[f"{prefix}_zcr"] = float(zc)

    # "energy" in time-domain 
    feats[f"{prefix}_energy_td"] = float(np.sum(x**2) / x.size)

    return feats


def _compute_freq_features(x: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Frequency-domain features using real FFT:
    - dominant frequency
    - spectral energy
    - spectral entropy (normalized 0..1)
    """
    feats = {}

    # Real FFT and corresponding frequencies
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / FS)

    power = np.abs(X) ** 2

    # Avoid all-zero power
    total_power = np.sum(power)
    if total_power <= 0:
        feats[f"{prefix}_dom_freq"] = 0.0
        feats[f"{prefix}_spec_energy"] = 0.0
        feats[f"{prefix}_spec_entropy"] = 0.0
        return feats

    # Dominant frequency: peak in power spectrum
    if power.size > 1:
        idx = np.argmax(power[1:]) + 1
    else:
        idx = 0
    dom_freq = freqs[idx]

    # Spectral energy (FFT-based)
    spec_energy = float(total_power / power.size)

    # Spectral entropy
    p = power / total_power  # normalize to a probability distribution
    # Avoid log(0)
    p_nonzero = p[p > 0]
    H = -np.sum(p_nonzero * np.log2(p_nonzero))
    # Normalize by log2(N) to get 0..1
    H_norm = float(H / np.log2(p_nonzero.size)) if p_nonzero.size > 1 else 0.0

    feats[f"{prefix}_dom_freq"] = float(dom_freq)
    feats[f"{prefix}_spec_energy"] = spec_energy
    feats[f"{prefix}_spec_entropy"] = H_norm

    return feats


# FOLDER 1 + 2: FILE-BASED TIME SERIES 

def compute_basic_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute time + frequency features per column of a multichannel time series.

    For each column k in df, we produce features like:
        col0_mean, col0_std, col0_rms, col0_zcr, col0_energy_td,
        col0_dom_freq, col0_spec_energy, col0_spec_entropy, ...
    """
    feats: Dict[str, float] = {}

    for col in df.columns:
        prefix = f"col{col}"
        x = _prepare_signal(df[col])

        # Time-domain
        feats.update(_compute_time_features(x, prefix=prefix))

        # Frequency-domain
        feats.update(_compute_freq_features(x, prefix=prefix))

    return feats


def extract_features_from_csv(csv_path: str) -> Dict[str, float]:
    """
    Load a raw time-series CSV and compute statistical + spectral features.
    """
    df = load_one_csv(csv_path)
    feats = compute_basic_stats(df)
    feats["file_path"] = csv_path  # keep for traceability/debugging
    return feats


def build_feature_dataset(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Loop over all file entries in index_df (from folders 1+2),
    computing a feature row per file.

    index_df rows must contain:
        - file_path
        - class_idx
        - class_name
        - subject
        - letter
        - trial
    """
    feature_records = []

    for _, row in index_df.iterrows():
        csv_path = row["file_path"]
        feats = extract_features_from_csv(csv_path)

        # Add label/meta info
        feats.update({
            "class_idx":  row["class_idx"],
            "class_name": row["class_name"],
            "subject":    row["subject"],
            "letter":     row["letter"],
            "trial":      row["trial"],
        })
        feature_records.append(feats)

    return pd.DataFrame(feature_records)


# FOLDER 3: SUBSAMPLE WINDOWS 

def build_feature_dataset_from_subsamples(subsample_df: pd.DataFrame) -> pd.DataFrame:
    """
    subsample_df: rows = windows, sensor columns + meta:
        sensor_cols: measurement across time (already flattened)
        meta_cols:  subject, window_len, class_idx, class_name, serial_no

    Here we treat each row as a 1D signal and compute:
    - mean, std, min, max, median, RMS, ZCR, energy_td
    - dom_freq, spec_energy, spec_entropy
    """
    meta_cols = ["subject", "window_len", "class_idx", "class_name", "serial_no"]
    sensor_cols = [c for c in subsample_df.columns if c not in meta_cols]

    feature_records = []

    for _, row in subsample_df.iterrows():
        sensor_series = row[sensor_cols].astype(float)
        x = _prepare_signal(sensor_series)

        # Time-domain features (no prefix to keep column names short)
        feats = {
            "mean":   float(np.mean(x)),
            "std":    float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
            "min":    float(np.min(x)),
            "max":    float(np.max(x)),
            "median": float(np.median(x)),
            "rms":    float(np.sqrt(np.mean(x**2))),
        }

        # ZCR
        if x.size > 1:
            signs = np.sign(x)
            zc = np.mean(signs[1:] * signs[:-1] < 0)
        else:
            zc = 0.0
        feats["zcr"] = float(zc)

        # Time-domain energy
        feats["energy_td"] = float(np.sum(x**2) / x.size)

        # Frequency-domain features
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(x.size, d=1.0 / FS)
        power = np.abs(X) ** 2
        total_power = np.sum(power)

        if total_power <= 0:
            dom_freq = 0.0
            spec_energy = 0.0
            spec_entropy = 0.0
        else:
            if power.size > 1:
                idx = np.argmax(power[1:]) + 1
            else:
                idx = 0
            dom_freq = float(freqs[idx])
            spec_energy = float(total_power / power.size)

            p = power / total_power
            p_nonzero = p[p > 0]
            H = -np.sum(p_nonzero * np.log2(p_nonzero))
            spec_entropy = float(
                H / np.log2(p_nonzero.size)
            ) if p_nonzero.size > 1 else 0.0

        feats["dom_freq"] = dom_freq
        feats["spec_energy"] = spec_energy
        feats["spec_entropy"] = spec_entropy

        # Attach meta
        feats.update({
            "class_idx":  row["class_idx"],
            "class_name": row["class_name"],
            "subject":    row["subject"],
            "window_len": row["window_len"],
            "serial_no":  row["serial_no"],
        })

        feature_records.append(feats)

    return pd.DataFrame(feature_records)
