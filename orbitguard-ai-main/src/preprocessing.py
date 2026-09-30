"""
Preprocessing utilities: feature scaling and sliding-window construction.

Scalers must always be FIT on normal training data only, then reused
(never refit) on validation/test/live data — this module's function
signatures are designed to make that the natural way to call them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_scaler(train_df: pd.DataFrame, feature_columns: list) -> StandardScaler:
    """Fit a StandardScaler on training features only."""
    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns].values)
    return scaler


def transform_features(df: pd.DataFrame, feature_columns: list, scaler: StandardScaler) -> np.ndarray:
    """Apply an already-fitted scaler. Never fits."""
    return scaler.transform(df[feature_columns].values)


def make_windows(features: np.ndarray, window_size: int, step_size: int = 1) -> np.ndarray:
    """
    Build sliding windows for the LSTM Autoencoder.

    features: 2D array, shape (n_samples, n_features), already scaled.
    Returns: 3D array, shape (n_windows, window_size, n_features).
    """
    n_samples = features.shape[0]
    if n_samples < window_size:
        return np.empty((0, window_size, features.shape[1]))

    windows = []
    for start in range(0, n_samples - window_size + 1, step_size):
        windows.append(features[start:start + window_size])
    return np.stack(windows, axis=0)


def window_end_indices(n_samples: int, window_size: int, step_size: int = 1) -> list:
    """
    Return the original-dataframe row index corresponding to the LAST row of
    each window, in the same order as make_windows(). Used to align window-
    level anomaly scores back to timestamps/rows for display and fusion.
    """
    if n_samples < window_size:
        return []
    return list(range(window_size - 1, n_samples, step_size))
