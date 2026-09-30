"""
Loads trained models (if present) and computes raw anomaly scores.

Every function here fails loudly and clearly (ModelNotTrainedError) rather
than silently falling back to fake/placeholder scores. Callers (dashboard,
evaluate.py) are expected to catch this and show "Model not trained yet."
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np

from src.config_utils import resolve_path
from src.preprocessing import transform_features, make_windows, window_end_indices


class ModelNotTrainedError(Exception):
    """Raised when a required trained model / scaler / metadata file is missing."""


def _require_file(path: str, what: str) -> str:
    if not os.path.exists(path):
        raise ModelNotTrainedError(
            f"{what} not found at {path}. Model has not been trained yet — "
            f"run the corresponding training script first."
        )
    return path


def isolation_forest_paths() -> dict:
    models_dir = resolve_path("models")
    return {
        "model": os.path.join(models_dir, "isolation_forest.joblib"),
        "scaler": os.path.join(models_dir, "isolation_forest_scaler.joblib"),
        "meta": os.path.join(models_dir, "isolation_forest_meta.json"),
    }


def lstm_paths() -> dict:
    models_dir = resolve_path("models")
    return {
        "model": os.path.join(models_dir, "lstm_autoencoder.keras"),
        "scaler": os.path.join(models_dir, "lstm_scaler.joblib"),
        "meta": os.path.join(models_dir, "lstm_meta.json"),
    }


def isolation_forest_available() -> bool:
    paths = isolation_forest_paths()
    return all(os.path.exists(p) for p in paths.values())


def lstm_available() -> bool:
    paths = lstm_paths()
    return all(os.path.exists(p) for p in paths.values())


def load_isolation_forest():
    paths = isolation_forest_paths()
    _require_file(paths["model"], "Isolation Forest model")
    _require_file(paths["scaler"], "Isolation Forest scaler")
    _require_file(paths["meta"], "Isolation Forest metadata")
    model = joblib.load(paths["model"])
    scaler = joblib.load(paths["scaler"])
    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, scaler, meta


def load_lstm_autoencoder():
    paths = lstm_paths()
    _require_file(paths["model"], "LSTM Autoencoder model")
    _require_file(paths["scaler"], "LSTM Autoencoder scaler")
    _require_file(paths["meta"], "LSTM Autoencoder metadata")

    import tensorflow as tf  # local import: keep TF optional/lazy for dashboard startup

    model = tf.keras.models.load_model(paths["model"])
    scaler = joblib.load(paths["scaler"])
    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, scaler, meta


def isolation_forest_scores(df, feature_columns: list) -> np.ndarray:
    """
    Return raw Isolation Forest anomaly scores, one per row of df.
    Higher = more anomalous (we negate sklearn's score_samples, which is
    higher-is-more-normal by default).
    """
    model, scaler, meta = load_isolation_forest()
    trained_features = meta.get("feature_columns", feature_columns)
    missing = [c for c in trained_features if c not in df.columns]
    if missing:
        raise ModelNotTrainedError(
            f"Isolation Forest was trained on feature(s) {missing} which are "
            f"not present in the current data."
        )
    X = transform_features(df, trained_features, scaler)
    raw = -model.score_samples(X)  # invert so higher = more anomalous
    return raw


def lstm_scores(df, feature_columns: list):
    """
    Return (window_end_row_indices, per_window_scores, per_feature_errors).

    per_window_scores: 1D array, mean squared reconstruction error per window.
    per_feature_errors: 2D array (n_windows, n_features), mean squared error
        per feature within each window — used for attribution.
    window_end_row_indices: row index in df that each window "ends" at, i.e.
        the row the score should be attributed to for display/fusion.
    """
    model, scaler, meta = load_lstm_autoencoder()
    trained_features = meta.get("feature_columns", feature_columns)
    missing = [c for c in trained_features if c not in df.columns]
    if missing:
        raise ModelNotTrainedError(
            f"LSTM Autoencoder was trained on feature(s) {missing} which are "
            f"not present in the current data."
        )
    window_size = int(meta["window_size"])
    step_size = int(meta.get("step_size", 1))

    X = transform_features(df, trained_features, scaler)
    windows = make_windows(X, window_size, step_size)
    if windows.shape[0] == 0:
        return [], np.array([]), np.empty((0, len(trained_features)))

    reconstructed = model.predict(windows, verbose=0)
    squared_error = np.square(windows - reconstructed)  # (n_windows, window_size, n_features)

    per_window_scores = squared_error.mean(axis=(1, 2))
    per_feature_errors = squared_error.mean(axis=1)  # (n_windows, n_features)

    end_indices = window_end_indices(len(df), window_size, step_size)
    return end_indices, per_window_scores, per_feature_errors
