"""
Feature-level attribution for LSTM Autoencoder anomalies.

Uses the per-feature reconstruction error already computed in
anomaly_detection.lstm_scores() — no SHAP, no extra model calls. The
attribution for a given window is simply which features had the largest
mean squared reconstruction error within that window, expressed both as
raw error and as a share (%) of the total error across features.
"""

from __future__ import annotations

import numpy as np


def top_contributing_features(
    per_feature_errors_row: np.ndarray,
    feature_names: list,
    top_n: int = 3,
) -> list:
    """
    Given the per-feature reconstruction error for a SINGLE window/row,
    return the top_n contributing features as a list of dicts:
        {"feature": name, "error": float, "share_pct": float}
    ordered from highest to lowest contribution.
    """
    if len(per_feature_errors_row) != len(feature_names):
        raise ValueError(
            f"per_feature_errors_row length ({len(per_feature_errors_row)}) does not "
            f"match feature_names length ({len(feature_names)})"
        )

    total = float(np.sum(per_feature_errors_row))
    order = np.argsort(per_feature_errors_row)[::-1]

    results = []
    for idx in order[:top_n]:
        error = float(per_feature_errors_row[idx])
        share = (error / total * 100.0) if total > 1e-12 else 0.0
        results.append(
            {
                "feature": feature_names[idx],
                "error": error,
                "share_pct": share,
            }
        )
    return results
