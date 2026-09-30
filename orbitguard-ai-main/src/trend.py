"""
Simple, robust trend detection for a recent window of telemetry values.

Uses a linear least-squares fit (slope) over the most recent N points,
normalized by the series' own standard deviation, rather than trying to
"predict the future". Classifies into Increasing / Decreasing / Stable /
Fluctuating.
"""

from __future__ import annotations

import numpy as np


def detect_trend(values: np.ndarray, window: int = 10, flat_threshold: float = 0.01) -> str:
    """
    values: 1D array of recent readings for a single telemetry feature,
        in chronological order (oldest first).
    Returns one of: "Increasing", "Decreasing", "Stable", "Fluctuating",
        or "Insufficient data" if fewer than 3 points are available.
    """
    values = np.asarray(values, dtype=float)
    recent = values[-window:] if len(values) > window else values

    if len(recent) < 3:
        return "Insufficient data"

    x = np.arange(len(recent))
    # Linear fit: recent = slope * x + intercept
    slope, _intercept = np.polyfit(x, recent, 1)

    std = float(np.std(recent))
    if std < 1e-9:
        return "Stable"

    normalized_slope = slope / std

    # Fluctuation check: how much the series reverses direction vs. moves
    # consistently (residual variance around the linear fit vs. total variance).
    fitted = slope * x + _intercept
    residual_std = float(np.std(recent - fitted))
    if residual_std > std * 0.75 and abs(normalized_slope) < 0.5:
        return "Fluctuating"

    if abs(normalized_slope) < flat_threshold:
        return "Stable"
    return "Increasing" if normalized_slope > 0 else "Decreasing"


def detect_trends_for_features(recent_df, feature_columns: list, window: int, flat_threshold: float) -> dict:
    """Return {feature_name: trend_label} for each feature over the recent window."""
    trends = {}
    for col in feature_columns:
        trends[col] = detect_trend(recent_df[col].values, window=window, flat_threshold=flat_threshold)
    return trends
