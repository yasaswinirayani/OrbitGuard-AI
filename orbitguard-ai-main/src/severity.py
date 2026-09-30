"""
Data-driven severity bucketing.

Thresholds are computed as percentiles of the fused anomaly score
distribution on a REFERENCE set (normally the validation split, which is
assumed to be mostly-normal behavior). This avoids hard-coding arbitrary
score cutoffs like "score > 0.8 = critical".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SeverityThresholds:
    medium: float
    high: float
    critical: float

    def classify(self, score: float) -> str:
        if score >= self.critical:
            return "Critical"
        if score >= self.high:
            return "High"
        if score >= self.medium:
            return "Medium"
        return "Low"


def compute_severity_thresholds(reference_scores: np.ndarray, severity_cfg: dict) -> SeverityThresholds:
    if len(reference_scores) == 0:
        raise ValueError("Cannot compute severity thresholds from an empty reference score array.")

    medium_p = float(severity_cfg.get("medium_percentile", 90))
    high_p = float(severity_cfg.get("high_percentile", 97))
    critical_p = float(severity_cfg.get("critical_percentile", 99.5))

    return SeverityThresholds(
        medium=float(np.percentile(reference_scores, medium_p)),
        high=float(np.percentile(reference_scores, high_p)),
        critical=float(np.percentile(reference_scores, critical_p)),
    )
