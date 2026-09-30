"""
Score fusion: combine Isolation Forest and LSTM Autoencoder anomaly scores
into a single fused score, using min-max normalization + weighted average.

final_score = alpha * isolation_score_norm + (1 - alpha) * lstm_score_norm

Normalization is always fit on a REFERENCE set of scores (e.g. the
validation split) and then applied consistently to new scores, so that a
single new/live score can still be placed on a comparable 0-1 scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoreNormalizer:
    """Min-max normalizer fit on a reference distribution of raw scores."""
    min_val: float
    max_val: float

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        span = self.max_val - self.min_val
        if span <= 1e-12:
            # Degenerate reference distribution (all identical) — avoid
            # divide-by-zero; everything is "at the reference level".
            return np.zeros_like(raw_scores, dtype=float)
        normalized = (raw_scores - self.min_val) / span
        return np.clip(normalized, 0.0, 1.0)


def fit_normalizer(reference_raw_scores: np.ndarray) -> ScoreNormalizer:
    return ScoreNormalizer(
        min_val=float(np.min(reference_raw_scores)),
        max_val=float(np.max(reference_raw_scores)),
    )


def fuse_scores(
    isolation_scores_norm: np.ndarray,
    lstm_scores_norm: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Weighted average of two already-normalized (0-1) score arrays."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"fusion alpha must be in [0, 1], got {alpha}")
    return alpha * isolation_scores_norm + (1.0 - alpha) * lstm_scores_norm
