"""
Evaluate Isolation Forest, LSTM Autoencoder, and their fusion on the TEST
split of the real dataset.

If ground-truth labels are configured, computes Precision / Recall / F1 /
False Positive Rate for each approach using severity-derived thresholds.
If no labels are available, prints/saves unsupervised statistics only
(score distributions) and clearly states that supervised metrics are
unavailable. No metric is ever fabricated or hard-coded.

Usage:
    python src/evaluate.py [--config config/config.yaml]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_utils import load_config, resolve_path, ConfigError  # noqa: E402
from src.data_loader import load_dataset, chronological_split, filter_normal_rows, DatasetError  # noqa: E402
from src.anomaly_detection import (  # noqa: E402
    isolation_forest_available,
    lstm_available,
    isolation_forest_scores,
    lstm_scores,
    ModelNotTrainedError,
)
from src.fusion import fit_normalizer, fuse_scores  # noqa: E402
from src.severity import compute_severity_thresholds  # noqa: E402


def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute precision, recall, F1, and false positive rate from scratch."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate OrbitGuard AI models on real data.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    try:
        cfg = load_config(resolve_path(args.config))
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}")
        return 1

    try:
        loaded = load_dataset(cfg)
    except DatasetError as exc:
        print(f"[DATASET ERROR] {exc}")
        return 1

    if not isolation_forest_available() and not lstm_available():
        print("[EVALUATION UNAVAILABLE] Neither model has been trained yet. "
              "Run src/train_isolation_forest.py and src/train_lstm_autoencoder.py first.")
        return 1

    train_df, val_df, test_df = chronological_split(loaded.df, cfg)
    normal_val_df = filter_normal_rows(val_df, loaded)

    results_rows = []
    fused_test_scores = None
    fused_test_row_indices = None

    # ---- Isolation Forest ----
    if_test_scores_norm = None
    if isolation_forest_available():
        try:
            if_val_raw = isolation_forest_scores(normal_val_df, loaded.feature_columns)
            if_test_raw = isolation_forest_scores(test_df, loaded.feature_columns)
            normalizer = fit_normalizer(if_val_raw)
            if_test_scores_norm = normalizer.transform(if_test_raw)
            print(f"Isolation Forest: scored {len(test_df)} test rows. "
                  f"Mean normalized score: {float(np.mean(if_test_scores_norm)):.4f}")
        except ModelNotTrainedError as exc:
            print(f"[ISOLATION FOREST] {exc}")
    else:
        print("[ISOLATION FOREST] Model not trained yet — skipping.")

    # ---- LSTM Autoencoder ----
    lstm_test_scores_norm = None
    lstm_end_indices = None
    per_feature_errors_test = None
    if lstm_available():
        try:
            val_end_idx, lstm_val_raw, _ = lstm_scores(normal_val_df, loaded.feature_columns)
            lstm_end_indices, lstm_test_raw, per_feature_errors_test = lstm_scores(test_df, loaded.feature_columns)
            if len(lstm_val_raw) == 0 or len(lstm_test_raw) == 0:
                print("[LSTM AUTOENCODER] Not enough rows to build even one window in "
                      "validation or test split — skipping.")
            else:
                normalizer = fit_normalizer(lstm_val_raw)
                lstm_test_scores_norm = normalizer.transform(lstm_test_raw)
                print(f"LSTM Autoencoder: scored {len(lstm_test_raw)} test windows. "
                      f"Mean normalized score: {float(np.mean(lstm_test_scores_norm)):.4f}")
        except ModelNotTrainedError as exc:
            print(f"[LSTM AUTOENCODER] {exc}")
    else:
        print("[LSTM AUTOENCODER] Model not trained yet — skipping.")

    # ---- Fusion (only if both available and aligned) ----
    fusion_cfg = cfg.get("fusion", {})
    alpha = float(fusion_cfg.get("alpha", 0.5))
    if if_test_scores_norm is not None and lstm_test_scores_norm is not None:
        # Align on the rows where an LSTM window ends (IF has a score per row).
        aligned_if = if_test_scores_norm[lstm_end_indices]
        fused_test_scores = fuse_scores(aligned_if, lstm_test_scores_norm, alpha)
        fused_test_row_indices = lstm_end_indices
        print(f"Fusion (alpha={alpha}): mean fused score on test set: "
              f"{float(np.mean(fused_test_scores)):.4f}")
    else:
        print("[FUSION] Requires both models trained and scorable — skipping.")

    # ---- Metrics (only if labels available) ----
    metrics_available = loaded.label_column is not None
    metric_rows = []

    if metrics_available:
        y_test_full = test_df[loaded.label_column].values
        # Severity thresholds from validation (assumed mostly-normal) fused scores,
        # used here as the anomaly/not-anomaly decision boundary ("Medium+").
        severity_cfg = cfg.get("severity", {})

        def add_metrics(name: str, scores_norm, row_indices, reference_scores):
            if scores_norm is None or len(scores_norm) == 0:
                return
            thresholds = compute_severity_thresholds(reference_scores, severity_cfg)
            y_true = y_test_full[row_indices] if row_indices is not None else y_test_full
            y_pred = (np.asarray(scores_norm) >= thresholds.medium).astype(int)
            m = binary_classification_metrics(y_true, y_pred)
            m["model"] = name
            metric_rows.append(m)
            print(f"[{name}] precision={m['precision']:.3f} recall={m['recall']:.3f} "
                  f"f1={m['f1_score']:.3f} fpr={m['false_positive_rate']:.3f}")

        if if_test_scores_norm is not None:
            if_val_raw_ref = isolation_forest_scores(normal_val_df, loaded.feature_columns)
            ref_norm = fit_normalizer(if_val_raw_ref).transform(if_val_raw_ref)
            add_metrics("isolation_forest", if_test_scores_norm, None, ref_norm)

        if lstm_test_scores_norm is not None:
            _, lstm_val_raw_ref, _ = lstm_scores(normal_val_df, loaded.feature_columns)
            ref_norm = fit_normalizer(lstm_val_raw_ref).transform(lstm_val_raw_ref)
            add_metrics("lstm_autoencoder", lstm_test_scores_norm, lstm_end_indices, ref_norm)

        if fused_test_scores is not None:
            if_val_raw_ref = isolation_forest_scores(normal_val_df, loaded.feature_columns)
            _, lstm_val_raw_ref, _ = lstm_scores(normal_val_df, loaded.feature_columns)
            val_end_idx_ref, _, _ = lstm_scores(normal_val_df, loaded.feature_columns)
            if_val_norm_ref = fit_normalizer(if_val_raw_ref).transform(if_val_raw_ref)
            lstm_val_norm_ref = fit_normalizer(lstm_val_raw_ref).transform(lstm_val_raw_ref)
            aligned_if_val_ref = if_val_norm_ref[val_end_idx_ref] if len(val_end_idx_ref) else np.array([])
            if len(aligned_if_val_ref) == len(lstm_val_norm_ref) and len(aligned_if_val_ref) > 0:
                fused_ref = fuse_scores(aligned_if_val_ref, lstm_val_norm_ref, alpha)
                add_metrics("fusion", fused_test_scores, fused_test_row_indices, fused_ref)
    else:
        print("[METRICS] No label_column configured in config.yaml — "
              "supervised metrics (precision/recall/F1) cannot be calculated. "
              "Only unsupervised score statistics are reported above.")

    # ---- Save results ----
    outputs_dir = resolve_path("outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    metrics_path = os.path.join(outputs_dir, "metrics.csv")

    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
        print(f"Saved evaluation metrics to {metrics_path}")
    else:
        # Still write a file so the dashboard has something deterministic to check,
        # but make it explicit that no supervised metrics were computed.
        pd.DataFrame([{"note": "No labeled evaluation metrics available. "
                                "Configure dataset.label_column to enable this."}]).to_csv(
            metrics_path, index=False
        )
        print(f"No supervised metrics computed. Wrote explanatory note to {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
