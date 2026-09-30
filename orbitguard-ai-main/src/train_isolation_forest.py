"""
Train the Isolation Forest point-anomaly model on REAL normal telemetry data.

Usage:
    python src/train_isolation_forest.py [--config config/config.yaml]

Trains ONLY on data/telemetry.csv (or whatever dataset.path points to).
No synthetic data is generated at any point in this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
from sklearn.ensemble import IsolationForest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_utils import load_config, resolve_path, ConfigError  # noqa: E402
from src.data_loader import load_dataset, chronological_split, filter_normal_rows, DatasetError  # noqa: E402
from src.preprocessing import fit_scaler, transform_features  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Isolation Forest on real telemetry data.")
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

    train_df, _, _ = chronological_split(loaded.df, cfg)
    normal_train_df = filter_normal_rows(train_df, loaded)

    if len(normal_train_df) < 10:
        print(
            f"[DATA ERROR] Only {len(normal_train_df)} normal training rows available "
            f"after filtering — need at least 10 to train a meaningful model."
        )
        return 1

    print(f"Training Isolation Forest on {len(normal_train_df)} normal rows, "
          f"{len(loaded.feature_columns)} features: {loaded.feature_columns}")

    scaler = fit_scaler(normal_train_df, loaded.feature_columns)
    X_train = transform_features(normal_train_df, loaded.feature_columns, scaler)

    if_cfg = cfg.get("isolation_forest", {})
    model = IsolationForest(
        n_estimators=int(if_cfg.get("n_estimators", 200)),
        contamination=if_cfg.get("contamination", "auto"),
        random_state=int(if_cfg.get("random_state", 42)),
    )
    model.fit(X_train)

    models_dir = resolve_path("models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "isolation_forest.joblib")
    scaler_path = os.path.join(models_dir, "isolation_forest_scaler.joblib")
    meta_path = os.path.join(models_dir, "isolation_forest_meta.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": loaded.feature_columns,
                "n_training_rows": int(len(normal_train_df)),
                "n_estimators": int(if_cfg.get("n_estimators", 200)),
                "contamination": if_cfg.get("contamination", "auto"),
            },
            f,
            indent=2,
        )

    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved metadata to {meta_path}")
    print("Isolation Forest training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
