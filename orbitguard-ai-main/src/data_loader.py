"""
Dataset adapter for OrbitGuard AI.

Loads a REAL telemetry CSV supplied by the user (never generates or assumes
synthetic data). Column names are driven entirely by config/config.yaml so
this module makes no hard-coded assumptions about the dataset's schema.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config_utils import PROJECT_ROOT, resolve_path


class DatasetError(Exception):
    """Raised when the dataset is missing, malformed, or misconfigured."""


@dataclass
class LoadedDataset:
    df: pd.DataFrame
    timestamp_column: str
    feature_columns: list
    label_column: str | None
    normal_label_value: object = 0


def _dataset_missing_message(path: str) -> str:
    return (
        f"Telemetry dataset not found at: {path}\n"
        f"Please place your real telemetry CSV in the 'data/' folder "
        f"(see data/README.md) and make sure 'dataset.path' in "
        f"config/config.yaml points to it."
    )


def load_dataset(cfg: dict) -> LoadedDataset:
    """
    Load and lightly validate the real telemetry CSV described by cfg.
    Raises DatasetError with a clear, specific message on any problem.
    Does NOT scale, window, or otherwise transform the data.
    """
    dataset_cfg = cfg.get("dataset", {})
    raw_path = dataset_cfg.get("path")
    if not raw_path:
        raise DatasetError("config.yaml: 'dataset.path' is not set.")

    path = resolve_path(raw_path)
    if not os.path.exists(path):
        raise DatasetError(_dataset_missing_message(path))

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - surface any parse error clearly
        raise DatasetError(f"Failed to read CSV at {path}: {exc}") from exc

    if df.empty:
        raise DatasetError(f"Dataset at {path} is empty (0 rows).")

    timestamp_column = dataset_cfg.get("timestamp_column")
    if not timestamp_column or timestamp_column not in df.columns:
        raise DatasetError(
            f"Timestamp column '{timestamp_column}' not found in dataset columns: "
            f"{list(df.columns)}. Check 'dataset.timestamp_column' in config.yaml."
        )

    try:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    except Exception as exc:  # noqa: BLE001
        raise DatasetError(
            f"Could not parse column '{timestamp_column}' as timestamps: {exc}"
        ) from exc

    # Sort chronologically and drop duplicate timestamps (keep first occurrence).
    df = df.sort_values(timestamp_column).reset_index(drop=True)
    before = len(df)
    df = df.drop_duplicates(subset=[timestamp_column], keep="first").reset_index(drop=True)
    dropped = before - len(df)

    label_column = dataset_cfg.get("label_column")
    if label_column is not None and label_column not in df.columns:
        raise DatasetError(
            f"Configured label_column '{label_column}' not found in dataset columns: "
            f"{list(df.columns)}."
        )

    feature_columns = dataset_cfg.get("feature_columns") or []
    if feature_columns:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise DatasetError(
                f"Configured feature column(s) not found in dataset: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
    else:
        # Auto-select: all numeric columns except timestamp / label.
        exclude = {timestamp_column}
        if label_column:
            exclude.add(label_column)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [c for c in numeric_cols if c not in exclude]
        if not feature_columns:
            raise DatasetError(
                "No numeric feature columns could be auto-detected. Set "
                "'dataset.feature_columns' explicitly in config.yaml."
            )

    # Handle missing values in feature columns: forward-fill then back-fill,
    # which is standard practice for telemetry time series (avoids fabricating
    # values via interpolation across large gaps).
    missing_before = int(df[feature_columns].isna().sum().sum())
    if missing_before > 0:
        df[feature_columns] = df[feature_columns].ffill().bfill()
    missing_after = int(df[feature_columns].isna().sum().sum())
    if missing_after > 0:
        raise DatasetError(
            f"{missing_after} missing values remain in feature columns after "
            f"fill — some feature column(s) may be entirely empty. Check your CSV."
        )

    # Coerce features to numeric, catching non-numeric junk explicitly.
    for col in feature_columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad = int(coerced.isna().sum()) - int(df[col].isna().sum())
        if bad > 0:
            raise DatasetError(
                f"Column '{col}' contains {bad} non-numeric value(s) that "
                f"cannot be used as a telemetry feature."
            )
        df[col] = coerced

    normal_label_value = dataset_cfg.get("normal_label_value", 0)

    return LoadedDataset(
        df=df,
        timestamp_column=timestamp_column,
        feature_columns=feature_columns,
        label_column=label_column,
        normal_label_value=normal_label_value,
    )


def chronological_split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a dataframe chronologically (no shuffling) into train/val/test."""
    split_cfg = cfg.get("split", {})
    train_frac = float(split_cfg.get("train_split", 0.7))
    val_frac = float(split_cfg.get("validation_split", 0.15))
    test_frac = float(split_cfg.get("test_split", 0.15))

    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0, atol=1e-6):
        raise DatasetError(
            f"split.train_split + validation_split + test_split must sum to 1.0 "
            f"(got {total})."
        )

    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    if len(train_df) == 0:
        raise DatasetError(
            "Training split has 0 rows — dataset too small or train_split too low."
        )

    return train_df, val_df, test_df


def filter_normal_rows(df: pd.DataFrame, loaded: LoadedDataset) -> pd.DataFrame:
    """
    Return only rows considered 'normal', for training.
    If no label column is configured, ALL rows are treated as normal
    (standard unsupervised assumption — document this clearly to the user).
    """
    if loaded.label_column is None:
        return df
    return df[df[loaded.label_column] == loaded.normal_label_value].reset_index(drop=True)
