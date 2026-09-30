# Data Folder

Place your **real CubeSat telemetry CSV** here.

By default, the pipeline looks for:

```
data/telemetry.csv
```

You can change this path in `config/config.yaml` under `dataset.path`.

## Expected format

A CSV file with:

- One row per telemetry reading (chronologically ordered, or with a
  timestamp column so it can be sorted).
- A timestamp column (name configurable via `dataset.timestamp_column`).
- One or more numeric telemetry columns (temperature, voltage, current,
  gyro readings, signal strength, etc.) — configurable via
  `dataset.feature_columns`.
- Optionally, a binary anomaly label column (0 = normal, 1 = anomaly) —
  configurable via `dataset.label_column`. This is used **only** for
  evaluation, never for training.

## Important

**No synthetic or example dataset is included in this repository.**
This folder intentionally contains only this README. Training scripts
will refuse to run and will print a clear error message until a real
dataset is placed here and `config/config.yaml` is updated to match its
column names.
