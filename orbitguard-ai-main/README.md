# 🛰️ OrbitGuard AI

**AI-Based CubeSat Telemetry Anomaly Detection and Early-Warning System**

OrbitGuard AI learns normal CubeSat behavior from real telemetry and flags
both point anomalies (a single reading out of range) and temporal
anomalies (a sequence that drifts or oscillates abnormally), with
feature-level attribution, subsystem mapping, data-driven severity, and a
live-replay dashboard.

> **This repository contains the complete pipeline but does not contain
> models trained on simulated data.** Nothing in this project generates
> or trains on synthetic telemetry. You must supply your own real
> CubeSat telemetry dataset and train the models yourself using the
> scripts below.

---

## Problem statement

CubeSats generate continuous telemetry (temperature, power, attitude,
communication signal, etc.). Manually watching dashboards for subtle,
early warning signs of subsystem failure does not scale. OrbitGuard AI
automates this by learning what "normal" looks like for a given
satellite and surfacing early-warning alerts — including slow drifts
that no single-value threshold would catch — before they become
critical failures.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full pipeline
diagram and design rationale. In short:

- **Isolation Forest** — point-anomaly detection on individual telemetry
  snapshots.
- **LSTM Autoencoder** — temporal-anomaly detection on sliding windows of
  consecutive readings, via reconstruction error.
- **Fusion** — a configurable weighted combination of both models' scores.
- **Attribution** — per-feature reconstruction error identifies which
  telemetry parameters are driving a detected anomaly.
- **Subsystem mapping** — a configurable feature → subsystem lookup.
- **Severity** — percentile-based thresholds derived from the model's own
  score distribution on validation data (never hard-coded cutoffs).
- **Trend detection** — simple linear-slope classification
  (Increasing / Decreasing / Stable / Fluctuating).

## ML approach

Both models are trained **unsupervised**, using only data assumed (or
labeled) to be "normal" behavior. If your dataset includes ground-truth
anomaly labels, they are used *only* for evaluation (precision, recall,
F1, false-positive rate) — never as a training signal.

---

## Project structure

```
orbitguard-ai/
├── app.py                        # Streamlit dashboard
├── requirements.txt
├── README.md
├── .gitignore
├── config/
│   ├── config.yaml                # dataset schema, model & fusion settings
│   └── subsystems.yaml            # feature -> subsystem mapping rules
├── data/
│   └── README.md                  # where to place your real dataset
├── models/                        # trained model files land here (gitignored)
├── outputs/                       # evaluation results land here (gitignored)
├── src/
│   ├── config_utils.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_isolation_forest.py
│   ├── train_lstm_autoencoder.py
│   ├── anomaly_detection.py
│   ├── fusion.py
│   ├── attribution.py
│   ├── severity.py
│   ├── trend.py
│   ├── subsystem_mapping.py
│   └── evaluate.py
└── docs/
    └── architecture.md
```

---

## 1. Install dependencies

Requires Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## 2. Add your real dataset

Place your real CubeSat telemetry CSV at:

```
data/telemetry.csv
```

(or any path — just update `dataset.path` in `config/config.yaml`).
See [`data/README.md`](data/README.md) for the expected format.

## 3. Configure the dataset schema

Edit `config/config.yaml`:

- `dataset.timestamp_column` — your timestamp column's exact name.
- `dataset.feature_columns` — list your numeric telemetry columns, or
  leave `[]` to auto-select all numeric columns.
- `dataset.label_column` — set this if you have ground-truth anomaly
  labels (0 = normal, 1 = anomaly); otherwise leave `null`.
- `lstm.window_size` — how many consecutive readings form one sequence
  for the LSTM Autoencoder. Must be small enough that your training
  split has at least a few times this many rows.

Also edit `config/subsystems.yaml` so its keyword rules match your
dataset's actual column names (e.g. if your battery voltage column is
called `batt_v_mv`, the existing `voltage` keyword under `Power` will
already match it — but check and extend as needed).

## 4. Train the models

Each step is explicit — nothing trains automatically.

```bash
python src/train_isolation_forest.py
python src/train_lstm_autoencoder.py
```

This produces, in `models/`:

```
isolation_forest.joblib
isolation_forest_scaler.joblib
isolation_forest_meta.json
lstm_autoencoder.keras
lstm_scaler.joblib
lstm_meta.json
```

## 5. Evaluate

```bash
python src/evaluate.py
```

Writes `outputs/metrics.csv`. If `dataset.label_column` is set, this
includes real precision/recall/F1/false-positive-rate per model,
computed from actual predictions — never fabricated. If no labels are
configured, only unsupervised score statistics are reported, and the
metrics file states clearly that supervised metrics are unavailable.

## 6. Launch the dashboard

```bash
python -m streamlit run app.py
```

The dashboard has three tabs:

- **Overview** — telemetry and anomaly-score timelines over the test split.
- **Live Replay** — replays real rows from your dataset in sequence (at a
  configurable interval) to simulate a live telemetry feed, scoring each
  new row/window as it "arrives" and showing alerts, severity, top
  contributing parameters, affected subsystem, and trends.
- **Evaluation** — displays `outputs/metrics.csv` if it exists.

If a model hasn't been trained yet, the dashboard will say so explicitly
(**"Model not trained yet"**) rather than showing placeholder numbers.

---

## How live replay works

Live replay does **not** generate synthetic telemetry. It steps through
real rows from your configured split (`train` / `validation` / `test` /
`full`) at a configurable interval, feeding each row (and the growing
history of rows) through the trained models exactly as if it had just
arrived from the satellite.

## How attribution works

For any window scored by the LSTM Autoencoder, OrbitGuard already
computes the mean squared reconstruction error separately per feature.
Attribution simply ranks features by that per-feature error and reports
the top contributors and their share of total reconstruction error — see
[`docs/architecture.md`](docs/architecture.md) for why this is preferred
over a SHAP-based approach here.

## How subsystem mapping works

`config/subsystems.yaml` defines keyword rules per subsystem (e.g. any
feature name containing `"voltage"`, `"current"`, or `"battery"` maps to
`Power`). A feature that matches no rule is reported as
`"Unknown / Unmapped"` — the system never guesses a subsystem.

## How severity is calculated

Severity thresholds (Medium / High / Critical) are percentiles of the
fused anomaly score distribution computed on validation data, configured
in `config/config.yaml` under `severity.*_percentile`. This makes
severity self-calibrating to your dataset instead of relying on an
arbitrary fixed cutoff.

## Adding a new telemetry parameter

1. Add the column to your CSV.
2. Add it to `dataset.feature_columns` in `config/config.yaml` (or leave
   auto-detection on).
3. Add a subsystem rule for it in `config/subsystems.yaml` if it should
   map to something other than `Unknown / Unmapped`.
4. Re-run both training scripts and `evaluate.py` — scalers and models
   must be retrained whenever the feature set changes.

## Limitations

See [`docs/architecture.md`](docs/architecture.md#known-limitations).

---

## No fake results, ever

- The dashboard will say **"Model not trained yet"** if `models/` is
  empty — it will not fabricate scores.
- `evaluate.py` will say **"Evaluation results unavailable — train and
  evaluate the model first"** rather than showing invented metrics.
- No accuracy, precision, recall, F1, anomaly percentage, or latency
  number anywhere in this codebase is hard-coded — every displayed
  metric comes from actually running the models on your data.
