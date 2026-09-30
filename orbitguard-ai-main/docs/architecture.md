# OrbitGuard AI — Architecture

## Pipeline overview

```
Real telemetry CSV (data/telemetry.csv)
        │
        ▼
data_loader.py       — load, validate, sort, dedupe, fill missing values
        │
        ▼
chronological split  — train / validation / test (no shuffling)
        │
        ▼
preprocessing.py     — StandardScaler (fit on train only) + sliding windows
        │
   ┌────┴─────┐
   ▼          ▼
Isolation    LSTM
Forest       Autoencoder
(point       (temporal
anomalies)   anomalies)
   │          │
   └────┬─────┘
        ▼
fusion.py             — min-max normalize each model's scores against a
                         reference (validation) distribution, then weighted
                         average: alpha * IF + (1-alpha) * LSTM
        ▼
severity.py            — percentile thresholds computed from the reference
                          distribution classify the fused score into
                          Low / Medium / High / Critical
        ▼
attribution.py         — per-feature reconstruction error (LSTM) ranks
                          which telemetry parameters contributed most
        ▼
subsystem_mapping.py   — maps top contributing features to subsystems
                          (Power, Thermal, Attitude, Communication, ...)
        ▼
trend.py                — linear-slope trend classification per feature
        ▼
app.py (Streamlit)     — dashboard: overview, live replay, evaluation
```

## Why two models?

- **Isolation Forest** scores a single telemetry snapshot in isolation. It
  is fast, needs no sequence context, and is good at catching a reading
  that is simply out of normal range.
- **LSTM Autoencoder** scores a *sequence* of readings. It is needed to
  catch anomalies where every individual value looks plausible but the
  *pattern* over time is not — e.g. a slow monotonic drift in temperature
  that never leaves the "normal" range for any single reading.

Score fusion combines both signals into one operator-facing score rather
than forcing the operator to reconcile two separate outputs.

## Why percentile-based severity instead of fixed thresholds?

A fixed cutoff (e.g. "score > 0.8 = critical") is only meaningful relative
to the scale of the specific scoring function and dataset it was tuned on.
Because Isolation Forest and LSTM reconstruction error have very different
native scales, and because every satellite's normal operating envelope is
different, OrbitGuard instead computes thresholds as percentiles of the
score distribution on a reference set of (assumed-mostly-normal)
validation data. This makes severity self-calibrating to whatever dataset
is supplied, and the percentiles themselves remain fully configurable in
`config/config.yaml`.

## Why feature-level attribution uses reconstruction error, not SHAP

The LSTM Autoencoder already computes, as a side effect of scoring, the
squared reconstruction error for every individual feature at every
timestep. Averaging that error per feature across a window directly
answers "which parameters were hardest for the model to reconstruct" —
which is exactly the attribution question — without the added complexity,
runtime cost, or approximation error of a separate SHAP explainer.

## Data leakage avoidance

- Scalers are fit ONLY on the training split's normal rows, then reused
  (never refit) for validation, test, and live-replay data.
- Splits are chronological, not shuffled, since telemetry is a time
  series and shuffling would let the model "see the future" during
  training.
- If a label column is used to filter "normal" rows for training, those
  labels are never used as a model input feature.

## Known limitations

- Severity thresholds assume the validation split is mostly free of
  anomalies. If your validation data contains a large proportion of
  anomalies, thresholds will be miscalibrated — consider labeling data
  or choosing a cleaner validation window.
- The LSTM Autoencoder requires enough consecutive normal rows to build
  multiple windows (`window_size` rows minimum per window). Very small
  datasets may not support meaningful LSTM training.
- Live replay is a replay of historical data for demonstration purposes,
  not a live telemetry feed from an actual satellite.
