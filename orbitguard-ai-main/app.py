"""
OrbitGuard AI — CubeSat Early-Warning System
Streamlit + Plotly dashboard.

This dashboard NEVER trains models on startup. It reads whatever trained
model files already exist in models/, and clearly states "Model not
trained yet" for anything missing. Live replay streams rows from the
user's real CSV — it never generates synthetic values.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config_utils import load_config, load_subsystems, resolve_path, ConfigError
from src.data_loader import load_dataset, chronological_split, filter_normal_rows, DatasetError
from src.anomaly_detection import (
    isolation_forest_available,
    lstm_available,
    isolation_forest_scores,
    lstm_scores,
    ModelNotTrainedError,
)
from src.fusion import fit_normalizer, fuse_scores
from src.severity import compute_severity_thresholds
from src.attribution import top_contributing_features
from src.subsystem_mapping import affected_subsystems
from src.trend import detect_trends_for_features

st.set_page_config(page_title="OrbitGuard AI", layout="wide", page_icon="🛰️")


# =============================================================================
# Setup / caching
# =============================================================================

@st.cache_data(show_spinner=False)
def _cached_config():
    cfg = load_config()
    subsystems_cfg = load_subsystems()
    return cfg, subsystems_cfg


@st.cache_data(show_spinner=False)
def _cached_dataset(_cfg_marker: str):
    """_cfg_marker forces cache invalidation if the config changes between runs."""
    cfg, _ = _cached_config()
    loaded = load_dataset(cfg)
    return loaded


def _config_marker(cfg: dict) -> str:
    return str(sorted(cfg.get("dataset", {}).items()))


# =============================================================================
# Reference (validation) scoring — used for normalization + severity thresholds
# =============================================================================

@st.cache_data(show_spinner=False)
def _reference_scores(_marker: str):
    """
    Compute normalized reference scores (Isolation Forest, LSTM, fused) on the
    normal validation split, plus fitted normalizers and severity thresholds.
    Returns a dict of everything downstream code needs, with None for any
    model that isn't trained yet.
    """
    cfg, _ = _cached_config()
    loaded = _cached_dataset(_marker)
    _, val_df, _ = chronological_split(loaded.df, cfg)
    normal_val_df = filter_normal_rows(val_df, loaded)

    result = {
        "if_normalizer": None,
        "lstm_normalizer": None,
        "if_severity": None,
        "lstm_severity": None,
        "fusion_severity": None,
    }

    severity_cfg = cfg.get("severity", {})

    if_val_norm = None
    if isolation_forest_available():
        try:
            if_val_raw = isolation_forest_scores(normal_val_df, loaded.feature_columns)
            normalizer = fit_normalizer(if_val_raw)
            if_val_norm = normalizer.transform(if_val_raw)
            result["if_normalizer"] = normalizer
            result["if_severity"] = compute_severity_thresholds(if_val_norm, severity_cfg)
        except ModelNotTrainedError:
            pass

    lstm_val_norm = None
    val_end_idx = None
    if lstm_available():
        try:
            val_end_idx, lstm_val_raw, _ = lstm_scores(normal_val_df, loaded.feature_columns)
            if len(lstm_val_raw) > 0:
                normalizer = fit_normalizer(lstm_val_raw)
                lstm_val_norm = normalizer.transform(lstm_val_raw)
                result["lstm_normalizer"] = normalizer
                result["lstm_severity"] = compute_severity_thresholds(lstm_val_norm, severity_cfg)
        except ModelNotTrainedError:
            pass

    if if_val_norm is not None and lstm_val_norm is not None and val_end_idx:
        alpha = float(cfg.get("fusion", {}).get("alpha", 0.5))
        aligned_if = if_val_norm[val_end_idx]
        if len(aligned_if) == len(lstm_val_norm):
            fused_val = fuse_scores(aligned_if, lstm_val_norm, alpha)
            result["fusion_severity"] = compute_severity_thresholds(fused_val, severity_cfg)

    return result


# =============================================================================
# UI helper components
# =============================================================================

def render_header():
    st.title("🛰️ OrbitGuard AI — CubeSat Early-Warning System")
    st.caption("AI-based telemetry anomaly detection and early-warning system for CubeSats")


def render_status_badges(if_ready: bool, lstm_ready: bool):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Isolation Forest", "Trained ✅" if if_ready else "Not trained yet ⚠️")
    with col2:
        st.metric("LSTM Autoencoder", "Trained ✅" if lstm_ready else "Not trained yet ⚠️")
    with col3:
        fusion_ready = if_ready and lstm_ready
        st.metric("Fusion", "Available ✅" if fusion_ready else "Unavailable ⚠️")


def score_to_color(severity: str) -> str:
    return {
        "Low": "#2ecc71",
        "Medium": "#f1c40f",
        "High": "#e67e22",
        "Critical": "#e74c3c",
    }.get(severity, "#95a5a6")


def render_alert(severity: str, subsystems: list, top_features: list, trends: dict, fused_score):
    if severity in ("High", "Critical"):
        box = st.error
    elif severity == "Medium":
        box = st.warning
    else:
        box = st.success

    lines = []
    if severity in ("Low",):
        lines.append("✅ **NORMAL** — no significant anomaly detected.")
    else:
        lines.append("🚨 **ANOMALY DETECTED**")
        lines.append(f"**Severity:** {severity}")
        if subsystems:
            lines.append(f"**Affected subsystem(s):** {', '.join(subsystems)}")
        if top_features:
            feat_str = ", ".join(f["feature"] for f in top_features)
            lines.append(f"**Main contributing parameters:** {feat_str}")
        trend_lines = [f"{name}: {trend}" for name, trend in trends.items() if trend not in ("Stable", "Insufficient data")]
        if trend_lines:
            lines.append(f"**Notable trends:** {'; '.join(trend_lines)}")
        if fused_score is not None:
            lines.append(f"**Anomaly score:** {fused_score:.3f}")

    box("\n\n".join(lines))


# =============================================================================
# Main app
# =============================================================================

def main():
    render_header()

    try:
        cfg, subsystems_cfg = _cached_config()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    marker = _config_marker(cfg)

    try:
        loaded = _cached_dataset(marker)
    except DatasetError as exc:
        st.warning(str(exc))
        st.info("Once your dataset is in place, reload this page to continue.")
        st.stop()

    if_ready = isolation_forest_available()
    lstm_ready = lstm_available()
    render_status_badges(if_ready, lstm_ready)

    if not if_ready and not lstm_ready:
        st.info(
            "**Model not trained yet.** Train the models from the command line, then reload:\n\n"
            "```\npython src/train_isolation_forest.py\npython src/train_lstm_autoencoder.py\npython src/evaluate.py\n```"
        )

    st.divider()

    train_df, val_df, test_df = chronological_split(loaded.df, cfg)

    tab_overview, tab_live, tab_eval = st.tabs(["📊 Overview", "▶️ Live Replay", "📈 Evaluation"])

    # -------------------------------------------------------------------
    # OVERVIEW TAB — static view over the test split
    # -------------------------------------------------------------------
    with tab_overview:
        render_overview(cfg, subsystems_cfg, loaded, test_df, marker, if_ready, lstm_ready)

    # -------------------------------------------------------------------
    # LIVE REPLAY TAB
    # -------------------------------------------------------------------
    with tab_live:
        render_live_replay(cfg, subsystems_cfg, loaded, train_df, val_df, test_df, marker, if_ready, lstm_ready)

    # -------------------------------------------------------------------
    # EVALUATION TAB
    # -------------------------------------------------------------------
    with tab_eval:
        render_evaluation()


def render_overview(cfg, subsystems_cfg, loaded, test_df, marker, if_ready, lstm_ready):
    st.subheader("Telemetry Overview (test split)")

    feature = st.selectbox("Telemetry parameter", loaded.feature_columns, key="overview_feature")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_df[loaded.timestamp_column], y=test_df[feature],
        mode="lines", name=feature,
    ))
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10),
                       title=f"{feature} over time")
    st.plotly_chart(fig, use_container_width=True)

    if not if_ready and not lstm_ready:
        return

    st.subheader("Anomaly Score Timeline")
    ref = _reference_scores(marker)
    alpha = float(cfg.get("fusion", {}).get("alpha", 0.5))

    if_scores_norm = None
    if if_ready and ref["if_normalizer"] is not None:
        raw = isolation_forest_scores(test_df, loaded.feature_columns)
        if_scores_norm = ref["if_normalizer"].transform(raw)

    lstm_scores_norm = None
    lstm_end_idx = None
    per_feature_errors = None
    if lstm_ready and ref["lstm_normalizer"] is not None:
        lstm_end_idx, raw, per_feature_errors = lstm_scores(test_df, loaded.feature_columns)
        if len(raw) > 0:
            lstm_scores_norm = ref["lstm_normalizer"].transform(raw)

    fig2 = go.Figure()
    if if_scores_norm is not None:
        fig2.add_trace(go.Scatter(
            x=test_df[loaded.timestamp_column], y=if_scores_norm,
            mode="lines", name="Isolation Forest (normalized)",
        ))
    if lstm_scores_norm is not None and lstm_end_idx:
        x_vals = test_df[loaded.timestamp_column].values[lstm_end_idx]
        fig2.add_trace(go.Scatter(
            x=x_vals, y=lstm_scores_norm,
            mode="lines", name="LSTM Autoencoder (normalized)",
        ))
    if if_scores_norm is not None and lstm_scores_norm is not None and lstm_end_idx:
        aligned_if = if_scores_norm[lstm_end_idx]
        fused = fuse_scores(aligned_if, lstm_scores_norm, alpha)
        x_vals = test_df[loaded.timestamp_column].values[lstm_end_idx]
        fig2.add_trace(go.Scatter(
            x=x_vals, y=fused, mode="lines", name="Fused score",
            line=dict(width=3, color="black"),
        ))
    fig2.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10),
                        yaxis_title="Normalized anomaly score")
    st.plotly_chart(fig2, use_container_width=True)


def render_live_replay(cfg, subsystems_cfg, loaded, train_df, val_df, test_df, marker, if_ready, lstm_ready):
    st.subheader("Live Telemetry Replay")
    st.caption(
        "Replays rows from your real dataset in sequence to simulate a live "
        "telemetry stream. No synthetic values are generated."
    )

    source_map = {"train": train_df, "validation": val_df, "test": test_df, "full": loaded.df}
    default_source = cfg.get("live_replay", {}).get("source", "test")

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        source_choice = st.selectbox("Replay source", list(source_map.keys()),
                                      index=list(source_map.keys()).index(default_source)
                                      if default_source in source_map else 0)
    with col_b:
        interval = st.number_input("Interval (seconds)", min_value=0.05, max_value=5.0,
                                    value=float(cfg.get("live_replay", {}).get("default_interval_seconds", 1.0)),
                                    step=0.05)
    with col_c:
        n_steps = st.number_input("Steps per run", min_value=5, max_value=500, value=50, step=5)

    replay_df = source_map[source_choice].reset_index(drop=True)

    if "replay_position" not in st.session_state:
        st.session_state.replay_position = 0

    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.button("▶️ Run live simulation", use_container_width=True)
    with col2:
        reset = st.button("⏮️ Reset to start", use_container_width=True)
    with col3:
        st.metric("Current position", f"{st.session_state.replay_position} / {len(replay_df)}")

    if reset:
        st.session_state.replay_position = 0

    placeholder_metrics = st.empty()
    placeholder_chart = st.empty()
    placeholder_alert = st.empty()

    if not if_ready and not lstm_ready:
        st.info("Train at least one model to see live anomaly scoring during replay.")

    if start:
        if len(replay_df) < 2:
            st.warning("Not enough rows in this split to replay.")
            return

        ref = _reference_scores(marker)
        history_rows = []

        end = min(st.session_state.replay_position + int(n_steps), len(replay_df))
        for i in range(st.session_state.replay_position, end):
            history_rows.append(replay_df.iloc[[i]])
            history_df = pd.concat(history_rows, ignore_index=True)
            current_row = replay_df.iloc[i]

            # --- Metrics row: current telemetry values ---
            with placeholder_metrics.container():
                cols = st.columns(min(len(loaded.feature_columns), 6))
                for j, feat in enumerate(loaded.feature_columns[:6]):
                    cols[j % len(cols)].metric(feat, f"{current_row[feat]:.3f}")

            # --- Chart: recent history for first feature as a quick visual ---
            with placeholder_chart.container():
                chart_feature = loaded.feature_columns[0]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_df[loaded.timestamp_column], y=history_df[chart_feature],
                    mode="lines+markers", name=chart_feature,
                ))
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                                   title=f"{chart_feature} (live replay)")
                st.plotly_chart(fig, use_container_width=True, key=f"live_chart_{i}")

            # --- Scoring + alert (only if enough history for LSTM window, else IF only) ---
            severity = "Low"
            subsystems = []
            top_feats = []
            trends = {}
            fused_val = None

            if if_ready and ref["if_normalizer"] is not None and ref["if_severity"] is not None:
                raw = isolation_forest_scores(history_df.tail(1), loaded.feature_columns)
                if_norm = ref["if_normalizer"].transform(raw)[0]
                severity = ref["if_severity"].classify(if_norm)
                fused_val = if_norm

            if lstm_ready and ref["lstm_normalizer"] is not None and len(history_df) >= 2:
                end_idx, raw, per_feat_err = lstm_scores(history_df, loaded.feature_columns)
                if len(raw) > 0:
                    lstm_norm = ref["lstm_normalizer"].transform(raw)[-1]
                    top_feats = top_contributing_features(per_feat_err[-1], loaded.feature_columns, top_n=3)
                    subsystems = affected_subsystems(top_feats, subsystems_cfg)
                    if ref["if_normalizer"] is not None and fused_val is not None:
                        alpha = float(cfg.get("fusion", {}).get("alpha", 0.5))
                        fused_val = alpha * fused_val + (1 - alpha) * lstm_norm
                        if ref["fusion_severity"] is not None:
                            severity = ref["fusion_severity"].classify(fused_val)
                    elif ref["lstm_severity"] is not None:
                        fused_val = lstm_norm
                        severity = ref["lstm_severity"].classify(lstm_norm)

            trend_window = int(cfg.get("trend", {}).get("window", 10))
            flat_thresh = float(cfg.get("trend", {}).get("slope_flat_threshold", 0.01))
            trends = detect_trends_for_features(history_df.tail(trend_window), loaded.feature_columns,
                                                 trend_window, flat_thresh)

            with placeholder_alert.container():
                render_alert(severity, subsystems, top_feats, trends, fused_val)

            time.sleep(float(interval))

        st.session_state.replay_position = end
        st.rerun()


def render_evaluation():
    st.subheader("Model Evaluation")
    metrics_path = resolve_path("outputs/metrics.csv")
    if not os.path.exists(metrics_path):
        st.info(
            "**Evaluation results unavailable — train and evaluate the model first.**\n\n"
            "```\npython src/evaluate.py\n```"
        )
        return

    df = pd.read_csv(metrics_path)
    if "note" in df.columns and len(df.columns) == 1:
        st.warning(df["note"].iloc[0])
        return

    st.dataframe(df, use_container_width=True)

    if "model" in df.columns and "f1_score" in df.columns:
        fig = go.Figure(go.Bar(x=df["model"], y=df["f1_score"]))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                           title="F1 Score by Model", yaxis_title="F1 score")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
