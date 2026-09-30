"""
Train the LSTM Autoencoder temporal-anomaly model on REAL normal telemetry data.

Usage:
    python src/train_lstm_autoencoder.py [--config config/config.yaml]

Trains ONLY on data/telemetry.csv (or whatever dataset.path points to).
No synthetic sequences are generated at any point in this script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_utils import load_config, resolve_path, ConfigError  # noqa: E402
from src.data_loader import load_dataset, chronological_split, filter_normal_rows, DatasetError  # noqa: E402
from src.preprocessing import fit_scaler, transform_features, make_windows  # noqa: E402


def build_lstm_autoencoder(window_size: int, n_features: int, latent_dim: int, learning_rate: float):
    """Build a simple LSTM Autoencoder (encoder-decoder) in Keras."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=(window_size, n_features))
    encoded = layers.LSTM(latent_dim, activation="tanh", return_sequences=False)(inputs)
    repeated = layers.RepeatVector(window_size)(encoded)
    decoded = layers.LSTM(latent_dim, activation="tanh", return_sequences=True)(repeated)
    outputs = layers.TimeDistributed(layers.Dense(n_features))(decoded)

    autoencoder = models.Model(inputs, outputs, name="lstm_autoencoder")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return autoencoder


def main() -> int:
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder on real telemetry data.")
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

    train_df, val_df, _ = chronological_split(loaded.df, cfg)
    normal_train_df = filter_normal_rows(train_df, loaded)
    normal_val_df = filter_normal_rows(val_df, loaded)

    lstm_cfg = cfg.get("lstm", {})
    window_size = int(lstm_cfg.get("window_size", 30))
    step_size = int(lstm_cfg.get("step_size", 1))

    if len(normal_train_df) < window_size * 2:
        print(
            f"[DATA ERROR] Only {len(normal_train_df)} normal training rows available, "
            f"but window_size={window_size} requires substantially more rows "
            f"(at least {window_size * 2}) to train a meaningful model. "
            f"Reduce 'lstm.window_size' in config.yaml or supply more data."
        )
        return 1

    print(f"Training LSTM Autoencoder on {len(normal_train_df)} normal rows, "
          f"window_size={window_size}, features: {loaded.feature_columns}")

    scaler = fit_scaler(normal_train_df, loaded.feature_columns)
    X_train_scaled = transform_features(normal_train_df, loaded.feature_columns, scaler)
    train_windows = make_windows(X_train_scaled, window_size, step_size)

    val_windows = None
    if len(normal_val_df) >= window_size:
        X_val_scaled = transform_features(normal_val_df, loaded.feature_columns, scaler)
        val_windows = make_windows(X_val_scaled, window_size, step_size)

    if train_windows.shape[0] == 0:
        print("[DATA ERROR] No training windows could be constructed. "
              "Check window_size against the number of available rows.")
        return 1

    n_features = len(loaded.feature_columns)
    latent_dim = int(lstm_cfg.get("latent_dim", 16))
    epochs = int(lstm_cfg.get("epochs", 50))
    batch_size = int(lstm_cfg.get("batch_size", 32))
    learning_rate = float(lstm_cfg.get("learning_rate", 0.001))
    patience = int(lstm_cfg.get("validation_patience", 5))

    model = build_lstm_autoencoder(window_size, n_features, latent_dim, learning_rate)
    model.summary()

    import tensorflow as tf

    callbacks = []
    fit_kwargs = dict(
        x=train_windows,
        y=train_windows,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    if val_windows is not None and val_windows.shape[0] > 0:
        fit_kwargs["validation_data"] = (val_windows, val_windows)
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            )
        )
    else:
        print("[WARNING] No validation windows available — training without "
              "early stopping or validation monitoring.")
    fit_kwargs["callbacks"] = callbacks

    history = model.fit(**fit_kwargs)

    models_dir = resolve_path("models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "lstm_autoencoder.keras")
    scaler_path = os.path.join(models_dir, "lstm_scaler.joblib")
    meta_path = os.path.join(models_dir, "lstm_meta.json")

    model.save(model_path)

    import joblib
    joblib.dump(scaler, scaler_path)

    final_train_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1]) if "val_loss" in history.history else None

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_columns": loaded.feature_columns,
                "window_size": window_size,
                "step_size": step_size,
                "latent_dim": latent_dim,
                "n_training_windows": int(train_windows.shape[0]),
                "epochs_run": len(history.history["loss"]),
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss,
            },
            f,
            indent=2,
        )

    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")
    print(f"Saved metadata to {meta_path}")
    print(f"Final training loss: {final_train_loss:.6f}")
    if final_val_loss is not None:
        print(f"Final validation loss: {final_val_loss:.6f}")
    print("LSTM Autoencoder training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
