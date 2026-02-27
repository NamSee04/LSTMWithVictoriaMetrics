"""
LSTM Anomaly Detection Model — PyTorch-based LSTM for time-series anomaly detection.

The model learns to predict the next value in a time series. Anomalies are detected
when the prediction error exceeds a learned threshold (mean + sigma * std of errors).

Output format matches vmanomaly: anomaly_score, yhat, yhat_lower, yhat_upper.
"""

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Force CPU
DEVICE = torch.device("cpu")


class LSTMNetwork(nn.Module):
    """PyTorch LSTM network for sequence-to-one prediction."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=DEVICE)

        out, _ = self.lstm(x, (h0, c0))
        # Take the last time step output
        out = self.fc(out[:, -1, :])
        return out


def _create_sequences(data: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Create input sequences and target values for LSTM training."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i : i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)


class LSTMAnomaly:
    """LSTM-based anomaly detection model."""

    def __init__(self, config: dict[str, Any]):
        self.sequence_length: int = config.get("sequence_length", 60)
        self.hidden_size: int = config.get("hidden_size", 64)
        self.num_layers: int = config.get("num_layers", 2)
        self.dropout: float = config.get("dropout", 0.2)
        self.epochs: int = config.get("epochs", 50)
        self.learning_rate: float = config.get("learning_rate", 0.001)
        self.threshold_sigma: float = config.get("threshold_sigma", 3.0)

        # Per-series state: each unique labelset gets its own model
        self._models: dict[str, LSTMNetwork] = {}
        self._scalers: dict[str, MinMaxScaler] = {}
        self._thresholds: dict[str, tuple[float, float]] = {}  # (mean_error, std_error)

        logger.info(
            f"[Model] LSTM config: seq_len={self.sequence_length}, "
            f"hidden={self.hidden_size}, layers={self.num_layers}, "
            f"epochs={self.epochs}, lr={self.learning_rate}, "
            f"threshold_sigma={self.threshold_sigma}"
        )

    def _get_or_create_model(self, series_key: str) -> LSTMNetwork:
        """Get existing model for a series or create a new one."""
        if series_key not in self._models:
            model = LSTMNetwork(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(DEVICE)
            self._models[series_key] = model
            self._scalers[series_key] = MinMaxScaler(feature_range=(0, 1))
            logger.info(f"[Model] Created new LSTM model for series '{series_key}'")
        return self._models[series_key]

    def fit(self, series_key: str, df: pd.DataFrame) -> bool:
        """
        Train LSTM on a single time series.

        Args:
            series_key: Unique identifier for this series.
            df: DataFrame with ['timestamp', 'value'] columns.

        Returns:
            True if training succeeded, False otherwise.
        """
        values = df["value"].values.astype(np.float32)

        if len(values) < self.sequence_length + 10:
            logger.warning(
                f"[Model] Not enough data for '{series_key}': "
                f"{len(values)} points, need >= {self.sequence_length + 10}"
            )
            return False

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
        self._scalers[series_key] = scaler

        # Create sequences
        X, y = _create_sequences(scaled, self.sequence_length)
        X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(DEVICE)  # (N, seq_len, 1)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1).to(DEVICE)  # (N, 1)

        # Create or get model
        model = self._get_or_create_model(series_key)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.debug(f"[Model] '{series_key}' epoch {epoch+1}/{self.epochs}, loss={loss.item():.6f}")

        # Calculate error distribution for threshold
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy().flatten()
        errors = np.abs(y - predictions)
        mean_err = float(np.mean(errors))
        std_err = float(np.std(errors))
        self._thresholds[series_key] = (mean_err, std_err)

        logger.info(
            f"[Model] Fitted '{series_key}': loss={loss.item():.6f}, "
            f"error_mean={mean_err:.6f}, error_std={std_err:.6f}"
        )
        return True

    def infer(
        self, series_key: str, df: pd.DataFrame
    ) -> Optional[dict[str, list[tuple[float, float]]]]:
        """
        Run inference on new data points.

        Args:
            series_key: Unique identifier for this series.
            df: DataFrame with ['timestamp', 'value'] columns.

        Returns:
            Dict mapping output variable names to list of (timestamp, value) tuples:
            - 'anomaly_score': 0-1+ score (>1 = anomaly)
            - 'yhat': predicted value
            - 'yhat_lower': lower bound
            - 'yhat_upper': upper bound
            Returns None if model not yet trained or insufficient data.
        """
        if series_key not in self._models:
            logger.warning(f"[Model] No trained model for '{series_key}', skipping inference")
            return None

        if series_key not in self._thresholds:
            logger.warning(f"[Model] No threshold computed for '{series_key}', skipping inference")
            return None

        values = df["value"].values.astype(np.float32)
        timestamps = df["timestamp"].values

        if len(values) < self.sequence_length:
            logger.warning(
                f"[Model] Not enough data for inference on '{series_key}': "
                f"{len(values)} points, need >= {self.sequence_length}"
            )
            return None

        model = self._models[series_key]
        scaler = self._scalers[series_key]
        mean_err, std_err = self._thresholds[series_key]

        # Normalize
        scaled = scaler.transform(values.reshape(-1, 1)).flatten()

        # Create sequences for inference
        X, y_actual_scaled = _create_sequences(scaled, self.sequence_length)
        X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(DEVICE)

        # Predict
        model.eval()
        with torch.no_grad():
            predictions_scaled = model(X_tensor).cpu().numpy().flatten()

        # Inverse transform predictions back to original scale
        predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        actuals = values[self.sequence_length:]
        infer_timestamps = timestamps[self.sequence_length:]

        # Calculate anomaly threshold in scaled space
        threshold = mean_err + self.threshold_sigma * std_err
        if threshold == 0:
            threshold = 1e-6  # prevent division by zero

        # Calculate errors and scores in scaled space
        errors_scaled = np.abs(y_actual_scaled - predictions_scaled)
        anomaly_scores = errors_scaled / threshold

        # Compute bounds in original scale
        # yhat_lower/upper based on threshold mapped back to original scale
        threshold_original = abs(
            scaler.inverse_transform([[threshold]])[0][0]
            - scaler.inverse_transform([[0.0]])[0][0]
        )
        yhat_lower = predictions - threshold_original
        yhat_upper = predictions + threshold_original

        # Build output
        results: dict[str, list[tuple[float, float]]] = {
            "anomaly_score": [],
            "yhat": [],
            "yhat_lower": [],
            "yhat_upper": [],
        }

        for i in range(len(infer_timestamps)):
            ts = float(infer_timestamps[i])
            results["anomaly_score"].append((ts, float(anomaly_scores[i])))
            results["yhat"].append((ts, float(predictions[i])))
            results["yhat_lower"].append((ts, float(yhat_lower[i])))
            results["yhat_upper"].append((ts, float(yhat_upper[i])))

        logger.info(
            f"[Model] Infer '{series_key}': {len(results['anomaly_score'])} points, "
            f"max_score={max(s for _, s in results['anomaly_score']):.4f}"
        )
        return results

    def save(self, directory: str) -> None:
        """Save all trained models to disk."""
        os.makedirs(directory, exist_ok=True)
        for series_key, model in self._models.items():
            safe_key = series_key.replace("/", "_").replace(",", "_").replace("=", "_")
            path = os.path.join(directory, f"lstm_{safe_key}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "scaler_min": self._scalers[series_key].data_min_,
                "scaler_max": self._scalers[series_key].data_max_,
                "threshold": self._thresholds.get(series_key, (0.0, 0.0)),
            }, path)
            logger.info(f"[Model] Saved model for '{series_key}' to {path}")

    def load(self, directory: str) -> None:
        """Load trained models from disk."""
        if not os.path.exists(directory):
            logger.info(f"[Model] No saved models directory: {directory}")
            return

        for fname in os.listdir(directory):
            if not fname.endswith(".pt"):
                continue
            path = os.path.join(directory, fname)
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

            # Reconstruct series key from filename
            series_key = fname[5:-3]  # strip 'lstm_' and '.pt'
            model = self._get_or_create_model(series_key)
            model.load_state_dict(checkpoint["model_state_dict"])

            scaler = self._scalers[series_key]
            scaler.fit(np.array([[0], [1]]))  # dummy fit
            scaler.data_min_ = checkpoint["scaler_min"]
            scaler.data_max_ = checkpoint["scaler_max"]
            scaler.scale_ = 1.0 / (scaler.data_max_ - scaler.data_min_)
            scaler.min_ = -scaler.data_min_ * scaler.scale_

            self._thresholds[series_key] = checkpoint["threshold"]
            logger.info(f"[Model] Loaded model for '{series_key}' from {path}")
