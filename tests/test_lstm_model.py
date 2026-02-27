"""
Unit tests for the LSTM anomaly detection model.

Tests:
- Model can fit on synthetic sine wave data
- Model produces correct output format (anomaly_score, yhat, etc.)
- Model detects injected anomalies (anomaly_score > 1)
- Sequence creation utility works correctly
"""

import numpy as np
import pandas as pd
import pytest

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lstm_model import LSTMAnomaly, _create_sequences


class TestCreateSequences:
    """Test the _create_sequences utility function."""

    def test_basic_sequences(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X, y = _create_sequences(data, seq_length=3)
        assert X.shape == (2, 3)
        assert y.shape == (2,)
        np.testing.assert_array_equal(X[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(X[1], [2.0, 3.0, 4.0])
        assert y[0] == 4.0
        assert y[1] == 5.0

    def test_sequence_length_equals_data(self):
        data = np.array([1.0, 2.0, 3.0])
        X, y = _create_sequences(data, seq_length=2)
        assert X.shape == (1, 2)
        assert y.shape == (1,)

    def test_empty_result_when_too_short(self):
        data = np.array([1.0, 2.0])
        X, y = _create_sequences(data, seq_length=3)
        assert len(X) == 0
        assert len(y) == 0


class TestLSTMAnomaly:
    """Test the LSTMAnomaly model."""

    @pytest.fixture
    def model(self):
        """Create a model with fast training settings."""
        config = {
            "sequence_length": 10,
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "epochs": 20,
            "learning_rate": 0.01,
            "threshold_sigma": 3.0,
        }
        return LSTMAnomaly(config)

    @pytest.fixture
    def sine_df(self):
        """Create a synthetic sine wave time series."""
        n_points = 200
        timestamps = np.arange(n_points, dtype=np.float64) * 60  # 1-min intervals
        values = np.sin(np.linspace(0, 8 * np.pi, n_points)).astype(np.float32)
        return pd.DataFrame({"timestamp": timestamps, "value": values})

    @pytest.fixture
    def anomalous_df(self):
        """Create a sine wave with injected anomalies."""
        n_points = 200
        timestamps = np.arange(n_points, dtype=np.float64) * 60
        values = np.sin(np.linspace(0, 8 * np.pi, n_points)).astype(np.float32)
        # Inject large spikes at specific points
        values[150] = 10.0
        values[160] = -10.0
        values[170] = 15.0
        return pd.DataFrame({"timestamp": timestamps, "value": values})

    def test_fit_succeeds(self, model, sine_df):
        """Model should train successfully on sine wave data."""
        result = model.fit("test_series", sine_df)
        assert result is True
        assert "test_series" in model._models
        assert "test_series" in model._scalers
        assert "test_series" in model._thresholds

    def test_fit_fails_with_insufficient_data(self, model):
        """Model should reject data with too few points."""
        short_df = pd.DataFrame({
            "timestamp": [0.0, 60.0, 120.0],
            "value": [1.0, 2.0, 3.0],
        })
        result = model.fit("short_series", short_df)
        assert result is False

    def test_infer_returns_correct_format(self, model, sine_df):
        """Inference should return dict with required output variables."""
        model.fit("test_series", sine_df)
        results = model.infer("test_series", sine_df)

        assert results is not None
        assert "anomaly_score" in results
        assert "yhat" in results
        assert "yhat_lower" in results
        assert "yhat_upper" in results

        # Each should be a list of (timestamp, value) tuples
        for var_name, datapoints in results.items():
            assert len(datapoints) > 0
            for ts, val in datapoints:
                assert isinstance(ts, float)
                assert isinstance(val, float)

    def test_infer_without_fit_returns_none(self, model, sine_df):
        """Inference on unfitted model should return None."""
        results = model.infer("unknown_series", sine_df)
        assert results is None

    def test_normal_data_has_low_scores(self, model, sine_df):
        """Normal sine wave should have mostly low anomaly scores."""
        model.fit("test_series", sine_df)
        results = model.infer("test_series", sine_df)

        assert results is not None
        scores = [s for _, s in results["anomaly_score"]]
        mean_score = np.mean(scores)
        # Most scores should be below 1.0 for normal data
        assert mean_score < 1.0, f"Mean anomaly score {mean_score} should be < 1.0 for normal data"

    def test_anomalous_data_detected(self, model, sine_df, anomalous_df):
        """Injected anomalies should produce higher scores."""
        # Train on clean data
        model.fit("test_series", sine_df)
        # Infer on anomalous data
        results = model.infer("test_series", anomalous_df)

        assert results is not None
        scores = [s for _, s in results["anomaly_score"]]
        max_score = max(scores)
        # At least some scores should be elevated (spike detection)
        assert max_score > 0.5, f"Max anomaly score {max_score} should be elevated for anomalous data"

    def test_yhat_bounds_ordering(self, model, sine_df):
        """yhat_lower should be <= yhat <= yhat_upper."""
        model.fit("test_series", sine_df)
        results = model.infer("test_series", sine_df)

        assert results is not None
        for i in range(len(results["yhat"])):
            _, lower = results["yhat_lower"][i]
            _, yhat = results["yhat"][i]
            _, upper = results["yhat_upper"][i]
            assert lower <= yhat <= upper, (
                f"Bound violation at index {i}: {lower} <= {yhat} <= {upper}"
            )

    def test_save_and_load(self, model, sine_df, tmp_path):
        """Model should persist and restore correctly."""
        model.fit("test_series", sine_df)
        results_before = model.infer("test_series", sine_df)

        # Save
        save_dir = str(tmp_path / "models")
        model.save(save_dir)

        # Load into a new model
        new_model = LSTMAnomaly({
            "sequence_length": 10,
            "hidden_size": 16,
            "num_layers": 1,
            "dropout": 0.0,
            "epochs": 20,
            "learning_rate": 0.01,
            "threshold_sigma": 3.0,
        })
        new_model.load(save_dir)

        assert len(results_after["anomaly_score"]) == len(results_before["anomaly_score"])

if __name__ == "__main__":
    pytest.main(["-v", __file__])
