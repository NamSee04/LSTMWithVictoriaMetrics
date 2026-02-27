"""
Unit tests for VmReader and VmWriter.

Uses mock HTTP responses to test without a running VictoriaMetrics instance.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vm_reader import VmReader, _parse_duration_to_seconds, _labelset_key
from src.vm_writer import VmWriter


class TestParseDuration:
    def test_seconds(self):
        assert _parse_duration_to_seconds("30s") == 30

    def test_minutes(self):
        assert _parse_duration_to_seconds("5m") == 300

    def test_hours(self):
        assert _parse_duration_to_seconds("1h") == 3600

    def test_days(self):
        assert _parse_duration_to_seconds("1d") == 86400

    def test_weeks(self):
        assert _parse_duration_to_seconds("2w") == 1209600

    def test_invalid_unit(self):
        with pytest.raises(ValueError):
            _parse_duration_to_seconds("5x")


class TestLabelsetKey:
    def test_basic_labels(self):
        key = _labelset_key({"instance": "node1:9100", "job": "exporter"})
        assert key == "instance=node1:9100,job=exporter"

    def test_ignores_name(self):
        key = _labelset_key({"__name__": "cpu_total", "instance": "node1"})
        assert key == "instance=node1"

    def test_no_labels(self):
        key = _labelset_key({"__name__": "cpu_total"})
        assert key == "__no_labels__"

    def test_empty_dict(self):
        key = _labelset_key({})
        assert key == "__no_labels__"


class TestVmReader:
    @pytest.fixture
    def reader(self):
        return VmReader({
            "datasource_url": "http://localhost:8428",
            "sampling_period": "1m",
            "queries": {
                "cpu": {"expr": "rate(cpu_total[5m])"},
                "memory": "node_memory_usage",
            },
        })

    @patch("src.vm_reader.requests.get")
    def test_read_parses_response(self, mock_get, reader):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {
                        "metric": {"instance": "node1:9100", "job": "exporter"},
                        "values": [
                            [1700000000, "0.5"],
                            [1700000060, "0.6"],
                            [1700000120, "0.55"],
                        ],
                    }
                ],
            },
        }
        mock_get.return_value = mock_response

        result = reader.read("cpu", 1700000000, 1700000120)

        assert len(result) == 1
        key = "instance=node1:9100,job=exporter"
        assert key in result
        df = result[key]
        assert len(df) == 3
        assert list(df.columns) == ["timestamp", "value"]
        assert df["value"].iloc[0] == 0.5

    @patch("src.vm_reader.requests.get")
    def test_read_handles_error(self, mock_get, reader):
        mock_get.side_effect = Exception("Connection refused")
        result = reader.read("cpu", 1700000000, 1700000120)
        assert result == {}

    @patch("src.vm_reader.requests.get")
    def test_health_check_success(self, mock_get, reader):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        assert reader.health_check() is True

    @patch("src.vm_reader.requests.get")
    def test_health_check_failure(self, mock_get, reader):
        mock_get.side_effect = Exception("Connection refused")
        assert reader.health_check() is False


class TestVmWriter:
    @pytest.fixture
    def writer(self):
        return VmWriter({
            "datasource_url": "http://localhost:8428",
            "metric_format": {
                "__name__": "lstm_anomaly_$VAR",
                "for": "$QUERY_KEY",
            },
        })

    def test_format_metric_name(self, writer):
        name = writer._format_metric_name("anomaly_score", "cpu_usage")
        assert name == "lstm_anomaly_anomaly_score"

    def test_build_labels(self, writer):
        labels = writer._build_labels("cpu_usage", {"instance": "node1"})
        assert labels["for"] == "cpu_usage"
        assert labels["instance"] == "node1"

    def test_prometheus_line_format(self, writer):
        line = writer._to_prometheus_line(
            "lstm_anomaly_anomaly_score",
            {"for": "cpu", "instance": "node1"},
            1.5,
            1700000000000,
        )
        assert "lstm_anomaly_anomaly_score" in line
        assert 'for="cpu"' in line
        assert 'instance="node1"' in line
        assert "1.5" in line
        assert "1700000000000" in line

    @patch("src.vm_writer.requests.post")
    def test_write_success(self, mock_post, writer):
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        results = {
            "anomaly_score": [(1700000000.0, 0.5), (1700000060.0, 1.2)],
            "yhat": [(1700000000.0, 100.0), (1700000060.0, 105.0)],
        }

        success = writer.write("cpu_usage", {"instance": "node1"}, results)
        assert success is True
        mock_post.assert_called_once()

        # Verify the payload
        call_args = mock_post.call_args
        payload = call_args[1]["data"] if "data" in call_args[1] else call_args[0][1]
        assert "lstm_anomaly_anomaly_score" in payload
        assert "lstm_anomaly_yhat" in payload

    @patch("src.vm_writer.requests.post")
    def test_write_failure(self, mock_post, writer):
        mock_post.side_effect = Exception("Connection refused")
        results = {"anomaly_score": [(1700000000.0, 0.5)]}
        success = writer.write("cpu_usage", {}, results)
        assert success is False
