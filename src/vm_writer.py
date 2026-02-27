"""
VictoriaMetrics Writer — pushes anomaly detection results back to VictoriaMetrics.

Uses the Prometheus text exposition format via /api/v1/import/prometheus endpoint.
"""

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


class VmWriter:
    """Writes anomaly scores and predictions back to VictoriaMetrics."""

    def __init__(self, config: dict[str, Any]):
        self.datasource_url = config["datasource_url"].rstrip("/")
        self.metric_format: dict[str, str] = config.get("metric_format", {
            "__name__": "lstm_anomaly_$VAR",
            "for": "$QUERY_KEY",
        })
        self.timeout = 10

    def _format_metric_name(self, var: str, query_key: str) -> str:
        """Replace $VAR and $QUERY_KEY placeholders in __name__."""
        return self.metric_format["__name__"].replace("$VAR", var).replace("$QUERY_KEY", query_key)

    def _build_labels(self, query_key: str, original_labels: dict[str, str]) -> dict[str, str]:
        """Build label set from metric_format config + original labels."""
        labels = {}
        for key, val in self.metric_format.items():
            if key == "__name__":
                continue
            labels[key] = val.replace("$VAR", "").replace("$QUERY_KEY", query_key)

        # Merge original labels (they take precedence for conflicts except reserved keys)
        for k, v in original_labels.items():
            if k != "__name__" and k not in labels:
                labels[k] = v

        return labels

    def _to_prometheus_line(
        self,
        metric_name: str,
        labels: dict[str, str],
        value: float,
        timestamp_ms: int,
    ) -> str:
        """Format a single metric in Prometheus text exposition format."""
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{metric_name}{{{label_str}}} {value} {timestamp_ms}"
        return f"{metric_name} {value} {timestamp_ms}"

    def write(
        self,
        query_key: str,
        original_labels: dict[str, str],
        results: dict[str, list[tuple[float, float]]],
    ) -> bool:
        """
        Write model results to VictoriaMetrics.

        Args:
            query_key: The query alias this result belongs to.
            original_labels: Original metric labels from the input series.
            results: Dict mapping variable name (e.g. 'anomaly_score', 'yhat')
                     to list of (timestamp, value) tuples.

        Returns:
            True if write succeeded, False otherwise.
        """
        lines = []
        for var_name, datapoints in results.items():
            metric_name = self._format_metric_name(var_name, query_key)
            labels = self._build_labels(query_key, original_labels)

            for ts, val in datapoints:
                timestamp_ms = int(ts * 1000)
                line = self._to_prometheus_line(metric_name, labels, val, timestamp_ms)
                lines.append(line)

        if not lines:
            logger.debug(f"[Writer] No data to write for '{query_key}'")
            return True

        payload = "\n".join(lines) + "\n"
        url = f"{self.datasource_url}/api/v1/import/prometheus"

        try:
            resp = requests.post(
                url,
                data=payload,
                headers={"Content-Type": "text/plain"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            logger.info(f"[Writer] Wrote {len(lines)} datapoints for '{query_key}'")
            return True
        except requests.RequestException as e:
            logger.error(f"[Writer] Failed to write for '{query_key}': {e}")
            return False

    def health_check(self) -> bool:
        """Check if VictoriaMetrics writer endpoint is reachable."""
        try:
            resp = requests.get(f"{self.datasource_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
