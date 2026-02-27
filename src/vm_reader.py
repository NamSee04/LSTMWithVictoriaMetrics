"""
VictoriaMetrics Reader — queries metrics from VictoriaMetrics via /api/v1/query_range.

Returns data as {query_alias: {labelset_key: pd.DataFrame}} where each DataFrame
has columns ['timestamp', 'value'].
"""

import logging
import time
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def _parse_duration_to_seconds(duration_str: str) -> int:
    """Parse duration string like '1m', '1h', '1d', '1w' to seconds."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    unit = duration_str[-1]
    if unit not in units:
        raise ValueError(f"Unknown duration unit '{unit}' in '{duration_str}'. Supported: {list(units.keys())}")
    return int(duration_str[:-1]) * units[unit]


def _labelset_key(metric: dict[str, str]) -> str:
    """Create a stable string key from a Prometheus metric labelset."""
    filtered = {k: v for k, v in sorted(metric.items()) if k != "__name__"}
    if not filtered:
        return "__no_labels__"
    return ",".join(f"{k}={v}" for k, v in filtered.items())


class VmReader:
    """Reads time-series data from VictoriaMetrics."""

    def __init__(self, config: dict[str, Any]):
        self.datasource_url = config["datasource_url"].rstrip("/")
        self.sampling_period = config.get("sampling_period", "1m")
        self.queries: dict[str, dict[str, Any]] = config.get("queries", {})
        self.timeout = _parse_duration_to_seconds(config.get("timeout", "30s"))

    def read(self, query_alias: str, start: float, end: float) -> dict[str, pd.DataFrame]:
        """
        Execute a range query and return data grouped by labelset.

        Args:
            query_alias: Key from the queries config section.
            start: Start timestamp (UNIX seconds).
            end: End timestamp (UNIX seconds).

        Returns:
            Dictionary mapping labelset_key -> DataFrame with ['timestamp', 'value'] columns.
        """
        query_cfg = self.queries[query_alias]
        expr = query_cfg if isinstance(query_cfg, str) else query_cfg["expr"]
        step = query_cfg.get("step", self.sampling_period) if isinstance(query_cfg, dict) else self.sampling_period

        url = f"{self.datasource_url}/api/v1/query_range"
        params = {
            "query": expr,
            "start": start,
            "end": end,
            "step": step,
        }

        logger.info(f"[Reader] Querying '{query_alias}': {expr} [{start} -> {end}], step={step}")

        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"[Reader] Query failed for '{query_alias}': {e}")
            return {}

        data = resp.json()
        if data.get("status") != "success":
            logger.error(f"[Reader] Query returned non-success status: {data}")
            return {}

        results: dict[str, pd.DataFrame] = {}
        for series in data.get("data", {}).get("result", []):
            metric = series.get("metric", {})
            values = series.get("values", [])

            key = _labelset_key(metric)
            timestamps = [float(v[0]) for v in values]
            vals = [float(v[1]) for v in values]

            df = pd.DataFrame({"timestamp": timestamps, "value": vals})
            results[key] = df
            logger.debug(f"[Reader] '{query_alias}' [{key}]: {len(df)} points")

        logger.info(f"[Reader] '{query_alias}': {len(results)} series returned")
        return results

    def read_all(self, start: float, end: float) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Execute all configured queries.

        Returns:
            {query_alias: {labelset_key: DataFrame}}
        """
        all_data: dict[str, dict[str, pd.DataFrame]] = {}
        for alias in self.queries:
            all_data[alias] = self.read(alias, start, end)
        return all_data

    def health_check(self) -> bool:
        """Check if VictoriaMetrics is reachable."""
        try:
            resp = requests.get(f"{self.datasource_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
