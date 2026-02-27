"""
Periodic Scheduler — orchestrates the fit/infer loop.

Triggers model training (fit) and inference (infer) at configurable intervals,
coordinating the reader → model → writer pipeline.
"""

import logging
import threading
import time
from typing import Any

from .vm_reader import VmReader, _parse_duration_to_seconds, _labelset_key
from .vm_writer import VmWriter
from .lstm_model import LSTMAnomaly

logger = logging.getLogger(__name__)


class PeriodicScheduler:
    """Runs fit and infer cycles at configured intervals."""

    def __init__(
        self,
        config: dict[str, Any],
        reader: VmReader,
        writer: VmWriter,
        model: LSTMAnomaly,
    ):
        self.fit_every = _parse_duration_to_seconds(config.get("fit_every", "1h"))
        self.infer_every = _parse_duration_to_seconds(config.get("infer_every", "1m"))
        self.fit_window = _parse_duration_to_seconds(config.get("fit_window", "1d"))

        self.reader = reader
        self.writer = writer
        self.model = model
        self._running = False
        self._last_fit_time: float = 0.0

        logger.info(
            f"[Scheduler] fit_every={self.fit_every}s, "
            f"infer_every={self.infer_every}s, "
            f"fit_window={self.fit_window}s"
        )

    def _run_fit(self) -> None:
        """Execute a fit (training) cycle on all configured queries."""
        now = time.time()
        start = now - self.fit_window
        end = now

        logger.info(f"[Scheduler] Starting FIT cycle (window: {self.fit_window}s)")

        try:
            all_data = self.reader.read_all(start, end)
        except Exception as e:
            logger.error(f"[Scheduler] FIT read failed: {e}")
            return

        fit_count = 0
        for query_alias, series_dict in all_data.items():
            for labelset_key, df in series_dict.items():
                full_key = f"{query_alias}::{labelset_key}"
                try:
                    success = self.model.fit(full_key, df)
                    if success:
                        fit_count += 1
                except Exception as e:
                    logger.error(f"[Scheduler] FIT failed for '{full_key}': {e}")

        self._last_fit_time = now
        logger.info(f"[Scheduler] FIT cycle complete: {fit_count} models trained")

    def _run_infer(self) -> None:
        """Execute an inference cycle and write results."""
        now = time.time()
        # For inference, we need at least sequence_length data points
        # Use a window of 2x sequence_length to ensure enough data
        infer_window = max(
            self.model.sequence_length * 2 * 60,  # assume ~1min sampling
            self.infer_every * 10,
        )
        start = now - infer_window
        end = now

        logger.info("[Scheduler] Starting INFER cycle")

        try:
            all_data = self.reader.read_all(start, end)
        except Exception as e:
            logger.error(f"[Scheduler] INFER read failed: {e}")
            return

        infer_count = 0
        for query_alias, series_dict in all_data.items():
            for labelset_key, df in series_dict.items():
                full_key = f"{query_alias}::{labelset_key}"

                try:
                    results = self.model.infer(full_key, df)
                    if results is None:
                        continue

                    # Parse original labels from labelset_key
                    original_labels = {}
                    if labelset_key != "__no_labels__":
                        for pair in labelset_key.split(","):
                            if "=" in pair:
                                k, v = pair.split("=", 1)
                                original_labels[k] = v

                    # Write results
                    self.writer.write(query_alias, original_labels, results)
                    infer_count += 1
                except Exception as e:
                    logger.error(f"[Scheduler] INFER failed for '{full_key}': {e}")

        logger.info(f"[Scheduler] INFER cycle complete: {infer_count} series processed")

    def _should_fit(self) -> bool:
        """Check if enough time has passed for a new fit cycle."""
        if self._last_fit_time == 0.0:
            return True
        return (time.time() - self._last_fit_time) >= self.fit_every

    def run(self) -> None:
        """Start the main scheduler loop (blocking)."""
        self._running = True
        logger.info("[Scheduler] Starting scheduler loop")

        # Health check
        if not self.reader.health_check():
            logger.warning("[Scheduler] VictoriaMetrics reader health check failed — will retry")
        if not self.writer.health_check():
            logger.warning("[Scheduler] VictoriaMetrics writer health check failed — will retry")

        # Initial fit
        logger.info("[Scheduler] Running initial FIT...")
        self._run_fit()

        while self._running:
            cycle_start = time.time()

            try:
                # Check if we need to retrain
                if self._should_fit():
                    self._run_fit()

                # Always run inference
                self._run_infer()

            except KeyboardInterrupt:
                logger.info("[Scheduler] Interrupted by user")
                self.stop()
                break
            except Exception as e:
                logger.error(f"[Scheduler] Unexpected error in main loop: {e}", exc_info=True)

            # Sleep until next infer cycle
            elapsed = time.time() - cycle_start
            sleep_time = max(0, self.infer_every - elapsed)
            if sleep_time > 0:
                logger.debug(f"[Scheduler] Sleeping {sleep_time:.1f}s until next cycle")
                time.sleep(sleep_time)

    def stop(self) -> None:
        """Stop the scheduler loop."""
        logger.info("[Scheduler] Stopping scheduler")
        self._running = False
