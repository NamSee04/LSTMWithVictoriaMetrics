#!/usr/bin/env python3
"""
LSTM Anomaly Detection for VictoriaMetrics — Entry Point.

Usage:
    python -m src.main --config config.yaml
    python src/main.py --config config.yaml
"""

import argparse
import logging
import signal
import sys

import yaml

from .vm_reader import VmReader
from .vm_writer import VmWriter
from .lstm_model import LSTMAnomaly
from .scheduler import PeriodicScheduler

logger = logging.getLogger("lstm_anomaly")


def setup_logging(level: str = "INFO") -> None:
    """Configure logging format and level."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LSTM Anomaly Detection for VictoriaMetrics"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Load configuration
    config = load_config(args.config)

    # Initialize components
    reader = VmReader(config["reader"])
    writer = VmWriter(config["writer"])
    model = LSTMAnomaly(config["model"])
    scheduler = PeriodicScheduler(config["scheduler"], reader, writer, model)

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, stopping...")
        scheduler.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Print banner
    logger.info("=" * 60)
    logger.info("  LSTM Anomaly Detection for VictoriaMetrics")
    logger.info(f"  Reader: {reader.datasource_url}")
    logger.info(f"  Writer: {writer.datasource_url}")
    logger.info(f"  Queries: {list(reader.queries.keys())}")
    logger.info("=" * 60)

    # Start scheduler
    try:
        scheduler.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Save models on exit
        model.save("./model_checkpoints")
        logger.info("Models saved. Goodbye!")


if __name__ == "__main__":
    main()
