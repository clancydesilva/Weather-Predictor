"""
src/logger.py
─────────────
Centralised logging configuration for the Cork City Weather Predictor.

Sets up file + console handlers with consistent formatting.
Import this once at the top of each module that needs logging:

    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("Starting retrain...")

Log files
---------
    logs/api.log      — API requests and reload events
    logs/pipeline.log — fetch, retrain, and data pipeline events

All log files rotate at 5 MB, keeping 3 backups. This keeps the
logs/ directory small even after months of nightly retraining.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Project root = parent of this file's directory
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, log_file: str = "pipeline.log") -> logging.Logger:
    """
    Return a logger that writes to both the console and a rotating file.

    Parameters
    ----------
    name     : typically __name__ of the calling module
    log_file : filename under logs/ (default: pipeline.log)
    """
    logger = logging.getLogger(name)

    # Only configure handlers once per logger name
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    # ── Console handler (INFO and above) ──────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # ── Rotating file handler (DEBUG and above) ───────────────────────────────
    log_path = _LOG_DIR / log_file
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoids duplicate console output)
    logger.propagate = False

    return logger


def get_api_logger(name: str) -> logging.Logger:
    """Convenience wrapper — logs to logs/api.log instead of pipeline.log."""
    return get_logger(name, log_file="api.log")
