"""
utils.py — Logging, config loader, and shared helpers.
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

import yaml


# ─────────────────────────────────────────────
# Config Loader
# ─────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load and return the YAML config as a nested dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─────────────────────────────────────────────
# Logger Factory
# ─────────────────────────────────────────────

def get_logger(name: str, cfg: Dict[str, Any] = None) -> logging.Logger:
    """
    Return a named logger with console + optional file handler.
    If cfg is provided it reads logging.level and logging.file.
    """
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    level_str = "INFO"
    log_file = None
    if cfg:
        level_str = cfg.get("logging", {}).get("level", "INFO")
        log_file = cfg.get("logging", {}).get("file")

    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────
# Time Helpers
# ─────────────────────────────────────────────

def utc_now() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def ms_to_datetime(ms: int) -> datetime:
    """Convert Binance millisecond timestamp to UTC datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def datetime_to_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp."""
    return int(dt.timestamp() * 1000)


def sleep_until_next_candle(interval_seconds: int = 300) -> None:
    """
    Block until the next candle boundary (aligned to interval).
    e.g. for 5m candles: sleeps until next :00, :05, :10, …
    """
    now = time.time()
    remainder = now % interval_seconds
    wait = interval_seconds - remainder
    time.sleep(wait)


# ─────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────

def pct_change(old: float, new: float) -> float:
    """Return percentage change from old to new."""
    if old == 0:
        return 0.0
    return (new - old) / old


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value between low and high."""
    return max(low, min(high, value))


def round_price(price: float, tick: float = 0.01) -> float:
    """Round price to nearest tick size."""
    return round(round(price / tick) * tick, 10)
