"""
feature_engineering.py — Compute technical indicators and ML features
from OHLCV DataFrames. All calculations are vectorised with pandas/numpy.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from utils import get_logger

logger = get_logger("FeatureEngineering")


# ─────────────────────────────────────────────
# Individual Indicator Functions
# ─────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing.
    Returns values in [0, 100].
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD Line, Signal Line, Histogram.
    Returns (macd_line, signal_line, histogram).
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (volatility proxy)."""
    high_low = df["high"] - df["low"]
    high_pc = (df["high"] - df["close"].shift(1)).abs()
    low_pc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def rolling_volatility(series: pd.Series, period: int = 14) -> pd.Series:
    """Rolling std-dev of log returns (annualised daily-bar basis)."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(period).std()


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Current volume relative to rolling mean."""
    vol_mean = df["volume"].rolling(period).mean()
    return df["volume"] / vol_mean.replace(0, np.nan)


def hl_range(df: pd.DataFrame) -> pd.Series:
    """High-low range as a fraction of close (intrabar volatility)."""
    return (df["high"] - df["low"]) / df["close"]


# ─────────────────────────────────────────────
# Main Feature Builder
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicator columns to an OHLCV DataFrame.

    Input columns required: open, high, low, close, volume, timestamp
    Returns a new DataFrame with additional feature columns.
    No rows are dropped — NaN handling is left to the caller.
    """
    df = df.copy()

    # ── Trend Indicators ─────────────────────
    df["ema_9"] = ema(df["close"], 9)
    df["ema_21"] = ema(df["close"], 21)
    df["ema_50"] = ema(df["close"], 50)

    # ── Momentum ─────────────────────────────
    df["rsi_14"] = rsi(df["close"], 14)

    macd_line, signal_line, histogram = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram

    # ── Volatility ───────────────────────────
    df["atr_14"] = atr(df, 14)
    df["volatility"] = rolling_volatility(df["close"], 14)
    df["hl_range"] = hl_range(df)

    # ── Returns ──────────────────────────────
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    # ── Volume ───────────────────────────────
    df["volume_ratio"] = volume_ratio(df, 20)

    logger.debug(f"Features built. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# Label Generator
# ─────────────────────────────────────────────

def build_labels(df: pd.DataFrame, lookahead: int = 3) -> pd.Series:
    """
    Binary classification target:
      1  → close N candles ahead is higher than current close  (UP)
      0  → close N candles ahead is lower or equal             (DOWN/FLAT)

    The last `lookahead` rows will have NaN labels.
    """
    future_close = df["close"].shift(-lookahead)
    label = (future_close > df["close"]).astype(float)
    label.iloc[-lookahead:] = np.nan
    return label


# ─────────────────────────────────────────────
# Feature/Label Dataset Preparation
# ─────────────────────────────────────────────

def prepare_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    lookahead: int = 3,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Given a raw OHLCV DataFrame:
    1. Build features
    2. Build labels
    3. Drop rows with any NaN in features or label
    4. Return (X, y)
    """
    df_feat = build_features(df)
    df_feat["label"] = build_labels(df_feat, lookahead)

    # Drop rows with NaN in any feature or label
    cols_needed = feature_cols + ["label"]
    df_clean = df_feat.dropna(subset=cols_needed).copy()

    X = df_clean[feature_cols].reset_index(drop=True)
    y = df_clean["label"].reset_index(drop=True)

    logger.info(
        f"Dataset prepared: {len(X)} rows, {len(feature_cols)} features, "
        f"label balance UP={y.mean():.2%} DOWN={(1-y.mean()):.2%}"
    )
    return X, y


# ─────────────────────────────────────────────
# Live Feature Vector
# ─────────────────────────────────────────────

def get_latest_feature_vector(
    df: pd.DataFrame, feature_cols: List[str]
) -> pd.DataFrame:
    """
    Build features for a live OHLCV DataFrame and return the
    single-row feature vector for the most recent candle.
    Returns a DataFrame with one row (matching model input shape).
    """
    df_feat = build_features(df)
    # Use the last row that has no NaN in required features
    last_valid = df_feat.dropna(subset=feature_cols).iloc[[-1]]
    return last_valid[feature_cols].reset_index(drop=True)
