"""
feature_engineering.py — Compute technical indicators and ML features
from OHLCV DataFrames. All calculations are vectorised with pandas/numpy.

Enhanced with 15+ additional indicators:
- Stochastic Oscillator
- Bollinger Bands
- Money Flow Index (MFI)
- On-Balance Volume (OBV)
- VWAP
- Additional EMAs & SMAs
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


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


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


def stochastic(
    df: pd.DataFrame, period: int = 14, k_smooth: int = 3, d_smooth: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator.
    Returns (%K, %D) where values are in [0, 100].
    
    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = EMA of %K
    """
    low_min = df["low"].rolling(window=period).min()
    high_max = df["high"].rolling(window=period).max()
    
    k_percent = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-9)
    k_smooth_val = k_percent.ewm(span=k_smooth, adjust=False).mean()
    d_smooth_val = k_smooth_val.ewm(span=d_smooth, adjust=False).mean()
    
    return k_smooth_val, d_smooth_val


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.
    Returns (upper, middle, lower).
    """
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def bollinger_position(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Position within Bollinger Bands [0, 1].
    0 = at lower band, 1 = at upper band.
    """
    upper, middle, lower = bollinger_bands(series, period, num_std)
    position = (series - lower) / (upper - lower + 1e-9)
    return position.clip(0, 1)


def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Money Flow Index (MFI).
    Combines price action and volume.
    Returns values in [0, 100].
    """
    # Typical Price
    tp = (df["high"] + df["low"] + df["close"]) / 3
    
    # Raw Money Flow
    rmf = tp * df["volume"]
    
    # Positive/Negative Money Flow
    positive_mf = rmf.where(tp > tp.shift(1), 0)
    negative_mf = rmf.where(tp < tp.shift(1), 0)
    
    # Money Flow Ratio
    positive_sum = positive_mf.rolling(window=period).sum()
    negative_sum = negative_mf.rolling(window=period).sum()
    
    mfr = positive_sum / (negative_sum + 1e-9)
    mfi = 100 - (100 / (1 + mfr))
    
    return mfi


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume (OBV).
    Cumulative volume indicator.
    """
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = df["volume"].iloc[0]
    
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + df["volume"].iloc[i]
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - df["volume"].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]
    
    return obv


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume-Weighted Average Price (VWAP).
    Tracks the average price weighted by volume.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cumul_vol = df["volume"].cumsum()
    cumul_tp_vol = (tp * df["volume"]).cumsum()
    vwap_val = cumul_tp_vol / cumul_vol
    return vwap_val


def vwap_position(df: pd.DataFrame) -> pd.Series:
    """
    Position relative to VWAP.
    Positive = price above VWAP, Negative = below.
    """
    vwap_val = vwap(df)
    return (df["close"] - vwap_val) / df["close"]


# ─────────────────────────────────────────────
# Main Feature Builder
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicator columns to an OHLCV DataFrame.

    Input columns required: open, high, low, close, volume, timestamp
    Returns a new DataFrame with additional feature columns.
    No rows are dropped — NaN handling is left to the caller.
    
    Total Features: 27+
    """
    df = df.copy()

    # ── Trend Indicators ─────────────────────
    df["ema_9"] = ema(df["close"], 9)
    df["ema_21"] = ema(df["close"], 21)
    df["ema_50"] = ema(df["close"], 50)
    df["ema_200"] = ema(df["close"], 200)
    df["sma_50"] = sma(df["close"], 50)

    # ── Momentum ─────────────────────────────
    df["rsi_14"] = rsi(df["close"], 14)
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(float)
    df["rsi_oversold"] = (df["rsi_14"] < 30).astype(float)

    macd_line, signal_line, histogram = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram
    df["macd_positive"] = (df["macd_hist"] > 0).astype(float)

    stoch_k, stoch_d = stochastic(df, 14, 3, 3)
    df["stochastic_k"] = stoch_k
    df["stochastic_d"] = stoch_d
    df["stoch_overbought"] = (stoch_k > 80).astype(float)
    df["stoch_oversold"] = (stoch_k < 20).astype(float)

    # ── Volatility ───────────────────────────
    df["atr_14"] = atr(df, 14)
    df["atr_21"] = atr(df, 21)
    df["volatility"] = rolling_volatility(df["close"], 14)
    df["hl_range"] = hl_range(df)

    # ── Bollinger Bands ──────────────────────
    upper, middle, lower = bollinger_bands(df["close"], 20, 2.0)
    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower
    df["bb_position"] = bollinger_position(df["close"], 20, 2.0)
    df["bb_width"] = (upper - lower) / middle

    # ── Volume Analysis ──────────────────────
    df["volume_ratio"] = volume_ratio(df, 20)
    df["mfi_14"] = money_flow_index(df, 14)
    df["obv"] = on_balance_volume(df)
    df["obv_ema"] = ema(df["obv"], 14)

    # ── Price Action ─────────────────────────
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    # ── VWAP ─────────────────────────────────
    df["vwap"] = vwap(df)
    df["vwap_position"] = vwap_position(df)

    logger.debug(f"Features built. Shape: {df.shape}, Features: {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']])}")
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
