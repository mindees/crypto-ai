"""Causal technical indicators.

Every function takes a tz-aware UTC-indexed OHLCV DataFrame and returns a
DataFrame of feature columns with the **same index**. Values for row ``t``
use only OHLCV data at or before ``t`` (no lookahead). This is enforced by
``tests/test_no_lookahead.py``.

Implementation note: we deliberately keep these pandas-native to avoid hard-
depending on TA-Lib (which is painful to install on Windows). When TA-Lib
is available later we can plug it in behind the same callable signatures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV df missing columns: {sorted(missing)}")


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Returns + EMAs + EMA stack
# ---------------------------------------------------------------------------

def returns_and_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    return pd.DataFrame({
        "returns_1": close.pct_change(),
        "log_returns_1": np.log(close).diff(),
    }, index=df.index)


def ema_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    spans = (9, 21, 50, 120, 200)
    out: dict[str, pd.Series] = {}
    for span in spans:
        out[f"ema_{span}"] = close.ewm(span=span, adjust=False, min_periods=span).mean()

    # EMA stack score: +1 for each ascending alignment, -1 for each descending.
    # Range: -10 to +10. Bullish = +10 (9>21>50>120>200), bearish = -10.
    pairs = [(9, 21), (21, 50), (50, 120), (120, 200)]
    stack_score = pd.Series(0.0, index=df.index)
    for short, long in pairs:
        e_short = out[f"ema_{short}"]
        e_long = out[f"ema_{long}"]
        stack_score = stack_score + np.where(e_short > e_long, 1.0,
                                              np.where(e_short < e_long, -1.0, 0.0))
    out["ema_stack_score"] = stack_score

    # Price relative to the long EMAs
    out["price_above_ema_200"] = (close > out["ema_200"]).astype("float64")
    out["price_above_ema_120"] = (close > out["ema_120"]).astype("float64")
    out["ema_120_cycle_signal"] = np.where(close > out["ema_120"], 1.0, -1.0)
    out["ema_120_cycle_signal"] = pd.Series(out["ema_120_cycle_signal"], index=df.index)

    # Golden / death cross flags (50 vs 200)
    cross_up = (out["ema_50"] > out["ema_200"]) & (
        out["ema_50"].shift(1) <= out["ema_200"].shift(1)
    )
    cross_dn = (out["ema_50"] < out["ema_200"]) & (
        out["ema_50"].shift(1) >= out["ema_200"].shift(1)
    )
    out["golden_cross"] = cross_up.astype("float64")
    out["death_cross"] = cross_dn.astype("float64")

    # SMAs (just 50 and 200 — used for cross flags and longer-term context)
    out["sma_50"] = close.rolling(50, min_periods=50).mean()
    out["sma_200"] = close.rolling(200, min_periods=200).mean()
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# RSI, MACD, Bollinger, ATR
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    rsi_val[avg_loss == 0] = 100.0
    return rsi_val


def rsi_features(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    r = rsi(df["close"], length=length)
    slope = r.diff()
    return pd.DataFrame({
        f"rsi_{length}": r,
        f"rsi_{length}_slope": slope,
        f"rsi_{length}_overbought": (r > 70).astype("float64"),
        f"rsi_{length}_oversold": (r < 30).astype("float64"),
        f"rsi_{length}_zscore_50": _rolling_zscore(r, 50),
    }, index=df.index)


def macd_features(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    cross_dn = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": signal_line,
        "macd_hist": hist,
        "macd_hist_slope": hist.diff(),
        "macd_cross_up": cross_up.astype("float64"),
        "macd_cross_down": cross_dn.astype("float64"),
    }, index=df.index)


def bollinger_features(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    close = df["close"]
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    bandwidth = (upper - lower) / ma.replace(0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    mean_reversion_signal = np.where(
        close > upper, -1.0, np.where(close < lower, 1.0, 0.0)
    )
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_mid": ma,
        "bb_pct_b": pct_b,
        "bb_bandwidth": bandwidth,
        "bb_mean_reversion_signal": pd.Series(mean_reversion_signal, index=df.index),
    }, index=df.index)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr_features(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    tr = true_range(df)
    atr = tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    return pd.DataFrame({
        "true_range": tr,
        "atr": atr,
        "atr_pct": atr / df["close"].replace(0, np.nan),
    }, index=df.index)


# ---------------------------------------------------------------------------
# Volume + VWAP + OBV + realized vol
# ---------------------------------------------------------------------------

def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    vol = df["volume"]
    out = pd.DataFrame(index=df.index)
    out["volume_log"] = np.log1p(vol)
    out["volume_zscore_20"] = _rolling_zscore(vol, 20)
    out["volume_zscore_50"] = _rolling_zscore(vol, 50)
    out["volume_spike"] = (out["volume_zscore_20"] > 2.0).astype("float64")
    return out


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff().fillna(0.0))
    return (sign * df["volume"]).cumsum()


def vwap_rolling(df: pd.DataFrame, window: int = 24) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (typical * df["volume"]).rolling(window, min_periods=window).sum()
    v = df["volume"].rolling(window, min_periods=window).sum()
    return pv / v.replace(0, np.nan)


def vwap_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "obv": obv(df),
        "vwap_24": vwap_rolling(df, 24),
        "vwap_72": vwap_rolling(df, 72),
    }, index=df.index)


def realized_volatility(df: pd.DataFrame, windows: tuple[int, ...] = (20, 50, 100)) -> pd.DataFrame:
    rets = np.log(df["close"]).diff()
    out = pd.DataFrame(index=df.index)
    for w in windows:
        out[f"realized_volatility_{w}"] = rets.rolling(w, min_periods=w).std(ddof=0)
    out["realized_volatility"] = out[f"realized_volatility_{windows[0]}"]
    return out


# ---------------------------------------------------------------------------
# Distance-to-extreme features
# ---------------------------------------------------------------------------

def distance_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    # All-time running max/min are causal because cummax uses only the past.
    ath = close.cummax()
    atl = close.cummin()
    out = pd.DataFrame(index=df.index)
    out["distance_from_ath_pct"] = (close / ath - 1.0)
    out["distance_from_atl_pct"] = (close / atl - 1.0)

    # 52-week and 200-week windows. Length depends on the bar timeframe; the
    # caller (build_matrix) supplies the right window per timeframe.
    for w_name, w in (("52w_24h", 365), ("52w_1h", 365 * 24), ("52w_4h", 365 * 6)):
        if len(df) < w:
            continue
        out[f"distance_from_{w_name}_high"] = (
            close / close.rolling(w, min_periods=w).max() - 1.0
        )
        out[f"distance_from_{w_name}_low"] = (
            close / close.rolling(w, min_periods=w).min() - 1.0
        )
    return out


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the standard indicator stack on a single OHLCV frame.

    Returns a DataFrame with the same index, joined causally.
    """
    _ensure_ohlcv(df)
    parts = [
        returns_and_log_returns(df),
        ema_features(df),
        rsi_features(df),
        macd_features(df),
        bollinger_features(df),
        atr_features(df),
        volume_features(df),
        vwap_features(df),
        realized_volatility(df),
        distance_features(df),
    ]
    return pd.concat(parts, axis=1)
