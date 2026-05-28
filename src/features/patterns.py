"""Candlestick pattern features (causal, deterministic).

These are simplified pandas-native pattern recognizers. Each returns 0/1
flags. When TA-Lib is available a future revision can swap in CDL functions
behind the same callables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def _range(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - df["low"]).replace(0, np.nan)


def _upper_wick(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)


def _lower_wick(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]


def candle_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    rng = _range(df)
    body = _body(df)
    up_w = _upper_wick(df)
    lo_w = _lower_wick(df)
    return pd.DataFrame({
        "candle_body_pct": body / rng,
        "candle_upper_wick_pct": up_w / rng,
        "candle_lower_wick_pct": lo_w / rng,
        "candle_bullish": (df["close"] > df["open"]).astype("float64"),
        "candle_bearish": (df["close"] < df["open"]).astype("float64"),
    }, index=df.index)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

def doji_flag(df: pd.DataFrame, body_threshold: float = 0.1) -> pd.Series:
    body = _body(df)
    rng = _range(df)
    return ((body / rng) < body_threshold).astype("float64")


def hammer_flag(df: pd.DataFrame) -> pd.Series:
    body = _body(df)
    rng = _range(df)
    lower = _lower_wick(df)
    upper = _upper_wick(df)
    return (
        (lower >= 2 * body) & (upper <= body) & (body / rng < 0.4)
    ).astype("float64")


def shooting_star_flag(df: pd.DataFrame) -> pd.Series:
    body = _body(df)
    rng = _range(df)
    lower = _lower_wick(df)
    upper = _upper_wick(df)
    return (
        (upper >= 2 * body) & (lower <= body) & (body / rng < 0.4)
    ).astype("float64")


def bullish_engulfing_flag(df: pd.DataFrame) -> pd.Series:
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    return (
        (prev_close < prev_open)              # prior bearish
        & (df["close"] > df["open"])          # current bullish
        & (df["open"] <= prev_close)
        & (df["close"] >= prev_open)
    ).astype("float64")


def bearish_engulfing_flag(df: pd.DataFrame) -> pd.Series:
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    return (
        (prev_close > prev_open)              # prior bullish
        & (df["close"] < df["open"])          # current bearish
        & (df["open"] >= prev_close)
        & (df["close"] <= prev_open)
    ).astype("float64")


def inside_bar_flag(df: pd.DataFrame) -> pd.Series:
    return (
        (df["high"] <= df["high"].shift(1))
        & (df["low"] >= df["low"].shift(1))
    ).astype("float64")


def outside_bar_flag(df: pd.DataFrame) -> pd.Series:
    return (
        (df["high"] > df["high"].shift(1))
        & (df["low"] < df["low"].shift(1))
    ).astype("float64")


def pin_bar_flag(df: pd.DataFrame) -> pd.Series:
    """Pin bar = long wick on ONE side (≥2× body) AND small opposite wick."""
    body = _body(df)
    lower = _lower_wick(df)
    upper = _upper_wick(df)
    bull_pin = (lower >= 2 * body) & (upper <= 0.5 * body)
    bear_pin = (upper >= 2 * body) & (lower <= 0.5 * body)
    return (bull_pin | bear_pin).astype("float64")


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def compute_all_patterns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "doji": doji_flag(df),
        "hammer": hammer_flag(df),
        "shooting_star": shooting_star_flag(df),
        "bullish_engulfing": bullish_engulfing_flag(df),
        "bearish_engulfing": bearish_engulfing_flag(df),
        "inside_bar": inside_bar_flag(df),
        "outside_bar": outside_bar_flag(df),
        "pin_bar": pin_bar_flag(df),
    }, index=df.index)
    return pd.concat([candle_anatomy(df), out], axis=1)
