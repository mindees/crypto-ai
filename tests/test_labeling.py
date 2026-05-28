"""Triple-barrier label tests with hand-crafted price paths."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labels.labeling import (
    DIRECTION_AMBIGUOUS,
    DIRECTION_DOWN,
    DIRECTION_SIDEWAYS,
    DIRECTION_UP,
    triple_barrier_labels,
    trade_quality_labels,
)


def _ohlcv_from_closes(closes: list[float], *, freq: str = "1h", hl_band: float = 0.0) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    close = np.array(closes, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + hl_band
    low = np.minimum(open_, close) - hl_band
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": np.ones(n)},
        index=idx,
    )


def test_triple_barrier_upper_hit_first():
    # Start at 100; ATR-like move = 5 (so atr_multiple=1 with constant atr=5 gives barriers at 95/105).
    # Path: 100 → 102 → 104 → 106 (hits upper at bar 3).
    df = _ohlcv_from_closes([100, 102, 104, 106, 105])
    atr = pd.Series([5.0] * len(df), index=df.index)
    out = triple_barrier_labels(df, atr_multiple=1.0, vertical_barrier_bars=4, atr_series=atr)
    assert out["direction"].iloc[0] == DIRECTION_UP


def test_triple_barrier_lower_hit_first():
    df = _ohlcv_from_closes([100, 98, 96, 94, 96])
    atr = pd.Series([5.0] * len(df), index=df.index)
    out = triple_barrier_labels(df, atr_multiple=1.0, vertical_barrier_bars=4, atr_series=atr)
    assert out["direction"].iloc[0] == DIRECTION_DOWN


def test_triple_barrier_neither_hit_is_sideways():
    df = _ohlcv_from_closes([100, 101, 100, 101, 100])
    atr = pd.Series([5.0] * len(df), index=df.index)
    out = triple_barrier_labels(df, atr_multiple=1.0, vertical_barrier_bars=4, atr_series=atr)
    assert out["direction"].iloc[0] == DIRECTION_SIDEWAYS


def test_triple_barrier_same_bar_both_hit_is_ambiguous():
    # Bar 1: high above upper barrier AND low below lower barrier in the same bar
    df = pd.DataFrame({
        "open":   [100, 100],
        "high":   [101, 110],  # bar 1 high > 105 upper
        "low":    [ 99,  90],  # bar 1 low  < 95 lower
        "close":  [100, 100],
        "volume": [  1,   1],
    }, index=pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"))
    atr = pd.Series([5.0, 5.0], index=df.index)
    out = triple_barrier_labels(df, atr_multiple=1.0, vertical_barrier_bars=3, atr_series=atr)
    assert out["direction"].iloc[0] == DIRECTION_AMBIGUOUS


def test_triple_barrier_skips_bars_without_atr():
    df = _ohlcv_from_closes([100, 101, 102])
    atr = pd.Series([np.nan, np.nan, np.nan], index=df.index)
    out = triple_barrier_labels(df, atr_multiple=1.0, vertical_barrier_bars=2, atr_series=atr)
    # No valid ATR -> all rows stay as DIRECTION_AMBIGUOUS sentinel
    assert (out["direction"] == DIRECTION_AMBIGUOUS).all()


def test_trade_quality_only_true_for_real_directional_hits():
    direction = pd.DataFrame({
        "direction": [DIRECTION_UP, DIRECTION_DOWN, DIRECTION_SIDEWAYS, DIRECTION_AMBIGUOUS],
        "barrier_rr": [1.0, -1.0, 0.0, 0.0],
    })
    tq = trade_quality_labels(direction)
    # The default min_rr_for_good=2.0 with the "relax by 0.5" makes 1.0 acceptable
    # for clearly-directional hits but excludes sideways and ambiguous.
    assert tq.iloc[0] == 1
    assert tq.iloc[1] == 1
    assert tq.iloc[2] == 0
    assert tq.iloc[3] == 0
