"""Strategies the backtester can drive — the ML model + honest baselines.

Each strategy is a callable that takes a row (the latest bar's features +
optionally the model's per-bar predictions) and returns a signal dict::

    {"side": +1|-1|0, "confidence": float, "reason": str}

side 0 means "no trade". The engine ignores trades when the broker can't
open (already in a position, daily-loss reached, etc.).

Baselines exist per the spec's "honest comparator" requirement:
* majority_class — always pick the most common label class (no trades)
* random         — coin-flip side
* buy_and_hold   — long once at the start, never exits
* ema_trend      — long when EMA9 > EMA21, short when below
* rsi_macd       — RSI mean-reversion gated by MACD trend
* no_trade       — flat forever (the floor)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class StrategySignal:
    side: int
    confidence: float
    reason: str


def model_signal(
    *,
    direction_probs: np.ndarray,
    trade_quality_prob: float,
    long_threshold: float = 0.58,
    short_threshold: float = 0.58,
    quality_threshold: float = 0.60,
    no_trade_threshold: float = 0.58,
) -> StrategySignal:
    """Translate per-bar model outputs into a trade signal."""
    down_p, side_p, up_p = float(direction_probs[0]), float(direction_probs[1]), float(direction_probs[2])
    top = max(down_p, side_p, up_p)
    if top < no_trade_threshold:
        return StrategySignal(0, top, "confidence below no_trade threshold")
    if up_p >= long_threshold and trade_quality_prob >= quality_threshold:
        return StrategySignal(+1, up_p, "long_bias")
    if down_p >= short_threshold and trade_quality_prob >= quality_threshold:
        return StrategySignal(-1, down_p, "short_bias")
    return StrategySignal(0, top, "quality threshold not met")


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def buy_and_hold_signal(bar_index: int) -> StrategySignal:
    if bar_index == 0:
        return StrategySignal(+1, 1.0, "buy_and_hold initial entry")
    return StrategySignal(0, 0.0, "buy_and_hold hold")


def majority_class_signal() -> StrategySignal:
    return StrategySignal(0, 0.0, "majority_class baseline never trades")


def ema_trend_signal(row: pd.Series) -> StrategySignal:
    e9 = row.get("ema_9")
    e21 = row.get("ema_21")
    if pd.isna(e9) or pd.isna(e21):
        return StrategySignal(0, 0.0, "ema unavailable")
    if e9 > e21:
        return StrategySignal(+1, 0.6, "ema9 > ema21")
    if e9 < e21:
        return StrategySignal(-1, 0.6, "ema9 < ema21")
    return StrategySignal(0, 0.0, "ema flat")


def rsi_macd_signal(row: pd.Series) -> StrategySignal:
    rsi = row.get("rsi_14")
    macd_h = row.get("macd_hist")
    if pd.isna(rsi) or pd.isna(macd_h):
        return StrategySignal(0, 0.0, "rsi/macd unavailable")
    if rsi < 30 and macd_h > 0:
        return StrategySignal(+1, 0.6, "rsi oversold + macd bullish")
    if rsi > 70 and macd_h < 0:
        return StrategySignal(-1, 0.6, "rsi overbought + macd bearish")
    return StrategySignal(0, 0.0, "no rsi/macd setup")


def random_signal(rng: np.random.Generator) -> StrategySignal:
    coin = rng.random()
    if coin < 0.33:
        return StrategySignal(+1, 0.33, "random long")
    if coin < 0.66:
        return StrategySignal(-1, 0.33, "random short")
    return StrategySignal(0, 0.34, "random flat")


def no_trade_signal() -> StrategySignal:
    return StrategySignal(0, 0.0, "no_trade baseline")
