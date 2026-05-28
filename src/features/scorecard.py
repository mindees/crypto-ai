"""Transparent rule-derived scorecard.

This is **separate from the ML model**. It produces a deterministic, fully
inspectable assessment of the latest bar using the features already
computed by the matrix builder. The spec requires that missing data is
reported as the literal string ``"unavailable"`` — not guessed, not zero,
not the last known value.

The scorecard powers the prediction-JSON ``scorecard`` field and the alert
payloads. It is intentionally readable, not optimized for speed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

UNAVAILABLE = "unavailable"


def _get(row: pd.Series | dict, key: str, default: Any = UNAVAILABLE) -> Any:
    val = row.get(key) if hasattr(row, "get") else (row[key] if key in row else default)
    if val is None:
        return default
    if isinstance(val, float) and np.isnan(val):
        return default
    return val


def _classify(value: Any, thresholds: list[tuple[float, str]]) -> str:
    """Translate a numeric value into a label given (threshold, label) pairs sorted ascending."""
    if value is UNAVAILABLE or not isinstance(value, (int, float)):
        return UNAVAILABLE
    for thresh, label in thresholds:
        if value <= thresh:
            return label
    return thresholds[-1][1]


def trade_checklist(row: pd.Series | dict) -> dict[str, Any]:
    """Per the spec's "BTC/ETH Futures Trade Checklist" — every field present."""
    ema_stack = _get(row, "ema_stack_score")
    ema120 = _get(row, "ema_120_cycle_signal")
    rsi_v = _get(row, "rsi_14")
    macd_h = _get(row, "macd_hist")
    bb_state = _get(row, "bb_mean_reversion_signal")
    structure = _get(row, "market_structure_score")
    funding = _get(row, "funding_rate")
    oi_change = _get(row, "open_interest_change_pct")
    governor = _get(row, "funding_oi_governor_risk")
    taker_delta = _get(row, "taker_delta_proxy")
    cvd = _get(row, "cvd_proxy")
    basis = _get(row, "basis_futures_vs_spot")
    long_short_ratio = _get(row, "deriv_global_long_short_ratio")
    fear_greed = _get(row, "fear_greed")
    btc_dominance = _get(row, "btc_dominance_pct", UNAVAILABLE)
    atr_ = _get(row, "atr")
    close = _get(row, "close")

    return {
        "market_structure_score": structure,
        "structure_state": _classify(structure, [
            (-1.5, "lower_lows_lower_highs"), (-0.5, "weak_bear"),
            (0.5, "neutral"), (1.5, "weak_bull"), (2.0, "higher_highs_higher_lows"),
        ]),
        "trend_direction": _classify(ema_stack, [
            (-3, "strong_down"), (-1, "down"), (1, "neutral"), (3, "up"), (4, "strong_up"),
        ]),
        "ema_stack_score": ema_stack,
        "ema_120_cycle": _classify(ema120, [(-0.5, "below"), (0.5, "neutral"), (1.0, "above")]),
        "rsi_14": rsi_v,
        "rsi_state": _classify(rsi_v, [(30, "oversold"), (50, "below_mid"), (70, "above_mid"), (100, "overbought")]),
        "macd_hist": macd_h,
        "macd_state": (
            "bullish_histogram_rising" if isinstance(macd_h, (int, float)) and macd_h > 0
            else "bearish_histogram_falling" if isinstance(macd_h, (int, float)) and macd_h < 0
            else UNAVAILABLE
        ),
        "bollinger_state": _classify(bb_state, [(-0.5, "above_upper"), (0.5, "neutral"), (1.0, "below_lower")]),
        "funding_rate": funding,
        "funding_state": _classify(funding, [
            (-0.0003, "very_negative"), (-0.0001, "slightly_negative"),
            (0.0001, "neutral"), (0.0003, "slightly_positive"), (1.0, "very_positive"),
        ]),
        "open_interest_change_pct": oi_change,
        "funding_oi_governor": "elevated" if governor == 1.0 else "normal" if governor == 0.0 else UNAVAILABLE,
        "taker_delta_proxy": taker_delta,
        "cvd_proxy": cvd,
        "basis_futures_vs_spot": basis,
        "long_short_ratio": long_short_ratio,
        "fear_greed": fear_greed,
        "fear_greed_state": _classify(fear_greed, [
            (20, "extreme_fear"), (40, "fear"), (60, "neutral"), (80, "greed"), (100, "extreme_greed"),
        ]),
        "btc_dominance": btc_dominance,
        "atr": atr_,
        "atr_pct": _get(row, "atr_pct"),
        "current_price": close,
        # Suggested exits computed in the predict layer; here we expose only the components.
        "min_rr_target": 2.0,
        "max_risk_per_trade_pct": 1.0,
        "max_daily_loss_pct": 4.0,
    }


def buy_checklist(row: pd.Series | dict) -> dict[str, Any]:
    """Per the spec's "BTC/ETH Buy Checklist"."""
    close = _get(row, "close")
    return {
        "current_price": close,
        "distance_from_ath_pct": _get(row, "distance_from_ath_pct"),
        "ema_120_cycle_state": _classify(_get(row, "ema_120_cycle_signal"),
                                          [(-0.5, "below"), (0.5, "neutral"), (1.0, "above")]),
        "sma_50": _get(row, "sma_50"),
        "sma_200": _get(row, "sma_200"),
        "rsi_14": _get(row, "rsi_14"),
        "macd_hist": _get(row, "macd_hist"),
        "trend_direction": _classify(_get(row, "ema_stack_score"),
                                      [(-3, "strong_down"), (-1, "down"), (1, "neutral"), (3, "up"), (4, "strong_up")]),
        "fear_greed": _get(row, "fear_greed"),
        "btc_active_addresses": _get(row, "btc_active_addresses"),
        "btc_hash_rate": _get(row, "btc_hash_rate"),
        "btc_miner_revenue_usd": _get(row, "btc_miner_revenue_usd"),
        # DCA-vs-lump-sum is a rule layered above the raw scorecard
        "dca_suggested": (
            True if isinstance(_get(row, "fear_greed"), (int, float)) and _get(row, "fear_greed") > 60
            else False if isinstance(_get(row, "fear_greed"), (int, float)) else UNAVAILABLE
        ),
        "risk_disclaimer": "Decision-support only. Not financial advice. Validate manually before trading.",
    }


def scorecard_for_row(row: pd.Series | dict) -> dict[str, Any]:
    return {
        "trade_checklist": trade_checklist(row),
        "buy_checklist": buy_checklist(row),
    }


def scorecard_for_latest(features_df: pd.DataFrame) -> dict[str, Any]:
    if features_df.empty:
        return {"error": "empty features dataframe"}
    return scorecard_for_row(features_df.iloc[-1])
