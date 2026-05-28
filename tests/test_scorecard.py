"""Scorecard tests.

The spec is emphatic: missing data must be reported as the literal string
``"unavailable"``, never guessed, never zero, never the last known value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.scorecard import (
    UNAVAILABLE,
    buy_checklist,
    scorecard_for_row,
    trade_checklist,
)


def test_unavailable_when_all_inputs_missing():
    row = {}
    out = trade_checklist(row)
    # Every value-bearing field should be UNAVAILABLE when nothing is supplied.
    for key in (
        "trend_direction", "rsi_state", "macd_state", "bollinger_state",
        "funding_state", "fear_greed_state", "ema_120_cycle",
    ):
        assert out[key] == UNAVAILABLE, f"{key} should be unavailable when input is missing"


def test_unavailable_for_nan_inputs():
    row = {
        "rsi_14": float("nan"),
        "macd_hist": float("nan"),
        "fear_greed": float("nan"),
        "ema_stack_score": float("nan"),
    }
    out = trade_checklist(row)
    assert out["rsi_14"] == UNAVAILABLE
    assert out["macd_state"] == UNAVAILABLE
    assert out["fear_greed_state"] == UNAVAILABLE
    assert out["trend_direction"] == UNAVAILABLE


def test_states_categorize_correctly_when_present():
    row = {
        "rsi_14": 75.0,
        "macd_hist": 1.2,
        "ema_stack_score": 4.0,
        "fear_greed": 85.0,
        "funding_rate": 0.0005,
        "bb_mean_reversion_signal": -1.0,
        "ema_120_cycle_signal": 1.0,
        "market_structure_score": 2.0,
        "funding_oi_governor_risk": 1.0,
        "close": 45000.0,
        "atr": 500.0,
        "atr_pct": 0.011,
    }
    out = trade_checklist(row)
    assert out["rsi_state"] == "overbought"
    assert out["macd_state"] == "bullish_histogram_rising"
    assert out["trend_direction"] in ("up", "strong_up")
    assert out["fear_greed_state"] == "extreme_greed"
    assert out["funding_state"] == "very_positive"
    assert out["bollinger_state"] == "above_upper"
    assert out["funding_oi_governor"] == "elevated"


def test_scorecard_round_trip_with_pandas_series():
    s = pd.Series({
        "rsi_14": 40.0,
        "macd_hist": -0.5,
        "ema_stack_score": -3.0,
        "fear_greed": 10.0,
        "funding_rate": -0.0005,
        "close": 30000.0,
    })
    out = scorecard_for_row(s)
    assert "trade_checklist" in out
    assert "buy_checklist" in out
    assert out["trade_checklist"]["rsi_state"] == "below_mid"
    assert out["trade_checklist"]["fear_greed_state"] == "extreme_fear"
    assert out["trade_checklist"]["trend_direction"] == "strong_down"


def test_buy_checklist_includes_risk_disclaimer():
    out = buy_checklist({})
    assert "risk_disclaimer" in out
    assert "Not financial advice" in out["risk_disclaimer"]


def test_buy_checklist_dca_logic():
    out_high = buy_checklist({"fear_greed": 80.0})
    out_low = buy_checklist({"fear_greed": 10.0})
    out_na = buy_checklist({})
    assert out_high["dca_suggested"] is True
    assert out_low["dca_suggested"] is False
    assert out_na["dca_suggested"] == UNAVAILABLE
