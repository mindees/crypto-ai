"""Schema tests for the feature matrix.

* Required columns exist
* No forbidden future-looking columns (e.g. ``label_*``) sneak into the
  feature matrix — labels live in their own parquet.
* Numeric dtypes for indicator outputs
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.indicators import compute_all_indicators
from src.features.patterns import compute_all_patterns
from src.features.structure import compute_all_structure


def _make_ohlcv(n: int = 300, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    rets = rng.normal(0.0, 0.008, size=n)
    close = 30_000.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) * 1.003,
        "low":  np.minimum(open_, close) * 0.997,
        "close": close,
        "volume": np.abs(rng.normal(1000, 200, size=n)),
    }, index=idx)


REQUIRED_INDICATOR_COLUMNS = {
    "returns_1", "log_returns_1",
    "ema_9", "ema_21", "ema_50", "ema_120", "ema_200",
    "ema_stack_score", "ema_120_cycle_signal",
    "rsi_14", "macd_line", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_pct_b", "bb_bandwidth",
    "atr", "atr_pct", "true_range",
    "obv", "vwap_24",
    "realized_volatility", "realized_volatility_20",
    "distance_from_ath_pct",
}

REQUIRED_STRUCTURE_COLUMNS = {
    "swing_high_flag", "swing_low_flag",
    "market_structure_score",
    "range_20_high", "range_20_low",
    "smc_fvg_up_proxy",
}

REQUIRED_PATTERN_COLUMNS = {
    "doji", "hammer", "shooting_star",
    "bullish_engulfing", "bearish_engulfing",
    "inside_bar", "outside_bar",
    "candle_body_pct",
}

# A guardrail: nothing in the feature matrix may resemble a label column.
FORBIDDEN_LABEL_LIKE_PREFIXES = ("label_", "y_", "target_")


def test_indicator_required_columns_present():
    df = _make_ohlcv()
    out = compute_all_indicators(df)
    missing = REQUIRED_INDICATOR_COLUMNS - set(out.columns)
    assert not missing, f"indicators missing columns: {missing}"


def test_structure_required_columns_present():
    df = _make_ohlcv()
    out = compute_all_structure(df)
    missing = REQUIRED_STRUCTURE_COLUMNS - set(out.columns)
    assert not missing, f"structure missing columns: {missing}"


def test_patterns_required_columns_present():
    df = _make_ohlcv()
    out = compute_all_patterns(df)
    missing = REQUIRED_PATTERN_COLUMNS - set(out.columns)
    assert not missing, f"patterns missing columns: {missing}"


def test_no_label_like_columns_in_feature_modules():
    df = _make_ohlcv()
    parts = [
        compute_all_indicators(df),
        compute_all_patterns(df),
        compute_all_structure(df),
    ]
    for part in parts:
        for col in part.columns:
            for bad in FORBIDDEN_LABEL_LIKE_PREFIXES:
                assert not col.startswith(bad), (
                    f"feature column {col!r} looks like a label — labels must live in "
                    "the labeling pipeline, never in the feature matrix."
                )


def test_indicator_outputs_are_numeric_and_finite_where_defined():
    df = _make_ohlcv()
    out = compute_all_indicators(df)
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            arr = out[col].dropna().to_numpy()
            assert not np.isinf(arr).any(), f"infinite values found in {col}"


def test_index_preserved_across_feature_pipeline():
    df = _make_ohlcv()
    out = compute_all_indicators(df)
    pd.testing.assert_index_equal(out.index, df.index)
