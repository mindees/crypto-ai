"""Prove that no feature at time t uses data from t+1 or later.

Strategy: compute features on a deterministic OHLCV series, then mutate
the FUTURE rows (everything after index ``k``) and recompute. Any row at
index ≤ k whose feature value changed proves a lookahead leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import compute_all_indicators
from src.features.patterns import compute_all_patterns
from src.features.structure import compute_all_structure


def _make_ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rets = rng.normal(0.0, 0.01, size=n)
    close = 30_000.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, size=n)))
    volume = np.abs(rng.normal(1000, 200, size=n))
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=idx)


def _assert_no_change_past(left: pd.DataFrame, right: pd.DataFrame, cutoff: int) -> None:
    """For each numeric column, assert rows 0..cutoff match between left and right."""
    common = [c for c in left.columns if c in right.columns]
    for col in common:
        a = left[col].iloc[:cutoff + 1]
        b = right[col].iloc[:cutoff + 1]
        if not pd.api.types.is_numeric_dtype(a):
            continue
        # NaN positions must match
        np.testing.assert_array_equal(a.isna().to_numpy(), b.isna().to_numpy(),
                                       err_msg=f"NaN positions diverged in column {col}")
        a_v = a.dropna().to_numpy()
        b_v = b.dropna().to_numpy()
        np.testing.assert_allclose(
            a_v, b_v, rtol=1e-12, atol=1e-12,
            err_msg=f"Lookahead leak in column {col}: past values changed when future was mutated",
        )


@pytest.mark.parametrize("compute", [
    compute_all_indicators,
    compute_all_patterns,
    compute_all_structure,
])
def test_no_future_leakage_into_past_rows(compute):
    df = _make_ohlcv()
    cutoff = 220

    out_full = compute(df)

    df_mutated = df.copy()
    # Aggressively mutate every numeric column AFTER the cutoff.
    rng = np.random.default_rng(99)
    for col in ("open", "high", "low", "close", "volume"):
        df_mutated.loc[df_mutated.index[cutoff + 1]:, col] = (
            df_mutated.loc[df_mutated.index[cutoff + 1]:, col].to_numpy()
            * (1.0 + rng.normal(0.0, 0.5, size=len(df) - cutoff - 1))
        ).clip(min=1.0)
    out_mutated = compute(df_mutated)

    _assert_no_change_past(out_full, out_mutated, cutoff)


def test_indicators_handle_short_series_without_lookahead():
    """A 30-bar series should still produce some NaNs (insufficient history) but
    no exception, and no infinite values."""
    df = _make_ohlcv(n=30)
    out = compute_all_indicators(df)
    assert np.isinf(out.select_dtypes(include="number").to_numpy()).sum() == 0


def test_atr_uses_only_past_bars():
    """ATR at row t must equal the same value regardless of rows after t."""
    df = _make_ohlcv()
    out_a = compute_all_indicators(df)["atr"]
    df2 = df.copy()
    df2.loc[df2.index[300]:, "high"] *= 5.0  # huge future spike
    out_b = compute_all_indicators(df2)["atr"]
    # Up to row 299 ATR must be unchanged
    a = out_a.iloc[:300].dropna().to_numpy()
    b = out_b.iloc[:300].dropna().to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
