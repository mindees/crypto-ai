"""Causal market-structure features.

Swing detection here uses a fixed-window "fractal" rule: a bar is a swing
high if its high is the maximum over a window of N bars on each side. To
preserve causality we evaluate swings on the **left side only** — the bar
becomes a confirmed swing N bars after the fact. Tests check that today's
features never reference tomorrow's data.

For higher-level structure (Wyckoff phase, ICT order blocks, SMC FVGs) we
emit clearly-labelled simplified proxies — the spec explicitly says these
must be marked as proxies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Swings via causal left-window fractals
# ---------------------------------------------------------------------------

def _causal_swing_high(high: pd.Series, window: int) -> pd.Series:
    """1 where bar t's high is >= the previous ``window`` bars' highs.

    Causal: uses only data on or before t. A "swing" by this rule is a local
    maximum confirmed at the moment it occurs, not on a future-confirmed pivot.
    """
    rolling_max = high.shift(1).rolling(window, min_periods=window).max()
    return ((high > rolling_max) & rolling_max.notna()).astype("float64")


def _causal_swing_low(low: pd.Series, window: int) -> pd.Series:
    rolling_min = low.shift(1).rolling(window, min_periods=window).min()
    return ((low < rolling_min) & rolling_min.notna()).astype("float64")


def swing_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    sh = _causal_swing_high(high, window)
    sl = _causal_swing_low(low, window)

    # Track the most recent swing high/low value as a forward-fill — causal
    # because the swing's value only enters the series once the bar prints.
    last_sh_value = high.where(sh > 0).ffill()
    last_sl_value = low.where(sl > 0).ffill()

    # Higher-high / lower-low flags compare consecutive swings.
    sh_values_only = high.where(sh > 0).dropna()
    sl_values_only = low.where(sl > 0).dropna()
    hh_flag = pd.Series(0.0, index=df.index)
    lh_flag = pd.Series(0.0, index=df.index)
    hl_flag = pd.Series(0.0, index=df.index)
    ll_flag = pd.Series(0.0, index=df.index)

    prev = None
    for ts, val in sh_values_only.items():
        if prev is not None:
            if val > prev:
                hh_flag.loc[ts] = 1.0
            elif val < prev:
                lh_flag.loc[ts] = 1.0
        prev = val
    prev = None
    for ts, val in sl_values_only.items():
        if prev is not None:
            if val > prev:
                hl_flag.loc[ts] = 1.0
            elif val < prev:
                ll_flag.loc[ts] = 1.0
        prev = val

    # Forward-fill the most recent regime-tag (-1 bear, +1 bull, 0 neutral).
    hh_running = hh_flag.replace(0.0, np.nan).ffill()
    ll_running = ll_flag.replace(0.0, np.nan).ffill()
    lh_running = lh_flag.replace(0.0, np.nan).ffill()
    hl_running = hl_flag.replace(0.0, np.nan).ffill()

    structure_score = pd.Series(0.0, index=df.index)
    structure_score = structure_score + hh_running.fillna(0.0) + hl_running.fillna(0.0)
    structure_score = structure_score - lh_running.fillna(0.0) - ll_running.fillna(0.0)
    structure_score = structure_score.clip(-2.0, 2.0)

    # Distance to last swing levels
    dist_to_sh = (df["close"] / last_sh_value - 1.0)
    dist_to_sl = (df["close"] / last_sl_value - 1.0)

    return pd.DataFrame({
        "swing_high_flag": sh,
        "swing_low_flag": sl,
        "swing_hh_flag": hh_flag,
        "swing_lh_flag": lh_flag,
        "swing_hl_flag": hl_flag,
        "swing_ll_flag": ll_flag,
        "last_swing_high": last_sh_value,
        "last_swing_low": last_sl_value,
        "distance_to_last_swing_high": dist_to_sh,
        "distance_to_last_swing_low": dist_to_sl,
        "market_structure_score": structure_score,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Range / breakout / liquidity sweep proxies
# ---------------------------------------------------------------------------

def range_breakout_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Recent range computed from PRIOR bars only (shift 1) — causal.
    rolling_high = high.shift(1).rolling(window, min_periods=window).max()
    rolling_low = low.shift(1).rolling(window, min_periods=window).min()
    range_size = rolling_high - rolling_low
    range_pct = range_size / close.replace(0, np.nan)

    breakout_up = (close > rolling_high).astype("float64")
    breakout_down = (close < rolling_low).astype("float64")
    inside_range = ((close >= rolling_low) & (close <= rolling_high)).astype("float64")

    # Liquidity-sweep proxy: wick pierces prior range high/low then closes back inside.
    swept_up = ((high > rolling_high) & (close < rolling_high)).astype("float64")
    swept_down = ((low < rolling_low) & (close > rolling_low)).astype("float64")

    return pd.DataFrame({
        f"range_{window}_high": rolling_high,
        f"range_{window}_low": rolling_low,
        f"range_{window}_pct": range_pct,
        f"range_{window}_breakout_up": breakout_up,
        f"range_{window}_breakout_down": breakout_down,
        f"range_{window}_inside": inside_range,
        f"range_{window}_sweep_up": swept_up,
        f"range_{window}_sweep_down": swept_down,
    }, index=df.index)


# ---------------------------------------------------------------------------
# SMC / ICT / Wyckoff PROXIES — simplified and clearly marked
# ---------------------------------------------------------------------------

def smc_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simplified SMC/ICT proxies. Not a full implementation.

    * FVG (fair value gap): bar t has FVG-up if low[t] > high[t-2].
    * Order-block proxy: a bullish OB is approximated by the most recent
      bearish candle preceding a strong upward move (3-bar close shift).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    open_ = df["open"]

    fvg_up = ((low > high.shift(2)) & high.shift(2).notna()).astype("float64")
    fvg_down = ((high < low.shift(2)) & low.shift(2).notna()).astype("float64")

    # Order-block proxy: bearish candle (close<open) whose NEXT 3-bar return is +.
    # Causal note: the OB flag is set on bar t once bar t+3 is known, so we
    # only emit it at t+3 (shifted) to keep things strictly causal.
    fwd3_ret = close.pct_change(3)
    bearish_bar = (close < open_)
    raw_ob_up = (bearish_bar & (fwd3_ret > 0.02))
    ob_up_proxy = raw_ob_up.shift(3).fillna(False).astype("float64")
    bullish_bar = (close > open_)
    raw_ob_dn = (bullish_bar & (fwd3_ret < -0.02))
    ob_down_proxy = raw_ob_dn.shift(3).fillna(False).astype("float64")

    # CHoCH (Change of Character) proxy: a swing low (5-window) where the
    # subsequent close breaks the prior swing high. Causal via shift.
    sh = _causal_swing_high(high, 5)
    sl = _causal_swing_low(low, 5)
    last_sh = high.where(sh > 0).ffill()
    last_sl = low.where(sl > 0).ffill()
    choch_up = ((close > last_sh.shift(1)) & last_sl.shift(1).notna()).astype("float64")
    choch_down = ((close < last_sl.shift(1)) & last_sh.shift(1).notna()).astype("float64")

    return pd.DataFrame({
        "smc_fvg_up_proxy": fvg_up,
        "smc_fvg_down_proxy": fvg_down,
        "smc_order_block_up_proxy": ob_up_proxy,
        "smc_order_block_down_proxy": ob_down_proxy,
        "smc_choch_up_proxy": choch_up,
        "smc_choch_down_proxy": choch_down,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def compute_all_structure(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        swing_features(df, window=10),
        range_breakout_features(df, window=20),
        range_breakout_features(df, window=50),
        smc_proxy_features(df),
    ], axis=1)
