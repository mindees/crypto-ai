"""Label generation: direction (triple-barrier), regime, cycle, trade-quality.

All labels are derived from a (symbol, timeframe) feature matrix or OHLCV
frame. Labels are *targets* and may use forward-looking data — that's
expected and OK. **Features** elsewhere remain strictly causal.

Output schema (one parquet per symbol×timeframe)::

    direction         {0=down, 1=sideways, 2=up}  + 'ambiguous' as -1
    regime            6-class                       (per spec)
    cycle             4-class                       (BTC halving anchored)
    trade_quality     {0=bad_or_no_trade, 1=good_trade}
    barrier_hit_at_ms barrier hit timestamp (or vertical-barrier ts)
    barrier_rr        realized R multiple at hit
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.features.indicators import atr_features, ema_features, realized_volatility
from src.utils.io import read_yaml, repo_root
from src.utils.logging import get_logger

_log = get_logger("labels")


DIRECTION_DOWN = 0
DIRECTION_SIDEWAYS = 1
DIRECTION_UP = 2
DIRECTION_AMBIGUOUS = -1

REGIME_CLASSES = (
    "trending_up", "trending_down", "ranging_low_vol",
    "ranging_high_vol", "breakout", "capitulation",
)
CYCLE_CLASSES = ("accumulation", "bull", "distribution", "bear")


# ---------------------------------------------------------------------------
# Triple-barrier direction labels
# ---------------------------------------------------------------------------

@dataclass
class TripleBarrierConfig:
    atr_multiple: float
    vertical_barrier_bars: int
    fee_bps_per_side: float = 4.0
    slippage_bps_per_side: float = 5.0


def triple_barrier_labels(
    ohlcv: pd.DataFrame,
    *,
    atr_multiple: float,
    vertical_barrier_bars: int,
    atr_series: pd.Series | None = None,
) -> pd.DataFrame:
    """For each bar, scan forward up to ``vertical_barrier_bars`` and label by which
    barrier is hit first based on intrabar high/low path.

    Returns a DataFrame indexed identically to ``ohlcv`` with columns
    ``direction``, ``barrier_hit_offset`` (in bars; 0 if same bar, NaN if no
    label), and ``barrier_rr`` (realized R at the hit).
    """
    n = len(ohlcv)
    close = ohlcv["close"].to_numpy()
    high = ohlcv["high"].to_numpy()
    low = ohlcv["low"].to_numpy()
    if atr_series is None:
        atr_series = atr_features(ohlcv)["atr"]
    atr = atr_series.reindex(ohlcv.index).to_numpy()

    direction = np.full(n, DIRECTION_AMBIGUOUS, dtype=np.int8)
    offset = np.full(n, np.nan)
    rr = np.full(n, np.nan)

    for i in range(n - 1):
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue
        entry = close[i]
        upper = entry + atr_multiple * atr_i
        lower = entry - atr_multiple * atr_i
        max_j = min(i + vertical_barrier_bars, n - 1)

        # Look at j = i+1 ... max_j; intrabar uses (high[j], low[j])
        hit_dir = DIRECTION_SIDEWAYS
        hit_j = max_j
        hit_rr = (close[max_j] - entry) / (atr_multiple * atr_i)

        for j in range(i + 1, max_j + 1):
            up_hit = high[j] >= upper
            dn_hit = low[j] <= lower
            if up_hit and dn_hit:
                # Conservative: mark ambiguous unless the open clearly tilts one way.
                hit_dir = DIRECTION_AMBIGUOUS
                hit_j = j
                hit_rr = 0.0
                break
            if up_hit:
                hit_dir = DIRECTION_UP
                hit_j = j
                hit_rr = 1.0
                break
            if dn_hit:
                hit_dir = DIRECTION_DOWN
                hit_j = j
                hit_rr = -1.0
                break

        direction[i] = hit_dir
        offset[i] = hit_j - i
        rr[i] = hit_rr

    return pd.DataFrame({
        "direction": pd.Series(direction, index=ohlcv.index, dtype="int8"),
        "barrier_hit_offset": pd.Series(offset, index=ohlcv.index),
        "barrier_rr": pd.Series(rr, index=ohlcv.index),
    }, index=ohlcv.index)


# ---------------------------------------------------------------------------
# Regime labels — rule-based
# ---------------------------------------------------------------------------

def regime_labels(ohlcv: pd.DataFrame) -> pd.Series:
    emas = ema_features(ohlcv)
    atr_df = atr_features(ohlcv)
    rv = realized_volatility(ohlcv)["realized_volatility_20"]

    ema_50_slope = emas["ema_50"].pct_change(periods=10)
    ema_stack = emas["ema_stack_score"]
    atr_pct = atr_df["atr_pct"]

    # Percentile thresholds on the realized volatility distribution — uses the
    # whole training window's distribution at training time; in production this
    # would be recomputed on the training fold.
    rv_low = rv.quantile(0.33)
    rv_high = rv.quantile(0.66)
    atr_low = atr_pct.quantile(0.33)
    atr_high = atr_pct.quantile(0.85)

    labels = pd.Series("ranging_low_vol", index=ohlcv.index, dtype="object")

    trending_up = (ema_stack >= 2) & (ema_50_slope > 0)
    trending_dn = (ema_stack <= -2) & (ema_50_slope < 0)
    breakout = (atr_pct > atr_high) & (ema_stack.abs() >= 2)
    ranging_high = (rv > rv_high) & (ema_stack.abs() < 2)
    ranging_low = (rv <= rv_low) & (ema_stack.abs() < 2)

    # Capitulation: massive single-bar drop + high realized vol
    cap = (ohlcv["close"].pct_change() < -0.10) & (rv > rv_high)

    labels.loc[trending_up] = "trending_up"
    labels.loc[trending_dn] = "trending_down"
    labels.loc[ranging_low] = "ranging_low_vol"
    labels.loc[ranging_high] = "ranging_high_vol"
    labels.loc[breakout] = "breakout"
    labels.loc[cap] = "capitulation"
    return labels


# ---------------------------------------------------------------------------
# Cycle labels — BTC halving anchored
# ---------------------------------------------------------------------------

def _load_halving_dates(root: Path) -> list[pd.Timestamp]:
    path = root / "reference" / "halvings.csv"
    df = pd.read_csv(path)
    out = [pd.Timestamp(d, tz="UTC") for d in df["date_utc"].tolist()]
    return out


def cycle_labels(ohlcv: pd.DataFrame, *, root: Path | None = None) -> pd.Series:
    """4-class cycle phase: accumulation, bull, distribution, bear.

    Combines: months since last BTC halving, drawdown from cumulative high,
    position relative to the 200-week MA.
    """
    root = root or repo_root()
    halvings = _load_halving_dates(root)

    months_since_halving = pd.Series(0.0, index=ohlcv.index)
    for i, ts in enumerate(ohlcv.index):
        prev = [h for h in halvings if h <= ts]
        if not prev:
            months_since_halving.iloc[i] = np.nan
        else:
            delta_days = (ts - prev[-1]).days
            months_since_halving.iloc[i] = delta_days / 30.4375

    close = ohlcv["close"]
    cum_max = close.cummax()
    drawdown = close / cum_max - 1.0

    # 200-week MA → 200 bars for weekly, ~1400 for daily, etc. We approximate
    # using rolling close mean over (200 * bars_per_week) but for portability
    # use a long EMA fallback when window > len(df).
    target_window = max(1, int(200 * 7))  # for 1d frames; longer windows degrade naturally
    if target_window >= len(close):
        mma = close.ewm(span=min(target_window, len(close)), min_periods=1).mean()
    else:
        mma = close.rolling(target_window, min_periods=target_window).mean()
    above_200w = (close > mma).astype("int")

    labels = pd.Series("accumulation", index=ohlcv.index, dtype="object")

    bull = (months_since_halving > 4) & (months_since_halving < 22) & (drawdown > -0.3) & (above_200w == 1)
    distribution = (months_since_halving >= 14) & (months_since_halving <= 24) & (drawdown > -0.2) & (above_200w == 1)
    bear = (drawdown < -0.4) & (above_200w == 0)
    accumulation = (drawdown < -0.5) | ((months_since_halving > 30) & (above_200w == 0))

    labels.loc[bull] = "bull"
    labels.loc[distribution] = "distribution"
    labels.loc[bear] = "bear"
    labels.loc[accumulation] = "accumulation"
    return labels


# ---------------------------------------------------------------------------
# Trade-quality labels — binary
# ---------------------------------------------------------------------------

def trade_quality_labels(
    direction_df: pd.DataFrame,
    ohlcv: pd.DataFrame | None = None,
    *,
    atr_series: pd.Series | None = None,
    vertical_barrier_bars: int = 64,
    target_rr: float = 2.0,
    fee_bps_per_side: float = 4.0,
    slippage_bps_per_side: float = 5.0,
) -> pd.Series:
    """1 = the bar's directional target would have reached ≥ ``target_rr`` R
    before hitting a -1R stop, after fees + slippage.

    When ``ohlcv`` is not provided (e.g. lightweight tests with pre-built
    direction frames), we fall back to: directional hit AND net positive
    realized R after costs.
    """
    if ohlcv is None:
        # Cheap fallback for tests that construct a synthetic direction frame.
        cost = 2 * (fee_bps_per_side + slippage_bps_per_side) / 10_000.0
        net_rr = direction_df["barrier_rr"].abs() - cost
        good = (
            direction_df["direction"].isin([DIRECTION_UP, DIRECTION_DOWN])
            & (net_rr > 0)
        )
        return good.astype("int8").rename("trade_quality")

    if atr_series is None:
        atr_series = atr_features(ohlcv)["atr"]
    atr = atr_series.reindex(ohlcv.index).to_numpy()
    high = ohlcv["high"].to_numpy()
    low = ohlcv["low"].to_numpy()
    close = ohlcv["close"].to_numpy()
    direction = direction_df["direction"].to_numpy()
    n = len(ohlcv)
    out = np.zeros(n, dtype=np.int8)
    cost_rr = 2 * (fee_bps_per_side + slippage_bps_per_side) / 10_000.0
    # cost_rr is in PRICE-fraction terms; convert to R by dividing by (atr/entry)
    for i in range(n):
        if direction[i] not in (DIRECTION_UP, DIRECTION_DOWN):
            continue
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue
        entry = close[i]
        r_unit = atr_i
        side = 1 if direction[i] == DIRECTION_UP else -1
        target = entry + side * target_rr * r_unit
        stop = entry - side * 1.0 * r_unit
        end_j = min(i + vertical_barrier_bars, n - 1)
        reached_target = False
        reached_stop = False
        for j in range(i + 1, end_j + 1):
            if side > 0:
                if low[j] <= stop:
                    reached_stop = True
                    break
                if high[j] >= target:
                    reached_target = True
                    break
            else:
                if high[j] >= stop:
                    reached_stop = True
                    break
                if low[j] <= target:
                    reached_target = True
                    break
        if reached_target and not reached_stop:
            # Subtract fee/slippage from the realized R
            net_r = target_rr - (cost_rr * entry / r_unit)
            if net_r > 0:
                out[i] = 1
    return pd.Series(out, index=ohlcv.index, name="trade_quality")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _load_ohlcv_parquet(symbol: str, timeframe: str, *, market: str = "spot", root: Path | None = None) -> pd.DataFrame | None:
    root = root or repo_root()
    path = (
        root / "data" / "processed" / "ohlcv"
        / f"source=binance" / f"market_type={market}"
        / f"symbol={symbol}" / f"timeframe={timeframe}" / "data.parquet"
    )
    if not path.exists():
        return None
    df = pq.read_table(path).to_pandas()
    if "open_time" not in df.columns:
        return None
    df["timestamp_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp_utc").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def label_one_combo(
    symbol: str,
    timeframe: str,
    *,
    market: str = "spot",
    sample: bool = False,
    root: Path | None = None,
) -> pd.DataFrame | None:
    root = root or repo_root()
    cfg = read_yaml(root / "configs" / "config.yaml")
    tb_cfg_root = (cfg.get("labels") or {}).get("triple_barrier") or {}
    tb_cfg = tb_cfg_root.get(timeframe)
    if tb_cfg is None:
        _log.warning("no triple_barrier config for timeframe %s; skipping", timeframe)
        return None

    ohlcv = _load_ohlcv_parquet(symbol, timeframe, market=market, root=root)
    if ohlcv is None:
        _log.warning("no OHLCV parquet for %s/%s; skipping", symbol, timeframe)
        return None
    if sample:
        ohlcv = ohlcv.tail(500)

    tb = triple_barrier_labels(
        ohlcv,
        atr_multiple=float(tb_cfg["atr_multiple"]),
        vertical_barrier_bars=int(tb_cfg["vertical_barrier_bars"]),
    )
    regime = regime_labels(ohlcv)
    cycle = cycle_labels(ohlcv, root=root)
    tq = trade_quality_labels(
        tb, ohlcv=ohlcv,
        vertical_barrier_bars=int(tb_cfg["vertical_barrier_bars"]) * 2,
    )

    out = tb.copy()
    out["regime"] = regime.astype("string")
    out["cycle"] = cycle.astype("string")
    out["trade_quality"] = tq.astype("int8")

    out_path = (
        root / "data" / "labels"
        / f"source=binance" / f"market_type={market}"
        / f"symbol={symbol}" / f"timeframe={timeframe}" / "labels.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = out.reset_index().rename(columns={"index": "timestamp_utc"})
    table.to_parquet(out_path, index=False)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.labels.labeling")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--timeframes", nargs="+", required=True)
    p.add_argument("--market", default="spot")
    p.add_argument("--sample", type=lambda s: s.strip().lower() in {"1", "true", "yes", "y", "t"},
                   default=False)
    args = p.parse_args(argv)

    rows_printed = 0
    print(f"{'symbol':<9} {'tf':<4} {'rows':>6} {'dir_up':>7} {'dir_dn':>7} {'dir_sd':>7} {'dir_amb':>7} "
          f"{'tq_good':>8} {'regime_modes'}")

    for symbol in args.symbols:
        for tf in args.timeframes:
            out = label_one_combo(symbol, tf, sample=args.sample)
            if out is None:
                print(f"{symbol:<9} {tf:<4} (no OHLCV — labels skipped)")
                continue
            d = out["direction"]
            r = out["regime"].value_counts().head(3).to_dict()
            print(
                f"{symbol:<9} {tf:<4} {len(out):>6} "
                f"{(d==DIRECTION_UP).sum():>7} {(d==DIRECTION_DOWN).sum():>7} "
                f"{(d==DIRECTION_SIDEWAYS).sum():>7} {(d==DIRECTION_AMBIGUOUS).sum():>7} "
                f"{out['trade_quality'].sum():>8} {r}"
            )
            rows_printed += 1

    return 0 if rows_printed else 2


if __name__ == "__main__":
    sys.exit(main())
