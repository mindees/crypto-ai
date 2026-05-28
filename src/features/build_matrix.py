"""Orchestrator: build the (symbol, timeframe) feature matrix from all sources.

Joins, in causal fashion:

* OHLCV (Phase 1 binance bulk parquet) — the row index
* Technical indicators ([[features.indicators]])
* Market structure / SMC proxies ([[features.structure]])
* Candle patterns ([[features.patterns]])
* Derivatives flow ([[features.flow]])
* Sentiment ([[features.sentiment]])
* On-chain ([[features.onchain]])
* Macro ([[features.macro]])
* Cycle anchor: months since the last BTC halving

Output: ``data/features/source=binance/market_type=<m>/symbol=<s>/timeframe=<t>/features.parquet``
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.features.flow import build_flow_features
from src.features.indicators import compute_all_indicators
from src.features.macro import build_macro_features
from src.features.onchain import build_onchain_features
from src.features.patterns import compute_all_patterns
from src.features.sentiment import build_sentiment_features
from src.features.structure import compute_all_structure
from src.utils.io import repo_root
from src.utils.logging import get_logger

_log = get_logger("features.build_matrix")


def _load_ohlcv(
    symbol: str, timeframe: str, *, market: str, root: Path
) -> pd.DataFrame | None:
    path = (
        root / "data" / "processed" / "ohlcv"
        / f"source=binance" / f"market_type={market}"
        / f"symbol={symbol}" / f"timeframe={timeframe}" / "data.parquet"
    )
    if not path.exists():
        _log.warning("missing OHLCV parquet: %s", path)
        return None
    df = pq.read_table(path).to_pandas()
    if "open_time" not in df.columns:
        return None
    df["timestamp_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp_utc").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def _load_dominance(target_index: pd.DatetimeIndex, root: Path) -> pd.DataFrame:
    path = root / "data" / "processed" / "coingecko" / "global_dominance.parquet"
    if not path.exists():
        return pd.DataFrame(index=target_index)
    df = pq.read_table(path).to_pandas()
    if "timestamp_utc" not in df.columns:
        return pd.DataFrame(index=target_index)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc").set_index("timestamp_utc")
    df = df[~df.index.duplicated(keep="last")]
    keep = [c for c in ("btc_dominance_pct", "eth_dominance_pct",
                        "total_market_cap_usd", "total_volume_usd_24h") if c in df.columns]
    if not keep:
        return pd.DataFrame(index=target_index)
    target_df = pd.DataFrame(index=target_index).sort_index()
    target_df["__key"] = target_df.index
    src = df[keep].reset_index().rename(columns={"timestamp_utc": "__key"})
    merged = pd.merge_asof(
        target_df.reset_index(drop=True).sort_values("__key"),
        src.sort_values("__key"),
        on="__key", direction="backward",
    )
    merged.index = target_df.index
    return merged.drop(columns="__key")


def _cycle_anchor(target_index: pd.DatetimeIndex, root: Path) -> pd.DataFrame:
    path = root / "reference" / "halvings.csv"
    halvings = [pd.Timestamp(d, tz="UTC") for d in pd.read_csv(path)["date_utc"].tolist()]
    months = np.full(len(target_index), np.nan)
    for i, ts in enumerate(target_index):
        prev = [h for h in halvings if h <= ts]
        if prev:
            months[i] = (ts - prev[-1]).days / 30.4375
    return pd.DataFrame({"cycle_months_since_halving": months}, index=target_index)


def build_feature_matrix(
    symbol: str,
    timeframe: str,
    *,
    market: str = "spot",
    sample: bool = False,
    root: Path | None = None,
) -> pd.DataFrame | None:
    root = root or repo_root()
    ohlcv = _load_ohlcv(symbol, timeframe, market=market, root=root)
    if ohlcv is None:
        return None
    if sample:
        ohlcv = ohlcv.tail(500)

    parts = [
        ohlcv,
        compute_all_indicators(ohlcv),
        compute_all_structure(ohlcv),
        compute_all_patterns(ohlcv),
        build_sentiment_features(ohlcv.index, root=root),
        build_onchain_features(ohlcv.index, symbol, root=root),
        build_macro_features(ohlcv.index, root=root),
        _load_dominance(ohlcv.index, root),
        _cycle_anchor(ohlcv.index, root),
    ]

    # Derivatives features only meaningful for the futures market; for spot
    # frames we still try because funding/OI provide market-wide context
    # (Binance has only USDT-M derivatives, no spot derivatives).
    spot_ohlcv = ohlcv if market == "spot" else _load_ohlcv(symbol, timeframe, market="spot", root=root)
    parts.append(build_flow_features(ohlcv, symbol, root=root, spot_ohlcv=spot_ohlcv))

    feature_df = pd.concat(parts, axis=1)
    # De-duplicate column names if any module collided (keep first)
    feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]
    return feature_df


def write_feature_matrix(
    feature_df: pd.DataFrame, symbol: str, timeframe: str, *, market: str, root: Path,
) -> Path:
    out = (
        root / "data" / "features"
        / f"source=binance" / f"market_type={market}"
        / f"symbol={symbol}" / f"timeframe={timeframe}" / "features.parquet"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    table = feature_df.reset_index().rename(columns={"index": "timestamp_utc"})
    table.to_parquet(out, index=False)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.features.build_matrix")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--timeframes", nargs="+", required=True)
    p.add_argument("--market", default="spot")
    p.add_argument("--sample", type=lambda s: s.strip().lower() in {"1", "true", "yes", "y", "t"},
                   default=False)
    args = p.parse_args(argv)

    root = repo_root()
    print(f"{'symbol':<9} {'tf':<4} {'rows':>6} {'cols':>5}  feature_matrix")
    n_written = 0
    for symbol in args.symbols:
        for tf in args.timeframes:
            df = build_feature_matrix(symbol, tf, market=args.market, sample=args.sample, root=root)
            if df is None or df.empty:
                print(f"{symbol:<9} {tf:<4}    -    -   (no OHLCV; skipped)")
                continue
            out = write_feature_matrix(df, symbol, tf, market=args.market, root=root)
            n_written += 1
            print(f"{symbol:<9} {tf:<4} {len(df):>6} {df.shape[1]:>5}  {out.relative_to(root)}")
    return 0 if n_written else 2


if __name__ == "__main__":
    sys.exit(main())
