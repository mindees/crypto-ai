"""Macro feature transformations.

Reads any available yfinance/FRED parquet outputs from Phase 2 and joins
them causally onto a target index. When Yahoo Finance was rate-limiting at
Phase-2 time the source files won't exist — in that case this module emits
an empty DataFrame (the matrix builder tolerates missing columns).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.utils.io import repo_root
from src.utils.logging import get_logger

_log = get_logger("features.macro")

KNOWN_SERIES = (
    "sp500_close",
    "nasdaq_close",
    "dxy_close",
    "vix_close",
    "fed_funds_rate_pct",
    "cpi_all_urban",
)


def _load_series(name: str, root: Path) -> pd.DataFrame | None:
    path = root / "data" / "processed" / "macro" / f"{name}.parquet"
    if not path.exists():
        return None
    df = pq.read_table(path).to_pandas()
    if "timestamp_utc" not in df.columns or "value" not in df.columns:
        return None
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    df = df.set_index("timestamp_utc")
    return df[~df.index.duplicated(keep="last")]


def _causal_join_series(target_index: pd.DatetimeIndex, src: pd.DataFrame) -> pd.Series:
    target_df = pd.DataFrame(index=target_index).sort_index()
    target_df["__key"] = target_df.index
    src_reset = src[["value"]].reset_index().rename(columns={"timestamp_utc": "__key"})
    merged = pd.merge_asof(
        target_df.reset_index(drop=True).sort_values("__key"),
        src_reset.sort_values("__key"),
        on="__key", direction="backward",
    )
    return pd.Series(merged["value"].to_numpy(), index=target_df.index)


def build_macro_features(target_index: pd.DatetimeIndex, *, root: Path | None = None) -> pd.DataFrame:
    root = root or repo_root()
    out = pd.DataFrame(index=target_index)
    for name in KNOWN_SERIES:
        df = _load_series(name, root)
        if df is None or df.empty:
            continue
        s = _causal_join_series(target_index, df)
        out[f"macro_{name}"] = s
        # Daily return for equity series (where it makes sense)
        if name in {"sp500_close", "nasdaq_close", "dxy_close", "vix_close"}:
            out[f"macro_{name}_ret"] = s.pct_change()
    return out
