"""Sentiment features built from the Fear & Greed parquet (Phase 2 output).

Causally joins F&G daily values onto a target OHLCV index. The reading for
each calendar day is treated as available at 00:00:01 UTC of that day —
F&G is published once per day around midnight UTC, so a bar that closes
later in the same day can safely use that day's reading.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.utils.io import repo_root
from src.utils.logging import get_logger

_log = get_logger("features.sentiment")


def _load_fear_greed(root: Path | None = None) -> pd.DataFrame | None:
    root = root or repo_root()
    path = root / "data" / "processed" / "sentiment" / "fear_greed.parquet"
    if not path.exists():
        return None
    df = pq.read_table(path).to_pandas()
    if "timestamp_utc" not in df.columns:
        return None
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    df = df.set_index("timestamp_utc")
    return df[~df.index.duplicated(keep="last")]


def build_sentiment_features(
    target_index: pd.DatetimeIndex, *, root: Path | None = None
) -> pd.DataFrame:
    fg = _load_fear_greed(root)
    if fg is None or fg.empty:
        return pd.DataFrame(index=target_index)

    src = fg[["fear_greed_value"]].copy()
    src["fear_greed_value"] = pd.to_numeric(src["fear_greed_value"], errors="coerce")

    # merge_asof backward — picks each bar's most recently-published F&G
    target_df = pd.DataFrame(index=target_index).sort_index()
    target_df["__key"] = target_df.index
    src_reset = src.reset_index().rename(columns={"timestamp_utc": "__key"})
    merged = pd.merge_asof(
        target_df.reset_index(drop=True).sort_values("__key"),
        src_reset.sort_values("__key"),
        on="__key",
        direction="backward",
    )
    merged.index = target_df.index
    merged = merged.drop(columns="__key")
    merged.columns = ["fear_greed"]

    out = pd.DataFrame(index=target_index)
    out["fear_greed"] = merged["fear_greed"]
    out["fear_greed_zscore_30"] = (
        out["fear_greed"] - out["fear_greed"].rolling(30, min_periods=30).mean()
    ) / out["fear_greed"].rolling(30, min_periods=30).std(ddof=0).replace(0, np.nan)
    out["fear_greed_delta"] = out["fear_greed"].diff()
    out["fear_greed_extreme_fear"] = (out["fear_greed"] < 20).astype("float64")
    out["fear_greed_extreme_greed"] = (out["fear_greed"] > 80).astype("float64")
    return out
