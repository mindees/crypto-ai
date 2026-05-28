"""Lightweight runtime validators used across ingestion and feature code."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def assert_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def assert_ohlcv_schema(df: pd.DataFrame) -> None:
    assert_columns(df, REQUIRED_OHLCV_COLUMNS)


def assert_monotonic_utc(series: pd.Series) -> None:
    if not pd.api.types.is_datetime64_any_dtype(series):
        raise TypeError("Series must be a datetime dtype")
    tz = getattr(series.dt, "tz", None)
    if tz is None:
        raise ValueError("Series must be tz-aware UTC")
    if str(tz) not in ("UTC", "tzutc()", "datetime.timezone.utc"):
        raise ValueError(f"Series must be UTC, got {tz}")
    if not series.is_monotonic_increasing:
        raise ValueError("Series must be monotonic increasing")


def assert_no_future_leakage(reference_ts: pd.Timestamp, feature_ts: pd.Timestamp) -> None:
    """Tiny guard: a feature timestamp must not exceed the reference (label) timestamp."""
    if feature_ts > reference_ts:
        raise ValueError(
            f"Future leakage detected: feature_ts {feature_ts} > reference_ts {reference_ts}"
        )
