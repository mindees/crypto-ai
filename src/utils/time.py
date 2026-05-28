"""UTC-first time helpers.

All timestamps in this project are UTC. These helpers exist so callers never
have to think about timezone conversion at call sites.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

UTC = timezone.utc

# Bar-length-in-milliseconds lookup for every supported Binance interval.
TIMEFRAME_TO_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
    # 1mo is calendar-month-based; rough approximation only used for bookkeeping.
    "1mo": 30 * 24 * 60 * 60_000,
}


def now_utc() -> datetime:
    return datetime.now(tz=UTC)


def to_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def ms_to_utc(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)


def utc_to_ms(ts: datetime) -> int:
    return int(to_utc(ts).timestamp() * 1000)


def timeframe_ms(timeframe: str) -> int:
    if timeframe not in TIMEFRAME_TO_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    return TIMEFRAME_TO_MS[timeframe]


def ensure_utc_index(df: pd.DataFrame, column: str = "open_time") -> pd.DataFrame:
    """Coerce a candle DataFrame's open_time column to a tz-aware UTC DatetimeIndex."""
    if column not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        raise KeyError(f"DataFrame has neither '{column}' column nor a DatetimeIndex")
    if column in df.columns:
        idx = pd.to_datetime(df[column], utc=True, errors="raise")
        df = df.set_index(idx)
        df.index.name = column
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    return df
