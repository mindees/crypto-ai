"""Binance USDT-margined futures derivatives metrics via REST.

The spec warns explicitly that "REST endpoints with recent-only windows must
be marked as recent-only" — we respect that here. Each metric reports its
coverage and known limitations in the source registry.

Metrics:

* ``funding_rate``         — ``/fapi/v1/fundingRate``. Funding happens every
  8h. Paginated; effectively goes back to symbol launch. The deepest /
  cheapest historical derivatives series Binance exposes for free.
* ``open_interest_hist``   — ``/futures/data/openInterestHist``. Recent-only
  (~30 days at hourly resolution). One row per requested period.
* ``global_long_short_ratio`` — ``/futures/data/globalLongShortAccountRatio``.
  Recent-only.
* ``top_trader_position_ratio`` — ``/futures/data/topLongShortPositionRatio``.
  Recent-only.
* ``taker_long_short_volume`` — ``/futures/data/takerlongshortRatio``.
  Returns buy/sell volume ratios. Recent-only.

CLI::

    python -m src.ingest.derivatives --symbols BTCUSDT ETHUSDT
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.ingest._adapter_base import (
    AdapterResult,
    finalize_dataframe,
    populate_timestamps,
    print_result,
    update_source_registry,
    write_parquet_idempotent,
)
from src.ingest import _http
from src.utils.io import repo_root
from src.utils.logging import get_logger

_log = get_logger("ingest.derivatives")

FAPI_BASE = "https://fapi.binance.com"

# Approximate Binance USDT-M perpetual launches; used as the starting hint
# when paging funding-rate history. The REST endpoint will silently return an
# empty page for periods before launch — safe.
FUTURES_LAUNCH_MS: dict[str, int] = {
    "BTCUSDT": 1568937600000,   # 2019-09-20 (BTCUSDT perp launch)
    "ETHUSDT": 1574208000000,   # 2019-11-20 (ETHUSDT perp launch)
}

RECENT_ONLY_LIMITATION = (
    "REST endpoint is recent-only (~30 days at hourly resolution). "
    "Older data is not retrievable for free."
)
FUNDING_LIMITATION = (
    "Funding events every 8h. Paginated through full history; capped at "
    "1000 events per request. Earlier than symbol launch returns empty."
)


# ---------------------------------------------------------------------------
# Low-level fetch helpers
# ---------------------------------------------------------------------------

def _get_json(url: str) -> list | dict | None:
    try:
        raw = _http.get_bytes(url)
    except (urllib.error.HTTPError, OSError) as exc:
        _log.warning("GET %s failed: %s", url, exc)
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        _log.warning("GET %s returned non-JSON: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Funding rate (paginated full history)
# ---------------------------------------------------------------------------

def fetch_funding_rate(symbol: str, *, start_ms: int | None = None) -> AdapterResult:
    res = AdapterResult(
        name=f"deriv_{symbol}_funding_rate",
        source=f"binance fapi:fundingRate:{symbol}",
        available=False,
        known_limitations=[FUNDING_LIMITATION],
    )
    if start_ms is None:
        start_ms = FUTURES_LAUNCH_MS.get(symbol, 1568937600000)

    rows: list[dict] = []
    cursor = start_ms
    end_cap = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    page_count = 0
    while True:
        page_count += 1
        url = (
            f"{FAPI_BASE}/fapi/v1/fundingRate?"
            f"symbol={symbol}&startTime={cursor}&endTime={end_cap}&limit=1000"
        )
        payload = _get_json(url)
        if payload is None:
            res.reason = f"network failure after {page_count - 1} pages"
            return res
        if not isinstance(payload, list):
            res.reason = f"unexpected payload shape: {type(payload).__name__}"
            return res
        if not payload:
            break
        rows.extend(payload)
        last_ms = int(payload[-1]["fundingTime"])
        if len(payload) < 1000 or last_ms >= end_cap:
            break
        cursor = last_ms + 1
        # tiny politeness sleep — Binance fapi is generous but not unlimited
        time.sleep(0.10)

    if not rows:
        res.reason = "no funding events returned"
        return res

    df = pd.DataFrame(rows)
    df["timestamp_utc"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce").astype("float64")
    if "markPrice" in df.columns:
        df["mark_price"] = pd.to_numeric(df["markPrice"], errors="coerce").astype("float64")
    keep = ["timestamp_utc", "funding_rate"] + (["mark_price"] if "mark_price" in df.columns else [])
    df = df[keep]
    df = finalize_dataframe(df, index_name="timestamp_utc")
    res.df = df
    res.metric_columns = [c for c in df.columns]
    res.available = True
    populate_timestamps(res)
    return res


# ---------------------------------------------------------------------------
# Recent-only series
# ---------------------------------------------------------------------------

def _fetch_recent_series(
    symbol: str,
    endpoint_path: str,
    *,
    metric_name: str,
    fields: dict[str, str],  # source_key -> canonical column
    period: str = "1h",
    limit: int = 500,
) -> AdapterResult:
    res = AdapterResult(
        name=f"deriv_{symbol}_{metric_name}",
        source=f"binance fapi:{endpoint_path}:{symbol}",
        available=False,
        known_limitations=[RECENT_ONLY_LIMITATION],
    )
    qs = urllib.parse.urlencode({"symbol": symbol, "period": period, "limit": limit})
    url = f"{FAPI_BASE}{endpoint_path}?{qs}"
    payload = _get_json(url)
    if payload is None:
        res.reason = "network failure"
        return res
    if not isinstance(payload, list) or not payload:
        res.reason = "empty payload"
        return res

    df = pd.DataFrame(payload)
    if "timestamp" not in df.columns:
        res.reason = f"unexpected schema: columns={list(df.columns)}"
        return res
    df["timestamp_utc"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    out = {"timestamp_utc": df["timestamp_utc"]}
    for src_key, canon in fields.items():
        if src_key not in df.columns:
            continue
        out[canon] = pd.to_numeric(df[src_key], errors="coerce").astype("float64")
    df_out = pd.DataFrame(out)
    df_out = finalize_dataframe(df_out, index_name="timestamp_utc")
    res.df = df_out
    res.metric_columns = [c for c in df_out.columns]
    res.available = True
    populate_timestamps(res)
    return res


def fetch_open_interest_hist(symbol: str, *, period: str = "1h", limit: int = 500) -> AdapterResult:
    return _fetch_recent_series(
        symbol, "/futures/data/openInterestHist",
        metric_name="open_interest_hist",
        fields={
            "sumOpenInterest": "open_interest",
            "sumOpenInterestValue": "open_interest_usd",
        },
        period=period, limit=limit,
    )


def fetch_global_long_short_ratio(symbol: str, *, period: str = "1h", limit: int = 500) -> AdapterResult:
    return _fetch_recent_series(
        symbol, "/futures/data/globalLongShortAccountRatio",
        metric_name="global_long_short_ratio",
        fields={
            "longAccount": "global_long_account_pct",
            "shortAccount": "global_short_account_pct",
            "longShortRatio": "global_long_short_ratio",
        },
        period=period, limit=limit,
    )


def fetch_top_trader_position_ratio(symbol: str, *, period: str = "1h", limit: int = 500) -> AdapterResult:
    return _fetch_recent_series(
        symbol, "/futures/data/topLongShortPositionRatio",
        metric_name="top_trader_position_ratio",
        fields={
            "longAccount": "top_long_position_pct",
            "shortAccount": "top_short_position_pct",
            "longShortRatio": "top_long_short_ratio",
        },
        period=period, limit=limit,
    )


def fetch_taker_long_short_volume(symbol: str, *, period: str = "1h", limit: int = 500) -> AdapterResult:
    return _fetch_recent_series(
        symbol, "/futures/data/takerlongshortRatio",
        metric_name="taker_long_short_volume",
        fields={
            "buyVol": "taker_buy_volume",
            "sellVol": "taker_sell_volume",
            "buySellRatio": "taker_buy_sell_ratio",
        },
        period=period, limit=limit,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def fetch_all(symbols: Iterable[str], *, period: str = "1h", limit: int = 500) -> list[AdapterResult]:
    results: list[AdapterResult] = []
    for symbol in symbols:
        results.append(fetch_funding_rate(symbol))
        results.append(fetch_open_interest_hist(symbol, period=period, limit=limit))
        results.append(fetch_global_long_short_ratio(symbol, period=period, limit=limit))
        results.append(fetch_top_trader_position_ratio(symbol, period=period, limit=limit))
        results.append(fetch_taker_long_short_volume(symbol, period=period, limit=limit))
    return results


def _output_path(symbol: str, metric: str, root: Path) -> Path:
    return root / "data" / "processed" / "derivatives" / symbol / f"{metric}.parquet"


def _parse_metric_name(name: str) -> tuple[str, str] | None:
    """'deriv_BTCUSDT_funding_rate' -> ('BTCUSDT', 'funding_rate')."""
    if not name.startswith("deriv_"):
        return None
    rest = name.removeprefix("deriv_")
    for symbol in ("BTCUSDT", "ETHUSDT"):
        prefix = f"{symbol}_"
        if rest.startswith(prefix):
            return symbol, rest.removeprefix(prefix)
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ingest.derivatives")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--period", default="1h")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args(argv)

    results = fetch_all(args.symbols, period=args.period, limit=args.limit)

    root = repo_root()
    if not args.no_persist:
        for res in results:
            if not res.available:
                continue
            parsed = _parse_metric_name(res.name)
            if parsed is None:
                continue
            symbol, metric = parsed
            out = _output_path(symbol, metric, root)
            write_parquet_idempotent(res.df, out)
            res.parquet_path = str(out.relative_to(root))
        update_source_registry(results)

    for res in results:
        print_result(res)

    any_available = any(r.available for r in results)
    return 0 if any_available else 2


if __name__ == "__main__":
    sys.exit(main())
