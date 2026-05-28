"""Macro context: equity index returns via yfinance, FRED if a key is set.

Default series via yfinance (no key, free, daily close):

* ``^GSPC``  — S&P 500 close
* ``^IXIC``  — Nasdaq Composite close
* ``DX-Y.NYB`` — US Dollar Index (DXY) proxy; sometimes intermittently
  unavailable on Yahoo, so we attempt and tolerate failure
* ``^VIX``   — volatility index

FRED series (only when ``FRED_API_KEY`` is set and
``sources.enable_fred`` is true in the config):

* ``FEDFUNDS`` — effective federal funds rate
* ``CPIAUCSL`` — consumer price index (all urban consumers)

All series are stored individually so missing series never block others.

CLI::

    python -m src.ingest.macro
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.ingest._adapter_base import (
    AdapterResult,
    finalize_dataframe,
    populate_timestamps,
    print_result,
    update_source_registry,
    write_parquet_idempotent,
)
from src.utils.io import read_yaml, repo_root
from src.utils.logging import get_logger

_log = get_logger("ingest.macro")

YFINANCE_TICKERS: dict[str, str] = {
    "sp500_close": "^GSPC",
    "nasdaq_close": "^IXIC",
    "dxy_close": "DX-Y.NYB",
    "vix_close": "^VIX",
}

FRED_SERIES: dict[str, str] = {
    "fed_funds_rate_pct": "FEDFUNDS",
    "cpi_all_urban": "CPIAUCSL",
}

NAME = "macro"
SOURCE = "yfinance + optional FRED"
KNOWN_LIMITATIONS = [
    "Equity series are daily close; rolling correlations vs BTC/ETH must respect causality.",
    "Yahoo Finance occasionally rejects requests; missing tickers are tolerated, not fatal.",
    "FRED series only fetched when FRED_API_KEY is set AND sources.enable_fred is true.",
]


def _yfinance_history(
    ticker: str, *, period_years: int = 12, max_retries: int = 3,
) -> tuple[pd.DataFrame | None, str | None]:
    """Returns (DataFrame, error_reason). On success error_reason is None."""
    try:
        import yfinance as yf  # noqa: WPS433
    except ImportError:
        return None, "yfinance package not installed"

    end = datetime.now(tz=timezone.utc).date()
    start = end - timedelta(days=int(period_years * 365.25))

    last_err: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = yf.download(
                    ticker, start=start.isoformat(), end=end.isoformat(),
                    progress=False, auto_adjust=False, threads=False,
                )
        except Exception as exc:  # noqa: BLE001 — yfinance raises a grab-bag
            last_err = f"{type(exc).__name__}: {exc}"
            _log.warning("yfinance %s attempt %d/%d failed: %s",
                         ticker, attempt, max_retries, last_err)
            if attempt < max_retries:
                import time as _time
                _time.sleep(2 ** attempt)
            continue

        if data is None or data.empty:
            # The most common "empty" failure on Yahoo is a quiet 429 rate
            # limit — surface that hypothesis in the reason text.
            last_err = (
                "yfinance returned an empty frame "
                "(Yahoo Finance is likely rate-limiting — try again later)"
            )
            if attempt < max_retries:
                import time as _time
                _time.sleep(2 ** attempt)
            continue

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        if "Close" not in data.columns:
            return None, f"unexpected yfinance schema; columns={list(data.columns)}"
        out = data[["Close"]].rename(columns={"Close": "value"}).copy()
        out.index = pd.to_datetime(out.index, utc=True)
        out.index.name = "timestamp_utc"
        return out, None

    return None, last_err


def _fred_series(series_id: str, api_key: str) -> pd.DataFrame | None:
    try:
        from fredapi import Fred  # noqa: WPS433
    except ImportError:
        _log.warning("fredapi not installed; skipping FRED series %s", series_id)
        return None
    try:
        fred = Fred(api_key=api_key)
        s = fred.get_series(series_id)
    except Exception as exc:  # noqa: BLE001
        _log.warning("FRED failed for %s: %s", series_id, exc)
        return None
    if s is None or len(s) == 0:
        return None
    df = pd.DataFrame({"value": s})
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp_utc"
    return df


def _config_enable_fred() -> bool:
    cfg_path = repo_root() / "configs" / "config.yaml"
    try:
        cfg = read_yaml(cfg_path)
    except Exception:  # noqa: BLE001 — config absence shouldn't kill ingest
        return False
    return bool((cfg.get("sources") or {}).get("enable_fred"))


def fetch() -> list[AdapterResult]:
    """Returns one AdapterResult per macro series (success or unavailable)."""
    results: list[AdapterResult] = []
    fred_key = os.environ.get("FRED_API_KEY")
    fred_enabled = _config_enable_fred() and bool(fred_key)

    for series_name, ticker in YFINANCE_TICKERS.items():
        res = AdapterResult(
            name=f"macro_{series_name}",
            source=f"yfinance:{ticker}",
            available=False,
            known_limitations=list(KNOWN_LIMITATIONS),
        )
        df, err = _yfinance_history(ticker)
        if df is None or df.empty:
            res.reason = err or f"yfinance returned no data for {ticker}"
            results.append(res)
            continue
        df = finalize_dataframe(df, index_name="timestamp_utc")
        res.df = df
        res.metric_columns = ["value"]
        res.available = True
        populate_timestamps(res)
        results.append(res)

    for series_name, series_id in FRED_SERIES.items():
        res = AdapterResult(
            name=f"macro_{series_name}",
            source=f"fred:{series_id}",
            available=False,
            known_limitations=list(KNOWN_LIMITATIONS),
        )
        if not fred_enabled:
            res.reason = (
                "FRED disabled: needs FRED_API_KEY env var AND sources.enable_fred=true in config"
            )
            results.append(res)
            continue
        df = _fred_series(series_id, fred_key)
        if df is None or df.empty:
            res.reason = f"FRED returned no data for {series_id}"
            results.append(res)
            continue
        df = finalize_dataframe(df, index_name="timestamp_utc")
        res.df = df
        res.metric_columns = ["value"]
        res.available = True
        populate_timestamps(res)
        results.append(res)

    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ingest.macro")
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args(argv)

    results = fetch()
    for res in results:
        if res.available and not args.no_persist:
            short = res.name.removeprefix("macro_")
            out = repo_root() / "data" / "processed" / "macro" / f"{short}.parquet"
            write_parquet_idempotent(res.df, out)
            res.parquet_path = str(out.relative_to(repo_root()))

    if not args.no_persist:
        update_source_registry(results)
    for res in results:
        print_result(res)

    any_available = any(r.available for r in results)
    return 0 if any_available else 2


if __name__ == "__main__":
    sys.exit(main())
