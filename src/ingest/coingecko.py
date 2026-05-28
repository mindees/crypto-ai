"""CoinGecko free/demo API: BTC dominance, ETH dominance, global market cap.

The free tier:
* No API key required for the public endpoints used here.
* Rate-limited (~30 req/min). Handled via ``_http``'s 429 retry/backoff.
* Historical depth is shallow on the no-key path — the ``global`` endpoint
  returns only the current snapshot, not a time series. We still call it so
  the daily delta pipeline can append a row per run and build up history
  going forward.

CLI::

    python -m src.ingest.coingecko

Output: ``data/processed/coingecko/global_dominance.parquet`` with columns
``btc_dominance_pct``, ``eth_dominance_pct``, ``total_market_cap_usd``,
``total_volume_usd_24h``.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from datetime import datetime, timezone

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

_log = get_logger("ingest.coingecko")

GLOBAL_URL = "https://api.coingecko.com/api/v3/global"

NAME = "coingecko_global"
SOURCE = "coingecko.com (free)"
KNOWN_LIMITATIONS = [
    "Free tier rate limit (~30 req/min); 429s are auto-retried with backoff.",
    "/global endpoint is a snapshot, not a time series — daily delta pipeline appends one row per run.",
    "CoinGecko has occasional schema changes; we ignore unknown keys.",
]


def fetch() -> AdapterResult:
    result = AdapterResult(name=NAME, source=SOURCE, available=False,
                           known_limitations=list(KNOWN_LIMITATIONS))
    try:
        raw = _http.get_bytes(GLOBAL_URL)
    except (urllib.error.HTTPError, OSError) as exc:
        result.reason = f"network error fetching {GLOBAL_URL}: {exc}"
        _log.warning(result.reason)
        return result

    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        result.reason = f"could not parse response JSON: {exc}"
        return result

    data = payload.get("data") or {}
    if not isinstance(data, dict) or not data:
        result.reason = f"unexpected /global schema; top-level data: {type(data).__name__}"
        return result

    btc_pct = (data.get("market_cap_percentage") or {}).get("btc")
    eth_pct = (data.get("market_cap_percentage") or {}).get("eth")
    total_mc = (data.get("total_market_cap") or {}).get("usd")
    total_vol = (data.get("total_volume") or {}).get("usd")

    if btc_pct is None and total_mc is None:
        result.reason = "no usable fields in /global response"
        return result

    now_utc = datetime.now(tz=timezone.utc).replace(microsecond=0)
    row = {
        "timestamp_utc": now_utc,
        "btc_dominance_pct": float(btc_pct) if btc_pct is not None else float("nan"),
        "eth_dominance_pct": float(eth_pct) if eth_pct is not None else float("nan"),
        "total_market_cap_usd": float(total_mc) if total_mc is not None else float("nan"),
        "total_volume_usd_24h": float(total_vol) if total_vol is not None else float("nan"),
    }
    df = pd.DataFrame([row])
    df = finalize_dataframe(df, index_name="timestamp_utc")
    result.df = df
    result.metric_columns = [
        "btc_dominance_pct", "eth_dominance_pct",
        "total_market_cap_usd", "total_volume_usd_24h",
    ]
    result.available = True
    populate_timestamps(result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ingest.coingecko")
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args(argv)

    result = fetch()
    if result.available and not args.no_persist:
        out = repo_root() / "data" / "processed" / "coingecko" / "global_dominance.parquet"
        write_parquet_idempotent(result.df, out)
        result.parquet_path = str(out.relative_to(repo_root()))
        update_source_registry([result])

    print_result(result)
    return 0 if result.available else 2


if __name__ == "__main__":
    sys.exit(main())
