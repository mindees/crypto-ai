"""Alternative.me Crypto Fear & Greed Index.

Free, no API key. ``limit=0`` returns the full history (daily values since
2018-02-01). The index is mostly Bitcoin-driven; documented as such.

CLI::

    python -m src.ingest.sentiment

Output: ``data/processed/sentiment/fear_greed.parquet`` with columns
``fear_greed_value`` (0–100) and ``fear_greed_classification`` (categorical).
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from pathlib import Path

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

_log = get_logger("ingest.sentiment")

FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"

NAME = "fear_greed_index"
SOURCE = "alternative.me"
KNOWN_LIMITATIONS = [
    "Mostly Bitcoin/crypto-market sentiment, not ETH-specific.",
    "Daily resolution only; published once per day around 00:00 UTC.",
    "First available value: 2018-02-01.",
]


def fetch() -> AdapterResult:
    result = AdapterResult(name=NAME, source=SOURCE, available=False,
                           known_limitations=list(KNOWN_LIMITATIONS))
    try:
        raw = _http.get_bytes(FNG_URL)
    except (urllib.error.HTTPError, OSError) as exc:
        result.reason = f"network error fetching {FNG_URL}: {exc}"
        _log.warning(result.reason)
        return result

    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        result.reason = f"could not parse response JSON: {exc}"
        return result

    data = payload.get("data") or []
    if not data:
        result.reason = "response payload had empty 'data' array"
        return result

    df = pd.DataFrame(data)
    if "timestamp" not in df.columns or "value" not in df.columns:
        result.reason = f"unexpected schema: columns={list(df.columns)}"
        return result

    df["timestamp_utc"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="s", utc=True)
    df["fear_greed_value"] = pd.to_numeric(df["value"], errors="raise").astype("int16")
    df["fear_greed_classification"] = df.get(
        "value_classification", pd.Series([None] * len(df))
    ).astype("string")
    df = df[["timestamp_utc", "fear_greed_value", "fear_greed_classification"]]

    df = finalize_dataframe(df, index_name="timestamp_utc")
    result.df = df
    result.metric_columns = ["fear_greed_value", "fear_greed_classification"]
    result.available = True
    populate_timestamps(result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ingest.sentiment")
    parser.add_argument("--no-persist", action="store_true",
                        help="Fetch and report, but don't write parquet/registry.")
    args = parser.parse_args(argv)

    result = fetch()
    if result.available and not args.no_persist:
        out = repo_root() / "data" / "processed" / "sentiment" / "fear_greed.parquet"
        write_parquet_idempotent(result.df, out)
        result.parquet_path = str(out.relative_to(repo_root()))
        update_source_registry([result])

    print_result(result)
    return 0 if result.available else 2


if __name__ == "__main__":
    sys.exit(main())
