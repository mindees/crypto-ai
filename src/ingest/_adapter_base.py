"""Shared base for non-OHLCV ingestion adapters (Phase 2).

Every adapter returns an ``AdapterResult`` whether it succeeds, partially
succeeds, or fails. Callers never have to wrap in try/except just to detect
unavailability — the result object reports it explicitly.

The base also owns idempotent Parquet persistence and source-registry updates
so per-adapter code stays focused on "fetch and parse".
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.io import read_yaml, repo_root, write_yaml
from src.utils.logging import get_logger

_log = get_logger("ingest.adapter")


@dataclass
class AdapterResult:
    """Outcome of one adapter's fetch.

    ``df`` is None when the adapter is unavailable; in that case ``reason``
    must be set. When available, ``df`` is a tz-aware UTC-indexed DataFrame
    and ``metric_columns`` lists the actual data columns.
    """
    name: str
    source: str
    available: bool
    df: pd.DataFrame | None = None
    metric_columns: list[str] = field(default_factory=list)
    reason: str | None = None
    first_timestamp_utc: str | None = None
    last_timestamp_utc: str | None = None
    row_count: int = 0
    known_limitations: list[str] = field(default_factory=list)
    parquet_path: str | None = None

    def to_registry_entry(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "available": self.available,
            "first_timestamp_utc": self.first_timestamp_utc,
            "last_timestamp_utc": self.last_timestamp_utc,
            "row_count": self.row_count,
            "metric_columns": list(self.metric_columns),
            "known_limitations": list(self.known_limitations),
            "reason": self.reason,
            "parquet_path": self.parquet_path,
        }


def finalize_dataframe(df: pd.DataFrame, *, index_name: str = "timestamp_utc") -> pd.DataFrame:
    """Coerce a fetched DataFrame to the project's canonical shape.

    * tz-aware UTC index named ``timestamp_utc``
    * sorted ascending
    * duplicate timestamps dropped (keep last — usually the more authoritative)
    """
    if df.empty:
        return df

    if df.index.name != index_name:
        if index_name in df.columns:
            df = df.set_index(index_name)
        else:
            raise ValueError(
                f"DataFrame has no '{index_name}' column and index name is {df.index.name!r}"
            )

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="raise")
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = index_name
    return df


def write_parquet_idempotent(df: pd.DataFrame, path: Path) -> None:
    """Merge ``df`` with any existing parquet at ``path`` (dedupe on index), write atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pq.read_table(path).to_pandas()
        existing = finalize_dataframe(existing, index_name=df.index.name)
        merged = pd.concat([existing, df]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
    else:
        merged = df

    table = pa.Table.from_pandas(merged.reset_index(), preserve_index=False)
    tmp = path.with_suffix(path.suffix + ".part")
    pq.write_table(table, tmp, compression="snappy")
    tmp.replace(path)


def populate_timestamps(result: AdapterResult) -> None:
    if result.df is None or result.df.empty:
        return
    result.row_count = len(result.df)
    result.first_timestamp_utc = result.df.index.min().isoformat()
    result.last_timestamp_utc = result.df.index.max().isoformat()


def update_source_registry(results: Iterable[AdapterResult], *, root: Path | None = None) -> Path:
    """Merge adapter results into ``metadata/source_registry.yaml``.

    Entries are keyed by ``name`` so re-runs update in place rather than
    appending duplicates. Binance bulk ingest's entries (which use a
    different schema) are preserved untouched.
    """
    root = root or repo_root()
    path = root / "metadata" / "source_registry.yaml"
    payload = read_yaml(path) if path.exists() else {"schema_version": 1, "sources": []}
    payload.setdefault("schema_version", 1)
    payload.setdefault("sources", [])

    existing = list(payload.get("sources") or [])
    by_name = {e.get("name"): i for i, e in enumerate(existing) if "name" in e}

    for r in results:
        entry = r.to_registry_entry()
        if r.name in by_name:
            existing[by_name[r.name]] = entry
        else:
            existing.append(entry)

    payload["sources"] = existing
    payload["generated_at_utc"] = datetime.now(tz=timezone.utc).isoformat()
    write_yaml(path, payload)
    return path


def print_result(result: AdapterResult) -> None:
    status = "ok" if result.available else "UNAVAILABLE"
    print(f"\n=== {result.name} ({result.source}) ===")
    print(f"status: {status}")
    if not result.available:
        print(f"reason: {result.reason}")
        return
    print(f"first:  {result.first_timestamp_utc}")
    print(f"last:   {result.last_timestamp_utc}")
    print(f"rows:   {result.row_count}")
    print(f"metrics: {', '.join(result.metric_columns)}")
    if result.known_limitations:
        print("limitations:")
        for lim in result.known_limitations:
            print(f"  - {lim}")
    print(f"saved:  {result.parquet_path or '(not persisted)'}")
