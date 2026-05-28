"""Data-validation CLI for CI / daily jobs.

Walks the processed OHLCV parquet partitions and checks each for:

* required OHLCV columns
* monotonic, tz-aware UTC index (open_time)
* duplicate open_time count
* basic sanity (high >= low, no negative volume)

Prints a per-partition report and exits 0 even when nothing is present (so a
fresh checkout in CI doesn't fail). Use ``--strict`` to fail on issues.

CLI::

    python -m src.utils.validation_cli
    python -m src.utils.validation_cli --strict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.utils.io import repo_root
from src.utils.logging import get_logger

_log = get_logger("utils.validation_cli")

REQUIRED = ("open_time", "open", "high", "low", "close", "volume")


def _validate_one(path: Path) -> dict:
    issues: list[str] = []
    df = pq.read_table(path).to_pandas()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        issues.append(f"missing columns {missing}")
        return {"path": str(path), "rows": len(df), "issues": issues}

    ot = pd.to_datetime(df["open_time"], unit="ms", utc=True, errors="coerce")
    if ot.isna().any():
        issues.append("unparseable open_time values")
    if not ot.is_monotonic_increasing:
        issues.append("open_time not monotonic increasing")
    dups = int(ot.duplicated().sum())
    if dups:
        issues.append(f"{dups} duplicate open_time rows")
    if (df["high"] < df["low"]).any():
        issues.append("high < low on some rows")
    if (df["volume"] < 0).any():
        issues.append("negative volume on some rows")

    return {"path": str(path), "rows": len(df), "duplicates": dups, "issues": issues}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.utils.validation_cli")
    p.add_argument("--strict", action="store_true", help="exit non-zero if any issues found")
    args = p.parse_args(argv)

    root = repo_root()
    base = root / "data" / "processed" / "ohlcv"
    parquets = sorted(base.rglob("data.parquet")) if base.exists() else []

    if not parquets:
        print("no processed OHLCV parquet found — nothing to validate (fresh checkout?).")
        return 0

    any_issues = False
    print(f"{'rows':>8}  {'dups':>5}  partition")
    for path in parquets:
        report = _validate_one(path)
        rel = path.relative_to(root)
        if report["issues"]:
            any_issues = True
            print(f"{report['rows']:>8}  {report.get('duplicates', 0):>5}  {rel}  ISSUES: {report['issues']}")
        else:
            print(f"{report['rows']:>8}  {report.get('duplicates', 0):>5}  {rel}  ok")

    if any_issues and args.strict:
        print("validation FAILED (strict mode)")
        return 1
    print("validation complete." + (" (issues found, non-strict)" if any_issues else " all clean."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
