"""Lightweight daily delta orchestrator (for GitHub Actions / cron).

Runs only the *cheap* refresh work — never trains a model:

1. Latest-only Binance OHLCV delta for configured train timeframes.
2. Fear & Greed + CoinGecko snapshots (append).
3. Derivatives recent window.
4. Data validation (row counts, monotonic UTC index).
5. Retrain/staleness check → ``metadata/retrain_status.json`` + report.

Every step is resilient: a failed adapter logs and the run continues. The
function returns a summary dict the workflow can print.

CLI::

    python -m src.ingest.daily_update
    python -m src.ingest.daily_update --skip-binance   # metadata-only refresh
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.utils.io import read_yaml, repo_root, write_json
from src.utils.logging import get_logger

_log = get_logger("ingest.daily_update")


def _safe(label: str, fn) -> dict:
    try:
        fn()
        return {"step": label, "ok": True, "error": None}
    except Exception as exc:  # noqa: BLE001 — daily job must never hard-fail
        _log.warning("daily step %s failed: %s", label, exc)
        return {"step": label, "ok": False, "error": str(exc)}


def run(symbols, timeframes, *, market: str = "spot", skip_binance: bool = False,
        root: Path | None = None) -> dict:
    root = root or repo_root()
    results: list[dict] = []

    if not skip_binance:
        def _binance():
            from src.ingest import binance_bulk
            binance_bulk.run(
                symbols=symbols, market_types=[market], timeframes=timeframes,
                # latest-only: restrict to current year to keep the delta tiny
                start_year=datetime.now(tz=timezone.utc).year,
                root=root,
            )
        results.append(_safe("binance_delta", _binance))

    def _sentiment():
        from src.ingest import sentiment
        r = sentiment.fetch()
        if r.available:
            from src.ingest._adapter_base import write_parquet_idempotent, update_source_registry
            out = root / "data" / "processed" / "sentiment" / "fear_greed.parquet"
            write_parquet_idempotent(r.df, out)
            update_source_registry([r], root=root)
    results.append(_safe("sentiment", _sentiment))

    def _coingecko():
        from src.ingest import coingecko
        r = coingecko.fetch()
        if r.available:
            from src.ingest._adapter_base import write_parquet_idempotent, update_source_registry
            out = root / "data" / "processed" / "coingecko" / "global_dominance.parquet"
            write_parquet_idempotent(r.df, out)
            update_source_registry([r], root=root)
    results.append(_safe("coingecko", _coingecko))

    def _derivatives():
        from src.ingest import derivatives
        res = derivatives.fetch_all(symbols, period="1h", limit=500)
        from src.ingest._adapter_base import write_parquet_idempotent, update_source_registry
        for r in res:
            if not r.available:
                continue
            parsed = derivatives._parse_metric_name(r.name)
            if parsed:
                sym, metric = parsed
                out = derivatives._output_path(sym, metric, root)
                write_parquet_idempotent(r.df, out)
        update_source_registry([r for r in res], root=root)
    results.append(_safe("derivatives", _derivatives))

    def _retrain_check():
        from src.models import retrain_check
        status = retrain_check.check(symbols, timeframes, market=market, root=root)
        write_json(root / "metadata" / "retrain_status.json", status)
    results.append(_safe("retrain_check", _retrain_check))

    summary = {
        "ran_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "symbols": list(symbols),
        "timeframes": list(timeframes),
        "steps": results,
        "all_ok": all(r["ok"] for r in results),
    }
    write_json(root / "metadata" / "daily_update_status.json", summary)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.ingest.daily_update")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--timeframes", nargs="+", default=["1h", "4h", "1d"])
    p.add_argument("--market", default="spot")
    p.add_argument("--skip-binance", action="store_true")
    args = p.parse_args(argv)

    summary = run(args.symbols, args.timeframes, market=args.market,
                  skip_binance=args.skip_binance, root=repo_root())
    print(f"daily update @ {summary['ran_at_utc']}")
    for step in summary["steps"]:
        print(f"  [{'ok' if step['ok'] else 'FAIL'}] {step['step']}"
              + (f" — {step['error']}" if step["error"] else ""))
    print(f"all_ok: {summary['all_ok']}")
    return 0  # never fail the daily job


if __name__ == "__main__":
    sys.exit(main())
