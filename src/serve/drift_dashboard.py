"""Drift dashboard generator/locator.

The actual chart + HTML generation lives in ``src.models.drift_viz``. This
module is the serving-side entry point: it (re)generates the dashboard and
returns the path, and is what ``/drift/latest`` in the API surfaces.

CLI::

    python -m src.serve.drift_dashboard --sample true
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.models import drift_viz
from src.utils.io import repo_root
from src.utils.logging import get_logger

_log = get_logger("serve.drift_dashboard")


def generate(*, symbol: str = "BTCUSDT", timeframe: str = "1h", sample: bool = False) -> Path | None:
    """Generate the dashboard via drift_viz and return the newest HTML path."""
    drift_viz.main([
        "--symbol", symbol, "--timeframe", timeframe,
        "--sample", "true" if sample else "false",
    ])
    reports = repo_root() / "reports"
    dashboards = sorted(reports.glob("drift_dashboard_*.html"))
    return dashboards[-1] if dashboards else None


def latest_dashboard_path() -> Path | None:
    reports = repo_root() / "reports"
    dashboards = sorted(reports.glob("drift_dashboard_*.html"))
    return dashboards[-1] if dashboards else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.serve.drift_dashboard")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--sample", type=lambda s: s.strip().lower() in {"1", "true", "yes", "y", "t"},
                   default=False)
    args = p.parse_args(argv)

    path = generate(symbol=args.symbol, timeframe=args.timeframe, sample=args.sample)
    if path is None:
        print("no dashboard generated")
        return 2
    print(f"dashboard: {path.relative_to(repo_root())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
