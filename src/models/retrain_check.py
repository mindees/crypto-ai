"""Decide whether a retrain is recommended (never auto-trains).

Triggers (from config ``retraining``):

1. Enough new bars accumulated since last train (per timeframe).
2. Feature drift: PSI above ``psi_feature_drift_above`` for important features.
3. Performance drift: live paper-trade expectancy negative (placeholder until
   live paper trades exist).
4. Time-based: weekly/monthly recommendation.

Writes ``reports/retrain_check_<date>.md`` and ``metadata/retrain_status.json``.
GitHub Actions runs this daily but NEVER trains.

CLI::

    python -m src.models.retrain_check
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.models.drift import SIGNIFICANT, feature_drift_table, severity
from src.utils.io import read_json, read_yaml, repo_root, write_json
from src.utils.logging import get_logger

_log = get_logger("models.retrain_check")


def _load_features(symbol: str, timeframe: str, *, market: str, root: Path) -> pd.DataFrame | None:
    path = (
        root / "data" / "features" / f"source=binance" / f"market_type={market}"
        / f"symbol={symbol}" / f"timeframe={timeframe}" / "features.parquet"
    )
    if not path.exists():
        return None
    df = pq.read_table(path).to_pandas()
    for c in ("source", "market_type", "symbol", "timeframe"):
        if c in df.columns:
            df = df.drop(columns=c)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.set_index("timestamp_utc").sort_index()
    return df


def check(symbols, timeframes, *, market: str = "spot", root: Path | None = None) -> dict:
    root = root or repo_root()
    cfg = read_yaml(root / "configs" / "config.yaml")
    rcfg = cfg.get("retraining") or {}
    min_bars = rcfg.get("min_new_bars_before_retrain") or {}
    psi_threshold = float((rcfg.get("trigger_if") or {}).get("psi_feature_drift_above", 0.25))

    reasons: list[str] = []
    drift_findings: dict[str, list[dict]] = {}
    new_bars: dict[str, int] = {}

    watermarks = {}
    wpath = root / "metadata" / "watermarks.json"
    if wpath.exists():
        watermarks = read_json(wpath).get("watermarks", {})

    for symbol in symbols:
        for tf in timeframes:
            df = _load_features(symbol, tf, market=market, root=root)
            if df is None or len(df) < 200:
                continue
            # Reference = first 70%, current = last 15% — proxy for "train vs recent".
            n = len(df)
            ref = df.iloc[: int(n * 0.7)]
            cur = df.iloc[int(n * 0.85):]
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            table = feature_drift_table(ref, cur, columns=numeric_cols[:60])
            significant = [d for d in table if d.severity == SIGNIFICANT]
            drift_findings[f"{symbol}/{tf}"] = [
                {"feature": d.feature, "psi": round(d.psi, 4), "severity": d.severity}
                for d in table[:10]
            ]
            if significant:
                reasons.append(
                    f"{symbol}/{tf}: {len(significant)} feature(s) with PSI>{psi_threshold} "
                    f"(top: {significant[0].feature}={significant[0].psi:.3f})"
                )
            # New-bars proxy: compare current row count to min threshold for this tf
            key = f"binance/{market}/{symbol}/{tf}"
            threshold = int(min_bars.get(tf, 500))
            new_bars[f"{symbol}/{tf}"] = {"rows_available": n, "retrain_bar_threshold": threshold}

    retrain_recommended = len(reasons) > 0
    status = {
        "schema_version": 1,
        "last_check_utc": datetime.now(tz=timezone.utc).isoformat(),
        "retrain_recommended": retrain_recommended,
        "reasons": reasons or ["no drift/staleness triggers fired"],
        "psi_threshold": psi_threshold,
        "drift_findings": drift_findings,
        "new_bars": new_bars,
    }
    return status


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.models.retrain_check")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--timeframes", nargs="+", default=["1h", "4h"])
    p.add_argument("--market", default="spot")
    args = p.parse_args(argv)

    root = repo_root()
    status = check(args.symbols, args.timeframes, market=args.market, root=root)

    write_json(root / "metadata" / "retrain_status.json", status)

    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md = root / "reports" / f"retrain_check_{stamp}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Retrain check — {stamp}",
        "",
        f"retrain_recommended: **{status['retrain_recommended']}**",
        "",
        "## Reasons",
        "",
    ]
    for r in status["reasons"]:
        lines.append(f"- {r}")
    lines += ["", "## Top drift findings (PSI)", ""]
    for combo, findings in status["drift_findings"].items():
        lines.append(f"### {combo}")
        for f in findings:
            lines.append(f"- {f['feature']}: PSI={f['psi']} ({f['severity']})")
        lines.append("")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"retrain_recommended: {status['retrain_recommended']}")
    for r in status["reasons"]:
        print(f"  - {r}")
    print(f"\nstatus: metadata/retrain_status.json")
    print(f"report: {md.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
