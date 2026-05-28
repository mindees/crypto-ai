"""Local prediction scheduler.

A thin loop that, on each tick: (optionally) refreshes data, runs prediction,
writes ``reports/latest_predictions.json``, and dispatches alerts for any
actionable signal that passes the gates + cooldown.

Cooldown is tracked per (asset, timeframe) in
``reports/alert_cooldowns.json`` so restarts don't spam.

CLI::

    python -m src.serve.scheduler --refresh-minutes 15
    python -m src.serve.scheduler --once          # single tick then exit (smoke)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.serve.alert_templates import AlertGateConfig, build_alert_payload, should_alert
from src.serve.alerts import dispatch
from src.utils.io import read_json, read_yaml, repo_root, write_json
from src.utils.logging import get_logger

_log = get_logger("serve.scheduler")


def _cooldown_path(root: Path) -> Path:
    return root / "reports" / "alert_cooldowns.json"


def _load_cooldowns(root: Path) -> dict:
    path = _cooldown_path(root)
    return read_json(path) if path.exists() else {}


def _save_cooldowns(root: Path, data: dict) -> None:
    write_json(_cooldown_path(root), data)


def run_prediction(root: Path, *, symbols, timeframes) -> bool:
    try:
        res = subprocess.run(
            [sys.executable, "-m", "src.models.predict", "--latest",
             "--symbols", *symbols, "--timeframes", *timeframes],
            cwd=str(root), check=False, timeout=900,
        )
        return res.returncode == 0
    except Exception as exc:  # noqa: BLE001
        _log.warning("prediction subprocess failed: %s", exc)
        return False


def tick(root: Path, *, symbols, timeframes, gate: AlertGateConfig, run_predict: bool) -> dict:
    if run_predict:
        run_prediction(root, symbols=symbols, timeframes=timeframes)

    pred_path = root / "reports" / "latest_predictions.json"
    if not pred_path.exists():
        return {"alerts_sent": 0, "note": "no latest_predictions.json"}
    data = read_json(pred_path)

    cooldowns = _load_cooldowns(root)
    now = datetime.now(tz=timezone.utc)
    sent = 0
    decisions = []
    for pred in data.get("predictions", []):
        payload = build_alert_payload(prediction=pred, cooldown_minutes=gate.cooldown_minutes)
        key = f"{pred.get('asset')}/{pred.get('timeframe')}"
        last_iso = cooldowns.get(key)
        last_dt = datetime.fromisoformat(last_iso) if last_iso else None
        send, reason = should_alert(payload, gate, last_alert_utc=last_dt, now=now)
        decisions.append({"combo": key, "signal": payload["signal"], "send": send, "reason": reason})
        if send:
            results = dispatch(payload)
            any_sent = any(r.sent for r in results)
            if any_sent:
                cooldowns[key] = now.isoformat()
                sent += 1
            for r in results:
                _log.info("alert %s: attempted=%s sent=%s (%s)", r.channel, r.attempted, r.sent, r.reason)
    _save_cooldowns(root, cooldowns)
    return {"alerts_sent": sent, "decisions": decisions}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.serve.scheduler")
    p.add_argument("--refresh-minutes", type=int, default=15)
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--timeframes", nargs="+", default=["1h", "4h"])
    p.add_argument("--once", action="store_true", help="single tick then exit")
    p.add_argument("--no-predict", action="store_true",
                   help="skip the TF prediction subprocess; just evaluate existing predictions")
    args = p.parse_args(argv)

    root = repo_root()
    cfg = read_yaml(root / "configs" / "config.yaml")
    alerting = ((cfg.get("serving") or {}).get("alerting")) or {}
    gate = AlertGateConfig(
        min_signal_confidence=float(alerting.get("min_signal_confidence", 0.65)),
        min_trade_quality_probability=float(alerting.get("min_trade_quality_probability", 0.60)),
        cooldown_minutes=int(alerting.get("cooldown_minutes", 60)),
    )

    if args.once:
        result = tick(root, symbols=args.symbols, timeframes=args.timeframes,
                      gate=gate, run_predict=not args.no_predict)
        print(f"tick complete. alerts_sent={result['alerts_sent']}")
        for d in result.get("decisions", []):
            print(f"  {d['combo']}: {d['signal']} -> send={d['send']} ({d['reason']})")
        return 0

    _log.info("scheduler started; refresh every %d min", args.refresh_minutes)
    try:
        while True:
            result = tick(root, symbols=args.symbols, timeframes=args.timeframes,
                          gate=gate, run_predict=not args.no_predict)
            _log.info("tick: alerts_sent=%d", result["alerts_sent"])
            time.sleep(args.refresh_minutes * 60)
    except KeyboardInterrupt:
        _log.info("scheduler stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
