"""A/B comparison of shadow logs — production vs candidate.

Reads ``reports/shadow/shadow_predictions_<candidate_id>.jsonl`` and computes
the comparison metrics the spec lists, then writes
``shadow_compare_<candidate_id>.md`` and ``.json``.

Promotion-after-shadow gates (min shadow days/signals etc.) are reported but
NOT auto-applied — promotion still requires an explicit ``promote`` command.

CLI::

    python -m src.models.ab_compare --candidate latest --sample true
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.models import registry
from src.utils.io import read_yaml, repo_root, write_json
from src.utils.logging import get_logger

_log = get_logger("models.ab_compare")


def _find_latest_run_dir(root: Path) -> Path | None:
    base = root / "artifacts" / "runs"
    if not base.exists():
        return None
    cands = sorted([p for p in base.iterdir() if p.is_dir() and (p / "model.keras").exists()])
    return cands[-1] if cands else None


def compare(candidate_id: str, *, root: Path, cfg: dict) -> dict | None:
    jsonl = root / "reports" / "shadow" / f"shadow_predictions_{candidate_id}.jsonl"
    if not jsonl.exists():
        return None

    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return None

    n = len(rows)
    agree = sum(1 for r in rows if r["production_signal"] == r["candidate_signal"])
    prod_trades = sum(1 for r in rows if r["production_signal"] != "no_trade")
    cand_trades = sum(1 for r in rows if r["candidate_signal"] != "no_trade")
    cand_only = sum(1 for r in rows
                    if r["candidate_signal"] != "no_trade" and r["production_signal"] == "no_trade")
    prod_only = sum(1 for r in rows
                    if r["production_signal"] != "no_trade" and r["candidate_signal"] == "no_trade")

    cand_conf = [max(r["candidate_probs"].values()) for r in rows]
    prod_conf = [max(r["production_probs"].values()) for r in rows]
    cand_tq = [r["candidate_trade_quality"] for r in rows]
    prod_tq = [r["production_trade_quality"] for r in rows]

    shadow_cfg = (cfg.get("retraining") or {}).get("shadow_ab_testing") or {}
    min_signals = int(shadow_cfg.get("min_shadow_signals", 30))

    gate_signals_ok = cand_trades >= min_signals
    alert_freq_safe = cand_trades <= max(1, prod_trades) * 3  # candidate not wildly more chatty

    summary = {
        "candidate_id": candidate_id,
        "rows": n,
        "signal_agreement_rate": agree / n,
        "production_trade_signals": prod_trades,
        "candidate_trade_signals": cand_trades,
        "candidate_only_signals": cand_only,
        "production_only_signals": prod_only,
        "avg_candidate_confidence": sum(cand_conf) / n,
        "avg_production_confidence": sum(prod_conf) / n,
        "avg_candidate_trade_quality": sum(cand_tq) / n,
        "avg_production_trade_quality": sum(prod_tq) / n,
        "promotion_gates": {
            "min_shadow_signals": min_signals,
            "candidate_signals_meet_minimum": gate_signals_ok,
            "alert_frequency_safe": alert_freq_safe,
        },
        "promotion_recommended_after_shadow": gate_signals_ok and alert_freq_safe,
    }
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.models.ab_compare")
    p.add_argument("--candidate", default="latest")
    p.add_argument("--sample", type=lambda s: s.strip().lower() in {"1", "true", "yes", "y", "t"},
                   default=False)
    args = p.parse_args(argv)

    root = repo_root()
    cfg = read_yaml(root / "configs" / "config.yaml")

    if args.candidate == "latest":
        run_dir = _find_latest_run_dir(root)
        candidate_id = run_dir.name if run_dir else None
    else:
        candidate_id = args.candidate
    if candidate_id is None:
        print("No candidate found.")
        return 2

    summary = compare(candidate_id, root=root, cfg=cfg)
    if summary is None:
        print(f"No shadow log for {candidate_id}. Run `python -m src.models.shadow "
              f"--candidate {candidate_id} --sample true` first.")
        return 2

    out_md = root / "reports" / "shadow" / f"shadow_compare_{candidate_id}.md"
    out_json = root / "reports" / "shadow" / f"shadow_compare_{candidate_id}.json"
    write_json(out_json, summary)

    lines = [
        f"# Shadow A/B comparison — {candidate_id}",
        f"generated: {datetime.now(tz=timezone.utc).isoformat()}",
        "",
        f"- shadow rows: {summary['rows']}",
        f"- signal agreement rate: {summary['signal_agreement_rate']:.3f}",
        f"- production trade signals: {summary['production_trade_signals']}",
        f"- candidate trade signals: {summary['candidate_trade_signals']}",
        f"- candidate-only signals: {summary['candidate_only_signals']}",
        f"- production-only signals: {summary['production_only_signals']}",
        f"- avg candidate confidence: {summary['avg_candidate_confidence']:.3f}",
        f"- avg production confidence: {summary['avg_production_confidence']:.3f}",
        f"- avg candidate trade-quality: {summary['avg_candidate_trade_quality']:.3f}",
        f"- avg production trade-quality: {summary['avg_production_trade_quality']:.3f}",
        "",
        "## Promotion gates (informational — promotion still requires explicit command)",
        "",
        f"- min shadow signals required: {summary['promotion_gates']['min_shadow_signals']}",
        f"- candidate meets minimum: {summary['promotion_gates']['candidate_signals_meet_minimum']}",
        f"- alert frequency safe: {summary['promotion_gates']['alert_frequency_safe']}",
        "",
        f"**promotion recommended after shadow: {summary['promotion_recommended_after_shadow']}**",
        "",
        "_Run `python -m src.models.promote --model-id "
        f"{candidate_id}` to promote (still gated)._",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"agreement rate: {summary['signal_agreement_rate']:.3f}")
    print(f"candidate trade signals: {summary['candidate_trade_signals']} "
          f"(production {summary['production_trade_signals']})")
    print(f"promotion recommended after shadow: {summary['promotion_recommended_after_shadow']}")
    print(f"reports:\n  {out_md.relative_to(root)}\n  {out_json.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
