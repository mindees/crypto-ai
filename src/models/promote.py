"""Promote a candidate model to production — gated, never silent.

Promotion gates (from config ``retraining.promotion``):

1. Beats current production on direction macro F1 by ≥ min improvement.
2. Positive expectancy after fees/slippage (when backtest metrics exist).
3. Backtest profit factor ≥ minimum.
4. Max drawdown not worse than production by more than the configured pct.
5. (Data-coverage / calibration checks are placeholders until those metrics
   are recorded on the registry record.)

If there is no current production model, the first candidate that has metrics
is eligible (the "cold start" case).

CLI::

    python -m src.models.promote --latest --dry-run
    python -m src.models.promote --model-id <id>
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

from src.models import registry
from src.utils.io import read_yaml, repo_root
from src.utils.logging import get_logger

_log = get_logger("models.promote")


@dataclass
class PromotionDecision:
    candidate_id: str
    production_id: str | None
    approved: bool = False
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)


def evaluate_promotion(candidate: dict, production: dict | None, *, cfg: dict) -> PromotionDecision:
    promo = (cfg.get("retraining") or {}).get("promotion") or {}
    min_f1_impr = float(promo.get("min_direction_macro_f1_improvement", 0.01))
    min_pf = float(promo.get("min_backtest_profit_factor", 1.10))
    max_dd_worse = float(promo.get("max_drawdown_not_worse_by_pct", 10))
    require_beat = bool(promo.get("require_beats_current_production", True))
    require_pos_exp = bool(promo.get("require_positive_expectancy", True))

    cm = candidate.get("metrics", {})
    decision = PromotionDecision(
        candidate_id=candidate["model_id"],
        production_id=production["model_id"] if production else None,
    )

    # Gate 1: beats production macro F1
    if production is None:
        decision.checks["beats_production_macro_f1"] = True
        decision.reasons.append("cold start: no production model, candidate eligible")
    else:
        pm = production.get("metrics", {})
        impr = cm.get("direction_macro_f1", 0.0) - pm.get("direction_macro_f1", 0.0)
        ok = (impr >= min_f1_impr) if require_beat else True
        decision.checks["beats_production_macro_f1"] = ok
        decision.reasons.append(
            f"macro F1 improvement {impr:+.4f} vs required {min_f1_impr:+.4f}: {'PASS' if ok else 'FAIL'}"
        )

    # Gate 2: positive expectancy (only enforced when backtest metrics exist)
    exp_r = cm.get("expectancy_r", 0.0)
    pf = cm.get("backtest_profit_factor", 0.0)
    has_backtest = (pf > 0.0) or (exp_r != 0.0)
    if require_pos_exp and has_backtest:
        ok = exp_r > 0
        decision.checks["positive_expectancy"] = ok
        decision.reasons.append(f"expectancy {exp_r:+.4f}R: {'PASS' if ok else 'FAIL'}")
    else:
        decision.checks["positive_expectancy"] = True
        decision.reasons.append("expectancy gate skipped (no backtest metrics recorded yet)")

    # Gate 3: profit factor
    if has_backtest:
        ok = pf >= min_pf
        decision.checks["profit_factor"] = ok
        decision.reasons.append(f"profit factor {pf:.2f} vs required {min_pf:.2f}: {'PASS' if ok else 'FAIL'}")
    else:
        decision.checks["profit_factor"] = True
        decision.reasons.append("profit-factor gate skipped (no backtest metrics)")

    # Gate 4: drawdown not materially worse
    if production is not None and has_backtest:
        cand_dd = abs(cm.get("max_drawdown_pct", 0.0))
        prod_dd = abs(production.get("metrics", {}).get("max_drawdown_pct", 0.0))
        ok = cand_dd <= prod_dd * (1 + max_dd_worse / 100.0) or prod_dd == 0.0
        decision.checks["drawdown_acceptable"] = ok
        decision.reasons.append(
            f"max drawdown {cand_dd:.2f}% vs prod {prod_dd:.2f}% (+{max_dd_worse}% allowance): "
            f"{'PASS' if ok else 'FAIL'}"
        )
    else:
        decision.checks["drawdown_acceptable"] = True
        decision.reasons.append("drawdown gate skipped")

    decision.approved = all(decision.checks.values())
    return decision


def apply_promotion(decision: PromotionDecision, *, root=None) -> None:
    """Set candidate→production, demote old production→archived, update pointer."""
    if not decision.approved:
        registry.set_status(decision.candidate_id, "candidate", root=root,
                            promotion_decision="rejected",
                            note="; ".join(decision.reasons))
        return
    if decision.production_id and decision.production_id != decision.candidate_id:
        registry.set_status(decision.production_id, "archived", root=root,
                            note="archived on promotion of " + decision.candidate_id)
    registry.set_status(decision.candidate_id, "production", root=root,
                        promotion_decision="approved",
                        note="; ".join(decision.reasons))
    registry.set_current_model_pointer(decision.candidate_id, root=root)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.models.promote")
    p.add_argument("--latest", action="store_true", help="promote the latest candidate")
    p.add_argument("--model-id", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    root = repo_root()
    cfg = read_yaml(root / "configs" / "config.yaml")
    registry.sync_runs(root)

    if args.model_id:
        candidate = registry.get_model(args.model_id, root=root)
    else:
        candidate = registry.latest_candidate(root)
    if candidate is None:
        print("No candidate model to promote. Train + register a model first.")
        return 2

    production = registry.get_production(root)
    decision = evaluate_promotion(candidate, production, cfg=cfg)

    print(f"candidate:  {decision.candidate_id}")
    print(f"production: {decision.production_id or '(none)'}")
    print("checks:")
    for name, ok in decision.checks.items():
        print(f"  [{'x' if ok else ' '}] {name}")
    print("reasons:")
    for r in decision.reasons:
        print(f"  - {r}")
    print(f"\ndecision: {'APPROVE' if decision.approved else 'REJECT'}")

    if args.dry_run:
        print("(dry-run — no registry changes applied)")
        return 0

    apply_promotion(decision, root=root)
    print("applied." if decision.approved else "candidate left as-is (rejected).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
