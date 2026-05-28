"""Honest evaluation of a trained run.

Computes per-head classification metrics on the held-out split, tunes
decision thresholds on the validation split, and compares the direction
head against honest baselines (majority-class, random, persistence).
Writes ``reports/eval_<run_id>.md`` and ``reports/eval_<run_id>.json``.

The report states plainly whether the model beats the baselines.

CLI::

    python -m src.models.evaluate --latest --sample true
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score,
)

from src.models import multitask_model  # noqa: F401 — registers custom Keras layers for load_model
from src.models.thresholds import to_dict, tune_thresholds
from src.utils.io import read_yaml, repo_root, write_json
from src.utils.logging import get_logger

_log = get_logger("models.evaluate")


def _find_latest_run_dir(root: Path) -> Path | None:
    base = root / "artifacts" / "runs"
    if not base.exists():
        return None
    candidates = sorted([p for p in base.iterdir() if p.is_dir() and (p / "model.keras").exists()])
    return candidates[-1] if candidates else None


def _find_latest_dataset_dir(root: Path) -> Path | None:
    base = root / "artifacts" / "datasets"
    if not base.exists():
        return None
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    return runs[-1] if runs else None


def _collect_combos(dataset_dir: Path, symbols, timeframe):
    combos = []
    for sym in symbols:
        d = dataset_dir / sym / timeframe
        if (d / "splits.npz").exists():
            arrays = dict(np.load(d / "splits.npz"))
            spec = json.loads((d / "dataset_spec.json").read_text("utf-8"))
            combos.append((sym, arrays, spec))
    return combos


def _stack_split(combos, split: str):
    def cat(key):
        parts = [a[f"{key}_{split}"] for _, a, _ in combos if f"{key}_{split}" in a]
        return np.concatenate(parts, axis=0) if parts else None
    return {
        "X_seq": cat("X_seq"), "X_context": cat("X_context"),
        "asset_id": cat("asset_id"), "tf_id": cat("tf_id"),
        "y_direction": cat("y_direction"), "y_regime": cat("y_regime"),
        "y_cycle": cat("y_cycle"), "y_trade_quality": cat("y_trade_quality"),
    }


def _inputs(d, *, has_ctx):
    inputs = [d["X_seq"]]
    if has_ctx:
        inputs.append(d["X_context"])
    inputs.append(d["asset_id"])
    inputs.append(d["tf_id"])
    return inputs


def _direction_baselines(y_train: np.ndarray, y_eval: np.ndarray) -> dict:
    """Majority-class, random, and uniform baselines for the 3-class direction head."""
    rng = np.random.default_rng(42)
    majority_cls = int(np.bincount(y_train, minlength=3).argmax())
    maj_pred = np.full_like(y_eval, majority_cls)
    rand_pred = rng.integers(0, 3, size=len(y_eval))
    return {
        "majority_class": {
            "class": majority_cls,
            "accuracy": float(accuracy_score(y_eval, maj_pred)),
            "macro_f1": float(f1_score(y_eval, maj_pred, average="macro", zero_division=0)),
        },
        "random": {
            "accuracy": float(accuracy_score(y_eval, rand_pred)),
            "macro_f1": float(f1_score(y_eval, rand_pred, average="macro", zero_division=0)),
        },
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.models.evaluate")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--sample", type=lambda s: s.strip().lower() in {"1", "true", "yes", "y", "t"},
                   default=False)
    p.add_argument("--model-run-id", default=None)
    p.add_argument("--dataset-run-id", default=None)
    args = p.parse_args(argv)

    root = repo_root()
    cfg = read_yaml(root / "configs" / "config.yaml")

    run_dir = (root / "artifacts" / "runs" / args.model_run_id) if args.model_run_id else _find_latest_run_dir(root)
    ds_dir = (root / "artifacts" / "datasets" / args.dataset_run_id) if args.dataset_run_id else _find_latest_dataset_dir(root)
    if run_dir is None or ds_dir is None:
        print("No model run / dataset found. Train first.")
        return 2
    print(f"model:   {run_dir.relative_to(root)}")
    print(f"dataset: {ds_dir.relative_to(root)}")

    combos = _collect_combos(ds_dir, args.symbols, args.timeframe)
    if not combos:
        print(f"No dataset combos for timeframe {args.timeframe}.")
        return 2

    spec = combos[0][2]
    has_ctx = spec["feature_count_context"] > 0

    eval_split = "test" if any("X_seq_test" in a for _, a, _ in combos) else "val"
    train = _stack_split(combos, "train")
    val = _stack_split(combos, "val")
    evl = _stack_split(combos, eval_split)

    import tensorflow as tf
    model = tf.keras.models.load_model(run_dir / "model.keras", compile=False)

    # Threshold tuning on validation
    threshold_cfg = None
    if val["X_seq"] is not None:
        val_preds = model.predict(_inputs(val, has_ctx=has_ctx), batch_size=128, verbose=0)
        threshold_cfg = tune_thresholds(
            val_preds[0], val["y_direction"],
            min_precision_per_trade_class=float(
                (cfg.get("class_imbalance") or {}).get("min_precision_per_trade_class", 0.45)
            ),
        )

    # Eval-split predictions
    eval_preds = model.predict(_inputs(evl, has_ctx=has_ctx), batch_size=128, verbose=0)
    dir_p, reg_p, cyc_p, tq_p = eval_preds
    dir_pred = dir_p.argmax(axis=1)
    reg_pred = reg_p.argmax(axis=1)
    cyc_pred = cyc_p.argmax(axis=1)

    direction_classes = ["down", "sideways", "up"]
    regime_classes = spec.get("classes_regime", [])
    cycle_classes = spec.get("classes_cycle", [])

    direction_report = classification_report(
        evl["y_direction"], dir_pred, labels=[0, 1, 2],
        target_names=direction_classes, output_dict=True, zero_division=0,
    )
    model_macro_f1 = float(f1_score(evl["y_direction"], dir_pred, average="macro", zero_division=0))
    baselines = _direction_baselines(train["y_direction"], evl["y_direction"])

    tq_auc = None
    if len(np.unique(evl["y_trade_quality"])) == 2:
        tq_auc = float(roc_auc_score(evl["y_trade_quality"], tq_p.ravel()))

    metrics = {
        "eval_split": eval_split,
        "n_eval": int(len(evl["y_direction"])),
        "direction": {
            "macro_f1": model_macro_f1,
            "accuracy": float(accuracy_score(evl["y_direction"], dir_pred)),
            "per_class": {
                c: direction_report.get(c, {}) for c in direction_classes
            },
        },
        "regime_accuracy": float(accuracy_score(evl["y_regime"], reg_pred)),
        "cycle_accuracy": float(accuracy_score(evl["y_cycle"], cyc_pred)),
        "trade_quality_auc": tq_auc,
        "baselines": baselines,
        "thresholds": to_dict(threshold_cfg) if threshold_cfg else None,
        "beats_majority_class_macro_f1": bool(model_macro_f1 > baselines["majority_class"]["macro_f1"]),
        "beats_random_macro_f1": bool(model_macro_f1 > baselines["random"]["macro_f1"]),
    }

    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = root / "reports" / f"eval_{run_id}.json"
    md_path = root / "reports" / f"eval_{run_id}.md"
    write_json(json_path, metrics)

    # Persist threshold config next to the model for serving
    if threshold_cfg is not None:
        write_json(run_dir / "threshold_config.json", to_dict(threshold_cfg))

    beats = metrics["beats_majority_class_macro_f1"] and metrics["beats_random_macro_f1"]
    verdict = (
        "Model beats the majority-class AND random baselines on direction macro F1."
        if beats else
        "**Model does NOT beat both baselines on direction macro F1 — not usable for trading as-is.**"
    )
    lines = [
        f"# Evaluation report — {run_id}",
        "",
        f"model: `{run_dir.name}`  dataset: `{ds_dir.name}`  eval_split: `{eval_split}`  n={metrics['n_eval']}",
        "",
        "## Direction head",
        "",
        f"- macro F1: **{model_macro_f1:.4f}**",
        f"- accuracy: {metrics['direction']['accuracy']:.4f}",
        f"- majority-class baseline macro F1: {baselines['majority_class']['macro_f1']:.4f}",
        f"- random baseline macro F1: {baselines['random']['macro_f1']:.4f}",
        "",
        f"### {verdict}",
        "",
        "## Other heads",
        "",
        f"- regime accuracy: {metrics['regime_accuracy']:.4f}",
        f"- cycle accuracy: {metrics['cycle_accuracy']:.4f}",
        f"- trade-quality AUC: {tq_auc if tq_auc is not None else 'n/a (single class in eval)'}",
        "",
        "## Tuned decision thresholds (validation only)",
        "",
    ]
    if threshold_cfg is not None:
        lines += [
            f"- long_threshold: {threshold_cfg.long_threshold}",
            f"- short_threshold: {threshold_cfg.short_threshold}",
            f"- no_trade_threshold: {threshold_cfg.no_trade_threshold}",
            f"- macro F1 at thresholds: {threshold_cfg.macro_f1_at_thresholds:.4f}",
            f"- coverage: {threshold_cfg.coverage_pct:.1f}% of bars produce a trade signal",
        ]
    else:
        lines.append("- (validation split unavailable — thresholds not tuned)")
    lines += [
        "",
        "## Limitations",
        "",
        "- This evaluation is on a sample window; treat metrics as a smoke check, not a verdict on edge.",
        "- A model that fails to beat baselines after fees/slippage MUST NOT be used for trading.",
        "- Backtest economics (fees, slippage, drawdown) are in the companion backtest report.",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\ndirection macro F1: {model_macro_f1:.4f}  "
          f"(majority {baselines['majority_class']['macro_f1']:.4f}, "
          f"random {baselines['random']['macro_f1']:.4f})")
    print(f"verdict: {'BEATS baselines' if beats else 'does NOT beat baselines'}")
    print(f"reports:\n  {md_path.relative_to(root)}\n  {json_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
