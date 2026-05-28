"""Shadow mode — run a candidate model beside production without affecting it.

The production model produces the official signal. The candidate receives the
SAME feature windows; its predictions are logged but never drive alerts or
trades. Both are paper-traded on identical bars for later A/B comparison.

Outputs:

* ``reports/shadow/shadow_predictions_<candidate_id>.jsonl``
* ``reports/shadow/shadow_paper_trades_<candidate_id>.csv``

CLI::

    python -m src.models.shadow --candidate latest --sample true
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# TF before pyarrow (Windows DLL ordering) + registers custom layers.
from src.models import multitask_model  # noqa: F401,E402

import numpy as np

from src.backtest.strategies import model_signal
from src.models import registry
from src.utils.io import read_json, repo_root
from src.utils.logging import get_logger

_log = get_logger("models.shadow")


def _find_latest_run_dir(root: Path) -> Path | None:
    base = root / "artifacts" / "runs"
    if not base.exists():
        return None
    cands = sorted([p for p in base.iterdir() if p.is_dir() and (p / "model.keras").exists()])
    return cands[-1] if cands else None


def _find_latest_dataset_dir(root: Path) -> Path | None:
    base = root / "artifacts" / "datasets"
    if not base.exists():
        return None
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    return runs[-1] if runs else None


def _load_split(combo_dir: Path):
    if not (combo_dir / "splits.npz").exists():
        return None, None
    arrays = dict(np.load(combo_dir / "splits.npz"))
    spec = json.loads((combo_dir / "dataset_spec.json").read_text("utf-8"))
    return arrays, spec


def _inputs(arrays, split, *, has_ctx):
    inp = [arrays[f"X_seq_{split}"]]
    if has_ctx:
        inp.append(arrays[f"X_context_{split}"])
    inp.append(arrays[f"asset_id_{split}"])
    inp.append(arrays[f"tf_id_{split}"])
    return inp


def run_shadow(
    *, candidate_id: str, production_id: str | None, symbols, timeframe,
    dataset_dir: Path, root: Path, sample: bool, thresholds: dict,
) -> tuple[Path, Path, dict]:
    import tensorflow as tf

    run_base = root / "artifacts" / "runs"
    cand_model = tf.keras.models.load_model(run_base / candidate_id / "model.keras", compile=False)
    prod_model = (
        tf.keras.models.load_model(run_base / production_id / "model.keras", compile=False)
        if production_id and (run_base / production_id / "model.keras").exists()
        else cand_model
    )

    shadow_dir = root / "reports" / "shadow"
    shadow_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = shadow_dir / f"shadow_predictions_{candidate_id}.jsonl"
    trades_path = shadow_dir / f"shadow_paper_trades_{candidate_id}.csv"

    long_thr = float(thresholds.get("long_threshold", 0.58))
    short_thr = float(thresholds.get("short_threshold", 0.58))
    no_trade_thr = float(thresholds.get("no_trade_threshold", 0.58))

    n_rows = 0
    agreement = 0
    prod_signals = {"long_bias": 0, "short_bias": 0, "no_trade": 0}
    cand_signals = {"long_bias": 0, "short_bias": 0, "no_trade": 0}

    with open(jsonl_path, "w", encoding="utf-8") as jf, \
         open(trades_path, "w", newline="", encoding="utf-8") as tf_csv:
        tw = csv.writer(tf_csv)
        tw.writerow(["timestamp_idx", "asset", "timeframe", "which", "signal",
                     "up_p", "down_p", "trade_quality"])
        for symbol in symbols:
            combo_dir = dataset_dir / symbol / timeframe
            arrays, spec = _load_split(combo_dir)
            if arrays is None:
                continue
            has_ctx = spec["feature_count_context"] > 0
            split = "test" if "X_seq_test" in arrays else "val"
            inp = _inputs(arrays, split, has_ctx=has_ctx)
            if sample:
                inp = [a[-200:] for a in inp]

            prod_pred = prod_model.predict(inp, verbose=0)
            cand_pred = cand_model.predict(inp, verbose=0)
            prod_dir, _, _, prod_tq = prod_pred
            cand_dir, _, _, cand_tq = cand_pred

            for i in range(len(prod_dir)):
                p_sig = model_signal(
                    direction_probs=prod_dir[i], trade_quality_prob=float(prod_tq[i, 0]),
                    long_threshold=long_thr, short_threshold=short_thr,
                    no_trade_threshold=no_trade_thr,
                )
                c_sig = model_signal(
                    direction_probs=cand_dir[i], trade_quality_prob=float(cand_tq[i, 0]),
                    long_threshold=long_thr, short_threshold=short_thr,
                    no_trade_threshold=no_trade_thr,
                )
                p_action = {1: "long_bias", -1: "short_bias", 0: "no_trade"}[p_sig.side]
                c_action = {1: "long_bias", -1: "short_bias", 0: "no_trade"}[c_sig.side]
                prod_signals[p_action] += 1
                cand_signals[c_action] += 1
                if p_action == c_action:
                    agreement += 1
                n_rows += 1

                jf.write(json.dumps({
                    "timestamp_idx": i,
                    "asset": symbol,
                    "timeframe": timeframe,
                    "production_model_id": production_id or candidate_id,
                    "candidate_model_id": candidate_id,
                    "production_signal": p_action,
                    "candidate_signal": c_action,
                    "production_probs": {
                        "down": float(prod_dir[i, 0]), "sideways": float(prod_dir[i, 1]),
                        "up": float(prod_dir[i, 2]),
                    },
                    "candidate_probs": {
                        "down": float(cand_dir[i, 0]), "sideways": float(cand_dir[i, 1]),
                        "up": float(cand_dir[i, 2]),
                    },
                    "production_trade_quality": float(prod_tq[i, 0]),
                    "candidate_trade_quality": float(cand_tq[i, 0]),
                }) + "\n")
                tw.writerow([i, symbol, timeframe, "production", p_action,
                             float(prod_dir[i, 2]), float(prod_dir[i, 0]), float(prod_tq[i, 0])])
                tw.writerow([i, symbol, timeframe, "candidate", c_action,
                             float(cand_dir[i, 2]), float(cand_dir[i, 0]), float(cand_tq[i, 0])])

    summary = {
        "candidate_id": candidate_id,
        "production_id": production_id or "(none — candidate mirrored)",
        "rows": n_rows,
        "signal_agreement_rate": (agreement / n_rows) if n_rows else 0.0,
        "production_signal_counts": prod_signals,
        "candidate_signal_counts": cand_signals,
    }
    return jsonl_path, trades_path, summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.models.shadow")
    p.add_argument("--candidate", default="latest", help="candidate run_id or 'latest'")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--sample", type=lambda s: s.strip().lower() in {"1", "true", "yes", "y", "t"},
                   default=False)
    args = p.parse_args(argv)

    root = repo_root()
    registry.sync_runs(root)

    if args.candidate == "latest":
        run_dir = _find_latest_run_dir(root)
        candidate_id = run_dir.name if run_dir else None
    else:
        candidate_id = args.candidate
    if candidate_id is None:
        print("No candidate model found. Train one first.")
        return 2

    prod = registry.get_production(root)
    production_id = prod["model_id"] if prod else None

    ds_dir = _find_latest_dataset_dir(root)
    if ds_dir is None:
        print("No dataset found.")
        return 2

    thresholds = {}
    run_dir = root / "artifacts" / "runs" / candidate_id
    if (run_dir / "threshold_config.json").exists():
        thresholds = read_json(run_dir / "threshold_config.json")

    jsonl_path, trades_path, summary = run_shadow(
        candidate_id=candidate_id, production_id=production_id,
        symbols=args.symbols, timeframe=args.timeframe,
        dataset_dir=ds_dir, root=root, sample=args.sample, thresholds=thresholds,
    )

    print(f"candidate:  {summary['candidate_id']}")
    print(f"production: {summary['production_id']}")
    print(f"shadow rows: {summary['rows']}")
    print(f"signal agreement rate: {summary['signal_agreement_rate']:.3f}")
    print(f"production signals: {summary['production_signal_counts']}")
    print(f"candidate signals:  {summary['candidate_signal_counts']}")
    print(f"\nlogs:\n  {jsonl_path.relative_to(root)}\n  {trades_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
