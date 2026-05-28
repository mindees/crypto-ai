"""Produce the latest decision-support prediction per (symbol, timeframe).

Loads the production model (or the latest run), rebuilds the most recent
feature window using the SAME imputer/scaler that were fit at dataset-build
time, runs the model, applies tuned thresholds, attaches the rule-based
scorecard, and writes a prediction JSON.

Signals are intentionally hedged: long_bias / short_bias / no_trade /
range_wait / high_risk — never a hard "buy/sell".

CLI::

    python -m src.models.predict --latest --symbols BTCUSDT ETHUSDT --timeframes 1h 4h
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# IMPORTANT (Windows): import TensorFlow (via multitask_model) BEFORE pyarrow,
# or pyarrow's bundled DLLs break TF's native load. This import also registers
# the custom Keras layers needed by load_model.
from src.models import multitask_model  # noqa: F401,E402

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.features.scorecard import scorecard_for_row
from src.utils.io import read_json, read_yaml, repo_root, write_json
from src.utils.logging import get_logger

_log = get_logger("models.predict")

ASSET_TO_ID = {"BTCUSDT": 0, "ETHUSDT": 1}
TIMEFRAME_TO_ID = {"15m": 0, "1h": 1, "4h": 2, "1d": 3}

RISK_WARNING = "Decision-support only. Not financial advice. Validate manually before trading."


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


def _production_model_dir(root: Path) -> Path | None:
    """Honour artifacts/production/current_model.json when it points to a run."""
    ptr = root / "artifacts" / "production" / "current_model.json"
    if ptr.exists():
        data = read_json(ptr)
        ap = data.get("artifact_path")
        if ap:
            cand = root / ap
            if (cand / "model.keras").exists():
                return cand
    return None


def _load_features(symbol: str, timeframe: str, *, market: str, root: Path) -> pd.DataFrame | None:
    path = (
        root / "data" / "features"
        / f"source=binance" / f"market_type={market}"
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
    return df[~df.index.duplicated(keep="last")]


def _signal_from_probs(
    dir_probs: np.ndarray, tq_prob: float, thresholds: dict, *, governor_high_risk: bool,
) -> tuple[str, str]:
    down_p, side_p, up_p = (float(dir_probs[0]), float(dir_probs[1]), float(dir_probs[2]))
    top = max(down_p, side_p, up_p)
    long_thr = float(thresholds.get("long_threshold", 0.58))
    short_thr = float(thresholds.get("short_threshold", 0.58))
    no_trade_thr = float(thresholds.get("no_trade_threshold", 0.58))
    quality_thr = 0.60

    if governor_high_risk:
        return "high_risk", "funding/OI governor flagged elevated squeeze risk"
    if top < no_trade_thr:
        return "no_trade", "confidence below configured threshold"
    if side_p == top:
        return "range_wait", "model favours sideways/range"
    if up_p >= long_thr and tq_prob >= quality_thr:
        return "long_bias", "up probability and trade-quality above thresholds"
    if down_p >= short_thr and tq_prob >= quality_thr:
        return "short_bias", "down probability and trade-quality above thresholds"
    return "no_trade", "trade-quality below threshold"


def predict_combo(
    symbol: str, timeframe: str, *, model, model_id: str,
    dataset_dir: Path, run_dir: Path, market: str, root: Path,
) -> dict | None:
    combo_ds = dataset_dir / symbol / timeframe
    if not (combo_ds / "feature_schema.json").exists():
        _log.warning("no feature schema for %s/%s in dataset dir", symbol, timeframe)
        return None
    schema = read_json(combo_ds / "feature_schema.json")
    seq_cols = schema["seq_columns"]
    ctx_cols = schema["context_columns"]
    seq_len = int(schema["seq_len"])

    feats = _load_features(symbol, timeframe, market=market, root=root)
    if feats is None or len(feats) < seq_len:
        _log.warning("insufficient features for %s/%s (need %d)", symbol, timeframe, seq_len)
        return None

    seq_imputer = joblib.load(combo_ds / "seq_imputer.joblib")
    seq_scaler = joblib.load(combo_ds / "seq_scaler.joblib")
    has_ctx = bool(ctx_cols)
    ctx_imputer = joblib.load(combo_ds / "ctx_imputer.joblib") if has_ctx else None
    ctx_scaler = joblib.load(combo_ds / "ctx_scaler.joblib") if has_ctx else None

    window = feats.iloc[-seq_len:]
    for c in seq_cols:
        if c not in window.columns:
            window[c] = np.nan
    seq_raw = window[seq_cols].to_numpy(dtype=np.float64)
    seq_x = seq_scaler.transform(seq_imputer.transform(seq_raw)).astype(np.float32)
    seq_x = np.nan_to_num(seq_x, nan=0.0, posinf=0.0, neginf=0.0)[None, :, :]  # [1, L, F]

    inputs = [seq_x]
    if has_ctx:
        ctx_row = feats.iloc[[-1]]
        for c in ctx_cols:
            if c not in ctx_row.columns:
                ctx_row[c] = np.nan
        ctx_raw = ctx_row[ctx_cols].to_numpy(dtype=np.float64)
        ctx_x = ctx_scaler.transform(ctx_imputer.transform(ctx_raw)).astype(np.float32)
        ctx_x = np.nan_to_num(ctx_x, nan=0.0, posinf=0.0, neginf=0.0)
        inputs.append(ctx_x)
    inputs.append(np.array([ASSET_TO_ID.get(symbol, 0)], dtype=np.int32))
    inputs.append(np.array([TIMEFRAME_TO_ID.get(timeframe, 1)], dtype=np.int32))

    dir_p, reg_p, cyc_p, tq_p = model.predict(inputs, verbose=0)
    dir_p, reg_p, cyc_p = dir_p[0], reg_p[0], cyc_p[0]
    tq_prob = float(tq_p[0, 0])

    regime_classes = schema.get("regime_label_to_id") or {}
    cycle_classes = schema.get("cycle_label_to_id") or {}
    regime_names = {v: k for k, v in regime_classes.items()}
    cycle_names = {v: k for k, v in cycle_classes.items()}

    thresholds = {}
    if (run_dir / "threshold_config.json").exists():
        thresholds = read_json(run_dir / "threshold_config.json")

    latest_row = feats.iloc[-1]
    governor = latest_row.get("funding_oi_governor_risk")
    governor_high_risk = bool(governor == 1.0) if governor is not None and not pd.isna(governor) else False

    signal_action, signal_reason = _signal_from_probs(
        dir_p, tq_prob, thresholds, governor_high_risk=governor_high_risk,
    )
    sc = scorecard_for_row(latest_row)

    return {
        "timestamp_utc": feats.index[-1].isoformat(),
        "model_id": model_id,
        "asset": symbol,
        "timeframe": timeframe,
        "model_outputs": {
            "direction": {
                "down": float(dir_p[0]), "sideways": float(dir_p[1]), "up": float(dir_p[2]),
            },
            "regime": {
                "predicted": regime_names.get(int(reg_p.argmax()), "unknown"),
                "confidence": float(reg_p.max()),
            },
            "cycle": {
                "predicted": cycle_names.get(int(cyc_p.argmax()), "unknown"),
                "confidence": float(cyc_p.max()),
            },
            "trade_quality": {"probability": tq_prob},
        },
        "signal": {
            "action": signal_action,
            "reason": signal_reason,
            "long_threshold": float(thresholds.get("long_threshold", 0.58)),
            "short_threshold": float(thresholds.get("short_threshold", 0.58)),
            "quality_threshold": 0.60,
        },
        "scorecard": sc["trade_checklist"],
        "risk_warning": RISK_WARNING,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.models.predict")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--timeframes", nargs="+", default=["1h"])
    p.add_argument("--market", default="spot")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--sample", type=lambda s: s.strip().lower() in {"1", "true", "yes", "y", "t"},
                   default=False)
    p.add_argument("--model-run-id", default=None)
    p.add_argument("--dataset-run-id", default=None)
    args = p.parse_args(argv)

    root = repo_root()

    run_dir = None
    if args.model_run_id:
        run_dir = root / "artifacts" / "runs" / args.model_run_id
    else:
        run_dir = _production_model_dir(root) or _find_latest_run_dir(root)
    ds_dir = (root / "artifacts" / "datasets" / args.dataset_run_id) if args.dataset_run_id else _find_latest_dataset_dir(root)
    if run_dir is None or ds_dir is None:
        print("No model run / dataset found. Train first.")
        return 2
    model_id = run_dir.name
    print(f"model: {run_dir.relative_to(root)}")
    print(f"dataset: {ds_dir.relative_to(root)}")

    import tensorflow as tf
    model = tf.keras.models.load_model(run_dir / "model.keras", compile=False)

    predictions = []
    for symbol in args.symbols:
        for tf_ in args.timeframes:
            pred = predict_combo(
                symbol, tf_, model=model, model_id=model_id,
                dataset_dir=ds_dir, run_dir=run_dir, market=args.market, root=root,
            )
            if pred is None:
                print(f"  {symbol}/{tf_}: skipped (insufficient data / no schema)")
                continue
            predictions.append(pred)
            sig = pred["signal"]
            mo = pred["model_outputs"]["direction"]
            print(f"  {symbol}/{tf_}: {sig['action']:<11} "
                  f"down={mo['down']:.2f} side={mo['sideways']:.2f} up={mo['up']:.2f} "
                  f"tq={pred['model_outputs']['trade_quality']['probability']:.2f}")

    out_path = root / "reports" / "latest_predictions.json"
    write_json(out_path, {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "model_id": model_id,
        "predictions": predictions,
    })
    print(f"\nsaved: {out_path.relative_to(root)}  ({len(predictions)} predictions)")
    return 0 if predictions else 2


if __name__ == "__main__":
    sys.exit(main())
