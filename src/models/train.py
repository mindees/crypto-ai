"""Multi-task training loop.

Responsibilities:

* Load the latest dataset from ``artifacts/datasets/<run_id>/...``.
* Build the model with shapes derived from the dataset spec.
* Compute class weights from training labels only.
* Compile with multi-head losses (direction/regime/cycle = sparse CE,
  trade_quality = binary CE) weighted per spec.
* Optionally enable mixed precision (``model.mixed_precision: true``).
* Auto-pick ``MirroredStrategy`` when ≥ 2 physical GPUs are detected.
* Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger.
* Save final model + training history + summary into
  ``artifacts/runs/<run_id>/``.

CPU smoke gate::

    python -m src.models.train --timeframe 1h --sample true --epochs 2
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.models.multitask_model import ModelConfig, build_model
from src.utils.io import read_yaml, repo_root, write_json
from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

_log = get_logger("models.train")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def find_latest_dataset_dir(root: Path) -> Path | None:
    base = root / "artifacts" / "datasets"
    if not base.exists():
        return None
    runs = sorted([p for p in base.iterdir() if p.is_dir()])
    return runs[-1] if runs else None


def load_combo(
    dataset_run_dir: Path, symbol: str, timeframe: str,
) -> tuple[dict[str, np.ndarray], dict] | None:
    out_dir = dataset_run_dir / symbol / timeframe
    if not (out_dir / "splits.npz").exists():
        return None
    arrays = dict(np.load(out_dir / "splits.npz"))
    spec = json.loads((out_dir / "dataset_spec.json").read_text("utf-8"))
    return arrays, spec


def stack_combos(per_combo: list[tuple[dict[str, np.ndarray], dict]]):
    """Concatenate train/val/test arrays across multiple (symbol, tf) combos."""
    def cat(name: str) -> np.ndarray | None:
        arrays_to_cat = [c[0][name] for c in per_combo if name in c[0]]
        if not arrays_to_cat:
            return None
        return np.concatenate(arrays_to_cat, axis=0)

    stacked: dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        for k in ("X_seq", "X_context", "asset_id", "tf_id",
                  "y_direction", "y_regime", "y_cycle", "y_trade_quality"):
            arr = cat(f"{k}_{split}")
            if arr is not None:
                stacked[f"{k}_{split}"] = arr
    return stacked


# ---------------------------------------------------------------------------
# Compile helpers
# ---------------------------------------------------------------------------

def class_weights_from(y: np.ndarray, num_classes: int) -> dict[int, float]:
    counts = np.bincount(y.astype(np.int64), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    raw = 1.0 / np.sqrt(counts)
    raw = raw / raw.mean()
    return {i: float(raw[i]) for i in range(num_classes)}


def _per_sample_direction_weights(y: np.ndarray, weights: dict[int, float]) -> np.ndarray:
    return np.array([weights[int(c)] for c in y], dtype=np.float32)


class DirectionMacroF1(tf.keras.metrics.Metric):
    """Macro F1 over the direction head. Aggregates per-class TP/FP/FN per batch."""

    def __init__(self, num_classes: int = 3, name="direction_macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name="tp", shape=(num_classes,), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(num_classes,), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(num_classes,), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred_cls = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        for c in range(self.num_classes):
            tp = tf.reduce_sum(tf.cast((y_true == c) & (y_pred_cls == c), tf.float32))
            fp = tf.reduce_sum(tf.cast((y_true != c) & (y_pred_cls == c), tf.float32))
            fn = tf.reduce_sum(tf.cast((y_true == c) & (y_pred_cls != c), tf.float32))
            self.tp.assign_add(tf.one_hot(c, self.num_classes) * tp)
            self.fp.assign_add(tf.one_hot(c, self.num_classes) * fp)
            self.fn.assign_add(tf.one_hot(c, self.num_classes) * fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return tf.reduce_mean(f1)

    def reset_state(self):
        for v in (self.tp, self.fp, self.fn):
            v.assign(tf.zeros_like(v))


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def _setup_strategy_and_precision(cfg: dict) -> tuple[tf.distribute.Strategy, str]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    for g in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except (RuntimeError, ValueError):
            pass

    use_multi = bool((cfg.get("model") or {}).get("use_multi_gpu_if_available", True))
    if use_multi and len(physical_gpus) >= 2:
        strat = tf.distribute.MirroredStrategy()
        _log.info("using MirroredStrategy with %d GPUs", len(physical_gpus))
    else:
        strat = tf.distribute.get_strategy()  # default (single device / CPU)
        _log.info("single-device strategy (%d GPUs detected)", len(physical_gpus))

    want_mixed = bool((cfg.get("model") or {}).get("mixed_precision", True))
    policy = "float32"
    if want_mixed and len(physical_gpus) >= 1:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            policy = "mixed_float16"
            _log.info("mixed precision enabled (mixed_float16)")
        except Exception as exc:  # noqa: BLE001
            _log.warning("could not enable mixed precision: %s", exc)
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
    return strat, policy


def _inputs_for(arrays: dict[str, np.ndarray], split: str, *, has_context: bool) -> list[np.ndarray]:
    """Order MUST match the order of inputs passed to keras.Model()."""
    inputs = [arrays[f"X_seq_{split}"]]
    if has_context:
        inputs.append(arrays[f"X_context_{split}"])
    inputs.append(arrays[f"asset_id_{split}"])
    inputs.append(arrays[f"tf_id_{split}"])
    return inputs


def _targets_for(arrays: dict[str, np.ndarray], split: str) -> list[np.ndarray]:
    return [
        arrays[f"y_direction_{split}"],
        arrays[f"y_regime_{split}"],
        arrays[f"y_cycle_{split}"],
        arrays[f"y_trade_quality_{split}"].astype(np.float32),
    ]


def build_and_compile(
    arrays: dict[str, np.ndarray],
    spec: dict,
    *,
    cfg: dict,
    learning_rate: float | None = None,
    use_mtf_fusion: bool = False,
) -> tf.keras.Model:
    model_cfg_dict = cfg.get("model") or {}
    heads_cfg = model_cfg_dict.get("heads") or {}
    transformer_cfg = model_cfg_dict.get("transformer") or {}

    mcfg = ModelConfig(
        seq_len=int(spec["seq_len"]),
        feature_count_seq=int(spec["feature_count_seq"]),
        feature_count_context=int(spec["feature_count_context"]),
        hidden_size=int(model_cfg_dict.get("hidden_size", 64)),
        num_transformer_layers=int(transformer_cfg.get("num_layers", 2)),
        num_heads=int(transformer_cfg.get("num_heads", 4)),
        ff_dim=int(transformer_cfg.get("ff_dim", 128)),
        dropout=float(transformer_cfg.get("dropout", 0.15)),
        attention_dropout=float(transformer_cfg.get("attention_dropout", 0.10)),
        regime_classes=len(spec.get("classes_regime", [])) or 6,
        cycle_classes=len(spec.get("classes_cycle", [])) or 4,
        use_multi_timeframe_fusion=use_mtf_fusion,
    )

    model = build_model(mcfg)
    optimizer_cfg = model_cfg_dict.get("optimizer") or {}
    lr = learning_rate or float(optimizer_cfg.get("learning_rate", 3e-4))
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=float(optimizer_cfg.get("weight_decay", 1e-4)),
        clipnorm=float(optimizer_cfg.get("clipnorm", 1.0)),
    )

    direction_w = float((heads_cfg.get("direction") or {}).get("loss_weight", 1.0))
    regime_w = float((heads_cfg.get("regime") or {}).get("loss_weight", 0.5))
    cycle_w = float((heads_cfg.get("cycle") or {}).get("loss_weight", 0.25))
    tq_w = float((heads_cfg.get("trade_quality") or {}).get("loss_weight", 0.75))

    model.compile(
        optimizer=optimizer,
        loss={
            "direction": "sparse_categorical_crossentropy",
            "regime": "sparse_categorical_crossentropy",
            "cycle": "sparse_categorical_crossentropy",
            "trade_quality": "binary_crossentropy",
        },
        loss_weights={
            "direction": direction_w,
            "regime": regime_w,
            "cycle": cycle_w,
            "trade_quality": tq_w,
        },
        metrics={
            "direction": ["accuracy", DirectionMacroF1(num_classes=3)],
            "regime": ["accuracy"],
            "cycle": ["accuracy"],
            "trade_quality": [tf.keras.metrics.AUC(name="auc")],
        },
    )
    return model


def fit_model(
    model: tf.keras.Model,
    arrays: dict[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    out_dir: Path,
    has_context: bool,
    class_weight_direction: dict[int, float] | None = None,
    monitor: str = "val_direction_direction_macro_f1",
    monitor_mode: str = "max",
    patience: int = 8,
    log_label: str = "phase1",
):
    X_tr = _inputs_for(arrays, "train", has_context=has_context)
    y_tr = _targets_for(arrays, "train")
    X_v = _inputs_for(arrays, "val", has_context=has_context) if "X_seq_val" in arrays else None
    y_v = _targets_for(arrays, "val") if "X_seq_val" in arrays else None
    sample_weight = None
    if class_weight_direction is not None:
        w = _per_sample_direction_weights(arrays["y_direction_train"], class_weight_direction)
        ones = np.ones_like(w)
        # Keras 3 requires sample_weight to match y's structure. y_tr is a list
        # of 4 arrays (direction, regime, cycle, trade_quality) — supply one
        # weight array per output in the same order, with ones for non-direction
        # outputs to avoid affecting their loss.
        sample_weight = [w, ones, ones, ones]

    out_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.CSVLogger(str(out_dir / f"{log_label}_history.csv")),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode=monitor_mode, patience=patience, min_delta=0.002,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode=monitor_mode, factor=0.5, patience=max(2, patience // 3),
            min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(out_dir / f"{log_label}_best.keras"),
            monitor=monitor, mode=monitor_mode, save_best_only=True, verbose=0,
        ),
    ]
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_v, y_v) if X_v is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        sample_weight=sample_weight,
        verbose=2,
    )
    return history


def save_artifacts(
    model: tf.keras.Model, spec: dict, history: tf.keras.callbacks.History,
    out_dir: Path, *, run_id: str, cfg: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.keras")
    (out_dir / "model_summary.txt").write_text(
        _stringify_summary(model), encoding="utf-8",
    )
    write_json(out_dir / "training_history.json", {
        k: [float(v) for v in vs] for k, vs in history.history.items()
    })
    write_json(out_dir / "dataset_spec.json", spec)
    write_json(out_dir / "class_indices.json", {
        "direction": {"0": "down", "1": "sideways", "2": "up"},
        "regime": {str(i): n for i, n in enumerate(spec.get("classes_regime", []))},
        "cycle": {str(i): n for i, n in enumerate(spec.get("classes_cycle", []))},
        "trade_quality": {"0": "bad_or_no_trade", "1": "good_trade"},
    })
    write_json(out_dir / "run_metadata.json", {
        "run_id": run_id,
        "saved_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "tf_version": tf.__version__,
        "model_config": (cfg.get("model") or {}),
    })


def _stringify_summary(model: tf.keras.Model) -> str:
    rows: list[str] = []
    model.summary(print_fn=rows.append)
    return "\n".join(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.models.train")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sample", type=lambda s: s.strip().lower() in {"1", "true", "yes", "y", "t"},
                   default=False)
    p.add_argument("--dataset-run-id", default=None,
                   help="Specific dataset run dir under artifacts/datasets/")
    args = p.parse_args(argv)

    root = repo_root()
    cfg = read_yaml(root / "configs" / "config.yaml")
    set_global_seed(int(cfg["project"]["seed"]))

    if args.dataset_run_id:
        ds_dir = root / "artifacts" / "datasets" / args.dataset_run_id
    else:
        ds_dir = find_latest_dataset_dir(root)
    if ds_dir is None or not ds_dir.exists():
        print("No dataset found in artifacts/datasets/. Run build_dataset first.")
        return 2
    print(f"loading dataset from {ds_dir.relative_to(root)}")

    per_combo: list[tuple[dict[str, np.ndarray], dict]] = []
    for symbol in args.symbols:
        loaded = load_combo(ds_dir, symbol, args.timeframe)
        if loaded is None:
            print(f"  skip {symbol}/{args.timeframe} (no dataset)")
            continue
        per_combo.append(loaded)
    if not per_combo:
        print("no combos loaded — aborting")
        return 2

    arrays = stack_combos(per_combo)
    spec = per_combo[0][1]
    print(
        f"train={arrays['X_seq_train'].shape[0]}  "
        f"val={arrays.get('X_seq_val', np.empty((0,))).shape[0]}  "
        f"test={arrays.get('X_seq_test', np.empty((0,))).shape[0]}  "
        f"seq_len={spec['seq_len']}  feature_count_seq={spec['feature_count_seq']}  "
        f"feature_count_context={spec['feature_count_context']}"
    )

    strat, precision_policy = _setup_strategy_and_precision(cfg)
    with strat.scope():
        model = build_and_compile(arrays, spec, cfg=cfg)
    print(f"precision policy: {precision_policy}")

    class_w = class_weights_from(arrays["y_direction_train"], num_classes=3)
    print(f"direction class weights: {class_w}")

    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = root / "artifacts" / "runs" / run_id
    has_context = spec["feature_count_context"] > 0
    history = fit_model(
        model, arrays,
        epochs=args.epochs, batch_size=args.batch_size,
        out_dir=out_dir, has_context=has_context,
        class_weight_direction=class_w,
        patience=max(2, args.epochs // 2),
        log_label="train",
    )
    save_artifacts(model, spec, history, out_dir, run_id=run_id, cfg=cfg)

    print(f"\nrun saved: {out_dir.relative_to(root)}")
    last_metrics = {k: float(v[-1]) for k, v in history.history.items()}
    print("final metrics:")
    for k in sorted(last_metrics):
        print(f"  {k}: {last_metrics[k]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
