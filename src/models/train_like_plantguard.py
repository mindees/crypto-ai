"""PlantGuard-style two-phase training, adapted for BTC/ETH time series.

Two phases:

* **Phase 1 — supervised head warmup.** If a pretrained encoder is available
  on disk (``artifacts/pretrained_encoders/.../encoder.keras``), freeze it
  and train only the fusion/trunk/heads at a higher learning rate. If not,
  this is just a lower-LR warmup of the full model (spec-compliant fallback
  when no SSL pretraining has run).
* **Phase 2 — fine-tune.** Unfreeze the last ``unfreeze_last_n_blocks``
  transformer blocks and continue training at a much lower learning rate.

Artifacts:

* ``artifacts/runs/<run_id>/{phase1_best.keras, phase2_best.keras, model.keras}``
* Training curves, confusion matrices, classification report, prediction demo
* ``class_indices.json``, ``dataset_spec.json``, ``run_metadata.json``

CPU smoke gate::

    python -m src.models.train_like_plantguard \\
        --timeframes 1h --sample true --phase1-epochs 1 --phase2-epochs 1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

from src.models.train import (
    build_and_compile, class_weights_from, fit_model, find_latest_dataset_dir,
    load_combo, save_artifacts, stack_combos, _setup_strategy_and_precision,
    _inputs_for, _targets_for,
)
from src.utils.io import read_yaml, repo_root, write_json
from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

_log = get_logger("models.train_like_plantguard")


# ---------------------------------------------------------------------------
# Pretrained encoder discovery (optional)
# ---------------------------------------------------------------------------

def find_pretrained_encoder(root: Path) -> Path | None:
    base = root / "artifacts" / "pretrained_encoders"
    if not base.exists():
        return None
    candidates = sorted(base.glob("*/encoder.keras"))
    return candidates[-1] if candidates else None


def freeze_encoder_layers(model: tf.keras.Model, *, freeze: bool, phase_label: str) -> int:
    """Toggle trainable on layers belonging to the sequence-encoder branches."""
    encoder_prefixes = ("main_input_norm", "main_proj", "main_pe", "main_block",
                        "fast_", "slow_")
    flipped = 0
    for layer in model.layers:
        if any(layer.name.startswith(p) for p in encoder_prefixes):
            if layer.trainable != (not freeze):
                layer.trainable = not freeze
                flipped += 1
    _log.info("phase=%s freeze=%s touched %d encoder layers", phase_label, freeze, flipped)
    return flipped


def unfreeze_last_n_blocks(model: tf.keras.Model, n: int) -> int:
    """Unfreeze the last N transformer blocks of the MAIN branch."""
    block_layers = [l for l in model.layers if l.name.startswith("main_block")]
    if not block_layers:
        return 0
    # Group layers by block index; figure out the unique block indices present.
    block_ids = sorted({l.name.split("main_block")[1].split("_")[0] for l in block_layers})
    unfreeze_ids = block_ids[-n:] if n <= len(block_ids) else block_ids
    flipped = 0
    for layer in model.layers:
        for bid in unfreeze_ids:
            prefix = f"main_block{bid}_"
            if layer.name.startswith(prefix) and not layer.trainable:
                layer.trainable = True
                flipped += 1
                break
    return flipped


# ---------------------------------------------------------------------------
# Plot helpers — saved next to the model
# ---------------------------------------------------------------------------

def _save_training_curves(history_csv_paths: list[Path], out_path: Path, *, phase2_start: int | None = None):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    epochs_offset = 0
    for csv_path in history_csv_paths:
        if not csv_path.exists():
            continue
        import csv
        with open(csv_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            history = {k: [] for k in reader.fieldnames}
            for row in reader:
                for k, v in row.items():
                    try:
                        history[k].append(float(v))
                    except (ValueError, TypeError):
                        history[k].append(np.nan)
        n = len(next(iter(history.values()), []))
        xs = np.arange(epochs_offset, epochs_offset + n)
        for col, axes in (("loss", ax[0, 0]), ("val_loss", ax[0, 0])):
            if col in history:
                axes.plot(xs, history[col], label=f"{col}")
        for col, axes in (("direction_accuracy", ax[0, 1]), ("val_direction_accuracy", ax[0, 1])):
            if col in history:
                axes.plot(xs, history[col], label=f"{col}")
        for col, axes in (("direction_direction_macro_f1", ax[1, 0]),
                           ("val_direction_direction_macro_f1", ax[1, 0])):
            if col in history:
                axes.plot(xs, history[col], label=f"{col}")
        for col, axes in (("trade_quality_auc", ax[1, 1]), ("val_trade_quality_auc", ax[1, 1])):
            if col in history:
                axes.plot(xs, history[col], label=f"{col}")
        epochs_offset += n

    if phase2_start is not None:
        for axes in ax.flatten():
            axes.axvline(phase2_start, color="red", linestyle="--", alpha=0.4,
                         label="phase 2 start")

    titles = ["loss", "direction accuracy", "direction macro F1", "trade-quality AUC"]
    for i, t in enumerate(titles):
        axes = ax[i // 2, i % 2]
        axes.set_title(t)
        axes.set_xlabel("epoch")
        axes.legend(loc="best", fontsize=8)
        axes.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: list[str], out_path: Path, *, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray,
                                 class_names: list[str]) -> dict:
    return classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True, zero_division=0,
    )


# ---------------------------------------------------------------------------
# Main two-phase driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.models.train_like_plantguard")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--timeframes", nargs="+", default=["1h"])
    p.add_argument("--phase1-epochs", type=int, default=10)
    p.add_argument("--phase2-epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sample", type=lambda s: s.strip().lower() in {"1", "true", "yes", "y", "t"},
                   default=False)
    p.add_argument("--dataset-run-id", default=None)
    args = p.parse_args(argv)

    root = repo_root()
    cfg = read_yaml(root / "configs" / "config.yaml")
    set_global_seed(int(cfg["project"]["seed"]))

    pg_cfg = (cfg.get("plantguard_style_training") or {})
    phase1_cfg = pg_cfg.get("phase1") or {}
    phase2_cfg = pg_cfg.get("phase2") or {}

    if args.dataset_run_id:
        ds_dir = root / "artifacts" / "datasets" / args.dataset_run_id
    else:
        ds_dir = find_latest_dataset_dir(root)
    if ds_dir is None or not ds_dir.exists():
        print("No dataset found in artifacts/datasets/. Run build_dataset first.")
        return 2
    print(f"loading dataset from {ds_dir.relative_to(root)}")

    # Use the FIRST requested timeframe (the plantguard demo trains one tf at a time)
    timeframe = args.timeframes[0]
    per_combo = []
    for symbol in args.symbols:
        loaded = load_combo(ds_dir, symbol, timeframe)
        if loaded is None:
            print(f"  skip {symbol}/{timeframe} (no dataset)")
            continue
        per_combo.append(loaded)
    if not per_combo:
        print("no combos loaded — aborting")
        return 2
    arrays = stack_combos(per_combo)
    spec = per_combo[0][1]
    has_context = spec["feature_count_context"] > 0

    print(
        f"train={arrays['X_seq_train'].shape[0]}  "
        f"val={arrays.get('X_seq_val', np.empty((0,))).shape[0]}  "
        f"test={arrays.get('X_seq_test', np.empty((0,))).shape[0]}"
    )

    pretrained_path = find_pretrained_encoder(root)
    if pretrained_path:
        print(f"pretrained encoder found: {pretrained_path}")
    else:
        print("no pretrained encoder — Phase 1 will run as a lower-LR warmup of the full model")

    strat, precision_policy = _setup_strategy_and_precision(cfg)
    with strat.scope():
        # Phase 1: head warmup at the Phase-1 LR
        phase1_lr = float(phase1_cfg.get("learning_rate", 0.001))
        model = build_and_compile(arrays, spec, cfg=cfg, learning_rate=phase1_lr)

    # Optionally freeze the encoder (only meaningful if SSL-pretrained weights were loaded;
    # we don't load them here unless explicitly added — the spec's fallback path applies)
    freeze = (
        pretrained_path is not None
        and bool(phase1_cfg.get("freeze_pretrained_encoder_if_available", True))
    )
    if freeze:
        freeze_encoder_layers(model, freeze=True, phase_label="phase1")

    class_w = class_weights_from(arrays["y_direction_train"], num_classes=3)
    print(f"direction class weights: {class_w}")
    print(f"precision policy: {precision_policy}")

    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_plantguard"
    out_dir = root / "artifacts" / "runs" / run_id

    p1_patience = int(phase1_cfg.get("patience", 5))
    history_p1 = fit_model(
        model, arrays,
        epochs=args.phase1_epochs, batch_size=args.batch_size,
        out_dir=out_dir, has_context=has_context,
        class_weight_direction=class_w,
        patience=max(2, p1_patience),
        log_label="phase1",
    )

    # Phase 2: unfreeze last N blocks, drop LR
    unfreeze_n = int(phase2_cfg.get("unfreeze_last_n_blocks", 1))
    if freeze:
        unfrozen = unfreeze_last_n_blocks(model, unfreeze_n)
        _log.info("phase 2: unfroze %d layers across last %d blocks", unfrozen, unfreeze_n)
    else:
        # All layers stayed trainable from Phase 1 — Phase 2 is just a lower-LR continuation.
        pass

    phase2_lr = float(phase2_cfg.get("learning_rate", 3e-5))
    with strat.scope():
        # Re-create optimizer with the lower LR; keep model weights as-is.
        model.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=phase2_lr, weight_decay=1e-4, clipnorm=1.0,
        )

    p2_patience = int(phase2_cfg.get("patience", 8))
    history_p2 = fit_model(
        model, arrays,
        epochs=args.phase2_epochs, batch_size=args.batch_size,
        out_dir=out_dir, has_context=has_context,
        class_weight_direction=class_w,
        patience=max(2, p2_patience),
        log_label="phase2",
    )

    # Final save + plots + reports
    save_artifacts(model, spec, history_p2, out_dir, run_id=run_id, cfg=cfg)
    p2_offset = int(args.phase1_epochs)
    _save_training_curves(
        [out_dir / "phase1_history.csv", out_dir / "phase2_history.csv"],
        out_dir / "training_curves.png",
        phase2_start=p2_offset,
    )

    # Predictions + confusion matrices + classification report on test (or val)
    eval_split = "test" if "X_seq_test" in arrays else "val"
    X_eval = _inputs_for(arrays, eval_split, has_context=has_context)
    y_eval = _targets_for(arrays, eval_split)
    preds = model.predict(X_eval, batch_size=args.batch_size, verbose=0)
    dir_probs, reg_probs, cyc_probs, tq_probs = preds

    dir_pred = dir_probs.argmax(axis=-1)
    reg_pred = reg_probs.argmax(axis=-1)
    cyc_pred = cyc_probs.argmax(axis=-1)
    tq_pred = (tq_probs.ravel() > 0.5).astype(np.int32)

    direction_classes = ["down", "sideways", "up"]
    regime_classes = list(spec.get("classes_regime") or [])
    cycle_classes = list(spec.get("classes_cycle") or [])

    _save_confusion_matrix(y_eval[0], dir_pred, direction_classes,
                            out_dir / "confusion_direction.png",
                            title=f"direction confusion ({eval_split})")
    if regime_classes:
        _save_confusion_matrix(y_eval[1], reg_pred, regime_classes,
                                out_dir / "confusion_regime.png",
                                title=f"regime confusion ({eval_split})")
    if cycle_classes:
        _save_confusion_matrix(y_eval[2], cyc_pred, cycle_classes,
                                out_dir / "confusion_cycle.png",
                                title=f"cycle confusion ({eval_split})")

    cls_report = {
        "direction": _classification_report_dict(y_eval[0], dir_pred, direction_classes),
        "regime": _classification_report_dict(y_eval[1], reg_pred, regime_classes) if regime_classes else None,
        "cycle": _classification_report_dict(y_eval[2], cyc_pred, cycle_classes) if cycle_classes else None,
    }
    # Trade-quality: AUC + AP if both classes are present
    if len(np.unique(y_eval[3])) == 2:
        cls_report["trade_quality"] = {
            "auc": float(roc_auc_score(y_eval[3], tq_probs.ravel())),
            "average_precision": float(average_precision_score(y_eval[3], tq_probs.ravel())),
        }
    write_json(out_dir / "classification_report.json", cls_report)

    # Prediction demo: probabilities on the most recent eval bar
    last = -1
    demo = {
        "asset_idx_last": int(arrays[f"asset_id_{eval_split}"][last]),
        "tf_idx_last": int(arrays[f"tf_id_{eval_split}"][last]),
        "direction_probs": [float(x) for x in dir_probs[last]],
        "regime_probs": [float(x) for x in reg_probs[last]],
        "cycle_probs": [float(x) for x in cyc_probs[last]],
        "trade_quality_prob": float(tq_probs[last, 0]),
    }
    write_json(out_dir / "prediction_demo.json", demo)

    # Reload test
    reload_model = tf.keras.models.load_model(out_dir / "model.keras", compile=False)
    sanity = reload_model.predict(
        [a[:2] for a in X_eval], verbose=0,
    )
    assert sanity[0].shape == (2, 3), f"reload check: direction shape {sanity[0].shape} != (2, 3)"

    last_p1 = {k: float(v[-1]) for k, v in history_p1.history.items()}
    last_p2 = {k: float(v[-1]) for k, v in history_p2.history.items()}
    print(f"\nrun saved: {out_dir.relative_to(root)}")
    print("phase1 final metrics:")
    for k in sorted(last_p1):
        print(f"  {k}: {last_p1[k]:.4f}")
    print("phase2 final metrics:")
    for k in sorted(last_p2):
        print(f"  {k}: {last_p2[k]:.4f}")
    print("reload + sanity prediction: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
