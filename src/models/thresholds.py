"""Decision-threshold tuning on validation predictions only.

Given the model's softmax direction probabilities over the validation
window, pick (long_threshold, short_threshold, no_trade_threshold) that
maximise macro F1 while respecting per-class precision floors.

We never read test data — that's the leakage rule.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score


@dataclass
class ThresholdConfig:
    long_threshold: float
    short_threshold: float
    no_trade_threshold: float
    macro_f1_at_thresholds: float
    coverage_pct: float           # fraction of bars classified as something other than no_trade
    min_precision_floor: float


def _classify(
    probs: np.ndarray, *, long_thr: float, short_thr: float, no_trade_thr: float,
) -> np.ndarray:
    """Return per-bar class predictions; -1 means no_trade."""
    n = len(probs)
    out = np.full(n, -1, dtype=np.int64)
    top = probs.max(axis=1)
    argmax = probs.argmax(axis=1)
    # Apply per-class thresholds on top of the no_trade gate
    long_mask = (top >= no_trade_thr) & (argmax == 2) & (probs[:, 2] >= long_thr)
    short_mask = (top >= no_trade_thr) & (argmax == 0) & (probs[:, 0] >= short_thr)
    sideways_mask = (top >= no_trade_thr) & (argmax == 1)
    out[long_mask] = 2
    out[short_mask] = 0
    out[sideways_mask] = 1
    return out


def tune_thresholds(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    *,
    min_precision_per_trade_class: float = 0.45,
    grid: np.ndarray | None = None,
) -> ThresholdConfig:
    """Grid search over (long, short, no_trade) thresholds on the validation split."""
    if grid is None:
        grid = np.array([0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.65, 0.70])

    best: ThresholdConfig | None = None
    best_f1 = -1.0
    for no_trade_thr in grid:
        for long_thr in grid:
            if long_thr < no_trade_thr:
                continue
            for short_thr in grid:
                if short_thr < no_trade_thr:
                    continue
                preds = _classify(
                    val_probs,
                    long_thr=float(long_thr),
                    short_thr=float(short_thr),
                    no_trade_thr=float(no_trade_thr),
                )
                # Evaluate only the bars we DID predict (drop no_trade)
                mask = preds != -1
                if mask.sum() < max(20, len(y_val) // 20):
                    continue
                pred_subset = preds[mask]
                y_subset = y_val[mask]

                # Precision floor on long/short classes
                if len(np.unique(pred_subset)) < 2:
                    continue
                long_prec = precision_score(
                    y_subset, pred_subset, labels=[2], average="macro", zero_division=0,
                )
                short_prec = precision_score(
                    y_subset, pred_subset, labels=[0], average="macro", zero_division=0,
                )
                if min(long_prec, short_prec) < min_precision_per_trade_class:
                    continue

                f1 = f1_score(y_subset, pred_subset, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best = ThresholdConfig(
                        long_threshold=float(long_thr),
                        short_threshold=float(short_thr),
                        no_trade_threshold=float(no_trade_thr),
                        macro_f1_at_thresholds=float(f1),
                        coverage_pct=float(mask.mean() * 100),
                        min_precision_floor=float(min_precision_per_trade_class),
                    )

    if best is None:
        # Fallback to a safe default
        return ThresholdConfig(
            long_threshold=0.58, short_threshold=0.58, no_trade_threshold=0.58,
            macro_f1_at_thresholds=0.0, coverage_pct=0.0,
            min_precision_floor=float(min_precision_per_trade_class),
        )
    return best


def to_dict(cfg: ThresholdConfig) -> dict:
    return asdict(cfg)
