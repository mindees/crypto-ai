"""Population Stability Index (PSI) drift detection.

PSI compares a reference distribution (the training window) against a current
distribution (recent live bars). Standard interpretation:

* PSI < 0.10        → stable
* 0.10 ≤ PSI < 0.25 → moderate drift
* PSI ≥ 0.25        → significant drift

Used by retrain_check and drift_viz.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STABLE = "stable"
MODERATE = "moderate"
SIGNIFICANT = "significant"


def severity(psi: float) -> str:
    if psi < 0.10:
        return STABLE
    if psi < 0.25:
        return MODERATE
    return SIGNIFICANT


def compute_psi(reference: np.ndarray, current: np.ndarray, *, bins: int = 10) -> float:
    """PSI between two 1-D samples using quantile bins from the reference."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return 0.0

    # Quantile edges from reference; guard against duplicate edges (low-variance features).
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_hist / max(1, ref_hist.sum())
    cur_pct = cur_hist / max(1, cur_hist.sum())
    # Laplace smoothing to avoid div-by-zero / log(0)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    severity: str


def feature_drift_table(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    bins: int = 10,
) -> list[FeatureDrift]:
    cols = columns or [c for c in reference_df.columns
                       if c in current_df.columns
                       and pd.api.types.is_numeric_dtype(reference_df[c])]
    out: list[FeatureDrift] = []
    for c in cols:
        psi = compute_psi(reference_df[c].to_numpy(), current_df[c].to_numpy(), bins=bins)
        out.append(FeatureDrift(feature=c, psi=psi, severity=severity(psi)))
    out.sort(key=lambda d: d.psi, reverse=True)
    return out


def prediction_distribution_drift(
    ref_probs: np.ndarray, cur_probs: np.ndarray, *, n_classes: int = 3,
) -> list[FeatureDrift]:
    """PSI of each class's predicted-probability distribution."""
    out: list[FeatureDrift] = []
    names = ["down", "sideways", "up"][:n_classes]
    for i in range(n_classes):
        psi = compute_psi(ref_probs[:, i], cur_probs[:, i])
        out.append(FeatureDrift(feature=f"pred_{names[i]}", psi=psi, severity=severity(psi)))
    return out
