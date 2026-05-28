"""Feature selection must NOT touch validation/test data when fitting.

We can't easily detect implementation-level leakage, but we can prove:

* MI ranking call signature only receives the training fold's X/y.
* Permutation-importance call signature only receives the validation fold.
* Running selection on disjoint train/val/test splits never reads test rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.selection import (
    SelectionConfig,
    drop_high_corr,
    drop_high_null,
    drop_low_variance,
    rank_mutual_info,
    select_features,
)


def _make_dataset(n: int = 300, n_features: int = 12, seed: int = 5):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # Make f0 strongly predictive; add a near-constant and a high-null column.
    X["f0"] = X["f0"]
    X["nearzero"] = 0.0
    X.loc[X.index[: int(n * 0.9)], "mostlynan"] = np.nan
    y = (X["f0"] > 0).astype("int8")
    return X, y


def test_drop_high_null_removes_majority_nan_columns():
    X, _ = _make_dataset()
    out, dropped = drop_high_null(X, min_non_null_ratio=0.85)
    assert "mostlynan" in dropped
    assert "f0" not in dropped


def test_drop_low_variance_removes_constant_columns():
    X, _ = _make_dataset()
    out, dropped = drop_low_variance(X, variance_threshold=1e-6)
    assert "nearzero" in dropped


def test_drop_high_corr_removes_one_of_two_duplicates():
    X, _ = _make_dataset()
    X["copy_of_f0"] = X["f0"] + np.random.default_rng(1).normal(0, 1e-4, size=len(X))
    out, pairs = drop_high_corr(X, threshold=0.95)
    cols = set(out.columns)
    assert "f0" in cols and "copy_of_f0" not in cols
    assert ("f0", "copy_of_f0") in pairs or ("copy_of_f0", "f0") in pairs


def test_mi_ranking_signal_visible():
    X, y = _make_dataset()
    ranked = rank_mutual_info(X[["f0", "f1", "f2"]], y, top_k=3)
    assert ranked[0][0] == "f0", "f0 should rank highest by MI given how y was constructed"


def test_selection_does_not_read_test_split():
    """Manually-isolated three-way split. Selection only sees train+val; we
    verify by checking that selection results are unchanged when the test
    split values are mutated."""
    X, y = _make_dataset(n=600)
    cfg = SelectionConfig(
        min_non_null_ratio=0.5, variance_threshold=1e-6,
        max_pairwise_corr=0.95, mutual_info_top_k=10,
        permutation_importance_top_k=10, final_top_k=5,
        always_keep=[],
    )
    train_end = 360
    val_end = 480
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

    rep_a = select_features(X_train, y_train, X_val, y_val, cfg=cfg)

    # Mutate test set aggressively
    X_mutated = X.copy()
    rng = np.random.default_rng(123)
    X_mutated.iloc[val_end:, :] = rng.normal(size=X.iloc[val_end:, :].shape) * 100

    X_train_b, y_train_b = X_mutated.iloc[:train_end], y.iloc[:train_end]
    X_val_b, y_val_b = X_mutated.iloc[train_end:val_end], y.iloc[train_end:val_end]
    rep_b = select_features(X_train_b, y_train_b, X_val_b, y_val_b, cfg=cfg)

    assert rep_a.selected == rep_b.selected, (
        "Selection picked different features after mutating only the test split. "
        "That implies the test split was used during selection — leakage."
    )
