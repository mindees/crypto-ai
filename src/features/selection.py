"""Feature selection pipeline (causal, fold-aware).

Pipeline (per spec):

1. Drop features with too many missing values (``min_non_null_ratio``).
2. Drop near-zero-variance features (``variance_threshold``).
3. Drop one feature from each highly correlated pair (``max_pairwise_corr``).
4. Rank by mutual information on the **training window only**.
5. Rank by permutation importance on the **validation window only** using a
   lightweight gradient-boosted baseline.
6. Keep ``always_keep`` features regardless.
7. Take top ``final_top_k``.
8. Persist the selected feature list per fold.

The fold-aware contract is enforced by callers: they pass ``train_idx`` /
``val_idx`` boolean masks. Selection NEVER touches the test set.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier

from src.utils.io import read_yaml, repo_root, write_json
from src.utils.logging import get_logger

_log = get_logger("features.selection")


@dataclass
class SelectionConfig:
    min_non_null_ratio: float = 0.85
    variance_threshold: float = 1e-6
    max_pairwise_corr: float = 0.95
    mutual_info_top_k: int = 160
    permutation_importance_top_k: int = 140
    final_top_k: int = 120
    always_keep: list[str] = field(default_factory=list)


@dataclass
class SelectionReport:
    selected: list[str]
    dropped_nulls: list[str]
    dropped_variance: list[str]
    dropped_correlation: list[tuple[str, str]]
    mi_ranking: list[tuple[str, float]]
    permutation_ranking: list[tuple[str, float]]


def load_config(root: Path | None = None) -> SelectionConfig:
    root = root or repo_root()
    cfg = (read_yaml(root / "configs" / "config.yaml").get("features") or {})
    return SelectionConfig(
        min_non_null_ratio=float(cfg.get("min_non_null_ratio", 0.85)),
        variance_threshold=float(cfg.get("variance_threshold", 1e-6)),
        max_pairwise_corr=float(cfg.get("max_pairwise_corr", 0.95)),
        mutual_info_top_k=int(cfg.get("mutual_info_top_k", 160)),
        permutation_importance_top_k=int(cfg.get("permutation_importance_top_k", 140)),
        final_top_k=int(cfg.get("final_top_k", 120)),
        always_keep=list(cfg.get("always_keep") or []),
    )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def drop_high_null(X: pd.DataFrame, min_non_null_ratio: float) -> tuple[pd.DataFrame, list[str]]:
    keep_mask = X.notna().mean() >= min_non_null_ratio
    dropped = [c for c, k in keep_mask.items() if not k]
    return X.loc[:, keep_mask], dropped


def drop_low_variance(X: pd.DataFrame, variance_threshold: float) -> tuple[pd.DataFrame, list[str]]:
    numeric = X.select_dtypes(include="number")
    vars_ = numeric.var(ddof=0)
    keep = (vars_ > variance_threshold)
    dropped = [c for c, k in keep.items() if not k]
    return X.drop(columns=dropped), dropped


def drop_high_corr(
    X: pd.DataFrame, threshold: float, *, always_keep: Iterable[str] = (),
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    numeric = X.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return X, []
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    drop: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for col in upper.columns:
        for row in upper.index:
            v = upper.at[row, col]
            if pd.notna(v) and v > threshold:
                # Drop the lexicographically later name unless it's in always_keep
                drop_col = col if col not in always_keep else row
                if drop_col in always_keep:
                    continue
                if drop_col in drop:
                    continue
                drop.add(drop_col)
                pairs.append((row, col))
    return X.drop(columns=list(drop)), pairs


def rank_mutual_info(
    X: pd.DataFrame, y: pd.Series, top_k: int, *, random_state: int = 42,
) -> list[tuple[str, float]]:
    X_num = X.select_dtypes(include="number").fillna(0.0)
    if X_num.empty:
        return []
    discrete = y.nunique() < 20
    func = mutual_info_classif if discrete else None
    if func is None:
        return []
    scores = func(X_num.to_numpy(), y.to_numpy(), random_state=random_state)
    ranked = sorted(zip(X_num.columns, scores), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


def rank_permutation(
    X: pd.DataFrame, y: pd.Series, top_k: int, *, random_state: int = 42, n_repeats: int = 5,
) -> list[tuple[str, float]]:
    X_num = X.select_dtypes(include="number").fillna(0.0)
    if X_num.empty or y.nunique() < 2:
        return []
    model = GradientBoostingClassifier(
        n_estimators=80, max_depth=3, random_state=random_state,
    )
    try:
        model.fit(X_num, y)
        r = permutation_importance(
            model, X_num, y, n_repeats=n_repeats, random_state=random_state, n_jobs=1,
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("permutation importance failed: %s", exc)
        return []
    ranked = sorted(zip(X_num.columns, r.importances_mean), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_k]


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    cfg: SelectionConfig | None = None,
) -> SelectionReport:
    cfg = cfg or load_config()
    always = set(cfg.always_keep)

    # Stage 1–3 operate on the training window only.
    X, dropped_null = drop_high_null(X_train, cfg.min_non_null_ratio)
    X, dropped_var = drop_low_variance(X, cfg.variance_threshold)
    X, dropped_corr = drop_high_corr(X, cfg.max_pairwise_corr, always_keep=always)

    # Stage 4: mutual information
    mi_ranking = rank_mutual_info(X, y_train, cfg.mutual_info_top_k)
    mi_keep = [name for name, _ in mi_ranking]

    # Stage 5: permutation importance on validation window
    X_val_filtered = X_val[[c for c in mi_keep if c in X_val.columns]]
    perm_ranking = rank_permutation(X_val_filtered, y_val, cfg.permutation_importance_top_k)
    perm_keep = [name for name, _ in perm_ranking] or mi_keep

    # Stage 6–7: always_keep + top-k
    selected: list[str] = []
    for c in cfg.always_keep:
        if c in X.columns:
            selected.append(c)
    for c in perm_keep:
        if c not in selected:
            selected.append(c)
        if len(selected) >= cfg.final_top_k:
            break

    return SelectionReport(
        selected=selected,
        dropped_nulls=dropped_null,
        dropped_variance=dropped_var,
        dropped_correlation=dropped_corr,
        mi_ranking=mi_ranking,
        permutation_ranking=perm_ranking,
    )


# ---------------------------------------------------------------------------
# CLI — uses Phase-3 sample feature/label outputs to demonstrate the pipeline
# ---------------------------------------------------------------------------

def _load_sample_features_and_labels(
    symbol: str = "BTCUSDT", timeframe: str = "1h", root: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series] | None:
    root = root or repo_root()
    feat = (
        root / "data" / "features" / f"source=binance" / f"market_type=spot"
        / f"symbol={symbol}" / f"timeframe={timeframe}" / "features.parquet"
    )
    lab = (
        root / "data" / "labels" / f"source=binance" / f"market_type=spot"
        / f"symbol={symbol}" / f"timeframe={timeframe}" / "labels.parquet"
    )
    if not feat.exists() or not lab.exists():
        return None
    fdf = pq.read_table(feat).to_pandas()
    ldf = pq.read_table(lab).to_pandas()
    for tcol in ("timestamp_utc",):
        if tcol in fdf.columns:
            fdf[tcol] = pd.to_datetime(fdf[tcol], utc=True)
            fdf = fdf.set_index(tcol)
        if tcol in ldf.columns:
            ldf[tcol] = pd.to_datetime(ldf[tcol], utc=True)
            ldf = ldf.set_index(tcol)
    joined = fdf.join(ldf[["direction"]], how="inner")
    joined = joined[joined["direction"].isin([0, 2])]  # binary up vs down for selection demo
    y = (joined["direction"] == 2).astype("int8")
    X = joined.drop(columns=["direction"]).select_dtypes(include="number")
    return X, y


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.features.selection")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--sample", type=lambda s: s.strip().lower() in {"1", "true", "yes", "y", "t"},
                   default=False)
    args = p.parse_args(argv)

    loaded = _load_sample_features_and_labels(args.symbol, args.timeframe)
    if loaded is None:
        print("Selection skipped: feature/label parquet not found. "
              "Run build_matrix + labeling first.")
        return 2
    X, y = loaded
    n = len(X)
    if n < 200:
        print(f"Selection skipped: only {n} rows after joining; need ≥200.")
        return 2

    # Simple holdout split: first 70% train, next 30% val. This is for the
    # demo only; production uses purged walk-forward (Phase 4).
    cut = int(n * 0.7)
    X_train, y_train = X.iloc[:cut], y.iloc[:cut]
    X_val, y_val = X.iloc[cut:], y.iloc[cut:]

    cfg = load_config()
    report = select_features(X_train, y_train, X_val, y_val, cfg=cfg)

    root = repo_root()
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_path = root / "reports" / f"feature_selection_{stamp}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Feature selection report — {stamp}",
        "",
        f"symbol/timeframe: {args.symbol} / {args.timeframe}",
        f"rows: {n} (train {cut}, val {n - cut})",
        f"selected ({len(report.selected)}): {', '.join(report.selected[:30])}"
        + ("..." if len(report.selected) > 30 else ""),
        f"dropped — high-null: {len(report.dropped_nulls)}",
        f"dropped — low-variance: {len(report.dropped_variance)}",
        f"dropped — high-corr: {len(report.dropped_correlation)}",
        "",
        "## top 15 by mutual information",
        "",
    ]
    for name, score in report.mi_ranking[:15]:
        lines.append(f"- {name}: {score:.4f}")
    lines += ["", "## top 15 by permutation importance", ""]
    for name, score in report.permutation_ranking[:15]:
        lines.append(f"- {name}: {score:.4f}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path = root / "artifacts" / "runs" / f"selected_features_{stamp}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(json_path, {
        "stamp": stamp,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "selected": report.selected,
        "dropped_nulls_count": len(report.dropped_nulls),
        "dropped_variance_count": len(report.dropped_variance),
        "dropped_correlation_count": len(report.dropped_correlation),
        "mi_ranking_top": report.mi_ranking[:30],
        "permutation_ranking_top": report.permutation_ranking[:30],
    })

    print(f"\nselected features: {len(report.selected)}")
    print(f"  high-null dropped: {len(report.dropped_nulls)}")
    print(f"  low-variance dropped: {len(report.dropped_variance)}")
    print(f"  high-corr dropped: {len(report.dropped_correlation)}")
    print(f"report: {md_path.relative_to(root)}")
    print(f"json:   {json_path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
