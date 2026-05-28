"""Build sequence-windowed datasets for the multi-task model.

Outputs per (symbol, timeframe):

* X_seq[N, L, F]        — main-timeframe rolling windows of length L of F features
* X_context[N, C]       — slow context vector at the bar's open_time
* asset_id[N]           — integer asset id (0=BTCUSDT, 1=ETHUSDT)
* tf_id[N]              — integer timeframe id (0=15m, 1=1h, 2=4h, 3=1d)
* y_direction[N]        — {0=down, 1=sideways, 2=up}  (ambiguous rows dropped)
* y_regime[N]           — int 0..5
* y_cycle[N]            — int 0..3
* y_trade_quality[N]    — int 0/1

Plus per-fold scaler/imputer/feature schema written under
``artifacts/datasets/<run_id>/...``.

Splits use a single purged + embargoed walk-forward fold for the CPU smoke
test (deterministic, fast). The training loop reads ``num_walk_forward_splits``
from config to opt into more folds.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.utils.io import read_yaml, repo_root, write_json
from src.utils.logging import get_logger
from src.utils.seeds import set_global_seed

_log = get_logger("datasets.build_dataset")

ASSET_TO_ID: dict[str, int] = {"BTCUSDT": 0, "ETHUSDT": 1}
TIMEFRAME_TO_ID: dict[str, int] = {"15m": 0, "1h": 1, "4h": 2, "1d": 3}

REGIME_LABEL_TO_ID = {
    "trending_up": 0, "trending_down": 1, "ranging_low_vol": 2,
    "ranging_high_vol": 3, "breakout": 4, "capitulation": 5,
}
CYCLE_LABEL_TO_ID = {
    "accumulation": 0, "bull": 1, "distribution": 2, "bear": 3,
}

# Columns considered "context" rather than sequence — slow-moving features
# that don't need to be windowed because they barely change bar-to-bar.
CONTEXT_FEATURE_PATTERNS = (
    "cycle_months_since_halving",
    "btc_dominance_pct", "eth_dominance_pct",
    "total_market_cap_usd", "total_volume_usd_24h",
    "macro_",  # any macro feature
)


@dataclass
class DatasetSpec:
    symbol: str
    timeframe: str
    seq_len: int
    train_rows: int
    val_rows: int
    test_rows: int
    embargo_bars: int
    feature_count_seq: int
    feature_count_context: int
    classes_direction: list[str]
    classes_regime: list[str]
    classes_cycle: list[str]


_PARTITION_COLUMNS = ("source", "market_type", "symbol", "timeframe")


def _drop_partition_cols(df: pd.DataFrame) -> pd.DataFrame:
    """pyarrow injects partition keys as columns when reading a Hive-partitioned
    file. We don't want them — they're metadata, not features."""
    cols = [c for c in _PARTITION_COLUMNS if c in df.columns]
    return df.drop(columns=cols) if cols else df


def _load_features(symbol: str, timeframe: str, *, market: str, root: Path) -> pd.DataFrame | None:
    path = (
        root / "data" / "features"
        / f"source=binance" / f"market_type={market}"
        / f"symbol={symbol}" / f"timeframe={timeframe}" / "features.parquet"
    )
    if not path.exists():
        _log.warning("missing features parquet: %s", path)
        return None
    df = pq.read_table(path).to_pandas()
    df = _drop_partition_cols(df)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.set_index("timestamp_utc")
    df = df.sort_index()
    return df[~df.index.duplicated(keep="last")]


def _load_labels(symbol: str, timeframe: str, *, market: str, root: Path) -> pd.DataFrame | None:
    path = (
        root / "data" / "labels"
        / f"source=binance" / f"market_type={market}"
        / f"symbol={symbol}" / f"timeframe={timeframe}" / "labels.parquet"
    )
    if not path.exists():
        return None
    df = pq.read_table(path).to_pandas()
    df = _drop_partition_cols(df)
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.set_index("timestamp_utc")
    return df.sort_index()


def _split_columns(features_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition feature columns into (sequence, context)."""
    cols = [c for c in features_df.columns if features_df[c].dtype.kind in "fbi"]
    ctx = [c for c in cols if any(c.startswith(p) or c == p for p in CONTEXT_FEATURE_PATTERNS)]
    seq = [c for c in cols if c not in ctx]
    return seq, ctx


def _purged_walk_forward_split(
    n: int, *, train_frac: float = 0.7, val_frac: float = 0.15, embargo_bars: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    train = np.arange(0, train_end)
    val = np.arange(train_end + embargo_bars, val_end)
    test = np.arange(val_end + embargo_bars, n)
    return train, val, test


def _make_windows(
    arr: np.ndarray, seq_len: int, *, valid_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build [N, L, F] windows that END at each ``valid_rows[i]`` index.

    Returns (windows, the row indices each window corresponds to).
    """
    out = np.empty((len(valid_rows), seq_len, arr.shape[1]), dtype=np.float32)
    keep_mask = np.ones(len(valid_rows), dtype=bool)
    for k, i in enumerate(valid_rows):
        if i + 1 < seq_len:
            keep_mask[k] = False
            continue
        out[k] = arr[i + 1 - seq_len: i + 1]
    return out[keep_mask], valid_rows[keep_mask]


def build_one_combo(
    symbol: str,
    timeframe: str,
    *,
    market: str = "spot",
    sample: bool = False,
    seq_len: int | None = None,
    embargo_bars: int | None = None,
    root: Path | None = None,
    run_id: str | None = None,
    seq_cols_override: list[str] | None = None,
    ctx_cols_override: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], DatasetSpec, Path] | None:
    root = root or repo_root()
    cfg = read_yaml(root / "configs" / "config.yaml")
    set_global_seed(int(cfg["project"]["seed"]))
    embargo_bars = embargo_bars if embargo_bars is not None else int(
        (cfg.get("validation") or {}).get("embargo_bars", 10)
    )
    if seq_len is None:
        seq_len_map = (cfg.get("model") or {}).get("sequence_length") or {}
        seq_len = int(seq_len_map.get(timeframe, 64 if sample else 128))
    if sample:
        seq_len = min(seq_len, 32)

    features = _load_features(symbol, timeframe, market=market, root=root)
    labels = _load_labels(symbol, timeframe, market=market, root=root)
    if features is None or labels is None:
        _log.warning("skipping %s/%s — features or labels missing", symbol, timeframe)
        return None

    joined = features.join(labels, how="inner")
    # Drop ambiguous / unlabeled rows
    joined = joined[joined["direction"].isin([0, 1, 2])]
    if joined.empty:
        _log.warning("no usable rows after joining for %s/%s", symbol, timeframe)
        return None

    seq_cols, ctx_cols = _split_columns(joined.drop(columns=[
        "direction", "barrier_hit_offset", "barrier_rr",
        "regime", "cycle", "trade_quality",
    ], errors="ignore"))
    if seq_cols_override is not None:
        seq_cols = [c for c in seq_cols_override if c in joined.columns]
        # Pad with NaN-filled columns for any override col not actually present here
        for c in seq_cols_override:
            if c not in joined.columns:
                joined[c] = np.nan
        seq_cols = list(seq_cols_override)
    if ctx_cols_override is not None:
        for c in ctx_cols_override:
            if c not in joined.columns:
                joined[c] = np.nan
        ctx_cols = list(ctx_cols_override)

    n = len(joined)
    train_idx, val_idx, test_idx = _purged_walk_forward_split(n, embargo_bars=embargo_bars)

    seq_train_raw = joined.iloc[train_idx][seq_cols].to_numpy(dtype=np.float64)
    ctx_train_raw = joined.iloc[train_idx][ctx_cols].to_numpy(dtype=np.float64) if ctx_cols else np.zeros((len(train_idx), 0))

    # keep_empty_features=True so that columns that are all-NaN on the training
    # window (e.g. a derivatives feature that only exists for futures data on a
    # spot-market combo) are preserved as NaN rather than silently dropped.
    # Downstream the scaler is happy with NaN columns (it just leaves them).
    seq_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    ctx_imputer = (
        SimpleImputer(strategy="median", keep_empty_features=True)
        if ctx_cols else None
    )
    seq_scaler = StandardScaler()
    ctx_scaler = StandardScaler() if ctx_cols else None

    seq_imputer.fit(seq_train_raw)
    seq_train_imp = seq_imputer.transform(seq_train_raw)
    seq_scaler.fit(seq_train_imp)

    if ctx_cols:
        ctx_imputer.fit(ctx_train_raw)
        ctx_train_imp = ctx_imputer.transform(ctx_train_raw)
        ctx_scaler.fit(ctx_train_imp)

    # Transform the WHOLE joined frame; splits select rows after.
    seq_all = joined[seq_cols].to_numpy(dtype=np.float64)
    seq_all = seq_imputer.transform(seq_all)
    seq_all = seq_scaler.transform(seq_all).astype(np.float32)
    # Replace any residual NaN (from all-NaN columns or scaler edge cases) with
    # zero — TF doesn't accept NaN inputs and 0 is the post-standardization mean.
    seq_all = np.nan_to_num(seq_all, nan=0.0, posinf=0.0, neginf=0.0)
    if ctx_cols:
        ctx_all = joined[ctx_cols].to_numpy(dtype=np.float64)
        ctx_all = ctx_imputer.transform(ctx_all)
        ctx_all = ctx_scaler.transform(ctx_all).astype(np.float32)
        ctx_all = np.nan_to_num(ctx_all, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        ctx_all = np.zeros((n, 0), dtype=np.float32)

    direction_all = joined["direction"].to_numpy(dtype=np.int64)
    regime_all = joined["regime"].map(REGIME_LABEL_TO_ID).fillna(REGIME_LABEL_TO_ID["ranging_low_vol"]).to_numpy(dtype=np.int64)
    cycle_all = joined["cycle"].map(CYCLE_LABEL_TO_ID).fillna(CYCLE_LABEL_TO_ID["accumulation"]).to_numpy(dtype=np.int64)
    tq_all = joined["trade_quality"].to_numpy(dtype=np.int64)

    asset_id = np.full(n, ASSET_TO_ID.get(symbol, 0), dtype=np.int32)
    tf_id = np.full(n, TIMEFRAME_TO_ID.get(timeframe, 1), dtype=np.int32)

    out: dict[str, np.ndarray] = {}
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idx) == 0:
            continue
        x_seq, kept_idx = _make_windows(seq_all, seq_len, valid_rows=idx)
        if len(kept_idx) == 0:
            continue
        out[f"X_seq_{name}"] = x_seq
        out[f"X_context_{name}"] = ctx_all[kept_idx]
        out[f"asset_id_{name}"] = asset_id[kept_idx]
        out[f"tf_id_{name}"] = tf_id[kept_idx]
        out[f"y_direction_{name}"] = direction_all[kept_idx]
        out[f"y_regime_{name}"] = regime_all[kept_idx]
        out[f"y_cycle_{name}"] = cycle_all[kept_idx]
        out[f"y_trade_quality_{name}"] = tq_all[kept_idx]

    spec = DatasetSpec(
        symbol=symbol, timeframe=timeframe,
        seq_len=seq_len,
        train_rows=int(out.get("X_seq_train", np.empty((0,))).shape[0]) if "X_seq_train" in out else 0,
        val_rows=int(out.get("X_seq_val", np.empty((0,))).shape[0]) if "X_seq_val" in out else 0,
        test_rows=int(out.get("X_seq_test", np.empty((0,))).shape[0]) if "X_seq_test" in out else 0,
        embargo_bars=embargo_bars,
        feature_count_seq=len(seq_cols),
        feature_count_context=len(ctx_cols),
        classes_direction=["down", "sideways", "up"],
        classes_regime=list(REGIME_LABEL_TO_ID.keys()),
        classes_cycle=list(CYCLE_LABEL_TO_ID.keys()),
    )

    # Persist artifacts
    run_id = run_id or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = root / "artifacts" / "datasets" / run_id / symbol / timeframe
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "splits.npz", **out)
    joblib.dump(seq_imputer, out_dir / "seq_imputer.joblib")
    joblib.dump(seq_scaler, out_dir / "seq_scaler.joblib")
    if ctx_cols:
        joblib.dump(ctx_imputer, out_dir / "ctx_imputer.joblib")
        joblib.dump(ctx_scaler, out_dir / "ctx_scaler.joblib")
    write_json(out_dir / "feature_schema.json", {
        "seq_columns": seq_cols,
        "context_columns": ctx_cols,
        "seq_len": seq_len,
        "regime_label_to_id": REGIME_LABEL_TO_ID,
        "cycle_label_to_id": CYCLE_LABEL_TO_ID,
    })
    write_json(out_dir / "dataset_spec.json", {
        "symbol": symbol, "timeframe": timeframe, "seq_len": seq_len,
        "train_rows": spec.train_rows, "val_rows": spec.val_rows, "test_rows": spec.test_rows,
        "embargo_bars": embargo_bars,
        "feature_count_seq": spec.feature_count_seq,
        "feature_count_context": spec.feature_count_context,
        "classes_direction": spec.classes_direction,
        "classes_regime": spec.classes_regime,
        "classes_cycle": spec.classes_cycle,
    })
    return out, spec, out_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.datasets.build_dataset")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--timeframes", nargs="+", required=True)
    p.add_argument("--market", default="spot")
    p.add_argument("--sample", type=lambda s: s.strip().lower() in {"1", "true", "yes", "y", "t"},
                   default=False)
    args = p.parse_args(argv)

    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"run_id: {run_id}")

    # Pre-pass: discover the INTERSECTION of feature columns across all combos
    # so the resulting tensors are joinable across symbols. Without this, BTC
    # has on-chain columns ETH doesn't, and the train loop would fail to stack.
    root = repo_root()
    aligned_seq: list[str] | None = None
    aligned_ctx: list[str] | None = None
    for symbol in args.symbols:
        for tf in args.timeframes:
            features = _load_features(symbol, tf, market=args.market, root=root)
            labels = _load_labels(symbol, tf, market=args.market, root=root)
            if features is None or labels is None:
                continue
            joined = features.join(labels, how="inner").drop(columns=[
                "direction", "barrier_hit_offset", "barrier_rr",
                "regime", "cycle", "trade_quality",
            ], errors="ignore")
            seq, ctx = _split_columns(joined)
            if aligned_seq is None:
                aligned_seq, aligned_ctx = list(seq), list(ctx)
            else:
                aligned_seq = [c for c in aligned_seq if c in seq]
                aligned_ctx = [c for c in (aligned_ctx or []) if c in ctx]
    if aligned_seq is not None:
        print(f"aligned feature schema: {len(aligned_seq)} seq, {len(aligned_ctx or [])} ctx")

    print(f"{'symbol':<9} {'tf':<4} {'seq_len':>7} {'train':>6} {'val':>6} {'test':>6} {'fseq':>5} {'fctx':>5}")
    any_written = False
    for symbol in args.symbols:
        for tf in args.timeframes:
            built = build_one_combo(
                symbol, tf, market=args.market, sample=args.sample, run_id=run_id,
                seq_cols_override=aligned_seq,
                ctx_cols_override=aligned_ctx,
            )
            if built is None:
                print(f"{symbol:<9} {tf:<4}  (skipped - no features/labels)")
                continue
            _, spec, out_dir = built
            any_written = True
            print(
                f"{symbol:<9} {tf:<4} {spec.seq_len:>7} {spec.train_rows:>6} "
                f"{spec.val_rows:>6} {spec.test_rows:>6} "
                f"{spec.feature_count_seq:>5} {spec.feature_count_context:>5}"
            )
            print(f"  -> {out_dir.relative_to(repo_root())}")
    return 0 if any_written else 2


if __name__ == "__main__":
    sys.exit(main())
