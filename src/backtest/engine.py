"""Event-driven backtest engine.

Drives the broker over a feature-aligned OHLCV frame, emitting trades and
an equity curve.  Compares the model strategy against the spec-required
honest baselines and writes:

* ``reports/backtest_<run_id>.md``
* ``reports/backtest_<run_id>.json``
* ``reports/trades_<run_id>.csv``

CLI::

    python -m src.backtest.engine --latest --sample true
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# IMPORTANT (Windows): TensorFlow must be imported BEFORE pyarrow, otherwise
# pyarrow's bundled DLLs break TF's native runtime load (ERROR_DLL_INIT_FAILED
# 0x45A). Importing multitask_model here pulls in keras+tf first and also
# registers the custom Keras layers needed by load_model.
from src.models import multitask_model  # noqa: F401,E402

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.backtest.broker import Broker, BrokerConfig, SIDE_LONG, SIDE_SHORT
from src.backtest.costs import FeeModel, SlippageModel
from src.backtest.metrics import BacktestMetrics, compute_metrics
from src.backtest import strategies as strat
from src.utils.io import read_yaml, repo_root, write_json
from src.utils.logging import get_logger

_log = get_logger("backtest.engine")


# ---------------------------------------------------------------------------
# Helpers — model prediction loading
# ---------------------------------------------------------------------------

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


def _load_features_for_backtest(symbol: str, timeframe: str, *, market: str, root: Path) -> pd.DataFrame | None:
    """Return the OHLCV+features frame at the latest available depth."""
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
    return df


def _predict_per_bar(
    run_dir: Path, dataset_dir: Path, symbol: str, timeframe: str, *, batch_size: int = 128,
) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Run the trained model on every bar of the test split.

    Returns (dataframe of OHLCV at those bars, prediction array of shape
    [N, direction_p(3) + cycle_p(C) + regime_p(R) + trade_quality(1)]).
    """
    import tensorflow as tf  # already initialized at module import time

    combo_dir = dataset_dir / symbol / timeframe
    if not combo_dir.exists():
        return None
    arrays = dict(np.load(combo_dir / "splits.npz"))
    spec = json.loads((combo_dir / "dataset_spec.json").read_text("utf-8"))

    # Use test if present; otherwise val; otherwise train
    for split in ("test", "val", "train"):
        if f"X_seq_{split}" in arrays:
            break
    else:
        return None

    has_ctx = spec["feature_count_context"] > 0
    inputs = [arrays[f"X_seq_{split}"]]
    if has_ctx:
        inputs.append(arrays[f"X_context_{split}"])
    inputs.append(arrays[f"asset_id_{split}"])
    inputs.append(arrays[f"tf_id_{split}"])

    model = tf.keras.models.load_model(run_dir / "model.keras", compile=False)
    preds = model.predict(inputs, batch_size=batch_size, verbose=0)
    dir_p, reg_p, cyc_p, tq_p = preds

    # The dataset doesn't store original open_times; reconstruct from features
    feats = _load_features_for_backtest(symbol, timeframe, market="spot", root=repo_root())
    if feats is None:
        return None
    # We assume the dataset was built from the most-recent N rows that matched
    # the labels frame; pick the last len(arrays[split]) rows that have OHLCV.
    n_pred = len(dir_p)
    feats_tail = feats.iloc[-n_pred:].copy()
    out_preds = np.concatenate([dir_p, reg_p, cyc_p, tq_p.reshape(-1, 1)], axis=1)
    return feats_tail, out_preds


# ---------------------------------------------------------------------------
# Backtest run
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    name: str
    metrics: BacktestMetrics
    trades: list  # of Trade


def _drive_broker(
    df: pd.DataFrame,
    signal_fn,
    *,
    asset: str,
    timeframe: str,
    cfg: BrokerConfig,
    fee: FeeModel,
    slip: SlippageModel,
) -> StrategyResult:
    broker = Broker(cfg, fee, slip, asset=asset, timeframe=timeframe)
    bars_held_for_open = 0
    for i, (ts, row) in enumerate(df.iterrows()):
        bars_held_for_open += 1 if broker.position else 0
        if broker.position is not None:
            broker.update_bar(
                ts=ts, high=float(row["high"]), low=float(row["low"]),
                close=float(row["close"]), bars_held=bars_held_for_open,
            )
            if broker.position is None:
                bars_held_for_open = 0
        if broker.position is None and broker.can_open(ts):
            sig = signal_fn(i=i, row=row)
            if sig.side != 0:
                atr = float(row.get("atr", 0.0) or 0.0)
                broker.open(ts=ts, side=sig.side, entry_price=float(row["close"]), atr=atr)
                bars_held_for_open = 1
        broker.mark_equity(ts)
    # End-of-data flat
    if broker.position is not None:
        last_ts = df.index[-1]
        broker.force_close(last_ts, float(df["close"].iloc[-1]))
    metrics = compute_metrics(
        broker.history, broker.equity_curve,
        initial_equity=cfg.initial_equity, total_bars=len(df),
    )
    return StrategyResult(name=signal_fn.__name__, metrics=metrics, trades=broker.history)


def run_backtest(
    *,
    df: pd.DataFrame,
    preds: np.ndarray | None,
    direction_classes: int,
    regime_classes: int,
    cycle_classes: int,
    asset: str,
    timeframe: str,
    bcfg_yaml: dict,
) -> dict[str, StrategyResult]:
    cfg = BrokerConfig(
        initial_equity=float(bcfg_yaml.get("initial_equity", 10_000.0)),
        risk_per_trade_pct=float(bcfg_yaml.get("risk_per_trade_pct", 1.0)),
        max_risk_per_trade_pct=float(bcfg_yaml.get("max_risk_per_trade_pct", 2.0)),
        max_daily_loss_pct=float(bcfg_yaml.get("max_daily_loss_pct", 4.0)),
        stop_atr_multiple=float(((bcfg_yaml.get("exits") or {}).get("stop_atr_multiple", 1.5))),
        tp1_rr=float(((bcfg_yaml.get("exits") or {}).get("tp1_rr", 1.0))),
        tp2_rr=float(((bcfg_yaml.get("exits") or {}).get("tp2_rr", 2.0))),
        tp3_rr=float(((bcfg_yaml.get("exits") or {}).get("tp3_rr", 3.0))),
        max_holding_bars=int(((bcfg_yaml.get("exits") or {}).get("max_holding_bars") or {}).get(timeframe, 48)),
    )
    fee = FeeModel(bps_per_side=float(bcfg_yaml.get("fee_bps_per_side", 4.0)))
    slip_cfg = bcfg_yaml.get("slippage") or {}
    slip = SlippageModel(
        min_bps_per_side=float(slip_cfg.get("min_bps_per_side", 2.0)),
        max_bps_per_side=float(slip_cfg.get("max_bps_per_side", 15.0)),
        atr_fraction=float(slip_cfg.get("atr_fraction", 0.02)),
    )
    conf_thr = bcfg_yaml.get("confidence_thresholds") or {}
    long_thr = float(conf_thr.get("long", 0.58))
    short_thr = float(conf_thr.get("short", 0.58))
    no_trade_thr = float(conf_thr.get("no_trade_below", 0.58))
    quality_thr = 0.60

    rng = np.random.default_rng(42)

    # The model signal_fn reads per-bar predictions; baselines read row-only.
    pred_lookup: dict[pd.Timestamp, np.ndarray] = {}
    if preds is not None:
        for ts, p in zip(df.index, preds):
            pred_lookup[ts] = p

    def model_fn(*, i: int, row: pd.Series) -> strat.StrategySignal:
        ts = row.name if hasattr(row, "name") else df.index[i]
        p = pred_lookup.get(ts)
        if p is None:
            return strat.StrategySignal(0, 0.0, "no model prediction")
        dir_p = p[:direction_classes]
        # trade_quality is the very last element
        tq_p = float(p[-1])
        return strat.model_signal(
            direction_probs=dir_p, trade_quality_prob=tq_p,
            long_threshold=long_thr, short_threshold=short_thr,
            quality_threshold=quality_thr, no_trade_threshold=no_trade_thr,
        )

    def baseline_buy_hold(*, i: int, row: pd.Series) -> strat.StrategySignal:
        return strat.buy_and_hold_signal(i)

    def baseline_ema_trend(*, i: int, row: pd.Series) -> strat.StrategySignal:
        return strat.ema_trend_signal(row)

    def baseline_rsi_macd(*, i: int, row: pd.Series) -> strat.StrategySignal:
        return strat.rsi_macd_signal(row)

    def baseline_random(*, i: int, row: pd.Series) -> strat.StrategySignal:
        return strat.random_signal(rng)

    def baseline_no_trade(*, i: int, row: pd.Series) -> strat.StrategySignal:
        return strat.no_trade_signal()

    candidates = {
        "model": model_fn,
        "buy_and_hold": baseline_buy_hold,
        "ema_trend": baseline_ema_trend,
        "rsi_macd": baseline_rsi_macd,
        "random": baseline_random,
        "no_trade": baseline_no_trade,
    }
    results: dict[str, StrategyResult] = {}
    for name, fn in candidates.items():
        # Rename the function so StrategyResult.name is informative
        fn.__name__ = name
        if name == "model" and preds is None:
            continue
        results[name] = _drive_broker(
            df, fn, asset=asset, timeframe=timeframe, cfg=cfg, fee=fee, slip=slip,
        )
    return results


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _trades_csv(strategy: StrategyResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "strategy", "open_ts", "close_ts", "asset", "timeframe", "side",
            "entry_price", "exit_price", "size", "stop_price",
            "tp1", "tp2", "tp3", "realized_pnl", "rr_realized",
            "exit_reason", "fees_paid", "slippage_cost", "atr_at_entry",
        ])
        for t in strategy.trades:
            w.writerow([
                strategy.name, t.open_ts, t.close_ts, t.asset, t.timeframe, t.side,
                t.entry_price, t.exit_price, t.size, t.stop_price,
                t.tp1, t.tp2, t.tp3, t.realized_pnl, t.rr_realized,
                t.exit_reason, t.fees_paid, t.slippage_cost, t.atr_at_entry,
            ])


def _report_markdown(
    results_by_combo: dict[tuple[str, str], dict[str, StrategyResult]],
    out_path: Path,
    *,
    initial_equity: float,
    fee_bps: float,
) -> None:
    lines = [
        f"# Backtest report — {datetime.now(tz=timezone.utc).isoformat()}",
        "",
        f"initial_equity = {initial_equity:.0f}, fee_bps_per_side = {fee_bps:.1f}",
        "",
        "Strategy: **model** vs baselines (majority_class / random / buy_and_hold / ema_trend / rsi_macd / no_trade).",
        "",
    ]
    for (asset, tf), results in results_by_combo.items():
        lines += [f"## {asset} / {tf}", ""]
        lines += [
            "| Strategy | Trades | WinRate% | AvgR | ProfitFactor | TotalRet% | MaxDD% | Sharpe | LongWR% | ShortWR% |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, sr in results.items():
            m = sr.metrics
            pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
            lines.append(
                f"| {name} | {m.n_trades} | {m.win_rate:.1f} | {m.avg_r:.3f} | "
                f"{pf} | {m.total_return_pct:.2f} | {m.max_drawdown_pct:.2f} | "
                f"{m.sharpe:.2f} | {m.long_win_rate:.1f} | {m.short_win_rate:.1f} |"
            )
        # Honesty section
        model = results.get("model")
        baselines = {k: v for k, v in results.items() if k != "model"}
        if model is not None and baselines:
            best_baseline_ret = max(b.metrics.total_return_pct for b in baselines.values())
            verdict = (
                "**Model beats best baseline.**"
                if model.metrics.total_return_pct > best_baseline_ret
                else "**Model does NOT beat the best baseline** — do not use for trading."
            )
            lines += ["", verdict, ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report_json(
    results_by_combo: dict[tuple[str, str], dict[str, StrategyResult]],
    out_path: Path,
    *,
    initial_equity: float,
) -> None:
    payload = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "initial_equity": initial_equity,
        "combos": {},
    }
    for (asset, tf), results in results_by_combo.items():
        payload["combos"][f"{asset}/{tf}"] = {
            name: {**asdict(sr.metrics), "name": sr.name}
            for name, sr in results.items()
        }
    write_json(out_path, payload)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.backtest.engine")
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--timeframes", nargs="+", default=["1h"])
    p.add_argument("--latest", action="store_true",
                   help="Use the latest model + dataset under artifacts/")
    p.add_argument("--sample", type=lambda s: s.strip().lower() in {"1", "true", "yes", "y", "t"},
                   default=False, help="Restrict backtest to the latest 500 bars per combo.")
    p.add_argument("--dataset-run-id", default=None)
    p.add_argument("--model-run-id", default=None)
    args = p.parse_args(argv)

    root = repo_root()
    cfg = read_yaml(root / "configs" / "config.yaml")
    bcfg = cfg.get("backtest") or {}

    if args.model_run_id:
        run_dir = root / "artifacts" / "runs" / args.model_run_id
    else:
        run_dir = _find_latest_run_dir(root)
    if args.dataset_run_id:
        ds_dir = root / "artifacts" / "datasets" / args.dataset_run_id
    else:
        ds_dir = _find_latest_dataset_dir(root)
    if run_dir is None or ds_dir is None:
        print("No model run or dataset found. Train a model first.")
        return 2
    print(f"model: {run_dir.relative_to(root)}")
    print(f"dataset: {ds_dir.relative_to(root)}")

    direction_classes, regime_classes, cycle_classes = 3, 6, 4

    results_by_combo: dict[tuple[str, str], dict[str, StrategyResult]] = {}
    for symbol in args.symbols:
        for tf in args.timeframes:
            loaded = _predict_per_bar(run_dir, ds_dir, symbol, tf)
            if loaded is None:
                print(f"  {symbol}/{tf}: no model prediction available")
                feats = _load_features_for_backtest(symbol, tf, market="spot", root=root)
                if feats is None:
                    continue
                if args.sample:
                    feats = feats.tail(500)
                preds = None
                df_for_bt = feats
            else:
                df_for_bt, preds = loaded
                if args.sample:
                    df_for_bt = df_for_bt.tail(500)
                    if preds is not None:
                        preds = preds[-500:]

            # Backtest needs OHLCV + atr; make sure they're present
            if not {"open", "high", "low", "close"}.issubset(df_for_bt.columns):
                print(f"  {symbol}/{tf}: features file missing OHLC — skipped")
                continue
            results = run_backtest(
                df=df_for_bt, preds=preds,
                direction_classes=direction_classes,
                regime_classes=regime_classes,
                cycle_classes=cycle_classes,
                asset=symbol, timeframe=tf, bcfg_yaml=bcfg,
            )
            results_by_combo[(symbol, tf)] = results

    if not results_by_combo:
        print("No combos backtested.")
        return 2

    run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_path = root / "reports" / f"backtest_{run_id}.md"
    json_path = root / "reports" / f"backtest_{run_id}.json"
    trades_path = root / "reports" / f"trades_{run_id}.csv"

    _report_markdown(results_by_combo, md_path,
                      initial_equity=float(bcfg.get("initial_equity", 10_000.0)),
                      fee_bps=float(bcfg.get("fee_bps_per_side", 4.0)))
    _report_json(results_by_combo, json_path,
                  initial_equity=float(bcfg.get("initial_equity", 10_000.0)))

    # Concatenate all trades into one CSV for easy auditing
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trades_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "combo", "strategy", "open_ts", "close_ts", "side", "entry_price",
            "exit_price", "size", "stop_price", "realized_pnl", "rr_realized",
            "exit_reason", "fees_paid", "slippage_cost",
        ])
        for (asset, tf), results in results_by_combo.items():
            for name, sr in results.items():
                for t in sr.trades:
                    w.writerow([
                        f"{asset}/{tf}", name, t.open_ts, t.close_ts, t.side,
                        t.entry_price, t.exit_price, t.size, t.stop_price,
                        t.realized_pnl, t.rr_realized,
                        t.exit_reason, t.fees_paid, t.slippage_cost,
                    ])

    print(f"\nreports:")
    for path in (md_path, json_path, trades_path):
        print(f"  {path.relative_to(root)}")

    # Summary line per combo
    print("\nsummary:")
    for (asset, tf), results in results_by_combo.items():
        model = results.get("model")
        if model is None:
            print(f"  {asset}/{tf}: model strategy unavailable")
            continue
        bh = results.get("buy_and_hold")
        print(
            f"  {asset}/{tf}: model trades={model.metrics.n_trades} "
            f"return={model.metrics.total_return_pct:.2f}% "
            f"PF={'inf' if model.metrics.profit_factor == float('inf') else f'{model.metrics.profit_factor:.2f}'} "
            f"buy_hold={bh.metrics.total_return_pct if bh else float('nan'):.2f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
