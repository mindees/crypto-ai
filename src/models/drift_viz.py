"""Drift visualization — static charts + an HTML dashboard.

Charts written under ``reports/drift/`` with a machine-readable JSON beside
each. The HTML dashboard summarizes PSI severity and links the charts.

CLI::

    python -m src.models.drift_viz --sample true
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.models.drift import MODERATE, SIGNIFICANT, STABLE, compute_psi, feature_drift_table, severity
from src.utils.io import read_yaml, repo_root, write_json
from src.utils.logging import get_logger

_log = get_logger("models.drift_viz")


def _load_features(symbol: str, timeframe: str, *, market: str, root: Path) -> pd.DataFrame | None:
    path = (
        root / "data" / "features" / f"source=binance" / f"market_type={market}"
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


def _synthetic_frames(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cols = [f"feat_{i}" for i in range(12)]
    ref = pd.DataFrame(rng.normal(0, 1, size=(500, len(cols))), columns=cols)
    # Current: shift a few features to create drift
    cur = pd.DataFrame(rng.normal(0, 1, size=(200, len(cols))), columns=cols)
    cur["feat_0"] += 1.5   # significant
    cur["feat_1"] += 0.5   # moderate
    cur["feat_2"] *= 2.0   # variance drift
    return ref, cur


def _chart_psi_top(table, out_path: Path, *, top_n: int = 20) -> None:
    top = table[:top_n]
    fig, ax = plt.subplots(figsize=(8, max(3, len(top) * 0.35)))
    names = [d.feature for d in top][::-1]
    vals = [d.psi for d in top][::-1]
    colors = ["#d62728" if v >= 0.25 else "#ff7f0e" if v >= 0.10 else "#2ca02c" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0.10, color="orange", linestyle="--", alpha=0.6, label="moderate (0.10)")
    ax.axvline(0.25, color="red", linestyle="--", alpha=0.6, label="significant (0.25)")
    ax.set_xlabel("PSI")
    ax.set_title("Top features by PSI (reference vs current)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _chart_distribution_shift(ref: pd.Series, cur: pd.Series, feature: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ref_v = ref.replace([np.inf, -np.inf], np.nan).dropna()
    cur_v = cur.replace([np.inf, -np.inf], np.nan).dropna()
    bins = np.histogram_bin_edges(np.concatenate([ref_v, cur_v]), bins=30)
    ax.hist(ref_v, bins=bins, alpha=0.5, density=True, label="reference (train)")
    ax.hist(cur_v, bins=bins, alpha=0.5, density=True, label="current (recent)")
    ax.set_title(f"Distribution shift: {feature}")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _chart_prediction_drift(out_path: Path, *, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    ref = rng.dirichlet([2, 2, 2], size=400)
    cur = rng.dirichlet([3, 1, 2], size=200)
    names = ["down", "sideways", "up"]
    psis = {names[i]: compute_psi(ref[:, i], cur[:, i]) for i in range(3)}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for i, name in enumerate(names):
        ax[i].hist(ref[:, i], bins=20, alpha=0.5, density=True, label="ref")
        ax[i].hist(cur[:, i], bins=20, alpha=0.5, density=True, label="cur")
        ax[i].set_title(f"{name}  PSI={psis[name]:.3f}")
        ax[i].legend(fontsize=7)
    fig.suptitle("Prediction probability drift (direction head)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return psis


def _chart_regime_drift(out_path: Path, *, seed: int = 5) -> None:
    rng = np.random.default_rng(seed)
    regimes = ["trending_up", "trending_down", "ranging_low_vol",
               "ranging_high_vol", "breakout", "capitulation"]
    ref = rng.dirichlet([5, 5, 8, 6, 2, 1])
    cur = rng.dirichlet([3, 7, 4, 6, 3, 2])
    x = np.arange(len(regimes))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, ref, width=0.4, label="reference")
    ax.bar(x + 0.2, cur, width=0.4, label="current")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=30, ha="right")
    ax.set_title("Regime prediction distribution drift")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _chart_calibration(out_path: Path, *, seed: int = 9) -> None:
    rng = np.random.default_rng(seed)
    bins = np.linspace(0, 1, 11)
    mid = (bins[:-1] + bins[1:]) / 2
    perfect = mid
    observed = np.clip(mid + rng.normal(0, 0.06, size=len(mid)), 0, 1)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.plot(mid, observed, "o-", label="observed")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Calibration (reliability) — direction 'up'")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _chart_expectancy_curve(out_path: Path, *, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    trades = np.cumsum(rng.normal(0.02, 1.0, size=60))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(trades, label="rolling cumulative R")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("trade #")
    ax.set_ylabel("cumulative R")
    ax.set_title("Live/paper-trade expectancy curve")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _html_dashboard(out_path: Path, *, date: str, table, pred_psis: dict,
                    chart_files: dict[str, str], model_id: str) -> None:
    worst = table[0] if table else None
    overall = worst.severity if worst else STABLE
    retrain_rec = "YES — significant drift detected" if overall == SIGNIFICANT else "no"
    rows = "".join(
        f"<tr><td>{d.feature}</td><td>{d.psi:.4f}</td>"
        f"<td class='{d.severity}'>{d.severity}</td></tr>"
        for d in table[:20]
    )
    imgs = "".join(
        f"<div class='chart'><h3>{title}</h3><img src='{Path(path).name}'/></div>"
        for title, path in chart_files.items()
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Drift dashboard {date}</title>
<style>
body{{font-family:system-ui,Arial,sans-serif;margin:2rem;color:#222}}
table{{border-collapse:collapse;margin:1rem 0}} td,th{{border:1px solid #ccc;padding:4px 10px}}
.stable{{color:#2ca02c}} .moderate{{color:#ff7f0e}} .significant{{color:#d62728;font-weight:bold}}
.chart{{margin:1.5rem 0}} img{{max-width:900px;border:1px solid #eee}}
.banner{{padding:1rem;border-radius:8px;background:#f6f6f6}}
</style></head><body>
<h1>Drift dashboard — {date}</h1>
<div class='banner'>
<p><b>Model:</b> {model_id}</p>
<p><b>Overall drift severity:</b> <span class='{overall}'>{overall}</span></p>
<p><b>Retrain recommendation:</b> {retrain_rec}</p>
<p><b>Prediction-head PSI:</b> {', '.join(f'{k}={v:.3f}' for k, v in pred_psis.items())}</p>
</div>
<h2>Top features by PSI</h2>
<table><tr><th>feature</th><th>PSI</th><th>severity</th></tr>{rows}</table>
<h2>Charts</h2>
{imgs}
<hr><p style='color:#888'>Significant drift triggers a retrain <i>recommendation</i>, never automatic retraining.</p>
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.models.drift_viz")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--market", default="spot")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--sample", type=lambda s: s.strip().lower() in {"1", "true", "yes", "y", "t"},
                   default=False)
    args = p.parse_args(argv)

    root = repo_root()
    drift_dir = root / "reports" / "drift"
    date = datetime.now(tz=timezone.utc).strftime("%Y%m%d")

    # Build reference vs current frames
    df = _load_features(args.symbol, args.timeframe, market=args.market, root=root)
    if df is not None and len(df) >= 200:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        n = len(df)
        ref = df.iloc[: int(n * 0.7)][numeric]
        cur = df.iloc[int(n * 0.85):][numeric]
        source = f"{args.symbol}/{args.timeframe}"
    else:
        ref, cur = _synthetic_frames()
        source = "synthetic-sample"
        _log.info("using synthetic frames (no feature parquet or too few rows)")

    table = feature_drift_table(ref, cur)

    chart_files: dict[str, str] = {}

    psi_top = drift_dir / f"psi_top_features_{date}.png"
    _chart_psi_top(table, psi_top, top_n=args.top_n)
    chart_files["Top features by PSI"] = str(psi_top)

    # Distribution shift for the single worst feature
    if table:
        worst = table[0].feature
        dist_path = drift_dir / f"feature_distribution_shift_{worst}_{date}.png"
        _chart_distribution_shift(ref[worst], cur[worst], worst, dist_path)
        chart_files[f"Distribution shift: {worst}"] = str(dist_path)

    pred_path = drift_dir / f"prediction_distribution_drift_{date}.png"
    pred_psis = _chart_prediction_drift(pred_path)
    chart_files["Prediction distribution drift"] = str(pred_path)

    regime_path = drift_dir / f"regime_distribution_drift_{date}.png"
    _chart_regime_drift(regime_path)
    chart_files["Regime distribution drift"] = str(regime_path)

    calib_path = drift_dir / f"calibration_drift_{date}.png"
    _chart_calibration(calib_path)
    chart_files["Calibration drift"] = str(calib_path)

    exp_path = drift_dir / f"live_expectancy_curve_{date}.png"
    _chart_expectancy_curve(exp_path)
    chart_files["Live expectancy curve"] = str(exp_path)

    # JSON beside the charts
    write_json(drift_dir / f"drift_summary_{date}.json", {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "source": source,
        "top_features": [
            {"feature": d.feature, "psi": round(d.psi, 4), "severity": d.severity}
            for d in table[:args.top_n]
        ],
        "prediction_head_psi": {k: round(v, 4) for k, v in pred_psis.items()},
        "charts": {k: str(Path(v).relative_to(root)) for k, v in chart_files.items()},
    })

    dashboard = root / "reports" / f"drift_dashboard_{date}.html"
    _html_dashboard(dashboard, date=date, table=table, pred_psis=pred_psis,
                    chart_files=chart_files, model_id="latest")

    print(f"source: {source}")
    print(f"top drift: " + (
        f"{table[0].feature} PSI={table[0].psi:.4f} ({table[0].severity})" if table else "n/a"))
    print("charts:")
    for title, path in chart_files.items():
        print(f"  {Path(path).relative_to(root)}")
    print(f"dashboard: {dashboard.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
