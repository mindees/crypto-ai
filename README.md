<div align="center">

# 🪙 mindees · crypto-ai

### BTC/ETH multi-task, multi-timeframe market-intelligence pipeline

*Free data in. Causal features. Honest baselines. A probabilistic decision-support model out.*

[![CI](https://github.com/mindees/crypto-ai/actions/workflows/smoke_tests.yml/badge.svg)](https://github.com/mindees/crypto-ai/actions/workflows/smoke_tests.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![tensorflow](https://img.shields.io/badge/tensorflow-2.21-orange.svg)](https://www.tensorflow.org/)
[![tests](https://img.shields.io/badge/tests-88%20passing-brightgreen.svg)](#testing)
[![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#license)
[![status](https://img.shields.io/badge/build-8%2F8%20phases%20complete-success.svg)](#build-phases)
[![stars](https://img.shields.io/github/stars/mindees/crypto-ai?style=social)](https://github.com/mindees/crypto-ai/stargazers)

**Repository:** <https://github.com/mindees/crypto-ai>

</div>

> [!WARNING]
> **Decision-support only. Not financial advice.**
> Markets are noisy and adversarial. **No model reliably predicts BTC/ETH prices all the time.**
> A model that fails to beat honest baselines after fees and slippage **must not be used for trading.**
> Backtests are not promises. You are responsible for your own capital.

---

## Table of contents

- [What this is](#what-this-is)
- [What it does — and does not — do](#what-it-does--and-does-not--do)
- [Why "deepest verified free history"](#why-deepest-verified-free-history)
- [Pipeline at a glance](#pipeline-at-a-glance)
- [Quickstart](#quickstart)
- [Data sources](#data-sources)
- [Features & the rule-based scorecard](#features--the-rule-based-scorecard)
- [Labels](#labels)
- [Feature selection](#feature-selection)
- [Class imbalance](#class-imbalance)
- [Model architecture](#model-architecture)
- [PlantGuard-style training](#plantguard-style-training)
- [Evaluation & honest baselines](#evaluation--honest-baselines)
- [Backtesting](#backtesting)
- [Prediction output & how to read it](#prediction-output--how-to-read-it)
- [Serving (FastAPI) & alerts](#serving-fastapi--alerts)
- [Model registry, promotion & rollback](#model-registry-promotion--rollback)
- [Drift, retraining & shadow A/B](#drift-retraining--shadow-ab)
- [Compute: Kaggle, Colab & GitHub Actions](#compute-kaggle-colab--github-actions)
- [Command reference](#command-reference)
- [Repository layout](#repository-layout)
- [Configuration & API keys](#configuration--api-keys)
- [Testing](#testing)
- [Build phases](#build-phases)
- [Limitations & known gaps](#limitations--known-gaps)
- [License & disclaimer](#license--disclaimer)

---

## What this is

`crypto-ai` ingests the deepest **free, verifiable** historical data for **BTCUSDT** and
**ETHUSDT** (spot + USDT-margined perpetual futures), engineers strictly causal features,
and trains a **multi-task TensorFlow model** with four heads:

| Head | Type | Classes |
|---|---|---|
| **direction** | 3-class softmax | `down` · `sideways` · `up` |
| **regime** | 6-class softmax | `trending_up` · `trending_down` · `ranging_low_vol` · `ranging_high_vol` · `breakout` · `capitulation` |
| **cycle** | 4-class softmax | `accumulation` · `bull` · `distribution` · `bear` |
| **trade_quality** | binary sigmoid | would the trade reach ≥2R before stop, after fees? |

Everything is validated with **purged + embargoed walk-forward** splits (never random shuffle),
backtested with realistic fees + slippage, and compared against **honest baselines**. If the
model can't beat buy-and-hold / EMA / RSI-MACD / majority-class after costs, the reports say so
plainly.

## What it does — and does not — do

<table>
<tr><th>✅ Does</th><th>🚫 Does not</th></tr>
<tr><td valign="top">

- Free, public data only (Binance, blockchain.info, Alternative.me, CoinGecko, FRED/yfinance)
- Idempotent, checksum-verified ingestion
- Causal feature engineering (no lookahead — enforced by tests)
- Multi-timeframe transformer with cross-timeframe attention
- Honest evaluation + event-driven backtest vs baselines
- Model registry with gated promotion, rollback, shadow A/B
- PSI drift detection + visual dashboard + retrain recommendations
- FastAPI serving + disabled-by-default alerts

</td><td valign="top">

- Claim coverage "from asset genesis" (it can't — see below)
- Require any paid data source (paid adapters are disabled stubs)
- Auto-promote or auto-trade — promotion needs an explicit command
- Send alerts by default — every channel ships disabled
- Emit hard buy/sell calls — signals are hedged biases
- Pretend a weak model is strong — failure to beat baselines is reported

</td></tr>
</table>

## Why "deepest verified free history"

Binance does **not** have BTC from 2009 or ETH from Ethereum genesis. So this project never
claims "zero to today." Instead it **discovers and reports the first verified candle per
source**, and uses that wording everywhere:

> *"Deepest free verified history available from each configured source."*

Observed free coverage (auto-detected, written to `metadata/watermarks.json`):

| Market | Symbol | First verified candle |
|---|---|---|
| spot | BTCUSDT / ETHUSDT | **2017-08-17** |
| futures (USDT-M) | BTCUSDT / ETHUSDT | **2020-01-01** |
| on-chain (BTC, blockchain.info) | — | **2009** (daily) |
| Fear & Greed | — | **2018-02-01** |

> Binance launched USDT-M futures in Sep 2019, but the public bulk archives only start
> 2020-01 — so the pipeline reports 4 leading 404s and starts there. That honesty is the point.

## Pipeline at a glance

```
                 ┌──────────────────────────────────────────────────────────────┐
  FREE SOURCES   │  Binance bulk · ccxt · derivatives REST · blockchain.info ·   │
                 │  Alternative.me F&G · CoinGecko · yfinance/FRED               │
                 └───────────────┬──────────────────────────────────────────────┘
                                 ▼
   ingest/  ──►  data/processed/*.parquet  (idempotent, checksum-verified, UTC)
                                 ▼
  features/ ──►  causal indicators · structure · patterns · flow · onchain ·
                 sentiment · macro · scorecard            (no lookahead)
                                 ▼
   labels/  ──►  triple-barrier direction · regime · cycle · trade-quality
                                 ▼
 datasets/  ──►  purged+embargoed walk-forward windows · train-only scaler
                                 ▼
  models/   ──►  MTF Transformer (4 heads)  ──►  train · PlantGuard 2-phase
                                 ▼
            evaluate (vs baselines) · backtest (fees+slippage) · thresholds · predict
                                 ▼
  registry · promote · rollback · drift · drift_viz · shadow A/B · retrain_check
                                 ▼
  serve/    ──►  FastAPI · scheduler · drift dashboard · disabled-by-default alerts
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/mindees/crypto-ai.git
cd crypto-ai

# 2. Create a venv (Python 3.11 recommended; 3.13 also works with TF 2.21)
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt

# 4. Sanity checks
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
python -m src.utils.hardware
pytest tests/ -q          # 88 passing
```

End-to-end smoke run (CPU, tiny sample windows — proves the whole pipeline):

```bash
# Ingest 1d OHLCV for all four combos (deepest verified history)
python -m src.ingest.binance_bulk --symbols BTCUSDT ETHUSDT --market-types spot futures_um --timeframes 1d

# Free context adapters
python -m src.ingest.sentiment
python -m src.ingest.coingecko
python -m src.ingest.onchain
python -m src.ingest.derivatives --symbols BTCUSDT ETHUSDT

# Features → labels → dataset → train (sample) → evaluate → backtest → predict
python -m src.features.build_matrix --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --sample true
python -m src.labels.labeling       --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --sample true
python -m src.datasets.build_dataset --symbols BTCUSDT ETHUSDT --timeframes 1h --sample true
python -m src.models.train_like_plantguard --timeframes 1h --sample true --phase1-epochs 1 --phase2-epochs 1
python -m src.models.evaluate --latest --sample true
python -m src.backtest.engine --latest --sample true
python -m src.models.predict  --latest --symbols BTCUSDT ETHUSDT --timeframes 1h 4h
```

> Optional extras (TA-Lib, polars, vectorbt, SHAP) live in `requirements-optional.txt`.
> The codebase works **without** TA-Lib — indicators are pandas-native with a TA-Lib hook.

---

## Data sources

All free by default. Every adapter returns a dated DataFrame **or a clear "unavailable" log** —
nothing silently fabricates data.

| Source | Module | Data | Coverage / limitation |
|---|---|---|---|
| **Binance bulk** (`data.binance.vision`) | `ingest/binance_bulk.py` | spot + USDT-M OHLCV, all intervals | Primary OHLCV. Monthly + daily archives, SHA-256 verified. ms→µs timestamp switch (2025-01) handled; spot/futures column aliases normalized. |
| ccxt | `ingest/ccxt_incremental.py` | recent OHLCV delta | Fallback only. |
| Binance derivatives REST | `ingest/derivatives.py` | funding, OI, long/short, taker vol | Funding paginated to launch; **OI / ratios / taker are recent-only (~30 days)** — marked as such. |
| blockchain.info | `ingest/onchain.py` | BTC hash-rate, difficulty, miner rev, txs, addresses, supply, fees | Daily, back to 2009. |
| Etherscan | `ingest/onchain.py` | ETH supply / gas snapshots | **Disabled unless `ETHERSCAN_API_KEY` set**; snapshots only. |
| Alternative.me | `ingest/sentiment.py` | Fear & Greed index | Daily since 2018-02; mostly BTC-driven. |
| CoinGecko (free) | `ingest/coingecko.py` | BTC/ETH dominance, global mcap | Snapshot endpoint — accrues history per run; rate-limited. |
| yfinance / FRED | `ingest/macro.py` | S&P500, Nasdaq, DXY, VIX, FedFunds, CPI | yfinance is **frequently rate-limited (429)** from data-center IPs; FRED needs a free key + opt-in. |
| Glassnode / CryptoQuant / CoinGlass / Coinalyze / Amberdata | `ingest/paid_stubs.py` | — | **Disabled stubs.** Bring your own key + implement to enable. |

An `onchain_coverage_score` per asset is written to `metadata/onchain_coverage.json` so the
modeling layer knows whether on-chain features are strong or mostly missing.

---

## Features & the rule-based scorecard

All features are **causal** — row `t` uses only data at or before `t`. This is enforced by
`tests/test_no_lookahead.py`, which mutates future rows and asserts past features don't change.

- **Indicators** (`features/indicators.py`): EMA 9/21/50/120/200 + stack score, EMA120 cycle signal, golden/death cross, RSI(14) + slope/zscore, MACD, Bollinger %B/bandwidth, ATR, OBV, VWAP, realized volatility, distance-from-ATH/52w.
- **Structure** (`features/structure.py`): causal swing highs/lows, HH/HL/LH/LL, market-structure score, range/breakout, liquidity-sweep proxy, FVG / order-block / CHoCH **proxies** (clearly labelled).
- **Patterns** (`features/patterns.py`): doji, hammer, shooting star, engulfing, inside/outside bar, pin bar.
- **Flow** (`features/flow.py`): funding rate + z-score + extremes, OI change, price/OI quadrants, taker delta, **CVD proxy** (named a proxy — not true full-market CVD), basis, funding/OI governor.
- **Sentiment / on-chain / macro**: causally joined via `merge_asof(direction="backward")` so a bar never sees a value published after its close.

The **scorecard** (`features/scorecard.py`) is a transparent, rule-derived assessment **separate
from the ML model**. Missing inputs are reported as the literal string `"unavailable"` — never
guessed, never zero. It powers the `scorecard` field in the prediction JSON and alerts.

---

## Labels

`labels/labeling.py` produces four targets (targets may look forward; **features may not**):

- **Direction** — triple-barrier: upper = `close + k·ATR`, lower = `close − k·ATR`, vertical = `N` bars. Uses intrabar high/low path. Same-candle double-touch → `ambiguous` (excluded from training). Per-timeframe `k` and `N` in `configs/config.yaml`.
- **Regime** — rule-based from EMA slope, EMA-stack, ATR percentile, realized-vol percentile, structure.
- **Cycle** — anchored on BTC halving dates (`reference/halvings.csv`) + drawdown-from-ATH + 200-week-MA position. ETH inherits the BTC anchor plus its own drawdown.
- **Trade quality** — binary: did the directional target reach **≥2R before a −1R stop**, after fees + slippage?

Validation **purges overlapping label horizons** and applies an embargo so triple-barrier
windows don't leak across the train/val/test boundary.

---

## Feature selection

`features/selection.py` runs **inside each training fold** (never on test):

1. Drop features below `min_non_null_ratio` (0.85).
2. Drop near-zero-variance features.
3. Drop one of each highly-correlated pair (`max_pairwise_corr` = 0.95).
4. Rank by **mutual information** (training window only).
5. Rank by **permutation importance** of a light GBM (validation window only).
6. Keep `always_keep` features + top-K (`final_top_k` = 120).

`tests/test_feature_selection_no_leakage.py` proves selection is unchanged when the test split
is mutated — i.e., the test set is never read during selection.

---

## Class imbalance

- **Class weights** by default: `weight = 1/√frequency`, normalized to mean 1.0 (no SMOTE — synthetic candle windows are unrealistic).
- Optional **focal loss** flag.
- **Decision-threshold tuning** (`models/thresholds.py`) on validation only — maximizes macro F1 subject to per-trade-class precision floors, and allows a `no_trade` zone when confidence is low.
- Stratified reporting by regime.

---

## Model architecture

Default: **MTF Transformer with cross-timeframe attention** (`models/multitask_model.py`).

```
fast_seq ─┐
main_seq ─┼─► per-tf encoder (LayerNorm → Dense → PosEnc → 3× Transformer block → attention-pool)
slow_seq ─┘                    │
                               ▼
                 cross-timeframe MultiHeadAttention  ◄── asset & timeframe embeddings
                               │
            context branch ───►├─► shared trunk (Dense 256 → 128, BN + dropout)
                               ▼
        ┌──────────┬───────────┬──────────────┬────────────────┐
   direction(3)  regime(6)   cycle(4)   trade_quality(1, sigmoid)
```

- Optimizer AdamW (lr 3e-4, weight-decay 1e-4, clipnorm 1.0), cosine decay + warmup.
- Mixed precision auto-enabled on GPU; **`tf.distribute.MirroredStrategy()` auto-selected when ≥2 GPUs** (Kaggle dual-T4).
- A config switch `use_multi_timeframe_fusion: false` falls back to a compact single-timeframe transformer for CPU smoke tests.
- Custom layers are serialization-safe (reload + predict verified).

---

## PlantGuard-style training

`models/train_like_plantguard.py` adapts the polished Plant Guard image-classification workflow
to **causal time series** (no image augmentation, no MobileNet). Two phases:

1. **Phase 1 — head warmup.** If an SSL-pretrained encoder exists, freeze it and train fusion/trunk/heads at a higher LR. If not, run a lower-LR warmup of the full model (spec-compliant fallback).
2. **Phase 2 — fine-tune.** Unfreeze the last N transformer blocks at a much lower LR.

Artifacts per run (`artifacts/runs/<id>_plantguard/`): `model.keras`, `phase{1,2}_best.keras`,
`training_curves.png`, `confusion_{direction,regime,cycle}.png`, `classification_report.json`,
`prediction_demo.json`, `class_indices.json`, `dataset_spec.json`, + a reload + sanity-predict check.

```bash
python -m src.models.train_like_plantguard \
  --symbols BTCUSDT ETHUSDT --timeframes 1h \
  --phase1-epochs 10 --phase2-epochs 25
```

| Plant Guard concept | Time-series equivalent here |
|---|---|
| image class distribution | label distribution (direction/regime/cycle/quality) |
| sample images | sample market windows |
| MobileNet pretrained base | optional self-supervised time-series encoder |
| frozen base → fine-tune | freeze encoder → unfreeze last blocks |
| confusion matrix | direction / regime / cycle confusion matrices |
| single-image prediction demo | latest-market prediction demo |
| Flask inference | FastAPI prediction endpoint |

---

## Evaluation & honest baselines

`models/evaluate.py` reports per-head metrics on the held-out split and compares the direction
head against **majority-class** and **random** baselines, tunes thresholds on validation, and
writes `reports/eval_<id>.md` + `.json`. The report **states plainly whether the model beats
baselines** — and on a 1–2 epoch CPU smoke run it correctly says it does **not**.

## Backtesting

`backtest/` is a real **event-driven** simulator (not just classification metrics):

- Risk-based sizing, ATR stops, **TP1/TP2/TP3 partials** (33/33/34%), SL→breakeven after TP1, SL→TP1 after TP2, vertical-barrier exit.
- Fees (bps/side) + **ATR/volume-aware slippage**, intrabar high/low logic, max-daily-loss cap.
- Metrics: total return, CAGR, max drawdown, profit factor, expectancy, Sharpe, Sortino, win rate, avg/median R, fee + slippage drag, long/short split, worst-10 trades.
- Compares **model vs buy-and-hold / EMA-trend / RSI-MACD / random / no-trade**.
- Default leverage **1×** for evaluation honesty.

```bash
python -m src.backtest.engine --latest --sample true
# → reports/backtest_<id>.md · backtest_<id>.json · trades_<id>.csv
```

> **Futures leverage / margin / liquidation simulation** is specified as optional and ships
> **disabled** (`backtest.futures_margin.enabled: false`). The default evaluation is spot/1×.

---

## Prediction output & how to read it

`models/predict.py` loads the production model, rebuilds the latest feature window with the
**saved scaler/imputer**, applies tuned thresholds + the funding/OI governor, attaches the
scorecard, and writes `reports/latest_predictions.json`:

```jsonc
{
  "asset": "BTCUSDT", "timeframe": "1h", "model_id": "...",
  "model_outputs": {
    "direction": { "down": 0.21, "sideways": 0.28, "up": 0.51 },
    "regime":    { "predicted": "trending_up", "confidence": 0.63 },
    "cycle":     { "predicted": "bull", "confidence": 0.58 },
    "trade_quality": { "probability": 0.62 }
  },
  "signal": { "action": "no_trade", "reason": "confidence below threshold",
              "long_threshold": 0.58, "short_threshold": 0.58, "quality_threshold": 0.60 },
  "scorecard": { "trend_direction": "up", "rsi_14": 61.2, "funding_state": "slightly_positive", "...": "..." },
  "risk_warning": "Decision-support only. Not financial advice. Validate manually before trading."
}
```

**Signal vocabulary is intentionally hedged** — never a hard buy/sell:
`long_bias` · `short_bias` · `no_trade` · `range_wait` · `high_risk`.

---

## Serving (FastAPI) & alerts

The API (`serve/api.py`) is deliberately **TensorFlow-free** — it serves the JSON `predict.py`
writes, keeping the web tier light.

```bash
uvicorn src.serve.api:app --host 0.0.0.0 --port 8000
python -m src.serve.api --smoke-test     # hits every route via TestClient
```

| Endpoint | Returns |
|---|---|
| `GET /health` | service status + disclaimer |
| `GET /model/current` | production model pointer |
| `GET /registry` | full model registry |
| `GET /predict/latest` | latest predictions for all combos |
| `GET /predict/{asset}/{timeframe}` | one prediction |
| `GET /scorecard/{asset}/{timeframe}` | the rule-based scorecard |
| `GET /drift/latest` | most recent drift dashboard (HTML) |
| `POST /predict/refresh` | re-read predictions (`?run_predict=true` to regenerate) |

**Alerts** (`serve/alert_templates.py`, `serve/alerts.py`) are **disabled by default** and require
both a config flag **and** environment credentials. The canonical payload includes `asset,
timeframe, signal, direction_confidence, trade_quality_probability, regime, cycle_phase,
entry/stop/tp1/tp2/tp3_reference, estimated_rr, risk_per_trade_pct, leverage,
liquidation_buffer_pct, scorecard{...}, warnings[], cooldown_minutes`. Alerts fire **only** for
`long_bias`/`short_bias` that clear confidence + trade-quality + cooldown gates and a non-stale
model — never for `no_trade`/`range_wait`/`high_risk`.

```bash
python -m src.serve.alert_templates --sample true   # preview payload + Telegram/email render
python -m src.serve.scheduler --refresh-minutes 15   # local predict→alert loop (--once for one tick)
```

---

## Model registry, promotion & rollback

`metadata/model_registry.json` tracks every run (`candidate` → `production` → `archived` /
`rejected` / `rolled_back`). **Artifacts are never overwritten** — only pointers move.

```bash
python -m src.models.registry --list          # sync + list all runs
python -m src.models.promote  --latest --dry-run   # show gate decision, change nothing
python -m src.models.promote  --model-id <id>      # apply (gated)
python -m src.models.rollback --model-id <id>      # restore a previous production model
```

Promotion is **gated** (and never silent): beats production on direction macro F1 by ≥ threshold,
positive expectancy after costs, profit factor ≥ minimum, drawdown not materially worse.

## Drift, retraining & shadow A/B

- **Drift** (`models/drift.py`): PSI per feature — `<0.10` stable · `0.10–0.25` moderate · `≥0.25` significant.
- **Retrain check** (`models/retrain_check.py`): new-bars + PSI + performance triggers → `metadata/retrain_status.json` + report. **Recommends only — never auto-trains.**
- **Drift dashboard** (`models/drift_viz.py`): 6 charts + HTML dashboard under `reports/`.
- **Shadow A/B** (`models/shadow.py`, `models/ab_compare.py`): run a candidate beside production on identical bars, log both, compare agreement / signal counts / confidence. Promotion after shadow still requires an explicit command.

```bash
python -m src.models.retrain_check
python -m src.models.drift_viz   --sample true
python -m src.models.shadow      --candidate latest --sample true
python -m src.models.ab_compare  --candidate latest --sample true
```

---

## Compute: Kaggle, Colab & GitHub Actions

| Environment | Role | Notes |
|---|---|---|
| **Kaggle** | primary training | dual-T4 → `MirroredStrategy` auto-selected. Notebooks: `notebooks/kaggle_train.ipynb`, `notebooks/kaggle_train_plantguard_style.ipynb`. |
| **Colab** | secondary training | T4 single-GPU; mount Drive for resume across disconnects. Notebooks: `notebooks/colab_train.ipynb`, `notebooks/colab_train_plantguard_style.ipynb`. |
| **GitHub Actions** | automation only | **Never trains.** Uses `requirements-ci.txt` (no TF). |
| **Local** | CPU smoke + serving | Windows + TF ≥ 2.11 is CPU-only (use WSL2 for GPU). |

All four notebooks import the real `src/` modules (no duplicated logic) and support **resume**
from existing checkpoints.

**Workflows** (`.github/workflows/`):
- `daily_data.yml` — daily delta → validation → retrain check → conditional Kaggle push → commits **only metadata/reports** (large data is never committed).
- `smoke_tests.yml` — `pytest tests/` on push/PR + a guard that fails if large data is git-tracked.
- `weekly_retrain_notice.yml` — weekly retrain **recommendation** (no training).

If Kaggle secrets are absent, the daily job still runs local validation and logs that the push was skipped.

---

## Command reference

<details><summary><b>Click to expand the full CLI</b></summary>

```bash
# Ingestion
python -m src.ingest.binance_bulk --symbols BTCUSDT ETHUSDT --market-types spot futures_um --timeframes 1h 4h 1d
python -m src.ingest.derivatives  --symbols BTCUSDT ETHUSDT
python -m src.ingest.sentiment
python -m src.ingest.coingecko
python -m src.ingest.onchain
python -m src.ingest.macro
python -m src.ingest.daily_update --symbols BTCUSDT ETHUSDT --timeframes 1h 4h 1d

# Features / labels / selection
python -m src.features.build_matrix --symbols BTCUSDT ETHUSDT --timeframes 1h 4h
python -m src.labels.labeling       --symbols BTCUSDT ETHUSDT --timeframes 1h 4h
python -m src.features.selection    --symbol BTCUSDT --timeframe 1h

# Dataset / train
python -m src.datasets.build_dataset --symbols BTCUSDT ETHUSDT --timeframes 1h
python -m src.models.train                  --timeframe 1h --epochs 60
python -m src.models.train_like_plantguard  --timeframes 1h --phase1-epochs 10 --phase2-epochs 25

# Evaluate / backtest / predict
python -m src.models.evaluate --latest
python -m src.backtest.engine --latest
python -m src.models.predict  --latest --symbols BTCUSDT ETHUSDT --timeframes 1h 4h

# Lifecycle
python -m src.models.registry --list
python -m src.models.promote  --latest --dry-run
python -m src.models.rollback --model-id <id>
python -m src.models.retrain_check
python -m src.models.drift_viz  --sample true
python -m src.models.shadow     --candidate latest --sample true
python -m src.models.ab_compare --candidate latest --sample true

# Serve
uvicorn src.serve.api:app --host 0.0.0.0 --port 8000
python -m src.serve.scheduler --refresh-minutes 15
python -m src.utils.hardware
python -m src.utils.validation_cli
```

</details>

## Repository layout

```text
configs/            main YAML config
data/               raw / interim / processed / features / labels (gitignored except samples)
metadata/           source registry, watermarks, checksums, model registry, retrain/coverage
reference/          halvings.csv
src/
  ingest/           binance_bulk, ccxt, derivatives, onchain, sentiment, macro, coingecko,
                    paid_stubs, daily_update
  features/         indicators, structure, patterns, flow, onchain, sentiment, macro,
                    scorecard, selection, build_matrix
  labels/           labeling (triple-barrier, regime, cycle, trade-quality)
  datasets/         build_dataset (windowing, purged walk-forward, train-only scaler)
  models/           multitask_model, train, train_like_plantguard, evaluate, predict,
                    thresholds, registry, promote, rollback, drift, drift_viz,
                    shadow, ab_compare, retrain_check
  backtest/         engine, broker, metrics, strategies, costs
  serve/            api, scheduler, drift_dashboard, alert_templates, alerts
  utils/            io, time, logging, seeds, validation, validation_cli, hardware
notebooks/          kaggle/colab train + PlantGuard-style notebooks
.github/workflows/  daily_data, smoke_tests, weekly_retrain_notice
tests/              88 tests (no-lookahead, labeling, idempotency, schema, scorecard,
                    selection-no-leakage, backtest, registry, drift, shadow, serving, alerts)
artifacts/          training runs + production pointer
reports/            eval / backtest / drift / shadow / retrain reports
```

## Configuration & API keys

Everything is driven by [`configs/config.yaml`](configs/config.yaml). No keys are required for
the free defaults. Optional keys go in `.env` (gitignored):

| Variable | Needed for |
|---|---|
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Kaggle dataset push (daily workflow) |
| `FRED_API_KEY` | FRED macro series (also set `sources.enable_fred: true`) |
| `ETHERSCAN_API_KEY` | ETH on-chain (also set `sources.enable_etherscan: true`) |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Telegram alerts (also `enable_telegram: true`) |
| `DISCORD_WEBHOOK_URL` | Discord alerts |
| `SMTP_HOST/PORT/USER/PASS/FROM/TO` | email alerts |

## Testing

```bash
pytest tests/ -q        # 88 passing
```

Coverage highlights: no-lookahead causality, triple-barrier label cases, ingestion idempotency
(+ ms→µs + futures-alias handling), feature schema, scorecard "unavailable" handling,
feature-selection no-leakage, backtest mechanics, registry/promote/rollback, PSI + drift viz,
shadow A/B, FastAPI contract, alert gating.

## Build phases

| Phase | Scope | Status |
|---|---|:--:|
| 0 | Scaffold, config, utils, hardware detection | ✅ |
| 1 | Binance bulk ingestion (idempotent, checksum-verified) | ✅ |
| 2 | Derivatives, sentiment, CoinGecko, on-chain, macro, paid stubs | ✅ |
| 3 | Features, labels, feature selection | ✅ |
| 4 | Dataset builder, MTF model, train + PlantGuard two-phase | ✅ |
| 5 | Evaluation, backtest, thresholds, prediction | ✅ |
| 6 | Registry, promotion, rollback, drift viz, shadow A/B | ✅ |
| 7 | FastAPI serving, scheduler, alerts, drift dashboard | ✅ |
| 8 | GitHub Actions + Kaggle/Colab notebooks | ✅ |

## Limitations & known gaps

- **No GPU locally** (Windows + TF ≥ 2.11). Shipped models are CPU smoke runs that **honestly do not beat baselines** — real training is the Kaggle/Colab notebooks' job.
- **yfinance/Yahoo rate-limits** macro pulls from data-center IPs; FRED is the reliable macro path.
- **Derivatives ratios/OI/taker are recent-only (~30 days)** via free REST; only funding has deep history.
- **`src/strategies/` research comparators** (ema120_cycle, wyckoff_proxy, ict_smc_proxy, funding_arbitrage_research, …) are not yet built — the core honest baselines live in `src/backtest/strategies.py`.
- **Futures margin/liquidation/funding simulation** is specified-but-disabled; default eval is spot/1×.
- **SSL encoder pretraining** (masked-window modeling) is optional; PlantGuard Phase 1 currently runs as a full-model warmup.
- SMC/ICT/Wyckoff and CVD features are **proxies**, labelled as such — not institutional-grade.

## License & disclaimer

Released under the **MIT License**.

> This software is provided for research and educational purposes. It is **not investment
> advice** and must not be relied upon for trading decisions. Crypto markets are extremely
> volatile and adversarial. Past performance and backtest results do not predict future results.
> The authors and contributors accept no liability for any loss arising from use of this software.

<div align="center">

**[⬆ back to top](#-mindees--crypto-ai)** · <https://github.com/mindees/crypto-ai>

</div>
