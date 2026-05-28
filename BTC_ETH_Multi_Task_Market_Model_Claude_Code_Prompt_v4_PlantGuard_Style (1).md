## V4 ADDITIONS

This version includes the final production gaps plus a PlantGuard-style training workflow:

1. Drift visualization dashboard and saved charts.
2. Exact Telegram/Discord/email alert message structure.
3. Optional futures leverage, margin, funding-fee, and liquidation-risk simulation.
4. Shadow A/B testing so a candidate model can run beside production before promotion.
5. A notebook/script training flow inspired by the Plant Guard AI project:
   - GPU check
   - single config block
   - data sanity checks
   - class/label distribution charts
   - sample window visualizations
   - two-stage training
   - callbacks
   - training curves
   - confusion matrices
   - classification reports
   - prediction demo
   - model save
   - metadata save
   - reload test
   - deployment smoke test

Do not copy image-classification logic directly. Adapt the structure to causal BTC/ETH time-series modeling.


---

## ROLE

You are a senior quantitative ML engineer, data engineer, TensorFlow production engineer, and crypto market-systems engineer.

Build a reproducible, production-grade BTC/ETH market-intelligence pipeline that:

1. Ingests the deepest **free, verifiable historical data available** for BTC and ETH.
2. Uses Binance public data as the primary exchange-data source, starting from each symbol/source’s **first verified available candle**, not from a guessed “asset genesis” date.
3. Keeps the dataset updated daily.
4. Engineers price-action, technical, derivatives, on-chain, sentiment, macro, and cycle-context features.
5. Trains a **specified multi-timeframe, multi-task TensorFlow model**, not a vague generic model.
6. Handles class imbalance, feature selection, model versioning, model promotion, rollback, and retraining triggers.
7. Runs daily lightweight data refresh through GitHub Actions.
8. Trains on Kaggle using **dual T4 GPUs when available** and on Google Colab using **T4 GPU when available**.
9. Serves the latest prediction through JSON files, FastAPI, and optional alert hooks.
10. Produces honest evaluation reports, baselines, backtests, and limitations.

This is a probabilistic decision-support system, not a price oracle and not financial advice.

---

## NON-NEGOTIABLE GROUND RULES

### 1. No lookahead / no leakage

Every feature must be computed causally.

Only use information available at or before the close of the current bar.

All timestamps must be UTC.

Triple-barrier labels create overlapping windows, so validation must use purged + embargoed walk-forward splits. Never use random shuffling for time-series validation.

No future-fill. Macro/sentiment/on-chain data must be forward-filled only after their real publication timestamp or known data timestamp.

### 2. Honesty over hype

Never fabricate accuracy, win rate, profitability, confidence, or “AI mastery.”

Always compare against honest baselines:

- Majority-class baseline
- Random baseline
- Buy-and-hold baseline
- Simple EMA trend baseline
- RSI/MACD technical baseline
- Sideways-only baseline
- No-trade baseline

If the model does not beat baseline after fees/slippage, say so clearly in the report.

### 3. Reproducible

Pin dependency versions.

Set seeds for Python, NumPy, TensorFlow, and any other stochastic component.

Log:

- Config
- Git SHA
- Dataset version
- Feature schema version
- Labeling config
- Model architecture config
- Training window
- Validation window
- Test window
- Random seed
- Hardware used
- Model registry version
- Promotion status

A fresh clone plus documented commands must reproduce the same pipeline.

### 4. No guessed API paths

For Binance bulk data, read the exact current directory schema from:

- `https://github.com/binance/binance-public-data`
- `https://data.binance.vision`

Do not hardcode unverified paths.

For ccxt, use the currently installed ccxt exchange metadata and docs.

For Binance REST endpoints, verify the endpoint and response schema before using it.

If a source is unavailable because of network, region, geo-blocking, 403, 429, or schema change, fail gracefully with a clear log message.

### 5. Idempotent ingestion

Re-running ingestion must not duplicate rows.

Use:

- Checksum verification where available
- Deduplication on `(source, market_type, symbol, timeframe, open_time)`
- Watermarks for each dataset
- Atomic writes
- Data-quality reports

Daily updates must fetch only missing data.

### 6. Free-by-default, no paid dependency

The system must run end-to-end using currently available free/public sources and free compute tiers.

Do not add a required paid source.

Do not add a trial-based source that later becomes paid.

Free API keys are allowed only when the service has a real free tier, for example FRED or Etherscan.

If a useful source is paid, for example Glassnode, CryptoQuant, CoinGlass paid endpoints, Coinalyze paid endpoints, or Amberdata, do not integrate it as required. Add a disabled-by-default adapter stub behind:

```yaml
enable_paid_sources: false
```

The adapter must not run unless explicitly enabled by the user later.

Because free tiers and public APIs can change, every external adapter must fail gracefully and log which features are missing.

### 7. Do not claim impossible historical coverage

Do not say “from zero to today” unless the source truly provides that asset’s full history.

Use this wording in code, docs, and reports:

> “Deepest free verified history available from each configured source.”

For Binance spot and futures data, start from the first verified Binance candle for that symbol/source.

For BTC/ETH on-chain data, start from the first verified timestamp available from the chosen free source.

For derivatives data, start from the first verified timestamp available from Binance public dumps or endpoint limits.

---

## HARD REALITY OF DATA COVERAGE

### Price data

Binance public data is the primary source for BTCUSDT and ETHUSDT spot/futures data.

Binance does not cover BTC from 2009 or ETH from Ethereum genesis.

The system must discover and report the actual first available timestamp per source, market type, symbol, and timeframe.

Example report:

```text
source=binance_public_data
market_type=spot
symbol=BTCUSDT
timeframe=1m
first_open_time=<actual detected UTC timestamp>
last_open_time=<actual detected UTC timestamp>
rows=<count>
```

### Derivatives data

Binance REST endpoints for open interest, long/short ratios, taker buy/sell volume, and basis may expose only recent history windows.

Use Binance public bulk metric dumps where available for historical backfill.

Use REST mainly for recent deltas and fallback.

Log exact coverage per metric.

Do not silently pretend a short REST window is full history.

### Liquidations and heatmaps

Historical liquidation heatmaps and historical full-depth order-book archives are not reliable free full-history sources.

Do not integrate CoinGlass/Coinalyze heatmaps as required sources.

Use free Binance live liquidation/order-book streams only going forward if implemented, and store them from the time the system starts.

For historical “liquidation pressure,” create only proxy features such as:

- Funding extremes
- Open-interest changes
- Large wick events
- Volume spikes
- Price/OI divergence
- Taker imbalance

Name these clearly as proxies.

### CVD

Do not claim institutional-grade CVD unless true tick-level aggressor-side data is available.

For the free version, derive:

```text
taker_delta_proxy = taker_buy_volume - taker_sell_volume
cvd_proxy = cumulative_sum(taker_delta_proxy)
```

Name it `cvd_proxy` or `taker_delta_proxy`, not true full-market CVD.

---

## TARGET ASSETS

Use only:

```yaml
assets:
  - BTCUSDT
  - ETHUSDT
```

Market types:

```yaml
market_types:
  - spot
  - futures_um_perpetual
```

Primary trade timeframes:

```yaml
trade_timeframes:
  - 15m
  - 1h
  - 4h
  - 1d
```

Additional context timeframes:

```yaml
context_timeframes:
  - 1m
  - 3m
  - 5m
  - 30m
  - 2h
  - 6h
  - 8h
  - 12h
  - 3d
  - 1w
  - 1mo
```

Important: build ingestion for all supported intervals, but default model training should focus on:

```text
15m, 1h, 4h, 1d
```

Use `1w` and `1mo` as cycle/context features.

---

## TECH STACK

Use Python 3.11.

Pin versions in `requirements.txt`.

Core:

```text
python-dotenv
pydantic
PyYAML
requests
aiohttp
tqdm
numpy
pandas
pyarrow
polars optional
ccxt
scikit-learn
tensorflow
tensorboard
matplotlib
joblib
fastapi
uvicorn
pydantic-settings
```

Indicators:

```text
TA-Lib optional
pandas-ta-classic fallback
```

Important:

TA-Lib often fails on some environments because it may need the system TA-Lib C library.

Implement an indicator abstraction:

1. Try TA-Lib.
2. If unavailable, use pandas-ta-classic.
3. If both fail, implement minimum required indicators manually or fail clearly.

Macro/equities:

```text
yfinance
fredapi
```

FRED requires a free `FRED_API_KEY` if macro FRED features are enabled.

Backtesting:

```text
vectorbt optional
custom event-driven backtester required
```

Do not rely only on vectorbt. Build a custom minimal event-driven backtester so the project works even if vectorbt installation fails.

Infra:

```text
kaggle
dvc optional only if free/local
```

Do not require DVC remote storage unless configured by the user.

---

## COMPUTE STRATEGY

### Kaggle

Primary training environment.

Use Kaggle notebooks with **dual T4 GPUs when available**.

The code must detect available GPUs:

```python
tf.config.list_physical_devices("GPU")
```

If two T4 GPUs are available, use:

```python
tf.distribute.MirroredStrategy()
```

If only one GPU is available, use normal single-GPU training.

If no GPU is available, allow a small CPU smoke test only.

Training must support:

- Checkpointing
- Resume from checkpoint
- Mixed precision when safe
- Batch-size auto-scaling
- Bounded training windows for quota safety
- Saving artifacts back to Kaggle dataset or downloadable output

### Google Colab

Secondary training environment.

Use Colab T4 GPU when available.

Do not assume GPU availability.

Training must support:

- Mounting Google Drive
- Pulling latest dataset from Kaggle or GitHub metadata
- Saving checkpoints frequently
- Resuming after disconnect
- Exporting final artifacts

### GitHub Actions

GitHub Actions must not train heavy models.

Use GitHub Actions only for:

- Lightweight daily data delta
- Metadata update
- Data validation
- Schema validation
- Small smoke tests
- Optional trigger/report that retraining is due

Do not commit large raw or processed market datasets to GitHub.

Commit only:

- Code
- Config
- Small reference tables
- Metadata
- Dataset manifest
- Checksums
- Reports
- Small sample files for tests

Push full datasets to Kaggle dataset storage or another explicitly configured free storage target.

---

## DATA SOURCES

Use these in priority order.

### A. Binance public data — primary OHLCV source

Use:

```text
https://data.binance.vision
https://github.com/binance/binance-public-data
```

Markets:

```text
spot
futures/um
```

Symbols:

```text
BTCUSDT
ETHUSDT
```

Intervals:

```text
1m
3m
5m
15m
30m
1h
2h
4h
6h
8h
12h
1d
3d
1w
1mo
```

Tasks:

1. Discover available files.
2. Download monthly archives first.
3. Download daily archives for recent data not covered by monthly.
4. Verify `.CHECKSUM` files when available.
5. Unzip.
6. Normalize schema.
7. Convert to UTC timestamps.
8. Deduplicate.
9. Write partitioned Parquet.
10. Store source coverage metadata.

Partitioning:

```text
data/processed/ohlcv/source=binance/market_type=<spot|futures_um>/symbol=<BTCUSDT|ETHUSDT>/timeframe=<tf>/
```

### B. ccxt incremental OHLCV fallback

Use ccxt Binance OHLCV fetching only for recent deltas and fallback.

Implement pagination and rate-limit sleeps.

Never use ccxt as the only full-history source without confirming coverage.

### C. Binance derivatives / flow data

Use Binance public dumps where available and REST endpoints for recent data.

Metrics:

- Funding rate
- Open interest statistics
- Global long/short account ratio
- Top trader long/short ratio if available
- Taker buy/sell volume
- Basis / futures-spot premium
- Mark price / premium index where useful

Important:

Each endpoint/source must report:

```text
metric
source
first_timestamp
last_timestamp
row_count
known_limitations
```

REST endpoints with recent-only windows must be marked as recent-only.

### D. On-chain free sources

Use adapter pattern.

Do **not** rely on one fragile provider.

Free/default adapters:

1. Coin Metrics community data/API where available.
2. Blockchain.com charts API for BTC metrics where available.
3. Etherscan free API key for ETH metrics if `ETHERSCAN_API_KEY` is provided.
4. Beaconchain/beaconcha.in free-access endpoints only if stable and accessible.
5. Manual CSV fallback for important metrics if the user later downloads them manually.

BTC possible metrics:

- Active addresses
- Hash rate
- Miner revenue
- Supply
- Transaction count
- Fees
- Difficulty
- On-chain transfer volume where available
- CVDD and Balanced Price only if a verified free source or correctly implemented calculation exists

ETH possible metrics:

- Gas fees
- Supply proxy
- Burn proxy
- Validator/staking proxy if free
- Transaction count
- Active addresses where available

If a metric is unavailable, do not break the pipeline. Log and continue.

For on-chain feature quality, output:

```text
onchain_coverage_score
```

per asset/timeframe so the model/report knows whether on-chain features are strong or mostly missing.

### E. Sentiment

Use Alternative.me Crypto Fear & Greed Index:

```text
https://api.alternative.me/fng/?limit=0&format=json
```

Features:

- Fear & Greed level
- Delta
- Rolling mean
- Extreme fear flag
- Extreme greed flag

Document that this is mostly Bitcoin/crypto-market sentiment, not ETH-specific.

### F. Market cap / dominance

Use CoinGecko free/demo API where available, with rate-limit handling.

Features:

- BTC dominance
- ETH market cap
- BTC market cap
- Global crypto market cap
- ETH/BTC relative strength

If CoinGecko rate-limits or blocks, degrade gracefully.

### G. Macro

Use:

- FRED with `FRED_API_KEY` if provided
- yfinance for broad market proxies

Possible features:

- Fed Funds rate
- CPI / inflation series
- DXY proxy if available
- S&P 500 returns
- Nasdaq returns
- Rolling correlation between BTC/ETH and equities
- Risk-on/risk-off proxy

FRED features must be disabled by default unless a free key is provided.

Forward-fill macro data causally only.

### H. ETF flow placeholder

ETF flow data can be useful, but reliable free structured APIs may be limited.

Create a disabled or manual CSV adapter:

```text
data/manual/etf_flows.csv
```

Do not scrape unstable sites by default.

---

## STRATEGY RESEARCH MODULES

Create a strategy-research layer for baseline and comparison only.

These strategies must not be treated as guaranteed systems. Every claim must be tested on the local dataset with fees/slippage.

Implement baseline strategy modules:

```text
src/strategies/
  ema120_cycle.py
  macd_rsi_mtf.py
  bollinger_mean_reversion.py
  funding_oi_governor.py
  eth_btc_rotation.py
  wyckoff_proxy.py
  ict_smc_proxy.py
  dynamic_grid_research.py
  funding_arbitrage_research.py
```

Rules:

1. These are baselines/research comparators, not production guarantees.
2. Any strategy with unsupported data must mark itself unavailable.
3. No strategy can claim a fixed win rate unless reproduced by this project’s own backtest.
4. Funding arbitrage must model spot/futures basis risk, funding availability, borrow/fee assumptions, and execution costs.
5. Dynamic grid must be research-only unless risk is fully modeled.
6. SMC/ICT/Wyckoff patterns must be proxy implementations and labeled as such.

The ML model should be compared against these baseline strategies where feasible.

---

## DATA LAYERS

Use this structure:

```text
data/
  raw/              # raw downloads, gitignored
  interim/          # temporary cleaned files, gitignored
  processed/        # normalized parquet, gitignored unless tiny sample
  features/         # feature matrices, gitignored unless tiny sample
  labels/           # labels, gitignored unless tiny sample
  samples/          # tiny committed sample for tests
  manual/           # optional user-provided CSVs
metadata/
  source_registry.yaml
  dataset_manifest.json
  watermarks.json
  checksums.json
  feature_registry.json
  model_registry.json
artifacts/
  runs/
reports/
reference/
  halvings.csv
```

Do not commit large data files.

---

## CONFIG

Create one main config:

```text
configs/config.yaml
```

It must include:

```yaml
project:
  name: btc_eth_multitask_market_model
  timezone: UTC
  seed: 42

assets:
  - BTCUSDT
  - ETHUSDT

market_types:
  - spot
  - futures_um_perpetual

timeframes:
  raw:
    - 1m
    - 3m
    - 5m
    - 15m
    - 30m
    - 1h
    - 2h
    - 4h
    - 6h
    - 8h
    - 12h
    - 1d
    - 3d
    - 1w
    - 1mo
  train_default:
    - 15m
    - 1h
    - 4h
    - 1d
  cycle_context:
    - 1w
    - 1mo

sources:
  enable_paid_sources: false
  enable_fred: false
  enable_etherscan: false
  enable_live_stream_storage: false
  enable_coingecko: true
  enable_fear_greed: true

storage:
  commit_large_data_to_git: false
  kaggle_dataset_slug: null
  local_data_dir: data

features:
  max_features_after_selection: 120
  min_non_null_ratio: 0.85
  max_pairwise_corr: 0.95
  variance_threshold: 0.000001
  mutual_info_top_k: 160
  permutation_importance_top_k: 140
  final_top_k: 120
  always_keep:
    - close
    - volume
    - returns_1
    - atr
    - rsi_14
    - macd_hist
    - ema_stack_score
    - realized_volatility
    - funding_rate
    - open_interest_change
    - fear_greed
    - btc_dominance
    - cycle_months_since_halving

labels:
  triple_barrier:
    15m:
      atr_multiple: 1.5
      vertical_barrier_bars: 32
    1h:
      atr_multiple: 1.8
      vertical_barrier_bars: 48
    4h:
      atr_multiple: 2.0
      vertical_barrier_bars: 42
    1d:
      atr_multiple: 2.5
      vertical_barrier_bars: 30

class_imbalance:
  use_class_weights: true
  class_weight_method: inverse_sqrt_frequency
  use_focal_loss: false
  focal_gamma: 2.0
  tune_decision_thresholds: true
  threshold_metric: macro_f1
  min_precision_per_trade_class: 0.45

validation:
  method: purged_embargoed_walk_forward
  embargo_bars: 10
  min_train_period_days: 365
  validation_period_days: 90
  test_period_days: 90
  num_walk_forward_splits: 5

model:
  architecture: mtf_transformer_attention
  use_multi_timeframe_fusion: true
  alternative_architectures:
    - single_timeframe_transformer
    - tcn_attention
    - lstm_gru_attention
  mixed_precision: true
  use_multi_gpu_if_available: true
  sequence_length:
    15m: 128
    1h: 128
    4h: 96
    1d: 90
  hidden_size: 128
  transformer:
    num_layers: 3
    num_heads: 4
    ff_dim: 256
    dropout: 0.15
    attention_dropout: 0.10
  lstm_gru:
    lstm_units: 128
    gru_units: 64
    dropout: 0.20
  tcn:
    filters: 96
    kernel_size: 3
    dilation_rates: [1, 2, 4, 8, 16]
    dropout: 0.15
  tabular_branch:
    dense_units: [128, 64]
    dropout: 0.15
  fusion:
    method: cross_timeframe_attention
    timeframe_embedding_dim: 16
    asset_embedding_dim: 8
  heads:
    direction:
      classes:
        - down
        - sideways
        - up
      dense_units: [128, 64]
      dropout: 0.15
      loss_weight: 1.0
    regime:
      classes:
        - trending_up
        - trending_down
        - ranging_low_vol
        - ranging_high_vol
        - breakout
        - capitulation
      dense_units: [96, 48]
      dropout: 0.15
      loss_weight: 0.5
    cycle:
      classes:
        - accumulation
        - bull
        - distribution
        - bear
      dense_units: [64, 32]
      dropout: 0.10
      loss_weight: 0.25
    trade_quality:
      type: binary
      dense_units: [64, 32]
      dropout: 0.15
      loss_weight: 0.75
  optimizer:
    name: AdamW
    learning_rate: 0.0003
    weight_decay: 0.0001
    clipnorm: 1.0
  lr_schedule:
    name: cosine_decay_with_warmup
    warmup_epochs: 3
    min_lr: 0.00001
  batch_size:
    default: 256
    single_t4: 512
    dual_t4: 1024
    cpu_smoke: 32
  epochs: 60
  early_stopping:
    monitor: val_direction_macro_f1
    mode: max
    patience: 8
    min_delta: 0.002
    restore_best_weights: true
  checkpoint:
    save_best_only: true
    monitor: val_direction_macro_f1
    mode: max

pretraining:
  enabled: true
  method: masked_window_modeling
  epochs: 10
  learning_rate: 0.0005
  mask_ratio: 0.15
  save_encoder: true

plantguard_style_training:
  enabled: true
  phase1:
    name: supervised_head_warmup
    freeze_pretrained_encoder_if_available: true
    epochs: 10
    learning_rate: 0.001
    monitor: val_direction_macro_f1
    patience: 5
    reduce_lr_factor: 0.5
    reduce_lr_patience: 3
  phase2:
    name: fine_tune_last_encoder_blocks
    unfreeze_last_n_blocks: 1
    epochs: 25
    learning_rate: 0.00003
    monitor: val_direction_macro_f1
    patience: 8
    reduce_lr_factor: 0.3
    reduce_lr_patience: 4
  artifacts:
    save_training_curves: true
    save_confusion_matrices: true
    save_classification_report: true
    save_prediction_demo: true
    save_reload_test: true

backtest:
  engine: custom_event_driven
  optional_vectorbt_comparison: true
  initial_equity: 10000
  fee_bps_per_side: 4
  slippage:
    model: atr_and_volume
    min_bps_per_side: 2
    max_bps_per_side: 15
    atr_fraction: 0.02
  futures_margin:
    enabled: false
    default_leverage: 1
    max_leverage_allowed: 3
    margin_mode: isolated
    maintenance_margin_rate: 0.005
    liquidation_buffer_pct_min: 5.0
    include_funding_fees: true
    funding_interval_hours: 8
    reject_trade_if_liquidation_buffer_below_pct: 5.0
  min_rr: 2.0
  risk_per_trade_pct: 1.0
  max_risk_per_trade_pct: 2.0
  max_daily_loss_pct: 4.0
  max_open_positions: 1
  confidence_thresholds:
    long: 0.58
    short: 0.58
    no_trade_below: 0.58
  exits:
    use_triple_barrier: true
    stop_atr_multiple: 1.5
    tp1_rr: 1.0
    tp2_rr: 2.0
    tp3_rr: 3.0
    partials:
      tp1_close_pct: 0.33
      tp2_close_pct: 0.33
      tp3_close_pct: 0.34
    move_sl_to_breakeven_after_tp1: true
    move_sl_to_tp1_after_tp2: true
    max_holding_bars:
      15m: 32
      1h: 48
      4h: 42
      1d: 30

retraining:
  schedule: weekly_manual_gpu
  daily_data_refresh: true
  min_new_bars_before_retrain:
    15m: 2000
    1h: 500
    4h: 150
    1d: 30
  trigger_if:
    validation_direction_macro_f1_drop_abs: 0.05
    psi_feature_drift_above: 0.25
    live_trade_expectancy_below_zero_over_trades: 30
    calibration_ece_above: 0.12
  drift_visualization:
    enabled: true
    top_n_features: 30
    save_static_charts: true
    save_html_dashboard: true
  shadow_ab_testing:
    enabled: true
    min_shadow_days: 14
    min_shadow_signals: 30
    compare_paper_trades: true
    candidate_traffic_pct: 0
    production_remains_default: true
  promotion:
    require_beats_current_production: true
    min_direction_macro_f1_improvement: 0.01
    min_backtest_profit_factor: 1.10
    max_drawdown_not_worse_by_pct: 10
    require_positive_expectancy: true

serving:
  enable_fastapi: true
  host: 0.0.0.0
  port: 8000
  prediction_refresh_minutes: 15
  alerting:
    enable_telegram: false
    enable_discord: false
    enable_email: false
    min_signal_confidence: 0.65
    min_trade_quality_probability: 0.60
    cooldown_minutes: 60
    include_scorecard: true
    include_risk_warning: true
    include_model_id: true
```

---

## FEATURE ENGINEERING

All features must be causal.

Implement in:

```text
src/features/
```

### 1. Structure features

File:

```text
src/features/structure.py
```

Features:

- Swing highs
- Swing lows
- Higher-high / higher-low flags
- Lower-high / lower-low flags
- Market-structure score
- Support/resistance clustering from prior swings only
- Range vs trend classifier
- Breakout flag
- Break-and-retest proxy
- Liquidity sweep proxy
- Distance to support
- Distance to resistance
- Higher-timeframe bias from weekly/daily/4H, broadcast causally
- Wyckoff spring proxy
- CHoCH proxy
- FVG proxy
- Order-block proxy

### 2. Indicator features

File:

```text
src/features/indicators.py
```

Features:

- EMA 9/21/50/120/200
- EMA120 cycle signal
- EMA stack score
- SMA 50/200
- Golden cross / death cross
- RSI 14
- RSI slope
- RSI overbought/oversold
- RSI divergence proxy
- MACD line/signal/histogram
- MACD crossover
- Bollinger Bands %B
- Bollinger bandwidth
- Bollinger mean-reversion signal
- ATR
- OBV
- VWAP
- Rolling realized volatility
- Rolling volume z-score
- Distance from ATH
- Distance from 52-week high/low
- Price vs 200-week MA
- Volume profile / POC approximation over rolling windows

### 3. Candlestick and chart patterns

File:

```text
src/features/patterns.py
```

Features:

- Doji
- Engulfing
- Hammer
- Shooting star
- Morning/evening star where available
- Pin bar proxy
- Inside bar
- Outside bar
- Double top / double bottom proxy
- W/M pattern proxy
- Channel break proxy

Use TA-Lib CDL functions if available. Otherwise implement simplified deterministic versions.

### 4. Derivatives / flow

File:

```text
src/features/flow.py
```

Features:

- Funding rate level
- Funding rate z-score
- Funding extreme flag
- Open interest
- OI change
- Price up + OI up
- Price down + OI up
- Price up + OI down
- Price down + OI down
- Long/short ratio
- Long/short ratio z-score
- Taker buy volume
- Taker sell volume
- Taker buy/sell ratio
- Taker delta proxy
- CVD proxy
- Basis / premium
- Funding/OI governor risk score
- Liquidation pressure proxy

### 5. On-chain / sentiment / macro

Files:

```text
src/features/onchain.py
src/features/sentiment.py
src/features/macro.py
```

Features:

- Active addresses where available
- Hash rate for BTC where available
- Miner revenue for BTC where available
- CVDD / Balanced Price only if source or calculation is verifiably implemented
- Gas / burn / staking proxies for ETH where available
- Fear & Greed level
- Fear & Greed delta
- BTC dominance
- ETH/BTC relative trend
- ETH/BTC rotation signal
- Equity correlation
- DXY/rate/inflation proxies where available
- FOMC/CPI event flags only if manually configured or available causally
- On-chain coverage score

### 6. Checklist scorecard

Create:

```text
src/features/scorecard.py
```

This must produce a transparent rule-derived scorecard separate from ML predictions.

It should cover:

BTC/ETH Futures Trade Checklist:

- Market structure
- Trend direction
- Support/resistance
- Range/trend state
- Higher timeframe bias
- Entry timeframe confirmation
- Chart pattern flags
- EMA stack
- EMA120 cycle signal
- Volume/POC approximation
- RSI
- MACD
- Bollinger mean-reversion state
- Funding rate
- Open interest
- Funding/OI governor risk
- CVD proxy
- Taker imbalance
- Basis
- Long/short ratio
- BTC dominance for ETH
- Fear & Greed
- Macro/equity correlation
- Entry trigger
- Stop-loss suggestion
- TP1/TP2/TP3 suggestion
- Minimum R:R
- Risk percentage warning
- Trade journal fields

BTC/ETH Buy Checklist:

- Current price
- ATH distance
- 52-week high/low range
- Market cap/dominance where available
- 24h volume where available
- 50/200 MA
- EMA120 cycle state
- RSI
- MACD
- Trend direction
- On-chain metrics where available
- Macro/sentiment
- BTC-specific halving/cycle features
- ETH-specific gas/staking/burn proxies where available
- DCA vs lump-sum helper
- Risk and exit-plan fields

The scorecard must clearly mark unavailable data as:

```text
unavailable
```

not guessed.

---

## FEATURE SELECTION AND FEATURE QUALITY

Create:

```text
src/features/selection.py
```

Feature selection must run inside each training fold to avoid leakage.

Pipeline:

1. Drop features with too many missing values using `min_non_null_ratio`.
2. Drop constant and near-zero variance features.
3. Remove exact duplicates.
4. Remove one feature from each highly correlated pair above `max_pairwise_corr`.
5. Rank remaining features using:
   - Mutual information on training window only
   - Permutation importance from a lightweight baseline model on validation window only
   - Optional SHAP only if installed and computationally feasible
6. Keep `always_keep` features.
7. Keep top K features from config.
8. Save selected feature list per fold.

Important:

- Do not compute feature importance using the test set.
- Do not select features globally before time split.
- Report dropped features and reasons.

Output:

```text
artifacts/runs/<run_id>/selected_features_<fold>.json
reports/feature_selection_<run_id>.md
```

---

## LABELS

Implement:

```text
src/labels/labeling.py
```

### 1. Direction label

Use triple-barrier labeling.

For each bar:

- Upper barrier: `close + k * ATR`
- Lower barrier: `close - k * ATR`
- Vertical barrier: `N` future bars

Classes:

```text
up
down
sideways
```

Label by which barrier is hit first.

If neither upper nor lower is hit before the vertical barrier, label sideways.

Use high/low path within future candles, not only close-to-close.

Same-candle ambiguity:

If both upper and lower barriers are touched in the same candle, use a deterministic conservative rule:

- If candle opens closer to upper and closes lower, assume upper first only if path can be reasonably inferred.
- Otherwise assign `ambiguous` and exclude from training by default.
- Log ambiguous count.

Important:

Labels are allowed to look forward because they are targets, but features must not.

Validation must purge overlapping label horizons.

### 2. Market regime label

Rule-based label from:

- EMA slope
- ADX if available
- Bollinger bandwidth
- ATR percentile
- Realized volatility
- Market structure

Classes:

```text
trending_up
trending_down
ranging_low_vol
ranging_high_vol
breakout
capitulation
```

Document the exact rules.

Optional:

Add unsupervised GMM/HMM regime labels later, but do not make it required in Phase 1.

### 3. Four-year cycle phase label

Use BTC halving dates:

```text
2012-11-28
2016-07-09
2020-05-11
2024-04-19
projected_2028_halving: approximate, configurable, not fixed truth
```

Do not label cycle only by calendar.

Use combined rules:

- Months since last BTC halving
- Drawdown from ATH
- Position relative to 200-week MA
- Realized volatility regime
- BTC dominance trend
- Optional on-chain value metrics where available

Classes:

```text
accumulation
bull
distribution
bear
```

Apply ETH cycle context using BTC cycle anchor plus ETH’s own drawdown and ETH/BTC relative trend.

Document the rule table clearly.

### 4. Trade-quality label

Add a fourth model head:

```text
trade_quality
```

Binary label:

```text
good_trade
bad_or_no_trade
```

Definition:

A `good_trade` is a bar where the model’s direction target would have reached at least 2R before hitting stop, after estimated fees and slippage.

This label helps the model learn “should I trade?” instead of only “up/down/sideways.”

### 5. Class balance

For each label head, log:

- Class counts
- Class percentages
- Missing labels
- Ambiguous labels
- Time range
- Asset
- Timeframe

Use class weights by default:

```text
class_weight = 1 / sqrt(class_frequency)
```

Normalize class weights so mean weight is 1.0.

Do not use SMOTE for sequential candle data by default. SMOTE can create unrealistic synthetic time-series windows. Instead use:

- Class weights
- Focal loss option
- Threshold tuning
- Stratified reporting by regime
- Optional balanced batch sampler if implemented causally

---

## MULTI-TIMEFRAME FUSION

Do not leave this vague.

Default architecture:

```text
Multi-Timeframe Transformer with Cross-Timeframe Attention
```

For each prediction timeframe, build aligned inputs:

Example for 1h prediction:

```text
fast branch: 15m sequence ending at the same UTC time
main branch: 1h sequence
slow branch: 4h sequence
daily branch: 1d sequence
weekly/monthly tabular context
```

For 4h prediction:

```text
fast branch: 1h sequence
main branch: 4h sequence
slow branch: 1d sequence
weekly/monthly tabular context
```

For 15m prediction:

```text
fast branch: 5m sequence
main branch: 15m sequence
slow branch: 1h sequence
daily/weekly tabular context
```

For 1d prediction:

```text
fast branch: 4h sequence
main branch: 1d sequence
slow branch: 1w sequence
monthly tabular context
```

Implementation:

1. Each timeframe branch gets its own small encoder.
2. Add timeframe embedding to each branch.
3. Pool each branch using attention pooling.
4. Fuse branch embeddings using cross-timeframe attention.
5. Concatenate fused sequence representation with tabular slow features.
6. Feed shared trunk.
7. Feed task-specific heads.

Fallback:

If multi-timeframe fusion is too heavy for CPU smoke tests, implement a config switch:

```yaml
use_multi_timeframe_fusion: false
```

Then train per-timeframe single-input models.

---

## MODEL ARCHITECTURE

Create:

```text
src/models/multitask_model.py
```

### Default model: MTF Transformer Attention

Inputs:

```text
fast_sequence_input:  [batch, fast_seq_len, fast_feature_count]
main_sequence_input:  [batch, main_seq_len, main_feature_count]
slow_sequence_input:  [batch, slow_seq_len, slow_feature_count]
context_input:        [batch, context_feature_count]
asset_id_input:       [batch]
timeframe_id_input:   [batch]
```

Per-timeframe encoder:

```text
LayerNorm
Dense(hidden_size=128)
PositionalEncoding
TransformerBlock x 3:
  MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.10)
  Add + LayerNorm
  FeedForward(Dense 256 -> GELU -> Dropout 0.15 -> Dense 128)
  Add + LayerNorm
AttentionPooling
```

Embeddings:

```text
asset_embedding_dim = 8
timeframe_embedding_dim = 16
```

Fusion:

```text
Stack [fast_embedding, main_embedding, slow_embedding]
Cross-timeframe MultiHeadAttention(num_heads=4, key_dim=32)
Flatten or attention pool
Concatenate context branch + asset embedding + timeframe embedding
```

Context branch:

```text
Dense 128 + GELU + Dropout 0.15
Dense 64 + GELU + Dropout 0.15
```

Shared trunk:

```text
Dense 256 + GELU + BatchNorm + Dropout 0.20
Dense 128 + GELU + BatchNorm + Dropout 0.15
```

Heads:

Direction head:

```text
Dense 128 + GELU + Dropout 0.15
Dense 64 + GELU + Dropout 0.10
Dense 3 + softmax
```

Regime head:

```text
Dense 96 + GELU + Dropout 0.15
Dense 48 + GELU + Dropout 0.10
Dense K + softmax
```

Cycle head:

```text
Dense 64 + GELU + Dropout 0.10
Dense 32 + GELU
Dense 4 + softmax
```

Trade-quality head:

```text
Dense 64 + GELU + Dropout 0.15
Dense 32 + GELU
Dense 1 + sigmoid
```

Losses:

```text
direction: sparse_categorical_crossentropy or focal categorical CE
regime: sparse_categorical_crossentropy
cycle: sparse_categorical_crossentropy
trade_quality: binary_crossentropy
```

Loss weights from config:

```text
direction: 1.0
regime: 0.5
cycle: 0.25
trade_quality: 0.75
```

Optimizer:

```text
AdamW
learning_rate = 3e-4
weight_decay = 1e-4
clipnorm = 1.0
```

Learning-rate schedule:

```text
3-epoch warmup
cosine decay to 1e-5
```

Regularization:

```text
dropout 0.10–0.20
weight decay 1e-4
early stopping
gradient clipping
```

Metrics:

During training:

- Direction accuracy
- Direction macro F1 callback
- Regime accuracy
- Cycle accuracy
- Trade-quality AUC
- Loss per head

During evaluation:

- Full per-class metrics
- Calibration
- Backtest metrics

---

## HYPERPARAMETER STRATEGY

Do not do uncontrolled huge search.

Create:

```text
src/models/hparam_search.py
```

Implement a small controlled search over:

```yaml
learning_rate: [0.0001, 0.0003, 0.0007]
dropout: [0.10, 0.15, 0.20]
hidden_size: [96, 128, 192]
num_transformer_layers: [2, 3]
num_heads: [4]
sequence_length_multiplier: [0.75, 1.0]
```

Use only one representative fold for hparam search to save GPU quota.

Select by:

```text
primary: validation direction macro F1
secondary: validation trade-quality AUC
third: backtest profit factor
risk constraint: max drawdown not worse than baseline by configured limit
```

Save:

```text
reports/hparam_search_<run_id>.md
artifacts/runs/<run_id>/best_hparams.yaml
```

Default: skip search unless explicitly enabled.

---

## DATASET BUILDING

Create:

```text
src/datasets/build_dataset.py
```

Responsibilities:

1. Load normalized data.
2. Align timeframes.
3. Join slow features causally.
4. Build multi-timeframe sequence windows.
5. Build context vectors.
6. Split into train/validation/test using purged walk-forward.
7. Fit imputers/scalers only on training windows.
8. Run feature selection inside the training fold.
9. Save scalers.
10. Save feature schema.
11. Save label schema.
12. Save dataset manifest.

No scaler may be fit on validation/test data.

No feature selection may use test data.


---

## PLANTGUARD-STYLE TRAINING WORKFLOW, ADAPTED FOR BTC/ETH TIME SERIES

The training experience should feel like a polished ML project notebook/script similar to the referenced Plant Guard style:

```text
GPU check
single configuration block
dataset validation
class distribution chart
sample data visualization
model build
phase 1 training
phase 2 fine-tuning
training curves
confusion matrix
classification report
prediction demo
save model
save class/label mappings
reload test
deployment/inference smoke test
```

But because this is not image classification, adapt it correctly for financial time series.

Create both:

```text
src/models/train_like_plantguard.py
notebooks/kaggle_train_plantguard_style.ipynb
notebooks/colab_train_plantguard_style.ipynb
```

These must import the real project modules from `src/` and must not duplicate core logic.

### Required notebook/script cells or sections

#### Section 1 — Imports and GPU check

Show:

- Python version
- TensorFlow version
- Keras version
- GPU list
- Whether Kaggle dual T4 is detected
- Whether `tf.distribute.MirroredStrategy()` is active
- Mixed precision status
- Random seed

If GPU exists, enable memory growth where supported.

#### Section 2 — Configuration

Use a single visible config block.

Show:

- Asset list
- Timeframes
- Sequence lengths
- Batch size
- Train/validation/test date ranges
- Model architecture
- Phase 1 epochs
- Phase 2 epochs
- Learning rates
- Artifact paths
- Dataset manifest path
- Production/candidate model ID

The notebook must allow quick overrides at the top without editing internal code.

#### Section 3 — Dataset validation

Before training, print:

- Dataset path
- Dataset version/hash
- Assets found
- Timeframes found
- First timestamp per asset/timeframe
- Last timestamp per asset/timeframe
- Row count
- Missing candle estimate
- Duplicate count
- Label count
- Feature count before selection
- Feature count after selection
- Data coverage score
- On-chain coverage score

If anything critical is missing, fail clearly.

#### Section 4 — Label/class distribution chart

Generate and save:

```text
reports/training_plots/label_distribution_<run_id>.png
reports/training_plots/regime_distribution_<run_id>.png
reports/training_plots/cycle_distribution_<run_id>.png
reports/training_plots/trade_quality_distribution_<run_id>.png
```

Also print class percentages.

This mirrors the class-distribution chart idea from image classification, but for:

- Direction labels
- Regime labels
- Cycle labels
- Trade-quality labels

#### Section 5 — Sample market windows visualization

Generate sample charts before training:

```text
reports/training_plots/sample_window_<asset>_<timeframe>_<index>.png
```

Each sample chart should show:

- OHLC close line or candlestick approximation
- EMA 9/21/50/200
- RSI panel
- MACD panel
- Bollinger Bands if available
- Funding/OI subplot if available
- Label marker showing up/down/sideways
- Regime label
- Cycle label

This is the time-series equivalent of displaying sample images before model training.

#### Section 6 — Build model summary

Print model summary.

Save architecture diagram if possible:

```text
reports/training_plots/model_architecture_<run_id>.png
```

If graph plotting dependencies are unavailable, save text summary:

```text
reports/model_summary_<run_id>.txt
```

#### Section 7 — Phase 0 optional self-supervised pretraining

Because financial time series has no ImageNet-like pretrained backbone by default, implement optional self-supervised pretraining.

This is the correct equivalent of using MobileNetV2 pretrained weights in the Plant Guard project.

Config:

```yaml
pretraining:
  enabled: true
  method: masked_window_modeling
  epochs: 10
  learning_rate: 0.0005
  mask_ratio: 0.15
  save_encoder: true
```

Supported methods:

```text
masked_window_modeling
next_window_contrastive
autoencoder_reconstruction
```

Default method:

```text
masked_window_modeling
```

Goal:

Train the multi-timeframe encoder to understand BTC/ETH market windows before supervised labels.

Save:

```text
artifacts/pretrained_encoders/<encoder_id>/
  encoder.keras
  pretraining_config.yaml
  pretraining_loss_curve.png
  dataset_manifest.json
```

If pretraining is disabled, continue directly to Phase 1 supervised training.

#### Section 8 — Phase 1 supervised head training

This mirrors “train the classification head while base is frozen.”

For BTC/ETH:

1. Load pretrained encoder if available.
2. Freeze encoder branches.
3. Train only:
   - Fusion layer if configured
   - Shared trunk
   - Direction head
   - Regime head
   - Cycle head
   - Trade-quality head

If no pretrained encoder exists, do not freeze randomly initialized encoders. Instead run Phase 1 as a lower-LR warmup of the full model.

Config:

```yaml
plantguard_style_training:
  phase1:
    name: supervised_head_warmup
    freeze_pretrained_encoder_if_available: true
    epochs: 10
    learning_rate: 0.001
    monitor: val_direction_macro_f1
    patience: 5
    reduce_lr_factor: 0.5
    reduce_lr_patience: 3
```

Callbacks:

- EarlyStopping
- ModelCheckpoint
- ReduceLROnPlateau
- TensorBoard
- CSVLogger

Save:

```text
checkpoints/<run_id>/phase1_best.keras
reports/training_logs/phase1_history_<run_id>.csv
```

#### Section 9 — Phase 2 fine-tuning

This mirrors “unfreeze last layers and fine-tune with lower learning rate.”

For BTC/ETH:

1. Unfreeze the last N transformer blocks / TCN blocks / recurrent layers.
2. Keep early encoder blocks frozen if pretrained.
3. Use a lower LR.
4. Continue training with strict early stopping.

Config:

```yaml
plantguard_style_training:
  phase2:
    name: fine_tune_last_encoder_blocks
    unfreeze_last_n_blocks: 1
    epochs: 25
    learning_rate: 0.00003
    monitor: val_direction_macro_f1
    patience: 8
    reduce_lr_factor: 0.3
    reduce_lr_patience: 4
```

Save:

```text
checkpoints/<run_id>/phase2_best.keras
reports/training_logs/phase2_history_<run_id>.csv
```

#### Section 10 — Training curves

Generate and save:

```text
reports/training_plots/training_curves_<run_id>.png
```

Must include:

- Total loss
- Direction loss
- Regime loss
- Cycle loss
- Trade-quality loss
- Direction macro F1
- Trade-quality AUC
- Validation metrics
- Vertical line showing Phase 2 fine-tuning start

#### Section 11 — Confusion matrices

Generate normalized and raw confusion matrices for:

- Direction head
- Regime head
- Cycle head

Save:

```text
reports/training_plots/confusion_direction_<run_id>.png
reports/training_plots/confusion_regime_<run_id>.png
reports/training_plots/confusion_cycle_<run_id>.png
```

Trade-quality head should get:

```text
reports/training_plots/trade_quality_roc_<run_id>.png
reports/training_plots/trade_quality_pr_curve_<run_id>.png
```

#### Section 12 — Classification reports

Save text and JSON reports:

```text
reports/classification_report_<run_id>.txt
reports/classification_report_<run_id>.json
```

Include:

- Precision
- Recall
- F1
- Support
- Macro average
- Weighted average
- Per-asset breakdown
- Per-timeframe breakdown
- Per-regime breakdown

Also generate F1 bar charts:

```text
reports/training_plots/f1_direction_<run_id>.png
reports/training_plots/f1_regime_<run_id>.png
reports/training_plots/f1_cycle_<run_id>.png
```

#### Section 13 — Prediction demo

Run a latest-market prediction demo like the Plant Guard single-image prediction demo.

For BTC/ETH:

1. Select latest completed bar for BTCUSDT and ETHUSDT.
2. Build latest feature window.
3. Run model prediction.
4. Print top probabilities.
5. Generate a visual prediction card.

Save:

```text
reports/prediction_demo_<run_id>.json
reports/training_plots/prediction_demo_<asset>_<timeframe>_<run_id>.png
```

Prediction demo should show:

- Asset
- Timeframe
- Direction probabilities
- Regime
- Cycle phase
- Trade-quality probability
- Signal decision
- Scorecard summary
- Risk warning

#### Section 14 — Save final model and metadata

Save:

```text
artifacts/runs/<run_id>/
  model.keras
  model_best_phase1.keras
  model_best_phase2.keras
  scaler.joblib
  imputer.joblib
  selected_features.json
  label_mapping.json
  class_indices.json
  threshold_config.json
  calibration_config.json
  config.yaml
  model_summary.txt
  training_history.csv
  dataset_manifest.json
  git_sha.txt
```

`class_indices.json` must map all output heads:

```json
{
  "direction": {
    "0": "down",
    "1": "sideways",
    "2": "up"
  },
  "regime": {
    "0": "trending_up",
    "1": "trending_down"
  },
  "cycle": {
    "0": "accumulation",
    "1": "bull",
    "2": "distribution",
    "3": "bear"
  },
  "trade_quality": {
    "0": "bad_or_no_trade",
    "1": "good_trade"
  }
}
```

#### Section 15 — Reload test

Immediately reload the saved model and run predictions on a small validation batch.

Print:

- Reload successful
- Prediction shapes per head
- Sample probabilities
- Class mapping loaded
- Threshold config loaded

Fail if reload does not work.

#### Section 16 — Deployment smoke test

Run:

```bash
python -m src.models.predict --model-id <run_id> --sample true
python -m src.serve.api --smoke-test
```

Confirm:

- Latest prediction JSON exists.
- FastAPI health endpoint would load model.
- Model registry can register this run as candidate.

### PlantGuard-style command

Add command:

```bash
python -m src.models.train_like_plantguard \
  --config configs/config.yaml \
  --assets BTCUSDT ETHUSDT \
  --timeframes 1h 4h \
  --plantguard-style true
```

It should produce the same polished set of artifacts as the notebook.

### Important adaptation note

Do not use image augmentation, ImageDataGenerator, MobileNetV2, or image-specific preprocessing.

The equivalent concepts are:

```text
Image class distribution      -> label distribution
Sample images                 -> sample market windows
Image augmentation            -> time-series-safe augmentation only if explicitly enabled
MobileNetV2 pretrained base   -> optional self-supervised pretrained time-series encoder
Frozen base phase             -> freeze pretrained encoder and train supervised heads
Fine-tuning phase             -> unfreeze last encoder blocks with lower LR
Confusion matrix              -> direction/regime/cycle confusion matrices
Single-image prediction demo  -> latest-market prediction demo
Flask app inference           -> FastAPI prediction endpoint
```

Time-series-safe augmentation is disabled by default. If enabled later, only allow realistic augmentations such as tiny noise on normalized features, time masking, or feature masking during pretraining. Never shuffle time order.


---

## MODEL TRAINING

Create:

```text
src/models/train.py
```

Requirements:

- Walk-forward training loop
- Early stopping
- Checkpoints
- Resume support
- TensorBoard logs
- Mixed precision if GPU supports it
- Multi-GPU support through `tf.distribute.MirroredStrategy()` when Kaggle dual T4 is available
- Save model, scaler, config, feature schema, selected features, label schema, git SHA, and report metadata
- Save out-of-fold predictions for threshold tuning and calibration

Artifacts:

```text
artifacts/runs/<run_id>/
  model.keras
  scaler.joblib
  imputer.joblib
  selected_features.json
  config.yaml
  feature_schema.json
  label_schema.json
  metrics.json
  git_sha.txt
  dataset_manifest.json
  threshold_config.json
  calibration_config.json
```

---

## CLASS IMBALANCE AND THRESHOLD TUNING

Create:

```text
src/models/thresholds.py
```

Responsibilities:

1. Compute class weights from training data only.
2. Train with class weights by default.
3. Tune decision thresholds on validation predictions only.
4. Optimize macro F1 while enforcing minimum precision for actionable up/down classes.
5. Allow “no_trade” when confidence is below threshold.

Example logic:

```text
if max(direction_probs) < no_trade_threshold:
    signal = no_trade
elif up_prob >= long_threshold and trade_quality_prob >= quality_threshold:
    signal = long_bias
elif down_prob >= short_threshold and trade_quality_prob >= quality_threshold:
    signal = short_bias
else:
    signal = no_trade
```

Save thresholds per asset/timeframe.

Do not tune thresholds on test data.


---

## OPTIONAL FUTURES LEVERAGE, MARGIN, AND LIQUIDATION MODEL

Default evaluation must use 1x leverage for honesty.

However, if the user later enables futures leverage, the backtester must simulate margin and liquidation risk conservatively.

Create:

```text
src/backtest/margin.py
src/backtest/liquidation.py
src/backtest/funding.py
```

### Margin mode

Support:

```text
isolated
cross optional later
```

Default:

```text
isolated
```

Do not implement cross-margin unless it is clearly tested.

### Leverage rules

Config defaults:

```text
default_leverage = 1
max_leverage_allowed = 3
```

Reject any leverage above config max.

Never suggest high leverage.

### Liquidation approximation

For isolated USDT-margined futures, approximate liquidation risk conservatively.

For long:

```text
approx_liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin_rate + fee_buffer)
```

For short:

```text
approx_liquidation_price = entry_price * (1 + (1 / leverage) - maintenance_margin_rate - fee_buffer)
```

Use exchange-specific formula only if verified and implemented later.

Add safety buffer:

```text
liquidation_buffer_pct = distance(entry_price, approx_liquidation_price) / entry_price * 100
```

Reject trade if:

```text
liquidation_buffer_pct < reject_trade_if_liquidation_buffer_below_pct
```

### Funding fees

If holding futures positions across funding intervals, estimate funding fees:

```text
funding_fee = notional_position_value * funding_rate
```

Apply at each funding interval crossed by the position.

If funding data is unavailable, mark funding fee estimate as unavailable and optionally reject leveraged backtests unless configured otherwise.

### Margin call / liquidation event

A position is liquidated in simulation if candle high/low touches the approximate liquidation price before stop/TP, using conservative intrabar assumptions.

Liquidation event should record:

```text
timestamp
asset
timeframe
side
entry
liquidation_price
equity_before
equity_after
loss
reason
```

### Reporting

Backtest report must include:

- Leverage used
- Margin mode
- Liquidation count
- Funding fees paid/received
- Liquidation buffer statistics
- Trades rejected due to liquidation risk
- Max notional exposure
- Max margin used


---

## BACKTESTING

Create:

```text
src/backtest/
  engine.py
  broker.py
  metrics.py
  strategies.py
  costs.py
```

Backtesting must be a real event-driven simulation, not just classification metrics.

### Entry logic

For each asset/timeframe:

Long entry:

```text
up_prob >= long_threshold
trade_quality_prob >= quality_threshold
regime not in [capitulation if long disabled, ranging_low_vol if breakout strategy disabled]
scorecard risk not high
```

Short entry:

```text
down_prob >= short_threshold
trade_quality_prob >= quality_threshold
funding/OI governor does not signal squeeze risk
scorecard risk not high
```

No trade:

```text
confidence below threshold
high-risk governor state
data coverage too weak
model stale
```

### Position sizing

Risk-based sizing:

```text
risk_amount = equity * risk_per_trade_pct / 100
stop_distance = abs(entry_price - stop_price)
position_size = risk_amount / stop_distance
```

Caps:

- Max 1 open position per asset/timeframe by default.
- Max daily loss 4%.
- Max risk per trade 2%.
- No pyramiding unless configured later.
- Optional leverage cap, default 1x for evaluation honesty unless user configures futures leverage.

### Stops and targets

Use ATR-based stops by default.

```text
stop = entry - ATR * stop_atr_multiple for long
stop = entry + ATR * stop_atr_multiple for short
```

Targets:

```text
TP1 = 1R
TP2 = 2R
TP3 = 3R
```

Partials:

```text
TP1 close 33%
TP2 close 33%
TP3 close 34%
```

Stop management:

- After TP1, move SL to breakeven.
- After TP2, move SL to TP1.
- After TP3, position closed.

### Execution realism

Include:

- Fee per side from config.
- Slippage per side using ATR/volume-aware model.
- Intrabar high/low logic.
- Conservative same-candle ambiguity handling.
- No impossible fills.
- Max holding period from config.

### Metrics

Report:

- Total return
- CAGR if period long enough
- Max drawdown
- Profit factor
- Expectancy
- Sharpe
- Sortino
- Omega if implemented
- Win rate
- Average R
- Median R
- WR × R score
- Number of trades
- Average trade duration
- Exposure time
- Long vs short performance
- Asset-specific performance
- Timeframe-specific performance
- Regime-specific performance
- Fee drag
- Slippage drag
- Worst 10 trades
- Monthly returns

Backtest must compare:

- Model strategy
- Buy-and-hold
- Majority-class baseline
- EMA trend baseline
- RSI/MACD baseline
- Bollinger mean-reversion baseline where applicable
- No-trade baseline

Output:

```text
reports/backtest_<run_id>.md
reports/backtest_<run_id>.json
reports/trades_<run_id>.csv
```

---

## EVALUATION

Create:

```text
src/models/evaluate.py
```

Evaluation must use purged + embargoed walk-forward.

Report:

- Per-head accuracy
- Per-class precision
- Per-class recall
- Per-class F1
- Macro F1
- Weighted F1
- Confusion matrices
- Direction calibration / reliability table
- Threshold tuning result
- Baseline comparisons
- Fee/slippage-aware backtest result
- Model staleness/drift report
- Feature importance report
- Data coverage report
- Failure modes

Output:

```text
reports/eval_<run_id>.md
reports/eval_<run_id>.json
```

The report must honestly state where the model fails.


---

## DRIFT VISUALIZATION DASHBOARD

Create:

```text
src/models/drift_viz.py
src/serve/drift_dashboard.py
```

The system already detects PSI and drift. Add visual outputs so drift is easy to inspect.

### Static charts

Generate charts in:

```text
reports/drift/
```

Required charts:

```text
psi_top_features_<date>.png
feature_distribution_shift_<feature>_<date>.png
prediction_distribution_drift_<date>.png
regime_distribution_drift_<date>.png
calibration_drift_<date>.png
live_expectancy_curve_<date>.png
```

Chart requirements:

1. Use only training-reference distribution vs latest-live distribution.
2. Show top drifting features by PSI.
3. Show prediction probability drift for up/down/sideways.
4. Show regime prediction distribution drift.
5. Show calibration drift when realized labels become available.
6. Show rolling live/paper-trade expectancy.
7. Save a machine-readable JSON beside every chart.

### HTML dashboard

Generate:

```text
reports/drift_dashboard_<date>.html
```

The HTML dashboard must include:

- Model ID
- Dataset version
- Feature schema version
- Top PSI features
- Drift severity table
- Static chart links
- Retrain recommendation
- Staleness status
- Candidate vs production drift comparison if shadow mode is active

### Drift severity

Use:

```text
PSI < 0.10        = stable
0.10–0.25         = moderate drift
> 0.25            = significant drift
```

Significant drift should trigger a retrain recommendation, not automatic retraining.


---

## MODEL VERSIONING, REGISTRY, PROMOTION, AND ROLLBACK

Create:

```text
src/models/registry.py
src/models/promote.py
src/models/rollback.py
```

Create registry file:

```text
metadata/model_registry.json
```

Each model record:

```json
{
  "model_id": "2026-05-26_120000_btc_eth_mtf_transformer",
  "created_at_utc": "2026-05-26T12:00:00Z",
  "status": "candidate",
  "artifact_path": "artifacts/runs/<run_id>",
  "dataset_manifest_hash": "...",
  "feature_schema_hash": "...",
  "label_config_hash": "...",
  "git_sha": "...",
  "metrics": {
    "direction_macro_f1": 0.0,
    "trade_quality_auc": 0.0,
    "backtest_profit_factor": 0.0,
    "max_drawdown_pct": 0.0,
    "expectancy_r": 0.0
  },
  "promotion_decision": "not_evaluated",
  "notes": ""
}
```

Statuses:

```text
candidate
production
archived
rejected
rolled_back
```

Promotion rule:

A candidate model can become production only if:

1. It beats the current production model on validation macro F1 by at least config threshold.
2. It has positive expectancy after fees/slippage.
3. Its profit factor is above configured minimum.
4. Its max drawdown is not materially worse than current production.
5. Calibration is acceptable.
6. Data coverage is not worse than minimum requirement.

Rollback:

```bash
python -m src.models.rollback --model-id <previous_model_id>
```

Rollback must update registry and serving pointer:

```text
artifacts/production/current_model.json
```

Never overwrite old artifacts.


---

## SHADOW A/B TESTING

Create:

```text
src/models/shadow.py
src/models/ab_compare.py
```

The model registry can promote models, but candidate models must be able to run in shadow mode before full promotion.

### Shadow mode

Shadow mode means:

1. Production model continues to produce the official signal.
2. Candidate model receives the same latest features.
3. Candidate predictions are logged but not used for official alerts or trading.
4. Candidate paper trades are simulated using the same backtest/live paper-trade rules.
5. Production and candidate are compared on identical timestamps.

### Shadow logs

Save:

```text
reports/shadow/
  shadow_predictions_<candidate_model_id>.jsonl
  shadow_paper_trades_<candidate_model_id>.csv
  shadow_compare_<candidate_model_id>.md
  shadow_compare_<candidate_model_id>.json
```

Each shadow prediction row must include:

```json
{
  "timestamp_utc": "...",
  "asset": "BTCUSDT",
  "timeframe": "1h",
  "production_model_id": "...",
  "candidate_model_id": "...",
  "production_signal": "no_trade",
  "candidate_signal": "long_bias",
  "production_probs": {"down": 0.0, "sideways": 0.0, "up": 0.0},
  "candidate_probs": {"down": 0.0, "sideways": 0.0, "up": 0.0},
  "production_trade_quality": 0.0,
  "candidate_trade_quality": 0.0,
  "scorecard_snapshot_hash": "...",
  "data_coverage_score": 0.0
}
```

### A/B comparison metrics

Compare:

- Signal agreement rate
- Candidate-only signal count
- Production-only signal count
- Direction confidence distribution
- Trade-quality distribution
- Paper-trade expectancy
- Paper-trade profit factor
- Paper-trade max drawdown
- Calibration when labels become available
- Regime-specific performance
- Data coverage differences
- Alert frequency difference

### Promotion after shadow mode

A candidate can be promoted after shadow mode only if:

1. Minimum shadow days completed.
2. Minimum shadow signals completed.
3. Candidate does not increase alert/trade frequency dangerously.
4. Candidate paper-trade expectancy is positive.
5. Candidate max drawdown is acceptable.
6. Candidate performs better than or equal to production on configured promotion criteria.
7. No critical drift/data-coverage issue is detected.

Promotion must still require an explicit command:

```bash
python -m src.models.promote --model-id <candidate_model_id>
```

Do not auto-promote silently.


---

## RETRAINING, DRIFT, AND STALENESS

Create:

```text
src/models/drift.py
src/models/retrain_check.py
```

Daily GitHub Actions must not train, but it must produce retrain recommendation.

Retrain triggers:

1. Enough new bars accumulated:
   - 15m: 2000 new bars
   - 1h: 500 new bars
   - 4h: 150 new bars
   - 1d: 30 new bars

2. Feature drift:
   - Population Stability Index above 0.25 for important features
   - Large volatility regime shift
   - Funding/OI distribution shift

3. Performance drift:
   - Live paper-trade expectancy negative over last 30 trades
   - Direction macro F1 proxy drops by configured threshold where labels are available
   - Calibration ECE above configured threshold

4. Time-based:
   - Weekly retrain recommendation if GPU is available
   - Monthly full retrain recommendation

Output:

```text
reports/retrain_check_<date>.md
metadata/retrain_status.json
```

The report should say:

```text
retrain_recommended: true/false
reason: ...
```

Kaggle/Colab training remains manual or notebook-driven unless the user explicitly runs it.

---

## PREDICTION

Create:

```text
src/models/predict.py
```

It must:

1. Load production model artifact by default.
2. Load latest processed data.
3. Build latest features causally.
4. Apply selected features, scaler, thresholds, and calibration.
5. Output JSON for each asset/timeframe.

Example:

```json
{
  "timestamp_utc": "2026-05-26T00:00:00Z",
  "model_id": "production_model_id",
  "asset": "BTCUSDT",
  "timeframe": "1h",
  "model_outputs": {
    "direction": {
      "down": 0.21,
      "sideways": 0.28,
      "up": 0.51
    },
    "regime": {
      "predicted": "trending_up",
      "confidence": 0.63
    },
    "cycle": {
      "predicted": "bull",
      "confidence": 0.58
    },
    "trade_quality": {
      "probability": 0.62
    }
  },
  "signal": {
    "action": "no_trade",
    "reason": "confidence below configured threshold",
    "long_threshold": 0.58,
    "short_threshold": 0.58,
    "quality_threshold": 0.60
  },
  "scorecard": {
    "trend_direction": "up",
    "ema_stack": "bullish",
    "ema120_cycle": "above",
    "rsi": 61.2,
    "macd": "bullish_histogram_rising",
    "bollinger_state": "neutral",
    "funding_rate": "slightly_positive",
    "open_interest": "rising",
    "funding_oi_governor": "normal",
    "taker_delta_proxy": "positive",
    "fear_greed": "greed",
    "btc_dominance": "available_or_unavailable",
    "onchain_coverage_score": 0.0,
    "support": 0,
    "resistance": 0
  },
  "risk_warning": "Decision-support only. Not financial advice. Validate manually before trading."
}
```

Do not output a hard “guaranteed buy/sell.”

Use wording like:

```text
long_bias
short_bias
no_trade
range_wait
high_risk
```

---

## SERVING AND ALERTS

Create:

```text
src/serve/
  api.py
  scheduler.py
  alerts.py
```

### FastAPI

Endpoints:

```text
GET /health
GET /model/current
GET /predict/latest
GET /predict/{asset}/{timeframe}
GET /scorecard/{asset}/{timeframe}
GET /registry
POST /predict/refresh
```

`/predict/latest` returns latest prediction JSON.

`/model/current` returns production model metadata.

### Local scheduler

Optional local loop:

```bash
python -m src.serve.scheduler --refresh-minutes 15
```

It should:

1. Pull latest data delta if configured.
2. Build latest features.
3. Run prediction.
4. Save JSON to `reports/latest_predictions.json`.
5. Trigger alerts if enabled.

### Alerts

Create disabled-by-default alert adapters:

- Telegram
- Discord webhook
- Email SMTP

Config:

```yaml
alerting:
  enable_telegram: false
  enable_discord: false
  enable_email: false
```

Alert only when:

- Signal is long_bias or short_bias.
- Direction confidence is above configured minimum.
- Trade-quality probability is above configured threshold.
- Cooldown has passed.
- Model is not stale.
- Data coverage is acceptable.
- Shadow candidate alerts are disabled unless explicitly requested.

Never spam alerts.

### Alert message content

All alert adapters must use the same canonical alert payload.

Create:

```text
src/serve/alert_templates.py
```

Canonical JSON payload:

```json
{
  "alert_type": "model_signal",
  "timestamp_utc": "2026-05-26T12:00:00Z",
  "model_id": "production_model_id",
  "asset": "BTCUSDT",
  "timeframe": "1h",
  "signal": "long_bias",
  "direction_confidence": 0.67,
  "trade_quality_probability": 0.64,
  "regime": "trending_up",
  "cycle_phase": "bull",
  "entry_reference": 0.0,
  "stop_reference": 0.0,
  "tp1_reference": 0.0,
  "tp2_reference": 0.0,
  "tp3_reference": 0.0,
  "estimated_rr": 2.0,
  "risk_per_trade_pct": 1.0,
  "leverage": 1,
  "liquidation_buffer_pct": null,
  "scorecard": {
    "trend": "bullish",
    "ema_stack": "bullish",
    "rsi": 61.2,
    "macd": "bullish",
    "funding": "normal",
    "open_interest": "rising",
    "fear_greed": "greed",
    "data_coverage_score": 0.82
  },
  "warnings": [
    "Decision-support only. Not financial advice.",
    "Validate manually before trading."
  ],
  "cooldown_minutes": 60
}
```

Telegram/Discord human-readable template:

```text
🚨 BTCUSDT 1h — LONG_BIAS

Model: production_model_id
Confidence: 67%
Trade Quality: 64%
Regime: trending_up
Cycle: bull

Entry Ref: <price>
SL Ref: <price>
TP1 / TP2 / TP3: <price> / <price> / <price>
Estimated R:R: 1:2+
Risk: 1.0%
Leverage: 1x

Scorecard:
- Trend: bullish
- EMA Stack: bullish
- RSI: 61.2
- MACD: bullish
- Funding/OI: normal
- Fear & Greed: greed
- Data Coverage: 82%

Warnings:
Decision-support only. Not financial advice. Validate manually before trading.
```

Email template must include the same fields plus full JSON attached or pasted below the readable summary.

Do not send alerts for:

```text
no_trade
range_wait
high_risk
model_stale
data_coverage_low
```

---

## INFRASTRUCTURE

### GitHub Actions

Create:

```text
.github/workflows/daily_data.yml
.github/workflows/smoke_tests.yml
.github/workflows/weekly_retrain_notice.yml
```

#### daily_data.yml

Runs once daily UTC and manually through `workflow_dispatch`.

Steps:

1. Checkout.
2. Setup Python 3.11.
3. Install minimal ingestion dependencies.
4. Run daily delta update.
5. Run data validation.
6. Update metadata files.
7. Run retrain/staleness check.
8. Push dataset to Kaggle dataset if Kaggle secrets exist.
9. Commit only metadata/checksums/manifests/reports, not large data.

Secrets:

```text
KAGGLE_USERNAME
KAGGLE_KEY
FRED_API_KEY optional
ETHERSCAN_API_KEY optional
TELEGRAM_BOT_TOKEN optional
TELEGRAM_CHAT_ID optional
DISCORD_WEBHOOK_URL optional
SMTP_* optional
```

If Kaggle secrets are missing, the workflow should still run local validation and log that Kaggle push was skipped.

#### smoke_tests.yml

Run tests on PR/push:

```text
pytest tests/
```

Use tiny sample data only.

#### weekly_retrain_notice.yml

Do not train heavy models.

Only produce a report saying whether enough new data exists to retrain.

### Kaggle notebook

Create:

```text
notebooks/kaggle_train.ipynb
```

It must:

1. Install requirements.
2. Detect GPUs.
3. Use dual T4 via MirroredStrategy if available.
4. Pull latest Kaggle dataset.
5. Build features/labels or load prebuilt features.
6. Optionally run small hparam search.
7. Train selected timeframe models.
8. Save checkpoints.
9. Save final artifacts.
10. Run evaluation and backtest.
11. Register model as candidate.
12. Optionally promote if gates pass.
13. Save evaluation reports.
14. Export artifacts as Kaggle output/dataset.

### Colab notebook

Create:

```text
notebooks/colab_train.ipynb
```

It must:

1. Mount Google Drive optionally.
2. Install requirements.
3. Detect T4 GPU.
4. Pull data from Kaggle or uploaded archive.
5. Train/resume.
6. Save checkpoints to Drive.
7. Run evaluation and backtest.
8. Register candidate model.
9. Export final artifacts.

---

## REPO LAYOUT

Create:

```text
.
├── README.md
├── requirements.txt
├── environment.md
├── .gitignore
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── features/
│   ├── labels/
│   ├── samples/
│   └── manual/
├── metadata/
│   ├── source_registry.yaml
│   ├── dataset_manifest.json
│   ├── watermarks.json
│   ├── checksums.json
│   ├── feature_registry.json
│   ├── model_registry.json
│   └── retrain_status.json
├── reference/
│   └── halvings.csv
├── src/
│   ├── __init__.py
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── binance_bulk.py
│   │   ├── ccxt_incremental.py
│   │   ├── derivatives.py
│   │   ├── onchain.py
│   │   ├── sentiment.py
│   │   ├── macro.py
│   │   ├── coingecko.py
│   │   ├── paid_stubs.py
│   │   └── daily_update.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── structure.py
│   │   ├── indicators.py
│   │   ├── patterns.py
│   │   ├── flow.py
│   │   ├── onchain.py
│   │   ├── sentiment.py
│   │   ├── macro.py
│   │   ├── scorecard.py
│   │   ├── selection.py
│   │   └── build_matrix.py
│   ├── labels/
│   │   ├── __init__.py
│   │   └── labeling.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── build_dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── multitask_model.py
│   │   ├── train.py
│   │   ├── train_like_plantguard.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   ├── thresholds.py
│   │   ├── hparam_search.py
│   │   ├── registry.py
│   │   ├── promote.py
│   │   ├── rollback.py
│   │   ├── drift.py
│   │   ├── drift_viz.py
│   │   ├── shadow.py
│   │   ├── ab_compare.py
│   │   └── retrain_check.py
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── broker.py
│   │   ├── metrics.py
│   │   ├── strategies.py
│   │   ├── costs.py
│   │   ├── margin.py
│   │   ├── liquidation.py
│   │   └── funding.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── ema120_cycle.py
│   │   ├── macd_rsi_mtf.py
│   │   ├── bollinger_mean_reversion.py
│   │   ├── funding_oi_governor.py
│   │   ├── eth_btc_rotation.py
│   │   ├── wyckoff_proxy.py
│   │   ├── ict_smc_proxy.py
│   │   ├── dynamic_grid_research.py
│   │   └── funding_arbitrage_research.py
│   ├── serve/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── scheduler.py
│   │   ├── drift_dashboard.py
│   │   ├── alert_templates.py
│   │   └── alerts.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── time.py
│       ├── logging.py
│       ├── seeds.py
│       ├── validation.py
│       └── hardware.py
├── notebooks/
│   ├── kaggle_train.ipynb
│   ├── colab_train.ipynb
│   ├── kaggle_train_plantguard_style.ipynb
│   └── colab_train_plantguard_style.ipynb
├── .github/
│   └── workflows/
│       ├── daily_data.yml
│       ├── smoke_tests.yml
│       └── weekly_retrain_notice.yml
├── tests/
│   ├── test_no_lookahead.py
│   ├── test_labeling.py
│   ├── test_ingest_idempotent.py
│   ├── test_feature_schema.py
│   ├── test_scorecard.py
│   ├── test_feature_selection_no_leakage.py
│   ├── test_backtest_engine.py
│   ├── test_model_registry.py
│   ├── test_drift_visualization.py
│   ├── test_shadow_ab.py
│   ├── test_margin_liquidation.py
│   ├── test_alert_templates.py
│   └── test_serving_contract.py
├── artifacts/
│   ├── production/
│   │   └── current_model.json
│   └── runs/
└── reports/
```

---

## TESTS

Create tests using tiny sample data.

### test_no_lookahead.py

Prove that changing future rows does not change past features.

Also prove scalers are fit only on training data.

### test_labeling.py

Test triple-barrier labels with known artificial price paths.

Cases:

- Upper barrier hit first
- Lower barrier hit first
- Neither hit, sideways
- Both touched in same candle, ambiguous handled consistently

### test_ingest_idempotent.py

Run ingestion twice on the same sample and prove row count is unchanged.

### test_feature_schema.py

Ensure feature matrix contains expected columns and no forbidden future columns.

### test_scorecard.py

Ensure unavailable data is marked unavailable, not guessed.

### test_feature_selection_no_leakage.py

Ensure feature selection is fit only on training folds.

### test_backtest_engine.py

Test:

- Fees
- Slippage
- Stop loss
- TP1/TP2/TP3
- Breakeven stop move
- Max daily loss

### test_model_registry.py

Test:

- Register candidate
- Promote candidate
- Rollback to previous model
- No artifact overwrite

### test_serving_contract.py

Test:

- FastAPI health endpoint
- Latest prediction schema
- Current model endpoint

### test_drift_visualization.py

Test:

- PSI chart generation
- Drift dashboard HTML generation
- Drift JSON output schema

### test_shadow_ab.py

Test:

- Production and candidate predictions are both logged
- Candidate does not replace production signal in shadow mode
- A/B comparison report is generated

### test_margin_liquidation.py

Test:

- Leverage cap
- Approximate liquidation price
- Liquidation buffer rejection
- Funding fee application
- Liquidation event logging

### test_alert_templates.py

Test:

- Canonical alert JSON schema
- Telegram/Discord message includes required fields
- No alert is sent for no_trade/high_risk/model_stale

---

## BUILD PHASES

### Phase 0 — Scaffold

Create repo layout, config, requirements, environment docs, logging utils, seed utils, hardware detection, README skeleton, `.gitignore`, and halving reference file.

Gate:

```bash
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
python -m src.utils.hardware
pytest tests/ -q
```

If TA-Lib fails, document fallback and ensure pandas-ta-classic path works.

Stop after gate output.

### Phase 1 — Binance bulk ingestion

Implement Binance public data discovery, download, checksum verification, normalization, Parquet writing, and coverage reporting.

Gate:

```bash
python -m src.ingest.binance_bulk --symbols BTCUSDT ETHUSDT --market-types spot futures_um --timeframes 1m 1h 4h 1d --dry-run false
python -m src.ingest.binance_bulk --symbols BTCUSDT ETHUSDT --market-types spot futures_um --timeframes 1h --dry-run false
pytest tests/test_ingest_idempotent.py -q
```

Print:

- First timestamp
- Last timestamp
- Row count
- Duplicate count
- Missing candle estimate
- Checksum status

Stop after gate output.

### Phase 2 — Other data adapters

Implement derivatives, sentiment, CoinGecko/dominance, on-chain, macro, and paid-source stubs.

Gate:

```bash
python -m src.ingest.derivatives --symbols BTCUSDT ETHUSDT
python -m src.ingest.sentiment
python -m src.ingest.coingecko
python -m src.ingest.onchain
python -m src.ingest.macro
```

Each adapter must return a dated DataFrame or a clear unavailable log.

Stop after gate output.

### Phase 3 — Features, labels, and feature selection

Implement feature builders, scorecard, feature selection, triple-barrier labels, regime labels, cycle labels, and trade-quality labels.

Gate:

```bash
python -m src.features.build_matrix --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --sample true
python -m src.labels.labeling --symbols BTCUSDT ETHUSDT --timeframes 1h 4h --sample true
python -m src.features.selection --sample true
pytest tests/test_no_lookahead.py -q
pytest tests/test_labeling.py -q
pytest tests/test_feature_schema.py -q
pytest tests/test_scorecard.py -q
pytest tests/test_feature_selection_no_leakage.py -q
```

Print class balance and selected-feature report.

Stop after gate output.

### Phase 4 — Dataset and model training

Implement dataset builder, MTF Transformer model architecture, train loop, class weights, threshold preparation, checkpointing, resume, mixed precision, and multi-GPU support.

Gate CPU smoke test:

```bash
python -m src.datasets.build_dataset --symbols BTCUSDT ETHUSDT --timeframes 1h --sample true
python -m src.models.train --timeframe 1h --sample true --epochs 2
python -m src.models.train_like_plantguard --timeframes 1h --sample true --phase1-epochs 1 --phase2-epochs 1
```

Gate GPU detection:

```bash
python -m src.utils.hardware
```

On Kaggle, if dual T4 is available, confirm MirroredStrategy is used.

Stop after gate output.

### Phase 5 — Evaluation, backtesting, and prediction

Implement evaluate.py, backtest engine, thresholds.py, and predict.py.

Gate:

```bash
python -m src.models.evaluate --latest --sample true
python -m src.backtest.engine --latest --sample true
python -m src.models.predict --latest --symbols BTCUSDT ETHUSDT --timeframes 1h 4h
```

Must produce:

```text
reports/eval_<run_id>.md
reports/eval_<run_id>.json
reports/backtest_<run_id>.md
reports/backtest_<run_id>.json
reports/trades_<run_id>.csv
```

Report must clearly state whether model beats baselines.

Stop after gate output.

### Phase 6 — Model registry, promotion, rollback, drift visualization, shadow A/B, and retraining checks

Implement registry, promotion, rollback, drift, drift visualization, shadow A/B testing, and retrain_check.

Gate:

```bash
python -m src.models.registry --list
python -m src.models.promote --latest --dry-run
python -m src.models.retrain_check
python -m src.models.drift_viz --sample true
python -m src.models.shadow --candidate latest --sample true
python -m src.models.ab_compare --candidate latest --sample true
pytest tests/test_model_registry.py -q
pytest tests/test_drift_visualization.py -q
pytest tests/test_shadow_ab.py -q
```

Stop after gate output.

### Phase 7 — Serving, alert templates, alerts, and drift dashboard

Implement FastAPI serving, scheduler, drift dashboard, canonical alert templates, and disabled-by-default alerts.

Gate:

```bash
uvicorn src.serve.api:app --host 0.0.0.0 --port 8000
python -m src.serve.alert_templates --sample true
pytest tests/test_serving_contract.py -q
pytest tests/test_alert_templates.py -q
```

Endpoints must return valid JSON.

Alert template sample must include asset, timeframe, signal, confidence, trade quality, model ID, scorecard summary, risk levels, and disclaimer.

Stop after gate output.

### Phase 8 — Automation and notebooks

Implement GitHub Actions and notebooks.

Gate:

```bash
pytest tests/ -q
```

Also verify:

- `daily_data.yml` does not train
- Large data is not committed
- Kaggle push is skipped gracefully if secrets are missing
- Kaggle notebook imports from `src/`
- Colab notebook imports from `src/`
- Both notebooks support resume
- Retrain report is produced but no heavy training happens in GitHub Actions

Stop after gate output.

---

## README REQUIREMENTS

README must include:

1. What the project does.
2. What it does not do.
3. Why “deepest verified free history” is used instead of “from zero.”
4. Data-source table with coverage limitations.
5. PlantGuard-style training workflow explanation adapted for BTC/ETH time series.
6. Model architecture explanation.
7. Feature-selection explanation.
7. Class-imbalance strategy.
8. Backtesting assumptions.
9. Model registry and promotion process.
10. Retraining/staleness policy.
11. Drift visualization dashboard.
12. Shadow A/B testing process.
13. Futures leverage/liquidation simulation limits.
14. Serving/API usage.
15. Alert setup and alert message fields.
16. Kaggle setup.
14. Colab setup.
15. GitHub Actions setup.
16. Optional API keys:
   - KAGGLE_USERNAME
   - KAGGLE_KEY
   - FRED_API_KEY
   - ETHERSCAN_API_KEY
   - TELEGRAM_BOT_TOKEN
   - TELEGRAM_CHAT_ID
   - DISCORD_WEBHOOK_URL
17. How to run ingestion.
18. How to build features.
19. How to train.
20. How to evaluate.
21. How to backtest.
22. How to promote a model.
23. How to roll back a model.
24. How to predict.
25. How to interpret output.
26. Limitations.
27. Financial-risk disclaimer.

README must plainly state:

> Markets are noisy and adversarial. No model reliably predicts BTC/ETH prices all the time. This system is decision-support only, not financial advice. A model that fails to beat baselines after fees/slippage must not be used for trading.

---

## DELIVERABLES

Deliver working code for:

- Every repo file above
- Config
- Tests
- Kaggle notebook
- Colab notebook
- Kaggle PlantGuard-style training notebook
- Colab PlantGuard-style training notebook
- PlantGuard-style training script
- GitHub Actions workflows
- README
- Environment docs
- Evaluation report template
- Backtest report template
- Prediction JSON output
- FastAPI serving layer
- Optional disabled alert adapters
- Canonical alert templates
- Drift visualization dashboard
- Shadow A/B testing reports
- Futures margin/liquidation simulation
- Scorecard output
- Strategy baseline modules
- Paid-source disabled stubs
- Dataset manifests
- Coverage reports
- Feature-selection reports
- Model registry
- Promotion/rollback scripts
- Retrain/staleness reports

Start with Phase 0 now.

After Phase 0, show me:

1. Files created.
2. Installation result.
3. Hardware detection result.
4. Test result.
5. Any issues or assumptions.

Then wait for my go-ahead.
