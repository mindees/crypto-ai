# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-05-28

Initial public release. End-to-end pipeline complete across 8 build phases.

### Added
- **Ingestion** — Binance bulk OHLCV (idempotent, SHA-256 verified; handles the
  2025-01 ms→µs timestamp switch and spot/futures column aliases), ccxt fallback,
  derivatives REST (funding paginated to launch; OI/ratios/taker recent-only),
  blockchain.info on-chain (BTC, since 2009), Alternative.me Fear & Greed,
  CoinGecko dominance, yfinance/FRED macro, and disabled paid-source stubs.
- **Features** — causal indicators, market structure, candlestick patterns,
  derivatives flow (incl. CVD proxy), sentiment/on-chain/macro joins, and a
  transparent rule-based scorecard. Fold-aware feature selection (MI +
  permutation importance) with leakage tests.
- **Labels** — triple-barrier direction, rule-based regime, halving-anchored
  cycle, and ≥2R trade-quality.
- **Model** — multi-timeframe Transformer with cross-timeframe attention and four
  heads (direction / regime / cycle / trade-quality); MirroredStrategy +
  mixed-precision auto-detection; PlantGuard-style two-phase training.
- **Evaluation & backtest** — purged + embargoed walk-forward, honest baselines
  (majority/random/buy-hold/EMA/RSI-MACD/no-trade), event-driven backtester with
  fees, slippage, TP1/2/3 partials, and breakeven stop moves.
- **Lifecycle** — model registry, gated promotion, rollback, PSI drift detection,
  drift dashboard, shadow A/B testing, retrain recommendations.
- **Serving** — TensorFlow-free FastAPI app, local scheduler, disabled-by-default
  Telegram/Discord/email alerts with a canonical payload.
- **Automation** — GitHub Actions (daily delta, smoke tests, weekly retrain
  notice) + Kaggle/Colab training notebooks (standard + PlantGuard-style).
- **Tests** — 88 passing (no-lookahead, labeling, idempotency, schema, scorecard,
  selection-no-leakage, backtest, registry, drift, shadow, serving, alerts).

### Notes
- Local development is CPU-only; shipped smoke models honestly **do not** beat
  baselines. Real training runs on Kaggle/Colab GPU via the notebooks.
- Decision-support only. Not financial advice.

[0.1.0]: https://github.com/mindees/crypto-ai/releases/tag/v0.1.0
