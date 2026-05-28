# Contributing to crypto-ai

Thanks for your interest! This project is a **decision-support research pipeline**,
and contributions are welcome — code, tests, docs, data adapters, or bug reports.

## Ground rules (non-negotiable)

This repo lives or dies by **honesty over hype**. Any contribution must respect:

1. **No lookahead / no leakage.** Every feature is causal — row `t` uses only data
   at or before `t`. New features must pass `tests/test_no_lookahead.py`.
2. **No fabricated metrics.** Never hard-code, inflate, or cherry-pick accuracy,
   win rate, or profitability. If a model doesn't beat baselines after fees, the
   reports must say so.
3. **Free by default.** Don't add a required paid dependency. Paid sources stay as
   disabled stubs.
4. **Reproducible.** Pin versions, set seeds, keep runs logged.
5. **Not financial advice.** Don't add anything that implies guaranteed returns or
   removes the risk disclaimers.

## Dev setup

```bash
git clone https://github.com/mindees/crypto-ai.git
cd crypto-ai
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1   |   *nix: source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q          # should be all green before you start
```

## Workflow

1. Fork → branch from `main` (`feat/...`, `fix/...`, `docs/...`).
2. Make focused changes. Add/extend tests for anything you touch.
3. `pytest tests/ -q` must pass locally.
4. Open a PR using the template. Describe the *why*, not just the *what*.
5. CI (`smoke-tests`) must be green. The "no large data tracked" guard must pass —
   never commit files under `data/` or training artifacts.

## Style

- Python 3.11+ (3.13 works). pandas-native, TF/Keras 3.
- Keep modules focused; no premature abstractions.
- Comments explain *why*, not *what*. Default to no comment.
- New data adapters: return an `AdapterResult` (available **or** a clear
  "unavailable" reason) — never silently fabricate data.

## Good first issues

- Implement the `src/strategies/` research comparators (ema120_cycle, wyckoff_proxy, …).
- Add the futures margin/liquidation/funding simulation (`src/backtest/margin.py`).
- Add a self-supervised encoder pretraining step (masked-window modeling).
- Price-relative feature variants (reduce non-stationary PSI on raw price levels).

By contributing you agree your work is licensed under the repo's [MIT License](LICENSE).
