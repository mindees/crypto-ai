"""Backtest performance metrics.

Computed from a trade list + equity curve.  Defensive against edge cases
(empty trade list, single-trade series, all-zero variance).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.backtest.broker import Trade


@dataclass
class BacktestMetrics:
    n_trades: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    median_r: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    avg_holding_bars: float = 0.0
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    fee_drag_total: float = 0.0
    slippage_drag_total: float = 0.0
    worst_10_pnl: list[float] = field(default_factory=list)
    exposure_pct: float = 0.0


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or returns.std(ddof=0) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=0)) * np.sqrt(len(returns))


def _sortino(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf") if returns.mean() > 0 else 0.0
    return float(returns.mean() / downside.std(ddof=0)) * np.sqrt(len(returns))


def compute_metrics(
    trades: list[Trade],
    equity_curve: list[tuple],
    *,
    initial_equity: float,
    total_bars: int,
) -> BacktestMetrics:
    m = BacktestMetrics()
    m.n_trades = len(trades)
    if not trades:
        if equity_curve:
            eq = np.array([e for _, e in equity_curve])
            m.total_return_pct = float((eq[-1] / initial_equity - 1.0) * 100)
            m.max_drawdown_pct = _max_drawdown(eq) * 100
        return m

    rrs = np.array([t.rr_realized for t in trades], dtype=np.float64)
    pnls = np.array([t.realized_pnl for t in trades], dtype=np.float64)
    holds = np.array(
        [(t.close_ts - t.open_ts).total_seconds() if t.close_ts else 0 for t in trades],
        dtype=np.float64,
    )

    m.win_rate = float((rrs > 0).mean()) * 100
    m.avg_r = float(rrs.mean())
    m.median_r = float(np.median(rrs))
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    m.profit_factor = float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)
    m.expectancy_r = float(rrs.mean())
    m.avg_holding_bars = float(holds.mean()) if len(holds) else 0.0
    longs = [t for t in trades if t.side == 1]
    shorts = [t for t in trades if t.side == -1]
    m.long_trades = len(longs)
    m.short_trades = len(shorts)
    if longs:
        m.long_win_rate = float(np.mean([t.rr_realized > 0 for t in longs])) * 100
    if shorts:
        m.short_win_rate = float(np.mean([t.rr_realized > 0 for t in shorts])) * 100
    m.fee_drag_total = float(sum(t.fees_paid for t in trades))
    m.slippage_drag_total = float(sum(t.slippage_cost for t in trades))
    m.worst_10_pnl = sorted(pnls.tolist())[:10]

    if equity_curve:
        eq = np.array([e for _, e in equity_curve])
        m.total_return_pct = float((eq[-1] / initial_equity - 1.0) * 100)
        m.max_drawdown_pct = _max_drawdown(eq) * 100
        rets = np.diff(eq) / eq[:-1]
        m.sharpe = _sharpe(rets)
        m.sortino = _sortino(rets)

    if total_bars > 0:
        m.exposure_pct = float(sum(
            (t.close_ts - t.open_ts).total_seconds() / 3600.0 if t.close_ts else 0
            for t in trades
        ) / max(1.0, total_bars) * 100)

    return m
