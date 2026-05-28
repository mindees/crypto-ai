"""Single-position broker simulator.

Position state machine:

* Flat → Long/Short on a signal that passes confidence + risk gates.
* Long/Short → partial closes at TP1/TP2/TP3, then full exit.
* Stop moves to breakeven after TP1, to TP1 after TP2 (per config).
* Vertical-barrier exit at the end of the holding window.

All trades are recorded so backtest_<id>.csv is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from src.backtest.costs import FeeModel, SlippageModel


SIDE_LONG = 1
SIDE_SHORT = -1


@dataclass
class Trade:
    open_ts: datetime
    close_ts: datetime | None
    side: int
    entry_price: float
    exit_price: float | None
    size: float
    stop_price: float
    tp1: float
    tp2: float
    tp3: float
    closes: list[tuple[datetime, float, float, str]] = field(default_factory=list)  # (ts, price, size, reason)
    fees_paid: float = 0.0
    slippage_cost: float = 0.0
    realized_pnl: float = 0.0
    rr_realized: float = 0.0
    atr_at_entry: float = 0.0
    asset: str = ""
    timeframe: str = ""
    exit_reason: str = "open"


@dataclass
class BrokerConfig:
    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 1.0
    max_risk_per_trade_pct: float = 2.0
    max_daily_loss_pct: float = 4.0
    stop_atr_multiple: float = 1.5
    tp1_rr: float = 1.0
    tp2_rr: float = 2.0
    tp3_rr: float = 3.0
    tp1_close_pct: float = 0.33
    tp2_close_pct: float = 0.33
    tp3_close_pct: float = 0.34
    move_sl_to_breakeven_after_tp1: bool = True
    move_sl_to_tp1_after_tp2: bool = True
    max_holding_bars: int = 48


class Broker:
    """A single-position, single-symbol broker. Tracks equity + open trade."""

    def __init__(self, cfg: BrokerConfig, fee: FeeModel, slip: SlippageModel,
                 *, asset: str = "", timeframe: str = ""):
        self.cfg = cfg
        self.fee = fee
        self.slip = slip
        self.equity = cfg.initial_equity
        self.peak_equity = cfg.initial_equity
        self.daily_loss = 0.0
        self.day_key: tuple[int, int, int] | None = None
        self.position: Trade | None = None
        self.history: list[Trade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.asset = asset
        self.timeframe = timeframe

    # --- helpers ------------------------------------------------------------

    def _start_day_if_needed(self, ts: datetime) -> None:
        key = (ts.year, ts.month, ts.day)
        if self.day_key != key:
            self.day_key = key
            self.daily_loss = 0.0

    def _budget_ok(self) -> bool:
        max_loss = self.cfg.initial_equity * self.cfg.max_daily_loss_pct / 100.0
        return self.daily_loss < max_loss

    def position_size_for_risk(self, *, entry: float, stop: float) -> float:
        risk_amount = self.equity * self.cfg.risk_per_trade_pct / 100.0
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return 0.0
        return risk_amount / stop_distance

    # --- lifecycle ----------------------------------------------------------

    def can_open(self, ts: datetime) -> bool:
        self._start_day_if_needed(ts)
        return self.position is None and self._budget_ok()

    def open(self, *, ts: datetime, side: int, entry_price: float, atr: float) -> Trade | None:
        if not self.can_open(ts):
            return None
        slipped_entry = self.slip.slip_price(price=entry_price, atr=atr, side=side)
        r_unit = atr * self.cfg.stop_atr_multiple
        stop = slipped_entry - side * r_unit
        tp1 = slipped_entry + side * r_unit * self.cfg.tp1_rr
        tp2 = slipped_entry + side * r_unit * self.cfg.tp2_rr
        tp3 = slipped_entry + side * r_unit * self.cfg.tp3_rr
        size = self.position_size_for_risk(entry=slipped_entry, stop=stop)
        if size <= 0:
            return None
        # Entry fee
        notional = size * slipped_entry
        fee_cost = self.fee.cost(notional)
        self.equity -= fee_cost

        trade = Trade(
            open_ts=ts, close_ts=None, side=side,
            entry_price=slipped_entry, exit_price=None,
            size=size, stop_price=stop, tp1=tp1, tp2=tp2, tp3=tp3,
            fees_paid=fee_cost,
            slippage_cost=abs(slipped_entry - entry_price) * size,
            atr_at_entry=atr,
            asset=self.asset, timeframe=self.timeframe,
        )
        self.position = trade
        return trade

    def update_bar(self, *, ts: datetime, high: float, low: float, close: float,
                    bars_held: int) -> None:
        """Advance one bar: check stop, TP1/2/3, vertical-barrier exit."""
        if self.position is None:
            return
        t = self.position
        side = t.side

        # Order of checks: stop first (conservative), then TPs by closeness
        if (side == SIDE_LONG and low <= t.stop_price) or (
            side == SIDE_SHORT and high >= t.stop_price
        ):
            self._close_partial(ts, t.stop_price, t.size, "stop", final=True)
            return

        # TPs in order; on a bar that hits multiple, we close them sequentially.
        for level, frac, label in (
            (t.tp1, self.cfg.tp1_close_pct, "tp1"),
            (t.tp2, self.cfg.tp2_close_pct, "tp2"),
            (t.tp3, self.cfg.tp3_close_pct, "tp3"),
        ):
            already = any(c[3] == label for c in t.closes)
            if already:
                continue
            hit = (
                (side == SIDE_LONG and high >= level)
                or (side == SIDE_SHORT and low <= level)
            )
            if hit:
                close_size = t.size * frac
                self._close_partial(ts, level, close_size, label, final=(label == "tp3"))
                if label == "tp1" and self.cfg.move_sl_to_breakeven_after_tp1:
                    t.stop_price = t.entry_price
                if label == "tp2" and self.cfg.move_sl_to_tp1_after_tp2:
                    t.stop_price = t.tp1

        # Vertical-barrier exit
        if self.position is not None and bars_held >= self.cfg.max_holding_bars:
            remaining = self.position.size - sum(c[2] for c in self.position.closes)
            if remaining > 0:
                self._close_partial(ts, close, remaining, "vertical_barrier", final=True)

    def _close_partial(self, ts: datetime, price: float, close_size: float,
                        reason: str, *, final: bool) -> None:
        if self.position is None or close_size <= 0:
            return
        t = self.position
        side = t.side
        # Slippage on exit
        slipped_exit = self.slip.slip_price(price=price, atr=t.atr_at_entry, side=-side)
        notional = close_size * slipped_exit
        fee_cost = self.fee.cost(notional)
        pnl = (slipped_exit - t.entry_price) * close_size * side - fee_cost
        t.realized_pnl += pnl
        t.fees_paid += fee_cost
        t.slippage_cost += abs(slipped_exit - price) * close_size
        t.closes.append((ts, slipped_exit, close_size, reason))
        self.equity += pnl
        self.daily_loss += max(0.0, -pnl)
        self.peak_equity = max(self.peak_equity, self.equity)

        # If everything closed, retire trade
        closed_total = sum(c[2] for c in t.closes)
        if final or closed_total >= t.size - 1e-9:
            t.close_ts = ts
            t.exit_price = slipped_exit
            t.exit_reason = reason
            # Realized R = pnl / (initial risk)
            risk_per_unit = abs(t.entry_price - t.stop_price)  # original (pre-move) stop is on Trade now
            if risk_per_unit > 0:
                t.rr_realized = t.realized_pnl / (risk_per_unit * t.size)
            self.history.append(t)
            self.position = None

    def force_close(self, ts: datetime, price: float, reason: str = "end_of_data") -> None:
        if self.position is None:
            return
        remaining = self.position.size - sum(c[2] for c in self.position.closes)
        if remaining > 0:
            self._close_partial(ts, price, remaining, reason, final=True)

    def mark_equity(self, ts: datetime) -> None:
        self.equity_curve.append((ts, self.equity))
