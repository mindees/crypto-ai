"""Fee and slippage models for the backtester.

Both are intentionally explicit so a trade's actual cost is auditable from
the trade log alone (no hidden parameters).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeeModel:
    """Flat fee in basis points per side (taker)."""
    bps_per_side: float = 4.0

    def cost(self, notional: float) -> float:
        return abs(notional) * self.bps_per_side / 10_000.0


@dataclass(frozen=True)
class SlippageModel:
    """ATR/volume-aware slippage capped by ``min_bps`` / ``max_bps``."""
    min_bps_per_side: float = 2.0
    max_bps_per_side: float = 15.0
    atr_fraction: float = 0.02

    def slip_bps(self, *, atr: float, price: float) -> float:
        """Slippage in bps. Falls back to min when atr is unknown/invalid."""
        if price <= 0 or atr is None or atr <= 0:
            return self.min_bps_per_side
        raw_bps = (self.atr_fraction * atr / price) * 10_000.0
        return max(self.min_bps_per_side, min(self.max_bps_per_side, raw_bps))

    def slip_price(self, *, price: float, atr: float, side: int) -> float:
        """``side`` = +1 for long (worse fill = higher), -1 for short (worse fill = lower)."""
        bps = self.slip_bps(atr=atr, price=price)
        return price * (1.0 + side * bps / 10_000.0)
