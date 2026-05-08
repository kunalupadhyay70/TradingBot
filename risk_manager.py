"""
risk_manager.py — Position sizing, stop-loss/take-profit calculation,
and overtrading guard.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils import get_logger

logger = get_logger("RiskManager")


# ─────────────────────────────────────────────
# Position & Trade Records
# ─────────────────────────────────────────────

@dataclass
class Position:
    """Represents an open position."""
    side: str           # 'BUY' or 'SELL'
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: float = field(default_factory=time.time)

    @property
    def is_long(self) -> bool:
        return self.side == "BUY"

    def check_exit(self, current_price: float) -> Optional[str]:
        """
        Check if SL or TP is hit.
        Returns 'STOP_LOSS' | 'TAKE_PROFIT' | None.
        """
        if self.is_long:
            if current_price <= self.stop_loss:
                return "STOP_LOSS"
            if current_price >= self.take_profit:
                return "TAKE_PROFIT"
        else:  # short
            if current_price >= self.stop_loss:
                return "STOP_LOSS"
            if current_price <= self.take_profit:
                return "TAKE_PROFIT"
        return None

    def unrealized_pnl(self, current_price: float) -> float:
        """Return unrealised PnL in quote currency."""
        if self.is_long:
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity


# ─────────────────────────────────────────────
# Risk Manager
# ─────────────────────────────────────────────

class RiskManager:
    """
    Controls:
    - Fixed-% position sizing
    - Stop-loss and take-profit placement
    - Maximum open positions
    - Trade frequency guard (max trades per hour)
    """

    def __init__(self, cfg: Dict[str, Any]):
        risk_cfg = cfg["risk"]

        self.capital: float = risk_cfg["capital"]
        self.risk_per_trade_pct: float = risk_cfg["risk_per_trade_pct"]
        self.stop_loss_pct: float = risk_cfg["stop_loss_pct"]
        self.take_profit_pct: float = risk_cfg["take_profit_pct"]
        self.max_open_positions: int = risk_cfg["max_open_positions"]
        self.max_trades_per_hour: int = risk_cfg["max_trades_per_hour"]
        self.fee_pct: float = risk_cfg.get("fee_pct", 0.001)

        # Equity tracking
        self._equity: float = self.capital
        self._open_positions: List[Position] = []

        # Overtrading guard: record trade timestamps
        self._trade_timestamps: deque = deque(maxlen=self.max_trades_per_hour * 2)

    # ─────────────────────────────────────────
    # Checks
    # ─────────────────────────────────────────

    def can_trade(self) -> bool:
        """
        Returns False if:
        - Max open positions already reached
        - Too many trades in the last hour
        """
        if len(self._open_positions) >= self.max_open_positions:
            logger.debug("Cannot trade: max open positions reached")
            return False

        now = time.time()
        one_hour_ago = now - 3600
        recent = [t for t in self._trade_timestamps if t > one_hour_ago]
        if len(recent) >= self.max_trades_per_hour:
            logger.warning(
                f"Overtrading guard: {len(recent)} trades in last hour "
                f"(max={self.max_trades_per_hour})"
            )
            return False

        return True

    def has_open_position(self, side: str = None) -> bool:
        if side is None:
            return len(self._open_positions) > 0
        return any(p.side == side for p in self._open_positions)

    # ─────────────────────────────────────────
    # Position Sizing
    # ─────────────────────────────────────────

    def compute_position_size(self, price: float) -> float:
        """
        Fixed-% risk sizing:
          risk_amount = equity * risk_per_trade_pct
          quantity    = risk_amount / (price * stop_loss_pct)

        Returns quantity (in base asset units).
        """
        risk_amount = self._equity * self.risk_per_trade_pct
        # The most we can lose per unit = price * stop_loss_pct
        quantity = risk_amount / (price * self.stop_loss_pct)
        # Ensure we have enough equity to cover the position
        max_affordable = (self._equity * 0.95) / price  # keep 5% buffer
        quantity = min(quantity, max_affordable)
        return max(quantity, 0.0)

    # ─────────────────────────────────────────
    # SL / TP Levels
    # ─────────────────────────────────────────

    def compute_levels(
        self, side: str, entry_price: float
    ) -> tuple:
        """
        Compute (stop_loss, take_profit) prices.
        """
        if side == "BUY":
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # SELL (short)
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)
        return stop_loss, take_profit

    # ─────────────────────────────────────────
    # Position Lifecycle
    # ─────────────────────────────────────────

    def open_position(self, side: str, entry_price: float) -> Optional[Position]:
        """
        Open a new position if risk checks pass.
        Returns the Position object, or None if blocked.
        """
        if not self.can_trade():
            return None

        quantity = self.compute_position_size(entry_price)
        if quantity <= 0:
            logger.warning("Computed position size is 0. Skipping.")
            return None

        stop_loss, take_profit = self.compute_levels(side, entry_price)

        pos = Position(
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self._open_positions.append(pos)
        self._trade_timestamps.append(time.time())

        logger.info(
            f"Position OPENED: {side} {quantity:.6f} @ {entry_price:.2f} | "
            f"SL={stop_loss:.2f} TP={take_profit:.2f}"
        )
        return pos

    def close_position(
        self, pos: Position, exit_price: float, reason: str = ""
    ) -> float:
        """
        Close position and update equity.
        Returns realised PnL (after fees).
        """
        if pos not in self._open_positions:
            logger.warning("Tried to close a position not in open list")
            return 0.0

        gross_pnl = pos.unrealized_pnl(exit_price)
        # Fees: entry + exit
        fee = pos.entry_price * pos.quantity * self.fee_pct
        fee += exit_price * pos.quantity * self.fee_pct
        net_pnl = gross_pnl - fee

        self._equity += net_pnl
        self._open_positions.remove(pos)

        logger.info(
            f"Position CLOSED [{reason}]: {pos.side} {pos.quantity:.6f} "
            f"entry={pos.entry_price:.2f} exit={exit_price:.2f} | "
            f"PnL={net_pnl:+.2f} | Equity={self._equity:.2f}"
        )
        return net_pnl

    def check_and_close_positions(
        self, current_price: float
    ) -> List[tuple]:
        """
        Check all open positions against SL/TP.
        Returns list of (position, exit_reason, pnl) for closed ones.
        """
        closed = []
        for pos in list(self._open_positions):
            exit_reason = pos.check_exit(current_price)
            if exit_reason:
                pnl = self.close_position(pos, current_price, exit_reason)
                closed.append((pos, exit_reason, pnl))
        return closed

    # ─────────────────────────────────────────
    # Accessors
    # ─────────────────────────────────────────

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def open_positions(self) -> List[Position]:
        return list(self._open_positions)

    def update_equity(self, amount: float) -> None:
        """Directly adjust equity (used by execution engine)."""
        self._equity += amount

    def reset(self, capital: float = None) -> None:
        """Reset state (used by backtester)."""
        self._equity = capital or self.capital
        self._open_positions.clear()
        self._trade_timestamps.clear()
