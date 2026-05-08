"""
execution_engine.py — Simulates order execution with realistic fees and
slippage. Provides a clean interface for swapping in a real exchange client.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from risk_manager import Position, RiskManager
from signal_engine import TradeSignal
from utils import get_logger

logger = get_logger("ExecutionEngine")


# ─────────────────────────────────────────────
# Order Types
# ─────────────────────────────────────────────

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    """Represents a single order."""
    order_id: str
    side: str             # 'BUY' | 'SELL'
    order_type: str       # 'MARKET' | 'LIMIT'
    requested_price: float
    requested_qty: float
    filled_price: float = 0.0
    filled_qty: float = 0.0
    fee: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    timestamp: float = 0.0
    note: str = ""

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    def __repr__(self) -> str:
        return (
            f"Order({self.order_id[:8]} {self.side} {self.filled_qty:.6f} "
            f"@ {self.filled_price:.2f} [{self.status.value}])"
        )


# ─────────────────────────────────────────────
# Abstract Interface (for real exchange swap-in)
# ─────────────────────────────────────────────

class BaseExchangeClient:
    """
    Interface contract for exchange clients.
    Swap SimulatedExchangeClient for BinanceExchangeClient in live trading.
    """

    def place_market_order(self, symbol: str, side: str, quantity: float) -> Order:
        raise NotImplementedError

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        raise NotImplementedError

    def get_balance(self, asset: str) -> float:
        raise NotImplementedError


# ─────────────────────────────────────────────
# Simulated Exchange Client
# ─────────────────────────────────────────────

class SimulatedExchangeClient(BaseExchangeClient):
    """
    Simulates Binance market order execution with:
    - Configurable taker fee
    - Configurable slippage (adverse price movement on fill)
    """

    _order_counter = 0

    def __init__(self, fee_pct: float = 0.001, slippage_pct: float = 0.0005):
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self._balance: Dict[str, float] = {}

    @classmethod
    def _new_id(cls) -> str:
        cls._order_counter += 1
        return f"SIM-{cls._order_counter:06d}"

    def place_market_order(self, symbol: str, side: str, quantity: float) -> Order:
        raise NotImplementedError(
            "Use ExecutionEngine.execute() which needs a price argument."
        )

    def simulate_fill(
        self, symbol: str, side: str, quantity: float, market_price: float
    ) -> Order:
        """
        Simulate a market order fill with slippage.
        BUY  fills at market_price * (1 + slippage) — we pay a little more
        SELL fills at market_price * (1 - slippage) — we receive a little less
        """
        if side == "BUY":
            fill_price = market_price * (1 + self.slippage_pct)
        else:
            fill_price = market_price * (1 - self.slippage_pct)

        notional = fill_price * quantity
        fee = notional * self.fee_pct

        order = Order(
            order_id=self._new_id(),
            side=side,
            order_type="MARKET",
            requested_price=market_price,
            requested_qty=quantity,
            filled_price=fill_price,
            filled_qty=quantity,
            fee=fee,
            status=OrderStatus.FILLED,
            timestamp=time.time(),
            note=f"slippage={self.slippage_pct:.4%} fee={self.fee_pct:.4%}",
        )
        logger.info(f"Simulated fill: {order}")
        return order

    def get_balance(self, asset: str) -> float:
        return self._balance.get(asset, 0.0)


# ─────────────────────────────────────────────
# Execution Engine
# ─────────────────────────────────────────────

class ExecutionEngine:
    """
    Orchestrates trade execution:
    1. Receives a TradeSignal
    2. Checks risk manager approval
    3. Sends order to exchange client (simulated or real)
    4. Records fill in risk manager
    5. Returns filled Order or None
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        cfg: Dict[str, Any],
        exchange_client: BaseExchangeClient = None,
    ):
        self.risk = risk_manager
        self.symbol = cfg["trading"]["symbol"]
        fee_pct = cfg["risk"].get("fee_pct", 0.001)
        slippage_pct = cfg["risk"].get("slippage_pct", 0.0005)

        self.client: BaseExchangeClient = exchange_client or SimulatedExchangeClient(
            fee_pct=fee_pct, slippage_pct=slippage_pct
        )
        self._order_history: List[Order] = []
        self._position_order_map: Dict[str, Position] = {}  # order_id → Position

    def execute(self, signal: TradeSignal, current_price: float) -> Optional[Order]:
        """
        Execute a trade signal.

        Parameters
        ----------
        signal       : TradeSignal from SignalEngine
        current_price: Latest market price for sizing and fill simulation

        Returns
        -------
        Filled Order if executed, None if blocked or HOLD.
        """
        if signal.action == "HOLD":
            logger.debug(f"HOLD signal — no order: {signal.reason}")
            return None

        # Open position via risk manager (performs all checks)
        pos = self.risk.open_position(signal.action, current_price)
        if pos is None:
            logger.warning("Risk manager blocked trade execution")
            return None

        # Send order to exchange
        order = self.client.simulate_fill(
            symbol=self.symbol,
            side=signal.action,
            quantity=pos.quantity,
            market_price=current_price,
        )

        if not order.is_filled:
            logger.error(f"Order not filled: {order}")
            # Roll back: remove position
            if pos in self.risk.open_positions:
                self.risk._open_positions.remove(pos)
            return None

        # Adjust position entry price to actual fill price
        pos.entry_price = order.filled_price
        # Recompute SL/TP with actual fill price
        pos.stop_loss, pos.take_profit = self.risk.compute_levels(
            pos.side, order.filled_price
        )

        self._order_history.append(order)
        self._position_order_map[order.order_id] = pos

        logger.info(
            f"Executed {signal.action}: {order.filled_qty:.6f} {self.symbol} "
            f"@ {order.filled_price:.2f} | "
            f"SL={pos.stop_loss:.2f} TP={pos.take_profit:.2f} | "
            f"Fee={order.fee:.4f}"
        )
        return order

    def check_exits(self, current_price: float) -> List[tuple]:
        """
        Check all open positions for SL/TP. Close if triggered.
        Returns list of (order, exit_reason, pnl).
        """
        closed = self.risk.check_and_close_positions(current_price)
        results = []
        for pos, reason, pnl in closed:
            # Simulate the exit fill
            exit_side = "SELL" if pos.is_long else "BUY"
            exit_order = self.client.simulate_fill(
                symbol=self.symbol,
                side=exit_side,
                quantity=pos.quantity,
                market_price=current_price,
            )
            self._order_history.append(exit_order)
            results.append((exit_order, reason, pnl))
            logger.info(
                f"Exit [{reason}]: {exit_order} | Net PnL={pnl:+.2f} | "
                f"Equity={self.risk.equity:.2f}"
            )
        return results

    @property
    def order_history(self) -> List[Order]:
        return list(self._order_history)

    def performance_summary(self) -> Dict[str, Any]:
        """Return basic trade statistics from order history."""
        if not self._order_history:
            return {"total_orders": 0}
        total_fees = sum(o.fee for o in self._order_history)
        return {
            "total_orders": len(self._order_history),
            "total_fees_paid": round(total_fees, 4),
            "final_equity": round(self.risk.equity, 2),
        }
