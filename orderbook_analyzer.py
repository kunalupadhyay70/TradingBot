"""
orderbook_analyzer.py — Real-time order book metrics.

Computes:
  • Bid-Ask Spread (absolute and %)
  • Order Imbalance  = (bid_vol - ask_vol) / (bid_vol + ask_vol)
  • Momentum Score   = composite of imbalance trend + trade flow + order consumption
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from utils import get_logger

logger = get_logger("OrderBookAnalyzer")


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class OrderBookMetrics:
    """Output of one analyzer tick."""
    timestamp: float
    best_bid: float
    best_ask: float
    spread_abs: float         # best_ask - best_bid
    spread_pct: float         # spread_abs / mid_price
    bid_volume: float         # total volume across tracked depth
    ask_volume: float
    imbalance: float          # (bid_vol - ask_vol) / (bid_vol + ask_vol)
    momentum_score: float     # 0..1, positive = buy pressure
    signal: str               # 'BUY_PRESSURE' | 'SELL_PRESSURE' | 'NEUTRAL'

    def __repr__(self) -> str:
        return (
            f"OBMetrics(spread={self.spread_pct:.4%}, "
            f"imbalance={self.imbalance:+.3f}, "
            f"momentum={self.momentum_score:.3f}, "
            f"signal={self.signal})"
        )


# ─────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────

class OrderBookAnalyzer:
    """
    Stateful analyzer that accepts live order book snapshots and
    recent trade lists, then produces OrderBookMetrics.

    Rolling windows are used to detect imbalance trends and
    trade flow momentum.
    """

    def __init__(self, cfg: Dict):
        signal_cfg = cfg["signal"]
        self._imbalance_window: int = signal_cfg.get("rolling_imbalance_window", 10)
        self._imbalance_threshold: float = signal_cfg.get("imbalance_threshold", 0.15)
        self._momentum_threshold: float = signal_cfg.get("momentum_threshold", 0.40)
        self._max_spread_pct: float = signal_cfg.get("max_spread_pct", 0.05) / 100

        # Rolling history for imbalance trend
        self._imbalance_history: deque = deque(maxlen=self._imbalance_window)
        # Rolling trade flow (buy qty - sell qty) history
        self._trade_flow_history: deque = deque(maxlen=self._imbalance_window)
        # Previous top-of-book to detect rapid order consumption
        self._prev_bid_vol: Optional[float] = None
        self._prev_ask_vol: Optional[float] = None

    # ─────────────────────────────────────────
    # Main Compute Method
    # ─────────────────────────────────────────

    def analyze(
        self,
        order_book: Dict[str, List],
        recent_trades: List[Dict],
    ) -> OrderBookMetrics:
        """
        Compute all metrics from a live order book snapshot and
        a list of recent trades.

        order_book: {"bids": [[price, qty], …], "asks": [[price, qty], …]}
        recent_trades: [{"price": float, "qty": float, "is_buyer_maker": bool}, …]
        """
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        if not bids or not asks:
            logger.warning("Empty order book received")
            return self._empty_metrics()

        # ── Spread ───────────────────────────
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2
        spread_abs = best_ask - best_bid
        spread_pct = spread_abs / mid_price if mid_price > 0 else 0.0

        # ── Volume at Depth ──────────────────
        bid_volume = sum(float(q) for _, q in bids)
        ask_volume = sum(float(q) for _, q in asks)
        total_volume = bid_volume + ask_volume

        # ── Order Imbalance ──────────────────
        imbalance = 0.0
        if total_volume > 0:
            imbalance = (bid_volume - ask_volume) / total_volume
        self._imbalance_history.append(imbalance)

        # ── Trade Flow (buy vs sell pressure) ─
        buy_flow, sell_flow = self._compute_trade_flow(recent_trades)
        trade_total = buy_flow + sell_flow
        trade_imbalance = (
            (buy_flow - sell_flow) / trade_total if trade_total > 0 else 0.0
        )
        self._trade_flow_history.append(trade_imbalance)

        # ── Order Consumption ────────────────
        consumption_score = self._compute_consumption_score(bid_volume, ask_volume)

        # ── Composite Momentum Score ─────────
        momentum_score = self._compute_momentum_score(
            current_imbalance=imbalance,
            trade_imbalance=trade_imbalance,
            consumption_score=consumption_score,
        )

        # ── Signal Classification ─────────────
        signal = self._classify_signal(imbalance, momentum_score, spread_pct)

        # Update previous volumes
        self._prev_bid_vol = bid_volume
        self._prev_ask_vol = ask_volume

        metrics = OrderBookMetrics(
            timestamp=time.time(),
            best_bid=best_bid,
            best_ask=best_ask,
            spread_abs=spread_abs,
            spread_pct=spread_pct,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance=imbalance,
            momentum_score=momentum_score,
            signal=signal,
        )
        logger.debug(str(metrics))
        return metrics

    # ─────────────────────────────────────────
    # Sub-Computations
    # ─────────────────────────────────────────

    def _compute_trade_flow(
        self, trades: List[Dict]
    ) -> tuple:
        """
        Split recent trades into buy-initiated vs sell-initiated volume.
        In Binance aggTrade: is_buyer_maker=True means buyer is on book,
        so the aggressor is a seller.
        """
        buy_vol = 0.0
        sell_vol = 0.0
        for t in trades:
            qty = float(t.get("qty", 0))
            if t.get("is_buyer_maker", False):
                sell_vol += qty   # seller aggressed
            else:
                buy_vol += qty    # buyer aggressed
        return buy_vol, sell_vol

    def _compute_consumption_score(
        self, current_bid_vol: float, current_ask_vol: float
    ) -> float:
        """
        Detect rapid order consumption:
        A sharp drop in bid volume signals buying ate through the book.
        A sharp drop in ask volume signals selling ate through the book.
        Returns score in [-1, 1]: positive = bid side being eaten (buy pressure).
        """
        if self._prev_bid_vol is None or self._prev_ask_vol is None:
            return 0.0

        bid_consumed = (self._prev_bid_vol - current_bid_vol) / (
            self._prev_bid_vol + 1e-9
        )
        ask_consumed = (self._prev_ask_vol - current_ask_vol) / (
            self._prev_ask_vol + 1e-9
        )

        # Positive bid consumption → ask side being lifted → buy pressure
        score = float(np.clip(ask_consumed - bid_consumed, -1, 1))
        return score

    def _compute_imbalance_trend(self) -> float:
        """
        Slope of recent imbalance history.
        Positive → growing buy pressure; negative → growing sell pressure.
        Returns value in [-1, 1].
        """
        hist = list(self._imbalance_history)
        if len(hist) < 3:
            return 0.0
        # Linear regression slope (x = index, y = imbalance)
        x = np.arange(len(hist), dtype=float)
        y = np.array(hist, dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        # Normalise to [-1, 1] assuming max slope ≈ 0.1 per tick
        return float(np.clip(slope / 0.1, -1, 1))

    def _compute_trade_flow_trend(self) -> float:
        """Trend in trade-flow imbalance. Returns value in [-1, 1]."""
        hist = list(self._trade_flow_history)
        if len(hist) < 3:
            return 0.0
        x = np.arange(len(hist), dtype=float)
        y = np.array(hist, dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        return float(np.clip(slope / 0.1, -1, 1))

    def _compute_momentum_score(
        self,
        current_imbalance: float,
        trade_imbalance: float,
        consumption_score: float,
    ) -> float:
        """
        Combine four signals into a single momentum score in [0, 1].
        >0.5 means net buy pressure; <0.5 means net sell pressure.

        Components:
          1. Current order-book imbalance     (weight 0.35)
          2. Trade flow imbalance             (weight 0.35)
          3. Consumption signal               (weight 0.15)
          4. Imbalance trend                  (weight 0.15)
        """
        imbalance_trend = self._compute_imbalance_trend()

        # Normalise each component to [0, 1] (0.5 = neutral)
        def norm(v: float) -> float:
            return float(np.clip((v + 1) / 2, 0, 1))

        w1, w2, w3, w4 = 0.35, 0.35, 0.15, 0.15
        score = (
            w1 * norm(current_imbalance)
            + w2 * norm(trade_imbalance)
            + w3 * norm(consumption_score)
            + w4 * norm(imbalance_trend)
        )
        return float(np.clip(score, 0, 1))

    def _classify_signal(
        self, imbalance: float, momentum: float, spread_pct: float
    ) -> str:
        """Classify order book signal as BUY_PRESSURE / SELL_PRESSURE / NEUTRAL."""
        if spread_pct > self._max_spread_pct:
            return "NEUTRAL"   # spread too wide → unreliable
        if (
            imbalance >= self._imbalance_threshold
            and momentum >= 0.5 + self._momentum_threshold / 2
        ):
            return "BUY_PRESSURE"
        elif (
            imbalance <= -self._imbalance_threshold
            and momentum <= 0.5 - self._momentum_threshold / 2
        ):
            return "SELL_PRESSURE"
        return "NEUTRAL"

    def _empty_metrics(self) -> OrderBookMetrics:
        return OrderBookMetrics(
            timestamp=time.time(),
            best_bid=0.0,
            best_ask=0.0,
            spread_abs=0.0,
            spread_pct=999.0,
            bid_volume=0.0,
            ask_volume=0.0,
            imbalance=0.0,
            momentum_score=0.5,
            signal="NEUTRAL",
        )
