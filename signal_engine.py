"""
signal_engine.py — Combines ML model prediction and order book metrics
to produce a final trading signal: BUY | SELL | HOLD.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from model import DirectionModel
from orderbook_analyzer import OrderBookMetrics
from utils import get_logger

logger = get_logger("SignalEngine")


# ─────────────────────────────────────────────
# Signal Output
# ─────────────────────────────────────────────

@dataclass
class TradeSignal:
    action: str            # 'BUY' | 'SELL' | 'HOLD'
    model_direction: str   # 'UP' | 'DOWN'
    model_confidence: float
    ob_signal: str         # order book signal string
    ob_momentum: float
    ob_imbalance: float
    spread_pct: float
    reason: str            # human-readable explanation

    def __repr__(self) -> str:
        return (
            f"TradeSignal(action={self.action}, "
            f"model={self.model_direction}@{self.model_confidence:.2%}, "
            f"ob={self.ob_signal}, momentum={self.ob_momentum:.3f})"
        )


# ─────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────

class SignalEngine:
    """
    Fuses ML model output with order book analysis.

    Decision rules:
      BUY  → model says UP with confidence ≥ threshold
               AND order book shows BUY_PRESSURE
               AND spread is below max
      SELL → model says DOWN with confidence ≥ threshold
               AND order book shows SELL_PRESSURE
               AND spread is below max
      HOLD → any other combination
    """

    def __init__(self, model: DirectionModel, cfg: Dict[str, Any]):
        self.model = model
        signal_cfg = cfg["signal"]

        self.confidence_threshold: float = signal_cfg.get(
            "model_confidence_threshold", 0.60
        )
        self.imbalance_threshold: float = signal_cfg.get(
            "imbalance_threshold", 0.15
        )
        self.momentum_threshold: float = signal_cfg.get(
            "momentum_threshold", 0.40
        )
        self.max_spread_pct: float = signal_cfg.get("max_spread_pct", 0.05) / 100

    def generate(
        self,
        feature_vector: pd.DataFrame,
        ob_metrics: OrderBookMetrics,
    ) -> TradeSignal:
        """
        Generate a trading signal.

        Parameters
        ----------
        feature_vector : pd.DataFrame
            Single-row DataFrame with model features for the latest candle.
        ob_metrics : OrderBookMetrics
            Live order book analysis result.

        Returns
        -------
        TradeSignal
        """
        # ── Step 1: Model Prediction ──────────
        prob_down, prob_up = self.model.predict_proba(feature_vector)
        if prob_up >= prob_down:
            model_direction = "UP"
            model_confidence = prob_up
        else:
            model_direction = "DOWN"
            model_confidence = prob_down

        logger.debug(
            f"Model → {model_direction} @ {model_confidence:.2%}  "
            f"| OB → {ob_metrics.signal} momentum={ob_metrics.momentum_score:.3f}"
        )

        # ── Step 2: Spread Filter ─────────────
        if ob_metrics.spread_pct > self.max_spread_pct:
            return TradeSignal(
                action="HOLD",
                model_direction=model_direction,
                model_confidence=model_confidence,
                ob_signal=ob_metrics.signal,
                ob_momentum=ob_metrics.momentum_score,
                ob_imbalance=ob_metrics.imbalance,
                spread_pct=ob_metrics.spread_pct,
                reason=f"Spread too wide: {ob_metrics.spread_pct:.4%} > {self.max_spread_pct:.4%}",
            )

        # ── Step 3: Confidence Filter ─────────
        if model_confidence < self.confidence_threshold:
            return TradeSignal(
                action="HOLD",
                model_direction=model_direction,
                model_confidence=model_confidence,
                ob_signal=ob_metrics.signal,
                ob_momentum=ob_metrics.momentum_score,
                ob_imbalance=ob_metrics.imbalance,
                spread_pct=ob_metrics.spread_pct,
                reason=f"Model confidence too low: {model_confidence:.2%} < {self.confidence_threshold:.2%}",
            )

        # ── Step 4: Order Book Confirmation ───
        action, reason = self._apply_ob_filter(
            model_direction, ob_metrics
        )

        return TradeSignal(
            action=action,
            model_direction=model_direction,
            model_confidence=model_confidence,
            ob_signal=ob_metrics.signal,
            ob_momentum=ob_metrics.momentum_score,
            ob_imbalance=ob_metrics.imbalance,
            spread_pct=ob_metrics.spread_pct,
            reason=reason,
        )

    # ─────────────────────────────────────────
    # Order Book Filter
    # ─────────────────────────────────────────

    def _apply_ob_filter(
        self, model_direction: str, ob: OrderBookMetrics
    ) -> Tuple[str, str]:
        """
        Apply order book confirmation rules.
        Returns (action, reason).
        """
        imb = ob.imbalance
        mom = ob.momentum_score

        if model_direction == "UP":
            if ob.signal == "BUY_PRESSURE":
                return "BUY", (
                    f"Model UP @ confidence met | OB BUY_PRESSURE "
                    f"imbalance={imb:+.3f} momentum={mom:.3f}"
                )
            # Secondary check: strong imbalance even without signal classification
            if (
                imb >= self.imbalance_threshold
                and mom >= 0.5 + self.momentum_threshold / 2
            ):
                return "BUY", (
                    f"Model UP | OB imbalance={imb:+.3f} momentum={mom:.3f} "
                    f"(threshold override)"
                )
            return "HOLD", (
                f"Model UP but OB not confirming: "
                f"signal={ob.signal} imbalance={imb:+.3f}"
            )

        if model_direction == "DOWN":
            if ob.signal == "SELL_PRESSURE":
                return "SELL", (
                    f"Model DOWN @ confidence met | OB SELL_PRESSURE "
                    f"imbalance={imb:+.3f} momentum={mom:.3f}"
                )
            if (
                imb <= -self.imbalance_threshold
                and mom <= 0.5 - self.momentum_threshold / 2
            ):
                return "SELL", (
                    f"Model DOWN | OB imbalance={imb:+.3f} momentum={mom:.3f} "
                    f"(threshold override)"
                )
            return "HOLD", (
                f"Model DOWN but OB not confirming: "
                f"signal={ob.signal} imbalance={imb:+.3f}"
            )

        return "HOLD", "Unknown model direction"
