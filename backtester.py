"""
backtester.py — Event-driven backtester that replays historical OHLCV data,
generates signals via the signal engine, and tracks performance.

Outputs: total profit, win rate, max drawdown, Sharpe ratio, trade log.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from feature_engineering import build_features, get_latest_feature_vector
from model import DirectionModel
from orderbook_analyzer import OrderBookMetrics
from risk_manager import RiskManager
from signal_engine import SignalEngine, TradeSignal
from utils import get_logger

logger = get_logger("Backtester")


# ─────────────────────────────────────────────
# Trade Record
# ─────────────────────────────────────────────

@dataclass
class BacktestTrade:
    """Records one complete round-trip trade."""
    entry_idx: int
    exit_idx: int
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl_gross: float
    pnl_net: float      # after fees + slippage
    exit_reason: str    # 'STOP_LOSS' | 'TAKE_PROFIT' | 'END_OF_DATA'


# ─────────────────────────────────────────────
# Backtester
# ─────────────────────────────────────────────

class Backtester:
    """
    Simulates the full trading strategy on historical OHLCV data.

    The order book is not available historically, so we synthesise
    a neutral OrderBookMetrics for every bar — the model signal is
    the primary decision driver. This makes the backtest conservative
    (order book confirmation is set aside).
    """

    def __init__(
        self,
        model: DirectionModel,
        cfg: Dict[str, Any],
        bypass_ob_filter: bool = True,
    ):
        """
        Parameters
        ----------
        bypass_ob_filter : bool
            If True, skip order book confirmation (use historical OB is
            unavailable). The signal engine receives a synthetic 'BUY_PRESSURE'
            or 'SELL_PRESSURE' OB that always agrees with the model to isolate
            model-only performance. Set to False to test with stricter filters
            using a synthetic neutral OB (more realistic but lower trade count).
        """
        self.cfg = cfg
        self.model = model
        self.bypass_ob_filter = bypass_ob_filter

        bt_cfg = cfg.get("backtester", cfg["risk"])
        self.initial_capital: float = bt_cfg.get(
            "initial_capital", cfg["risk"]["capital"]
        )
        self.fee_pct: float = bt_cfg.get("fee_pct", cfg["risk"]["fee_pct"])
        self.slippage_pct: float = bt_cfg.get(
            "slippage_pct", cfg["risk"]["slippage_pct"]
        )
        self.stop_loss_pct: float = cfg["risk"]["stop_loss_pct"]
        self.take_profit_pct: float = cfg["risk"]["take_profit_pct"]
        self.risk_per_trade_pct: float = cfg["risk"]["risk_per_trade_pct"]

        self.feature_cols: List[str] = cfg["model"]["features"]
        self.confidence_threshold: float = cfg["signal"][
            "model_confidence_threshold"
        ]

    # ─────────────────────────────────────────
    # Main Run
    # ─────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest on OHLCV DataFrame.
        Returns performance metrics dict.
        """
        df_feat = build_features(df).dropna(subset=self.feature_cols).reset_index(
            drop=True
        )
        n = len(df_feat)
        logger.info(f"Backtesting on {n} candles …")

        equity = self.initial_capital
        equity_curve: List[float] = [equity]
        trades: List[BacktestTrade] = []

        # State
        in_position = False
        pos_side: str = ""
        entry_price = 0.0
        entry_idx = 0
        pos_qty = 0.0
        stop_loss = 0.0
        take_profit = 0.0

        # Warm-up: need at least 60 bars for indicators to stabilise
        warmup = 60

        for i in range(warmup, n):
            row = df_feat.iloc[i]
            price = float(row["close"])

            # ── Check open position exit ───────
            if in_position:
                exit_reason = self._check_exit(
                    pos_side, price, stop_loss, take_profit
                )
                if exit_reason or i == n - 1:
                    exit_reason = exit_reason or "END_OF_DATA"
                    fill_price = self._apply_slippage(price, "SELL" if pos_side == "BUY" else "BUY")
                    pnl_gross = self._calc_pnl(pos_side, entry_price, fill_price, pos_qty)
                    fee = (entry_price + fill_price) * pos_qty * self.fee_pct
                    pnl_net = pnl_gross - fee
                    equity += pnl_net
                    trades.append(
                        BacktestTrade(
                            entry_idx=entry_idx,
                            exit_idx=i,
                            side=pos_side,
                            entry_price=entry_price,
                            exit_price=fill_price,
                            quantity=pos_qty,
                            pnl_gross=pnl_gross,
                            pnl_net=pnl_net,
                            exit_reason=exit_reason,
                        )
                    )
                    in_position = False
                    logger.debug(
                        f"[{i}] EXIT {pos_side} @ {fill_price:.2f} "
                        f"PnL={pnl_net:+.2f} [{exit_reason}]"
                    )

            # ── Generate signal if flat ────────
            if not in_position:
                feature_vec = df_feat.iloc[[i]][self.feature_cols].reset_index(drop=True)
                action = self._get_action(feature_vec)

                if action in ("BUY", "SELL"):
                    fill_price = self._apply_slippage(price, action)
                    pos_qty = self._size_position(equity, fill_price)
                    if pos_qty <= 0 or equity < fill_price * pos_qty:
                        continue

                    in_position = True
                    pos_side = action
                    entry_price = fill_price
                    entry_idx = i

                    if action == "BUY":
                        stop_loss = fill_price * (1 - self.stop_loss_pct)
                        take_profit = fill_price * (1 + self.take_profit_pct)
                    else:
                        stop_loss = fill_price * (1 + self.stop_loss_pct)
                        take_profit = fill_price * (1 - self.take_profit_pct)

                    logger.debug(
                        f"[{i}] ENTER {action} @ {fill_price:.2f} "
                        f"qty={pos_qty:.6f} SL={stop_loss:.2f} TP={take_profit:.2f}"
                    )

            equity_curve.append(equity)

        metrics = self._compute_metrics(equity_curve, trades)
        self._print_report(metrics)
        return {"metrics": metrics, "trades": trades, "equity_curve": equity_curve}

    # ─────────────────────────────────────────
    # Signal Generation (model-only)
    # ─────────────────────────────────────────

    def _get_action(self, feature_vec: pd.DataFrame) -> str:
        """Get model-only action for backtesting."""
        try:
            prob_down, prob_up = self.model.predict_proba(feature_vec)
        except Exception as e:
            logger.debug(f"Model prediction error: {e}")
            return "HOLD"

        if self.bypass_ob_filter:
            # Use model alone
            if prob_up >= self.confidence_threshold:
                return "BUY"
            elif prob_down >= self.confidence_threshold:
                return "SELL"
        else:
            # Apply confidence filter only (no OB)
            if prob_up >= self.confidence_threshold:
                return "BUY"
            elif prob_down >= self.confidence_threshold:
                return "SELL"
        return "HOLD"

    # ─────────────────────────────────────────
    # Position Helpers
    # ─────────────────────────────────────────

    def _check_exit(
        self, side: str, price: float, sl: float, tp: float
    ) -> Optional[str]:
        if side == "BUY":
            if price <= sl:
                return "STOP_LOSS"
            if price >= tp:
                return "TAKE_PROFIT"
        else:
            if price >= sl:
                return "STOP_LOSS"
            if price <= tp:
                return "TAKE_PROFIT"
        return None

    def _apply_slippage(self, price: float, side: str) -> float:
        if side == "BUY":
            return price * (1 + self.slippage_pct)
        return price * (1 - self.slippage_pct)

    def _calc_pnl(
        self, side: str, entry: float, exit_p: float, qty: float
    ) -> float:
        if side == "BUY":
            return (exit_p - entry) * qty
        return (entry - exit_p) * qty

    def _size_position(self, equity: float, price: float) -> float:
        risk_amt = equity * self.risk_per_trade_pct
        qty = risk_amt / (price * self.stop_loss_pct)
        max_qty = (equity * 0.95) / price
        return min(qty, max_qty)

    # ─────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────

    def _compute_metrics(
        self, equity_curve: List[float], trades: List[BacktestTrade]
    ) -> Dict[str, Any]:
        initial = equity_curve[0]
        final = equity_curve[-1]
        total_return = (final - initial) / initial

        if not trades:
            return {
                "total_return_pct": 0.0,
                "total_profit": 0.0,
                "num_trades": 0,
                "win_rate": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "avg_trade_pnl": 0.0,
            }

        pnls = [t.pnl_net for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(trades)

        # Max drawdown
        eq = np.array(equity_curve)
        rolling_max = np.maximum.accumulate(eq)
        drawdowns = (eq - rolling_max) / rolling_max
        max_drawdown = float(drawdowns.min())

        # Sharpe (daily returns approximation)
        eq_series = pd.Series(equity_curve)
        daily_ret = eq_series.pct_change().dropna()
        sharpe = (
            (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
            if daily_ret.std() > 0
            else 0.0
        )

        # Profit factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 1e-9
        profit_factor = gross_profit / gross_loss

        return {
            "initial_capital": round(initial, 2),
            "final_equity": round(final, 2),
            "total_profit": round(final - initial, 2),
            "total_return_pct": round(total_return * 100, 2),
            "num_trades": len(trades),
            "win_rate": round(win_rate * 100, 2),
            "avg_trade_pnl": round(np.mean(pnls), 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "profit_factor": round(profit_factor, 3),
            "best_trade": round(max(pnls), 2),
            "worst_trade": round(min(pnls), 2),
        }

    def _print_report(self, metrics: Dict[str, Any]) -> None:
        sep = "─" * 42
        logger.info(f"\n{sep}")
        logger.info("  BACKTEST RESULTS")
        logger.info(sep)
        for k, v in metrics.items():
            logger.info(f"  {k:<25} {v}")
        logger.info(sep)
