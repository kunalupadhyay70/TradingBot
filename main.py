"""
main.py — Orchestrator for the crypto trading bot.

Workflow:
  1. Load config
  2. Train model on historical data (if model file not found)
  3. Run backtest (optional, prints metrics)
  4. Start WebSocket streams
  5. Enter real-time loop:
       → fetch latest candle
       → build features
       → get model prediction
       → analyze order book
       → generate signal
       → execute trade / check exits
       → repeat on next candle
"""

import os
import signal
import sys
import time
from typing import Any, Dict

from backtester import Backtester
from data_collector import DataCollector
from execution_engine import ExecutionEngine
from feature_engineering import get_latest_feature_vector, prepare_dataset
from model import DirectionModel
from orderbook_analyzer import OrderBookAnalyzer
from risk_manager import RiskManager
from signal_engine import SignalEngine
from utils import get_logger, load_config, sleep_until_next_candle

# ─────────────────────────────────────────────
# Global shutdown flag
# ─────────────────────────────────────────────
_RUNNING = True


def _handle_shutdown(sig, frame):
    global _RUNNING
    logger.info("Shutdown signal received. Stopping gracefully…")
    _RUNNING = False


# ─────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────

def build_components(cfg: Dict[str, Any]):
    """Instantiate all system components."""
    collector = DataCollector(cfg)
    model = DirectionModel(cfg)
    ob_analyzer = OrderBookAnalyzer(cfg)
    risk_mgr = RiskManager(cfg)
    signal_eng = SignalEngine(model, cfg)
    exec_eng = ExecutionEngine(risk_mgr, cfg)
    backtester = Backtester(model, cfg, bypass_ob_filter=True)
    return collector, model, ob_analyzer, risk_mgr, signal_eng, exec_eng, backtester


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_model(model: DirectionModel, collector: DataCollector, cfg: Dict) -> None:
    """Fetch historical data, build features, train and save model."""
    logger.info("=" * 55)
    logger.info("  TRAINING MODEL")
    logger.info("=" * 55)

    df = collector.fetch_candles(limit=cfg["trading"]["lookback_candles"])
    feature_cols = cfg["model"]["features"]
    lookahead = cfg["model"]["target_lookahead"]

    X, y = prepare_dataset(df, feature_cols, lookahead)
    metrics = model.train(X, y)
    model.save()

    logger.info(
        f"Training complete — Accuracy={metrics['accuracy']:.4f} "
        f"AUC={metrics['auc_roc']:.4f}"
    )


# ─────────────────────────────────────────────
# Backtest (optional)
# ─────────────────────────────────────────────

def run_backtest(backtester: Backtester, collector: DataCollector) -> None:
    """Run strategy on historical data and print metrics."""
    logger.info("=" * 55)
    logger.info("  RUNNING BACKTEST")
    logger.info("=" * 55)
    df = collector.fetch_candles()
    backtester.run(df)


# ─────────────────────────────────────────────
# Real-Time Trading Loop
# ─────────────────────────────────────────────

def trading_loop(
    collector: DataCollector,
    model: DirectionModel,
    ob_analyzer: OrderBookAnalyzer,
    signal_eng: SignalEngine,
    exec_eng: ExecutionEngine,
    cfg: Dict[str, Any],
) -> None:
    """
    Main real-time loop. Runs one iteration per closed 5-minute candle.
    """
    feature_cols = cfg["model"]["features"]
    interval_seconds = _interval_to_seconds(cfg["trading"]["interval"])

    logger.info("=" * 55)
    logger.info("  STARTING REAL-TIME TRADING LOOP")
    logger.info(f"  Symbol : {cfg['trading']['symbol']}")
    logger.info(f"  Interval: {cfg['trading']['interval']}")
    logger.info("=" * 55)

    # Kick off WebSocket streams
    collector.start_streams()
    logger.info("WebSocket streams started. Waiting for first candle close…")

    # Allow streams to warm up
    time.sleep(3)

    iteration = 0
    while _RUNNING:
        iteration += 1
        logger.info(f"\n{'─'*50}")
        logger.info(f"  Iteration #{iteration}")
        logger.info(f"{'─'*50}")

        try:
            # ── 1. Fetch latest candles ──────────────
            df = collector.fetch_candles(limit=200)

            # ── 2. Build live feature vector ─────────
            feature_vec = get_latest_feature_vector(df, feature_cols)
            if feature_vec.empty:
                logger.warning("Feature vector is empty — skipping iteration")
                _wait_for_next_candle(interval_seconds)
                continue

            # ── 3. Analyze order book ─────────────────
            ob_snapshot = collector.get_order_book()
            recent_trades = collector.get_recent_trades()
            ob_metrics = ob_analyzer.analyze(ob_snapshot, recent_trades)

            # ── 4. Generate signal ────────────────────
            signal = signal_eng.generate(feature_vec, ob_metrics)
            logger.info(f"Signal: {signal}")

            # ── 5. Check existing position exits ──────
            current_price = collector.get_current_price()
            exit_results = exec_eng.check_exits(current_price)
            for order, reason, pnl in exit_results:
                logger.info(
                    f"  └─ Position closed [{reason}] PnL={pnl:+.2f} "
                    f"Equity={exec_eng.risk.equity:.2f}"
                )

            # ── 6. Execute new signal ─────────────────
            if signal.action != "HOLD":
                order = exec_eng.execute(signal, current_price)
                if order:
                    logger.info(
                        f"  └─ Order executed: {order} "
                        f"Equity={exec_eng.risk.equity:.2f}"
                    )

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Loop error (iteration {iteration}): {e}", exc_info=True)

        # ── Wait for next candle ──────────────────
        _wait_for_next_candle(interval_seconds)

    # Cleanup
    logger.info("Stopping WebSocket streams…")
    collector.stop_streams()
    summary = exec_eng.performance_summary()
    logger.info(f"Session summary: {summary}")


def _interval_to_seconds(interval: str) -> int:
    """Convert Binance interval string to seconds (e.g. '5m' → 300)."""
    unit = interval[-1]
    value = int(interval[:-1])
    return {"m": 60, "h": 3600, "d": 86400}.get(unit, 60) * value


def _wait_for_next_candle(interval_seconds: int) -> None:
    """Sleep until just after the next candle boundary."""
    try:
        sleep_until_next_candle(interval_seconds)
    except Exception:
        time.sleep(interval_seconds)


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Load config ───────────────────────────
    cfg = load_config("config.yaml")
    logger = get_logger("Main", cfg)

    # Register graceful shutdown
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    # ── Build all components ──────────────────
    (
        collector,
        model,
        ob_analyzer,
        risk_mgr,
        signal_eng,
        exec_eng,
        backtester,
    ) = build_components(cfg)

    # ── Train model if not on disk ────────────
    model_loaded = model.load()
    if not model_loaded:
        train_model(model, collector, cfg)
    else:
        logger.info("Loaded pre-trained model from disk.")

    # ── Optionally run backtest ───────────────
    run_backtest_flag = "--backtest" in sys.argv or os.getenv("RUN_BACKTEST") == "1"
    if run_backtest_flag:
        run_backtest(backtester, collector)

    # ── Start trading ─────────────────────────
    trading_loop(collector, model, ob_analyzer, signal_eng, exec_eng, cfg)
