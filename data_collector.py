"""
data_collector.py — Fetch historical OHLCV from Binance REST API
and stream real-time order book + trade data via WebSocket.
"""

import json
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests
import websocket  # websocket-client library

from utils import get_logger, load_config, ms_to_datetime

logger = get_logger("DataCollector")


# ─────────────────────────────────────────────
# REST — Historical OHLCV
# ─────────────────────────────────────────────

class BinanceRESTClient:
    """Thin wrapper around Binance public REST endpoints."""

    BASE_URL = "https://api.binance.com"

    def __init__(self, base_url: str = None):
        self.base_url = base_url or self.BASE_URL
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles from Binance.

        Returns a DataFrame with columns:
            timestamp, open, high, low, close, volume
        """
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        url = f"{self.base_url}/api/v3/klines"
        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        df = pd.DataFrame(
            raw,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "num_trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ],
        )

        df["timestamp"] = df["open_time"].apply(ms_to_datetime)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info(f"Fetched {len(df)} candles for {symbol} [{interval}]")
        return df

    def get_order_book_snapshot(self, symbol: str, depth: int = 20) -> Dict:
        """
        Fetch current order book snapshot.

        Returns dict with keys: 'bids', 'asks'
        Each is a list of [price, quantity] pairs (floats).
        """
        url = f"{self.base_url}/api/v3/depth"
        params = {"symbol": symbol.upper(), "limit": depth}
        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        return {
            "bids": [[float(p), float(q)] for p, q in raw["bids"]],
            "asks": [[float(p), float(q)] for p, q in raw["asks"]],
            "lastUpdateId": raw["lastUpdateId"],
        }

    def get_ticker_price(self, symbol: str) -> float:
        """Return latest best ask price for symbol."""
        url = f"{self.base_url}/api/v3/ticker/price"
        resp = self.session.get(url, params={"symbol": symbol.upper()}, timeout=5)
        resp.raise_for_status()
        return float(resp.json()["price"])


# ─────────────────────────────────────────────
# WebSocket — Real-Time Streams
# ─────────────────────────────────────────────

class OrderBookStream:
    """
    Maintains a local order book using Binance diff depth stream.
    Thread-safe snapshot available at any time via `.snapshot()`.
    """

    def __init__(self, symbol: str, depth: int = 20, ws_base: str = None):
        self.symbol = symbol.lower()
        self.depth = depth
        self.ws_base = ws_base or "wss://stream.binance.com:9443"
        self._bids: Dict[float, float] = {}   # price -> qty
        self._asks: Dict[float, float] = {}
        self._lock = threading.Lock()
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_update_id = 0

    # ── public interface ──────────────────────

    def start(self) -> None:
        """Start background WebSocket thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"OrderBookStream started for {self.symbol.upper()}")

    def stop(self) -> None:
        self._running = False
        if self._ws:
            self._ws.close()

    def snapshot(self) -> Dict[str, List]:
        """
        Return sorted top-N bids and asks as [[price, qty], …].
        Bids descending, asks ascending.
        """
        with self._lock:
            bids = sorted(self._bids.items(), key=lambda x: -x[0])[: self.depth]
            asks = sorted(self._asks.items(), key=lambda x: x[0])[: self.depth]
        return {
            "bids": [[p, q] for p, q in bids],
            "asks": [[p, q] for p, q in asks],
        }

    # ── internals ─────────────────────────────

    def _run(self) -> None:
        stream = f"{self.ws_base}/ws/{self.symbol}@depth@100ms"
        self._ws = websocket.WebSocketApp(
            stream,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        while self._running:
            try:
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                logger.warning(f"OrderBookStream error: {e}. Reconnecting in 5s…")
                time.sleep(5)

    def _on_open(self, ws) -> None:
        logger.debug("OrderBook WebSocket connected")

    def _on_message(self, ws, message: str) -> None:
        data = json.loads(message)
        with self._lock:
            for price_str, qty_str in data.get("b", []):
                price, qty = float(price_str), float(qty_str)
                if qty == 0:
                    self._bids.pop(price, None)
                else:
                    self._bids[price] = qty
            for price_str, qty_str in data.get("a", []):
                price, qty = float(price_str), float(qty_str)
                if qty == 0:
                    self._asks.pop(price, None)
                else:
                    self._asks[price] = qty

    def _on_error(self, ws, error) -> None:
        logger.warning(f"OrderBook WS error: {error}")

    def _on_close(self, ws, code, msg) -> None:
        logger.debug(f"OrderBook WS closed: {code} {msg}")


class TradeStream:
    """
    Streams individual trades (aggTrade) and maintains a rolling
    window of recent trades for momentum analysis.
    """

    def __init__(
        self,
        symbol: str,
        window: int = 200,
        ws_base: str = None,
        on_trade: Optional[Callable] = None,
    ):
        self.symbol = symbol.lower()
        self.window = window
        self.ws_base = ws_base or "wss://stream.binance.com:9443"
        self.on_trade = on_trade  # optional callback
        self._trades: deque = deque(maxlen=window)
        self._lock = threading.Lock()
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"TradeStream started for {self.symbol.upper()}")

    def stop(self) -> None:
        self._running = False
        if self._ws:
            self._ws.close()

    def recent_trades(self) -> List[Dict]:
        """Return list of recent trade dicts (thread-safe copy)."""
        with self._lock:
            return list(self._trades)

    def _run(self) -> None:
        stream = f"{self.ws_base}/ws/{self.symbol}@aggTrade"
        self._ws = websocket.WebSocketApp(
            stream,
            on_message=self._on_message,
            on_error=lambda ws, e: logger.warning(f"TradeStream error: {e}"),
            on_close=lambda ws, c, m: logger.debug("TradeStream closed"),
        )
        while self._running:
            try:
                self._ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                logger.warning(f"TradeStream reconnecting after error: {e}")
                time.sleep(5)

    def _on_message(self, ws, message: str) -> None:
        data = json.loads(message)
        trade = {
            "price": float(data["p"]),
            "qty": float(data["q"]),
            "is_buyer_maker": data["m"],   # True = seller aggressor
            "time": data["T"],
        }
        with self._lock:
            self._trades.append(trade)
        if self.on_trade:
            self.on_trade(trade)


# ─────────────────────────────────────────────
# Convenience wrapper used by main.py
# ─────────────────────────────────────────────

class DataCollector:
    """
    High-level data access layer.
    Combines REST + streaming into one interface.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        ex_cfg = cfg["exchange"]
        trading_cfg = cfg["trading"]

        self.symbol = trading_cfg["symbol"]
        self.interval = trading_cfg["interval"]
        self.lookback = trading_cfg["lookback_candles"]
        self.depth = trading_cfg["order_book_depth"]

        self.rest = BinanceRESTClient(base_url=ex_cfg.get("base_url"))
        self.ob_stream = OrderBookStream(
            symbol=self.symbol,
            depth=self.depth,
            ws_base=ex_cfg.get("ws_url"),
        )
        self.trade_stream = TradeStream(
            symbol=self.symbol,
            ws_base=ex_cfg.get("ws_url"),
        )

    def start_streams(self) -> None:
        """Launch WebSocket threads."""
        self.ob_stream.start()
        self.trade_stream.start()
        time.sleep(2)  # allow initial subscription

    def stop_streams(self) -> None:
        self.ob_stream.stop()
        self.trade_stream.stop()

    def fetch_candles(self, limit: int = None) -> pd.DataFrame:
        """Fetch recent OHLCV candles via REST."""
        return self.rest.get_klines(
            symbol=self.symbol,
            interval=self.interval,
            limit=limit or self.lookback,
        )

    def get_order_book(self) -> Dict:
        """Return live order book snapshot from stream."""
        return self.ob_stream.snapshot()

    def get_recent_trades(self) -> List[Dict]:
        """Return recent trades from stream."""
        return self.trade_stream.recent_trades()

    def get_current_price(self) -> float:
        """Return latest market price via REST."""
        return self.rest.get_ticker_price(self.symbol)
