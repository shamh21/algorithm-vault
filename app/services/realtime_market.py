"""Realtime market snapshot support with HTTP fallback."""

from __future__ import annotations

import json
import time
from statistics import mean
from typing import Any


class RealtimeMarketService:
    """Caches Hyperliquid WebSocket feeds and falls back to HTTP snapshots."""

    def __init__(self, config: dict[str, Any], market_data: Any) -> None:
        self.config = config
        self.market_data = market_data
        self._mids: dict[str, float] = {}
        self._books: dict[str, dict[str, Any]] = {}
        self._trades: dict[str, list[dict[str, Any]]] = {}
        self._updated_at: dict[str, float] = {}

    def snapshot(self, symbol: str, mode: str, timeframe: str = "1m", user: str | None = None) -> dict[str, Any]:
        symbol = str(symbol or "").upper()
        if bool(self.config.get("REALTIME_MARKET_ENABLED", False)):
            cached = self._cached_snapshot(symbol, timeframe)
            if cached is not None:
                return cached
        return self._http_snapshot(symbol, mode, timeframe, user=user)

    def subscription_messages(self, symbol: str, user: str | None = None, dex: str | None = None) -> list[dict[str, Any]]:
        symbol = str(symbol or "").upper()
        all_mids: dict[str, Any] = {"type": "allMids"}
        if dex:
            all_mids["dex"] = dex
        subscriptions = [
            all_mids,
            {"type": "l2Book", "coin": symbol},
            {"type": "trades", "coin": symbol},
        ]
        if user:
            subscriptions.extend(
                [
                    {"type": "orderUpdates", "user": user},
                    {"type": "userEvents", "user": user},
                    {"type": "userFills", "user": user},
                ]
            )
        return [{"method": "subscribe", "subscription": subscription} for subscription in subscriptions]

    def connect_once(self, symbol: str, mode: str, user: str | None = None) -> bool:
        """Connect briefly to seed caches; callers can retry on their own schedule."""

        if not bool(self.config.get("REALTIME_MARKET_ENABLED", False)):
            return False
        try:
            import websocket  # type: ignore[import-not-found]
        except Exception:  # noqa: BLE001
            return False

        url = self._ws_url(mode)
        timeout = float(self.config.get("REALTIME_MARKET_CONNECT_TIMEOUT_SECONDS", 3.0) or 3.0)
        try:
            ws = websocket.create_connection(url, timeout=timeout)
            for message in self.subscription_messages(symbol, user=user):
                ws.send(json.dumps(message))
            deadline = time.time() + timeout
            while time.time() < deadline:
                self.ingest_message(ws.recv())
            ws.close()
        except Exception:  # noqa: BLE001
            return False
        return True

    def ingest_message(self, payload: str | dict[str, Any]) -> None:
        try:
            message = json.loads(payload) if isinstance(payload, str) else payload
        except json.JSONDecodeError:
            return
        if not isinstance(message, dict):
            return
        channel = str(message.get("channel") or "")
        data = message.get("data")
        now = time.time()

        if channel == "allMids" and isinstance(data, dict):
            mids = data.get("mids", data)
            if isinstance(mids, dict):
                for symbol, value in mids.items():
                    try:
                        self._mids[str(symbol).upper()] = float(value)
                        self._updated_at[f"mid:{str(symbol).upper()}"] = now
                    except (TypeError, ValueError):
                        continue
            return

        if channel == "l2Book" and isinstance(data, dict):
            symbol = str(data.get("coin") or "").upper()
            if symbol:
                self._books[symbol] = data
                self._updated_at[f"book:{symbol}"] = now
            return

        if channel == "trades" and isinstance(data, list):
            for trade in data:
                if not isinstance(trade, dict):
                    continue
                symbol = str(trade.get("coin") or "").upper()
                if not symbol:
                    continue
                bucket = self._trades.setdefault(symbol, [])
                bucket.append(trade)
                del bucket[:-50]
                self._updated_at[f"trades:{symbol}"] = now

    def _cached_snapshot(self, symbol: str, timeframe: str) -> dict[str, Any] | None:
        max_stale = float(self.config.get("REALTIME_MARKET_MAX_STALE_SECONDS", 15.0) or 15.0)
        now = time.time()
        book = self._books.get(symbol)
        mid = self._mids.get(symbol)
        if book is None or mid is None:
            return None
        if now - self._updated_at.get(f"book:{symbol}", 0.0) > max_stale:
            return None
        if now - self._updated_at.get(f"mid:{symbol}", 0.0) > max_stale:
            return None

        metrics = self._book_metrics(book)
        trades = list(self._trades.get(symbol, []))[-25:]
        volatility = self._trade_volatility_pct(trades)
        return {
            "source": "websocket",
            "symbol": symbol,
            "timeframe": timeframe,
            "mid": mid,
            "bid": metrics["bid"],
            "ask": metrics["ask"],
            "spread_bps": metrics["spread_bps"],
            "liquidity_usd": metrics["liquidity_usd"],
            "recent_trades": trades,
            "volatility_pct": volatility,
            "signal_stability": self._signal_stability(volatility, metrics["spread_bps"]),
            "is_realtime": True,
        }

    def _http_snapshot(self, symbol: str, mode: str, timeframe: str, user: str | None = None) -> dict[str, Any]:
        try:
            mid = float(self.market_data.get_mid_price(symbol, mode) or 0.0)
        except Exception:  # noqa: BLE001
            mid = 0.0
        try:
            book = self.market_data.get_order_book(symbol, mode)
        except Exception:  # noqa: BLE001
            book = {}
        try:
            candles = self.market_data.get_candles(symbol, timeframe, mode=mode, limit=32)
        except Exception:  # noqa: BLE001
            candles = []
        metrics = self._book_metrics(book)
        volatility = self._candle_volatility_pct(candles)
        return {
            "source": "http",
            "symbol": symbol,
            "timeframe": timeframe,
            "mid": mid,
            "bid": metrics["bid"],
            "ask": metrics["ask"],
            "spread_bps": metrics["spread_bps"],
            "liquidity_usd": metrics["liquidity_usd"],
            "recent_trades": [],
            "volatility_pct": volatility,
            "signal_stability": self._signal_stability(volatility, metrics["spread_bps"]),
            "is_realtime": False,
            "user": user,
        }

    def _book_metrics(self, book: dict[str, Any]) -> dict[str, float]:
        levels = book.get("levels", []) if isinstance(book, dict) else []
        bid = ask = liquidity = 0.0
        if isinstance(levels, list) and len(levels) >= 2:
            bid = self._level_price_size(levels[0][0])[0] if levels[0] else 0.0
            ask = self._level_price_size(levels[1][0])[0] if levels[1] else 0.0
            for side in levels[:2]:
                if not isinstance(side, list):
                    continue
                for row in side[:5]:
                    price, size = self._level_price_size(row)
                    liquidity += price * size
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        spread_bps = ((ask - bid) / mid) * 10_000 if mid > 0 and ask >= bid else 0.0
        return {"bid": bid, "ask": ask, "spread_bps": spread_bps, "liquidity_usd": liquidity}

    def _candle_volatility_pct(self, candles: list[dict[str, Any]]) -> float:
        closes = [self._safe_float(row.get("close", row.get("c"))) for row in candles if isinstance(row, dict)]
        closes = [value for value in closes if value > 0]
        if len(closes) < 3:
            return 0.0
        returns = [
            abs((closes[index] - closes[index - 1]) / closes[index - 1])
            for index in range(1, len(closes))
            if closes[index - 1] > 0
        ]
        return mean(returns) * 100 if returns else 0.0

    def _trade_volatility_pct(self, trades: list[dict[str, Any]]) -> float:
        prices = [self._safe_float(row.get("px")) for row in trades]
        prices = [value for value in prices if value > 0]
        if len(prices) < 3:
            return 0.0
        returns = [
            abs((prices[index] - prices[index - 1]) / prices[index - 1])
            for index in range(1, len(prices))
            if prices[index - 1] > 0
        ]
        return mean(returns) * 100 if returns else 0.0

    def _signal_stability(self, volatility_pct: float, spread_bps: float) -> float:
        spread_penalty = min(max(spread_bps, 0.0) / 100.0, 1.0)
        volatility_penalty = min(max(volatility_pct, 0.0) / 5.0, 1.0)
        return max(0.0, min(1.0, 1.0 - (spread_penalty * 0.45 + volatility_penalty * 0.55)))

    def _ws_url(self, mode: str) -> str:
        if str(mode or "").lower() == "live":
            return str(self.config.get("HL_WS_MAINNET_URL", "wss://api.hyperliquid.xyz/ws"))
        return str(self.config.get("HL_WS_TESTNET_URL", "wss://api.hyperliquid-testnet.xyz/ws"))

    def _level_price_size(self, row: Any) -> tuple[float, float]:
        if isinstance(row, dict):
            return self._safe_float(row.get("px")), self._safe_float(row.get("sz"))
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            return self._safe_float(row[0]), self._safe_float(row[1])
        return 0.0, 0.0

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
