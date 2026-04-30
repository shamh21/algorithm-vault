"""Market data helpers used by routes, historical simulations, and strategies."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any

from .hyperliquid_client import HyperliquidClient


TIMEFRAME_TO_DELTA = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
}


class MarketDataService:
    """Provides normalized market data for monitoring and simulation."""

    def __init__(self, config: dict[str, Any], client: HyperliquidClient) -> None:
        self.config = config
        self.client = client
        self._cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        mode: str = "testnet",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self._validate_symbol(symbol)

        if timeframe not in TIMEFRAME_TO_DELTA:
            supported = ", ".join(TIMEFRAME_TO_DELTA)
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: {supported}")

        candle_limit = self._resolve_limit(limit)
        cache_key = ("candles", str(mode or "testnet"), symbol.upper(), timeframe, candle_limit)
        cached = self._cache_get(cache_key, mode)
        if cached is not None:
            return [dict(row) for row in cached]
        interval = TIMEFRAME_TO_DELTA[timeframe]

        end = datetime.now(timezone.utc)
        start = end - (interval * candle_limit)

        rows = self.client.get_candles(
            mode,
            symbol,
            timeframe,
            int(start.timestamp() * 1000),
            int(end.timestamp() * 1000),
        )

        candles = [self._normalize_candle(row, timeframe) for row in rows]
        candles = [candle for candle in candles if candle is not None]

        result = sorted(candles, key=lambda candle: candle["timestamp"])[-candle_limit:]
        self._cache_set(cache_key, result, mode)
        return [dict(row) for row in result]

    def get_mid_price(self, symbol: str, mode: str) -> float:
        self._validate_symbol(symbol)

        mids = self.get_all_mids(mode)
        return self._safe_float(mids.get(symbol))

    def get_all_mids(self, mode: str) -> dict[str, Any]:
        cache_key = ("mids", str(mode or "testnet"))
        cached = self._cache_get(cache_key, mode)
        if cached is not None:
            return dict(cached)
        mids = self.client.get_all_mids(mode)
        result = dict(mids or {})
        self._cache_set(cache_key, result, mode)
        return dict(result)

    def get_dashboard_market_summary(
        self,
        symbols: list[str],
        timeframe: str,
        mode: str,
    ) -> list[dict[str, Any]]:
        mids = self.get_all_mids(mode)
        summary: list[dict[str, Any]] = []

        for symbol in symbols:
            try:
                self._validate_symbol(symbol)

                candles = self.get_candles(symbol, timeframe, mode=mode, limit=30)
                closes = [candle["close"] for candle in candles if candle["close"] > 0][-5:]
                mid = self._safe_float(mids.get(symbol))

                summary.append(
                    {
                        "symbol": symbol,
                        "mid": mid,
                        "recent_average": mean(closes) if closes else 0.0,
                        "change_pct": self._change_pct(candles),
                        "candle_count": len(candles),
                        "status": "ok",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                summary.append(
                    {
                        "symbol": symbol,
                        "mid": self._safe_float(mids.get(symbol)),
                        "recent_average": 0.0,
                        "change_pct": 0.0,
                        "candle_count": 0,
                        "status": "error",
                        "error": str(exc),
                    }
                )

        return summary

    def get_order_book(self, symbol: str, mode: str) -> dict[str, Any]:
        self._validate_symbol(symbol)
        cache_key = ("order_book", str(mode or "testnet"), symbol.upper())
        cached = self._cache_get(cache_key, mode)
        if cached is not None:
            return dict(cached)
        book = dict(self.client.get_order_book(mode, symbol) or {})
        self._cache_set(cache_key, book, mode)
        return dict(book)

    def cache_stats(self) -> dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "entries": len(self._cache),
            "hit_rate": self._cache_hits / total if total else 0.0,
        }

    def clear_cache(self) -> None:
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def _resolve_limit(self, limit: int | None) -> int:
        default_limit = self._safe_int(self.config.get("DASHBOARD_CANDLE_LIMIT"), 200)
        resolved = limit if limit is not None else default_limit
        return max(1, min(int(resolved), 5_000))

    def _cache_get(self, key: tuple[Any, ...], mode: str) -> Any | None:
        ttl = self._cache_ttl(mode)
        if ttl <= 0:
            self._cache_misses += 1
            return None
        cached_at, value = self._cache.get(key, (0.0, None))
        if value is not None and time.time() - cached_at < ttl:
            self._cache_hits += 1
            return value
        self._cache_misses += 1
        return None

    def _cache_set(self, key: tuple[Any, ...], value: Any, mode: str) -> None:
        if self._cache_ttl(mode) <= 0:
            return
        self._cache[key] = (time.time(), value)

    def _cache_ttl(self, mode: str) -> float:
        key = "MARKET_DATA_CACHE_TTL_SECONDS" if str(mode or "").lower() == "live" else "MARKET_DATA_RESEARCH_CACHE_TTL_SECONDS"
        try:
            return max(0.0, float(self.config.get(key, 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _normalize_candle(row: dict[str, Any], timeframe: str) -> dict[str, Any] | None:
        timestamp = MarketDataService._safe_int(row.get("t"))
        open_price = MarketDataService._safe_float(row.get("o"))
        high = MarketDataService._safe_float(row.get("h"))
        low = MarketDataService._safe_float(row.get("l"))
        close = MarketDataService._safe_float(row.get("c"))
        volume = MarketDataService._safe_float(row.get("v"))

        if timestamp <= 0 or close <= 0:
            return None

        return {
            "timestamp": timestamp,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "interval": row.get("i", timeframe),
        }

    @staticmethod
    def _change_pct(candles: list[dict[str, Any]]) -> float:
        valid = [candle for candle in candles if candle.get("close", 0.0) > 0]

        if len(valid) < 2:
            return 0.0

        start = float(valid[0]["close"])
        end = float(valid[-1]["close"])

        if start <= 0:
            return 0.0

        return ((end - start) / start) * 100

    @staticmethod
    def _validate_symbol(symbol: str) -> None:
        if not symbol or not isinstance(symbol, str):
            raise ValueError("symbol must be a non-empty string")

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        if value is None:
            return default

        try:
            return int(value)
        except (TypeError, ValueError):
            return default
