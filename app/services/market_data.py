"""Market data helpers used by routes, historical simulations, and strategies."""

from __future__ import annotations

import threading
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
    "4h": timedelta(hours=4),
}


class MarketDataService:
    """Provides normalized market data for monitoring and simulation."""

    def __init__(self, config: dict[str, Any], client: HyperliquidClient) -> None:
        self.config = config
        self.client = client
        self._cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
        self._failure_backoff: dict[tuple[Any, ...], tuple[float, str]] = {}
        self._inflight: dict[tuple[Any, ...], threading.Event] = {}
        self._cache_guard = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._stale_serves = 0

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        mode: str = "testnet",
        limit: int | None = None,
        retry: bool = True,
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
        backoff_error = self._failure_backoff_error(cache_key, mode)
        if backoff_error:
            stale = self._cache_get_stale(cache_key, mode)
            if stale is not None:
                self._stale_serves += 1
                return [dict(row) for row in stale]
            raise RuntimeError(backoff_error)
        is_owner, inflight = self._claim_inflight(cache_key)
        if not is_owner:
            inflight.wait(timeout=2.0)
            cached = self._cache_get(cache_key, mode)
            if cached is not None:
                return [dict(row) for row in cached]
        interval = TIMEFRAME_TO_DELTA[timeframe]

        end = datetime.now(timezone.utc)
        start = end - (interval * candle_limit)

        try:
            rows = self._client_get_candles(
                mode,
                symbol,
                timeframe,
                int(start.timestamp() * 1000),
                int(end.timestamp() * 1000),
                retry=retry and str(mode or "").lower() != "live",
            )
        except Exception as exc:  # noqa: BLE001
            stale = self._cache_get_stale(cache_key, mode)
            if stale is not None and self._looks_transient_provider_error(exc):
                self._stale_serves += 1
                return [dict(row) for row in stale]
            self._record_failure_backoff(cache_key, mode, exc)
            raise
        finally:
            if is_owner:
                self._release_inflight(cache_key)

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
        backoff_error = self._failure_backoff_error(cache_key, mode)
        if backoff_error:
            stale = self._cache_get_stale(cache_key, mode)
            if stale is not None:
                self._stale_serves += 1
                return dict(stale)
            raise RuntimeError(backoff_error)
        is_owner, inflight = self._claim_inflight(cache_key)
        if not is_owner:
            inflight.wait(timeout=1.5)
            cached = self._cache_get(cache_key, mode)
            if cached is not None:
                return dict(cached)
        try:
            mids = self._client_get_all_mids(mode, retry=str(mode or "").lower() != "live")
        except Exception as exc:  # noqa: BLE001
            stale = self._cache_get_stale(cache_key, mode)
            if stale is not None and self._looks_transient_provider_error(exc):
                self._stale_serves += 1
                return dict(stale)
            self._record_failure_backoff(cache_key, mode, exc)
            raise
        finally:
            if is_owner:
                self._release_inflight(cache_key)
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

    def get_order_book(self, symbol: str, mode: str, retry: bool = True) -> dict[str, Any]:
        self._validate_symbol(symbol)
        cache_key = ("order_book", str(mode or "testnet"), symbol.upper())
        cached = self._cache_get(cache_key, mode)
        if cached is not None:
            return dict(cached)
        backoff_error = self._failure_backoff_error(cache_key, mode)
        if backoff_error:
            stale = self._cache_get_stale(cache_key, mode)
            if stale is not None:
                self._stale_serves += 1
                return dict(stale)
            raise RuntimeError(backoff_error)
        is_owner, inflight = self._claim_inflight(cache_key)
        if not is_owner:
            inflight.wait(timeout=1.5)
            cached = self._cache_get(cache_key, mode)
            if cached is not None:
                return dict(cached)
        try:
            book = dict(self._client_get_order_book(mode, symbol, retry=retry and str(mode or "").lower() != "live") or {})
        except Exception as exc:  # noqa: BLE001
            stale = self._cache_get_stale(cache_key, mode)
            if stale is not None and self._looks_transient_provider_error(exc):
                self._stale_serves += 1
                return dict(stale)
            self._record_failure_backoff(cache_key, mode, exc)
            raise
        finally:
            if is_owner:
                self._release_inflight(cache_key)
        self._cache_set(cache_key, book, mode)
        return dict(book)

    def cache_stats(self) -> dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        with self._cache_guard:
            entries = len(self._cache)
            failure_backoffs = sum(1 for until, _ in self._failure_backoff.values() if until > time.time())
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "entries": entries,
            "hit_rate": self._cache_hits / total if total else 0.0,
            "stale_serves": self._stale_serves,
            "failure_backoffs": failure_backoffs,
        }

    def clear_cache(self) -> None:
        with self._cache_guard:
            self._cache.clear()
            self._failure_backoff.clear()
            inflight = list(self._inflight.values())
            self._inflight.clear()
        for event in inflight:
            event.set()
        self._cache_hits = 0
        self._cache_misses = 0
        self._stale_serves = 0

    def _client_get_candles(
        self,
        mode: str,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
        *,
        retry: bool,
    ) -> list[dict[str, Any]]:
        try:
            return self.client.get_candles(mode, symbol, timeframe, start_ms, end_ms, retry=retry)
        except TypeError as exc:
            if not self._looks_like_retry_kwarg_error(exc):
                raise
            return self.client.get_candles(mode, symbol, timeframe, start_ms, end_ms)

    def _client_get_order_book(self, mode: str, symbol: str, *, retry: bool) -> dict[str, Any]:
        try:
            return self.client.get_order_book(mode, symbol, retry=retry)
        except TypeError as exc:
            if not self._looks_like_retry_kwarg_error(exc):
                raise
            return self.client.get_order_book(mode, symbol)

    def _client_get_all_mids(self, mode: str, *, retry: bool) -> dict[str, Any]:
        try:
            return self.client.get_all_mids(mode, retry=retry)
        except TypeError as exc:
            if not self._looks_like_retry_kwarg_error(exc):
                raise
            return self.client.get_all_mids(mode)

    def _resolve_limit(self, limit: int | None) -> int:
        default_limit = self._safe_int(self.config.get("DASHBOARD_CANDLE_LIMIT"), 200)
        resolved = limit if limit is not None else default_limit
        return max(1, min(int(resolved), 5_000))

    def _cache_get(self, key: tuple[Any, ...], mode: str) -> Any | None:
        ttl = self._cache_ttl(key, mode)
        if ttl <= 0:
            self._cache_misses += 1
            return None
        with self._cache_guard:
            cached_at, value = self._cache.get(key, (0.0, None))
        if value is not None and time.time() - cached_at < ttl:
            self._cache_hits += 1
            return value
        self._cache_misses += 1
        return None

    def _cache_get_stale(self, key: tuple[Any, ...], mode: str) -> Any | None:
        stale_ttl = self._stale_cache_ttl(mode)
        if stale_ttl <= 0:
            return None
        with self._cache_guard:
            cached_at, value = self._cache.get(key, (0.0, None))
        if value is not None and time.time() - cached_at < stale_ttl:
            return value
        return None

    def _cache_set(self, key: tuple[Any, ...], value: Any, mode: str) -> None:
        if self._cache_ttl(key, mode) <= 0:
            return
        with self._cache_guard:
            self._cache[key] = (time.time(), value)
            self._failure_backoff.pop(key, None)

    def _failure_backoff_error(self, key: tuple[Any, ...], mode: str) -> str | None:
        if str(mode or "").lower() != "live":
            return None
        with self._cache_guard:
            until, message = self._failure_backoff.get(key, (0.0, ""))
        if until <= time.time():
            with self._cache_guard:
                self._failure_backoff.pop(key, None)
            return None
        return message or "Live market data temporarily unavailable; retrying after provider backoff."

    def _record_failure_backoff(self, key: tuple[Any, ...], mode: str, exc: Exception) -> None:
        if str(mode or "").lower() != "live" or not self._looks_transient_provider_error(exc):
            return
        with self._cache_guard:
            self._failure_backoff[key] = (
                time.time() + self._failure_backoff_seconds(),
                "Live market data temporarily unavailable; retrying after provider backoff.",
            )

    def _claim_inflight(self, key: tuple[Any, ...]) -> tuple[bool, threading.Event]:
        with self._cache_guard:
            event = self._inflight.get(key)
            if event is None:
                event = threading.Event()
                self._inflight[key] = event
                return True, event
            return False, event

    def _release_inflight(self, key: tuple[Any, ...]) -> None:
        with self._cache_guard:
            event = self._inflight.pop(key, None)
        if event is not None:
            event.set()

    def _failure_backoff_seconds(self) -> float:
        fallback = self._config_float("ONE_H10_MARKET_DATA_BACKOFF_SECONDS", 30.0)
        return max(1.0, self._config_float("MARKET_DATA_LIVE_FAILURE_BACKOFF_SECONDS", fallback))

    def _cache_ttl(self, cache_key: tuple[Any, ...], mode: str) -> float:
        mode_key = str(mode or "").lower()
        category = str(cache_key[0] if cache_key else "")
        if mode_key == "live":
            fallback = self._config_float("MARKET_DATA_CACHE_TTL_SECONDS", 5.0)
            if category == "candles":
                return self._config_float("MARKET_DATA_LIVE_CANDLE_CACHE_SECONDS", max(fallback, 55.0))
            if category == "order_book":
                return self._config_float("MARKET_DATA_LIVE_ORDER_BOOK_CACHE_SECONDS", fallback)
            if category == "mids":
                return self._config_float("MARKET_DATA_LIVE_MIDS_CACHE_SECONDS", fallback)
            return fallback
        return self._config_float("MARKET_DATA_RESEARCH_CACHE_TTL_SECONDS", 60.0)

    def _stale_cache_ttl(self, mode: str) -> float:
        key = "MARKET_DATA_LIVE_STALE_SECONDS" if str(mode or "").lower() == "live" else "MARKET_DATA_RESEARCH_STALE_SECONDS"
        return self._config_float(key, 300.0)

    def _config_float(self, key: str, default: float) -> float:
        try:
            return max(0.0, float(self.config.get(key, default) or 0.0))
        except (TypeError, ValueError):
            return max(0.0, default)

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
    def _looks_like_retry_kwarg_error(exc: TypeError) -> bool:
        text = str(exc)
        return "retry" in text and ("unexpected keyword" in text or "got an unexpected" in text)

    @staticmethod
    def _looks_transient_provider_error(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc!r} {exc}".lower()
        return any(
            marker in text
            for marker in (
                "429",
                "rate limit",
                "too many requests",
                "timeout",
                "timed out",
                "cloudfront",
                "connection reset",
                "connection aborted",
                "failed to resolve",
                "nameresolutionerror",
                "temporarily unavailable",
            )
        )

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
