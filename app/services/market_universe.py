"""Dynamic liquid-market discovery for optimizer and vault selection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .market_data import MarketDataService
from .tradability import (
    book_liquidity_usd,
    cost_drag_bps,
    level_price_size,
    market_structure_score,
    safe_float,
    spread_bps,
    volatility_pct,
    volatility_regime,
)


@dataclass(frozen=True, slots=True)
class UniverseCandidate:
    symbol: str
    mid: float
    spread_bps: float
    liquidity_usd: float
    volatility_pct: float
    candle_count: int
    score: float
    source: str = "dynamic_liquid"
    cost_drag_bps: float = 0.0
    market_structure_score: float = 0.0
    volatility_regime: str = "unknown"
    rejection_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "mid": self.mid,
            "spread_bps": self.spread_bps,
            "liquidity_usd": self.liquidity_usd,
            "volatility_pct": self.volatility_pct,
            "candle_count": self.candle_count,
            "score": self.score,
            "source": self.source,
            "cost_drag_bps": self.cost_drag_bps,
            "market_structure_score": self.market_structure_score,
            "volatility_regime": self.volatility_regime,
            "rejection_reason": self.rejection_reason,
        }


class MarketUniverseService:
    """Builds a tradable universe from exchange-discoverable pairs."""

    def __init__(self, config: dict[str, Any], market_data: MarketDataService) -> None:
        self.config = config
        self.market_data = market_data
        self._cache: dict[tuple[str, str], tuple[float, list[UniverseCandidate]]] = {}
        self._rejection_cache: dict[tuple[str, str], dict[str, int]] = {}
        self.last_rejections: dict[str, int] = {}

    def liquid_universe(self, mode: str = "testnet", timeframe: str = "5m") -> list[UniverseCandidate]:
        cache_key = (str(mode or "testnet"), str(timeframe or "5m"))
        cached_at, cached = self._cache.get(cache_key, (0.0, []))
        ttl = max(0, int(self.config.get("UNIVERSE_REFRESH_SECONDS", 300)))
        if cached and ttl and (time.time() - cached_at) < ttl:
            self.last_rejections = dict(self._rejection_cache.get(cache_key, {}))
            return list(cached)

        self.last_rejections = {}
        discovered = self._discover(cache_key[0], cache_key[1])
        if not discovered:
            discovered = self._configured_fallback(cache_key[0], cache_key[1])

        self._cache[cache_key] = (time.time(), discovered)
        self._rejection_cache[cache_key] = dict(self.last_rejections)
        return list(discovered)

    def symbols(self, mode: str = "testnet", timeframe: str = "5m") -> list[str]:
        return [candidate.symbol for candidate in self.liquid_universe(mode, timeframe)]

    def _discover(self, mode: str, timeframe: str) -> list[UniverseCandidate]:
        if not bool(self.config.get("DYNAMIC_UNIVERSE_ENABLED", False)):
            return []

        try:
            mids = self.market_data.client.get_all_mids(mode)
        except Exception:  # noqa: BLE001
            self._reject("mids_unavailable")
            return []

        blacklist = {str(item).upper() for item in self.config.get("UNIVERSE_SYMBOL_BLACKLIST", [])}
        candidates: list[UniverseCandidate] = []

        for symbol, mid in mids.items():
            symbol = str(symbol).upper()
            mid = safe_float(mid)
            if not symbol or symbol in blacklist or mid <= 0:
                if symbol in blacklist:
                    self._reject("blacklisted")
                continue
            candidate = self._candidate(symbol, mode, timeframe, mid, source="dynamic_liquid")
            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: max(1, int(self.config.get("UNIVERSE_MAX_SYMBOLS", 20)))]

    def _configured_fallback(self, mode: str, timeframe: str) -> list[UniverseCandidate]:
        candidates: list[UniverseCandidate] = []
        for symbol in self.config.get("ALLOWED_SYMBOLS", ["BTC"]):
            symbol = str(symbol).upper()
            try:
                mid = self.market_data.get_mid_price(symbol, mode)
            except Exception:  # noqa: BLE001
                mid = 0.0
            candidate = self._candidate(symbol, mode, timeframe, mid, source="configured")
            if candidate is not None:
                candidates.append(candidate)
            elif symbol:
                candidates.append(
                    UniverseCandidate(
                        symbol=symbol,
                        mid=mid,
                        spread_bps=0.0,
                        liquidity_usd=0.0,
                        volatility_pct=0.0,
                        candle_count=0,
                        score=0.0,
                        source="configured",
                        rejection_reason="configured_fallback_no_market_data",
                    )
                )
        return candidates[: max(1, int(self.config.get("UNIVERSE_MAX_SYMBOLS", 20)))]

    def _candidate(self, symbol: str, mode: str, timeframe: str, mid: float, *, source: str) -> UniverseCandidate | None:
        try:
            book = self.market_data.get_order_book(symbol, mode)
            candles = self.market_data.get_candles(symbol, timeframe, mode=mode, limit=80)
        except Exception:  # noqa: BLE001
            return None

        spread = spread_bps(book, mid)
        liquidity_usd = book_liquidity_usd(book, depth=max(1, int(self.config.get("VAULT_BOOK_DEPTH_LEVELS", 5))))
        volatility = volatility_pct(candles)
        regime, volatility_score = volatility_regime(volatility)
        candle_count = len(candles)
        cost_drag = cost_drag_bps(
            spread=spread,
            fee_bps=float(self.config.get("FEE_BPS", 5.0) or 5.0),
            slippage_bps=float(self.config.get("SIM_SLIPPAGE_BPS", 8.0) or 8.0),
        )
        liquidity_score = min(liquidity_usd / max(float(self.config.get("UNIVERSE_MIN_LIQUIDITY_USD", 25_000.0) or 25_000.0) * 4, 1.0), 1.0)
        structure_score = market_structure_score(
            liquidity_usd=liquidity_usd,
            spread=spread,
            volatility_score=volatility_score,
            min_liquidity_usd=float(self.config.get("UNIVERSE_MIN_LIQUIDITY_USD", 25_000.0) or 25_000.0),
            max_spread_bps=float(self.config.get("UNIVERSE_MAX_SPREAD_BPS", 15.0) or 15.0),
        )

        if source == "dynamic_liquid":
            if spread <= 0 or spread > float(self.config.get("UNIVERSE_MAX_SPREAD_BPS", 15.0)):
                self._reject("spread_above_threshold")
                return None
            if liquidity_usd < float(self.config.get("UNIVERSE_MIN_LIQUIDITY_USD", 25_000.0)):
                self._reject("liquidity_below_threshold")
                return None
            if candle_count < 30:
                self._reject("insufficient_history")
                return None

        score = (liquidity_score * 2.5) + (volatility_score * 1.4) + min(volatility, 2.0) - (cost_drag / 40)
        return UniverseCandidate(
            symbol=symbol,
            mid=mid,
            spread_bps=spread,
            liquidity_usd=liquidity_usd,
            volatility_pct=volatility,
            candle_count=candle_count,
            score=score,
            source=source,
            cost_drag_bps=cost_drag,
            market_structure_score=structure_score,
            volatility_regime=regime,
        )

    def _reject(self, reason: str) -> None:
        self.last_rejections[reason] = self.last_rejections.get(reason, 0) + 1

    def _spread_bps(self, book: dict[str, Any], mid: float) -> float:
        return spread_bps(book, mid)

    def _liquidity_usd(self, book: dict[str, Any]) -> float:
        return book_liquidity_usd(book, depth=max(1, int(self.config.get("VAULT_BOOK_DEPTH_LEVELS", 5))))

    _volatility_pct = staticmethod(volatility_pct)
    _volatility_regime = staticmethod(volatility_regime)
    _level_price_size = staticmethod(level_price_size)
    _safe_float = staticmethod(safe_float)
