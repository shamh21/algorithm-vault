"""Persistent leveraged-market discovery and 1H10 feature snapshots."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable

from sqlalchemy.exc import IntegrityError

from ..extensions import db
from ..models import LeveragedMarket, LeveragedMarketFeature, Setting, TradingConnection
from .provider_assets import normalize_provider, provider_collateral_asset


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    provider: str
    trading_connection_id: int | None
    active: int
    disabled: int
    features_attempted: int = 0
    features_skipped: int = 0
    feature_skip_reasons: tuple[str, ...] = ()
    feature_cursor: int = 0
    next_feature_cursor: int = 0
    skipped: bool = False
    reason: str = ""


@dataclass(slots=True)
class _FeatureSyncBudget:
    max_markets: int
    attempted: int = 0
    skipped: int = 0
    reasons: list[str] = field(default_factory=list)

    def has_capacity(self) -> bool:
        return self.max_markets > 0 and self.attempted < self.max_markets

    def add_skip(self, reason: str) -> None:
        self.skipped += 1
        if reason and reason not in self.reasons:
            self.reasons.append(reason)


class LeveragedMarketDiscoveryService:
    """Sync provider futures/perpetual listings for allocation and ML features."""

    SUPPORTED_PROVIDERS = {"hyperliquid", "kucoin"}

    def __init__(self, config: dict[str, Any], market_data: Any, trading_connections: Any, feature_factory: Any) -> None:
        self.config = config
        self.market_data = market_data
        self.trading_connections = trading_connections
        self.feature_factory = feature_factory
        self._feature_symbol_backoff_until: dict[tuple[str, str], float] = {}
        self._feature_provider_backoff_until: dict[str, float] = {}

    def sync_for_user(
        self,
        user_id: int,
        mode: str = "live",
        *,
        feature_symbols: Iterable[str] | None = None,
        feature_scope: str = "allowed",
        persist_features: bool = True,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        feature_budget = _FeatureSyncBudget(self._feature_max_markets_per_sync()) if persist_features else None
        scoped_symbols = self._feature_symbol_set(feature_symbols, feature_scope=feature_scope) if persist_features else set()
        try:
            connections = self.trading_connections.verified_tradable_connections(user_id)
        except AttributeError:
            connections = []
        for connection in connections:
            result = self.sync_for_connection(
                connection,
                mode=mode,
                feature_symbols=scoped_symbols,
                feature_scope=feature_scope,
                persist_features=persist_features,
                feature_budget=feature_budget,
            )
            results.append(asdict(result))
        return results

    def sync_one_h10_backfill_for_user(self, user_id: int, mode: str = "live") -> list[dict[str, Any]]:
        """Sync all active leveraged markets and backfill a rate-limited feature batch."""

        return self.sync_for_user(user_id, mode=mode, feature_scope="all", persist_features=True)

    def sync_for_connection(
        self,
        connection: TradingConnection,
        mode: str = "live",
        *,
        feature_symbols: Iterable[str] | None = None,
        feature_scope: str = "allowed",
        persist_features: bool = True,
        feature_budget: _FeatureSyncBudget | None = None,
    ) -> DiscoveryResult:
        provider = normalize_provider(connection.provider)
        if provider not in self.SUPPORTED_PROVIDERS:
            return DiscoveryResult(provider, connection.id, 0, 0, skipped=True, reason="provider_discovery_not_implemented")
        budget = feature_budget or (_FeatureSyncBudget(self._feature_max_markets_per_sync()) if persist_features else None)
        scoped_symbols = self._feature_symbol_set(feature_symbols, feature_scope=feature_scope) if persist_features else set()
        attempted_before = budget.attempted if budget is not None else 0
        skipped_before = budget.skipped if budget is not None else 0
        reasons_before = len(budget.reasons) if budget is not None else 0
        cursor_before = self._feature_cursor(provider, connection.id) if persist_features and str(feature_scope or "").lower() == "all" else 0
        try:
            connector = self.trading_connections.connector_for_user(connection.user_id, connection.id)
            raw_markets = connector.discover_leveraged_markets(mode)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Leveraged-market discovery failed for %s connection %s: %s", provider, connection.id, exc)
            return DiscoveryResult(provider, connection.id, 0, 0, skipped=True, reason=str(exc))

        seen: set[str] = set()
        active_count = 0
        now = datetime.utcnow()
        feature_candidates: list[LeveragedMarket] = []
        for raw in raw_markets:
            normalized = self._normalize(provider, raw)
            if not normalized:
                continue
            venue_symbol = str(normalized["venue_symbol"]).strip()
            symbol = str(normalized["symbol"]).upper()
            venue_key = venue_symbol
            if venue_key in seen:
                continue
            seen.add(venue_key)
            market = self._market_for_venue_symbol(provider, venue_symbol)
            if market is None:
                market = LeveragedMarket(provider=provider, venue_symbol=venue_symbol)
                db.session.add(market)
            market.venue_symbol = venue_symbol
            market.trading_connection_id = connection.id
            market.symbol = symbol
            market.status = str(normalized.get("status") or "active")
            market.settlement_asset = str(normalized.get("settlement_asset") or provider_collateral_asset(provider)).upper()
            market.max_leverage = self._safe_float(normalized.get("max_leverage"), 1.0)
            market.tick_size = self._safe_float(normalized.get("tick_size"))
            market.lot_size = self._safe_float(normalized.get("lot_size"))
            market.contract_size = self._safe_float(normalized.get("contract_size"))
            market.min_size = self._safe_float(normalized.get("min_size"))
            market.funding_rate = self._safe_float(normalized.get("funding_rate"))
            market.liquidity_usd = self._safe_float(normalized.get("liquidity_usd"))
            market.spread_bps = self._safe_float(normalized.get("spread_bps"))
            market.fee_bps = self._safe_float(normalized.get("fee_bps"))
            market.raw = dict(normalized.get("raw") or raw)
            market.last_seen_at = now
            active_count += 1
            if persist_features and self._market_in_feature_scope(market, scoped_symbols):
                feature_candidates.append(market)

        disabled_count = 0
        if seen:
            for stale in LeveragedMarket.query.filter_by(provider=provider, status="active").all():
                if str(stale.venue_symbol or "") in seen:
                    continue
                stale.status = "disabled"
                disabled_count += 1
        try:
            db.session.flush()
        except IntegrityError as exc:
            db.session.rollback()
            logger.warning(
                "Leveraged-market discovery hit duplicate venue symbol for %s connection %s; rolled back sync: %s",
                provider,
                connection.id,
                exc,
            )
            return DiscoveryResult(
                provider,
                connection.id,
                0,
                0,
                skipped=True,
                reason="leveraged_market_unique_conflict",
            )
        next_cursor = cursor_before
        if persist_features and budget is not None:
            ordered_candidates = self._ordered_feature_candidates(provider, connection.id, feature_candidates, cursor_before, feature_scope)
            for market in ordered_candidates:
                self._persist_feature_snapshots(market, budget)
            if str(feature_scope or "").lower() == "all" and feature_candidates:
                attempted_delta = max((budget.attempted if budget is not None else 0) - attempted_before, 0)
                next_cursor = (cursor_before + attempted_delta) % len(feature_candidates) if attempted_delta else cursor_before
                self._set_feature_cursor(provider, connection.id, next_cursor)
        attempted_after = budget.attempted if budget is not None else 0
        skipped_after = budget.skipped if budget is not None else 0
        reasons_after = tuple(budget.reasons[reasons_before:]) if budget is not None else ()
        return DiscoveryResult(
            provider,
            connection.id,
            active_count,
            disabled_count,
            features_attempted=attempted_after - attempted_before,
            features_skipped=skipped_after - skipped_before,
            feature_skip_reasons=reasons_after,
            feature_cursor=cursor_before,
            next_feature_cursor=next_cursor,
        )

    def active_markets(self, provider: str | None = None, symbols: list[str] | None = None) -> list[LeveragedMarket]:
        query = LeveragedMarket.query.filter_by(status="active")
        provider_key = normalize_provider(provider)
        if provider_key and provider_key != "global":
            query = query.filter_by(provider=provider_key)
        symbol_set = {str(symbol).upper() for symbol in (symbols or []) if str(symbol or "").strip()}
        if symbol_set:
            query = query.filter(LeveragedMarket.symbol.in_(sorted(symbol_set)))
        return query.order_by(LeveragedMarket.liquidity_usd.desc(), LeveragedMarket.max_leverage.desc(), LeveragedMarket.symbol.asc()).all()

    @staticmethod
    def _market_for_venue_symbol(provider: str, venue_symbol: str) -> LeveragedMarket | None:
        # Venue symbol is the provider's unique contract identity. Generic base
        # symbols like DOGE can map to multiple contracts, so matching by symbol
        # here can mutate one contract into another and violate the unique index.
        with db.session.no_autoflush:
            return LeveragedMarket.query.filter_by(provider=provider, venue_symbol=venue_symbol).first()

    def _normalize(self, provider: str, raw: dict[str, Any]) -> dict[str, Any]:
        if provider == "hyperliquid":
            return self._normalize_hyperliquid(raw)
        if provider == "kucoin":
            return self._normalize_kucoin(raw)
        return {}

    def _normalize_hyperliquid(self, raw: dict[str, Any]) -> dict[str, Any]:
        venue_symbol = str(raw.get("name") or raw.get("coin") or raw.get("symbol") or "").strip()
        if not venue_symbol or bool(raw.get("isDelisted", False)):
            return {}
        if self._is_hyperliquid_indexed_symbol(venue_symbol):
            return {}
        symbol = venue_symbol.upper()
        context = raw.get("_asset_context") if isinstance(raw.get("_asset_context"), dict) else {}
        mark = self._safe_float(context.get("markPx") or context.get("midPx"))
        open_interest = self._safe_float(context.get("openInterest"))
        return {
            "venue_symbol": venue_symbol,
            "symbol": symbol,
            "status": "active",
            "settlement_asset": "USDC",
            "max_leverage": self._safe_float(raw.get("maxLeverage"), self.config.get("MAX_LEVERAGE", 1.0)),
            "tick_size": 10 ** (-int(self._safe_float(raw.get("szDecimals"), 0))),
            "lot_size": 10 ** (-int(self._safe_float(raw.get("szDecimals"), 0))),
            "min_size": 10 ** (-int(self._safe_float(raw.get("szDecimals"), 0))),
            "funding_rate": self._safe_float(context.get("funding")),
            "liquidity_usd": mark * open_interest if mark > 0 and open_interest > 0 else self._safe_float(context.get("dayNtlVlm")),
            "spread_bps": self._safe_float(context.get("spreadBps")),
            "fee_bps": self._safe_float(self.config.get("FEE_BPS"), 5.0),
            "raw": raw,
        }

    def _normalize_kucoin(self, raw: dict[str, Any]) -> dict[str, Any]:
        venue_symbol = str(raw.get("symbol") or "").upper()
        if not venue_symbol:
            return {}
        status = str(raw.get("status") or raw.get("state") or "").lower()
        if status and status not in {"open", "active", "trading"}:
            return {}
        base = str(raw.get("baseCurrency") or raw.get("baseCurrencySymbol") or "").upper()
        quote = str(raw.get("quoteCurrency") or raw.get("quoteCurrencySymbol") or raw.get("settleCurrency") or "").upper()
        if not base:
            base = re.sub(r"(USDTM|USDM|USDCM|USDTP|PERP)$", "", venue_symbol)
        if base == "XBT":
            base = "BTC"
        return {
            "venue_symbol": venue_symbol,
            "symbol": base or venue_symbol,
            "status": "active",
            "settlement_asset": str(raw.get("settleCurrency") or quote or "USDT").upper(),
            "max_leverage": self._safe_float(raw.get("maxLeverage"), self.config.get("MAX_LEVERAGE", 1.0)),
            "tick_size": self._safe_float(raw.get("tickSize") or raw.get("priceIncrement")),
            "lot_size": self._safe_float(raw.get("lotSize") or raw.get("multiplier")),
            "contract_size": self._safe_float(raw.get("multiplier") or raw.get("contractSize")),
            "min_size": self._safe_float(raw.get("lotSize") or raw.get("minSize")),
            "funding_rate": self._safe_float(raw.get("fundingFeeRate") or raw.get("fundingRate")),
            "liquidity_usd": self._safe_float(raw.get("turnoverOf24h") or raw.get("volumeOf24h") or raw.get("openInterest")),
            "spread_bps": self._safe_float(raw.get("spreadBps")),
            "fee_bps": self._safe_float(self.config.get("FEE_BPS"), 5.0),
            "raw": raw,
        }

    def _persist_feature_snapshots(self, market: LeveragedMarket, budget: _FeatureSyncBudget) -> None:
        provider = normalize_provider(market.provider)
        symbol = str(market.symbol or "").upper()
        market_symbol = str(market.venue_symbol or market.symbol or "").strip()
        if provider == "hyperliquid" and self._is_hyperliquid_indexed_symbol(market_symbol):
            budget.add_skip(f"{provider}:{symbol}:indexed_symbol_skipped")
            return
        if not self._feature_provider_available(provider, budget):
            return
        if not self._feature_symbol_available(provider, symbol, budget):
            return
        timeframes = self.config.get("ONE_H10_FEATURE_TIMEFRAMES", ["15m", "1h", "4h"])
        if isinstance(timeframes, str):
            timeframes = [item.strip() for item in timeframes.split(",") if item.strip()]
        timeframes = [str(timeframe) for timeframe in list(timeframes or [])[:6]]
        pending_timeframes = [
            timeframe
            for timeframe in timeframes
            if not self._feature_is_fresh(
                LeveragedMarketFeature.query.filter_by(leveraged_market_id=market.id, timeframe=str(timeframe)).one_or_none()
            )
        ]
        if not pending_timeframes:
            return
        if not budget.has_capacity():
            budget.add_skip("feature_cap_reached")
            return
        budget.attempted += 1
        for timeframe in pending_timeframes:
            try:
                candles = self._feature_candles(market_symbol, str(timeframe))
                order_book = self._feature_order_book(market_symbol)
                payload = self.feature_factory.build(
                    symbol=symbol,
                    timeframe=str(timeframe),
                    candles=candles,
                    optimizer_context={
                        "provider": market.provider,
                        "collateral_asset": market.settlement_asset,
                        "liquidity_usd": market.liquidity_usd,
                        "spread_bps": market.spread_bps,
                        "fee_bps": market.fee_bps,
                        "max_leverage": market.max_leverage,
                    },
                    order_book=order_book,
                    funding={"funding_rate": market.funding_rate},
                    provider_context={"provider": market.provider, "collateral_asset": market.settlement_asset},
                )
                payload["feature_schema_version"] = "1h10_feature_v1"
                payload["ml_namespace"] = "1h10"
                payload["fibonacci_timing"] = self._fibonacci_timing_features(candles)
                payload["order_book_imbalance"] = self._order_book_imbalance(order_book)
                feature = LeveragedMarketFeature.query.filter_by(leveraged_market_id=market.id, timeframe=str(timeframe)).one_or_none()
                if feature is None:
                    feature = LeveragedMarketFeature(
                        leveraged_market=market,
                        provider=market.provider,
                        symbol=symbol,
                        timeframe=str(timeframe),
                    )
                    db.session.add(feature)
                feature.provider = market.provider
                feature.symbol = symbol
                feature.feature_schema_version = "1h10_feature_v1"
                feature.features = payload
            except Exception as exc:  # noqa: BLE001
                if self._is_rate_limited_exception(exc):
                    self._mark_provider_feature_backoff(provider, f"{provider}:rate_limited_feature_backoff", budget)
                    return
                if self._is_invalid_symbol_exception(exc, symbol):
                    self._mark_symbol_feature_backoff(provider, symbol, f"{provider}:{symbol}:invalid_symbol", budget)
                    return
                budget.add_skip(f"{provider}:{symbol}:{timeframe}:feature_unavailable")
                logger.debug("Skipping %s %s feature snapshot for %s: %s", market.provider, timeframe, symbol, exc)

    def _feature_candles(self, symbol: str, timeframe: str) -> list[dict[str, Any]]:
        try:
            return self.market_data.get_candles(symbol, timeframe, mode="live", limit=240, retry=False)
        except TypeError as exc:
            if not self._looks_like_retry_kwarg_error(exc):
                raise
            return self.market_data.get_candles(symbol, timeframe, mode="live", limit=240)

    def _feature_order_book(self, symbol: str) -> dict[str, Any]:
        try:
            try:
                value = self.market_data.get_order_book(symbol, "live", retry=False)
            except TypeError as exc:
                if not self._looks_like_retry_kwarg_error(exc):
                    raise
                value = self.market_data.get_order_book(symbol, "live")
        except Exception as exc:  # noqa: BLE001
            if self._is_rate_limited_exception(exc):
                raise
            if self._is_invalid_symbol_exception(exc, symbol):
                raise
            return {}
        return value if isinstance(value, dict) else {}

    def _feature_symbol_set(self, feature_symbols: Iterable[str] | None, *, feature_scope: str = "allowed") -> set[str] | None:
        if str(feature_scope or "").strip().lower() in {"all", "full", "all_pairs", "one_h10"}:
            return None
        raw = feature_symbols
        if raw is None:
            raw = self.config.get("ALLOWED_SYMBOLS", ["BTC", "ETH", "SOL"])
        if isinstance(raw, str) and raw.strip() == "*":
            return None
        return {str(symbol).strip().upper() for symbol in raw if str(symbol or "").strip()}

    @staticmethod
    def _market_in_feature_scope(market: LeveragedMarket, symbols: set[str] | None) -> bool:
        if symbols is None:
            return True
        if not symbols:
            return False
        return str(market.symbol or "").upper() in symbols or str(market.venue_symbol or "").upper() in symbols

    def _ordered_feature_candidates(
        self,
        provider: str,
        connection_id: int | None,
        candidates: list[LeveragedMarket],
        cursor: int,
        feature_scope: str,
    ) -> list[LeveragedMarket]:
        sorted_candidates = sorted(
            candidates,
            key=lambda market: (
                -self._safe_float(getattr(market, "liquidity_usd", 0.0)),
                str(getattr(market, "symbol", "") or ""),
                str(getattr(market, "venue_symbol", "") or ""),
            ),
        )
        if not sorted_candidates or str(feature_scope or "").lower() != "all":
            return sorted_candidates
        start = max(0, min(int(cursor or 0), len(sorted_candidates) - 1))
        return [*sorted_candidates[start:], *sorted_candidates[:start]]

    @staticmethod
    def _feature_cursor_key(provider: str, connection_id: int | None) -> str:
        return f"one_h10_feature_cursor:{normalize_provider(provider)}:{connection_id or 'global'}"

    def _feature_cursor(self, provider: str, connection_id: int | None) -> int:
        try:
            return max(0, int(Setting.get_json(self._feature_cursor_key(provider, connection_id), 0) or 0))
        except Exception:  # noqa: BLE001
            return 0

    def _set_feature_cursor(self, provider: str, connection_id: int | None, cursor: int) -> None:
        try:
            Setting.set_json(self._feature_cursor_key(provider, connection_id), max(0, int(cursor or 0)))
        except Exception:  # noqa: BLE001
            return

    def _feature_max_markets_per_sync(self) -> int:
        return max(0, int(self._safe_float(self.config.get("ONE_H10_FEATURE_MAX_MARKETS_PER_SYNC"), 10.0)))

    def _feature_refresh_seconds(self) -> float:
        return max(0.0, self._safe_float(self.config.get("ONE_H10_FEATURE_REFRESH_SECONDS"), 3600.0))

    def _feature_backoff_seconds(self) -> float:
        return max(1.0, self._safe_float(self.config.get("ONE_H10_FEATURE_RATE_LIMIT_BACKOFF_SECONDS"), 300.0))

    def _feature_is_fresh(self, feature: LeveragedMarketFeature | None) -> bool:
        if feature is None or feature.updated_at is None:
            return False
        refresh_seconds = self._feature_refresh_seconds()
        if refresh_seconds <= 0:
            return False
        return datetime.utcnow() - feature.updated_at < timedelta(seconds=refresh_seconds)

    def _feature_provider_available(self, provider: str, budget: _FeatureSyncBudget) -> bool:
        blocked_until = self._feature_provider_backoff_until.get(provider, 0.0)
        if blocked_until > time.time():
            budget.add_skip(f"{provider}:feature_backoff_active")
            return False
        return True

    def _feature_symbol_available(self, provider: str, symbol: str, budget: _FeatureSyncBudget) -> bool:
        blocked_until = self._feature_symbol_backoff_until.get((provider, symbol), 0.0)
        if blocked_until > time.time():
            budget.add_skip(f"{provider}:{symbol}:symbol_backoff_active")
            return False
        return True

    def _mark_provider_feature_backoff(self, provider: str, reason: str, budget: _FeatureSyncBudget) -> None:
        self._feature_provider_backoff_until[provider] = time.time() + self._feature_backoff_seconds()
        budget.add_skip(reason)
        logger.info("Backed off %s 1H10 feature fetching: %s", provider, reason)

    def _mark_symbol_feature_backoff(self, provider: str, symbol: str, reason: str, budget: _FeatureSyncBudget) -> None:
        self._feature_symbol_backoff_until[(provider, symbol)] = time.time() + self._feature_backoff_seconds()
        budget.add_skip(reason)
        logger.info("Backed off %s %s 1H10 feature fetching: %s", provider, symbol, reason)

    @staticmethod
    def _is_rate_limited_exception(exc: Exception) -> bool:
        text = repr(exc).lower()
        return "429" in text or "rate limit" in text or "too many requests" in text

    @staticmethod
    def _is_invalid_symbol_exception(exc: Exception, symbol: str) -> bool:
        if isinstance(exc, KeyError):
            return True
        text = repr(exc).upper()
        symbol = str(symbol or "").upper()
        if not symbol:
            return False
        return f"'{symbol}'" in text or f'"{symbol}"' in text or ("INVALID" in text and symbol in text)

    @staticmethod
    def _is_hyperliquid_indexed_symbol(symbol: str) -> bool:
        return str(symbol or "").strip().startswith(("#", "@"))

    @staticmethod
    def _looks_like_retry_kwarg_error(exc: TypeError) -> bool:
        text = str(exc)
        return "retry" in text and ("unexpected keyword" in text or "got an unexpected" in text)

    def _fibonacci_timing_features(self, candles: list[dict[str, Any]]) -> dict[str, Any]:
        closes = [self._safe_float(row.get("close")) for row in candles if isinstance(row, dict)]
        closes = [value for value in closes if value > 0]
        if len(closes) < 3:
            return {}
        high = max(closes)
        low = min(closes)
        span = high - low
        latest = closes[-1]
        if span <= 0:
            return {"range_position": 0.0, "bars_since_extreme": 0}
        high_index = max(index for index, value in enumerate(closes) if value == high)
        low_index = max(index for index, value in enumerate(closes) if value == low)
        return {
            "range_position": (latest - low) / span,
            "bars_since_high": len(closes) - high_index - 1,
            "bars_since_low": len(closes) - low_index - 1,
            "fib_time_13": len(closes) % 13,
            "fib_time_21": len(closes) % 21,
            "fib_time_34": len(closes) % 34,
            "fib_time_55": len(closes) % 55,
        }

    def _order_book_imbalance(self, order_book: dict[str, Any]) -> float:
        levels = order_book.get("levels") if isinstance(order_book, dict) else None
        if not isinstance(levels, list) or len(levels) < 2:
            return 0.0
        bid_size = sum(self._safe_float(level.get("sz") or level.get("size")) for level in levels[0][:10] if isinstance(level, dict))
        ask_size = sum(self._safe_float(level.get("sz") or level.get("size")) for level in levels[1][:10] if isinstance(level, dict))
        total = bid_size + ask_size
        return (bid_size - ask_size) / total if total > 0 else 0.0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
