"""Dashboard opportunity facade over existing market, scanner, and ML services."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..extensions import db
from ..models import LeveragedMarket, MarketForecast, StrategyRanking
from .provider_assets import normalize_provider


@dataclass
class _CachedOpportunities:
    expires_at: float
    stale_until: float
    payload: dict[str, Any]


class DashboardOpportunityScanner:
    """Ranks dashboard opportunities without owning live execution."""

    def __init__(
        self,
        config: dict[str, Any],
        leveraged_markets: Any,
        market_scanner: Any,
        projection_engine: Any,
        trading_connections: Any | None = None,
    ) -> None:
        self.config = config
        self.leveraged_markets = leveraged_markets
        self.market_scanner = market_scanner
        self.projection_engine = projection_engine
        self.trading_connections = trading_connections
        self._cache: dict[tuple[Any, ...], _CachedOpportunities] = {}
        self._lock = threading.Lock()
        self._refreshing: set[tuple[Any, ...]] = set()
        self._last_sync_by_user: dict[int, float] = {}
        self.metrics = {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "stale_serves": 0,
            "background_refreshes": 0,
            "last_scan_ms": 0.0,
            "last_count": 0,
        }

    def opportunities(
        self,
        *,
        user: Any,
        mode: str,
        market_mode: str,
        limit: int | None = None,
        cursor: str | None = None,
        offset: int = 0,
        refresh: bool = False,
    ) -> dict[str, Any]:
        self.metrics["requests"] += 1
        page_size = max(1, min(int(limit or self.config.get("DASHBOARD_PAGE_SIZE", 30) or 30), 150))
        try:
            resolved_offset = max(0, int(cursor if cursor not in {None, ""} else offset or 0))
        except (TypeError, ValueError):
            resolved_offset = 0
        resolved_limit = min(150, resolved_offset + page_size + 1)
        user_id = int(getattr(user, "id", 0) or 0)
        key = (user_id, str(mode or "live"), str(market_mode or "live"), resolved_limit)
        now = time.time()
        ttl = max(1.0, float(self.config.get("DASHBOARD_OPPORTUNITY_TTL_SECONDS", 10.0) or 10.0))
        stale_ttl = max(ttl, float(self.config.get("DASHBOARD_OPPORTUNITY_STALE_SECONDS", 45.0) or 45.0))
        if not refresh:
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None and now < cached.expires_at:
                    self.metrics["cache_hits"] += 1
                    return self._page_payload(dict(cached.payload), page_size=page_size, offset=resolved_offset, cache_hit=True)
                if cached is not None and now < cached.stale_until:
                    self.metrics["stale_serves"] += 1
                    self._refresh_async(key, user=user, mode=mode, market_mode=market_mode, limit=resolved_limit, ttl=ttl, stale_ttl=stale_ttl)
                    return self._page_payload(dict(cached.payload), page_size=page_size, offset=resolved_offset, cache_hit=True, stale=True)

        self.metrics["cache_misses"] += 1
        payload = self._build_payload(
            user=user,
            mode=mode,
            market_mode=market_mode,
            limit=resolved_limit,
        )
        with self._lock:
            self._cache[key] = _CachedOpportunities(expires_at=now + ttl, stale_until=now + stale_ttl, payload=dict(payload))
        return self._page_payload(payload, page_size=page_size, offset=resolved_offset)

    def _build_payload(
        self,
        *,
        user: Any,
        mode: str,
        market_mode: str,
        limit: int,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        user_id = int(getattr(user, "id", 0) or 0)
        self._maybe_sync_markets(user_id, market_mode)
        markets = self._active_markets_for_user(user_id)
        rows = self._rank_markets(markets, mode=market_mode, limit=limit)
        if not rows:
            rows = self._ranking_fallback(limit)
        rows.sort(key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
        rows = rows[:limit]
        payload = {
            "opportunities": rows,
            "count": len(rows),
            "mode": mode,
            "market_mode": market_mode,
            "updated_at": datetime.utcnow().isoformat(),
            "diagnostics": {
                "active_market_count": len(markets),
                "cache_hit": False,
                "forecast_persistence": "enabled",
                "preview_only": True,
            },
        }
        self.metrics["last_scan_ms"] = (time.perf_counter() - started_at) * 1000
        self.metrics["last_count"] = len(rows)
        return payload

    def _refresh_async(
        self,
        key: tuple[Any, ...],
        *,
        user: Any,
        mode: str,
        market_mode: str,
        limit: int,
        ttl: float,
        stale_ttl: float,
    ) -> None:
        if key in self._refreshing:
            return
        self._refreshing.add(key)
        try:
            from flask import current_app, has_app_context
            app = current_app._get_current_object() if has_app_context() else None
        except Exception:
            app = None

        def _run() -> None:
            try:
                if app is not None:
                    with app.app_context():
                        payload = self._build_payload(user=user, mode=mode, market_mode=market_mode, limit=limit)
                else:
                    payload = self._build_payload(user=user, mode=mode, market_mode=market_mode, limit=limit)
                now = time.time()
                with self._lock:
                    self._cache[key] = _CachedOpportunities(expires_at=now + ttl, stale_until=now + stale_ttl, payload=dict(payload))
                    self.metrics["background_refreshes"] += 1
            finally:
                with self._lock:
                    self._refreshing.discard(key)

        threading.Thread(target=_run, daemon=True).start()

    @staticmethod
    def _page_payload(
        payload: dict[str, Any],
        *,
        page_size: int,
        offset: int,
        cache_hit: bool = False,
        stale: bool = False,
    ) -> dict[str, Any]:
        rows = list(payload.get("opportunities") or [])[:150]
        page = rows[offset : offset + page_size]
        next_offset = offset + len(page)
        result = dict(payload)
        result["opportunities"] = page
        result["count"] = len(page)
        result["total"] = len(rows)
        result["next_cursor"] = str(next_offset) if next_offset < len(rows) else None
        result["has_more"] = next_offset < len(rows)
        diagnostics = dict(result.get("diagnostics") or {})
        diagnostics["cache_hit"] = cache_hit
        diagnostics["stale"] = stale
        result["diagnostics"] = diagnostics
        return result

    def chart_payload(
        self,
        *,
        provider: str,
        symbol: str,
        venue_symbol: str = "",
        timeframe: str = "live",
        market_mode: str = "live",
    ) -> dict[str, Any]:
        provider_key = normalize_provider(provider)
        symbol_key = str(symbol or "").upper()
        market = self._market(provider_key, symbol_key, venue_symbol)
        features = self._market_features(market) if market is not None else {}
        latest = self._latest_forecast(provider_key, symbol_key, timeframe)
        forecast = latest.payload if latest is not None else {}
        if not forecast and features:
            forecast = self.projection_engine.forecast_from_features(
                features,
                provider=provider_key,
                symbol=symbol_key,
                allocation_cap_usd=self._preview_allocation(),
                available_margin_usd=self._preview_allocation(),
                market=market,
            )
        return self.projection_engine.chart_payload(
            provider=provider_key,
            symbol=symbol_key,
            venue_symbol=venue_symbol or (str(getattr(market, "venue_symbol", "") or "") if market is not None else symbol_key),
            mode=market_mode,
            timeframe=timeframe,
            forecast=forecast,
            features=features,
        )

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            cache_entries = len(self._cache)
        return {
            "cache_entries": cache_entries,
            "requests": self.metrics["requests"],
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "stale_serves": self.metrics["stale_serves"],
            "background_refreshes": self.metrics["background_refreshes"],
            "last_scan_ms": self.metrics["last_scan_ms"],
            "last_count": self.metrics["last_count"],
            "forecast_rows": MarketForecast.query.count(),
        }

    def _maybe_sync_markets(self, user_id: int, market_mode: str) -> None:
        if user_id <= 0 or not bool(self.config.get("DASHBOARD_OPPORTUNITY_DISCOVERY_REFRESH_ENABLED", False)):
            return
        ttl = max(30.0, float(self.config.get("DASHBOARD_OPPORTUNITY_DISCOVERY_REFRESH_SECONDS", 300.0) or 300.0))
        now = time.time()
        if now - self._last_sync_by_user.get(user_id, 0.0) < ttl:
            return
        self._last_sync_by_user[user_id] = now
        try:
            self.leveraged_markets.sync_for_user(user_id, mode=market_mode, feature_scope="allowed", persist_features=False)
        except Exception:
            return

    def _active_markets_for_user(self, user_id: int) -> list[LeveragedMarket]:
        all_markets = list(self.leveraged_markets.active_markets())
        if user_id <= 0 or self.trading_connections is None:
            return all_markets
        try:
            connections = list(self.trading_connections.verified_tradable_connections(user_id))
        except Exception:
            return all_markets
        connection_ids = {int(connection.id) for connection in connections if getattr(connection, "id", None)}
        providers = {normalize_provider(getattr(connection, "provider", "")) for connection in connections}
        scoped = [
            market
            for market in all_markets
            if int(getattr(market, "trading_connection_id", 0) or 0) in connection_ids
            or normalize_provider(getattr(market, "provider", "")) in providers
        ]
        return scoped or all_markets

    def _rank_markets(self, markets: list[LeveragedMarket], *, mode: str, limit: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        provider_groups: dict[str, list[LeveragedMarket]] = {}
        for market in markets:
            provider_groups.setdefault(normalize_provider(market.provider), []).append(market)
        for provider, provider_markets in provider_groups.items():
            scored = self._scored_candidates(provider_markets, provider=provider, limit=limit)
            seen_market_ids: set[int] = set()
            for candidate in scored:
                market = self._market(provider, str(candidate.symbol or "").upper(), str(candidate.features.get("venue_symbol", "")))
                if market is not None and market.id is not None:
                    seen_market_ids.add(int(market.id))
                rows.append(self._row_from_candidate(candidate, market, provider=provider, mode=mode))
            for market in provider_markets:
                if market.id is not None and int(market.id) in seen_market_ids:
                    continue
                rows.append(self._row_from_market(market, mode=mode))
        return [row for row in rows if row]

    def _scored_candidates(self, markets: list[LeveragedMarket], *, provider: str, limit: int) -> list[Any]:
        try:
            return list(self.market_scanner.score_one_h10_markets(markets, provider=provider, limit=limit) or [])
        except Exception:
            return []

    def _row_from_candidate(self, candidate: Any, market: LeveragedMarket | None, *, provider: str, mode: str) -> dict[str, Any]:
        features = dict(getattr(candidate, "features", {}) or {})
        if market is not None:
            features = {**self._market_features(market), **features}
        symbol = str(getattr(candidate, "symbol", features.get("symbol", "")) or "").upper()
        forecast = self.projection_engine.forecast_from_features(
            features,
            provider=provider,
            symbol=symbol,
            allocation_cap_usd=self._preview_allocation(),
            available_margin_usd=self._preview_allocation(),
            market=market,
        )
        row = self._opportunity_row(
            provider=provider,
            symbol=symbol,
            venue_symbol=str(getattr(market, "venue_symbol", features.get("venue_symbol", symbol)) or symbol),
            base_score=float(getattr(candidate, "score", 0.0) or 0.0),
            source=str(getattr(candidate, "source", "market_scanner") or "market_scanner"),
            features=features,
            forecast=forecast,
        )
        self._persist(row, forecast, features)
        return row

    def _row_from_market(self, market: LeveragedMarket, *, mode: str) -> dict[str, Any]:
        provider = normalize_provider(market.provider)
        symbol = str(market.symbol or "").upper()
        features = self._market_features(market)
        forecast = self.projection_engine.forecast_from_features(
            features,
            provider=provider,
            symbol=symbol,
            allocation_cap_usd=self._preview_allocation(),
            available_margin_usd=self._preview_allocation(),
            market=market,
        )
        base_score = self._safe_float(market.liquidity_usd) / 100_000.0 - self._safe_float(market.spread_bps) / 20.0
        row = self._opportunity_row(
            provider=provider,
            symbol=symbol,
            venue_symbol=str(market.venue_symbol or symbol),
            base_score=base_score,
            source="active_leveraged_market",
            features=features,
            forecast=forecast,
        )
        self._persist(row, forecast, features)
        return row

    def _opportunity_row(
        self,
        *,
        provider: str,
        symbol: str,
        venue_symbol: str,
        base_score: float,
        source: str,
        features: dict[str, Any],
        forecast: dict[str, Any],
    ) -> dict[str, Any]:
        levels = self.projection_engine.price_levels([], forecast, features)
        confidence = max(0.0, min(self._safe_float(forecast.get("confidence")), 1.0))
        expected_bps = self._safe_float(forecast.get("expected_return_bps"), forecast.get("net_expected_return_bps", 0.0))
        model_agreement = self._safe_float(
            forecast.get("ml_agreement_score"),
            forecast.get("ml_consensus_multiplier", forecast.get("confidence", 0.0)),
        )
        fib = features.get("fibonacci_confluence") if isinstance(features.get("fibonacci_confluence"), dict) else {}
        fibonacci_alignment = self._safe_float(fib.get("score"), forecast.get("fibonacci_alignment", 0.0))
        liquidity = max(self._safe_float(features.get("liquidity_usd")), self._safe_float(features.get("liquidity_capacity_usd")))
        liquidity_score = max(0.0, min(liquidity / max(self._min_liquidity(), 1.0), 1.0))
        strategy_consensus = max(
            self._safe_float(features.get("one_hour_edge_v2")),
            self._safe_float(features.get("expected_execution_quality")),
            confidence,
        )
        score = (
            base_score
            + confidence * 4.0
            + max(expected_bps, 0.0) / 45.0
            + model_agreement * 1.5
            + fibonacci_alignment
            + liquidity_score
        )
        direction = str(forecast.get("predicted_side") or forecast.get("action") or "hold").lower()
        if direction not in {"buy", "sell", "hold"}:
            direction = "hold"
        return {
            "provider": normalize_provider(provider),
            "symbol": symbol,
            "venue_symbol": venue_symbol,
            "direction": direction,
            "score": score,
            "confidence": confidence,
            "predicted_roi": expected_bps / 100.0,
            "expected_return_bps": expected_bps,
            "duration": self._duration_label(forecast),
            "entry": levels["entry"],
            "exit": levels["exit"],
            "stop_loss": levels["stop_loss"],
            "risk_reward": levels["risk_reward"],
            "liquidity_score": liquidity_score,
            "slippage_bps": self._safe_float(forecast.get("spread_bps"), features.get("spread_bps", 0.0)),
            "strategy_consensus": max(0.0, min(strategy_consensus, 1.0)),
            "ml_model_agreement": max(0.0, min(model_agreement, 1.0)),
            "fibonacci_alignment": max(0.0, min(fibonacci_alignment, 1.0)),
            "blockers": [str(item) for item in (forecast.get("blockers", []) or []) if str(item)],
            "advisory_blockers": [str(item) for item in (forecast.get("advisory_blockers", []) or []) if str(item)],
            "source": source,
            "score_breakdown": dict(features.get("scanner_score_breakdown") or {}),
            "forecast_source": forecast.get("source"),
            "preview_only": True,
        }

    def _persist(self, row: dict[str, Any], forecast: dict[str, Any], features: dict[str, Any]) -> None:
        provider = normalize_provider(row.get("provider"))
        symbol = str(row.get("symbol") or "").upper()
        timeframe = "live"
        horizon = str(forecast.get("ml_horizon") or forecast.get("horizon") or "1h10")
        now = datetime.utcnow()
        ttl = max(10.0, float(self.config.get("DASHBOARD_FORECAST_TTL_SECONDS", 90.0) or 90.0))
        existing = (
            MarketForecast.query.filter_by(provider=provider, symbol=symbol, timeframe=timeframe, horizon=horizon)
            .order_by(MarketForecast.created_at.desc())
            .first()
        )
        record = existing if existing and existing.expires_at > now else MarketForecast()
        record.provider = provider
        record.venue_symbol = str(row.get("venue_symbol") or symbol)
        record.symbol = symbol
        record.timeframe = timeframe
        record.horizon = horizon
        record.side = str(row.get("direction") or "hold")
        record.confidence = self._safe_float(row.get("confidence"))
        record.expected_return_bps = self._safe_float(row.get("expected_return_bps"))
        record.entry_price = self._safe_float(row.get("entry"))
        record.exit_price = self._safe_float(row.get("exit"))
        record.stop_loss_price = self._safe_float(row.get("stop_loss"))
        record.risk_reward = self._safe_float(row.get("risk_reward"))
        record.liquidity_score = self._safe_float(row.get("liquidity_score"))
        record.slippage_bps = self._safe_float(row.get("slippage_bps"))
        record.model_agreement = self._safe_float(row.get("ml_model_agreement"))
        record.fibonacci_alignment = self._safe_float(row.get("fibonacci_alignment"))
        record.source = str(row.get("source") or "dashboard")
        overlays = self.projection_engine.overlays([], forecast, features)
        record.forecast_path = list(overlays.get("path", []) or [])
        record.zones = dict(overlays.get("zones", {}) or {})
        record.score_breakdown = dict(row.get("score_breakdown") or {})
        record.payload = {**dict(row), "forecast": forecast}
        record.created_at = now
        record.expires_at = now + timedelta(seconds=ttl)
        db.session.add(record)
        self._prune_forecasts(now)
        db.session.commit()

    def _prune_forecasts(self, now: datetime) -> None:
        MarketForecast.query.filter(MarketForecast.expires_at < now).delete(synchronize_session=False)
        max_rows = max(20, int(self.config.get("DASHBOARD_FORECAST_MAX_ROWS", 150) or 150))
        stale_ids = [
            row.id
            for row in MarketForecast.query.order_by(MarketForecast.created_at.desc()).offset(max_rows).all()
            if row.id is not None
        ]
        if stale_ids:
            MarketForecast.query.filter(MarketForecast.id.in_(stale_ids)).delete(synchronize_session=False)

    def _market_features(self, market: LeveragedMarket | None) -> dict[str, Any]:
        if market is None:
            return {}
        rows = list(getattr(market, "feature_rows", []) or [])
        priority = {"1m": 0, "5m": 1, "15m": 2, "1h": 3, "4h": 4}
        rows.sort(key=lambda row: priority.get(str(row.timeframe), 99))
        payload = dict(rows[0].features if rows else {})
        payload.update(
            {
                "provider": normalize_provider(market.provider),
                "symbol": str(market.symbol or "").upper(),
                "venue_symbol": str(market.venue_symbol or market.symbol or "").upper(),
                "liquidity_usd": max(self._safe_float(payload.get("liquidity_usd")), self._safe_float(market.liquidity_usd)),
                "spread_bps": max(self._safe_float(payload.get("spread_bps")), self._safe_float(market.spread_bps)),
                "fee_bps": self._safe_float(market.fee_bps, self.config.get("FEE_BPS", 5.0)),
                "max_leverage": self._safe_float(market.max_leverage, self.config.get("MAX_LEVERAGE", 1.0)),
                "funding_rate": self._safe_float(market.funding_rate),
                "one_h10_feature_timeframes": sorted({str(row.timeframe) for row in rows}),
            }
        )
        return payload

    def _market(self, provider: str, symbol: str, venue_symbol: str = "") -> LeveragedMarket | None:
        provider_key = normalize_provider(provider)
        venue_key = str(venue_symbol or "").upper()
        query = LeveragedMarket.query.filter_by(provider=provider_key, status="active")
        if venue_key:
            market = query.filter(LeveragedMarket.venue_symbol == venue_key).one_or_none()
            if market is not None:
                return market
        return query.filter(LeveragedMarket.symbol == str(symbol or "").upper()).first()

    def _latest_forecast(self, provider: str, symbol: str, timeframe: str) -> MarketForecast | None:
        return (
            MarketForecast.query.filter_by(provider=normalize_provider(provider), symbol=str(symbol or "").upper())
            .filter(MarketForecast.expires_at >= datetime.utcnow())
            .order_by(MarketForecast.created_at.desc())
            .first()
        )

    def _ranking_fallback(self, limit: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        rankings = (
            StrategyRanking.query.filter_by(rejected=False)
            .order_by(StrategyRanking.score.desc(), StrategyRanking.created_at.desc())
            .limit(limit)
            .all()
        )
        for ranking in rankings:
            symbol = str(ranking.symbol or "").upper()
            provider = normalize_provider(ranking.provider)
            expected_bps = self._safe_float(ranking.edge_score, ranking.recent_1h_return * 10_000)
            confidence = max(0.0, min(self._safe_float(ranking.ml_adjusted_score, ranking.score) / 10.0, 1.0))
            rows.append(
                {
                    "provider": provider,
                    "symbol": symbol,
                    "venue_symbol": symbol,
                    "direction": "hold",
                    "score": self._safe_float(ranking.score),
                    "confidence": confidence,
                    "predicted_roi": expected_bps / 100.0,
                    "expected_return_bps": expected_bps,
                    "duration": f"{ranking.lock_duration_hours or 1}h",
                    "entry": 0.0,
                    "exit": 0.0,
                    "stop_loss": 0.0,
                    "risk_reward": 0.0,
                    "liquidity_score": max(0.0, min(self._safe_float(ranking.capacity_usd) / self._min_liquidity(), 1.0)),
                    "slippage_bps": self._safe_float(ranking.cost_drag_bps),
                    "strategy_consensus": max(0.0, min(self._safe_float(ranking.window_stability), 1.0)),
                    "ml_model_agreement": confidence,
                    "fibonacci_alignment": 0.0,
                    "blockers": [],
                    "advisory_blockers": list(ranking.warnings or []),
                    "source": "strategy_ranking",
                    "score_breakdown": ranking.ml_explanation,
                    "preview_only": True,
                }
            )
        return rows

    def _duration_label(self, forecast: dict[str, Any]) -> str:
        seconds = self._safe_float(forecast.get("horizon_seconds"), 3600.0)
        if seconds >= 3600:
            return f"{seconds / 3600:.1f}h".replace(".0", "")
        return f"{int(seconds // 60)}m"

    def _preview_allocation(self) -> float:
        return max(1.0, float(self.config.get("DASHBOARD_FORECAST_PREVIEW_ALLOCATION_USD", 10.0) or 10.0))

    def _min_liquidity(self) -> float:
        return max(1.0, self._safe_float(self.config.get("ONE_H10_MIN_LIQUIDITY_USD"), self.config.get("VAULT_MIN_LIQUIDITY_USD", 1_000.0)))

    @staticmethod
    def _safe_float(value: Any, default: Any = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return float(default)
            except (TypeError, ValueError):
                return 0.0
