from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

from app.auth import password_hash
from app.extensions import db
from app.models import LeveragedMarket, LeveragedMarketFeature, MarketForecast, Setting, TradingConnection, User
from app.routes import dashboard as dashboard_routes
from app.services.hyperliquid_client import ClientSnapshot


def _add_market(provider: str, symbol: str, venue_symbol: str, *, close: float = 100.0) -> LeveragedMarket:
    market = LeveragedMarket(
        provider=provider,
        venue_symbol=venue_symbol,
        symbol=symbol,
        status="active",
        settlement_asset="USDC",
        max_leverage=5.0,
        liquidity_usd=250_000.0,
        spread_bps=1.2,
        fee_bps=4.0,
    )
    db.session.add(market)
    db.session.flush()
    features = LeveragedMarketFeature(
        leveraged_market_id=market.id,
        provider=provider,
        symbol=symbol,
        timeframe="15m",
    )
    features.features = {
        "close": close,
        "atr_pct": 0.012,
        "volatility": 0.011,
        "trend_strength": 0.018,
        "ema_trend": 1.6,
        "macd_histogram": 0.35,
        "rsi": 58,
        "order_book_imbalance": 0.2,
        "liquidity_usd": 250_000.0,
        "liquidity_capacity_usd": 200_000.0,
        "spread_bps": 1.2,
        "fibonacci_confluence": {"score": 0.78, "cluster_count": 3},
        "fibonacci_timing": {"range_position": 0.42},
    }
    db.session.add(features)
    return market


def _forecast(features: dict[str, Any], *, provider: str, symbol: str, **_: Any) -> dict[str, Any]:
    direction = "sell" if symbol.startswith("ETH") else "buy"
    return {
        "predicted_side": direction,
        "action": direction,
        "confidence": 0.84,
        "expected_return_bps": 62.0,
        "net_expected_return_bps": 51.0,
        "suggested_stop_loss_pct": 0.012,
        "suggested_take_profit_pct": 0.036,
        "horizon_seconds": 4200,
        "ml_agreement_score": 0.81,
        "fibonacci_alignment": 0.76,
        "blockers": [],
        "advisory_blockers": [],
        "source": "test_dashboard_forecast",
    }


def test_dashboard_opportunity_scanner_ranks_enabled_providers_and_persists(app, monkeypatch) -> None:
    app.config["DASHBOARD_FORECAST_MAX_ROWS"] = 3
    service = app.extensions["services"]["dashboard_opportunities"]
    monkeypatch.setattr(service.market_scanner, "score_one_h10_markets", lambda *args, **kwargs: [])
    monkeypatch.setattr(service.projection_engine, "forecast_from_features", _forecast)

    _add_market("hyperliquid", "BTC", "BTC", close=101.0)
    _add_market("kucoin", "ETH", "ETH-USDTM", close=2025.0)
    db.session.commit()

    payload = service.opportunities(user=None, mode="live", market_mode="live", limit=10, refresh=True)

    providers = {row["provider"] for row in payload["opportunities"]}
    assert {"hyperliquid", "kucoin"} <= providers
    first = payload["opportunities"][0]
    assert {
        "provider",
        "symbol",
        "venue_symbol",
        "direction",
        "score",
        "confidence",
        "predicted_roi",
        "duration",
        "entry",
        "exit",
        "stop_loss",
        "risk_reward",
        "liquidity_score",
        "slippage_bps",
        "strategy_consensus",
        "ml_model_agreement",
        "fibonacci_alignment",
        "blockers",
        "advisory_blockers",
    } <= set(first)
    assert first["preview_only"] is True
    assert MarketForecast.query.count() == 2

    stale = MarketForecast(
        provider="hyperliquid",
        venue_symbol="STALE",
        symbol="STALE",
        timeframe="live",
        horizon="1h10",
        created_at=datetime.utcnow() - timedelta(hours=2),
        expires_at=datetime.utcnow() - timedelta(minutes=1),
    )
    db.session.add(stale)
    db.session.commit()

    service.opportunities(user=None, mode="live", market_mode="live", limit=10, refresh=True)
    assert MarketForecast.query.filter_by(symbol="STALE").count() == 0
    assert MarketForecast.query.count() <= 3


def test_dashboard_opportunity_scanner_serves_stale_cache_while_refreshing(app, monkeypatch) -> None:
    service = app.extensions["services"]["dashboard_opportunities"]
    monkeypatch.setattr(service.market_scanner, "score_one_h10_markets", lambda *args, **kwargs: [])
    monkeypatch.setattr(service.projection_engine, "forecast_from_features", _forecast)
    _add_market("hyperliquid", "BTC", "BTC", close=101.0)
    db.session.commit()

    payload = service.opportunities(user=None, mode="live", market_mode="live", limit=10, refresh=True)
    key = (0, "live", "live", 11)
    cached = service._cache[key]
    cached.expires_at = 0.0
    cached.stale_until = time.time() + 30.0
    refreshes = {"count": 0}
    monkeypatch.setattr(service, "_refresh_async", lambda *args, **kwargs: refreshes.__setitem__("count", refreshes["count"] + 1))

    stale = service.opportunities(user=None, mode="live", market_mode="live", limit=10)

    assert stale["opportunities"] == payload["opportunities"]
    assert stale["diagnostics"]["cache_hit"] is True
    assert stale["diagnostics"]["stale"] is True
    assert refreshes["count"] == 1


def test_ml_projection_engine_aggregates_45m_from_15m(app) -> None:
    engine = app.extensions["services"]["ml_projection_engine"]
    base = 1_700_000_000
    candles = [
        {"time": base + index * 900, "open": 100 + index, "high": 102 + index, "low": 99 + index, "close": 101 + index, "volume": 10 + index}
        for index in range(6)
    ]
    engine.market_data.get_candles = lambda symbol, timeframe, mode, limit: list(candles)

    rows = engine.candles("BTC", "45m", mode="live", limit=2)

    assert len(rows) == 2
    assert rows[0]["open"] == 100
    assert rows[0]["high"] == 104
    assert rows[0]["low"] == 99
    assert rows[0]["close"] == 103
    assert rows[0]["volume"] == 33
    assert rows[1]["open"] == 103
    assert rows[1]["close"] == 106


def test_dashboard_forecast_routes_payload_shape_and_sse(app, monkeypatch) -> None:
    service = app.extensions["services"]["dashboard_opportunities"]
    monkeypatch.setattr(dashboard_routes, "require_admin", lambda: None)
    monkeypatch.setattr(service.market_scanner, "score_one_h10_markets", lambda *args, **kwargs: [])
    monkeypatch.setattr(service.projection_engine, "forecast_from_features", _forecast)
    service.projection_engine.market_data.get_candles = lambda symbol, timeframe, mode, limit: [
        {"time": 1_700_000_000 + index * 60, "open": 100, "high": 102, "low": 99, "close": 100 + index, "volume": 1000}
        for index in range(20)
    ]
    _add_market("hyperliquid", "BTC", "BTC", close=100.0)
    db.session.commit()

    client = app.test_client()
    opportunities = client.get("/admin/api/dashboard/opportunities")
    assert opportunities.status_code == 200
    assert "no-store" in opportunities.headers["Cache-Control"]
    assert opportunities.get_json()["opportunities"][0]["provider"] == "hyperliquid"

    chart = client.get("/admin/api/dashboard/chart?provider=hyperliquid&symbol=BTC&timeframe=45m")
    assert chart.status_code == 200
    chart_payload = chart.get_json()
    assert chart_payload["timeframe"] == "45m"
    assert len(chart_payload["candles"]) <= 150
    assert {"path", "zones", "confidence_band", "volatility_cone", "fibonacci_time_zones"} <= set(chart_payload["overlays"])

    stream = client.get("/admin/api/dashboard/stream?once=1")
    assert stream.status_code == 200
    assert stream.headers["Content-Type"].startswith("text/event-stream")
    assert "no-store" in stream.headers["Cache-Control"]
    assert b"event: opportunities" in stream.data
    assert b"event: activity" in stream.data

    activity = client.get("/admin/api/dashboard/activity")
    assert activity.status_code == 200
    assert "no-store" in activity.headers["Cache-Control"]
    assert {"items", "next_cursor", "has_more"} <= set(activity.get_json())


def test_dashboard_new_api_routes_remain_admin_protected(app) -> None:
    client = app.test_client()
    assert client.get("/admin/api/dashboard/opportunities").status_code in {302, 401, 403}
    assert client.get("/admin/api/dashboard/chart").status_code in {302, 401, 403}
    assert client.get("/admin/api/dashboard/stream?once=1").status_code in {302, 401, 403}
    assert client.get("/admin/api/dashboard/activity").status_code in {302, 401, 403}


def test_dashboard_render_refreshes_provider_snapshot_on_open(app, monkeypatch) -> None:
    admin = User(username="dash-admin", password_hash=password_hash("password123"), role="admin")
    db.session.add(admin)
    db.session.flush()
    connection = TradingConnection(
        user_id=admin.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        is_active=True,
        verification_status="verified",
    )
    db.session.add(connection)
    db.session.commit()
    monkeypatch.setattr(dashboard_routes, "require_admin", lambda: None)
    monkeypatch.setattr(dashboard_routes, "current_user", lambda: admin)

    calls = {"count": 0}

    def account_snapshot(*args: Any, **kwargs: Any) -> ClientSnapshot:
        calls["count"] += 1
        return ClientSnapshot(
            "live",
            [{"asset": "USDC", "type": "margin", "value": 500.0, "withdrawable": 500.0}],
            [{"symbol": "BTC", "quantity": 0.2, "entry_price": 100.0, "mark_price": 110.0, "unrealized_pnl": 2.0}],
            [{"symbol": "BTC", "side": "buy", "price": 99.0, "size": 0.1, "reduce_only": False}],
            [{"symbol": "BTC", "side": "sell", "price": 111.0, "size": 0.1, "closed_pnl": 1.25}],
            [],
        )

    services = app.extensions["services"]
    monkeypatch.setattr(services["trading_connections"], "account_snapshot", account_snapshot)
    services["market_data"].get_dashboard_market_summary = lambda symbols, timeframe, mode: [
        {"symbol": "BTC", "mid": 100.0, "change_pct": 0.0}
    ]
    services["market_data"].get_candles = lambda symbol, timeframe, mode, limit: [
        {"time": 1_700_000_000 + index * 60, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1}
        for index in range(80)
    ]

    response = app.test_client().get("/admin/dashboard")

    assert response.status_code == 200
    assert "no-store" in response.headers["Cache-Control"]
    assert calls["count"] == 1
    assert b"Dynamic Opportunities" in response.data
    assert b'data-dashboard-data-url="/admin/api/dashboard-data?refresh_exchange=1"' in response.data
    assert b"500.00" in response.data
    assert b"Provider Health" in response.data
    assert b"Manual Order Entry" not in response.data


def test_dashboard_data_refresh_returns_provider_health_and_cached_ops_rows(app, monkeypatch) -> None:
    admin = User(username="dash-api-admin", password_hash=password_hash("password123"), role="admin")
    db.session.add(admin)
    db.session.flush()
    connection = TradingConnection(
        user_id=admin.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        is_active=True,
        verification_status="verified",
    )
    db.session.add(connection)
    db.session.commit()
    monkeypatch.setattr(dashboard_routes, "require_admin", lambda: None)
    monkeypatch.setattr(dashboard_routes, "current_user", lambda: admin)
    services = app.extensions["services"]
    services["market_data"].get_dashboard_market_summary = lambda symbols, timeframe, mode: []
    services["market_data"].get_candles = lambda symbol, timeframe, mode, limit: []

    def live_snapshot(*args: Any, **kwargs: Any) -> ClientSnapshot:
        return ClientSnapshot(
            "live",
            [{"asset": "USDC", "type": "margin", "value": 500.0, "withdrawable": 500.0}],
            [{"symbol": "BTC", "quantity": 0.2, "entry_price": 100.0, "mark_price": 110.0, "unrealized_pnl": 2.0}],
            [{"symbol": "BTC", "side": "buy", "price": 99.0, "size": 0.1, "reduce_only": False}],
            [{"symbol": "BTC", "side": "sell", "price": 111.0, "size": 0.1, "closed_pnl": 1.25}],
            [],
        )

    monkeypatch.setattr(services["trading_connections"], "account_snapshot", live_snapshot)
    response = app.test_client().get("/admin/api/dashboard-data?refresh_exchange=1")
    payload = response.get_json()
    cached = Setting.get_json(f"exchange_balance_snapshot:{admin.id}", {})

    assert response.status_code == 200
    assert payload["account_snapshot"]["status"] == "live"
    assert payload["provider_health"]["status"] == "online"
    assert payload["positions"][0]["symbol"] == "BTC"
    assert payload["open_orders"][0]["symbol"] == "BTC"
    assert payload["recent_trades"][0]["closed_pnl"] == 1.25
    assert cached["positions"][0]["symbol"] == "BTC"
    assert cached["open_orders"][0]["price"] == 99.0
    assert cached["recent_fills"][0]["closed_pnl"] == 1.25

    services["dashboard_payload"]._cache.clear()
    services["dashboard_payload"]._inflight.clear()
    monkeypatch.setattr(
        services["trading_connections"],
        "account_snapshot",
        lambda *args, **kwargs: ClientSnapshot("live", [], [], [], [], ["Hyperliquid unavailable"]),
    )

    degraded = app.test_client().get("/admin/api/dashboard-data?refresh_exchange=1").get_json()

    assert degraded["account_snapshot"]["status"] == "degraded"
    assert degraded["positions"][0]["symbol"] == "BTC"
    assert degraded["provider_health"]["status"] == "degraded"
    assert "Hyperliquid unavailable" in degraded["provider_health"]["impact"]


def test_dashboard_service_worker_does_not_cache_authenticated_html_or_apis() -> None:
    source = open("static/js/sw.js", encoding="utf-8").read()
    app_shell = source.split("];", 1)[0]
    assert '"/admin/dashboard"' not in app_shell
    assert '"/wallet"' not in app_shell
    assert '"/vault"' not in app_shell
    assert "isApiRequest" in source
    assert 'cache: "no-store"' in source


def test_dashboard_frontend_uses_lazy_chart_module_and_abortable_requests() -> None:
    source = open("static/js/dashboard.js", encoding="utf-8").read()
    assert "AbortController" in source
    assert "chartModule" in source
    assert "AlgorithmVaultDashboardChart" in source
    assert "EventSource" in source
    assert "scheduleReconnect" in source
    assert "renderAccountSummary" in source
    assert "updateEmptyForecast" in source
    assert "data-provider-health-status" in open("templates/dashboard.html", encoding="utf-8").read()
