from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pyotp
import pytest

import app.routes.consumer as consumer_module
import app.services.risk_engine as risk_engine_module
from app.auth import encrypt_totp_secret, password_hash
from app.extensions import db
from app.ml.decision_engine import MLDecisionEngine
from app.ml.online_ranker import ONE_H10_HORIZON, OnlineRanker, extract_features, horizon_from_context
from app.models import (
    LeveragedMarket,
    LeveragedMarketFeature,
    MLModelState,
    MLOfflineModel,
    Order,
    Setting,
    StrategyRun,
    TradingConnection,
    User,
    VaultAllocationLeg,
    VaultCycle,
    WalletBalance,
)
from app.routes.consumer import (
    _cycle_summary,
    _one_h10_live_context,
    _one_h10_ml_readiness,
    _refresh_one_h10_cycle_ml_state,
    _resume_one_h10_active_runs,
)
from app.services.hyperliquid_client import ClientSnapshot
from app.services.one_h10_quality import one_h10_forecast_live_blockers, one_h10_profitability_payload
from app.services.order_manager import OrderIntent
from app.strategies.base import Signal


def _create_user(username: str = "oneh10") -> tuple[User, str]:
    secret = pyotp.random_base32()
    user = User(username=username, password_hash=password_hash("password123"), role="user")
    user.totp_secret_encrypted = encrypt_totp_secret(secret)
    user.two_factor_enabled_at = datetime.utcnow()
    db.session.add(user)
    db.session.commit()
    return user, secret


def _login(client, username: str, secret: str) -> None:
    client.post(
        "/login",
        data={"username": username, "password": "password123", "totp_code": pyotp.TOTP(secret).now()},
        follow_redirects=False,
    )


def _connection(app, user: User, provider: str, *, active: bool = False) -> TradingConnection:
    service = app.extensions["services"]["trading_connections"]
    connection = service.create_or_update(
        user_id=user.id,
        provider=provider,
        connection_type="cex_api_key",
        api_key="key" if provider == "kucoin" else "",
        api_secret="0x" + ("1" * 64),
        passphrase="passphrase" if provider == "kucoin" else "",
        wallet_address="0x" + ("2" * 40) if provider == "hyperliquid" else "",
        is_active=False,
    )
    connection.verification_status = "verified"
    connection.is_active = active
    db.session.commit()
    return connection


def _confirm_one_h10_live(app) -> None:
    app.config["EXPLICIT_LIVE_CONFIRMED"] = True
    app.config["SECONDARY_CONFIRMATION"] = True
    Setting.set_json("explicit_live_confirmed", True)
    Setting.set_json("secondary_confirmation", True)
    db.session.commit()


def _candles(count: int = 80) -> list[dict[str, Any]]:
    rows = []
    price = 100.0
    for index in range(count):
        price += 0.2 if index % 2 == 0 else -0.1
        rows.append(
            {
                "timestamp": index,
                "open": price - 0.1,
                "high": price + 0.2,
                "low": price - 0.2,
                "close": price,
                "volume": 1000 + index,
            }
        )
    return rows


def test_one_h10_selector_uses_separate_ml_namespace(app) -> None:
    selector = app.extensions["services"]["vault_strategy_selector"]
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: {"levels": [[{"px": "99.99", "sz": "1000"}], [{"px": "100.01", "sz": "1000"}]]}
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles(limit)

    selection = selector.select("USDC", 1, "live", 100.0, allowed_symbols=["BTC"], provider="hyperliquid")

    assert selection.profile == "1H10"
    assert selection.metadata["vault_cycle_name"] == "1H10"
    assert selection.metadata["ml_horizon"] == ONE_H10_HORIZON
    assert selection.parameters["one_h10_vault"] is True
    assert selection.metadata["target_amount_usd"] == 1000.0


def test_one_h10_online_ranker_does_not_update_generic_one_hour_state(app) -> None:
    ranker = OnlineRanker({"ML_RANKER_ENABLED": True, "ML_ALLOW_LIVE_UPDATES": True})
    features = extract_features({"algorithm_profile": "1H10", "horizon": "1h", "lock_duration_hours": 1, "net_return_after_costs": 0.1})

    assert features["horizon"] == "1h10"
    assert horizon_from_context({"algorithm_profile": "1H10", "horizon": "1h"}, 1) == "1h10"

    ranker.update(features, 0.5, mode="paper", source="test", horizon=features["horizon"])

    assert MLModelState.query.filter_by(model_key="online_ranker:1h10").one()
    assert MLModelState.query.filter_by(model_key="online_ranker:1h").one_or_none() is None


def test_verified_tradable_connections_returns_all_verified_connections(app) -> None:
    user, _ = _create_user("allverified")
    hyperliquid = _connection(app, user, "hyperliquid", active=True)
    kucoin = _connection(app, user, "kucoin", active=False)

    rows = app.extensions["services"]["trading_connections"].verified_tradable_connections(user.id)

    assert {row.id for row in rows} == {hyperliquid.id, kucoin.id}


def test_leveraged_market_discovery_persists_hyperliquid_and_kucoin(app, monkeypatch) -> None:
    user, _ = _create_user("discover")
    hyperliquid = _connection(app, user, "hyperliquid", active=True)
    kucoin = _connection(app, user, "kucoin", active=False)
    service = app.extensions["services"]["trading_connections"]
    markets = app.extensions["services"]["leveraged_markets"]
    market_data = app.extensions["services"]["market_data"]
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles(limit)
    market_data.get_order_book = lambda symbol, mode: {"levels": [[{"px": "99.9", "sz": "100"}], [{"px": "100.1", "sz": "80"}]]}
    db.session.add(LeveragedMarket(provider="hyperliquid", venue_symbol="OLD", symbol="OLD", status="active"))
    db.session.commit()

    class FakeConnector:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self.rows = rows

        def discover_leveraged_markets(self, mode: str) -> list[dict[str, Any]]:
            return self.rows

    def connector_for_user(user_id: int, connection_id: int | None = None):
        if connection_id == hyperliquid.id:
            return FakeConnector(
                [
                    {
                        "name": "BTC",
                        "maxLeverage": 50,
                        "szDecimals": 3,
                        "_asset_context": {"funding": "0.0001", "markPx": "100", "openInterest": "200"},
                    },
                    {"name": "#50", "maxLeverage": 3, "szDecimals": 2, "_asset_context": {"markPx": "1", "openInterest": "100"}},
                    {"name": "@149", "maxLeverage": 3, "szDecimals": 2, "_asset_context": {"markPx": "1", "openInterest": "100"}},
                ]
            )
        if connection_id == kucoin.id:
            return FakeConnector(
                [
                    {
                        "symbol": "ETHUSDTM",
                        "baseCurrency": "ETH",
                        "quoteCurrency": "USDT",
                        "settleCurrency": "USDT",
                        "maxLeverage": 75,
                        "multiplier": "0.01",
                        "tickSize": "0.01",
                        "fundingFeeRate": "0.0002",
                        "turnoverOf24h": "1000000",
                    }
                ]
            )
        return FakeConnector([])

    monkeypatch.setattr(service, "connector_for_user", connector_for_user)

    results = markets.sync_for_user(user.id)

    assert {row["provider"] for row in results} == {"hyperliquid", "kucoin"}
    btc = LeveragedMarket.query.filter_by(provider="hyperliquid", venue_symbol="BTC").one()
    eth = LeveragedMarket.query.filter_by(provider="kucoin", venue_symbol="ETHUSDTM").one()
    assert btc.trading_connection_id == hyperliquid.id
    assert eth.trading_connection_id == kucoin.id
    assert btc.max_leverage == 50
    assert eth.settlement_asset == "USDT"
    assert LeveragedMarket.query.filter_by(provider="hyperliquid", venue_symbol="OLD").one().status == "disabled"
    assert LeveragedMarket.query.filter_by(provider="hyperliquid", venue_symbol="#50").one_or_none() is None
    assert LeveragedMarket.query.filter_by(provider="hyperliquid", venue_symbol="@149").one_or_none() is None
    assert btc.feature_rows


def test_leveraged_market_discovery_prefers_exact_venue_symbol_for_duplicate_base_symbol(app, monkeypatch) -> None:
    user, _ = _create_user("duplicatebase")
    kucoin = _connection(app, user, "kucoin", active=True)
    service = app.extensions["services"]["trading_connections"]
    markets = app.extensions["services"]["leveraged_markets"]
    db.session.add_all(
        [
            LeveragedMarket(provider="kucoin", venue_symbol="DOGEUSDTM", symbol="DOGE", status="active"),
            LeveragedMarket(provider="kucoin", venue_symbol="DOGEUSDCM", symbol="DOGE", status="active"),
        ]
    )
    db.session.commit()

    class FakeConnector:
        def discover_leveraged_markets(self, mode: str) -> list[dict[str, Any]]:
            return [
                {
                    "symbol": "DOGEUSDCM",
                    "baseCurrency": "DOGE",
                    "quoteCurrency": "USDC",
                    "settleCurrency": "USDC",
                    "maxLeverage": 20,
                    "multiplier": "10",
                    "tickSize": "0.00001",
                    "fundingFeeRate": "0.001939",
                    "turnoverOf24h": "270.7716",
                }
            ]

    monkeypatch.setattr(service, "connector_for_user", lambda user_id, connection_id=None: FakeConnector())

    result = markets.sync_for_connection(kucoin, persist_features=False)

    assert result.skipped is False
    assert result.active == 1
    assert LeveragedMarket.query.filter_by(provider="kucoin", venue_symbol="DOGEUSDCM").count() == 1
    usdc_market = LeveragedMarket.query.filter_by(provider="kucoin", venue_symbol="DOGEUSDCM").one()
    usdt_market = LeveragedMarket.query.filter_by(provider="kucoin", venue_symbol="DOGEUSDTM").one()
    assert usdc_market.status == "active"
    assert usdc_market.trading_connection_id == kucoin.id
    assert usdc_market.settlement_asset == "USDC"
    assert usdc_market.contract_size == 10.0
    assert usdt_market.status == "disabled"


def test_leveraged_market_discovery_scopes_feature_snapshots_to_allowed_symbols(app, monkeypatch) -> None:
    app.config["ONE_H10_FEATURE_TIMEFRAMES"] = ["15m"]
    user, _ = _create_user("scopedfeatures")
    _connection(app, user, "hyperliquid", active=True)
    service = app.extensions["services"]["trading_connections"]
    markets = app.extensions["services"]["leveraged_markets"]
    market_data = app.extensions["services"]["market_data"]
    candle_calls: list[str] = []
    book_calls: list[str] = []

    class FakeConnector:
        def discover_leveraged_markets(self, mode: str) -> list[dict[str, Any]]:
            return [
                {"name": "BTC", "maxLeverage": 50, "szDecimals": 3, "_asset_context": {"markPx": "100", "openInterest": "200"}},
                {"name": "ETH", "maxLeverage": 50, "szDecimals": 3, "_asset_context": {"markPx": "100", "openInterest": "200"}},
                {"name": "KPEPE", "maxLeverage": 3, "szDecimals": 0, "_asset_context": {"markPx": "0.01", "openInterest": "1000"}},
            ]

    monkeypatch.setattr(service, "connector_for_user", lambda user_id, connection_id=None: FakeConnector())

    def candles(symbol: str, timeframe: str, mode: str, limit: int, **kwargs) -> list[dict[str, Any]]:
        candle_calls.append(symbol)
        return _candles(limit)

    def book(symbol: str, mode: str, **kwargs) -> dict[str, Any]:
        book_calls.append(symbol)
        return {"levels": [[{"px": "99.9", "sz": "100"}], [{"px": "100.1", "sz": "80"}]]}

    market_data.get_candles = candles
    market_data.get_order_book = book

    results = markets.sync_for_user(user.id, feature_symbols=["BTC"])

    assert results[0]["features_attempted"] == 1
    assert {row.venue_symbol for row in LeveragedMarket.query.filter_by(provider="hyperliquid").all()} == {"BTC", "ETH", "KPEPE"}
    assert candle_calls == ["BTC"]
    assert book_calls == ["BTC"]
    assert LeveragedMarket.query.filter_by(provider="hyperliquid", venue_symbol="BTC").one().feature_rows
    assert LeveragedMarket.query.filter_by(provider="hyperliquid", venue_symbol="ETH").one().feature_rows == []
    assert LeveragedMarket.query.filter_by(provider="hyperliquid", venue_symbol="KPEPE").one().feature_rows == []


def test_hyperliquid_k_prefixed_markets_use_case_sensitive_venue_symbol_for_features(app, monkeypatch) -> None:
    app.config["ONE_H10_FEATURE_TIMEFRAMES"] = ["15m"]
    user, _ = _create_user("casefeatures")
    _connection(app, user, "hyperliquid", active=True)
    service = app.extensions["services"]["trading_connections"]
    markets = app.extensions["services"]["leveraged_markets"]
    market_data = app.extensions["services"]["market_data"]
    candle_calls: list[str] = []
    book_calls: list[str] = []
    db.session.add(LeveragedMarket(provider="hyperliquid", venue_symbol="KPEPE", symbol="KPEPE", status="active"))
    db.session.commit()

    class FakeConnector:
        def discover_leveraged_markets(self, mode: str) -> list[dict[str, Any]]:
            return [{"name": "kPEPE", "maxLeverage": 3, "szDecimals": 0, "_asset_context": {"markPx": "0.01", "openInterest": "1000"}}]

    monkeypatch.setattr(service, "connector_for_user", lambda user_id, connection_id=None: FakeConnector())
    market_data.get_candles = lambda symbol, timeframe, mode, limit, **kwargs: candle_calls.append(symbol) or _candles(limit)
    market_data.get_order_book = lambda symbol, mode, **kwargs: (
        book_calls.append(symbol) or {"levels": [[{"px": "0.0099", "sz": "100"}], [{"px": "0.0101", "sz": "80"}]]}
    )

    results = markets.sync_for_user(user.id, feature_symbols=["KPEPE"])

    assert results[0]["features_attempted"] == 1
    market = LeveragedMarket.query.filter_by(provider="hyperliquid", venue_symbol="kPEPE").one()
    assert market.venue_symbol == "kPEPE"
    assert LeveragedMarket.query.filter_by(provider="hyperliquid", venue_symbol="KPEPE").one().status == "disabled"
    assert candle_calls == ["kPEPE"]
    assert book_calls == ["kPEPE"]
    assert market.feature_rows[0].symbol == "KPEPE"


def test_one_h10_backfill_feature_scope_scans_all_markets_with_cursor(app, monkeypatch) -> None:
    app.config["ONE_H10_FEATURE_TIMEFRAMES"] = ["15m"]
    app.config["ONE_H10_FEATURE_MAX_MARKETS_PER_SYNC"] = 2
    user, _ = _create_user("allfeaturebackfill")
    _connection(app, user, "hyperliquid", active=True)
    service = app.extensions["services"]["trading_connections"]
    markets = app.extensions["services"]["leveraged_markets"]
    market_data = app.extensions["services"]["market_data"]
    candle_calls: list[str] = []

    class FakeConnector:
        def discover_leveraged_markets(self, mode: str) -> list[dict[str, Any]]:
            return [
                {"name": "BTC", "maxLeverage": 50, "szDecimals": 3, "_asset_context": {"markPx": "100", "openInterest": "1000"}},
                {"name": "DOGE", "maxLeverage": 20, "szDecimals": 0, "_asset_context": {"markPx": "1", "openInterest": "900"}},
                {"name": "XRP", "maxLeverage": 10, "szDecimals": 0, "_asset_context": {"markPx": "1", "openInterest": "800"}},
            ]

    monkeypatch.setattr(service, "connector_for_user", lambda user_id, connection_id=None: FakeConnector())
    market_data.get_candles = lambda symbol, timeframe, mode, limit, **kwargs: candle_calls.append(symbol) or _candles(limit)
    market_data.get_order_book = lambda symbol, mode, **kwargs: {"levels": [[{"px": "99.9", "sz": "100"}], [{"px": "100.1", "sz": "80"}]]}

    results = markets.sync_one_h10_backfill_for_user(user.id)

    assert results[0]["features_attempted"] == 2
    assert set(candle_calls) == {"BTC", "DOGE"}
    assert results[0]["feature_cursor"] == 0
    assert results[0]["next_feature_cursor"] == 2
    assert LeveragedMarket.query.filter_by(provider="hyperliquid").count() == 3


def test_one_h10_backfill_cli_requires_confirmation_and_syncs_all_markets(app, monkeypatch) -> None:
    app.config["ONE_H10_FEATURE_TIMEFRAMES"] = ["15m"]
    user, _ = _create_user("backfillcli")
    _connection(app, user, "hyperliquid", active=True)
    service = app.extensions["services"]["trading_connections"]
    market_data = app.extensions["services"]["market_data"]

    class FakeConnector:
        def discover_leveraged_markets(self, mode: str) -> list[dict[str, Any]]:
            return [{"name": "DOGE", "maxLeverage": 20, "szDecimals": 0, "_asset_context": {"markPx": "1", "openInterest": "900"}}]

    monkeypatch.setattr(service, "connector_for_user", lambda user_id, connection_id=None: FakeConnector())
    market_data.get_candles = lambda symbol, timeframe, mode, limit, **kwargs: _candles(limit)
    market_data.get_order_book = lambda symbol, mode, **kwargs: {"levels": [[{"px": "0.99", "sz": "100"}], [{"px": "1.01", "sz": "80"}]]}

    denied = app.test_cli_runner().invoke(args=["one-h10-backfill", "--user-id", str(user.id)])
    accepted = app.test_cli_runner().invoke(args=["one-h10-backfill", "--user-id", str(user.id), "--confirm", "BACKFILL-1H10-FEATURES"])

    assert denied.exit_code != 0
    assert accepted.exit_code == 0
    assert '"provider": "hyperliquid"' in accepted.output
    assert LeveragedMarket.query.filter_by(provider="hyperliquid", venue_symbol="DOGE").one().feature_rows


def test_leveraged_market_feature_invalid_symbol_stops_remaining_timeframes(app, monkeypatch) -> None:
    app.config["ONE_H10_FEATURE_TIMEFRAMES"] = ["15m", "1h"]
    user, _ = _create_user("invalidfeature")
    _connection(app, user, "hyperliquid", active=True)
    service = app.extensions["services"]["trading_connections"]
    markets = app.extensions["services"]["leveraged_markets"]
    market_data = app.extensions["services"]["market_data"]
    candle_calls: list[tuple[str, str]] = []

    class FakeConnector:
        def discover_leveraged_markets(self, mode: str) -> list[dict[str, Any]]:
            return [{"name": "KPEPE", "maxLeverage": 3, "szDecimals": 0, "_asset_context": {"markPx": "0.01", "openInterest": "1000"}}]

    monkeypatch.setattr(service, "connector_for_user", lambda user_id, connection_id=None: FakeConnector())

    def candles(symbol: str, timeframe: str, mode: str, limit: int, **kwargs) -> list[dict[str, Any]]:
        candle_calls.append((symbol, timeframe))
        raise KeyError(symbol)

    market_data.get_candles = candles
    market_data.get_order_book = lambda symbol, mode, **kwargs: {}

    results = markets.sync_for_user(user.id, feature_symbols=["KPEPE"])

    assert candle_calls == [("KPEPE", "15m")]
    assert results[0]["features_attempted"] == 1
    assert "hyperliquid:KPEPE:invalid_symbol" in results[0]["feature_skip_reasons"]
    assert LeveragedMarket.query.filter_by(provider="hyperliquid", venue_symbol="KPEPE").one().feature_rows == []


def test_leveraged_market_feature_rate_limit_backs_off_provider(app, monkeypatch) -> None:
    app.config["ONE_H10_FEATURE_TIMEFRAMES"] = ["15m"]
    user, _ = _create_user("ratelimitfeatures")
    _connection(app, user, "hyperliquid", active=True)
    service = app.extensions["services"]["trading_connections"]
    markets = app.extensions["services"]["leveraged_markets"]
    market_data = app.extensions["services"]["market_data"]
    candle_calls: list[str] = []

    class FakeConnector:
        def discover_leveraged_markets(self, mode: str) -> list[dict[str, Any]]:
            return [
                {"name": "BTC", "maxLeverage": 50, "szDecimals": 3, "_asset_context": {"markPx": "100", "openInterest": "200"}},
                {"name": "ETH", "maxLeverage": 50, "szDecimals": 3, "_asset_context": {"markPx": "100", "openInterest": "200"}},
            ]

    monkeypatch.setattr(service, "connector_for_user", lambda user_id, connection_id=None: FakeConnector())

    def candles(symbol: str, timeframe: str, mode: str, limit: int, **kwargs) -> list[dict[str, Any]]:
        candle_calls.append(symbol)
        raise RuntimeError("429 too many requests")

    market_data.get_candles = candles
    market_data.get_order_book = lambda symbol, mode, **kwargs: {}

    results = markets.sync_for_user(user.id, feature_symbols=["BTC", "ETH"])

    assert candle_calls == ["BTC"]
    assert results[0]["features_attempted"] == 1
    assert "hyperliquid:rate_limited_feature_backoff" in results[0]["feature_skip_reasons"]
    assert "hyperliquid:feature_backoff_active" in results[0]["feature_skip_reasons"]


def test_hyperliquid_public_market_data_can_disable_retries(app, monkeypatch) -> None:
    client = app.extensions["services"]["hyperliquid_client"]
    app.config["EXCHANGE_RETRY_ATTEMPTS"] = 3

    class FakeInfo:
        calls = 0

        def candles_snapshot(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> list[dict[str, Any]]:
            self.calls += 1
            raise KeyError(symbol)

    fake = FakeInfo()
    monkeypatch.setattr(client, "_get_public_info", lambda mode: fake)

    with pytest.raises(KeyError):
        client.get_candles("live", "KPEPE", "15m", 1, 2, retry=False)

    assert fake.calls == 1


def _feature(market: LeveragedMarket, *, score: float = 1.0, rsi: float = 55.0, liquidity: float = 100_000.0) -> LeveragedMarketFeature:
    feature = LeveragedMarketFeature(
        leveraged_market=market,
        provider=market.provider,
        symbol=market.symbol,
        timeframe="15m",
    )
    feature.features = {
        "ml_namespace": "1h10",
        "rsi": rsi,
        "ema_fast": 101.0,
        "ema_slow": 100.0,
        "sma_fast": 100.5,
        "sma_slow": 99.5,
        "ema_trend": score,
        "trend_strength": score / 100.0,
        "macd_histogram": score,
        "volatility": 0.01,
        "liquidity_usd": liquidity,
        "spread_bps": 2.0,
        "funding_rate": 0.0,
        "order_book_imbalance": 0.2,
        "fibonacci_confluence": {"score": score, "cluster_count": 5},
        "fibonacci_levels": {"retracements": {"61.8": 100.0}},
        "fibonacci_timing": {"fib_time_13": 1},
        "volume_spike": {"ratio": 2.0},
    }
    db.session.add(feature)
    return feature


def test_one_h10_scanner_rejects_zero_spread_features(app) -> None:
    scanner = app.extensions["services"]["market_scanner"]
    market = LeveragedMarket(
        provider="hyperliquid",
        venue_symbol="BTC",
        symbol="BTC",
        status="active",
        settlement_asset="USDC",
        max_leverage=50,
        liquidity_usd=250_000,
        spread_bps=0.0,
    )
    db.session.add(market)
    db.session.flush()
    feature = _feature(market, liquidity=250_000)
    payload = dict(feature.features)
    payload["spread_bps"] = 0.0
    feature.features = payload
    db.session.commit()

    scored = scanner.score_one_h10_markets([market], provider="hyperliquid")

    assert scored == []
    assert scanner.last_scan_diagnostics["rejection_breakdown"] == {"spread_missing": 1}


def test_one_h10_ml_training_rows_use_persisted_feature_namespace(app) -> None:
    app.config["ML_ONE_H10_MIN_TRAINING_ROWS"] = 2
    btc = LeveragedMarket(
        provider="hyperliquid",
        venue_symbol="BTC",
        symbol="BTC",
        status="active",
        settlement_asset="USDC",
        max_leverage=50,
        liquidity_usd=100_000,
        spread_bps=2.0,
    )
    eth = LeveragedMarket(
        provider="kucoin",
        venue_symbol="ETHUSDTM",
        symbol="ETH",
        status="active",
        settlement_asset="USDT",
        max_leverage=20,
        liquidity_usd=250_000,
        spread_bps=2.0,
    )
    db.session.add_all([btc, eth])
    db.session.flush()
    _feature(btc, score=1.2)
    _feature(eth, score=0.8)
    db.session.commit()

    engine = MLDecisionEngine(app.config)
    rows = engine.training_rows("pytorch_fibonacci", "1h10", objective="one_h10")

    assert len(rows) == 2
    assert {row.provider for row in rows} == {"hyperliquid", "kucoin"}
    assert all(row.source == "one_h10_feature_bootstrap:pytorch_fibonacci" for row in rows)
    assert engine._objective("one_h10") == "one_h10"


def test_one_h10_provider_decision_can_use_global_promoted_model(app, tmp_path) -> None:
    app.config["ML_ALL_AREAS_ENABLED"] = True
    app.config["ML_FIBONACCI_MODEL_ENABLED"] = True
    app.config["ML_REQUIRE_PROMOTED_FOR_LIVE"] = True
    artifact = tmp_path / "global-one-h10.pt"
    artifact.write_text("{}", encoding="utf-8")
    db.session.add(
        MLOfflineModel(
            model_key="ml_suite:global:1h10:pytorch_fibonacci:test",
            provider="global",
            horizon="1h10",
            model_type="pytorch_fibonacci",
            status="promoted",
            feature_schema_version="ml_decision_v1",
            artifact_path=str(artifact),
            training_rows=20,
            validation_rows=5,
            validation_loss=0.01,
            metrics={"false_positive_rate": 0.0, "drift": 0.0},
            promoted_at=datetime.utcnow(),
        )
    )
    db.session.commit()

    engine = MLDecisionEngine(app.config)
    readiness = engine.family_readiness("pytorch_fibonacci", "1h10", provider="kucoin")

    assert readiness["ready"] is True
    assert readiness["provider"] == "kucoin"
    assert readiness["promoted_model"]["provider"] == "global"


def test_one_h10_scanner_ranks_all_persisted_markets_not_allowed_symbols_only(app) -> None:
    app.config["ALLOWED_SYMBOLS"] = ["BTC"]
    btc = LeveragedMarket(
        provider="hyperliquid",
        venue_symbol="BTC",
        symbol="BTC",
        status="active",
        settlement_asset="USDC",
        max_leverage=50,
        liquidity_usd=100_000,
        spread_bps=2.0,
    )
    doge = LeveragedMarket(
        provider="hyperliquid",
        venue_symbol="DOGE",
        symbol="DOGE",
        status="active",
        settlement_asset="USDC",
        max_leverage=20,
        liquidity_usd=250_000,
        spread_bps=2.0,
    )
    db.session.add_all([btc, doge])
    db.session.flush()
    _feature(btc, score=0.5, liquidity=100_000)
    _feature(doge, score=3.0, liquidity=250_000)
    db.session.commit()

    ranked = app.extensions["services"]["market_scanner"].score_one_h10_markets([btc, doge], provider="hyperliquid", limit=2)

    assert [candidate.symbol for candidate in ranked] == ["DOGE", "BTC"]
    assert ranked[0].features["ml_horizon"] == "1h10"
    assert ranked[0].features["fibonacci_levels"]["retracements"]["61.8"] == 100.0


def test_one_h10_provider_legs_skip_forecast_blocked_candidates(app, monkeypatch) -> None:
    user, _ = _create_user("blockedlegs")
    connection = _connection(app, user, "hyperliquid", active=True)
    market = LeveragedMarket(
        provider="hyperliquid",
        venue_symbol="DOGE",
        symbol="DOGE",
        status="active",
        settlement_asset="USDC",
        max_leverage=20,
        liquidity_usd=250_000,
        spread_bps=4.0,
        trading_connection_id=connection.id,
    )
    db.session.add(market)
    db.session.flush()
    feature = _feature(market, score=0.2, liquidity=250_000)
    feature.features = {
        **feature.features,
        "spread_bps": 4.0,
        "cost_drag_bps": 0.0,
        "one_h10_feature_updated_at": datetime.utcnow().isoformat(),
    }
    db.session.commit()
    service = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(
        service,
        "account_snapshot",
        lambda user_id, mode, connection_id=None: ClientSnapshot(
            mode,
            [{"asset": "USDC", "type": "margin", "value": 25.0, "withdrawable": 25.0}],
            [],
            [],
            [],
            [],
        ),
    )
    selection = SimpleNamespace(
        strategy_name="scalping",
        symbol="DOGE",
        timeframe="1m",
        parameters={"leverage": 1.0},
        metadata={"target_roi_pct": 1000.0},
        legs=[
            {
                "strategy_name": "scalping",
                "symbol": "DOGE",
                "timeframe": "1m",
                "parameters": {"leverage": 1.0},
                "allocation_cap_usd": 10.0,
                "leverage": 1.0,
            }
        ],
    )

    legs, history, blockers = consumer_module._one_h10_provider_legs(
        user=user,
        selection=selection,
        connections=[connection],
        starting_value_usd=10.0,
        settlement_asset="USDC",
        allowed_symbols=[],
        connection_blockers=[],
    )

    assert legs == []
    assert history[0]["legs"][0]["skip_reason"].startswith("one_h10_forecast_blocked:")
    assert any("cost_drag_above_threshold" in str(row.get("reason")) for row in blockers)


def test_one_h10_provider_legs_weight_accepted_forecasts_by_allocation_score(app, monkeypatch) -> None:
    user, _ = _create_user("allocationweights")
    connection = _connection(app, user, "hyperliquid", active=True)
    app.config["ONE_H10_MAX_PROVIDER_LEGS"] = 3
    markets = [
        LeveragedMarket(
            provider="hyperliquid",
            venue_symbol=symbol,
            symbol=symbol,
            status="active",
            settlement_asset="USDC",
            max_leverage=20,
            liquidity_usd=1_000_000,
            spread_bps=1.0,
            trading_connection_id=connection.id,
        )
        for symbol in ("DOGE", "BTC", "SKIP")
    ]
    db.session.add_all(markets)
    db.session.flush()
    for market in markets:
        feature = _feature(market, score=1.0, liquidity=1_000_000)
        feature.features = {
            **feature.features,
            "spread_bps": 1.0,
            "cost_drag_bps": 4.0,
            "one_h10_feature_updated_at": datetime.utcnow().isoformat(),
        }
    db.session.commit()
    service = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(
        service,
        "account_snapshot",
        lambda user_id, mode, connection_id=None: ClientSnapshot(
            mode,
            [{"asset": "USDC", "type": "margin", "value": 100.0, "withdrawable": 100.0}],
            [],
            [],
            [],
            [],
        ),
    )

    class AllocationForecast:
        scores = {"DOGE": 9.0, "BTC": 1.0, "SKIP": 0.01}

        def forecast(self, features, *, provider, symbol, allocation_cap_usd=0.0, available_margin_usd=0.0, market=None):
            score = self.scores[str(symbol)]
            return {
                "predicted_side": "buy",
                "action": "buy",
                "confidence": 0.9,
                "expected_return_bps": 80.0,
                "gross_expected_return_bps": 90.0,
                "net_expected_return_bps": 72.0,
                "execution_adjusted_net_return_bps": 64.8,
                "cost_drag_bps": 8.0,
                "estimated_slippage_bps": 1.0,
                "spread_bps": 1.0,
                "execution_quality": 0.9,
                "expected_execution_quality": 0.9,
                "capital_efficiency_score": 1.0,
                "risk_reward": 3.0,
                "expected_net_edge_passed": True,
                "profitability_score": 0.9 if symbol != "SKIP" else 0.1,
                "allocation_score": score,
                "suggested_notional_usd": min(float(allocation_cap_usd or 0.0), float(available_margin_usd or 0.0)),
                "suggested_leverage": 1.0,
                "suggested_order_type": "limit",
                "suggested_stop_loss_pct": 0.01,
                "suggested_take_profit_pct": 0.03,
                "blockers": ["low_profitability_score"] if symbol == "SKIP" else [],
                "advisory_blockers": [],
                "ml_namespace": "1h10",
                "ml_horizon": "1h10",
                "source": "one_h10_ml_profit_suite",
                "ml_ready": True,
            }

    monkeypatch.setitem(app.extensions["services"], "one_h10_forecast", AllocationForecast())
    selection = SimpleNamespace(
        strategy_name="scalping",
        symbol="DOGE",
        timeframe="1m",
        parameters={"leverage": 1.0},
        metadata={"target_roi_pct": 1000.0},
        legs=[
            {
                "strategy_name": "scalping",
                "symbol": "DOGE",
                "timeframe": "1m",
                "parameters": {"leverage": 1.0},
                "allocation_cap_usd": 100.0,
                "leverage": 1.0,
            }
        ],
    )

    legs, history, blockers = consumer_module._one_h10_provider_legs(
        user=user,
        selection=selection,
        connections=[connection],
        starting_value_usd=100.0,
        settlement_asset="USDC",
        allowed_symbols=[],
        connection_blockers=[],
    )

    allocations = {leg["symbol"]: leg["allocation_cap_usd"] for leg in legs}
    assert set(allocations) == {"DOGE", "BTC"}
    assert allocations["DOGE"] == pytest.approx(90.0)
    assert allocations["BTC"] == pytest.approx(10.0)
    assert all(leg["allocation_score"] > 0 for leg in legs)
    assert any(row["skip_reason"].endswith("low_profitability_score") for row in history[0]["legs"] if row.get("symbol") == "SKIP")
    assert any("low_profitability_score" in str(row.get("reason")) for row in blockers)


def test_one_h10_scanner_prefers_net_edge_after_costs(app) -> None:
    app.config["ONE_H10_MAX_SLIPPAGE_BPS"] = 30.0
    cheap = LeveragedMarket(
        provider="hyperliquid",
        venue_symbol="CHEAP",
        symbol="CHEAP",
        status="active",
        settlement_asset="USDC",
        max_leverage=20,
        liquidity_usd=250_000,
        spread_bps=0.5,
    )
    costly = LeveragedMarket(
        provider="hyperliquid",
        venue_symbol="COSTLY",
        symbol="COSTLY",
        status="active",
        settlement_asset="USDC",
        max_leverage=20,
        liquidity_usd=250_000,
        spread_bps=16.0,
    )
    db.session.add_all([cheap, costly])
    db.session.flush()
    cheap_feature = _feature(cheap, score=2.6, liquidity=250_000)
    costly_feature = _feature(costly, score=3.0, liquidity=250_000)
    cheap_feature.features = {**cheap_feature.features, "spread_bps": 0.5}
    costly_feature.features = {**costly_feature.features, "spread_bps": 16.0}
    db.session.commit()

    ranked = app.extensions["services"]["market_scanner"].score_one_h10_markets([cheap, costly], provider="hyperliquid", limit=2)

    assert [candidate.symbol for candidate in ranked] == ["CHEAP", "COSTLY"]
    assert ranked[0].features["net_expected_return_bps"] > 0
    assert ranked[0].features["cost_drag_bps"] < ranked[1].features["cost_drag_bps"]
    assert ranked[0].features["expected_execution_quality"] > ranked[1].features["expected_execution_quality"]
    assert ranked[0].features["target_progress"] >= ranked[1].features["target_progress"]
    assert ranked[0].features["risk_reward"] > ranked[1].features["risk_reward"]
    assert ranked[0].features["profitability_score"] > ranked[1].features["profitability_score"]
    assert ranked[0].features["allocation_score"] > ranked[1].features["allocation_score"]
    assert "net_expected_edge" in ranked[0].score_breakdown
    assert "profitability_score" in ranked[0].score_breakdown
    assert "allocation_score" in ranked[0].score_breakdown
    assert "cost_drag_penalty" in ranked[0].score_breakdown
    assert "target_progress" in ranked[0].score_breakdown


def test_one_h10_low_profitability_score_blocks_live_forecast(app) -> None:
    forecast = {
        "predicted_side": "buy",
        "action": "buy",
        "confidence": 0.9,
        "net_expected_return_bps": 6.0,
        "execution_adjusted_net_return_bps": 0.6,
        "cost_drag_bps": 10.0,
        "estimated_slippage_bps": 1.0,
        "spread_bps": 1.0,
        "execution_quality": 0.20,
        "expected_execution_quality": 0.20,
        "risk_reward": 0.5,
        "capital_efficiency_score": 0.1,
        "suggested_notional_usd": 10.0,
        "suggested_stop_loss_pct": 0.02,
        "suggested_take_profit_pct": 0.01,
        "created_at": datetime.utcnow().isoformat(),
        "blockers": [],
        "advisory_blockers": [],
    }
    forecast.update(one_h10_profitability_payload(forecast, app.config))

    blockers = one_h10_forecast_live_blockers(forecast, app.config)

    assert forecast["profitability_score"] < app.config["ONE_H10_MIN_PROFITABILITY_SCORE"]
    assert "low_profitability_score" in blockers


def test_one_h10_forecast_service_uses_1h10_namespace_and_bootstrap_blocker(app) -> None:
    service = app.extensions["services"]["one_h10_forecast"]
    forecast = service.forecast(
        {
            "symbol": "DOGE",
            "close": 100.0,
            "rsi": 52.0,
            "ema_trend": 1.0,
            "trend_strength": 0.02,
            "macd_histogram": 1.0,
            "volatility": 0.01,
            "spread_bps": 2.0,
            "order_book_imbalance": 0.4,
            "fibonacci_confluence": {"score": 1.5},
            "fibonacci_timing": {"range_position": 0.35},
        },
        provider="hyperliquid",
        symbol="DOGE",
        allocation_cap_usd=100.0,
        available_margin_usd=125.0,
    )

    assert forecast["ml_namespace"] == "1h10"
    assert forecast["ml_horizon"] == "1h10"
    assert forecast["objective"] == "one_h10"
    assert forecast["horizon_seconds"] == app.config["ONE_H10_HORIZON_SECONDS"]
    assert forecast["predicted_side"] in {"buy", "sell"}
    assert forecast["suggested_stop_loss_pct"] > 0
    assert forecast["suggested_take_profit_pct"] > 0
    assert forecast["estimated_fee_bps"] >= 0
    assert forecast["estimated_slippage_bps"] >= 0
    assert forecast["expected_execution_quality"] == pytest.approx(forecast["execution_quality"])
    assert forecast["target_multiplier"] == pytest.approx(10.0)
    assert 0 <= forecast["target_progress"] <= 1
    assert forecast["after_cost_pnl_estimate_usd"] == pytest.approx(
        forecast["suggested_notional_usd"] * forecast["net_expected_return_bps"] / 10_000.0
    )
    assert forecast["risk_reward"] >= app.config["ONE_H10_MIN_RISK_REWARD"]
    assert forecast["confidence_kind"] == "heuristic_confidence"
    assert forecast["confidence_calibrated"] is False
    assert forecast["decision_reason_code"]
    assert "ml_not_ready" in forecast["blockers"]
    assert "features_stale" in forecast["blockers"]
    assert "missing_fibonacci_features" in forecast["blockers"]


def test_one_h10_bootstrap_treats_low_confidence_as_advisory(app) -> None:
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    app.config["ONE_H10_MIN_FORECAST_CONFIDENCE"] = 0.50
    app.config["ONE_H10_MIN_BOOTSTRAP_CONFIDENCE"] = 0.03
    app.config["ONE_H10_DIRECTIONAL_THRESHOLD"] = 0.04
    app.config["ONE_H10_MIN_POSITION_FRACTION"] = 0.25
    service = app.extensions["services"]["one_h10_forecast"]

    forecast = service.forecast(
        {
            "symbol": "DOGE",
            "close": 100.0,
            "rsi": 52.0,
            "ema_trend": 0.1,
            "trend_strength": 0.02,
            "macd_histogram": 0.05,
            "volatility": 0.01,
            "spread_bps": 2.0,
            "order_book_imbalance": 0.05,
            "one_h10_feature_timeframes": ["15m", "1h", "4h"],
            "fibonacci_confluence": {"score": 0.2},
            "fibonacci_levels": {"retracements": {"61.8": 100.0}},
            "fibonacci_timing": {"range_position": 0.35},
        },
        provider="hyperliquid",
        symbol="DOGE",
        allocation_cap_usd=100.0,
        available_margin_usd=100.0,
    )

    assert forecast["predicted_side"] == "buy"
    assert 0 < forecast["position_fraction"] < 0.25
    assert "low_confidence" not in forecast["blockers"]
    assert "low_confidence" in forecast["advisory_blockers"]
    assert "ml_not_ready" in forecast["advisory_blockers"]


def test_one_h10_forecast_cost_adjusts_bootstrap_confidence_and_sizing(app) -> None:
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    service = app.extensions["services"]["one_h10_forecast"]
    base_features = {
        "symbol": "DOGE",
        "close": 100.0,
        "rsi": 54.0,
        "ema_trend": 0.25,
        "trend_strength": 0.025,
        "macd_histogram": 0.10,
        "volatility": 0.004,
        "order_book_imbalance": 0.10,
        "one_h10_feature_timeframes": ["15m", "1h", "4h"],
        "fibonacci_confluence": {"score": 0.8},
        "fibonacci_levels": {"retracements": {"61.8": 100.0}},
        "fibonacci_timing": {"range_position": 0.35},
        "liquidity_usd": 500_000.0,
    }

    cheap = service.forecast(
        {**base_features, "spread_bps": 0.5, "cost_drag_bps": 6.0},
        provider="hyperliquid",
        symbol="DOGE",
        allocation_cap_usd=100.0,
        available_margin_usd=100.0,
    )
    costly = service.forecast(
        {**base_features, "spread_bps": 18.0, "cost_drag_bps": 48.0},
        provider="hyperliquid",
        symbol="DOGE",
        allocation_cap_usd=100.0,
        available_margin_usd=100.0,
    )

    assert cheap["confidence"] > costly["confidence"]
    assert cheap["position_fraction"] > costly["position_fraction"]
    assert cheap["net_expected_return_bps"] > costly["net_expected_return_bps"]
    assert costly["suggested_order_type"] == "limit"
    assert "cost_drag_above_threshold" in costly["blockers"]
    assert "low_edge_after_costs" in costly["blockers"]


def test_one_h10_aggressive_sizing_requires_profitable_execution_quality(app) -> None:
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    app.config["MAX_LEVERAGE"] = 4.0
    app.config["ONE_H10_MAX_LEVERAGE"] = 4.0
    app.config["ONE_H10_MAX_COST_DRAG_BPS"] = 25.0
    app.config["ONE_H10_MAX_POSITION_FRACTION"] = 0.75
    service = app.extensions["services"]["one_h10_forecast"]
    base_features = {
        "symbol": "DOGE",
        "close": 100.0,
        "rsi": 54.0,
        "ema_trend": 2.0,
        "trend_strength": 0.05,
        "macd_histogram": 0.75,
        "volatility": 0.02,
        "order_book_imbalance": 0.35,
        "one_h10_feature_timeframes": ["15m", "1h", "4h"],
        "fibonacci_confluence": {"score": 1.2},
        "fibonacci_levels": {"retracements": {"61.8": 100.0}},
        "fibonacci_timing": {"range_position": 0.35},
        "liquidity_usd": 2_000_000.0,
        "max_leverage": 4.0,
    }

    strong = service.forecast(
        {**base_features, "spread_bps": 0.5, "cost_drag_bps": 4.0},
        provider="hyperliquid",
        symbol="DOGE",
        allocation_cap_usd=100.0,
        available_margin_usd=100.0,
    )
    weak = service.forecast(
        {**base_features, "spread_bps": 19.0, "cost_drag_bps": 80.0},
        provider="hyperliquid",
        symbol="DOGE",
        allocation_cap_usd=100.0,
        available_margin_usd=100.0,
    )

    assert strong["position_fraction"] > 0.25
    assert strong["position_fraction"] <= 0.75
    assert strong["suggested_leverage"] > 1.0
    assert strong["profitability_score"] >= app.config["ONE_H10_MIN_PROFITABILITY_SCORE"]
    assert weak["position_fraction"] <= 0.25
    assert weak["suggested_leverage"] == 1.0
    assert "cost_drag_above_threshold" in weak["blockers"]


def test_one_h10_forecast_rebuilds_zero_stored_cost_drag(app) -> None:
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    service = app.extensions["services"]["one_h10_forecast"]

    forecast = service.forecast(
        {
            "symbol": "PENDLE",
            "close": 100.0,
            "rsi": 54.0,
            "ema_trend": 0.3,
            "trend_strength": 0.02,
            "macd_histogram": 0.10,
            "volatility": 0.004,
            "order_book_imbalance": 0.10,
            "spread_bps": 7.0,
            "cost_drag_bps": 0.0,
            "one_h10_feature_timeframes": ["15m", "1h", "4h"],
            "fibonacci_confluence": {"score": 0.8},
            "fibonacci_levels": {"retracements": {"61.8": 100.0}},
            "fibonacci_timing": {"range_position": 0.35},
            "liquidity_usd": 500_000.0,
        },
        provider="hyperliquid",
        symbol="PENDLE",
        allocation_cap_usd=10.0,
        available_margin_usd=10.0,
    )

    assert forecast["cost_drag_bps"] == pytest.approx(25.0)
    assert forecast["net_expected_return_bps"] < forecast["gross_expected_return_bps"]
    assert "cost_drag_above_threshold" in forecast["blockers"]


def test_one_h10_forecast_marks_stale_persisted_features(app) -> None:
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    app.config["ONE_H10_MAX_FEATURE_AGE_SECONDS"] = 60
    service = app.extensions["services"]["one_h10_forecast"]

    forecast = service.forecast(
        {
            "symbol": "DOGE",
            "close": 100.0,
            "rsi": 54.0,
            "ema_trend": 0.3,
            "trend_strength": 0.02,
            "macd_histogram": 0.10,
            "volatility": 0.004,
            "order_book_imbalance": 0.10,
            "spread_bps": 0.5,
            "one_h10_feature_timeframes": ["15m", "1h", "4h"],
            "one_h10_feature_updated_at": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            "fibonacci_confluence": {"score": 0.8},
            "fibonacci_levels": {"retracements": {"61.8": 100.0}},
            "fibonacci_timing": {"range_position": 0.35},
            "liquidity_usd": 500_000.0,
        },
        provider="hyperliquid",
        symbol="DOGE",
        allocation_cap_usd=10.0,
        available_margin_usd=10.0,
    )

    assert "features_stale" in forecast["advisory_blockers"]
    assert "features_stale" in consumer_module._one_h10_forecast_live_blockers(forecast)


class _OneH10ForecastSuite:
    calls: list[tuple[str, str]] = []

    def decision(self, family: str, context: dict[str, Any], *, horizon: str = "1h", candles=None) -> dict[str, Any]:
        self.calls.append((family, horizon))
        raw_by_family = {
            "pytorch_fibonacci": {
                "target_zone_quality": 0.72,
                "suggested_stop_loss_pct": 0.012,
                "suggested_take_profit_pct": 0.04,
            },
            "pytorch_roi_target": {
                "target_probability": 0.82,
                "projected_roi_pct": 720.0,
                "distance_to_target_pct": 280.0,
            },
            "pytorch_extreme_upside": {
                "extreme_upside_probability": 0.88,
                "projected_roi_pct": 840.0,
                "suggested_notional_usdc": 80.0,
                "suggested_leverage": 2.7,
            },
            "pytorch_cap_policy": {
                "suggested_notional_usdc": 75.0,
                "suggested_leverage": 2.5,
            },
            "pytorch_exit_policy": {
                "suggested_stop_loss_pct": 0.018,
                "suggested_take_profit_pct": 0.16,
                "blockers": [],
            },
            "pytorch_execution_policy": {
                "order_type_suggestion": "limit",
                "maker_taker_preference": "maker",
            },
            "pytorch_risk_policy": {
                "approve": True,
                "confidence": 0.8,
            },
            "pytorch_optimizer_policy": {
                "optimizer_policy_score": 0.75,
                "skip_candidate": False,
            },
        }
        confidence = 0.8 if family != "pytorch_fibonacci" else 0.72
        return {
            "ready": True,
            "family": family,
            "action": "suggest",
            "confidence": confidence,
            "expected_return": 0.8,
            "blockers": [],
            "raw": raw_by_family.get(family, {}),
        }


class _OneH10DisagreementSuite:
    calls: list[tuple[str, str]]

    def __init__(self, *, conflicting: bool) -> None:
        self.conflicting = conflicting
        self.calls = []

    def decision(self, family: str, context: dict[str, Any], *, horizon: str = "1h", candles=None) -> dict[str, Any]:
        self.calls.append((family, horizon))
        if self.conflicting:
            raw_by_family = {
                "pytorch_fibonacci": {
                    "target_zone_quality": 0.90,
                    "predicted_side": "buy",
                    "suggested_stop_loss_pct": 0.012,
                    "suggested_take_profit_pct": 0.06,
                },
                "pytorch_roi_target": {"target_probability": 0.90, "projected_roi_pct": 800.0, "predicted_side": "buy"},
                "pytorch_extreme_upside": {
                    "extreme_upside_probability": 0.10,
                    "projected_roi_pct": 900.0,
                    "predicted_side": "sell",
                    "uncertainty": 0.85,
                },
                "pytorch_cap_policy": {"suggested_notional_usdc": 80.0, "suggested_leverage": 3.0},
                "pytorch_exit_policy": {"suggested_stop_loss_pct": 0.012, "suggested_take_profit_pct": 0.08},
                "pytorch_execution_policy": {"order_type_suggestion": "limit"},
                "pytorch_risk_policy": {"approve": True, "confidence": 0.20, "uncertainty": 0.85},
                "pytorch_optimizer_policy": {"optimizer_policy_score": 0.10, "skip_candidate": False},
            }
            confidence = 0.20 if family in {"pytorch_risk_policy", "pytorch_extreme_upside", "pytorch_optimizer_policy"} else 0.90
            uncertainty = 0.85 if family in {"pytorch_risk_policy", "pytorch_extreme_upside"} else 0.10
        else:
            raw_by_family = {
                "pytorch_fibonacci": {
                    "target_zone_quality": 0.90,
                    "predicted_side": "buy",
                    "suggested_stop_loss_pct": 0.012,
                    "suggested_take_profit_pct": 0.06,
                },
                "pytorch_roi_target": {"target_probability": 0.90, "projected_roi_pct": 800.0, "predicted_side": "buy"},
                "pytorch_extreme_upside": {"extreme_upside_probability": 0.88, "projected_roi_pct": 900.0, "predicted_side": "buy"},
                "pytorch_cap_policy": {"suggested_notional_usdc": 80.0, "suggested_leverage": 3.0},
                "pytorch_exit_policy": {"suggested_stop_loss_pct": 0.012, "suggested_take_profit_pct": 0.08},
                "pytorch_execution_policy": {"order_type_suggestion": "limit"},
                "pytorch_risk_policy": {"approve": True, "confidence": 0.90},
                "pytorch_optimizer_policy": {"optimizer_policy_score": 0.85, "skip_candidate": False},
            }
            confidence = 0.90
            uncertainty = 0.05
        return {
            "ready": True,
            "family": family,
            "action": "suggest",
            "confidence": confidence,
            "expected_return": 0.8,
            "uncertainty": uncertainty,
            "blockers": [],
            "raw": raw_by_family.get(family, {}),
        }


class _PassingOneH10Forecast:
    def forecast(
        self,
        features: dict[str, Any],
        *,
        provider: str,
        symbol: str,
        allocation_cap_usd: float = 0.0,
        available_margin_usd: float = 0.0,
        market: Any = None,
    ) -> dict[str, Any]:
        suggested_notional = min(
            value
            for value in [
                float(allocation_cap_usd or 5.0),
                float(available_margin_usd or allocation_cap_usd or 5.0),
                5.0,
            ]
            if value > 0
        )
        return {
            "predicted_side": "buy",
            "action": "buy",
            "confidence": 0.82,
            "expected_return_bps": 42.0,
            "gross_expected_return_bps": 54.0,
            "net_expected_return_bps": 28.0,
            "execution_adjusted_net_return_bps": 25.2,
            "cost_drag_bps": 8.0,
            "spread_bps": 1.0,
            "execution_quality": 0.9,
            "expected_execution_quality": 0.9,
            "capital_efficiency_score": 1.0,
            "expected_net_edge_passed": True,
            "risk_reward": 3.0,
            "profitability_score": 0.85,
            "allocation_score": 0.85,
            "target_progress": 0.01,
            "target_gap_pct": 99.0,
            "after_cost_pnl_estimate_usd": suggested_notional * 25.2 / 10_000.0,
            "suggested_notional_usd": suggested_notional,
            "suggested_leverage": 1.0,
            "suggested_order_type": "limit",
            "suggested_stop_loss_pct": 0.01,
            "suggested_take_profit_pct": 0.03,
            "directional_score": 0.6,
            "blockers": [],
            "advisory_blockers": [],
            "ml_namespace": "1h10",
            "ml_horizon": "1h10",
            "source": "one_h10_ml_profit_suite",
            "ml_ready": True,
            "ml_decision": {},
            "ml_policy_decisions": {},
            "provider": provider,
            "symbol": symbol,
        }


def test_one_h10_forecast_uses_profit_ml_suite_for_sizing_exits_and_execution(app) -> None:
    app.config["ONE_H10_ML_FORECAST_FAMILIES"] = [
        "pytorch_fibonacci",
        "pytorch_roi_target",
        "pytorch_extreme_upside",
        "pytorch_cap_policy",
        "pytorch_exit_policy",
        "pytorch_execution_policy",
        "pytorch_risk_policy",
        "pytorch_optimizer_policy",
    ]
    app.config["MAX_LEVERAGE"] = 3.0
    app.config["ONE_H10_MAX_LEVERAGE"] = 3.0
    app.config["ONE_H10_MAX_COST_DRAG_BPS"] = 25.0
    engine = _OneH10ForecastSuite()
    service = app.extensions["services"]["one_h10_forecast"]
    service.ml_decision_engine = engine

    forecast = service.forecast(
        {
            "symbol": "DOGE",
            "close": 100.0,
            "rsi": 50.0,
            "ema_trend": 0.15,
            "trend_strength": 0.02,
            "macd_histogram": 0.05,
            "volatility": 0.01,
            "spread_bps": 3.0,
            "max_leverage": 3.0,
            "order_book_imbalance": 0.1,
            "one_h10_feature_timeframes": ["15m", "1h", "4h"],
            "fibonacci_confluence": {"score": 0.5},
            "fibonacci_levels": {"retracements": {"61.8": 100.0}},
            "fibonacci_timing": {"range_position": 0.35},
        },
        provider="hyperliquid",
        symbol="DOGE",
        allocation_cap_usd=100.0,
        available_margin_usd=90.0,
    )

    assert forecast["source"] == "one_h10_ml_profit_suite"
    assert forecast["ml_ready"] is True
    assert forecast["ml_horizon"] == "1h10"
    assert forecast["predicted_side"] == "buy"
    assert forecast["suggested_leverage"] >= 2.5
    assert 0 < forecast["suggested_notional_usd"] < 80.0
    assert forecast["suggested_take_profit_pct"] == 0.16
    assert forecast["suggested_order_type"] == "limit"
    assert forecast["ml_profit_score"] > 0.4
    assert forecast["net_expected_return_bps"] < forecast["gross_expected_return_bps"]
    assert "cost_drag_above_threshold" not in forecast["decision_blockers"]
    assert forecast["roi_target_probability"] == 0.82
    assert forecast["extreme_upside_probability"] == 0.88
    assert all(horizon == "1h10" for _, horizon in engine.calls)
    assert {"pytorch_roi_target", "pytorch_extreme_upside", "pytorch_exit_policy"}.issubset({family for family, _ in engine.calls})


def test_one_h10_forecast_scales_sizing_when_ml_families_disagree(app) -> None:
    app.config["ONE_H10_ML_FORECAST_FAMILIES"] = [
        "pytorch_fibonacci",
        "pytorch_roi_target",
        "pytorch_extreme_upside",
        "pytorch_cap_policy",
        "pytorch_exit_policy",
        "pytorch_execution_policy",
        "pytorch_risk_policy",
        "pytorch_optimizer_policy",
    ]
    app.config["MAX_LEVERAGE"] = 3.0
    app.config["ONE_H10_MAX_LEVERAGE"] = 3.0
    app.config["ONE_H10_ML_EXPECTED_EDGE_CAP_BPS"] = 500.0
    service = app.extensions["services"]["one_h10_forecast"]
    features = {
        "symbol": "DOGE",
        "close": 100.0,
        "rsi": 50.0,
        "ema_trend": 0.15,
        "trend_strength": 0.02,
        "macd_histogram": 0.05,
        "volatility": 0.01,
        "spread_bps": 1.0,
        "max_leverage": 3.0,
        "order_book_imbalance": 0.1,
        "one_h10_feature_timeframes": ["15m", "1h", "4h"],
        "fibonacci_confluence": {"score": 0.8},
        "fibonacci_levels": {"retracements": {"61.8": 100.0}},
        "fibonacci_timing": {"range_position": 0.35},
        "liquidity_usd": 1_000_000.0,
    }

    service.ml_decision_engine = _OneH10DisagreementSuite(conflicting=False)
    aligned = service.forecast(
        features,
        provider="hyperliquid",
        symbol="DOGE",
        allocation_cap_usd=100.0,
        available_margin_usd=100.0,
    )
    service.ml_decision_engine = _OneH10DisagreementSuite(conflicting=True)
    conflicting = service.forecast(
        features,
        provider="hyperliquid",
        symbol="DOGE",
        allocation_cap_usd=100.0,
        available_margin_usd=100.0,
    )

    assert aligned["gross_expected_return_bps"] <= 500.0
    assert conflicting["gross_expected_return_bps"] <= 500.0
    assert conflicting["ml_agreement_score"] < aligned["ml_agreement_score"]
    assert conflicting["ml_consensus_multiplier"] < aligned["ml_consensus_multiplier"]
    assert conflicting["ml_profit_score"] < aligned["ml_profit_score"]
    assert conflicting["suggested_notional_usd"] < aligned["suggested_notional_usd"]
    assert conflicting["suggested_leverage"] < aligned["suggested_leverage"]
    assert "ml_model_disagreement" in conflicting["advisory_blockers"]
    assert "ml_calibration_weak" in conflicting["advisory_blockers"]


def test_one_h10_start_creates_provider_tagged_legs(app, monkeypatch) -> None:
    _confirm_one_h10_live(app)
    app.config["AGGRESSIVE_1H_MAX_COST_DRAG_BPS"] = 30.0
    user, secret = _create_user("route")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=500.0, estimated_usd_value=500.0))
    hyperliquid = _connection(app, user, "hyperliquid", active=True)
    kucoin = _connection(app, user, "kucoin", active=False)
    db.session.add_all(
        [
            LeveragedMarket(
                provider="hyperliquid",
                venue_symbol="kPEPE",
                symbol="KPEPE",
                status="active",
                settlement_asset="USDC",
                max_leverage=10,
                liquidity_usd=1_200_000,
            ),
            LeveragedMarket(
                provider="hyperliquid",
                venue_symbol="ZEC",
                symbol="ZEC",
                status="active",
                settlement_asset="USDC",
                max_leverage=10,
                liquidity_usd=1_100_000,
            ),
            LeveragedMarket(
                provider="hyperliquid",
                venue_symbol="ASTER",
                symbol="ASTER",
                status="active",
                settlement_asset="USDC",
                max_leverage=5,
                liquidity_usd=1_050_000,
            ),
            LeveragedMarket(
                provider="hyperliquid",
                venue_symbol="BTC",
                symbol="BTC",
                status="active",
                settlement_asset="USDC",
                max_leverage=50,
                liquidity_usd=1_000_000,
            ),
            LeveragedMarket(
                provider="kucoin",
                venue_symbol="XBTMM26",
                symbol="BTC",
                status="active",
                settlement_asset="XBT",
                max_leverage=20,
                liquidity_usd=1_100_000,
            ),
            LeveragedMarket(
                provider="kucoin",
                venue_symbol="ETHUSDM",
                symbol="ETH",
                status="active",
                settlement_asset="ETH",
                max_leverage=50,
                liquidity_usd=1_050_000,
            ),
            LeveragedMarket(
                provider="kucoin",
                venue_symbol="SOLUSDM",
                symbol="SOL",
                status="active",
                settlement_asset="SOL",
                max_leverage=50,
                liquidity_usd=1_000_000,
            ),
        ]
    )
    db.session.flush()
    for market in LeveragedMarket.query.all():
        _feature(market, score=1.0, liquidity=1_000_000)
    db.session.commit()
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: {"levels": [[{"px": "99.9", "sz": "1000"}], [{"px": "100.1", "sz": "1000"}]]}
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles(limit)
    started: list[int] = []
    app.extensions["services"]["strategy_manager"].start = lambda run_id: started.append(run_id)
    service = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(service, "can_trade", lambda user_id, mode, connection_id=None: True)

    def snapshot(user_id: int, mode: str, connection_id: int | None = None) -> ClientSnapshot:
        if connection_id == hyperliquid.id:
            return ClientSnapshot(mode, [{"asset": "USDC", "type": "margin", "value": 600.0, "withdrawable": 600.0}], [], [], [], [])
        return ClientSnapshot(mode, [{"asset": "USDT", "type": "futures", "value": 400.0, "withdrawable": 400.0}], [], [], [], [])

    monkeypatch.setattr(service, "account_snapshot", snapshot)
    discovery_calls: list[dict[str, Any]] = []

    def sync_for_user(user_id: int, mode: str = "live", **kwargs) -> list[dict[str, Any]]:
        discovery_calls.append({"user_id": user_id, "mode": mode, **kwargs})
        return []

    monkeypatch.setattr(app.extensions["services"]["leveraged_markets"], "sync_for_user", sync_for_user)
    monkeypatch.setitem(app.extensions["services"], "one_h10_forecast", _PassingOneH10Forecast())

    client = app.test_client()
    _login(client, user.username, secret)
    response = client.post(
        "/vault/start",
        data={
            "deposit_amount": "100",
            "deposit_asset": "USDC",
            "lock_duration": "1",
            "settlement_asset": "USDC",
            "allowed_symbols": "BTC",
            "one_h10_live_ack": "on",
        },
    )

    assert response.status_code == 302
    cycle = VaultCycle.query.filter_by(user_id=user.id).one()
    legs = VaultAllocationLeg.query.filter_by(vault_cycle_id=cycle.id).all()
    assert cycle.algorithm_profile == "1H10"
    assert cycle.selection_metadata["ml_horizon"] == "1h10"
    assert {leg.provider for leg in legs} == {"hyperliquid", "kucoin"}
    assert {leg.trading_connection_id for leg in legs} == {hyperliquid.id, kucoin.id}
    hyperliquid_legs = [leg for leg in legs if leg.provider == "hyperliquid"]
    kucoin_legs = [leg for leg in legs if leg.provider == "kucoin"]
    assert [leg.symbol for leg in hyperliquid_legs] == ["KPEPE", "ZEC", "ASTER"]
    assert [leg.details.get("venue_symbol") for leg in hyperliquid_legs] == ["kPEPE", "ZEC", "ASTER"]
    assert [leg.symbol for leg in kucoin_legs] == ["BTC", "ETH", "SOL"]
    assert [leg.details.get("venue_symbol") for leg in kucoin_legs] == ["XBTMM26", "ETHUSDM", "SOLUSDM"]
    assert [leg.details.get("app_symbol") for leg in kucoin_legs] == ["BTC", "ETH", "SOL"]
    assert [leg.details.get("provider_symbol") for leg in kucoin_legs] == ["XBTMM26", "ETHUSDM", "SOLUSDM"]
    assert [db.session.get(StrategyRun, leg.strategy_run_id).symbol for leg in kucoin_legs] == ["BTC", "ETH", "SOL"]
    assert [db.session.get(StrategyRun, leg.strategy_run_id).parameters["venue_symbol"] for leg in kucoin_legs] == [
        "XBTMM26",
        "ETHUSDM",
        "SOLUSDM",
    ]
    assert db.session.get(StrategyRun, hyperliquid_legs[0].strategy_run_id).parameters["venue_symbol"] == "kPEPE"
    assert all(db.session.get(LeveragedMarket, leg.details.get("market_id")).symbol == leg.symbol for leg in legs)
    assert all(db.session.get(StrategyRun, leg.strategy_run_id).trading_connection_id == leg.trading_connection_id for leg in legs)
    assert all((leg.details.get("one_h10_forecast") or {}).get("ml_namespace") == "1h10" for leg in legs)
    assert all(db.session.get(StrategyRun, leg.strategy_run_id).parameters["one_h10_forecast"]["ml_horizon"] == "1h10" for leg in legs)
    assert all(float(leg.details.get("allocation_score") or 0.0) > 0 for leg in legs)
    assert all(db.session.get(StrategyRun, leg.strategy_run_id).parameters["forecast_profitability_score"] is not None for leg in legs)
    assert all(
        db.session.get(StrategyRun, leg.strategy_run_id).parameters["forecast_execution_adjusted_net_return_bps"] is not None
        for leg in legs
    )
    history_legs = [
        history_leg for provider_row in cycle.selection_metadata["exchange_allocation_history"] for history_leg in provider_row["legs"]
    ]
    assert history_legs[0]["forecast"]["ml_horizon"] == "1h10"
    assert all(float(history_leg.get("allocation_score") or 0.0) > 0 for history_leg in history_legs)
    assert any(history_leg["venue_symbol"] == "kPEPE" for history_leg in history_legs)
    assert sum(float(leg.allocation_cap_usd or 0.0) for leg in legs) <= 100.0 + 1e-9
    assert started
    assert discovery_calls == [
        {
            "user_id": user.id,
            "mode": "live",
            "feature_scope": "all",
            "persist_features": False,
        }
    ]


def test_vault_routing_preview_reports_provider_allocations_without_creating_cycle(app, monkeypatch) -> None:
    _confirm_one_h10_live(app)
    app.config["ML_ALL_AREAS_ENABLED"] = False
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False
    user, secret = _create_user("routingpreview")
    hyperliquid = _connection(app, user, "hyperliquid", active=True)
    _connection(app, user, "kucoin", active=False)
    service = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(service, "can_trade", lambda user_id, mode, connection_id=None: True)

    def snapshot(user_id: int, mode: str, connection_id: int | None = None) -> ClientSnapshot:
        if connection_id == hyperliquid.id:
            return ClientSnapshot(mode, [{"asset": "USDC", "type": "margin", "value": 60.0, "withdrawable": 60.0}], [], [], [], [])
        return ClientSnapshot(mode, [{"asset": "USDT", "type": "futures", "value": 20.0, "withdrawable": 20.0}], [], [], [], [])

    monkeypatch.setattr(service, "account_snapshot", snapshot)

    client = app.test_client()
    _login(client, user.username, secret)
    response = client.get(
        "/api/vault/routing-preview?cycle_type=one_h10&amount=20&deposit_asset=USDC&settlement_asset=USDC&providers=hyperliquid&providers=kucoin"
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["cycle"]["type"] == "one_h10"
    assert payload["summary"]["ready_provider_count"] == 2
    rows = {row["provider"]: row for row in payload["providers"]}
    assert rows["hyperliquid"]["target_amount"] == pytest.approx(15.0)
    assert rows["kucoin"]["target_amount"] == pytest.approx(5.0)
    assert rows["hyperliquid"]["allocation_weight"] == pytest.approx(0.75)
    assert rows["kucoin"]["allocation_weight"] == pytest.approx(0.25)
    assert VaultCycle.query.filter_by(user_id=user.id).count() == 0


def test_one_h10_start_honors_selected_provider_filter(app, monkeypatch) -> None:
    _confirm_one_h10_live(app)
    app.config["AGGRESSIVE_1H_MAX_COST_DRAG_BPS"] = 30.0
    user, secret = _create_user("providerfilter")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=100.0, estimated_usd_value=100.0))
    hyperliquid = _connection(app, user, "hyperliquid", active=True)
    kucoin = _connection(app, user, "kucoin", active=False)
    db.session.add_all(
        [
            LeveragedMarket(
                provider="hyperliquid",
                venue_symbol="BTC",
                symbol="BTC",
                status="active",
                settlement_asset="USDC",
                max_leverage=50,
                liquidity_usd=1_000_000,
            ),
            LeveragedMarket(
                provider="kucoin",
                venue_symbol="XBTUSDTM",
                symbol="BTC",
                status="active",
                settlement_asset="USDT",
                max_leverage=50,
                liquidity_usd=1_000_000,
            ),
        ]
    )
    db.session.flush()
    for market in LeveragedMarket.query.all():
        _feature(market, score=1.0, liquidity=1_000_000)
    db.session.commit()
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: {"levels": [[{"px": "99.9", "sz": "1000"}], [{"px": "100.1", "sz": "1000"}]]}
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles(limit)
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    service = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(service, "can_trade", lambda user_id, mode, connection_id=None: True)

    def snapshot(user_id: int, mode: str, connection_id: int | None = None) -> ClientSnapshot:
        asset = "USDT" if connection_id == kucoin.id else "USDC"
        return ClientSnapshot(mode, [{"asset": asset, "type": "margin", "value": 50.0, "withdrawable": 50.0}], [], [], [], [])

    monkeypatch.setattr(service, "account_snapshot", snapshot)
    monkeypatch.setattr(app.extensions["services"]["leveraged_markets"], "sync_for_user", lambda *args, **kwargs: [])
    monkeypatch.setitem(app.extensions["services"], "one_h10_forecast", _PassingOneH10Forecast())

    client = app.test_client()
    _login(client, user.username, secret)
    response = client.post(
        "/vault/start",
        data={
            "deposit_amount": "10",
            "deposit_asset": "USDC",
            "lock_duration": "1",
            "settlement_asset": "USDC",
            "providers_submitted": "1",
            "providers": "kucoin",
            "one_h10_live_ack": "on",
        },
    )

    assert response.status_code == 302
    cycle = VaultCycle.query.filter_by(user_id=user.id).one()
    legs = VaultAllocationLeg.query.filter_by(vault_cycle_id=cycle.id).all()
    assert {leg.provider for leg in legs} == {"kucoin"}
    assert {leg.trading_connection_id for leg in legs} == {kucoin.id}
    assert hyperliquid.id not in {leg.trading_connection_id for leg in legs}
    assert cycle.selection_metadata["requested_provider_filter"] == ["kucoin"]


def test_one_h10_start_serializes_promoted_model_datetimes(app, monkeypatch) -> None:
    _confirm_one_h10_live(app)
    app.config["AGGRESSIVE_1H_MAX_COST_DRAG_BPS"] = 30.0
    user, secret = _create_user("routejson")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=100.0, estimated_usd_value=100.0))
    _connection(app, user, "hyperliquid", active=True)
    market = LeveragedMarket(
        provider="hyperliquid",
        venue_symbol="BTC",
        symbol="BTC",
        status="active",
        settlement_asset="USDC",
        max_leverage=50,
        liquidity_usd=1_000_000,
    )
    db.session.add(market)
    db.session.flush()
    _feature(market, score=1.0, liquidity=1_000_000)
    db.session.commit()
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: {"levels": [[{"px": "99.9", "sz": "1000"}], [{"px": "100.1", "sz": "1000"}]]}
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles(limit)
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    service = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(service, "can_trade", lambda user_id, mode, connection_id=None: True)
    monkeypatch.setattr(
        service,
        "account_snapshot",
        lambda user_id, mode, connection_id=None: ClientSnapshot(
            mode, [{"asset": "USDC", "type": "margin", "value": 50.0, "withdrawable": 50.0}], [], [], [], []
        ),
    )
    monkeypatch.setattr(app.extensions["services"]["leveraged_markets"], "sync_for_user", lambda *args, **kwargs: [])
    monkeypatch.setitem(app.extensions["services"], "one_h10_forecast", _PassingOneH10Forecast())

    class DateReadyEngine:
        def family_readiness(self, family: str, horizon: str = "1h", *, provider: str = "global") -> dict[str, Any]:
            return {
                "ready": True,
                "family": family,
                "horizon": horizon,
                "provider": provider,
                "blockers": [],
                "promoted_model": {
                    "status": "promoted",
                    "created_at": datetime.utcnow(),
                    "promoted_at": datetime.utcnow(),
                },
            }

    monkeypatch.setitem(app.extensions["services"], "ml_decision_engine", DateReadyEngine())

    client = app.test_client()
    _login(client, user.username, secret)
    response = client.post(
        "/vault/start",
        data={
            "deposit_amount": "10",
            "deposit_asset": "USDC",
            "lock_duration": "1",
            "settlement_asset": "USDC",
            "one_h10_live_ack": "on",
        },
    )

    assert response.status_code == 302
    cycle = VaultCycle.query.filter_by(user_id=user.id).one()
    promoted = cycle.selection_metadata["ml_readiness"]["families"]["pytorch_risk_policy"]["promoted_model"]
    assert isinstance(promoted["created_at"], str)
    assert isinstance(promoted["promoted_at"], str)


def test_one_h10_start_rejects_missing_live_ack(app) -> None:
    _confirm_one_h10_live(app)
    user, secret = _create_user("missingack")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=100.0, estimated_usd_value=100.0))
    _connection(app, user, "hyperliquid", active=True)
    db.session.commit()

    client = app.test_client()
    _login(client, user.username, secret)
    response = client.post(
        "/vault/start",
        data={"deposit_amount": "25", "deposit_asset": "USDC", "lock_duration": "1", "settlement_asset": "USDC"},
        follow_redirects=True,
    )

    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDC").one()
    assert response.status_code == 200
    assert VaultCycle.query.filter_by(user_id=user.id).count() == 0
    assert balance.available_balance == 100.0
    assert balance.locked_balance == 0.0
    assert b"Confirm the 1H10 acknowledgement" in response.data


def test_one_h10_start_rejects_when_live_flag_disabled(app) -> None:
    _confirm_one_h10_live(app)
    app.config["ONE_H10_LIVE_ENABLED"] = False
    user, secret = _create_user("disabled")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=100.0, estimated_usd_value=100.0))
    _connection(app, user, "hyperliquid", active=True)
    db.session.commit()

    client = app.test_client()
    _login(client, user.username, secret)
    response = client.post(
        "/vault/start",
        data={
            "deposit_amount": "25",
            "deposit_asset": "USDC",
            "lock_duration": "1",
            "settlement_asset": "USDC",
            "one_h10_live_ack": "on",
        },
        follow_redirects=True,
    )

    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDC").one()
    assert response.status_code == 200
    assert VaultCycle.query.filter_by(user_id=user.id).count() == 0
    assert balance.available_balance == 100.0
    assert balance.locked_balance == 0.0
    assert b"1H10 live execution is disabled" in response.data


def test_one_h10_live_context_reports_bootstrap_ready_without_promoted_ml_blocking(app, monkeypatch) -> None:
    _confirm_one_h10_live(app)
    app.config["ML_ALL_AREAS_ENABLED"] = False
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False
    user, _ = _create_user("livecontext")
    hyperliquid = _connection(app, user, "hyperliquid", active=True)
    _connection(app, user, "kucoin", active=False)
    service = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(service, "can_trade", lambda user_id, mode, connection_id=None: connection_id == hyperliquid.id)

    def snapshot(user_id: int, mode: str, connection_id: int | None = None) -> ClientSnapshot:
        if connection_id == hyperliquid.id:
            return ClientSnapshot(mode, [{"asset": "USDC", "type": "margin", "value": 125.0, "withdrawable": 125.0}], [], [], [], [])
        return ClientSnapshot(mode, [{"asset": "USDT", "type": "futures", "value": 0.0, "withdrawable": 0.0}], [], [], [], [])

    monkeypatch.setattr(service, "account_snapshot", snapshot)

    context = _one_h10_live_context(user)

    assert context["enabled"] is True
    assert context["ack_required"] is True
    assert context["enabled_provider_count"] == 1
    assert context["total_free_margin_usd"] == 125.0
    assert context["ml_readiness"]["ready"] is True
    assert context["ml_readiness"]["mode"] == "bootstrap"
    assert context["ml_readiness"]["display_status"] == "Bootstrap Ready"
    assert context["ml_readiness"]["blockers"] == []
    assert context["ml_readiness"]["promoted_ready"] is False
    assert context["ml_readiness"]["promoted_blockers"]
    assert "ml_not_ready" not in context["safety_blockers"]
    assert {provider["provider"] for provider in context["providers"]} == {"hyperliquid", "kucoin"}


def test_one_h10_live_context_blocks_when_promoted_ml_is_required(app, monkeypatch) -> None:
    _confirm_one_h10_live(app)
    app.config["ML_ALL_AREAS_ENABLED"] = False
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = True
    user, _ = _create_user("strictlivecontext")
    hyperliquid = _connection(app, user, "hyperliquid", active=True)
    service = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(service, "can_trade", lambda user_id, mode, connection_id=None: connection_id == hyperliquid.id)
    monkeypatch.setattr(
        service,
        "account_snapshot",
        lambda user_id, mode, connection_id=None: ClientSnapshot(
            mode,
            [{"asset": "USDC", "type": "margin", "value": 125.0, "withdrawable": 125.0}],
            [],
            [],
            [],
            [],
        ),
    )

    context = _one_h10_live_context(user)

    assert context["ml_readiness"]["ready"] is False
    assert context["ml_readiness"]["mode"] == "blocked"
    assert context["ml_readiness"]["blockers"]
    assert "ml_not_ready" in context["safety_blockers"]


def test_one_h10_bootstrap_readiness_exposes_promoted_blockers_as_advisory(app) -> None:
    app.config["ML_ALL_AREAS_ENABLED"] = False
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False

    readiness = _one_h10_ml_readiness("global")

    assert readiness["ready"] is True
    assert readiness["execution_ready"] is True
    assert readiness["promoted_ready"] is False
    assert readiness["mode"] == "bootstrap"
    assert readiness["blockers"] == []
    assert readiness["advisory_blockers"]
    assert readiness["promoted_blockers"] == readiness["advisory_blockers"]


def test_one_h10_cycle_summary_normalizes_stale_bootstrap_readiness(app) -> None:
    app.config["ML_ALL_AREAS_ENABLED"] = False
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False
    user, _ = _create_user("stalereadiness")
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDC",
        deposit_amount=25.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        status="active",
        algorithm_profile="1H10",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=25.0,
        current_estimated_value_usd=25.0,
    )
    cycle.selection_metadata = {
        "ml_readiness": {
            "ready": False,
            "source": "one_h10_live_execution_readiness",
            "blockers": ["pytorch_risk_policy:ML_ALL_AREAS_ENABLED=false"],
        }
    }
    db.session.add(cycle)
    db.session.commit()

    summary = _cycle_summary(cycle)

    assert summary["ml_readiness"]["ready"] is True
    assert summary["ml_readiness"]["mode"] == "bootstrap"
    assert "ml_not_ready" not in summary["blocker_categories"]


def test_vault_cycle_summary_persists_promoted_model_datetimes(app) -> None:
    user, _ = _create_user("summarydatetimes")
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDC",
        deposit_amount=25.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        status="active",
        algorithm_profile="1H10",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=25.0,
        current_estimated_value_usd=25.0,
    )
    db.session.add(cycle)
    db.session.flush()

    cycle.cycle_summary = {
        "ml_readiness": {
            "ready": True,
            "promoted_model": {
                "id": 1,
                "created_at": datetime.utcnow(),
                "promoted_at": datetime.utcnow(),
            },
        }
    }
    db.session.commit()

    payload = db.session.get(VaultCycle, cycle.id).cycle_summary
    assert payload["ml_readiness"]["ready"] is True
    assert isinstance(payload["ml_readiness"]["promoted_model"]["created_at"], str)


def test_one_h10_live_readiness_uses_execution_families_not_optional_signal_model(app, monkeypatch) -> None:
    app.config["ML_ALL_AREAS_ENABLED"] = True
    app.config["ML_FIBONACCI_MODEL_ENABLED"] = True
    app.config["ML_RISK_POLICY_ENABLED"] = True
    app.config["ML_EXIT_POLICY_ENABLED"] = True
    app.config["ML_CAP_POLICY_ENABLED"] = True
    app.config["ML_ORDER_POLICY_ENABLED"] = True
    app.config["ML_ROI_TARGET_POLICY_ENABLED"] = True

    class ReadyOneH10Engine:
        def family_readiness(self, family: str, horizon: str = "1h", *, provider: str = "global") -> dict[str, Any]:
            return {
                "ready": True,
                "family": family,
                "horizon": horizon,
                "provider": provider,
                "blockers": [],
                "promoted_model": {"status": "promoted", "horizon": horizon, "model_type": family},
            }

    monkeypatch.setitem(app.extensions["services"], "ml_decision_engine", ReadyOneH10Engine())

    readiness = _one_h10_ml_readiness("global")

    assert readiness["ready"] is True
    assert readiness["family"] == "one_h10_live_execution"
    assert readiness["objective"] == "one_h10"
    assert "pytorch_gru_signal" in readiness["ignored_optional_families"]


def test_one_h10_risk_rejects_orders_above_one_h10_leverage_cap(app) -> None:
    app.config["MAX_LEVERAGE"] = 5.0
    app.config["ONE_H10_MAX_LEVERAGE"] = 2.0
    risk = app.extensions["services"]["risk_engine"]
    intent = OrderIntent(
        symbol="BTC",
        side="buy",
        quantity=0.1,
        mode="live",
        leverage=3.0,
        stop_loss=95.0,
        take_profit=110.0,
        metadata={"one_h10_vault": True, "ml_horizon": "1h10", "algorithm_profile": "1H10"},
    )

    decision = risk.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert decision.approved is False
    assert decision.rule_name == "one_h10_leverage_cap"


def test_one_h10_bootstrap_live_approves_without_promoted_ml_policy(app) -> None:
    _confirm_one_h10_live(app)
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False
    app.config["ML_ALL_AREAS_ENABLED"] = False
    app.config["ML_RISK_POLICY_ENABLED"] = False

    decision = app.extensions["services"]["risk_engine"].evaluate(_one_h10_intent(), market_price=100.0, has_trading_access=True)

    assert decision.approved is True
    assert decision.details["ml_policy_authority"] == "one_h10_bootstrap"
    assert decision.details["ml_policy_decisions"]["ml_policy_decisions"]["one_h10_bootstrap_policy"]["horizon"] == "1h10"


def test_one_h10_bootstrap_live_still_rejects_dynamic_cap_breach(app) -> None:
    _confirm_one_h10_live(app)
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False
    intent = _one_h10_intent(allocation_cap_usd=10.0, available_margin_usd=10.0, account_equity_usd=10.0)
    intent.quantity = 0.2

    decision = app.extensions["services"]["risk_engine"].evaluate(intent, market_price=100.0, has_trading_access=True)

    assert decision.approved is False
    assert decision.rule_name == "safety_envelope_blocked"
    assert "safety_one_h10_dynamic_cap_breached" in decision.details["blockers"]


def test_one_h10_bootstrap_preserves_deterministic_signal_when_ml_signal_disabled(app) -> None:
    app.config["HIGH_UPSIDE_REQUIRE_ML_SIGNAL"] = True
    app.config["ML_SIGNAL_MODEL_ENABLED"] = False
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False
    manager = app.extensions["services"]["strategy_manager"]
    run = StrategyRun(strategy_name="scalping", symbol="DOGE", timeframe="1m", mode="live")
    run.parameters = {"high_upside_profile": True, "one_h10_vault": True, "ml_horizon": "1h10", "provider": "hyperliquid"}
    signal = Signal("buy", "base signal", "1m", 99.0, 102.0, 0.5, {"confidence": 0.8})

    selected = manager._high_upside_ml_signal(
        run,
        signal,
        [{"close": 100.0, "volume": 1000}, {"close": 101.0, "volume": 1100}],
        {"trend_strength": 1.0},
        "live",
    )

    assert selected.action == "buy"
    assert selected.position_fraction == 0.5
    assert selected.metadata["ml_signal_model"]["status"] == "disabled"
    assert selected.metadata["one_h10_bootstrap_live"] is True


def test_one_h10_forecast_signal_can_create_bootstrap_live_signal_by_default(app) -> None:
    manager = app.extensions["services"]["strategy_manager"]
    run = StrategyRun(strategy_name="scalping", symbol="DOGE", timeframe="1m", mode="live")
    run.parameters = {
        "one_h10_vault": True,
        "ml_horizon": "1h10",
        "one_h10_forecast": {
            "predicted_side": "buy",
            "confidence": 0.7,
            "position_fraction": 0.6,
            "suggested_stop_loss_pct": 0.01,
            "suggested_take_profit_pct": 0.03,
            "suggested_leverage": 2.0,
            "blockers": ["ml_not_ready"],
        },
    }
    base = Signal("hold", "base hold", "1m", None, None, 0.0, {})

    selected = manager._one_h10_forecast_signal(run, base, [{"close": 100.0}])

    assert selected.action == "hold"
    assert selected.position_fraction == 0.0
    assert selected.stop_loss is None
    assert selected.take_profit is None
    assert selected.metadata["no_trade_reason"] == "one_h10_promoted_ml_not_ready"
    assert selected.metadata["ml_signal_not_ready"] is True


def test_one_h10_forecast_signal_blocks_when_promoted_ml_is_required(app) -> None:
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = True
    manager = app.extensions["services"]["strategy_manager"]
    run = StrategyRun(strategy_name="scalping", symbol="DOGE", timeframe="1m", mode="live")
    run.parameters = {
        "one_h10_vault": True,
        "ml_horizon": "1h10",
        "one_h10_forecast": {
            "predicted_side": "buy",
            "confidence": 0.7,
            "position_fraction": 0.6,
            "suggested_stop_loss_pct": 0.01,
            "suggested_take_profit_pct": 0.03,
            "suggested_leverage": 2.0,
            "blockers": ["ml_not_ready"],
        },
    }
    base = Signal("hold", "base hold", "1m", None, None, 0.0, {})

    selected = manager._one_h10_forecast_signal(run, base, [{"close": 100.0}])

    assert selected.action == "hold"
    assert selected.position_fraction == 0.0
    assert selected.metadata["no_trade_reason"] == "one_h10_promoted_ml_not_ready"


def test_one_h10_forecast_signal_holds_on_cost_quality_blocker_without_fallback(app) -> None:
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    manager = app.extensions["services"]["strategy_manager"]
    run = StrategyRun(strategy_name="scalping", symbol="DOGE", timeframe="1m", mode="live")
    run.parameters = {
        "one_h10_vault": True,
        "ml_horizon": "1h10",
        "one_h10_forecast": {
            "ml_ready": True,
            "ml_horizon": "1h10",
            "predicted_side": "buy",
            "confidence": 0.8,
            "position_fraction": 0.6,
            "gross_expected_return_bps": 20.0,
            "expected_return_bps": 1.0,
            "net_expected_return_bps": 1.0,
            "cost_drag_bps": 30.0,
            "estimated_fee_bps": 5.0,
            "estimated_slippage_bps": 15.0,
            "spread_bps": 10.0,
            "execution_quality": 0.4,
            "expected_net_edge_passed": False,
            "risk_reward": 2.0,
            "suggested_notional_usd": 10.0,
            "suggested_stop_loss_pct": 0.01,
            "suggested_take_profit_pct": 0.02,
            "blockers": ["low_edge_after_costs"],
        },
    }
    base = Signal("buy", "base signal should not execute", "1m", 99.0, 102.0, 0.5, {})

    selected = manager._one_h10_forecast_signal(run, base, [{"close": 100.0}])

    assert selected.action == "hold"
    assert selected.position_fraction == 0.0
    assert selected.metadata["no_trade_reason"].startswith("one_h10_forecast_blocked:")
    assert "low_edge_after_costs" in selected.metadata["forecast_decision_blockers"]
    assert selected.metadata["decision_reason_code"] == "BELOW_EDGE_THRESHOLD"


def test_one_h10_risk_rejects_low_edge_forecast_before_live_order(app) -> None:
    _confirm_one_h10_live(app)
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    forecast = {
        "ml_ready": True,
        "ml_horizon": "1h10",
        "predicted_side": "buy",
        "confidence": 0.8,
        "gross_expected_return_bps": 12.0,
        "expected_return_bps": 1.0,
        "net_expected_return_bps": 1.0,
        "cost_drag_bps": 10.0,
        "estimated_fee_bps": 5.0,
        "estimated_slippage_bps": 3.0,
        "spread_bps": 1.0,
        "execution_quality": 0.9,
        "expected_net_edge_passed": False,
        "risk_reward": 2.0,
        "suggested_notional_usd": 25.0,
        "suggested_stop_loss_pct": 0.01,
        "suggested_take_profit_pct": 0.02,
        "blockers": [],
    }
    intent = _one_h10_intent(one_h10_forecast=forecast)

    decision = app.extensions["services"]["risk_engine"].evaluate(intent, market_price=100.0, has_trading_access=True)

    assert decision.approved is False
    assert decision.rule_name == "one_h10_signal_quality_blocked"
    assert "low_edge_after_costs" in decision.details["blockers"]
    assert decision.details["decision_reason_code"] == "BELOW_EDGE_THRESHOLD"


def test_one_h10_signal_model_forward_steps_use_one_hour_horizon() -> None:
    from app.ml.signal_model import MLSignalModel

    assert MLSignalModel._forward_steps("1h10", "1m") == 60
    assert MLSignalModel._forward_steps("1h10", "5m") == 12
    assert MLSignalModel._forward_steps("1h10", "15m") == 4


def test_one_h10_backtest_ml_first_uses_explicit_horizon(app) -> None:
    from app.backtesting.engine import BacktestConfig

    class CaptureDecisionEngine:
        horizon = ""

        def decision(self, family: str, context: dict[str, Any], *, horizon: str = "1h", candles=None) -> dict[str, Any]:
            self.horizon = horizon
            return {"ready": False, "action": "hold", "blockers": ["test_blocker"], "raw": {}}

    engine = app.extensions["services"]["backtest_engine"]
    capture = CaptureDecisionEngine()
    engine.ml_decision_engine = capture
    app.config["ML_FIRST_STRATEGIES_ENABLED"] = True
    config = BacktestConfig(
        strategy_name="scalping",
        symbol="BTC",
        timeframe="1m",
        mode="testnet",
        initial_balance=1000.0,
        fee_bps=5.0,
        slippage_bps=8.0,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
        position_size_fraction=0.1,
        parameters={"one_h10_vault": True, "vault_cycle_duration": "1h10", "lock_duration_hours": 2},
    )
    base = Signal("buy", "base", "1m", 99.0, 102.0, 0.2, {})

    selected = engine._ml_first_signal(config, base, _candles(40), {"trend_strength": 1.0})

    assert selected.action == "hold"
    assert capture.horizon == "1h10"


def test_one_h10_evaluation_report_includes_cost_metrics_and_small_sample_warning(app) -> None:
    from app.services.one_h10_evaluation import build_one_h10_evaluation_report

    class FakeEngine:
        calls = []

        def run(self, backtest, candles=None):
            self.calls.append(backtest)
            optimized = backtest.parameters["optimizer_variant"] == "net_expectancy"
            return {
                "total_return": 0.012 if optimized else 0.01,
                "net_return_after_costs": 0.009 if optimized else 0.006,
                "sharpe_like": 0.5 if optimized else 0.4,
                "sortino_like": 0.4 if optimized else 0.3,
                "max_drawdown": -0.015 if optimized else -0.02,
                "win_rate": 0.55 if optimized else 0.5,
                "profit_factor": 1.4 if optimized else 1.2,
                "average_return_per_trade": 0.003 if optimized else 0.002,
                "avg_loss": 0.001,
                "trade_count": 1,
                "fees_paid": 0.5,
                "funding_cost_estimate": 0.0,
                "cost_drag_bps": 11.0 if optimized else 13.0,
                "average_trade_duration_minutes": 12.0,
                "no_trade_reason": "",
                "trades": [{"direction": "long", "duration_minutes": 12.0, "return": 0.002}],
            }

    fake = FakeEngine()
    report = build_one_h10_evaluation_report(
        app.config,
        fake,
        symbol="btc",
        timeframe="1m",
        candles=[{"timestamp": 1}, {"timestamp": 2}],
    )

    assert report["horizon_seconds"] == app.config["ONE_H10_HORIZON_SECONDS"]
    assert report["config"]["fee_bps"] == app.config["FEE_BPS"]
    assert report["baseline"]["profit_factor"] == 1.2
    assert report["optimized"]["profit_factor"] == 1.4
    assert report["difference"]["net_return_after_costs"] == pytest.approx(0.003)
    assert "sample_size_below_threshold" in report["warnings"]
    assert [call.parameters["optimizer_variant"] for call in fake.calls] == ["baseline", "net_expectancy"]
    assert fake.calls[1].parameters["ml_horizon"] == "1h10"


def test_one_h10_market_data_failure_backs_off_without_limiting_cycle(app) -> None:
    user, _ = _create_user("mdbackoff")
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDT",
        deposit_amount=2.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        status="active",
        execution_substatus="executing",
        execution_mode="live",
        algorithm_profile="1H10",
        starting_value_usd=2.0,
        current_estimated_value_usd=2.0,
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
    )
    db.session.add(cycle)
    db.session.flush()
    run = StrategyRun(
        user_id=user.id,
        strategy_name="scalping",
        symbol="BTC",
        timeframe="1m",
        mode="live",
        status="running",
        manual_enabled=True,
    )
    run.parameters = {
        "vault_cycle_id": cycle.id,
        "one_h10_vault": True,
        "ml_horizon": "1h10",
        "objective": "one_h10",
    }
    db.session.add(run)
    db.session.flush()
    cycle.strategy_run_id = run.id
    db.session.commit()
    manager = app.extensions["services"]["strategy_manager"]

    handled = manager._handle_one_h10_market_data_failure(
        run,
        RuntimeError("Hyperliquid call failed after retries: live candles_snapshot BTC 1m"),
    )
    db.session.commit()

    assert handled is True
    assert run.status == "running"
    assert run.manual_enabled is True
    assert run.last_signal["action"] == "hold"
    assert run.parameters["one_h10_market_data_status"] == "backoff"
    assert manager._one_h10_market_data_backoff_remaining(run) > 0
    assert cycle.status == "active"
    assert cycle.execution_substatus == "executing"
    assert cycle.validation_failure_reason is None
    assert cycle.selection_metadata["one_h10_market_data_blocker"] == "features_stale"
    assert "no_order_failure_reason" not in cycle.selection_metadata
    assert "features_stale" in cycle.selection_metadata["risk_blockers"]


def test_one_h10_market_data_backoff_applies_to_same_provider_connection(app) -> None:
    user, _ = _create_user("providerbackoff")
    hyperliquid = _connection(app, user, "hyperliquid", active=True)
    manager = app.extensions["services"]["strategy_manager"]
    first = StrategyRun(
        user_id=user.id,
        trading_connection_id=hyperliquid.id,
        strategy_name="scalping",
        symbol="BTC",
        timeframe="1m",
        mode="live",
    )
    first.parameters = {"one_h10_vault": True, "ml_horizon": "1h10", "provider": "hyperliquid"}
    second = StrategyRun(
        user_id=user.id,
        trading_connection_id=hyperliquid.id,
        strategy_name="scalping",
        symbol="ETH",
        timeframe="1m",
        mode="live",
    )
    second.parameters = {"one_h10_vault": True, "ml_horizon": "1h10", "provider": "hyperliquid"}
    db.session.add_all([first, second])
    db.session.flush()

    handled = manager._handle_one_h10_market_data_failure(first, RuntimeError("429 too many requests: live all_mids"))
    db.session.commit()

    assert handled is True
    assert manager._one_h10_market_data_backoff_remaining(second) > 0
    payload = Setting.get_json(f"one_h10_market_data_backoff:hyperliquid:{hyperliquid.id}", {})
    assert payload["blocker_category"] == "rate_limited"
    assert "429" not in first.last_error


def test_one_h10_provider_market_data_uses_connection_connector(app, monkeypatch) -> None:
    user, _ = _create_user("providerdata")
    kucoin = _connection(app, user, "kucoin", active=True)
    run = StrategyRun(
        user_id=user.id,
        trading_connection_id=kucoin.id,
        strategy_name="scalping",
        symbol="BTC",
        timeframe="1m",
        mode="live",
    )
    run.parameters = {
        "one_h10_vault": True,
        "ml_horizon": "1h10",
        "provider": "kucoin",
        "venue_symbol": "XBTMM26",
    }
    manager = app.extensions["services"]["strategy_manager"]
    market_data = app.extensions["services"]["market_data"]
    monkeypatch.setattr(
        market_data, "get_candles", lambda *args, **kwargs: pytest.fail("Hyperliquid candles should not be used for KuCoin 1H10 legs")
    )
    monkeypatch.setattr(
        market_data, "get_mid_price", lambda *args, **kwargs: pytest.fail("Hyperliquid mids should not be used for KuCoin 1H10 legs")
    )

    class Connector:
        def get_candles(self, symbol, timeframe, mode, limit):
            assert (symbol, timeframe, mode, limit) == ("XBTMM26", "1m", "live", 10)
            return [{"timestamp": "2026-05-03T00:00:00+00:00", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}]

        def get_mid_price(self, symbol, mode):
            assert (symbol, mode) == ("XBTMM26", "live")
            return 1.5

    monkeypatch.setattr(app.extensions["services"]["trading_connections"], "connector_for_user", lambda *args, **kwargs: Connector())

    candles = manager._run_candles(run, "live", limit=10)
    assert candles[0]["close"] == 1.5
    assert (
        app.extensions["services"]["order_manager"]._safe_market_price(
            "BTC",
            "live",
            user_id=user.id,
            trading_connection_id=kucoin.id,
            provider="kucoin",
            venue_symbol="XBTMM26",
        )
        == 1.5
    )


def test_one_h10_active_backoff_shows_runtime_notice_not_repair_prompt(app) -> None:
    user, secret = _create_user("runtimeui")
    _connection(app, user, "hyperliquid", active=True)
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=5.0, locked_balance=2.0, estimated_usd_value=7.0))
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDT",
        deposit_amount=2.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        status="active",
        execution_substatus="executing",
        execution_mode="live",
        algorithm_profile="1H10",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=2.0,
        current_estimated_value_usd=2.0,
    )
    cycle.selection_metadata = {
        "no_order_failure_reason": "(429, None, 'null', None, {'Content-Type': 'application/json'})",
        "one_h10_market_data_error": "(429, None, 'null', None, {'Content-Type': 'application/json'})",
        "one_h10_market_data_blocker": "rate_limited",
        "one_h10_market_data_backoff_until": (datetime.utcnow() + timedelta(seconds=30)).isoformat(),
        "one_h10_runtime_notice": {
            "kind": "market_data_backoff",
            "message": "(429, None, 'null', None, {'Content-Type': 'application/json'})",
            "blocker_category": "rate_limited",
        },
    }
    db.session.add(cycle)
    db.session.commit()

    client = app.test_client()
    _login(client, user.username, secret)
    vault = client.get("/vault")
    detail = client.get(f"/vault/cycles/{cycle.id}")

    assert b"Market data backoff active" in detail.data
    assert b"Provider rate limited market data or account data" in detail.data
    assert b"Content-Type" not in detail.data
    assert b"flask repair-limited-cycle --cycle-id" not in vault.data
    assert b"flask repair-limited-cycle --cycle-id" not in detail.data


def _one_h10_intent(**metadata: Any) -> OrderIntent:
    payload = {
        "one_h10_vault": True,
        "ml_horizon": "1h10",
        "algorithm_profile": "1H10",
        "vault_cycle_name": "1H10",
        "objective": "one_h10",
        "ml_objective": "one_h10",
        "ml_policy_required": True,
        "one_h10_forecast": {
            "ml_ready": True,
            "ml_horizon": "1h10",
            "objective": "one_h10",
            "source": "promoted_1h10_ml",
            "predicted_side": "buy",
            "confidence": 0.8,
            "gross_expected_return_bps": 40.0,
            "expected_return_bps": 24.0,
            "net_expected_return_bps": 24.0,
            "cost_drag_bps": 10.0,
            "estimated_fee_bps": 5.0,
            "estimated_slippage_bps": 2.0,
            "spread_bps": 1.0,
            "execution_quality": 0.9,
            "expected_net_edge_passed": True,
            "risk_reward": 2.0,
            "suggested_leverage": 1.0,
            "suggested_notional_usd": 50.0,
            "suggested_stop_loss_pct": 0.01,
            "suggested_take_profit_pct": 0.02,
            "blockers": [],
        },
        "allocation_cap_usd": 100.0,
        "available_margin_usd": 100.0,
        "account_equity_usd": 100.0,
        "user_input_amount_usd": 100.0,
    }
    payload.update(metadata)
    return OrderIntent(
        symbol="BTC",
        side="buy",
        quantity=0.5,
        mode="live",
        leverage=1.0,
        stop_loss=95.0,
        take_profit=110.0,
        strategy_name="scalping",
        timeframe="1m",
        metadata=payload,
    )


def _enable_one_h10_ml_policy(app, *, cap_policy: bool = False) -> None:
    _confirm_one_h10_live(app)
    app.config["ML_ALL_AREAS_ENABLED"] = True
    app.config["ML_FIBONACCI_MODEL_ENABLED"] = True
    app.config["ML_RISK_POLICY_ENABLED"] = True
    app.config["ML_CAP_POLICY_ENABLED"] = cap_policy
    app.config["ML_POLICY_LIVE_AUTHORITY"] = "guarded"
    app.config["ML_LIVE_HARD_DAILY_LOSS_USDC"] = 100.0


class _ReadyOneH10Policy:
    readiness_horizons: list[str] = []
    decision_horizons: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def family_readiness(self, family: str, horizon: str = "1h", **kwargs) -> dict[str, Any]:
        self.__class__.readiness_horizons.append(horizon)
        return {"ready": True, "family": family, "horizon": horizon, "blockers": []}

    def decision(self, family: str, context: dict[str, Any], *, horizon: str = "1h", candles=None) -> dict[str, Any]:
        self.__class__.decision_horizons.append(horizon)
        raw = {"approve": True, "risk_budget_usdc": context.get("notional", 0.0)}
        action = "approve"
        if family == "pytorch_cap_policy":
            raw = {"suggested_notional_usdc": context.get("notional", 0.0), "suggested_leverage": context.get("leverage", 1.0)}
            action = "cap"
        return {"ready": True, "family": family, "action": action, "blockers": [], "raw": raw}


class _CapBreachOneH10Policy(_ReadyOneH10Policy):
    def decision(self, family: str, context: dict[str, Any], *, horizon: str = "1h", candles=None) -> dict[str, Any]:
        self.__class__.decision_horizons.append(horizon)
        if family == "pytorch_cap_policy":
            return {
                "ready": True,
                "family": family,
                "action": "cap",
                "blockers": [],
                "raw": {"suggested_notional_usdc": 150.0, "suggested_leverage": context.get("leverage", 1.0)},
            }
        return {"ready": True, "family": family, "action": "approve", "blockers": [], "raw": {"approve": True}}


class _LeverageBreachOneH10Policy(_ReadyOneH10Policy):
    def decision(self, family: str, context: dict[str, Any], *, horizon: str = "1h", candles=None) -> dict[str, Any]:
        self.__class__.decision_horizons.append(horizon)
        if family == "pytorch_cap_policy":
            return {
                "ready": True,
                "family": family,
                "action": "cap",
                "blockers": [],
                "raw": {"suggested_notional_usdc": context.get("notional", 0.0), "suggested_leverage": 3.0},
            }
        return {"ready": True, "family": family, "action": "approve", "blockers": [], "raw": {"approve": True}}


def test_one_h10_opening_order_bootstrap_approves_when_custom_ml_policy_not_ready(app) -> None:
    _enable_one_h10_ml_policy(app)
    app.config["ML_ALL_AREAS_ENABLED"] = False
    app.config["ML_FIBONACCI_MODEL_ENABLED"] = False
    app.config["ML_RISK_POLICY_ENABLED"] = False

    decision = app.extensions["services"]["risk_engine"].evaluate(_one_h10_intent(), market_price=100.0, has_trading_access=True)

    assert decision.approved is False
    assert decision.rule_name == "one_h10_fibonacci_ml_not_ready"
    assert "promoted_pytorch_fibonacci_missing" in decision.details["blockers"]


def test_one_h10_opening_order_rejects_when_promoted_ml_required_and_missing(app) -> None:
    _enable_one_h10_ml_policy(app)
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = True

    decision = app.extensions["services"]["risk_engine"].evaluate(_one_h10_intent(), market_price=100.0, has_trading_access=True)

    assert decision.approved is False
    assert decision.rule_name == "one_h10_fibonacci_ml_not_ready"
    assert "promoted_pytorch_fibonacci_missing" in decision.details["blockers"]


def test_one_h10_ready_policy_approves_with_1h10_horizon(app, monkeypatch) -> None:
    _enable_one_h10_ml_policy(app)
    _ReadyOneH10Policy.readiness_horizons = []
    _ReadyOneH10Policy.decision_horizons = []
    monkeypatch.setattr(risk_engine_module, "MLDecisionEngine", _ReadyOneH10Policy)

    decision = app.extensions["services"]["risk_engine"].evaluate(
        _one_h10_intent(horizon="1h"), market_price=100.0, has_trading_access=True
    )

    assert decision.approved is True
    assert all(horizon == "1h10" for horizon in _ReadyOneH10Policy.readiness_horizons)
    assert all(horizon == "1h10" for horizon in _ReadyOneH10Policy.decision_horizons)
    assert "1h" not in _ReadyOneH10Policy.readiness_horizons
    assert "1h" not in _ReadyOneH10Policy.decision_horizons


def test_one_h10_dynamic_cap_replaces_fixed_ml_hard_cap(app, monkeypatch) -> None:
    _enable_one_h10_ml_policy(app)
    app.config["ML_LIVE_HARD_CAP_USDC"] = 10.0
    monkeypatch.setattr(risk_engine_module, "MLDecisionEngine", _ReadyOneH10Policy)

    decision = app.extensions["services"]["risk_engine"].evaluate(_one_h10_intent(), market_price=100.0, has_trading_access=True)

    assert decision.approved is True
    assert decision.details["safety_envelope"]["details"]["one_h10_dynamic_cap_usd"] == 100.0


def test_one_h10_ml_suggested_notional_above_dynamic_cap_rejects(app, monkeypatch) -> None:
    _enable_one_h10_ml_policy(app, cap_policy=True)
    monkeypatch.setattr(risk_engine_module, "MLDecisionEngine", _CapBreachOneH10Policy)

    decision = app.extensions["services"]["risk_engine"].evaluate(_one_h10_intent(), market_price=100.0, has_trading_access=True)

    assert decision.approved is False
    assert decision.rule_name == "ml_policy_rejected"
    assert "ml_cap_policy_dynamic_cap_breach" in decision.details["blockers"]


def test_one_h10_ml_suggested_leverage_above_cap_rejects(app, monkeypatch) -> None:
    _enable_one_h10_ml_policy(app, cap_policy=True)
    app.config["MAX_LEVERAGE"] = 5.0
    app.config["ONE_H10_MAX_LEVERAGE"] = 2.0
    monkeypatch.setattr(risk_engine_module, "MLDecisionEngine", _LeverageBreachOneH10Policy)

    decision = app.extensions["services"]["risk_engine"].evaluate(_one_h10_intent(), market_price=100.0, has_trading_access=True)

    assert decision.approved is False
    assert decision.rule_name == "ml_policy_rejected"
    assert "ml_cap_policy_leverage_cap_breach" in decision.details["blockers"]


def test_one_h10_rejected_order_persists_provider_connection_and_forecast_metadata(app, monkeypatch) -> None:
    _confirm_one_h10_live(app)
    app.config["ONE_H10_MAX_LEVERAGE"] = 2.0
    order_manager = app.extensions["services"]["order_manager"]
    market_data = app.extensions["services"]["market_data"]
    trading_connections = app.extensions["services"]["trading_connections"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    monkeypatch.setattr(trading_connections, "can_trade", lambda user_id, mode, connection_id=None: True)
    intent = _one_h10_intent(
        provider="hyperliquid",
        trading_connection_id=123,
        settlement_asset="USDC",
        one_h10_forecast={"ml_horizon": "1h10", "predicted_side": "buy", "confidence": 0.7},
        forecast_predicted_side="buy",
        forecast_confidence=0.7,
    )
    intent.user_id = 1
    intent.trading_connection_id = 123
    intent.leverage = 3.0

    order = order_manager.place_order(intent)

    assert order.status == "rejected"
    assert order.trading_connection_id == 123
    assert order.details["provider"] == "hyperliquid"
    assert order.details["settlement_asset"] == "USDC"
    assert order.details["one_h10_forecast"]["ml_horizon"] == "1h10"
    assert order.details["forecast_predicted_side"] == "buy"
    assert order.exchange_order_id is None


def test_one_h10_reduce_only_exit_allowed_without_custom_ml_policy(app) -> None:
    _confirm_one_h10_live(app)
    intent = _one_h10_intent()
    intent.reduce_only = True
    intent.take_profit = None
    app.config["ML_RISK_POLICY_ENABLED"] = True

    decision = app.extensions["services"]["risk_engine"].evaluate(intent, market_price=100.0, has_trading_access=True)

    assert decision.rule_name != "ml_policy_rejected"
    assert decision.rule_name != "one_h10_ml_policy_not_enabled"
    assert decision.approved is True


def test_one_h10_cycle_summary_reports_target_roi_and_allocation_history(app) -> None:
    user, _ = _create_user("summary")
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDC",
        deposit_amount=100.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        status="active",
        algorithm_profile="1H10",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=100.0,
        current_estimated_value_usd=150.0,
    )
    cycle.selection_metadata = {
        "ml_horizon": "1h10",
        "target_amount_usd": 1000.0,
        "target_roi_pct": 1000.0,
        "ml_readiness": {"ready": False, "blockers": ["promoted_pytorch_risk_policy_missing"]},
        "exchange_allocation_history": [
            {
                "provider": "hyperliquid",
                "allocated_usd": 60.0,
                "legs": [{"symbol": "DOGE", "allocation_cap_usd": 60.0, "scanner_score": 3.5, "scanner_source": "1h10"}],
            }
        ],
        "provider_skip_reasons": [{"provider": "kucoin", "reason": "insufficient_free_margin"}],
    }
    db.session.add(cycle)
    db.session.flush()
    order = Order(
        user_id=user.id,
        client_order_id="rejected-1h10",
        mode="live",
        symbol="DOGE",
        side="buy",
        order_type="market",
        status="rejected",
        quantity=1.0,
        leverage=5.0,
        stop_loss=95.0,
        take_profit=110.0,
        rejection_reason="Projected slippage exceeds threshold.",
    )
    order.details = {
        "vault_cycle_id": cycle.id,
        "provider": "hyperliquid",
        "trading_connection_id": 10,
        "settlement_asset": "USDC",
        "blocker_category": "slippage_too_high",
    }
    db.session.add(order)
    db.session.commit()

    summary = _cycle_summary(cycle)

    assert summary["target_amount_usd"] == 1000.0
    assert summary["final_settlement_amount"] == 150.0
    assert summary["roi_pct"] == 50.0
    assert summary["exchange_allocation_history"][0]["provider"] == "hyperliquid"
    assert summary["provider_skip_reasons"][0]["reason"] == "insufficient_free_margin"
    assert summary["ranked_candidates"][0]["symbol"] == "DOGE"
    assert "insufficient_margin" in summary["blocker_categories"]
    assert "ml_not_ready" in summary["blocker_categories"]
    assert "slippage_too_high" in summary["blocker_categories"]
    assert summary["rejected_intents"][0]["client_order_id"] == "rejected-1h10"
    assert summary["rejected_order_count"] == 1


def test_one_h10_summary_treats_forecast_hold_low_confidence_as_advisory(app) -> None:
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False
    user, _ = _create_user("advisorysummary")
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDT",
        deposit_amount=5.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        status="active",
        algorithm_profile="1H10",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=5.0,
        current_estimated_value_usd=5.0,
    )
    cycle.selection_metadata = {
        "target_amount_usd": 50.0,
        "target_roi_pct": 1000.0,
        "blocker_categories": ["ml_hold"],
    }
    db.session.add(cycle)
    db.session.flush()
    leg = VaultAllocationLeg(
        vault_cycle_id=cycle.id,
        provider="hyperliquid",
        symbol="DOGE",
        timeframe="1m",
        allocation_cap_usd=5.0,
        status="active",
    )
    leg.details = {
        "forecast_blockers": ["forecast_hold", "low_confidence"],
        "forecast_advisory_blockers": ["forecast_hold", "low_confidence"],
    }
    db.session.add(leg)
    db.session.commit()

    summary = _cycle_summary(cycle)

    assert summary["forecast_blockers"] == []
    assert "forecast_hold" in summary["forecast_advisory_blockers"]
    assert "low_confidence" in summary["forecast_advisory_blockers"]
    assert "ml_hold" not in summary["blocker_categories"]


def test_one_h10_active_cycle_refreshes_stale_bootstrap_forecasts(app, monkeypatch) -> None:
    user, _ = _create_user("refreshforecast")
    stale_forecast = {
        "source": "one_h10_bootstrap_forecast",
        "ml_ready": False,
        "predicted_side": "hold",
        "confidence": 0.01,
        "blockers": [],
        "advisory_blockers": ["ml_not_ready", "forecast_hold"],
    }
    ready_forecast = {
        "source": "one_h10_ml_profit_suite",
        "ml_ready": True,
        "predicted_side": "buy",
        "action": "buy",
        "confidence": 0.88,
        "expected_return_bps": 45.0,
        "suggested_notional_usd": 12.0,
        "suggested_leverage": 2.0,
        "suggested_order_type": "market",
        "suggested_stop_loss_pct": 0.01,
        "suggested_take_profit_pct": 0.03,
        "blockers": [],
        "advisory_blockers": ["ml_fibonacci_confidence_below_minimum"],
    }

    class FakeForecastService:
        def forecast(self, features, **kwargs):
            assert features["scanner"] == "stored"
            assert kwargs["provider"] == "hyperliquid"
            assert kwargs["symbol"] == "DOGE"
            return ready_forecast

    monkeypatch.setitem(app.extensions["services"], "one_h10_forecast", FakeForecastService())
    monkeypatch.setattr(
        consumer_module,
        "_one_h10_ml_readiness",
        lambda provider="global": {
            "ready": True,
            "promoted_ready": True,
            "mode": "promoted",
            "display_status": "Ready",
            "blockers": [],
            "promoted_blockers": [],
            "source": "one_h10_live_execution_readiness",
        },
    )
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDC",
        deposit_amount=30.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        status="active",
        execution_substatus="executing",
        algorithm_profile="1H10",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=30.0,
        current_estimated_value_usd=30.0,
    )
    cycle.selection_metadata = {
        "ml_readiness": {
            "ready": False,
            "source": "one_h10_live_execution_readiness",
            "blockers": ["promoted_pytorch_fibonacci_missing"],
        },
        "blocker_categories": ["ml_not_ready"],
        "provider_allocation_history": [
            {
                "provider": "hyperliquid",
                "trading_connection_id": 11,
                "legs": [{"symbol": "DOGE", "market_id": 99, "forecast": stale_forecast}],
            }
        ],
        "exchange_allocation_history": [
            {
                "provider": "hyperliquid",
                "trading_connection_id": 11,
                "legs": [{"symbol": "DOGE", "market_id": 99, "forecast": stale_forecast}],
            }
        ],
    }
    db.session.add(cycle)
    db.session.flush()
    run = StrategyRun(
        user_id=user.id,
        trading_connection_id=11,
        strategy_name="scalping",
        symbol="DOGE",
        timeframe="1m",
        mode="live",
        status="stopped",
        manual_enabled=False,
    )
    run.parameters = {
        "one_h10_vault": True,
        "provider": "hyperliquid",
        "scanner_features": {"scanner": "stored", "one_h10_forecast": stale_forecast},
        "one_h10_forecast": stale_forecast,
        "forecast_advisory_blockers": ["ml_not_ready", "forecast_hold"],
    }
    db.session.add(run)
    db.session.flush()
    leg = VaultAllocationLeg(
        vault_cycle_id=cycle.id,
        strategy_run_id=run.id,
        provider="hyperliquid",
        trading_connection_id=11,
        symbol="DOGE",
        timeframe="1m",
        allocation_cap_usd=30.0,
        status="active",
    )
    leg.details = {
        "provider": "hyperliquid",
        "trading_connection_id": 11,
        "available_margin_usd": 30.0,
        "market_id": 99,
        "scanner_features": {"scanner": "stored", "one_h10_forecast": stale_forecast},
        "one_h10_forecast": stale_forecast,
        "forecast_advisory_blockers": ["ml_not_ready", "forecast_hold"],
    }
    db.session.add(leg)
    db.session.commit()

    assert _refresh_one_h10_cycle_ml_state(cycle) is True
    db.session.commit()

    assert cycle.selection_metadata["ml_readiness"]["ready"] is True
    assert cycle.selection_metadata["blocker_categories"] == []
    assert leg.details["one_h10_forecast"]["ml_ready"] is True
    assert leg.details["forecast_predicted_side"] == "buy"
    assert run.parameters["one_h10_forecast"]["source"] == "one_h10_ml_profit_suite"
    history_forecast = cycle.selection_metadata["provider_allocation_history"][0]["legs"][0]["forecast"]
    assert history_forecast["ml_ready"] is True
    summary = _cycle_summary(cycle)
    assert "ml_not_ready" not in summary["forecast_advisory_blockers"]
    assert summary["ml_readiness"]["display_status"] == "Ready"


def test_one_h10_active_cycle_resumes_stopped_strategy_runs(app) -> None:
    app.config["ONE_H10_AUTO_RESUME_ACTIVE_RUNS"] = True
    user, _ = _create_user("resumerun")
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDC",
        deposit_amount=30.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        status="active",
        execution_substatus="executing",
        algorithm_profile="1H10",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=30.0,
        current_estimated_value_usd=30.0,
    )
    db.session.add(cycle)
    db.session.flush()
    run = StrategyRun(
        user_id=user.id,
        strategy_name="scalping",
        symbol="DOGE",
        timeframe="1m",
        mode="live",
        status="stopped",
        manual_enabled=False,
    )
    run.parameters = {"one_h10_vault": True}
    db.session.add(run)
    db.session.flush()
    leg = VaultAllocationLeg(
        vault_cycle_id=cycle.id,
        strategy_run_id=run.id,
        provider="hyperliquid",
        symbol="DOGE",
        timeframe="1m",
        allocation_cap_usd=30.0,
        status="active",
    )
    db.session.add(leg)
    db.session.commit()

    assert _resume_one_h10_active_runs(cycle) == [run.id]
    assert run.manual_enabled is True
    assert run.status == "starting"
    assert cycle.selection_metadata["one_h10_auto_resumed_run_ids"] == [run.id]


def test_one_h10_queued_worker_state_is_visible_in_cycle_summary(app) -> None:
    app.config["WORKER_MODE"] = "web"
    app.config["ENABLE_IN_PROCESS_WORKERS"] = False
    user, _ = _create_user("queued1h10")
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDC",
        deposit_amount=25.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        lock_duration_seconds=app.config["ONE_H10_HORIZON_SECONDS"],
        status="active",
        execution_substatus="executing",
        execution_mode="live",
        algorithm_profile="1H10",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=25.0,
        current_estimated_value_usd=25.0,
    )
    db.session.add(cycle)
    db.session.flush()
    run = StrategyRun(
        user_id=user.id,
        strategy_name="scalping",
        symbol="BTC",
        timeframe="1m",
        mode="live",
        status="starting",
        manual_enabled=True,
    )
    run.parameters = {
        "vault_cycle_id": cycle.id,
        "one_h10_vault": True,
        "ml_horizon": "1h10",
        "algorithm_profile": "1H10",
    }
    db.session.add(run)
    db.session.flush()
    leg = VaultAllocationLeg(
        vault_cycle_id=cycle.id,
        strategy_run_id=run.id,
        symbol="BTC",
        timeframe="1m",
        provider="hyperliquid",
        allocation_cap_usd=25.0,
        leverage=1.0,
        status="active",
    )
    leg.details = {"one_h10_vault": True, "ml_horizon": "1h10", "provider": "hyperliquid"}
    db.session.add(leg)
    db.session.commit()

    consumer_module._start_strategy_runs([run.id])
    db.session.refresh(run)

    summary = _cycle_summary(
        cycle,
        performance={"realized_pnl": 0.0, "unrealized_pnl": 0.0, "total_pnl": 0.0, "has_trading_data": False},
    )

    assert run.last_signal["metadata"]["trade_decision_stage"] == "queued_for_worker"
    assert summary["trade_decision"]["stage"] == "queued_for_worker"
    assert summary["trade_decision"]["broker_order_submitted"] is False
    assert summary["trade_decision_legs"][0]["stage"] == "queued_for_worker"
    assert summary["trade_decision_legs"][0]["reason"] == "strategy_worker_pending"
    assert summary["worker"]["strategy_run_queue"] == "dedicated_worker"
    assert summary["worker"]["queued_run_count"] == 1
    assert summary["worker"]["runtime_health"]["health"] == "blocked"
    assert summary["worker"]["runtime_health"]["recent_heartbeat"] is False
    assert "no recent dedicated worker heartbeat" in summary["worker"]["runtime_health"]["blockers"][0]
    assert summary["worker"]["live_order_path"] == "VaultCycle -> StrategyRun -> Worker -> RiskEngine -> OrderManager"


def test_one_h10_rebalance_preserves_distinct_provider_symbols(app, monkeypatch) -> None:
    manager = app.extensions["services"]["strategy_manager"]
    user, _ = _create_user("distinctrebalance")
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDC",
        deposit_amount=30.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        status="active",
        execution_substatus="executing",
        algorithm_profile="1H10",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=30.0,
        current_estimated_value_usd=30.0,
    )
    db.session.add(cycle)
    db.session.flush()
    current = StrategyRun(
        user_id=user.id,
        trading_connection_id=2,
        strategy_name="scalping",
        symbol="ETH",
        timeframe="1m",
        mode="live",
        status="running",
        manual_enabled=True,
    )
    current.parameters = {
        "one_h10_vault": True,
        "vault_cycle_id": cycle.id,
        "provider": "hyperliquid",
        "app_symbol": "ETH",
        "venue_symbol": "ETH",
        "provider_symbol": "ETH",
        "allocation_cap_usd": 10.0,
        "available_margin_usd": 30.0,
    }
    sibling = StrategyRun(
        user_id=user.id,
        trading_connection_id=2,
        strategy_name="scalping",
        symbol="BTC",
        timeframe="1m",
        mode="live",
        status="running",
        manual_enabled=True,
    )
    sibling.parameters = {
        "one_h10_vault": True,
        "vault_cycle_id": cycle.id,
        "provider": "hyperliquid",
        "app_symbol": "BTC",
        "venue_symbol": "BTC",
        "provider_symbol": "BTC",
    }
    db.session.add_all([current, sibling])
    db.session.flush()
    current_leg = VaultAllocationLeg(
        vault_cycle_id=cycle.id,
        strategy_run_id=current.id,
        provider="hyperliquid",
        trading_connection_id=2,
        symbol="ETH",
        timeframe="1m",
        allocation_cap_usd=10.0,
        status="active",
    )
    sibling_leg = VaultAllocationLeg(
        vault_cycle_id=cycle.id,
        strategy_run_id=sibling.id,
        provider="hyperliquid",
        trading_connection_id=2,
        symbol="BTC",
        timeframe="1m",
        allocation_cap_usd=10.0,
        status="active",
    )
    db.session.add_all([current_leg, sibling_leg])
    db.session.flush()
    current.parameters = {**current.parameters, "vault_leg_id": current_leg.id}
    sibling.parameters = {**sibling.parameters, "vault_leg_id": sibling_leg.id}
    current_leg.details = {"app_symbol": "ETH", "venue_symbol": "ETH", "provider_symbol": "ETH"}
    sibling_leg.details = {"app_symbol": "BTC", "venue_symbol": "BTC", "provider_symbol": "BTC"}
    db.session.commit()

    candidates = [
        SimpleNamespace(
            symbol="BTC",
            score=100.0,
            technical_score=100.0,
            ml_score=0.0,
            hot_score=0.0,
            source="test",
            rejection_reason="",
            stale_data=False,
            score_breakdown={},
            features={"symbol": "BTC", "app_symbol": "BTC", "venue_symbol": "BTC", "market_id": 1},
        ),
        SimpleNamespace(
            symbol="DOGE",
            score=90.0,
            technical_score=90.0,
            ml_score=0.0,
            hot_score=0.0,
            source="test",
            rejection_reason="",
            stale_data=False,
            score_breakdown={},
            features={"symbol": "DOGE", "app_symbol": "DOGE", "venue_symbol": "DOGE", "market_id": 2},
        ),
    ]
    monkeypatch.setattr(manager.order_manager, "current_position", lambda *args, **kwargs: {"quantity": 0.0})
    monkeypatch.setattr(
        app.extensions["services"]["leveraged_markets"], "active_markets", lambda **kwargs: [SimpleNamespace(id=1), SimpleNamespace(id=2)]
    )
    monkeypatch.setattr(app.extensions["services"]["market_scanner"], "score_one_h10_markets", lambda *args, **kwargs: candidates)

    class FakeForecast:
        def forecast(self, features, *, symbol, **kwargs):
            return {
                "symbol": symbol,
                "predicted_side": "buy",
                "confidence": 0.75,
                "expected_return_bps": 10.0,
                "suggested_notional_usd": 5.0,
                "suggested_leverage": 1.0,
                "suggested_order_type": "limit",
                "suggested_stop_loss_pct": 0.01,
                "suggested_take_profit_pct": 0.03,
                "blockers": [],
                "advisory_blockers": [],
            }

    monkeypatch.setitem(app.extensions["services"], "one_h10_forecast", FakeForecast())

    manager._one_h10_rebalance_tick(current)

    assert current.symbol == "DOGE"
    assert current.parameters["venue_symbol"] == "DOGE"
    assert current.parameters["one_h10_forecast"]["symbol"] == "DOGE"
    assert current_leg.symbol == "DOGE"
    assert current_leg.details["venue_symbol"] == "DOGE"
    assert current_leg.details["one_h10_forecast"]["symbol"] == "DOGE"
