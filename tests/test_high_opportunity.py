from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pyotp

from app.auth import encrypt_totp_secret, password_hash
from app.backtesting.engine import BacktestConfig, BacktestEngine
from app.extensions import db
from app.models import (
    MLTrainingEvent,
    OptimizerRun,
    Setting,
    StrategyRanking,
    StrategyRun,
    StrategyValidation,
    User,
    VaultAllocationLeg,
    VaultCycle,
    WalletBalance,
)
from app.services.hyperliquid_client import ClientSnapshot
from app.services.order_manager import OrderIntent
from app.strategies.base import Signal


def _book(spread: float = 0.1, size: str = "1000") -> dict[str, Any]:
    bid = 100.0 - spread / 2
    ask = 100.0 + spread / 2
    return {"levels": [[{"px": str(bid), "sz": size}], [{"px": str(ask), "sz": size}]]}


def _candles(count: int = 100, step: float = 0.08) -> list[dict[str, Any]]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    price = 100.0
    rows = []
    for index in range(count):
        price += step if index % 2 == 0 else -step / 2
        rows.append(
            {
                "timestamp": int((start + timedelta(minutes=index)).timestamp() * 1000),
                "open": price - 0.02,
                "high": price + 0.08,
                "low": price - 0.08,
                "close": price,
                "volume": 1000,
            }
        )
    return rows


def _correlated_pair_candles(symbol: str, count: int = 120) -> list[dict[str, Any]]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for index in range(count):
        base = 100.0 + index * 0.08 + (index % 7) * 0.03
        price = base * 0.95 if symbol == "ETH" else base
        rows.append(
            {
                "timestamp": int((start + timedelta(minutes=index)).timestamp() * 1000),
                "open": price - 0.02,
                "high": price + 0.08,
                "low": price - 0.08,
                "close": price,
                "volume": 1000,
            }
        )
    return rows


def _create_user(username: str = "opportunity", role: str = "user") -> tuple[User, str]:
    secret = pyotp.random_base32()
    user = User(username=username, password_hash=password_hash("password123"), role=role)
    user.totp_secret_encrypted = encrypt_totp_secret(secret)
    user.two_factor_enabled_at = datetime.utcnow()
    db.session.add(user)
    db.session.commit()
    return user, secret


def _login(client, username: str, secret: str):
    response = client.post(
        "/login",
        data={"username": username, "password": "password123", "totp_code": pyotp.TOTP(secret).now()},
        follow_redirects=False,
    )
    user = User.query.filter_by(username=username).one_or_none()
    if user is not None and response.status_code in {302, 303}:
        _create_live_connection(client.application, user)
    return response


def _create_live_connection(app, user):
    connection = app.extensions["services"]["trading_connections"].create_or_update(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        api_secret="0x" + ("1" * 64),
        wallet_address="0x" + ("2" * 40),
        is_active=True,
    )
    connection.verification_status = "verified"
    connection.is_active = True
    db.session.commit()
    app.extensions["services"]["trading_connections"].account_snapshot = lambda user_id, mode, connection_id=None: ClientSnapshot(
        mode,
        [{"asset": "USDC", "type": "margin", "value": 1_000.0, "withdrawable": 1_000.0}],
        [],
        [],
        [],
        [],
    )
    return connection


def _confirm_one_h10_live(app) -> None:
    app.config["EXPLICIT_LIVE_CONFIRMED"] = True
    app.config["SECONDARY_CONFIRMATION"] = True
    Setting.set_json("explicit_live_confirmed", True)
    Setting.set_json("secondary_confirmation", True)
    db.session.commit()


class _BuyThenReduceStrategy:
    parameters: dict[str, Any] = {}

    def generate_signal(self, *, symbol: str, timeframe: str, candles: list[dict[str, Any]], position: dict[str, Any]) -> Signal:
        if position.get("quantity", 0.0):
            return Signal("reduce", "close", timeframe, None, None, 0.0)
        return Signal("buy", "open", timeframe, candles[-1]["close"] * 0.99, candles[-1]["close"] * 1.02, 0.2)


class _Registry:
    def build(self, name: str, parameters: dict[str, Any] | None = None):
        return _BuyThenReduceStrategy()


class _MarketData:
    def get_candles(self, *args, **kwargs):
        return []


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
        provider_floor = 10.0 if str(provider).lower() == "hyperliquid" else 5.0
        suggested_notional = min(
            value
            for value in [
                float(allocation_cap_usd or provider_floor),
                float(available_margin_usd or allocation_cap_usd or provider_floor),
                provider_floor,
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
            "cost_drag_bps": 8.0,
            "spread_bps": 1.0,
            "execution_quality": 0.9,
            "capital_efficiency_score": 1.0,
            "expected_net_edge_passed": True,
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


def test_dynamic_universe_filters_liquid_pairs_and_falls_back(app) -> None:
    app.config["DYNAMIC_UNIVERSE_ENABLED"] = True
    app.config["UNIVERSE_MIN_LIQUIDITY_USD"] = 50_000.0
    app.config["UNIVERSE_MAX_SPREAD_BPS"] = 20.0
    app.config["UNIVERSE_SYMBOL_BLACKLIST"] = ["DOGE"]
    service = app.extensions["services"]["market_universe"]
    service._cache.clear()
    market_data = app.extensions["services"]["market_data"]
    market_data.client.get_all_mids = lambda mode: {"BTC": 100.0, "ETH": 100.0, "DOGE": 1.0, "XRP": 100.0}
    market_data.get_order_book = lambda symbol, mode: {
        "BTC": _book(spread=0.1, size="1000"),
        "ETH": _book(spread=5.0, size="1000"),
        "XRP": _book(spread=0.1, size="10"),
    }[symbol]
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles()

    candidates = service.liquid_universe("testnet", "1m")

    assert [candidate.symbol for candidate in candidates] == ["BTC"]
    assert candidates[0].cost_drag_bps > 0
    assert candidates[0].market_structure_score > 0
    assert candidates[0].volatility_regime in {"compressed", "tradable", "elevated", "dislocated"}
    assert service.last_rejections["blacklisted"] == 1
    assert service.last_rejections["spread_above_threshold"] == 1
    assert service.last_rejections["liquidity_below_threshold"] == 1

    service._cache.clear()
    market_data.client.get_all_mids = lambda mode: {}
    fallback = service.liquid_universe("testnet", "1m")
    assert [candidate.symbol for candidate in fallback] == app.config["ALLOWED_SYMBOLS"]


def test_dynamic_universe_cache_preserves_rejection_reasons(app) -> None:
    app.config["DYNAMIC_UNIVERSE_ENABLED"] = True
    app.config["UNIVERSE_MIN_LIQUIDITY_USD"] = 50_000.0
    app.config["UNIVERSE_MAX_SPREAD_BPS"] = 20.0
    service = app.extensions["services"]["market_universe"]
    service._cache.clear()
    service._rejection_cache.clear()
    market_data = app.extensions["services"]["market_data"]
    market_data.client.get_all_mids = lambda mode: {"BTC": 100.0, "ETH": 100.0}
    market_data.get_order_book = lambda symbol, mode: {
        "BTC": _book(spread=0.1, size="1000"),
        "ETH": _book(spread=5.0, size="1000"),
    }[symbol]
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles()

    assert [candidate.symbol for candidate in service.liquid_universe("testnet", "1m")] == ["BTC"]
    assert service.last_rejections == {"spread_above_threshold": 1}

    service.last_rejections = {"stale_reason": 99}
    assert [candidate.symbol for candidate in service.liquid_universe("testnet", "1m")] == ["BTC"]
    assert service.last_rejections == {"spread_above_threshold": 1}


def test_market_scanner_includes_tradability_in_dynamic_scores(app) -> None:
    app.config["DYNAMIC_UNIVERSE_ENABLED"] = True
    app.config["UNIVERSE_MIN_LIQUIDITY_USD"] = 10_000.0
    app.config["UNIVERSE_MAX_SPREAD_BPS"] = 30.0
    scanner = app.extensions["services"]["market_scanner"]
    scanner._score_cache.clear()
    scanner._hot_cache.clear()
    market_data = app.extensions["services"]["market_data"]
    market_data.client.get_all_mids = lambda mode: {"BTC": 100.0, "ETH": 100.0}
    market_data.get_order_book = lambda symbol, mode: _book(spread=0.02 if symbol == "BTC" else 0.20, size="1000")
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles()
    app.extensions["services"]["market_universe"]._cache.clear()

    scored = scanner.score_candidates(
        ["BTC", "ETH"],
        mode="testnet",
        timeframe="1m",
        duration_seconds=3600,
        strategy_name="scalping",
        optimizer_profile="dynamic_intraday",
    )

    assert [candidate.symbol for candidate in scored] == ["BTC", "ETH"]
    assert scored[0].features["tradability"]["spread_bps"] < scored[1].features["tradability"]["spread_bps"]
    assert scored[0].features["cost_drag_bps"] < scored[1].features["cost_drag_bps"]
    assert scored[0].features["market_structure_score"] >= scored[1].features["market_structure_score"]
    assert "upside_screen_score" in scored[0].features
    assert "scanner_score_breakdown" in scored[0].features
    assert "momentum_acceleration" in scored[0].features
    assert "volume_impulse_persistence" in scored[0].features
    assert "breakout_proximity_bps" in scored[0].features
    assert "cost_adjusted_expected_move" in scored[0].features
    assert "net_roi_score" in scored[0].features
    assert "net_roi_v2_score" in scored[0].features
    assert "roi_quality_grade" in scored[0].features
    assert "roi_rejection_risk" in scored[0].features
    assert "regime_support" in scored[0].features
    assert "sustained_volume_impulse" in scored[0].features
    assert "pullback_quality" in scored[0].features
    assert "breakout_retest_success" in scored[0].features
    assert "volatility_expansion_after_compression" in scored[0].features
    assert "cost_adjusted_expected_move_persistence" in scored[0].features
    assert "expected_fill_quality" in scored[0].features
    assert "churn_penalty" in scored[0].features
    assert "one_hour_edge_v2" in scored[0].features
    assert "one_hour_edge_grade" in scored[0].features
    assert "expected_execution_quality" in scored[0].features
    assert "candidate_quality_breakdown" in scored[0].features


def test_market_scanner_high_upside_diagnostics_align_accepted_and_rejected(app) -> None:
    app.config["DYNAMIC_UNIVERSE_ENABLED"] = True
    app.config["HIGH_UPSIDE_PROFILE_ENABLED"] = True
    app.config["UNIVERSE_MIN_LIQUIDITY_USD"] = 10_000.0
    app.config["UNIVERSE_MAX_SPREAD_BPS"] = 30.0
    app.config["AGGRESSIVE_1H_MAX_COST_DRAG_BPS"] = 22.0
    scanner = app.extensions["services"]["market_scanner"]
    scanner._score_cache.clear()
    scanner._hot_cache.clear()
    scanner._diagnostic_cache.clear()
    market_data = app.extensions["services"]["market_data"]
    market_data.client.get_all_mids = lambda mode: {"BTC": 100.0, "ETH": 100.0, "SOL": 100.0, "DOGE": 100.0}
    market_data.get_order_book = lambda symbol, mode: {
        "BTC": _book(spread=0.02, size="1000"),
        "ETH": _book(spread=0.16, size="1000"),
        "SOL": _book(spread=0.50, size="1000"),
        "DOGE": _book(spread=0.02, size="10"),
    }[symbol]
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles(limit, step=0.12 if symbol == "BTC" else 0.04)
    app.extensions["services"]["market_universe"]._cache.clear()

    scored = scanner.score_candidates(
        ["BTC", "ETH", "SOL", "DOGE"],
        mode="testnet",
        timeframe="1m",
        duration_seconds=3600,
        strategy_name="scalping",
        optimizer_profile="dynamic_intraday",
    )
    diagnostics = scanner.last_scan_diagnostics

    assert [candidate.symbol for candidate in scored] == ["BTC"]
    assert diagnostics["scan_key"]["optimizer_profile"] == "dynamic_intraday"
    assert diagnostics["accepted"][0]["symbol"] == "BTC"
    assert diagnostics["accepted"][0]["score_breakdown"]["cost_penalty"] <= 0
    assert diagnostics["accepted"][0]["net_roi_score"] > 0
    assert diagnostics["accepted"][0]["net_roi_v2_score"] > 0
    assert diagnostics["accepted"][0]["roi_quality_grade"] in {"A", "B", "C", "D"}
    assert diagnostics["accepted"][0]["regime_support"] in {"regime-supported", "regime-neutral", "regime-fragile"}
    assert diagnostics["accepted"][0]["expected_fill_quality"] >= app.config["NET_ROI_MIN_FILL_QUALITY"]
    assert diagnostics["accepted"][0]["one_hour_edge_grade"] in {"A", "B", "C", "D"}
    assert "one_hour_edge_v2" in diagnostics["accepted"][0]["score_breakdown"]
    assert "candidate_quality_breakdown" in diagnostics["accepted"][0]
    assert "volume_persistence" in diagnostics["accepted"][0]["score_breakdown"]
    assert "net_roi" in diagnostics["accepted"][0]["score_breakdown"]
    assert "net_roi_v2" in diagnostics["accepted"][0]["score_breakdown"]
    assert "offline_ml_status" in diagnostics["accepted"][0]
    assert "market_data_cache" in diagnostics
    accepted_keys = set(diagnostics["accepted"][0])
    for row in diagnostics["rejected"]:
        assert set(row) == accepted_keys
    assert {row["symbol"]: row["rejection_reason"] for row in diagnostics["rejected"]} == {
        "ETH": "cost_drag_above_threshold",
        "SOL": "spread_above_threshold",
        "DOGE": "liquidity_below_threshold",
    }
    assert diagnostics["rejection_breakdown"] == {
        "cost_drag_above_threshold": 1,
        "spread_above_threshold": 1,
        "liquidity_below_threshold": 1,
    }
    assert diagnostics["rejection_rate"] == 0.75


def test_market_scanner_cache_preserves_scan_diagnostics(app) -> None:
    app.config["DYNAMIC_UNIVERSE_ENABLED"] = True
    app.config["HIGH_UPSIDE_PROFILE_ENABLED"] = True
    app.config["UNIVERSE_MIN_LIQUIDITY_USD"] = 10_000.0
    app.config["UNIVERSE_MAX_SPREAD_BPS"] = 30.0
    app.config["AGGRESSIVE_1H_MAX_COST_DRAG_BPS"] = 22.0
    scanner = app.extensions["services"]["market_scanner"]
    scanner._score_cache.clear()
    scanner._hot_cache.clear()
    scanner._diagnostic_cache.clear()
    market_data = app.extensions["services"]["market_data"]
    market_data.client.get_all_mids = lambda mode: {"BTC": 100.0, "ETH": 100.0}
    market_data.get_order_book = lambda symbol, mode: _book(spread=0.02 if symbol == "BTC" else 0.16, size="1000")
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles(limit, step=0.12)
    app.extensions["services"]["market_universe"]._cache.clear()

    scanner.score_candidates(
        ["BTC", "ETH"],
        mode="testnet",
        timeframe="1m",
        duration_seconds=3600,
        strategy_name="scalping",
        optimizer_profile="dynamic_intraday",
    )
    first = dict(scanner.last_scan_diagnostics)
    scanner.last_scan_diagnostics = {"rejection_breakdown": {"stale": 99}}
    scanner.score_candidates(
        ["BTC", "ETH"],
        mode="testnet",
        timeframe="1m",
        duration_seconds=3600,
        strategy_name="scalping",
        optimizer_profile="dynamic_intraday",
    )

    assert scanner.last_scan_diagnostics["cache_hit"] is True
    assert scanner.last_scan_diagnostics["rejection_breakdown"] == first["rejection_breakdown"]


def test_market_data_cache_reuses_provider_snapshots(app) -> None:
    market_data = app.extensions["services"]["market_data"]
    market_data.clear_cache()
    app.config["MARKET_DATA_RESEARCH_CACHE_TTL_SECONDS"] = 60
    calls = {"candles": 0, "book": 0, "mids": 0}

    def candles(mode, symbol, timeframe, start_ms, end_ms):
        calls["candles"] += 1
        return [{"t": index + 1, "o": "100", "h": "101", "l": "99", "c": str(100 + index), "v": "10"} for index in range(5)]

    def book(mode, symbol):
        calls["book"] += 1
        return _book(spread=0.02, size="1000")

    def mids(mode):
        calls["mids"] += 1
        return {"BTC": 100.0}

    market_data.client.get_candles = candles
    market_data.client.get_order_book = book
    market_data.client.get_all_mids = mids

    assert market_data.get_candles("BTC", "1m", mode="testnet", limit=5)
    assert market_data.get_candles("BTC", "1m", mode="testnet", limit=5)
    assert market_data.get_order_book("BTC", "testnet")
    assert market_data.get_order_book("BTC", "testnet")
    assert market_data.get_all_mids("testnet") == {"BTC": 100.0}
    assert market_data.get_all_mids("testnet") == {"BTC": 100.0}

    assert calls == {"candles": 1, "book": 1, "mids": 1}
    stats = market_data.cache_stats()
    assert stats["hits"] >= 3
    assert stats["hit_rate"] > 0


def test_market_scanner_rejects_stale_realtime_data_with_aligned_diagnostics(app) -> None:
    app.config["REALTIME_MARKET_ENABLED"] = True
    app.config["REALTIME_MARKET_MAX_STALE_SECONDS"] = 1.0
    app.config["UNIVERSE_MIN_LIQUIDITY_USD"] = 1_000.0
    app.config["UNIVERSE_MAX_SPREAD_BPS"] = 30.0
    scanner = app.extensions["services"]["market_scanner"]
    scanner._score_cache.clear()
    scanner._hot_cache.clear()
    scanner._diagnostic_cache.clear()
    market_data = app.extensions["services"]["market_data"]
    market_data.client.get_all_mids = lambda mode: {"BTC": 100.0}
    market_data.get_order_book = lambda symbol, mode: _book(spread=0.02, size="1000")
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles(limit, step=0.12)
    app.extensions["services"]["market_universe"]._cache.clear()

    scored = scanner.score_candidates(
        ["BTC"],
        mode="live",
        timeframe="1m",
        duration_seconds=3600,
        strategy_name="scalping",
        optimizer_profile="aggressive_1h",
    )

    assert scored == []
    rejected = scanner.last_scan_diagnostics["rejected"][0]
    assert rejected["rejection_reason"] == "stale_market_data"
    assert rejected["stale_data"] is True
    assert rejected["stale_data_age_seconds"] > 0


def test_optimizer_accepts_horizon_amount_dynamic_universe_and_leverage(app) -> None:
    app.config["LEVERAGE_OPTIMIZER_ENABLED"] = True
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(
        symbols=["BTC"],
        profile="aggressive_1h",
        allocation_amount_usd=250.0,
        lock_duration_hours=4,
        universe_mode="dynamic_liquid",
        max_parallel_legs=3,
        allow_leverage_experiment=True,
    )

    assert config.training_window_hours == 120
    assert config.testing_window_hours == 4
    assert config.step_hours == 1
    assert config.allocation_amount_usd == 250.0
    assert config.universe_mode == "dynamic_liquid"
    assert config.max_parallel_legs == 3
    assert any(row.get("leverage", 1.0) > 1.0 for row in optimizer._parameter_sets("scalping", 10, config))


def test_backtest_models_leverage_funding_and_liquidation_buffer() -> None:
    engine = BacktestEngine({}, _Registry(), _MarketData())
    base = {
        "strategy_name": "test",
        "symbol": "BTC",
        "timeframe": "15m",
        "mode": "testnet",
        "initial_balance": 1_000.0,
        "fee_bps": 5.0,
        "slippage_bps": 5.0,
        "stop_loss_pct": 0.01,
        "take_profit_pct": 0.02,
        "position_size_fraction": 0.2,
        "parameters": {},
        "max_daily_loss": 100.0,
        "max_drawdown_pct": 0.5,
        "intrabar_model": "conservative",
    }
    one_x = engine.run(BacktestConfig(**base, leverage=1.0), _candles())
    three_x = engine.run(BacktestConfig(**base, leverage=3.0, funding_cost_bps=1.0), _candles())
    blocked = engine.run(
        BacktestConfig(**base, leverage=10.0, min_liquidation_buffer_pct=0.2),
        _candles(),
    )

    assert three_x["capital_turnover_rate"] > one_x["capital_turnover_rate"]
    assert three_x["funding_cost_estimate"] > 0
    assert blocked["trade_count"] == 0
    assert any(event["rule"] == "liquidation_buffer_too_tight" for event in blocked["risk_events"])


def test_vault_cycle_creates_multi_leg_strategy_runs(app, monkeypatch) -> None:
    app.config["DYNAMIC_UNIVERSE_ENABLED"] = True
    app.config["LEVERAGE_OPTIMIZER_ENABLED"] = True
    app.config["VAULT_MAX_PARALLEL_LEGS"] = 3
    _confirm_one_h10_live(app)
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: _book(spread=0.1, size="1000")
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles()
    app.extensions["services"]["market_universe"].symbols = lambda mode, timeframe: ["BTC", "ETH", "SOL"]
    monkeypatch.setitem(app.extensions["services"], "one_h10_forecast", _PassingOneH10Forecast())
    started: list[int] = []
    app.extensions["services"]["strategy_manager"].start = lambda run_id: started.append(run_id)

    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    for index, symbol in enumerate(["BTC", "ETH", "SOL"]):
        ranking = StrategyRanking(
            optimizer_run_id=optimizer_run.id,
            strategy_name="scalping",
            symbol=symbol,
            timeframe="1m",
            profile="aggressive_1h",
            experimental=True,
            risk_label="Very High Risk",
            score=3.0 - index * 0.2,
            recent_1h_return=0.03,
            max_drawdown=-0.05,
            profit_factor=1.5,
            trade_count=12,
            edge_score=20.0 - index,
            execution_style="maker_limit",
            universe_source="dynamic_liquid",
            rejected=False,
        )
        ranking.parameters = {"risk_fraction": 0.02, "leverage": 2.0}
        db.session.add(ranking)
    user, secret = _create_user()
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=500.0, estimated_usd_value=500.0))
    db.session.commit()

    client = app.test_client()
    _login(client, user.username, secret)
    response = client.post(
        "/vault/start",
        data={
            "deposit_amount": "120",
            "deposit_asset": "USDC",
            "lock_duration": "1",
            "settlement_asset": "USDC",
            "one_h10_live_ack": "on",
        },
    )

    assert response.status_code == 302
    cycle = VaultCycle.query.filter_by(user_id=user.id).one()
    legs = VaultAllocationLeg.query.filter_by(vault_cycle_id=cycle.id).all()
    assert len(legs) == 3
    assert len(started) == 3
    assert StrategyRun.query.count() == 3
    assert sum(leg.allocation_cap_usd for leg in legs) <= 120.0 + 1e-9
    assert all(db.session.get(StrategyRun, leg.strategy_run_id).parameters["vault_leg_id"] == leg.id for leg in legs)


def test_one_h10_all_market_scanner_ignores_legacy_pair_stat_arb_legs(app, monkeypatch) -> None:
    app.config["PAIR_SCREENING_ENABLED"] = True
    app.config["PAIR_TRADING_ENABLED"] = True
    app.config["PAIR_MIN_CORRELATION"] = 0.75
    app.config["PAIR_MAX_SPREAD_ZSCORE"] = 2.5
    app.config["PAIR_MIN_LIQUIDITY_USD"] = 25_000.0
    app.config["PAIR_MAX_SPREAD_BPS"] = 20.0
    _confirm_one_h10_live(app)
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: _book(spread=0.02, size="1000")
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _correlated_pair_candles(symbol, limit)
    monkeypatch.setitem(app.extensions["services"], "one_h10_forecast", _PassingOneH10Forecast())
    started: list[int] = []
    app.extensions["services"]["strategy_manager"].start = lambda run_id: started.append(run_id)
    user, secret = _create_user("pairuser")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=500.0, estimated_usd_value=500.0))
    db.session.commit()

    client = app.test_client()
    _login(client, user.username, secret)
    response = client.post(
        "/vault/start",
        data={
            "deposit_amount": "120",
            "deposit_asset": "USDC",
            "lock_duration": "1",
            "settlement_asset": "USDC",
            "one_h10_live_ack": "on",
        },
    )

    assert response.status_code == 302
    cycle = VaultCycle.query.filter_by(user_id=user.id).one()
    legs = VaultAllocationLeg.query.filter_by(vault_cycle_id=cycle.id).order_by(VaultAllocationLeg.id).all()
    assert len(legs) == 3
    assert len(started) == 3
    assert all(leg.details["one_h10_vault"] is True for leg in legs)
    assert all(leg.details.get("pair_mode") is None for leg in legs)
    assert all(db.session.get(StrategyRun, leg.strategy_run_id).parameters["one_h10_all_pairs"] is True for leg in legs)
    assert sum(float(leg.allocation_cap_usd or 0.0) for leg in legs) <= 120.0 + 1e-9


def test_vault_cycle_creates_strategy_basket_without_dynamic_universe(app) -> None:
    app.config["DYNAMIC_UNIVERSE_ENABLED"] = False
    app.config["VAULT_STRATEGY_BASKET_ENABLED"] = True
    app.config["VAULT_MAX_PARALLEL_LEGS"] = 3
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: _book(spread=0.1, size="1000")
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles()
    started: list[int] = []
    app.extensions["services"]["strategy_manager"].start = lambda run_id: started.append(run_id)

    optimizer_run = OptimizerRun(profile="aggressive_risk_adjusted", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    for index, strategy in enumerate(["scalping", "rsi_mean_reversion", "volatility_breakout"]):
        ranking = StrategyRanking(
            optimizer_run_id=optimizer_run.id,
            strategy_name=strategy,
            symbol="BTC",
            timeframe="5m",
            profile="aggressive_risk_adjusted",
            score=4.0 - index * 0.2,
            net_return_after_costs=0.04,
            recent_performance_score=0.03,
            recent_1h_return=0.02,
            max_drawdown=-0.04,
            profit_factor=1.5,
            trade_count=18,
            edge_score=18.0 - index,
            rejected=False,
        )
        ranking.parameters = {"risk_fraction": 0.02, "leverage": 1.0}
        db.session.add(ranking)
    user, secret = _create_user(username="strategy-basket")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=500.0, estimated_usd_value=500.0))
    db.session.commit()

    client = app.test_client()
    _login(client, user.username, secret)
    response = client.post(
        "/vault/start",
        data={
            "deposit_amount": "120",
            "deposit_asset": "USDC",
            "lock_duration": "24",
            "settlement_asset": "USDC",
        },
    )

    assert response.status_code == 302
    cycle = VaultCycle.query.filter_by(user_id=user.id).one()
    legs = VaultAllocationLeg.query.filter_by(vault_cycle_id=cycle.id).all()
    assert len(legs) == 3
    assert {leg.strategy_run.strategy_name for leg in legs} == {"scalping", "rsi_mean_reversion", "volatility_breakout"}
    assert len(started) == 3
    assert cycle.selection_metadata["allocation_mode"] == "ranked_strategy_basket"


def test_one_hour_ensemble_allocator_weights_and_caps_strategy_legs(app) -> None:
    app.config["DYNAMIC_UNIVERSE_ENABLED"] = False
    app.config["ENSEMBLE_1H_ENABLED"] = True
    app.config["ENSEMBLE_1H_MAX_LEGS"] = 3
    app.config["ENSEMBLE_1H_MAX_SYMBOL_PCT"] = 0.50
    app.config["ENSEMBLE_1H_MAX_STRATEGY_PCT"] = 0.60
    app.config["ENSEMBLE_1H_MIN_EDGE_BPS"] = 5.0
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: _book(spread=0.1, size="1000")
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles()

    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    for index, strategy in enumerate(["scalping", "rsi_mean_reversion", "volatility_breakout"]):
        ranking = StrategyRanking(
            optimizer_run_id=optimizer_run.id,
            strategy_name=strategy,
            symbol="BTC",
            timeframe="1m",
            profile="aggressive_1h",
            experimental=True,
            risk_label="Very High Risk",
            score=3.0 - index * 0.2,
            net_return_after_costs=0.05 - index * 0.005,
            recent_performance_score=0.03,
            recent_1h_return=0.025 - index * 0.002,
            max_drawdown=-0.04,
            profit_factor=1.6,
            win_rate=0.58,
            trade_count=18,
            trades_per_day=24.0,
            edge_score=18.0 - index,
            expectancy=1.4,
            max_favorable_excursion=0.012,
            max_adverse_excursion=-0.004,
            capacity_usd=10_000.0,
            rejected=False,
        )
        ranking.parameters = {"risk_fraction": 0.02, "leverage": 1.0}
        db.session.add(ranking)
    db.session.commit()

    selection = app.extensions["services"]["vault_strategy_selector"].select("BTC", 1, "paper", 120.0)

    assert selection.metadata["allocation_mode"] == "1h_live_aggressive_ensemble"
    assert selection.metadata["ensemble_id"].startswith("ensemble-1h-btc")
    assert len(selection.legs) >= 2
    assert {leg["strategy_name"] for leg in selection.legs}.issubset({"scalping", "rsi_mean_reversion", "volatility_breakout"})
    assert sum(leg["allocation_cap_usd"] for leg in selection.legs) <= 60.0 + 1e-9
    assert all(leg["ensemble_weight"] > 0 for leg in selection.legs)
    assert all("fib_confluence" in leg for leg in selection.legs)


def test_enhanced_ensemble_selects_weighted_legs_with_v2_metadata(app) -> None:
    app.config["DYNAMIC_UNIVERSE_ENABLED"] = False
    app.config["ENSEMBLE_ENHANCED_ENABLED"] = True
    app.config["ENSEMBLE_MAX_LEGS"] = 5
    app.config["ENSEMBLE_MAX_SYMBOL_PCT"] = 0.50
    app.config["ENSEMBLE_MAX_STRATEGY_PCT"] = 0.60
    app.config["ENSEMBLE_MIN_EDGE_BPS"] = 5.0
    app.config["ENSEMBLE_MIN_SHARPE"] = 0.4
    app.config["FIB_CONFLUENCE_THRESHOLD"] = 0.01
    app.config["FIB_CONFLUENCE_CLUSTER_TOLERANCE_BPS"] = 300.0
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: _book(spread=0.1, size="1000")
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles(max(limit, 140))

    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    for index, strategy in enumerate(["scalping", "rsi_mean_reversion", "volatility_breakout", "breakout", "ema_crossover"]):
        ranking = StrategyRanking(
            optimizer_run_id=optimizer_run.id,
            strategy_name=strategy,
            symbol="BTC",
            timeframe="1m",
            profile="aggressive_1h",
            experimental=True,
            risk_label="Very High Risk",
            score=4.0 - index * 0.15,
            net_return_after_costs=0.055 - index * 0.004,
            recent_performance_score=0.03,
            recent_1h_return=0.025 - index * 0.001,
            max_drawdown=-0.035,
            profit_factor=1.7,
            sharpe_like=0.8,
            sortino_like=1.0,
            win_rate=0.6,
            trade_count=20,
            trades_per_day=24.0,
            edge_score=22.0 - index,
            expectancy=1.6,
            max_favorable_excursion=0.014,
            max_adverse_excursion=-0.004,
            capacity_usd=10_000.0,
            rejected=False,
        )
        ranking.parameters = {"risk_fraction": 0.02, "leverage": 1.0}
        db.session.add(ranking)
    db.session.commit()

    selection = app.extensions["services"]["vault_strategy_selector"].select("BTC", 1, "paper", 150.0)

    assert selection.metadata["allocation_mode"] == "enhanced_1h_ensemble"
    assert selection.metadata["ensemble_version"] == "enhanced_v2"
    assert selection.metadata["confluence_score"] >= 0.01
    assert len(selection.legs) >= 2
    assert len(selection.legs) <= 5
    assert sum(leg["allocation_cap_usd"] for leg in selection.legs) <= 150.0 + 1e-9
    assert all(leg["allocation_cap_usd"] <= 75.0 + 1e-9 for leg in selection.legs)
    assert all(leg["ensemble_version"] == "enhanced_v2" for leg in selection.legs)
    assert all("multi_timeframe_confluence" in leg for leg in selection.legs)
    assert set(selection.metadata["selected_strategies"]) == {leg["strategy_name"] for leg in selection.legs}


def test_enhanced_ensemble_skips_hard_rejected_candidate_despite_high_score(app) -> None:
    from app.services.ensemble_allocator import EnhancedEnsembleAllocator

    app.config["DYNAMIC_UNIVERSE_ENABLED"] = False
    app.config["ENSEMBLE_ENHANCED_ENABLED"] = True
    app.config["ENSEMBLE_MAX_LEGS"] = 3
    app.config["ENSEMBLE_MAX_SYMBOL_PCT"] = 1.0
    app.config["ENSEMBLE_MIN_EDGE_BPS"] = 5.0
    app.config["ENSEMBLE_MIN_SHARPE"] = 0.4
    app.config["FIB_CONFLUENCE_THRESHOLD"] = 0.01
    app.config["FIB_CONFLUENCE_CLUSTER_TOLERANCE_BPS"] = 300.0
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: _book(spread=0.1, size="1000")
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles(max(limit, 140))

    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    candidates = [
        ("scalping", 50.0, "slippage_cap"),
        ("rsi_mean_reversion", 12.0, ""),
        ("volatility_breakout", 11.0, ""),
    ]
    for strategy, score, no_trade_reason in candidates:
        ranking = StrategyRanking(
            optimizer_run_id=optimizer_run.id,
            strategy_name=strategy,
            symbol="BTC",
            timeframe="1m",
            profile="aggressive_1h",
            score=score,
            net_return_after_costs=0.05,
            recent_1h_return=0.02,
            max_drawdown=-0.03,
            profit_factor=1.6,
            sharpe_like=0.8,
            sortino_like=1.0,
            win_rate=0.6,
            trade_count=18,
            trades_per_day=24.0,
            edge_score=20.0,
            expectancy=1.5,
            capacity_usd=10_000.0,
            no_trade_reason=no_trade_reason,
            rejected=False,
        )
        ranking.parameters = {"risk_fraction": 0.02, "leverage": 1.0}
        db.session.add(ranking)
    db.session.commit()

    ranked, skipped = EnhancedEnsembleAllocator(app.config).rank(
        StrategyRanking.query.order_by(StrategyRanking.score.desc()).all(),
        duration_hours=1,
        metadata={"multi_timeframe_confluence": {"score": 0.7}},
    )

    assert {candidate.ranking.strategy_name for candidate in ranked} == {"rsi_mean_reversion", "volatility_breakout"}
    assert {item.get("skip_reason") for item in skipped if item.get("strategy_name") == "scalping"} == {"ranking_has_no_trade_reason"}


def test_enhanced_ensemble_fails_closed_when_confluence_missing(app) -> None:
    app.config["ENSEMBLE_ENHANCED_ENABLED"] = True
    app.config["FIB_CONFLUENCE_THRESHOLD"] = 0.5
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: _book(spread=0.1, size="1000")
    market_data.get_candles = lambda symbol, timeframe, mode, limit: []

    selection = app.extensions["services"]["vault_strategy_selector"].select("BTC", 1, "paper", 150.0)

    assert selection.metadata["allocation_mode"] == "single_best"
    assert selection.metadata["ensemble_version"] == "enhanced_v2"
    assert selection.metadata["skip_reason"] in {"multi_timeframe_data_unavailable", "multi_timeframe_confluence_failed"}
    assert len(selection.legs) == 1


def test_duration_experimental_ensemble_selects_24h_library_and_effective_weights(app) -> None:
    app.config["DYNAMIC_UNIVERSE_ENABLED"] = False
    app.config["EXPERIMENTAL_DURATION_ENSEMBLE_ENABLED"] = True
    app.config["EXPERIMENTAL_DURATION_ENSEMBLE_MAX_LEGS"] = 4
    app.config["FIB_CONFLUENCE_THRESHOLD"] = 0.0
    app.config["FIB_CONFLUENCE_CLUSTER_TOLERANCE_BPS"] = 300.0
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: _book(spread=0.1, size="1000")
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles(max(limit, 140))

    optimizer_run = OptimizerRun(profile="aggressive_risk_adjusted", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    for index, (strategy, symbol) in enumerate(
        [
            ("rsi_mean_reversion", "BTC"),
            ("ema_crossover", "ETH"),
            ("volatility_breakout", "SOL"),
            ("scalping", "BTC"),
        ]
    ):
        ranking = StrategyRanking(
            optimizer_run_id=optimizer_run.id,
            strategy_name=strategy,
            symbol=symbol,
            timeframe="15m",
            profile="aggressive_risk_adjusted",
            score=6.0 + index,
            net_return_after_costs=0.08 - index * 0.01,
            recent_performance_score=0.04,
            max_drawdown=-0.04,
            profit_factor=1.5,
            sharpe_like=0.8,
            sortino_like=1.0,
            win_rate=0.6,
            trade_count=18,
            trades_per_day=8.0,
            edge_score=18.0,
            expectancy=1.4,
            capacity_usd=10_000.0,
            lock_duration_hours=24,
            rejected=False,
        )
        ranking.parameters = {"risk_fraction": 0.02, "leverage": 1.0}
        db.session.add(ranking)
    db.session.commit()

    selection = app.extensions["services"]["vault_strategy_selector"].select("BTC", 24, "paper", 240.0)

    assert selection.metadata["allocation_mode"] == "duration_experimental_ensemble"
    assert selection.metadata["ensemble_version"] == "duration_experimental_v1"
    assert selection.metadata["duration_bucket"] == "24h"
    assert "scalping" not in {leg["strategy_name"] for leg in selection.legs}
    assert len(selection.legs) >= 2
    assert sum(leg["allocation_cap_usd"] for leg in selection.legs) <= 240.0 + 1e-9
    assert round(sum(leg["effective_allocation_weight"] for leg in selection.legs), 6) == 1.0
    assert all("target_ensemble_weight" in leg for leg in selection.legs)
    assert all("effective_allocation_weight" in leg for leg in selection.legs)


def test_duration_experimental_ensemble_records_cap_limited_allocation(app) -> None:
    app.config["EXPERIMENTAL_DURATION_ENSEMBLE_ENABLED"] = True
    app.config["EXPERIMENTAL_DURATION_ENSEMBLE_MAX_LEGS"] = 3
    app.config["ENSEMBLE_MAX_SYMBOL_PCT"] = 0.70
    app.config["ENSEMBLE_MAX_STRATEGY_PCT"] = 0.70
    app.config["FIB_CONFLUENCE_THRESHOLD"] = 0.0
    app.config["FIB_CONFLUENCE_CLUSTER_TOLERANCE_BPS"] = 300.0
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: _book(spread=0.1, size="1000")
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles(max(limit, 140))

    optimizer_run = OptimizerRun(profile="aggressive_risk_adjusted", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    for index, strategy in enumerate(["rsi_mean_reversion", "ema_crossover", "volatility_breakout"]):
        ranking = StrategyRanking(
            optimizer_run_id=optimizer_run.id,
            strategy_name=strategy,
            symbol="BTC",
            timeframe="15m",
            profile="aggressive_risk_adjusted",
            score=8.0 - index,
            net_return_after_costs=0.07 - index * 0.005,
            recent_performance_score=0.04,
            max_drawdown=-0.04,
            profit_factor=1.5,
            sharpe_like=0.8,
            sortino_like=1.0,
            win_rate=0.6,
            trade_count=18,
            trades_per_day=8.0,
            edge_score=18.0,
            expectancy=1.4,
            capacity_usd=10_000.0,
            lock_duration_hours=24,
            rejected=False,
        )
        ranking.parameters = {"risk_fraction": 0.02, "leverage": 1.0}
        db.session.add(ranking)
    db.session.commit()

    selection = app.extensions["services"]["vault_strategy_selector"].select("BTC", 24, "paper", 100.0)

    assert selection.metadata["allocation_mode"] == "duration_experimental_ensemble"
    assert selection.metadata["allocation_conservation"]["allocated_usd"] <= 100.0 + 1e-9
    assert selection.metadata["allocation_conservation"]["within_total_cap"] is True
    assert round(sum(leg["effective_allocation_weight"] for leg in selection.legs), 6) == 1.0
    assert selection.metadata["cap_blocked_count"] >= 1 or any(leg["cap_limited"] for leg in selection.legs)


def test_duration_experimental_ensemble_fallback_metadata_when_candidates_blocked(app) -> None:
    app.config["EXPERIMENTAL_DURATION_ENSEMBLE_ENABLED"] = True
    app.config["FIB_CONFLUENCE_THRESHOLD"] = 0.5
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: _book(spread=0.1, size="1000")
    market_data.get_candles = lambda symbol, timeframe, mode, limit: []

    selection = app.extensions["services"]["vault_strategy_selector"].select("BTC", 24, "paper", 100.0)

    assert selection.metadata["allocation_mode"] == "single_best"
    assert selection.metadata["ensemble_version"] == "duration_experimental_v1"
    assert selection.metadata["skip_reason"]
    assert selection.metadata["rejected_candidate_count"] == 0
    assert selection.metadata["cap_blocked_count"] == 0


def test_max_return_vault_metadata_includes_market_structure(app) -> None:
    app.config["MAX_RETURN_OPTIMIZER_ENABLED"] = True
    app.config["MARKET_STRUCTURE_FEATURES_ENABLED"] = True
    app.config["MARKET_STRUCTURE_DEPTH_USD_SCALE"] = 50_000.0
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: _book(spread=0.1, size="1000")
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles(max(limit, 140))

    selection = app.extensions["services"]["vault_strategy_selector"].select("BTC", 24, "paper", 100.0)

    assert selection.metadata["profit_objective_version"] == "max_return_v3"
    assert selection.metadata["market_structure"]["enabled"] is True
    assert selection.metadata["market_structure_score"] > 0.0
    assert selection.legs[0]["profit_objective_version"] == "max_return_v3"
    assert selection.legs[0]["market_structure_score"] == selection.metadata["market_structure_score"]


def test_completed_enhanced_cycle_updates_bandit_from_cycle_and_legs(app) -> None:
    from app.routes.consumer import _learn_from_completed_cycle

    app.config["ML_RANKER_ENABLED"] = True
    app.config["ENSEMBLE_LEARNING_ENABLED"] = True
    app.config["ENSEMBLE_LEARNING_DECAY"] = 0.8
    primary_run = StrategyRun(strategy_name="scalping", symbol="BTC", timeframe="1m", mode="paper", status="running")
    secondary_run = StrategyRun(strategy_name="rsi_mean_reversion", symbol="ETH", timeframe="1m", mode="paper", status="running")
    db.session.add_all([primary_run, secondary_run])
    db.session.flush()
    cycle = VaultCycle(
        strategy_run_id=primary_run.id,
        deposit_asset="USDC",
        deposit_amount=100.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        status="complete",
        execution_mode="paper",
        algorithm_profile="Aggressive",
        selected_strategy_name="enhanced_ensemble",
        selected_timeframe="1m",
        starting_value_usd=100.0,
        current_estimated_value_usd=106.0,
        unlocks_at=datetime.utcnow(),
    )
    cycle.selection_metadata = {
        "allocation_mode": "enhanced_1h_ensemble",
        "ensemble_id": "ensemble-v2-test",
        "ensemble_version": "enhanced_v2",
        "optimizer_profile": "aggressive_1h",
        "symbol": "BTC",
        "total_pnl_usd": 6.0,
        "confluence_score": 0.7,
        "multi_timeframe_confluence": {
            "score": 0.7,
            "cluster_count": 3,
            "volume_confirmation": True,
            "rsi_confirmation": True,
            "trend_regime": "bullish",
        },
    }
    db.session.add(cycle)
    db.session.flush()
    leg_one = VaultAllocationLeg(
        vault_cycle_id=cycle.id,
        strategy_run_id=primary_run.id,
        symbol="BTC",
        timeframe="1m",
        allocation_cap_usd=60.0,
        realized_pnl_usd=4.0,
    )
    leg_one.details = {"ensemble_weight": 0.6, "edge_score": 22.0, "optimizer_profile": "aggressive_1h"}
    leg_two = VaultAllocationLeg(
        vault_cycle_id=cycle.id,
        strategy_run_id=secondary_run.id,
        symbol="ETH",
        timeframe="1m",
        allocation_cap_usd=40.0,
        realized_pnl_usd=2.0,
    )
    leg_two.details = {"ensemble_weight": 0.4, "edge_score": 18.0, "optimizer_profile": "aggressive_1h"}
    db.session.add_all([leg_one, leg_two])
    db.session.commit()

    _learn_from_completed_cycle(cycle)

    sources = [event.source for event in MLTrainingEvent.query.order_by(MLTrainingEvent.id.asc()).all()]
    assert "vault_cycle" in sources
    assert "vault_cycle_ensemble" in sources
    assert sources.count("vault_leg") == 2


def test_live_aggressive_leverage_cap_blocks_over_cap_orders(app) -> None:
    app.config["ALLOW_AGGRESSIVE_LIVE_TRADING"] = True
    app.config["AGGRESSIVE_MAX_LIVE_LEVERAGE"] = 2.0
    Setting.set_json("explicit_live_confirmed", True)
    Setting.set_json("secondary_confirmation", True)
    db.session.add(
        StrategyValidation(
            strategy_name="scalping",
            symbol="BTC",
            timeframe="1m",
            stage="shadow_live",
            status="passed",
            completed_at=datetime.utcnow(),
        )
    )
    db.session.commit()
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = OrderIntent(
        symbol="BTC",
        side="buy",
        quantity=0.1,
        mode="live",
        stop_loss=95.0,
        leverage=3.0,
        strategy_name="scalping",
        timeframe="1m",
        metadata={"optimizer_profile": "aggressive_1h"},
    )

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "aggressive_live_leverage_cap"


def test_admin_backtests_no_longer_renders_opportunity_lab(app) -> None:
    app.config["DYNAMIC_UNIVERSE_ENABLED"] = True
    admin, secret = _create_user(username="opportunity-admin", role="admin")

    def fail_liquid_universe(*args, **kwargs):
        raise AssertionError("Backtests index should not load opportunity diagnostics.")

    app.extensions["services"]["market_universe"].liquid_universe = fail_liquid_universe

    client = app.test_client()
    _login(client, admin.username, secret)
    response = client.get("/admin/backtests")

    assert response.status_code == 200
    assert b"Paper Vault" in response.data
    assert b"Opportunity Lab" not in response.data
    assert b"dynamic_liquid" not in response.data
