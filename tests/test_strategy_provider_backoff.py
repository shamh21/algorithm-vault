from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest

from app.auth import password_hash
from app.extensions import db
from app.models import Setting, StrategyRun, TradingConnection, User, VaultAllocationLeg, VaultCycle
from app.routes.consumer import _cycle_summary, _refresh_cycle_performance
from app.services.hyperliquid_client import ClientSnapshot
from app.services.market_data import MarketDataService


def _run_with_connection(provider: str = "hyperliquid", *, one_h10: bool = False) -> StrategyRun:
    user = User(username=f"{provider}-backoff", password_hash=password_hash("password123"))
    db.session.add(user)
    db.session.flush()
    connection = TradingConnection(
        user_id=user.id,
        provider=provider,
        connection_type="cex_api_key",
        verification_status="verified",
        is_active=True,
        last_verified_at=datetime.utcnow(),
    )
    db.session.add(connection)
    db.session.flush()
    run = StrategyRun(
        user_id=user.id,
        trading_connection_id=connection.id,
        strategy_name="scalping",
        symbol="BTC",
        timeframe="1m",
        mode="live",
        status="running",
        manual_enabled=True,
    )
    run.parameters = {
        "provider": provider,
        "one_h10_vault": one_h10,
        "ml_horizon": "1h10" if one_h10 else "1h",
    }
    db.session.add(run)
    db.session.commit()
    return run


def test_provider_runtime_backoff_records_rate_limit_for_all_live_runs(app) -> None:
    manager = app.extensions["services"]["strategy_manager"]
    run = _run_with_connection("hyperliquid")

    handled = manager._handle_provider_runtime_failure(
        run,
        RuntimeError("Hyperliquid call failed: live all_mids attempt=3/3 error=(429, None, 'null')"),
    )
    db.session.commit()

    payload = Setting.get_json(f"strategy_provider_runtime_backoff:hyperliquid:{run.trading_connection_id}", {})
    assert handled is True
    assert payload["blocker_category"] == "rate_limited"
    assert payload["provider"] == "hyperliquid"
    assert manager._runtime_provider_backoff_remaining(run) > 0
    assert run.status == "running"
    assert run.manual_enabled is True
    assert run.last_signal["metadata"]["no_trade_reason"] == "provider_runtime_backoff"


def test_one_h10_runtime_backoff_updates_legacy_market_data_key(app) -> None:
    manager = app.extensions["services"]["strategy_manager"]
    run = _run_with_connection("kucoin", one_h10=True)

    handled = manager._handle_provider_runtime_failure(
        run,
        RuntimeError("KuCoin request failed after 2 attempt(s): Read timed out. candles_snapshot BTC 1m"),
    )
    db.session.commit()

    generic = Setting.get_json(f"strategy_provider_runtime_backoff:kucoin:{run.trading_connection_id}", {})
    one_h10 = Setting.get_json(f"one_h10_market_data_backoff:kucoin:{run.trading_connection_id}", {})
    assert handled is True
    assert generic["blocker_category"] == "network_timeout"
    assert one_h10["blocker_category"] == "network_timeout"
    assert run.last_signal["metadata"]["one_h10_vault"] is True


def test_provider_runtime_backoff_handles_timestamp_skew_without_api_storm(app) -> None:
    manager = app.extensions["services"]["strategy_manager"]
    run = _run_with_connection("kucoin")

    handled = manager._handle_provider_runtime_failure(
        run,
        RuntimeError('{"code":"400002","msg":"Invalid KC-API-TIMESTAMP"}'),
    )

    payload = Setting.get_json(f"strategy_provider_runtime_backoff:kucoin:{run.trading_connection_id}", {})
    assert handled is True
    assert payload["blocker_category"] == "timestamp_skew"
    assert run.last_signal["metadata"]["no_trade_reason"] == "provider_runtime_backoff"


def test_live_strategy_candle_fetch_uses_single_attempt_before_provider_backoff(app, monkeypatch) -> None:
    manager = app.extensions["services"]["strategy_manager"]
    run = _run_with_connection("hyperliquid")
    captured: dict[str, object] = {}

    def fake_candles(symbol, timeframe, mode="live", limit=None, retry=True):
        captured.update({"symbol": symbol, "timeframe": timeframe, "mode": mode, "limit": limit, "retry": retry})
        return [
            {"timestamp": index, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}
            for index in range(3)
        ]

    monkeypatch.setattr(manager.market_data, "get_candles", fake_candles)

    candles = manager._run_candles(run, "live", limit=200)

    assert len(candles) == 3
    assert captured["retry"] is False


def test_market_data_serves_stale_live_candles_during_rate_limit() -> None:
    calls = {"candles": 0}

    class FakeClient:
        def get_candles(self, mode, symbol, timeframe, start_ms, end_ms, *, retry=True):
            calls["candles"] += 1
            if calls["candles"] == 1:
                return [
                    {"t": 1, "o": "100", "h": "101", "l": "99", "c": "100", "v": "10"},
                    {"t": 2, "o": "100", "h": "102", "l": "99", "c": "101", "v": "11"},
                ]
            raise RuntimeError("Hyperliquid call failed: live candles_snapshot BTC 1m error=(429, None, 'null')")

    market_data = MarketDataService(
        {
            "MARKET_DATA_LIVE_CANDLE_CACHE_SECONDS": 1.0,
            "MARKET_DATA_LIVE_STALE_SECONDS": 300.0,
            "DASHBOARD_CANDLE_LIMIT": 200,
        },
        FakeClient(),
    )

    first = market_data.get_candles("BTC", "1m", mode="live", limit=2)
    market_data._cache[("candles", "live", "BTC", "1m", 2)] = (time.time() - 2.0, first)
    second = market_data.get_candles("BTC", "1m", mode="live", limit=2, retry=False)

    assert second == first
    assert calls["candles"] == 2
    assert market_data.cache_stats()["stale_serves"] == 1


def test_live_mids_failure_backoff_prevents_repeated_provider_calls() -> None:
    calls: list[bool] = []

    class FakeClient:
        def get_all_mids(self, mode, *, retry=True):
            calls.append(bool(retry))
            raise RuntimeError("Hyperliquid call failed: live all_mids error=(429, None, 'null')")

    market_data = MarketDataService(
        {
            "MARKET_DATA_LIVE_MIDS_CACHE_SECONDS": 1.0,
            "MARKET_DATA_LIVE_FAILURE_BACKOFF_SECONDS": 30.0,
            "DASHBOARD_CANDLE_LIMIT": 200,
        },
        FakeClient(),
    )

    with pytest.raises(RuntimeError):
        market_data.get_all_mids("live")
    with pytest.raises(RuntimeError):
        market_data.get_all_mids("live")

    assert calls == [False]
    assert market_data.cache_stats()["failure_backoffs"] == 1


def test_trading_connection_snapshot_cache_prevents_repeated_live_position_calls(app, monkeypatch) -> None:
    run = _run_with_connection("hyperliquid")
    service = app.extensions["services"]["trading_connections"]
    calls = {"snapshots": 0}

    class FakeConnector:
        def account_snapshot(self, mode: str) -> ClientSnapshot:
            calls["snapshots"] += 1
            return ClientSnapshot(
                mode,
                [{"asset": "USDC", "type": "margin", "value": 25.0, "withdrawable": 25.0}],
                [{"symbol": "BTC", "quantity": 0.1, "unrealized_pnl": 1.25}],
                [],
                [],
                [],
            )

    monkeypatch.setattr(service, "_connector_for_connection", lambda connection: FakeConnector())

    first = service.account_snapshot(run.user_id, "live", run.trading_connection_id)
    second = service.account_snapshot(run.user_id, "live", run.trading_connection_id)

    assert calls["snapshots"] == 1
    assert first.positions == second.positions == [{"symbol": "BTC", "quantity": 0.1, "unrealized_pnl": 1.25}]


def test_current_position_does_not_fetch_live_mid_for_flat_cached_position(app, monkeypatch) -> None:
    run = _run_with_connection("hyperliquid")
    manager = app.extensions["services"]["order_manager"]
    calls = {"snapshots": 0, "mids": 0}

    def snapshot(user_id: int, mode: str, connection_id: int | None = None) -> ClientSnapshot:
        calls["snapshots"] += 1
        return ClientSnapshot(mode, [], [], [], [], [])

    def fail_mid(*args, **kwargs):
        calls["mids"] += 1
        raise AssertionError("flat live position should not fetch all_mids")

    monkeypatch.setattr(manager.trading_connections, "account_snapshot", snapshot)
    monkeypatch.setattr(manager.market_data, "get_mid_price", fail_mid)

    position = manager.current_position("BTC", "live", run.user_id, run.trading_connection_id)

    assert position["symbol"] == "BTC"
    assert position["quantity"] == 0.0
    assert position["mark_price"] == 0.0
    assert calls == {"snapshots": 1, "mids": 0}


def test_cycle_summary_uses_one_cached_account_snapshot_per_provider(app, monkeypatch) -> None:
    run = _run_with_connection("hyperliquid", one_h10=True)
    cycle = VaultCycle(
        user_id=run.user_id,
        trading_connection_id=run.trading_connection_id,
        strategy_run_id=run.id,
        deposit_asset="USDC",
        deposit_amount=10.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        status="active",
        execution_substatus="executing",
        execution_mode="live",
        algorithm_profile="1H10",
        selected_strategy_name="scalping",
        selected_timeframe="1m",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=10.0,
        current_estimated_value_usd=10.0,
    )
    db.session.add(cycle)
    db.session.flush()
    run.parameters = {
        **dict(run.parameters or {}),
        "vault_cycle_id": cycle.id,
        "one_h10_vault": True,
        "provider": "hyperliquid",
        "app_symbol": "BTC",
        "venue_symbol": "BTC",
    }
    db.session.add_all(
        [
            VaultAllocationLeg(
                vault_cycle_id=cycle.id,
                strategy_run_id=run.id,
                symbol="BTC",
                timeframe="1m",
                provider="hyperliquid",
                trading_connection_id=run.trading_connection_id,
                allocation_cap_usd=5.0,
                leverage=1.0,
            ),
            VaultAllocationLeg(
                vault_cycle_id=cycle.id,
                strategy_run_id=run.id,
                symbol="ETH",
                timeframe="1m",
                provider="hyperliquid",
                trading_connection_id=run.trading_connection_id,
                allocation_cap_usd=5.0,
                leverage=1.0,
            ),
        ]
    )
    db.session.flush()
    for leg in cycle.allocation_legs:
        leg.details = {
            "provider": "hyperliquid",
            "app_symbol": leg.symbol,
            "venue_symbol": leg.symbol,
            "provider_symbol": leg.symbol,
            "trading_connection_id": run.trading_connection_id,
        }
    db.session.commit()
    service = app.extensions["services"]["trading_connections"]
    calls = {"snapshots": 0}

    def snapshot(user_id: int, mode: str, connection_id: int | None = None) -> ClientSnapshot:
        calls["snapshots"] += 1
        return ClientSnapshot(
            mode,
            [],
            [
                {"symbol": "BTC", "quantity": 0.01, "unrealized_pnl": 1.25},
                {"symbol": "ETH", "quantity": 0.1, "unrealized_pnl": 2.0},
            ],
            [],
            [],
            [],
        )

    monkeypatch.setattr(service, "account_snapshot", snapshot)

    summary = _cycle_summary(cycle)

    assert calls["snapshots"] == 1
    assert summary["unrealized_pnl_usd"] == pytest.approx(3.25)
    assert {leg["symbol"]: leg["unrealized_pnl_usd"] for leg in summary["legs"]} == {"BTC": 1.25, "ETH": 2.0}


def test_backoff_active_cycle_summary_skips_live_position_and_market_reads(app, monkeypatch) -> None:
    run = _run_with_connection("hyperliquid", one_h10=True)
    retry_after = (datetime.utcnow() + timedelta(minutes=5)).isoformat()
    cycle = VaultCycle(
        user_id=run.user_id,
        trading_connection_id=run.trading_connection_id,
        strategy_run_id=run.id,
        deposit_asset="USDC",
        deposit_amount=10.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        status="active",
        execution_substatus="executing",
        execution_mode="live",
        algorithm_profile="1H10",
        selected_strategy_name="scalping",
        selected_timeframe="1m",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=10.0,
        current_estimated_value_usd=10.4,
    )
    cycle.selection_metadata = {
        "unrealized_pnl_usd": 0.4,
        "total_pnl_usd": 0.4,
        "one_h10_runtime_notice": {
            "kind": "market_data_backoff",
            "message": "Provider rate limited market data or account data; retrying after backoff.",
            "retry_after": retry_after,
        },
    }
    db.session.add(cycle)
    db.session.flush()
    leg = VaultAllocationLeg(
        vault_cycle_id=cycle.id,
        strategy_run_id=run.id,
        symbol="BTC",
        timeframe="1m",
        provider="hyperliquid",
        trading_connection_id=run.trading_connection_id,
        allocation_cap_usd=10.0,
        leverage=1.0,
        unrealized_pnl_usd=0.4,
    )
    leg.details = {
        "provider": "hyperliquid",
        "app_symbol": "BTC",
        "venue_symbol": "BTC",
        "provider_symbol": "BTC",
        "trading_connection_id": run.trading_connection_id,
    }
    db.session.add(leg)
    db.session.commit()

    def fail_snapshot(*args, **kwargs):
        raise AssertionError("backoff detail should not fetch live positions")

    def fail_market(*args, **kwargs):
        raise AssertionError("backoff detail should not fetch market data")

    service = app.extensions["services"]["trading_connections"]
    market_data = app.extensions["services"]["market_data"]
    monkeypatch.setattr(service, "account_snapshot", fail_snapshot)
    monkeypatch.setattr(market_data, "get_mid_price", fail_market)
    monkeypatch.setattr(market_data, "get_order_book", fail_market)

    performance = _refresh_cycle_performance(cycle)
    summary = _cycle_summary(cycle, performance=performance)

    assert performance["unrealized_pnl"] == pytest.approx(0.4)
    assert cycle.current_estimated_value_usd == pytest.approx(10.4)
    assert summary["runtime_notice"]["retry_after"] == retry_after
