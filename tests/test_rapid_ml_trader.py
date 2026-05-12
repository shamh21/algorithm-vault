from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from app.extensions import db
from app.models import LeveragedMarket, Order, RapidMLDecision, RapidMLSession, Setting, TradingConnection, User
from app.services.hyperliquid_client import ClientSnapshot


def _user_with_connections() -> tuple[User, TradingConnection, TradingConnection]:
    user = User(username="rapid-user", password_hash="x")
    db.session.add(user)
    db.session.flush()
    hyperliquid = TradingConnection(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="api_wallet",
        verification_status="verified",
        is_active=True,
        wallet_address="0x0000000000000000000000000000000000000001",
    )
    kucoin = TradingConnection(
        user_id=user.id,
        provider="kucoin",
        connection_type="futures",
        verification_status="verified",
        is_active=False,
    )
    db.session.add_all([hyperliquid, kucoin])
    db.session.commit()
    return user, hyperliquid, kucoin


def _install_ready_rapid_fakes(app, monkeypatch, *, same_symbol: bool = False) -> None:
    services = app.extensions["services"]
    trader = services["rapid_ml_trader"]
    connections = services["trading_connections"]
    app.config["RAPID_ML_SYMBOLS"] = "BTC,ETH"
    app.config["KUCOIN_SYMBOL_MAP_JSON"] = {"BTC": "XBTUSDTM", "ETH": "ETHUSDTM"}
    app.config["KUCOIN_CONTRACT_SPECS_JSON"] = {
        "XBTUSDTM": {"contract_size": 0.001, "size_step": 1, "min_size": 1},
        "ETHUSDTM": {"contract_size": 0.01, "size_step": 1, "min_size": 1},
    }

    def fake_snapshot(user_id: int, mode: str, connection_id: int | None = None) -> ClientSnapshot:
        connection = db.session.get(TradingConnection, int(connection_id or 0))
        asset = "USDT" if connection and connection.provider == "kucoin" else "USDC"
        return ClientSnapshot(
            mode,
            [{"asset": asset, "type": "margin", "value": 100.0, "withdrawable": 100.0, "updated_at": datetime.utcnow()}],
            [],
            [],
            [],
            [],
        )

    monkeypatch.setattr(connections, "can_trade", lambda user_id, mode, connection_id=None: True)
    monkeypatch.setattr(connections, "account_snapshot", fake_snapshot)
    monkeypatch.setattr(trader, "_reference_price", lambda provider, symbol, connection_id, venue_symbol=None: 100.0 if symbol == "BTC" else 50.0)
    monkeypatch.setattr(trader, "_spread_bps", lambda provider, symbol, connection_id, venue_symbol=None: 1.0)
    monkeypatch.setattr(trader, "_live_feature_context", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        trader,
        "_ml_readiness",
        lambda provider: {
            "ready": True,
            "provider": provider,
            "blockers": [],
            "families": {},
            "offline_ranker": {"ready": True, "blockers": []},
        },
    )

    def high_opportunity(context: dict[str, Any]) -> bool:
        provider = context.get("provider")
        symbol = context.get("symbol")
        if same_symbol:
            return symbol == "BTC"
        return (provider == "hyperliquid" and symbol == "BTC") or (provider == "kucoin" and symbol == "ETH")

    def fake_decision(family: str, context: dict[str, Any], *, horizon: str = "1h", candles: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        high = high_opportunity(context)
        if family == "pytorch_gru_signal":
            return {
                "family": family,
                "action": "buy" if high else "hold",
                "confidence": 0.9 if high else 0.1,
                "expected_return": 0.004 if high else 0.0,
                "blockers": [],
                "ready": True,
            }
        if family == "pytorch_risk_policy":
            return {"family": family, "action": "approve" if high else "reject", "confidence": 0.9, "expected_return": 0.002, "blockers": []}
        if family == "pytorch_ops_anomaly":
            return {"family": family, "action": "observe", "confidence": 0.1, "expected_return": 0.0, "blockers": [], "raw": {"ops_anomaly_score": 0.0}}
        if family == "pytorch_allocator":
            return {"family": family, "action": "allocate", "confidence": 0.8, "expected_return": 0.002, "blockers": [], "raw": {"sizing_score": 0.8}}
        if family == "pytorch_roi_target":
            return {"family": family, "action": "target_met_candidate", "confidence": 0.8, "expected_return": 0.004 if high else 0.0, "blockers": []}
        return {"family": family, "action": "route", "confidence": 0.8, "expected_return": 0.002, "blockers": []}

    def fake_offline(context: dict[str, Any], horizon: str, *, base_score: float | None = None, rejected: bool = False) -> dict[str, Any]:
        return {
            "status": "promoted",
            "prediction": 0.002 if high_opportunity(context) else 0.0,
            "blockers": [],
            "model_id": 1,
        }

    monkeypatch.setattr(services["ml_decision_engine"], "decision", fake_decision)
    monkeypatch.setattr(services["offline_ranker"], "score_payload", fake_offline)


def _payload(result) -> dict[str, Any]:
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def _rapid_source_order(
    *,
    user: User,
    connection: TradingConnection,
    symbol: str = "BTC",
    side: str = "sell",
    stop_loss: float = 105.0,
    take_profit: float = 99.0,
) -> Order:
    order = Order(
        user_id=user.id,
        trading_connection_id=connection.id,
        client_order_id=f"rapid-source-{symbol}-{connection.id}",
        mode="live",
        symbol=symbol,
        side=side,
        order_type="market",
        status="filled",
        quantity=1.0,
        filled_quantity=1.0,
        average_fill_price=100.0,
        reduce_only=False,
        leverage=1.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_status="approved",
    )
    order.details = {
        "rapid_ml": True,
        "provider": connection.provider,
        "venue_symbol": symbol,
        "rapid_ml_session_id": 1,
        "reference_price": 100.0,
    }
    db.session.add(order)
    db.session.commit()
    return order


def _snapshot_with_position(provider: str, position: dict[str, Any]) -> ClientSnapshot:
    asset = "USDT" if provider == "kucoin" else "USDC"
    return ClientSnapshot(
        "live",
        [{"asset": asset, "type": "margin", "value": 100.0, "withdrawable": 100.0, "updated_at": datetime.utcnow()}],
        [position],
        [],
        [],
        [],
    )


def test_hyperliquid_sizing_blocks_below_exchange_min_notional(app) -> None:
    service = app.extensions["services"]["rapid_ml_trader"]
    app.config["HYPERLIQUID_MIN_ORDER_VALUE_USD"] = 10.0
    app.config["RAPID_ML_MIN_NOTIONAL_BUFFER_USD"] = 0.50

    quantity, sizing = service._quantity_for("hyperliquid", "HYPE", 1.0, 42.58, None)

    assert quantity == 0.0
    assert "hyperliquid_min_order_value_exceeds_allocation" in sizing["blockers"]
    assert sizing["min_required_allocation_usd"] == 10.5


def test_live_rapid_ml_preview_allocates_both_providers_without_order(app, monkeypatch) -> None:
    user, _, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    monkeypatch.setattr(
        app.extensions["services"]["order_manager"],
        "place_order",
        lambda intent: (_ for _ in ()).throw(AssertionError("preview must not submit")),
    )

    result = app.test_cli_runner().invoke(
        args=[
            "live-rapid-ml-trader",
            "--user-id",
            str(user.id),
            "--capital-usd",
            "100",
            "--provider",
            "both",
            "--duration-minutes",
            "0",
        ]
    )

    payload = _payload(result)
    assert payload["submitted_count"] == 0
    assert payload["preview_ready_count"] == 2
    assert payload["capital"]["daily_loss_cap_usd"] == 5.0
    assert payload["capital"]["per_position_cap_usd"] == 100.0
    assert payload["capital"]["ml_sizing_enabled"] is True
    assert payload["capital"]["rapid_ml_fixed_hard_cap_usd"] == 0.0
    assert payload["safety"]["profitability_gate"]["min_edge_bps"] == 5.0
    profitability = payload["cycles"][0]["executions"][0]["order_intent"]["metadata"]["profitability"]
    assert profitability["edge_bps_after_costs"] > profitability["min_edge_bps"]
    assert profitability["positive_edge_source_count"] >= profitability["required_positive_edge_sources"]
    assert Order.query.count() == 0
    assert RapidMLSession.query.count() == 1
    assert RapidMLDecision.query.count() == 2


def test_live_rapid_ml_compact_output_omits_deep_cycles(app, monkeypatch) -> None:
    user, _, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)

    result = app.test_cli_runner().invoke(
        args=[
            "live-rapid-ml-trader",
            "--user-id",
            str(user.id),
            "--capital-usd",
            "100",
            "--provider",
            "both",
            "--duration-minutes",
            "0",
            "--compact",
        ]
    )

    payload = _payload(result)
    assert "cycles" not in payload
    assert len(payload["providers"]) == 2
    assert payload["providers"][0]["candidate_summaries"]
    summary = payload["providers"][0]["candidate_summaries"][0]
    assert summary["signal_action"] == "buy"
    assert summary["risk_policy_action"] == "approve"
    assert summary["signal_expected_return"] == 0.004
    assert summary["risk_policy_expected_return"] == 0.002
    assert payload["profitability_gate"]["min_edge_bps"] == 5.0
    assert payload["submitted_count"] == 0
    assert Order.query.count() == 0


def test_live_rapid_ml_preview_duration_runs_multiple_read_only_cycles(app, monkeypatch) -> None:
    user, _, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    connections = app.extensions["services"]["trading_connections"]
    snapshot_calls = 0

    def counted_snapshot(user_id: int, mode: str, connection_id: int | None = None) -> ClientSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return ClientSnapshot(
            mode,
            [{"asset": "USDC", "type": "margin", "value": 100.0, "withdrawable": 100.0, "updated_at": datetime.utcnow()}],
            [],
            [],
            [],
            [],
        )

    monkeypatch.setattr(connections, "account_snapshot", counted_snapshot)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    result = app.test_cli_runner().invoke(
        args=[
            "live-rapid-ml-trader",
            "--user-id",
            str(user.id),
            "--capital-usd",
            "100",
            "--provider",
            "hyperliquid",
            "--duration-minutes",
            "0.01",
            "--decision-interval-ms",
            "250",
            "--compact",
        ]
    )

    payload = _payload(result)
    assert payload["cycle_count"] == 2
    assert payload["latest_cycle"] == 2
    assert payload["preview_ready_count"] == 2
    assert payload["submitted_count"] == 0
    assert Order.query.count() == 0
    assert RapidMLDecision.query.count() == 2
    assert snapshot_calls == 1


def test_live_rapid_ml_profitability_gate_blocks_after_cost_edge(app, monkeypatch) -> None:
    user, _, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    app.config["RAPID_ML_MIN_EDGE_BPS"] = 100.0

    result = app.test_cli_runner().invoke(
        args=[
            "live-rapid-ml-trader",
            "--user-id",
            str(user.id),
            "--capital-usd",
            "100",
            "--provider",
            "hyperliquid",
        ]
    )

    payload = _payload(result)
    candidates = payload["cycles"][0]["analyses"][0]["candidates"]
    blockers = [blocker for candidate in candidates for blocker in candidate.get("blockers", [])]
    assert "ml_edge_below_cost_threshold" in blockers


def test_live_rapid_ml_uses_provider_specific_symbol_universe(app, monkeypatch) -> None:
    user, _, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    app.config["RAPID_ML_SYMBOLS"] = "BTC,ETH,SOL"
    app.config["RAPID_ML_SYMBOLS_HYPERLIQUID"] = "LINK,WIF,PENDLE,AAVE"
    app.config["RAPID_ML_SYMBOLS_KUCOIN"] = "BTC"
    app.config["RAPID_ML_MAX_SYMBOLS_PER_PROVIDER"] = 3
    app.config["ALLOWED_SYMBOLS"] = ["LINK", "WIF", "PENDLE", "BTC"]

    result = app.test_cli_runner().invoke(
        args=[
            "live-rapid-ml-trader",
            "--user-id",
            str(user.id),
            "--capital-usd",
            "100",
            "--provider",
            "both",
            "--duration-minutes",
            "0",
        ]
    )

    payload = _payload(result)
    analyses = payload["cycles"][0]["analyses"]
    symbols_by_provider = {
        item["provider"]: [candidate["symbol"] for candidate in item["candidates"]]
        for item in analyses
    }
    assert symbols_by_provider["hyperliquid"] == ["LINK", "WIF", "PENDLE"]
    assert symbols_by_provider["kucoin"] == ["BTC"]
    assert payload["preview_ready_count"] == 0
    assert payload["submitted_count"] == 0
    assert Order.query.count() == 0


def test_live_rapid_ml_auto_universe_uses_active_futures_without_allowed_symbols(app, monkeypatch) -> None:
    user, hyperliquid, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    app.config["RAPID_ML_SYMBOLS"] = ""
    app.config["RAPID_ML_SYMBOLS_HYPERLIQUID"] = ""
    app.config["RAPID_ML_UNIVERSE_REFRESH_SECONDS"] = 0
    app.config["ALLOWED_SYMBOLS"] = ["BTC"]
    db.session.add_all(
        [
            LeveragedMarket(
                provider="hyperliquid",
                trading_connection_id=hyperliquid.id,
                venue_symbol="WIF",
                symbol="WIF",
                status="active",
                settlement_asset="USDC",
                max_leverage=10,
                liquidity_usd=2_000_000,
            ),
            LeveragedMarket(
                provider="hyperliquid",
                trading_connection_id=hyperliquid.id,
                venue_symbol="BTC",
                symbol="BTC",
                status="active",
                settlement_asset="USDC",
                max_leverage=50,
                liquidity_usd=1_000_000,
            ),
        ]
    )
    db.session.commit()

    result = app.test_cli_runner().invoke(
        args=[
            "live-rapid-ml-trader",
            "--user-id",
            str(user.id),
            "--capital-usd",
            "100",
            "--provider",
            "hyperliquid",
            "--duration-minutes",
            "0",
        ]
    )

    payload = _payload(result)
    candidates = payload["cycles"][0]["analyses"][0]["candidates"]
    assert [candidate["symbol"] for candidate in candidates[:2]] == ["WIF", "BTC"]
    assert not any("symbol_not_allowed:WIF" in blocker for candidate in candidates for blocker in candidate.get("blockers", []))
    selected = payload["cycles"][0]["analyses"][0]["selected"]
    assert selected["symbol"] == "BTC"
    assert selected["rapid_ml_all_futures_universe"] is True


def test_live_rapid_ml_blocks_unapproved_symbols_before_order_creation(app, monkeypatch) -> None:
    user, _, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    app.config["RAPID_ML_AUTO_FUTURES_UNIVERSE_ENABLED"] = False
    app.config["ALLOWED_SYMBOLS"] = ["BTC"]
    app.config["RAPID_ML_SYMBOLS_HYPERLIQUID"] = "WIF"
    app.config["RAPID_ML_LIVE_ENABLED"] = True
    app.config["RAPID_ML_PREVIEW_ONLY"] = False
    app.config["CANARY_PREVIEW_ONLY"] = False

    result = app.test_cli_runner().invoke(
        args=[
            "live-rapid-ml-trader",
            "--user-id",
            str(user.id),
            "--capital-usd",
            "100",
            "--provider",
            "hyperliquid",
            "--submit",
            "--confirm",
            "RAPID-ML-LIVE",
        ],
    )

    payload = _payload(result)
    candidates = payload["cycles"][0]["analyses"][0]["candidates"]
    assert candidates[0]["blockers"] == ["symbol_not_allowed:WIF"]
    assert payload["submitted_count"] == 0
    assert Order.query.count() == 0


def test_live_rapid_ml_profitability_gate_requires_model_agreement(app, monkeypatch) -> None:
    user, _, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    services = app.extensions["services"]

    def one_source_decision(
        family: str,
        context: dict[str, Any],
        *,
        horizon: str = "1h",
        candles: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if family == "pytorch_gru_signal":
            return {
                "family": family,
                "action": "buy",
                "confidence": 0.9,
                "expected_return": 0.004,
                "blockers": [],
                "ready": True,
            }
        if family == "pytorch_risk_policy":
            return {"family": family, "action": "approve", "confidence": 0.9, "expected_return": 0.0, "blockers": []}
        if family == "pytorch_ops_anomaly":
            return {"family": family, "action": "observe", "confidence": 0.1, "expected_return": 0.0, "blockers": [], "raw": {"ops_anomaly_score": 0.0}}
        if family == "pytorch_allocator":
            return {"family": family, "action": "allocate", "confidence": 0.8, "expected_return": 0.0, "blockers": [], "raw": {"sizing_score": 0.8}}
        if family == "pytorch_roi_target":
            return {"family": family, "action": "target_met_candidate", "confidence": 0.8, "expected_return": 0.0, "blockers": []}
        return {"family": family, "action": "route", "confidence": 0.8, "expected_return": 0.0, "blockers": []}

    monkeypatch.setattr(services["ml_decision_engine"], "decision", one_source_decision)
    monkeypatch.setattr(
        services["offline_ranker"],
        "score_payload",
        lambda context, horizon, base_score=None, rejected=False: {
            "status": "promoted",
            "prediction": 0.0,
            "blockers": [],
            "model_id": 1,
        },
    )

    result = app.test_cli_runner().invoke(
        args=[
            "live-rapid-ml-trader",
            "--user-id",
            str(user.id),
            "--capital-usd",
            "100",
            "--provider",
            "hyperliquid",
        ]
    )

    payload = _payload(result)
    candidates = payload["cycles"][0]["analyses"][0]["candidates"]
    blockers = [blocker for candidate in candidates for blocker in candidate.get("blockers", [])]
    assert "ml_edge_agreement_below_threshold" in blockers
    assert payload["preview_ready_count"] == 0
    assert Order.query.count() == 0


def test_live_rapid_ml_profitability_counts_only_approved_sources(app) -> None:
    service = app.extensions["services"]["rapid_ml_trader"]

    payload = service._profitability_model(
        "hyperliquid",
        {"spread_bps": 1.0},
        confidence=0.9,
        signal={"action": "hold", "expected_return": 0.004, "blockers": []},
        roi={"action": "target_unlikely", "expected_return": 0.004, "blockers": []},
        offline={"status": "promoted", "prediction": 0.004, "blockers": []},
    )

    assert payload["positive_edge_sources"] == ["offline_ranker"]
    assert payload["positive_edge_source_count"] == 1
    assert "ml_edge_agreement_below_threshold" in payload["blockers"]


def test_live_rapid_ml_preserves_live_feature_scores(app, monkeypatch) -> None:
    user, _, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    services = app.extensions["services"]
    trader = services["rapid_ml_trader"]
    captured: list[dict[str, Any]] = []

    monkeypatch.setattr(
        trader,
        "_live_feature_context",
        lambda *args, **kwargs: {"score": 42.0, "expected_return": 0.003, "rapid_feature_candles": []},
    )

    def fake_decision(
        family: str,
        context: dict[str, Any],
        *,
        horizon: str = "1h",
        candles: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if family == "pytorch_gru_signal":
            captured.append({"score": context.get("score"), "expected_return": context.get("expected_return")})
            return {"family": family, "action": "buy", "confidence": 0.9, "expected_return": 0.004, "blockers": [], "ready": True}
        if family == "pytorch_risk_policy":
            return {"family": family, "action": "approve", "confidence": 0.9, "expected_return": 0.002, "blockers": []}
        if family == "pytorch_ops_anomaly":
            return {"family": family, "action": "observe", "confidence": 0.1, "expected_return": 0.0, "blockers": [], "raw": {"ops_anomaly_score": 0.0}}
        if family == "pytorch_allocator":
            return {"family": family, "action": "allocate", "confidence": 0.8, "expected_return": 0.002, "blockers": [], "raw": {"sizing_score": 0.8}}
        if family == "pytorch_roi_target":
            return {"family": family, "action": "target_met_candidate", "confidence": 0.8, "expected_return": 0.004, "blockers": []}
        return {"family": family, "action": "route", "confidence": 0.8, "expected_return": 0.002, "blockers": []}

    monkeypatch.setattr(services["ml_decision_engine"], "decision", fake_decision)

    result = app.test_cli_runner().invoke(
        args=["live-rapid-ml-trader", "--user-id", str(user.id), "--capital-usd", "100", "--provider", "hyperliquid"]
    )

    payload = _payload(result)
    assert payload["preview_ready_count"] == 1
    assert captured
    assert all(item["score"] == 42.0 for item in captured)
    assert all(item["expected_return"] == 0.003 for item in captured)


def test_live_rapid_ml_submit_requires_exact_confirmation(app) -> None:
    user, _, _ = _user_with_connections()
    result = app.test_cli_runner().invoke(
        args=[
            "live-rapid-ml-trader",
            "--user-id",
            str(user.id),
            "--capital-usd",
            "100",
            "--submit",
            "--confirm",
            "WRONG",
        ]
    )

    assert result.exit_code != 0
    assert "RAPID-ML-LIVE" in result.output
    assert Order.query.count() == 0


def test_live_rapid_ml_missing_promoted_models_blocks_live(app, monkeypatch) -> None:
    user, _, _ = _user_with_connections()
    connections = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(connections, "can_trade", lambda user_id, mode, connection_id=None: True)
    monkeypatch.setattr(
        connections,
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

    result = app.test_cli_runner().invoke(
        args=["live-rapid-ml-trader", "--user-id", str(user.id), "--capital-usd", "100", "--provider", "hyperliquid"]
    )

    payload = _payload(result)
    provider_blockers = payload["cycles"][0]["analyses"][0]["blockers"]
    assert any("promoted" in blocker or "ML_ALL_AREAS_ENABLED=false" in blocker for blocker in provider_blockers)
    assert payload["submitted_count"] == 0
    assert Order.query.count() == 0


def test_live_rapid_ml_daily_loss_cap_cannot_be_disabled(app) -> None:
    user = User(username="loss-cap-user", password_hash="x")
    db.session.add(user)
    db.session.commit()
    app.config["RAPID_ML_MAX_DAILY_LOSS_PCT"] = 0.0

    payload = app.extensions["services"]["rapid_ml_trader"].run(user_id=user.id, capital_usd=100.0, provider="hyperliquid")

    assert payload["capital"]["daily_loss_cap_pct"] > 0
    assert payload["capital"]["daily_loss_cap_pct"] <= 0.10
    assert payload["capital"]["daily_loss_cap_usd"] > 0


def test_live_rapid_ml_submit_env_flags_block_order(app, monkeypatch) -> None:
    user, _, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    monkeypatch.setattr(
        app.extensions["services"]["order_manager"],
        "place_order",
        lambda intent: (_ for _ in ()).throw(AssertionError("submit flags are missing")),
    )

    result = app.test_cli_runner().invoke(
        args=[
            "live-rapid-ml-trader",
            "--user-id",
            str(user.id),
            "--capital-usd",
            "100",
            "--provider",
            "both",
            "--submit",
            "--confirm",
            "RAPID-ML-LIVE",
        ]
    )

    payload = _payload(result)
    assert payload["status"] == "submit_blocked"
    assert "RAPID_ML_LIVE_ENABLED=false" in payload["blockers"]
    assert "RAPID_ML_PREVIEW_ONLY=true" in payload["blockers"]
    assert payload["submitted_count"] == 0
    assert Order.query.count() == 0


def test_live_rapid_ml_submit_places_one_order_per_ready_provider(app, monkeypatch) -> None:
    user, _, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    app.config.update(
        {
            "RAPID_ML_LIVE_ENABLED": True,
            "RAPID_ML_PREVIEW_ONLY": False,
            "CANARY_PREVIEW_ONLY": False,
            "EXPLICIT_LIVE_CONFIRMED": True,
            "SECONDARY_CONFIRMATION": True,
            "ML_LIVE_HARD_CAP_USDC": 100.0,
            "ML_LIVE_HARD_DAILY_LOSS_USDC": 10.0,
        }
    )
    Setting.set_json("explicit_live_confirmed", True)
    Setting.set_json("secondary_confirmation", True)

    def fake_place_order(intent) -> Order:
        order = Order(
            user_id=intent.user_id,
            trading_connection_id=intent.trading_connection_id,
            client_order_id=intent.idempotency_key,
            mode=intent.mode,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            status="submitted",
            quantity=intent.quantity,
            reduce_only=intent.reduce_only,
            leverage=intent.leverage,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
            risk_status="approved",
        )
        order.details = dict(intent.metadata)
        db.session.add(order)
        db.session.flush()
        return order

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fake_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "live-rapid-ml-trader",
            "--user-id",
            str(user.id),
            "--capital-usd",
            "100",
            "--provider",
            "both",
            "--duration-minutes",
            "0",
            "--submit",
            "--confirm",
            "RAPID-ML-LIVE",
        ]
    )

    payload = _payload(result)
    assert payload["submitted_count"] == 2
    assert payload["real_order_submitted"] is True
    assert Order.query.count() == 2
    assert {order.details["provider"] for order in Order.query.all()} == {"hyperliquid", "kucoin"}


def test_live_rapid_ml_preview_reports_protective_close_in_compact_output(app, monkeypatch) -> None:
    user, hyperliquid, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    _rapid_source_order(user=user, connection=hyperliquid, symbol="BTC", side="sell", stop_loss=105.0, take_profit=99.0)
    snapshot = _snapshot_with_position(
        "hyperliquid",
        {"symbol": "BTC", "quantity": -1.0, "entry_price": 100.0, "mark_price": 98.0, "unrealized_pnl": 2.0},
    )
    monkeypatch.setattr(app.extensions["services"]["trading_connections"], "account_snapshot", lambda *args, **kwargs: snapshot)
    monkeypatch.setattr(
        app.extensions["services"]["order_manager"],
        "place_order",
        lambda intent: (_ for _ in ()).throw(AssertionError("preview must not submit close orders")),
    )

    result = app.test_cli_runner().invoke(
        args=[
            "live-rapid-ml-trader",
            "--user-id",
            str(user.id),
            "--capital-usd",
            "100",
            "--provider",
            "hyperliquid",
            "--duration-minutes",
            "0",
            "--compact",
        ]
    )

    payload = _payload(result)
    management = payload["providers"][0]["position_management"]
    managed_position = management["positions"][0]
    assert management["status"] == "would_close"
    assert management["preview_ready"] is True
    assert managed_position["trigger"] == "take_profit"
    assert managed_position["would_close"] is True
    assert payload["submitted_count"] == 0


def test_live_rapid_ml_submit_closes_short_take_profit_reduce_only(app, monkeypatch) -> None:
    user, hyperliquid, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    app.config.update(
        {
            "RAPID_ML_LIVE_ENABLED": True,
            "RAPID_ML_PREVIEW_ONLY": False,
            "CANARY_PREVIEW_ONLY": False,
            "EXPLICIT_LIVE_CONFIRMED": True,
            "SECONDARY_CONFIRMATION": True,
            "ML_LIVE_HARD_CAP_USDC": 100.0,
            "ML_LIVE_HARD_DAILY_LOSS_USDC": 10.0,
        }
    )
    Setting.set_json("explicit_live_confirmed", True)
    Setting.set_json("secondary_confirmation", True)
    _rapid_source_order(user=user, connection=hyperliquid, symbol="BTC", side="sell", stop_loss=105.0, take_profit=99.0)
    snapshot = _snapshot_with_position(
        "hyperliquid",
        {"symbol": "BTC", "quantity": -1.0, "entry_price": 100.0, "mark_price": 98.0, "unrealized_pnl": 2.0},
    )
    monkeypatch.setattr(app.extensions["services"]["trading_connections"], "account_snapshot", lambda *args, **kwargs: snapshot)
    submitted_intents = []

    def fake_place_order(intent) -> Order:
        submitted_intents.append(intent)
        order = Order(
            user_id=intent.user_id,
            trading_connection_id=intent.trading_connection_id,
            client_order_id=intent.idempotency_key,
            mode=intent.mode,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            status="submitted",
            quantity=intent.quantity,
            reduce_only=intent.reduce_only,
            leverage=intent.leverage,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
            risk_status="approved",
        )
        order.details = dict(intent.metadata)
        db.session.add(order)
        db.session.flush()
        return order

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fake_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "live-rapid-ml-trader",
            "--user-id",
            str(user.id),
            "--capital-usd",
            "100",
            "--provider",
            "hyperliquid",
            "--duration-minutes",
            "0",
            "--submit",
            "--confirm",
            "RAPID-ML-LIVE",
        ]
    )

    payload = _payload(result)
    assert payload["submitted_count"] == 1
    assert len(submitted_intents) == 1
    intent = submitted_intents[0]
    assert intent.reduce_only is True
    assert intent.side == "buy"
    assert intent.quantity == 1.0
    assert intent.metadata["rapid_ml_exit"] is True
    assert intent.metadata["rapid_ml_exit_trigger"] == "take_profit"
    assert "exchange_response" not in intent.metadata
    assert "risk_decision" not in intent.metadata


def test_live_rapid_ml_snapshot_unavailable_uses_local_rapid_source_reduce_only(app, monkeypatch) -> None:
    user, hyperliquid, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    source = _rapid_source_order(user=user, connection=hyperliquid, symbol="HYPE", side="sell", stop_loss=110.0, take_profit=90.0)
    analysis = {
        "provider": "hyperliquid",
        "connection_id": hyperliquid.id,
        "snapshot": {
            "positions": [],
            "positions_count": 0,
            "balances": [],
            "alerts": ["Exchange data unavailable: timeout"],
            "blockers": ["provider_snapshot_alerts", "usdc_balance_unavailable"],
        },
        "candidates": [],
        "selected": None,
    }
    submitted_intents = []

    def fake_place_order(intent) -> Order:
        submitted_intents.append(intent)
        order = Order(
            user_id=intent.user_id,
            trading_connection_id=intent.trading_connection_id,
            client_order_id=intent.idempotency_key,
            mode=intent.mode,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            status="submitted",
            quantity=intent.quantity,
            reduce_only=intent.reduce_only,
            leverage=intent.leverage,
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
            risk_status="approved",
        )
        order.details = dict(intent.metadata)
        db.session.add(order)
        db.session.flush()
        return order

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fake_place_order)

    result = app.extensions["services"]["rapid_ml_trader"]._manage_provider_positions(
        user_id=user.id,
        session_id=1,
        analysis=analysis,
        submit=True,
    )

    assert result["status"] == "submitted"
    assert result["submitted_count"] == 1
    assert result["positions"][0]["source_order"]["id"] == source.id
    assert result["positions"][0]["trigger"] == "provider_snapshot_unavailable_rapid_ml_close"
    assert len(submitted_intents) == 1
    intent = submitted_intents[0]
    assert intent.reduce_only is True
    assert intent.side == "buy"
    assert intent.symbol == "HYPE"
    assert intent.metadata["provider_snapshot_fallback"] is True
    assert intent.metadata["rapid_ml_exit_source_order_id"] == source.id


def test_live_rapid_ml_snapshot_unavailable_without_source_does_not_report_flat(app, monkeypatch) -> None:
    user, hyperliquid, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    monkeypatch.setattr(
        app.extensions["services"]["order_manager"],
        "place_order",
        lambda intent: (_ for _ in ()).throw(AssertionError("no local rapid source means no close")),
    )
    analysis = {
        "provider": "hyperliquid",
        "connection_id": hyperliquid.id,
        "snapshot": {
            "positions": [],
            "positions_count": 0,
            "balances": [],
            "alerts": ["Exchange data unavailable: timeout"],
            "blockers": ["provider_snapshot_alerts", "usdc_balance_unavailable"],
        },
        "candidates": [],
        "selected": None,
    }

    result = app.extensions["services"]["rapid_ml_trader"]._manage_provider_positions(
        user_id=user.id,
        session_id=1,
        analysis=analysis,
        submit=True,
    )

    assert result["status"] == "snapshot_unavailable"
    assert result["positions"] == []
    assert "provider_snapshot_alerts" in result["blockers"]


def test_live_rapid_ml_snapshot_fallback_skips_source_after_successful_exit(app, monkeypatch) -> None:
    user, hyperliquid, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    source = _rapid_source_order(user=user, connection=hyperliquid, symbol="HYPE", side="sell", stop_loss=110.0, take_profit=90.0)
    exit_order = Order(
        user_id=user.id,
        trading_connection_id=hyperliquid.id,
        client_order_id="rapid-exit-hype",
        mode="live",
        symbol="HYPE",
        side="buy",
        order_type="market",
        status="filled",
        quantity=1.0,
        filled_quantity=1.0,
        average_fill_price=100.0,
        reduce_only=True,
        leverage=1.0,
        risk_status="approved",
    )
    exit_order.details = {"rapid_ml": True, "rapid_ml_exit": True, "rapid_ml_exit_source_order_id": source.id}
    db.session.add(exit_order)
    db.session.commit()
    monkeypatch.setattr(
        app.extensions["services"]["order_manager"],
        "place_order",
        lambda intent: (_ for _ in ()).throw(AssertionError("source with successful exit must not close again")),
    )
    analysis = {
        "provider": "hyperliquid",
        "connection_id": hyperliquid.id,
        "snapshot": {
            "positions": [],
            "positions_count": 0,
            "balances": [],
            "alerts": ["Exchange data unavailable: timeout"],
            "blockers": ["provider_snapshot_alerts", "usdc_balance_unavailable"],
        },
        "candidates": [],
        "selected": None,
    }

    result = app.extensions["services"]["rapid_ml_trader"]._manage_provider_positions(
        user_id=user.id,
        session_id=1,
        analysis=analysis,
        submit=True,
    )

    assert result["status"] == "snapshot_unavailable"
    assert result["positions"] == []


def test_live_rapid_ml_closes_long_stop_loss_reduce_only(app, monkeypatch) -> None:
    user, hyperliquid, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    source = _rapid_source_order(user=user, connection=hyperliquid, symbol="BTC", side="buy", stop_loss=95.0, take_profit=110.0)
    analysis = {
        "provider": "hyperliquid",
        "connection_id": hyperliquid.id,
        "snapshot": {
            "positions": [{"symbol": "BTC", "quantity": 1.0, "entry_price": 100.0, "mark_price": 94.0}],
            "positions_count": 1,
            "blockers": [],
        },
        "candidates": [{"symbol": "BTC", "action": "buy", "side": "buy", "opportunity_score": 1.0, "expected_edge_bps_after_costs": 100.0, "blockers": []}],
        "selected": {"symbol": "BTC", "action": "buy", "side": "buy", "opportunity_score": 1.0, "expected_edge_bps_after_costs": 100.0, "blockers": []},
    }
    submitted_intents = []

    def fake_place_order(intent) -> Order:
        submitted_intents.append(intent)
        return Order(client_order_id=intent.idempotency_key, mode="live", symbol=intent.symbol, side=intent.side, status="submitted", quantity=intent.quantity)

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fake_place_order)

    result = app.extensions["services"]["rapid_ml_trader"]._manage_provider_positions(
        user_id=user.id,
        session_id=1,
        analysis=analysis,
        submit=True,
    )

    assert source.id == result["positions"][0]["source_order"]["id"]
    assert result["positions"][0]["trigger"] == "stop_loss"
    assert submitted_intents[0].reduce_only is True
    assert submitted_intents[0].side == "sell"


def test_live_rapid_ml_rotation_closes_when_new_candidate_is_materially_stronger(app, monkeypatch) -> None:
    user, hyperliquid, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    _rapid_source_order(user=user, connection=hyperliquid, symbol="HYPE", side="sell", stop_loss=110.0, take_profit=90.0)
    analysis = {
        "provider": "hyperliquid",
        "connection_id": hyperliquid.id,
        "snapshot": {
            "positions": [{"symbol": "HYPE", "quantity": -1.0, "entry_price": 100.0, "mark_price": 100.0}],
            "positions_count": 1,
            "blockers": [],
        },
        "candidates": [
            {"symbol": "HYPE", "action": "sell", "side": "sell", "opportunity_score": 1.0, "expected_edge_bps_after_costs": 120.0, "blockers": []},
            {"symbol": "XRP", "action": "sell", "side": "sell", "opportunity_score": 1.2, "expected_edge_bps_after_costs": 120.0, "blockers": []},
        ],
        "selected": {"symbol": "XRP", "action": "sell", "side": "sell", "opportunity_score": 1.2, "expected_edge_bps_after_costs": 120.0, "blockers": []},
    }
    submitted_intents = []
    monkeypatch.setattr(
        app.extensions["services"]["order_manager"],
        "place_order",
        lambda intent: submitted_intents.append(intent)
        or Order(client_order_id=intent.idempotency_key, mode="live", symbol=intent.symbol, side=intent.side, status="submitted", quantity=intent.quantity),
    )

    result = app.extensions["services"]["rapid_ml_trader"]._manage_provider_positions(
        user_id=user.id,
        session_id=1,
        analysis=analysis,
        submit=True,
    )

    assert result["positions"][0]["trigger"] == "ml_rotate_stronger_candidate"
    assert result["positions"][0]["rotation_score_delta"] == 0.19999999999999996
    assert submitted_intents[0].reduce_only is True
    assert submitted_intents[0].side == "buy"


def test_live_rapid_ml_does_not_close_manual_or_wrong_connection_positions(app, monkeypatch) -> None:
    user, hyperliquid, kucoin = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    _rapid_source_order(user=user, connection=kucoin, symbol="BTC", side="sell", stop_loss=105.0, take_profit=99.0)
    analysis = {
        "provider": "hyperliquid",
        "connection_id": hyperliquid.id,
        "snapshot": {
            "positions": [{"symbol": "BTC", "quantity": -1.0, "entry_price": 100.0, "mark_price": 98.0}],
            "positions_count": 1,
            "blockers": [],
        },
        "candidates": [{"symbol": "BTC", "action": "sell", "side": "sell", "opportunity_score": 1.0, "expected_edge_bps_after_costs": 100.0, "blockers": []}],
        "selected": {"symbol": "BTC", "action": "sell", "side": "sell", "opportunity_score": 1.0, "expected_edge_bps_after_costs": 100.0, "blockers": []},
    }
    monkeypatch.setattr(
        app.extensions["services"]["order_manager"],
        "place_order",
        lambda intent: (_ for _ in ()).throw(AssertionError("manual or wrong-connection positions must not close")),
    )

    result = app.extensions["services"]["rapid_ml_trader"]._manage_provider_positions(
        user_id=user.id,
        session_id=1,
        analysis=analysis,
        submit=True,
    )

    assert result["submitted_count"] == 0
    assert result["positions"][0]["source_order"] is None
    assert "position_not_rapid_ml_owned" in result["positions"][0]["blockers"]


def test_live_rapid_ml_kucoin_missing_contract_specs_fails_closed(app, monkeypatch) -> None:
    user, _, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch)
    app.config["KUCOIN_CONTRACT_SPECS_JSON"] = {}

    result = app.test_cli_runner().invoke(
        args=["live-rapid-ml-trader", "--user-id", str(user.id), "--capital-usd", "100", "--provider", "kucoin"]
    )

    payload = _payload(result)
    candidates = payload["cycles"][0]["analyses"][0]["candidates"]
    assert any("kucoin_contract_specs_missing" in blocker for candidate in candidates for blocker in candidate.get("blockers", []))
    assert payload["submitted_count"] == 0
    assert Order.query.count() == 0


def test_live_rapid_ml_correlation_guard_blocks_duplicate_same_symbol(app, monkeypatch) -> None:
    user, _, _ = _user_with_connections()
    _install_ready_rapid_fakes(app, monkeypatch, same_symbol=True)

    result = app.test_cli_runner().invoke(
        args=[
            "live-rapid-ml-trader",
            "--user-id",
            str(user.id),
            "--capital-usd",
            "100",
            "--provider",
            "both",
        ]
    )

    payload = _payload(result)
    blockers = [
        blocker
        for execution in payload["cycles"][0]["executions"]
        for blocker in execution.get("blockers", [])
    ]
    assert "global_correlation_same_symbol_side" in blockers
    assert payload["preview_ready_count"] == 1


def test_live_rapid_ml_order_throttle_blocks_too_fast(app) -> None:
    service = app.extensions["services"]["rapid_ml_trader"]
    blocker = service._throttle_blocker("hyperliquid", 1.0, {"hyperliquid": time.monotonic()})
    assert blocker == "order_rate_throttle"


def test_live_rapid_ml_rejected_order_burst_trips_circuit_breaker(app) -> None:
    user, hyperliquid, _ = _user_with_connections()
    for index in range(3):
        order = Order(
            user_id=user.id,
            trading_connection_id=hyperliquid.id,
            client_order_id=f"rejected-{index}",
            mode="live",
            symbol="BTC",
            side="buy",
            order_type="market",
            status="rejected",
            quantity=0.01,
            reduce_only=False,
            leverage=1.0,
            risk_status="approved",
        )
        order.details = {"provider": "hyperliquid", "rapid_ml": True}
        db.session.add(order)
    db.session.commit()

    blockers = app.extensions["services"]["rapid_ml_trader"]._circuit_breakers("hyperliquid", hyperliquid.id)

    assert "rejected_order_burst_circuit_breaker" in blockers


def test_live_rapid_ml_provider_failure_breaker_does_not_count_itself(app) -> None:
    user, hyperliquid, _ = _user_with_connections()
    session = RapidMLSession(
        user_id=user.id,
        provider_scope="hyperliquid",
        capital_usd=10.0,
        status="blocked",
    )
    db.session.add(session)
    db.session.flush()
    for _ in range(3):
        decision = RapidMLDecision(
            session_id=session.id,
            user_id=user.id,
            trading_connection_id=hyperliquid.id,
            provider="hyperliquid",
            status="blocked",
        )
        decision.blockers = ["provider_failure_burst_circuit_breaker"]
        db.session.add(decision)
    db.session.commit()

    blockers = app.extensions["services"]["rapid_ml_trader"]._circuit_breakers("hyperliquid", hyperliquid.id)

    assert "provider_failure_burst_circuit_breaker" not in blockers


def test_live_rapid_ml_provider_failure_breaker_counts_raw_provider_failures(app) -> None:
    user, hyperliquid, _ = _user_with_connections()
    session = RapidMLSession(
        user_id=user.id,
        provider_scope="hyperliquid",
        capital_usd=10.0,
        status="blocked",
    )
    db.session.add(session)
    db.session.flush()
    for _ in range(3):
        decision = RapidMLDecision(
            session_id=session.id,
            user_id=user.id,
            trading_connection_id=hyperliquid.id,
            provider="hyperliquid",
            status="blocked",
        )
        decision.blockers = ["provider_request_failed:timeout"]
        db.session.add(decision)
    db.session.commit()

    blockers = app.extensions["services"]["rapid_ml_trader"]._circuit_breakers("hyperliquid", hyperliquid.id)

    assert "provider_failure_burst_circuit_breaker" in blockers
