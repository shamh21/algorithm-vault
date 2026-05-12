from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pyotp

from app.auth import encrypt_totp_secret, password_hash
from app.extensions import db
from app.models import (
    DepositAddress,
    Fill,
    Order,
    Setting,
    VaultCycle,
    WalletAddress,
    WalletAuditLog,
    WalletBalance,
    WalletTransaction,
    WalletWithdrawal,
    User,
)
from app.services.order_manager import OrderIntent
from app.services.self_custody_wallet import AddressBalance
from app.services.hyperliquid_client import ClientSnapshot


def _candles():
    return [
        {"timestamp": index, "open": 100, "high": 101, "low": 99, "close": 100 + index * 0.01, "volume": 1000}
        for index in range(80)
    ]


def _patch_market_data(app) -> None:
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: {
        "levels": [[{"px": "99.95", "sz": "1000"}], [{"px": "100.05", "sz": "1000"}]]
    }
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles()


def _create_user(username="multi"):
    user = User(username=username, password_hash=password_hash("password123"), role="user")
    secret = pyotp.random_base32()
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


def _start_cycle(client, amount: str, duration: str = "24"):
    data = {
        "deposit_amount": amount,
        "deposit_asset": "USDC",
        "lock_duration": duration,
        "settlement_asset": "USDC",
    }
    if str(duration) == "1":
        _confirm_one_h10_live(client.application)
        data["one_h10_live_ack"] = "on"
    return client.post(
        "/vault/start",
        data=data,
        follow_redirects=True,
    )


def _seed_custody_usdc(user: User, amount: float = 1000.0) -> None:
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=amount, estimated_usd_value=amount))
    db.session.commit()


def test_multiple_vault_cycles_can_run_at_once(app, monkeypatch) -> None:
    _patch_market_data(app)
    _confirm_one_h10_live(app)
    monkeypatch.setitem(app.extensions["services"], "one_h10_forecast", _PassingOneH10Forecast())
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    user, secret = _create_user()
    _seed_custody_usdc(user)
    client = app.test_client()
    _login(client, user.username, secret)
    client.get("/wallet")

    first = _start_cycle(client, "100", "1")
    second = _start_cycle(client, "100", "24")

    assert first.status_code == 200
    assert second.status_code == 200
    assert VaultCycle.query.filter_by(user_id=user.id, status="active").count() == 2
    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDC").one()
    assert balance.available_balance == 800
    assert balance.locked_balance == 200
    vault = client.get("/vault")
    assert b"2 Active Cycles" in vault.data


def test_vault_concentration_rejects_excess_asset_exposure(app) -> None:
    _patch_market_data(app)
    app.config["VAULT_MAX_ASSET_EXPOSURE_PCT"] = 0.20
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    user, secret = _create_user(username="concentration")
    _seed_custody_usdc(user)
    client = app.test_client()
    _login(client, user.username, secret)
    client.get("/wallet")

    assert _start_cycle(client, "150", "24").status_code == 200
    blocked = _start_cycle(client, "100", "48")

    assert VaultCycle.query.filter_by(user_id=user.id, status="active").count() == 1
    assert b"asset exposure would exceed" in blocked.data


def test_cycle_settlement_persists_trade_risk_leverage_reward_summary(app, monkeypatch) -> None:
    _patch_market_data(app)
    monkeypatch.setitem(app.extensions["services"], "one_h10_forecast", _PassingOneH10Forecast())
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    app.extensions["services"]["strategy_manager"].stop = lambda run_id: None
    app.extensions["services"]["order_manager"].current_position = lambda *args, **kwargs: {"unrealized_pnl": 2.0}
    user, secret = _create_user(username="summary")
    _seed_custody_usdc(user)
    client = app.test_client()
    _login(client, user.username, secret)
    client.get("/wallet")
    _start_cycle(client, "100", "1")
    cycle = VaultCycle.query.filter_by(user_id=user.id).one()
    leg = cycle.allocation_legs[0]
    app.extensions["services"]["order_manager"].current_position = (
        lambda symbol, *args, **kwargs: {"unrealized_pnl": 2.0 if symbol == leg.symbol else 0.0}
    )
    order = Order(
        user_id=user.id,
        trading_connection_id=cycle.trading_connection_id,
        client_order_id="summary-order",
        mode="live",
        symbol=leg.symbol,
        side="sell",
        order_type="market",
        status="filled",
        strategy_name=leg.strategy_run.strategy_name,
        quantity=1.0,
        filled_quantity=1.0,
        average_fill_price=100,
        leverage=2.0,
        stop_loss=95,
        take_profit=110,
    )
    order.details = {"vault_cycle_id": cycle.id, "vault_leg_id": leg.id, "slippage_bps": 3.5}
    db.session.add(order)
    db.session.flush()
    db.session.add(Fill(order_id=order.id, symbol=leg.symbol, side="sell", quantity=1.0, price=100, fee=1.0, pnl=10.0))
    cycle.unlocks_at = datetime.utcnow() - timedelta(seconds=1)
    db.session.commit()

    response = client.get("/vault")
    refreshed = db.session.get(VaultCycle, cycle.id)

    assert response.status_code == 200
    assert refreshed.status == "complete"
    assert refreshed.final_settlement_amount == 111
    assert refreshed.cycle_summary["order_count"] == 1
    assert refreshed.cycle_summary["fill_count"] == 1
    assert refreshed.cycle_summary["realized_pnl_usd"] == 9
    assert refreshed.cycle_summary["unrealized_pnl_usd"] == 2
    assert refreshed.cycle_summary["max_leverage"] == 2
    assert refreshed.cycle_summary["risk_reward"] == 2
    assert WalletTransaction.query.filter_by(user_id=user.id, transaction_type="settlement").count() == 1
    detail = client.get(f"/vault/cycles/{cycle.id}")
    for removed_copy in [
        b"Testing view",
        b"Token And Profit Logic",
        b"Cycle Mechanics",
        b"1H10 Diagnostics",
        b"Scanner, ML, And Blockers",
        b"Submitted Orders",
        b"Rejected Orders",
        b"Failed Orders",
    ]:
        assert removed_copy not in detail.data
    assert b"summary-order" in detail.data
    assert b"Trade Summary" in detail.data
    assert b"Filled" in detail.data
    assert b"2.00x" in detail.data


def test_all_wallet_assets_render_in_allocation_and_settlement_selectors(app) -> None:
    _patch_market_data(app)
    user, secret = _create_user(username="selectors")
    client = app.test_client()
    _login(client, user.username, secret)
    vault = client.get("/vault")
    wallet = client.get("/wallet")

    for asset in (b"USDC", b"USDT", b"BTC", b"ETH", b"SOL", b"XRP"):
        assert asset in vault.data
        assert asset in wallet.data
    assert b'<option value="SOL"' in vault.data
    assert b'<option value="XRP"' in vault.data


def test_standard_duration_cycles_start_and_settle(app, monkeypatch) -> None:
    _patch_market_data(app)
    monkeypatch.setitem(app.extensions["services"], "one_h10_forecast", _PassingOneH10Forecast())
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    app.extensions["services"]["strategy_manager"].stop = lambda run_id: None
    app.extensions["services"]["order_manager"].current_position = lambda *args, **kwargs: {"unrealized_pnl": 0.0}
    user, secret = _create_user(username="durations")
    _seed_custody_usdc(user)
    client = app.test_client()
    _login(client, user.username, secret)
    client.get("/wallet")

    for duration in ("1", "24", "48", "168"):
        response = _start_cycle(client, "50", duration)
        assert response.status_code == 200

    cycles = VaultCycle.query.filter_by(user_id=user.id).order_by(VaultCycle.lock_duration_hours.asc()).all()
    assert [cycle.lock_duration_hours for cycle in cycles] == [1, 24, 48, 168]
    for cycle in cycles:
        cycle.unlocks_at = datetime.utcnow() - timedelta(seconds=1)
    db.session.commit()

    settled = client.get("/vault")

    assert settled.status_code == 200
    assert VaultCycle.query.filter_by(user_id=user.id, status="complete").count() == 4
    assert WalletTransaction.query.filter_by(user_id=user.id, transaction_type="settlement").count() == 4
    assert all(cycle.cycle_summary.get("legs") for cycle in VaultCycle.query.filter_by(user_id=user.id).all())


def test_one_hour_high_return_mode_still_respects_risk_gates(app) -> None:
    app.config["ALLOW_AGGRESSIVE_LIVE_TRADING"] = True
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = OrderIntent(
        symbol="BTC",
        side="buy",
        quantity=0.1,
        mode="live",
        leverage=1.0,
        strategy_name="scalping",
        timeframe="1m",
        metadata={"optimizer_profile": "aggressive_1h", "account_equity_usd": 2_000.0},
    )

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "stop_loss_required"


def test_self_custody_disabled_by_default_and_secret_columns_absent(app) -> None:
    service = app.extensions["services"]["self_custody_wallet"]
    forbidden = {"private_key", "mnemonic", "seed", "xpub", "derivation"}
    model_columns = {
        name.lower()
        for model in (WalletAddress, WalletWithdrawal)
        for name in model.__table__.columns.keys()
    }

    assert service.enabled is False
    assert service.validate_address("Ethereum", "0xcfc7d08f480E6F8c3631268ed49B44cdff389677")
    assert not service.supports_network("SOL", "Solana")
    assert not any(any(term in column for term in forbidden) for column in model_columns)


def test_rotation_with_funds_creates_pending_draft_without_auto_sweep(app) -> None:
    _patch_market_data(app)
    app.config["WALLET_SELF_CUSTODY_ENABLED"] = True
    app.config["WALLET_WITHDRAWAL_FEE_BPS"] = 100
    app.config["WALLET_WITHDRAWAL_FIXED_FEE_ETH"] = 0.001
    app.config["DEPOSIT_ADDRESS_BOOK"] = {
        "USDC": {
            "Ethereum": [
                "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ]
        }
    }
    service = app.extensions["services"]["self_custody_wallet"]
    service.check_address_balance = lambda asset, network, address: AddressBalance(100.0, asset, 0.05, True)
    user, secret = _create_user(username="draft")
    client = app.test_client()
    _login(client, user.username, secret)
    client.get("/wallet/deposit/USDC")

    response = client.post("/wallet/rotate-address/USDC", data={"confirm_rotate": "on"})
    withdrawal = WalletWithdrawal.query.one()

    assert response.status_code == 302
    assert withdrawal.status == "pending_approval"
    assert withdrawal.workflow_type == "rotated_address_draft"
    assert withdrawal.destination_address == "0xcfc7d08f480E6F8c3631268ed49B44cdff389677"
    assert withdrawal.fee_eth == 0.0015
    assert WalletAuditLog.query.filter_by(action="rotation_sweep_workflow_created").count() == 1


def test_auto_sweep_is_blocked_without_required_mainnet_gates_and_idempotent(app) -> None:
    _patch_market_data(app)
    app.config["WALLET_SELF_CUSTODY_ENABLED"] = True
    app.config["WALLET_AUTO_SWEEP_ENABLED"] = True
    service = app.extensions["services"]["self_custody_wallet"]
    service.check_address_balance = lambda asset, network, address: AddressBalance(1.0, asset, 1.0, True)
    user, _ = _create_user(username="autosweep")
    old = DepositAddress(
        user_id=user.id,
        asset="ETH",
        network="Ethereum",
        address="0x1111111111111111111111111111111111111111",
        is_active=False,
    )
    new = DepositAddress(
        user_id=user.id,
        asset="ETH",
        network="Ethereum",
        address="0x2222222222222222222222222222222222222222",
        is_active=True,
    )
    db.session.add_all([old, new])
    db.session.commit()

    first = service.handle_rotated_address(user.id, "ETH", "Ethereum", old, new)
    second = service.handle_rotated_address(user.id, "ETH", "Ethereum", old, new)

    assert first.id == second.id
    assert WalletWithdrawal.query.count() == 1
    assert first.status == "blocked"
    assert "wallet RPC is not configured" in first.failure_reason


def test_realtime_market_falls_back_to_http_and_uses_fresh_websocket_cache(app) -> None:
    _patch_market_data(app)
    service = app.extensions["services"]["realtime_market"]

    fallback = service.snapshot("BTC", "testnet", timeframe="1m")
    messages = service.subscription_messages("BTC", user="0x1111111111111111111111111111111111111111")
    app.config["REALTIME_MARKET_ENABLED"] = True
    service.ingest_message({"channel": "allMids", "data": {"mids": {"BTC": "101"}}})
    service.ingest_message({"channel": "l2Book", "data": {"coin": "BTC", "levels": [[{"px": "100", "sz": "5"}], [{"px": "102", "sz": "5"}]]}})
    service.ingest_message({"channel": "trades", "data": [{"coin": "BTC", "px": "100", "sz": "1"}, {"coin": "BTC", "px": "101", "sz": "1"}]})
    cached = service.snapshot("BTC", "testnet", timeframe="1m")

    assert fallback["source"] == "http"
    assert {"method": "subscribe", "subscription": {"type": "allMids"}} in messages
    assert {"method": "subscribe", "subscription": {"type": "l2Book", "coin": "BTC"}} in messages
    assert {"method": "subscribe", "subscription": {"type": "trades", "coin": "BTC"}} in messages
    assert {"method": "subscribe", "subscription": {"type": "userFills", "user": "0x1111111111111111111111111111111111111111"}} in messages
    assert cached["source"] == "websocket"
    assert cached["mid"] == 101
    assert cached["spread_bps"] > 0
