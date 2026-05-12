from __future__ import annotations

import json
from datetime import datetime, timedelta

import pyotp

from app.auth import encrypt_totp_secret, password_hash
from app.extensions import db
from app.models import TradingConnection, User, VaultCycle, WalletBalance, WalletTransaction


def _create_user(username: str = "sufyanh", *, role: str = "admin") -> User:
    user = User(username=username, password_hash=password_hash("password123"), role=role)
    user.totp_secret_encrypted = encrypt_totp_secret(pyotp.random_base32())
    user.two_factor_enabled_at = datetime.utcnow()
    db.session.add(user)
    db.session.commit()
    return user


def _login(client, user: User) -> None:
    with client.session_transaction() as session:
        session["user_id"] = user.id
        session["two_factor_verified"] = True


def _create_verified_connection(user: User, *, provider: str = "hyperliquid", active: bool = True) -> TradingConnection:
    connection = TradingConnection(
        user_id=user.id,
        provider=provider,
        connection_type="cex_api_key",
        is_active=active,
        verification_status="verified",
        last_verified_at=datetime.utcnow(),
    )
    db.session.add(connection)
    db.session.commit()
    return connection


def _patch_dashboard_market_data(app) -> None:
    market_data = app.extensions["services"]["market_data"]
    candles = [
        {"timestamp": i, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}
        for i in range(80)
    ]
    market_data.get_dashboard_market_summary = lambda symbols, timeframe, mode: [
        {"symbol": symbol, "mid": 100.0, "recent_average": 100.0, "change_pct": 0.0}
        for symbol in symbols
    ]
    market_data.get_candles = lambda symbol, timeframe, mode, limit: candles


def test_profile_wallet_check_reports_sufyanh_local_funds_and_warnings(app) -> None:
    user = _create_user()
    db.session.add_all(
        [
            WalletBalance(user_id=user.id, asset="BTC", available_balance=0.0000294848, estimated_usd_value=2.38),
            WalletBalance(user_id=user.id, asset="ETH", available_balance=0.0120367629, estimated_usd_value=27.87),
            WalletBalance(user_id=user.id, asset="USDC", available_balance=5.0001690445, estimated_usd_value=5.0001690445),
            WalletBalance(user_id=user.id, asset="USDT", available_balance=0.9972689688, estimated_usd_value=0.9972689688),
        ]
    )
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDC",
        deposit_amount=5.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        unlocks_at=datetime.utcnow() - timedelta(hours=1),
        status="complete",
        execution_substatus="complete",
    )
    db.session.add(cycle)
    db.session.flush()
    db.session.add_all(
        [
            WalletTransaction(user_id=user.id, vault_cycle_id=cycle.id, asset="USDC", amount=5.0, transaction_type="settlement", status="complete"),
            WalletTransaction(user_id=user.id, vault_cycle_id=cycle.id, asset="USDC", amount=5.1, transaction_type="settlement", status="complete"),
        ]
    )
    _create_verified_connection(user, provider="hyperliquid", active=True)
    _create_verified_connection(user, provider="kucoin", active=False)
    db.session.commit()

    result = app.test_cli_runner().invoke(args=["profile-wallet-check", "--username", "sufyanh"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["exists"] is True
    assert payload["user"]["username"] == "sufyanh"
    assert payload["user"]["two_factor_enabled"] is True
    assert payload["wallet"]["source"] == "local_app_wallet"
    assert payload["wallet"]["locked_total"] == 0.0
    assert payload["activity"]["order_count"] == 0
    assert [item["provider"] for item in payload["trading_connections"]] == ["hyperliquid", "kucoin"]
    assert any("duplicate_complete_settlement_transactions" in item for item in payload["reconciliation_warnings"])


def test_profile_wallet_check_missing_user_exits_nonzero(app) -> None:
    result = app.test_cli_runner().invoke(args=["profile-wallet-check", "--username", "missing"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["exists"] is False
    assert payload["blockers"] == ["profile_not_found"]


def test_wallet_and_dashboard_pages_do_not_require_fresh_provider_snapshot(app) -> None:
    user = _create_user()
    _create_verified_connection(user)
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=10.0, estimated_usd_value=10.0))
    db.session.commit()
    _patch_dashboard_market_data(app)
    app.extensions["services"]["trading_connections"].account_snapshot = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("normal page render must not request a fresh provider snapshot")
    )
    client = app.test_client()
    _login(client, user)

    home = client.get("/")
    wallet = client.get("/wallet")
    dashboard = client.get("/admin/dashboard")
    dashboard_api = client.get("/admin/api/dashboard-data")

    assert home.status_code == 200
    assert wallet.status_code == 200
    assert dashboard.status_code == 200
    assert dashboard_api.status_code == 200
    assert b"Refresh Snapshot" not in wallet.data
    assert b"Exchange Margin" not in wallet.data


def test_wallet_exchange_snapshot_refresh_param_no_longer_renders_margin(app) -> None:
    user = _create_user()
    _create_verified_connection(user)
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=10.0, estimated_usd_value=10.0))
    db.session.commit()
    app.extensions["services"]["trading_connections"].account_snapshot = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("wallet page must not refresh exchange margin snapshots")
    )
    client = app.test_client()
    _login(client, user)

    response = client.get("/wallet?refresh_exchange=1")

    assert response.status_code == 200
    assert b"Exchange Margin" not in response.data
    assert b"Refresh Snapshot" not in response.data
