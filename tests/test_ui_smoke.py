from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import pyotp
import pytest
from cryptography.fernet import Fernet
from werkzeug.security import check_password_hash

from app import create_app
from app.auth import decrypt_totp_secret, encrypt_totp_secret, password_hash
from app.extensions import db
from app.models import (
    AuditLog,
    BacktestRun,
    DepositAddress,
    Fill,
    LeveragedMarket,
    OptimizerRun,
    Order,
    ReferralInviteCode,
    Setting,
    StrategyRanking,
    StrategyRun,
    StrategyValidation,
    User,
    VaultAllocationLeg,
    VaultCycle,
    WalletAddress,
    WalletBalance,
    WalletLedgerEvent,
    WalletTransaction,
    WalletWithdrawal,
)
from app.services.hyperliquid_client import ClientSnapshot
from app.services.wallet_custody import BroadcastResult, GeneratedWallet, RealWalletCustodyService, WalletBalanceSnapshot


def _candles():
    return [
        {
            "timestamp": index,
            "open": 100 + index,
            "high": 101 + index,
            "low": 99 + index,
            "close": 100 + index,
            "volume": 1000,
        }
        for index in range(80)
    ]


def _patch_market_data(app) -> None:
    market_data = app.extensions["services"]["market_data"]
    market_data.get_dashboard_market_summary = lambda symbols, timeframe, mode: [
        {"symbol": symbol, "mid": 100.0, "recent_average": 100.0, "change_pct": 0.0} for symbol in symbols
    ]
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles()
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: {"levels": [[{"px": "99.95", "sz": "1"}], [{"px": "100.05", "sz": "1"}]]}


def _patch_deep_book(app, spread: float = 0.1, size: str = "1000") -> None:
    bid = 100.0 - spread / 2
    ask = 100.0 + spread / 2
    app.extensions["services"]["market_data"].get_order_book = lambda symbol, mode: {
        "levels": [[{"px": str(bid), "sz": size}], [{"px": str(ask), "sz": size}]]
    }


def _create_user(username="alice", role="user", enabled_2fa=True):
    user = User(username=username, password_hash=password_hash("password123"), role=role)
    secret = pyotp.random_base32()
    user.totp_secret_encrypted = encrypt_totp_secret(secret)
    if enabled_2fa:
        user.two_factor_enabled_at = datetime.utcnow()
    db.session.add(user)
    db.session.commit()
    return user, secret


def _login(client, username: str, secret: str, password: str = "password123"):
    response = client.post(
        "/login",
        data={"username": username, "password": password, "totp_code": pyotp.TOTP(secret).now()},
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


def _seed_backtest_market(provider: str = "hyperliquid", symbol: str = "BTC", venue_symbol: str | None = None) -> LeveragedMarket:
    market = LeveragedMarket(
        provider=provider,
        venue_symbol=venue_symbol or symbol,
        symbol=symbol,
        status="active",
        settlement_asset="USDC" if provider == "hyperliquid" else "USDT",
        max_leverage=20.0,
        liquidity_usd=250_000.0,
        spread_bps=2.5,
        fee_bps=5.0,
    )
    db.session.add(market)
    db.session.commit()
    return market


def _seed_wallet_balance(user: User, asset: str = "USDC", available: float = 10_000.0, estimated_usd: float | None = None) -> WalletBalance:
    balance = WalletBalance(
        user_id=user.id,
        asset=asset,
        available_balance=available,
        locked_balance=0.0,
        estimated_usd_value=available if estimated_usd is None else estimated_usd,
    )
    db.session.add(balance)
    db.session.commit()
    return balance


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


class _LiveWalletAdapter:
    def __init__(self) -> None:
        self.broadcasts = 0

    def supports(self, asset: str, network: str) -> bool:
        return asset.upper() in {"ETH", "USDC"} and network == "Ethereum"

    def generate_wallet(self, asset: str, network: str) -> GeneratedWallet:
        return GeneratedWallet(
            address="0x1234567890abcdef1234567890abcdef12345678",
            private_key="11" * 32,
            public_key="0x1234567890abcdef1234567890abcdef12345678",
            key_type="secp256k1",
            provider="fake_evm",
        )

    def get_balance(self, address: str, asset: str, network: str) -> WalletBalanceSnapshot:
        amount = 2.0 if asset.upper() == "ETH" else 1000.0
        return WalletBalanceSnapshot(amount=amount, asset=asset, checked=True, confirmations=12)

    def estimate_fee(self, asset: str, network: str, destination: str, amount: float) -> float:
        return 0.001

    def sign_and_broadcast(self, withdrawal: WalletWithdrawal, private_key: str) -> BroadcastResult:
        assert private_key == "11" * 32
        self.broadcasts += 1
        return BroadcastResult("submitted", "0xroutehash", {"ok": True})

    def confirm_transaction(self, provider_reference: str, asset: str, network: str) -> dict:
        return {"confirmed": True}


class _LiveUsdtWalletAdapter(_LiveWalletAdapter):
    def supports(self, asset: str, network: str) -> bool:
        return asset.upper() in {"ETH", "USDT"} and network == "Ethereum"

    def get_balance(self, address: str, asset: str, network: str) -> WalletBalanceSnapshot:
        amount = 10.0 if asset.upper() == "USDT" else 0.01
        return WalletBalanceSnapshot(amount=amount, asset=asset, checked=True, confirmations=12)


class _FailingSyncCustody:
    enabled = True

    def __init__(self) -> None:
        self.sync_calls = 0

    def sync_user(self, user_id: int) -> None:
        self.sync_calls += 1
        raise RuntimeError(f"custody sync should be skipped for user {user_id}")


def _enable_live_wallets(app) -> tuple[_LiveWalletAdapter, RealWalletCustodyService]:
    app.config["USE_REAL_ADDRESSES"] = True
    app.config["WALLET_REAL_CUSTODY_ENABLED"] = True
    app.config["WALLET_ALLOW_IN_APP_KEYGEN"] = True
    app.config["WALLET_WITHDRAWALS_ENABLED"] = True
    app.config["WALLET_REQUIRE_WITHDRAWAL_APPROVAL"] = True
    app.config["TOTP_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    app.config["WALLET_EVM_RPC_URL"] = "https://evm.example.invalid"
    app.config["WALLET_EVM_TOKEN_CONTRACTS"] = {
        "ETHEREUM": {
            "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            "USDC_DECIMALS": 6,
        }
    }
    Setting.set_json("use_real_addresses", True)
    fake = _LiveWalletAdapter()
    custody = RealWalletCustodyService(app.config, adapters=[fake])
    app.extensions["services"]["wallet_custody"] = custody
    return fake, custody


def _enable_live_usdt_wallets(app) -> tuple[_LiveUsdtWalletAdapter, RealWalletCustodyService]:
    app.config["USE_REAL_ADDRESSES"] = True
    app.config["WALLET_REAL_CUSTODY_ENABLED"] = True
    app.config["WALLET_ALLOW_IN_APP_KEYGEN"] = True
    app.config["WALLET_WITHDRAWALS_ENABLED"] = True
    app.config["WALLET_REQUIRE_WITHDRAWAL_APPROVAL"] = False
    app.config["TOTP_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    app.config["WALLET_EVM_RPC_URL"] = "https://evm.example.invalid"
    app.config["WALLET_EVM_TOKEN_CONTRACTS"] = {
        "ETHEREUM": {
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "USDT_DECIMALS": 6,
        }
    }
    Setting.set_json("use_real_addresses", True)
    fake = _LiveUsdtWalletAdapter()
    custody = RealWalletCustodyService(app.config, adapters=[fake])
    app.extensions["services"]["wallet_custody"] = custody
    app.extensions["services"]["self_custody_wallet"].custody = custody
    return fake, custody


def test_signup_enabled_without_invite_code() -> None:
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite://",
            "SIGNUP_INVITE_CODE": "",
            "ADMIN_PASSWORD": "",
        }
    )
    with app.app_context():
        client = app.test_client()
        response = client.get("/register")
        assert response.status_code == 200
        assert b"Create Account" in response.data
        assert b"Invite Code" not in response.data


def test_signup_requires_invite_and_stores_password_hash(app) -> None:
    app.config["SIGNUP_INVITE_CODE"] = "join-code"
    client = app.test_client()

    bad = client.post(
        "/register",
        data={
            "username": "newuser",
            "password": "password123",
            "confirm_password": "password123",
            "invite_code": "wrong",
        },
    )
    assert bad.status_code == 302
    assert User.query.filter_by(username="newuser").one_or_none() is None

    good = client.post(
        "/register",
        data={
            "username": "newuser",
            "password": "password123",
            "confirm_password": "password123",
            "invite_code": "join-code",
        },
    )
    assert good.status_code == 302
    assert good.location == "/setup-2fa"
    user = User.query.filter_by(username="newuser").one()
    assert user.password_hash != "password123"
    assert check_password_hash(user.password_hash, "password123")
    assert not user.two_factor_enabled


def test_managed_invite_code_binds_user_referral_percent(app) -> None:
    invite = ReferralInviteCode(code="VIP50", label="VIP", percent_profit=42.5, is_active=True)
    db.session.add(invite)
    db.session.commit()
    client = app.test_client()

    response = client.post(
        "/register",
        data={
            "username": "vipuser",
            "password": "password123",
            "confirm_password": "password123",
            "invite_code": "VIP50",
        },
    )

    assert response.status_code == 302
    user = User.query.filter_by(username="vipuser").one()
    db.session.refresh(invite)
    assert user.referral_invite_code_id == invite.id
    assert invite.usage_count == 1


def test_admin_invite_codes_page_renders_and_creates_code(app) -> None:
    admin, secret = _create_user(username="inviteadmin", role="admin")
    client = app.test_client()
    _login(client, admin.username, secret)

    page = client.get("/admin/invite-codes")
    assert page.status_code == 200
    assert b"Invite Codes" in page.data

    response = client.post(
        "/admin/invite-codes",
        data={
            "code": "PROFIT25",
            "label": "Partner",
            "percent_profit": "25",
            "max_uses": "10",
            "is_active": "on",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    invite = ReferralInviteCode.query.filter_by(code="PROFIT25").one()
    assert invite.percent_profit == 25.0
    assert invite.max_uses == 10


def test_login_without_2fa_redirects_to_setup(app) -> None:
    user = User(username="no2fa", password_hash=password_hash("password123"), role="user")
    db.session.add(user)
    db.session.commit()
    client = app.test_client()

    response = client.post("/login", data={"username": "no2fa", "password": "password123"})

    assert response.status_code == 302
    assert response.location == "/setup-2fa"


def test_2fa_setup_verifies_code_and_rejects_invalid(app) -> None:
    user = User(username="setup", password_hash=password_hash("password123"), role="user")
    db.session.add(user)
    db.session.commit()
    client = app.test_client()
    client.post("/login", data={"username": "setup", "password": "password123"})

    setup = client.get("/setup-2fa")
    assert setup.status_code == 200
    assert b"Manual Entry Secret" in setup.data
    secret = decrypt_totp_secret(user)
    assert secret

    invalid = client.post("/setup-2fa", data={"totp_code": "000000"})
    assert invalid.status_code == 200
    assert b"Invalid authenticator code" in invalid.data

    valid = client.post("/setup-2fa", data={"totp_code": pyotp.TOTP(secret).now()})
    assert valid.status_code == 302
    assert db.session.get(User, user.id).two_factor_enabled


def test_consumer_pages_render_wallet_and_vault_experience(app) -> None:
    _patch_market_data(app)
    _, secret = _create_user()
    client = app.test_client()
    _login(client, "alice", secret)

    home = client.get("/")
    wallet = client.get("/wallet")
    convert = client.get("/convert")
    vault = client.get("/vault")
    activity = client.get("/activity")
    settings = client.get("/settings/")

    assert home.status_code == 200
    assert b"Total Wallet Balance" in home.data
    assert b"Past Account P&amp;L" in home.data
    assert b"Automated Strategies" not in home.data
    assert b"Market Monitor" not in home.data
    assert b"Automation Activity" not in home.data
    assert b"static/js/command-center.js" in home.data
    for removed_copy in [
        b"Vault Pulse",
        b"Risk Notice",
        b"Start a Cycle",
        b"View Wallet",
        b"Locked Funds",
        b"Active Provider",
        b"No active vault cycle",
        b"Wallet Updates",
    ]:
        assert removed_copy not in home.data
    assert wallet.status_code == 200
    assert b"Verified On-chain Value" in wallet.data
    assert b"Balance Trend" in wallet.data
    assert b"Asset Mix" in wallet.data
    assert b"Deposit" in wallet.data
    assert b"Withdraw" in wallet.data
    assert b"Convert" in wallet.data
    assert b"Buy with Card" in wallet.data
    assert b"Buy with Apple Pay" in wallet.data
    assert b'data-onramp-method="apple_pay"' in wallet.data
    assert b"wallet-card-buy-data" in wallet.data
    assert b"wallet-apple-pay-data" in wallet.data
    assert b"wallet-card-buy-1" in wallet.data
    assert b"CARD_GATEWAY_API_KEY" not in wallet.data
    assert b"APPLE_PAY_GATEWAY_API_KEY" not in wallet.data
    assert b"Settlement Currency" not in wallet.data
    assert b"Exchange Margin" not in wallet.data
    assert b"Risk Notice" not in wallet.data
    assert b'data-bottom-nav-section="activity"' not in wallet.data
    assert convert.status_code == 200
    assert b"Convert Assets" in convert.data
    assert b"Move available wallet balances" in convert.data
    assert b"Convertible Assets" not in convert.data
    assert b"data-convert-form" in convert.data
    assert b"data-convert-offline" in convert.data
    assert b"data-convert-preview" in convert.data
    assert b"No convertible balance" in convert.data
    assert b"USDC" in convert.data
    assert b"XRP" in convert.data
    assert vault.status_code == 200
    assert b"Vault Cycle" in vault.data
    assert b"Start 1H10 Cycle" in vault.data
    assert b"Candidate Symbols" not in vault.data
    assert b"Operator Readiness" not in vault.data
    assert b"Vault risk notice" not in vault.data
    assert b"Algorithm Profile" not in vault.data
    assert b'data-bottom-nav-section="activity"' not in vault.data
    assert activity.status_code == 302
    assert activity.headers["Location"].endswith("/")
    assert settings.status_code == 200
    assert b"App Mode" in settings.data
    assert b"Providers" in settings.data
    for removed_copy in [
        b"Wallet Preferences",
        b"Wallet, vault cycles",
        b"Risk Notices",
        b"Address Mode",
        b"Live is disabled until",
    ]:
        assert removed_copy not in settings.data


def test_wallet_page_renders_when_live_sync_is_disabled(app) -> None:
    _patch_market_data(app)
    user, secret = _create_user(username="wallet-nosync")
    _seed_wallet_balance(user)
    failing_custody = _FailingSyncCustody()
    app.extensions["services"]["wallet_custody"] = failing_custody
    app.config["WALLET_PAGE_LIVE_SYNC_ENABLED"] = False
    client = app.test_client()
    _login(client, user.username, secret)

    response = client.get("/wallet")

    assert response.status_code == 200
    assert b"Verified On-chain Value" in response.data
    assert b"Buy with Card" in response.data
    assert failing_custody.sync_calls == 0


def test_activity_page_redirects_to_dashboard(app) -> None:
    _patch_market_data(app)
    user, secret = _create_user(username="activitypages")
    now = datetime.utcnow()
    for index in range(55):
        db.session.add(
            WalletTransaction(
                user_id=user.id,
                asset="USDT",
                amount=float(index),
                transaction_type="deposit",
                status="complete",
                note=f"wallet-note-{index}",
                created_at=now + timedelta(minutes=index),
            )
        )
        db.session.add(
            VaultCycle(
                user_id=user.id,
                deposit_asset="USDC",
                deposit_amount=float(index),
                settlement_asset="USDC",
                lock_duration_hours=1,
                lock_duration_seconds=3600,
                status="complete",
                execution_substatus="complete",
                algorithm_profile=f"CycleType{index}",
                started_at=now + timedelta(minutes=index),
                unlocks_at=now + timedelta(minutes=index, hours=1),
                settled_at=now + timedelta(minutes=index, hours=1),
                starting_value_usd=float(index),
                current_estimated_value_usd=float(index),
            )
        )
    db.session.commit()
    client = app.test_client()
    _login(client, user.username, secret)

    activity = client.get("/activity")
    activity_with_query = client.get("/activity?activity_page=2&cycle_page=2")

    assert activity.status_code == 302
    assert activity.headers["Location"].endswith("/")
    assert activity_with_query.status_code == 302
    assert activity_with_query.headers["Location"].endswith("/")
    assert WalletTransaction.query.filter_by(user_id=user.id).count() == 55
    assert VaultCycle.query.filter_by(user_id=user.id).count() == 55


def test_wallet_activity_is_paginated_and_retained_to_50_records(app) -> None:
    _patch_market_data(app)
    user, secret = _create_user(username="paginated")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=100.0, estimated_usd_value=100.0))
    now = datetime.utcnow()
    for index in range(55):
        db.session.add(
            WalletTransaction(
                user_id=user.id,
                asset="USDC",
                amount=float(index),
                transaction_type="deposit",
                status="complete",
                created_at=now + timedelta(minutes=index),
            )
        )
    db.session.commit()
    client = app.test_client()
    _login(client, user.username, secret)

    first_page = client.get("/wallet")

    assert first_page.status_code == 200
    assert WalletTransaction.query.filter_by(user_id=user.id).count() == 50
    assert b"54 USDC" in first_page.data
    assert b"50 USDC" in first_page.data
    assert b"49 USDC" not in first_page.data
    assert b"Page 1 of 10" in first_page.data

    second_page = client.get("/wallet?activity_page=2")

    assert second_page.status_code == 200
    assert b"49 USDC" in second_page.data
    assert b"54 USDC" not in second_page.data


def test_convert_page_converts_between_wallet_assets(app) -> None:
    _patch_market_data(app)
    user, secret = _create_user(username="convertwallet")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=100.0, estimated_usd_value=100.0))
    db.session.add(WalletBalance(user_id=user.id, asset="ETH", available_balance=0.0, estimated_usd_value=0.0))
    db.session.commit()
    client = app.test_client()
    _login(client, user.username, secret)

    page = client.get("/convert?from_asset=USDC&to_asset=ETH&amount=50")

    assert page.status_code == 200
    assert b"0.50000000 ETH" in page.data
    assert b"$50.00" in page.data
    assert b"Convert" in page.data
    assert b"data-convert-form" in page.data
    assert b"Ready to preview" not in page.data

    response = client.post(
        "/convert",
        data={"from_asset": "USDC", "to_asset": "ETH", "amount": "50"},
        follow_redirects=False,
    )

    assert response.status_code == 302
    confirmation = client.get(response.location)
    assert confirmation.status_code == 200
    assert b"Converted 50.00000000 USDC to 0.50000000 ETH." in confirmation.data
    usdc = WalletBalance.query.filter_by(user_id=user.id, asset="USDC").one()
    eth = WalletBalance.query.filter_by(user_id=user.id, asset="ETH").one()
    assert usdc.available_balance == pytest.approx(50.0)
    assert usdc.estimated_usd_value == pytest.approx(50.0)
    assert eth.available_balance == pytest.approx(0.5)
    assert eth.estimated_usd_value == pytest.approx(50.0)
    transactions = WalletTransaction.query.filter_by(user_id=user.id, transaction_type="conversion").order_by(WalletTransaction.asset).all()
    assert len(transactions) == 2
    assert [(tx.asset, tx.amount) for tx in transactions] == [("ETH", pytest.approx(0.5)), ("USDC", pytest.approx(-50.0))]


def test_convert_page_rejects_invalid_amounts(app) -> None:
    _patch_market_data(app)
    user, secret = _create_user(username="convertreject")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=25.0, estimated_usd_value=25.0))
    db.session.commit()
    client = app.test_client()
    _login(client, user.username, secret)

    response = client.post("/convert", data={"from_asset": "USDC", "to_asset": "ETH", "amount": "50"})

    assert response.status_code == 200
    assert b"Amount exceeds available USDC balance." in response.data
    assert b"Review required" in response.data
    assert b'aria-invalid="true"' in response.data
    assert WalletBalance.query.filter_by(user_id=user.id, asset="USDC").one().available_balance == 25.0
    assert WalletTransaction.query.filter_by(user_id=user.id, transaction_type="conversion").count() == 0


def test_convert_page_disables_empty_state_without_available_funds(app) -> None:
    _patch_market_data(app)
    user, secret = _create_user(username="convertempty")
    client = app.test_client()
    _login(client, user.username, secret)

    response = client.get("/convert")

    assert response.status_code == 200
    assert b"No convertible balance" in response.data
    assert b'data-can-convert="false"' in response.data
    assert b'disabled aria-disabled="true"' in response.data
    assert b"static/js/convert.js" in response.data
    assert b"Network unavailable" in response.data


def test_vault_cycles_are_paginated_and_retained_to_50_records(app) -> None:
    _patch_market_data(app)
    user, secret = _create_user(username="vaultpages")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=100.0, estimated_usd_value=100.0))
    now = datetime.utcnow()
    oldest_cycle_id = None
    oldest_leg_id = None
    for index in range(55):
        cycle = VaultCycle(
            user_id=user.id,
            deposit_asset="USDC",
            deposit_amount=float(index),
            settlement_asset="USDC",
            lock_duration_hours=1,
            lock_duration_seconds=3600,
            status="complete",
            execution_substatus="complete",
            algorithm_profile="1H10",
            started_at=now + timedelta(minutes=index),
            unlocks_at=now + timedelta(minutes=index, hours=1),
            settled_at=now + timedelta(minutes=index, hours=1),
            starting_value_usd=float(index),
            current_estimated_value_usd=float(index),
        )
        db.session.add(cycle)
        db.session.flush()
        leg = VaultAllocationLeg(vault_cycle_id=cycle.id, symbol="BTC", timeframe="1m", allocation_cap_usd=float(index))
        db.session.add(leg)
        db.session.flush()
        db.session.add(
            WalletTransaction(
                user_id=user.id,
                vault_cycle_id=cycle.id,
                asset="USDC",
                amount=float(index),
                transaction_type="settlement",
                status="complete",
                note=f"cycle-{index}",
            )
        )
        db.session.add(
            Order(
                user_id=user.id,
                vault_cycle_id=cycle.id,
                vault_leg_id=leg.id,
                client_order_id=f"cycle-retention-{index}",
                mode="live",
                symbol="BTC",
                side="buy",
                quantity=1.0,
                status="filled",
            )
        )
        if index == 0:
            oldest_cycle_id = cycle.id
            oldest_leg_id = leg.id
    db.session.commit()
    client = app.test_client()
    _login(client, user.username, secret)

    first_page = client.get("/vault")

    assert first_page.status_code == 200
    assert VaultCycle.query.filter_by(user_id=user.id).count() == 50
    assert VaultAllocationLeg.query.filter_by(vault_cycle_id=oldest_cycle_id).count() == 0
    assert WalletTransaction.query.filter_by(user_id=user.id, note="cycle-0").one().vault_cycle_id is None
    old_order = Order.query.filter_by(client_order_id="cycle-retention-0").one()
    assert old_order.vault_cycle_id is None
    assert old_order.vault_leg_id is None
    assert oldest_leg_id is not None
    assert b"54.000000 USDC" in first_page.data
    assert b"50.000000 USDC" in first_page.data
    assert b"49.000000 USDC" not in first_page.data
    assert b"Page 1 of 10" in first_page.data

    second_page = client.get("/vault?cycle_page=2")

    assert second_page.status_code == 200
    assert b"49.000000 USDC" in second_page.data
    assert b"54.000000 USDC" not in second_page.data


def test_admin_routes_require_login_and_2fa(app) -> None:
    client = app.test_client()
    assert client.get("/admin/panic").status_code == 404

    for path in ["/admin/dashboard", "/admin/backtests"]:
        response = client.get(path)
        assert response.status_code == 302
        assert "/login" in response.location

    for path in ["/admin/orders", "/admin/orders/", "/orders/", "/panic", "/panic/"]:
        response = client.get(path)
        assert response.status_code == 404

    admin = User(username="admin", password_hash=password_hash("password123"), role="admin")
    db.session.add(admin)
    db.session.commit()
    response = client.post("/login", data={"username": "admin", "password": "password123"})
    assert response.status_code == 302
    assert response.location == "/setup-2fa"
    blocked = client.get("/admin/dashboard")
    assert blocked.status_code == 302
    assert "/setup-2fa" in blocked.location


def test_admin_with_2fa_can_access_admin_and_user_cannot(app) -> None:
    _patch_market_data(app)
    _, admin_secret = _create_user(username="admin2", role="admin")
    _, user_secret = _create_user(username="regular", role="user")
    client = app.test_client()

    _login(client, "regular", user_secret)
    denied = client.get("/admin/dashboard")
    assert denied.status_code == 302
    assert denied.location == "/"
    client.post("/logout")

    _login(client, "admin2", admin_secret)
    dashboard = client.get("/admin/dashboard")
    backtests = client.get("/admin/backtests")
    risk = client.get("/admin/risk")
    strategies = client.get("/admin/strategies")
    readiness = client.get("/admin/live-readiness")

    assert dashboard.status_code == 200
    assert b"Automation Rankings" in dashboard.data
    assert b"Open Orders" in dashboard.data
    assert b"Manual Order Entry" not in dashboard.data
    assert b"Start Strategy" not in dashboard.data
    assert b"/admin/strategies/start" not in dashboard.data
    assert b"/admin/panic" not in dashboard.data
    assert b"Emergency Stop" not in dashboard.data
    assert client.get("/admin/orders").status_code == 404
    assert client.get("/admin/panic").status_code == 404
    assert backtests.status_code == 200
    assert b"Paper Vault" in backtests.data
    assert b"Vault backtesting" in backtests.data
    assert b"Vault Allocation Assets" in backtests.data
    assert b"Test Allocation Amount" in backtests.data
    assert b"Vault Cycle" in backtests.data
    assert b"Vault Cycle Duration" not in backtests.data
    assert b"Run Backtest" in backtests.data
    assert b"static/js/backtests.js" in backtests.data
    assert b"static/js/vendor/lightweight-charts.standalone.production.js" in backtests.data
    assert b'name="strategy_name"' not in backtests.data
    assert b'name="leverage"' not in backtests.data
    assert b'name="fee_bps"' not in backtests.data
    assert b'name="slippage_bps"' not in backtests.data
    assert b'name="stop_loss_pct"' not in backtests.data
    assert b'name="take_profit_pct"' not in backtests.data
    assert b'name="sizing_mode"' not in backtests.data
    assert b"Auto Universe" in backtests.data
    assert b"All enabled leveraged pairs" in backtests.data
    assert b"Auto-Optimized Metrics" not in backtests.data
    for removed_copy in [
        b"Short-Term Optimizer",
        b'value="dynamic_intraday"',
        b"Dynamic Intraday",
        b"Upside Scanner Diagnostics",
        b"High-upside profile",
        b"Latest Fibonacci Zones",
        b"Opportunity Lab",
        b"Saved Backtests",
        b"Optimizer rankings",
        b"Top Accepted",
        b"Top Rejected",
        b"Manual Strategy",
        b"Historical candle simulation only",
        b"Crypto trading carries significant risk",
    ]:
        assert removed_copy not in backtests.data
    assert risk.status_code == 200
    assert b"Risk Engine Controls" in risk.data
    assert b"Adaptive ML Slippage Engine" in risk.data
    assert b"static/js/risk.js" in risk.data
    assert b"audit-feed" in risk.data
    assert b"Audit Events" in risk.data
    assert b"Risk Status" not in risk.data
    assert b"Admin Shortcuts" not in risk.data
    assert b"Recent Risk Events" not in risk.data
    assert b"Max Notional" not in risk.data
    assert b"Max " + b"Slippage" not in risk.data
    assert b"/admin/orders" not in risk.data
    assert strategies.status_code == 200
    assert b"Strategy Diagnostics" in strategies.data
    assert b"Why Candidates Rank" in strategies.data
    assert readiness.status_code == 200
    assert b"Live Readiness" in readiness.data


def test_manual_trading_routes_are_not_registered(app) -> None:
    endpoints = {rule.endpoint for rule in app.url_map.iter_rules()}
    rules = {rule.rule for rule in app.url_map.iter_rules()}

    assert not any(endpoint.startswith("orders.") for endpoint in endpoints)
    assert "dashboard.start_strategy" not in endpoints
    assert "dashboard.stop_strategy" not in endpoints
    assert (
        not {
            "/admin/orders/",
            "/admin/orders",
            "/admin/orders/place",
            "/admin/orders/<int:order_id>/cancel",
            "/admin/strategies/start",
            "/admin/strategies/<int:run_id>/stop",
            "/orders/",
            "/orders/place",
            "/orders/<int:order_id>/cancel",
            "/strategies/start",
            "/strategies/<int:run_id>/stop",
        }
        & rules
    )


def test_admin_login_defaults_to_consumer_home_instead_of_heavy_dashboard(app) -> None:
    _, admin_secret = _create_user(username="admin-home", role="admin")
    client = app.test_client()

    response = _login(client, "admin-home", admin_secret)

    assert response.status_code == 302
    assert response.location == "/"


def test_removed_manual_trading_posts_return_404_without_service_calls(app, monkeypatch) -> None:
    def fail_order_submit(*args, **kwargs):
        raise AssertionError("manual order route called order_manager.place_order")

    def fail_strategy_start(*args, **kwargs):
        raise AssertionError("manual strategy start route called strategy_manager.start")

    def fail_strategy_stop(*args, **kwargs):
        raise AssertionError("manual strategy stop route called strategy_manager.stop")

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_order_submit)
    monkeypatch.setattr(app.extensions["services"]["strategy_manager"], "start", fail_strategy_start)
    monkeypatch.setattr(app.extensions["services"]["strategy_manager"], "stop", fail_strategy_stop)

    client = app.test_client()
    for path in [
        "/admin/orders/place",
        "/admin/orders/123/cancel",
        "/orders/place",
        "/orders/123/cancel",
        "/admin/strategies/start",
        "/admin/strategies/123/stop",
        "/strategies/start",
        "/strategies/123/stop",
    ]:
        response = client.post(path)
        assert response.status_code == 404


def test_backtests_json_run_uses_paper_allocation_and_returns_charts(app) -> None:
    _patch_market_data(app)
    admin, admin_secret = _create_user(username="backtestjson", role="admin")
    _seed_wallet_balance(admin, "USDC", 1_000.0)
    _seed_backtest_market()
    client = app.test_client()
    _login(client, "backtestjson", admin_secret)

    response = client.post(
        "/admin/backtests/run",
        data={
            "strategy_name": "ema_crossover",
            "symbol": "BTC",
            "timeframe": "live",
            "allocation_amount_usd": "500",
            "allocation_assets": "USDC",
            "cycle_duration": "1h10",
            "leverage": "99",
            "fee_bps": "999",
            "slippage_bps": "999",
            "stop_loss_pct": "0.99",
            "take_profit_pct": "9.99",
            "sizing_mode": "fixed_fraction",
        },
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["run_id"]
    assert {"metrics", "charts", "summary", "result", "autopilot", "asset_breakdown", "execution_quality"} <= set(payload)
    assert payload["execution_mode"] == "backtest"
    assert payload["simulation_scope"]["creates_backtest_run"] is True
    assert payload["simulation_scope"]["creates_vault_cycle"] is False
    assert payload["simulation_scope"]["starts_strategy_runs"] is False
    assert payload["simulation_scope"]["queues_worker"] is False
    assert payload["simulation_scope"]["submits_broker_order"] is False
    assert payload["trade_decision"]["stage"] == "simulated"
    assert payload["trade_decision"]["broker_order_submitted"] is False
    assert payload["autopilot"]["enabled"] is True
    assert payload["execution_quality"]["venue_count"] == 1
    assert payload["execution_quality"]["fee_bps"] != 999
    assert payload["execution_quality"]["slippage_bps"] != 999
    assert payload["result"]["target_multiplier"] == pytest.approx(10.0)
    assert payload["result"]["target_roi_pct"] == pytest.approx(1000.0)
    assert payload["result"]["objective_horizon_seconds"] == app.config["ONE_H10_HORIZON_SECONDS"]
    assert payload["result"]["hit_target"] in {True, False}
    assert payload["result"]["objective_gap_pct"] >= 0
    assert payload["target_progress"] >= 0
    assert payload["metrics"]["target_progress"] >= 0
    assert {"net_pnl", "closed_trades", "open_trades", "average_trade", "profit_factor"} <= set(payload["metrics"])
    assert payload["metrics"]["closed_trades"] >= 0
    assert payload["metrics"]["open_trades"] >= 0
    assert payload["portfolio_diagnostics"]["allocation_policy"] == "after_cost_pnl_weighted"
    assert "allocation_plan" in payload
    assert "skipped_candidates" in payload
    assert "asset_diagnostics" in payload
    assert "market_history_validation" in payload
    assert payload["result"]["screener_source"]
    assert payload["result"]["ml_families_used"]
    assert payload["asset_breakdown"]
    assert "allocation_weight" in payload["asset_breakdown"][0]
    assert "net_expected_return_bps" in payload["asset_breakdown"][0]
    assert "closed_trades" in payload["asset_breakdown"][0]
    assert "open_trades" in payload["asset_breakdown"][0]
    assert payload["metrics"]["ending_balance"] >= 0
    assert payload["charts"]["equity"]
    assert payload["charts"]["pnl"]
    assert payload["charts"]["drawdown"]
    assert payload["charts"]["growth"]
    assert "trade_timeline" in payload["charts"]
    assert payload["summary"]["allocation"] == 500.0
    assert payload["summary"]["allocation_assets"] == ["USDC"]
    assert payload["asset_diagnostics"][0]["status"] in {"simulated", "skipped"}
    assert payload["asset_diagnostics"][0]["market_history_validation"]["valid_candle_count"] >= 30
    assert payload["simulation_scope"]["creates_vault_cycle"] is False
    assert Order.query.count() == 0
    assert StrategyRun.query.count() == 0
    assert VaultCycle.query.count() == 0
    run = BacktestRun.query.one()
    assert run.strategy_name == "portfolio_vault_cycle_auto"
    assert run.timeframe == "1h10"
    assert run.parameters["initial_balance"] == 500.0
    assert run.parameters["allocation_amount_usd"] == 500.0
    assert run.parameters["allocation_assets"] == ["USDC"]
    assert run.parameters["parameters"]["simulated_capital_only"] is True
    assert run.parameters["parameters"]["execution_mode"] == "backtest"
    assert run.parameters["parameters"]["broker_order_submitted"] is False
    assert run.parameters["parameters"]["paper_balance_usd"] == 10_000.0
    assert run.parameters["parameters"]["one_h10_vault"] is True
    assert run.parameters["parameters"]["ml_horizon"] == "1h10"
    assert run.parameters["parameters"]["lock_duration_seconds"] == app.config["ONE_H10_HORIZON_SECONDS"]
    assert run.parameters["parameters"]["target_multiplier"] == pytest.approx(10.0)
    assert run.parameters["parameters"]["asset_breakdown"]
    assert run.parameters["parameters"]["allocation_plan"]
    assert "leverage" not in run.parameters
    assert "fee_bps" not in run.parameters
    assert run.result["autopilot"]["enabled"] is True


def test_backtests_after_cost_allocator_skips_higher_raw_return_high_cost_candidate(app, monkeypatch) -> None:
    admin, admin_secret = _create_user(username="aftercostadmin", role="admin")
    _seed_wallet_balance(admin, "USDC", 1_000.0)
    _seed_backtest_market(provider="hyperliquid", symbol="BTC", venue_symbol="BTC")
    _seed_backtest_market(provider="hyperliquid", symbol="ETH", venue_symbol="ETH")
    simulator = app.extensions["services"]["backtest_vault_simulator"]
    candidates = [
        {
            "provider": "hyperliquid",
            "provider_label": "Hyperliquid",
            "symbol": "BTC",
            "venue_symbol": "BTC",
            "vault_allocation_asset": "USDC",
            "liquidity_usd": 500_000.0,
            "screener_score": 4.0,
            "screener_source": "test",
            "screener_features": {
                "net_expected_return_bps": 80.0,
                "cost_drag_bps": 6.0,
                "expected_execution_quality": 0.92,
                "liquidity_usd": 500_000.0,
            },
        },
        {
            "provider": "hyperliquid",
            "provider_label": "Hyperliquid",
            "symbol": "ETH",
            "venue_symbol": "ETH",
            "vault_allocation_asset": "USDC",
            "liquidity_usd": 500_000.0,
            "screener_score": 5.0,
            "screener_source": "test",
            "screener_features": {
                "net_expected_return_bps": -12.0,
                "cost_drag_bps": 42.0,
                "expected_execution_quality": 0.35,
                "liquidity_usd": 500_000.0,
            },
        },
    ]

    def fake_result(row: dict[str, Any], *, allocation: float, cycle_duration_minutes: int, user: Any | None = None) -> dict[str, Any]:
        roi = 0.08 if row["symbol"] == "BTC" else 0.22
        return {
            "vault_simulation": True,
            "summary": {
                "symbol": row["symbol"],
                "provider": row["provider"],
                "provider_label": row["provider_label"],
                "vault_allocation_asset": "USDC",
                "allocation": allocation,
                "quote_asset": "USDC",
            },
            "metrics": {
                "roi": roi,
                "pnl": allocation * roi,
                "win_rate": 1.0,
                "max_drawdown": -0.02,
                "trades": 3,
                "fees": 1.0,
                "ending_balance": allocation * (1.0 + roi),
            },
            "charts": {
                "equity": [{"x": 1, "y": allocation}, {"x": 2, "y": allocation * (1.0 + roi)}],
                "pnl": [{"x": 1, "y": 0.0}, {"x": 2, "y": allocation * roi}],
                "drawdown": [{"x": 1, "y": 0.0}, {"x": 2, "y": -0.02}],
                "growth": [{"x": 1, "y": 0.0}, {"x": 2, "y": roi}],
                "trade_timeline": [],
            },
            "execution_quality": {
                "fee_bps": 5.0,
                "slippage_bps": 1.0,
                "fill_quality": row["screener_features"]["expected_execution_quality"],
                "liquidity_usd": row["liquidity_usd"],
            },
            "strategy_weights": [],
            "status": "simulated",
            "status_label": "Simulated",
            "screener_source": "test",
        }

    monkeypatch.setattr(simulator, "_rank_rows_for_one_h10", lambda rows, user=None: candidates)
    monkeypatch.setattr(simulator, "_simulate_asset_row", fake_result)
    client = app.test_client()
    _login(client, "aftercostadmin", admin_secret)

    response = client.post(
        "/admin/backtests/run",
        data={"allocation_amount_usd": "500", "allocation_assets": "USDC", "cycle_duration": "1h10"},
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert [row["asset"] for row in payload["allocation_plan"] if row["allocated"]] == ["BTC"]
    assert payload["allocation_plan"][0]["allocation_weight"] == pytest.approx(1.0)
    assert payload["asset_breakdown"][0]["asset"] == "BTC"
    assert payload["asset_breakdown"][0]["allocation_weight"] == pytest.approx(1.0)
    assert payload["skipped_candidates"][0]["asset"] == "ETH"
    assert payload["skipped_candidates"][0]["skip_reason"] == "after_cost_edge_below_threshold"


def test_backtest_strategy_weights_disable_negative_and_no_trade_strategies(app) -> None:
    simulator = app.extensions["services"]["backtest_vault_simulator"]

    weights = simulator._strategy_weights(
        [
            {
                "strategy_name": "strong_after_cost",
                "label": "Strong",
                "score": 0.44,
                "total_return": 0.05,
                "net_return_after_costs": 0.05,
                "max_drawdown": -0.02,
                "win_rate": 0.75,
                "trade_count": 4,
                "error": "",
            },
            {
                "strategy_name": "negative_after_cost",
                "label": "Negative",
                "score": 0.20,
                "total_return": -0.01,
                "net_return_after_costs": -0.01,
                "max_drawdown": -0.01,
                "win_rate": 0.40,
                "trade_count": 3,
                "error": "",
            },
            {
                "strategy_name": "no_trades",
                "label": "No Trades",
                "score": 0.30,
                "total_return": 0.02,
                "net_return_after_costs": 0.02,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "trade_count": 0,
                "error": "",
            },
        ]
    )

    by_name = {row["strategy_name"]: row for row in weights}
    assert by_name["strong_after_cost"]["enabled"] is True
    assert by_name["negative_after_cost"]["enabled"] is False
    assert by_name["negative_after_cost"]["disabled_reason"] == "negative_after_cost_return"
    assert by_name["no_trades"]["enabled"] is False
    assert by_name["no_trades"]["disabled_reason"] == "no_after_cost_trades"


def test_backtests_json_run_persists_multiple_vault_allocation_assets(app) -> None:
    _patch_market_data(app)
    admin, admin_secret = _create_user(username="backtestmultiasset", role="admin")
    _seed_wallet_balance(admin, "USDC", 400.0)
    _seed_wallet_balance(admin, "ETH", 3.0, estimated_usd=300.0)
    _seed_backtest_market(provider="hyperliquid", symbol="BTC", venue_symbol="BTC")
    _seed_backtest_market(provider="hyperliquid", symbol="ETH", venue_symbol="ETH")
    client = app.test_client()
    _login(client, "backtestmultiasset", admin_secret)

    response = client.post(
        "/admin/backtests/run",
        data={
            "allocation_amount_usd": "600",
            "allocation_assets": ["USDC", "ETH"],
            "cycle_duration": "1h10",
        },
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["summary"]["allocation_assets"] == ["USDC", "ETH"]
    run = BacktestRun.query.one()
    assert run.parameters["allocation_assets"] == ["USDC", "ETH"]
    assert run.parameters["parameters"]["selected_allocation_assets"] == ["USDC", "ETH"]


def test_backtests_continue_when_one_asset_has_insufficient_history(app) -> None:
    _patch_market_data(app)
    admin, admin_secret = _create_user(username="backtestpartialhistory", role="admin")
    _seed_wallet_balance(admin, "USDC", 1_000.0)
    _seed_backtest_market(provider="hyperliquid", symbol="BTC", venue_symbol="BTC")
    _seed_backtest_market(provider="hyperliquid", symbol="ETH", venue_symbol="ETH")
    market_data = app.extensions["services"]["market_data"]
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles()[:12] if symbol == "ETH" else _candles()
    client = app.test_client()
    _login(client, "backtestpartialhistory", admin_secret)

    response = client.post(
        "/admin/backtests/run",
        data={"allocation_amount_usd": "500", "allocation_assets": "USDC", "cycle_duration": "1h10"},
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    diagnostics = {row["asset"]: row for row in payload["asset_diagnostics"]}
    assert diagnostics["ETH"]["status"] == "insufficient_history"
    assert diagnostics["ETH"]["error_code"] == "insufficient_market_history"
    assert diagnostics["ETH"]["market_history_validation"]["valid_candle_count"] == 12
    assert "BTC" in diagnostics
    assert BacktestRun.query.count() == 1
    assert Order.query.count() == 0
    assert StrategyRun.query.count() == 0
    assert VaultCycle.query.count() == 0


def test_backtests_kucoin_uses_verified_connector_venue_symbol_for_candles(app, monkeypatch) -> None:
    _patch_market_data(app)
    admin, admin_secret = _create_user(username="backtestkucoinhistory", role="admin")
    _seed_wallet_balance(admin, "USDT", 1_000.0)
    client = app.test_client()
    _login(client, "backtestkucoinhistory", admin_secret)
    connection = app.extensions["services"]["trading_connections"].create_or_update(
        user_id=admin.id,
        provider="kucoin",
        connection_type="cex_api_key",
        api_key="kucoin-key",
        api_secret="kucoin-secret",
        passphrase="kucoin-passphrase",
        is_active=True,
    )
    connection.verification_status = "verified"
    connection.is_active = True
    market = _seed_backtest_market(provider="kucoin", symbol="BTC", venue_symbol="BTCUSDTM")
    market.trading_connection_id = connection.id
    db.session.commit()
    calls: list[tuple[str, str, str, int]] = []

    class FakeKucoinConnector:
        def get_candles(self, symbol: str, timeframe: str, mode: str, limit: int) -> list[dict[str, Any]]:
            calls.append((symbol, timeframe, mode, limit))
            return _candles()

    trading_connections = app.extensions["services"]["trading_connections"]
    original_connector_for_user = trading_connections.connector_for_user

    def connector_for_user(user_id: int, connection_id: int | None = None):
        if connection_id == connection.id:
            return FakeKucoinConnector()
        return original_connector_for_user(user_id, connection_id)

    monkeypatch.setattr(trading_connections, "connector_for_user", connector_for_user)
    monkeypatch.setattr(app.extensions["services"]["backtest_vault_simulator"], "_one_h10_screener_rows", lambda markets: [])
    app.extensions["services"]["market_data"].get_candles = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("KuCoin backtest should use the verified connector candle loader")
    )

    response = client.post(
        "/admin/backtests/run",
        data={"allocation_amount_usd": "500", "allocation_assets": "USDT", "cycle_duration": "1h10"},
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert calls
    assert calls[0][0] == "BTCUSDTM"
    assert payload["asset_diagnostics"][0]["market_history_validation"]["provider_source"] == "kucoin_connector"
    assert payload["summary"]["collateral_asset"] == "USDT"
    assert Order.query.count() == 0


def test_backtests_run_records_unavailable_asset_without_eligible_leveraged_pairs(app) -> None:
    _patch_market_data(app)
    admin, admin_secret = _create_user(username="backtestnomarkets", role="admin")
    _seed_wallet_balance(admin, "XRP", 500.0, estimated_usd=500.0)
    client = app.test_client()
    _login(client, "backtestnomarkets", admin_secret)

    response = client.post(
        "/admin/backtests/run",
        data={
            "allocation_amount_usd": "500",
            "allocation_assets": "XRP",
            "cycle_duration": "1h10",
        },
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["asset_breakdown"][0]["asset"] == "XRP"
    assert payload["asset_breakdown"][0]["status"] == "skipped"
    assert payload["asset_breakdown"][0]["error_code"] == "market_unavailable"
    assert "No active leveraged market" in payload["asset_breakdown"][0]["error"]
    assert payload["asset_diagnostics"][0]["status"] == "skipped"
    assert BacktestRun.query.count() == 1


def test_backtests_symbol_api_returns_paginated_exchange_universe(app) -> None:
    _patch_market_data(app)
    admin, admin_secret = _create_user(username="backtestsymbols", role="admin")
    _seed_wallet_balance(admin, "USDC", 750.0)
    db.session.add(
        LeveragedMarket(
            provider="kucoin",
            venue_symbol="BTCUSDTM",
            symbol="BTC",
            status="active",
            settlement_asset="USDT",
            max_leverage=20,
            liquidity_usd=250_000,
            spread_bps=2.5,
            fee_bps=6.0,
        )
    )
    db.session.add(
        LeveragedMarket(
            provider="hyperliquid",
            venue_symbol="ETH",
            symbol="ETH",
            status="active",
            settlement_asset="USDC",
            max_leverage=15,
            liquidity_usd=200_000,
            spread_bps=3.0,
            fee_bps=5.0,
        )
    )
    db.session.commit()
    client = app.test_client()
    _login(client, "backtestsymbols", admin_secret)
    connection = app.extensions["services"]["trading_connections"].create_or_update(
        user_id=admin.id,
        provider="kucoin",
        connection_type="cex_api_key",
        api_key="kucoin-key",
        api_secret="kucoin-secret",
        passphrase="kucoin-passphrase",
        is_active=True,
    )
    connection.verification_status = "verified"
    connection.is_active = True
    db.session.commit()

    response = client.get("/admin/backtests/api/symbols?q=btc&limit=1")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["count"] == 1
    assert payload["symbols"][0]["provider"] == "kucoin"
    assert payload["symbols"][0]["venue_symbol"] == "BTCUSDTM"
    assert payload["symbols"][0]["compatibility_badges"] == ["KuCoin", "USDT"]
    assert {"USDC", "USDT", "BTC", "ETH", "SOL", "XRP"} <= {row["asset"] for row in payload["allocation_assets"]}
    assert payload["allocation_cap_usd"] == 750.0
    assert payload["has_more"] is False


def test_backtests_symbol_api_caps_max_allocation_at_paper_balance_max(app) -> None:
    _patch_market_data(app)
    app.config["BACKTEST_PAPER_BALANCE_USD"] = 1_500_000.0
    app.config["PAPER_BALANCE_MAX"] = 1_000_000.0
    admin, admin_secret = _create_user(username="backtestmaxcap", role="admin")
    _seed_wallet_balance(admin, "USDC", 2_000_000.0)
    client = app.test_client()
    _login(client, "backtestmaxcap", admin_secret)

    response = client.get("/admin/backtests/api/symbols?limit=1")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["paper_balance_usd"] == 1_000_000.0
    assert payload["allocation_cap_usd"] == 1_000_000.0
    usdc = next(row for row in payload["allocation_assets"] if row["asset"] == "USDC")
    assert usdc["available_usd"] == 2_000_000.0


def test_backtests_quote_api_returns_precise_symbol_conversion(app) -> None:
    _patch_market_data(app)
    _, admin_secret = _create_user(username="backtestquote", role="admin")
    client = app.test_client()
    _login(client, "backtestquote", admin_secret)

    response = client.get("/admin/backtests/api/quote?provider=kucoin&symbol=BTC&venue_symbol=BTCUSDTM&allocation_usd=10000")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["allocation_usd"] == 10_000.0
    assert payload["mid"] == 100.0
    assert payload["asset_amount"] == pytest.approx(100.0)
    assert payload["asset_amount_formatted"] == "100.000000"
    assert payload["quote_asset"] == "USDT"
    assert payload["price_status"] == "priced"
    assert payload["price_source"] == "market_data"


def test_backtests_rejects_allocation_above_paper_balance(app) -> None:
    _patch_market_data(app)
    admin, admin_secret = _create_user(username="backtestcap", role="admin")
    _seed_wallet_balance(admin, "USDC", 20_000.0)
    client = app.test_client()
    _login(client, "backtestcap", admin_secret)

    response = client.post(
        "/admin/backtests/run",
        data={
            "strategy_name": "ema_crossover",
            "symbol": "BTC",
            "timeframe": "live",
            "allocation_amount_usd": "10000.01",
            "allocation_assets": "USDC",
            "cycle_duration": "1h10",
        },
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error_code"] == "invalid_backtest_input"
    assert "selected Vault allocation funds" in payload["error"]
    assert BacktestRun.query.count() == 0


def test_backtests_rejects_unknown_vault_allocation_asset(app) -> None:
    _patch_market_data(app)
    admin, admin_secret = _create_user(username="backtestbadasset", role="admin")
    _seed_wallet_balance(admin, "USDC", 500.0)
    client = app.test_client()
    _login(client, "backtestbadasset", admin_secret)

    response = client.post(
        "/admin/backtests/run",
        data={
            "allocation_amount_usd": "100",
            "allocation_assets": "DOGE",
            "cycle_duration": "1h10",
        },
        headers={"Accept": "application/json"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error_code"] == "invalid_backtest_input"
    assert payload["error"] == "Unsupported Vault allocation asset selected: DOGE."
    assert BacktestRun.query.count() == 0


def test_backtests_page_skips_optimizer_scanner_and_ml_services(app, monkeypatch) -> None:
    _, admin_secret = _create_user(username="backtestlean", role="admin")

    def fail_service_call(*args, **kwargs):
        raise AssertionError("backtests index loaded removed diagnostics")

    monkeypatch.setattr(app.extensions["services"]["market_universe"], "liquid_universe", fail_service_call)
    monkeypatch.setattr(app.extensions["services"]["market_scanner"], "score_candidates", fail_service_call)
    monkeypatch.setattr(app.extensions["services"]["feature_engine"], "snapshot", fail_service_call)
    monkeypatch.setattr(app.extensions["services"]["offline_ranker"], "readiness", fail_service_call)

    client = app.test_client()
    _login(client, "backtestlean", admin_secret)
    response = client.get("/admin/backtests")

    assert response.status_code == 200
    assert b"Paper Vault" in response.data
    assert b"Opportunity Lab" not in response.data
    assert b"Upside Scanner Diagnostics" not in response.data


def test_backtests_form_fallback_persists_without_javascript(app) -> None:
    _patch_market_data(app)
    admin, admin_secret = _create_user(username="backtestform", role="admin")
    _seed_wallet_balance(admin, "USDC", 1_000.0)
    _seed_backtest_market()
    client = app.test_client()
    _login(client, "backtestform", admin_secret)

    response = client.post(
        "/admin/backtests/run",
        data={
            "strategy_name": "ema_crossover",
            "symbol": "BTC",
            "timeframe": "5m",
            "allocation_amount_usd": "750",
            "allocation_assets": "USDC",
            "cycle_duration": "4h",
            "leverage": "99",
            "fee_bps": "999",
            "slippage_bps": "999",
            "stop_loss_pct": "0.99",
            "take_profit_pct": "9.99",
            "sizing_mode": "fixed_fraction",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Vault simulation completed." in response.data
    assert BacktestRun.query.count() == 1
    run = BacktestRun.query.one()
    assert run.strategy_name == "portfolio_vault_cycle_auto"
    assert run.parameters["initial_balance"] == 750.0
    assert run.parameters["allocation_assets"] == ["USDC"]
    assert run.parameters["parameters"]["vault_cycle_duration"] == "1h10"
    assert "leverage" not in run.parameters


def test_aggressive_optimizer_hidden_and_forced_submission_blocked(app) -> None:
    _, admin_secret = _create_user(username="optimizeradmin", role="admin")
    client = app.test_client()
    _login(client, "optimizeradmin", admin_secret)

    page = client.get("/admin/backtests")
    assert page.status_code == 200
    assert b"Aggressive 1H Experimental" not in page.data

    response = client.post(
        "/admin/backtests/optimize",
        data={"profile": "aggressive_1h", "strategy_name": "scalping", "symbols": "BTC", "timeframes": "1m"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Aggressive 1H Experimental optimization is disabled." in response.data
    assert OptimizerRun.query.count() == 0


def test_extreme_roi_optimizer_hidden_and_forced_submission_blocked(app) -> None:
    _, admin_secret = _create_user(username="extremeadmin", role="admin")
    client = app.test_client()
    _login(client, "extremeadmin", admin_secret)

    page = client.get("/admin/backtests")
    assert page.status_code == 200
    assert b"Extreme ROI Experimental" not in page.data

    response = client.post(
        "/admin/backtests/optimize",
        data={"profile": "extreme_roi_experimental", "strategy_name": "scalping", "symbols": "BTC", "timeframes": "1m"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Extreme ROI Experimental optimization is disabled." in response.data
    assert OptimizerRun.query.count() == 0


def test_backtests_page_hides_optimizer_internals_even_when_rankings_exist(app) -> None:
    app.config["AGGRESSIVE_1H_ENABLED"] = True
    _, admin_secret = _create_user(username="optimizeradmin2", role="admin")
    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    ranking = StrategyRanking(
        optimizer_run_id=optimizer_run.id,
        strategy_name="scalping",
        symbol="BTC",
        timeframe="1m",
        profile="aggressive_1h",
        experimental=True,
        risk_label="Very High Risk",
        score=1.25,
        total_return=0.04,
        net_return_after_costs=0.035,
        recent_performance_score=0.03,
        recent_1h_return=0.02,
        estimated_fees=1.25,
        edge_score=9.5,
        expectancy=1.75,
        cost_drag_bps=3.25,
        convex_edge_score=12.345,
        mfe_mae_ratio=2.5,
        capacity_multiple=3.0,
        cost_adjusted_recent_1h_return=0.018,
        decay_penalty=0.125,
        turnover_after_fees=1.2,
        window_stability=0.88,
        accepted_window_ratio=0.67,
        win_rate=0.58,
        no_trade_reason="low_edge_after_costs",
        max_drawdown=-0.08,
        profit_factor=1.4,
        trade_count=12,
        rejected=True,
        rejection_reason="low_trade_count",
    )
    ranking.warnings = ["Aggressive 1H mode is experimental and can lose capital quickly. Past backtests do not guarantee future returns."]
    db.session.add(ranking)
    db.session.commit()

    client = app.test_client()
    _login(client, "optimizeradmin2", admin_secret)
    page = client.get("/admin/backtests")

    assert page.status_code == 200
    assert b"Paper Vault" in page.data
    for removed_copy in [
        b"Aggressive 1H Experimental",
        b"Very High Risk",
        b"Aggressive Experiment Comparison",
        b"Profit Lab",
        b"Edge / Cost",
        b"Convex Edge",
        b"low_edge_after_costs",
        b"low_trade_count",
    ]:
        assert removed_copy not in page.data


def test_deposit_page_and_address_rotation(app) -> None:
    _patch_market_data(app)
    app.config["DEPOSIT_ADDRESS_BOOK"] = {
        "USDC": {
            "Ethereum": [
                "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            ]
        }
    }
    user, secret = _create_user()
    client = app.test_client()
    _login(client, user.username, secret)
    client.get("/wallet")

    deposit = client.get("/wallet/deposit/USDC")
    assert deposit.status_code == 200
    assert b"USDC Deposit Address" in deposit.data
    assert b"data:image/svg+xml;base64" in deposit.data
    old = DepositAddress.query.filter_by(user_id=user.id, asset="USDC", is_active=True).one()

    rotate = client.post("/wallet/rotate-address/USDC", data={"confirm_rotate": "on"})
    assert rotate.status_code == 302
    new = DepositAddress.query.filter_by(user_id=user.id, asset="USDC", is_active=True).one()
    assert new.id != old.id
    assert db.session.get(DepositAddress, old.id).is_active is False


def test_missing_deposit_address_fails_closed_without_placeholder(app) -> None:
    _patch_market_data(app)
    user, secret = _create_user(username="noaddr")
    client = app.test_client()
    _login(client, user.username, secret)

    deposit = client.get("/wallet/deposit/USDC")

    assert deposit.status_code == 200
    assert b"No deposit address configured" in deposit.data
    assert b"TEST-USDC" not in deposit.data
    assert b"data:image/svg+xml;base64" not in deposit.data
    assert DepositAddress.query.filter_by(user_id=user.id, asset="USDC").count() == 0


def test_configured_deposit_addresses_are_used_and_rotated(app) -> None:
    _patch_market_data(app)
    Setting.set_json("use_real_addresses", True)
    app.config["DEPOSIT_ADDRESS_BOOK"] = {
        "USDC": {
            "Ethereum": [
                "0x1111111111111111111111111111111111111111",
                "0x2222222222222222222222222222222222222222",
            ]
        }
    }
    db.session.commit()
    user, secret = _create_user(username="realaddr")
    client = app.test_client()
    _login(client, user.username, secret)

    deposit = client.get("/wallet/deposit/USDC")
    assert deposit.status_code == 200
    assert b"0x1111111111111111111111111111111111111111" in deposit.data

    rotate = client.post("/wallet/rotate-address/USDC", data={"confirm_rotate": "on"})
    assert rotate.status_code == 302
    active = DepositAddress.query.filter_by(user_id=user.id, asset="USDC", is_active=True).one()
    assert active.address == "0x2222222222222222222222222222222222222222"


def test_rotation_without_replacement_keeps_current_address_active(app) -> None:
    _patch_market_data(app)
    Setting.set_json("use_real_addresses", True)
    app.config["DEPOSIT_ADDRESS_BOOK"] = {"USDC": {"Ethereum": ["0x3333333333333333333333333333333333333333"]}}
    db.session.commit()
    user, secret = _create_user(username="onaddr")
    client = app.test_client()
    _login(client, user.username, secret)
    client.get("/wallet/deposit/USDC")
    old = DepositAddress.query.filter_by(user_id=user.id, asset="USDC", is_active=True).one()

    response = client.post("/wallet/rotate-address/USDC", data={"confirm_rotate": "on"}, follow_redirects=True)

    assert response.status_code == 200
    assert b"No replacement deposit address is configured for that asset/network." in response.data
    refreshed = db.session.get(DepositAddress, old.id)
    assert refreshed.is_active is True
    assert refreshed.expired_at is None
    assert DepositAddress.query.filter_by(user_id=user.id, asset="USDC").count() == 1


def test_generated_rotation_duplicate_replacement_fails_closed_without_500(app) -> None:
    _patch_market_data(app)
    _enable_live_wallets(app)
    user, secret = _create_user(username="dupaddr")
    client = app.test_client()
    _login(client, user.username, secret)

    deposit = client.get("/wallet/deposit/ETH")
    assert deposit.status_code == 200
    old = DepositAddress.query.filter_by(user_id=user.id, asset="ETH", is_active=True).one()

    response = client.post("/wallet/rotate-address/ETH", data={"confirm_rotate": "on"}, follow_redirects=True)

    assert response.status_code == 200
    assert b"No replacement deposit address is configured for that asset/network." in response.data
    refreshed = db.session.get(DepositAddress, old.id)
    assert refreshed.is_active is True
    assert refreshed.expired_at is None
    assert DepositAddress.query.filter_by(user_id=user.id, asset="ETH").count() == 1
    wallet_address = WalletAddress.query.filter_by(user_id=user.id, asset="ETH", address=old.address).one()
    assert wallet_address.status == "active"


def test_configured_deposit_lookup_is_case_insensitive(app) -> None:
    _patch_market_data(app)
    Setting.set_json("use_real_addresses", True)
    app.config["DEPOSIT_ADDRESS_BOOK"] = {"usdc": {"ethereum": ["0x4444444444444444444444444444444444444444"]}}
    db.session.commit()
    user, secret = _create_user(username="caseaddr")
    client = app.test_client()
    _login(client, user.username, secret)

    deposit = client.get("/wallet/deposit/USDC?network=Ethereum")

    assert deposit.status_code == 200
    assert b"0x4444444444444444444444444444444444444444" in deposit.data


def test_generated_live_deposit_address_page_creates_linked_wallet_record(app) -> None:
    _patch_market_data(app)
    _, custody = _enable_live_wallets(app)
    user, secret = _create_user(username="generateddeposit")
    client = app.test_client()
    _login(client, user.username, secret)

    deposit = client.get("/wallet/deposit/ETH")

    assert deposit.status_code == 200
    assert b"ETH Deposit Address" in deposit.data
    assert b"TEST-ETH" not in deposit.data
    address = DepositAddress.query.filter_by(user_id=user.id, asset="ETH", is_active=True).one()
    wallet_address = WalletAddress.query.filter_by(user_id=user.id, asset="ETH", address=address.address).one()
    assert wallet_address.deposit_address_id == address.id
    assert wallet_address.encrypted_metadata["custody"] == "in_app"
    assert custody.readiness()["real_custody_enabled"] is True


def test_wallet_models_do_not_store_custody_secrets(app) -> None:
    forbidden = {"private_key", "mnemonic", "seed", "xpub", "derivation"}
    model_columns = {
        column.name.lower() for model in (DepositAddress, WalletBalance, WalletTransaction) for column in model.__table__.columns
    }

    assert not any(any(term in column for term in forbidden) for column in model_columns)


def test_admin_can_view_deposit_address_history_but_consumer_cannot(app) -> None:
    _patch_market_data(app)
    app.config["DEPOSIT_ADDRESS_BOOK"] = {"USDC": {"Ethereum": ["0x5555555555555555555555555555555555555555"]}}
    user, user_secret = _create_user(username="consumer")
    _, admin_secret = _create_user(username="addradmin", role="admin")
    client = app.test_client()

    _login(client, user.username, user_secret)
    client.get("/wallet/deposit/USDC")
    consumer_history = client.get("/admin/deposit-addresses")
    assert consumer_history.status_code == 302
    client.post("/logout")

    _login(client, "addradmin", admin_secret)
    history = client.get("/admin/deposit-addresses")
    assert history.status_code == 200
    assert b"Deposit Address History" in history.data
    assert b"USDC" in history.data


def test_withdraw_rejects_invalid_2fa_and_over_balance_then_locks_valid_amount(app) -> None:
    _patch_market_data(app)
    _, custody = _enable_live_wallets(app)
    user, secret = _create_user()
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=1000.0, estimated_usd_value=1000.0))
    db.session.commit()
    custody.get_or_create_address(user_id=user.id, asset="USDC", network="Ethereum")
    db.session.commit()
    client = app.test_client()
    _login(client, user.username, secret)
    client.get("/wallet")

    over_balance = client.post(
        "/wallet/withdraw/USDC",
        data={
            "withdraw_address": "0x1111111111111111111111111111111111111111",
            "amount": "2000",
            "network": "Ethereum",
            "totp_code": pyotp.TOTP(secret).now(),
        },
    )
    assert over_balance.status_code == 200
    assert b"exceeds available balance" in over_balance.data

    invalid = client.post(
        "/wallet/withdraw/USDC",
        data={
            "withdraw_address": "0x1111111111111111111111111111111111111111",
            "amount": "10",
            "network": "Ethereum",
            "totp_code": "000000",
        },
    )
    assert invalid.status_code == 200
    assert b"Invalid authenticator code" in invalid.data

    available_before_valid = WalletBalance.query.filter_by(user_id=user.id, asset="USDC").one().available_balance
    valid = client.post(
        "/wallet/withdraw/USDC",
        data={
            "withdraw_address": "0x1111111111111111111111111111111111111111",
            "amount": "10",
            "network": "Ethereum",
            "totp_code": pyotp.TOTP(secret).now(),
        },
    )
    assert valid.status_code == 302
    tx = WalletTransaction.query.filter_by(user_id=user.id, transaction_type="withdrawal").one()
    withdrawal = WalletWithdrawal.query.filter_by(user_id=user.id).one()
    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDC").one()
    assert tx.status == "pending_approval"
    assert tx.withdraw_address == "0x1111111111111111111111111111111111111111"
    assert withdrawal.status == "pending_approval"
    assert withdrawal.workflow_type == "manual_withdrawal"
    assert balance.available_balance == available_before_valid - 10
    assert balance.locked_balance == 10


def test_wallet_page_does_not_render_exchange_margin_or_overwrite_custody_usdt_balance(app) -> None:
    _patch_market_data(app)
    user, secret = _create_user(username="custodyusdt")
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=10.0, estimated_usd_value=10.0))
    db.session.add(
        WalletLedgerEvent(
            user_id=user.id,
            asset="USDT",
            network="Ethereum",
            address="0x" + ("a" * 40),
            event_type="deposit",
            provider_reference="test-usdt-recovery",
            idempotency_key="deposit:test-usdt-recovery",
            amount=10.0,
            confirmations=12,
            status="complete",
        )
    )
    db.session.commit()
    client = app.test_client()
    _login(client, user.username, secret)
    app.extensions["services"]["trading_connections"].account_snapshot = lambda user_id, mode, connection_id=None: ClientSnapshot(
        mode,
        [{"asset": "USDT", "type": "margin", "value": 0.0000000065, "withdrawable": 0.0000000065}],
        [],
        [],
        [],
        [],
    )

    response = client.get("/wallet?refresh_exchange=1")

    assert response.status_code == 200
    assert b"Exchange Margin" not in response.data
    assert b"0.0000000065" not in response.data
    assert b"10.000000" in response.data
    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDT").one()
    assert balance.available_balance == 10.0
    snapshot = Setting.get_json(f"exchange_balance_snapshot:{user.id}", {})
    assert snapshot == {}


def test_live_wallet_withdrawal_requires_admin_approval_and_releases_on_reject(app) -> None:
    _patch_market_data(app)
    fake, custody = _enable_live_wallets(app)
    user, secret = _create_user(username="livewithdraw")
    admin, admin_secret = _create_user(username="walletadmin", role="admin")
    source = custody.get_or_create_address(user_id=user.id, asset="ETH", network="Ethereum")
    db.session.add(WalletBalance(user_id=user.id, asset="ETH", available_balance=2.0, locked_balance=0.0))
    db.session.commit()

    client = app.test_client()
    _login(client, user.username, secret)
    response = client.post(
        "/wallet/withdraw/ETH",
        data={
            "withdraw_address": "0x1111111111111111111111111111111111111111",
            "amount": "1",
            "network": "Ethereum",
            "totp_code": pyotp.TOTP(secret).now(),
        },
    )

    assert response.status_code == 302
    withdrawal = WalletWithdrawal.query.filter_by(user_id=user.id).one()
    assert withdrawal.status == "pending_approval"
    assert withdrawal.source_wallet_address_id is None
    assert WalletAddress.query.filter_by(id=source.id).one().encrypted_metadata["custody"] == "in_app"
    balance = WalletBalance.query.filter_by(user_id=user.id, asset="ETH").one()
    assert balance.available_balance == 3.0
    assert balance.locked_balance == 1.0
    tx = WalletTransaction.query.filter_by(user_id=user.id, transaction_type="withdrawal").one()
    assert tx.status == "pending_approval"

    client.post("/logout")
    _login(client, admin.username, admin_secret)
    listing = client.get("/admin/wallet-withdrawals")
    assert listing.status_code == 200
    assert b"Wallet Withdrawals" in listing.data

    approved = client.post(f"/admin/wallet-withdrawals/{withdrawal.id}/approve", follow_redirects=True)

    assert approved.status_code == 200
    db.session.refresh(withdrawal)
    db.session.refresh(tx)
    assert withdrawal.status == "submitted"
    assert withdrawal.provider_reference == "0xroutehash"
    assert tx.status == "pending_withdrawal"
    assert fake.broadcasts == 1
    db.session.refresh(balance)
    assert balance.available_balance == 3.0
    assert balance.locked_balance == 1.0

    balance.available_balance -= 0.25
    balance.locked_balance += 0.25
    pending_reject = app.extensions["services"]["self_custody_wallet"].create_manual_withdrawal(
        user_id=user.id,
        asset="ETH",
        network="Ethereum",
        destination_address="0x2222222222222222222222222222222222222222",
        amount=0.25,
    )
    db.session.add(
        WalletTransaction(
            user_id=user.id,
            asset="ETH",
            amount=0.25,
            transaction_type="withdrawal",
            status="pending_approval",
            network="Ethereum",
            withdraw_address="0x2222222222222222222222222222222222222222",
            note=f"Withdrawal workflow {pending_reject.id}: pending_approval.",
        )
    )
    db.session.commit()

    Setting.set_json("panic_lock", True)
    blocked = client.post(f"/admin/wallet-withdrawals/{pending_reject.id}/approve", follow_redirects=True)
    assert b"Panic lock is active" in blocked.data
    db.session.refresh(pending_reject)
    assert pending_reject.status == "pending_approval"

    Setting.set_json("panic_lock", False)
    rejected = client.post(f"/admin/wallet-withdrawals/{pending_reject.id}/reject", follow_redirects=True)

    assert rejected.status_code == 200
    db.session.refresh(pending_reject)
    db.session.refresh(balance)
    assert pending_reject.status == "rejected"
    assert balance.available_balance == 3.0
    assert balance.locked_balance == 1.0


def test_usdt_withdraw_max_uses_on_chain_token_balance(app) -> None:
    _patch_market_data(app)
    fake, custody = _enable_live_usdt_wallets(app)
    user, secret = _create_user(username="usdtmax")
    custody.get_or_create_address(user_id=user.id, asset="USDT", network="Ethereum")
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=7.0, locked_balance=0.0))
    db.session.commit()

    client = app.test_client()
    _login(client, user.username, secret)
    form = client.get("/wallet/withdraw/USDT?network=Ethereum&max=1")
    assert form.status_code == 200
    assert b"Withdraw USDT" in form.data
    assert b"Withdraw max 10.000000 USDT" in form.data

    response = client.post(
        "/wallet/withdraw/USDT",
        data={
            "withdraw_address": "0x0eA336f8CFD67Ee22EeaF8198BB287A953c04761",
            "amount": "0",
            "withdraw_max": "1",
            "network": "Ethereum",
            "totp_code": pyotp.TOTP(secret).now(),
        },
    )

    assert response.status_code == 302
    withdrawal = WalletWithdrawal.query.filter_by(user_id=user.id).one()
    assert withdrawal.amount == 10.0
    assert withdrawal.destination_address == "0x0eA336f8CFD67Ee22EeaF8198BB287A953c04761"
    assert withdrawal.status == "submitted"
    assert fake.broadcasts == 1
    tx = WalletTransaction.query.filter_by(user_id=user.id, transaction_type="withdrawal").one()
    assert tx.amount == 10.0
    assert tx.status == "pending_withdrawal"


def test_withdrawal_materializes_verified_onchain_surplus_before_locking(app) -> None:
    _patch_market_data(app)
    fake, custody = _enable_live_usdt_wallets(app)
    user, secret = _create_user(username="usdtmaterialize")
    wallet_address = custody.get_or_create_address(user_id=user.id, asset="USDT", network="Ethereum")
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=7.0, locked_balance=0.0, estimated_usd_value=7.0))
    db.session.add(
        WalletLedgerEvent(
            user_id=user.id,
            wallet_address_id=wallet_address.id,
            asset="USDT",
            network="Ethereum",
            address=wallet_address.address,
            event_type="deposit",
            provider_reference="historical-10",
            idempotency_key="deposit:historical-10",
            amount=10.0,
            confirmations=12,
            status="complete",
        )
    )
    db.session.commit()

    client = app.test_client()
    _login(client, user.username, secret)
    response = client.post(
        "/wallet/withdraw/USDT",
        data={
            "withdraw_address": "0x0eA336f8CFD67Ee22EeaF8198BB287A953c04761",
            "amount": "10",
            "network": "Ethereum",
            "totp_code": pyotp.TOTP(secret).now(),
        },
    )

    assert response.status_code == 302
    assert fake.broadcasts == 1
    withdrawal = WalletWithdrawal.query.filter_by(user_id=user.id).one()
    assert withdrawal.amount == 10.0
    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDT").one()
    assert balance.available_balance == pytest.approx(0.0)
    assert balance.locked_balance == pytest.approx(10.0)
    reconciliation = WalletTransaction.query.filter_by(user_id=user.id, transaction_type="onchain_reconciliation").one()
    assert reconciliation.amount == pytest.approx(3.0)


def test_admin_platform_treasury_page_renders(app) -> None:
    _patch_market_data(app)
    admin, secret = _create_user(username="treasuryadmin", role="admin")
    client = app.test_client()
    _login(client, admin.username, secret)

    response = client.get("/admin/platform-treasury")

    assert response.status_code == 200
    assert b"Platform Treasury" in response.data
    assert b"Reserve" in response.data
    assert b"gas sponsorship" in response.data


def test_vault_cycle_start_locks_wallet_and_links_strategy(app) -> None:
    _patch_market_data(app)
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    user, secret = _create_user()
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=1000.0, estimated_usd_value=1000.0))
    db.session.commit()
    client = app.test_client()
    _login(client, user.username, secret)
    client.get("/wallet")

    response = client.post(
        "/vault/start",
        data={
            "deposit_amount": "125",
            "deposit_asset": "USDC",
            "lock_duration": "24",
            "settlement_asset": "USDC",
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    cycle = VaultCycle.query.filter_by(user_id=user.id).one()
    transaction = WalletTransaction.query.filter_by(user_id=user.id, transaction_type="allocation").one()
    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDC").one()
    run = db.session.get(StrategyRun, cycle.strategy_run_id)

    assert cycle.status == "active"
    assert cycle.algorithm_profile == "Conservative"
    assert cycle.selected_strategy_name == "mean_reversion"
    assert cycle.selected_timeframe == "5m"
    assert cycle.starting_value_usd == 125
    assert transaction.amount == 125
    assert balance.available_balance == 875
    assert balance.locked_balance == 125
    assert run is not None
    assert run.parameters["allocation_cap_usd"] == 125
    assert run.parameters["vault_cycle_id"] == cycle.id


def test_vault_cycle_defaults_settlement_to_allocation_asset(app) -> None:
    _patch_market_data(app)
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    user, secret = _create_user(username="autosettle")
    db.session.add(WalletBalance(user_id=user.id, asset="ETH", available_balance=10.0, estimated_usd_value=1000.0))
    db.session.commit()
    client = app.test_client()
    _login(client, user.username, secret)
    client.get("/wallet")

    response = client.post(
        "/vault/start",
        data={
            "deposit_amount": "1",
            "deposit_asset": "ETH",
            "lock_duration": "24",
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    cycle = VaultCycle.query.filter_by(user_id=user.id).one()
    run = db.session.get(StrategyRun, cycle.strategy_run_id)
    assert cycle.deposit_asset == "ETH"
    assert cycle.settlement_asset == "ETH"
    assert run.parameters["settlement_asset"] == "ETH"


def test_one_hour_vault_cycle_uses_short_horizon_strategy(app, monkeypatch) -> None:
    _patch_market_data(app)
    _patch_deep_book(app)
    _confirm_one_h10_live(app)
    monkeypatch.setitem(app.extensions["services"], "one_h10_forecast", _PassingOneH10Forecast())
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    user, secret = _create_user(username="onehour")
    db.session.add_all(
        [
            LeveragedMarket(
                provider="hyperliquid",
                venue_symbol="BTC",
                symbol="BTC",
                status="active",
                settlement_asset="USDC",
                max_leverage=50,
                liquidity_usd=250_000,
                spread_bps=2.0,
                fee_bps=5.0,
            ),
            LeveragedMarket(
                provider="hyperliquid",
                venue_symbol="ETH",
                symbol="ETH",
                status="active",
                settlement_asset="USDC",
                max_leverage=25,
                liquidity_usd=200_000,
                spread_bps=2.0,
                fee_bps=5.0,
            ),
            LeveragedMarket(
                provider="hyperliquid",
                venue_symbol="SOL",
                symbol="SOL",
                status="active",
                settlement_asset="USDC",
                max_leverage=20,
                liquidity_usd=150_000,
                spread_bps=2.0,
                fee_bps=5.0,
            ),
        ]
    )
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=1000.0, estimated_usd_value=1000.0))
    db.session.commit()
    client = app.test_client()
    _login(client, user.username, secret)
    client.get("/wallet")

    response = client.post(
        "/vault/start",
        data={
            "deposit_amount": "50",
            "deposit_asset": "USDC",
            "lock_duration": "1",
            "settlement_asset": "USDC",
            "one_h10_live_ack": "on",
        },
    )

    assert response.status_code == 302
    cycle = VaultCycle.query.filter_by(user_id=user.id).one()
    run = db.session.get(StrategyRun, cycle.strategy_run_id)
    assert cycle.lock_duration_hours == 1
    assert cycle.algorithm_profile == "1H10"
    assert cycle.selected_strategy_name == "scalping"
    assert cycle.selected_timeframe == "1m"
    legs = VaultAllocationLeg.query.filter_by(vault_cycle_id=cycle.id).all()
    assert legs
    assert sum(float(leg.allocation_cap_usd or 0.0) for leg in legs) == pytest.approx(50.0)
    assert run.parameters["allocation_cap_usd"] <= 50
    assert run.parameters["one_h10_all_pairs"] is True
    assert run.parameters["ml_horizon"] == "1h10"


def test_vault_estimated_value_reflects_linked_strategy_pnl(app) -> None:
    _patch_market_data(app)
    user, secret = _create_user(username="pnluser")
    client = app.test_client()
    _login(client, user.username, secret)
    run = StrategyRun(user_id=user.id, strategy_name="mean_reversion", symbol="BTC", timeframe="5m", mode="live", status="running")
    db.session.add(run)
    db.session.flush()
    cycle = VaultCycle(
        user_id=user.id,
        strategy_run_id=run.id,
        deposit_asset="USDC",
        deposit_amount=100,
        settlement_asset="USDC",
        lock_duration_hours=24,
        status="active",
        execution_substatus="executing",
        algorithm_profile="Conservative",
        selected_strategy_name="mean_reversion",
        selected_timeframe="5m",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=24),
        starting_value_usd=100,
        current_estimated_value_usd=100,
    )
    db.session.add(cycle)
    db.session.flush()
    run.parameters = {"consumer_vault": True, "vault_cycle_id": cycle.id}
    order = Order(
        user_id=user.id,
        client_order_id="vault-pnl-order",
        mode="live",
        symbol="BTC",
        side="sell",
        order_type="market",
        status="filled",
        strategy_name="mean_reversion",
        quantity=1.0,
        filled_quantity=1.0,
        average_fill_price=105,
        reduce_only=True,
    )
    order.details = {"vault_cycle_id": cycle.id}
    db.session.add(order)
    db.session.flush()
    db.session.add(Fill(order_id=order.id, symbol="BTC", side="sell", quantity=1.0, price=105, fee=0.25, pnl=5.0))
    other_order = Order(
        user_id=user.id,
        client_order_id="other-vault-pnl-order",
        mode="live",
        symbol="BTC",
        side="sell",
        order_type="market",
        status="filled",
        strategy_name="mean_reversion",
        quantity=1.0,
        filled_quantity=1.0,
        average_fill_price=120,
        reduce_only=True,
    )
    other_order.details = {"vault_cycle_id": cycle.id + 999}
    db.session.add(other_order)
    db.session.flush()
    db.session.add(Fill(order_id=other_order.id, symbol="BTC", side="sell", quantity=1.0, price=120, fee=0.0, pnl=100.0))
    db.session.commit()

    response = client.get("/vault")
    assert response.status_code == 200
    refreshed = db.session.get(VaultCycle, cycle.id)
    assert refreshed.current_estimated_value_usd == 104.75
    assert refreshed.selection_metadata["total_pnl_usd"] == 4.75
    assert b"$4.75" in response.data


def test_vault_strategy_selector_aggressive_and_limited_market_conditions(app) -> None:
    _patch_market_data(app)
    _patch_deep_book(app)
    selector = app.extensions["services"]["vault_strategy_selector"]

    aggressive = selector.select("BTC", 168, "paper")
    assert aggressive.profile == "Aggressive"
    assert aggressive.strategy_name == "volatility_breakout"
    assert aggressive.parameters["risk_fraction"] <= app.config["VAULT_MAX_RISK_FRACTION"]
    assert aggressive.parameters["risk_fraction"] > app.config["RISK_PER_TRADE_PCT"]

    _patch_deep_book(app, spread=2.0, size="1")
    limited = selector.select("BTC", 168, "paper")
    assert limited.profile == "Conservative"
    assert any("limited execution" in reason for reason in limited.metadata["selection_reasons"])


def test_vault_selector_uses_only_aggressive_compatible_rankings_for_one_hour(app) -> None:
    _patch_market_data(app)
    _patch_deep_book(app)
    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    rejected_shape = StrategyRanking(
        optimizer_run_id=optimizer_run.id,
        strategy_name="mean_reversion",
        symbol="BTC",
        timeframe="5m",
        profile="short_term",
        experimental=False,
        score=10.0,
        rejected=False,
        profit_factor=2.0,
        trade_count=50,
    )
    aggressive = StrategyRanking(
        optimizer_run_id=optimizer_run.id,
        strategy_name="scalping",
        symbol="BTC",
        timeframe="1m",
        profile="aggressive_1h",
        experimental=True,
        risk_label="Very High Risk",
        score=2.0,
        rejected=False,
        recent_1h_return=0.03,
        max_drawdown=-0.05,
        profit_factor=1.4,
        trade_count=12,
    )
    aggressive.parameters = {"risk_fraction": 0.02, "stop_loss_pct": 0.003, "take_profit_pct": 0.006}
    db.session.add_all([rejected_shape, aggressive])
    db.session.commit()

    selection = app.extensions["services"]["vault_strategy_selector"].select("BTC", 1, "paper")

    assert selection.strategy_name == "scalping"
    assert selection.timeframe == "1m"
    assert selection.metadata["optimizer_profile"] == "aggressive_1h"
    assert selection.metadata["optimizer_recent_1h_return"] == 0.03


def test_live_vault_cycle_starts_with_active_connection(app) -> None:
    _patch_market_data(app)
    _patch_deep_book(app)
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    Setting.set_json("current_mode", "live")
    db.session.commit()
    user, secret = _create_user()
    connection = _create_live_connection(app, user)
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=1000.0, estimated_usd_value=1000.0))
    db.session.commit()
    client = app.test_client()
    _login(client, user.username, secret)
    client.get("/wallet")

    response = client.post(
        "/vault/start",
        data={
            "deposit_amount": "100",
            "deposit_asset": "USDC",
            "lock_duration": "168",
            "settlement_asset": "USDC",
        },
    )

    assert response.status_code == 302
    cycle = VaultCycle.query.filter_by(user_id=user.id).one()
    run = db.session.get(StrategyRun, cycle.strategy_run_id)
    assert cycle.trading_connection_id == connection.id
    assert run.trading_connection_id == connection.id
    assert cycle.execution_substatus == "executing"
    assert cycle.live_validation_status == "not_required"
    assert cycle.execution_mode == "live"
    assert run.mode == "live"
    assert run.parameters["live_validation_status"] == "not_required"

    vault = client.get("/vault")
    assert b"Executing" in vault.data
    assert b"Algorithm Profile" not in vault.data
    assert b"volatility_breakout" not in vault.data


def test_live_vault_cycle_blocks_when_exchange_ip_check_fails(app) -> None:
    _patch_market_data(app)
    _patch_deep_book(app)
    _confirm_one_h10_live(app)
    app.extensions["services"]["strategy_manager"].start = lambda run_id: (_ for _ in ()).throw(AssertionError("strategy must not start"))
    user, secret = _create_user(username="ipblocked")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=1000.0, estimated_usd_value=1000.0))
    db.session.commit()
    client = app.test_client()
    _login(client, user.username, secret)
    connection = app.extensions["services"]["trading_connections"].active_tradable_connection(user.id)
    failure = '{"code":"400006","msg":"Invalid request ip, the current clientIp is:209.52.132.232"}'
    app.extensions["services"]["trading_connections"].account_snapshot = lambda user_id, mode, connection_id=None: ClientSnapshot(
        mode,
        [],
        [],
        [],
        [],
        [failure],
    )

    response = client.post(
        "/vault/start",
        data={
            "deposit_amount": "100",
            "deposit_asset": "USDC",
            "lock_duration": "1",
            "settlement_asset": "USDC",
            "one_h10_live_ack": "on",
        },
        follow_redirects=True,
    )

    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDC").one()
    health = Setting.get_json(f"connection_health:{connection.id}", {})
    assert response.status_code == 200
    assert VaultCycle.query.filter_by(user_id=user.id).count() == 0
    assert balance.available_balance == 1000.0
    assert balance.locked_balance == 0.0
    assert health["can_trade"] is False
    assert health["provider_code"] == "400006"
    assert health["client_ip"] == "209.52.132.232"
    assert b"Whitelist current client IP 209.52.132.232" in response.data


def test_live_vault_cycle_uses_recent_timeout_backoff_before_snapshot(app) -> None:
    _patch_market_data(app)
    _patch_deep_book(app)
    _confirm_one_h10_live(app)
    app.extensions["services"]["strategy_manager"].start = lambda run_id: (_ for _ in ()).throw(AssertionError("strategy must not start"))
    user, secret = _create_user(username="timeoutbackoff")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=1000.0, estimated_usd_value=1000.0))
    db.session.commit()
    client = app.test_client()
    _login(client, user.username, secret)
    connection = app.extensions["services"]["trading_connections"].active_tradable_connection(user.id)
    Setting.set_json(
        f"connection_health:{connection.id}",
        {
            "connection_id": connection.id,
            "provider": connection.provider,
            "mode": "live",
            "last_checked_at": datetime.utcnow().isoformat() + "Z",
            "can_trade": False,
            "alerts": ["Read timed out."],
            "failure_reason": "HTTPSConnectionPool(host='api.hyperliquid.xyz', port=443): Read timed out. (read timeout=10.0)",
            "transient_failure": True,
            "failure_category": "network_timeout",
        },
    )
    db.session.commit()
    app.extensions["services"]["trading_connections"].account_snapshot = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("fresh snapshot must wait for backoff")
    )

    response = client.post(
        "/vault/start",
        data={
            "deposit_amount": "100",
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
    assert balance.available_balance == 1000.0
    assert balance.locked_balance == 0.0
    assert b"temporarily unavailable" in response.data


def test_vault_cycle_keeps_small_available_reserve_when_asset_locked(app) -> None:
    _patch_market_data(app)
    _confirm_one_h10_live(app)
    app.extensions["services"]["strategy_manager"].start = lambda run_id: (_ for _ in ()).throw(AssertionError("strategy must not start"))
    user, secret = _create_user(username="smallreserve")
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=5.0, locked_balance=5.0, estimated_usd_value=10.0))
    db.session.commit()
    client = app.test_client()
    _login(client, user.username, secret)

    response = client.post(
        "/vault/start",
        data={
            "deposit_amount": "5",
            "deposit_asset": "USDT",
            "lock_duration": "1",
            "settlement_asset": "USDT",
            "one_h10_live_ack": "on",
        },
        follow_redirects=True,
    )

    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDT").one()
    assert response.status_code == 200
    assert VaultCycle.query.filter_by(user_id=user.id).count() == 0
    assert balance.available_balance == 5.0
    assert balance.locked_balance == 5.0
    assert b"Keep at least $5.00 available" in response.data


def test_no_order_cycle_failure_is_visible_in_vault_and_detail(app) -> None:
    _patch_market_data(app)
    user, secret = _create_user(username="visiblefail")
    client = app.test_client()
    _login(client, user.username, secret)
    run = StrategyRun(
        user_id=user.id,
        strategy_name="scalping",
        symbol="BTC",
        timeframe="1m",
        mode="live",
        status="error",
        last_error='{"code":"400006","msg":"Invalid request ip, the current clientIp is:209.52.132.232"}',
    )
    db.session.add(run)
    db.session.flush()
    cycle = VaultCycle(
        user_id=user.id,
        strategy_run_id=run.id,
        deposit_asset="USDC",
        deposit_amount=5.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        status="active",
        execution_substatus="limited",
        execution_mode="live",
        validation_failure_reason=run.last_error,
        algorithm_profile="Aggressive",
        selected_strategy_name="scalping",
        selected_timeframe="1m",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=5.0,
        current_estimated_value_usd=5.0,
    )
    db.session.add(cycle)
    db.session.flush()
    run.parameters = {"consumer_vault": True, "vault_cycle_id": cycle.id}
    db.session.add(
        VaultAllocationLeg(vault_cycle_id=cycle.id, strategy_run_id=run.id, symbol="BTC", timeframe="1m", allocation_cap_usd=5.0)
    )
    db.session.commit()

    vault = client.get("/vault")
    detail = client.get(f"/vault/cycles/{cycle.id}")

    assert b"No live order submitted" not in vault.data
    assert b"No live order submitted" not in detail.data
    assert b"No live orders" in detail.data
    assert b"Invalid request ip" not in detail.data
    assert b"flask repair-limited-cycle --cycle-id" not in vault.data
    assert b"flask repair-limited-cycle --cycle-id" not in detail.data
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


def test_repair_limited_cycle_releases_no_order_failed_cycle(app) -> None:
    user, _ = _create_user(username="repairable")
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=5.0, locked_balance=5.0, estimated_usd_value=10.0))
    run = StrategyRun(
        user_id=user.id,
        strategy_name="scalping",
        symbol="BTC",
        timeframe="1m",
        mode="live",
        status="error",
        last_error='{"code":"400006","msg":"Invalid request ip, the current clientIp is:209.52.132.232"}',
    )
    db.session.add(run)
    db.session.flush()
    cycle = VaultCycle(
        user_id=user.id,
        strategy_run_id=run.id,
        deposit_asset="USDT",
        deposit_amount=5.0,
        settlement_asset="USDT",
        lock_duration_hours=1,
        status="active",
        execution_substatus="limited",
        execution_mode="live",
        validation_failure_reason=run.last_error,
        algorithm_profile="Aggressive",
        selected_strategy_name="scalping",
        selected_timeframe="1m",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=5.0,
        current_estimated_value_usd=5.0,
    )
    db.session.add(cycle)
    db.session.flush()
    run.parameters = {"consumer_vault": True, "vault_cycle_id": cycle.id}
    db.session.add(
        VaultAllocationLeg(vault_cycle_id=cycle.id, strategy_run_id=run.id, symbol="BTC", timeframe="1m", allocation_cap_usd=5.0)
    )
    db.session.commit()

    preview = app.test_cli_runner().invoke(args=["repair-limited-cycle", "--cycle-id", str(cycle.id)])
    assert preview.exit_code == 0
    assert json.loads(preview.output)["preview_only"] is True
    wrong = app.test_cli_runner().invoke(args=["repair-limited-cycle", "--cycle-id", str(cycle.id), "--confirm", "NOPE"])
    assert wrong.exit_code == 1
    assert "confirmation_required" in json.loads(wrong.output)["blockers"]

    result = app.test_cli_runner().invoke(args=["repair-limited-cycle", "--cycle-id", str(cycle.id), "--confirm", "REPAIR-LIMITED-CYCLE"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    db.session.refresh(cycle)
    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDT").one()
    assert payload["repaired"] is True
    assert cycle.status == "complete"
    assert cycle.execution_substatus == "failed_no_execution"
    assert cycle.final_settlement_amount == 5.0
    assert balance.available_balance == 10.0
    assert balance.locked_balance == 0.0
    assert WalletTransaction.query.filter_by(user_id=user.id, vault_cycle_id=cycle.id, transaction_type="settlement").count() == 1
    assert AuditLog.query.filter_by(action="repair_limited_cycle").count() == 1

    again = app.test_cli_runner().invoke(args=["repair-limited-cycle", "--cycle-id", str(cycle.id), "--confirm", "REPAIR-LIMITED-CYCLE"])
    assert again.exit_code == 0
    assert json.loads(again.output)["already_repaired"] is True
    db.session.refresh(balance)
    assert balance.available_balance == 10.0
    assert WalletTransaction.query.filter_by(user_id=user.id, vault_cycle_id=cycle.id, transaction_type="settlement").count() == 1


def test_repair_limited_cycle_refuses_cycles_with_orders(app) -> None:
    user, _ = _create_user(username="orderedcycle")
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=5.0, locked_balance=5.0, estimated_usd_value=10.0))
    run = StrategyRun(
        user_id=user.id,
        strategy_name="scalping",
        symbol="BTC",
        timeframe="1m",
        mode="live",
        status="error",
        last_error="Invalid request ip",
    )
    db.session.add(run)
    db.session.flush()
    cycle = VaultCycle(
        user_id=user.id,
        strategy_run_id=run.id,
        deposit_asset="USDT",
        deposit_amount=5.0,
        settlement_asset="USDT",
        lock_duration_hours=1,
        status="active",
        execution_substatus="limited",
        execution_mode="live",
        validation_failure_reason="Invalid request ip",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=5.0,
        current_estimated_value_usd=5.0,
    )
    db.session.add(cycle)
    db.session.flush()
    order = Order(
        user_id=user.id,
        client_order_id="repair-block-order",
        mode="live",
        symbol="BTC",
        side="buy",
        order_type="market",
        status="submitted",
        quantity=0.001,
    )
    order.details = {"vault_cycle_id": cycle.id}
    db.session.add(order)
    db.session.commit()

    result = app.test_cli_runner().invoke(args=["repair-limited-cycle", "--cycle-id", str(cycle.id), "--confirm", "REPAIR-LIMITED-CYCLE"])
    payload = json.loads(result.output)
    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDT").one()
    assert result.exit_code == 1
    assert "cycle_has_orders" in payload["blockers"]
    assert balance.available_balance == 5.0
    assert balance.locked_balance == 5.0


def test_vault_live_gate_passes_and_failure_limits_cycle(app) -> None:
    _patch_market_data(app)
    manager = app.extensions["services"]["strategy_manager"]
    user, _ = _create_user()
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDC",
        deposit_amount=50,
        settlement_asset="USDC",
        lock_duration_hours=24,
        status="active",
        execution_substatus="validating_market",
        execution_mode="live",
        live_validation_status="pending",
        validation_started_at=datetime.utcnow() - timedelta(minutes=20),
        started_at=datetime.utcnow() - timedelta(minutes=20),
        unlocks_at=datetime.utcnow() + timedelta(hours=24),
        starting_value_usd=50,
        current_estimated_value_usd=50,
    )
    db.session.add(cycle)
    db.session.flush()
    run = StrategyRun(
        strategy_name="mean_reversion",
        symbol="BTC",
        timeframe="5m",
        mode="live",
        status="running",
    )
    run.parameters = {"consumer_vault": True, "vault_cycle_id": cycle.id, "live_validation_status": "pending"}
    db.session.add(run)
    db.session.flush()
    cycle.strategy_run_id = run.id
    validation = StrategyValidation(
        strategy_name="mean_reversion",
        symbol="BTC",
        timeframe="5m",
        stage="shadow_live",
        status="pending",
    )
    validation.metrics = {"observations": 2, "actionable_signals": 2, "missed_fills": 0, "avg_spread_bps": 1.0}
    db.session.add(validation)
    db.session.commit()

    manager._evaluate_vault_live_gate(run)
    assert cycle.execution_substatus == "executing"
    assert cycle.live_validation_status == "passed"
    assert run.parameters["live_validation_status"] == "passed"

    cycle.live_validation_status = "pending"
    cycle.execution_substatus = "validating_market"
    run.parameters = {"consumer_vault": True, "vault_cycle_id": cycle.id, "live_validation_status": "pending"}
    validation.status = "pending"
    validation.completed_at = None
    validation.metrics = {"observations": 2, "actionable_signals": 2, "missed_fills": 0, "avg_spread_bps": 100.0}
    db.session.commit()

    manager._evaluate_vault_live_gate(run)
    assert cycle.execution_substatus == "limited"
    assert cycle.live_validation_status == "failed"
    assert run.mode == "live"
