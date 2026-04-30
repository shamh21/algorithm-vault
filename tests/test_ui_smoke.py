from __future__ import annotations

from datetime import datetime, timedelta

import pyotp
from cryptography.fernet import Fernet
from werkzeug.security import check_password_hash

from app import create_app
from app.auth import decrypt_totp_secret, encrypt_totp_secret, password_hash
from app.extensions import db
from app.models import DepositAddress, Fill, OptimizerRun, Order, Setting, StrategyRanking, StrategyRun, StrategyValidation, User, VaultCycle, WalletAddress, WalletBalance, WalletTransaction, WalletWithdrawal
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
        {"symbol": symbol, "mid": 100.0, "recent_average": 100.0, "change_pct": 0.0}
        for symbol in symbols
    ]
    market_data.get_candles = lambda symbol, timeframe, mode, limit: _candles()
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: {
        "levels": [[{"px": "99.95", "sz": "1"}], [{"px": "100.05", "sz": "1"}]]
    }


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


class _LiveWalletAdapter:
    def __init__(self) -> None:
        self.broadcasts = 0

    def supports(self, asset: str, network: str) -> bool:
        return asset.upper() == "ETH" and network == "Ethereum"

    def generate_wallet(self, asset: str, network: str) -> GeneratedWallet:
        return GeneratedWallet(
            address="0x1234567890abcdef1234567890abcdef12345678",
            private_key="11" * 32,
            public_key="0x1234567890abcdef1234567890abcdef12345678",
            key_type="secp256k1",
            provider="fake_evm",
        )

    def get_balance(self, address: str, asset: str, network: str) -> WalletBalanceSnapshot:
        return WalletBalanceSnapshot(amount=0.0, asset=asset, checked=True, confirmations=12)

    def estimate_fee(self, asset: str, network: str, destination: str, amount: float) -> float:
        return 0.001

    def sign_and_broadcast(self, withdrawal: WalletWithdrawal, private_key: str) -> BroadcastResult:
        assert private_key == "11" * 32
        self.broadcasts += 1
        return BroadcastResult("submitted", "0xroutehash", {"ok": True})

    def confirm_transaction(self, provider_reference: str, asset: str, network: str) -> dict:
        return {"confirmed": True}


def _enable_live_wallets(app) -> tuple[_LiveWalletAdapter, RealWalletCustodyService]:
    app.config["USE_REAL_ADDRESSES"] = True
    app.config["WALLET_REAL_CUSTODY_ENABLED"] = True
    app.config["WALLET_ALLOW_IN_APP_KEYGEN"] = True
    app.config["WALLET_WITHDRAWALS_ENABLED"] = True
    app.config["WALLET_REQUIRE_WITHDRAWAL_APPROVAL"] = True
    app.config["TOTP_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    app.config["WALLET_EVM_RPC_URL"] = "https://evm.example.invalid"
    Setting.set_json("use_real_addresses", True)
    fake = _LiveWalletAdapter()
    custody = RealWalletCustodyService(app.config, adapters=[fake])
    app.extensions["services"]["wallet_custody"] = custody
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
    vault = client.get("/vault")
    activity = client.get("/activity")
    settings = client.get("/settings/")

    assert home.status_code == 200
    assert b"Algorithm Vault" in home.data
    assert b"Start a Cycle" in home.data
    assert wallet.status_code == 200
    assert b"Total Portfolio Value" in wallet.data
    assert b"Deposit" in wallet.data
    assert b"Withdraw" in wallet.data
    assert vault.status_code == 200
    assert b"Start Algorithm Cycle" in vault.data
    assert b"Algorithm Profile" not in vault.data
    assert activity.status_code == 200
    assert b"Wallet and Vault History" in activity.data
    assert settings.status_code == 200
    assert b"Wallet Preferences" in settings.data


def test_admin_routes_require_login_and_2fa(app) -> None:
    client = app.test_client()
    for path in ["/admin/dashboard", "/admin/orders", "/admin/backtests", "/admin/panic"]:
        response = client.get(path)
        assert response.status_code == 302
        assert "/login" in response.location

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
    orders = client.get("/admin/orders")
    backtests = client.get("/admin/backtests")
    panic = client.get("/admin/panic")
    risk = client.get("/admin/risk")
    strategies = client.get("/admin/strategies")
    readiness = client.get("/admin/live-readiness")

    assert dashboard.status_code == 200
    assert b"Automation Rankings" in dashboard.data
    assert orders.status_code == 200
    assert b"Manual Order Entry" in orders.data
    assert backtests.status_code == 200
    assert b"Short-Term Optimizer" in backtests.data
    assert b'value="dynamic_intraday"' in backtests.data
    assert b"Dynamic Intraday" in backtests.data
    assert b"Upside Scanner Diagnostics" in backtests.data
    assert b"High-upside profile" in backtests.data
    assert panic.status_code == 200
    assert b"Panic" in panic.data
    assert risk.status_code == 200
    assert b"Risk Diagnostics" in risk.data
    assert strategies.status_code == 200
    assert b"Strategy and Vault Internals" in strategies.data
    assert readiness.status_code == 200
    assert b"Live Readiness" in readiness.data


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


def test_aggressive_optimizer_visible_and_rankings_show_experimental_fields(app) -> None:
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
    ranking.warnings = [
        "Aggressive 1H mode is experimental and can lose capital quickly. Past backtests do not guarantee future returns."
    ]
    db.session.add(ranking)
    db.session.commit()

    client = app.test_client()
    _login(client, "optimizeradmin2", admin_secret)
    page = client.get("/admin/backtests")

    assert page.status_code == 200
    assert b"Aggressive 1H Experimental" in page.data
    assert b"Very High Risk" in page.data
    assert b"3.50%" in page.data
    assert b"2.00%" in page.data
    assert b"$1.25" in page.data
    assert b"9.50 bps" in page.data
    assert b"58.00%" in page.data
    assert b"88.00%" in page.data
    assert b"67.00% accepted" in page.data
    assert b"1.7500" in page.data
    assert b"low_edge_after_costs" in page.data
    assert b"Aggressive Experiment Comparison" in page.data
    assert b"Profit Lab" in page.data
    assert b"Edge / Cost" in page.data
    assert b"Convex Edge" in page.data
    assert b"12.345" in page.data
    assert b"2.50x" in page.data
    assert b"3.00x" in page.data
    assert b"0.125" in page.data
    assert b"low_trade_count" in page.data


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
    app.config["DEPOSIT_ADDRESS_BOOK"] = {
        "USDC": {
            "Ethereum": ["0x3333333333333333333333333333333333333333"]
        }
    }
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
    app.config["DEPOSIT_ADDRESS_BOOK"] = {
        "usdc": {
            "ethereum": ["0x4444444444444444444444444444444444444444"]
        }
    }
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
        name.lower()
        for model in (DepositAddress, WalletBalance, WalletTransaction)
        for name in model.__table__.columns.keys()
    }

    assert not any(any(term in column for term in forbidden) for column in model_columns)


def test_admin_can_view_deposit_address_history_but_consumer_cannot(app) -> None:
    _patch_market_data(app)
    app.config["DEPOSIT_ADDRESS_BOOK"] = {
        "USDC": {
            "Ethereum": ["0x5555555555555555555555555555555555555555"]
        }
    }
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
    app.config["WALLET_WITHDRAWALS_ENABLED"] = True
    user, secret = _create_user()
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

    valid = client.post(
        "/wallet/withdraw/USDC",
        data={
            "withdraw_address": "0x1111111111111111111111111111111111111111",
            "amount": "10",
            "network": "Ethereum",
            "totp_code": pyotp.TOTP(secret).now(),
        },
    )
    assert valid.status_code == 200
    assert b"real wallet address mode is required" in valid.data
    tx = WalletTransaction.query.filter_by(user_id=user.id, transaction_type="withdrawal").one()
    withdrawal = WalletWithdrawal.query.filter_by(user_id=user.id).one()
    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDC").one()
    assert tx.status == "failed"
    assert tx.withdraw_address == "0x1111111111111111111111111111111111111111"
    assert withdrawal.status == "failed"
    assert withdrawal.workflow_type == "manual_withdrawal"
    assert balance.available_balance == 1000
    assert balance.locked_balance == 0


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
    assert balance.available_balance == 1.0
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
    assert balance.available_balance == 1.0
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
    assert balance.available_balance == 1.0
    assert balance.locked_balance == 1.0


def test_vault_cycle_start_locks_wallet_and_links_strategy(app) -> None:
    _patch_market_data(app)
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    user, secret = _create_user()
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


def test_one_hour_vault_cycle_uses_short_horizon_strategy(app) -> None:
    _patch_market_data(app)
    _patch_deep_book(app)
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    user, secret = _create_user(username="onehour")
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
        },
    )

    assert response.status_code == 302
    cycle = VaultCycle.query.filter_by(user_id=user.id).one()
    run = db.session.get(StrategyRun, cycle.strategy_run_id)
    assert cycle.lock_duration_hours == 1
    assert cycle.algorithm_profile == "Aggressive"
    assert cycle.selected_strategy_name == "scalping"
    assert cycle.selected_timeframe == "1m"
    assert run.parameters["allocation_cap_usd"] == 50


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
    assert b"Algorithm Profile" in vault.data
    assert b"volatility_breakout" not in vault.data


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
