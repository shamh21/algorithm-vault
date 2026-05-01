from __future__ import annotations

from datetime import datetime

import pyotp
import pytest

from app.auth import decrypt_totp_secret, encrypt_totp_secret, password_hash
from app.extensions import db
from app.models import Order, Setting, TradingConnection, User
from app.services.hyperliquid_client import ClientSnapshot
from app.services.live_provider_adapters import BinanceFuturesConnector, UniswapDelegatedConnector
from app.services.order_manager import OrderIntent


def _create_user(username: str) -> User:
    user = User(username=username, password_hash=password_hash("password123"), role="user")
    db.session.add(user)
    db.session.commit()
    return user


def _enable_2fa(user: User) -> str:
    secret = pyotp.random_base32()
    user.totp_secret_encrypted = encrypt_totp_secret(secret)
    user.two_factor_enabled_at = datetime.utcnow()
    db.session.commit()
    return secret


def _create_connection(app, user: User, *, provider: str = "hyperliquid") -> TradingConnection:
    connection = app.extensions["services"]["trading_connections"].create_or_update(
        user_id=user.id,
        provider=provider,
        connection_type="cex_api_key",
        api_key="0x" + ("a" * 40),
        api_secret="0x" + ("1" * 64),
        passphrase="connection-passphrase",
        wallet_address="0x" + ("2" * 40),
        is_active=True,
    )
    if provider == "hyperliquid":
        connection.verification_status = "verified"
        connection.is_active = True
    db.session.commit()
    return connection


def test_provider_specs_mark_only_hyperliquid_tradable(app) -> None:
    specs = app.extensions["services"]["trading_connections"].provider_specs()

    assert specs["hyperliquid"]["tradable"] is True
    assert specs["hyperliquid"]["verification_supported"] is True
    assert specs["binance"]["tradable"] is True
    assert specs["kucoin"]["verification_supported"] is True
    assert specs["dydx"]["connection_type"] == "permissioned_key"
    assert specs["uniswap"]["connection_type"] == "wallet_delegation"


def test_trading_connection_encrypts_and_decrypts_only_for_execution(app) -> None:
    user = _create_user("encrypt")
    service = app.extensions["services"]["trading_connections"]

    connection = service.create_or_update(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        api_key="account-label",
        api_secret="super-secret-key",
        passphrase="exchange-passphrase",
        wallet_address="0x" + ("3" * 40),
        is_active=True,
    )
    db.session.commit()

    assert connection.encrypted_api_secret != "super-secret-key"
    assert "super-secret-key" not in connection.encrypted_api_secret

    credentials = service.credentials_for_execution(user.id, connection.id)

    assert credentials.api_key == "account-label"
    assert credentials.api_secret == "super-secret-key"
    assert credentials.passphrase == "exchange-passphrase"
    assert credentials.wallet_address == "0x" + ("3" * 40)
    assert connection.verification_status == "needs_verification"
    assert not connection.is_active


def test_trading_connection_rejects_seed_phrases_and_cross_user_access(app) -> None:
    owner = _create_user("owner")
    other = _create_user("other")
    service = app.extensions["services"]["trading_connections"]
    connection = _create_connection(app, owner)

    with pytest.raises(ValueError, match="Hyperliquid API wallet/agent secret"):
        service.create_or_update(
            user_id=owner.id,
            provider="hyperliquid",
            connection_type="cex_api_key",
            api_secret="alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima",
        )

    with pytest.raises(PermissionError):
        service.get_for_user(other.id, connection.id)


def test_dydx_permissioned_key_metadata_is_required_and_secret_is_encrypted(app) -> None:
    user = _create_user("dydx")
    service = app.extensions["services"]["trading_connections"]

    with pytest.raises(ValueError):
        service.create_or_update(
            user_id=user.id,
            provider="dydx",
            connection_type="permissioned_key",
            api_secret="permissioned-private-key",
            wallet_address="dydx1owner",
            metadata={"subaccount_number": "0"},
        )

    connection = service.create_or_update(
        user_id=user.id,
        provider="dydx",
        connection_type="permissioned_key",
        api_secret="permissioned-private-key",
        wallet_address="dydx1owner",
        metadata={"subaccount_number": "0", "authenticator_id": "auth-1"},
    )
    db.session.commit()

    assert connection.encrypted_api_secret != "permissioned-private-key"
    assert connection.provider_metadata["authenticator_id"] == "auth-1"
    assert service.credentials_for_execution(user.id, connection.id).api_secret == "permissioned-private-key"


def test_uniswap_delegation_requires_caps_and_rejects_private_fields(app) -> None:
    user = _create_user("uniswap")
    service = app.extensions["services"]["trading_connections"]

    with pytest.raises(ValueError):
        service.create_or_update(
            user_id=user.id,
            provider="uniswap",
            connection_type="wallet_delegation",
            api_secret="private-key-not-allowed",
            wallet_address="0x" + ("1" * 40),
        )

    connection = service.create_or_update(
        user_id=user.id,
        provider="uniswap",
        connection_type="wallet_delegation",
        wallet_address="0x" + ("1" * 40),
        metadata={
            "chain_id": "1",
            "delegation_status": "approved",
            "delegation_expires_at": "2099-01-01T00:00",
            "max_notional_usd": "100",
            "daily_loss_usd": "25",
            "allowed_tokens": "ETH,BTC",
            "session_topic": "wc-session",
        },
    )
    db.session.commit()

    assert connection.verification_status == "needs_verification"
    assert connection.provider_metadata["max_notional_usd"] == "100"


def test_new_provider_remains_inactive_until_verified(app) -> None:
    user = _create_user("inactive")
    service = app.extensions["services"]["trading_connections"]
    connection = _create_connection(app, user, provider="binance")

    assert connection.verification_status == "needs_verification"
    assert not connection.is_active
    assert service.can_trade(user.id, "live", connection.id) is False

    with pytest.raises(ValueError):
        service.activate_verified(user.id, connection.id)


def test_verify_connection_sets_verified_or_action_needed(app, monkeypatch) -> None:
    user = _create_user("verify")
    service = app.extensions["services"]["trading_connections"]
    connection = service.create_or_update(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        api_secret="secret",
        wallet_address="0x" + ("7" * 40),
    )

    class GoodConnector:
        def can_trade(self, mode: str) -> bool:
            return True

        def account_snapshot(self, mode: str) -> ClientSnapshot:
            return ClientSnapshot(mode, [{"asset": "USDC", "value": 100.0}], [], [], [], [])

    monkeypatch.setattr(service, "_connector_for_connection", lambda record: GoodConnector())

    result = service.verify_connection(user.id, connection.id)

    assert result["ok"] is True
    assert connection.verification_status == "verified"
    assert connection.last_verification_error is None

    class BadConnector:
        def can_trade(self, mode: str) -> bool:
            return True

        def account_snapshot(self, mode: str) -> ClientSnapshot:
            return ClientSnapshot(mode, [], [], [], [], ["bad credentials"])

    connection.verification_status = "needs_verification"
    monkeypatch.setattr(service, "_connector_for_connection", lambda record: BadConnector())

    result = service.verify_connection(user.id, connection.id)

    assert result["ok"] is False
    assert connection.verification_status == "action_needed"
    assert connection.last_verification_error == "bad credentials"


def test_verify_binance_connection_can_activate_after_provider_success(app, monkeypatch) -> None:
    user = _create_user("binanceverify")
    service = app.extensions["services"]["trading_connections"]
    connection = _create_connection(app, user, provider="binance")

    class GoodConnector:
        def can_trade(self, mode: str) -> bool:
            return mode == "live"

        def account_snapshot(self, mode: str) -> ClientSnapshot:
            return ClientSnapshot(mode, [{"asset": "USDT", "type": "futures", "value": 100.0}], [], [], [], [])

    monkeypatch.setattr(service, "_connector_for_connection", lambda record: GoodConnector())

    result = service.verify_connection(user.id, connection.id)
    assert result["ok"] is True
    assert connection.verification_status == "verified"

    activated = service.activate_verified(user.id, connection.id)
    assert activated.is_active
    assert service.active_tradable_connection(user.id).provider == "binance"


def test_activate_verified_denies_unverified_providers(app) -> None:
    user = _create_user("activate")
    service = app.extensions["services"]["trading_connections"]
    unverified = _create_connection(app, user, provider="binance")

    with pytest.raises(ValueError):
        service.activate_verified(user.id, unverified.id)

    hyperliquid = _create_connection(app, user, provider="hyperliquid")
    hyperliquid.verification_status = "verified"
    db.session.commit()

    activated = service.activate_verified(user.id, hyperliquid.id)

    assert activated.is_active
    assert service.active_tradable_connection(user.id).id == hyperliquid.id


def test_live_order_uses_authenticated_users_connection(app, monkeypatch) -> None:
    user = _create_user("trader")
    connection = _create_connection(app, user)
    manager = app.extensions["services"]["order_manager"]
    calls: list[tuple[int, int | None]] = []

    class FakeConnector:
        def can_trade(self, mode: str) -> bool:
            return mode == "live"

        def place_order(self, *args):
            return {"status": "filled", "exchange_order_id": "user-fill-1", "fill_price": 100.0}

        def get_positions(self, mode: str):
            return [{"symbol": "BTC", "quantity": 1.0, "entry_price": 100.0, "mark_price": 100.0, "unrealized_pnl": 0.0, "leverage": 1.0}]

    def connector_for_user(user_id: int, connection_id: int | None = None):
        calls.append((user_id, connection_id))
        return FakeConnector()

    monkeypatch.setattr(app.extensions["services"]["market_data"], "get_mid_price", lambda symbol, mode: 100.0)
    monkeypatch.setattr(app.extensions["services"]["trading_connections"], "connector_for_user", connector_for_user)

    order = manager.place_order(
        OrderIntent(
            symbol="BTC",
            side="buy",
            quantity=0.1,
            mode="live",
            stop_loss=95.0,
            user_id=user.id,
            trading_connection_id=connection.id,
        )
    )

    assert order.status == "filled"
    assert order.user_id == user.id
    assert order.trading_connection_id == connection.id
    assert calls[0] == (user.id, connection.id)
    assert Order.query.filter_by(user_id=user.id, trading_connection_id=connection.id).one()


def test_binance_connector_signs_usdm_order(monkeypatch) -> None:
    class Creds:
        api_key = "key"
        api_secret = "secret"
        passphrase = ""
        wallet_address = ""

    class Response:
        status_code = 200
        text = "{}"

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    calls = []

    class Session:
        def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            if url.endswith("/fapi/v1/leverage"):
                return Response({"leverage": 2})
            return Response({"status": "NEW", "orderId": 123, "price": "100.00"})

    connector = BinanceFuturesConnector(
        {"ENABLE_LIVE_TRADING": True, "BINANCE_FUTURES_BASE_URL": "https://example.test"},
        Creds(),
    )
    connector.session = Session()
    monkeypatch.setattr("app.services.live_provider_adapters.time.time", lambda: 1_700_000_000)

    result = connector.place_order("live", "BTC", "buy", 0.1, "limit", 100.0, False, 2.0, 0.0)

    assert result["exchange_order_id"] == "123"
    assert calls[-1][2]["headers"]["X-MBX-APIKEY"] == "key"
    assert calls[-1][2]["params"]["symbol"] == "BTCUSDT"
    assert "signature" in calls[-1][2]["params"]


def test_uniswap_delegated_connector_fails_closed_without_delegation(app) -> None:
    class Creds:
        api_key = ""
        api_secret = ""
        passphrase = ""
        wallet_address = "0x" + ("1" * 40)

    connector = UniswapDelegatedConnector({"ENABLE_LIVE_TRADING": True, "UNISWAP_API_KEY": "api-key"}, Creds(), {})

    assert connector.can_trade("paper") is False
    with pytest.raises(RuntimeError, match="delegation"):
        connector.can_trade("live")


def test_register_2fa_connection_onboarding_then_live_home(app, monkeypatch) -> None:
    app.config["APP_MODE"] = "live"
    Setting.set_json("current_mode", "live")
    db.session.commit()
    client = app.test_client()

    registered = client.post(
        "/register",
        data={"username": "onboard", "password": "password123", "confirm_password": "password123"},
    )
    assert registered.status_code == 302
    assert registered.location == "/setup-2fa"

    user = User.query.filter_by(username="onboard").one()
    client.get("/setup-2fa")
    secret = decrypt_totp_secret(user)
    enabled = client.post("/setup-2fa", data={"totp_code": pyotp.TOTP(secret).now()})

    assert enabled.status_code == 302
    assert enabled.location == "/settings/connections"

    saved = client.post(
        "/settings/connections",
        data={
            "provider": "hyperliquid",
            "connection_type": "cex_api_key",
            "api_secret": "0x" + ("4" * 64),
            "wallet_address": "0x" + ("5" * 40),
        },
    )

    assert saved.status_code == 302
    assert "/settings/connections/hyperliquid" in saved.location
    connection = TradingConnection.query.filter_by(user_id=user.id).one()
    assert connection.provider == "hyperliquid"
    assert connection.verification_status == "needs_verification"
    assert not connection.is_active

    class GoodConnector:
        def can_trade(self, mode: str) -> bool:
            return True

        def account_snapshot(self, mode: str) -> ClientSnapshot:
            return ClientSnapshot(mode, [{"asset": "USDC", "type": "margin", "value": 500.0, "withdrawable": 500.0}], [], [], [], [])

    service = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(service, "_connector_for_connection", lambda record: GoodConnector())

    verified = client.post(f"/settings/connections/{connection.id}/verify")

    assert verified.status_code == 302
    assert db.session.get(TradingConnection, connection.id).verification_status == "verified"

    activated = client.post(f"/settings/connections/{connection.id}/activate")

    assert activated.status_code == 302
    assert activated.location == "/"
    assert db.session.get(TradingConnection, connection.id).is_active

    home = client.get("/")
    assert home.status_code == 200
    assert b"Portfolio Value" in home.data


def test_unsupported_provider_does_not_satisfy_live_onboarding(app) -> None:
    app.config["APP_MODE"] = "live"
    Setting.set_json("current_mode", "live")
    user = _create_user("draftonly")
    secret = _enable_2fa(user)
    _create_connection(app, user, provider="binance")
    client = app.test_client()
    client.post("/login", data={"username": user.username, "password": "password123", "totp_code": pyotp.TOTP(secret).now()})

    response = client.get("/")

    assert response.status_code == 302
    assert response.location == "/settings/connections"


def test_connection_wizard_pages_render_provider_specific_workflows(app) -> None:
    user = _create_user("wizard")
    secret = _enable_2fa(user)
    client = app.test_client()
    client.post("/login", data={"username": user.username, "password": "password123", "totp_code": pyotp.TOTP(secret).now()})

    hyperliquid = client.get("/settings/connections/hyperliquid")
    binance = client.get("/settings/connections/binance")
    uniswap = client.get("/settings/connections/uniswap")
    dydx = client.get("/settings/connections/dydx")

    assert hyperliquid.status_code == 200
    assert b"Easy Connect" in hyperliquid.data
    assert b"Verify And Activate" in hyperliquid.data
    assert b"Hyperliquid Setup" in hyperliquid.data
    assert binance.status_code == 200
    assert b"Live USD-M futures" in binance.data
    assert b"Save Connection" in binance.data
    assert uniswap.status_code == 200
    assert b"Public Wallet Address" in uniswap.data
    assert b"Delegation Expiry" in uniswap.data
    assert dydx.status_code == 200
    assert b"Permissioned Trading Private Key" in dydx.data
