from __future__ import annotations

from datetime import datetime

import pyotp
import pytest
from cryptography.fernet import Fernet

from app.auth import decrypt_totp_secret, encrypt_totp_secret, password_hash
from app.extensions import db
from app.models import Order, Setting, TradingConnection, User
from app.services.connection_health import build_connection_health, parse_exchange_failure, store_connection_health
from app.services.hyperliquid_client import ClientSnapshot, HyperliquidClient
from app.services.live_provider_adapters import BinanceFuturesConnector, KucoinFuturesConnector, ProviderRequestError, UniswapDelegatedConnector
from app.services.order_manager import OrderIntent, OrderManager


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
        api_secret="0x" + ("1" * 64),
        passphrase="exchange-passphrase",
        wallet_address="0x" + ("3" * 40),
        is_active=True,
    )
    db.session.commit()

    assert connection.encrypted_api_secret != "0x" + ("1" * 64)
    assert "0x" + ("1" * 64) not in connection.encrypted_api_secret

    credentials = service.credentials_for_execution(user.id, connection.id)

    assert credentials.api_key == "account-label"
    assert credentials.api_secret == "0x" + ("1" * 64)
    assert credentials.passphrase == "exchange-passphrase"
    assert credentials.wallet_address == "0x" + ("3" * 40)
    assert connection.verification_status == "needs_verification"
    assert not connection.is_active


def test_stale_required_connection_secret_has_actionable_error(app) -> None:
    user = _create_user("stale-required")
    service = app.extensions["services"]["trading_connections"]
    old_fernet = Fernet(Fernet.generate_key())
    connection = service.create_or_update(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        api_secret="0x" + ("1" * 64),
        wallet_address="0x" + ("3" * 40),
    )
    connection.encrypted_api_secret = old_fernet.encrypt(b"0x" + b"4" * 64).decode("utf-8")
    db.session.commit()

    with pytest.raises(RuntimeError, match="Saved API Wallet Secret cannot be decrypted"):
        service.credentials_for_execution(user.id, connection.id)

    result = service.verify_connection(user.id, connection.id)
    assert result["ok"] is False
    assert "Re-enter or delete this connection" in result["error"]
    assert connection.verification_status == "action_needed"


def test_stale_optional_hyperliquid_label_does_not_block_required_credentials(app) -> None:
    user = _create_user("stale-optional")
    service = app.extensions["services"]["trading_connections"]
    old_fernet = Fernet(Fernet.generate_key())
    connection = service.create_or_update(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        api_key="account-label",
        api_secret="0x" + ("1" * 64),
        wallet_address="0x" + ("3" * 40),
    )
    connection.encrypted_api_key = old_fernet.encrypt(b"old-label").decode("utf-8")
    db.session.commit()

    credentials = service.credentials_for_execution(user.id, connection.id)

    assert credentials.api_key == ""
    assert credentials.api_secret == "0x" + ("1" * 64)
    assert credentials.wallet_address == "0x" + ("3" * 40)


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


def test_hyperliquid_accepts_full_0x_api_wallet_private_key(app) -> None:
    user = _create_user("hl-secret")
    service = app.extensions["services"]["trading_connections"]

    connection = service.create_or_update(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        api_secret="0x" + ("abcdef1234567890" * 4),
        wallet_address="0x" + ("3" * 40),
    )
    db.session.commit()

    credentials = service.credentials_for_execution(user.id, connection.id)
    assert credentials.api_secret == "0x" + ("abcdef1234567890" * 4)


def test_hyperliquid_rejects_truncated_or_invalid_api_wallet_private_key(app) -> None:
    user = _create_user("hl-invalid-secret")
    service = app.extensions["services"]["trading_connections"]

    with pytest.raises(ValueError, match="full 0x private key"):
        service.create_or_update(
            user_id=user.id,
            provider="hyperliquid",
            connection_type="cex_api_key",
            api_secret="0x4e4etc....",
            wallet_address="0x" + ("3" * 40),
        )

    with pytest.raises(ValueError, match="no spaces"):
        service.create_or_update(
            user_id=user.id,
            provider="hyperliquid",
            connection_type="cex_api_key",
            api_secret="0x" + ("1" * 32) + " " + ("2" * 32),
            wallet_address="0x" + ("3" * 40),
        )


def test_long_exchange_tokens_are_not_seed_phrases_for_other_cex_providers(app) -> None:
    user = _create_user("long-token")
    service = app.extensions["services"]["trading_connections"]
    long_secret = "abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"

    connection = service.create_or_update(
        user_id=user.id,
        provider="kucoin",
        connection_type="cex_api_key",
        api_key="key123",
        api_secret=long_secret,
        passphrase="pass123",
    )
    db.session.commit()

    credentials = service.credentials_for_execution(user.id, connection.id)
    assert credentials.api_secret == long_secret


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
            "allocation_budget_usd": "100",
            "daily_loss_usd": "25",
            "allowed_tokens": "ETH,BTC",
            "session_topic": "wc-session",
        },
    )
    db.session.commit()

    assert connection.verification_status == "needs_verification"
    assert connection.provider_metadata["allocation_budget_usd"] == "100"


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
        api_secret="0x" + ("1" * 64),
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


def test_activate_verified_keeps_existing_enabled_providers_active(app) -> None:
    user = _create_user("multiactive")
    service = app.extensions["services"]["trading_connections"]
    hyperliquid = _create_connection(app, user, provider="hyperliquid")
    binance = _create_connection(app, user, provider="binance")
    binance.verification_status = "verified"
    binance.is_active = False
    db.session.commit()

    activated = service.activate_verified(user.id, binance.id)

    db.session.refresh(hyperliquid)
    db.session.refresh(binance)
    enabled_ids = {connection.id for connection in service.enabled_tradable_connections(user.id)}
    assert activated.id == binance.id
    assert hyperliquid.is_active is True
    assert binance.is_active is True
    assert enabled_ids == {hyperliquid.id, binance.id}


def test_settings_provider_cards_support_multi_enable_and_disable(app) -> None:
    user = _create_user("settingscards")
    secret = _enable_2fa(user)
    service = app.extensions["services"]["trading_connections"]
    hyperliquid = _create_connection(app, user, provider="hyperliquid")
    binance = _create_connection(app, user, provider="binance")
    binance.verification_status = "verified"
    binance.is_active = True
    store_connection_health(hyperliquid, build_connection_health(hyperliquid, can_trade=True))
    db.session.commit()

    client = app.test_client()
    client.post("/login", data={"username": user.username, "password": "password123", "totp_code": pyotp.TOTP(secret).now()})

    settings = client.get("/settings/")
    connections = client.get("/settings/connections")

    assert settings.status_code == 200
    assert b"2 Enabled" in settings.data
    assert b"Hyperliquid" in settings.data
    assert b"Binance" in settings.data
    assert b"Online" in settings.data
    for removed_copy in [b"Wallet Preferences", b"Risk Notices", b"Address Mode", b"Live is disabled until"]:
        assert removed_copy not in settings.data
    assert connections.status_code == 200
    assert b"Verify & Enable" in connections.data
    assert b"Disable" in connections.data
    assert b'role="switch"' in connections.data

    disabled = client.post(f"/settings/connections/{binance.id}/disable")

    assert disabled.status_code == 302
    db.session.refresh(hyperliquid)
    db.session.refresh(binance)
    assert hyperliquid.is_active is True
    assert binance.is_active is False
    assert binance.verification_status == "verified"
    assert binance.encrypted_api_key
    assert {connection.id for connection in service.enabled_tradable_connections(user.id)} == {hyperliquid.id}


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


def test_kucoin_connector_signs_futures_order_with_contract_sizing(monkeypatch) -> None:
    class Creds:
        api_key = "key"
        api_secret = "secret"
        passphrase = "passphrase"
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
            return Response({"code": "200000", "data": {"orderId": "ku-123", "clientOid": "client-1"}})

    connector = KucoinFuturesConnector(
        {
            "ENABLE_LIVE_TRADING": True,
            "KUCOIN_FUTURES_BASE_URL": "https://example.test",
            "KUCOIN_CONTRACT_SPECS_JSON": '{"XBTUSDTM":{"contract_size":0.001,"size_step":1,"min_size":1}}',
        },
        Creds(),
    )
    connector.session = Session()
    monkeypatch.setattr("app.services.live_provider_adapters.time.time", lambda: 1_700_000_000)

    result = connector.place_order("live", "BTC", "buy", 0.1, "limit", 100.0, False, 2.0, 0.0)

    body = calls[-1][2]["data"]
    headers = calls[-1][2]["headers"]
    assert result["status"] == "submitted"
    assert result["exchange_order_id"] == "ku-123"
    assert headers["KC-API-KEY"] == "key"
    assert headers["KC-API-KEY-VERSION"] == "2"
    assert '"symbol":"XBTUSDTM"' in body
    assert '"marginMode":"ISOLATED"' in body
    assert '"positionSide":"BOTH"' in body
    assert '"size":100' in body


def test_kucoin_connector_fails_closed_on_unaligned_contract_quantity() -> None:
    class Creds:
        api_key = "key"
        api_secret = "secret"
        passphrase = "passphrase"
        wallet_address = ""

    connector = KucoinFuturesConnector(
        {
            "ENABLE_LIVE_TRADING": True,
            "KUCOIN_CONTRACT_SPECS_JSON": '{"XBTUSDTM":{"contract_size":0.001,"size_step":1,"min_size":1}}',
        },
        Creds(),
    )

    with pytest.raises(ValueError, match="does not align"):
        connector.place_order("live", "BTC", "buy", 0.1005, "market", None, False, 1.0, 0.0)


def test_kucoin_connector_reads_futures_candles_and_mid_price() -> None:
    class Creds:
        api_key = "key"
        api_secret = "secret"
        passphrase = "passphrase"
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
            if "/kline/query" in url:
                return Response(
                    {
                        "code": "200000",
                        "data": [
                            [1_766_000_000, "100", "105", "99", "104", "1000"],
                            [1_766_000_060, "104", "106", "103", "105", "900"],
                        ],
                    }
                )
            return Response({"code": "200000", "data": {"bestBidPrice": "104.9", "bestAskPrice": "105.1"}})

    connector = KucoinFuturesConnector(
        {
            "ENABLE_LIVE_TRADING": True,
            "KUCOIN_FUTURES_BASE_URL": "https://example.test",
            "KUCOIN_SYMBOL_MAP_JSON": '{"BTC":"XBTUSDTM"}',
        },
        Creds(),
    )
    connector.session = Session()

    candles = connector.get_candles("XBTUSDTM", "1m", "live", 2)
    mid = connector.get_mid_price("XBTUSDTM", "live")

    assert [row["close"] for row in candles] == [104.0, 105.0]
    assert mid == pytest.approx(105.0)
    kline_call = calls[0]
    assert kline_call[2]["params"]["symbol"] == "XBTUSDTM"
    assert kline_call[2]["params"]["granularity"] == 1
    assert kline_call[2]["params"]["from"] > 1_000_000_000_000
    assert kline_call[2]["params"]["to"] > 1_000_000_000_000


def test_kucoin_connector_normalizes_positions_orders_and_fills() -> None:
    class Creds:
        api_key = "key"
        api_secret = "secret"
        passphrase = "passphrase"
        wallet_address = ""

    class Response:
        status_code = 200
        text = "{}"

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class Session:
        def request(self, method, url, **kwargs):
            if url.endswith("/api/v1/positions"):
                return Response({"code": "200000", "data": [{"symbol": "XBTUSDTM", "currentQty": 2, "avgEntryPrice": "100", "markPrice": "110", "unrealisedPnl": "1.5"}]})
            params = kwargs.get("params") or {}
            if params.get("status") == "active":
                return Response({"code": "200000", "data": {"items": [{"symbol": "XBTUSDTM", "id": "order-1", "side": "buy", "price": "100", "size": "2"}]}})
            if url.endswith("/api/v1/recentDoneOrders"):
                return Response({"code": "200000", "data": {"items": [{"symbol": "XBTUSDTM", "side": "sell", "price": "111", "size": "1", "realisedPnl": "2"}]}})
            return Response({"code": "200000", "data": {}})

    connector = KucoinFuturesConnector({"ENABLE_LIVE_TRADING": True}, Creds())
    connector.session = Session()

    positions = connector.get_positions("live")
    orders = connector.get_open_orders("live")
    fills = connector.get_recent_fills("live")

    assert positions[0]["symbol"] == "BTC"
    assert positions[0]["quantity"] == 2.0
    assert orders[0]["order_id"] == "order-1"
    assert fills[0]["closed_pnl"] == 2.0


def test_kucoin_provider_error_redacts_and_categorizes_failures() -> None:
    error = ProviderRequestError('{"code":"400006","msg":"Invalid request ip, the current clientIp is:209.52.132.232","KC-API-KEY":"secret-key"}')
    parsed = parse_exchange_failure(str(error))

    assert "secret-key" not in str(error)
    assert parsed["provider_code"] == "400006"
    assert parsed["client_ip"] == "209.52.132.232"
    assert parsed["failure_category"] == "ip_whitelist"


def test_hyperliquid_rejected_order_sets_local_rejection_reason(app, monkeypatch) -> None:
    user = _create_user("hl-rejected")
    connection = _create_connection(app, user)
    manager = app.extensions["services"]["order_manager"]

    class FakeConnector:
        def can_trade(self, mode: str) -> bool:
            return True

        def place_order(self, *args):
            return {"status": "rejected", "exchange_order_id": None, "error": "insufficient margin", "raw": {"response": "rejected"}}

        def get_positions(self, mode: str):
            return []

    monkeypatch.setattr(app.extensions["services"]["market_data"], "get_mid_price", lambda symbol, mode: 100.0)
    monkeypatch.setattr(app.extensions["services"]["trading_connections"], "connector_for_user", lambda user_id, connection_id=None: FakeConnector())

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

    assert order.status == "rejected"
    assert order.rejection_reason == "insufficient margin"
    assert order.details["exchange_error"] == "insufficient margin"


def test_hyperliquid_normalizes_top_level_sdk_rejection() -> None:
    result = HyperliquidClient._normalize_order_response({"status": "err", "response": "insufficient margin"}, submitted_price=100.0)

    assert result["status"] == "rejected"
    assert result["error"] == "insufficient margin"


def test_hyperliquid_market_order_uses_sdk_price_normalizer(monkeypatch) -> None:
    captured: dict[str, float] = {}

    class FakeExchange:
        def _slippage_price(self, symbol: str, is_buy: bool, slippage_pct: float) -> float:
            assert symbol == "BTC"
            assert is_buy is False
            assert slippage_pct == pytest.approx(0.0001)
            return 79971.0

        def update_leverage(self, leverage: int, symbol: str):
            return {"status": "ok"}

        def order(self, symbol, is_buy, quantity, price, order_type, reduce_only=False):
            captured.update({"quantity": quantity, "price": price})
            return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": 123}}]}}}

    client = HyperliquidClient(
        {
            "ENABLE_LIVE_TRADING": True,
            "HL_ACCOUNT_ADDRESS": "0x" + ("1" * 40),
            "HL_SECRET_KEY": "0x" + ("2" * 64),
            "HL_MAINNET_BASE_URL": "https://api.hyperliquid.xyz",
            "HL_TESTNET_BASE_URL": "https://api.hyperliquid-testnet.xyz",
        }
    )
    monkeypatch.setattr(client, "_get_exchange", lambda mode: FakeExchange())

    result = client.place_order("live", "BTC", "sell", 0.00001, "market", None, False, 1.0, 0.0001)

    assert result["status"] == "open"
    assert result["submitted_price"] == 79971.0
    assert captured == {"quantity": 0.00001, "price": 79971.0}


def test_hyperliquid_order_size_uses_asset_size_decimals(monkeypatch) -> None:
    captured: dict[str, float] = {}

    class FakeInfo:
        name_to_coin = {"TRX": "TRX"}
        coin_to_asset = {"TRX": 7}
        asset_to_sz_decimals = {7: 0}

    class FakeExchange:
        info = FakeInfo()

        def _slippage_price(self, symbol: str, is_buy: bool, slippage_pct: float) -> float:
            return 0.34905

        def update_leverage(self, leverage: int, symbol: str):
            return {"status": "ok"}

        def order(self, symbol, is_buy, quantity, price, order_type, reduce_only=False):
            captured.update({"quantity": quantity, "price": price})
            return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"filled": {"oid": 456, "avgPx": "0.34905", "totalSz": "2"}}]}}}

    client = HyperliquidClient(
        {
            "ENABLE_LIVE_TRADING": True,
            "HL_ACCOUNT_ADDRESS": "0x" + ("1" * 40),
            "HL_SECRET_KEY": "0x" + ("2" * 64),
            "HL_MAINNET_BASE_URL": "https://api.hyperliquid.xyz",
            "HL_TESTNET_BASE_URL": "https://api.hyperliquid-testnet.xyz",
        }
    )
    monkeypatch.setattr(client, "_get_exchange", lambda mode: FakeExchange())

    result = client.place_order("live", "TRX", "sell", 2.864631, "market", None, False, 1.0, 0.0001)

    assert result["status"] == "filled"
    assert result["submitted_quantity"] == 2.0
    assert result["filled_quantity"] == 2.0
    assert captured == {"quantity": 2.0, "price": 0.34905}


def test_order_submit_slippage_caps_requested_value() -> None:
    assert OrderManager._submission_slippage_pct(0.0001, 0.015) == pytest.approx(0.0001)
    assert OrderManager._submission_slippage_pct(0.02, 0.015) == pytest.approx(0.015)
    assert OrderManager._submission_slippage_pct(0.0, 0.015) == pytest.approx(0.0075)


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
    assert "/settings/connections/hyperliquid" in activated.location
    assert db.session.get(TradingConnection, connection.id).is_active

    home = client.get("/")
    assert home.status_code == 200
    assert b"Portfolio Balance" in home.data


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
    assert b"Providers" in hyperliquid.data
    assert b"Verify & Enable" in hyperliquid.data
    assert b"Hyperliquid" in hyperliquid.data
    assert binance.status_code == 200
    assert b"Binance" in binance.data
    assert b"Save Connection" in binance.data
    assert uniswap.status_code == 200
    assert b"Public Wallet Address" in uniswap.data
    assert b"Delegation Expiry" in uniswap.data
    assert dydx.status_code == 200
    assert b"Permissioned Trading Private Key" in dydx.data
