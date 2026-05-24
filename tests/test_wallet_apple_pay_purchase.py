from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta
from typing import Any

import requests

from app.auth import password_hash
from app.csrf import CSRF_SESSION_KEY
from app.extensions import db
from app.models import DepositAddress, User, WalletApplePayPurchaseOrder, WalletBalance, WalletLedgerEvent, WalletTransaction
from app.services.wallet_apple_pay_purchase import WalletApplePayPurchaseError, WalletApplePayPurchaseService


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _FakeSession:
    def __init__(
        self,
        *,
        quote_payload: dict[str, Any] | None = None,
        gateway_payload: dict[str, Any] | None = None,
        swap_payload: dict[str, Any] | None = None,
        signer_payload: dict[str, Any] | None = None,
        exc: Exception | None = None,
        fail_gateway: bool = False,
        fail_swap: bool = False,
    ) -> None:
        self.quote_payload = quote_payload or {"gas": "50000", "gasPrice": "20000000000", "quoteId": "quote-123"}
        self.gateway_payload = gateway_payload or {"status": "captured", "payment_id": "pay-123", "capture_id": "cap-123"}
        self.swap_payload = swap_payload or {"swap_id": "swap-123", "tx": {"to": "0x111", "data": "0x", "value": "0"}}
        self.signer_payload = signer_payload or {"status": "complete", "tx_hash": "0xuser", "treasury_tx_hash": "0xtreasury"}
        self.exc = exc
        self.fail_gateway = fail_gateway
        self.fail_swap = fail_swap
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, *, params: dict[str, Any], headers: dict[str, str], timeout: float) -> _FakeResponse:
        self.calls.append({"method": "GET", "url": url, "params": params, "headers": headers, "timeout": timeout})
        if self.exc is not None:
            raise self.exc
        if self.fail_swap and url.endswith("/swap"):
            raise requests.Timeout("swap timeout")
        return _FakeResponse(self.swap_payload if url.endswith("/swap") else self.quote_payload)

    def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append({"method": "POST", "url": url, **kwargs})
        if self.exc is not None:
            raise self.exc
        if "apple-pay-gateway.example" in url:
            if self.fail_gateway:
                raise requests.Timeout("gateway timeout")
            return _FakeResponse(self.gateway_payload)
        if "card-gateway.example" in url:
            if self.fail_gateway:
                raise requests.Timeout("gateway timeout")
            return _FakeResponse(self.gateway_payload)
        if "apple-pay-signer.example" in url:
            return _FakeResponse(self.signer_payload)
        return _FakeResponse({"epochTimestamp": 1, "merchantSessionIdentifier": "merchant-session"})


def _configure_apple_pay(app, *, fake_session: _FakeSession | None = None) -> _FakeSession:
    fake = fake_session or _FakeSession()
    app.config.update(
        APPLE_PAY_DIRECT_ENABLED=True,
        APPLE_PAY_CRYPTO_SALE_APPROVED=True,
        APPLE_PAY_MERCHANT_ID="merchant.com.algvault",
        APPLE_PAY_DISPLAY_NAME="AlgVault",
        APPLE_PAY_DOMAIN="algvault.app",
        APPLE_PAY_DOMAIN_ASSOCIATION="apple-domain-association-test",
        APPLE_PAY_COUNTRY_CODE="CA",
        APPLE_PAY_SUPPORTED_NETWORKS=["visa", "masterCard"],
        APPLE_PAY_MERCHANT_CERT_PEM="-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----",
        APPLE_PAY_MERCHANT_KEY_PEM="-----BEGIN PRIVATE KEY-----\nMIIB\n-----END PRIVATE KEY-----",
        APPLE_PAY_GATEWAY_AUTHORIZE_URL="https://apple-pay-gateway.example/authorize",
        APPLE_PAY_GATEWAY_API_KEY="gw_test_key",
        APPLE_PAY_GATEWAY_WEBHOOK_SECRET="gw_webhook_secret",
        APPLE_PAY_BUY_ALLOWED_ASSETS={
            "USDT": {
                "Ethereum": {
                    "chain_id": 1,
                    "token_address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                    "decimals": 6,
                    "source_asset": "USDT",
                    "source_token_address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                }
            }
        },
        APPLE_PAY_TREASURY_SOURCE_ADDRESS="0x2222222222222222222222222222222222222222",
        APPLE_PAY_TREASURY_FEE_ADDRESS="0x3333333333333333333333333333333333333333",
        APPLE_PAY_TREASURY_SIGNER_URL="https://apple-pay-signer.example/submit",
        APPLE_PAY_TREASURY_SIGNER_TOKEN="signer-token",
        APPLE_PAY_TREASURY_FEE_BPS=250.0,
        APPLE_PAY_EXECUTION_FEE_BUFFER_BPS=500.0,
        APPLE_PAY_MIN_FIAT_USD=10.0,
        APPLE_PAY_MAX_FIAT_USD=5000.0,
        APPLE_PAY_TIMEOUT_SECONDS=7.0,
        PLATFORM_TREASURY_ETH_USD_FALLBACK=3000.0,
        ONEINCH_API_KEY="oneinch-key",
        ONEINCH_API_BASE_URL="https://api.1inch.dev/swap/v6.0",
        ENABLE_LIVE_TRADING=False,
    )
    app.extensions["services"]["wallet_apple_pay_purchase"].session = fake
    return fake


def _configure_card_buy(app, *, fake_session: _FakeSession | None = None) -> _FakeSession:
    fake = _configure_apple_pay(app, fake_session=fake_session)
    app.config.update(
        CARD_BUY_ENABLED=True,
        CARD_GATEWAY_TOKENIZATION_URL="https://card-gateway.example/tokenize",
        CARD_GATEWAY_AUTHORIZE_URL="https://card-gateway.example/authorize",
        CARD_GATEWAY_API_KEY="card_test_key",
        CARD_GATEWAY_WEBHOOK_SECRET="card_webhook_secret",
        CARD_GATEWAY_PUBLIC_CONFIG={"publishable_key": "pk_test_card", "secret_key": "do-not-leak"},
        WALLET_BUY_PLATFORM_FEE_BPS=250.0,
        WALLET_BUY_QUOTE_TTL_SECONDS=300,
    )
    return fake


def _treasury_transfer_allowed_assets() -> dict[str, Any]:
    return {
        "USDC": {
            "Ethereum": {
                "fulfillment_kind": "treasury_transfer",
                "chain_id": 1,
                "token_address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                "decimals": 6,
            }
        },
        "USDT": {
            "Ethereum": {
                "fulfillment_kind": "treasury_transfer",
                "chain_id": 1,
                "token_address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                "decimals": 6,
            }
        },
        "ETH": {"Ethereum": {"fulfillment_kind": "treasury_transfer", "chain_id": 1, "decimals": 18}},
        "BTC": {"Bitcoin": {"fulfillment_kind": "treasury_transfer", "decimals": 8}},
        "SOL": {"Solana": {"fulfillment_kind": "treasury_transfer", "decimals": 9}},
        "XRP": {"XRP Ledger": {"fulfillment_kind": "treasury_transfer", "decimals": 6}},
        "ALGV": {"Ethereum": {"fulfillment_kind": "treasury_transfer", "chain_id": 1}},
    }


def _treasury_source_wallets() -> dict[str, Any]:
    return {
        "USDC": {"Ethereum": {"source_address": "0x2222222222222222222222222222222222222222", "signer_route": "evm"}},
        "USDT": {"Ethereum": {"source_address": "0x2222222222222222222222222222222222222222", "signer_route": "evm"}},
        "ETH": {"Ethereum": {"source_address": "0x2222222222222222222222222222222222222222", "signer_route": "evm"}},
        "BTC": {"Bitcoin": {"source_address": "bc1qtreasurywallet", "signer_route": "btc"}},
        "SOL": {"Solana": {"source_address": "SoLTreasuryWallet111111111111111111111111", "signer_route": "sol"}},
        "XRP": {"XRP Ledger": {"source_address": "rTreasuryWallet111111111111111111111", "signer_route": "xrp"}},
    }


def _configure_treasury_transfer_apple_pay(app, *, fake_session: _FakeSession | None = None) -> _FakeSession:
    fake = _configure_apple_pay(app, fake_session=fake_session)
    app.config.update(
        APPLE_PAY_BUY_ALLOWED_ASSETS=_treasury_transfer_allowed_assets(),
        APPLE_PAY_TREASURY_SOURCE_WALLETS=_treasury_source_wallets(),
        APPLE_PAY_TREASURY_SOURCE_ADDRESS="",
        APPLE_PAY_TREASURY_FEE_ADDRESS="",
        APPLE_PAY_ASSET_PRICE_USD={
            "BTC": 100_000.0,
            "ETH": 3_000.0,
            "SOL": 150.0,
            "XRP": 0.6,
        },
        ONEINCH_API_KEY="",
        WALLET_EVM_RPC_URL="https://evm.example.invalid",
        WALLET_BTC_INDEXER_URL="https://btc.example.invalid",
        WALLET_SOLANA_RPC_URL="https://sol.example.invalid",
        WALLET_XRP_RPC_URL="https://xrp.example.invalid",
    )
    return fake


def _user_with_deposit(asset: str = "USDT", network: str = "Ethereum") -> tuple[User, DepositAddress]:
    asset_key = asset.upper()
    addresses = {
        "BTC": "bc1quserwallet111111111111111111111111111",
        "ETH": "0x1111111111111111111111111111111111111111",
        "SOL": "SoLUserWallet1111111111111111111111111111",
        "USDC": "0x4444444444444444444444444444444444444444",
        "USDT": "0x5555555555555555555555555555555555555555",
        "XRP": "rUserWallet111111111111111111111111",
    }
    user = User(
        username=f"apple-pay-user-{asset_key.lower()}-{DepositAddress.query.count() + 1}",
        password_hash=password_hash("password123"),
        role="user",
        totp_secret_encrypted="configured",
        two_factor_enabled_at=datetime.utcnow(),
    )
    db.session.add(user)
    db.session.flush()
    deposit = DepositAddress(
        user_id=user.id,
        asset=asset_key,
        network=network,
        address=addresses.get(asset_key, "0x1111111111111111111111111111111111111111"),
        version=1,
        is_active=True,
    )
    balance = WalletBalance(
        user_id=user.id,
        asset=asset_key,
        available_balance=0.0,
        locked_balance=0.0,
        estimated_usd_value=0.0,
        active_deposit_address=deposit,
    )
    db.session.add_all([deposit, balance])
    db.session.commit()
    return user, deposit


def _login(client, user: User) -> str:
    csrf_token = "csrf-apple-pay-test"
    with client.session_transaction() as session:
        session["user_id"] = user.id
        session["two_factor_verified"] = True
        session[CSRF_SESSION_KEY] = csrf_token
    return csrf_token


def _signed_headers(secret: str, body: bytes) -> dict[str, str]:
    timestamp = str(int(time.time()))
    signature = hmac.new(secret.encode("utf-8"), timestamp.encode("utf-8") + b"." + body, hashlib.sha256).hexdigest()
    return {
        "X-AlgVault-ApplePay-Timestamp": timestamp,
        "X-AlgVault-ApplePay-Signature": "sha256=" + signature,
        "Content-Type": "application/json",
    }


def _signed_card_headers(secret: str, body: bytes) -> dict[str, str]:
    timestamp = str(int(time.time()))
    signature = hmac.new(secret.encode("utf-8"), timestamp.encode("utf-8") + b"." + body, hashlib.sha256).hexdigest()
    return {
        "X-AlgVault-Card-Timestamp": timestamp,
        "X-AlgVault-Card-Signature": "sha256=" + signature,
        "Content-Type": "application/json",
    }


def test_apple_pay_readiness_fails_closed_when_env_missing(app) -> None:
    service = WalletApplePayPurchaseService(app.config)

    readiness = service.readiness()

    assert readiness["ready"] is False
    assert "APPLE_PAY_DIRECT_ENABLED must be true" in readiness["blockers"]
    assert "APPLE_PAY_CRYPTO_SALE_APPROVED must be true" in readiness["blockers"]
    assert "APPLE_PAY_DOMAIN_ASSOCIATION must be configured" in readiness["blockers"]
    assert "APPLE_PAY_BUY_ALLOWED_ASSETS_JSON must allow at least one configured asset/network" in readiness["blockers"]


def test_apple_pay_readiness_requires_domain_association(app) -> None:
    _configure_apple_pay(app)
    app.config["APPLE_PAY_DOMAIN_ASSOCIATION"] = ""
    service = WalletApplePayPurchaseService(app.config)

    readiness = service.readiness()

    assert readiness["ready"] is False
    assert "APPLE_PAY_DOMAIN_ASSOCIATION must be configured" in readiness["blockers"]


def test_apple_pay_readiness_requires_order_migration(app, monkeypatch) -> None:
    _configure_apple_pay(app)
    service = WalletApplePayPurchaseService(app.config)
    monkeypatch.setattr(service, "_orders_table_ready", lambda: False)

    readiness = service.readiness()

    assert readiness["ready"] is False
    assert "Apple Pay purchase order migration must be applied" in readiness["blockers"]


def test_apple_pay_readiness_requires_oneinch_only_for_evm_swap_assets(app) -> None:
    _configure_apple_pay(app)
    app.config["ONEINCH_API_KEY"] = ""
    service = WalletApplePayPurchaseService(app.config)

    evm_readiness = service.readiness()

    assert evm_readiness["ready"] is False
    assert "ONEINCH_API_KEY must be configured when evm_swap assets are enabled" in evm_readiness["blockers"]

    _configure_treasury_transfer_apple_pay(app)
    treasury_readiness = service.readiness()

    assert treasury_readiness["ready"] is True
    assert "ONEINCH_API_KEY must be configured when evm_swap assets are enabled" not in treasury_readiness["blockers"]
    assert sorted(treasury_readiness["allowed_assets"]) == ["BTC", "ETH", "SOL", "USDC", "USDT", "XRP"]
    assert "ALGV" not in treasury_readiness["allowed_assets"]


def test_apple_pay_quote_uses_net_from_charge_fee_math(app) -> None:
    fake = _configure_apple_pay(app)
    user, _deposit = _user_with_deposit()
    client = app.test_client()
    csrf_token = _login(client, user)

    response = client.post(
        "/wallet/apple-pay/quote",
        json={"asset": "USDT", "network": "Ethereum", "fiat_currency": "USD", "fiat_amount": 100},
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "apple-pay-100"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    order = payload["order"]
    assert order["fiat_gross_amount"] == 100.0
    assert order["treasury_fee_usd"] == 2.5
    assert order["execution_fee_usd"] == 3.15
    assert order["net_asset_amount"] == 94.35
    assert payload["apple_pay_request"]["total"]["amount"] == "100.00"
    assert WalletApplePayPurchaseOrder.query.count() == 1
    quote_call = next(call for call in fake.calls if call["method"] == "GET")
    assert quote_call["url"].endswith("/1/quote")
    assert quote_call["headers"]["Authorization"] == "Bearer oneinch-key"


def test_apple_pay_treasury_transfer_quotes_all_supported_assets_without_oneinch(app, monkeypatch) -> None:
    fake = _configure_treasury_transfer_apple_pay(app)

    for asset, network in [
        ("BTC", "Bitcoin"),
        ("SOL", "Solana"),
        ("XRP", "XRP Ledger"),
        ("ETH", "Ethereum"),
        ("USDC", "Ethereum"),
        ("USDT", "Ethereum"),
    ]:
        user, deposit = _user_with_deposit(asset, network)
        service = app.extensions["services"]["wallet_apple_pay_purchase"]
        monkeypatch.setattr(service, "_estimate_treasury_transfer_fee_usd", lambda **_kwargs: (1.0, 0.00001))
        monkeypatch.setattr(service, "_assert_treasury_inventory", lambda **_kwargs: None)

        order = service.create_quote_order(
            user=user,
            asset=asset,
            network=network,
            fiat_currency="USD",
            fiat_amount=100,
            deposit_address=deposit,
            idempotency_key=f"apple-pay-{asset.lower()}",
        )

        assert order.asset == asset
        assert order.details["quote"]["fulfillment_kind"] == "treasury_transfer"

    assert [call for call in fake.calls if call["method"] == "GET"] == []


def test_apple_pay_treasury_transfer_fails_closed_when_source_wallet_missing(app) -> None:
    _configure_treasury_transfer_apple_pay(app)
    app.config["APPLE_PAY_TREASURY_SOURCE_WALLETS"] = {}
    service = WalletApplePayPurchaseService(app.config)

    readiness = service.readiness()

    assert readiness["ready"] is False
    assert "APPLE_PAY_TREASURY_SOURCE_WALLETS_JSON must configure BTC on Bitcoin" in readiness["blockers"]


def test_apple_pay_treasury_transfer_fails_closed_when_price_missing(app, monkeypatch) -> None:
    _configure_treasury_transfer_apple_pay(app)
    user, deposit = _user_with_deposit("BTC", "Bitcoin")
    service = app.extensions["services"]["wallet_apple_pay_purchase"]
    monkeypatch.setattr(service, "_asset_usd_price", lambda asset: 0.0 if asset == "BTC" else 1.0)

    try:
        service.create_quote_order(
            user=user,
            asset="BTC",
            network="Bitcoin",
            fiat_currency="USD",
            fiat_amount=100,
            deposit_address=deposit,
            idempotency_key="apple-pay-btc-no-price",
        )
    except WalletApplePayPurchaseError as exc:
        assert exc.code == "apple_pay_price_unavailable"
    else:
        raise AssertionError("expected treasury transfer quote to fail without pricing")


def test_apple_pay_authorize_captures_gateway_without_storing_payment_token(app) -> None:
    fake = _configure_apple_pay(app)
    user, _deposit = _user_with_deposit()
    client = app.test_client()
    csrf_token = _login(client, user)
    quote_response = client.post(
        "/wallet/apple-pay/quote",
        json={"asset": "USDT", "network": "Ethereum", "fiat_currency": "USD", "fiat_amount": 100},
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "apple-pay-auth"},
    )
    order_id = quote_response.get_json()["order"]["order_id"]

    response = client.post(
        "/wallet/apple-pay/authorize",
        json={"order_id": order_id, "payment_token": {"paymentData": {"secret": "do-not-store"}}},
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "apple-pay-auth"},
    )

    assert response.status_code == 200
    order = WalletApplePayPurchaseOrder.query.filter_by(public_id=order_id).one()
    assert order.status == "fulfillment_pending"
    assert order.gateway_payment_id == "pay-123"
    assert "do-not-store" not in order.metadata_json
    gateway_call = next(call for call in fake.calls if call["method"] == "POST" and "gateway" in call["url"])
    assert gateway_call["json"]["apple_pay_token"]["paymentData"]["secret"] == "do-not-store"


def test_card_buy_readiness_and_quote_return_gateway_config_without_secret(app) -> None:
    _configure_card_buy(app)
    user, _deposit = _user_with_deposit()
    client = app.test_client()
    csrf_token = _login(client, user)

    response = client.post(
        "/wallet/card/quote",
        json={"asset": "USDT", "network": "Ethereum", "fiat_currency": "USD", "fiat_amount": 100},
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "card-quote-100"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    order = payload["order"]
    assert order["payment_method"] == "card"
    assert order["purchase_amount"] == 100.0
    assert order["algvault_fee"] == 2.5
    assert order["total_charged"] == 100.0
    assert order["estimated_receive_amount"] == 94.35
    assert order["expires_at"]
    assert payload["gateway"]["tokenization_url"] == "https://card-gateway.example/tokenize"
    assert payload["gateway"]["public_config"] == {"publishable_key": "pk_test_card"}
    assert "do-not-leak" not in json.dumps(payload)
    assert "card_test_key" not in json.dumps(payload)
    stored = WalletApplePayPurchaseOrder.query.one()
    assert stored.payment_method == "card"


def test_card_authorize_uses_tokenized_gateway_without_storing_token(app) -> None:
    fake = _configure_card_buy(app)
    user, _deposit = _user_with_deposit()
    client = app.test_client()
    csrf_token = _login(client, user)
    quote_response = client.post(
        "/wallet/card/quote",
        json={"asset": "USDT", "network": "Ethereum", "fiat_currency": "USD", "fiat_amount": 100},
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "card-auth"},
    )
    order_id = quote_response.get_json()["order"]["order_id"]

    response = client.post(
        "/wallet/card/authorize",
        json={"order_id": order_id, "gateway_payment_token": {"token": "tok_card_secret"}},
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "card-auth"},
    )

    assert response.status_code == 200
    order = WalletApplePayPurchaseOrder.query.filter_by(public_id=order_id).one()
    assert order.status == "fulfillment_pending"
    assert order.gateway_payment_id == "pay-123"
    assert "tok_card_secret" not in order.metadata_json
    gateway_call = next(call for call in fake.calls if call["method"] == "POST" and "card-gateway" in call["url"])
    assert gateway_call["headers"]["Authorization"] == "Bearer card_test_key"
    assert gateway_call["json"]["payment_method"] == "card"
    assert gateway_call["json"]["gateway_payment_token"]["token"] == "tok_card_secret"


def test_card_authorize_rejects_expired_quote(app) -> None:
    _configure_card_buy(app)
    user, _deposit = _user_with_deposit()
    client = app.test_client()
    csrf_token = _login(client, user)
    quote_response = client.post(
        "/wallet/card/quote",
        json={"asset": "USDT", "network": "Ethereum", "fiat_currency": "USD", "fiat_amount": 100},
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "card-expired"},
    )
    order_id = quote_response.get_json()["order"]["order_id"]
    order = WalletApplePayPurchaseOrder.query.filter_by(public_id=order_id).one()
    order.expires_at = datetime.utcnow() - timedelta(seconds=1)
    db.session.commit()

    response = client.post(
        "/wallet/card/authorize",
        json={"order_id": order_id, "gateway_payment_token": {"token": "tok_card_secret"}},
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "card-expired"},
    )

    assert response.status_code == 400
    assert response.get_json()["code"] == "wallet_buy_quote_expired"
    assert WalletApplePayPurchaseOrder.query.filter_by(public_id=order_id).one().status == "expired"


def test_card_gateway_webhook_updates_status_without_wallet_credit(app) -> None:
    _configure_card_buy(app)
    user, deposit = _user_with_deposit()
    order = WalletApplePayPurchaseOrder(
        user_id=user.id,
        deposit_address_id=deposit.id,
        asset="USDT",
        network="Ethereum",
        destination_address=deposit.address,
        fiat_currency="USD",
        fiat_gross_amount=100.0,
        treasury_fee_usd=2.5,
        execution_fee_usd=3.15,
        net_asset_amount=94.35,
        payment_method="card",
        status="payment_authorized",
        idempotency_key="webhook-card-buy",
        gateway_payment_id="pay-123",
    )
    db.session.add(order)
    db.session.commit()
    client = app.test_client()
    body = json.dumps({"external_order_id": order.public_id, "gateway_payment_id": "pay-123", "status": "captured"}).encode()

    invalid = client.post(
        "/api/wallet/card/gateway/webhook",
        data=body,
        headers={
            "X-AlgVault-Card-Timestamp": str(int(time.time())),
            "X-AlgVault-Card-Signature": "sha256=bad",
            "Content-Type": "application/json",
        },
    )
    valid = client.post("/api/wallet/card/gateway/webhook", data=body, headers=_signed_card_headers("card_webhook_secret", body))

    assert invalid.status_code == 401
    assert valid.status_code == 200
    refreshed = db.session.get(WalletApplePayPurchaseOrder, order.id)
    assert refreshed.status == "fulfillment_pending"
    assert WalletTransaction.query.count() == 0
    assert WalletLedgerEvent.query.count() == 0


def test_apple_pay_rejects_invalid_request_inputs(app) -> None:
    _configure_apple_pay(app)
    user, _deposit = _user_with_deposit()
    client = app.test_client()
    csrf_token = _login(client, user)

    response = client.post(
        "/wallet/apple-pay/quote",
        json={"asset": "USDT", "network": "Ethereum", "fiat_currency": "USD", "fiat_amount": 1},
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "apple-pay-too-small"},
    )

    assert response.status_code == 400
    assert response.get_json()["code"] == "apple_pay_bad_amount"


def test_apple_pay_gateway_failure_records_failed_order_without_wallet_credit(app) -> None:
    _configure_apple_pay(app, fake_session=_FakeSession(fail_gateway=True))
    user, _deposit = _user_with_deposit()
    client = app.test_client()
    csrf_token = _login(client, user)

    quote_response = client.post(
        "/wallet/apple-pay/quote",
        json={"asset": "USDT", "network": "Ethereum", "fiat_currency": "USD", "fiat_amount": 100},
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "apple-pay-gateway-failure"},
    )
    order_id = quote_response.get_json()["order"]["order_id"]
    response = client.post(
        "/wallet/apple-pay/authorize",
        json={"order_id": order_id, "payment_token": {"paymentData": {"secret": "do-not-store"}}},
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "apple-pay-gateway-failure"},
    )

    assert response.status_code == 503
    order = WalletApplePayPurchaseOrder.query.filter_by(public_id=order_id).one()
    assert order.status == "failed"
    assert "timeout" in order.failure_reason
    assert WalletTransaction.query.count() == 0
    assert WalletLedgerEvent.query.count() == 0


def test_apple_pay_gateway_webhook_updates_payment_status_without_wallet_credit(app) -> None:
    _configure_apple_pay(app)
    user, deposit = _user_with_deposit()
    order = WalletApplePayPurchaseOrder(
        user_id=user.id,
        deposit_address_id=deposit.id,
        asset="USDT",
        network="Ethereum",
        destination_address=deposit.address,
        fiat_currency="USD",
        fiat_gross_amount=100.0,
        treasury_fee_usd=2.5,
        execution_fee_usd=3.15,
        net_asset_amount=94.35,
        status="payment_authorized",
        idempotency_key="webhook-apple-pay",
        gateway_payment_id="pay-123",
    )
    db.session.add(order)
    db.session.commit()
    client = app.test_client()
    body = json.dumps({"external_order_id": order.public_id, "gateway_payment_id": "pay-123", "status": "captured"}).encode()

    invalid = client.post(
        "/api/wallet/apple-pay/gateway/webhook",
        data=body,
        headers={
            "X-AlgVault-ApplePay-Timestamp": str(int(time.time())),
            "X-AlgVault-ApplePay-Signature": "sha256=bad",
            "Content-Type": "application/json",
        },
    )
    valid = client.post("/api/wallet/apple-pay/gateway/webhook", data=body, headers=_signed_headers("gw_webhook_secret", body))

    assert invalid.status_code == 401
    assert valid.status_code == 200
    refreshed = db.session.get(WalletApplePayPurchaseOrder, order.id)
    assert refreshed.status == "fulfillment_pending"
    assert WalletTransaction.query.count() == 0
    assert WalletLedgerEvent.query.count() == 0


def test_apple_pay_treasury_transfer_fulfillment_uses_signer_without_oneinch(app) -> None:
    fake = _configure_treasury_transfer_apple_pay(app)
    user, deposit = _user_with_deposit("BTC", "Bitcoin")
    order = WalletApplePayPurchaseOrder(
        user_id=user.id,
        deposit_address_id=deposit.id,
        asset="BTC",
        network="Bitcoin",
        destination_address=deposit.address,
        fiat_currency="USD",
        fiat_gross_amount=100.0,
        treasury_fee_usd=2.5,
        execution_fee_usd=1.0,
        net_asset_amount=0.000965,
        status="fulfillment_pending",
        idempotency_key="fulfillment-btc-treasury-transfer",
    )
    order.details = {"quote": {"fulfillment_kind": "treasury_transfer"}}
    db.session.add(order)
    db.session.commit()

    result = app.extensions["services"]["wallet_apple_pay_purchase"].process_pending_orders()

    assert result["orders"][0]["status"] == "complete"
    refreshed = db.session.get(WalletApplePayPurchaseOrder, order.id)
    assert refreshed.status == "complete"
    assert refreshed.fulfillment_tx_hash == "0xuser"
    assert [call for call in fake.calls if call["method"] == "GET"] == []
    signer_call = next(call for call in fake.calls if call["method"] == "POST" and "signer" in call["url"])
    assert signer_call["json"]["fulfillment_kind"] == "treasury_transfer"
    assert signer_call["json"]["asset"] == "BTC"
    assert signer_call["json"]["source_address"] == "bc1qtreasurywallet"


def test_apple_pay_fulfillment_failure_records_failed_order(app) -> None:
    _configure_apple_pay(app, fake_session=_FakeSession(fail_swap=True))
    user, deposit = _user_with_deposit()
    order = WalletApplePayPurchaseOrder(
        user_id=user.id,
        deposit_address_id=deposit.id,
        asset="USDT",
        network="Ethereum",
        destination_address=deposit.address,
        fiat_currency="USD",
        fiat_gross_amount=100.0,
        treasury_fee_usd=2.5,
        execution_fee_usd=3.15,
        net_asset_amount=94.35,
        status="fulfillment_pending",
        idempotency_key="fulfillment-fails",
    )
    db.session.add(order)
    db.session.commit()

    result = app.extensions["services"]["wallet_apple_pay_purchase"].process_pending_orders()

    assert result["processed"] == 1
    refreshed = db.session.get(WalletApplePayPurchaseOrder, order.id)
    assert refreshed.status == "failed"
    assert "timeout" in refreshed.failure_reason
