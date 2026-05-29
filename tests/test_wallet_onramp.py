from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Any

import pytest
import requests

from app.auth import password_hash
from app.csrf import CSRF_SESSION_KEY
from app.extensions import db
from app.models import DepositAddress, User, WalletBalance, WalletLedgerEvent, WalletOnrampOrder, WalletTransaction
from app.services.wallet_onramp import WalletOnrampService


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _FakeSession:
    def __init__(self, payload: dict[str, Any] | None = None, exc: Exception | None = None) -> None:
        self.payload = payload or {
            "checkout_url": "https://checkout.example/session/abc",
            "provider_order_id": "provider-abc",
            "status": "checkout_created",
        }
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str], timeout: float) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if self.exc is not None:
            raise self.exc
        return _FakeResponse(self.payload)


def _configure_onramp(app, *, enabled: bool = True, fake_session: _FakeSession | None = None) -> _FakeSession:
    fake = fake_session or _FakeSession()
    app.config.update(
        ONRAMP_PROVIDER_ENABLED=enabled,
        ONRAMP_PROVIDER="custom_hosted",
        ONRAMP_CUSTOM_SESSION_URL="https://onramp.example/session",
        ONRAMP_CUSTOM_API_KEY="sk_test_onramp",
        ONRAMP_CUSTOM_WEBHOOK_SECRET="whsec_test_onramp",
        ONRAMP_CUSTOM_ALLOWED_ASSETS={"USDT": ["Ethereum"]},
        ONRAMP_CUSTOM_MIN_FIAT_USD=10.0,
        ONRAMP_CUSTOM_MAX_FIAT_USD=5000.0,
        ONRAMP_CUSTOM_TIMEOUT_SECONDS=7.0,
        ONRAMP_CUSTOM_WEBHOOK_TOLERANCE_SECONDS=300,
        ENABLE_LIVE_TRADING=False,
    )
    app.extensions["services"]["wallet_onramp"].session = fake
    return fake


def _user_with_deposit() -> tuple[User, DepositAddress]:
    user = User(
        username="onramp-user",
        password_hash=password_hash("password123"),
        role="user",
        totp_secret_encrypted="configured",
        two_factor_enabled_at=datetime.utcnow(),
    )
    db.session.add(user)
    db.session.flush()
    deposit = DepositAddress(
        user_id=user.id,
        asset="USDT",
        network="Ethereum",
        address="0x1111111111111111111111111111111111111111",
        version=1,
        is_active=True,
    )
    balance = WalletBalance(
        user_id=user.id,
        asset="USDT",
        available_balance=0.0,
        locked_balance=0.0,
        estimated_usd_value=0.0,
        active_deposit_address=deposit,
    )
    db.session.add_all([deposit, balance])
    db.session.commit()
    return user, deposit


def _login(client, user: User) -> str:
    csrf_token = "csrf-onramp-test"
    with client.session_transaction() as session:
        session["user_id"] = user.id
        session["two_factor_verified"] = True
        session[CSRF_SESSION_KEY] = csrf_token
    return csrf_token


def _signed_headers(secret: str, body: bytes) -> dict[str, str]:
    timestamp = str(int(time.time()))
    signature = hmac.new(secret.encode("utf-8"), timestamp.encode("utf-8") + b"." + body, hashlib.sha256).hexdigest()
    return {
        "X-AlgVault-Onramp-Timestamp": timestamp,
        "X-AlgVault-Onramp-Signature": "sha256=" + signature,
        "Content-Type": "application/json",
    }


def test_onramp_readiness_is_disabled_when_env_missing(app) -> None:
    service = WalletOnrampService(app.config)

    readiness = service.readiness()

    assert readiness["ready"] is False
    assert "ONRAMP_PROVIDER_ENABLED must be true" in readiness["blockers"]
    assert "ONRAMP_CUSTOM_SESSION_URL must be configured" in readiness["blockers"]
    assert "ONRAMP_CUSTOM_WEBHOOK_SECRET must be configured" in readiness["blockers"]


def test_onramp_readiness_requires_custom_hosted_provider(app) -> None:
    _configure_onramp(app)
    app.config["ONRAMP_PROVIDER"] = "coinbase"
    service = WalletOnrampService(app.config)

    readiness = service.readiness()

    assert readiness["ready"] is False
    assert readiness["provider"] == "custom_hosted"
    assert "ONRAMP_PROVIDER must be custom_hosted" in readiness["blockers"]
    assert "coinbase" not in json.dumps(readiness).lower()


def test_onramp_readiness_does_not_fall_back_to_legacy_or_coinbase_allowlists(app) -> None:
    _configure_onramp(app)
    app.config.update(
        ONRAMP_CUSTOM_ALLOWED_ASSETS={},
        ONRAMP_SUPPORTED_ASSETS=["USDT"],
        ONRAMP_COINBASE_ALLOWED_ASSETS={"USDT": ["Ethereum"]},
    )
    service = WalletOnrampService(app.config)

    readiness = service.readiness()

    assert readiness["ready"] is False
    assert readiness["allowed_assets"] == {}
    assert "ONRAMP_CUSTOM_ALLOWED_ASSETS_JSON must allow at least one asset/network" in readiness["blockers"]


def test_wallet_onramp_limits_do_not_drive_direct_apple_pay_wallet_ui(app) -> None:
    _configure_onramp(app)
    app.config.update(ONRAMP_CUSTOM_MIN_FIAT_USD=5.0, ONRAMP_CUSTOM_MAX_FIAT_USD=5.0)
    user, _deposit = _user_with_deposit()
    client = app.test_client()
    _login(client, user)

    response = client.get("/wallet/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "wallet-apple-pay-data" in html
    assert "wallet-polish-1" in html
    assert "wallet-card-buy-data" in html
    assert 'data-onramp-method="card"' in html
    assert 'data-onramp-method="apple_pay"' in html
    assert "Buy with Apple Pay" in html
    assert 'data-onramp-amount-preset="5"' not in html


def test_onramp_session_creates_order_with_deposit_destination_and_idempotency(app) -> None:
    fake = _configure_onramp(app)
    user, deposit = _user_with_deposit()
    client = app.test_client()
    csrf_token = _login(client, user)

    response = client.post(
        "/wallet/onramp/session",
        json={
            "asset": "USDT",
            "network": "Ethereum",
            "fiat_currency": "USD",
            "fiat_amount": 100,
            "payment_method": "card",
        },
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "buy-100-usdt"},
    )
    repeat = client.post(
        "/wallet/onramp/session",
        json={
            "asset": "USDT",
            "network": "Ethereum",
            "fiat_currency": "USD",
            "fiat_amount": 100,
            "payment_method": "card",
        },
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "buy-100-usdt"},
    )

    assert response.status_code == 200
    assert repeat.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["checkout_url"] == "https://checkout.example/session/abc"
    assert WalletOnrampOrder.query.count() == 1
    order = WalletOnrampOrder.query.one()
    assert order.deposit_address_id == deposit.id
    assert order.destination_address == deposit.address
    assert order.provider_order_id == "provider-abc"
    assert order.status == "checkout_created"
    assert len(fake.calls) == 1
    provider_payload = fake.calls[0]["json"]
    assert provider_payload["external_order_id"] == order.public_id
    assert provider_payload["user_id"] == user.id
    assert provider_payload["username"] == user.username
    assert provider_payload["asset"] == "USDT"
    assert provider_payload["network"] == "Ethereum"
    assert provider_payload["destination_address"] == deposit.address
    assert provider_payload["fiat_currency"] == "USD"
    assert provider_payload["fiat_amount"] == 100.0
    assert provider_payload["payment_method"] == "card"
    assert provider_payload["return_url"].endswith(f"/wallet/?onramp_order={order.public_id}")
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer sk_test_onramp"


def test_onramp_session_fails_closed_when_provider_is_not_custom_hosted(app) -> None:
    fake = _configure_onramp(app)
    app.config["ONRAMP_PROVIDER"] = "coinbase"
    user, _deposit = _user_with_deposit()
    client = app.test_client()
    csrf_token = _login(client, user)

    response = client.post(
        "/wallet/onramp/session",
        json={
            "asset": "USDT",
            "network": "Ethereum",
            "fiat_currency": "USD",
            "fiat_amount": 125,
            "payment_method": "apple_pay",
        },
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "bad-provider-buy"},
    )

    assert response.status_code == 503
    assert response.get_json()["code"] == "onramp_not_ready"
    assert WalletOnrampOrder.query.count() == 0
    assert fake.calls == []


def test_onramp_webhook_fails_closed_when_provider_is_not_custom_hosted(app) -> None:
    _configure_onramp(app)
    app.config["ONRAMP_PROVIDER"] = "coinbase"
    service = WalletOnrampService(app.config)
    body = b'{"external_order_id":"order"}'

    assert service.verify_webhook_signature(body, _signed_headers("whsec_test_onramp", body)) is False


@pytest.mark.parametrize(
    ("body", "expected_code"),
    [
        ({"asset": "DOGE", "network": "Ethereum", "fiat_amount": 100, "payment_method": "card"}, "onramp_bad_asset"),
        ({"asset": "USDT", "network": "Solana", "fiat_amount": 100, "payment_method": "card"}, "onramp_bad_network"),
        ({"asset": "USDT", "network": "Ethereum", "fiat_amount": 1, "payment_method": "card"}, "onramp_bad_amount"),
        ({"asset": "USDT", "network": "Ethereum", "fiat_amount": 100, "payment_method": "cash"}, "onramp_bad_payment_method"),
    ],
)
def test_onramp_session_rejects_invalid_requests(app, body: dict[str, Any], expected_code: str) -> None:
    _configure_onramp(app)
    user, _deposit = _user_with_deposit()
    client = app.test_client()
    csrf_token = _login(client, user)

    response = client.post(
        "/wallet/onramp/session",
        json={"fiat_currency": "USD", **body},
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": f"bad-{expected_code}"},
    )

    assert response.status_code == 400
    assert response.get_json()["code"] == expected_code


def test_onramp_provider_failure_records_failed_order_without_success(app) -> None:
    _configure_onramp(app, fake_session=_FakeSession(exc=requests.Timeout("session timeout")))
    user, _deposit = _user_with_deposit()
    client = app.test_client()
    csrf_token = _login(client, user)

    response = client.post(
        "/wallet/onramp/session",
        json={
            "asset": "USDT",
            "network": "Ethereum",
            "fiat_currency": "USD",
            "fiat_amount": 100,
            "payment_method": "apple_pay",
        },
        headers={"X-CSRF-Token": csrf_token, "Idempotency-Key": "provider-failure"},
    )

    assert response.status_code == 503
    assert response.get_json()["code"] == "onramp_provider_failed"
    order = WalletOnrampOrder.query.one()
    assert order.status == "failed"
    assert "timeout" in order.failure_reason
    assert WalletTransaction.query.count() == 0
    assert WalletLedgerEvent.query.count() == 0


def test_onramp_webhook_signature_and_status_update_never_credit_balance(app) -> None:
    _configure_onramp(app)
    user, deposit = _user_with_deposit()
    order = WalletOnrampOrder(
        user_id=user.id,
        deposit_address_id=deposit.id,
        asset="USDT",
        network="Ethereum",
        destination_address=deposit.address,
        fiat_currency="USD",
        fiat_amount=100.0,
        payment_method="card",
        provider="custom_hosted",
        provider_order_id="provider-abc",
        checkout_url="https://checkout.example/session/abc",
        status="checkout_created",
        idempotency_key="webhook-order",
    )
    db.session.add(order)
    db.session.commit()
    client = app.test_client()
    body = json.dumps(
        {
            "external_order_id": order.public_id,
            "provider_order_id": "provider-abc",
            "status": "completed",
            "tx_hash": "0xabc",
            "amount_crypto": "100.0",
        }
    ).encode("utf-8")

    invalid = client.post(
        "/api/wallet/onramp/custom/webhook",
        data=body,
        headers={
            "X-AlgVault-Onramp-Timestamp": str(int(time.time())),
            "X-AlgVault-Onramp-Signature": "sha256=bad",
            "Content-Type": "application/json",
        },
    )
    valid = client.post(
        "/api/wallet/onramp/custom/webhook",
        data=body,
        headers=_signed_headers("whsec_test_onramp", body),
    )

    assert invalid.status_code == 401
    assert valid.status_code == 200
    refreshed = db.session.get(WalletOnrampOrder, order.id)
    assert refreshed.status == "complete"
    assert refreshed.completed_at is not None
    assert WalletTransaction.query.count() == 0
    assert WalletLedgerEvent.query.count() == 0
