"""Hosted wallet on-ramp session tracking.

The provider may collect card or Apple Pay payment details, but AlgVault never
does. A completed on-ramp order is not a wallet credit; existing custody and
on-chain reconciliation remain the only balance-crediting path.
"""

from __future__ import annotations

import hashlib
import hmac
import time
import uuid
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import requests

from ..extensions import db
from ..models import DepositAddress, User, WalletOnrampOrder

PAYMENT_METHODS = {"card": "Card", "apple_pay": "Apple Pay"}
TERMINAL_STATUSES = {"complete", "failed", "canceled", "expired"}


class WalletOnrampError(ValueError):
    """Raised when an on-ramp request cannot be safely created."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "onramp_error",
        order: WalletOnrampOrder | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.order = order


class WalletOnrampService:
    """Creates and tracks custom hosted on-ramp checkout sessions."""

    def __init__(self, config: dict[str, Any], session: requests.Session | None = None) -> None:
        self.config = config
        self.session = session or requests.Session()

    def readiness(self) -> dict[str, Any]:
        blockers: list[str] = []
        configured_provider = self.configured_provider
        if not self.provider_enabled:
            blockers.append("ONRAMP_PROVIDER_ENABLED must be true")
        if configured_provider != "custom_hosted":
            blockers.append("ONRAMP_PROVIDER must be custom_hosted")
        if not self.session_url:
            blockers.append("ONRAMP_CUSTOM_SESSION_URL must be configured")
        elif not self._valid_checkout_origin(self.session_url):
            blockers.append("ONRAMP_CUSTOM_SESSION_URL must be an http or https URL")
        if not self.api_key:
            blockers.append("ONRAMP_CUSTOM_API_KEY must be configured")
        if not self.webhook_secret:
            blockers.append("ONRAMP_CUSTOM_WEBHOOK_SECRET must be configured")
        if not self.allowed_assets:
            blockers.append("ONRAMP_CUSTOM_ALLOWED_ASSETS_JSON must allow at least one asset/network")
        min_amount = self.min_fiat_usd
        max_amount = self.max_fiat_usd
        if min_amount <= 0:
            blockers.append("ONRAMP_CUSTOM_MIN_FIAT_USD must be greater than zero")
        if max_amount < min_amount:
            blockers.append("ONRAMP_CUSTOM_MAX_FIAT_USD must be greater than or equal to minimum")
        return {
            "ready": not blockers,
            "enabled": self.provider_enabled,
            "provider": self.provider,
            "blockers": blockers,
            "payment_methods": [{"key": key, "label": label} for key, label in PAYMENT_METHODS.items()],
            "fiat_currency": "USD",
            "min_fiat_usd": min_amount,
            "max_fiat_usd": max_amount,
            "allowed_assets": self.allowed_assets,
        }

    def create_session(
        self,
        *,
        user: User,
        asset: str,
        network: str,
        fiat_currency: str,
        fiat_amount: float,
        payment_method: str,
        deposit_address: DepositAddress,
        idempotency_key: str | None,
        return_url: str,
        cancel_url: str,
        webhook_url: str,
        client_ip: str | None = None,
    ) -> WalletOnrampOrder:
        asset_key = self._normalize_asset(asset)
        network_name = str(network or "").strip()
        payment_key = str(payment_method or "").strip().lower()
        currency = str(fiat_currency or "USD").strip().upper()
        amount = self._coerce_amount(fiat_amount)
        idem = self._normalize_idempotency_key(idempotency_key)
        idem = f"u{user.id}:{idem}"[:220]

        self._validate_request(asset_key, network_name, currency, amount, payment_key, deposit_address)
        existing = WalletOnrampOrder.query.filter_by(user_id=user.id, idempotency_key=idem).one_or_none()
        if existing is not None:
            if existing.checkout_url and existing.status not in TERMINAL_STATUSES:
                return existing
            raise WalletOnrampError("This buy request already finished. Start a new buy request.", code="onramp_duplicate_terminal")

        order = WalletOnrampOrder(
            user_id=user.id,
            deposit_address_id=deposit_address.id,
            asset=asset_key,
            network=network_name,
            destination_address=str(deposit_address.address or "").strip(),
            fiat_currency=currency,
            fiat_amount=amount,
            payment_method=payment_key,
            provider=self.provider,
            status="created",
            idempotency_key=idem,
        )
        order.details = {"return_url": return_url, "cancel_url": cancel_url}
        db.session.add(order)
        db.session.flush()

        return_url = self._order_url(return_url, order)
        cancel_url = self._order_url(cancel_url, order)
        order.details = {"return_url": return_url, "cancel_url": cancel_url}
        payload = self._session_payload(
            user,
            order,
            return_url=return_url,
            cancel_url=cancel_url,
            webhook_url=webhook_url,
            client_ip=client_ip,
        )
        try:
            provider_payload = self._request_hosted_session(payload, idem)
        except Exception as exc:  # noqa: BLE001
            order.status = "failed"
            order.failure_reason = self._safe_error(exc)
            details = order.details
            details["provider_error"] = order.failure_reason
            order.details = details
            db.session.flush()
            raise WalletOnrampError(
                "Card checkout is temporarily unavailable. Try again later.",
                code="onramp_provider_failed",
                order=order,
            ) from exc

        checkout_url = str(provider_payload.get("checkout_url") or provider_payload.get("paymentLink") or "").strip()
        provider_order_id = str(provider_payload.get("provider_order_id") or provider_payload.get("order_id") or "").strip()
        if not checkout_url or not self._valid_checkout_origin(checkout_url):
            order.status = "failed"
            order.failure_reason = "Provider returned an invalid checkout URL."
            db.session.flush()
            raise WalletOnrampError(
                "Checkout URL was not returned by the provider.",
                code="onramp_invalid_checkout",
                order=order,
            )

        order.checkout_url = checkout_url
        order.provider_order_id = provider_order_id or None
        order.status = self._normalize_status(provider_payload.get("status") or "checkout_created")
        order.expires_at = self._parse_datetime(provider_payload.get("expires_at"))
        details = order.details
        details["provider_response"] = {
            "status": provider_payload.get("status"),
            "expires_at": provider_payload.get("expires_at"),
        }
        order.details = details
        db.session.flush()
        return order

    def handle_webhook(self, payload: dict[str, Any]) -> WalletOnrampOrder:
        external_order_id = str(payload.get("external_order_id") or payload.get("order_id") or "").strip()
        provider_order_id = str(payload.get("provider_order_id") or payload.get("providerOrderId") or "").strip()
        order = None
        if external_order_id:
            order = WalletOnrampOrder.query.filter_by(public_id=external_order_id).one_or_none()
        if order is None and provider_order_id:
            order = WalletOnrampOrder.query.filter_by(provider_order_id=provider_order_id).one_or_none()
        if order is None:
            raise WalletOnrampError("On-ramp order was not found.", code="onramp_order_not_found")

        status = self._normalize_status(payload.get("status") or order.status)
        order.status = status
        if provider_order_id and not order.provider_order_id:
            order.provider_order_id = provider_order_id
        if status in {"failed", "canceled", "expired"}:
            order.failure_reason = str(payload.get("failure_reason") or payload.get("reason") or order.failure_reason or "").strip()
        if status == "complete" and order.completed_at is None:
            order.completed_at = datetime.utcnow()
        details = order.details
        details["last_webhook"] = self._webhook_details(payload)
        order.details = details
        db.session.flush()
        return order

    def verify_webhook_signature(self, body: bytes, headers: Any) -> bool:
        if self.configured_provider != "custom_hosted":
            return False
        secret = self.webhook_secret
        if not secret:
            return False
        timestamp = str(
            headers.get("X-AlgVault-Onramp-Timestamp") or headers.get("X-Onramp-Timestamp") or headers.get("X-Timestamp") or ""
        ).strip()
        supplied = str(
            headers.get("X-AlgVault-Onramp-Signature") or headers.get("X-Onramp-Signature") or headers.get("X-Signature") or ""
        ).strip()
        if supplied.startswith("sha256="):
            supplied = supplied.removeprefix("sha256=")
        try:
            ts = int(timestamp)
        except (TypeError, ValueError):
            return False
        if abs(int(time.time()) - ts) > self.webhook_tolerance_seconds:
            return False
        expected = hmac.new(secret.encode("utf-8"), timestamp.encode("utf-8") + b"." + body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, supplied)

    def status_payload(self, order: WalletOnrampOrder) -> dict[str, Any]:
        return {
            "order_id": order.public_id,
            "asset": order.asset,
            "network": order.network,
            "fiat_currency": order.fiat_currency,
            "fiat_amount": order.fiat_amount,
            "payment_method": order.payment_method,
            "status": order.status,
            "checkout_url": order.checkout_url if order.status not in TERMINAL_STATUSES else "",
            "failure_reason": order.failure_reason,
            "created_at": order.created_at.isoformat() if order.created_at else "",
            "updated_at": order.updated_at.isoformat() if order.updated_at else "",
        }

    @property
    def provider(self) -> str:
        return "custom_hosted"

    @property
    def configured_provider(self) -> str:
        return str(self.config.get("ONRAMP_PROVIDER", "custom_hosted") or "custom_hosted").strip().lower()

    @property
    def provider_enabled(self) -> bool:
        configured = self.config.get("ONRAMP_PROVIDER_ENABLED")
        if configured is not None:
            return bool(configured)
        return bool(self.config.get("ONRAMP_ENABLED", False))

    @property
    def session_url(self) -> str:
        return str(self.config.get("ONRAMP_CUSTOM_SESSION_URL", "") or "").strip()

    @property
    def api_key(self) -> str:
        return str(self.config.get("ONRAMP_CUSTOM_API_KEY", "") or "").strip()

    @property
    def webhook_secret(self) -> str:
        return str(self.config.get("ONRAMP_CUSTOM_WEBHOOK_SECRET", "") or "").strip()

    @property
    def min_fiat_usd(self) -> float:
        return float(self.config.get("ONRAMP_CUSTOM_MIN_FIAT_USD", 10.0) or 10.0)

    @property
    def max_fiat_usd(self) -> float:
        return float(self.config.get("ONRAMP_CUSTOM_MAX_FIAT_USD", 5_000.0) or 5_000.0)

    @property
    def webhook_tolerance_seconds(self) -> int:
        return int(self.config.get("ONRAMP_CUSTOM_WEBHOOK_TOLERANCE_SECONDS", 300) or 300)

    @property
    def allowed_assets(self) -> dict[str, list[str]]:
        raw = self.config.get("ONRAMP_CUSTOM_ALLOWED_ASSETS")
        return self._allowed_assets_from_config(raw)

    def _allowed_assets_from_config(self, raw: Any) -> dict[str, list[str]]:
        allowed: dict[str, list[str]] = {}
        if not isinstance(raw, dict):
            return allowed
        for asset, value in raw.items():
            asset_key = self._normalize_asset(asset)
            networks: list[str] = []
            if isinstance(value, list):
                networks = [str(item).strip() for item in value if str(item).strip()]
            elif isinstance(value, dict):
                configured = value.get("networks") or value.get("network")
                if isinstance(configured, list):
                    networks = [str(item).strip() for item in configured if str(item).strip()]
                elif configured:
                    networks = [str(configured).strip()]
            elif value:
                networks = [str(value).strip()]
            if asset_key and networks:
                allowed[asset_key] = list(dict.fromkeys(networks))
        return allowed

    def _validate_request(
        self,
        asset: str,
        network: str,
        fiat_currency: str,
        fiat_amount: float,
        payment_method: str,
        deposit_address: DepositAddress,
    ) -> None:
        readiness = self.readiness()
        if not readiness["ready"]:
            raise WalletOnrampError("Buy with card is not available until on-ramp provider readiness is complete.", code="onramp_not_ready")
        if payment_method not in PAYMENT_METHODS:
            raise WalletOnrampError("Choose card or Apple Pay.", code="onramp_bad_payment_method")
        if fiat_currency != "USD":
            raise WalletOnrampError("Only USD card buys are currently supported.", code="onramp_bad_currency")
        if fiat_amount < self.min_fiat_usd or fiat_amount > self.max_fiat_usd:
            raise WalletOnrampError(
                f"Enter a USD amount between ${self.min_fiat_usd:.0f} and ${self.max_fiat_usd:.0f}.",
                code="onramp_bad_amount",
            )
        networks = self.allowed_assets.get(asset, [])
        if network not in networks:
            raise WalletOnrampError("This asset/network is not enabled for card buys.", code="onramp_asset_not_allowed")
        if not str(getattr(deposit_address, "address", "") or "").strip():
            raise WalletOnrampError("A destination deposit address is required before checkout.", code="onramp_no_deposit_address")

    def _request_hosted_session(self, payload: dict[str, Any], idempotency_key: str) -> dict[str, Any]:
        response = self.session.post(
            self.session_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
            },
            timeout=float(self.config.get("ONRAMP_CUSTOM_TIMEOUT_SECONDS", 12.0) or 12.0),
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise WalletOnrampError("Provider returned an invalid response.", code="onramp_provider_response")
        return data

    def _session_payload(
        self,
        user: User,
        order: WalletOnrampOrder,
        *,
        return_url: str,
        cancel_url: str,
        webhook_url: str,
        client_ip: str | None = None,
    ) -> dict[str, Any]:
        return {
            "external_order_id": order.public_id,
            "user_id": order.user_id,
            "username": user.username,
            "asset": order.asset,
            "network": order.network,
            "destination_address": order.destination_address,
            "fiat_currency": order.fiat_currency,
            "fiat_amount": order.fiat_amount,
            "payment_method": order.payment_method,
            "return_url": return_url,
            "cancel_url": cancel_url,
            "webhook_url": webhook_url,
            "client_ip": client_ip,
        }

    @staticmethod
    def _order_url(value: str, order: WalletOnrampOrder) -> str:
        return str(value or "").replace("__ORDER_ID__", order.public_id)

    @staticmethod
    def _normalize_asset(value: Any) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _normalize_idempotency_key(value: str | None) -> str:
        raw = str(value or "").strip()
        return raw[:220] if raw else f"wallet-onramp-{uuid.uuid4().hex}"

    @staticmethod
    def _coerce_amount(value: Any) -> float:
        try:
            amount = float(value)
        except (TypeError, ValueError):
            raise WalletOnrampError("Enter a valid USD amount.", code="onramp_bad_amount") from None
        if amount <= 0:
            raise WalletOnrampError("Enter a USD amount greater than zero.", code="onramp_bad_amount")
        return round(amount, 2)

    @staticmethod
    def _normalize_status(value: Any) -> str:
        raw = str(value or "").strip().lower().replace("-", "_")
        if raw in {"success", "succeeded", "completed", "settled"}:
            return "complete"
        if raw in {"cancelled"}:
            return "canceled"
        if raw in {"created", "checkout_created", "pending", "processing", "complete", "failed", "canceled", "expired"}:
            return raw
        return "pending"

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None

    @staticmethod
    def _valid_checkout_origin(value: str) -> bool:
        try:
            parsed = urlparse(str(value or ""))
        except Exception:  # noqa: BLE001
            return False
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        text = str(exc)
        for marker in ("Authorization", "Bearer", "api_key", "secret", "token", "KEY_SECRET", "API_KEY"):
            text = text.replace(marker, "[redacted]")
        return text[:280]

    @staticmethod
    def _webhook_details(payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "external_order_id",
            "provider_order_id",
            "status",
            "asset",
            "network",
            "destination_address",
            "tx_hash",
            "transaction_hash",
            "amount_crypto",
            "fiat_amount",
            "fiat_currency",
            "failure_reason",
            "reason",
        }
        return {key: payload.get(key) for key in allowed if key in payload}
