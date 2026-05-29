"""Direct wallet card purchase support.

AlgVault owns the pre-checkout quote and order tracking. Raw card credentials
are never stored by AlgVault; tokenized Apple Pay or card gateway references are
passed once to the configured payment gateway, then discarded. Wallet balances
are not credited by payment events alone.
"""

from __future__ import annotations

import hashlib
import hmac
import tempfile
import time
import uuid
from contextlib import ExitStack
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from flask import current_app, has_app_context
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import SQLAlchemyError

from ..extensions import db
from ..models import (
    DepositAddress,
    User,
    WalletApplePayPurchaseOrder,
    WalletLedgerEvent,
    WalletTransaction,
    WorkerLease,
)
from ..runtime import get_current_mode, market_mode_for
from .vault_allocation_assets import asset_usd_price as shared_asset_usd_price
from .wallet_custody import BitcoinWalletAdapter, EvmWalletAdapter, SolanaWalletAdapter, WalletChainAdapter, XrpWalletAdapter

APPLE_PAY_METHOD = "apple_pay"
CARD_METHOD = "card"
BUY_PAYMENT_METHODS = {CARD_METHOD: "Card", APPLE_PAY_METHOD: "Apple Pay"}
APPLE_PAY_TERMINAL_STATUSES = {"complete", "failed", "canceled", "expired", "refunded"}
APPLE_PAY_PROCESSING_STATUSES = {
    "payment_authorized",
    "payment_captured",
    "fulfillment_pending",
    "swap_submitted",
    "transfer_submitted",
}
APPLE_PAY_FULFILLMENT_KINDS = {"evm_swap", "treasury_transfer"}
APPLE_PAY_EXCLUDED_ASSETS = {"ALGV"}
APPLE_PAY_TREASURY_TRANSFER_ASSETS = {"BTC", "ETH", "SOL", "USDC", "USDT", "XRP"}
APPLE_PAY_EVM_ASSETS = {"ETH", "USDC", "USDT"}
CARD_BUY_ASSETS = {"ETH", "USDC", "USDT"}
TREASURY_FEE_ASSET = "ETH"
WALLET_BUY_PLATFORM_FEE_BPS = 250.0
APPLE_PAY_NETWORK_CANONICAL = {
    "amex": "amex",
    "discover": "discover",
    "mastercard": "masterCard",
    "visa": "visa",
}
EVM_CHAIN_IDS = {
    "ethereum": 1,
    "base": 8453,
    "arbitrum": 42161,
    "optimism": 10,
    "polygon": 137,
    "avalanche": 43114,
    "bsc": 56,
}


class WalletApplePayPurchaseError(ValueError):
    """Raised when a direct Apple Pay purchase cannot be safely advanced."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "apple_pay_purchase_error",
        order: WalletApplePayPurchaseOrder | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.order = order


class WalletApplePayPurchaseService:
    """Quotes and tracks direct card and Apple Pay crypto purchases."""

    def __init__(self, config: dict[str, Any], session: requests.Session | None = None) -> None:
        self.config = config
        self.session = session or requests.Session()

    def readiness(self) -> dict[str, Any]:
        blockers: list[str] = []
        allowed_assets = self.allowed_assets
        if not self.enabled:
            blockers.append("APPLE_PAY_DIRECT_ENABLED must be true")
        if not self.crypto_sale_approved:
            blockers.append("APPLE_PAY_CRYPTO_SALE_APPROVED must be true")
        if not self.merchant_id:
            blockers.append("APPLE_PAY_MERCHANT_ID must be configured")
        if not self.display_name:
            blockers.append("APPLE_PAY_DISPLAY_NAME must be configured")
        if not self.domain_name:
            blockers.append("APPLE_PAY_DOMAIN must be configured")
        if not self.domain_association:
            blockers.append("APPLE_PAY_DOMAIN_ASSOCIATION must be configured")
        if not self._has_merchant_certificate():
            blockers.append("Apple merchant identity certificate must be configured")
        if not self._has_merchant_key():
            blockers.append("Apple merchant identity private key must be configured")
        if not self.gateway_authorize_url:
            blockers.append("APPLE_PAY_GATEWAY_AUTHORIZE_URL must be configured")
        elif not self._valid_url(self.gateway_authorize_url):
            blockers.append("APPLE_PAY_GATEWAY_AUTHORIZE_URL must be an http or https URL")
        if not self.gateway_api_key:
            blockers.append("APPLE_PAY_GATEWAY_API_KEY must be configured")
        if not self.gateway_webhook_secret:
            blockers.append("APPLE_PAY_GATEWAY_WEBHOOK_SECRET must be configured")
        if self._uses_fulfillment_kind(allowed_assets, "evm_swap") and not self._oneinch_provider_configured():
            blockers.append("ONEINCH_API_KEY must be configured when evm_swap assets are enabled")
        if not allowed_assets:
            blockers.append("APPLE_PAY_BUY_ALLOWED_ASSETS_JSON must allow at least one configured asset/network")
        blockers.extend(self._allowed_asset_readiness_blockers(allowed_assets))
        if not self.treasury_signer_url:
            blockers.append("APPLE_PAY_TREASURY_SIGNER_URL must be configured")
        if not self.treasury_signer_token:
            blockers.append("APPLE_PAY_TREASURY_SIGNER_TOKEN must be configured")
        if self.enabled and not self._orders_table_ready():
            blockers.append("Apple Pay purchase order migration must be applied")
        if self.min_fiat_usd <= 0:
            blockers.append("APPLE_PAY_MIN_FIAT_USD must be greater than zero")
        if self.max_fiat_usd < self.min_fiat_usd:
            blockers.append("APPLE_PAY_MAX_FIAT_USD must be greater than or equal to minimum")
        if not bool(self.config.get("WORKER_APPLE_PAY_FULFILLMENT_ENABLED", True)):
            blockers.append("WORKER_APPLE_PAY_FULFILLMENT_ENABLED must be true")
        worker_status = self._apple_pay_fulfillment_worker_status(payment_method=APPLE_PAY_METHOD)
        if worker_status.get("required") and not worker_status.get("recent"):
            blockers.append("apple_pay_fulfillment worker heartbeat must be recent")
        return {
            "ready": not blockers,
            "enabled": self.enabled,
            "provider": "direct_apple_pay",
            "blockers": blockers,
            "payment_methods": [{"key": APPLE_PAY_METHOD, "label": "Apple Pay"}],
            "fiat_currency": "USD",
            "country_code": self.country_code,
            "merchant_capabilities": ["supports3DS"],
            "supported_networks": self.apple_pay_networks,
            "min_fiat_usd": self.min_fiat_usd,
            "max_fiat_usd": self.max_fiat_usd,
            "treasury_fee_bps": self.treasury_fee_bps,
            "allowed_assets": {asset: list(networks.keys()) for asset, networks in allowed_assets.items()},
            "fulfillment_worker": worker_status,
            "card_buy": self.card_readiness(),
        }

    def card_readiness(self) -> dict[str, Any]:
        blockers: list[str] = []
        allowed_assets, asset_blockers = self._card_allowed_assets(self.allowed_assets)
        if not self.card_enabled:
            blockers.append("CARD_BUY_ENABLED must be true")
        if not self.crypto_sale_approved:
            blockers.append("APPLE_PAY_CRYPTO_SALE_APPROVED must be true")
        if not self.card_gateway_tokenization_url:
            blockers.append("CARD_GATEWAY_TOKENIZATION_URL must be configured")
        elif not self._valid_url(self.card_gateway_tokenization_url):
            blockers.append("CARD_GATEWAY_TOKENIZATION_URL must be an http or https URL")
        if not self.card_gateway_authorize_url:
            blockers.append("CARD_GATEWAY_AUTHORIZE_URL must be configured")
        elif not self._valid_url(self.card_gateway_authorize_url):
            blockers.append("CARD_GATEWAY_AUTHORIZE_URL must be an http or https URL")
        if not self.card_gateway_api_key:
            blockers.append("CARD_GATEWAY_API_KEY must be configured")
        if not self.card_gateway_webhook_secret:
            blockers.append("CARD_GATEWAY_WEBHOOK_SECRET must be configured")
        if not self.card_gateway_public_config:
            blockers.append("CARD_GATEWAY_PUBLIC_CONFIG_JSON must expose gateway public configuration")
        if not allowed_assets:
            blockers.append("APPLE_PAY_BUY_ALLOWED_ASSETS_JSON must allow at least one EVM card-buy asset/network")
        blockers.extend(asset_blockers)
        blockers.extend(self._allowed_asset_readiness_blockers(allowed_assets))
        if not self.treasury_fee_address:
            blockers.append("APPLE_PAY_TREASURY_FEE_ADDRESS must be configured")
        if not self.treasury_signer_url:
            blockers.append("APPLE_PAY_TREASURY_SIGNER_URL must be configured")
        if not self.treasury_signer_token:
            blockers.append("APPLE_PAY_TREASURY_SIGNER_TOKEN must be configured")
        if abs(float(self.treasury_fee_bps or 0.0) - WALLET_BUY_PLATFORM_FEE_BPS) > 1e-9:
            blockers.append("WALLET_BUY_PLATFORM_FEE_BPS must be 250")
        if self._eth_usd_price_context()[0] <= 0:
            blockers.append("ETH treasury fee pricing must be available")
        if not bool(self.config.get("WORKER_APPLE_PAY_FULFILLMENT_ENABLED", True)):
            blockers.append("WORKER_APPLE_PAY_FULFILLMENT_ENABLED must be true")
        worker_status = self._apple_pay_fulfillment_worker_status(payment_method=CARD_METHOD)
        if worker_status.get("required") and not worker_status.get("recent"):
            blockers.append("apple_pay_fulfillment worker heartbeat must be recent")
        if self.card_enabled and not self._orders_table_ready():
            blockers.append("Wallet purchase order migration must be applied")
        if self.min_fiat_usd <= 0:
            blockers.append("APPLE_PAY_MIN_FIAT_USD must be greater than zero")
        if self.max_fiat_usd < self.min_fiat_usd:
            blockers.append("APPLE_PAY_MAX_FIAT_USD must be greater than or equal to minimum")
        return {
            "ready": not blockers,
            "enabled": self.card_enabled,
            "provider": "custom_card_gateway",
            "blockers": blockers,
            "payment_methods": [{"key": CARD_METHOD, "label": BUY_PAYMENT_METHODS[CARD_METHOD]}],
            "fiat_currency": "USD",
            "min_fiat_usd": self.min_fiat_usd,
            "max_fiat_usd": self.max_fiat_usd,
            "treasury_fee_bps": self.treasury_fee_bps,
            "allowed_assets": {asset: list(networks.keys()) for asset, networks in allowed_assets.items()},
            "gateway": self.card_gateway_client_config(),
            "fulfillment_worker": worker_status,
        }

    def create_quote_order(
        self,
        *,
        user: User,
        asset: str,
        network: str,
        fiat_currency: str,
        fiat_amount: float,
        deposit_address: DepositAddress,
        idempotency_key: str | None,
        payment_method: str = APPLE_PAY_METHOD,
    ) -> WalletApplePayPurchaseOrder:
        asset_key = self._normalize_asset(asset)
        network_name = str(network or "").strip()
        currency = str(fiat_currency or "USD").strip().upper()
        gross_amount = self._coerce_money(fiat_amount)
        payment_key = self._normalize_payment_method(payment_method)
        idem = f"u{user.id}:{self._normalize_idempotency_key(idempotency_key)}"[:220]

        self._validate_quote_request(asset_key, network_name, currency, gross_amount, deposit_address, payment_method=payment_key)
        existing = WalletApplePayPurchaseOrder.query.filter_by(user_id=user.id, idempotency_key=idem).one_or_none()
        if existing is not None:
            if existing.status in {"quoted", *APPLE_PAY_PROCESSING_STATUSES}:
                return existing
            raise WalletApplePayPurchaseError(
                f"This {self._payment_method_label(payment_key)} request already finished. Start a new buy request.",
                code="apple_pay_duplicate_terminal",
            )

        quote = self._quote_payload(
            asset=asset_key,
            network=network_name,
            fiat_currency=currency,
            fiat_gross_amount=gross_amount,
            destination_address=str(deposit_address.address or "").strip(),
        )
        order = WalletApplePayPurchaseOrder(
            user_id=user.id,
            deposit_address_id=deposit_address.id,
            asset=asset_key,
            network=network_name,
            destination_address=str(deposit_address.address or "").strip(),
            fiat_currency=currency,
            fiat_gross_amount=quote["fiat_gross_amount"],
            treasury_fee_usd=quote["treasury_fee_usd"],
            execution_fee_usd=quote["execution_fee_usd"],
            net_asset_amount=quote["net_asset_amount"],
            payment_method=payment_key,
            oneinch_quote_id=quote.get("oneinch_quote_id") or None,
            status="quoted",
            idempotency_key=idem,
            expires_at=datetime.utcnow() + timedelta(seconds=self.quote_ttl_seconds),
        )
        order.details = {"quote": quote}
        db.session.add(order)
        db.session.flush()
        return order

    def create_card_quote_order(
        self,
        *,
        user: User,
        asset: str,
        network: str,
        fiat_currency: str,
        fiat_amount: float,
        deposit_address: DepositAddress,
        idempotency_key: str | None,
    ) -> WalletApplePayPurchaseOrder:
        return self.create_quote_order(
            user=user,
            asset=asset,
            network=network,
            fiat_currency=fiat_currency,
            fiat_amount=fiat_amount,
            deposit_address=deposit_address,
            idempotency_key=idempotency_key,
            payment_method=CARD_METHOD,
        )

    def merchant_session(self, *, validation_url: str, initiative_context: str | None = None) -> dict[str, Any]:
        if not self.readiness()["ready"]:
            raise WalletApplePayPurchaseError(
                "Apple Pay is not available until direct purchase readiness is complete.",
                code="apple_pay_not_ready",
            )
        if not self._valid_url(validation_url):
            raise WalletApplePayPurchaseError("Apple merchant validation URL is invalid.", code="apple_pay_bad_validation_url")
        parsed = urlparse(validation_url)
        if not parsed.netloc.endswith("apple.com"):
            raise WalletApplePayPurchaseError(
                "Apple merchant validation must use an Apple validation URL.", code="apple_pay_bad_validation_url"
            )
        payload = {
            "merchantIdentifier": self.merchant_id,
            "displayName": self.display_name,
            "initiative": "web",
            "initiativeContext": initiative_context or self.domain_name,
        }
        return self._request_apple_merchant_session(validation_url, payload)

    def authorize_payment(
        self,
        *,
        user: User,
        order_id: str,
        payment_token: dict[str, Any],
        idempotency_key: str | None,
        payment_method: str | None = None,
    ) -> WalletApplePayPurchaseOrder:
        order = WalletApplePayPurchaseOrder.query.filter_by(public_id=str(order_id or "").strip(), user_id=user.id).one_or_none()
        if order is None:
            raise WalletApplePayPurchaseError("Apple Pay order was not found.", code="apple_pay_order_not_found")
        expected_method = self._normalize_payment_method(payment_method or order.payment_method or APPLE_PAY_METHOD)
        if self._normalize_payment_method(order.payment_method) != expected_method:
            raise WalletApplePayPurchaseError(
                f"{self._payment_method_label(expected_method)} order was not found.",
                code="apple_pay_order_not_found",
            )
        if order.status != "quoted":
            if order.status in APPLE_PAY_PROCESSING_STATUSES | APPLE_PAY_TERMINAL_STATUSES:
                return order
            raise WalletApplePayPurchaseError(
                f"{self._payment_method_label(expected_method)} order is not ready for authorization.",
                code="apple_pay_bad_order_state",
                order=order,
            )
        if self._quote_expired(order):
            order.status = "expired"
            db.session.flush()
            raise WalletApplePayPurchaseError(
                "This quote expired. Refresh the quote before authorizing payment.",
                code="wallet_buy_quote_expired",
                order=order,
            )
        if not isinstance(payment_token, dict) or not payment_token:
            raise WalletApplePayPurchaseError(
                f"{self._payment_method_label(expected_method)} payment token is required.",
                code="apple_pay_missing_token",
                order=order,
            )
        self._revalidate_order_quote(order)
        readiness = self.card_readiness() if expected_method == CARD_METHOD else self.readiness()
        if not readiness["ready"]:
            raise WalletApplePayPurchaseError(
                f"{self._payment_method_label(expected_method)} is not available until purchase readiness is complete.",
                code="apple_pay_not_ready",
                order=order,
            )

        payload = self._gateway_authorize_payload(order, payment_token)
        idem = self._normalize_idempotency_key(idempotency_key or order.idempotency_key)
        try:
            provider_payload = self._request_gateway_authorize(payload, idem, payment_method=expected_method)
        except Exception as exc:  # noqa: BLE001
            order.status = "failed"
            order.failure_reason = self._safe_error(exc)
            order.details = {**order.details, "gateway_error": order.failure_reason}
            db.session.flush()
            raise WalletApplePayPurchaseError(
                f"{self._payment_method_label(expected_method)} authorization is temporarily unavailable.",
                code="apple_pay_gateway_failed",
                order=order,
            ) from exc

        gateway_status = self._normalize_gateway_status(provider_payload.get("status") or provider_payload.get("payment_status"))
        order.gateway_payment_id = (
            str(
                provider_payload.get("gateway_payment_id") or provider_payload.get("payment_id") or provider_payload.get("id") or ""
            ).strip()
            or None
        )
        order.gateway_capture_id = (
            str(provider_payload.get("gateway_capture_id") or provider_payload.get("capture_id") or "").strip() or None
        )
        order.status = "fulfillment_pending" if gateway_status == "payment_captured" else gateway_status
        order.details = {**order.details, "gateway_response": self._safe_gateway_details(provider_payload)}
        db.session.flush()
        return order

    def authorize_card_payment(
        self,
        *,
        user: User,
        order_id: str,
        gateway_payment_token: dict[str, Any],
        idempotency_key: str | None,
    ) -> WalletApplePayPurchaseOrder:
        return self.authorize_payment(
            user=user,
            order_id=order_id,
            payment_token=gateway_payment_token,
            idempotency_key=idempotency_key,
            payment_method=CARD_METHOD,
        )

    def handle_gateway_webhook(self, payload: dict[str, Any]) -> WalletApplePayPurchaseOrder:
        external_order_id = str(payload.get("external_order_id") or payload.get("order_id") or "").strip()
        gateway_payment_id = str(payload.get("gateway_payment_id") or payload.get("payment_id") or "").strip()
        order = None
        if external_order_id:
            order = WalletApplePayPurchaseOrder.query.filter_by(public_id=external_order_id).one_or_none()
        if order is None and gateway_payment_id:
            order = WalletApplePayPurchaseOrder.query.filter_by(gateway_payment_id=gateway_payment_id).one_or_none()
        if order is None:
            raise WalletApplePayPurchaseError("Apple Pay order was not found.", code="apple_pay_order_not_found")

        status = self._normalize_gateway_status(payload.get("status") or payload.get("payment_status") or order.status)
        if status == "payment_captured" and order.status not in APPLE_PAY_TERMINAL_STATUSES:
            order.status = "fulfillment_pending"
        elif status in {"failed", "canceled", "refunded", "refund_pending"}:
            order.status = status
        if gateway_payment_id and not order.gateway_payment_id:
            order.gateway_payment_id = gateway_payment_id
        capture_id = str(payload.get("gateway_capture_id") or payload.get("capture_id") or "").strip()
        if capture_id and not order.gateway_capture_id:
            order.gateway_capture_id = capture_id
        refund_id = str(payload.get("gateway_refund_id") or payload.get("refund_id") or "").strip()
        if refund_id:
            order.gateway_refund_id = refund_id
        if order.status in {"failed", "canceled", "refunded"}:
            order.failure_reason = str(payload.get("failure_reason") or payload.get("reason") or order.failure_reason or "").strip()
        order.details = {**order.details, "last_gateway_webhook": self._webhook_details(payload)}
        db.session.flush()
        return order

    def verify_gateway_webhook_signature(self, body: bytes, headers: Any) -> bool:
        secret = self.gateway_webhook_secret
        if not secret:
            return False
        timestamp = str(
            headers.get("X-AlgVault-ApplePay-Timestamp") or headers.get("X-ApplePay-Gateway-Timestamp") or headers.get("X-Timestamp") or ""
        ).strip()
        supplied = str(
            headers.get("X-AlgVault-ApplePay-Signature") or headers.get("X-ApplePay-Gateway-Signature") or headers.get("X-Signature") or ""
        ).strip()
        if supplied.startswith("sha256="):
            supplied = supplied.removeprefix("sha256=")
        try:
            ts = int(timestamp)
        except (TypeError, ValueError):
            return False
        if abs(int(time.time()) - ts) > self.gateway_webhook_tolerance_seconds:
            return False
        expected = hmac.new(secret.encode("utf-8"), timestamp.encode("utf-8") + b"." + body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, supplied)

    def verify_card_gateway_webhook_signature(self, body: bytes, headers: Any) -> bool:
        secret = self.card_gateway_webhook_secret
        if not secret:
            return False
        timestamp = str(
            headers.get("X-AlgVault-Card-Timestamp") or headers.get("X-Card-Gateway-Timestamp") or headers.get("X-Timestamp") or ""
        ).strip()
        supplied = str(
            headers.get("X-AlgVault-Card-Signature") or headers.get("X-Card-Gateway-Signature") or headers.get("X-Signature") or ""
        ).strip()
        if supplied.startswith("sha256="):
            supplied = supplied.removeprefix("sha256=")
        try:
            ts = int(timestamp)
        except (TypeError, ValueError):
            return False
        if abs(int(time.time()) - ts) > self.gateway_webhook_tolerance_seconds:
            return False
        expected = hmac.new(secret.encode("utf-8"), timestamp.encode("utf-8") + b"." + body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, supplied)

    def process_pending_orders(self, *, limit: int = 10) -> dict[str, Any]:
        orders = (
            WalletApplePayPurchaseOrder.query.filter_by(status="fulfillment_pending")
            .order_by(WalletApplePayPurchaseOrder.created_at.asc(), WalletApplePayPurchaseOrder.id.asc())
            .limit(max(1, min(int(limit or 10), 50)))
            .all()
        )
        results = []
        for order in orders:
            try:
                self.fulfill_order(order)
                results.append({"order_id": order.public_id, "status": order.status})
            except WalletApplePayPurchaseError as exc:
                results.append({"order_id": order.public_id, "status": "failed", "code": exc.code, "message": str(exc)})
        return {"processed": len(results), "orders": results}

    def fulfill_order(self, order: WalletApplePayPurchaseOrder) -> WalletApplePayPurchaseOrder:
        if order.status != "fulfillment_pending":
            return order
        readiness = self.card_readiness() if self._normalize_payment_method(order.payment_method) == CARD_METHOD else self.readiness()
        if not readiness["ready"]:
            raise WalletApplePayPurchaseError(
                f"{self._payment_method_label(order.payment_method)} fulfillment is blocked until purchase readiness is complete.",
                code="apple_pay_fulfillment_not_ready",
                order=order,
            )
        try:
            if self._fulfillment_kind(order.asset, order.network) == "treasury_transfer":
                swap_payload: dict[str, Any] = {}
                signer_payload = self._request_treasury_transfer_signer(order)
            else:
                swap_payload = self._request_oneinch_swap(order)
                signer_payload = self._request_evm_swap_signer(order, swap_payload)
        except WalletApplePayPurchaseError:
            raise
        except Exception as exc:  # noqa: BLE001
            order.status = "failed"
            order.failure_reason = self._safe_error(exc)
            order.details = {**order.details, "fulfillment_error": order.failure_reason}
            db.session.flush()
            raise WalletApplePayPurchaseError(
                "Apple Pay fulfillment failed and requires review.",
                code="apple_pay_fulfillment_failed",
                order=order,
            ) from exc

        if swap_payload:
            order.oneinch_swap_id = (
                str(swap_payload.get("txHash") or swap_payload.get("swap_id") or swap_payload.get("id") or "").strip() or None
            )
        order.fulfillment_tx_hash = str(signer_payload.get("tx_hash") or signer_payload.get("transaction_hash") or "").strip() or None
        order.treasury_tx_hash = str(signer_payload.get("treasury_tx_hash") or "").strip() or None
        normalized_status = self._normalize_fulfillment_status(signer_payload.get("status") or "transfer_submitted")
        validation_error = self._fulfillment_response_error(order, normalized_status)
        if validation_error:
            order.status = "failed"
            order.failure_reason = validation_error
            order.details = {
                **order.details,
                "oneinch_swap": self._safe_gateway_details(swap_payload) if swap_payload else {},
                "signer_response": self._safe_gateway_details(signer_payload),
                "fulfillment_error": validation_error,
            }
            db.session.flush()
            raise WalletApplePayPurchaseError(validation_error, code="apple_pay_fulfillment_failed", order=order)
        order.status = normalized_status
        if order.status == "complete":
            order.completed_at = datetime.utcnow()
        order.details = {
            **order.details,
            "oneinch_swap": self._safe_gateway_details(swap_payload) if swap_payload else {},
            "signer_response": self._safe_gateway_details(signer_payload),
        }
        db.session.flush()
        return order

    def status_payload(self, order: WalletApplePayPurchaseOrder) -> dict[str, Any]:
        quote = order.details.get("quote") if isinstance(order.details, dict) else {}
        expired = self._quote_expired(order) and order.status == "quoted"
        status = "expired" if expired else order.status
        treasury_fee = self._order_treasury_fee_details(order)
        return {
            "order_id": order.public_id,
            "asset": order.asset,
            "network": order.network,
            "fiat_currency": order.fiat_currency,
            "fiat_gross_amount": order.fiat_gross_amount,
            "treasury_fee_usd": order.treasury_fee_usd,
            "execution_fee_usd": order.execution_fee_usd,
            "net_asset_amount": order.net_asset_amount,
            "purchase_amount": order.fiat_gross_amount,
            "algvault_fee": order.treasury_fee_usd,
            "algvault_fee_label": "AlgVault fee, paid to ETH treasury",
            "provider_network_estimate": order.execution_fee_usd,
            "total_charged": order.fiat_gross_amount,
            "estimated_receive_amount": order.net_asset_amount,
            "treasury_fee_address": self.treasury_fee_address,
            "treasury_fee_asset": treasury_fee["treasury_fee_asset"],
            "treasury_fee_eth_amount": treasury_fee["treasury_fee_eth_amount"],
            "treasury_fee_eth_price_usd": treasury_fee["treasury_fee_eth_price_usd"],
            "treasury_fee_price_source": treasury_fee["treasury_fee_price_source"],
            "payment_method": order.payment_method,
            "status": status,
            "failure_reason": order.failure_reason,
            "line_items": quote.get("line_items", []) if isinstance(quote, dict) else [],
            "created_at": order.created_at.isoformat() if order.created_at else "",
            "updated_at": order.updated_at.isoformat() if order.updated_at else "",
            "expires_at": order.expires_at.isoformat() if order.expires_at else "",
        }

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("APPLE_PAY_DIRECT_ENABLED", False))

    @property
    def card_enabled(self) -> bool:
        return bool(self.config.get("CARD_BUY_ENABLED", False))

    @property
    def crypto_sale_approved(self) -> bool:
        return bool(self.config.get("APPLE_PAY_CRYPTO_SALE_APPROVED", False))

    @property
    def merchant_id(self) -> str:
        return str(self.config.get("APPLE_PAY_MERCHANT_ID", "") or "").strip()

    @property
    def display_name(self) -> str:
        return str(self.config.get("APPLE_PAY_DISPLAY_NAME", "AlgVault") or "AlgVault").strip()

    @property
    def domain_name(self) -> str:
        return str(self.config.get("APPLE_PAY_DOMAIN", "") or "").strip()

    @property
    def domain_association(self) -> str:
        return str(self.config.get("APPLE_PAY_DOMAIN_ASSOCIATION", "") or "").strip()

    @property
    def country_code(self) -> str:
        return str(self.config.get("APPLE_PAY_COUNTRY_CODE", "CA") or "CA").strip().upper()

    @property
    def apple_pay_networks(self) -> list[str]:
        raw = self.config.get("APPLE_PAY_SUPPORTED_NETWORKS")
        if isinstance(raw, list):
            raw_values = [str(item).strip() for item in raw if str(item).strip()]
        else:
            raw_values = [item.strip() for item in str(raw or "visa,masterCard,amex,discover").split(",") if item.strip()]
        values: list[str] = []
        for item in raw_values:
            normalized = self._canonical_apple_pay_network(item)
            if normalized and normalized not in values:
                values.append(normalized)
        return values or ["visa", "masterCard"]

    @property
    def gateway_authorize_url(self) -> str:
        return str(self.config.get("APPLE_PAY_GATEWAY_AUTHORIZE_URL", "") or "").strip()

    @property
    def card_gateway_tokenization_url(self) -> str:
        return str(self.config.get("CARD_GATEWAY_TOKENIZATION_URL", "") or "").strip()

    @property
    def card_gateway_authorize_url(self) -> str:
        return str(self.config.get("CARD_GATEWAY_AUTHORIZE_URL", "") or "").strip()

    @property
    def card_gateway_api_key(self) -> str:
        return str(self.config.get("CARD_GATEWAY_API_KEY", "") or "").strip()

    @property
    def card_gateway_webhook_secret(self) -> str:
        return str(self.config.get("CARD_GATEWAY_WEBHOOK_SECRET", "") or "").strip()

    @property
    def card_gateway_public_config(self) -> dict[str, Any]:
        raw = self.config.get("CARD_GATEWAY_PUBLIC_CONFIG") or {}
        if not isinstance(raw, dict):
            return {}
        blocked = {"secret", "private", "api_key", "apikey", "bearer", "token", "password"}
        return {
            str(key): value
            for key, value in raw.items()
            if isinstance(value, str | int | float | bool) and not any(marker in str(key).lower().replace("-", "_") for marker in blocked)
        }

    def card_gateway_client_config(self) -> dict[str, Any]:
        return {
            "tokenization_url": self.card_gateway_tokenization_url,
            "public_config": self.card_gateway_public_config,
        }

    @property
    def gateway_api_key(self) -> str:
        return str(self.config.get("APPLE_PAY_GATEWAY_API_KEY", "") or "").strip()

    @property
    def gateway_webhook_secret(self) -> str:
        return str(self.config.get("APPLE_PAY_GATEWAY_WEBHOOK_SECRET", "") or "").strip()

    @property
    def gateway_webhook_tolerance_seconds(self) -> int:
        return int(self.config.get("APPLE_PAY_GATEWAY_WEBHOOK_TOLERANCE_SECONDS", 300) or 300)

    @property
    def oneinch_api_key(self) -> str:
        return str(self.config.get("ONEINCH_API_KEY", "") or "").strip()

    @property
    def oneinch_base_url(self) -> str:
        return str(self.config.get("ONEINCH_API_BASE_URL", "https://api.1inch.com/swap/v6.1") or "").strip().rstrip("/")

    def _oneinch_provider_configured(self) -> bool:
        return bool(self.oneinch_api_key)

    def _internal_treasury_signer_available(self) -> bool:
        if not bool(self.config.get("WALLET_BUY_INTERNAL_TREASURY_SIGNER_ENABLED", True)):
            return False
        if not bool(self.config.get("WALLET_INTERNAL_MPC_SIGNER_ENABLED", False)):
            return False
        if not str(self.config.get("WALLET_MPC_SIGNER_TOKEN", "") or "").strip():
            return False
        if not str(self.config.get("WALLET_MPC_SIGNER_ENCRYPTION_KEY", "") or "").strip():
            return False
        return bool(str(self.config.get("PUBLIC_APP_ORIGIN") or self.config.get("PUBLIC_API_ORIGIN") or "").strip())

    @property
    def treasury_source_address(self) -> str:
        return str(self.config.get("APPLE_PAY_TREASURY_SOURCE_ADDRESS", "") or "").strip()

    @property
    def treasury_fee_address(self) -> str:
        return str(self.config.get("APPLE_PAY_TREASURY_FEE_ADDRESS", "") or "").strip()

    @property
    def treasury_signer_url(self) -> str:
        configured = str(self.config.get("APPLE_PAY_TREASURY_SIGNER_URL", "") or "").strip()
        if configured:
            return configured
        if self._internal_treasury_signer_available():
            origin = str(self.config.get("PUBLIC_APP_ORIGIN") or self.config.get("PUBLIC_API_ORIGIN") or "").strip().rstrip("/")
            if origin:
                return f"{origin}/_internal/mpc-signer/wallet-buy/treasury-transfer"
        return ""

    @property
    def treasury_signer_token(self) -> str:
        configured = str(self.config.get("APPLE_PAY_TREASURY_SIGNER_TOKEN", "") or "").strip()
        if configured:
            return configured
        if self._internal_treasury_signer_available():
            return str(self.config.get("WALLET_MPC_SIGNER_TOKEN", "") or "").strip()
        return ""

    @property
    def treasury_signer_provider(self) -> str:
        if str(self.config.get("APPLE_PAY_TREASURY_SIGNER_URL", "") or "").strip():
            return "external"
        return "internal_mpc" if self._internal_treasury_signer_available() else ""

    @property
    def treasury_fee_bps(self) -> float:
        return self._wallet_buy_treasury_fee_bps()

    @property
    def execution_fee_buffer_bps(self) -> float:
        return max(0.0, float(self.config.get("APPLE_PAY_EXECUTION_FEE_BUFFER_BPS", 500.0) or 500.0))

    @property
    def min_fiat_usd(self) -> float:
        return float(self.config.get("APPLE_PAY_MIN_FIAT_USD", 10.0) or 10.0)

    @property
    def max_fiat_usd(self) -> float:
        return float(self.config.get("APPLE_PAY_MAX_FIAT_USD", 5_000.0) or 5_000.0)

    @property
    def timeout_seconds(self) -> float:
        return float(self.config.get("APPLE_PAY_TIMEOUT_SECONDS", 12.0) or 12.0)

    @property
    def quote_ttl_seconds(self) -> int:
        return max(30, int(self.config.get("WALLET_BUY_QUOTE_TTL_SECONDS", 300) or 300))

    @property
    def allowed_assets(self) -> dict[str, dict[str, dict[str, Any]]]:
        raw = self.config.get("APPLE_PAY_BUY_ALLOWED_ASSETS") or {}
        return self._allowed_assets_from_config(raw)

    @property
    def treasury_source_wallets(self) -> dict[str, Any]:
        raw = self.config.get("APPLE_PAY_TREASURY_SOURCE_WALLETS") or {}
        return raw if isinstance(raw, dict) else {}

    def _quote_payload(
        self,
        *,
        asset: str,
        network: str,
        fiat_currency: str,
        fiat_gross_amount: float,
        destination_address: str,
    ) -> dict[str, Any]:
        if self._fulfillment_kind(asset, network) == "treasury_transfer":
            return self._treasury_transfer_quote_payload(
                asset=asset,
                network=network,
                fiat_currency=fiat_currency,
                fiat_gross_amount=fiat_gross_amount,
                destination_address=destination_address,
            )
        return self._evm_swap_quote_payload(
            asset=asset,
            network=network,
            fiat_currency=fiat_currency,
            fiat_gross_amount=fiat_gross_amount,
            destination_address=destination_address,
        )

    def _evm_swap_quote_payload(
        self,
        *,
        asset: str,
        network: str,
        fiat_currency: str,
        fiat_gross_amount: float,
        destination_address: str,
    ) -> dict[str, Any]:
        self._assert_oneinch_swap_provider_available()
        treasury_fee = self._round_money(fiat_gross_amount * self.treasury_fee_bps / 10_000)
        treasury_fee_details = self._treasury_fee_quote(treasury_fee)
        quote_amount = max(0.0, fiat_gross_amount - treasury_fee)
        oneinch_quote = self._request_oneinch_quote(asset=asset, network=network, amount_usd=quote_amount)
        execution_fee = self._execution_fee_from_oneinch(oneinch_quote)
        net_swap_amount_usd = max(0.0, fiat_gross_amount - treasury_fee - execution_fee)
        dst_amount = self._destination_amount_from_oneinch(asset, oneinch_quote)
        if dst_amount > 0:
            asset_price = self._asset_usd_price(asset)
            execution_asset_fee = execution_fee / asset_price if asset_price > 0 else 0.0
            net_amount = self._round_asset_amount(asset, max(0.0, dst_amount - execution_asset_fee))
        else:
            net_amount = self._round_asset_amount(asset, net_swap_amount_usd / max(self._asset_usd_price(asset), 1e-12))
        if net_amount <= 0:
            raise WalletApplePayPurchaseError("Apple Pay amount is too small after fees.", code="apple_pay_amount_too_small")
        cfg = self._network_asset_config(asset, network)
        source_decimals = int(cfg.get("decimals", 6) or 6)
        source_units = str(int(max(0.0, net_swap_amount_usd) * (10**source_decimals)))
        line_items = [
            {"label": f"Estimated {asset} delivered", "amount": self._money_text(net_amount)},
            {"label": "AlgVault fee, paid to ETH treasury", "amount": self._money_text(treasury_fee)},
            {"label": "Network execution estimate", "amount": self._money_text(execution_fee)},
            {"label": "Total charged", "amount": self._money_text(fiat_gross_amount)},
        ]
        return {
            "asset": asset,
            "network": network,
            "destination_address": destination_address,
            "fiat_currency": fiat_currency,
            "fiat_gross_amount": self._round_money(fiat_gross_amount),
            "treasury_fee_usd": treasury_fee,
            **treasury_fee_details,
            "execution_fee_usd": execution_fee,
            "net_asset_amount": net_amount,
            "fulfillment_kind": "evm_swap",
            "oneinch_quote_id": str(oneinch_quote.get("quoteId") or oneinch_quote.get("quote_id") or "").strip(),
            "oneinch_source_amount_units": source_units,
            "line_items": line_items,
            "apple_pay_request": {
                "countryCode": self.country_code,
                "currencyCode": fiat_currency,
                "merchantCapabilities": ["supports3DS"],
                "supportedNetworks": self.apple_pay_networks,
                "lineItems": line_items,
                "total": {"label": self.display_name, "amount": self._money_text(fiat_gross_amount)},
            },
        }

    def _treasury_transfer_quote_payload(
        self,
        *,
        asset: str,
        network: str,
        fiat_currency: str,
        fiat_gross_amount: float,
        destination_address: str,
    ) -> dict[str, Any]:
        asset_price = self._asset_usd_price(asset)
        if asset_price <= 0:
            raise WalletApplePayPurchaseError(f"{asset} reference pricing is unavailable.", code="apple_pay_price_unavailable")
        treasury_fee = self._round_money(fiat_gross_amount * self.treasury_fee_bps / 10_000)
        treasury_fee_details = self._treasury_fee_quote(treasury_fee)
        quote_amount_usd = max(0.0, fiat_gross_amount - treasury_fee)
        estimated_asset_amount = quote_amount_usd / asset_price
        execution_fee, fee_native_amount = self._estimate_treasury_transfer_fee_usd(
            asset=asset,
            network=network,
            destination_address=destination_address,
            amount=estimated_asset_amount,
        )
        net_usd = max(0.0, fiat_gross_amount - treasury_fee - execution_fee)
        net_amount = self._round_asset_amount(asset, net_usd / asset_price)
        if net_amount <= 0:
            raise WalletApplePayPurchaseError("Apple Pay amount is too small after fees.", code="apple_pay_amount_too_small")
        self._assert_treasury_inventory(
            asset=asset,
            network=network,
            destination_address=destination_address,
            amount=net_amount,
            fee_native_amount=fee_native_amount,
        )
        line_items = [
            {"label": f"Estimated {asset} delivered", "amount": self._money_text(net_amount)},
            {"label": "AlgVault fee, paid to ETH treasury", "amount": self._money_text(treasury_fee)},
            {"label": "Network execution estimate", "amount": self._money_text(execution_fee)},
            {"label": "Total charged", "amount": self._money_text(fiat_gross_amount)},
        ]
        return {
            "asset": asset,
            "network": network,
            "destination_address": destination_address,
            "fiat_currency": fiat_currency,
            "fiat_gross_amount": self._round_money(fiat_gross_amount),
            "treasury_fee_usd": treasury_fee,
            **treasury_fee_details,
            "execution_fee_usd": execution_fee,
            "net_asset_amount": net_amount,
            "fulfillment_kind": "treasury_transfer",
            "asset_price_usd": asset_price,
            "line_items": line_items,
            "apple_pay_request": {
                "countryCode": self.country_code,
                "currencyCode": fiat_currency,
                "merchantCapabilities": ["supports3DS"],
                "supportedNetworks": self.apple_pay_networks,
                "lineItems": line_items,
                "total": {"label": self.display_name, "amount": self._money_text(fiat_gross_amount)},
            },
        }

    def _validate_quote_request(
        self,
        asset: str,
        network: str,
        fiat_currency: str,
        fiat_amount: float,
        deposit_address: DepositAddress,
        *,
        payment_method: str = APPLE_PAY_METHOD,
    ) -> None:
        payment_key = self._normalize_payment_method(payment_method)
        readiness = self.card_readiness() if payment_key == CARD_METHOD else self.readiness()
        if not readiness["ready"]:
            if (
                payment_key == CARD_METHOD
                and self._is_swap_fulfillment_asset(asset, network)
                and self.card_enabled
                and not self._oneinch_provider_configured()
            ):
                raise WalletApplePayPurchaseError(
                    "A low-fee swap provider is required for card buy fulfillment. Configure ONEINCH_API_KEY or disable card buys for EVM swap assets.",
                    code="apple_pay_oneinch_provider_unavailable",
                )
            raise WalletApplePayPurchaseError(
                f"{self._payment_method_label(payment_key)} is not available until purchase readiness is complete.",
                code="apple_pay_not_ready",
            )
        if fiat_currency != "USD":
            raise WalletApplePayPurchaseError(
                f"Only USD {self._payment_method_label(payment_key)} buys are currently supported.",
                code="apple_pay_bad_currency",
            )
        if fiat_amount < self.min_fiat_usd or fiat_amount > self.max_fiat_usd:
            raise WalletApplePayPurchaseError(
                f"Enter a USD amount between ${self.min_fiat_usd:.0f} and ${self.max_fiat_usd:.0f}.",
                code="apple_pay_bad_amount",
            )
        networks = self._allowed_assets_for_payment_method(payment_key).get(asset, {})
        if network not in networks:
            raise WalletApplePayPurchaseError(
                f"This asset/network is not enabled for {self._payment_method_label(payment_key)} buys.",
                code="apple_pay_asset_not_allowed",
            )
        if payment_key == CARD_METHOD and self._fulfillment_kind(asset, network) != "treasury_transfer":
            raise WalletApplePayPurchaseError(
                "Card buys are limited to EVM treasury-transfer fulfillment for this release.",
                code="apple_pay_asset_not_allowed",
            )
        if not str(getattr(deposit_address, "address", "") or "").strip():
            raise WalletApplePayPurchaseError(
                "A destination deposit address is required before checkout.", code="apple_pay_no_deposit_address"
            )

    def _request_oneinch_quote(self, *, asset: str, network: str, amount_usd: float) -> dict[str, Any]:
        self._assert_oneinch_swap_provider_available()
        override = self.config.get("APPLE_PAY_EXECUTION_FEE_USD_OVERRIDE")
        if override is not None and float(override) >= 0:
            return {"gas": 0, "gasPrice": 0, "execution_fee_usd_override": float(override)}
        cfg = self._network_asset_config(asset, network)
        decimals = int(cfg.get("decimals", 6) or 6)
        amount_units = str(int(max(0.0, float(amount_usd or 0.0)) * (10**decimals)))
        chain_id = int(cfg["chain_id"])
        url = f"{self.oneinch_base_url}/{chain_id}/quote"
        params: dict[str, Any] = {
            "src": cfg.get("source_token_address") or cfg.get("token_address"),
            "dst": cfg.get("token_address"),
            "amount": amount_units,
            "includeGas": "true",
        }
        params.update(self._oneinch_partner_params())
        try:
            response = self.session.get(
                url,
                params=params,
                headers={"Authorization": f"Bearer {self.oneinch_api_key}"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise WalletApplePayPurchaseError("1inch fee quote is temporarily unavailable.", code="apple_pay_oneinch_quote_failed") from exc
        if not isinstance(data, dict):
            raise WalletApplePayPurchaseError("1inch returned an invalid quote.", code="apple_pay_oneinch_quote_failed")
        return data

    def _request_oneinch_swap(self, order: WalletApplePayPurchaseOrder) -> dict[str, Any]:
        self._assert_oneinch_swap_provider_available()
        cfg = self._network_asset_config(order.asset, order.network)
        decimals = int(cfg.get("decimals", 6) or 6)
        quote = order.details.get("quote", {}) if isinstance(order.details, dict) else {}
        amount_units = str(quote.get("oneinch_source_amount_units") or "").strip()
        if not amount_units:
            amount_units = str(int(max(0.0, float(order.net_asset_amount or 0.0)) * (10**decimals)))
        chain_id = int(cfg["chain_id"])
        url = f"{self.oneinch_base_url}/{chain_id}/swap"
        params: dict[str, Any] = {
            "src": cfg.get("source_token_address") or cfg.get("token_address"),
            "dst": cfg.get("token_address"),
            "amount": amount_units,
            "from": self._treasury_source_address(order.asset, order.network),
            "receiver": order.destination_address,
            "slippage": float(self.config.get("APPLE_PAY_ONEINCH_SLIPPAGE_PCT", 0.5) or 0.5),
            "disableEstimate": "false",
            "includeGas": "true",
        }
        params.update(self._oneinch_partner_params())
        response = self.session.get(
            url,
            params=params,
            headers={"Authorization": f"Bearer {self.oneinch_api_key}"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise WalletApplePayPurchaseError("1inch returned an invalid swap.", code="apple_pay_oneinch_swap_failed", order=order)
        return data

    def _request_evm_swap_signer(self, order: WalletApplePayPurchaseOrder, swap_payload: dict[str, Any]) -> dict[str, Any]:
        source_cfg = self._treasury_source_config(order.asset, order.network)
        treasury_fee = self._order_treasury_fee_details(order)
        response = self.session.post(
            self.treasury_signer_url,
            json={
                "external_order_id": order.public_id,
                "fulfillment_kind": "evm_swap",
                "asset": order.asset,
                "network": order.network,
                "source_address": self._treasury_source_address(order.asset, order.network),
                "destination_address": order.destination_address,
                "treasury_fee_address": self.treasury_fee_address,
                "net_asset_amount": order.net_asset_amount,
                "treasury_fee_usd": order.treasury_fee_usd,
                "treasury_fee_asset": treasury_fee["treasury_fee_asset"],
                "treasury_fee_eth_amount": treasury_fee["treasury_fee_eth_amount"],
                "treasury_fee_eth_price_usd": treasury_fee["treasury_fee_eth_price_usd"],
                "treasury_fee_price_source": treasury_fee["treasury_fee_price_source"],
                "require_treasury_fee_transfer": self._treasury_fee_transfer_required(order),
                "signer_route": source_cfg.get("signer_route") or source_cfg.get("route") or "",
                "signer_key_id": source_cfg.get("signer_key_id") or source_cfg.get("key_id") or "",
                "oneinch_swap": swap_payload.get("tx") or swap_payload,
            },
            headers={"Authorization": f"Bearer {self.treasury_signer_token}", "Content-Type": "application/json"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise WalletApplePayPurchaseError("Treasury signer returned an invalid response.", code="apple_pay_signer_failed", order=order)
        return data

    def _request_treasury_transfer_signer(self, order: WalletApplePayPurchaseOrder) -> dict[str, Any]:
        source_cfg = self._treasury_source_config(order.asset, order.network)
        treasury_fee = self._order_treasury_fee_details(order)
        response = self.session.post(
            self.treasury_signer_url,
            json={
                "external_order_id": order.public_id,
                "fulfillment_kind": "treasury_transfer",
                "asset": order.asset,
                "network": order.network,
                "source_address": self._treasury_source_address(order.asset, order.network),
                "destination_address": order.destination_address,
                "amount": order.net_asset_amount,
                "net_asset_amount": order.net_asset_amount,
                "fiat_currency": order.fiat_currency,
                "fiat_gross_amount": order.fiat_gross_amount,
                "treasury_fee_usd": order.treasury_fee_usd,
                "treasury_fee_address": self.treasury_fee_address,
                "treasury_fee_asset": treasury_fee["treasury_fee_asset"],
                "treasury_fee_eth_amount": treasury_fee["treasury_fee_eth_amount"],
                "treasury_fee_eth_price_usd": treasury_fee["treasury_fee_eth_price_usd"],
                "treasury_fee_price_source": treasury_fee["treasury_fee_price_source"],
                "require_treasury_fee_transfer": self._treasury_fee_transfer_required(order),
                "execution_fee_usd": order.execution_fee_usd,
                "signer_route": source_cfg.get("signer_route") or source_cfg.get("route") or "",
                "signer_key_id": source_cfg.get("signer_key_id") or source_cfg.get("key_id") or "",
                "signer_metadata": self._safe_signer_metadata(source_cfg.get("signer_metadata")),
            },
            headers={"Authorization": f"Bearer {self.treasury_signer_token}", "Content-Type": "application/json"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise WalletApplePayPurchaseError("Treasury signer returned an invalid response.", code="apple_pay_signer_failed", order=order)
        return data

    def _request_gateway_authorize(self, payload: dict[str, Any], idempotency_key: str, *, payment_method: str) -> dict[str, Any]:
        payment_key = self._normalize_payment_method(payment_method)
        authorize_url = self.card_gateway_authorize_url if payment_key == CARD_METHOD else self.gateway_authorize_url
        api_key = self.card_gateway_api_key if payment_key == CARD_METHOD else self.gateway_api_key
        response = self.session.post(
            authorize_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise WalletApplePayPurchaseError("Gateway returned an invalid response.", code="apple_pay_gateway_response")
        return data

    def _request_apple_merchant_session(self, validation_url: str, payload: dict[str, Any]) -> dict[str, Any]:
        cert_path = str(self.config.get("APPLE_PAY_MERCHANT_CERT_PATH", "") or "").strip()
        key_path = str(self.config.get("APPLE_PAY_MERCHANT_KEY_PATH", "") or "").strip()
        cert = (cert_path, key_path) if cert_path and key_path else None
        if cert is None:
            cert_pem = str(self.config.get("APPLE_PAY_MERCHANT_CERT_PEM", "") or "").strip()
            key_pem = str(self.config.get("APPLE_PAY_MERCHANT_KEY_PEM", "") or "").strip()
            if cert_pem and key_pem:
                with ExitStack() as stack:
                    cert_file = stack.enter_context(tempfile.NamedTemporaryFile("w", suffix=".pem", delete=True))
                    key_file = stack.enter_context(tempfile.NamedTemporaryFile("w", suffix=".pem", delete=True))
                    cert_file.write(cert_pem)
                    key_file.write(key_pem)
                    cert_file.flush()
                    key_file.flush()
                    response = self.session.post(
                        validation_url,
                        json=payload,
                        cert=(cert_file.name, key_file.name),
                        timeout=self.timeout_seconds,
                    )
                    response.raise_for_status()
                    data = response.json()
                    if not isinstance(data, dict):
                        raise WalletApplePayPurchaseError(
                            "Apple merchant validation returned an invalid response.",
                            code="apple_pay_merchant_response",
                        )
                    return data
        response = self.session.post(
            validation_url,
            json=payload,
            cert=cert,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise WalletApplePayPurchaseError("Apple merchant validation returned an invalid response.", code="apple_pay_merchant_response")
        return data

    def _gateway_authorize_payload(self, order: WalletApplePayPurchaseOrder, payment_token: dict[str, Any]) -> dict[str, Any]:
        quote = order.details.get("quote", {}) if isinstance(order.details, dict) else {}
        payment_key = self._normalize_payment_method(order.payment_method)
        payload = {
            "external_order_id": order.public_id,
            "user_id": order.user_id,
            "asset": order.asset,
            "network": order.network,
            "destination_address": order.destination_address,
            "fiat_currency": order.fiat_currency,
            "fiat_amount": order.fiat_gross_amount,
            "payment_method": payment_key,
            "line_items": quote.get("line_items", []),
            "treasury_fee": self._order_treasury_fee_details(order),
            "capture": True,
        }
        if payment_key == CARD_METHOD:
            payload["gateway_payment_token"] = payment_token
        else:
            payload["apple_pay_token"] = payment_token
        return payload

    def _revalidate_order_quote(self, order: WalletApplePayPurchaseOrder) -> None:
        quote = order.details.get("quote", {}) if isinstance(order.details, dict) else {}
        if self._is_swap_fulfillment_asset(order.asset, order.network):
            self._assert_oneinch_swap_provider_available()
        expected_fee = self._round_money(float(order.fiat_gross_amount or 0.0) * self.treasury_fee_bps / 10_000)
        if abs(float(order.treasury_fee_usd or 0.0) - expected_fee) > 0.009:
            raise WalletApplePayPurchaseError(
                "Quote fee validation failed. Refresh the quote before authorizing payment.",
                code="wallet_buy_quote_invalid",
                order=order,
            )
        if str(quote.get("asset") or order.asset).upper() != str(order.asset or "").upper():
            raise WalletApplePayPurchaseError("Quote asset validation failed.", code="wallet_buy_quote_invalid", order=order)
        if str(quote.get("network") or order.network) != str(order.network or ""):
            raise WalletApplePayPurchaseError("Quote network validation failed.", code="wallet_buy_quote_invalid", order=order)
        if float(order.net_asset_amount or 0.0) <= 0 or float(order.fiat_gross_amount or 0.0) <= 0:
            raise WalletApplePayPurchaseError("Quote amount validation failed.", code="wallet_buy_quote_invalid", order=order)

    def _quote_expired(self, order: WalletApplePayPurchaseOrder) -> bool:
        expires_at = getattr(order, "expires_at", None)
        return bool(expires_at and datetime.utcnow() > expires_at)

    def _destination_amount_from_oneinch(self, asset: str, payload: dict[str, Any]) -> float:
        raw = payload.get("dstAmount")
        if raw is None:
            raw = payload.get("toAmount")
        if raw is None:
            return 0.0
        try:
            units = Decimal(str(raw))
        except Exception:  # noqa: BLE001
            return 0.0
        if units <= 0:
            return 0.0
        return float(units / Decimal(10 ** self._asset_decimals(asset)))

    def _oneinch_partner_params(self) -> dict[str, Any]:
        try:
            fee = float(self.config.get("ONEINCH_PARTNER_FEE_PCT", 0.0) or 0.0)
        except (TypeError, ValueError):
            fee = 0.0
        referrer = str(self.config.get("ONEINCH_PARTNER_FEE_REFERRER") or self.config.get("ONEINCH_REFERRER") or "").strip()
        if 0 < fee <= 3 and referrer:
            return {"fee": fee, "referrer": referrer}
        return {}

    def _execution_fee_from_oneinch(self, payload: dict[str, Any]) -> float:
        override = payload.get("execution_fee_usd_override")
        if override is not None:
            return self._round_money(float(override))
        gas = self._first_float(payload, "gas", "estimatedGas", "estimated_gas")
        tx = payload.get("tx") if isinstance(payload.get("tx"), dict) else {}
        gas = gas or self._first_float(tx, "gas", "estimatedGas")
        gas_price = self._first_float(payload, "gasPrice", "gas_price", "gasPriceWei") or self._first_float(tx, "gasPrice", "gas_price")
        if gas <= 0 or gas_price <= 0:
            raise WalletApplePayPurchaseError("1inch fee quote is unavailable.", code="apple_pay_oneinch_fee_unavailable")
        eth_price = max(0.0, float(self.config.get("PLATFORM_TREASURY_ETH_USD_FALLBACK", 3000.0) or 0.0))
        if eth_price <= 0:
            raise WalletApplePayPurchaseError("ETH reference pricing is unavailable.", code="apple_pay_eth_price_unavailable")
        fee_usd = gas * gas_price / 10**18 * eth_price
        fee_usd *= 1 + self.execution_fee_buffer_bps / 10_000
        return self._round_money(max(0.01, fee_usd))

    def _estimate_treasury_transfer_fee_usd(
        self,
        *,
        asset: str,
        network: str,
        destination_address: str,
        amount: float,
    ) -> tuple[float, float]:
        adapter = self._adapter_for(asset, network)
        if adapter is None:
            raise WalletApplePayPurchaseError(
                f"No custody fee estimator supports {asset} on {network}.",
                code="apple_pay_transfer_fee_unavailable",
            )
        try:
            fee_native = max(0.0, float(adapter.estimate_fee(asset, network, destination_address, amount) or 0.0))
        except Exception as exc:  # noqa: BLE001
            raise WalletApplePayPurchaseError(
                f"{asset} network fee estimate is unavailable.",
                code="apple_pay_transfer_fee_unavailable",
            ) from exc
        if fee_native <= 0:
            raise WalletApplePayPurchaseError(
                f"{asset} network fee estimate is unavailable.",
                code="apple_pay_transfer_fee_unavailable",
            )
        fee_asset = self._native_fee_asset(asset, network)
        fee_price = self._asset_usd_price(fee_asset)
        if fee_price <= 0:
            raise WalletApplePayPurchaseError(
                f"{fee_asset} fee pricing is unavailable.",
                code="apple_pay_price_unavailable",
            )
        fee_usd = fee_native * fee_price
        fee_usd *= 1 + self.execution_fee_buffer_bps / 10_000
        return self._round_money(max(0.01, fee_usd)), fee_native

    def _assert_treasury_inventory(
        self,
        *,
        asset: str,
        network: str,
        destination_address: str,
        amount: float,
        fee_native_amount: float,
    ) -> None:
        source_address = self._treasury_source_address(asset, network)
        if not source_address:
            raise WalletApplePayPurchaseError(
                f"Treasury source wallet is not configured for {asset} on {network}.",
                code="apple_pay_treasury_source_missing",
            )
        adapter = self._adapter_for(asset, network)
        if adapter is None:
            raise WalletApplePayPurchaseError(
                f"No custody inventory checker supports {asset} on {network}.",
                code="apple_pay_inventory_unavailable",
            )
        try:
            snapshot = adapter.get_balance(source_address, asset, network)
        except Exception as exc:  # noqa: BLE001
            raise WalletApplePayPurchaseError(
                f"{asset} treasury inventory check is unavailable.",
                code="apple_pay_inventory_unavailable",
            ) from exc
        if not getattr(snapshot, "checked", False):
            raise WalletApplePayPurchaseError(
                f"{asset} treasury inventory check is unavailable.",
                code="apple_pay_inventory_unavailable",
            )
        required_amount = amount + fee_native_amount if self._native_fee_asset(asset, network) == asset else amount
        if float(getattr(snapshot, "amount", 0.0) or 0.0) + 1e-12 < required_amount:
            raise WalletApplePayPurchaseError(
                f"Treasury inventory is insufficient for {asset} fulfillment.",
                code="apple_pay_inventory_insufficient",
            )
        if self._is_evm_network(network) and asset != "ETH" and fee_native_amount > 0:
            try:
                gas_snapshot = adapter.get_balance(source_address, "ETH", network)
            except Exception as exc:  # noqa: BLE001
                raise WalletApplePayPurchaseError(
                    "Treasury gas inventory check is unavailable.",
                    code="apple_pay_inventory_unavailable",
                ) from exc
            if not getattr(gas_snapshot, "checked", False):
                raise WalletApplePayPurchaseError(
                    "Treasury gas inventory check is unavailable.",
                    code="apple_pay_inventory_unavailable",
                )
            if float(getattr(gas_snapshot, "amount", 0.0) or 0.0) + 1e-18 < fee_native_amount:
                raise WalletApplePayPurchaseError(
                    "Treasury ETH gas inventory is insufficient for token fulfillment.",
                    code="apple_pay_inventory_insufficient",
                )

    def _treasury_fee_quote(self, treasury_fee_usd: float) -> dict[str, Any]:
        fee_usd = self._round_money(max(0.0, float(treasury_fee_usd or 0.0)))
        eth_price, price_source = self._eth_usd_price_context()
        if fee_usd > 0 and eth_price <= 0:
            raise WalletApplePayPurchaseError("ETH reference pricing is unavailable.", code="apple_pay_eth_price_unavailable")
        fee_eth = self._round_asset_amount(TREASURY_FEE_ASSET, fee_usd / eth_price) if fee_usd > 0 else 0.0
        return {
            "treasury_fee_usd": fee_usd,
            "treasury_fee_asset": TREASURY_FEE_ASSET,
            "treasury_fee_eth_amount": fee_eth,
            "treasury_fee_eth_price_usd": self._round_money(eth_price) if eth_price > 0 else 0.0,
            "treasury_fee_price_source": price_source,
            "treasury_fee_address": self.treasury_fee_address,
        }

    def _order_treasury_fee_details(self, order: WalletApplePayPurchaseOrder) -> dict[str, Any]:
        quote = order.details.get("quote", {}) if isinstance(order.details, dict) else {}
        if isinstance(quote, dict) and quote.get("treasury_fee_asset") and quote.get("treasury_fee_eth_amount") is not None:
            return {
                "treasury_fee_usd": float(quote.get("treasury_fee_usd") or order.treasury_fee_usd or 0.0),
                "treasury_fee_asset": str(quote.get("treasury_fee_asset") or TREASURY_FEE_ASSET),
                "treasury_fee_eth_amount": float(quote.get("treasury_fee_eth_amount") or 0.0),
                "treasury_fee_eth_price_usd": float(quote.get("treasury_fee_eth_price_usd") or 0.0),
                "treasury_fee_price_source": str(quote.get("treasury_fee_price_source") or ""),
                "treasury_fee_address": str(quote.get("treasury_fee_address") or self.treasury_fee_address),
            }
        try:
            return self._treasury_fee_quote(float(order.treasury_fee_usd or 0.0))
        except WalletApplePayPurchaseError:
            return {
                "treasury_fee_usd": float(order.treasury_fee_usd or 0.0),
                "treasury_fee_asset": TREASURY_FEE_ASSET,
                "treasury_fee_eth_amount": 0.0,
                "treasury_fee_eth_price_usd": 0.0,
                "treasury_fee_price_source": "",
                "treasury_fee_address": self.treasury_fee_address,
            }

    def _eth_usd_price_context(self) -> tuple[float, str]:
        overrides = self.config.get("APPLE_PAY_ASSET_PRICE_USD") or {}
        if isinstance(overrides, dict):
            try:
                configured = float(overrides.get("ETH") or overrides.get("eth") or 0.0)
            except (TypeError, ValueError):
                configured = 0.0
            if configured > 0:
                return configured, "asset_price_override"
        market_price = shared_asset_usd_price(TREASURY_FEE_ASSET, self._market_price_lookup)
        if market_price > 0:
            return market_price, "market_data"
        try:
            fallback = float(self.config.get("PLATFORM_TREASURY_ETH_USD_FALLBACK", 0.0) or 0.0)
        except (TypeError, ValueError):
            fallback = 0.0
        if fallback > 0:
            return fallback, "platform_treasury_eth_usd_fallback"
        return 0.0, ""

    def _fulfillment_response_error(self, order: WalletApplePayPurchaseOrder, normalized_status: str) -> str:
        if normalized_status != "complete":
            return ""
        if not order.fulfillment_tx_hash:
            return "Treasury signer did not return a user delivery transaction hash."
        if self._treasury_fee_transfer_required(order) and not order.treasury_tx_hash:
            return "Treasury signer did not return an ETH fee transfer transaction hash."
        return ""

    def _treasury_fee_transfer_required(self, order: WalletApplePayPurchaseOrder) -> bool:
        return self._normalize_payment_method(order.payment_method) == CARD_METHOD and float(order.treasury_fee_usd or 0.0) > 0

    def _apple_pay_fulfillment_worker_status(self, *, payment_method: str | None = None) -> dict[str, Any]:
        required = self._requires_recent_fulfillment_worker(payment_method=payment_method)
        if not required:
            return {"required": False, "recent": True, "status": "not_required"}
        stale_after = self._worker_stale_after_seconds()
        status = {
            "required": True,
            "recent": False,
            "status": "missing",
            "stale_after_seconds": stale_after,
            "heartbeat_lag_seconds": None,
        }
        if not has_app_context():
            return status
        try:
            lease = WorkerLease.query.filter_by(lease_name="apple_pay_fulfillment:singleton").one_or_none()
        except SQLAlchemyError:
            return {**status, "status": "unavailable"}
        if lease is None:
            return status
        lag = self._elapsed_seconds(datetime.utcnow(), lease.heartbeat_at) if lease.heartbeat_at else None
        recent = lag is not None and lag <= stale_after
        return {
            **status,
            "recent": recent,
            "status": "recent" if recent else "stale",
            "lease_status": lease.status,
            "heartbeat_lag_seconds": lag,
            "expires_at": lease.expires_at.isoformat() if lease.expires_at else None,
        }

    def _requires_recent_fulfillment_worker(self, *, payment_method: str | None = None) -> bool:
        if bool(self.config.get("TESTING", False)):
            return False
        target = str(self.config.get("DEPLOYMENT_TARGET", "local") or "local").strip().lower()
        if target not in {"vercel", "production", "prod", "vps", "postgres", "staging"}:
            return False
        if not bool(self.config.get("WORKER_APPLE_PAY_FULFILLMENT_ENABLED", True)):
            return False
        if payment_method == CARD_METHOD:
            return bool(self.card_enabled)
        if payment_method == APPLE_PAY_METHOD:
            return bool(self.enabled)
        return bool(self.enabled or self.card_enabled)

    def _worker_stale_after_seconds(self) -> float:
        try:
            poll_seconds = max(1, int(self.config.get("WORKER_POLL_SECONDS", 15) or 15))
            lease_ttl_seconds = max(1, int(self.config.get("WORKER_LEASE_TTL_SECONDS", 120) or 120))
        except (TypeError, ValueError):
            poll_seconds = 15
            lease_ttl_seconds = 120
        return float(max(60, poll_seconds * 6, lease_ttl_seconds))

    @staticmethod
    def _elapsed_seconds(now: datetime, then: datetime) -> float:
        if then.tzinfo is not None and now.tzinfo is None:
            now = now.replace(tzinfo=then.tzinfo)
        return max(0.0, (now - then).total_seconds())

    @staticmethod
    def _canonical_apple_pay_network(value: str) -> str:
        cleaned = str(value or "").strip()
        key = "".join(ch for ch in cleaned.lower() if ch.isalnum())
        return APPLE_PAY_NETWORK_CANONICAL.get(key, cleaned)

    def _allowed_assets_for_payment_method(self, payment_method: str) -> dict[str, dict[str, dict[str, Any]]]:
        if self._normalize_payment_method(payment_method) == CARD_METHOD:
            return self._card_allowed_assets(self.allowed_assets)[0]
        return self.allowed_assets

    def _card_allowed_assets(
        self,
        allowed_assets: dict[str, dict[str, dict[str, Any]]],
    ) -> tuple[dict[str, dict[str, dict[str, Any]]], list[str]]:
        card_assets: dict[str, dict[str, dict[str, Any]]] = {}
        blockers: list[str] = []
        for asset, networks in allowed_assets.items():
            asset_key = self._normalize_asset(asset)
            if asset_key not in CARD_BUY_ASSETS:
                blockers.append(f"Card buys do not support {asset_key} in this release")
                continue
            for network, cfg in networks.items():
                if not self._is_evm_network(network):
                    blockers.append(f"Card buys require an EVM network for {asset_key} on {network}")
                    continue
                if str(cfg.get("fulfillment_kind") or "").strip().lower() != "treasury_transfer":
                    blockers.append(f"Card buys require treasury_transfer fulfillment for {asset_key} on {network}")
                    continue
                card_assets.setdefault(asset_key, {})[network] = cfg
        return card_assets, blockers

    def _allowed_assets_from_config(self, raw: Any) -> dict[str, dict[str, dict[str, Any]]]:
        if not isinstance(raw, dict):
            return {}
        allowed: dict[str, dict[str, dict[str, Any]]] = {}
        for asset, value in raw.items():
            asset_key = self._normalize_asset(asset)
            if not asset_key or asset_key in APPLE_PAY_EXCLUDED_ASSETS:
                continue
            networks: dict[str, dict[str, Any]] = {}
            if isinstance(value, list):
                for network in value:
                    cfg = self._network_asset_config_from_defaults(asset_key, str(network).strip(), {})
                    if cfg:
                        networks[str(network).strip()] = cfg
            elif isinstance(value, dict):
                configured = value.get("networks")
                if isinstance(configured, list):
                    for network in configured:
                        cfg = self._network_asset_config_from_defaults(asset_key, str(network).strip(), value)
                        if cfg:
                            networks[str(network).strip()] = cfg
                else:
                    for network, network_cfg in value.items():
                        if network in {"networks", "network"}:
                            continue
                        cfg_raw = network_cfg if isinstance(network_cfg, dict) else value
                        cfg = self._network_asset_config_from_defaults(asset_key, str(network).strip(), cfg_raw)
                        if cfg:
                            networks[str(network).strip()] = cfg
            if networks:
                allowed[asset_key] = networks
        return allowed

    def _allowed_asset_readiness_blockers(self, allowed_assets: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
        blockers: list[str] = []
        for asset, networks in allowed_assets.items():
            if asset in APPLE_PAY_EXCLUDED_ASSETS:
                blockers.append(f"{asset} is not enabled for Apple Pay buys")
                continue
            for network, cfg in networks.items():
                kind = str(cfg.get("fulfillment_kind") or "").strip()
                if kind not in APPLE_PAY_FULFILLMENT_KINDS:
                    blockers.append(f"Unsupported Apple Pay fulfillment kind for {asset} on {network}")
                if not self._treasury_source_address(asset, network):
                    blockers.append(f"APPLE_PAY_TREASURY_SOURCE_WALLETS_JSON must configure {asset} on {network}")
                if kind == "treasury_transfer":
                    endpoint_blocker = self._chain_endpoint_blocker(asset, network)
                    if endpoint_blocker:
                        blockers.append(endpoint_blocker)
        return blockers

    @staticmethod
    def _uses_fulfillment_kind(allowed_assets: dict[str, dict[str, dict[str, Any]]], kind: str) -> bool:
        return any(str(cfg.get("fulfillment_kind") or "") == kind for networks in allowed_assets.values() for cfg in networks.values())

    def _network_asset_config(self, asset: str, network: str) -> dict[str, Any]:
        cfg = self.allowed_assets.get(asset, {}).get(network)
        if not cfg:
            raise WalletApplePayPurchaseError(
                "This asset/network is not configured for Apple Pay buys.", code="apple_pay_asset_not_allowed"
            )
        return cfg

    def _network_asset_config_from_defaults(self, asset: str, network: str, raw: dict[str, Any]) -> dict[str, Any]:
        if not network:
            return {}
        fulfillment_kind = str(raw.get("fulfillment_kind") or raw.get("fulfillmentKind") or raw.get("kind") or "").strip().lower()
        chain_id = raw.get("chain_id") or raw.get("chainId") or self._chain_id_for_network(network)
        token_address = raw.get("token_address") or raw.get("tokenAddress") or self._token_contract(asset, network)
        if not fulfillment_kind:
            fulfillment_kind = "evm_swap" if chain_id and token_address else "treasury_transfer"
        if fulfillment_kind not in APPLE_PAY_FULFILLMENT_KINDS:
            return {}
        if fulfillment_kind == "evm_swap" and (not chain_id or not token_address):
            return {}
        source_asset = self._normalize_asset(raw.get("source_asset") or raw.get("sourceAsset") or asset)
        source_token = (
            raw.get("source_token_address") or raw.get("sourceTokenAddress") or self._token_contract(source_asset, network) or token_address
        )
        cfg: dict[str, Any] = {
            "fulfillment_kind": fulfillment_kind,
            "decimals": int(raw.get("decimals") or self._asset_decimals(asset)),
            "source_asset": source_asset,
        }
        if chain_id:
            cfg["chain_id"] = int(chain_id)
        if token_address:
            cfg["token_address"] = str(token_address).strip()
        if source_token:
            cfg["source_token_address"] = str(source_token).strip()
        if fulfillment_kind == "treasury_transfer":
            if asset not in APPLE_PAY_TREASURY_TRANSFER_ASSETS:
                return {}
            if self._is_evm_network(network) and asset != "ETH" and not token_address:
                return {}
        return cfg

    def _fulfillment_kind(self, asset: str, network: str) -> str:
        cfg = self._network_asset_config(asset, network)
        kind = str(cfg.get("fulfillment_kind") or "").strip().lower()
        return kind if kind in APPLE_PAY_FULFILLMENT_KINDS else "evm_swap"

    def _treasury_source_config(self, asset: str, network: str) -> dict[str, Any]:
        wallets = self.treasury_source_wallets
        asset_key = self._normalize_asset(asset)
        network_key = str(network or "").strip()
        candidates = (
            wallets.get(asset_key),
            wallets.get(asset_key.lower()),
            wallets.get(f"{asset_key}:{network_key}"),
            wallets.get(f"{asset_key}:{network_key}".lower()),
            wallets.get(f"{asset_key}_{network_key}"),
            wallets.get(f"{asset_key}_{network_key}".lower()),
        )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return {"source_address": candidate.strip()}
            if isinstance(candidate, dict):
                nested = (
                    candidate.get(network_key)
                    or candidate.get(network_key.upper())
                    or candidate.get(network_key.lower())
                    or candidate.get(self._network_key(network_key))
                )
                if isinstance(nested, str) and nested.strip():
                    return {"source_address": nested.strip()}
                if isinstance(nested, dict):
                    return nested
                if candidate.get("source_address") or candidate.get("address"):
                    return candidate
        if self.treasury_source_address:
            return {"source_address": self.treasury_source_address}
        return {}

    def _treasury_source_address(self, asset: str, network: str) -> str:
        cfg = self._treasury_source_config(asset, network)
        return str(cfg.get("source_address") or cfg.get("address") or "").strip()

    @staticmethod
    def _safe_signer_metadata(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        blocked = {"token", "secret", "private_key", "signer_key_id"}
        return {str(key): item for key, item in value.items() if all(marker not in str(key).lower() for marker in blocked)}

    def _adapter_for(self, asset: str, network: str) -> WalletChainAdapter | None:
        adapters: tuple[WalletChainAdapter, ...] = (
            EvmWalletAdapter(self.config),
            BitcoinWalletAdapter(self.config),
            SolanaWalletAdapter(self.config),
            XrpWalletAdapter(self.config),
        )
        for adapter in adapters:
            if adapter.supports(asset, network):
                return adapter
        return None

    def _chain_endpoint_blocker(self, asset: str, network: str) -> str:
        if self._is_evm_network(network):
            if not self._configured_evm_rpc_url(network):
                return f"WALLET_EVM_RPC_URL or WALLET_EVM_NETWORKS_JSON must configure RPC for {network}"
            return ""
        if asset == "BTC" and not str(self.config.get("WALLET_BTC_INDEXER_URL", "") or "").strip():
            return "WALLET_BTC_INDEXER_URL must be configured for BTC Apple Pay buys"
        if asset == "SOL" and not str(self.config.get("WALLET_SOLANA_RPC_URL", "") or "").strip():
            return "WALLET_SOLANA_RPC_URL must be configured for SOL Apple Pay buys"
        if asset == "XRP" and not str(self.config.get("WALLET_XRP_RPC_URL", "") or "").strip():
            return "WALLET_XRP_RPC_URL must be configured for XRP Apple Pay buys"
        return ""

    def _configured_evm_rpc_url(self, network: str) -> str:
        global_url = str(self.config.get("WALLET_EVM_RPC_URL", "") or "").strip()
        if global_url:
            return global_url
        configured = self.config.get("WALLET_EVM_NETWORKS") or {}
        network_key = str(network or "").strip()
        if isinstance(configured, dict):
            for candidate in (network_key, network_key.upper(), network_key.lower(), self._network_key(network_key)):
                row = configured.get(candidate)
                if isinstance(row, dict) and str(row.get("rpc_url") or "").strip():
                    return str(row.get("rpc_url") or "").strip()
        return ""

    @staticmethod
    def _is_evm_network(network: str) -> bool:
        return "".join(ch for ch in str(network or "").upper() if ch.isalnum()) in {
            "ETHEREUM",
            "ARBITRUM",
            "OPTIMISM",
            "BASE",
            "POLYGON",
            "AVALANCHE",
            "BSC",
        }

    def _native_fee_asset(self, asset: str, network: str) -> str:
        return "ETH" if self._is_evm_network(network) else self._normalize_asset(asset)

    def _asset_usd_price(self, asset: str) -> float:
        asset_key = self._normalize_asset(asset)
        if asset_key == TREASURY_FEE_ASSET:
            return self._eth_usd_price_context()[0]
        overrides = self.config.get("APPLE_PAY_ASSET_PRICE_USD") or {}
        if isinstance(overrides, dict):
            try:
                configured = float(overrides.get(asset_key) or overrides.get(asset_key.lower()) or 0.0)
            except (TypeError, ValueError):
                configured = 0.0
            if configured > 0:
                return configured
        return shared_asset_usd_price(asset_key, self._market_price_lookup)

    @staticmethod
    def _market_price_lookup(asset: str) -> float:
        if not has_app_context():
            return 0.0
        try:
            return float(
                current_app.extensions["services"]["market_data"].get_mid_price(
                    asset,
                    market_mode_for(get_current_mode()),
                )
                or 0.0
            )
        except Exception:  # noqa: BLE001
            return 0.0

    def _chain_id_for_network(self, network: str) -> int | None:
        configured = self.config.get("WALLET_EVM_NETWORKS")
        key = str(network or "").strip()
        if isinstance(configured, dict):
            row = configured.get(key) or configured.get(key.upper()) or configured.get(key.lower())
            if isinstance(row, dict) and row.get("chain_id"):
                return int(row["chain_id"])
        return EVM_CHAIN_IDS.get(key.lower())

    def _token_contract(self, asset: str, network: str) -> str:
        configured = self.config.get("WALLET_EVM_TOKEN_CONTRACTS")
        asset_key = self._normalize_asset(asset)
        network_key = str(network or "").strip()
        if isinstance(configured, dict):
            by_asset = configured.get(asset_key) or configured.get(asset_key.lower())
            if isinstance(by_asset, dict):
                return str(
                    by_asset.get(network_key) or by_asset.get(network_key.upper()) or by_asset.get(network_key.lower()) or ""
                ).strip()
            by_network = configured.get(network_key) or configured.get(network_key.upper()) or configured.get(network_key.lower())
            if isinstance(by_network, dict):
                return str(by_network.get(asset_key) or by_network.get(asset_key.lower()) or "").strip()
        return ""

    def _has_merchant_certificate(self) -> bool:
        path = str(self.config.get("APPLE_PAY_MERCHANT_CERT_PATH", "") or "").strip()
        if path and Path(path).exists():
            return True
        return bool(str(self.config.get("APPLE_PAY_MERCHANT_CERT_PEM", "") or "").strip())

    def _has_merchant_key(self) -> bool:
        path = str(self.config.get("APPLE_PAY_MERCHANT_KEY_PATH", "") or "").strip()
        if path and Path(path).exists():
            return True
        return bool(str(self.config.get("APPLE_PAY_MERCHANT_KEY_PEM", "") or "").strip())

    def _orders_table_ready(self) -> bool:
        if not has_app_context():
            return False
        try:
            return bool(sa_inspect(db.engine).has_table(WalletApplePayPurchaseOrder.__tablename__))
        except SQLAlchemyError:
            return False

    @staticmethod
    def _normalize_asset(value: Any) -> str:
        return str(value or "").strip().upper()

    @staticmethod
    def _normalize_payment_method(value: Any) -> str:
        raw = str(value or "").strip().lower().replace("-", "_")
        return CARD_METHOD if raw == CARD_METHOD else APPLE_PAY_METHOD

    @staticmethod
    def _payment_method_label(value: Any) -> str:
        return BUY_PAYMENT_METHODS.get(WalletApplePayPurchaseService._normalize_payment_method(value), "Apple Pay")

    @staticmethod
    def _network_key(value: Any) -> str:
        return "".join(ch for ch in str(value or "").upper() if ch.isalnum())

    @staticmethod
    def _normalize_idempotency_key(value: str | None) -> str:
        raw = str(value or "").strip()
        return raw[:220] if raw else f"wallet-apple-pay-{uuid.uuid4().hex}"

    @staticmethod
    def _coerce_money(value: Any) -> float:
        try:
            amount = float(value)
        except (TypeError, ValueError):
            raise WalletApplePayPurchaseError("Enter a valid USD amount.", code="apple_pay_bad_amount") from None
        if amount <= 0:
            raise WalletApplePayPurchaseError("Enter a USD amount greater than zero.", code="apple_pay_bad_amount")
        return WalletApplePayPurchaseService._round_money(amount)

    @staticmethod
    def _round_money(value: float) -> float:
        return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    @staticmethod
    def _round_asset(value: float) -> float:
        return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))

    @staticmethod
    def _asset_decimals(asset: str) -> int:
        return {"BTC": 8, "ETH": 18, "SOL": 9, "XRP": 6, "USDC": 6, "USDT": 6}.get(
            WalletApplePayPurchaseService._normalize_asset(asset),
            6,
        )

    @staticmethod
    def _round_asset_amount(asset: str, value: float) -> float:
        precision = "0." + ("0" * (WalletApplePayPurchaseService._asset_decimals(asset) - 1)) + "1"
        return float(Decimal(str(value)).quantize(Decimal(precision), rounding=ROUND_HALF_UP))

    @staticmethod
    def _money_text(value: float) -> str:
        return f"{float(value or 0.0):.2f}"

    @staticmethod
    def _first_float(payload: dict[str, Any], *keys: str) -> float:
        for key in keys:
            try:
                value = payload.get(key)
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    def _wallet_buy_treasury_fee_bps(self) -> float:
        try:
            configured = float(
                self.config.get(
                    "WALLET_BUY_PLATFORM_FEE_BPS",
                    self.config.get("APPLE_PAY_TREASURY_FEE_BPS", WALLET_BUY_PLATFORM_FEE_BPS),
                )
                or WALLET_BUY_PLATFORM_FEE_BPS
            )
        except (TypeError, ValueError):
            configured = WALLET_BUY_PLATFORM_FEE_BPS
        return configured

    def _is_swap_fulfillment_asset(self, asset: str, network: str) -> bool:
        return self._fulfillment_kind(asset, network) == "evm_swap"

    def _assert_oneinch_swap_provider_available(self) -> None:
        if not self._oneinch_provider_configured():
            raise WalletApplePayPurchaseError(
                "A low-fee swap provider is required for card buy fulfillment. Configure ONEINCH_API_KEY or disable card buys for EVM swap assets.",
                code="apple_pay_oneinch_provider_unavailable",
            )

    @staticmethod
    def _valid_url(value: str) -> bool:
        try:
            parsed = urlparse(str(value or ""))
        except Exception:  # noqa: BLE001
            return False
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        text = str(exc)
        for marker in ("Authorization", "Bearer", "api_key", "secret", "token", "TOKEN", "API_KEY"):
            text = text.replace(marker, "[redacted]")
        return text[:280]

    @staticmethod
    def _safe_gateway_details(payload: dict[str, Any]) -> dict[str, Any]:
        blocked = {
            "apple_pay_token",
            "payment_token",
            "token",
            "card",
            "cryptogram",
            "eciIndicator",
            "private_key",
            "signer_key_id",
        }
        return {
            key: value
            for key, value in payload.items()
            if key not in blocked and "secret" not in key.lower() and "token" not in key.lower()
        }

    @staticmethod
    def _webhook_details(payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "external_order_id",
            "gateway_payment_id",
            "payment_id",
            "gateway_capture_id",
            "capture_id",
            "gateway_refund_id",
            "refund_id",
            "status",
            "payment_status",
            "failure_reason",
            "reason",
        }
        return {key: payload.get(key) for key in allowed if key in payload}

    @staticmethod
    def _normalize_gateway_status(value: Any) -> str:
        raw = str(value or "").strip().lower().replace("-", "_")
        if raw in {"captured", "capture_succeeded", "succeeded", "paid", "payment_captured"}:
            return "payment_captured"
        if raw in {"authorized", "payment_authorized"}:
            return "payment_authorized"
        if raw in {"refund_pending", "refunded", "failed", "canceled", "cancelled"}:
            return "canceled" if raw == "cancelled" else raw
        return "payment_captured" if raw == "complete" else raw or "payment_authorized"

    @staticmethod
    def _normalize_fulfillment_status(value: Any) -> str:
        raw = str(value or "").strip().lower().replace("-", "_")
        if raw in {"success", "succeeded", "completed", "settled"}:
            return "complete"
        if raw in {"transfer_submitted", "swap_submitted", "complete", "failed"}:
            return raw
        return "transfer_submitted"

    @staticmethod
    def no_wallet_credit_created() -> bool:
        return WalletTransaction.query.count() == 0 and WalletLedgerEvent.query.count() == 0
