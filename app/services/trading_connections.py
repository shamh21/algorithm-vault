"""User-scoped trading connection storage and connector adapters."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from cryptography.fernet import Fernet, InvalidToken
from flask import current_app

from ..extensions import db
from ..models import TradingConnection
from .hyperliquid_client import ClientSnapshot, HyperliquidClient
from .live_provider_adapters import (
    BinanceFuturesConnector,
    DydxV4Connector,
    KucoinFuturesConnector,
    UniswapDelegatedConnector,
)

SUPPORTED_PROVIDERS = {"hyperliquid", "binance", "kucoin", "uniswap", "dydx"}
SUPPORTED_CONNECTION_TYPES = {"cex_api_key", "dex_wallet", "permissioned_key", "wallet_delegation"}
VERIFIED_STATUS = "verified"
NEEDS_VERIFICATION_STATUS = "needs_verification"
ACTION_NEEDED_STATUS = "action_needed"
NOT_SUPPORTED_STATUS = "not_supported"


PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "hyperliquid": {
        "key": "hyperliquid",
        "label": "Hyperliquid",
        "connection_type": "cex_api_key",
        "tradable": True,
        "verification_supported": True,
        "summary": "Live-ready perpetuals connection for balances, positions, orders, and vault execution.",
        "setup_hint": "Create or use a Hyperliquid API wallet/agent with trading permissions, then enter the API wallet secret and account address. Never enter a recovery phrase.",
        "help_steps": [
            "Create an API wallet/agent with trading permission only.",
            "Keep withdrawal permission disabled unless Vault Cycle automatic settlement is explicitly enabled and capped.",
            "Enter the API wallet/agent private key or secret, plus the account address, then verify.",
            "Do not paste a seed phrase or your main wallet recovery phrase.",
        ],
        "fields": [
            {"name": "api_secret", "label": "API Wallet Secret", "type": "password", "required": True, "placeholder": "0x... API wallet/agent secret, encrypted at rest"},
            {"name": "wallet_address", "label": "Account Address", "type": "text", "required": True, "placeholder": "0x..."},
            {"name": "api_key", "label": "Account Label", "type": "text", "required": False, "placeholder": "Optional label or address"},
        ],
        "capabilities": ["Live orders", "Balances", "Positions", "Open orders", "USDC bridge withdrawals"],
    },
    "binance": {
        "key": "binance",
        "label": "Binance",
        "connection_type": "cex_api_key",
        "tradable": True,
        "verification_supported": True,
        "summary": "Live USD-M futures connection for balances, positions, orders, and vault execution.",
        "setup_hint": "Create a Binance USD-M Futures API key with futures trading enabled. Do not enable withdrawals.",
        "help_steps": [
            "Enable USD-M Futures on Binance before creating the key.",
            "Grant futures trading permission only; leave withdrawals disabled.",
            "Use IP restrictions when Binance offers them, then verify the connection.",
        ],
        "fields": [
            {"name": "api_key", "label": "API Key", "type": "text", "required": True, "placeholder": "Binance API key"},
            {"name": "api_secret", "label": "API Secret", "type": "password", "required": True, "placeholder": "Encrypted at rest"},
        ],
        "capabilities": ["USD-M futures orders", "Balances", "Positions", "Open orders", "Panic flatten"],
    },
    "kucoin": {
        "key": "kucoin",
        "label": "KuCoin",
        "connection_type": "cex_api_key",
        "tradable": True,
        "verification_supported": True,
        "summary": "Live KuCoin connection for spot balances, spot order testing, and guarded spot/futures execution where explicitly configured.",
        "setup_hint": "Create a KuCoin API key with General permission for read-only checks and Spot permission for spot order placement/cancellation. Keep Withdrawal disabled.",
        "help_steps": [
            "Create a KuCoin API key with General permission; add Spot permission only when spot order placement is required.",
            "Enter the key, secret, and API passphrase exactly as created.",
            "Keep Withdrawal disabled. Do not enable futures, margin, transfer, copy-trading, or earn permissions for spot-only validation.",
        ],
        "fields": [
            {"name": "api_key", "label": "API Key", "type": "text", "required": True, "placeholder": "KuCoin API key"},
            {"name": "api_secret", "label": "API Secret", "type": "password", "required": True, "placeholder": "Encrypted at rest"},
            {"name": "passphrase", "label": "Passphrase", "type": "password", "required": True, "placeholder": "KuCoin API passphrase"},
        ],
        "capabilities": ["Spot balances", "Spot market metadata", "Spot test orders", "Spot order create/cancel", "Futures execution only when explicitly configured"],
    },
    "uniswap": {
        "key": "uniswap",
        "label": "Uniswap",
        "connection_type": "wallet_delegation",
        "tradable": True,
        "verification_supported": True,
        "summary": "Wallet-delegated Uniswap swaps through the Uniswap Trading API.",
        "setup_hint": "Connect a wallet through WalletConnect/Reown, then save delegation limits. Seed phrases and private keys are never accepted.",
        "help_steps": [
            "Connect the wallet you want to trade from and approve a limited delegation.",
            "Set an expiry, allowed tokens/protocols, and daily loss cap.",
            "Verification fails closed until delegation status is approved and the Uniswap API key is configured.",
        ],
        "fields": [
            {"name": "wallet_address", "label": "Public Wallet Address", "type": "text", "required": True, "placeholder": "0x..."},
            {"name": "chain_id", "label": "Chain ID", "type": "number", "required": True, "placeholder": "1", "storage": "metadata"},
            {"name": "delegation_status", "label": "Delegation Status", "type": "text", "required": True, "placeholder": "approved", "storage": "metadata"},
            {"name": "delegation_expires_at", "label": "Delegation Expiry", "type": "datetime-local", "required": True, "placeholder": "YYYY-MM-DDTHH:MM", "storage": "metadata"},
            {"name": "daily_loss_usd", "label": "Daily Loss Cap USD", "type": "number", "required": True, "placeholder": "25", "storage": "metadata"},
            {"name": "allowed_tokens", "label": "Allowed Tokens", "type": "text", "required": True, "placeholder": "ETH,BTC", "storage": "metadata"},
            {"name": "protocols", "label": "Protocols", "type": "text", "required": False, "placeholder": "V2,V3,V4", "storage": "metadata"},
            {"name": "session_topic", "label": "WalletConnect Session", "type": "text", "required": True, "placeholder": "Reown session topic/reference", "storage": "metadata"},
        ],
        "capabilities": ["Delegated swaps", "Permit2-aware routing", "Daily loss cap"],
    },
    "dydx": {
        "key": "dydx",
        "label": "dYdX",
        "connection_type": "permissioned_key",
        "tradable": True,
        "verification_supported": True,
        "summary": "Live dYdX v4 perpetuals connection using permissioned API trading keys.",
        "setup_hint": "Use a dYdX permissioned API trading key. Seed phrases are rejected; use only the limited trading private key.",
        "help_steps": [
            "Create a permissioned API key from dYdX API Trading Keys.",
            "Save the one-time private key, owner wallet address, subaccount number, and authenticator id.",
            "Verification checks the permissioned setup and requires the dYdX order-signing SDK/executor.",
        ],
        "fields": [
            {"name": "wallet_address", "label": "Owner Wallet Address", "type": "text", "required": True, "placeholder": "dydx1..."},
            {"name": "api_key", "label": "API Wallet Address", "type": "text", "required": False, "placeholder": "Optional permissioned wallet address"},
            {"name": "api_secret", "label": "Permissioned Trading Private Key", "type": "password", "required": True, "placeholder": "Encrypted at rest"},
            {"name": "subaccount_number", "label": "Subaccount Number", "type": "number", "required": True, "placeholder": "0", "storage": "metadata"},
            {"name": "authenticator_id", "label": "Authenticator ID", "type": "text", "required": True, "placeholder": "Authenticator id", "storage": "metadata"},
        ],
        "capabilities": ["dYdX perpetuals", "Permissioned order keys", "Positions", "Panic flatten"],
    },
}


@dataclass(frozen=True, slots=True)
class TradingCredentials:
    """Decrypted credentials kept in memory only for one execution path."""

    provider: str
    connection_type: str
    api_key: str
    api_secret: str
    passphrase: str
    wallet_address: str


def _balance_available(balances: list[dict[str, Any]], asset: str) -> float:
    asset_key = str(asset or "").upper().strip()
    for row in balances or []:
        row_asset = str(row.get("asset") or row.get("currency") or "").upper().strip()
        if row_asset != asset_key:
            continue
        for key in ("withdrawable", "available", "available_balance", "free", "value", "total"):
            try:
                value = float(row.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value
    return 0.0


class TradingConnector(Protocol):
    """Provider adapter interface for user-scoped live execution."""

    def can_trade(self, mode: str) -> bool:
        ...

    def account_snapshot(self, mode: str) -> ClientSnapshot:
        ...

    def place_order(
        self,
        mode: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        limit_price: float | None,
        reduce_only: bool,
        leverage: float,
        slippage_pct: float,
        *,
        client_order_id: str | None = None,
        time_in_force: str | None = None,
    ) -> dict[str, Any]:
        ...

    def cancel_order(
        self,
        mode: str,
        symbol: str,
        exchange_order_id: str,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        ...

    def cancel_all_orders(self, mode: str) -> list[dict[str, Any]]:
        ...

    def flatten_all_positions(self, mode: str) -> list[dict[str, Any]]:
        ...

    def get_positions(self, mode: str) -> list[dict[str, Any]]:
        ...

    def get_recent_fills(self, mode: str) -> list[dict[str, Any]]:
        ...

    def discover_leveraged_markets(self, mode: str) -> list[dict[str, Any]]:
        ...

    def withdraw_from_bridge(self, mode: str, amount: float, destination: str) -> dict[str, Any]:
        ...

    def deposit_address(self, mode: str, asset: str, network: str | None = None) -> dict[str, Any]:
        ...

    def reserve_funds(self, mode: str, asset: str, amount: float) -> dict[str, Any]:
        ...

    def withdraw_to_address(
        self,
        mode: str,
        asset: str,
        amount: float,
        destination: str,
        network: str | None = None,
        memo: str | None = None,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        ...

    def transfer_status(self, mode: str, provider_reference: str, transfer_type: str | None = None) -> dict[str, Any]:
        ...

    def convert_stablecoin(
        self,
        mode: str,
        from_asset: str,
        to_asset: str,
        amount: float,
        max_slippage_bps: float,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        ...


class HyperliquidTradingConnector:
    """Hyperliquid adapter backed by a per-user encrypted key/address."""

    def __init__(self, config: dict[str, Any], credentials: TradingCredentials) -> None:
        if not credentials.api_secret:
            raise RuntimeError("Hyperliquid API secret is required.")
        if not credentials.wallet_address and not credentials.api_key:
            raise RuntimeError("Hyperliquid account address is required.")

        client_config = dict(config)
        client_config["HL_SECRET_KEY"] = credentials.api_secret
        client_config["HL_ACCOUNT_ADDRESS"] = credentials.wallet_address or credentials.api_key
        self.client = HyperliquidClient(client_config)

    def can_trade(self, mode: str) -> bool:
        return self.client.can_trade(mode)

    def account_snapshot(self, mode: str) -> ClientSnapshot:
        return self.client.account_snapshot(mode)

    def place_order(
        self,
        mode: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        limit_price: float | None,
        reduce_only: bool,
        leverage: float,
        slippage_pct: float,
        *,
        client_order_id: str | None = None,
        time_in_force: str | None = None,
    ) -> dict[str, Any]:
        return self.client.place_order(
            mode,
            symbol,
            side,
            quantity,
            order_type,
            limit_price,
            reduce_only,
            leverage,
            slippage_pct,
            client_order_id=client_order_id,
            time_in_force=time_in_force,
        )

    def cancel_order(
        self,
        mode: str,
        symbol: str,
        exchange_order_id: str,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        return self.client.cancel_order(mode, symbol, exchange_order_id, client_order_id=client_order_id)

    def cancel_all_orders(self, mode: str) -> list[dict[str, Any]]:
        return self.client.cancel_all_orders(mode)

    def flatten_all_positions(self, mode: str) -> list[dict[str, Any]]:
        return self.client.flatten_all_positions(mode)

    def get_positions(self, mode: str) -> list[dict[str, Any]]:
        return self.client.get_positions(mode)

    def get_recent_fills(self, mode: str) -> list[dict[str, Any]]:
        return self.client.get_recent_fills(mode)

    def get_open_orders(self, mode: str) -> list[dict[str, Any]]:
        return self.client.get_open_orders(mode)

    def get_balances(self, mode: str) -> list[dict[str, Any]]:
        return self.client.get_balances(mode)

    def get_order_status(
        self,
        mode: str,
        *,
        exchange_order_id: str | int | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        return self.client.get_order_status(mode, exchange_order_id=exchange_order_id, client_order_id=client_order_id)

    def live_test_order_plan(self, mode: str = "testnet", *, side: str = "buy", require_funds: bool = True) -> dict[str, Any]:
        return self.client.live_test_order_plan(mode, side=side, require_funds=require_funds)

    def discover_leveraged_markets(self, mode: str) -> list[dict[str, Any]]:
        if mode != "live":
            return []
        meta, contexts = self.client.get_perp_meta_and_asset_contexts(mode)
        universe = meta.get("universe", []) if isinstance(meta, dict) else []
        rows: list[dict[str, Any]] = []
        for index, asset in enumerate(universe):
            if not isinstance(asset, dict):
                continue
            row = dict(asset)
            if index < len(contexts) and isinstance(contexts[index], dict):
                row["_asset_context"] = dict(contexts[index])
            rows.append(row)
        return rows

    def withdraw_from_bridge(self, mode: str, amount: float, destination: str) -> dict[str, Any]:
        return self.client.withdraw_from_bridge(mode, amount, destination)

    def deposit_address(self, mode: str, asset: str, network: str | None = None) -> dict[str, Any]:
        asset_key = str(asset or "").upper().strip()
        network_key = str(network or "Arbitrum").strip() or "Arbitrum"
        if mode != "live" or asset_key != "USDC":
            raise RuntimeError("Hyperliquid Vault Cycle funding supports live USDC only.")
        return {
            "asset": asset_key,
            "network": network_key,
            "address": getattr(self.client, "account_address", "") or "",
            "memo": "",
            "status": "manual_bridge_required",
            "raw": {
                "rail": "hyperliquid_bridge2",
                "credit_rule": "USDC bridge deposits credit the sending account on Hyperliquid.",
            },
        }

    def reserve_funds(self, mode: str, asset: str, amount: float) -> dict[str, Any]:
        snapshot = self.account_snapshot(mode)
        asset_key = str(asset or "").upper().strip()
        requested = max(0.0, float(amount or 0.0))
        available = _balance_available(snapshot.balances, asset_key)
        if available + 1e-9 < requested:
            raise RuntimeError(f"Hyperliquid {asset_key} reserve unavailable: {available:.6f} < {requested:.6f}.")
        return {
            "status": "confirmed",
            "provider_reference": f"hyperliquid-reserve-{int(time.time() * 1000)}",
            "asset": asset_key,
            "confirmed_amount": requested,
            "available_before": available,
            "raw": {"reserve_type": "exchange_balance"},
        }

    def withdraw_to_address(
        self,
        mode: str,
        asset: str,
        amount: float,
        destination: str,
        network: str | None = None,
        memo: str | None = None,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        asset_key = str(asset or "").upper().strip()
        if asset_key != "USDC":
            raise RuntimeError("Hyperliquid bridge withdrawals support USDC only.")
        response = self.withdraw_from_bridge(mode, amount, destination)
        return {
            **response,
            "asset": asset_key,
            "network": str(network or "Arbitrum"),
            "client_reference": client_reference,
        }

    def transfer_status(self, mode: str, provider_reference: str, transfer_type: str | None = None) -> dict[str, Any]:
        return {
            "status": "submitted" if provider_reference else "unknown",
            "provider_reference": str(provider_reference or ""),
            "transfer_type": transfer_type or "withdrawal",
            "raw": {},
        }

    def convert_stablecoin(
        self,
        mode: str,
        from_asset: str,
        to_asset: str,
        amount: float,
        max_slippage_bps: float,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError("Hyperliquid connector does not provide USDT/USDC conversion.")


class UnsupportedTradingConnector:
    """Fail-closed placeholder for future providers."""

    def __init__(self, provider: str) -> None:
        self.provider = provider

    def can_trade(self, mode: str) -> bool:
        return False

    def account_snapshot(self, mode: str) -> ClientSnapshot:
        return ClientSnapshot(mode, [], [], [], [], [f"{self.provider} connector is not implemented yet."])

    def place_order(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(f"{self.provider} connector is not implemented yet.")

    def cancel_order(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(f"{self.provider} connector is not implemented yet.")

    def cancel_all_orders(self, mode: str) -> list[dict[str, Any]]:
        return []

    def flatten_all_positions(self, mode: str) -> list[dict[str, Any]]:
        return []

    def get_positions(self, mode: str) -> list[dict[str, Any]]:
        return []

    def discover_leveraged_markets(self, mode: str) -> list[dict[str, Any]]:
        return []

    def withdraw_from_bridge(self, mode: str, amount: float, destination: str) -> dict[str, Any]:
        raise RuntimeError(f"{self.provider} connector does not support withdrawals.")

    def deposit_address(self, mode: str, asset: str, network: str | None = None) -> dict[str, Any]:
        raise RuntimeError(f"{self.provider} connector does not support deposit address lookup.")

    def reserve_funds(self, mode: str, asset: str, amount: float) -> dict[str, Any]:
        raise RuntimeError(f"{self.provider} connector does not support exchange reserve accounting.")

    def withdraw_to_address(
        self,
        mode: str,
        asset: str,
        amount: float,
        destination: str,
        network: str | None = None,
        memo: str | None = None,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(f"{self.provider} connector does not support automated withdrawals.")

    def transfer_status(self, mode: str, provider_reference: str, transfer_type: str | None = None) -> dict[str, Any]:
        return {"status": "unsupported", "provider_reference": str(provider_reference or ""), "raw": {}}

    def convert_stablecoin(
        self,
        mode: str,
        from_asset: str,
        to_asset: str,
        amount: float,
        max_slippage_bps: float,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(f"{self.provider} connector does not support stablecoin conversion.")


class TradingConnectionService:
    """Owns credential encryption, user isolation, and connector selection."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._snapshot_cache: dict[tuple[int, int, str], tuple[float, ClientSnapshot]] = {}
        self._snapshot_backoff: dict[tuple[int, int, str], tuple[float, str]] = {}
        self._snapshot_inflight: dict[tuple[int, int, str], threading.Event] = {}
        self._snapshot_guard = threading.Lock()

    def create_or_update(
        self,
        *,
        user_id: int,
        provider: str,
        connection_type: str,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        wallet_address: str = "",
        metadata: dict[str, Any] | None = None,
        is_active: bool = True,
    ) -> TradingConnection:
        provider = self._normalize_provider(provider)
        spec = self.provider_spec(provider)
        connection_type = self._normalize_connection_type(connection_type or spec["connection_type"])
        self._reject_seed_phrase(api_key, "API key")
        self._reject_seed_phrase(api_secret, "API secret")
        self._reject_seed_phrase(passphrase, "Passphrase")
        for key, value in (metadata or {}).items():
            self._reject_seed_phrase(str(value), str(key).replace("_", " ").title())
        self._validate_provider_secret_shape(provider, api_secret)

        connection = TradingConnection.query.filter_by(
            user_id=user_id,
            provider=provider,
            connection_type=connection_type,
        ).one_or_none()
        if connection is None:
            connection = TradingConnection(user_id=user_id, provider=provider, connection_type=connection_type)
            db.session.add(connection)

        metadata = {key: str(value).strip() for key, value in (metadata or {}).items() if str(value).strip()}
        existing_metadata = connection.provider_metadata if connection.id else {}
        merged_metadata = {
            **existing_metadata,
            **self._metadata_for_spec(spec),
            **metadata,
        }

        self._validate_provider_fields(spec, connection, api_key, api_secret, passphrase, wallet_address, merged_metadata)

        if api_key:
            connection.encrypted_api_key = self._encrypt(api_key)
        if api_secret:
            connection.encrypted_api_secret = self._encrypt(api_secret)
        if passphrase:
            connection.encrypted_passphrase = self._encrypt(passphrase)
        if wallet_address or connection.wallet_address is None:
            connection.wallet_address = wallet_address.strip()
        connection.connection_type = connection_type
        connection.provider_metadata = merged_metadata
        connection.last_verified_at = None
        connection.last_verification_error = None
        connection.verification_status = NEEDS_VERIFICATION_STATUS if spec["tradable"] else NOT_SUPPORTED_STATUS
        connection.is_active = False

        if is_active and self._is_verified_tradable(connection):
            connection.is_active = True

        db.session.flush()
        return connection

    def delete(self, *, user_id: int, connection_id: int) -> None:
        connection = self.get_for_user(user_id, connection_id)
        db.session.delete(connection)

    def get_for_user(self, user_id: int, connection_id: int) -> TradingConnection:
        connection = db.session.get(TradingConnection, int(connection_id))
        if connection is None or connection.user_id != int(user_id):
            raise PermissionError("Trading connection was not found for this user.")
        return connection

    def active_connection(self, user_id: int, provider: str | None = None) -> TradingConnection | None:
        query = TradingConnection.query.filter_by(user_id=int(user_id), is_active=True)
        if provider:
            query = query.filter_by(provider=self._normalize_provider(provider))
        return query.order_by(TradingConnection.updated_at.desc(), TradingConnection.id.desc()).first()

    def active_tradable_connection(self, user_id: int, provider: str | None = None) -> TradingConnection | None:
        query = TradingConnection.query.filter_by(
            user_id=int(user_id),
            is_active=True,
            verification_status=VERIFIED_STATUS,
        )
        if provider:
            query = query.filter_by(provider=self._normalize_provider(provider))
        for connection in query.order_by(TradingConnection.updated_at.desc(), TradingConnection.id.desc()).all():
            if self.provider_spec(connection.provider)["tradable"]:
                return connection
        return None

    def verified_tradable_connections(self, user_id: int, providers: list[str] | None = None) -> list[TradingConnection]:
        provider_filter = {self._normalize_provider(provider) for provider in (providers or []) if str(provider or "").strip()}
        query = TradingConnection.query.filter_by(user_id=int(user_id), verification_status=VERIFIED_STATUS)
        connections: list[TradingConnection] = []
        for connection in query.order_by(TradingConnection.updated_at.desc(), TradingConnection.id.desc()).all():
            if provider_filter and connection.provider not in provider_filter:
                continue
            if self.provider_spec(connection.provider)["tradable"]:
                connections.append(connection)
        return connections

    def enabled_tradable_connections(self, user_id: int, providers: list[str] | None = None) -> list[TradingConnection]:
        provider_filter = {self._normalize_provider(provider) for provider in (providers or []) if str(provider or "").strip()}
        query = TradingConnection.query.filter_by(
            user_id=int(user_id),
            is_active=True,
            verification_status=VERIFIED_STATUS,
        )
        connections: list[TradingConnection] = []
        for connection in query.order_by(TradingConnection.updated_at.desc(), TradingConnection.id.desc()).all():
            if provider_filter and connection.provider not in provider_filter:
                continue
            if self.provider_spec(connection.provider)["tradable"]:
                connections.append(connection)
        return connections

    def has_active_connection(self, user_id: int) -> bool:
        return self.active_tradable_connection(user_id) is not None

    def can_trade(self, user_id: int | None, mode: str, connection_id: int | None = None) -> bool:
        if mode != "live":
            return False
        if not bool(self.config.get("ENABLE_LIVE_TRADING", False)):
            return False
        if user_id is None:
            return False
        try:
            connection = self.get_for_user(user_id, connection_id) if connection_id else self.active_tradable_connection(user_id)
            requires_active = connection_id is None
            if connection is None or (requires_active and not connection.is_active) or not self._is_verified_tradable(connection):
                return False
            return self._connector_for_connection(connection).can_trade(mode)
        except Exception:  # noqa: BLE001 - connection checks fail closed on any provider/config error.
            return False

    def account_snapshot(self, user_id: int | None, mode: str, connection_id: int | None = None) -> ClientSnapshot:
        if user_id is None:
            return ClientSnapshot(mode, [], [], [], [], ["No authenticated user is available for live account data."])
        connection = self.get_for_user(user_id, connection_id) if connection_id else self.active_tradable_connection(user_id)
        if connection is None:
            return ClientSnapshot(mode, [], [], [], [], ["Connect and verify a trading account before live trading."])
        cache_key = (int(user_id), int(connection.id or 0), str(mode))

        while True:
            with self._snapshot_guard:
                cached = self._cached_snapshot_nolock(cache_key, mode)
                if cached is not None:
                    return cached

                backoff_alert = self._snapshot_backoff_alert_nolock(cache_key)
                if backoff_alert:
                    stale = self._cached_snapshot_nolock(cache_key, mode, allow_stale=True)
                    if stale is not None:
                        return stale
                    return ClientSnapshot(mode, [], [], [], [], [backoff_alert])

                inflight = self._snapshot_inflight.get(cache_key)
                if inflight is not None:
                    event = inflight
                else:
                    event = threading.Event()
                    self._snapshot_inflight[cache_key] = event
                    break

            event.wait(max(0.1, float(self._snapshot_cache_seconds(mode) or 0.0)))
            with self._snapshot_guard:
                # If another caller completed work, serve what they stored.
                cached = self._cached_snapshot_nolock(cache_key, mode)
                if cached is not None:
                    return cached

        try:
            connector = self.connector_for_user(int(user_id), int(connection.id or connection_id or 0) or None)
            snapshot_reader = getattr(connector, "account_snapshot", None)
            if callable(snapshot_reader):
                snapshot = snapshot_reader(mode)
            else:
                positions_reader = getattr(connector, "get_positions", None)
                positions = positions_reader(mode) if callable(positions_reader) else []
                snapshot = ClientSnapshot(mode, [], positions or [], [], [], [])
            clone = self._clone_snapshot(snapshot)
        except Exception as exc:  # noqa: BLE001
            if self._is_transient_provider_error(exc):
                self._record_snapshot_backoff(cache_key, exc)
                stale = self._cached_snapshot_nolock(cache_key, mode, allow_stale=True)
                if stale is not None:
                    return stale
            else:
                with self._snapshot_guard:
                    self._snapshot_backoff.pop(cache_key, None)
            return ClientSnapshot(mode, [], [], [], [], [str(exc)])
        finally:
            with self._snapshot_guard:
                inflight = self._snapshot_inflight.pop(cache_key, None)
                if inflight is not None:
                    inflight.set()

        if self._snapshot_has_transient_alert(clone):
            self._record_snapshot_backoff(cache_key, "; ".join(str(alert) for alert in clone.alerts))
        elif not clone.alerts:
            self._snapshot_backoff.pop(cache_key, None)

        with self._snapshot_guard:
            self._snapshot_cache[cache_key] = (time.time(), self._clone_snapshot(clone))

        return self._clone_snapshot(clone)

    def invalidate_account_snapshot(
        self,
        user_id: int,
        mode: str | None = None,
        connection_id: int | None = None,
    ) -> None:
        connection_filter = int(connection_id) if connection_id is not None else None
        mode_filter = str(mode).lower() if mode else None
        with self._snapshot_guard:
            for key in list(self._snapshot_cache.keys()):
                cache_user_id, cache_connection_id, cache_mode = key
                if int(cache_user_id) != int(user_id):
                    continue
                if connection_filter is not None and cache_connection_id != connection_filter:
                    continue
                if mode_filter is not None and cache_mode != mode_filter:
                    continue
                self._snapshot_cache.pop(key, None)
                self._snapshot_backoff.pop(key, None)

    def connector_for_user(self, user_id: int, connection_id: int | None = None) -> TradingConnector:
        connection = self.get_for_user(user_id, connection_id) if connection_id else self.active_tradable_connection(user_id)
        if connection is None:
            raise RuntimeError("Connect and verify a trading account before live trading.")
        return self._connector_for_connection(connection)

    def provider_specs(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in PROVIDER_SPECS.items()}

    def provider_spec(self, provider: str) -> dict[str, Any]:
        return dict(PROVIDER_SPECS[self._normalize_provider(provider)])

    def verify_connection(self, user_id: int, connection_id: int) -> dict[str, Any]:
        connection = self.get_for_user(user_id, connection_id)
        spec = self.provider_spec(connection.provider)
        connection.provider_metadata = {
            **self._metadata_for_spec(spec),
            **connection.provider_metadata,
        }

        if not spec["tradable"] or not spec["verification_supported"]:
            connection.is_active = False
            connection.verification_status = NOT_SUPPORTED_STATUS
            connection.last_verified_at = None
            connection.last_verification_error = f"{spec['label']} is saved as a draft. Live trading support is not implemented yet."
            db.session.flush()
            return {"ok": False, "connection": connection, "error": connection.last_verification_error}

        try:
            self._validate_saved_connection(spec, connection)
            connector = self._connector_for_connection(connection)
            if not connector.can_trade("live"):
                raise RuntimeError("Credentials are present but cannot trade in live mode.")
            snapshot = connector.account_snapshot("live")
            if snapshot.alerts:
                raise RuntimeError("; ".join(snapshot.alerts))
        except Exception as exc:  # noqa: BLE001
            normalized = self._normalize_provider_verification_error(connection.provider, exc)
            connection.is_active = False
            connection.verification_status = str(normalized.get("verification_status") or ACTION_NEEDED_STATUS)
            connection.last_verified_at = None
            connection.last_verification_error = str(normalized.get("client_error") or "Connection verification failed.")
            if normalized.get("diagnostics"):
                metadata = connection.provider_metadata
                metadata["provider_diagnostics"] = normalized["diagnostics"]
                connection.provider_metadata = metadata
            db.session.flush()
            return {"ok": False, "connection": connection, "error": connection.last_verification_error, "diagnostics": normalized.get("diagnostics", {})}

        connection.verification_status = VERIFIED_STATUS
        connection.last_verified_at = datetime.utcnow()
        connection.last_verification_error = None
        metadata = connection.provider_metadata
        metadata["last_verified_mode"] = "live"
        metadata["last_verified_balances"] = len(snapshot.balances)
        connection.provider_metadata = metadata
        db.session.flush()
        return {"ok": True, "connection": connection, "snapshot": snapshot}

    def _normalize_provider_verification_error(self, provider: str, exc: Exception) -> dict[str, Any]:
        raw = str(exc or "")
        provider_code = str(getattr(exc, "provider_code", "") or "")
        provider_key = self._normalize_provider(provider)
        if provider_key == "kucoin" and (provider_code == "400302" or self._is_kucoin_geo_restriction(raw)):
            detected_area = self._detected_area_from_text(raw) or ("US" if self._mentions_us(raw) else "restricted region")
            diagnostics = {
                "providerCode": provider_code or ("400302" if "400302" in raw else ""),
                "detectedArea": str(detected_area).upper() if len(str(detected_area)) <= 3 else str(detected_area),
                "egressRegion": os.environ.get("VERCEL_REGION") or os.environ.get("AWS_REGION") or "unknown",
                "maskedIp": self._mask_ip(raw),
            }
            diagnostics = {key: value for key, value in diagnostics.items() if value}
            client_error = f"Provider detected restricted region: {diagnostics.get('detectedArea', 'restricted region')}. Recheck after server-region update or use another supported exchange."
            current_app.logger.warning("KuCoin verification geo restriction diagnostics=%s", diagnostics)
            return {
                "verification_status": "geo_restricted",
                "client_error": client_error,
                "diagnostics": diagnostics,
            }
        return {
            "verification_status": ACTION_NEEDED_STATUS,
            "client_error": self._sanitize_provider_error(raw) or "Connection verification failed.",
            "diagnostics": {},
        }

    @staticmethod
    def _is_kucoin_geo_restriction(message: str) -> bool:
        lowered = str(message or "").lower()
        return "400302" in str(message or "") or any(
            phrase in lowered
            for phrase in (
                "unavailable in your current area",
                "unavailable in the detected region",
                "provider detected restricted region",
                "restricted region",
            )
        )

    @staticmethod
    def _mentions_us(message: str) -> bool:
        lowered = f" {str(message or '').lower()} "
        return " us " in lowered or " u.s." in lowered or "united states" in lowered

    @staticmethod
    def _sanitize_provider_error(message: str) -> str:
        text = re.sub(r"\b(\d{1,3})\.(\d{1,3})\.\d{1,3}\.\d{1,3}\b", r"\1.\2.xxx.xxx", str(message or ""))
        return text[:500]

    @staticmethod
    def _mask_ip(message: str) -> str:
        match = re.search(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b", str(message or ""))
        return f"{match.group(1)}.{match.group(2)}.xxx.xxx" if match else ""

    @staticmethod
    def _detected_area_from_text(message: str) -> str:
        text = str(message or "")
        for pattern in (r'"(?:detectedArea|detected_area|area|country|region)"\s*:\s*"?([A-Za-z]{2,32})', r"detected region[:= ]+([A-Za-z]{2,32})", r"current area[:= ]+([A-Za-z]{2,32})"):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return ""
        if isinstance(payload, dict):
            for key in ("detectedArea", "detected_area", "area", "country", "region"):
                if payload.get(key):
                    return str(payload[key])
        return ""

    def activate_verified(self, user_id: int, connection_id: int) -> TradingConnection:
        connection = self.get_for_user(user_id, connection_id)
        if not self._is_verified_tradable(connection):
            raise ValueError("Only verified live-ready connections can be activated.")
        connection.is_active = True
        db.session.flush()
        return connection

    def disable(self, user_id: int, connection_id: int) -> TradingConnection:
        connection = self.get_for_user(user_id, connection_id)
        connection.is_active = False
        db.session.flush()
        return connection

    def credentials_for_execution(self, user_id: int, connection_id: int) -> TradingCredentials:
        connection = self.get_for_user(user_id, connection_id)
        spec = self.provider_spec(connection.provider)
        return TradingCredentials(
            provider=connection.provider,
            connection_type=connection.connection_type,
            api_key=self._decrypt_connection_secret(spec, "api_key", connection.encrypted_api_key),
            api_secret=self._decrypt_connection_secret(spec, "api_secret", connection.encrypted_api_secret),
            passphrase=self._decrypt_connection_secret(spec, "passphrase", connection.encrypted_passphrase),
            wallet_address=connection.wallet_address or "",
        )

    def _connector_for_connection(self, connection: TradingConnection) -> TradingConnector:
        spec = self.provider_spec(connection.provider)
        if not spec["tradable"]:
            return UnsupportedTradingConnector(connection.provider)
        credentials = self.credentials_for_execution(connection.user_id, connection.id)
        if credentials.provider == "hyperliquid":
            return HyperliquidTradingConnector(self.config, credentials)
        if credentials.provider == "binance":
            return BinanceFuturesConnector(self.config, credentials, connection.provider_metadata)
        if credentials.provider == "kucoin":
            return KucoinFuturesConnector(self.config, credentials, connection.provider_metadata)
        if credentials.provider == "dydx":
            return DydxV4Connector(self.config, credentials, connection.provider_metadata)
        if credentials.provider == "uniswap":
            return UniswapDelegatedConnector(self.config, credentials, connection.provider_metadata)
        return UnsupportedTradingConnector(credentials.provider)

    def _is_verified_tradable(self, connection: TradingConnection) -> bool:
        return (
            connection.verification_status == VERIFIED_STATUS
            and self.provider_spec(connection.provider)["tradable"]
        )

    @staticmethod
    def _metadata_for_spec(spec: dict[str, Any]) -> dict[str, Any]:
        return {
            "provider_label": spec["label"],
            "tradable": bool(spec["tradable"]),
            "verification_supported": bool(spec["verification_supported"]),
            "capabilities": list(spec.get("capabilities", [])),
        }

    def _validate_provider_fields(
        self,
        spec: dict[str, Any],
        connection: TradingConnection,
        api_key: str,
        api_secret: str,
        passphrase: str,
        wallet_address: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if spec["connection_type"] in {"dex_wallet", "wallet_delegation"} and any([api_key, api_secret, passphrase]):
            raise ValueError("Wallet providers accept public wallet addresses only. Private keys and seed phrases are not accepted.")

        metadata = metadata or {}
        saved = {
            "api_key": bool(api_key or connection.encrypted_api_key),
            "api_secret": bool(api_secret or connection.encrypted_api_secret),
            "passphrase": bool(passphrase or connection.encrypted_passphrase),
            "wallet_address": bool(wallet_address or connection.wallet_address),
        }
        missing = []
        for field in spec["fields"]:
            if not field.get("required"):
                continue
            name = field["name"]
            if field.get("storage") == "metadata":
                if not str(metadata.get(name, "")).strip():
                    missing.append(field["label"])
            elif not saved.get(name):
                missing.append(field["label"])
        if missing:
            raise ValueError(f"Missing required field(s): {', '.join(missing)}.")

    @staticmethod
    def _validate_saved_connection(spec: dict[str, Any], connection: TradingConnection) -> None:
        saved = {
            "api_key": bool(connection.encrypted_api_key),
            "api_secret": bool(connection.encrypted_api_secret),
            "passphrase": bool(connection.encrypted_passphrase),
            "wallet_address": bool(connection.wallet_address),
        }
        metadata = connection.provider_metadata
        missing = []
        for field in spec["fields"]:
            if not field.get("required"):
                continue
            name = field["name"]
            if field.get("storage") == "metadata":
                if not str(metadata.get(name, "")).strip():
                    missing.append(field["label"])
            elif not saved.get(name):
                missing.append(field["label"])
        if missing:
            raise RuntimeError(f"Missing required field(s): {', '.join(missing)}.")

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        value = re.sub(r"[^a-z0-9_]", "", str(provider or "").strip().lower())
        if value not in SUPPORTED_PROVIDERS:
            raise ValueError("Unsupported trading provider.")
        return value

    def _cached_snapshot(self, key: tuple[int, int, str], mode: str, *, allow_stale: bool = False) -> ClientSnapshot | None:
        with self._snapshot_guard:
            return self._cached_snapshot_nolock(key, mode, allow_stale=allow_stale)

    def _cached_snapshot_nolock(self, key: tuple[int, int, str], mode: str, *, allow_stale: bool = False) -> ClientSnapshot | None:
        cached_at, snapshot = self._snapshot_cache.get(key, (0.0, None))
        if snapshot is None:
            return None
        ttl = self._snapshot_cache_seconds(mode)
        if allow_stale:
            ttl = max(ttl, self._snapshot_stale_seconds(mode))
        if ttl <= 0 or time.time() - cached_at > ttl:
            return None
        return self._clone_snapshot(snapshot)

    def _snapshot_cache_seconds(self, mode: str) -> float:
        if str(mode or "").lower() != "live":
            return 0.0
        try:
            return max(0.0, float(self.config.get("TRADING_CONNECTION_LIVE_SNAPSHOT_CACHE_SECONDS", self.config.get("ONE_H10_ACCOUNT_REFRESH_SECONDS", 20.0)) or 0.0))
        except (TypeError, ValueError):
            return 20.0

    def _snapshot_stale_seconds(self, mode: str) -> float:
        if str(mode or "").lower() != "live":
            return 0.0
        try:
            return max(self._snapshot_cache_seconds(mode), float(self.config.get("TRADING_CONNECTION_LIVE_SNAPSHOT_STALE_SECONDS", 120.0) or 0.0))
        except (TypeError, ValueError):
            return 120.0

    def _snapshot_backoff_seconds(self) -> float:
        try:
            return max(1.0, float(self.config.get("TRADING_CONNECTION_LIVE_SNAPSHOT_BACKOFF_SECONDS", self.config.get("ONE_H10_MARKET_DATA_BACKOFF_SECONDS", 30.0)) or 0.0))
        except (TypeError, ValueError):
            return 30.0

    def _snapshot_backoff_alert(self, key: tuple[int, int, str]) -> str | None:
        with self._snapshot_guard:
            return self._snapshot_backoff_alert_nolock(key)

    def _snapshot_backoff_alert_nolock(self, key: tuple[int, int, str]) -> str | None:
        until, reason = self._snapshot_backoff.get(key, (0.0, ""))
        if until <= time.time():
            self._snapshot_backoff.pop(key, None)
            return None
        return reason or "Live account snapshot is temporarily unavailable; retrying after provider backoff."

    def _record_snapshot_backoff(self, key: tuple[int, int, str], exc: object) -> None:
        with self._snapshot_guard:
            self._snapshot_backoff[key] = (
                time.time() + self._snapshot_backoff_seconds(),
                "Live account snapshot is temporarily unavailable; retrying after provider backoff.",
            )

    def _record_snapshot_backoff_nolock(self, key: tuple[int, int, str], exc: object) -> None:
        if not self._is_transient_provider_error(exc):
            return
        self._record_snapshot_backoff(key, exc)

    @classmethod
    def _snapshot_has_transient_alert(cls, snapshot: ClientSnapshot) -> bool:
        return any(cls._is_transient_provider_error(alert) for alert in (snapshot.alerts or []))

    @staticmethod
    def _is_transient_provider_error(value: object) -> bool:
        text = repr(value).lower()
        return any(
            marker in text
            for marker in (
                "429",
                "rate limit",
                "too many requests",
                "timed out",
                "timeout",
                "temporarily unavailable",
                "hyperliquid call failed",
                "cloudfront",
                "connection reset",
                "dns",
            )
        )

    @staticmethod
    def _clone_snapshot(snapshot: ClientSnapshot) -> ClientSnapshot:
        return ClientSnapshot(
            snapshot.mode,
            [dict(item) for item in snapshot.balances],
            [dict(item) for item in snapshot.positions],
            [dict(item) for item in snapshot.open_orders],
            [dict(item) for item in snapshot.recent_fills],
            [str(item) for item in snapshot.alerts],
        )

    @staticmethod
    def _normalize_connection_type(connection_type: str) -> str:
        value = re.sub(r"[^a-z0-9_]", "", str(connection_type or "").strip().lower())
        if value not in SUPPORTED_CONNECTION_TYPES:
            raise ValueError("Unsupported connection type.")
        return value

    @staticmethod
    def _reject_seed_phrase(value: str, label: str) -> None:
        raw = str(value or "").strip()
        if not raw or re.fullmatch(r"0x[0-9a-fA-F]{64}", raw):
            return
        words = re.findall(r"[A-Za-z]+", raw)
        parts = raw.split()
        if len(parts) in {12, 15, 18, 21, 24} and len(words) == len(parts):
            raise ValueError(
                f"{label} looks like a seed phrase. Seed phrases are not accepted. "
                "Use an exchange API secret or a Hyperliquid API wallet/agent secret instead."
            )

    @staticmethod
    def _validate_provider_secret_shape(provider: str, api_secret: str) -> None:
        if provider != "hyperliquid" or not api_secret:
            return
        value = str(api_secret).strip()
        if re.fullmatch(r"0x[0-9a-fA-F]{64}", value):
            return
        if "..." in value or len(value) < 66:
            raise ValueError("Hyperliquid API Wallet Secret must be the full 0x private key with 64 hex characters. Do not paste a shortened example.")
        if any(character.isspace() for character in value):
            raise ValueError("Hyperliquid API Wallet Secret must be one 0x private key string with no spaces.")
        raise ValueError("Hyperliquid API Wallet Secret must be a 0x private key with exactly 64 hex characters.")

    @staticmethod
    def _fernet() -> Fernet:
        configured = str(current_app.config.get("TOTP_ENCRYPTION_KEY", "") or "").strip()
        if configured:
            try:
                return Fernet(configured.encode("utf-8"))
            except Exception:  # noqa: BLE001
                pass
        raw = str(current_app.config.get("SECRET_KEY", "dev-secret-change-me")).encode("utf-8")
        key = base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
        return Fernet(key)

    def _encrypt(self, value: str) -> str:
        if not value:
            return ""
        return self._fernet().encrypt(value.encode("utf-8")).decode("utf-8")

    def _decrypt(self, value: str | None) -> str:
        if not value:
            return ""
        return self._fernet().decrypt(value.encode("utf-8")).decode("utf-8")

    def _decrypt_connection_secret(self, spec: dict[str, Any], field_name: str, value: str | None) -> str:
        if not value:
            return ""
        try:
            return self._decrypt(value)
        except InvalidToken as exc:
            if self._field_required(spec, field_name):
                label = self._field_label(spec, field_name)
                raise RuntimeError(
                    f"Saved {label} cannot be decrypted with current TOTP_ENCRYPTION_KEY. "
                    "Re-enter or delete this connection."
                ) from exc
            return ""

    @staticmethod
    def _field_required(spec: dict[str, Any], field_name: str) -> bool:
        return any(field.get("name") == field_name and bool(field.get("required")) for field in spec.get("fields", []))

    @staticmethod
    def _field_label(spec: dict[str, Any], field_name: str) -> str:
        for field in spec.get("fields", []):
            if field.get("name") == field_name:
                return str(field.get("label") or field_name.replace("_", " ").title())
        return field_name.replace("_", " ").title()
