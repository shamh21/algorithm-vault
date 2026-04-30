"""Deposit address provider backed by configured custody addresses."""

from __future__ import annotations

import os
import re
from typing import Any, Protocol

from flask import current_app, has_app_context

from ..extensions import db
from ..models import DepositAddress, Setting, WalletAddress


class DepositAddressProvider(Protocol):
    """Public-address provider interface for deposit address assignment."""

    def next_address(self, user_id: int, asset: str, network: str) -> str | None:
        ...

    def has_configured_addresses(self, asset: str, network: str) -> bool:
        ...

    def configured_networks(self, asset: str) -> list[str]:
        ...

    def configured_assets(self) -> list[str]:
        ...


class ConfiguredAddressPoolProvider:
    """Returns only configured public custody or exchange deposit addresses."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def next_address(self, user_id: int, asset: str, network: str) -> str | None:
        asset_key = self._asset_key(asset)
        network_key = self._network_key(network)
        addresses = self._configured_addresses(asset_key, network_key)

        if not addresses:
            return None

        used = self._used_addresses(user_id, asset_key, network_key)

        for address in addresses:
            if address not in used and self._looks_valid(asset_key, network_key, address):
                return address

        return None

    def has_configured_addresses(self, asset: str, network: str) -> bool:
        return bool(self._configured_addresses(self._asset_key(asset), self._network_key(network)))

    def configured_networks(self, asset: str) -> list[str]:
        asset_key = self._asset_key(asset)
        book = self.config.get("DEPOSIT_ADDRESS_BOOK", {}) or {}
        networks: list[str] = []

        for configured_asset, raw_networks in book.items():
            if self._asset_key(configured_asset) != asset_key or not isinstance(raw_networks, dict):
                continue
            networks.extend(str(network) for network in raw_networks if str(network).strip())

        prefix = f"DEPOSIT_ADDRESS_{asset_key}_"
        for key in os.environ:
            if key.startswith(prefix):
                networks.append(key.removeprefix(prefix))

        return list(dict.fromkeys(networks))

    def configured_assets(self) -> list[str]:
        book = self.config.get("DEPOSIT_ADDRESS_BOOK", {}) or {}
        assets = [self._asset_key(asset) for asset in book if self._asset_key(asset)]

        for key in os.environ:
            if not key.startswith("DEPOSIT_ADDRESS_"):
                continue
            parts = key.split("_")
            if len(parts) >= 3:
                assets.append(self._asset_key(parts[2]))

        return list(dict.fromkeys(assets))

    def _configured_addresses(self, asset_key: str, network_key: str) -> list[str]:
        book = self.config.get("DEPOSIT_ADDRESS_BOOK", {}) or {}
        asset_book = book.get(asset_key, {})
        if not asset_book:
            for configured_asset, networks in book.items():
                if self._asset_key(configured_asset) == asset_key and isinstance(networks, dict):
                    asset_book = networks
                    break

        configured = asset_book.get(network_key) or asset_book.get(network_key.lower()) or []
        if not configured:
            for configured_network, value in asset_book.items():
                if self._network_key(configured_network) == network_key:
                    configured = value
                    break

        env_key = f"DEPOSIT_ADDRESS_{asset_key}_{network_key}"
        env_value = os.getenv(env_key, "").strip()

        addresses: list[str] = []

        if isinstance(configured, str):
            addresses.extend(self._split_addresses(configured))
        elif isinstance(configured, list):
            addresses.extend(str(item).strip() for item in configured if str(item).strip())

        if env_value:
            addresses.extend(self._split_addresses(env_value))

        return list(dict.fromkeys(addresses))

    def _used_addresses(self, user_id: int, asset_key: str, network_key: str) -> set[str]:
        return {
            row.address
            for row in DepositAddress.query.filter_by(user_id=user_id, asset=asset_key).all()
            if self._network_key(row.network) == network_key
        }

    def _looks_valid(self, asset: str, network: str, address: str) -> bool:
        network_key = self._network_key(network)
        address = str(address or "").strip()

        if not address:
            return False

        if network_key in {"ETHEREUM", "ARBITRUM", "OPTIMISM", "BASE", "POLYGON", "AVALANCHE", "BSC"}:
            return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", address))

        if network_key == "BITCOIN":
            return bool(re.fullmatch(r"(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,90}", address))

        if network_key == "SOLANA":
            return bool(re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", address))

        if network_key == "TRON":
            return bool(re.fullmatch(r"T[1-9A-HJ-NP-Za-km-z]{33}", address))

        if network_key == "XRPLEDGER":
            return bool(re.fullmatch(r"r[1-9A-HJ-NP-Za-km-z]{24,34}", address))

        return len(address) >= 24

    @staticmethod
    def _asset_key(asset: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", str(asset or "")).upper()

    @staticmethod
    def _network_key(network: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", str(network or "")).upper()

    @staticmethod
    def _split_addresses(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]


class WalletAddressService:
    """Coordinates configured public address providers without custody secrets."""

    def __init__(self, config: dict[str, Any], providers: list[DepositAddressProvider] | None = None) -> None:
        self.config = config
        self.providers = providers or [ConfiguredAddressPoolProvider(config)]

    def next_address(self, user_id: int, asset: str, network: str) -> str | None:
        for provider in self.providers:
            address = provider.next_address(user_id, asset, network)
            if address:
                return address
        return None

    def has_configured_addresses(self, asset: str, network: str) -> bool:
        return any(provider.has_configured_addresses(asset, network) for provider in self.providers)

    def configured_networks(self, asset: str) -> list[str]:
        networks: list[str] = []
        for provider in self.providers:
            networks.extend(provider.configured_networks(asset))
        return list(dict.fromkeys(networks))

    def configured_assets(self) -> list[str]:
        assets: list[str] = []
        for provider in self.providers:
            assets.extend(provider.configured_assets())
        return list(dict.fromkeys(assets))


def generate_deposit_address(asset: str, user: Any, network: str = "native", *, force_new: bool = False) -> str:
    """Generate or acquire a deposit address for the current address mode."""

    asset_key = _asset_key(asset)
    network_name = str(network or "native").strip() or "native"
    user_id = int(getattr(user, "id", user) or 0)
    if user_id <= 0:
        raise ValueError("A persisted user is required to generate a deposit address.")

    service = _service()
    address = service.next_address(user_id, asset_key, network_name)
    if address:
        return address
    if has_app_context():
        custody = current_app.extensions.get("services", {}).get("wallet_custody")
        if custody is not None and custody.enabled:
            wallet_address = custody.get_or_create_address(
                user_id=user_id,
                asset=asset_key,
                network=network_name,
                force_new=force_new,
            )
            return wallet_address.address
    raise RuntimeError(f"No real wallet address provider is configured for {asset_key} on {network_name}.")


def get_or_create_address(asset: str, user: Any, network: str | None = None) -> DepositAddress:
    """Return an active deposit address, creating one when needed."""

    asset_key = _asset_key(asset)
    network_name = str(network or "native").strip() or "native"
    user_id = int(getattr(user, "id", user) or 0)
    if user_id <= 0:
        raise ValueError("A persisted user is required to fetch a deposit address.")

    existing = (
        DepositAddress.query.filter_by(user_id=user_id, asset=asset_key, network=network_name, is_active=True)
        .order_by(DepositAddress.version.desc())
        .first()
    )
    if existing is not None:
        _link_generated_wallet_address(existing)
        return existing

    latest = (
        DepositAddress.query.filter_by(user_id=user_id, asset=asset_key, network=network_name)
        .order_by(DepositAddress.version.desc())
        .first()
    )
    address = DepositAddress(
        user_id=user_id,
        asset=asset_key,
        network=network_name,
        address=generate_deposit_address(asset_key, user_id, network_name),
        version=(latest.version if latest is not None else 0) + 1,
        is_active=True,
    )
    db.session.add(address)
    db.session.flush()
    _link_generated_wallet_address(address)
    return address


def validate_withdraw_address(address: str, asset: str, network: str | None = None) -> bool:
    """Validate a destination withdrawal address for common crypto networks."""

    address = str(address or "").strip()
    if not address:
        return False

    asset_key = _asset_key(asset)
    network_key = _network_key(network or asset_key)
    if address.startswith("TEST-"):
        return False
    if network_key in {"ETHEREUM", "ARBITRUM", "OPTIMISM", "BASE", "POLYGON", "AVALANCHE", "BSC"}:
        return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", address))
    if network_key == "TRON":
        return bool(re.fullmatch(r"T[1-9A-HJ-NP-Za-km-z]{33}", address))
    if network_key == "SOLANA" or asset_key == "SOL":
        return bool(re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", address))
    if network_key == "XRPLEDGER" or asset_key == "XRP":
        return bool(re.fullmatch(r"r[1-9A-HJ-NP-Za-km-z]{24,34}", address))
    if network_key == "BITCOIN" or asset_key == "BTC":
        return bool(re.fullmatch(r"(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,90}", address))
    if asset_key in {"ETH", "USDC", "USDT"}:
        return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", address))
    return bool(re.fullmatch(r"[A-Za-z0-9:_-]{24,128}", address))


def use_real_addresses(config: dict[str, Any] | None = None) -> bool:
    """Return the runtime address mode, using settings when available."""

    default = bool((config or {}).get("USE_REAL_ADDRESSES", False))
    if has_app_context():
        return bool(Setting.get_json("use_real_addresses", default))
    return default


def _link_generated_wallet_address(deposit_address: DepositAddress) -> None:
    wallet_address = (
        WalletAddress.query.filter_by(
            user_id=deposit_address.user_id,
            asset=deposit_address.asset,
            network=deposit_address.network,
            address=deposit_address.address,
            status="active",
        )
        .order_by(WalletAddress.rotation_index.desc())
        .first()
    )
    if wallet_address is not None and wallet_address.deposit_address_id != deposit_address.id:
        wallet_address.deposit_address_id = deposit_address.id


def _service() -> WalletAddressService:
    if has_app_context():
        return current_app.extensions["services"]["wallet_address_service"]
    raise RuntimeError("Wallet address generation requires an application context.")


def _asset_key(asset: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(asset or "")).upper()


def _network_key(network: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(network or "")).upper()
