"""Shared Vault allocation asset helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ..models import WalletBalance

BASE_VAULT_ALLOCATION_ASSETS = ("USDC", "USDT", "BTC", "ETH", "SOL", "XRP")
DEFAULT_ASSET_NETWORKS = {
    "BTC": ("Bitcoin",),
    "ETH": ("Ethereum",),
    "SOL": ("Solana",),
    "XRP": ("XRP Ledger",),
    "USDC": ("Ethereum",),
    "USDT": ("Ethereum",),
}
STABLE_ASSETS = {"USDC", "USDT"}


@dataclass(frozen=True, slots=True)
class VaultAllocationAssetView:
    asset: str
    label: str
    available_balance: float
    locked_balance: float
    total_balance: float
    estimated_usd_value: float
    price_usd: float
    available_usd: float
    cap_usd: float
    networks: tuple[str, ...]
    state: str
    configured: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "label": self.label,
            "available_balance": self.available_balance,
            "locked_balance": self.locked_balance,
            "total_balance": self.total_balance,
            "estimated_usd_value": self.estimated_usd_value,
            "price_usd": self.price_usd,
            "available_usd": self.available_usd,
            "cap_usd": self.cap_usd,
            "networks": list(self.networks),
            "state": self.state,
            "configured": self.configured,
        }


def normalize_asset(value: Any) -> str:
    return str(value or "").strip().upper()


def supported_vault_allocation_assets(configured_assets: Iterable[Any] | None = None) -> tuple[str, ...]:
    configured = [normalize_asset(asset) for asset in configured_assets or () if normalize_asset(asset)]
    return tuple(dict.fromkeys((*BASE_VAULT_ALLOCATION_ASSETS, *configured)))


def functional_wallet_network(asset: str, network: str) -> bool:
    asset_key = normalize_asset(asset)
    network_key = "".join(ch for ch in str(network or "").upper() if ch.isalnum())
    if asset_key in {"ETH", "USDC", "USDT"}:
        return network_key in {"ETHEREUM", "ARBITRUM", "OPTIMISM", "BASE", "POLYGON", "AVALANCHE", "BSC"}
    if asset_key == "BTC":
        return network_key == "BITCOIN"
    if asset_key == "SOL":
        return network_key == "SOLANA"
    if asset_key == "XRP":
        return network_key == "XRPLEDGER"
    return bool(network_key)


def vault_asset_networks(asset: str, configured_networks: Iterable[Any] | None = None) -> tuple[str, ...]:
    asset_key = normalize_asset(asset)
    networks = tuple(dict.fromkeys((*DEFAULT_ASSET_NETWORKS.get(asset_key, ("native",)), *(configured_networks or ()))))
    functional = tuple(str(network) for network in networks if functional_wallet_network(asset_key, str(network)))
    return functional or DEFAULT_ASSET_NETWORKS.get(asset_key, ("native",))


def asset_usd_price(asset: str, price_lookup: Callable[[str], float] | None = None) -> float:
    asset_key = normalize_asset(asset)
    if asset_key in STABLE_ASSETS:
        return 1.0
    if price_lookup is None:
        return 0.0
    try:
        price = float(price_lookup(asset_key) or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0
    return price if price > 0 else 0.0


def allocation_asset_views(
    *,
    user_id: int | None = None,
    balances: Iterable[WalletBalance] | None = None,
    configured_assets: Iterable[Any] | None = None,
    configured_networks: Callable[[str], Iterable[Any]] | None = None,
    price_lookup: Callable[[str], float] | None = None,
) -> list[VaultAllocationAssetView]:
    assets = supported_vault_allocation_assets(configured_assets)
    balance_rows = list(balances) if balances is not None else _query_balances(user_id)
    balances_by_asset = {normalize_asset(getattr(row, "asset", "")): row for row in balance_rows}
    configured_set = {normalize_asset(asset) for asset in configured_assets or () if normalize_asset(asset)}

    views: list[VaultAllocationAssetView] = []
    for asset in assets:
        row = balances_by_asset.get(asset)
        available = _safe_float(getattr(row, "available_balance", 0.0) if row is not None else 0.0)
        locked = _safe_float(getattr(row, "locked_balance", 0.0) if row is not None else 0.0)
        total = _safe_float(getattr(row, "total_balance", available + locked) if row is not None else available + locked)
        estimated = _safe_float(getattr(row, "estimated_usd_value", 0.0) if row is not None else 0.0)
        price = asset_usd_price(asset, price_lookup)
        if price <= 0 and total > 0 and estimated > 0:
            price = estimated / max(total, 1e-9)
        available_usd = available * price if price > 0 else 0.0
        networks = vault_asset_networks(asset, configured_networks(asset) if configured_networks is not None else ())
        views.append(
            VaultAllocationAssetView(
                asset=asset,
                label=asset,
                available_balance=available,
                locked_balance=locked,
                total_balance=total,
                estimated_usd_value=estimated,
                price_usd=price,
                available_usd=max(available_usd, 0.0),
                cap_usd=max(available_usd, 0.0),
                networks=networks,
                state=_asset_state(available_usd, price),
                configured=asset in configured_set,
            )
        )
    return views


def default_vault_allocation_asset(assets: Iterable[VaultAllocationAssetView | dict[str, Any]]) -> str:
    rows = list(assets)
    funded = [row for row in rows if _row_float(row, "available_balance") > 0]
    for asset in ("USDC", "USDT"):
        match = next((row for row in funded if _row_asset(row) == asset), None)
        if match is not None:
            return _row_asset(match)
    if funded:
        return _row_asset(funded[0])
    return _row_asset(rows[0]) if rows else "USDC"


def selected_assets_from_values(values: Iterable[Any], supported_assets: Iterable[Any]) -> tuple[str, ...]:
    supported = tuple(normalize_asset(asset) for asset in supported_assets if normalize_asset(asset))
    supported_set = set(supported)
    selected = tuple(dict.fromkeys(normalize_asset(value) for value in values if normalize_asset(value)))
    invalid = [asset for asset in selected if asset not in supported_set]
    if invalid:
        raise ValueError(f"Unsupported Vault allocation asset selected: {', '.join(invalid)}.")
    return selected or (default_vault_allocation_asset([{"asset": asset} for asset in supported]),)


def selected_allocation_cap_usd(assets: Iterable[VaultAllocationAssetView | dict[str, Any]], selected_assets: Iterable[Any]) -> float:
    selected = {normalize_asset(asset) for asset in selected_assets if normalize_asset(asset)}
    return sum(_row_float(row, "available_usd") for row in assets if _row_asset(row) in selected)


def _query_balances(user_id: int | None) -> list[WalletBalance]:
    if user_id is None:
        return []
    return WalletBalance.query.filter_by(user_id=int(user_id)).order_by(WalletBalance.asset.asc()).all()


def _asset_state(available_usd: float, price: float) -> str:
    if available_usd > 0:
        return "ready"
    if price <= 0:
        return "price_unavailable"
    return "empty"


def _row_asset(row: VaultAllocationAssetView | dict[str, Any]) -> str:
    if isinstance(row, VaultAllocationAssetView):
        return row.asset
    return normalize_asset(row.get("asset"))


def _row_float(row: VaultAllocationAssetView | dict[str, Any], key: str) -> float:
    if isinstance(row, VaultAllocationAssetView):
        return _safe_float(getattr(row, key, 0.0))
    return _safe_float(row.get(key))


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed == parsed else fallback
