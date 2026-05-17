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
    price_status: str
    price_source: str
    price_label: str
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
            "price_status": self.price_status,
            "price_source": self.price_source,
            "price_label": self.price_label,
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
    balances_by_asset = _aggregate_balance_rows(balance_rows)
    configured_set = {normalize_asset(asset) for asset in configured_assets or () if normalize_asset(asset)}

    views: list[VaultAllocationAssetView] = []
    for asset in assets:
        row = balances_by_asset.get(asset)
        available = _safe_float(row.get("available_balance") if row is not None else 0.0)
        locked = _safe_float(row.get("locked_balance") if row is not None else 0.0)
        total = _safe_float(row.get("total_balance") if row is not None else available + locked)
        estimated = _safe_float(row.get("estimated_usd_value") if row is not None else 0.0)
        price, price_source, price_status = _asset_price_context(asset, price_lookup, total, estimated)
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
                price_status=price_status,
                price_source=price_source,
                price_label=_price_label(price, price_status),
                networks=networks,
                state=_asset_state(available_usd, price, total),
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


def _aggregate_balance_rows(rows: Iterable[WalletBalance]) -> dict[str, dict[str, float]]:
    aggregated: dict[str, dict[str, float]] = {}
    for row in rows:
        asset = normalize_asset(getattr(row, "asset", ""))
        if not asset:
            continue
        available = _safe_float(getattr(row, "available_balance", 0.0))
        locked = _safe_float(getattr(row, "locked_balance", 0.0))
        total = _safe_float(getattr(row, "total_balance", available + locked), available + locked)
        estimated = _safe_float(getattr(row, "estimated_usd_value", 0.0))
        target = aggregated.setdefault(
            asset,
            {
                "available_balance": 0.0,
                "locked_balance": 0.0,
                "total_balance": 0.0,
                "estimated_usd_value": 0.0,
            },
        )
        target["available_balance"] += max(available, 0.0)
        target["locked_balance"] += max(locked, 0.0)
        target["total_balance"] += max(total, 0.0)
        target["estimated_usd_value"] += max(estimated, 0.0)
    return aggregated


def _asset_price_context(
    asset: str,
    price_lookup: Callable[[str], float] | None,
    total_balance: float,
    estimated_usd_value: float,
) -> tuple[float, str, str]:
    asset_key = normalize_asset(asset)
    if asset_key in STABLE_ASSETS:
        return 1.0, "stable_usd", "priced"
    price = asset_usd_price(asset_key, price_lookup)
    if price > 0:
        return price, "market_data", "priced"
    if total_balance > 0 and estimated_usd_value > 0:
        return estimated_usd_value / max(total_balance, 1e-9), "wallet_estimate", "estimated"
    return 0.0, "unavailable", "unavailable"


def _price_label(price: float, status: str) -> str:
    if status == "unavailable" or price <= 0:
        return "price unavailable"
    if price >= 1:
        return f"${price:,.2f}"
    return f"${price:,.6f}"


def _asset_state(available_usd: float, price: float, total_balance: float = 0.0) -> str:
    if available_usd > 0:
        return "ready"
    if price <= 0 and total_balance > 0:
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
