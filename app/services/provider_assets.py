"""Provider-specific collateral and feature helpers."""

from __future__ import annotations

from typing import Any


PROVIDER_COLLATERAL_ASSETS = {
    "hyperliquid": "USDC",
    "kucoin": "USDT",
}


def normalize_provider(provider: Any, *, default: str = "global") -> str:
    value = str(provider or "").strip().lower().replace("-", "_")
    return value or default


def provider_collateral_asset(provider: Any) -> str:
    return PROVIDER_COLLATERAL_ASSETS.get(normalize_provider(provider), "USDC")


def provider_feature_context(provider: Any) -> dict[str, Any]:
    provider_key = normalize_provider(provider)
    collateral = provider_collateral_asset(provider_key)
    return {
        "provider": provider_key,
        "execution_venue": provider_key,
        "collateral_asset": collateral,
        "provider_is_hyperliquid": provider_key == "hyperliquid",
        "provider_is_kucoin": provider_key == "kucoin",
        "collateral_is_usdc": collateral == "USDC",
        "collateral_is_usdt": collateral == "USDT",
    }
