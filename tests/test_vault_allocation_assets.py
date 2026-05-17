from __future__ import annotations

from app.services.vault_allocation_assets import (
    allocation_asset_views,
    selected_allocation_cap_usd,
    selected_assets_from_values,
    supported_vault_allocation_assets,
)


def test_supported_vault_allocation_assets_include_configured_assets() -> None:
    assets = supported_vault_allocation_assets(["arb", "USDC", "OP"])

    assert assets[:6] == ("USDC", "USDT", "BTC", "ETH", "SOL", "XRP")
    assert assets[-2:] == ("ARB", "OP")


def test_allocation_asset_views_compute_caps_and_networks() -> None:
    views = allocation_asset_views(
        configured_assets=["ARB"],
        configured_networks=lambda asset: ("Arbitrum",) if asset == "ARB" else (),
        price_lookup=lambda asset: 2.0 if asset == "ARB" else 1.0,
    )

    arb = next(row for row in views if row.asset == "ARB")
    assert "Arbitrum" in arb.networks
    assert arb.available_usd == 0.0
    assert arb.state == "empty"
    assert selected_allocation_cap_usd(views, ["USDC", "ARB"]) == 0.0


def test_selected_assets_reject_unsupported_values() -> None:
    try:
        selected_assets_from_values(["USDC", "DOGE"], ["USDC", "USDT"])
    except ValueError as exc:
        assert "DOGE" in str(exc)
    else:
        raise AssertionError("unsupported asset was accepted")
