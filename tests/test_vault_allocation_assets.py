from __future__ import annotations

from types import SimpleNamespace

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


def test_allocation_asset_views_normalize_and_aggregate_duplicate_assets() -> None:
    views = allocation_asset_views(
        balances=[
            SimpleNamespace(asset="eth", available_balance=1.0, locked_balance=0.1, total_balance=1.1, estimated_usd_value=2_200.0),
            SimpleNamespace(asset="ETH", available_balance=0.5, locked_balance=0.2, total_balance=0.7, estimated_usd_value=1_400.0),
            SimpleNamespace(asset="usdt", available_balance=25.0, locked_balance=0.0, total_balance=25.0, estimated_usd_value=25.0),
        ],
        price_lookup=lambda asset: 2_000.0 if asset == "ETH" else 0.0,
    )

    eth = next(row for row in views if row.asset == "ETH")
    usdt = next(row for row in views if row.asset == "USDT")
    assert eth.available_balance == 1.5
    assert eth.locked_balance == 0.30000000000000004
    assert eth.total_balance == 1.8
    assert eth.available_usd == 3_000.0
    assert eth.price_source == "market_data"
    assert eth.price_status == "priced"
    assert usdt.available_usd == 25.0
    assert usdt.price_source == "stable_usd"
    assert selected_allocation_cap_usd(views, ["ETH", "USDT"]) == 3_025.0


def test_allocation_asset_views_use_verified_estimate_or_exclude_unpriced_assets() -> None:
    views = allocation_asset_views(
        balances=[
            SimpleNamespace(asset="BTC", available_balance=0.1, locked_balance=0.0, total_balance=0.1, estimated_usd_value=5_000.0),
            SimpleNamespace(asset="SOL", available_balance=2.0, locked_balance=0.0, total_balance=2.0, estimated_usd_value=0.0),
        ],
        price_lookup=lambda asset: 0.0,
    )

    btc = next(row for row in views if row.asset == "BTC")
    sol = next(row for row in views if row.asset == "SOL")
    assert btc.available_usd == 5_000.0
    assert btc.price_source == "wallet_estimate"
    assert btc.price_status == "estimated"
    assert sol.available_usd == 0.0
    assert sol.cap_usd == 0.0
    assert sol.state == "price_unavailable"
    assert sol.price_label == "price unavailable"
    assert selected_allocation_cap_usd(views, ["BTC", "SOL"]) == 5_000.0


def test_selected_assets_reject_unsupported_values() -> None:
    try:
        selected_assets_from_values(["USDC", "DOGE"], ["USDC", "USDT"])
    except ValueError as exc:
        assert "DOGE" in str(exc)
    else:
        raise AssertionError("unsupported asset was accepted")
