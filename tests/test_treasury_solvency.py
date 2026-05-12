from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from cryptography.fernet import Fernet

from app.auth import password_hash
from app.extensions import db
from app.models import (
    PlatformTreasuryReserveJob,
    TreasuryAlert,
    TreasuryReserveState,
    User,
    VaultCycle,
    VaultCycleSettlement,
    WalletAccount,
    WalletAddress,
    WalletBalance,
    WalletWithdrawal,
)
from app.services.wallet_custody import EvmWalletAdapter


def _configure(app) -> None:
    app.config.update(
        {
            "PLATFORM_GAS_TREASURY_ENABLED": True,
            "TREASURY_ENCRYPTION_KEY": Fernet.generate_key().decode("utf-8"),
            "TREASURY_SOLVENCY_ENABLED": True,
            "TREASURY_SOLVENCY_BASE_SAFETY_MULTIPLIER": 1.0,
            "TREASURY_REBALANCE_TARGET_RATIO": 3.0,
            "TREASURY_SOLVENCY_WITHDRAWAL_MIN_RATIO": 1.10,
            "TREASURY_REBALANCE_SOURCE_ASSET": "USDC",
            "TREASURY_REBALANCE_MIN_ETH": 0.0,
            "TREASURY_REBALANCE_MAX_ETH": 0.0,
            "WALLET_EVM_MIN_GAS_PRICE_GWEI": 1.0,
            "WALLET_EVM_NETWORKS": {"ETHEREUM": {"rpc_url": "https://evm.example.invalid", "chain_id": 1}},
            "WALLET_EVM_TOKEN_CONTRACTS": {
                "ETHEREUM": {
                    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                    "USDT_DECIMALS": 6,
                    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                    "USDC_DECIMALS": 6,
                }
            },
        }
    )


def _user(username: str = "solvency") -> User:
    user = User(username=username, password_hash=password_hash("password123"), role="user")
    db.session.add(user)
    db.session.flush()
    return user


def _wallet(user: User, asset: str = "USDT") -> WalletAddress:
    account = WalletAccount(user_id=user.id, provider="self_custody", asset=asset, network="Ethereum")
    db.session.add(account)
    db.session.flush()
    address = WalletAddress(
        user_id=user.id,
        wallet_account_id=account.id,
        asset=asset,
        network="Ethereum",
        address=f"0x{user.id:040x}",
        status="active",
    )
    db.session.add(address)
    db.session.flush()
    return address


def _cycle(user: User) -> VaultCycle:
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDT",
        deposit_amount=10.0,
        settlement_asset="USDT",
        lock_duration_hours=1,
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=10.0,
        current_estimated_value_usd=10.0,
    )
    db.session.add(cycle)
    db.session.flush()
    return cycle


def _patch_gas(monkeypatch, fee: float = 0.001) -> None:
    monkeypatch.setattr(EvmWalletAdapter, "estimate_fee", lambda self, asset, network, destination, amount: fee)
    monkeypatch.setattr(
        EvmWalletAdapter,
        "_gas_price_payload",
        lambda self, network: {"gas_price_wei": 1_000_000_000, "fee_source": "test"},
    )


def test_treasury_liability_aggregates_balances_withdrawals_and_settlements(app, monkeypatch) -> None:
    _configure(app)
    _patch_gas(monkeypatch)
    user = _user()
    _wallet(user)
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=10.0, estimated_usd_value=10.0))
    db.session.add(
        WalletWithdrawal(
            user_id=user.id,
            asset="USDT",
            network="Ethereum",
            destination_address="0x1111111111111111111111111111111111111111",
            amount=2.0,
            status="pending_submission",
            idempotency_token="manual:solvency:1",
        )
    )
    cycle = _cycle(user)
    db.session.add(VaultCycleSettlement(vault_cycle_id=cycle.id, user_id=user.id, settlement_asset="USDT", status="withdrawing", final_amount=5.0))
    treasury = app.extensions["services"]["platform_treasury"]
    wallet = treasury.create_wallet(created_by_user_id=user.id)
    monkeypatch.setattr(treasury, "eth_balance", lambda address, network="Ethereum": 0.01 if address == wallet.address else 0.0)

    state = app.extensions["services"]["treasury_solvency"].recalculate(network="Ethereum")

    assert state.raw_estimated_liability == pytest.approx(0.003)
    assert state.total_estimated_liability == pytest.approx(0.003)
    assert state.reserve_ratio == pytest.approx(0.01 / 0.003)
    assert state.health_status == "healthy"
    assert state.active_balance_count == 1
    assert state.pending_withdrawal_count == 1
    assert state.active_settlement_count == 1


def test_treasury_rebalance_creates_idempotent_cex_job_and_alert(app, monkeypatch) -> None:
    _configure(app)
    _patch_gas(monkeypatch)
    user = _user("rebalance")
    _wallet(user)
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=10.0, estimated_usd_value=10.0))
    treasury = app.extensions["services"]["platform_treasury"]
    wallet = treasury.create_wallet(created_by_user_id=user.id)
    monkeypatch.setattr(treasury, "eth_balance", lambda address, network="Ethereum": 0.0005 if address == wallet.address else 0.0)

    first = app.extensions["services"]["treasury_solvency"].rebalance_if_needed(network="Ethereum")
    second = app.extensions["services"]["treasury_solvency"].rebalance_if_needed(network="Ethereum")

    job = PlatformTreasuryReserveJob.query.filter_by(job_type="solvency_rebalance").one()
    alert = TreasuryAlert.query.filter_by(event_type="treasury_rebalance_queued").one()
    assert first["created"] is True
    assert second["status"] == "existing"
    assert job.conversion_asset == "USDC"
    assert job.reserve_eth_target == pytest.approx(first["state"]["deficit_eth"])
    assert job.details["conversion_mode"] == "cex"
    assert job.details["dex_interface"]["status"] == "disabled"
    assert alert.severity == "warning"


def test_withdrawal_safety_queues_and_releases_after_reserve_recovery(app, monkeypatch) -> None:
    _configure(app)
    _patch_gas(monkeypatch)
    user = _user("queuesafe")
    source = _wallet(user)
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=10.0, locked_balance=2.0, estimated_usd_value=12.0))
    withdrawal = WalletWithdrawal(
        user_id=user.id,
        source_wallet_address_id=source.id,
        asset="USDT",
        network="Ethereum",
        destination_address="0x1111111111111111111111111111111111111111",
        amount=2.0,
        status="pending_submission",
        idempotency_token="manual:solvency:queue",
    )
    db.session.add(withdrawal)
    treasury = app.extensions["services"]["platform_treasury"]
    wallet = treasury.create_wallet(created_by_user_id=user.id)
    balance = {"eth": 0.0001}
    monkeypatch.setattr(treasury, "eth_balance", lambda address, network="Ethereum": balance["eth"] if address == wallet.address else 0.0)
    solvency = app.extensions["services"]["treasury_solvency"]

    unsafe = solvency.evaluate_withdrawal(withdrawal, projected_spend_eth=0.001, estimated_gas_eth=0.001)

    assert unsafe["safe"] is False
    assert withdrawal.status == "queued_treasury_solvency"
    assert withdrawal.treasury_safety_status == "queued_treasury_solvency"

    balance["eth"] = 1.0
    assert solvency.release_queued_withdrawal_if_safe(withdrawal) is True
    assert withdrawal.status == "pending_submission"
    assert withdrawal.failure_reason is None
