from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

import pyotp
import pytest
from cryptography.fernet import Fernet

from app.auth import encrypt_totp_secret, password_hash
from app.extensions import db
from app.models import (
    PlatformTreasuryEvent,
    PlatformTreasuryReserveJob,
    PlatformTreasuryWallet,
    ReferralInviteCode,
    Setting,
    User,
    WalletAccount,
    WalletAddress,
    WalletAuditLog,
    WalletBalance,
    WalletLedgerEvent,
    WalletTransaction,
    WalletWithdrawal,
)
from app.services import wallet_custody as wallet_custody_module
from app.services.wallet_custody import BroadcastResult, EvmWalletAdapter, GeneratedWallet, RealWalletCustodyService, WalletBalanceSnapshot


def _create_user(username: str = "custody") -> tuple[User, str]:
    user = User(username=username, password_hash=password_hash("password123"), role="user")
    secret = pyotp.random_base32()
    user.totp_secret_encrypted = encrypt_totp_secret(secret)
    user.two_factor_enabled_at = datetime.utcnow()
    db.session.add(user)
    db.session.commit()
    return user, secret


class _FakeAdapter:
    def __init__(self, amount: float = 5.0, confirmations: int = 12, checked: bool = True) -> None:
        self.amount = amount
        self.confirmations = confirmations
        self.checked = checked
        self.broadcasts = 0

    def supports(self, asset: str, network: str) -> bool:
        return asset.upper() in {"ETH", "USDC", "USDT"} and network == "Ethereum"

    def generate_wallet(self, asset: str, network: str) -> GeneratedWallet:
        return GeneratedWallet(
            address="0x1234567890abcdef1234567890abcdef12345678",
            private_key="11" * 32,
            public_key="0x1234567890abcdef1234567890abcdef12345678",
            key_type="secp256k1",
            provider="fake_evm",
        )

    def get_balance(self, address: str, asset: str, network: str) -> WalletBalanceSnapshot:
        return WalletBalanceSnapshot(
            amount=self.amount,
            asset=asset,
            checked=self.checked,
            confirmations=self.confirmations,
            provider_reference=f"fake:{address}:{asset}:{self.amount}",
        )

    def estimate_fee(self, asset: str, network: str, destination: str, amount: float) -> float:
        return 0.001

    def sign_and_broadcast(self, withdrawal: WalletWithdrawal, private_key: str) -> BroadcastResult:
        assert private_key == "11" * 32
        self.broadcasts += 1
        return BroadcastResult("submitted", "0xtxhash", {"ok": True})

    def confirm_transaction(self, provider_reference: str, asset: str, network: str) -> dict[str, Any]:
        return {"confirmed": True}


class _ConfirmingAdapter(_FakeAdapter):
    def __init__(self, receipt: dict[str, Any]) -> None:
        super().__init__()
        self.receipt = receipt

    def confirm_transaction(self, provider_reference: str, asset: str, network: str) -> dict[str, Any]:
        return {"confirmed": self.receipt.get("status") == "0x1", "raw": self.receipt}


class _FakeTreasuryConversionConnector:
    def __init__(self) -> None:
        self.conversions: list[dict[str, Any]] = []
        self.withdrawals: list[dict[str, Any]] = []

    def convert_stablecoin(
        self,
        mode: str,
        from_asset: str,
        to_asset: str,
        amount: float,
        max_slippage_bps: float,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        self.conversions.append({"from_asset": from_asset, "to_asset": to_asset, "amount": amount, "client_reference": client_reference})
        return {
            "status": "confirmed",
            "provider_reference": f"convert-{client_reference}",
            "confirmed_amount": amount / 3000.0,
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
        self.withdrawals.append({"asset": asset, "amount": amount, "destination": destination, "client_reference": client_reference})
        return {
            "status": "confirmed",
            "provider_reference": f"withdraw-{client_reference}",
            "confirmed_amount": amount,
        }


def _enable_generated_wallets(app) -> None:
    app.config["USE_REAL_ADDRESSES"] = True
    app.config["WALLET_REAL_CUSTODY_ENABLED"] = True
    app.config["WALLET_ALLOW_IN_APP_KEYGEN"] = True
    app.config["TOTP_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    app.config["WALLET_EVM_RPC_URL"] = "https://evm.example.invalid"
    app.config["WALLET_EVM_TOKEN_CONTRACTS"] = {
        "ETHEREUM": {
            "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
            "USDC_DECIMALS": 6,
            "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
            "USDT_DECIMALS": 6,
        }
    }
    app.config["WALLET_BTC_INDEXER_URL"] = "https://btc.example.invalid"
    app.config["WALLET_SOLANA_RPC_URL"] = "https://sol.example.invalid"
    app.config["WALLET_XRP_RPC_URL"] = "https://xrp.example.invalid"
    Setting.set_json("use_real_addresses", True)


def test_real_wallet_generates_mainnet_style_addresses_without_test_prefix(app) -> None:
    _enable_generated_wallets(app)
    custody = app.extensions["services"]["wallet_custody"]
    user, _ = _create_user()

    eth = custody.get_or_create_address(user_id=user.id, asset="ETH", network="Ethereum")
    usdc = custody.get_or_create_address(user_id=user.id, asset="USDC", network="Ethereum")
    usdt = custody.get_or_create_address(user_id=user.id, asset="USDT", network="Ethereum")
    btc = custody.get_or_create_address(user_id=user.id, asset="BTC", network="Bitcoin")
    sol = custody.get_or_create_address(user_id=user.id, asset="SOL", network="Solana")
    xrp = custody.get_or_create_address(user_id=user.id, asset="XRP", network="XRP Ledger")

    assert re.fullmatch(r"0x[a-fA-F0-9]{40}", eth.address)
    assert re.fullmatch(r"0x[a-fA-F0-9]{40}", usdc.address)
    assert re.fullmatch(r"0x[a-fA-F0-9]{40}", usdt.address)
    assert re.fullmatch(r"1[1-9A-HJ-NP-Za-km-z]{25,34}", btc.address)
    assert re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", sol.address)
    assert re.fullmatch(r"r[1-9A-HJ-NP-Za-km-z]{24,34}", xrp.address)
    assert all(not item.address.startswith("TEST-") for item in (eth, usdc, usdt, btc, sol, xrp))
    private_key = custody._private_key(eth)
    assert private_key
    assert private_key not in eth.encrypted_metadata_json


def test_force_new_real_wallet_address_generates_replacement(app) -> None:
    _enable_generated_wallets(app)
    custody = app.extensions["services"]["wallet_custody"]
    user, _ = _create_user("rotatebtc")

    first = custody.get_or_create_address(user_id=user.id, asset="BTC", network="Bitcoin")
    second = custody.get_or_create_address(user_id=user.id, asset="BTC", network="Bitcoin", force_new=True)

    assert second.id != first.id
    assert second.address != first.address
    assert second.rotation_index == first.rotation_index + 1
    assert first.status == "active"
    assert second.status == "active"


def test_real_wallet_generation_fails_closed_when_keygen_disabled(app) -> None:
    _enable_generated_wallets(app)
    app.config["WALLET_ALLOW_IN_APP_KEYGEN"] = False
    custody = app.extensions["services"]["wallet_custody"]
    user, _ = _create_user("nokeygen")

    try:
        custody.get_or_create_address(user_id=user.id, asset="ETH", network="Ethereum")
    except RuntimeError as exc:
        assert "key generation is disabled" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("wallet generation should fail closed")


def test_deposit_sync_credits_once_with_idempotent_ledger(app) -> None:
    _enable_generated_wallets(app)
    fake = _FakeAdapter(amount=5.0)
    custody = RealWalletCustodyService(app.config, adapters=[fake])
    app.extensions["services"]["wallet_custody"] = custody
    user, _ = _create_user("sync")
    wallet_address = custody.get_or_create_address(user_id=user.id, asset="ETH", network="Ethereum")

    first = custody.sync_address(wallet_address)
    second = custody.sync_address(wallet_address)

    balance = WalletBalance.query.filter_by(user_id=user.id, asset="ETH").one()
    assert first["credited"] == 5.0
    assert second["credited"] == 0.0
    assert balance.available_balance == 5.0
    assert WalletLedgerEvent.query.count() == 1
    assert WalletTransaction.query.filter_by(transaction_type="deposit").count() == 1


def test_deposit_sync_deducts_treasury_gas_reserve_and_queues_job(app, monkeypatch) -> None:
    _enable_generated_wallets(app)
    app.config["PLATFORM_GAS_TREASURY_ENABLED"] = True
    app.config["TREASURY_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    app.config["PLATFORM_TREASURY_ETH_USD_FALLBACK"] = 3000.0
    fake = _FakeAdapter(amount=10.0)
    custody = RealWalletCustodyService(app.config, adapters=[fake])
    app.extensions["services"]["wallet_custody"] = custody
    user, _ = _create_user("reservedepsync")
    wallet_address = custody.get_or_create_address(user_id=user.id, asset="USDT", network="Ethereum")
    treasury = app.extensions["services"]["platform_treasury"]
    treasury.create_wallet(created_by_user_id=user.id)
    monkeypatch.setattr(EvmWalletAdapter, "estimate_fee", lambda self, asset, network, destination, amount: 0.001)
    monkeypatch.setattr(treasury, "_asset_usd_price", lambda asset: 3000.0 if asset == "ETH" else 1.0)

    result = custody.sync_address(wallet_address)

    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDT").one()
    job = PlatformTreasuryReserveJob.query.filter_by(user_id=user.id, job_type="deposit_gas_reserve").one()
    event = PlatformTreasuryEvent.query.filter_by(event_type="deposit_gas_reserve_deducted").one()
    assert result["credited"] == 10.0
    assert balance.available_balance == pytest.approx(4.0)
    assert job.conversion_amount == pytest.approx(6.0)
    assert job.reserve_eth_target == pytest.approx(0.002)
    assert job.status == "pending"
    assert event.gas_reserve_contribution == pytest.approx(0.002)


def test_process_reserve_jobs_converts_queued_deposit_reserve(app, monkeypatch) -> None:
    _enable_generated_wallets(app)
    app.config["PLATFORM_GAS_TREASURY_ENABLED"] = True
    app.config["TREASURY_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    fake = _FakeAdapter(amount=10.0)
    custody = RealWalletCustodyService(app.config, adapters=[fake])
    app.extensions["services"]["wallet_custody"] = custody
    user, _ = _create_user("reservebatch")
    wallet_address = custody.get_or_create_address(user_id=user.id, asset="USDT", network="Ethereum")
    treasury = app.extensions["services"]["platform_treasury"]
    wallet = treasury.create_wallet(created_by_user_id=user.id)
    conversion = _FakeTreasuryConversionConnector()
    monkeypatch.setattr(treasury, "_conversion_connector", lambda: conversion)
    monkeypatch.setattr(EvmWalletAdapter, "estimate_fee", lambda self, asset, network, destination, amount: 0.001)
    monkeypatch.setattr(treasury, "_asset_usd_price", lambda asset: 3000.0 if asset == "ETH" else 1.0)

    custody.sync_address(wallet_address)
    jobs = treasury.process_reserve_jobs()

    job = PlatformTreasuryReserveJob.query.filter_by(user_id=user.id, job_type="deposit_gas_reserve").one()
    event = PlatformTreasuryEvent.query.filter_by(platform_treasury_job_id=job.id, event_type="deposit_gas_reserve_converted_to_eth").one()
    assert len(jobs) == 1
    assert job.status == "complete"
    assert job.converted_eth_amount == pytest.approx(0.002)
    assert event.destination_address == wallet.address
    assert len(conversion.conversions) == 1
    assert len(conversion.withdrawals) == 1


def test_process_reserve_jobs_marks_missing_conversion_connector_retryable(app, monkeypatch) -> None:
    _enable_generated_wallets(app)
    app.config["PLATFORM_GAS_TREASURY_ENABLED"] = True
    app.config["TREASURY_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    fake = _FakeAdapter(amount=10.0)
    custody = RealWalletCustodyService(app.config, adapters=[fake])
    app.extensions["services"]["wallet_custody"] = custody
    user, _ = _create_user("reserveretry")
    wallet_address = custody.get_or_create_address(user_id=user.id, asset="USDT", network="Ethereum")
    treasury = app.extensions["services"]["platform_treasury"]
    treasury.create_wallet(created_by_user_id=user.id)
    monkeypatch.setattr(EvmWalletAdapter, "estimate_fee", lambda self, asset, network, destination, amount: 0.001)
    monkeypatch.setattr(treasury, "_asset_usd_price", lambda asset: 3000.0 if asset == "ETH" else 1.0)

    custody.sync_address(wallet_address)
    treasury.process_reserve_jobs()

    job = PlatformTreasuryReserveJob.query.filter_by(user_id=user.id, job_type="deposit_gas_reserve").one()
    assert job.status == "retryable"
    assert "CONVERSION_USER_ID" in job.failure_reason


def test_reconcile_custody_balance_restores_completed_deposits_idempotently(app) -> None:
    _enable_generated_wallets(app)
    custody = app.extensions["services"]["wallet_custody"]
    user, _ = _create_user("reconcileusdt")
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=0.0000000065, locked_balance=2.0))
    db.session.add(
        WalletLedgerEvent(
            user_id=user.id,
            asset="USDT",
            network="Ethereum",
            address="0x" + ("b" * 40),
            event_type="deposit",
            provider_reference="recovered-usdt",
            idempotency_key="deposit:recovered-usdt",
            amount=10.0,
            confirmations=12,
            status="complete",
        )
    )
    db.session.commit()

    first = custody.reconcile_custody_balance(user.id, "USDT")
    db.session.commit()
    second = custody.reconcile_custody_balance(user.id, "USDT")
    db.session.commit()

    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDT").one()
    assert first["changed"] is True
    assert first["available_balance"] == 8.0
    assert second["changed"] is False
    assert balance.available_balance == 8.0
    assert balance.locked_balance == 2.0
    assert WalletAuditLog.query.filter_by(user_id=user.id, action="wallet_balance_reconciled").count() == 2


def test_evm_balance_uses_lowercase_required_confirmation_key(monkeypatch) -> None:
    config = {
        "WALLET_EVM_NETWORKS": {"ETHEREUM": {"rpc_url": "https://evm.example.invalid", "chain_id": 1}},
        "WALLET_EVM_TOKEN_CONTRACTS": {
            "ETHEREUM": {
                "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                "USDT_DECIMALS": 6,
            }
        },
        "WALLET_REQUIRED_CONFIRMATIONS": {"ethereum": 12.0},
    }
    calls: list[str] = []

    def fake_rpc(url: str, method: str, params: list[Any]) -> str:
        calls.append(method)
        if method == "eth_call":
            return hex(10 * 10**6)
        if method == "eth_blockNumber":
            return "0x123"
        raise AssertionError(f"unexpected RPC method {method}")

    monkeypatch.setattr(wallet_custody_module, "_json_rpc", fake_rpc)

    snapshot = EvmWalletAdapter(config).get_balance(
        "0xBff167B7407f4Bfa125D0b03325B8cCb4a885051",
        "USDT",
        "Ethereum",
    )

    assert snapshot.checked is True
    assert snapshot.amount == 10.0
    assert snapshot.confirmations == 12
    assert calls == ["eth_call", "eth_blockNumber"]


def test_json_rpc_sends_wallet_sync_user_agent(monkeypatch) -> None:
    captured_headers: dict[str, str] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return b'{"jsonrpc":"2.0","id":1,"result":"0x1"}'

    def fake_urlopen(request, timeout: float):
        captured_headers.update(dict(request.header_items()))
        assert timeout == 5.0
        return _Response()

    monkeypatch.setattr(wallet_custody_module.urllib.request, "urlopen", fake_urlopen)

    result = wallet_custody_module._json_rpc("https://evm.example.invalid", "eth_blockNumber", [])

    assert result == "0x1"
    assert captured_headers["Content-type"] == "application/json"
    assert captured_headers["User-agent"] == "TradingBotWalletSync/1.0"


def test_evm_fee_estimation_uses_estimate_gas_and_fee_history(monkeypatch) -> None:
    config = {
        "WALLET_EVM_NETWORKS": {"ETHEREUM": {"rpc_url": "https://evm.example.invalid", "chain_id": 1}},
        "WALLET_EVM_TOKEN_CONTRACTS": {
            "ETHEREUM": {
                "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                "USDT_DECIMALS": 6,
            }
        },
        "WALLET_EVM_MIN_GAS_PRICE_GWEI": 1.0,
        "WALLET_EVM_GAS_LIMIT_BUFFER_MULTIPLIER": 1.2,
    }
    adapter = EvmWalletAdapter(config)
    calls: list[str] = []

    def fake_rpc(method: str, params: list[Any], *, network: str) -> Any:
        calls.append(method)
        if method == "eth_estimateGas":
            assert params[0]["to"] == "0xdAC17F958D2ee523a2206206994597C13D831ec7"
            return hex(100_000)
        if method == "eth_feeHistory":
            return {
                "baseFeePerGas": [hex(1_000_000_000), hex(1_000_000_000)],
                "reward": [[hex(500_000_000)]],
            }
        raise AssertionError(f"unexpected RPC method {method}")

    monkeypatch.setattr(adapter, "_rpc", fake_rpc)

    fee = adapter.estimate_fee("USDT", "Ethereum", "0x1111111111111111111111111111111111111111", 1.0)

    assert fee == pytest.approx(0.00018)
    assert calls == ["eth_estimateGas", "eth_feeHistory"]


def _recovery_custody(app, *, amount: float = 5.0) -> RealWalletCustodyService:
    _enable_generated_wallets(app)
    fake = _FakeAdapter(amount=amount)
    custody = RealWalletCustodyService(app.config, adapters=[fake])
    app.extensions["services"]["wallet_custody"] = custody
    return custody


def test_recover_evm_token_deposit_preview_does_not_create_wallet_or_credit(app) -> None:
    custody = _recovery_custody(app)
    user, _ = _create_user("previewrecover")
    source = custody.get_or_create_address(user_id=user.id, asset="ETH", network="Ethereum")

    result = custody.recover_evm_token_deposit(
        user_id=user.id,
        asset="USDT",
        address=source.address.lower(),
        tx_hash="0x" + ("a" * 64),
        confirm=False,
    )

    assert result["preview_only"] is True
    assert result["ready"] is True
    assert result["recovered"] is False
    assert WalletAddress.query.filter_by(user_id=user.id, asset="USDT").count() == 0
    assert WalletBalance.query.filter_by(user_id=user.id, asset="USDT").one_or_none() is None
    assert WalletLedgerEvent.query.count() == 0
    assert WalletAuditLog.query.filter_by(action="recover_evm_token_deposit_preview", status="ready").count() == 1


def test_recover_evm_token_deposit_links_existing_evm_key_and_credits_once(app) -> None:
    custody = _recovery_custody(app, amount=7.25)
    user, _ = _create_user("confirmrecover")
    source = custody.get_or_create_address(user_id=user.id, asset="ETH", network="Ethereum")
    tx_hash = "0x" + ("b" * 64)

    first = custody.recover_evm_token_deposit(
        user_id=user.id,
        asset="USDT",
        address=source.address,
        tx_hash=tx_hash,
        confirm=True,
    )
    second = custody.recover_evm_token_deposit(
        user_id=user.id,
        asset="USDT",
        address=source.address,
        tx_hash=tx_hash,
        confirm=True,
    )

    recovered = WalletAddress.query.filter_by(user_id=user.id, asset="USDT", network="Ethereum", address=source.address).one()
    assert first["recovered"] is True
    assert first["created_wallet_address"] is True
    assert first["credited"] == 7.25
    assert second["recovered"] is True
    assert second["created_wallet_address"] is False
    assert second["credited"] == 0.0
    assert recovered.deposit_address_id is None
    assert recovered.rotated_from_id == source.id
    assert recovered.encrypted_metadata["encrypted_private_key"] == source.encrypted_metadata["encrypted_private_key"]
    assert recovered.encrypted_metadata["recovery"]["tx_hash"] == tx_hash
    assert WalletBalance.query.filter_by(user_id=user.id, asset="USDT").one().available_balance == 7.25
    assert WalletLedgerEvent.query.filter_by(asset="USDT", address=source.address).count() == 1
    assert WalletTransaction.query.filter_by(user_id=user.id, asset="USDT", transaction_type="deposit").count() == 1


def test_recover_evm_token_deposit_cli_requires_exact_confirmation(app) -> None:
    custody = _recovery_custody(app, amount=3.0)
    user, _ = _create_user("clirecover")
    source = custody.get_or_create_address(user_id=user.id, asset="ETH", network="Ethereum")
    tx_hash = "0x" + ("c" * 64)

    preview = app.test_cli_runner().invoke(
        args=[
            "recover-evm-token-deposit",
            "--user-id",
            str(user.id),
            "--asset",
            "USDT",
            "--address",
            source.address,
            "--tx-hash",
            tx_hash,
        ]
    )
    assert preview.exit_code == 0
    assert '"preview_only": true' in preview.output
    assert WalletAddress.query.filter_by(user_id=user.id, asset="USDT").count() == 0

    confirmed = app.test_cli_runner().invoke(
        args=[
            "recover-evm-token-deposit",
            "--user-id",
            str(user.id),
            "--asset",
            "USDT",
            "--address",
            source.address,
            "--tx-hash",
            tx_hash,
            "--confirm",
            "RECOVER-EVM-TOKEN",
        ]
    )

    assert confirmed.exit_code == 0
    assert '"preview_only": false' in confirmed.output
    assert '"credited": 3.0' in confirmed.output
    assert WalletAddress.query.filter_by(user_id=user.id, asset="USDT").count() == 1


@pytest.mark.parametrize(
    ("case", "mutate", "expected_blocker"),
    [
        ("wrong_user", lambda custody, user, source: _create_user("otherrecover")[0].id, "Address is not an existing generated in-app EVM wallet for this user"),
        ("unsupported_token", lambda custody, user, source: "asset:ETH", "Only supported ERC-20 recovery assets are allowed"),
        ("missing_contract", lambda custody, user, source: custody.config.__setitem__("WALLET_EVM_TOKEN_CONTRACTS", {}), "USDT token contract is not configured"),
        ("missing_key", lambda custody, user, source: _remove_wallet_private_key(source), "Source wallet private key is unavailable or cannot be decrypted"),
        ("non_evm_address", lambda custody, user, source: "address:bc1notanevmaddress", "Recovery address must be a valid EVM 0x address"),
        ("active_duplicate", lambda custody, user, source: _create_duplicate_target_wallet(custody, user, source), "Active USDT/Ethereum wallet address already exists for this address"),
    ],
)
def test_recover_evm_token_deposit_fails_closed_for_invalid_inputs(app, case, mutate, expected_blocker) -> None:
    custody = _recovery_custody(app)
    user, _ = _create_user(f"fail{case}")
    source = custody.get_or_create_address(user_id=user.id, asset="ETH", network="Ethereum")
    user_id = user.id
    asset = "USDT"
    address = source.address
    result = mutate(custody, user, source)
    if isinstance(result, int):
        user_id = result
    elif isinstance(result, str) and result.startswith("asset:"):
        asset = result.split(":", 1)[1]
    elif isinstance(result, str) and result.startswith("address:"):
        address = result.split(":", 1)[1]

    payload = custody.recover_evm_token_deposit(
        user_id=user_id,
        asset=asset,
        address=address,
        tx_hash="0x" + ("d" * 64),
        confirm=True,
    )

    assert payload["ready"] is False
    assert any(expected_blocker in blocker for blocker in payload["blockers"])
    assert payload["recovered"] is False
    assert WalletLedgerEvent.query.filter_by(asset="USDT").count() == 0


def _remove_wallet_private_key(wallet_address: WalletAddress) -> None:
    metadata = wallet_address.encrypted_metadata
    metadata.pop("encrypted_private_key", None)
    wallet_address.encrypted_metadata = metadata
    db.session.flush()


def _create_duplicate_target_wallet(custody: RealWalletCustodyService, user: User, source: WalletAddress) -> None:
    account = custody._account_for(user.id, "USDT", "Ethereum")
    duplicate = WalletAddress(
        wallet_account_id=account.id,
        user_id=user.id,
        asset="USDT",
        network="Ethereum",
        address=source.address,
        status="active",
    )
    duplicate.encrypted_metadata = {
        "custody": "external",
        "encrypted_private_key": source.encrypted_metadata["encrypted_private_key"],
    }
    db.session.add(duplicate)
    db.session.flush()


def test_real_withdrawal_requires_live_mode_and_broadcasts_when_live(app) -> None:
    _enable_generated_wallets(app)
    fake = _FakeAdapter(amount=5.0)
    custody = RealWalletCustodyService(app.config, adapters=[fake])
    app.extensions["services"]["wallet_custody"] = custody
    service = app.extensions["services"]["self_custody_wallet"]
    user, _ = _create_user("withdraw")
    wallet_address = custody.get_or_create_address(user_id=user.id, asset="ETH", network="Ethereum")
    db.session.add(WalletBalance(user_id=user.id, asset="ETH", available_balance=2.0))
    withdrawal = service.create_manual_withdrawal(
        user_id=user.id,
        asset="ETH",
        network="Ethereum",
        destination_address="0x1111111111111111111111111111111111111111",
        amount=1.0,
    )
    withdrawal.source_wallet_address_id = wallet_address.id
    withdrawal.status = "pending_submission"
    db.session.flush()

    failed = service.submit_withdrawal(withdrawal, mode="paper")
    assert failed.status == "failed"
    assert "live mode" in failed.failure_reason

    withdrawal.status = "pending_submission"
    withdrawal.failure_reason = None
    submitted = service.submit_withdrawal(withdrawal, mode="live")
    assert submitted.status == "submitted"
    assert submitted.provider_reference == "0xtxhash"
    assert fake.broadcasts == 1


def test_evm_token_withdrawal_fails_before_broadcast_without_eth_for_gas(app, monkeypatch) -> None:
    _enable_generated_wallets(app)
    adapter = EvmWalletAdapter(app.config)
    calls: list[tuple[str, list[Any]]] = []

    def fake_rpc(method: str, params: list[Any], *, network: str) -> Any:
        calls.append((method, params))
        if method == "eth_getTransactionCount":
            return "0x0"
        if method == "eth_gasPrice":
            return hex(20_000_000_000)
        if method == "eth_getBalance":
            return "0x0"
        if method == "eth_sendRawTransaction":
            raise AssertionError("broadcast should not be attempted without gas")
        return "0x0"

    monkeypatch.setattr(adapter, "_rpc", fake_rpc)
    user, _ = _create_user("nogas")
    wallet_address = app.extensions["services"]["wallet_custody"].get_or_create_address(
        user_id=user.id,
        asset="USDT",
        network="Ethereum",
    )
    withdrawal = WalletWithdrawal(
        user_id=user.id,
        source_wallet_address_id=wallet_address.id,
        asset="USDT",
        network="Ethereum",
        destination_address="0x1111111111111111111111111111111111111111",
        amount=1.0,
    )

    with pytest.raises(RuntimeError, match="insufficient ETH"):
        adapter.sign_and_broadcast(withdrawal, "11" * 32)

    assert [call[0] for call in calls] == [
        "eth_getTransactionCount",
        "eth_feeHistory",
        "eth_gasPrice",
        "eth_getBalance",
        "eth_estimateGas",
    ]


def test_evm_broadcast_hash_missing_on_chain_returns_failed_status(app, monkeypatch) -> None:
    _enable_generated_wallets(app)
    adapter = EvmWalletAdapter(app.config)

    def fake_rpc(method: str, params: list[Any], *, network: str) -> Any:
        if method == "eth_getTransactionCount":
            return "0x0"
        if method == "eth_gasPrice":
            return hex(1_000_000_000)
        if method == "eth_getBalance":
            return hex(10**18)
        if method == "eth_call":
            return hex(2 * 10**6)
        if method == "eth_sendRawTransaction":
            return "0xmissing"
        if method == "eth_getTransactionByHash":
            return None
        return "0x0"

    monkeypatch.setattr(adapter, "_rpc", fake_rpc)
    app.config["WALLET_BROADCAST_VERIFY_ATTEMPTS"] = 1
    user, _ = _create_user("missingtx")
    wallet_address = app.extensions["services"]["wallet_custody"].get_or_create_address(
        user_id=user.id,
        asset="USDT",
        network="Ethereum",
    )
    withdrawal = WalletWithdrawal(
        user_id=user.id,
        source_wallet_address_id=wallet_address.id,
        asset="USDT",
        network="Ethereum",
        destination_address="0x1111111111111111111111111111111111111111",
        amount=1.0,
    )

    result = adapter.sign_and_broadcast(withdrawal, "11" * 32)

    assert result.status == "failed_broadcast_not_found"
    assert result.provider_reference == "0xmissing"
    assert result.raw["broadcast_visible"] is False
    assert result.raw["token_contract"] == "0xdAC17F958D2ee523a2206206994597C13D831ec7"
    assert result.raw["amount_units"] == 1_000_000


def test_evm_token_withdrawal_applies_minimum_gas_price_floor(app, monkeypatch) -> None:
    _enable_generated_wallets(app)
    app.config["WALLET_EVM_MIN_GAS_PRICE_GWEI"] = 2.0
    app.config["WALLET_BROADCAST_VERIFY_ATTEMPTS"] = 1
    adapter = EvmWalletAdapter(app.config)

    def fake_rpc(method: str, params: list[Any], *, network: str) -> Any:
        if method == "eth_getTransactionCount":
            return "0x0"
        if method == "eth_gasPrice":
            return hex(136_742_204)
        if method == "eth_getBalance":
            return hex(10**18)
        if method == "eth_call":
            return hex(2 * 10**6)
        if method == "eth_sendRawTransaction":
            return "0xsent"
        if method == "eth_getTransactionByHash":
            return {"hash": params[0]}
        return "0x0"

    monkeypatch.setattr(adapter, "_rpc", fake_rpc)
    user, _ = _create_user("gasfloor")
    wallet_address = app.extensions["services"]["wallet_custody"].get_or_create_address(
        user_id=user.id,
        asset="USDT",
        network="Ethereum",
    )
    withdrawal = WalletWithdrawal(
        user_id=user.id,
        source_wallet_address_id=wallet_address.id,
        asset="USDT",
        network="Ethereum",
        destination_address="0x1111111111111111111111111111111111111111",
        amount=1.0,
    )

    result = adapter.sign_and_broadcast(withdrawal, "11" * 32)

    assert result.status == "submitted"
    assert result.raw["rpc_gas_price_wei"] == 136_742_204
    assert result.raw["gas_price_wei"] == 2_000_000_000


def test_evm_token_withdrawal_refuses_broadcast_without_token_balance(app, monkeypatch) -> None:
    _enable_generated_wallets(app)
    app.config["WALLET_BROADCAST_VERIFY_ATTEMPTS"] = 1
    adapter = EvmWalletAdapter(app.config)

    def fake_rpc(method: str, params: list[Any], *, network: str) -> Any:
        if method == "eth_getTransactionCount":
            return "0x0"
        if method == "eth_gasPrice":
            return hex(2_000_000_000)
        if method == "eth_getBalance":
            return hex(10**18)
        if method == "eth_call":
            return "0x0"
        if method == "eth_sendRawTransaction":
            raise AssertionError("broadcast should not be attempted without token balance")
        return "0x0"

    monkeypatch.setattr(adapter, "_rpc", fake_rpc)
    user, _ = _create_user("notokens")
    wallet_address = app.extensions["services"]["wallet_custody"].get_or_create_address(
        user_id=user.id,
        asset="USDT",
        network="Ethereum",
    )
    withdrawal = WalletWithdrawal(
        user_id=user.id,
        source_wallet_address_id=wallet_address.id,
        asset="USDT",
        network="Ethereum",
        destination_address="0x1111111111111111111111111111111111111111",
        amount=1.0,
    )

    with pytest.raises(RuntimeError, match="insufficient USDT token balance"):
        adapter.sign_and_broadcast(withdrawal, "11" * 32)


def test_withdrawal_preflight_prefers_source_with_token_and_gas(app) -> None:
    _enable_generated_wallets(app)
    user, _ = _create_user("sourcepref")
    account = WalletAccount(user_id=user.id, provider="self_custody", asset="USDT", network="Ethereum")
    db.session.add(account)
    db.session.flush()
    no_gas = WalletAddress(
        wallet_account_id=account.id,
        user_id=user.id,
        asset="USDT",
        network="Ethereum",
        address="0x1111111111111111111111111111111111111111",
        status="active",
        rotation_index=2,
    )
    ready = WalletAddress(
        wallet_account_id=account.id,
        user_id=user.id,
        asset="USDT",
        network="Ethereum",
        address="0x2222222222222222222222222222222222222222",
        status="active",
        rotation_index=1,
    )
    db.session.add_all([no_gas, ready])
    db.session.flush()

    class _AddressAwareAdapter(_FakeAdapter):
        def get_balance(self, address: str, asset: str, network: str) -> WalletBalanceSnapshot:
            if asset.upper() == "ETH":
                amount = 0.0 if address == no_gas.address else 0.01
            else:
                amount = 5.0
            return WalletBalanceSnapshot(amount=amount, asset=asset, checked=True, confirmations=12)

    custody = RealWalletCustodyService(app.config, adapters=[_AddressAwareAdapter()])
    withdrawal = WalletWithdrawal(
        user_id=user.id,
        asset="USDT",
        network="Ethereum",
        destination_address="0x3333333333333333333333333333333333333333",
        amount=1.0,
    )

    result = custody.withdrawal_preflight(withdrawal)

    assert result["ready"] is True
    assert withdrawal.source_wallet_address_id == ready.id


def test_reconcile_failed_withdrawal_releases_locked_balance_once(app) -> None:
    _enable_generated_wallets(app)
    user, _ = _create_user("reconcilefailed")
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=1.0, locked_balance=0.99))
    withdrawal = WalletWithdrawal(
        user_id=user.id,
        asset="USDT",
        network="Ethereum",
        destination_address="0x1111111111111111111111111111111111111111",
        amount=0.99,
        status="submitted",
        provider_reference="0xfailed",
        idempotency_token="manual:failed",
    )
    db.session.add(withdrawal)
    db.session.flush()
    db.session.add(
        WalletTransaction(
            user_id=user.id,
            asset="USDT",
            amount=0.99,
            transaction_type="withdrawal",
            status="pending_withdrawal",
            network="Ethereum",
            note=f"Withdrawal workflow {withdrawal.id}: submitted.",
        )
    )
    db.session.flush()
    custody = RealWalletCustodyService(app.config, adapters=[_ConfirmingAdapter({"status": "0x0", "logs": []})])

    first = custody.reconcile_withdrawal(withdrawal, commit=True)
    second = custody.reconcile_withdrawal(withdrawal, commit=True)

    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDT").one()
    tx = WalletTransaction.query.filter_by(user_id=user.id, transaction_type="withdrawal").one()
    assert first["status"] == "failed"
    assert second["terminal"] is True
    assert withdrawal.status == "failed"
    assert tx.status == "failed"
    assert balance.available_balance == pytest.approx(1.99)
    assert balance.locked_balance == 0.0


def test_reconcile_successful_erc20_withdrawal_clears_lock(app) -> None:
    _enable_generated_wallets(app)
    user, _ = _create_user("reconcilesuccess")
    source = app.extensions["services"]["wallet_custody"].get_or_create_address(
        user_id=user.id,
        asset="USDT",
        network="Ethereum",
    )
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=1.0, locked_balance=1.0))
    withdrawal = WalletWithdrawal(
        user_id=user.id,
        source_wallet_address_id=source.id,
        asset="USDT",
        network="Ethereum",
        destination_address="0x1111111111111111111111111111111111111111",
        amount=1.0,
        status="submitted",
        provider_reference="0xcomplete",
        idempotency_token="manual:complete",
    )
    db.session.add(withdrawal)
    db.session.flush()
    db.session.add(
        WalletTransaction(
            user_id=user.id,
            asset="USDT",
            amount=1.0,
            transaction_type="withdrawal",
            status="pending_withdrawal",
            network="Ethereum",
            note=f"Withdrawal workflow {withdrawal.id}: submitted.",
        )
    )
    source_topic = "0x" + source.address.lower().replace("0x", "").rjust(64, "0")
    destination_topic = "0x" + withdrawal.destination_address.lower().replace("0x", "").rjust(64, "0")
    receipt = {
        "status": "0x1",
        "logs": [
            {
                "address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                "topics": [
                    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef",
                    source_topic,
                    destination_topic,
                ],
                "data": hex(1_000_000),
            }
        ],
    }
    custody = RealWalletCustodyService(app.config, adapters=[_ConfirmingAdapter(receipt)])

    result = custody.reconcile_withdrawal(withdrawal, commit=True)

    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDT").one()
    tx = WalletTransaction.query.filter_by(user_id=user.id, transaction_type="withdrawal").one()
    assert result["status"] == "complete"
    assert withdrawal.status == "complete"
    assert tx.status == "complete"
    assert balance.available_balance == 1.0
    assert balance.locked_balance == 0.0


def test_platform_treasury_create_and_topup_keeps_private_key_encrypted(app, monkeypatch) -> None:
    _enable_generated_wallets(app)
    app.config["PLATFORM_GAS_TREASURY_ENABLED"] = True
    app.config["TREASURY_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    user, _ = _create_user("treasurytopup")
    source = app.extensions["services"]["wallet_custody"].get_or_create_address(
        user_id=user.id,
        asset="USDT",
        network="Ethereum",
    )
    withdrawal = WalletWithdrawal(
        user_id=user.id,
        source_wallet_address_id=source.id,
        asset="USDT",
        network="Ethereum",
        destination_address="0x1111111111111111111111111111111111111111",
        amount=1.0,
        status="pending_gas_topup",
        idempotency_token="manual:topup",
    )
    db.session.add(withdrawal)
    db.session.flush()
    treasury = app.extensions["services"]["platform_treasury"]
    wallet = treasury.create_wallet(created_by_user_id=user.id)
    monkeypatch.setattr(treasury, "eth_balance", lambda address, network="Ethereum": 1.0 if address == wallet.address else 0.0)
    monkeypatch.setattr(treasury, "_send_eth", lambda **kwargs: {"provider_reference": "0xtopup", "fee_eth": 0.0001})
    monkeypatch.setattr(EvmWalletAdapter, "estimate_fee", lambda self, asset, network, destination, amount: 0.001)

    result = treasury.top_up_withdrawal_gas(withdrawal)

    event = PlatformTreasuryEvent.query.filter_by(wallet_withdrawal_id=withdrawal.id, event_type="withdrawal_gas_topup").one()
    stored = db.session.get(PlatformTreasuryWallet, wallet.id)
    assert result["status"] == "submitted"
    assert event.provider_reference == "0xtopup"
    assert stored.address not in stored.encrypted_private_key
    assert "0xtopup" in withdrawal.metadata_json


def test_platform_treasury_topup_retries_after_failed_event(app, monkeypatch) -> None:
    _enable_generated_wallets(app)
    app.config["PLATFORM_GAS_TREASURY_ENABLED"] = True
    app.config["TREASURY_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    user, _ = _create_user("topupretry")
    source = app.extensions["services"]["wallet_custody"].get_or_create_address(
        user_id=user.id,
        asset="USDT",
        network="Ethereum",
    )
    withdrawal = WalletWithdrawal(
        user_id=user.id,
        source_wallet_address_id=source.id,
        asset="USDT",
        network="Ethereum",
        destination_address="0x1111111111111111111111111111111111111111",
        amount=1.0,
        status="pending_gas_topup",
        idempotency_token="manual:topupretry",
    )
    db.session.add(withdrawal)
    db.session.flush()
    treasury = app.extensions["services"]["platform_treasury"]
    wallet = treasury.create_wallet(created_by_user_id=user.id)
    treasury._record_event(
        treasury_wallet_id=wallet.id,
        user_id=user.id,
        wallet_withdrawal_id=withdrawal.id,
        event_type="withdrawal_gas_topup",
        status="failed",
        network="Ethereum",
        amount=0.002,
        provider_reference="0xfailedgas",
        source_address=wallet.address,
        destination_address=source.address,
        idempotency_key=f"treasury:withdrawal-gas:{withdrawal.id}",
    )
    monkeypatch.setattr(treasury, "eth_balance", lambda address, network="Ethereum": 1.0 if address == wallet.address else 0.0)
    monkeypatch.setattr(treasury, "_send_eth", lambda **kwargs: {"provider_reference": "0xretrygas", "fee_eth": 0.0001})
    monkeypatch.setattr(EvmWalletAdapter, "estimate_fee", lambda self, asset, network, destination, amount: 0.001)

    result = treasury.top_up_withdrawal_gas(withdrawal)

    events = PlatformTreasuryEvent.query.filter_by(wallet_withdrawal_id=withdrawal.id, event_type="withdrawal_gas_topup").all()
    assert result["status"] == "submitted"
    assert result["provider_reference"] == "0xretrygas"
    assert len(events) == 2
    assert any(event.idempotency_key.endswith(":retry:2") for event in events)


def test_platform_treasury_topup_respects_pause_panic_and_daily_limit(app, monkeypatch) -> None:
    _enable_generated_wallets(app)
    app.config["PLATFORM_GAS_TREASURY_ENABLED"] = True
    app.config["TREASURY_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    user, _ = _create_user("topuplimits")
    source = app.extensions["services"]["wallet_custody"].get_or_create_address(
        user_id=user.id,
        asset="USDT",
        network="Ethereum",
    )
    withdrawal = WalletWithdrawal(
        user_id=user.id,
        source_wallet_address_id=source.id,
        asset="USDT",
        network="Ethereum",
        destination_address="0x1111111111111111111111111111111111111111",
        amount=1.0,
        status="pending_gas_topup",
        idempotency_token="manual:topuplimits",
    )
    db.session.add(withdrawal)
    db.session.flush()
    treasury = app.extensions["services"]["platform_treasury"]
    wallet = treasury.create_wallet(created_by_user_id=user.id)
    monkeypatch.setattr(treasury, "eth_balance", lambda address, network="Ethereum": 1.0 if address == wallet.address else 0.0)
    monkeypatch.setattr(EvmWalletAdapter, "estimate_fee", lambda self, asset, network, destination, amount: 0.001)

    treasury.set_paused(True, user_id=user.id)
    with pytest.raises(RuntimeError, match="pause"):
        treasury.top_up_withdrawal_gas(withdrawal)

    treasury.set_paused(False, user_id=user.id)
    Setting.set_json("panic_lock", True)
    with pytest.raises(RuntimeError, match="Panic lock"):
        treasury.top_up_withdrawal_gas(withdrawal)

    Setting.set_json("panic_lock", False)
    app.config["PLATFORM_GAS_TREASURY_DAILY_LIMIT_ETH"] = 0.001
    with pytest.raises(RuntimeError, match="daily gas top-up limit"):
        treasury.top_up_withdrawal_gas(withdrawal)


def test_withdrawal_queue_retries_after_gas_topup_confirmation(app) -> None:
    _enable_generated_wallets(app)
    app.config["PLATFORM_GAS_TREASURY_ENABLED"] = True
    app.config["TREASURY_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    user, _ = _create_user("queuegas")
    source = app.extensions["services"]["wallet_custody"].get_or_create_address(
        user_id=user.id,
        asset="USDT",
        network="Ethereum",
    )
    withdrawal = WalletWithdrawal(
        user_id=user.id,
        source_wallet_address_id=source.id,
        asset="USDT",
        network="Ethereum",
        destination_address="0x1111111111111111111111111111111111111111",
        amount=1.0,
        status="pending_gas_topup",
        idempotency_token="manual:queuegas",
    )
    db.session.add(withdrawal)
    db.session.flush()
    treasury = app.extensions["services"]["platform_treasury"]
    wallet = treasury.create_wallet(created_by_user_id=user.id)
    treasury._record_event(
        treasury_wallet_id=wallet.id,
        user_id=user.id,
        wallet_withdrawal_id=withdrawal.id,
        event_type="withdrawal_gas_topup",
        status="complete",
        network="Ethereum",
        amount=0.002,
        provider_reference="0xgasdone",
        source_address=wallet.address,
        destination_address=source.address,
        idempotency_key=f"treasury:withdrawal-gas:{withdrawal.id}",
    )

    class _Submitter:
        def submit_withdrawal(self, item, *, mode: str):
            item.status = "submitted"
            item.provider_reference = "0xwithdrawn"
            return item

    app.extensions["services"]["self_custody_wallet"] = _Submitter()

    result = treasury.process_withdrawal_queue()

    assert result[0]["status"] == "submitted"
    assert withdrawal.status == "submitted"
    assert withdrawal.provider_reference == "0xwithdrawn"


def test_platform_treasury_process_cli_runs_bounded_batch(app) -> None:
    runner = app.test_cli_runner()

    result = runner.invoke(args=["platform-treasury", "process", "--reserve-limit", "3", "--withdrawal-limit", "2"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["processed"] is True
    assert payload["reserve_job_count"] == 0
    assert payload["withdrawal_count"] == 0


def test_generation_fails_closed_without_valid_encryption_key(app) -> None:
    _enable_generated_wallets(app)
    app.config["TOTP_ENCRYPTION_KEY"] = ""
    custody = app.extensions["services"]["wallet_custody"]
    user, _ = _create_user("missingkey")

    try:
        custody.get_or_create_address(user_id=user.id, asset="ETH", network="Ethereum")
    except RuntimeError as exc:
        assert "TOTP_ENCRYPTION_KEY" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("wallet generation should require a valid encryption key")


def test_unsupported_token_network_combinations_fail_closed(app) -> None:
    _enable_generated_wallets(app)
    custody = app.extensions["services"]["wallet_custody"]
    user, _ = _create_user("unsupportedtokens")

    for asset, network in (("USDC", "Solana"), ("USDT", "Tron")):
        try:
            custody.get_or_create_address(user_id=user.id, asset=asset, network=network)
        except RuntimeError as exc:
            assert "no custody adapter supports" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"{asset}/{network} should fail closed")


def test_deposit_sync_does_not_credit_unconfirmed_balances(app) -> None:
    _enable_generated_wallets(app)
    fake = _FakeAdapter(amount=2.0, confirmations=0)
    custody = RealWalletCustodyService(app.config, adapters=[fake])
    app.extensions["services"]["wallet_custody"] = custody
    user, _ = _create_user("unconfirmed")
    wallet_address = custody.get_or_create_address(user_id=user.id, asset="ETH", network="Ethereum")

    result = custody.sync_address(wallet_address)

    assert result["credited"] == 0.0
    assert result["unconfirmed"] is True
    assert WalletBalance.query.filter_by(user_id=user.id, asset="ETH").one_or_none() is None
    assert WalletLedgerEvent.query.count() == 0


def test_withdrawal_approval_submits_live_and_rejection_releases_funds(app) -> None:
    _enable_generated_wallets(app)
    fake = _FakeAdapter(amount=5.0)
    custody = RealWalletCustodyService(app.config, adapters=[fake])
    app.extensions["services"]["wallet_custody"] = custody
    service = app.extensions["services"]["self_custody_wallet"]
    user, _ = _create_user("approval")
    wallet_address = custody.get_or_create_address(user_id=user.id, asset="ETH", network="Ethereum")
    balance = WalletBalance(user_id=user.id, asset="ETH", available_balance=2.0, locked_balance=0.0)
    db.session.add(balance)
    balance.available_balance -= 1.0
    balance.locked_balance += 1.0
    withdrawal = service.create_manual_withdrawal(
        user_id=user.id,
        asset="ETH",
        network="Ethereum",
        destination_address="0x1111111111111111111111111111111111111111",
        amount=1.0,
    )
    withdrawal.source_wallet_address_id = wallet_address.id
    db.session.flush()

    submitted = service.approve_withdrawal(withdrawal, approved_by_user_id=None, mode="live")

    assert submitted.status == "submitted"
    assert submitted.approved_at is not None
    assert fake.broadcasts == 1
    assert balance.available_balance == 1.0
    assert balance.locked_balance == 1.0

    balance.available_balance -= 0.5
    balance.locked_balance += 0.5
    rejected = service.create_manual_withdrawal(
        user_id=user.id,
        asset="ETH",
        network="Ethereum",
        destination_address="0x2222222222222222222222222222222222222222",
        amount=0.5,
    )
    rejected.source_wallet_address_id = wallet_address.id
    service.reject_withdrawal(rejected, rejected_by_user_id=None, reason="manual review failed")
    custody.release_failed_withdrawal(rejected)

    assert rejected.status == "rejected"
    assert balance.available_balance == 1.0
    assert balance.locked_balance == 1.0


def test_wallet_readiness_cli_reports_live_wallet_blockers(app) -> None:
    runner = app.test_cli_runner()

    result = runner.invoke(args=["wallet-readiness"])

    assert result.exit_code == 0
    payload = result.output
    assert '"use_real_addresses": true' in payload
    assert '"valid_encryption_key": false' in payload
    assert "USE_REAL_ADDRESSES is disabled" not in payload
