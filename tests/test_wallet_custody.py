from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pyotp
from cryptography.fernet import Fernet

from app.auth import encrypt_totp_secret, password_hash
from app.extensions import db
from app.models import Setting, User, WalletAddress, WalletBalance, WalletLedgerEvent, WalletTransaction, WalletWithdrawal
from app.services.wallet_custody import BroadcastResult, GeneratedWallet, RealWalletCustodyService, WalletBalanceSnapshot


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
    assert '"valid_encryption_key": false' in payload
    assert "USE_REAL_ADDRESSES is disabled" in payload
