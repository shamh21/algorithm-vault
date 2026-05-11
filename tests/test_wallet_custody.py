from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pyotp
import pytest
from cryptography.fernet import Fernet

from app.auth import encrypt_totp_secret, password_hash
from app.extensions import db
from app.models import Setting, User, WalletAddress, WalletAuditLog, WalletBalance, WalletLedgerEvent, WalletTransaction, WalletWithdrawal
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

    assert [call[0] for call in calls] == ["eth_getTransactionCount", "eth_gasPrice", "eth_getBalance"]


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
