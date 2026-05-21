from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from app.auth import password_hash
from app.extensions import db
from app.models import User, WalletAccount, WalletAddress
from app.services.oneinch_funding import OneInchFundingConnector
from app.services.wallet_custody import BroadcastResult, EvmWalletAdapter, WalletBalanceSnapshot


def _user(username: str = "oneinch") -> User:
    user = User(username=username, password_hash=password_hash("password123"), role="user")
    db.session.add(user)
    db.session.flush()
    return user


def _wallet(user: User, asset: str, network: str, address: str) -> WalletAddress:
    account = WalletAccount(user_id=user.id, provider="mpc_signer", asset=asset, network=network, status="active")
    account.encrypted_metadata = {"custody": "mpc", "network": network}
    db.session.add(account)
    db.session.flush()
    wallet = WalletAddress(
        wallet_account_id=account.id,
        user_id=user.id,
        asset=asset,
        network=network,
        address=address,
        status="active",
        rotation_index=1,
        onchain_balance=100.0,
        onchain_checked_at=datetime.utcnow(),
        onchain_status="checked",
        onchain_confirmations=1,
    )
    wallet.encrypted_metadata = {"custody": "mpc", "signer_key_id": f"key-{asset.lower()}"}
    db.session.add(wallet)
    db.session.flush()
    return wallet


def _config(app) -> None:
    app.config.update(
        {
            "VAULT_CYCLE_ONEINCH_AUTO_CONVERSION_ENABLED": True,
            "VAULT_CYCLE_ONEINCH_API_URL": "https://api.1inch.dev/swap/v6.1",
            "VAULT_CYCLE_ONEINCH_API_KEY": "test-key",
            "VAULT_CYCLE_ONEINCH_NETWORK": "Arbitrum",
            "VAULT_CYCLE_ONEINCH_CONFIRMATION_ATTEMPTS": 1,
            "WALLET_REAL_CUSTODY_ENABLED": True,
            "WALLET_CONVERSION_SIGNER_TRANSACTIONS_ENABLED": True,
            "HYPERLIQUID_BRIDGE2_CONTRACT_ADDRESS": "0x2Df1c51E09aECF9cacB7bc98cB1742757f163dF7",
            "WALLET_EVM_NETWORKS": {"ARBITRUM": {"rpc_url": "https://arb.example.invalid", "chain_id": 42161}},
            "WALLET_EVM_TOKEN_CONTRACTS": {
                "ARBITRUM": {
                    "USDT": "0x" + ("1" * 40),
                    "USDT_DECIMALS": 6,
                    "USDC": "0x" + ("2" * 40),
                    "USDC_DECIMALS": 6,
                }
            },
        }
    )


class _FakeCustody:
    def __init__(self) -> None:
        self.transactions: list[dict[str, Any]] = []

    def supports(self, asset: str, network: str) -> bool:
        return asset.upper() in {"USDT", "USDC"} and network == "Arbitrum"

    def sign_and_broadcast_evm_transaction(self, **kwargs):
        self.transactions.append(kwargs)
        suffix = len(self.transactions)
        return BroadcastResult("submitted", f"0x{'a' * 63}{suffix}", {"tx_hash": f"0x{'a' * 63}{suffix}"})


def test_oneinch_connector_swaps_usdt_to_usdc_then_bridges_to_hyperliquid(app, monkeypatch) -> None:
    _config(app)
    user = _user()
    source = _wallet(user, "USDT", "Arbitrum", "0x" + ("9" * 40))
    custody = _FakeCustody()
    calls: list[str] = []

    def fake_http(url: str, headers: dict[str, str]) -> dict[str, Any]:
        calls.append(url)
        assert headers["Authorization"] == "Bearer test-key"
        if "/approve/allowance?" in url:
            return {"allowance": "0"}
        if "/approve/transaction?" in url:
            return {"to": "0x" + ("1" * 40), "data": "0x095ea7b3", "value": "0x0", "chainId": 42161}
        if "/quote?" in url:
            return {"dstAmount": "34990000"}
        if "/swap?" in url:
            return {"tx": {"to": "0x" + ("3" * 40), "data": "0x1234", "value": "0x0", "chainId": 42161}}
        raise AssertionError(url)

    monkeypatch.setattr(
        EvmWalletAdapter,
        "get_balance",
        lambda self, address, asset, network: WalletBalanceSnapshot(100.0, asset, True, confirmations=1),
    )
    monkeypatch.setattr(
        EvmWalletAdapter,
        "confirm_transaction",
        lambda self, provider_reference, asset, network: {"confirmed": True, "raw": {"status": "0x1"}},
    )
    connector = OneInchFundingConnector(
        app.config,
        user_id=user.id,
        wallet_custody=custody,
        hyperliquid_account_address=source.address,
        http_get=fake_http,
        sleep=lambda _: None,
    )

    conversion = connector.convert_stablecoin("live", "USDT", "USDC", 35.0, 10.0, client_reference="fund:convert")
    withdrawal = connector.withdraw_to_address(
        "live",
        "USDC",
        conversion["confirmed_amount"],
        source.address,
        network="Arbitrum",
        client_reference="fund:withdraw",
    )

    assert conversion["status"] == "confirmed"
    assert conversion["confirmed_amount"] == pytest.approx(34.95501)
    assert withdrawal["destination"] == "0x2Df1c51E09aECF9cacB7bc98cB1742757f163dF7"
    assert withdrawal["hyperliquid_account_address"] == source.address
    assert len(custody.transactions) == 3
    assert custody.transactions[-1]["transaction"]["to"] == "0x" + ("2" * 40)
    assert custody.transactions[-1]["transaction"]["data"].startswith("0xa9059cbb")
    assert any("/42161/quote?" in call for call in calls)
    assert any("/42161/swap?" in call for call in calls)


def test_oneinch_route_blocks_when_bridge_source_does_not_match_hyperliquid_account(app, monkeypatch) -> None:
    _config(app)
    user = _user("oneinch-mismatch")
    _wallet(user, "USDT", "Arbitrum", "0x" + ("8" * 40))
    monkeypatch.setattr(
        EvmWalletAdapter,
        "get_balance",
        lambda self, address, asset, network: WalletBalanceSnapshot(100.0, asset, True, confirmations=1),
    )
    connector = OneInchFundingConnector(
        app.config,
        user_id=user.id,
        wallet_custody=_FakeCustody(),
        hyperliquid_account_address="0x" + ("9" * 40),
    )

    route = connector.route_check(
        from_asset="USDT",
        to_asset="USDC",
        amount=35.0,
        hyperliquid_account_address="0x" + ("9" * 40),
    )

    assert route.ready is False
    assert "hyperliquid_bridge_source_wallet_mismatch" in route.blockers
