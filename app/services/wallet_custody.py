"""Real in-app wallet custody adapters with fail-closed chain operations."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import secrets
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from eth_account import Account
from flask import current_app, has_app_context

from ..extensions import db
from ..models import DepositAddress, Setting, WalletAccount, WalletAddress, WalletAuditLog, WalletBalance, WalletLedgerEvent, WalletTransaction, WalletWithdrawal


EVM_NETWORKS = {"ETHEREUM", "ARBITRUM", "OPTIMISM", "BASE", "POLYGON", "AVALANCHE", "BSC"}
EVM_ASSETS = {"ETH", "USDC", "USDT"}
BTC_NETWORKS = {"BITCOIN"}
SOL_NETWORKS = {"SOLANA"}
XRP_NETWORKS = {"XRPLEDGER"}
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


@dataclass(frozen=True, slots=True)
class GeneratedWallet:
    address: str
    private_key: str
    public_key: str
    key_type: str
    provider: str


@dataclass(frozen=True, slots=True)
class WalletBalanceSnapshot:
    amount: float
    asset: str
    checked: bool
    confirmations: int = 0
    provider_reference: str = ""
    reason: str = ""
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class BroadcastResult:
    status: str
    provider_reference: str
    raw: dict[str, Any]


class WalletChainAdapter(Protocol):
    def supports(self, asset: str, network: str) -> bool:
        ...

    def generate_wallet(self, asset: str, network: str) -> GeneratedWallet:
        ...

    def get_balance(self, address: str, asset: str, network: str) -> WalletBalanceSnapshot:
        ...

    def estimate_fee(self, asset: str, network: str, destination: str, amount: float) -> float:
        ...

    def sign_and_broadcast(self, withdrawal: WalletWithdrawal, private_key: str) -> BroadcastResult:
        ...

    def confirm_transaction(self, provider_reference: str, asset: str, network: str) -> dict[str, Any]:
        ...


class RealWalletCustodyService:
    """Generates encrypted real wallets, syncs deposits, and broadcasts withdrawals."""

    def __init__(self, config: dict[str, Any], adapters: list[WalletChainAdapter] | None = None) -> None:
        self.config = config
        self.adapters = adapters or [
            EvmWalletAdapter(config),
            BitcoinWalletAdapter(config),
            SolanaWalletAdapter(config),
            XrpWalletAdapter(config),
        ]

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("WALLET_REAL_CUSTODY_ENABLED", False))

    def can_generate(self) -> bool:
        return self.enabled and bool(self.config.get("WALLET_ALLOW_IN_APP_KEYGEN", False))

    def supports(self, asset: str, network: str) -> bool:
        return self._adapter_for(asset, network) is not None

    def readiness(self) -> dict[str, Any]:
        assets = ("ETH", "USDC", "USDT", "BTC", "SOL", "XRP")
        networks = {
            "ETH": ["Ethereum"],
            "USDC": ["Ethereum"],
            "USDT": ["Ethereum"],
            "BTC": ["Bitcoin"],
            "SOL": ["Solana"],
            "XRP": ["XRP Ledger"],
        }
        supported: list[dict[str, Any]] = []
        blockers: list[str] = []
        for asset, asset_networks in networks.items():
            for network in asset_networks:
                pair_blockers = self.generation_blockers(asset, network)
                supported.append(
                    {
                        "asset": asset,
                        "network": network,
                        "ready": not pair_blockers,
                        "blockers": pair_blockers,
                    }
                )
                blockers.extend(f"{asset}/{network}: {reason}" for reason in pair_blockers)
        return {
            "use_real_addresses": self._real_address_mode_enabled(),
            "real_custody_enabled": self.enabled,
            "keygen_enabled": bool(self.config.get("WALLET_ALLOW_IN_APP_KEYGEN", False)),
            "valid_encryption_key": self._has_valid_encryption_key(),
            "withdrawals_enabled": bool(self.config.get("WALLET_WITHDRAWALS_ENABLED", False)),
            "approval_required": bool(self.config.get("WALLET_REQUIRE_WITHDRAWAL_APPROVAL", True)),
            "supported": supported,
            "ready": not blockers,
            "blockers": list(dict.fromkeys(blockers)),
        }

    def generation_blockers(self, asset: str, network: str) -> list[str]:
        asset_key = self._asset_key(asset)
        network_name = str(network or "").strip() or self._default_network(asset_key)
        network_key = self._network_key(network_name)
        blockers: list[str] = []

        if not self._real_address_mode_enabled():
            blockers.append("USE_REAL_ADDRESSES is disabled")
        if not self.enabled:
            blockers.append("WALLET_REAL_CUSTODY_ENABLED is disabled")
        if not bool(self.config.get("WALLET_ALLOW_IN_APP_KEYGEN", False)):
            blockers.append("wallet key generation is disabled (WALLET_ALLOW_IN_APP_KEYGEN=false)")
        if not self._has_valid_encryption_key():
            blockers.append("TOTP_ENCRYPTION_KEY must be a valid Fernet key")
        if self._adapter_for(asset_key, network_name) is None:
            blockers.append("no custody adapter supports this asset/network")

        if network_key in EVM_NETWORKS:
            if not self._evm_rpc_url(network_name):
                blockers.append("EVM RPC URL is not configured")
            if asset_key in {"USDC", "USDT"} and not EvmWalletAdapter(self.config)._token_contract(asset_key, network_name):
                blockers.append(f"{asset_key} token contract is not configured")
        elif network_key in BTC_NETWORKS:
            if not str(self.config.get("WALLET_BTC_INDEXER_URL", "") or "").strip():
                blockers.append("Bitcoin indexer URL is not configured")
        elif network_key in SOL_NETWORKS:
            if asset_key != "SOL":
                blockers.append("SPL token support is not configured")
            if not str(self.config.get("WALLET_SOLANA_RPC_URL", "") or "").strip():
                blockers.append("Solana RPC URL is not configured")
        elif network_key in XRP_NETWORKS:
            if not str(self.config.get("WALLET_XRP_RPC_URL", "") or "").strip():
                blockers.append("XRP RPC URL is not configured")

        return list(dict.fromkeys(blockers))

    def get_or_create_address(
        self,
        *,
        user_id: int,
        asset: str,
        network: str,
        deposit_address_id: int | None = None,
        force_new: bool = False,
    ) -> WalletAddress:
        asset_key = self._asset_key(asset)
        network_name = str(network or "").strip() or self._default_network(asset_key)
        blockers = self.generation_blockers(asset_key, network_name)
        if blockers:
            raise RuntimeError("Real wallet generation is not ready: " + "; ".join(blockers))
        adapter = self._require_adapter(asset_key, network_name)

        existing = (
            WalletAddress.query.filter_by(
                user_id=user_id,
                asset=asset_key,
                network=network_name,
                status="active",
            )
            .order_by(WalletAddress.rotation_index.desc())
            .first()
        )
        if existing is not None and not force_new:
            if deposit_address_id is not None and existing.deposit_address_id is None:
                existing.deposit_address_id = deposit_address_id
            return existing

        generated = adapter.generate_wallet(asset_key, network_name)
        duplicate = (
            WalletAddress.query.filter_by(
                user_id=user_id,
                asset=asset_key,
                network=network_name,
                address=generated.address,
            )
            .order_by(WalletAddress.rotation_index.desc())
            .first()
        )
        if duplicate is not None:
            raise RuntimeError("Generated replacement wallet address matched an existing address.")
        account = self._account_for(user_id, asset_key, network_name)
        latest = (
            WalletAddress.query.filter_by(user_id=user_id, asset=asset_key, network=network_name)
            .order_by(WalletAddress.rotation_index.desc())
            .first()
        )
        wallet_address = WalletAddress(
            wallet_account_id=account.id,
            user_id=user_id,
            deposit_address_id=deposit_address_id,
            asset=asset_key,
            network=network_name,
            address=generated.address,
            status="active",
            rotation_index=(latest.rotation_index if latest else 0) + 1,
        )
        wallet_address.encrypted_metadata = {
            "custody": "in_app",
            "provider": generated.provider,
            "key_type": generated.key_type,
            "public_key": generated.public_key,
            "encrypted_private_key": self._encrypt(generated.private_key),
            "sync_status": "not_synced",
            "last_sync_cursor": "",
        }
        db.session.add(wallet_address)
        db.session.flush()
        self._audit(
            user_id=user_id,
            wallet_account_id=account.id,
            action="wallet_address_generated",
            status="active",
            message=f"Generated real {asset_key} wallet address on {network_name}.",
            metadata={
                "asset": asset_key,
                "network": network_name,
                "address": generated.address,
                "provider": generated.provider,
                "key_type": generated.key_type,
            },
        )
        return wallet_address

    def sync_user(self, user_id: int) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "synced": 0, "credited": 0.0, "errors": []}
        summary = {"enabled": True, "synced": 0, "credited": 0.0, "errors": []}
        addresses = WalletAddress.query.filter_by(user_id=user_id, status="active").all()
        for wallet_address in addresses:
            try:
                result = self.sync_address(wallet_address)
                summary["synced"] += 1
                summary["credited"] += float(result.get("credited", 0.0) or 0.0)
            except Exception as exc:  # noqa: BLE001
                summary["errors"].append(str(exc))
        return summary

    def sync_address(self, wallet_address: WalletAddress) -> dict[str, Any]:
        adapter = self._require_adapter(wallet_address.asset, wallet_address.network)
        snapshot = adapter.get_balance(wallet_address.address, wallet_address.asset, wallet_address.network)
        metadata = wallet_address.encrypted_metadata
        metadata["last_sync_status"] = "checked" if snapshot.checked else "failed"
        metadata["last_sync_reason"] = snapshot.reason
        wallet_address.encrypted_metadata = metadata
        if not snapshot.checked:
            self._audit(
                user_id=wallet_address.user_id,
                wallet_account_id=wallet_address.wallet_account_id,
                action="wallet_deposit_sync_failed",
                status="failed",
                message=snapshot.reason or "Wallet balance sync failed.",
                metadata={"asset": wallet_address.asset, "network": wallet_address.network, "address": wallet_address.address},
            )
            return {"credited": 0.0, "checked": False}

        required = int(self._duration_map_value("WALLET_REQUIRED_CONFIRMATIONS", wallet_address.network, 1))
        if snapshot.confirmations < required:
            return {"credited": 0.0, "checked": True, "unconfirmed": True}

        credited = self._credited_amount(wallet_address)
        delta = max(0.0, float(snapshot.amount or 0.0) - credited)
        if delta <= 1e-12:
            return {"credited": 0.0, "checked": True}

        provider_reference = snapshot.provider_reference or f"balance:{wallet_address.network}:{wallet_address.address}:{wallet_address.asset}:{snapshot.amount:.12f}"
        idempotency_key = f"deposit:{wallet_address.network}:{wallet_address.address}:{wallet_address.asset}:{provider_reference}"
        existing = WalletLedgerEvent.query.filter_by(idempotency_key=idempotency_key).one_or_none()
        if existing is not None:
            return {"credited": 0.0, "checked": True, "duplicate": True}

        event = WalletLedgerEvent(
            user_id=wallet_address.user_id,
            wallet_address_id=wallet_address.id,
            deposit_address_id=wallet_address.deposit_address_id,
            asset=wallet_address.asset,
            network=wallet_address.network,
            address=wallet_address.address,
            event_type="deposit",
            provider_reference=provider_reference,
            idempotency_key=idempotency_key,
            amount=delta,
            confirmations=snapshot.confirmations,
            status="complete",
        )
        event.details = snapshot.metadata or {}
        db.session.add(event)
        balance = WalletBalance.query.filter_by(user_id=wallet_address.user_id, asset=wallet_address.asset).one_or_none()
        if balance is None:
            balance = WalletBalance(user_id=wallet_address.user_id, asset=wallet_address.asset)
            db.session.add(balance)
        balance.available_balance = float(balance.available_balance or 0.0) + delta
        db.session.add(
            WalletTransaction(
                user_id=wallet_address.user_id,
                asset=wallet_address.asset,
                amount=delta,
                transaction_type="deposit",
                status="complete",
                network=wallet_address.network,
                note=f"Confirmed on-chain deposit {provider_reference}.",
            )
        )
        self._audit(
            user_id=wallet_address.user_id,
            wallet_account_id=wallet_address.wallet_account_id,
            action="wallet_deposit_credited",
            status="complete",
            message=f"Credited {delta:.8f} {wallet_address.asset} from confirmed on-chain balance.",
            metadata={
                "asset": wallet_address.asset,
                "network": wallet_address.network,
                "address": wallet_address.address,
                "provider_reference": provider_reference,
                "amount": delta,
            },
        )
        db.session.flush()
        return {"credited": delta, "checked": True}

    def sign_and_broadcast(self, withdrawal: WalletWithdrawal, *, mode: str) -> BroadcastResult:
        if str(mode or "").lower() != "live":
            raise RuntimeError("Real wallet withdrawals can only be broadcast in live mode.")
        if not self.enabled:
            raise RuntimeError("Real wallet custody is disabled.")
        wallet_address = self._withdrawal_source(withdrawal)
        adapter = self._require_adapter(withdrawal.asset, withdrawal.network)
        private_key = self._private_key(wallet_address)
        return adapter.sign_and_broadcast(withdrawal, private_key)

    def release_failed_withdrawal(self, withdrawal: WalletWithdrawal) -> None:
        balance = WalletBalance.query.filter_by(user_id=withdrawal.user_id, asset=withdrawal.asset).one_or_none()
        if balance is None:
            return
        amount = float(withdrawal.amount or 0.0)
        balance.locked_balance = max(0.0, float(balance.locked_balance or 0.0) - amount)
        balance.available_balance = float(balance.available_balance or 0.0) + amount

    def _withdrawal_source(self, withdrawal: WalletWithdrawal) -> WalletAddress:
        if withdrawal.source_wallet_address_id:
            source = db.session.get(WalletAddress, int(withdrawal.source_wallet_address_id))
        else:
            source = (
                WalletAddress.query.filter_by(
                    user_id=withdrawal.user_id,
                    asset=withdrawal.asset,
                    network=withdrawal.network,
                    status="active",
                )
                .order_by(WalletAddress.rotation_index.desc())
                .first()
            )
        if source is None:
            raise RuntimeError("No active source wallet address is available for withdrawal.")
        return source

    def _private_key(self, wallet_address: WalletAddress) -> str:
        encrypted = str((wallet_address.encrypted_metadata or {}).get("encrypted_private_key") or "")
        if not encrypted:
            raise RuntimeError("Wallet private key is unavailable.")
        return self._decrypt(encrypted)

    def _account_for(self, user_id: int, asset: str, network: str) -> WalletAccount:
        account = WalletAccount.query.filter_by(
            user_id=user_id,
            provider=str(self.config.get("WALLET_PROVIDER", "in_app_custody") or "in_app_custody"),
            asset=asset,
            network=network,
        ).one_or_none()
        if account is None:
            account = WalletAccount(
                user_id=user_id,
                provider=str(self.config.get("WALLET_PROVIDER", "in_app_custody") or "in_app_custody"),
                asset=asset,
                network=network,
                status="active",
            )
            account.encrypted_metadata = {"custody": "in_app", "network": network}
            db.session.add(account)
            db.session.flush()
        return account

    def _credited_amount(self, wallet_address: WalletAddress) -> float:
        events = WalletLedgerEvent.query.filter_by(
            wallet_address_id=wallet_address.id,
            asset=wallet_address.asset,
            network=wallet_address.network,
            event_type="deposit",
            status="complete",
        ).all()
        return sum(float(event.amount or 0.0) for event in events)

    def _adapter_for(self, asset: str, network: str) -> WalletChainAdapter | None:
        for adapter in self.adapters:
            if adapter.supports(asset, network):
                return adapter
        return None

    def _require_adapter(self, asset: str, network: str) -> WalletChainAdapter:
        adapter = self._adapter_for(asset, network)
        if adapter is None:
            raise RuntimeError(f"No real custody adapter supports {asset} on {network}.")
        return adapter

    def _duration_map_value(self, key: str, bucket: str, default: float) -> float:
        mapping = self.config.get(key) or {}
        if not isinstance(mapping, dict):
            return default
        for candidate in (bucket, str(bucket).lower(), self._network_key(bucket), self._network_key(bucket).lower()):
            if candidate in mapping:
                return _float(mapping.get(candidate), default)
        return default

    def _real_address_mode_enabled(self) -> bool:
        enabled = bool(self.config.get("USE_REAL_ADDRESSES", False))
        if has_app_context():
            enabled = bool(Setting.get_json("use_real_addresses", enabled))
        return enabled

    def _has_valid_encryption_key(self) -> bool:
        configured = str(self.config.get("TOTP_ENCRYPTION_KEY", "") or "").strip()
        if not configured:
            return False
        try:
            Fernet(configured.encode("utf-8"))
        except Exception:  # noqa: BLE001
            return False
        return True

    def _evm_rpc_url(self, network: str) -> str:
        return EvmWalletAdapter(self.config)._rpc_url(network)

    def _encrypt(self, value: str) -> str:
        return _fernet(self.config).encrypt(value.encode("utf-8")).decode("utf-8")

    def _decrypt(self, value: str) -> str:
        return _fernet(self.config).decrypt(value.encode("utf-8")).decode("utf-8")

    def _audit(
        self,
        *,
        user_id: int | None,
        action: str,
        status: str,
        message: str,
        wallet_account_id: int | None = None,
        wallet_withdrawal_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WalletAuditLog:
        entry = WalletAuditLog(
            user_id=user_id,
            wallet_account_id=wallet_account_id,
            wallet_withdrawal_id=wallet_withdrawal_id,
            category="wallet",
            action=action,
            status=status,
            message=message,
        )
        entry.details = metadata or {}
        db.session.add(entry)
        db.session.flush()
        return entry

    @staticmethod
    def _asset_key(asset: str) -> str:
        return "".join(ch for ch in str(asset or "").upper() if ch.isalnum())

    @staticmethod
    def _network_key(network: str) -> str:
        return "".join(ch for ch in str(network or "").upper() if ch.isalnum())

    @staticmethod
    def _default_network(asset: str) -> str:
        if asset == "BTC":
            return "Bitcoin"
        if asset == "SOL":
            return "Solana"
        if asset == "XRP":
            return "XRP Ledger"
        return "Ethereum"


class EvmWalletAdapter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def supports(self, asset: str, network: str) -> bool:
        return _network_key(network) in EVM_NETWORKS and _asset_key(asset) in EVM_ASSETS

    def generate_wallet(self, asset: str, network: str) -> GeneratedWallet:
        account = Account.create(secrets.token_hex(32))
        return GeneratedWallet(
            address=account.address,
            private_key=account.key.hex(),
            public_key=account.address,
            key_type="secp256k1",
            provider="evm",
        )

    def get_balance(self, address: str, asset: str, network: str) -> WalletBalanceSnapshot:
        rpc_url = self._rpc_url(network)
        if not rpc_url:
            return WalletBalanceSnapshot(0.0, _asset_key(asset), False, reason="EVM RPC URL is not configured")
        asset_key = _asset_key(asset)
        try:
            if asset_key == "ETH":
                raw = self._rpc("eth_getBalance", [address, "latest"], network=network)
                amount = int(str(raw or "0x0"), 16) / 10**18
            else:
                contract = self._token_contract(asset_key, network)
                if not contract:
                    return WalletBalanceSnapshot(0.0, asset_key, False, reason=f"{asset_key} token contract is not configured")
                selector = "0x70a08231"
                data = selector + address.lower().replace("0x", "").rjust(64, "0")
                raw = self._rpc("eth_call", [{"to": contract, "data": data}, "latest"], network=network)
                decimals = self._token_decimals(asset_key, network)
                amount = int(str(raw or "0x0"), 16) / 10**decimals
            block = self._rpc("eth_blockNumber", [], network=network)
            return WalletBalanceSnapshot(
                amount=amount,
                asset=asset_key,
                checked=True,
                confirmations=max(int(self.config.get("WALLET_REQUIRED_CONFIRMATIONS", {}).get(_network_key(network), 1) or 1), 1),
                provider_reference=f"evm-balance:{_network_key(network)}:{address}:{asset_key}:{amount:.12f}:{block}",
                metadata={"block": block},
            )
        except Exception as exc:  # noqa: BLE001
            return WalletBalanceSnapshot(0.0, asset_key, False, reason=f"EVM balance check failed: {exc}")

    def estimate_fee(self, asset: str, network: str, destination: str, amount: float) -> float:
        gas = 21_000 if _asset_key(asset) == "ETH" else 70_000
        try:
            gas_price = int(str(self._rpc("eth_gasPrice", [], network=network) or "0x0"), 16)
        except Exception:  # noqa: BLE001
            gas_price = 0
        return gas * gas_price / 10**18

    def sign_and_broadcast(self, withdrawal: WalletWithdrawal, private_key: str) -> BroadcastResult:
        network = withdrawal.network
        asset = _asset_key(withdrawal.asset)
        source = current_app.extensions["services"]["wallet_custody"]._withdrawal_source(withdrawal) if has_app_context() else None
        from_address = source.address if source is not None else Account.from_key(private_key).address
        nonce = int(str(self._rpc("eth_getTransactionCount", [from_address, "pending"], network=network) or "0x0"), 16)
        gas_price = int(str(self._rpc("eth_gasPrice", [], network=network) or "0x0"), 16)
        chain_id = int(self._chain_config(network).get("chain_id", 1) or 1)
        if asset == "ETH":
            value = int(float(withdrawal.amount or 0.0) * 10**18)
            tx = {
                "chainId": chain_id,
                "nonce": nonce,
                "to": withdrawal.destination_address,
                "value": value,
                "gas": 21_000,
                "gasPrice": gas_price,
                "data": b"",
            }
        else:
            contract = self._token_contract(asset, network)
            if not contract:
                raise RuntimeError(f"{asset} token contract is not configured")
            decimals = self._token_decimals(asset, network)
            amount_units = int(float(withdrawal.amount or 0.0) * 10**decimals)
            data = "0xa9059cbb" + withdrawal.destination_address.lower().replace("0x", "").rjust(64, "0") + hex(amount_units)[2:].rjust(64, "0")
            tx = {
                "chainId": chain_id,
                "nonce": nonce,
                "to": contract,
                "value": 0,
                "gas": 70_000,
                "gasPrice": gas_price,
                "data": data,
            }
        signed = Account.sign_transaction(tx, private_key)
        raw_tx = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
        raw_hex = raw_tx.hex()
        if not raw_hex.startswith("0x"):
            raw_hex = "0x" + raw_hex
        tx_hash = self._rpc("eth_sendRawTransaction", [raw_hex], network=network)
        return BroadcastResult("submitted", str(tx_hash or ""), {"tx_hash": tx_hash})

    def confirm_transaction(self, provider_reference: str, asset: str, network: str) -> dict[str, Any]:
        receipt = self._rpc("eth_getTransactionReceipt", [provider_reference], network=network)
        return {"confirmed": bool(receipt and receipt.get("status") == "0x1"), "raw": receipt}

    def _rpc_url(self, network: str) -> str:
        chain_config = self._chain_config(network)
        return str(chain_config.get("rpc_url") or self.config.get("WALLET_EVM_RPC_URL", "") or "").strip()

    def _chain_config(self, network: str) -> dict[str, Any]:
        mapping = self.config.get("WALLET_EVM_NETWORKS") or {}
        if not isinstance(mapping, dict):
            return {}
        key = _network_key(network)
        return mapping.get(key) or mapping.get(key.lower()) or {}

    def _token_contract(self, asset: str, network: str) -> str:
        mapping = self.config.get("WALLET_EVM_TOKEN_CONTRACTS") or {}
        if not isinstance(mapping, dict):
            return ""
        network_key = _network_key(network)
        asset_key = _asset_key(asset)
        nested = mapping.get(network_key) or mapping.get(network_key.lower()) or {}
        if isinstance(nested, dict):
            return str(nested.get(asset_key) or nested.get(asset_key.lower()) or "").strip()
        return str(mapping.get(asset_key) or mapping.get(asset_key.lower()) or "").strip()

    def _token_decimals(self, asset: str, network: str) -> int:
        mapping = self.config.get("WALLET_EVM_TOKEN_CONTRACTS") or {}
        network_key = _network_key(network)
        asset_key = _asset_key(asset)
        nested = mapping.get(network_key) or mapping.get(network_key.lower()) or {}
        if isinstance(nested, dict):
            decimals = nested.get(f"{asset_key}_DECIMALS") or nested.get(f"{asset_key.lower()}_decimals")
            if decimals is not None:
                return int(decimals)
        return 6 if asset_key in {"USDC", "USDT"} else 18

    def _rpc(self, method: str, params: list[Any], *, network: str) -> Any:
        return _json_rpc(self._rpc_url(network), method, params)


class BitcoinWalletAdapter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def supports(self, asset: str, network: str) -> bool:
        return _asset_key(asset) == "BTC" and _network_key(network) in BTC_NETWORKS

    def generate_wallet(self, asset: str, network: str) -> GeneratedWallet:
        private_key = ec.generate_private_key(ec.SECP256K1())
        private_value = private_key.private_numbers().private_value.to_bytes(32, "big")
        public_key = private_key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.CompressedPoint)
        payload = b"\x00" + _hash160(public_key)
        address = _base58check(payload)
        return GeneratedWallet(address, private_value.hex(), public_key.hex(), "secp256k1-p2pkh", "bitcoin")

    def get_balance(self, address: str, asset: str, network: str) -> WalletBalanceSnapshot:
        base_url = str(self.config.get("WALLET_BTC_INDEXER_URL", "") or "").rstrip("/")
        if not base_url:
            return WalletBalanceSnapshot(0.0, "BTC", False, reason="Bitcoin indexer URL is not configured")
        try:
            payload = _http_json(f"{base_url}/address/{address}")
            chain_stats = payload.get("chain_stats", {})
            funded = int(chain_stats.get("funded_txo_sum", 0) or 0)
            spent = int(chain_stats.get("spent_txo_sum", 0) or 0)
            amount = max(funded - spent, 0) / 10**8
            tx_count = int(chain_stats.get("tx_count", 0) or 0)
            return WalletBalanceSnapshot(amount, "BTC", True, confirmations=1 if tx_count else 0, provider_reference=f"btc-balance:{address}:{amount:.12f}:{tx_count}", metadata=payload)
        except Exception as exc:  # noqa: BLE001
            return WalletBalanceSnapshot(0.0, "BTC", False, reason=f"Bitcoin balance check failed: {exc}")

    def estimate_fee(self, asset: str, network: str, destination: str, amount: float) -> float:
        base_url = str(self.config.get("WALLET_BTC_INDEXER_URL", "") or "").rstrip("/")
        if not base_url:
            return 0.0
        try:
            estimates = _http_json(f"{base_url}/fee-estimates")
            sats_per_vbyte = _float(estimates.get("2") or estimates.get("3") or estimates.get("6"), 10.0)
        except Exception:  # noqa: BLE001
            sats_per_vbyte = 10.0
        return (226 * sats_per_vbyte) / 10**8

    def sign_and_broadcast(self, withdrawal: WalletWithdrawal, private_key: str) -> BroadcastResult:
        base_url = str(self.config.get("WALLET_BTC_INDEXER_URL", "") or "").rstrip("/")
        if not base_url:
            raise RuntimeError("Bitcoin indexer URL is not configured")
        try:
            from bitcoinlib.keys import Key
            from bitcoinlib.transactions import Transaction
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("bitcoinlib is required for Bitcoin signing") from exc

        source = current_app.extensions["services"]["wallet_custody"]._withdrawal_source(withdrawal) if has_app_context() else None
        source_address = source.address if source is not None else Key(private_key, network="bitcoin").address()
        utxos = _http_json_list(f"{base_url}/address/{source_address}/utxo")
        if not utxos:
            raise RuntimeError("No confirmed Bitcoin UTXOs are available for withdrawal")
        try:
            estimates = _http_json(f"{base_url}/fee-estimates")
            sats_per_vbyte = _float(estimates.get("2") or estimates.get("3") or estimates.get("6"), 10.0)
        except Exception:  # noqa: BLE001
            sats_per_vbyte = 10.0

        amount_sats = int(float(withdrawal.amount or 0.0) * 10**8)
        if amount_sats <= 0:
            raise RuntimeError("Bitcoin withdrawal amount must be positive")
        selected: list[dict[str, Any]] = []
        total_sats = 0
        fee_sats = 0
        for utxo in sorted(utxos, key=lambda item: int(item.get("value", 0) or 0), reverse=True):
            selected.append(utxo)
            total_sats += int(utxo.get("value", 0) or 0)
            estimated_vbytes = (180 * len(selected)) + (34 * 2) + 10
            fee_sats = max(250, int(math.ceil(estimated_vbytes * sats_per_vbyte)))
            if total_sats >= amount_sats + fee_sats:
                break
        if total_sats < amount_sats + fee_sats:
            raise RuntimeError("Insufficient confirmed Bitcoin balance after network fee")

        key = Key(private_key, network="bitcoin")
        tx = Transaction(network="bitcoin", witness_type="legacy")
        for utxo in selected:
            tx.add_input(
                utxo["txid"],
                int(utxo["vout"]),
                keys=[key],
                address=source_address,
                value=int(utxo["value"]),
                witness_type="legacy",
            )
        tx.add_output(amount_sats, address=withdrawal.destination_address)
        change_sats = total_sats - amount_sats - fee_sats
        if change_sats > 546:
            tx.add_output(change_sats, address=source_address, change=True)
        tx.fee = fee_sats
        tx.sign([key])
        raw_hex = tx.raw_hex()
        txid = _http_text_post(f"{base_url}/tx", raw_hex)
        return BroadcastResult("submitted", str(txid or ""), {"tx_hash": txid, "fee_sats": fee_sats})

    def confirm_transaction(self, provider_reference: str, asset: str, network: str) -> dict[str, Any]:
        base_url = str(self.config.get("WALLET_BTC_INDEXER_URL", "") or "").rstrip("/")
        if not base_url or not provider_reference:
            return {"confirmed": False, "raw": {}}
        raw = _http_json(f"{base_url}/tx/{provider_reference}")
        status = raw.get("status", {}) if isinstance(raw, dict) else {}
        return {"confirmed": bool(status.get("confirmed")), "raw": raw}


class SolanaWalletAdapter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def supports(self, asset: str, network: str) -> bool:
        return _asset_key(asset) == "SOL" and _network_key(network) in SOL_NETWORKS

    def generate_wallet(self, asset: str, network: str) -> GeneratedWallet:
        private_key = ed25519.Ed25519PrivateKey.generate()
        raw_private = private_key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
        raw_public = private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return GeneratedWallet(_base58(raw_public), raw_private.hex(), raw_public.hex(), "ed25519", "solana")

    def get_balance(self, address: str, asset: str, network: str) -> WalletBalanceSnapshot:
        rpc_url = str(self.config.get("WALLET_SOLANA_RPC_URL", "") or "").strip()
        if not rpc_url:
            return WalletBalanceSnapshot(0.0, _asset_key(asset), False, reason="Solana RPC URL is not configured")
        if _asset_key(asset) != "SOL":
            return WalletBalanceSnapshot(0.0, _asset_key(asset), False, reason="SPL token indexer is not configured")
        try:
            raw = _json_rpc(rpc_url, "getBalance", [address])
            value = raw.get("value", 0) if isinstance(raw, dict) else 0
            amount = int(value or 0) / 10**9
            return WalletBalanceSnapshot(amount, "SOL", True, confirmations=1, provider_reference=f"sol-balance:{address}:{amount:.12f}", metadata={"raw": raw})
        except Exception as exc:  # noqa: BLE001
            return WalletBalanceSnapshot(0.0, "SOL", False, reason=f"Solana balance check failed: {exc}")

    def estimate_fee(self, asset: str, network: str, destination: str, amount: float) -> float:
        return 0.000005

    def sign_and_broadcast(self, withdrawal: WalletWithdrawal, private_key: str) -> BroadcastResult:
        rpc_url = str(self.config.get("WALLET_SOLANA_RPC_URL", "") or "").strip()
        if not rpc_url:
            raise RuntimeError("Solana RPC URL is not configured")
        if _asset_key(withdrawal.asset) != "SOL":
            raise RuntimeError("SPL token withdrawals require a configured token-account signer path")
        try:
            from solders.hash import Hash
            from solders.keypair import Keypair
            from solders.message import Message
            from solders.pubkey import Pubkey
            from solders.system_program import TransferParams, transfer
            from solders.transaction import Transaction
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("solana/solders SDK is required for Solana signing") from exc

        keypair = Keypair.from_seed(bytes.fromhex(private_key))
        destination = Pubkey.from_string(withdrawal.destination_address)
        lamports = int(float(withdrawal.amount or 0.0) * 10**9)
        if lamports <= 0:
            raise RuntimeError("Solana withdrawal amount must be positive")
        blockhash_raw = _json_rpc(rpc_url, "getLatestBlockhash", [{"commitment": "finalized"}])
        blockhash_value = str(blockhash_raw.get("value", {}).get("blockhash", "") if isinstance(blockhash_raw, dict) else "")
        if not blockhash_value:
            raise RuntimeError("Solana latest blockhash is unavailable")
        blockhash = Hash.from_string(blockhash_value)
        instruction = transfer(
            TransferParams(
                from_pubkey=keypair.pubkey(),
                to_pubkey=destination,
                lamports=lamports,
            )
        )
        message = Message.new_with_blockhash([instruction], keypair.pubkey(), blockhash)
        transaction = Transaction([keypair], message, blockhash)
        encoded = base64.b64encode(bytes(transaction)).decode("ascii")
        signature = _json_rpc(rpc_url, "sendTransaction", [encoded, {"encoding": "base64", "preflightCommitment": "confirmed"}])
        return BroadcastResult("submitted", str(signature or ""), {"signature": signature})

    def confirm_transaction(self, provider_reference: str, asset: str, network: str) -> dict[str, Any]:
        rpc_url = str(self.config.get("WALLET_SOLANA_RPC_URL", "") or "").strip()
        if not rpc_url or not provider_reference:
            return {"confirmed": False, "raw": {}}
        raw = _json_rpc(rpc_url, "getSignatureStatuses", [[provider_reference], {"searchTransactionHistory": True}])
        statuses = raw.get("value", []) if isinstance(raw, dict) else []
        status = statuses[0] if statuses else None
        return {
            "confirmed": bool(status and status.get("confirmationStatus") in {"confirmed", "finalized"} and not status.get("err")),
            "raw": raw,
        }


class XrpWalletAdapter:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def supports(self, asset: str, network: str) -> bool:
        return _asset_key(asset) == "XRP" and _network_key(network) in XRP_NETWORKS

    def generate_wallet(self, asset: str, network: str) -> GeneratedWallet:
        try:
            from xrpl.wallet import Wallet
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("xrpl-py is required for XRP wallet generation") from exc

        wallet = Wallet.create()
        return GeneratedWallet(
            wallet.classic_address,
            wallet.seed,
            wallet.public_key,
            "xrpl-seed",
            "xrp",
        )

    def get_balance(self, address: str, asset: str, network: str) -> WalletBalanceSnapshot:
        rpc_url = str(self.config.get("WALLET_XRP_RPC_URL", "") or "").strip()
        if not rpc_url:
            return WalletBalanceSnapshot(0.0, "XRP", False, reason="XRP RPC URL is not configured")
        try:
            raw = _json_rpc(rpc_url, "account_info", [{"account": address, "ledger_index": "validated"}])
            account_data = raw.get("account_data", {}) if isinstance(raw, dict) else {}
            amount = int(account_data.get("Balance", 0) or 0) / 10**6
            ledger_index = raw.get("ledger_index") if isinstance(raw, dict) else ""
            return WalletBalanceSnapshot(amount, "XRP", True, confirmations=1, provider_reference=f"xrp-balance:{address}:{amount:.12f}:{ledger_index}", metadata={"raw": raw})
        except Exception as exc:  # noqa: BLE001
            return WalletBalanceSnapshot(0.0, "XRP", False, reason=f"XRP balance check failed: {exc}")

    def estimate_fee(self, asset: str, network: str, destination: str, amount: float) -> float:
        return 0.000012

    def sign_and_broadcast(self, withdrawal: WalletWithdrawal, private_key: str) -> BroadcastResult:
        rpc_url = str(self.config.get("WALLET_XRP_RPC_URL", "") or "").strip()
        if not rpc_url:
            raise RuntimeError("XRP RPC URL is not configured")
        if not private_key.startswith("s"):
            raise RuntimeError("Existing XRP key material is not an XRPL seed and cannot be signed by this adapter")
        try:
            from xrpl.clients import JsonRpcClient
            from xrpl.models.transactions import Payment
            from xrpl.transaction import submit_and_wait
            from xrpl.utils import xrp_to_drops
            from xrpl.wallet import Wallet
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("xrpl-py is required for XRP signing") from exc

        wallet = Wallet.from_seed(private_key)
        payment = Payment(
            account=wallet.classic_address,
            destination=withdrawal.destination_address,
            amount=xrp_to_drops(float(withdrawal.amount or 0.0)),
        )
        response = submit_and_wait(payment, JsonRpcClient(rpc_url), wallet)
        result = response.result if isinstance(response.result, dict) else {}
        tx_hash = str(result.get("hash") or result.get("tx_json", {}).get("hash") or "")
        return BroadcastResult("submitted", tx_hash, {"result": result})

    def confirm_transaction(self, provider_reference: str, asset: str, network: str) -> dict[str, Any]:
        rpc_url = str(self.config.get("WALLET_XRP_RPC_URL", "") or "").strip()
        if not rpc_url or not provider_reference:
            return {"confirmed": False, "raw": {}}
        raw = _json_rpc(rpc_url, "tx", [{"transaction": provider_reference, "binary": False}])
        return {"confirmed": bool(isinstance(raw, dict) and raw.get("validated") and raw.get("meta", {}).get("TransactionResult") == "tesSUCCESS"), "raw": raw}


def _json_rpc(url: str, method: str, params: list[Any]) -> Any:
    if not url:
        raise RuntimeError("RPC URL is not configured")
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode("utf-8")
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=5.0) as response:
        body = json.loads(response.read().decode("utf-8"))
    if "error" in body:
        raise RuntimeError(body["error"])
    return body.get("result")


def _http_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(request, timeout=5.0) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body if isinstance(body, dict) else {}


def _http_json_list(url: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    with urllib.request.urlopen(request, timeout=5.0) as response:
        body = json.loads(response.read().decode("utf-8"))
    return [item for item in body if isinstance(item, dict)] if isinstance(body, list) else []


def _http_text_post(url: str, payload: str) -> str:
    request = urllib.request.Request(
        url,
        data=payload.encode("utf-8"),
        headers={"Content-Type": "text/plain"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10.0) as response:
        return response.read().decode("utf-8").strip()


def _fernet(config: dict[str, Any]) -> Fernet:
    configured = str(config.get("TOTP_ENCRYPTION_KEY", "") or "").strip()
    if configured:
        try:
            return Fernet(configured.encode("utf-8"))
        except Exception:  # noqa: BLE001
            pass
    raw = str(config.get("SECRET_KEY", "dev-secret-change-me")).encode("utf-8")
    key = base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
    return Fernet(key)


def _hash160(payload: bytes) -> bytes:
    sha = hashlib.sha256(payload).digest()
    ripe = hashlib.new("ripemd160")
    ripe.update(sha)
    return ripe.digest()


def _base58check(payload: bytes) -> str:
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return _base58(payload + checksum)


def _base58(payload: bytes, *, alphabet: str = BASE58_ALPHABET) -> str:
    value = int.from_bytes(payload, "big")
    encoded = ""
    while value:
        value, remainder = divmod(value, 58)
        encoded = alphabet[remainder] + encoded
    leading = 0
    for byte in payload:
        if byte == 0:
            leading += 1
        else:
            break
    return alphabet[0] * leading + (encoded or alphabet[0])


def _asset_key(asset: str) -> str:
    return "".join(ch for ch in str(asset or "").upper() if ch.isalnum())


def _network_key(network: str) -> str:
    return "".join(ch for ch in str(network or "").upper() if ch.isalnum())


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
