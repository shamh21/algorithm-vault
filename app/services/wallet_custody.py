"""Real in-app wallet custody adapters with fail-closed chain operations."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import secrets
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from typing import Any, Protocol

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from eth_account import Account
from flask import current_app, has_app_context

from ..extensions import db
from ..models import WalletAccount, WalletAddress, WalletAuditLog, WalletBalance, WalletLedgerEvent, WalletTransaction, WalletWithdrawal
from .failures import WalletBroadcastError, WalletCustodyError


EVM_NETWORKS = {"ETHEREUM", "ARBITRUM", "OPTIMISM", "BASE", "POLYGON", "AVALANCHE", "BSC"}
EVM_ASSETS = {"ETH", "USDC", "USDT"}
ERC20_TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
BTC_NETWORKS = {"BITCOIN"}
SOL_NETWORKS = {"SOLANA"}
XRP_NETWORKS = {"XRPLEDGER"}
ACTIVE_WITHDRAWAL_RESERVE_STATUSES = {
    "pending_approval",
    "pending_submission",
    "pending_gas_topup",
    "queued_treasury_solvency",
    "submitted",
}
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
XRP_BASE58_ALPHABET = "rpshnaf39wBUDNEGHJKLM4PQRST7VWXYZ2bcdeCg65jkm8oFqi1tuvAxyz"


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
        self._persist_onchain_snapshot(wallet_address, snapshot)
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
        if has_app_context():
            try:
                current_app.extensions["services"]["platform_treasury"].reserve_for_deposit(
                    wallet_address,
                    event,
                    amount=delta,
                )
            except Exception as exc:  # noqa: BLE001
                self._audit(
                    user_id=wallet_address.user_id,
                    wallet_account_id=wallet_address.wallet_account_id,
                    action="wallet_deposit_gas_reserve_failed",
                    status="failed",
                    message=f"Deposit credited, but treasury gas reserve processing failed: {exc}",
                    metadata={
                        "asset": wallet_address.asset,
                        "network": wallet_address.network,
                        "wallet_ledger_event_id": event.id,
                        "reason": str(exc),
                    },
                )
        db.session.flush()
        return {"credited": delta, "checked": True}

    def onchain_balance_for_user_asset(
        self,
        user_id: int,
        asset: str,
        network: str,
        *,
        refresh: bool = True,
    ) -> dict[str, Any]:
        """Return raw active-address on-chain balance for one user asset."""

        asset_key = self._asset_key(asset)
        network_name = str(network or "").strip() or self._default_network(asset_key)
        addresses = (
            WalletAddress.query.filter_by(user_id=user_id, asset=asset_key, network=network_name, status="active")
            .order_by(WalletAddress.rotation_index.desc(), WalletAddress.created_at.desc(), WalletAddress.id.desc())
            .all()
        )
        if not self.enabled:
            return self._empty_onchain_payload(user_id, asset_key, network_name, "disabled", "Real wallet custody is disabled.")
        adapter = self._adapter_for(asset_key, network_name)
        if adapter is None:
            return self._empty_onchain_payload(user_id, asset_key, network_name, "unsupported", "No custody adapter supports this asset/network.")
        if not addresses:
            return self._empty_onchain_payload(user_id, asset_key, network_name, "no_active_address", "No active wallet address is available.")

        checked_at = datetime.utcnow()
        rows: list[dict[str, Any]] = []
        for source in addresses:
            if refresh:
                snapshot = adapter.get_balance(source.address, asset_key, network_name)
                self._persist_onchain_snapshot(source, snapshot, checked_at=checked_at)
            status = str(source.onchain_status or "unknown")
            checked = status == "checked"
            amount = max(0.0, float(source.onchain_balance or 0.0)) if checked else 0.0
            rows.append(
                {
                    "wallet_address_id": source.id,
                    "address": source.address,
                    "amount": amount,
                    "asset": asset_key,
                    "network": network_name,
                    "checked": checked,
                    "status": status,
                    "reason": str(source.onchain_reason or ""),
                    "confirmations": int(source.onchain_confirmations or 0),
                    "provider_reference": str(source.onchain_provider_reference or ""),
                    "checked_at": source.onchain_checked_at.isoformat() if source.onchain_checked_at else None,
                }
            )
        db.session.flush()

        checked_rows = [row for row in rows if row["checked"]]
        amount = sum(float(row["amount"] or 0.0) for row in checked_rows)
        confirmations = max((int(row["confirmations"] or 0) for row in checked_rows), default=0)
        latest_checked_at = max(
            (source.onchain_checked_at for source in addresses if source.onchain_checked_at is not None),
            default=None,
        )
        status = "checked" if checked_rows else "unavailable"
        reasons = [str(row.get("reason") or row.get("status") or "") for row in rows if row.get("reason") or row.get("status")]
        return {
            "user_id": user_id,
            "asset": asset_key,
            "network": network_name,
            "amount": amount,
            "checked": bool(checked_rows),
            "status": status,
            "reason": "; ".join(dict.fromkeys(reasons)),
            "confirmations": confirmations,
            "checked_at": latest_checked_at.isoformat() if latest_checked_at else None,
            "addresses": rows,
        }

    def verified_spendable_amount(self, user_id: int, asset: str, network: str) -> float:
        """Return custody-verified spendable funds after active locks/reservations."""

        asset_key = self._asset_key(asset)
        network_name = str(network or "").strip() or self._default_network(asset_key)
        network_key = self._network_key(network_name)
        if not self.enabled or self._adapter_for(asset_key, network_name) is None:
            return 0.0

        payload = self.onchain_balance_for_user_asset(user_id, asset_key, network_name)
        if not bool(payload.get("checked")):
            return 0.0

        adapter = self._require_adapter(asset_key, network_name)
        gas_adapter = self._adapter_for("ETH", network_name)
        required = int(self._duration_map_value("WALLET_REQUIRED_CONFIRMATIONS", network_name, 1))
        treasury_ready = self._treasury_ready_for_token_gas()
        max_source_amount = 0.0

        for row in list(payload.get("addresses") or []):
            if not bool(row.get("checked")) or int(row.get("confirmations") or 0) < required:
                continue
            amount = max(0.0, float(row.get("amount") or 0.0))
            address = str(row.get("address") or "")
            if network_key in EVM_NETWORKS and asset_key in EVM_ASSETS:
                gas_fee = max(0.0, float(adapter.estimate_fee(asset_key, network_name, address, amount) or 0.0))
                if asset_key == "ETH":
                    amount = max(0.0, amount - gas_fee)
                else:
                    gas_snapshot = gas_adapter.get_balance(address, "ETH", network_name) if gas_adapter is not None else None
                    gas_ok = bool(
                        gas_snapshot is not None
                        and gas_snapshot.checked
                        and float(gas_snapshot.amount or 0.0) + 1e-18 >= gas_fee
                    )
                    if not gas_ok and not treasury_ready:
                        amount = 0.0
            max_source_amount = max(max_source_amount, amount)

        reserved = self._reserved_amount(user_id, asset_key)
        return max(0.0, max_source_amount - reserved)

    def materialize_onchain_surplus(self, user_id: int, asset: str, network: str, required_amount: float) -> dict[str, Any]:
        """Credit verified on-chain surplus into app accounting before spending it."""

        asset_key = self._asset_key(asset)
        network_name = str(network or "").strip() or self._default_network(asset_key)
        required = max(0.0, float(required_amount or 0.0))
        if required <= 0:
            return {"credited": 0.0, "required_amount": required, "changed": False}
        if not self.enabled or self._adapter_for(asset_key, network_name) is None:
            return {"credited": 0.0, "required_amount": required, "changed": False, "skipped": "custody_unavailable"}

        spendable = self.verified_spendable_amount(user_id, asset_key, network_name)
        if spendable + 1e-9 < required:
            raise WalletCustodyError(
                f"Verified on-chain {asset_key} balance is {spendable:.6f}, below the requested {required:.6f}.",
                code="onchain_balance_insufficient",
                context={"asset": asset_key, "network": network_name, "spendable": spendable, "required_amount": required},
            )

        balance = WalletBalance.query.filter_by(user_id=user_id, asset=asset_key).one_or_none()
        if balance is None:
            balance = WalletBalance(user_id=user_id, asset=asset_key)
            db.session.add(balance)
            db.session.flush()

        available_before = max(0.0, float(balance.available_balance or 0.0))
        locked = max(0.0, float(balance.locked_balance or 0.0))
        if available_before + 1e-9 >= required:
            return {
                "credited": 0.0,
                "required_amount": required,
                "available_before": available_before,
                "available_after": available_before,
                "spendable": spendable,
                "changed": False,
            }

        onchain = self.onchain_balance_for_user_asset(user_id, asset_key, network_name, refresh=False)
        onchain_amount = max(0.0, float(onchain.get("amount") or 0.0)) if bool(onchain.get("checked")) else 0.0
        app_total = available_before + locked
        surplus = max(0.0, onchain_amount - app_total)
        needed = max(0.0, required - available_before)
        credit = min(surplus, needed)
        if credit <= 1e-12 or available_before + credit + 1e-9 < required:
            raise WalletCustodyError(
                "Verified on-chain funds exceed app available funds, but no unreserved surplus can be credited.",
                code="onchain_surplus_unavailable",
                context={
                    "asset": asset_key,
                    "network": network_name,
                    "onchain_balance": onchain_amount,
                    "available_balance": available_before,
                    "locked_balance": locked,
                    "required_amount": required,
                    "spendable": spendable,
                },
            )

        balance.available_balance = available_before + credit
        if asset_key in {"USDC", "USDT", "USD"}:
            balance.estimated_usd_value = float(balance.available_balance or 0.0) + float(balance.locked_balance or 0.0)
        transaction = WalletTransaction(
            user_id=user_id,
            asset=asset_key,
            amount=credit,
            transaction_type="onchain_reconciliation",
            status="complete",
            network=network_name,
            note="Credited verified on-chain surplus before wallet operation.",
        )
        db.session.add(transaction)
        self._audit(
            user_id=user_id,
            action="onchain_surplus_materialized",
            status="complete",
            message=f"Credited {credit:.8f} {asset_key} from verified on-chain surplus.",
            metadata={
                "asset": asset_key,
                "network": network_name,
                "credited": credit,
                "required_amount": required,
                "available_before": available_before,
                "available_after": balance.available_balance,
                "locked_balance": locked,
                "onchain_balance": onchain_amount,
                "spendable": spendable,
            },
        )
        db.session.flush()
        return {
            "credited": credit,
            "required_amount": required,
            "available_before": available_before,
            "available_after": float(balance.available_balance or 0.0),
            "spendable": spendable,
            "onchain_balance": onchain_amount,
            "changed": True,
        }

    def recover_evm_token_deposit(
        self,
        *,
        user_id: int,
        asset: str,
        address: str,
        tx_hash: str,
        confirm: bool = False,
    ) -> dict[str, Any]:
        asset_key = self._asset_key(asset)
        network = "Ethereum"
        address_value = str(address or "").strip()
        tx_hash_value = str(tx_hash or "").strip()
        blockers: list[str] = []

        if asset_key not in {"USDC", "USDT"}:
            blockers.append("Only supported ERC-20 recovery assets are allowed: USDC, USDT")
        if not re.fullmatch(r"0x[a-fA-F0-9]{40}", address_value):
            blockers.append("Recovery address must be a valid EVM 0x address")
        if not tx_hash_value:
            blockers.append("Transaction hash is required")
        if not self.enabled:
            blockers.append("Real wallet custody is disabled")
        if not self._has_valid_encryption_key():
            blockers.append("TOTP_ENCRYPTION_KEY must be a valid Fernet key")
        if not self._adapter_for(asset_key, network):
            blockers.append("No EVM adapter supports the requested token")
        if not EvmWalletAdapter(self.config)._rpc_url(network):
            blockers.append("EVM RPC URL is not configured")
        if asset_key in {"USDC", "USDT"} and not EvmWalletAdapter(self.config)._token_contract(asset_key, network):
            blockers.append(f"{asset_key} token contract is not configured")

        source = self._recoverable_evm_source(user_id, address_value, asset_key)
        if source is None:
            blockers.append("Address is not an existing generated in-app EVM wallet for this user")
        else:
            try:
                self._private_key(source)
            except Exception:  # noqa: BLE001
                blockers.append("Source wallet private key is unavailable or cannot be decrypted")

        duplicate = self._wallet_address_by_address(user_id, asset_key, network, address_value, status="active")
        reusable_recovery = self._is_matching_recovery(duplicate, source, tx_hash_value) if duplicate is not None else False
        if duplicate is not None and not reusable_recovery:
            blockers.append(f"Active {asset_key}/{network} wallet address already exists for this address")

        payload: dict[str, Any] = {
            "preview_only": not confirm,
            "ready": not blockers,
            "recovered": False,
            "asset": asset_key,
            "network": network,
            "address": address_value,
            "tx_hash": tx_hash_value,
            "blockers": list(dict.fromkeys(blockers)),
            "source_wallet_address_id": source.id if source is not None else None,
            "source_deposit_address_id": source.deposit_address_id if source is not None else None,
            "existing_recovery_wallet_address_id": duplicate.id if duplicate is not None and reusable_recovery else None,
            "sync": {},
        }

        self._audit(
            user_id=user_id,
            wallet_account_id=source.wallet_account_id if source is not None else None,
            action="recover_evm_token_deposit_preview" if not confirm else "recover_evm_token_deposit_validation",
            status="ready" if not blockers else "blocked",
            message=f"ERC-20 {asset_key} recovery validation for {address_value}.",
            metadata=payload,
        )
        if blockers or not confirm:
            return payload

        if duplicate is not None and reusable_recovery:
            recovered = duplicate
            created = False
        else:
            assert source is not None
            recovered = self._create_recovered_evm_token_wallet(
                source=source,
                asset=asset_key,
                network=network,
                address=address_value,
                tx_hash=tx_hash_value,
            )
            created = True

        sync_result = self.sync_address(recovered)
        if not bool(sync_result.get("checked", False)):
            self._audit(
                user_id=user_id,
                wallet_account_id=recovered.wallet_account_id,
                action="recover_evm_token_deposit_sync_failed",
                status="failed",
                message=f"Recovered {asset_key} wallet linked, but token balance sync failed.",
                metadata={**payload, "wallet_address_id": recovered.id, "sync": sync_result},
            )
        else:
            self._audit(
                user_id=user_id,
                wallet_account_id=recovered.wallet_account_id,
                action="recover_evm_token_deposit_complete",
                status="complete",
                message=f"Recovered {asset_key} ERC-20 deposit tracking for {address_value}.",
                metadata={**payload, "wallet_address_id": recovered.id, "sync": sync_result},
            )

        payload.update(
            {
                "recovered": True,
                "created_wallet_address": created,
                "wallet_address_id": recovered.id,
                "wallet_account_id": recovered.wallet_account_id,
                "sync": sync_result,
                "credited": float(sync_result.get("credited", 0.0) or 0.0),
            }
        )
        return payload

    def sign_and_broadcast(self, withdrawal: WalletWithdrawal, *, mode: str) -> BroadcastResult:
        if str(mode or "").lower() != "live":
            raise WalletCustodyError("Real wallet withdrawals can only be broadcast in live mode.", code="wallet_live_mode_required")
        if not self.enabled:
            raise WalletCustodyError("Real wallet custody is disabled.", code="wallet_custody_disabled")
        preflight = self.withdrawal_preflight(withdrawal)
        if preflight.get("blockers") and not preflight.get("gas_topup_required"):
            blockers = [str(item) for item in preflight["blockers"]]
            raise WalletCustodyError("; ".join(blockers), code="wallet_preflight_blocked", context={"blockers": blockers})
        if preflight.get("gas_topup_required"):
            raise WalletCustodyError(
                "source wallet has insufficient ETH for token transfer gas",
                code="wallet_gas_topup_required",
            )
        wallet_address = self._withdrawal_source(withdrawal)
        adapter = self._require_adapter(withdrawal.asset, withdrawal.network)
        private_key = self._private_key(wallet_address)
        try:
            return adapter.sign_and_broadcast(withdrawal, private_key)
        except WalletCustodyError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise WalletBroadcastError(
                "Wallet broadcast failed through custody adapter.",
                code="wallet_broadcast_failed",
                context={"withdrawal_id": withdrawal.id, "asset": withdrawal.asset, "network": withdrawal.network},
            ) from exc

    def release_failed_withdrawal(self, withdrawal: WalletWithdrawal) -> None:
        balance = WalletBalance.query.filter_by(user_id=withdrawal.user_id, asset=withdrawal.asset).one_or_none()
        if balance is None:
            return
        amount = float(withdrawal.amount or 0.0)
        locked = max(0.0, float(balance.locked_balance or 0.0))
        released = min(locked, amount)
        balance.locked_balance = max(0.0, locked - released)
        balance.available_balance = float(balance.available_balance or 0.0) + released
        if withdrawal.asset in {"USDC", "USDT", "USD"}:
            balance.estimated_usd_value = float(balance.available_balance or 0.0) + float(balance.locked_balance or 0.0)

    def complete_withdrawal_lock(self, withdrawal: WalletWithdrawal) -> None:
        balance = WalletBalance.query.filter_by(user_id=withdrawal.user_id, asset=withdrawal.asset).one_or_none()
        if balance is None:
            return
        locked = max(0.0, float(balance.locked_balance or 0.0))
        balance.locked_balance = max(0.0, locked - min(locked, float(withdrawal.amount or 0.0)))
        if withdrawal.asset in {"USDC", "USDT", "USD"}:
            balance.estimated_usd_value = float(balance.available_balance or 0.0) + float(balance.locked_balance or 0.0)

    def withdrawal_preflight(self, withdrawal: WalletWithdrawal) -> dict[str, Any]:
        """Select a source wallet and verify EVM token/gas readiness before signing."""

        asset = self._asset_key(withdrawal.asset)
        network = str(withdrawal.network or "").strip() or "native"
        network_key = self._network_key(network)
        if network_key not in EVM_NETWORKS or asset not in EVM_ASSETS:
            return {"ready": True, "blockers": [], "source_wallet_address_id": withdrawal.source_wallet_address_id}

        adapter = self._require_adapter(asset, network)
        gas_adapter = self._require_adapter("ETH", network)
        amount = max(0.0, float(withdrawal.amount or 0.0))
        candidates = self._withdrawal_source_candidates(withdrawal)
        gas_fee = max(0.0, float(adapter.estimate_fee(asset, network, withdrawal.destination_address, amount) or 0.0))
        rows: list[dict[str, Any]] = []
        selected: WalletAddress | None = None
        selected_needs_gas = False

        for candidate in candidates:
            token_snapshot = adapter.get_balance(candidate.address, asset, network)
            token_balance = max(0.0, float(token_snapshot.amount or 0.0)) if token_snapshot.checked else 0.0
            gas_snapshot = gas_adapter.get_balance(candidate.address, "ETH", network)
            eth_balance = max(0.0, float(gas_snapshot.amount or 0.0)) if gas_snapshot.checked else 0.0
            has_token = token_balance + 1e-12 >= amount
            gas_needed = gas_fee if asset != "ETH" else amount + gas_fee
            has_gas = eth_balance + 1e-18 >= gas_needed
            row = {
                "wallet_address_id": candidate.id,
                "address": candidate.address,
                "token_balance": token_balance,
                "eth_balance": eth_balance,
                "estimated_gas_eth": gas_fee,
                "has_token": has_token,
                "has_gas": has_gas,
                "token_checked": bool(token_snapshot.checked),
                "gas_checked": bool(gas_snapshot.checked),
                "token_reason": token_snapshot.reason,
                "gas_reason": gas_snapshot.reason,
            }
            rows.append(row)
            if has_token and has_gas:
                selected = candidate
                selected_needs_gas = False
                break
            if has_token and selected is None:
                selected = candidate
                selected_needs_gas = asset != "ETH"

        blockers: list[str] = []
        if selected is None:
            blockers.append(f"no active {asset} wallet has enough on-chain balance for withdrawal")
        elif selected_needs_gas:
            blockers.append("source wallet has insufficient ETH for token transfer gas")
        if selected is not None:
            withdrawal.source_wallet_address_id = selected.id
        return {
            "ready": selected is not None and not selected_needs_gas,
            "gas_topup_required": bool(selected is not None and selected_needs_gas),
            "blockers": blockers,
            "source_wallet_address_id": selected.id if selected is not None else None,
            "source_address": selected.address if selected is not None else "",
            "estimated_gas_eth": gas_fee,
            "candidates": rows,
        }

    def verified_withdrawable_amount(self, user_id: int, asset: str, network: str, local_available: float) -> float:
        asset_key = self._asset_key(asset)
        network_name = str(network or "").strip() or self._default_network(asset_key)
        app_available = max(0.0, float(local_available or 0.0))
        if not self.enabled or self._adapter_for(asset_key, network_name) is None:
            return app_available
        return self.verified_spendable_amount(user_id, asset_key, network_name)

    def reconcile_withdrawal(self, withdrawal: WalletWithdrawal, *, commit: bool = False) -> dict[str, Any]:
        if withdrawal.status in {"complete", "failed", "rejected"}:
            return {"withdrawal_id": withdrawal.id, "status": withdrawal.status, "changed": False, "terminal": True}
        if withdrawal.status == "pending_gas_topup":
            return {"withdrawal_id": withdrawal.id, "status": withdrawal.status, "changed": False, "pending_gas_topup": True}
        if withdrawal.status != "submitted" or not withdrawal.provider_reference:
            return {"withdrawal_id": withdrawal.id, "status": withdrawal.status, "changed": False, "skipped": True}
        adapter = self._require_adapter(withdrawal.asset, withdrawal.network)
        confirmation = adapter.confirm_transaction(withdrawal.provider_reference, withdrawal.asset, withdrawal.network)
        receipt = confirmation.get("raw") if isinstance(confirmation, dict) else None
        if not receipt:
            return {"withdrawal_id": withdrawal.id, "status": withdrawal.status, "changed": False, "receipt": None}
        details = withdrawal.details
        details["reconciled_onchain"] = {"receipt": receipt}
        result = {
            "withdrawal_id": withdrawal.id,
            "tx_hash": withdrawal.provider_reference,
            "receipt_status": receipt.get("status"),
            "changed": False,
            "status": withdrawal.status,
        }
        if receipt.get("status") == "0x1" and self._receipt_matches_withdrawal(withdrawal, receipt):
            result["status"] = "complete"
            result["changed"] = withdrawal.status != "complete"
            if commit:
                withdrawal.status = "complete"
                withdrawal.completed_at = datetime.utcnow()
                details["reconciled_onchain"].update({"status": "complete"})
                withdrawal.details = details
                self.complete_withdrawal_lock(withdrawal)
                self._update_withdrawal_transaction(withdrawal, "complete", f"Withdrawal workflow {withdrawal.id} confirmed on-chain.")
                if has_app_context():
                    solvency = current_app.extensions.get("services", {}).get("treasury_solvency")
                    if solvency is not None:
                        solvency.record_withdrawal_gas_usage(withdrawal, withdrawal.details.get("provider_response", {}), receipt=receipt)
                self._audit(
                    user_id=withdrawal.user_id,
                    wallet_withdrawal_id=withdrawal.id,
                    action="withdrawal_reconciled_complete",
                    status="complete",
                    message="Withdrawal confirmed on-chain during reconciliation.",
                    metadata=details["reconciled_onchain"],
                )
            return result
        if receipt.get("status") == "0x0":
            result["status"] = "failed"
            result["changed"] = withdrawal.status != "failed"
            if commit:
                withdrawal.status = "failed"
                withdrawal.failure_reason = "On-chain receipt status 0x0; no matching transfer logs. Funds unlocked during reconciliation."
                withdrawal.completed_at = datetime.utcnow()
                details["reconciled_onchain"].update({"status": "failed", "reason": withdrawal.failure_reason})
                withdrawal.details = details
                self.release_failed_withdrawal(withdrawal)
                self._update_withdrawal_transaction(withdrawal, "failed", withdrawal.failure_reason)
                self._audit(
                    user_id=withdrawal.user_id,
                    wallet_withdrawal_id=withdrawal.id,
                    action="withdrawal_reconciled_failed",
                    status="failed",
                    message=withdrawal.failure_reason,
                    metadata=details["reconciled_onchain"],
                )
            return result
        result["matched_transfer"] = self._receipt_matches_withdrawal(withdrawal, receipt)
        if commit:
            withdrawal.details = details
        return result

    def reconcile_custody_balance(self, user_id: int, asset: str) -> dict[str, Any]:
        """Repair the custody balance row from completed custody ledger activity."""
        asset_key = self._asset_key(asset)
        balance = WalletBalance.query.filter_by(user_id=user_id, asset=asset_key).one_or_none()
        if balance is None:
            balance = WalletBalance(user_id=user_id, asset=asset_key)
            db.session.add(balance)
            db.session.flush()

        deposit_total = sum(
            float(event.amount or 0.0)
            for event in WalletLedgerEvent.query.filter_by(
                user_id=user_id,
                asset=asset_key,
                event_type="deposit",
                status="complete",
            ).all()
        )
        completed_withdrawal_total = sum(
            float(withdrawal.amount or 0.0)
            for withdrawal in WalletWithdrawal.query.filter_by(
                user_id=user_id,
                asset=asset_key,
                workflow_type="manual_withdrawal",
                status="complete",
            ).all()
        )
        locked = max(0.0, float(balance.locked_balance or 0.0))
        expected_available = max(0.0, deposit_total - completed_withdrawal_total - locked)
        before = float(balance.available_balance or 0.0)
        balance.available_balance = expected_available
        if asset_key in {"USDC", "USDT", "USD"}:
            balance.estimated_usd_value = expected_available + locked
        self._audit(
            user_id=user_id,
            action="wallet_balance_reconciled",
            status="complete",
            message=f"Reconciled {asset_key} custody balance from completed wallet ledger events.",
            metadata={
                "asset": asset_key,
                "before_available": before,
                "after_available": expected_available,
                "deposit_total": deposit_total,
                "completed_withdrawal_total": completed_withdrawal_total,
                "locked_balance": locked,
            },
        )
        db.session.flush()
        return {
            "user_id": user_id,
            "asset": asset_key,
            "before_available": before,
            "available_balance": expected_available,
            "locked_balance": locked,
            "deposit_total": deposit_total,
            "completed_withdrawal_total": completed_withdrawal_total,
            "changed": not math.isclose(before, expected_available, rel_tol=1e-12, abs_tol=1e-12),
        }

    @staticmethod
    def _empty_onchain_payload(user_id: int, asset: str, network: str, status: str, reason: str) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "asset": asset,
            "network": network,
            "amount": 0.0,
            "checked": False,
            "status": status,
            "reason": reason,
            "confirmations": 0,
            "checked_at": None,
            "addresses": [],
        }

    def _persist_onchain_snapshot(
        self,
        wallet_address: WalletAddress,
        snapshot: WalletBalanceSnapshot,
        *,
        checked_at: datetime | None = None,
    ) -> None:
        timestamp = checked_at or datetime.utcnow()
        status = "checked" if snapshot.checked else "failed"
        amount = max(0.0, float(snapshot.amount or 0.0)) if snapshot.checked else 0.0
        wallet_address.onchain_balance = amount
        wallet_address.onchain_checked_at = timestamp
        wallet_address.onchain_status = status
        wallet_address.onchain_reason = str(snapshot.reason or "")
        wallet_address.onchain_confirmations = int(snapshot.confirmations or 0)
        wallet_address.onchain_provider_reference = str(snapshot.provider_reference or "")[:220]
        metadata = wallet_address.encrypted_metadata
        metadata["last_sync_status"] = status
        metadata["last_sync_reason"] = snapshot.reason
        metadata["last_onchain_balance"] = amount
        metadata["last_onchain_checked_at"] = timestamp.isoformat() + "Z"
        metadata["last_onchain_confirmations"] = int(snapshot.confirmations or 0)
        metadata["last_onchain_provider_reference"] = str(snapshot.provider_reference or "")
        wallet_address.encrypted_metadata = metadata

    def _reserved_amount(self, user_id: int, asset: str) -> float:
        balance = WalletBalance.query.filter_by(user_id=user_id, asset=asset).one_or_none()
        locked = max(0.0, float(balance.locked_balance or 0.0)) if balance is not None else 0.0
        pending = sum(
            float(withdrawal.amount or 0.0)
            for withdrawal in WalletWithdrawal.query.filter(
                WalletWithdrawal.user_id == user_id,
                WalletWithdrawal.asset == asset,
                WalletWithdrawal.status.in_(ACTIVE_WITHDRAWAL_RESERVE_STATUSES),
            ).all()
        )
        return max(locked, pending)

    @staticmethod
    def _treasury_ready_for_token_gas() -> bool:
        if not has_app_context():
            return False
        try:
            treasury_status = current_app.extensions["services"]["platform_treasury"].status(include_events=False)
            ready = bool(treasury_status.get("ready"))
            solvency_state = ((treasury_status.get("solvency") or {}).get("state") or {})
            if solvency_state and str(solvency_state.get("health_status") or "") in {"critical", "emergency"}:
                return False
            return ready
        except Exception:
            return False

    def _withdrawal_source(self, withdrawal: WalletWithdrawal) -> WalletAddress:
        candidates = self._withdrawal_source_candidates(withdrawal)
        source = candidates[0] if candidates else None
        if source is None:
            raise RuntimeError("No active source wallet address is available for withdrawal.")
        return source

    def _withdrawal_source_candidates(self, withdrawal: WalletWithdrawal) -> list[WalletAddress]:
        if withdrawal.source_wallet_address_id:
            source = db.session.get(WalletAddress, int(withdrawal.source_wallet_address_id))
            return [source] if source is not None else []
        return (
            WalletAddress.query.filter_by(
                user_id=withdrawal.user_id,
                asset=withdrawal.asset,
                network=withdrawal.network,
                status="active",
            )
            .order_by(WalletAddress.rotation_index.desc(), WalletAddress.created_at.desc(), WalletAddress.id.desc())
            .all()
        )

    def _receipt_matches_withdrawal(self, withdrawal: WalletWithdrawal, receipt: dict[str, Any]) -> bool:
        asset = self._asset_key(withdrawal.asset)
        if asset == "ETH":
            tx = EvmWalletAdapter(self.config)._rpc("eth_getTransactionByHash", [withdrawal.provider_reference], network=withdrawal.network)  # noqa: SLF001
            if not isinstance(tx, dict):
                return False
            if str(tx.get("to") or "").lower() != str(withdrawal.destination_address or "").lower():
                return False
            value = int(str(tx.get("value") or "0x0"), 16) / 10**18
            return value + 1e-12 >= float(withdrawal.amount or 0.0)
        adapter = EvmWalletAdapter(self.config)
        contract = adapter._token_contract(asset, withdrawal.network).lower()  # noqa: SLF001
        decimals = adapter._token_decimals(asset, withdrawal.network)  # noqa: SLF001
        amount_units = _decimal_units(withdrawal.amount, decimals)
        source = self._withdrawal_source(withdrawal)
        source_topic = "0x" + source.address.lower().replace("0x", "").rjust(64, "0")
        destination_topic = "0x" + withdrawal.destination_address.lower().replace("0x", "").rjust(64, "0")
        for log in receipt.get("logs") or []:
            if str(log.get("address") or "").lower() != contract:
                continue
            topics = [str(topic).lower() for topic in (log.get("topics") or [])]
            if len(topics) < 3 or topics[0] != ERC20_TRANSFER_TOPIC:
                continue
            if topics[1] != source_topic or topics[2] != destination_topic:
                continue
            transferred = int(str(log.get("data") or "0x0"), 16)
            if transferred >= amount_units:
                return True
        return False

    def _update_withdrawal_transaction(self, withdrawal: WalletWithdrawal, status: str, note: str) -> None:
        transaction = (
            WalletTransaction.query.filter(
                WalletTransaction.user_id == withdrawal.user_id,
                WalletTransaction.asset == withdrawal.asset,
                WalletTransaction.transaction_type == "withdrawal",
                WalletTransaction.note.like(f"%Withdrawal workflow {withdrawal.id}:%"),
            )
            .order_by(WalletTransaction.created_at.desc())
            .first()
        )
        if transaction is None:
            return
        transaction.status = status
        transaction.note = note

    def _recoverable_evm_source(self, user_id: int, address: str, target_asset: str) -> WalletAddress | None:
        matches = [
            row
            for row in WalletAddress.query.filter_by(user_id=user_id, network="Ethereum").order_by(WalletAddress.id.desc()).all()
            if str(row.address or "").lower() == str(address or "").lower()
        ]
        for row in matches:
            metadata = row.encrypted_metadata
            if row.asset == target_asset:
                continue
            if metadata.get("custody") == "in_app":
                return row
        return None

    def _wallet_address_by_address(
        self,
        user_id: int,
        asset: str,
        network: str,
        address: str,
        *,
        status: str | None = None,
    ) -> WalletAddress | None:
        query = WalletAddress.query.filter_by(user_id=user_id, asset=asset, network=network)
        if status is not None:
            query = query.filter_by(status=status)
        for row in query.order_by(WalletAddress.id.desc()).all():
            if str(row.address or "").lower() == str(address or "").lower():
                return row
        return None

    @staticmethod
    def _is_matching_recovery(existing: WalletAddress | None, source: WalletAddress | None, tx_hash: str) -> bool:
        if existing is None or source is None:
            return False
        recovery = (existing.encrypted_metadata or {}).get("recovery") or {}
        return (
            recovery.get("type") == "evm_token_sent_to_existing_address"
            and int(recovery.get("source_wallet_address_id") or 0) == int(source.id)
            and str(recovery.get("tx_hash") or "").lower() == str(tx_hash or "").lower()
        )

    def _create_recovered_evm_token_wallet(
        self,
        *,
        source: WalletAddress,
        asset: str,
        network: str,
        address: str,
        tx_hash: str,
    ) -> WalletAddress:
        source_metadata = dict(source.encrypted_metadata or {})
        account = self._account_for(source.user_id, asset, network)
        latest = (
            WalletAddress.query.filter_by(user_id=source.user_id, asset=asset, network=network)
            .order_by(WalletAddress.rotation_index.desc())
            .first()
        )
        recovered = WalletAddress(
            wallet_account_id=account.id,
            user_id=source.user_id,
            asset=asset,
            network=network,
            address=address,
            status="active",
            rotation_index=(latest.rotation_index if latest else 0) + 1,
            rotated_from_id=source.id,
        )
        recovered.encrypted_metadata = {
            "custody": "in_app",
            "provider": source_metadata.get("provider", "evm"),
            "key_type": source_metadata.get("key_type", "secp256k1"),
            "public_key": source_metadata.get("public_key", address),
            "encrypted_private_key": source_metadata.get("encrypted_private_key", ""),
            "sync_status": "not_synced",
            "last_sync_cursor": "",
            "recovery": {
                "type": "evm_token_sent_to_existing_address",
                "source_wallet_address_id": source.id,
                "source_wallet_asset": source.asset,
                "source_deposit_address_id": source.deposit_address_id,
                "tx_hash": tx_hash,
            },
        }
        db.session.add(recovered)
        db.session.flush()
        self._audit(
            user_id=source.user_id,
            wallet_account_id=account.id,
            action="recover_evm_token_wallet_linked",
            status="active",
            message=f"Linked recovered {asset} wallet tracking to existing EVM custody address.",
            metadata={
                "asset": asset,
                "network": network,
                "address": address,
                "source_wallet_address_id": source.id,
                "source_deposit_address_id": source.deposit_address_id,
                "tx_hash": tx_hash,
            },
        )
        return recovered

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
        return True

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
            confirmation_map = self.config.get("WALLET_REQUIRED_CONFIRMATIONS", {})
            network_key = _network_key(network)
            confirmations = 1
            if isinstance(confirmation_map, dict):
                confirmations = int(
                    confirmation_map.get(network_key)
                    or confirmation_map.get(network_key.lower())
                    or confirmation_map.get(network)
                    or confirmation_map.get(str(network).lower())
                    or 1
                )
            return WalletBalanceSnapshot(
                amount=amount,
                asset=asset_key,
                checked=True,
                confirmations=max(confirmations, 1),
                provider_reference=f"evm-balance:{_network_key(network)}:{address}:{asset_key}:{amount:.12f}:{block}",
                metadata={"block": block},
            )
        except Exception as exc:  # noqa: BLE001
            return WalletBalanceSnapshot(0.0, asset_key, False, reason=f"EVM balance check failed: {exc}")

    def estimate_fee(self, asset: str, network: str, destination: str, amount: float) -> float:
        asset_key = _asset_key(asset)
        gas = self._estimated_gas_limit(asset_key, network, destination, amount)
        gas_price = self._gas_price_wei(network)
        return gas * gas_price / 10**18

    def sign_and_broadcast(self, withdrawal: WalletWithdrawal, private_key: str) -> BroadcastResult:
        network = withdrawal.network
        asset = _asset_key(withdrawal.asset)
        source = current_app.extensions["services"]["wallet_custody"]._withdrawal_source(withdrawal) if has_app_context() else None
        from_address = source.address if source is not None else Account.from_key(private_key).address
        rpc_url = self._rpc_url(network)
        nonce = int(str(self._rpc("eth_getTransactionCount", [from_address, "pending"], network=network) or "0x0"), 16)
        gas_price_payload = self._gas_price_payload(network)
        rpc_gas_price = int(gas_price_payload.get("rpc_gas_price_wei", 0) or 0)
        gas_price = int(gas_price_payload.get("gas_price_wei", 0) or 0)
        chain_id = int(self._chain_config(network).get("chain_id", 1) or 1)
        eth_balance_wei = int(str(self._rpc("eth_getBalance", [from_address, "pending"], network=network) or "0x0"), 16)
        token_contract = ""
        amount_units = 0
        if asset == "ETH":
            value = int(float(withdrawal.amount or 0.0) * 10**18)
            gas_limit = self._estimated_gas_limit(asset, network, withdrawal.destination_address, withdrawal.amount, from_address=from_address)
            required_wei = value + gas_limit * gas_price
            if eth_balance_wei < required_wei:
                raise RuntimeError(
                    "source wallet has insufficient ETH for withdrawal amount plus gas"
                )
            tx = {
                "chainId": chain_id,
                "nonce": nonce,
                "to": withdrawal.destination_address,
                "value": value,
                "gas": gas_limit,
                "gasPrice": gas_price,
                "data": b"",
            }
        else:
            contract = self._token_contract(asset, network)
            if not contract:
                raise RuntimeError(f"{asset} token contract is not configured")
            token_contract = contract
            decimals = self._token_decimals(asset, network)
            amount_units = _decimal_units(withdrawal.amount, decimals)
            gas_limit = self._estimated_gas_limit(asset, network, withdrawal.destination_address, withdrawal.amount, from_address=from_address)
            required_gas_wei = gas_limit * gas_price
            if eth_balance_wei < required_gas_wei:
                raise RuntimeError(
                    f"source wallet has insufficient ETH for {asset} token transfer gas"
                )
            token_balance_units = self._token_balance_units(from_address, asset, network)
            if token_balance_units < amount_units:
                raise RuntimeError(f"source wallet has insufficient {asset} token balance")
            data = "0xa9059cbb" + withdrawal.destination_address.lower().replace("0x", "").rjust(64, "0") + hex(amount_units)[2:].rjust(64, "0")
            tx = {
                "chainId": chain_id,
                "nonce": nonce,
                "to": contract,
                "value": 0,
                "gas": gas_limit,
                "gasPrice": gas_price,
                "data": data,
            }
        signed = Account.sign_transaction(tx, private_key)
        raw_tx = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
        raw_hex = raw_tx.hex()
        if not raw_hex.startswith("0x"):
            raw_hex = "0x" + raw_hex
        tx_hash = self._rpc("eth_sendRawTransaction", [raw_hex], network=network)
        tx_hash_value = str(tx_hash or "")
        tx_found = self._transaction_visible(tx_hash_value, network)
        raw = {
            "tx_hash": tx_hash_value,
            "rpc_url": rpc_url,
            "nonce": nonce,
            "rpc_gas_price_wei": rpc_gas_price,
            "gas_price_wei": gas_price,
            "fee_source": gas_price_payload.get("fee_source", "eth_gasPrice"),
            "gas_limit": int(tx.get("gas", 0) or 0),
            "source": from_address,
            "destination": withdrawal.destination_address,
            "asset": asset,
            "token_contract": token_contract,
            "amount_units": amount_units if asset != "ETH" else value,
            "raw_transaction": raw_hex,
            "broadcast_visible": tx_found,
        }
        if tx_hash_value and not tx_found:
            return BroadcastResult("failed_broadcast_not_found", tx_hash_value, raw)
        return BroadcastResult("submitted", tx_hash_value, raw)

    def confirm_transaction(self, provider_reference: str, asset: str, network: str) -> dict[str, Any]:
        receipt = self._rpc("eth_getTransactionReceipt", [provider_reference], network=network)
        return {"confirmed": bool(receipt and receipt.get("status") == "0x1"), "raw": receipt}

    def _transaction_visible(self, tx_hash: str, network: str) -> bool:
        if not tx_hash:
            return False
        attempts = max(1, int(float(self.config.get("WALLET_BROADCAST_VERIFY_ATTEMPTS", 3) or 3)))
        delay_seconds = max(0.0, float(self.config.get("WALLET_BROADCAST_VERIFY_DELAY_SECONDS", 0.5) or 0.5))
        for attempt in range(attempts):
            try:
                if self._rpc("eth_getTransactionByHash", [tx_hash], network=network):
                    return True
            except Exception:
                pass
            if attempt < attempts - 1 and delay_seconds > 0:
                time.sleep(delay_seconds)
        return False

    def _minimum_gas_price_wei(self) -> int:
        gwei = max(0.0, float(self.config.get("WALLET_EVM_MIN_GAS_PRICE_GWEI", 2.0) or 0.0))
        return int(gwei * 10**9)

    def _gas_price_wei(self, network: str) -> int:
        return int(self._gas_price_payload(network).get("gas_price_wei", 0) or 0)

    def _gas_price_payload(self, network: str) -> dict[str, Any]:
        minimum = self._minimum_gas_price_wei()
        try:
            fee_history = self._rpc("eth_feeHistory", ["0x3", "pending", [50]], network=network)
            if isinstance(fee_history, dict):
                base_fees = fee_history.get("baseFeePerGas") or []
                rewards = fee_history.get("reward") or []
                latest_base = int(str(base_fees[-1] if base_fees else "0x0"), 16)
                reward_values = [
                    int(str(row[0]), 16)
                    for row in rewards
                    if isinstance(row, list) and row
                ]
                priority = max(reward_values or [minimum // 10, 1])
                estimated = latest_base + priority
                return {
                    "gas_price_wei": max(estimated, minimum),
                    "rpc_gas_price_wei": estimated,
                    "fee_source": "eth_feeHistory",
                    "base_fee_wei": latest_base,
                    "priority_fee_wei": priority,
                }
        except Exception:
            pass
        try:
            rpc_gas_price = int(str(self._rpc("eth_gasPrice", [], network=network) or "0x0"), 16)
        except Exception:  # noqa: BLE001
            rpc_gas_price = 0
        return {
            "gas_price_wei": max(rpc_gas_price, minimum),
            "rpc_gas_price_wei": rpc_gas_price,
            "fee_source": "eth_gasPrice",
        }

    def _estimated_gas_limit(
        self,
        asset: str,
        network: str,
        destination: str,
        amount: float,
        *,
        from_address: str | None = None,
    ) -> int:
        asset_key = _asset_key(asset)
        fallback_key = "WALLET_EVM_FALLBACK_ETH_GAS_LIMIT" if asset_key == "ETH" else "WALLET_EVM_FALLBACK_ERC20_GAS_LIMIT"
        fallback = max(21_000, int(float(self.config.get(fallback_key, 21_000 if asset_key == "ETH" else 70_000) or 0)))
        buffer = max(1.0, float(self.config.get("WALLET_EVM_GAS_LIMIT_BUFFER_MULTIPLIER", 1.20) or 1.0))
        tx: dict[str, Any]
        if asset_key == "ETH":
            tx = {
                "to": str(destination or "").strip(),
                "value": hex(int(float(amount or 0.0) * 10**18)),
            }
        else:
            contract = self._token_contract(asset_key, network)
            if not contract:
                return int(math.ceil(fallback * buffer))
            decimals = self._token_decimals(asset_key, network)
            tx = {
                "to": contract,
                "value": "0x0",
                "data": self._erc20_transfer_data(destination, _decimal_units(amount, decimals)),
            }
        estimate_from = str(from_address or self.config.get("WALLET_EVM_ESTIMATE_FROM_ADDRESS", "") or "").strip()
        if re.fullmatch(r"0x[a-fA-F0-9]{40}", estimate_from):
            tx["from"] = estimate_from
        try:
            estimated = int(str(self._rpc("eth_estimateGas", [tx], network=network) or "0x0"), 16)
        except Exception:
            estimated = fallback
        if estimated <= 0:
            estimated = fallback
        return int(math.ceil(max(estimated, fallback) * buffer))

    @staticmethod
    def _erc20_transfer_data(destination: str, amount_units: int) -> str:
        return "0xa9059cbb" + str(destination or "").lower().replace("0x", "").rjust(64, "0") + hex(int(amount_units or 0))[2:].rjust(64, "0")

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

    def _token_balance_units(self, address: str, asset: str, network: str) -> int:
        contract = self._token_contract(asset, network)
        if not contract:
            return 0
        data = "0x70a08231" + str(address or "").lower().replace("0x", "").rjust(64, "0")
        raw = self._rpc("eth_call", [{"to": contract, "data": data}, "latest"], network=network)
        return int(str(raw or "0x0"), 16)

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
        except Exception:  # noqa: BLE001
            private_key = ec.generate_private_key(ec.SECP256K1())
            private_value = private_key.private_numbers().private_value.to_bytes(32, "big")
            public_key = private_key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.CompressedPoint)
            return GeneratedWallet(
                _xrp_classic_address(public_key),
                private_value.hex(),
                public_key.hex(),
                "secp256k1",
                "xrp",
            )

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
        try:
            from xrpl.clients import JsonRpcClient
            from xrpl.models.transactions import Payment
            from xrpl.transaction import submit_and_wait
            from xrpl.utils import xrp_to_drops
            from xrpl.wallet import Wallet
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("xrpl-py is required for XRP signing") from exc

        private_key_value = str(private_key or "").strip()
        if private_key_value.startswith("s"):
            wallet = Wallet.from_seed(private_key_value)
        else:
            public_key = _secp256k1_public_key_from_private(private_key_value)
            wallet = Wallet(
                public_key=public_key,
                private_key=private_key_value.upper(),
                master_address=_xrp_classic_address(bytes.fromhex(public_key)),
            )
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
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "TradingBotWalletSync/1.0"},
        method="POST",
    )
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


def _xrp_classic_address(public_key: bytes) -> str:
    return _base58check_with_alphabet(b"\x00" + _hash160(public_key), XRP_BASE58_ALPHABET)


def _secp256k1_public_key_from_private(private_key_hex: str) -> str:
    private_value = int(str(private_key_hex or "").strip(), 16)
    private_key = ec.derive_private_key(private_value, ec.SECP256K1())
    return private_key.public_key().public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.CompressedPoint,
    ).hex().upper()


def _base58check_with_alphabet(payload: bytes, alphabet: str) -> str:
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return _base58(payload + checksum, alphabet=alphabet)


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


def _decimal_units(amount: Any, decimals: int) -> int:
    value = Decimal(str(amount or "0"))
    scale = Decimal(10) ** int(decimals)
    return int((value * scale).to_integral_value(rounding=ROUND_DOWN))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
