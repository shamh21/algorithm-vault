"""Fail-closed self-custody wallet workflow support."""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from flask import current_app, has_app_context

from ..extensions import db
from ..models import DepositAddress, WalletAccount, WalletAddress, WalletAuditLog, WalletWithdrawal
from .wallet_addresses import use_real_addresses, validate_withdraw_address


EVM_NETWORKS = {"ETHEREUM", "ARBITRUM", "OPTIMISM", "BASE", "POLYGON", "AVALANCHE", "BSC"}
EVM_ASSETS = {"ETH", "USDC", "USDT"}
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AddressBalance:
    amount: float
    asset: str
    amount_eth: float
    checked: bool
    reason: str = ""

    @property
    def has_funds(self) -> bool:
        return self.checked and (self.amount > 0 or self.amount_eth > 0)


class SelfCustodyWalletService:
    """Records public EVM wallet state and drafts withdrawals without custody secrets."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("WALLET_SELF_CUSTODY_ENABLED", False)) or bool(
            self.config.get("WALLET_REAL_CUSTODY_ENABLED", False)
        )

    def supports_network(self, asset: str, network: str) -> bool:
        return self._asset_key(asset) in EVM_ASSETS and self._network_key(network) in EVM_NETWORKS

    def validate_address(self, network: str, address: str) -> bool:
        if self._network_key(network) not in EVM_NETWORKS:
            return False
        return bool(re.fullmatch(r"0x[a-fA-F0-9]{40}", str(address or "").strip()))

    def account_for(self, user_id: int, asset: str, network: str) -> WalletAccount:
        asset_key = self._asset_key(asset)
        network_name = str(network or "").strip() or "Ethereum"
        account = WalletAccount.query.filter_by(
            user_id=user_id,
            provider=str(self.config.get("WALLET_PROVIDER", "self_custody") or "self_custody"),
            asset=asset_key,
            network=network_name,
        ).one_or_none()
        if account is None:
            account = WalletAccount(
                user_id=user_id,
                provider=str(self.config.get("WALLET_PROVIDER", "self_custody") or "self_custody"),
                asset=asset_key,
                network=network_name,
                status="active",
            )
            account.encrypted_metadata = {"custody": "external", "stores_public_addresses_only": True}
            db.session.add(account)
            db.session.flush()
        return account

    def record_public_address(
        self,
        user_id: int,
        asset: str,
        network: str,
        address: str,
        *,
        deposit_address_id: int | None = None,
        status: str = "active",
        rotated_from_id: int | None = None,
    ) -> WalletAddress | None:
        if not self.supports_network(asset, network) or not self.validate_address(network, address):
            return None

        account = self.account_for(user_id, asset, network)
        existing = (
            WalletAddress.query.filter_by(
                user_id=user_id,
                asset=self._asset_key(asset),
                network=str(network or "").strip() or "Ethereum",
                address=address,
            )
            .order_by(WalletAddress.id.desc())
            .first()
        )
        if existing is not None:
            existing.status = status
            if deposit_address_id is not None:
                existing.deposit_address_id = deposit_address_id
            if rotated_from_id is not None:
                existing.rotated_from_id = rotated_from_id
            return existing

        latest = (
            WalletAddress.query.filter_by(
                user_id=user_id,
                asset=self._asset_key(asset),
                network=str(network or "").strip() or "Ethereum",
            )
            .order_by(WalletAddress.rotation_index.desc())
            .first()
        )
        wallet_address = WalletAddress(
            wallet_account_id=account.id,
            user_id=user_id,
            deposit_address_id=deposit_address_id,
            asset=self._asset_key(asset),
            network=str(network or "").strip() or "Ethereum",
            address=address,
            status=status,
            rotation_index=(latest.rotation_index if latest else 0) + 1,
            rotated_from_id=rotated_from_id,
        )
        wallet_address.encrypted_metadata = {"custody": "external", "validated_address_format": True}
        db.session.add(wallet_address)
        db.session.flush()
        return wallet_address

    def handle_rotated_address(
        self,
        user_id: int,
        asset: str,
        network: str,
        old_address: DepositAddress | None,
        replacement_address: DepositAddress | None,
    ) -> WalletWithdrawal | None:
        if old_address is None or replacement_address is None:
            return None
        if not self.enabled:
            self._audit(
                user_id=user_id,
                action="rotation_sweep_skipped",
                status="disabled",
                message="Self-custody sweep workflow is disabled.",
                metadata={"asset": asset, "network": network, "source_deposit_address_id": old_address.id},
            )
            return None
        if not self.supports_network(asset, network):
            self._audit(
                user_id=user_id,
                action="rotation_sweep_rejected",
                status="unsupported_network",
                message="Rotated-address sweep is supported for EVM assets only.",
                metadata={"asset": asset, "network": network, "source_deposit_address_id": old_address.id},
            )
            return None

        source = self.record_public_address(
            user_id,
            asset,
            network,
            old_address.address,
            deposit_address_id=old_address.id,
            status="inactive",
        )
        replacement = self.record_public_address(
            user_id,
            asset,
            network,
            replacement_address.address,
            deposit_address_id=replacement_address.id,
            status="active",
            rotated_from_id=source.id if source is not None else None,
        )
        balance = self.check_address_balance(asset, network, old_address.address)
        if not balance.checked:
            self._audit(
                user_id=user_id,
                wallet_account_id=source.wallet_account_id if source is not None else None,
                action="rotation_balance_check_unavailable",
                status="blocked",
                message="Old address balance could not be checked, so no sweep was created.",
                metadata={"asset": asset, "network": network, "reason": balance.reason},
            )
            return None
        if not balance.has_funds:
            self._audit(
                user_id=user_id,
                wallet_account_id=source.wallet_account_id if source is not None else None,
                action="rotation_no_funds",
                status="complete",
                message="Old address had no detectable funds at rotation time.",
                metadata={"asset": asset, "network": network},
            )
            return None

        withdrawal = self._create_rotation_withdrawal(
            user_id=user_id,
            asset=asset,
            network=network,
            source_deposit_address=old_address,
            source_wallet_address=source,
            replacement_wallet_address=replacement,
            balance=balance,
        )
        self._audit(
            user_id=user_id,
            wallet_account_id=source.wallet_account_id if source is not None else None,
            wallet_withdrawal_id=withdrawal.id,
            action="rotation_sweep_workflow_created",
            status=withdrawal.status,
            message="Rotated-address funds require a gated sweep workflow.",
            metadata=withdrawal.details,
        )
        return withdrawal

    def check_address_balance(self, asset: str, network: str, address: str) -> AddressBalance:
        if not self.enabled:
            return AddressBalance(0.0, self._asset_key(asset), 0.0, False, "self-custody disabled")
        if not self.supports_network(asset, network) or not self.validate_address(network, address):
            return AddressBalance(0.0, self._asset_key(asset), 0.0, False, "unsupported network or address")

        rpc_url = str(self.config.get("WALLET_EVM_RPC_URL", "") or "").strip()
        if not rpc_url:
            return AddressBalance(0.0, self._asset_key(asset), 0.0, False, "wallet RPC not configured")

        if self._asset_key(asset) != "ETH":
            return AddressBalance(0.0, self._asset_key(asset), 0.0, False, "token balance indexer not configured")

        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_getBalance",
                "params": [address, "latest"],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            rpc_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=float(self.config.get("REALTIME_MARKET_CONNECT_TIMEOUT_SECONDS", 3.0) or 3.0),
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            return AddressBalance(0.0, "ETH", 0.0, False, f"wallet RPC failed: {exc}")

        raw_balance = str(body.get("result") or "0x0")
        try:
            wei = int(raw_balance, 16)
        except ValueError:
            return AddressBalance(0.0, "ETH", 0.0, False, "wallet RPC returned invalid balance")
        amount_eth = wei / 10**18
        return AddressBalance(amount_eth, "ETH", amount_eth, True)

    def create_manual_withdrawal(
        self,
        *,
        user_id: int,
        asset: str,
        network: str,
        destination_address: str,
        amount: float,
        trading_connection_id: int | None = None,
    ) -> WalletWithdrawal:
        """Create an idempotent manual withdrawal record."""

        approval_required = use_real_addresses(self.config) and bool(
            self.config.get("WALLET_REQUIRE_WITHDRAWAL_APPROVAL", True)
        )
        withdrawal = WalletWithdrawal(
            user_id=user_id,
            trading_connection_id=trading_connection_id,
            asset=self._asset_key(asset),
            network=str(network or "").strip() or "native",
            destination_address=str(destination_address or "").strip(),
            amount=float(amount or 0.0),
            amount_eth=float(amount or 0.0) if self._asset_key(asset) == "ETH" else 0.0,
            status="pending_approval" if approval_required else "pending_submission",
            workflow_type="manual_withdrawal",
            idempotency_token=f"manual:{user_id}:{self._asset_key(asset)}:{uuid.uuid4().hex}",
        )
        withdrawal.details = {
            "address_mode": "real" if use_real_addresses(self.config) else "test",
            "provider": self.config.get("WALLET_PROVIDER", "self_custody"),
            "approval_required": approval_required,
        }
        db.session.add(withdrawal)
        db.session.flush()
        return withdrawal

    def approve_withdrawal(
        self,
        withdrawal: WalletWithdrawal,
        *,
        approved_by_user_id: int | None,
        mode: str,
    ) -> WalletWithdrawal:
        """Approve and submit a gated manual withdrawal."""

        if withdrawal.status != "pending_approval":
            raise RuntimeError("Only pending-approval withdrawals can be approved.")
        withdrawal.status = "pending_submission"
        withdrawal.approved_at = datetime.utcnow()
        details = withdrawal.details
        details["approved_by_user_id"] = approved_by_user_id
        details["approved_at"] = withdrawal.approved_at.isoformat()
        withdrawal.details = details
        self._audit(
            user_id=withdrawal.user_id,
            wallet_withdrawal_id=withdrawal.id,
            action="withdrawal_approved",
            status=withdrawal.status,
            message="Withdrawal approved for live submission.",
            metadata=details,
        )
        return self.submit_withdrawal(withdrawal, mode=mode)

    def reject_withdrawal(
        self,
        withdrawal: WalletWithdrawal,
        *,
        rejected_by_user_id: int | None,
        reason: str,
    ) -> WalletWithdrawal:
        """Reject a pending withdrawal without broadcasting it."""

        if withdrawal.status not in {"pending_approval", "pending_submission"}:
            raise RuntimeError("Only pending withdrawals can be rejected.")
        withdrawal.status = "rejected"
        withdrawal.failure_reason = reason or "Withdrawal rejected by admin."
        withdrawal.completed_at = datetime.utcnow()
        details = withdrawal.details
        details["rejected_by_user_id"] = rejected_by_user_id
        details["rejected_at"] = withdrawal.completed_at.isoformat()
        details["rejection_reason"] = withdrawal.failure_reason
        withdrawal.details = details
        self._audit(
            user_id=withdrawal.user_id,
            wallet_withdrawal_id=withdrawal.id,
            action="withdrawal_rejected",
            status=withdrawal.status,
            message=withdrawal.failure_reason,
            metadata=details,
        )
        return withdrawal

    def submit_withdrawal(self, withdrawal: WalletWithdrawal, *, mode: str = "live") -> WalletWithdrawal:
        """Submit a manual withdrawal workflow through the real custody service."""

        if withdrawal.status == "pending_approval":
            withdrawal.status = "failed"
            withdrawal.failure_reason = "withdrawal requires admin approval before broadcast"
            withdrawal.completed_at = datetime.utcnow()
            return withdrawal
        if withdrawal.amount <= 0:
            withdrawal.status = "failed"
            withdrawal.failure_reason = "withdrawal amount must be positive"
            withdrawal.completed_at = datetime.utcnow()
            return withdrawal
        if not validate_withdraw_address(withdrawal.destination_address, withdrawal.asset, withdrawal.network):
            withdrawal.status = "failed"
            withdrawal.failure_reason = "destination address is malformed for the selected asset/network"
            withdrawal.completed_at = datetime.utcnow()
            return withdrawal

        if not use_real_addresses(self.config):
            withdrawal.status = "failed"
            withdrawal.failure_reason = "real wallet address mode is required for withdrawals"
            withdrawal.completed_at = datetime.utcnow()
            return withdrawal

        try:
            return self._submit_real_withdrawal(withdrawal, mode=mode)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Withdrawal submission failed for withdrawal %s.", withdrawal.id)
            withdrawal.status = "failed"
            withdrawal.failure_reason = str(exc)
            withdrawal.completed_at = datetime.utcnow()
            details = withdrawal.details
            details["exception"] = str(exc)
            withdrawal.details = details
            self._audit(
                user_id=withdrawal.user_id,
                wallet_withdrawal_id=withdrawal.id,
                action="withdrawal_submission_failed",
                status=withdrawal.status,
                message=f"Withdrawal submission failed: {exc}",
                metadata=withdrawal.details,
            )
            return withdrawal

    def _submit_real_withdrawal(self, withdrawal: WalletWithdrawal, *, mode: str) -> WalletWithdrawal:
        if not has_app_context():
            raise RuntimeError("Real withdrawal submission requires an application context")
        if str(mode or "").lower() != "live":
            raise RuntimeError("Real wallet withdrawals can only be broadcast in live mode")
        custody = current_app.extensions["services"]["wallet_custody"]
        result = custody.sign_and_broadcast(withdrawal, mode=mode)
        withdrawal.status = str(result.status or "submitted")
        withdrawal.provider_reference = str(result.provider_reference or "")
        details = withdrawal.details
        details["provider_response"] = result.raw
        details["api_mode"] = "live"
        withdrawal.details = details
        self._audit(
            user_id=withdrawal.user_id,
            wallet_withdrawal_id=withdrawal.id,
            action="withdrawal_submitted",
            status=withdrawal.status,
            message="Withdrawal submitted to the configured wallet custody adapter.",
            metadata=withdrawal.details,
        )
        return withdrawal

    def _create_rotation_withdrawal(
        self,
        *,
        user_id: int,
        asset: str,
        network: str,
        source_deposit_address: DepositAddress,
        source_wallet_address: WalletAddress | None,
        replacement_wallet_address: WalletAddress | None,
        balance: AddressBalance,
    ) -> WalletWithdrawal:
        destination = str(self.config.get("WALLET_SWEEP_DESTINATION_ETH", "") or "").strip()
        amount_eth = float(balance.amount_eth or 0.0)
        fee_eth = self._fee_eth(amount_eth)
        idempotency_token = f"rotate:{source_deposit_address.id}:{destination.lower()}"
        existing = WalletWithdrawal.query.filter_by(idempotency_token=idempotency_token).one_or_none()
        if existing is not None:
            return existing

        status = "pending_approval"
        workflow_type = "rotated_address_draft"
        failure_reason = None
        auto_sweep = bool(self.config.get("WALLET_AUTO_SWEEP_ENABLED", False))
        if auto_sweep:
            workflow_type = "rotated_address_auto_sweep"
            status, failure_reason = self._auto_sweep_gate_status(amount_eth, fee_eth, destination)
        elif not bool(self.config.get("WALLET_REQUIRE_WITHDRAWAL_APPROVAL", True)):
            status = "blocked"
            failure_reason = "automatic sweep disabled and approval gate not configured"

        withdrawal = WalletWithdrawal(
            user_id=user_id,
            wallet_account_id=source_wallet_address.wallet_account_id if source_wallet_address is not None else None,
            source_wallet_address_id=source_wallet_address.id if source_wallet_address is not None else None,
            source_deposit_address_id=source_deposit_address.id,
            asset=self._asset_key(asset),
            network=str(network or "").strip() or "Ethereum",
            destination_address=destination,
            amount=float(balance.amount or 0.0),
            amount_eth=amount_eth,
            fee_eth=fee_eth,
            status=status,
            workflow_type=workflow_type,
            idempotency_token=idempotency_token,
            failure_reason=failure_reason,
        )
        withdrawal.details = {
            "source_address": source_deposit_address.address,
            "replacement_wallet_address_id": replacement_wallet_address.id if replacement_wallet_address is not None else None,
            "auto_sweep_enabled": auto_sweep,
            "approval_required": bool(self.config.get("WALLET_REQUIRE_WITHDRAWAL_APPROVAL", True)),
            "fee_policy": {
                "bps": float(self.config.get("WALLET_WITHDRAWAL_FEE_BPS", 0.0) or 0.0),
                "fixed_eth": float(self.config.get("WALLET_WITHDRAWAL_FIXED_FEE_ETH", 0.0) or 0.0),
                "max_eth": float(self.config.get("WALLET_WITHDRAWAL_FEE_MAX_ETH", 0.0) or 0.0),
            },
            "failure_reason": failure_reason,
        }
        db.session.add(withdrawal)
        db.session.flush()
        return withdrawal

    def _auto_sweep_gate_status(self, amount_eth: float, fee_eth: float, destination: str) -> tuple[str, str | None]:
        if not self.validate_address("Ethereum", destination):
            return "blocked", "sweep destination is not an EVM address"
        if amount_eth <= 0:
            return "blocked", "no ETH-equivalent amount is available"
        if fee_eth < 0 or fee_eth >= amount_eth:
            return "blocked", "fee exceeds available sweep amount"
        max_sweep = float(self.config.get("WALLET_MAX_SWEEP_ETH", 0.0) or 0.0)
        if max_sweep > 0 and amount_eth > max_sweep:
            return "blocked", "sweep amount exceeds configured cap"
        if not str(self.config.get("WALLET_EVM_RPC_URL", "") or "").strip():
            return "blocked", "wallet RPC is not configured"
        return "blocked", "signer and broadcast provider are not configured"

    def _fee_eth(self, amount_eth: float) -> float:
        bps = max(0.0, float(self.config.get("WALLET_WITHDRAWAL_FEE_BPS", 0.0) or 0.0))
        fixed = max(0.0, float(self.config.get("WALLET_WITHDRAWAL_FIXED_FEE_ETH", 0.0) or 0.0))
        cap = max(0.0, float(self.config.get("WALLET_WITHDRAWAL_FEE_MAX_ETH", 0.0) or 0.0))
        fee = (max(0.0, amount_eth) * bps / 10_000) + fixed
        return min(fee, cap) if cap > 0 else fee

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
        return re.sub(r"[^A-Za-z0-9]", "", str(asset or "")).upper()

    @staticmethod
    def _network_key(network: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", str(network or "")).upper()
