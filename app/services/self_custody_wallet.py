"""Fail-closed self-custody wallet workflow support."""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from flask import current_app, has_app_context

from ..extensions import db
from ..models import DepositAddress, Setting, WalletAccount, WalletAddress, WalletAuditLog, WalletWithdrawal
from .failures import WalletCustodyError
from .wallet_addresses import use_real_addresses, validate_withdraw_address
from .withdrawal_config import wallet_withdrawals_enabled

EVM_NETWORKS = {"ETHEREUM", "ARBITRUM", "OPTIMISM", "BASE", "POLYGON", "AVALANCHE", "BSC"}
EVM_ASSETS = {"ETH", "USDC", "USDT"}
logger = logging.getLogger(__name__)


def _sum_withdrawal_amounts(query) -> float:
    return sum(float(row.amount or 0.0) for row in query.all())


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
        return bool(self.config.get("WALLET_SELF_CUSTODY_ENABLED", False)) or bool(self.config.get("WALLET_REAL_CUSTODY_ENABLED", False))

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

        approval_required = use_real_addresses(self.config) and bool(self.config.get("WALLET_REQUIRE_WITHDRAWAL_APPROVAL", True))
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
            "address_mode": "real",
            "provider": self.config.get("WALLET_PROVIDER", "self_custody"),
            "approval_required": approval_required,
        }
        db.session.add(withdrawal)
        db.session.flush()
        self._audit(
            user_id=withdrawal.user_id,
            wallet_withdrawal_id=withdrawal.id,
            action="withdrawal_requested",
            status=withdrawal.status,
            message="Withdrawal request recorded for gated wallet workflow.",
            metadata={
                "asset": withdrawal.asset,
                "network": withdrawal.network,
                "amount": withdrawal.amount,
                "approval_required": approval_required,
            },
        )
        self._apply_treasury_safety(withdrawal, return_status=withdrawal.status)
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

        if withdrawal.status not in {"pending_approval", "pending_submission", "pending_gas_topup", "queued_treasury_solvency"}:
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

        try:
            return self._submit_real_withdrawal(withdrawal, mode=mode)
        except WalletCustodyError as exc:
            logger.exception(
                "Withdrawal custody failure for withdrawal %s.",
                withdrawal.id,
                extra={"withdrawal_id": withdrawal.id, "error_code": exc.code, "context": exc.context},
            )
            withdrawal.status = "failed"
            withdrawal.failure_reason = exc.message
            withdrawal.completed_at = datetime.utcnow()
            details = withdrawal.details
            details["exception"] = exc.code
            withdrawal.details = details
            self._audit(
                user_id=withdrawal.user_id,
                wallet_withdrawal_id=withdrawal.id,
                action="withdrawal_failed",
                status=withdrawal.status,
                message=f"Withdrawal custody failure: {exc.message}",
                metadata={
                    "error_code": exc.code,
                    "context": exc.context,
                    **self._support_impersonation_metadata(withdrawal),
                },
            )
            return withdrawal
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
            raise WalletCustodyError("Real withdrawal submission requires an application context", code="wallet_context_missing")
        if str(mode or "").lower() != "live":
            raise WalletCustodyError("Real wallet withdrawals can only be broadcast in live mode", code="wallet_live_mode_required")
        safety_blockers = self._withdrawal_safety_blockers(withdrawal)
        if safety_blockers:
            return self._block_withdrawal_by_safety_gate(withdrawal, safety_blockers)
        custody = current_app.extensions["services"]["wallet_custody"]
        preflight = custody.withdrawal_preflight(withdrawal)
        details = withdrawal.details
        details["preflight"] = preflight
        withdrawal.details = details
        solvency = current_app.extensions["services"].get("treasury_solvency")
        if solvency is not None:
            safety = solvency.evaluate_withdrawal(
                withdrawal,
                projected_spend_eth=0.0,
                estimated_gas_eth=float(preflight.get("estimated_gas_eth", 0.0) or 0.0),
                persist=True,
            )
            if not safety.get("safe", True):
                return withdrawal
        if preflight.get("gas_topup_required"):
            topup = current_app.extensions["services"]["platform_treasury"].top_up_withdrawal_gas(withdrawal)
            if str(topup.get("status") or "") == "not_required":
                preflight = custody.withdrawal_preflight(withdrawal)
                details = withdrawal.details
                details["preflight"] = preflight
                details["gas_topup"] = topup
                withdrawal.details = details
            elif str(topup.get("status") or "") == "queued_treasury_solvency":
                self._audit(
                    user_id=withdrawal.user_id,
                    wallet_withdrawal_id=withdrawal.id,
                    action="withdrawal_queued_treasury_solvency",
                    status=withdrawal.status,
                    message="Withdrawal is queued until the platform treasury reserve recovers.",
                    metadata={"preflight": preflight, "gas_topup": topup, **self._support_impersonation_metadata(withdrawal)},
                )
                return withdrawal
            elif str(topup.get("status") or "") in {"submitted", "pending", "complete"}:
                self._audit(
                    user_id=withdrawal.user_id,
                    wallet_withdrawal_id=withdrawal.id,
                    action="withdrawal_pending_gas_topup",
                    status=withdrawal.status,
                    message="Withdrawal is waiting for platform treasury gas top-up before token broadcast.",
                    metadata={"preflight": preflight, "gas_topup": topup, **self._support_impersonation_metadata(withdrawal)},
                )
                return withdrawal
        blockers = [str(item) for item in preflight.get("blockers", []) if str(item)]
        if blockers:
            raise WalletCustodyError("; ".join(blockers), code="wallet_preflight_blocked", context={"blockers": blockers})
        result = custody.sign_and_broadcast(withdrawal, mode=mode)
        self._audit(
            user_id=withdrawal.user_id,
            wallet_withdrawal_id=withdrawal.id,
            action="withdrawal_signed",
            status="signed",
            message="Withdrawal signed by the configured custody adapter.",
            metadata={"asset": withdrawal.asset, "network": withdrawal.network, **self._support_impersonation_metadata(withdrawal)},
        )
        withdrawal.status = str(result.status or "submitted")
        withdrawal.provider_reference = str(result.provider_reference or "")
        details = withdrawal.details
        details["provider_response"] = result.raw
        details["api_mode"] = "live"
        withdrawal.details = details
        if withdrawal.status.startswith("failed"):
            withdrawal.failure_reason = {
                "failed_broadcast_not_found": "withdrawal broadcast returned a transaction hash, but the transaction was not found on-chain",
            }.get(withdrawal.status, withdrawal.status.replace("_", " "))
            withdrawal.completed_at = datetime.utcnow()
        self._audit(
            user_id=withdrawal.user_id,
            wallet_withdrawal_id=withdrawal.id,
            action="withdrawal_submitted",
            status=withdrawal.status,
            message="Withdrawal submitted to the configured wallet custody adapter.",
            metadata=withdrawal.details,
        )
        self._audit(
            user_id=withdrawal.user_id,
            wallet_withdrawal_id=withdrawal.id,
            action="withdrawal_broadcast",
            status=withdrawal.status,
            message="Withdrawal broadcast result recorded.",
            metadata={"provider_reference_present": bool(withdrawal.provider_reference), "status": withdrawal.status},
        )
        if solvency is not None:
            solvency.record_withdrawal_gas_usage(withdrawal, result.raw)
        return withdrawal

    def _withdrawal_safety_blockers(self, withdrawal: WalletWithdrawal) -> list[str]:
        config = self.config
        blockers: list[str] = []
        if not wallet_withdrawals_enabled(config) and not bool(config.get("TESTING", False)):
            blockers.append("wallet withdrawals are disabled by configuration")
        if bool(config.get("WALLET_EMERGENCY_STOP", False)):
            blockers.append("wallet emergency stop is active")
        try:
            if Setting.get_json("panic_lock", False):
                blockers.append("panic lock is active")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to read panic lock before withdrawal %s.", withdrawal.id)
            blockers.append("panic lock state could not be verified")

        deployment_target = str(config.get("DEPLOYMENT_TARGET", "local") or "local").strip().lower()
        production_like = deployment_target in {"vps", "production", "prod", "postgres", "staging", "vercel"}
        custody_mode = str(config.get("WALLET_CUSTODY_MODE", "local_dev") or "local_dev").strip().lower()
        if production_like:
            if custody_mode not in {"kms", "hsm", "mpc"}:
                blockers.append("production withdrawals require kms, hsm, or mpc custody mode")
            if bool(config.get("WALLET_SIGNER_ISOLATION_REQUIRED", True)) and not bool(
                config.get("WALLET_SIGNER_ISOLATION_CONFIRMED", False)
            ):
                blockers.append("production withdrawals require confirmed signer isolation")
            if not bool(config.get("WALLET_SDK_CHECKS_PASSED", False)):
                blockers.append("wallet SDK/integration checks are not marked passing")

        blockers.extend(self._withdrawal_velocity_blockers(withdrawal, require_limits=production_like))
        return blockers

    def _withdrawal_velocity_blockers(self, withdrawal: WalletWithdrawal, *, require_limits: bool) -> list[str]:
        since = datetime.utcnow() - timedelta(days=1)
        amount = max(0.0, float(withdrawal.amount or 0.0))
        asset = self._asset_key(withdrawal.asset)
        active_statuses = {
            "pending_approval",
            "pending_submission",
            "pending_gas_topup",
            "queued_treasury_solvency",
            "submitted",
            "complete",
        }
        base_query = WalletWithdrawal.query.filter(WalletWithdrawal.created_at >= since).filter(
            WalletWithdrawal.status.in_(active_statuses)
        )
        if withdrawal.id is not None:
            base_query = base_query.filter(WalletWithdrawal.id != withdrawal.id)

        blockers: list[str] = []
        wallet_limit = max(0.0, float(self.config.get("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_WALLET", 0.0) or 0.0))
        if wallet_limit <= 0 and require_limits:
            blockers.append("per-wallet daily withdrawal limit is not configured")
        elif wallet_limit > 0:
            used = _sum_withdrawal_amounts(base_query.filter(WalletWithdrawal.user_id == withdrawal.user_id))
            if used + amount > wallet_limit + 1e-12:
                blockers.append("per-wallet daily withdrawal limit exceeded")

        asset_limits = self.config.get("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_ASSET") or {}
        asset_limit = 0.0
        if isinstance(asset_limits, dict):
            asset_limit = max(0.0, float(asset_limits.get(asset.lower(), asset_limits.get(asset, 0.0)) or 0.0))
        if asset_limit <= 0 and require_limits:
            blockers.append(f"daily withdrawal limit for {asset} is not configured")
        elif asset_limit > 0:
            used = _sum_withdrawal_amounts(base_query.filter(WalletWithdrawal.asset == asset))
            if used + amount > asset_limit + 1e-12:
                blockers.append(f"daily withdrawal limit for {asset} exceeded")

        destination_limit = max(0.0, float(self.config.get("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_DESTINATION", 0.0) or 0.0))
        if destination_limit <= 0 and require_limits:
            blockers.append("per-destination daily withdrawal limit is not configured")
        elif destination_limit > 0:
            used = _sum_withdrawal_amounts(base_query.filter(WalletWithdrawal.destination_address == withdrawal.destination_address))
            if used + amount > destination_limit + 1e-12:
                blockers.append("per-destination daily withdrawal limit exceeded")

        global_limit = max(0.0, float(self.config.get("WALLET_DAILY_GLOBAL_WITHDRAWAL_LIMIT", 0.0) or 0.0))
        if global_limit <= 0 and require_limits:
            blockers.append("global daily withdrawal limit is not configured")
        elif global_limit > 0:
            used = _sum_withdrawal_amounts(base_query)
            if used + amount > global_limit + 1e-12:
                blockers.append("global daily withdrawal limit exceeded")
        return blockers

    def _block_withdrawal_by_safety_gate(self, withdrawal: WalletWithdrawal, blockers: list[str]) -> WalletWithdrawal:
        withdrawal.status = "failed_safety_gate"
        withdrawal.failure_reason = "; ".join(blockers)
        withdrawal.completed_at = datetime.utcnow()
        details = withdrawal.details
        details["safety_gate_blockers"] = blockers
        details["custody_mode"] = str(self.config.get("WALLET_CUSTODY_MODE", "local_dev") or "local_dev")
        withdrawal.details = details
        self._audit(
            user_id=withdrawal.user_id,
            wallet_withdrawal_id=withdrawal.id,
            action="withdrawal_blocked_by_safety_gate",
            status=withdrawal.status,
            message="Withdrawal blocked by production custody safety gate.",
            metadata={"blockers": blockers, "custody_mode": details["custody_mode"], **self._support_impersonation_metadata(withdrawal)},
        )
        return withdrawal

    @staticmethod
    def _support_impersonation_metadata(withdrawal: WalletWithdrawal) -> dict[str, Any]:
        details = withdrawal.details
        if not details.get("support_impersonation"):
            return {}
        keys = (
            "support_impersonation",
            "impersonator_user_id",
            "impersonator_username",
            "target_user_id",
            "target_username",
            "grant_public_id",
            "approval_bypassed_by_support_impersonation",
            "approval_bypassed_at",
        )
        return {key: details[key] for key in keys if key in details}

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

    def _apply_treasury_safety(self, withdrawal: WalletWithdrawal, *, return_status: str) -> None:
        if not has_app_context():
            return
        solvency = current_app.extensions.get("services", {}).get("treasury_solvency")
        if solvency is None:
            return
        try:
            safety = solvency.evaluate_withdrawal(withdrawal, projected_spend_eth=0.0, persist=True)
        except Exception as exc:  # noqa: BLE001
            safety = {
                "safe": False,
                "status": "queued_treasury_solvency",
                "reason": str(exc),
                "estimated_gas_eth": 0.0,
                "reserve_ratio": 0.0,
                "health_status": "emergency",
            }
            solvency.apply_withdrawal_safety(withdrawal, safety)
        if not safety.get("safe", True):
            details = withdrawal.details
            details["return_status_after_solvency"] = return_status
            withdrawal.details = details

    @staticmethod
    def _asset_key(asset: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", str(asset or "")).upper()

    @staticmethod
    def _network_key(network: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", str(network or "")).upper()
