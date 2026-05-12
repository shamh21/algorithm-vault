"""Durable transfer, reserve, conversion, and withdrawal handling for Vault Cycles."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from ..extensions import db
from ..models import AuditLog, DepositAddress, VaultCycle, VaultCycleAllocation, VaultCycleRiskEvent, VaultCycleTransfer


class VaultCycleTransferService:
    """Creates idempotent transfer records and delegates real exchange operations."""

    def __init__(self, config: dict[str, Any], trading_connections: Any) -> None:
        self.config = config
        self.trading_connections = trading_connections

    def funding_blockers(self) -> list[str]:
        blockers: list[str] = []
        if not bool(self.config.get("VAULT_CYCLE_REAL_TRANSFERS_ENABLED", False)):
            blockers.append("VAULT_CYCLE_REAL_TRANSFERS_ENABLED is disabled")
        if not bool(self.config.get("ENABLE_LIVE_TRADING", False)):
            blockers.append("live trading is disabled")
        return blockers

    def reserve_allocation(self, cycle: VaultCycle, allocation: VaultCycleAllocation) -> VaultCycleTransfer:
        """Reserve already available collateral on the exchange and persist confirmation."""

        self._require_real_transfers()
        amount = max(0.0, float(allocation.allocated_amount or allocation.target_amount or 0.0))
        transfer = self._get_or_create_transfer(
            cycle=cycle,
            allocation=allocation,
            direction="fund_exchange",
            transfer_type="exchange_reserve",
            asset=allocation.collateral_asset,
            amount=amount,
            idempotency_key=f"vault-cycle:{cycle.id}:allocation:{allocation.id}:reserve",
        )
        if transfer.status in {"confirmed", "complete"}:
            return transfer
        connector = self.trading_connections.connector_for_user(cycle.user_id, allocation.trading_connection_id)
        try:
            result = connector.reserve_funds(cycle.execution_mode, allocation.collateral_asset, amount)
        except Exception as exc:  # noqa: BLE001
            self._fail_transfer(transfer, str(exc))
            self.record_risk_event(
                cycle,
                allocation=allocation,
                transfer=transfer,
                category="transfer",
                severity="error",
                rule_name="exchange_reserve_failed",
                reason=str(exc),
            )
            raise
        transfer.status = "confirmed"
        transfer.provider_reference = str(result.get("provider_reference") or transfer.idempotency_key)
        transfer.confirmed_amount = float(result.get("confirmed_amount", amount) or amount)
        transfer.submitted_at = datetime.utcnow()
        transfer.confirmed_at = transfer.submitted_at
        transfer.details = {"provider_response": result, "reserve_confirmed": True}
        allocation.status = "funded"
        allocation.funded_at = transfer.confirmed_at
        self.audit(cycle, "allocation_reserved", f"Reserved {transfer.confirmed_amount:.6f} {transfer.asset} on {allocation.provider}.", allocation)
        return transfer

    def convert_allocation_to_settlement(self, cycle: VaultCycle, allocation: VaultCycleAllocation, amount: float) -> VaultCycleTransfer | None:
        collateral = str(allocation.collateral_asset or "").upper()
        settlement = str(cycle.settlement_asset or "").upper()
        if collateral == settlement:
            return None
        if not bool(self.config.get("VAULT_CYCLE_CONVERSION_ENABLED", False)):
            raise RuntimeError("stablecoin_conversion_route_unavailable")
        if allocation.provider != str(self.config.get("VAULT_CYCLE_CONVERSION_PROVIDER", "kucoin")).lower():
            raise RuntimeError("stablecoin_conversion_provider_mismatch")
        transfer = self._get_or_create_transfer(
            cycle=cycle,
            allocation=allocation,
            direction="conversion",
            transfer_type="stablecoin_conversion",
            asset=collateral,
            amount=amount,
            idempotency_key=f"vault-cycle:{cycle.id}:allocation:{allocation.id}:convert:{collateral}:{settlement}",
        )
        if transfer.status in {"confirmed", "complete", "submitted"}:
            return transfer
        connector = self.trading_connections.connector_for_user(cycle.user_id, allocation.trading_connection_id)
        try:
            result = connector.convert_stablecoin(
                cycle.execution_mode,
                collateral,
                settlement,
                amount,
                float(self.config.get("VAULT_CYCLE_STABLECOIN_MAX_SLIPPAGE_BPS", 10.0) or 10.0),
                client_reference=transfer.idempotency_key,
            )
        except Exception as exc:  # noqa: BLE001
            self._fail_transfer(transfer, str(exc))
            self.record_risk_event(
                cycle,
                allocation=allocation,
                transfer=transfer,
                category="conversion",
                severity="error",
                rule_name="stablecoin_conversion_failed",
                reason=str(exc),
            )
            raise
        transfer.status = str(result.get("status") or "submitted")
        transfer.provider_reference = str(result.get("provider_reference") or "")
        transfer.confirmed_amount = float(result.get("confirmed_amount", 0.0) or 0.0)
        transfer.destination = settlement
        transfer.submitted_at = datetime.utcnow()
        transfer.details = {"provider_response": result, "to_asset": settlement}
        if transfer.status in {"confirmed", "complete"}:
            transfer.confirmed_at = datetime.utcnow()
        self.audit(cycle, "allocation_converted", f"Submitted {collateral}->{settlement} conversion for {allocation.provider}.", allocation)
        return transfer

    def withdraw_allocation_to_vault(self, cycle: VaultCycle, allocation: VaultCycleAllocation, amount: float) -> VaultCycleTransfer:
        self._require_real_transfers()
        if not bool(self.config.get("VAULT_CYCLE_AUTOMATIC_WITHDRAWALS_ENABLED", False)):
            raise RuntimeError("VAULT_CYCLE_AUTOMATIC_WITHDRAWALS_ENABLED is disabled")
        settlement = str(cycle.settlement_asset or allocation.collateral_asset or "").upper()
        destination = self.settlement_destination(cycle.user_id, settlement)
        if not self._is_allowed_destination(settlement, destination["network"], destination["address"]):
            raise RuntimeError("settlement destination is not in VAULT_CYCLE_ALLOWED_WITHDRAWAL_ADDRESSES_JSON")
        self._enforce_transfer_caps(cycle, settlement, amount)
        transfer = self._get_or_create_transfer(
            cycle=cycle,
            allocation=allocation,
            direction="withdraw_to_vault",
            transfer_type="exchange_withdrawal",
            asset=settlement,
            amount=amount,
            idempotency_key=f"vault-cycle:{cycle.id}:allocation:{allocation.id}:withdraw:{settlement}",
        )
        transfer.network = destination["network"]
        transfer.destination = destination["address"]
        if transfer.status in {"confirmed", "complete", "submitted"}:
            return transfer
        connector = self.trading_connections.connector_for_user(cycle.user_id, allocation.trading_connection_id)
        try:
            result = connector.withdraw_to_address(
                cycle.execution_mode,
                settlement,
                amount,
                destination["address"],
                network=destination["network"],
                memo=destination.get("memo"),
                client_reference=transfer.idempotency_key,
            )
        except Exception as exc:  # noqa: BLE001
            self._fail_transfer(transfer, str(exc))
            self.record_risk_event(
                cycle,
                allocation=allocation,
                transfer=transfer,
                category="withdrawal",
                severity="error",
                rule_name="exchange_withdrawal_failed",
                reason=str(exc),
            )
            raise
        transfer.status = str(result.get("status") or "submitted")
        transfer.provider_reference = str(result.get("provider_reference") or "")
        transfer.confirmed_amount = float(result.get("confirmed_amount", amount) or amount)
        transfer.fee_amount = float(result.get("fee_amount", 0.0) or 0.0)
        transfer.fee_asset = str(result.get("fee_asset") or settlement)
        transfer.submitted_at = datetime.utcnow()
        transfer.details = {"provider_response": result}
        if transfer.status in {"confirmed", "complete"}:
            transfer.confirmed_at = datetime.utcnow()
        self.audit(cycle, "allocation_withdrawal_submitted", f"Submitted {settlement} withdrawal from {allocation.provider}.", allocation)
        return transfer

    def refresh_transfer(self, transfer: VaultCycleTransfer) -> VaultCycleTransfer:
        if transfer.status in {"confirmed", "complete", "failed", "blocked"}:
            return transfer
        if not transfer.trading_connection_id or not transfer.provider_reference:
            return transfer
        cycle = transfer.vault_cycle
        connector = self.trading_connections.connector_for_user(transfer.user_id, transfer.trading_connection_id)
        try:
            result = connector.transfer_status(cycle.execution_mode, transfer.provider_reference, transfer.transfer_type)
        except Exception as exc:  # noqa: BLE001
            transfer.attempts = int(transfer.attempts or 0) + 1
            details = transfer.details
            details["last_status_error"] = str(exc)
            transfer.details = details
            return transfer
        status = str(result.get("status") or transfer.status)
        if status in {"confirmed", "complete"}:
            transfer.status = "confirmed"
            transfer.confirmed_at = datetime.utcnow()
            if result.get("confirmed_amount") is not None:
                transfer.confirmed_amount = float(result.get("confirmed_amount") or 0.0)
        else:
            transfer.status = status
        details = transfer.details
        details["last_status_response"] = result
        transfer.details = details
        return transfer

    def settlement_destination(self, user_id: int, asset: str) -> dict[str, str]:
        asset_key = str(asset or "").upper().strip()
        configured_network = self._default_network(asset_key)
        query = DepositAddress.query.filter_by(user_id=user_id, asset=asset_key, is_active=True)
        if configured_network:
            preferred = query.filter_by(network=configured_network).order_by(DepositAddress.version.desc()).first()
            if preferred is not None:
                return {"address": preferred.address, "network": preferred.network, "memo": ""}
        address = query.order_by(DepositAddress.version.desc()).first()
        if address is None:
            raise RuntimeError(f"No active vault settlement address is available for {asset_key}.")
        return {"address": address.address, "network": address.network, "memo": ""}

    def record_risk_event(
        self,
        cycle: VaultCycle,
        *,
        allocation: VaultCycleAllocation | None = None,
        transfer: VaultCycleTransfer | None = None,
        category: str,
        severity: str,
        rule_name: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> VaultCycleRiskEvent:
        event = VaultCycleRiskEvent(
            vault_cycle_id=cycle.id,
            allocation_id=allocation.id if allocation is not None else None,
            transfer_id=transfer.id if transfer is not None else None,
            user_id=cycle.user_id,
            trading_connection_id=allocation.trading_connection_id if allocation is not None else cycle.trading_connection_id,
            provider=allocation.provider if allocation is not None else "",
            category=category,
            severity=severity,
            rule_name=rule_name,
            reason=reason,
            status="recorded",
        )
        event.details = metadata or {}
        db.session.add(event)
        return event

    def audit(self, cycle: VaultCycle, action: str, message: str, allocation: VaultCycleAllocation | None = None, metadata: dict[str, Any] | None = None) -> AuditLog:
        entry = AuditLog(
            user_id=cycle.user_id,
            trading_connection_id=allocation.trading_connection_id if allocation is not None else cycle.trading_connection_id,
            category="vault_cycle",
            action=action,
            message=message,
        )
        entry.details = {
            "vault_cycle_id": cycle.id,
            "allocation_id": allocation.id if allocation is not None else None,
            "provider": allocation.provider if allocation is not None else None,
            **(metadata or {}),
        }
        db.session.add(entry)
        return entry

    def _get_or_create_transfer(
        self,
        *,
        cycle: VaultCycle,
        allocation: VaultCycleAllocation | None,
        direction: str,
        transfer_type: str,
        asset: str,
        amount: float,
        idempotency_key: str,
    ) -> VaultCycleTransfer:
        existing = VaultCycleTransfer.query.filter_by(idempotency_key=idempotency_key).one_or_none()
        if existing is not None:
            return existing
        transfer = VaultCycleTransfer(
            vault_cycle_id=cycle.id,
            allocation_id=allocation.id if allocation is not None else None,
            user_id=cycle.user_id,
            trading_connection_id=allocation.trading_connection_id if allocation is not None else cycle.trading_connection_id,
            provider=allocation.provider if allocation is not None else "",
            direction=direction,
            transfer_type=transfer_type,
            asset=str(asset or "").upper(),
            requested_amount=max(0.0, float(amount or 0.0)),
            idempotency_key=idempotency_key,
            status="pending",
        )
        db.session.add(transfer)
        db.session.flush()
        return transfer

    def _fail_transfer(self, transfer: VaultCycleTransfer, reason: str) -> None:
        transfer.status = "failed"
        transfer.failure_reason = reason
        transfer.attempts = int(transfer.attempts or 0) + 1
        transfer.details = {**transfer.details, "failure_reason": reason}

    def _require_real_transfers(self) -> None:
        blockers = self.funding_blockers()
        if blockers:
            raise RuntimeError("; ".join(blockers))

    def _enforce_transfer_caps(self, cycle: VaultCycle, asset: str, amount: float) -> None:
        amount_usd = max(0.0, float(amount or 0.0))
        max_transfer = float(self.config.get("VAULT_CYCLE_MAX_TRANSFER_USD", 0.0) or 0.0)
        if max_transfer > 0 and amount_usd > max_transfer + 1e-9:
            raise RuntimeError("withdrawal amount exceeds VAULT_CYCLE_MAX_TRANSFER_USD")
        daily_cap = float(self.config.get("VAULT_CYCLE_DAILY_WITHDRAWAL_CAP_USD", 0.0) or 0.0)
        if daily_cap <= 0:
            return
        since = datetime.utcnow() - timedelta(hours=24)
        total = sum(
            float(row.requested_amount or 0.0)
            for row in VaultCycleTransfer.query.filter(
                VaultCycleTransfer.user_id == cycle.user_id,
                VaultCycleTransfer.direction == "withdraw_to_vault",
                VaultCycleTransfer.status.notin_(["failed", "blocked"]),
                VaultCycleTransfer.created_at >= since,
            ).all()
        )
        if total + amount_usd > daily_cap + 1e-9:
            raise RuntimeError("withdrawal amount exceeds VAULT_CYCLE_DAILY_WITHDRAWAL_CAP_USD")

    def _is_allowed_destination(self, asset: str, network: str, address: str) -> bool:
        book = self.config.get("VAULT_CYCLE_ALLOWED_WITHDRAWAL_ADDRESSES") or {}
        asset_book = book.get(str(asset or "").upper(), {})
        if not isinstance(asset_book, dict) or not asset_book:
            return False
        address_key = str(address or "").strip().lower()
        network_key = self._network_key(network)
        for configured_network, addresses in asset_book.items():
            if self._network_key(configured_network) != network_key:
                continue
            return address_key in {str(item).strip().lower() for item in addresses or []}
        return False

    def _default_network(self, asset: str) -> str:
        mapping = self.config.get("VAULT_CYCLE_DEFAULT_NETWORK_BY_ASSET") or {}
        if isinstance(mapping, dict):
            value = str(mapping.get(str(asset or "").upper(), "") or "").strip()
            if value:
                return value
        return "Arbitrum" if str(asset or "").upper() == "USDC" else "Ethereum"

    @staticmethod
    def _network_key(network: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", str(network or "")).upper()
