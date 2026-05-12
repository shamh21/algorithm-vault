"""Reporting payloads for Vault Cycle dashboards and API polling."""

from __future__ import annotations

from typing import Any

from ..models import VaultCycle
from .vault_coherence import extract_cycle_coherence_payload


class VaultCycleReportingService:
    """Builds compact JSON-safe Vault Cycle status payloads."""

    def status_payload(self, cycle: VaultCycle) -> dict[str, Any]:
        settlement = cycle.vault_settlement
        allocations = [
            {
                "id": allocation.id,
                "provider": allocation.provider,
                "trading_connection_id": allocation.trading_connection_id,
                "settlement_asset": allocation.settlement_asset,
                "collateral_asset": allocation.collateral_asset,
                "target_amount": float(allocation.target_amount or 0.0),
                "allocated_amount": float(allocation.allocated_amount or 0.0),
                "allocation_weight": float(allocation.allocation_weight or 0.0),
                "status": allocation.status,
                "risk_adjusted_score": float(allocation.risk_adjusted_score or 0.0),
                "scores": allocation.scores,
                "constraints": allocation.constraints,
            }
            for allocation in cycle.exchange_allocations
        ]
        transfers = [
            {
                "id": transfer.id,
                "allocation_id": transfer.allocation_id,
                "provider": transfer.provider,
                "direction": transfer.direction,
                "transfer_type": transfer.transfer_type,
                "asset": transfer.asset,
                "network": transfer.network,
                "requested_amount": float(transfer.requested_amount or 0.0),
                "confirmed_amount": float(transfer.confirmed_amount or 0.0),
                "fee_amount": float(transfer.fee_amount or 0.0),
                "status": transfer.status,
                "provider_reference": transfer.provider_reference,
                "failure_reason": transfer.failure_reason,
                "created_at": transfer.created_at.isoformat() if transfer.created_at else None,
                "confirmed_at": transfer.confirmed_at.isoformat() if transfer.confirmed_at else None,
            }
            for transfer in cycle.vault_transfers
        ]
        risk_events = [
            {
                "id": event.id,
                "allocation_id": event.allocation_id,
                "transfer_id": event.transfer_id,
                "provider": event.provider,
                "category": event.category,
                "severity": event.severity,
                "rule_name": event.rule_name,
                "reason": event.reason,
                "status": event.status,
                "created_at": event.created_at.isoformat() if event.created_at else None,
            }
            for event in cycle.vault_risk_events
        ]
        payload = {
            "cycle_id": cycle.id,
            "status": cycle.status,
            "execution_substatus": cycle.execution_substatus,
            "algorithm_profile": cycle.algorithm_profile,
            "deposit_asset": cycle.deposit_asset,
            "deposit_amount": float(cycle.deposit_amount or 0.0),
            "settlement_asset": cycle.settlement_asset,
            "starting_value_usd": float(cycle.starting_value_usd or 0.0),
            "current_estimated_value_usd": float(cycle.current_estimated_value_usd or 0.0),
            "final_settlement_amount": float(cycle.final_settlement_amount or 0.0),
            "started_at": cycle.started_at.isoformat() if cycle.started_at else None,
            "unlocks_at": cycle.unlocks_at.isoformat() if cycle.unlocks_at else None,
            "settled_at": cycle.settled_at.isoformat() if cycle.settled_at else None,
            "metadata": cycle.selection_metadata,
            "active_trading_enforcement": cycle.selection_metadata.get("active_trading_enforcement", {}),
            "allocations": allocations,
            "transfers": transfers,
            "risk_events": risk_events,
            "settlement": {
                "id": settlement.id if settlement else None,
                "status": settlement.status if settlement else "not_started",
                "final_amount": float(settlement.final_amount or 0.0) if settlement else 0.0,
                "final_value_usd": float(settlement.final_value_usd or 0.0) if settlement else 0.0,
                "net_pnl_usd": float(settlement.net_pnl_usd or 0.0) if settlement else 0.0,
                "fees_usd": float(settlement.fees_usd or 0.0) if settlement else 0.0,
                "roi_pct": float(settlement.roi_pct or 0.0) if settlement else 0.0,
                "failure_reason": settlement.failure_reason if settlement else None,
                "details": settlement.details if settlement else {},
            },
        }
        payload.update(extract_cycle_coherence_payload(cycle))
        return payload
