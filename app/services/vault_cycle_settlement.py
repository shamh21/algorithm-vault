"""End-of-cycle closeout and settlement orchestration for Vault Cycles."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from flask import current_app, has_app_context

from ..extensions import db
from ..models import (
    Fill,
    Order,
    VaultCycle,
    VaultCycleAllocation,
    VaultCycleSettlement,
    VaultCycleTrade,
    WalletBalance,
    WalletTransaction,
)
from .invite_profit_share import InviteProfitShareError


class VaultCycleSettlementService:
    """Closes trading, reconciles PnL, and settles confirmed funds to the wallet."""

    def __init__(self, config: dict[str, Any], trading_connections: Any, strategy_manager: Any, transfer_service: Any) -> None:
        self.config = config
        self.trading_connections = trading_connections
        self.strategy_manager = strategy_manager
        self.transfer_service = transfer_service

    def settle_cycle(self, cycle: VaultCycle) -> dict[str, Any]:
        if not self.is_vault_cycle_engine_cycle(cycle):
            return {"handled": False}
        settlement = self._settlement_for(cycle)
        if settlement.status == "complete":
            return {"handled": True, "status": "complete", "settlement_id": settlement.id}

        cycle.status = "settling"
        cycle.execution_substatus = "closing_positions"
        settlement.status = "closing_positions"
        self._stop_strategy_runs(cycle)
        try:
            self._flatten_allocations(cycle)
        except Exception as exc:  # noqa: BLE001
            self._pause_recovery(cycle, settlement, "closing_positions_failed", str(exc))
            raise

        self.sync_trade_links(cycle)
        gross_value = self._gross_cycle_value(cycle)
        cycle.current_estimated_value_usd = gross_value
        allocation_values = self._allocation_settlement_values(cycle, gross_value)
        pending: list[str] = []
        confirmed_total = 0.0
        fee_total = 0.0

        cycle.execution_substatus = "converting"
        settlement.status = "converting"
        for allocation, amount in allocation_values:
            try:
                conversion = self.transfer_service.convert_allocation_to_settlement(cycle, allocation, amount)
            except Exception as exc:  # noqa: BLE001
                self._pause_recovery(cycle, settlement, "conversion_failed", str(exc))
                raise
            if conversion is not None:
                self.transfer_service.refresh_transfer(conversion)
                if conversion.status not in {"confirmed", "complete"}:
                    pending.append(f"conversion:{conversion.id}:{conversion.status}")

        cycle.execution_substatus = "withdrawing"
        settlement.status = "withdrawing"
        for allocation, amount in allocation_values:
            try:
                withdrawal = self.transfer_service.withdraw_allocation_to_vault(cycle, allocation, amount)
            except Exception as exc:  # noqa: BLE001
                self._pause_recovery(cycle, settlement, "withdrawal_failed", str(exc))
                raise
            self.transfer_service.refresh_transfer(withdrawal)
            fee_total += float(withdrawal.fee_amount or 0.0)
            if withdrawal.status in {"confirmed", "complete"}:
                confirmed_total += float(withdrawal.confirmed_amount or withdrawal.requested_amount or 0.0)
                allocation.status = "settled"
                allocation.completed_at = datetime.utcnow()
            else:
                pending.append(f"withdrawal:{withdrawal.id}:{withdrawal.status}")

        settlement.gross_value_usd = gross_value
        settlement.fees_usd = fee_total
        settlement.gross_pnl_usd = gross_value - float(cycle.starting_value_usd or 0.0)
        settlement.net_pnl_usd = settlement.gross_pnl_usd - fee_total
        settlement.roi_pct = (settlement.net_pnl_usd / max(float(cycle.starting_value_usd or 0.0), 1e-9)) * 100.0
        settlement.details = {
            **settlement.details,
            "allocation_values": [
                {"allocation_id": allocation.id, "provider": allocation.provider, "amount": amount}
                for allocation, amount in allocation_values
            ],
            "pending": pending,
        }

        if pending:
            cycle.execution_substatus = "settlement_pending_recovery"
            settlement.status = "pending_confirmation"
            settlement.withdrawal_status = "pending_confirmation"
            return {"handled": True, "status": "pending_confirmation", "pending": pending, "settlement_id": settlement.id}

        self._credit_wallet(cycle, settlement, confirmed_total)
        return {"handled": True, "status": "complete", "settlement_id": settlement.id}

    def sync_trade_links(self, cycle: VaultCycle) -> list[VaultCycleTrade]:
        allocations = {allocation.trading_connection_id: allocation for allocation in cycle.exchange_allocations}
        linked: list[VaultCycleTrade] = []
        orders = Order.query.filter_by(user_id=cycle.user_id, vault_cycle_id=cycle.id).order_by(Order.created_at.asc()).all()
        for order in orders:
            allocation = allocations.get(order.trading_connection_id)
            fills: list[Fill | None] = list(order.fills) or [None]
            for fill in fills:
                existing = VaultCycleTrade.query.filter_by(
                    vault_cycle_id=cycle.id,
                    order_id=order.id,
                    fill_id=fill.id if fill is not None else None,
                ).one_or_none()
                if existing is not None:
                    linked.append(existing)
                    continue
                quantity = float((fill.quantity if fill is not None else order.filled_quantity) or 0.0)
                price = float((fill.price if fill is not None else order.average_fill_price) or 0.0)
                trade = VaultCycleTrade(
                    vault_cycle_id=cycle.id,
                    allocation_id=allocation.id if allocation is not None else None,
                    order_id=order.id,
                    fill_id=fill.id if fill is not None else None,
                    user_id=cycle.user_id,
                    trading_connection_id=order.trading_connection_id,
                    provider=allocation.provider if allocation is not None else str(order.details.get("provider") or ""),
                    symbol=order.symbol,
                    side=order.side,
                    status=order.status,
                    quantity=quantity,
                    notional_usd=abs(quantity * price),
                    fee_usd=float(fill.fee or 0.0) if fill is not None else 0.0,
                    realized_pnl_usd=float(fill.pnl or 0.0) if fill is not None else 0.0,
                )
                trade.details = {"client_order_id": order.client_order_id, "exchange_order_id": order.exchange_order_id}
                db.session.add(trade)
                linked.append(trade)
        return linked

    @staticmethod
    def is_vault_cycle_engine_cycle(cycle: VaultCycle) -> bool:
        metadata = cycle.selection_metadata if cycle is not None else {}
        return str(cycle.algorithm_profile or "").lower() == "vaultcycle" or bool(metadata.get("vault_cycle_engine"))

    def _settlement_for(self, cycle: VaultCycle) -> VaultCycleSettlement:
        settlement = VaultCycleSettlement.query.filter_by(vault_cycle_id=cycle.id).one_or_none()
        if settlement is None:
            settlement = VaultCycleSettlement(
                vault_cycle_id=cycle.id,
                user_id=cycle.user_id,
                settlement_asset=cycle.settlement_asset,
                status="pending",
                starting_value_usd=float(cycle.starting_value_usd or 0.0),
            )
            db.session.add(settlement)
            db.session.flush()
        return settlement

    def _stop_strategy_runs(self, cycle: VaultCycle) -> None:
        run_ids = {leg.strategy_run_id for leg in cycle.allocation_legs if leg.strategy_run_id}
        if cycle.strategy_run_id:
            run_ids.add(cycle.strategy_run_id)
        for run_id in sorted(run_id for run_id in run_ids if run_id):
            self.strategy_manager.stop(run_id)

    def _flatten_allocations(self, cycle: VaultCycle) -> None:
        for allocation in cycle.exchange_allocations:
            connector = self.trading_connections.connector_for_user(cycle.user_id, allocation.trading_connection_id)
            cancel_results = connector.cancel_all_orders(cycle.execution_mode)
            flatten_results = connector.flatten_all_positions(cycle.execution_mode)
            positions = connector.get_positions(cycle.execution_mode)
            open_positions = [row for row in positions if abs(float(row.get("quantity", 0.0) or 0.0)) > 1e-9]
            if open_positions:
                raise RuntimeError(f"{allocation.provider} positions remain open after flatten attempt")
            self.transfer_service.audit(
                cycle,
                "allocation_flattened",
                f"Closed open orders and positions on {allocation.provider}.",
                allocation,
                metadata={"cancel_results": cancel_results, "flatten_results": flatten_results},
            )

    def _gross_cycle_value(self, cycle: VaultCycle) -> float:
        orders = Order.query.filter_by(user_id=cycle.user_id, vault_cycle_id=cycle.id).all()
        realized = 0.0
        fees = 0.0
        for order in orders:
            for fill in order.fills:
                realized += float(fill.pnl or 0.0)
                fees += float(fill.fee or 0.0) + float(getattr(fill, "funding_fee", 0.0) or 0.0)
        if orders:
            return max(float(cycle.starting_value_usd or 0.0) + realized - fees, 0.0)
        return max(float(cycle.current_estimated_value_usd or cycle.starting_value_usd or 0.0), 0.0)

    def _allocation_settlement_values(self, cycle: VaultCycle, gross_value: float) -> list[tuple[VaultCycleAllocation, float]]:
        allocations = list(cycle.exchange_allocations)
        if not allocations:
            return []
        weight_total = sum(float(allocation.allocation_weight or 0.0) for allocation in allocations)
        if weight_total <= 0:
            equal = 1.0 / max(len(allocations), 1)
            return [(allocation, gross_value * equal) for allocation in allocations]
        return [
            (allocation, gross_value * (float(allocation.allocation_weight or 0.0) / weight_total))
            for allocation in allocations
        ]

    def _credit_wallet(self, cycle: VaultCycle, settlement: VaultCycleSettlement, confirmed_total: float) -> None:
        if WalletTransaction.query.filter_by(
            user_id=cycle.user_id,
            vault_cycle_id=cycle.id,
            transaction_type="settlement",
            status="complete",
        ).first() is not None:
            return
        credit_amount = float(confirmed_total or 0.0)
        treasury_deductions: dict[str, Any] | None = None
        if has_app_context() and bool(self.config.get("PLATFORM_GAS_TREASURY_ENABLED", False)):
            treasury_deductions = current_app.extensions["services"]["platform_treasury"].apply_vault_settlement_deductions(
                cycle,
                settlement,
                confirmed_total,
            )
            credit_amount = float(treasury_deductions.get("user_credit_amount", credit_amount) or 0.0)
        invite_deductions: dict[str, Any] | None = None
        if has_app_context():
            invite_profit_share = current_app.extensions.get("services", {}).get("invite_profit_share")
            if invite_profit_share is not None:
                try:
                    invite_deductions = invite_profit_share.process_cycle(
                        cycle,
                        settlement,
                        available_credit_amount=credit_amount,
                    )
                except InviteProfitShareError as exc:
                    self._pause_recovery(cycle, settlement, "invite_profit_share_failed", str(exc))
                    return
                if invite_deductions.get("applied"):
                    credit_amount = max(0.0, credit_amount - float(invite_deductions.get("payout_amount", 0.0) or 0.0))
        deposit_balance = WalletBalance.query.filter_by(user_id=cycle.user_id, asset=cycle.deposit_asset).one_or_none()
        if deposit_balance is not None:
            deposit_balance.locked_balance = max(0.0, float(deposit_balance.locked_balance or 0.0) - float(cycle.deposit_amount or 0.0))
        settlement_balance = WalletBalance.query.filter_by(user_id=cycle.user_id, asset=cycle.settlement_asset).one_or_none()
        if settlement_balance is None:
            settlement_balance = WalletBalance(user_id=cycle.user_id, asset=cycle.settlement_asset)
            db.session.add(settlement_balance)
        settlement_balance.available_balance = float(settlement_balance.available_balance or 0.0) + credit_amount
        settlement_balance.estimated_usd_value = settlement_balance.total_balance if cycle.settlement_asset in {"USDC", "USDT"} else settlement_balance.estimated_usd_value
        cycle.final_settlement_amount = credit_amount
        cycle.current_estimated_value_usd = credit_amount if cycle.settlement_asset in {"USDC", "USDT"} else settlement.final_value_usd
        cycle.status = "complete"
        cycle.execution_substatus = "complete"
        cycle.settled_at = datetime.utcnow()
        settlement.final_amount = credit_amount
        settlement.final_value_usd = credit_amount if cycle.settlement_asset in {"USDC", "USDT"} else settlement.final_value_usd
        if cycle.settlement_asset in {"USDC", "USDT"}:
            treasury_total = 0.0
            if treasury_deductions is not None:
                treasury_total += float(treasury_deductions.get("gas_reserve_asset", 0.0) or 0.0)
                treasury_total += float(treasury_deductions.get("profit_share_asset", 0.0) or 0.0)
            if invite_deductions is not None and invite_deductions.get("applied"):
                treasury_total += float(invite_deductions.get("payout_amount", 0.0) or 0.0)
            settlement.fees_usd = float(settlement.fees_usd or 0.0) + treasury_total
            settlement.net_pnl_usd = float(settlement.gross_pnl_usd or 0.0) - float(settlement.fees_usd or 0.0)
        settlement.status = "complete"
        settlement.withdrawal_status = "confirmed"
        settlement.completed_at = cycle.settled_at
        settlement.details = {
            **settlement.details,
            "credited_wallet": True,
            "credited_amount": credit_amount,
            "invite_profit_share": invite_deductions,
        }
        db.session.add(
            WalletTransaction(
                vault_cycle_id=cycle.id,
                user_id=cycle.user_id,
                asset=cycle.settlement_asset,
                amount=credit_amount,
                transaction_type="settlement",
                status="complete",
                note="Vault Cycle settled after confirmed exchange withdrawals.",
            )
        )
        self.transfer_service.audit(cycle, "cycle_settled", f"Vault Cycle settled to {credit_amount:.6f} {cycle.settlement_asset}.")

    def _pause_recovery(self, cycle: VaultCycle, settlement: VaultCycleSettlement, status: str, reason: str) -> None:
        cycle.status = "settling"
        cycle.execution_substatus = "settlement_pending_recovery"
        settlement.status = "pending_recovery"
        settlement.failure_reason = reason
        settlement.details = {**settlement.details, "recovery_status": status, "failure_reason": reason}
