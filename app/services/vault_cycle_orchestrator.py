"""Broader Vault Cycle orchestration for allocation, funding, trading, and recovery."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from flask import current_app, has_app_context

from ..extensions import db
from ..models import (
    AuditLog,
    Setting,
    StrategyRun,
    User,
    VaultAllocationLeg,
    VaultCycle,
    VaultCycleAllocation,
    WalletBalance,
    WalletTransaction,
)
from .worker_lease import in_process_workers_enabled


class VaultCycleOrchestrator:
    """Creates and resumes institutional-style Vault Cycle state machines."""

    IDEMPOTENCY_PREFIX = "vault_cycle_engine_idem"

    def __init__(
        self,
        config: dict[str, Any],
        trading_connections: Any,
        allocator: Any,
        transfer_service: Any,
        settlement_service: Any,
        strategy_selector: Any,
        strategy_manager: Any,
        trading_enforcer: Any | None = None,
    ) -> None:
        self.config = config
        self.trading_connections = trading_connections
        self.allocator = allocator
        self.transfer_service = transfer_service
        self.settlement_service = settlement_service
        self.strategy_selector = strategy_selector
        self.strategy_manager = strategy_manager
        self.trading_enforcer = trading_enforcer

    def start_cycle(
        self,
        *,
        user: User,
        amount: float,
        deposit_asset: str,
        settlement_asset: str,
        duration_seconds: int,
        providers: list[str] | None = None,
        allowed_symbols: list[str] | None = None,
        max_leverage: float | None = None,
        max_positions: int | None = None,
        idempotency_key: str | None = None,
        start_strategy_runs: bool = True,
    ) -> dict[str, Any]:
        self._validate_engine_ready()
        existing = self._existing_cycle(user.id, idempotency_key)
        if existing is not None:
            return {
                "cycle": existing,
                "created": False,
                "run_ids": [leg.strategy_run_id for leg in existing.allocation_legs if leg.strategy_run_id],
            }

        amount = max(0.0, float(amount or 0.0))
        deposit = str(deposit_asset or "").upper().strip()
        settlement = str(settlement_asset or deposit).upper().strip()
        if deposit not in {"USDC", "USDT"} or settlement not in {"USDC", "USDT"}:
            raise ValueError("Vault Cycle v1 supports USDC and USDT settlement currencies only.")
        if amount <= 0:
            raise ValueError("Vault Cycle amount must be greater than zero.")
        if duration_seconds <= 0:
            raise ValueError("Vault Cycle duration must be positive.")

        wallet_balance = WalletBalance.query.filter_by(user_id=user.id, asset=deposit).one_or_none()
        verified_spendable = self._verified_spendable_amount(user.id, deposit)
        if verified_spendable is not None:
            if verified_spendable + 1e-9 < amount:
                raise ValueError("Vault Cycle amount exceeds verified on-chain wallet balance.")
            self._materialize_onchain_surplus(user.id, deposit, amount)
            wallet_balance = WalletBalance.query.filter_by(user_id=user.id, asset=deposit).one_or_none()
        if wallet_balance is None or float(wallet_balance.available_balance or 0.0) + 1e-9 < amount:
            raise ValueError("Vault Cycle amount exceeds available wallet balance.")

        provider_filter = [
            str(provider).strip().lower()
            for provider in (providers or self.config.get("VAULT_CYCLE_PROVIDERS", []))
            if str(provider).strip()
        ]
        connections = self.trading_connections.enabled_tradable_connections(user.id, providers=provider_filter)
        if not connections:
            raise RuntimeError("No verified, active Hyperliquid or KuCoin connection is available for Vault Cycle.")

        plans, blockers = self.allocator.allocate(
            user_id=user.id,
            amount_usd=amount,
            settlement_asset=settlement,
            connections=connections,
            allowed_symbols=allowed_symbols,
            provider_filter=provider_filter,
        )
        if not plans:
            raise RuntimeError(self._allocation_failure_message(blockers))

        now = datetime.utcnow()
        wallet_balance.available_balance = float(wallet_balance.available_balance or 0.0) - amount
        wallet_balance.locked_balance = float(wallet_balance.locked_balance or 0.0) + amount
        wallet_balance.estimated_usd_value = (
            wallet_balance.total_balance if deposit in {"USDC", "USDT"} else wallet_balance.estimated_usd_value
        )

        cycle = VaultCycle(
            user_id=user.id,
            trading_connection_id=plans[0].connection.id,
            deposit_asset=deposit,
            deposit_amount=amount,
            settlement_asset=settlement,
            lock_duration_hours=max(1, int((duration_seconds + 3599) // 3600)),
            lock_duration_seconds=duration_seconds,
            status="active",
            execution_substatus="allocating",
            execution_mode="live",
            live_validation_status="passed",
            algorithm_profile="VaultCycle",
            selected_strategy_name="vault_cycle_allocator",
            selected_timeframe=str(self.config.get("DEFAULT_TIMEFRAME", "15m")),
            started_at=now,
            unlocks_at=now + timedelta(seconds=duration_seconds),
            starting_value_usd=amount,
            current_estimated_value_usd=amount,
        )
        cycle.selection_metadata = {
            "vault_cycle_engine": True,
            "state_machine": "initialized",
            "idempotency_key": idempotency_key or "",
            "provider_filter": provider_filter,
            "allocation_blockers": blockers,
            "allowed_symbols": allowed_symbols or [],
            "max_leverage": max_leverage,
            "max_positions": max_positions,
        }
        db.session.add(cycle)
        db.session.flush()

        allocations: list[VaultCycleAllocation] = []
        for plan in plans:
            constraints = dict(plan.constraints)
            if max_leverage is not None:
                constraints["requested_max_leverage"] = float(max_leverage)
            if max_positions is not None:
                constraints["requested_max_positions"] = int(max_positions)
            allocation = VaultCycleAllocation(
                vault_cycle_id=cycle.id,
                user_id=user.id,
                trading_connection_id=plan.connection.id,
                provider=plan.provider,
                settlement_asset=plan.settlement_asset,
                collateral_asset=plan.collateral_asset,
                target_amount=plan.target_amount,
                allocated_amount=plan.target_amount,
                allocation_weight=plan.allocation_weight,
                status="pending_funding",
                risk_adjusted_score=float(plan.scores.get("risk_adjusted_score", 0.0) or 0.0),
                opportunity_score=float(plan.scores.get("opportunity_score", 0.0) or 0.0),
                liquidity_score=float(plan.scores.get("liquidity_score", 0.0) or 0.0),
                slippage_bps=float(plan.scores.get("spread_bps", 0.0) or 0.0),
                max_leverage=float(plan.scores.get("max_leverage", self.config.get("MAX_LEVERAGE", 1.0)) or 1.0),
                max_symbol_exposure_usd=float(plan.target_amount)
                * float(self.config.get("VAULT_CYCLE_MAX_SYMBOL_ALLOCATION_PCT", 0.5) or 0.5),
                max_concurrent_positions=max(1, int(max_positions or constraints.get("max_concurrent_positions", 1) or 1)),
            )
            allocation.scores = plan.scores
            allocation.constraints = constraints
            db.session.add(allocation)
            allocations.append(allocation)
        db.session.flush()

        cycle.execution_substatus = "funding_exchange"
        metadata = cycle.selection_metadata
        metadata["state_machine"] = "funding_exchange"
        cycle.selection_metadata = metadata
        for allocation in allocations:
            self.transfer_service.prepare_allocation_funding(cycle, allocation)
            self.transfer_service.reserve_allocation(cycle, allocation)

        cycle.execution_substatus = "trading"
        run_ids = self._create_strategy_runs(cycle, allocations, allowed_symbols=allowed_symbols)
        metadata = cycle.selection_metadata
        metadata["state_machine"] = "trading"
        metadata["allocation_count"] = len(allocations)
        metadata["run_ids"] = run_ids
        metadata["provider_allocation_history"] = [
            {
                "provider": allocation.provider,
                "allocation_usd": allocation.allocated_amount,
                "weight": allocation.allocation_weight,
                "score": allocation.risk_adjusted_score,
                "collateral_asset": allocation.collateral_asset,
                "status": allocation.status,
            }
            for allocation in allocations
        ]
        metadata["exchange_allocation_history"] = metadata["provider_allocation_history"]
        cycle.selection_metadata = metadata
        cycle.selected_strategy_name = "vault_cycle_multi_strategy"
        if self.trading_enforcer is not None:
            self.trading_enforcer.mark_cycle_started(cycle)
        db.session.add(
            WalletTransaction(
                vault_cycle_id=cycle.id,
                user_id=user.id,
                asset=deposit,
                amount=amount,
                transaction_type="allocation",
                status="complete",
                note="Vault Cycle allocation locked and exchange reserves confirmed.",
            )
        )
        self._audit(cycle, "cycle_started", f"Vault Cycle started with {len(allocations)} exchange allocation(s).")
        if idempotency_key:
            Setting.set_json(self._idempotency_key(user.id, idempotency_key), {"cycle_id": cycle.id})

        if start_strategy_runs:
            self._start_or_queue_strategy_runs(run_ids)
        return {"cycle": cycle, "created": True, "run_ids": run_ids}

    def _start_or_queue_strategy_runs(self, run_ids: list[int]) -> None:
        if in_process_workers_enabled(self.config):
            for run_id in run_ids:
                self.strategy_manager.start(run_id)
            return
        queued_ids: list[int] = []
        for run_id in dict.fromkeys(run_ids):
            run = db.session.get(StrategyRun, int(run_id))
            if run is None:
                continue
            run.manual_enabled = True
            if run.status not in {"running", "starting"}:
                run.status = "queued"
            queued_ids.append(int(run.id))
        if queued_ids:
            audit = AuditLog(
                category="worker",
                action="vault_cycle_runs_queued_for_worker",
                message="Vault Cycle strategy runs queued for dedicated worker startup.",
            )
            audit.details = {"run_ids": queued_ids}
            db.session.add(audit)

    def resume_due_cycles(self, user_id: int | None = None) -> list[dict[str, Any]]:
        query = VaultCycle.query.filter(VaultCycle.status.in_(["active", "settling"]))
        if user_id is not None:
            query = query.filter_by(user_id=int(user_id))
        now = datetime.utcnow()
        results: list[dict[str, Any]] = []
        for cycle in query.filter(VaultCycle.unlocks_at <= now).order_by(VaultCycle.unlocks_at.asc()).all():
            if not self.settlement_service.is_vault_cycle_engine_cycle(cycle):
                continue
            results.append(self.settlement_service.settle_cycle(cycle))
        return results

    def _create_strategy_runs(
        self,
        cycle: VaultCycle,
        allocations: list[VaultCycleAllocation],
        *,
        allowed_symbols: list[str] | None,
    ) -> list[int]:
        run_ids: list[int] = []
        duration_hours = max(1, int(cycle.lock_duration_hours or 1))
        for index, allocation in enumerate(allocations):
            selection = self.strategy_selector.select(
                cycle.deposit_asset,
                duration_hours,
                "live",
                allocation.allocated_amount,
                allowed_symbols=allowed_symbols,
                provider=allocation.provider,
            )
            params = dict(selection.parameters or {})
            scores = allocation.scores
            params.update(
                {
                    "vault_cycle_id": cycle.id,
                    "vault_cycle_allocation_id": allocation.id,
                    "consumer_vault": True,
                    "vault_cycle_engine": True,
                    "algorithm_profile": "VaultCycle",
                    "vault_cycle_name": "Vault Cycle",
                    "settlement_asset": cycle.settlement_asset,
                    "collateral_asset": allocation.collateral_asset,
                    "allocation_cap_usd": float(allocation.allocated_amount or 0.0),
                    "allocation_weight": float(allocation.allocation_weight or 0.0),
                    "provider": allocation.provider,
                    "execution_venue": allocation.provider,
                    "trading_connection_id": allocation.trading_connection_id,
                    "user_id": cycle.user_id,
                    "risk_adjusted_score": allocation.risk_adjusted_score,
                    "scanner_score_breakdown": scores,
                    "max_concurrent_positions": allocation.max_concurrent_positions,
                    "lock_duration_seconds": cycle.lock_duration_seconds,
                    "allowed_symbols": allowed_symbols or [],
                }
            )
            leverage_cap = min(
                float(params.get("leverage", 1.0) or 1.0),
                float(allocation.max_leverage or 1.0),
                float(self.config.get("MAX_LEVERAGE", 3.0) or 3.0),
            )
            params["leverage"] = max(1.0, leverage_cap)
            run = StrategyRun(
                strategy_name=str(selection.strategy_name),
                symbol=str(selection.symbol),
                timeframe=str(selection.timeframe),
                mode="live",
                user_id=cycle.user_id,
                trading_connection_id=allocation.trading_connection_id,
                status="starting",
                lock_duration_seconds=cycle.lock_duration_seconds,
                manual_enabled=True,
            )
            run.parameters = params
            db.session.add(run)
            db.session.flush()
            leg = VaultAllocationLeg(
                vault_cycle_id=cycle.id,
                strategy_run_id=run.id,
                symbol=run.symbol,
                timeframe=run.timeframe,
                provider=allocation.provider,
                trading_connection_id=allocation.trading_connection_id,
                allocation_cap_usd=float(allocation.allocated_amount or 0.0),
                leverage=params["leverage"],
                status="active",
            )
            leg.details = {
                "vault_cycle_allocation_id": allocation.id,
                "provider": allocation.provider,
                "execution_venue": allocation.provider,
                "settlement_asset": cycle.settlement_asset,
                "collateral_asset": allocation.collateral_asset,
                "allocation_weight": allocation.allocation_weight,
                "allocation_mode": "vault_cycle_dynamic",
                "risk_adjusted_score": allocation.risk_adjusted_score,
                "scanner_score_breakdown": scores,
                "best_symbol": scores.get("best_symbol"),
                "best_market_id": scores.get("best_market_id"),
            }
            db.session.add(leg)
            db.session.flush()
            params["vault_leg_id"] = leg.id
            run.parameters = params
            if index == 0:
                cycle.strategy_run_id = run.id
            run_ids.append(run.id)
        return run_ids

    def _validate_engine_ready(self) -> None:
        blockers = []
        if not bool(self.config.get("VAULT_CYCLE_ENGINE_ENABLED", False)):
            blockers.append("VAULT_CYCLE_ENGINE_ENABLED is disabled")
        blockers.extend(self.transfer_service.funding_blockers())
        if blockers:
            raise RuntimeError("; ".join(blockers))

    def _existing_cycle(self, user_id: int, idempotency_key: str | None) -> VaultCycle | None:
        if not idempotency_key:
            return None
        payload = Setting.get_json(self._idempotency_key(user_id, idempotency_key), {})
        cycle_id = payload.get("cycle_id") if isinstance(payload, dict) else None
        if not cycle_id:
            return None
        return VaultCycle.query.filter_by(user_id=user_id, id=int(cycle_id)).one_or_none()

    def _idempotency_key(self, user_id: int, idempotency_key: str) -> str:
        return f"{self.IDEMPOTENCY_PREFIX}:{int(user_id)}:{str(idempotency_key).strip()[:80]}"

    @staticmethod
    def _allocation_failure_message(blockers: list[dict[str, Any]]) -> str:
        if not blockers:
            return "No exchange allocation passed Vault Cycle screening."
        reasons = [str(item.get("reason") or item) for item in blockers[:3]]
        return "No exchange allocation passed Vault Cycle screening: " + "; ".join(reasons)

    def _verified_spendable_amount(self, user_id: int, asset: str) -> float | None:
        if not has_app_context():
            return None
        network = self._default_network(asset)
        try:
            custody = current_app.extensions.get("services", {}).get("wallet_custody")
            if custody is None or not getattr(custody, "enabled", False) or not custody.supports(asset, network):
                return None
            return float(custody.verified_spendable_amount(user_id, asset, network) or 0.0)
        except Exception as exc:  # noqa: BLE001
            current_app.logger.warning("Vault Cycle on-chain spendable check failed closed for %s/%s: %s", asset, network, exc)
            return 0.0

    def _materialize_onchain_surplus(self, user_id: int, asset: str, amount: float) -> None:
        if not has_app_context():
            return
        network = self._default_network(asset)
        custody = current_app.extensions.get("services", {}).get("wallet_custody")
        if custody is None or not getattr(custody, "enabled", False) or not custody.supports(asset, network):
            return
        custody.materialize_onchain_surplus(user_id, asset, network, amount)

    @staticmethod
    def _default_network(asset: str) -> str:
        asset_key = str(asset or "").upper().strip()
        if asset_key == "BTC":
            return "Bitcoin"
        if asset_key == "SOL":
            return "Solana"
        if asset_key == "XRP":
            return "XRP Ledger"
        return "Ethereum"

    def _audit(self, cycle: VaultCycle, action: str, message: str) -> None:
        entry = AuditLog(
            user_id=cycle.user_id,
            trading_connection_id=cycle.trading_connection_id,
            category="vault_cycle",
            action=action,
            message=message,
        )
        entry.details = {"vault_cycle_id": cycle.id, "metadata": cycle.selection_metadata}
        db.session.add(entry)
