"""Active trading supervision for Vault Cycle engine cycles."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import or_

from ..extensions import db
from ..models import AuditLog, Order, Setting, StrategyRun, VaultAllocationLeg, VaultCycle, VaultCycleAllocation, VaultCycleRiskEvent
from .db_retry import commit_with_retry
from .provider_assets import normalize_provider
from .worker_lease import in_process_workers_enabled


ACTIVE_ORDER_STATUSES = {"pending", "submitted", "open"}
ACCEPTED_ORDER_STATUSES = {"pending", "submitted", "open", "filled"}
REJECTED_ORDER_STATUSES = {"rejected", "failed"}


class VaultCycleTradingEnforcer:
    """Keeps funded Vault Cycle engine allocations actively supervised."""

    def __init__(
        self,
        config: dict[str, Any],
        trading_connections: Any,
        strategy_selector: Any,
        strategy_manager: Any,
        order_manager: Any,
        leveraged_markets: Any,
        market_data: Any,
        settlement_service: Any,
    ) -> None:
        self.config = config
        self.trading_connections = trading_connections
        self.strategy_selector = strategy_selector
        self.strategy_manager = strategy_manager
        self.order_manager = order_manager
        self.leveraged_markets = leveraged_markets
        self.market_data = market_data
        self.settlement_service = settlement_service

    def mark_cycle_started(self, cycle: VaultCycle) -> None:
        """Persist initial enforcement metadata during cycle creation."""

        if not self._enabled():
            return
        metadata = dict(cycle.selection_metadata or {})
        metadata["active_trading_enforcement"] = {
            **self._config_payload(),
            "status": "initialized",
            "started_at": datetime.utcnow().isoformat(),
            "last_tick_at": None,
            "last_rescreen_at": None,
            "rotation_history": [],
        }
        cycle.selection_metadata = metadata
        for leg in list(cycle.allocation_legs or []):
            details = dict(leg.details or {})
            details.setdefault("active_trading_enforced", True)
            details.setdefault("strategy_activation_reason", "vault_cycle_start")
            leg.details = details

    def enforce_active_cycles(self, user_id: int | None = None) -> list[dict[str, Any]]:
        """Inspect active cycles and apply restart, re-screen, and rotation actions."""

        if not self._enabled():
            return []
        query = VaultCycle.query.filter_by(status="active")
        if user_id is not None:
            query = query.filter_by(user_id=int(user_id))
        now = datetime.utcnow()
        results: list[dict[str, Any]] = []
        run_ids_to_stop: set[int] = set()
        run_ids_to_start: set[int] = set()
        changed = False
        for cycle in query.order_by(VaultCycle.started_at.asc(), VaultCycle.id.asc()).all():
            if cycle.unlocks_at and cycle.unlocks_at <= now:
                continue
            if not self.settlement_service.is_vault_cycle_engine_cycle(cycle):
                continue
            result = self.enforce_cycle(cycle, now=now)
            results.append(result)
            changed = changed or bool(result.get("changed"))
            run_ids_to_stop.update(int(item) for item in result.get("stop_run_ids", []) if item)
            run_ids_to_start.update(int(item) for item in result.get("start_run_ids", []) if item)

        if changed:
            commit_with_retry()
        run_ids_to_start.difference_update(run_ids_to_stop)
        for run_id in sorted(run_ids_to_stop):
            self.strategy_manager.stop(run_id)
        self._start_or_queue_strategy_runs(sorted(run_ids_to_start))
        return results

    def _start_or_queue_strategy_runs(self, run_ids: list[int]) -> None:
        if not run_ids:
            return
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
                action="vault_cycle_enforcement_queued_for_worker",
                message="Vault Cycle enforcement queued strategy runs for dedicated worker startup.",
            )
            audit.details = {"run_ids": queued_ids}
            db.session.add(audit)
            commit_with_retry()

    def enforce_cycle(self, cycle: VaultCycle, *, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.utcnow()
        run_ids_to_start: set[int] = set()
        run_ids_to_stop: set[int] = set()
        changed = False
        metadata = dict(cycle.selection_metadata or {})
        enforcement = dict(metadata.get("active_trading_enforcement") or {})
        if not enforcement:
            self.mark_cycle_started(cycle)
            metadata = dict(cycle.selection_metadata or {})
            enforcement = dict(metadata.get("active_trading_enforcement") or {})
            changed = True

        linked = self.settlement_service.sync_trade_links(cycle)
        changed = changed or bool(linked)

        allocations = self._funded_allocations(cycle)
        restarted = self._ensure_strategy_runs(cycle, allocations, now=now)
        run_ids_to_start.update(restarted)
        changed = changed or bool(restarted)

        orders = self._cycle_orders(cycle)
        activity = self._activity_snapshot(cycle, orders, now=now)
        activity["restarted_run_ids"] = sorted(restarted)

        rescreen_due = self._rescreen_due(cycle, activity, enforcement, now)
        candidates: list[dict[str, Any]] = []
        rotations: list[dict[str, Any]] = []
        rebalances: list[dict[str, Any]] = []
        no_valid_setup = False

        if rescreen_due and allocations:
            candidates = self._rank_opportunities(cycle, allocations)
            if candidates:
                rotations, stop_ids, start_ids = self._rotate_idle_legs(cycle, allocations, candidates, now=now)
                run_ids_to_stop.update(stop_ids)
                run_ids_to_start.update(start_ids)
                changed = changed or bool(rotations)
                rebalances = self._soft_rebalance_allocations(cycle, allocations, candidates, now=now)
                changed = changed or bool(rebalances)
            else:
                no_valid_setup = True
                self._record_event(
                    cycle,
                    category="activity",
                    severity="info",
                    rule_name="no_valid_setup",
                    reason="No valid setup cleared Vault Cycle opportunity thresholds.",
                    metadata={"activity": activity},
                )
                self._audit(
                    cycle,
                    "active_trading_no_valid_setup",
                    "Vault Cycle remained idle because no valid setup cleared quality thresholds.",
                    metadata={"activity": activity},
                )
                changed = True

            enforcement["last_rescreen_at"] = now.isoformat()
            enforcement["last_rescreen_reason"] = activity.get("rescreen_reason")
            enforcement["last_candidate_count"] = len(candidates)
            enforcement["last_rotation_count"] = len(rotations)
            changed = True

        enforcement.update(
            {
                **self._config_payload(),
                "status": "monitoring",
                "last_tick_at": now.isoformat(),
                "last_activity_snapshot": activity,
                "last_candidates": candidates[:8],
                "last_rotations": rotations,
                "last_rebalances": rebalances,
                "no_valid_setup": no_valid_setup,
            }
        )
        if rotations:
            history = list(enforcement.get("rotation_history") or [])
            history.extend(rotations)
            enforcement["rotation_history"] = history[-20:]
        metadata["active_trading_enforcement"] = enforcement
        cycle.selection_metadata = metadata
        if activity.get("rescreen_reason"):
            cycle.execution_substatus = "trading_rescreened" if rescreen_due else cycle.execution_substatus
        changed = True
        return {
            "cycle_id": cycle.id,
            "changed": changed,
            "activity": activity,
            "rescreen_due": rescreen_due,
            "candidate_count": len(candidates),
            "rotations": rotations,
            "rebalances": rebalances,
            "start_run_ids": sorted(run_ids_to_start),
            "stop_run_ids": sorted(run_ids_to_stop),
        }

    def _ensure_strategy_runs(
        self,
        cycle: VaultCycle,
        allocations: list[VaultCycleAllocation],
        *,
        now: datetime,
    ) -> set[int]:
        run_ids: set[int] = set()
        stale_after = max(60.0, float(self.config.get("STRATEGY_POLL_SECONDS", 20) or 20) * 5.0)
        for leg in list(cycle.allocation_legs or []):
            if str(leg.status or "").lower() != "active":
                continue
            run = leg.strategy_run
            if run is None:
                allocation = self._allocation_for_leg(leg, allocations)
                if allocation is None:
                    continue
                run = self._create_strategy_run(cycle, allocation, leg=leg, reason="missing_strategy_run_recreated")
                params = dict(run.parameters or {})
                params["vault_leg_id"] = leg.id
                run.parameters = params
                leg.strategy_run_id = run.id
                run_ids.add(run.id)
                continue
            heartbeat = run.last_heartbeat_at
            heartbeat_stale = heartbeat is None or (now - heartbeat).total_seconds() > stale_after
            active = run.manual_enabled and str(run.status or "").lower() in {"starting", "running"} and not heartbeat_stale
            if active:
                continue
            run.manual_enabled = True
            run.status = "starting"
            run.last_error = None
            details = dict(leg.details or {})
            details["strategy_activation_reason"] = "active_trading_enforcer_restart"
            details["restarted_at"] = now.isoformat()
            leg.details = details
            self._audit(
                cycle,
                "active_trading_strategy_restarted",
                f"Restarted strategy run {run.id} for active Vault Cycle leg.",
                allocation=self._allocation_for_leg(leg, allocations),
                metadata={"run_id": run.id, "leg_id": leg.id},
            )
            run_ids.add(run.id)
        return run_ids

    def _rotate_idle_legs(
        self,
        cycle: VaultCycle,
        allocations: list[VaultCycleAllocation],
        candidates: list[dict[str, Any]],
        *,
        now: datetime,
    ) -> tuple[list[dict[str, Any]], set[int], set[int]]:
        rotations: list[dict[str, Any]] = []
        stop_ids: set[int] = set()
        start_ids: set[int] = set()
        min_score = self._min_opportunity_score()
        min_delta = self._rotation_score_delta()
        for allocation in allocations:
            provider_candidates = [
                item
                for item in candidates
                if int(item.get("allocation_id") or 0) == int(allocation.id) and float(item.get("score") or 0.0) >= min_score
            ]
            if not provider_candidates:
                continue
            best = provider_candidates[0]
            current_leg = self._active_leg_for_allocation(cycle, allocation)
            current_score = self._live_current_score(current_leg, allocation, provider_candidates)
            if current_leg is not None and str(current_leg.symbol or "").upper() == str(best.get("symbol") or "").upper():
                self._update_leg_candidate_state(current_leg, best, now)
                continue
            score_delta = float(best.get("score") or 0.0) - current_score
            if current_leg is not None and score_delta + 1e-9 < min_delta:
                self._update_leg_candidate_state(current_leg, best, now, skipped_reason="rotation_delta_below_threshold")
                continue
            if current_leg is not None and self._leg_has_open_exposure(current_leg):
                self._update_leg_candidate_state(current_leg, best, now, skipped_reason="open_exposure_blocks_rotation")
                continue

            new_run = self._create_strategy_run(cycle, allocation, candidate=best, reason="stronger_opportunity_rotation")
            new_leg = VaultAllocationLeg(
                vault_cycle_id=cycle.id,
                strategy_run_id=new_run.id,
                symbol=str(best.get("symbol") or ""),
                timeframe=str(best.get("timeframe") or "5m"),
                provider=allocation.provider,
                trading_connection_id=allocation.trading_connection_id,
                allocation_cap_usd=self._effective_allocation_cap(allocation),
                leverage=float(new_run.parameters.get("leverage", allocation.max_leverage or 1.0) or 1.0),
                status="active",
            )
            new_leg.details = {
                "vault_cycle_allocation_id": allocation.id,
                "provider": allocation.provider,
                "execution_venue": allocation.provider,
                "venue_symbol": best.get("venue_symbol"),
                "app_symbol": best.get("symbol"),
                "active_trading_enforced": True,
                "strategy_activation_reason": "stronger_opportunity_rotation",
                "opportunity_score": best.get("score"),
                "score_breakdown": best.get("score_breakdown", {}),
                "market_regime": best.get("market_regime"),
                "rotated_at": now.isoformat(),
            }
            db.session.add(new_leg)
            db.session.flush()
            params = dict(new_run.parameters or {})
            params["vault_leg_id"] = new_leg.id
            new_run.parameters = params
            if cycle.strategy_run_id is None or (current_leg is not None and cycle.strategy_run_id == current_leg.strategy_run_id):
                cycle.strategy_run_id = new_run.id

            old_run_id = current_leg.strategy_run_id if current_leg is not None else None
            old_leg_id = current_leg.id if current_leg is not None else None
            old_symbol = current_leg.symbol if current_leg is not None else ""
            if current_leg is not None:
                old_details = dict(current_leg.details or {})
                old_details.update(
                    {
                        "status_before_rotation": current_leg.status,
                        "rotation_reason": "stronger_opportunity_available",
                        "rotation_score_delta": score_delta,
                        "rotated_to_leg_id": new_leg.id,
                        "rotated_to_symbol": best.get("symbol"),
                        "rotated_at": now.isoformat(),
                    }
                )
                current_leg.details = old_details
                current_leg.status = "rotated"
                if current_leg.strategy_run is not None:
                    current_leg.strategy_run.manual_enabled = False
                    current_leg.strategy_run.status = "stopped"
                    current_leg.strategy_run.last_heartbeat_at = now
                if old_run_id:
                    stop_ids.add(int(old_run_id))

            rotation = {
                "allocation_id": allocation.id,
                "provider": allocation.provider,
                "old_leg_id": old_leg_id,
                "old_run_id": old_run_id,
                "old_symbol": old_symbol,
                "new_leg_id": new_leg.id,
                "new_run_id": new_run.id,
                "new_symbol": new_leg.symbol,
                "strategy_name": new_run.strategy_name,
                "score": float(best.get("score") or 0.0),
                "previous_score": current_score,
                "score_delta": score_delta,
                "rotated_at": now.isoformat(),
            }
            rotations.append(rotation)
            self._audit(
                cycle,
                "active_trading_leg_rotated",
                f"Rotated {allocation.provider} Vault Cycle leg from {old_symbol or 'none'} to {new_leg.symbol}.",
                allocation=allocation,
                metadata=rotation,
            )
            self._record_event(
                cycle,
                allocation=allocation,
                category="activity",
                severity="info",
                rule_name="leg_rotated",
                reason="Stronger executable opportunity cleared rotation threshold.",
                metadata=rotation,
            )
            start_ids.add(int(new_run.id))
        return rotations, stop_ids, start_ids

    def _soft_rebalance_allocations(
        self,
        cycle: VaultCycle,
        allocations: list[VaultCycleAllocation],
        candidates: list[dict[str, Any]],
        *,
        now: datetime,
    ) -> list[dict[str, Any]]:
        best_by_allocation: dict[int, dict[str, Any]] = {}
        for item in candidates:
            allocation_id = int(item.get("allocation_id") or 0)
            if allocation_id and allocation_id not in best_by_allocation:
                best_by_allocation[allocation_id] = item
        scored = [
            (allocation, float(best_by_allocation.get(allocation.id, {}).get("score") or 0.0))
            for allocation in allocations
        ]
        eligible = [(allocation, score) for allocation, score in scored if score >= self._min_opportunity_score()]
        if not eligible:
            return []
        score_sum = sum(score for _, score in eligible)
        total_allocated = sum(float(allocation.allocated_amount or 0.0) for allocation in allocations)
        rebalances: list[dict[str, Any]] = []
        for allocation, score in scored:
            constraints = dict(allocation.constraints or {})
            previous_cap = float(constraints.get("effective_allocation_cap_usd", allocation.allocated_amount or 0.0) or 0.0)
            if score_sum > 0 and score >= self._min_opportunity_score():
                desired = total_allocated * (score / score_sum)
                effective_cap = min(float(allocation.allocated_amount or 0.0), desired)
            else:
                effective_cap = min(float(allocation.allocated_amount or 0.0), float(allocation.allocated_amount or 0.0) * 0.25)
            constraints.update(
                {
                    "effective_allocation_cap_usd": max(0.0, effective_cap),
                    "soft_rebalance_score": score,
                    "soft_rebalance_updated_at": now.isoformat(),
                    "exchange_rebalance_enabled": bool(self.config.get("VAULT_CYCLE_EXCHANGE_REBALANCE_ENABLED", False)),
                }
            )
            allocation.constraints = constraints
            scores = dict(allocation.scores or {})
            scores["active_trading_opportunity_score"] = score
            scores["active_trading_rebalanced_at"] = now.isoformat()
            allocation.scores = scores
            row = {
                "allocation_id": allocation.id,
                "provider": allocation.provider,
                "score": score,
                "previous_effective_cap_usd": previous_cap,
                "effective_cap_usd": max(0.0, effective_cap),
            }
            rebalances.append(row)
        if rebalances:
            self._audit(
                cycle,
                "active_trading_soft_rebalance",
                "Updated effective Vault Cycle exchange deployment caps from live opportunity scores.",
                metadata={"rebalances": rebalances},
            )
        return rebalances

    def _rank_opportunities(self, cycle: VaultCycle, allocations: list[VaultCycleAllocation]) -> list[dict[str, Any]]:
        allowed_symbols = self._allowed_symbols(cycle)
        candidates: list[dict[str, Any]] = []
        for allocation in allocations:
            markets = self.leveraged_markets.active_markets(provider=allocation.provider, symbols=allowed_symbols)
            for market in markets:
                row = self._score_market(allocation, market)
                if row["score"] >= self._min_opportunity_score():
                    candidates.append(row)
        candidates.sort(
            key=lambda item: (
                float(item.get("score") or 0.0),
                float((item.get("score_breakdown") or {}).get("liquidity_score") or 0.0),
                str(item.get("symbol") or ""),
            ),
            reverse=True,
        )
        return candidates

    def _score_market(self, allocation: VaultCycleAllocation, market: Any) -> dict[str, Any]:
        symbol = str(getattr(market, "symbol", "") or "").upper()
        venue_symbol = str(getattr(market, "venue_symbol", "") or symbol)
        raw = getattr(market, "raw", {}) if hasattr(market, "raw") else {}
        features = self._live_market_features(venue_symbol or symbol)
        liquidity = max(float(getattr(market, "liquidity_usd", 0.0) or 0.0), float(features.get("liquidity_usd", 0.0) or 0.0))
        spread = float(features.get("spread_bps") or getattr(market, "spread_bps", 0.0) or 0.0)
        funding_rate = float(getattr(market, "funding_rate", 0.0) or 0.0)
        liquidity_score = min(math.log10(max(liquidity, 1.0)) / 8.0, 1.0)
        max_spread = max(float(self.config.get("VAULT_MAX_SPREAD_BPS", 25.0) or 25.0), 1.0)
        spread_quality = 1.0 - min(max(spread, 0.0) / max_spread, 1.0)
        funding_score = 1.0 - min(abs(funding_rate) * 100.0, 1.0)
        leverage_score = min(float(getattr(market, "max_leverage", 1.0) or 1.0) / max(float(self.config.get("MAX_LEVERAGE", 3.0) or 3.0), 1.0), 1.0)
        structure_score = self._bounded_float(raw.get("market_structure_score"), 0.55)
        ml_score = self._bounded_float(raw.get("ml_score", raw.get("rank_score")), 0.0)
        volatility_score = min(abs(float(features.get("volatility_pct", 0.0) or 0.0)) / 0.35, 1.0)
        momentum_score = min(abs(float(features.get("momentum_pct", 0.0) or 0.0)) / 1.5, 1.0)
        opportunity_base = max(
            0.0,
            min((liquidity_score * 0.55) + (leverage_score * 0.25) + (spread_quality * 0.20) - min(abs(funding_rate) * 100.0, 0.35), 1.0),
        )
        score = max(
            0.0,
            min(
                (opportunity_base * 0.30)
                + (liquidity_score * 0.18)
                + (spread_quality * 0.15)
                + (funding_score * 0.10)
                + (structure_score * 0.10)
                + (ml_score * 0.10)
                + (volatility_score * 0.04)
                + (momentum_score * 0.03),
                1.0,
            ),
        )
        regime = self._market_regime(features, liquidity_score)
        strategy_name, timeframe = self._strategy_for_regime(regime)
        return {
            "allocation_id": allocation.id,
            "provider": allocation.provider,
            "trading_connection_id": allocation.trading_connection_id,
            "symbol": symbol,
            "venue_symbol": venue_symbol,
            "market_id": getattr(market, "id", None),
            "score": score,
            "strategy_name": strategy_name,
            "timeframe": timeframe,
            "market_regime": regime,
            "score_breakdown": {
                "opportunity_base": opportunity_base,
                "liquidity_score": liquidity_score,
                "spread_quality_score": spread_quality,
                "funding_score": funding_score,
                "structure_score": structure_score,
                "ml_score": ml_score,
                "volatility_score": volatility_score,
                "momentum_score": momentum_score,
                "liquidity_usd": liquidity,
                "spread_bps": spread,
                "funding_rate": funding_rate,
            },
        }

    def _create_strategy_run(
        self,
        cycle: VaultCycle,
        allocation: VaultCycleAllocation,
        *,
        leg: VaultAllocationLeg | None = None,
        candidate: dict[str, Any] | None = None,
        reason: str,
    ) -> StrategyRun:
        symbol = str((candidate or {}).get("symbol") or (leg.symbol if leg is not None else "") or self._allocation_best_symbol(allocation) or "").upper()
        venue_symbol = str((candidate or {}).get("venue_symbol") or symbol)
        provider = normalize_provider(allocation.provider)
        duration_hours = max(1, int(cycle.lock_duration_hours or 1))
        selection = self.strategy_selector.select(
            cycle.deposit_asset,
            duration_hours,
            "live",
            allocation.allocated_amount,
            allowed_symbols=[symbol] if symbol else self._allowed_symbols(cycle),
            provider=provider,
        )
        strategy_name = str((candidate or {}).get("strategy_name") or selection.strategy_name)
        timeframe = str((candidate or {}).get("timeframe") or selection.timeframe)
        params = dict(selection.parameters or {})
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
                "allocation_cap_usd": self._effective_allocation_cap(allocation),
                "allocation_weight": float(allocation.allocation_weight or 0.0),
                "provider": provider,
                "execution_venue": provider,
                "trading_connection_id": allocation.trading_connection_id,
                "user_id": cycle.user_id,
                "venue_symbol": venue_symbol,
                "provider_symbol": venue_symbol,
                "app_symbol": symbol,
                "market_id": (candidate or {}).get("market_id"),
                "active_trading_enforced": True,
                "strategy_activation_reason": reason,
                "opportunity_score": (candidate or {}).get("score", allocation.opportunity_score),
                "scanner_score_breakdown": (candidate or {}).get("score_breakdown", allocation.scores),
                "lock_duration_seconds": cycle.lock_duration_seconds,
                "allowed_symbols": [symbol] if symbol else self._allowed_symbols(cycle),
            }
        )
        leverage_cap = min(
            float(params.get("leverage", 1.0) or 1.0),
            float(allocation.max_leverage or 1.0),
            float(self.config.get("MAX_LEVERAGE", 3.0) or 3.0),
        )
        params["leverage"] = max(1.0, leverage_cap)
        run = StrategyRun(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
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
        return run

    def _activity_snapshot(self, cycle: VaultCycle, orders: list[Order], *, now: datetime) -> dict[str, Any]:
        accepted = [order for order in orders if str(order.status or "").lower() in ACCEPTED_ORDER_STATUSES]
        rejected = [order for order in orders if str(order.status or "").lower() in REJECTED_ORDER_STATUSES]
        last_order_at = max((order.created_at for order in orders if order.created_at), default=None)
        last_heartbeat_at = max(
            (
                leg.strategy_run.last_heartbeat_at
                for leg in cycle.allocation_legs
                if leg.strategy_run is not None and leg.strategy_run.last_heartbeat_at is not None
            ),
            default=None,
        )
        idle_anchor = last_order_at or cycle.started_at or cycle.created_at or now
        idle_seconds = max((now - idle_anchor).total_seconds(), 0.0)
        accepted_notional = sum(self._order_notional(order) for order in accepted if not order.reduce_only)
        allocated = sum(float(allocation.allocated_amount or 0.0) for allocation in cycle.exchange_allocations) or float(cycle.starting_value_usd or 0.0)
        utilization = min(accepted_notional / max(allocated, 1e-9), 1.0)
        no_trade_reasons = list(
            dict.fromkeys(
                str((leg.strategy_run.last_signal.get("metadata") or {}).get("no_trade_reason") or leg.strategy_run.last_signal.get("rationale") or "")
                for leg in cycle.allocation_legs
                if leg.strategy_run is not None and isinstance(leg.strategy_run.last_signal, dict)
            )
        )
        no_trade_reasons = [reason for reason in no_trade_reasons if reason]
        provider_rows: list[dict[str, Any]] = []
        for allocation in cycle.exchange_allocations:
            provider_orders = [order for order in orders if order.trading_connection_id == allocation.trading_connection_id]
            provider_accepted = [order for order in provider_orders if str(order.status or "").lower() in ACCEPTED_ORDER_STATUSES]
            provider_notional = sum(self._order_notional(order) for order in provider_accepted if not order.reduce_only)
            provider_rows.append(
                {
                    "allocation_id": allocation.id,
                    "provider": allocation.provider,
                    "allocated_amount": float(allocation.allocated_amount or 0.0),
                    "accepted_order_attempts": len(provider_accepted),
                    "rejected_order_attempts": sum(1 for order in provider_orders if str(order.status or "").lower() in REJECTED_ORDER_STATUSES),
                    "utilization_pct": provider_notional / max(float(allocation.allocated_amount or 0.0), 1e-9),
                    "last_order_at": max((order.created_at.isoformat() for order in provider_orders if order.created_at), default=None),
                }
            )
        rescreen_reason = ""
        if idle_seconds >= self._max_idle_seconds():
            rescreen_reason = "max_idle_duration_exceeded"
        elif len(accepted) < self._min_trades_per_cycle() and idle_seconds >= self._rescreen_seconds():
            rescreen_reason = "minimum_trade_target_behind"
        elif utilization < self._target_utilization_pct() and idle_seconds >= self._rescreen_seconds():
            rescreen_reason = "capital_utilization_target_behind"
        return {
            "trade_count": len([order for order in accepted if not order.reduce_only]),
            "accepted_order_attempts": len(accepted),
            "rejected_order_attempts": len(rejected),
            "last_order_at": last_order_at.isoformat() if last_order_at else None,
            "last_signal_heartbeat_at": last_heartbeat_at.isoformat() if last_heartbeat_at else None,
            "idle_duration_seconds": idle_seconds,
            "accepted_notional_usd": accepted_notional,
            "capital_utilization_pct": utilization,
            "target_utilization_pct": self._target_utilization_pct(),
            "minimum_trades_per_cycle": self._min_trades_per_cycle(),
            "provider_activity": provider_rows,
            "no_trade_reasons": no_trade_reasons[:8],
            "rescreen_reason": rescreen_reason,
        }

    def _rescreen_due(
        self,
        cycle: VaultCycle,
        activity: dict[str, Any],
        enforcement: dict[str, Any],
        now: datetime,
    ) -> bool:
        if not activity.get("rescreen_reason"):
            return False
        last_rescreen = self._parse_datetime(enforcement.get("last_rescreen_at"))
        if last_rescreen is not None and (now - last_rescreen).total_seconds() < self._rescreen_seconds():
            return False
        if Setting.get_json("panic_lock", False):
            return False
        return cycle.unlocks_at is None or cycle.unlocks_at > now

    def _leg_has_open_exposure(self, leg: VaultAllocationLeg) -> bool:
        if Order.query.filter_by(vault_leg_id=leg.id).filter(Order.status.in_(list(ACTIVE_ORDER_STATUSES))).count() > 0:
            return True
        if leg.strategy_run is None:
            return False
        try:
            position = self.order_manager.current_position(
                leg.symbol,
                "live",
                leg.strategy_run.user_id,
                leg.trading_connection_id,
            )
        except Exception:  # noqa: BLE001
            return True
        try:
            return abs(float(position.get("quantity", 0.0) or 0.0)) > 1e-9
        except (TypeError, ValueError):
            return True

    def _cycle_orders(self, cycle: VaultCycle) -> list[Order]:
        legacy_with_space = f'%"vault_cycle_id": {int(cycle.id)}%'
        legacy_without_space = f'%"vault_cycle_id":{int(cycle.id)}%'
        return (
            Order.query.filter(
                Order.user_id == cycle.user_id,
                or_(
                    Order.vault_cycle_id == cycle.id,
                    Order.metadata_json.like(legacy_with_space),
                    Order.metadata_json.like(legacy_without_space),
                ),
            )
            .order_by(Order.created_at.asc(), Order.id.asc())
            .all()
        )

    def _live_market_features(self, symbol: str) -> dict[str, Any]:
        candles: list[dict[str, Any]] = []
        try:
            candles = self.market_data.get_candles(symbol, "1m", mode="live", limit=80)
        except Exception:  # noqa: BLE001
            candles = []
        closes = [self._safe_float(row.get("close", row.get("c"))) for row in candles if isinstance(row, dict)]
        closes = [value for value in closes if value > 0]
        volumes = [self._safe_float(row.get("volume", row.get("v"))) for row in candles if isinstance(row, dict)]
        momentum_pct = ((closes[-1] - closes[0]) / closes[0]) * 100.0 if len(closes) >= 2 and closes[0] > 0 else 0.0
        returns = [abs((closes[index] - closes[index - 1]) / closes[index - 1]) for index in range(1, len(closes)) if closes[index - 1] > 0]
        volatility_pct = (sum(returns) / len(returns) * 100.0) if returns else 0.0
        recent_volume = sum(volumes[-10:]) / max(len(volumes[-10:]), 1) if volumes else 0.0
        prior_volume = sum(volumes[-30:-10]) / max(len(volumes[-30:-10]), 1) if len(volumes) > 10 else 0.0
        volume_impulse = recent_volume / max(prior_volume, 1e-9) if prior_volume > 0 else 0.0
        spread_bps = 0.0
        liquidity_usd = 0.0
        try:
            book = self.market_data.get_order_book(symbol, "live")
            levels = book.get("levels") if isinstance(book, dict) else []
            if isinstance(levels, list) and len(levels) >= 2 and levels[0] and levels[1]:
                bid = self._level_price_size(levels[0][0])[0]
                ask = self._level_price_size(levels[1][0])[0]
                mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
                spread_bps = ((ask - bid) / mid) * 10_000 if mid > 0 and ask >= bid else 0.0
                for side in levels[:2]:
                    for row in side[:5]:
                        price, size = self._level_price_size(row)
                        liquidity_usd += price * size
        except Exception:  # noqa: BLE001
            pass
        return {
            "momentum_pct": momentum_pct,
            "volatility_pct": volatility_pct,
            "volume_impulse": volume_impulse,
            "spread_bps": spread_bps,
            "liquidity_usd": liquidity_usd,
        }

    def _market_regime(self, features: dict[str, Any], liquidity_score: float) -> str:
        volatility = abs(float(features.get("volatility_pct", 0.0) or 0.0))
        momentum = float(features.get("momentum_pct", 0.0) or 0.0)
        volume_impulse = float(features.get("volume_impulse", 0.0) or 0.0)
        if volatility >= 0.20 and liquidity_score >= 0.55:
            return "high_volatility_liquid"
        if volume_impulse >= 1.5 and abs(momentum) >= 0.35:
            return "expansion_breakout"
        if abs(momentum) >= 0.50:
            return "trend_continuation"
        return "ranging"

    @staticmethod
    def _strategy_for_regime(regime: str) -> tuple[str, str]:
        if regime == "high_volatility_liquid":
            return "scalping", "1m"
        if regime == "expansion_breakout":
            return "volatility_breakout", "5m"
        if regime == "trend_continuation":
            return "ema_crossover", "5m"
        return "mean_reversion", "5m"

    def _record_event(
        self,
        cycle: VaultCycle,
        *,
        category: str,
        severity: str,
        rule_name: str,
        reason: str,
        allocation: VaultCycleAllocation | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = VaultCycleRiskEvent(
            vault_cycle_id=cycle.id,
            allocation_id=allocation.id if allocation is not None else None,
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

    def _audit(
        self,
        cycle: VaultCycle,
        action: str,
        message: str,
        *,
        allocation: VaultCycleAllocation | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        audit = AuditLog(
            user_id=cycle.user_id,
            trading_connection_id=allocation.trading_connection_id if allocation is not None else cycle.trading_connection_id,
            category="vault_cycle",
            action=action,
            message=message,
        )
        audit.details = {"vault_cycle_id": cycle.id, "allocation_id": allocation.id if allocation is not None else None, **(metadata or {})}
        db.session.add(audit)

    def _funded_allocations(self, cycle: VaultCycle) -> list[VaultCycleAllocation]:
        return [
            allocation
            for allocation in list(cycle.exchange_allocations or [])
            if str(allocation.status or "").lower() in {"funded", "active"}
        ]

    @staticmethod
    def _allocation_for_leg(leg: VaultAllocationLeg, allocations: list[VaultCycleAllocation]) -> VaultCycleAllocation | None:
        details = dict(leg.details or {})
        raw_id = details.get("vault_cycle_allocation_id")
        for allocation in allocations:
            if raw_id is not None and int(allocation.id) == int(raw_id):
                return allocation
            if allocation.trading_connection_id == leg.trading_connection_id and normalize_provider(allocation.provider) == normalize_provider(leg.provider):
                return allocation
        return None

    @staticmethod
    def _active_leg_for_allocation(cycle: VaultCycle, allocation: VaultCycleAllocation) -> VaultAllocationLeg | None:
        for leg in list(cycle.allocation_legs or []):
            if str(leg.status or "").lower() != "active":
                continue
            details = dict(leg.details or {})
            raw_id = details.get("vault_cycle_allocation_id")
            if raw_id is not None and int(raw_id) == int(allocation.id):
                return leg
            if leg.trading_connection_id == allocation.trading_connection_id and normalize_provider(leg.provider) == normalize_provider(allocation.provider):
                return leg
        return None

    def _update_leg_candidate_state(
        self,
        leg: VaultAllocationLeg,
        candidate: dict[str, Any],
        now: datetime,
        *,
        skipped_reason: str = "",
    ) -> None:
        details = dict(leg.details or {})
        details.update(
            {
                "latest_opportunity_score": candidate.get("score"),
                "latest_opportunity_symbol": candidate.get("symbol"),
                "latest_opportunity_checked_at": now.isoformat(),
            }
        )
        if skipped_reason:
            details["rotation_skipped_reason"] = skipped_reason
        leg.details = details

    @staticmethod
    def _current_leg_score(leg: VaultAllocationLeg | None, allocation: VaultCycleAllocation) -> float:
        if leg is not None:
            details = dict(leg.details or {})
            for key in ("latest_opportunity_score", "opportunity_score", "risk_adjusted_score"):
                value = details.get(key)
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    parsed = 0.0
                if parsed > 0:
                    return parsed
            breakdown = details.get("scanner_score_breakdown") if isinstance(details.get("scanner_score_breakdown"), dict) else {}
            for key in ("risk_adjusted_score", "opportunity_score"):
                try:
                    parsed = float(breakdown.get(key, 0.0) or 0.0)
                except (TypeError, ValueError):
                    parsed = 0.0
                if parsed > 0:
                    return parsed
        return max(float(allocation.risk_adjusted_score or 0.0), float(allocation.opportunity_score or 0.0))

    def _live_current_score(
        self,
        leg: VaultAllocationLeg | None,
        allocation: VaultCycleAllocation,
        candidates: list[dict[str, Any]],
    ) -> float:
        if leg is not None:
            symbol = str(leg.symbol or "").upper()
            for candidate in candidates:
                if str(candidate.get("symbol") or "").upper() == symbol:
                    return float(candidate.get("score") or 0.0)
        return self._current_leg_score(leg, allocation)

    def _effective_allocation_cap(self, allocation: VaultCycleAllocation) -> float:
        constraints = dict(allocation.constraints or {})
        effective = self._safe_float(constraints.get("effective_allocation_cap_usd"), 0.0)
        if effective > 0:
            return min(effective, float(allocation.allocated_amount or 0.0))
        return float(allocation.allocated_amount or 0.0)

    @staticmethod
    def _allocation_best_symbol(allocation: VaultCycleAllocation) -> str:
        scores = allocation.scores if isinstance(allocation.scores, dict) else {}
        return str(scores.get("best_symbol") or "").upper()

    def _allowed_symbols(self, cycle: VaultCycle) -> list[str]:
        metadata = dict(cycle.selection_metadata or {})
        values = metadata.get("allowed_symbols") or self.config.get("ALLOWED_SYMBOLS", [])
        return [str(symbol).upper() for symbol in values or [] if str(symbol).strip()]

    @staticmethod
    def _order_notional(order: Order) -> float:
        details = dict(order.details or {})
        for key in ("notional", "notional_usd", "target_notional_usd", "allocation_usd"):
            try:
                value = float(details.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value
        quantity = abs(float(order.filled_quantity or order.quantity or 0.0))
        price = float(order.average_fill_price or order.limit_price or details.get("reference_price") or 0.0)
        return quantity * price if quantity > 0 and price > 0 else 0.0

    def _config_payload(self) -> dict[str, Any]:
        return {
            "max_idle_seconds": self._max_idle_seconds(),
            "rescreen_seconds": self._rescreen_seconds(),
            "minimum_trades_per_cycle": self._min_trades_per_cycle(),
            "target_utilization_pct": self._target_utilization_pct(),
            "minimum_opportunity_score": self._min_opportunity_score(),
            "rotation_score_delta": self._rotation_score_delta(),
            "exchange_rebalance_enabled": bool(self.config.get("VAULT_CYCLE_EXCHANGE_REBALANCE_ENABLED", False)),
        }

    def _enabled(self) -> bool:
        return bool(self.config.get("VAULT_CYCLE_ACTIVITY_ENFORCEMENT_ENABLED", True))

    def _max_idle_seconds(self) -> float:
        return max(1.0, float(self.config.get("VAULT_CYCLE_MAX_IDLE_SECONDS", 300.0) or 300.0))

    def _rescreen_seconds(self) -> float:
        return max(1.0, float(self.config.get("VAULT_CYCLE_RESCREEN_SECONDS", 180.0) or 180.0))

    def _min_trades_per_cycle(self) -> int:
        return max(0, int(float(self.config.get("VAULT_CYCLE_MIN_TRADES_PER_CYCLE", 1) or 1)))

    def _target_utilization_pct(self) -> float:
        return min(1.0, max(0.0, float(self.config.get("VAULT_CYCLE_TARGET_UTILIZATION_PCT", 0.60) or 0.60)))

    def _min_opportunity_score(self) -> float:
        return min(1.0, max(0.0, float(self.config.get("VAULT_CYCLE_MIN_OPPORTUNITY_SCORE", 0.60) or 0.60)))

    def _rotation_score_delta(self) -> float:
        return max(0.0, float(self.config.get("VAULT_CYCLE_ROTATION_SCORE_DELTA", 0.10) or 0.10))

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", ""))
        except ValueError:
            return None

    @staticmethod
    def _level_price_size(row: Any) -> tuple[float, float]:
        if isinstance(row, dict):
            return VaultCycleTradingEnforcer._safe_float(row.get("px", row.get("price"))), VaultCycleTradingEnforcer._safe_float(row.get("sz", row.get("size")))
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            return VaultCycleTradingEnforcer._safe_float(row[0]), VaultCycleTradingEnforcer._safe_float(row[1])
        return 0.0, 0.0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bounded_float(value: Any, default: float = 0.0) -> float:
        return max(0.0, min(VaultCycleTradingEnforcer._safe_float(value, default), 1.0))
