"""In-process background runner for strategy execution."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any

from flask import Flask

from ..extensions import db
from ..ml.online_ranker import OnlineRanker, extract_features, horizon_from_duration, outcome_from_result
from ..features.engine import FeatureEngine
from ..models import AuditLog, Order, ShadowLiveObservation, StrategyRun, StrategyValidation, Setting, VaultAllocationLeg, VaultCycle
from ..strategies.registry import StrategyRegistry
from .db_retry import commit_with_retry, write_with_retry
from .order_manager import OrderIntent, OrderManager
from .market_data import MarketDataService
from .signal_quality import SignalQualityEvaluator


logger = logging.getLogger(__name__)


class StrategyManager:
    """Runs strategy loops inside background threads for local deployments."""

    def __init__(
        self,
        app: Flask,
        config: dict[str, Any],
        registry: StrategyRegistry,
        market_data: MarketDataService,
        order_manager: OrderManager,
        feature_engine: FeatureEngine | None = None,
        online_ranker: OnlineRanker | None = None,
        realtime_market: Any | None = None,
    ) -> None:
        self.app = app
        self.config = config
        self.registry = registry
        self.market_data = market_data
        self.order_manager = order_manager
        self.feature_engine = feature_engine or FeatureEngine()
        self.online_ranker = online_ranker or OnlineRanker(config)
        self.realtime_market = realtime_market
        self.signal_quality = SignalQualityEvaluator(config, self.online_ranker)
        self._threads: dict[int, threading.Thread] = {}
        self._stop_events: dict[int, threading.Event] = {}

    def start(self, run_id: int) -> None:
        if run_id in self._threads and self._threads[run_id].is_alive():
            return
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run_loop,
            args=(run_id, stop_event),
            daemon=True,
            name=f"strategy-{run_id}",
        )
        self._threads[run_id] = thread
        self._stop_events[run_id] = stop_event
        thread.start()

    def stop(self, run_id: int) -> None:
        event = self._stop_events.get(run_id)
        if event:
            event.set()
        thread = self._threads.get(run_id)
        if thread and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=min(float(self.config.get("STRATEGY_POLL_SECONDS", 20)), 2.0))
        with self.app.app_context():
            def mark_stopped() -> None:
                run = StrategyRun.query.get(run_id)
                if run:
                    if run.status == "stopped" and not run.manual_enabled:
                        return
                    run.status = "stopped"
                    run.manual_enabled = False
                    run.last_heartbeat_at = datetime.utcnow()

            write_with_retry(mark_stopped)

    def stop_all(self) -> None:
        for run_id in list(self._stop_events):
            self.stop(run_id)

    def _run_loop(self, run_id: int, stop_event: threading.Event) -> None:
        with self.app.app_context():
            run = StrategyRun.query.get(run_id)
            if run is None:
                return
            run.status = "running"
            run.manual_enabled = True
            run.last_error = None
            run.last_heartbeat_at = datetime.utcnow()
            commit_with_retry()

        while not stop_event.is_set():
            poll_seconds = float(self.config.get("STRATEGY_POLL_SECONDS", 20))
            with self.app.app_context():
                run = StrategyRun.query.get(run_id)
                if run is None:
                    return
                if not run.manual_enabled or Setting.get_json("panic_lock", False):
                    run.status = "stopped"
                    run.last_heartbeat_at = datetime.utcnow()
                    commit_with_retry()
                    return
                poll_seconds = self._poll_seconds(run)

                try:
                    started_at = time.perf_counter()
                    market_mode = self._market_mode(run.mode)
                    if run.mode not in {"shadow_live"}:
                        self.order_manager.enforce_protective_exit(
                            run.symbol,
                            run.mode,
                            run.user_id,
                            run.trading_connection_id,
                        )
                    candles = self.market_data.get_candles(run.symbol, run.timeframe, mode=market_mode, limit=200)
                    strategy = self.registry.build(run.strategy_name, run.parameters)
                    signal = strategy.generate_signal(
                        symbol=run.symbol,
                        timeframe=run.timeframe,
                        candles=candles,
                        position=self.order_manager.current_position(
                            run.symbol,
                            run.mode,
                            run.user_id,
                            run.trading_connection_id,
                        ),
                    )
                    previous_signal = dict(run.last_signal or {})
                    previous_heartbeat_at = run.last_heartbeat_at
                    feature_payload = self._feature_payload(run.symbol, run.timeframe, candles, signal)
                    run.last_signal = signal.as_dict()
                    run.last_heartbeat_at = datetime.utcnow()
                    run.status = "running"
                    run.last_error = None
                    commit_with_retry()

                    if self._vault_live_gate_pending(run):
                        self._record_shadow_live_observation(run, signal, started_at, feature_payload)
                        self._evaluate_vault_live_gate(run)
                        continue

                    if run.mode == "shadow_live":
                        self._record_shadow_live_observation(run, signal, started_at, feature_payload)
                        continue

                    if signal.action != "hold":
                        edge_payload: dict[str, Any] = {}
                        if signal.action == "reduce":
                            current = self.order_manager.current_position(
                                run.symbol,
                                run.mode,
                                run.user_id,
                                run.trading_connection_id,
                            )
                            current_qty = abs(float(current.get("quantity", 0.0)))
                            quantity = current_qty if current_qty > 0 else 0.0
                            side = "sell" if float(current.get("quantity", 0.0)) > 0 else "buy"
                        else:
                            side = "buy" if signal.action == "buy" else "sell"
                            mid = self.market_data.get_mid_price(run.symbol, market_mode)
                            forced_side = str(run.parameters.get("pair_forced_side") or "").lower()
                            if forced_side in {"buy", "sell"}:
                                side = forced_side
                            edge_payload = self._signal_edge_payload(
                                run,
                                signal,
                                feature_payload,
                                mid,
                                market_mode,
                                previous_signal=previous_signal,
                                previous_heartbeat_at=previous_heartbeat_at,
                            )
                            if edge_payload:
                                signal_payload = dict(run.last_signal or signal.as_dict())
                                signal_payload.update(edge_payload)
                                run.last_signal = signal_payload
                                commit_with_retry()
                                if edge_payload.get("no_trade_reason"):
                                    self._record_no_trade(run, edge_payload)
                                    self._refresh_vault_estimate(run)
                                    continue
                            sizing_base = self._entry_sizing_base(run)
                            quantity = max((sizing_base * signal.position_fraction) / max(mid, 1e-9), 0.0)
                            leverage = self._dynamic_leverage(run, edge_payload, feature_payload)

                        if quantity > 0:
                            order_type, limit_price = self._execution_order_shape(run, signal, side, mid if signal.action != "reduce" else 0.0, edge_payload)
                            stop_loss, take_profit = self._pair_adjusted_exit_prices(run, signal, side, mid if signal.action != "reduce" else 0.0)
                            intent = OrderIntent(
                                symbol=run.symbol,
                                side=side,
                                quantity=quantity,
                                mode=run.mode,
                                order_type=order_type,
                                limit_price=limit_price,
                                reduce_only=(signal.action == "reduce"),
                                leverage=leverage if signal.action != "reduce" else float(run.parameters.get("leverage", 1.0)),
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                strategy_name=run.strategy_name,
                                timeframe=run.timeframe,
                                slippage_pct=float(self.config["MAX_SLIPPAGE_PCT"]) / 2,
                                user_id=run.user_id,
                                trading_connection_id=run.trading_connection_id,
                                idempotency_key=self._strategy_order_key(run, signal, side),
                                metadata={
                                    "rationale": signal.rationale,
                                    "signal_action": signal.action,
                                    "feature_snapshot": feature_payload,
                                    "vault_cycle_id": run.parameters.get("vault_cycle_id"),
                                    "vault_leg_id": run.parameters.get("vault_leg_id"),
                                    "execution_mode": run.parameters.get("execution_mode"),
                                    "optimizer_profile": run.parameters.get("optimizer_profile"),
                                    "experimental": run.parameters.get("experimental"),
                                    "risk_label": run.parameters.get("risk_label"),
                                    "algorithm_profile": run.parameters.get("algorithm_profile"),
                                    "consumer_vault": run.parameters.get("consumer_vault"),
                                    "execution_style": run.parameters.get("execution_style"),
                                    "high_upside_profile": run.parameters.get("high_upside_profile"),
                                    "duration_hours": run.parameters.get("duration_hours", run.parameters.get("lock_duration_hours")),
                                    "max_drawdown": run.parameters.get("max_drawdown"),
                                    "expected_stop_loss": stop_loss,
                                    "expected_take_profit": take_profit,
                                    "signal_metadata": getattr(signal, "metadata", {}) or {},
                                    "edge_score": edge_payload.get("edge_score", 0.0),
                                    "cost_drag_bps": edge_payload.get("cost_drag_bps", 0.0),
                                    "net_roi_score": edge_payload.get("net_roi_score", run.parameters.get("net_roi_score", 0.0)),
                                    "net_roi_v2_score": edge_payload.get("net_roi_v2_score", run.parameters.get("net_roi_v2_score", 0.0)),
                                    "roi_quality_grade": edge_payload.get("roi_quality_grade", run.parameters.get("roi_quality_grade", "D")),
                                    "roi_rejection_risk": edge_payload.get("roi_rejection_risk", run.parameters.get("roi_rejection_risk", "high")),
                                    "regime_support": edge_payload.get("regime_support", run.parameters.get("regime_support", "regime-neutral")),
                                    "regime_bucket": edge_payload.get("regime_bucket", run.parameters.get("regime_bucket", {})),
                                    "tail_loss_penalty": edge_payload.get("tail_loss_penalty", run.parameters.get("tail_loss_penalty", 0.0)),
                                    "downside_asymmetry_penalty": edge_payload.get(
                                        "downside_asymmetry_penalty",
                                        run.parameters.get("downside_asymmetry_penalty", 0.0),
                                    ),
                                    "cost_adjusted_breakout_potential": edge_payload.get(
                                        "cost_adjusted_breakout_potential",
                                        run.parameters.get("cost_adjusted_breakout_potential", 0.0),
                                    ),
                                    "signal_quality_breakdown": edge_payload.get("signal_quality_breakdown", {}),
                                    "expected_fill_quality": edge_payload.get("expected_fill_quality", run.parameters.get("expected_fill_quality", 0.0)),
                                    "churn_penalty": edge_payload.get("churn_penalty", run.parameters.get("churn_penalty", 0.0)),
                                    "edge_after_cost_bps": edge_payload.get("edge_after_cost_bps", run.parameters.get("edge_after_cost_bps", 0.0)),
                                    "upside_screen_score": run.parameters.get("upside_screen_score", edge_payload.get("upside_screen_score", 0.0)),
                                    "volume_impulse": run.parameters.get("volume_impulse"),
                                    "liquidity_capacity": run.parameters.get("liquidity_capacity", run.parameters.get("liquidity_capacity_usd")),
                                    "volatility_regime": run.parameters.get("volatility_regime"),
                                    "scanner_source": run.parameters.get("scanner_source"),
                                    "convex_edge_score": run.parameters.get("convex_edge_score", run.parameters.get("optimizer_convex_edge_score")),
                                    "mfe_mae_ratio": run.parameters.get("mfe_mae_ratio", run.parameters.get("optimizer_mfe_mae_ratio")),
                                    "rejection_reason": run.parameters.get("rejection_reason"),
                                    "signal_confidence": edge_payload.get("confidence", 0.0),
                                    "quality_reasons": edge_payload.get("quality_reasons", []),
                                    "fibonacci_alignment": edge_payload.get("fibonacci_alignment", {}),
                                    "feature_confluence": edge_payload.get("feature_confluence", {}),
                                    "ml_signal_quality": edge_payload.get("ml_signal_quality", {}),
                                    "market_source": edge_payload.get("market_source"),
                                    "signal_stability": edge_payload.get("signal_stability"),
                                    "dynamic_leverage": leverage if signal.action != "reduce" else float(run.parameters.get("leverage", 1.0)),
                                    "pair_group_id": run.parameters.get("pair_group_id"),
                                    "pair_mode": run.parameters.get("pair_mode"),
                                    "pair_symbol": run.parameters.get("pair_symbol"),
                                    "pair_role": run.parameters.get("pair_role"),
                                    "hedge_ratio": run.parameters.get("hedge_ratio"),
                                    "spread_zscore": run.parameters.get("spread_zscore"),
                                    "spread_half_life": run.parameters.get("spread_half_life"),
                                    "pair_score": run.parameters.get("pair_score"),
                                    "correlation": run.parameters.get("correlation"),
                                    "pair_signal": run.parameters.get("pair_signal", {}),
                                    "pair_forced_side": run.parameters.get("pair_forced_side"),
                                }
                                | (getattr(signal, "metadata", {}) or {}),
                            )
                            order = self.order_manager.place_order(intent)
                            self._refresh_vault_estimate(run)
                            self._learn_from_order_outcome(run, order)
                            if run.mode == "live" and order.status == "failed":
                                run.status = "error"
                                run.manual_enabled = False
                                run.last_error = "Live order failed; live trading is blocked until review."
                                self._mark_vault_limited(run, "Live order failed; live trading is blocked until review.")
                                audit = AuditLog(
                                    category="strategy",
                                    action="live_failure_block",
                                    message=f"Strategy run {run.id} stopped after a failed live order.",
                                    user_id=run.user_id,
                                    trading_connection_id=run.trading_connection_id,
                                )
                                audit.details = {"run_id": run.id, "order_id": order.id}
                                db.session.add(audit)
                                commit_with_retry()
                                return
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Strategy loop failed for run %s", run_id)
                    run = StrategyRun.query.get(run_id)
                    if run:
                        run.status = "error"
                        run.last_error = str(exc)
                        run.last_heartbeat_at = datetime.utcnow()
                        self._mark_vault_limited(run, str(exc))
                        commit_with_retry()
                    audit = AuditLog(
                        category="strategy",
                        action="error",
                        message=f"Strategy run {run_id} failed: {exc}",
                        user_id=run.user_id if run else None,
                        trading_connection_id=run.trading_connection_id if run else None,
                    )
                    audit.details = {"run_id": run_id}
                    db.session.add(audit)
                    commit_with_retry()
                    return

            stop_event.wait(poll_seconds)

    def _record_shadow_live_observation(
        self,
        run: StrategyRun,
        signal: Any,
        started_at: float,
        feature_payload: dict[str, Any],
    ) -> None:
        live_mid = self._safe_mid(run.symbol, "live")
        expected_slippage_bps = float(self.config["SIM_SLIPPAGE_BPS"])
        expected_price = live_mid
        if signal.action == "buy":
            expected_price = live_mid * (1 + expected_slippage_bps / 10_000)
        elif signal.action == "sell":
            expected_price = live_mid * (1 - expected_slippage_bps / 10_000)
        spread_bps = self._safe_spread_bps(run.symbol, "live", live_mid)
        validation = self._shadow_validation(run)
        observation = ShadowLiveObservation(
            validation_id=validation.id,
            strategy_name=run.strategy_name,
            symbol=run.symbol,
            timeframe=run.timeframe,
            signal_action=signal.action,
            expected_price=expected_price,
            live_mid=live_mid,
            expected_slippage_bps=expected_slippage_bps,
            observed_spread_bps=spread_bps,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            missed_fill=(signal.action in {"buy", "sell"} and live_mid <= 0),
            drawdown=0.0,
        )
        observation.details = {
            "rationale": signal.rationale,
            "feature_snapshot": feature_payload,
            "rule_decision": (getattr(signal, "metadata", {}) or {}).get("rule_decision", {}),
            "pattern_prediction": (getattr(signal, "metadata", {}) or {}).get("pattern_prediction", {}),
            "external_scores": (getattr(signal, "metadata", {}) or {}).get("external_scores", {}),
        }
        db.session.add(observation)
        self._update_shadow_validation(validation)
        commit_with_retry()

    def _shadow_validation(self, run: StrategyRun) -> StrategyValidation:
        validation = (
            StrategyValidation.query.filter_by(
                strategy_name=run.strategy_name,
                symbol=run.symbol,
                timeframe=run.timeframe,
                stage="shadow_live",
            )
            .order_by(StrategyValidation.started_at.desc())
            .first()
        )
        if validation is not None and validation.status in {"pending", "passed"}:
            return validation
        validation = StrategyValidation(
            strategy_name=run.strategy_name,
            symbol=run.symbol,
            timeframe=run.timeframe,
            stage="shadow_live",
            status="pending",
        )
        validation.parameters = run.parameters
        db.session.add(validation)
        db.session.flush()
        return validation

    def _update_shadow_validation(self, validation: StrategyValidation) -> None:
        observations = ShadowLiveObservation.query.filter_by(validation_id=validation.id).all()
        actionable = [row for row in observations if row.signal_action in {"buy", "sell", "reduce"}]
        started_at = validation.started_at
        elapsed_hours = (datetime.utcnow() - started_at).total_seconds() / 3600 if started_at else 0.0
        missed = len([row for row in actionable if row.missed_fill])
        avg_latency = sum(row.latency_ms for row in observations) / max(len(observations), 1)
        avg_spread = sum(row.observed_spread_bps for row in observations) / max(len(observations), 1)
        validation.metrics = {
            "observations": len(observations),
            "actionable_signals": len(actionable),
            "missed_fills": missed,
            "avg_latency_ms": avg_latency,
            "avg_spread_bps": avg_spread,
            "elapsed_hours": elapsed_hours,
        }
        if (
            validation.status == "pending"
            and len(actionable) >= int(self.config["SHADOW_LIVE_MIN_TRADES"])
            and elapsed_hours >= float(self.config["SHADOW_LIVE_MIN_HOURS"])
            and missed == 0
        ):
            validation.status = "passed"
            validation.completed_at = datetime.utcnow()

    def _vault_live_gate_pending(self, run: StrategyRun) -> bool:
        # Live onboarding now uses hard risk limits instead of per-cycle shadow probation.
        return False

    def _evaluate_vault_live_gate(self, run: StrategyRun) -> None:
        cycle = self._vault_cycle(run)
        if cycle is None:
            return
        validation = self._shadow_validation(run)
        metrics = validation.metrics
        observations = int(metrics.get("observations", 0) or 0)
        actionable = int(metrics.get("actionable_signals", 0) or 0)
        avg_spread = float(metrics.get("avg_spread_bps", 0.0) or 0.0)
        missed = int(metrics.get("missed_fills", 0) or 0)
        elapsed_minutes = self._validation_elapsed_minutes(cycle)
        min_minutes = float(self.config.get("VAULT_SHADOW_VALIDATION_MINUTES", 15.0))
        min_signals = int(self.config.get("VAULT_SHADOW_MIN_SIGNALS", 2))
        max_spread = float(self.config.get("VAULT_MAX_SPREAD_BPS", 25.0))
        max_slippage = float(self.config.get("VAULT_MAX_SLIPPAGE_BPS", 20.0))
        signal_stability = self._signal_stability(run)
        estimated_slippage = avg_spread + float(self.config.get("SIM_SLIPPAGE_BPS", 8.0))

        failure_reason = ""
        if avg_spread > max_spread:
            failure_reason = "Execution limited due to market conditions."
        elif estimated_slippage > max_slippage:
            failure_reason = "Execution limited due to projected slippage."
        elif missed > 0:
            failure_reason = "Execution limited because live price validation was incomplete."
        elif observations >= min_signals and signal_stability < float(self.config.get("VAULT_SIGNAL_STABILITY_THRESHOLD", 0.65)):
            failure_reason = "Execution limited due to unstable signals."

        if failure_reason:
            cycle.execution_substatus = "limited"
            cycle.live_validation_status = "failed"
            cycle.validation_completed_at = datetime.utcnow()
            cycle.validation_failure_reason = failure_reason
            params = run.parameters
            params["live_validation_status"] = "failed"
            params["fallback_reason"] = failure_reason
            run.parameters = params
            audit = AuditLog(category="vault", action="live_validation_failed", message=f"Vault cycle {cycle.id} live validation failed.")
            audit.details = {"cycle_id": cycle.id, "run_id": run.id, "metrics": metrics, "reason": failure_reason}
            db.session.add(audit)
            commit_with_retry()
            return

        enough_time = elapsed_minutes >= min_minutes
        enough_signals = actionable >= min_signals or observations >= min_signals
        if enough_time or enough_signals:
            validation.status = "passed"
            validation.completed_at = datetime.utcnow()
            cycle.execution_substatus = "executing"
            cycle.execution_mode = "live"
            cycle.live_validation_status = "passed"
            cycle.validation_completed_at = datetime.utcnow()
            cycle.validation_failure_reason = None
            params = run.parameters
            params["live_validation_status"] = "passed"
            params["execution_mode"] = "live"
            run.parameters = params
            audit = AuditLog(category="vault", action="live_validation_passed", message=f"Vault cycle {cycle.id} switched to live execution.")
            audit.details = {"cycle_id": cycle.id, "run_id": run.id, "metrics": metrics}
            db.session.add(audit)
            commit_with_retry()

    def _validation_elapsed_minutes(self, cycle: VaultCycle) -> float:
        started = cycle.validation_started_at or cycle.started_at
        return (datetime.utcnow() - started).total_seconds() / 60.0 if started else 0.0

    def _signal_stability(self, run: StrategyRun) -> float:
        params = run.parameters
        actions = list(params.get("validation_signal_actions", []))
        action = run.last_signal.get("action", "hold")
        actions.append(action)
        actions = actions[-10:]
        params["validation_signal_actions"] = actions
        run.parameters = params
        if not actions:
            return 1.0
        most_common = max(actions.count(item) for item in set(actions))
        return most_common / len(actions)

    def _mark_vault_limited(self, run: StrategyRun, reason: str) -> None:
        cycle = self._vault_cycle(run)
        if cycle is None or cycle.status != "active":
            return
        cycle.execution_substatus = "limited"
        cycle.live_validation_status = "failed" if cycle.live_validation_status == "pending" else cycle.live_validation_status
        cycle.validation_failure_reason = reason
        cycle.validation_completed_at = cycle.validation_completed_at or datetime.utcnow()

    def _refresh_vault_estimate(self, run: StrategyRun) -> None:
        cycle = self._vault_cycle(run)
        if cycle is None:
            return
        realized = 0.0
        has_trading_data = False
        leg_totals: dict[int, float] = {}
        query = Order.query.filter_by(mode=run.mode)
        if run.user_id is not None:
            query = query.filter_by(user_id=run.user_id)
        for order in query.order_by(Order.created_at.asc()).all():
            if order.details.get("vault_cycle_id") != cycle.id:
                continue
            has_trading_data = True
            order_pnl = sum(float(fill.pnl or 0.0) - float(fill.fee or 0.0) for fill in order.fills)
            realized += order_pnl
            leg_id = order.details.get("vault_leg_id")
            if leg_id:
                leg_totals[int(leg_id)] = leg_totals.get(int(leg_id), 0.0) + order_pnl
        unrealized = 0.0
        legs = VaultAllocationLeg.query.filter_by(vault_cycle_id=cycle.id).all()
        if legs:
            for leg in legs:
                leg.realized_pnl_usd = leg_totals.get(leg.id, 0.0)
                try:
                    position = self.order_manager.current_position(
                        leg.symbol,
                        run.mode,
                        run.user_id,
                        run.trading_connection_id,
                    )
                    leg.unrealized_pnl_usd = float(position.get("unrealized_pnl", 0.0) or 0.0)
                except Exception:  # noqa: BLE001
                    leg.unrealized_pnl_usd = 0.0
                unrealized += leg.unrealized_pnl_usd
        else:
            try:
                position = self.order_manager.current_position(
                    run.symbol,
                    run.mode,
                    run.user_id,
                    run.trading_connection_id,
                )
                unrealized = float(position.get("unrealized_pnl", 0.0) or 0.0)
            except Exception:  # noqa: BLE001
                unrealized = 0.0
        total = realized + unrealized
        if has_trading_data:
            cycle.current_estimated_value_usd = max(float(cycle.starting_value_usd or 0.0) + total, 0.0)
        metadata = cycle.selection_metadata
        metadata["realized_pnl_usd"] = realized
        metadata["unrealized_pnl_usd"] = unrealized
        metadata["total_pnl_usd"] = total
        cycle.selection_metadata = metadata
        commit_with_retry()

    def _learn_from_order_outcome(self, run: StrategyRun, order: Order) -> None:
        if not bool(self.config.get("ML_RANKER_ENABLED", False)):
            return
        if order.status != "filled" or not order.fills:
            return
        if order.mode != "live" and not self.online_ranker.should_update_from_mode(order.mode):
            return
        realized = sum(float(fill.pnl or 0.0) - float(fill.fee or 0.0) for fill in order.fills)
        notional = abs(float(order.filled_quantity or order.quantity or 0.0) * float(order.average_fill_price or 0.0))
        if not order.reduce_only and abs(realized) < 1e-12:
            return
        details = order.details or {}
        duration_hours = 0
        cycle = self._vault_cycle(run)
        if cycle is not None:
            duration_hours = int(cycle.lock_duration_hours or 0)
        horizon = horizon_from_duration(duration_hours or details.get("lock_duration_hours") or 24)
        feature_snapshot = details.get("feature_snapshot") if isinstance(details.get("feature_snapshot"), dict) else {}
        features = extract_features(
            {
                **feature_snapshot,
                "strategy_name": run.strategy_name,
                "symbol": run.symbol,
                "timeframe": run.timeframe,
                "optimizer_profile": run.parameters.get("optimizer_profile"),
                "lock_duration_hours": duration_hours,
                "horizon": horizon,
                "net_return_after_costs": realized / max(notional, 1.0),
                "total_return": realized / max(notional, 1.0),
                "edge_score": details.get("edge_score", 0.0),
                "cost_drag_bps": details.get("cost_drag_bps", 0.0),
                "net_roi_score": details.get("net_roi_score", 0.0),
                "expected_fill_quality": details.get("expected_fill_quality", 0.0),
                "churn_penalty": details.get("churn_penalty", 0.0),
                "edge_after_cost_bps": details.get("edge_after_cost_bps", 0.0),
                "upside_screen_score": details.get("upside_screen_score", 0.0),
                "volume_impulse": details.get("volume_impulse", 0.0),
                "liquidity_capacity": details.get("liquidity_capacity", 0.0),
                "volatility_regime": details.get("volatility_regime", ""),
                "scanner_source": details.get("scanner_source", ""),
                "convex_edge_score": details.get("convex_edge_score", 0.0),
                "mfe_mae_ratio": details.get("mfe_mae_ratio", 0.0),
                "rejection_reason": details.get("rejection_reason", ""),
                "high_upside_profile": bool(details.get("high_upside_profile", False)),
                "leverage": order.leverage,
                "trade_count": 1,
            }
        )
        self.online_ranker.update(
            features,
            outcome_from_result(
                {
                    "net_return_after_costs": realized / max(notional, 1.0),
                    "total_return": realized / max(notional, 1.0),
                    "profit_factor": 1.2 if realized > 0 else 0.8,
                    "consistency": 1.0 if realized > 0 else 0.0,
                    "window_stability": 1.0,
                    "trade_count": 1,
                    "edge_score": details.get("edge_score", 0.0),
                    "cost_drag_bps": details.get("cost_drag_bps", 0.0),
                    "net_roi_score": details.get("net_roi_score", 0.0),
                    "expected_fill_quality": details.get("expected_fill_quality", 0.0),
                    "churn_penalty": details.get("churn_penalty", 0.0),
                    "upside_screen_score": details.get("upside_screen_score", 0.0),
                    "convex_edge_score": details.get("convex_edge_score", 0.0),
                }
            ),
            horizon=horizon,
            source="strategy_order",
            source_id=order.id,
            mode=order.mode,
            metadata={
                "run_id": run.id,
                "vault_cycle_id": run.parameters.get("vault_cycle_id"),
                "vault_leg_id": run.parameters.get("vault_leg_id"),
                "realized_after_fees": realized,
                "status": "quarantined" if order.mode == "live" else "trained",
                "high_upside_profile": bool(details.get("high_upside_profile", False)),
            },
        )
        commit_with_retry()

    def _vault_cycle(self, run: StrategyRun) -> VaultCycle | None:
        cycle_id = run.parameters.get("vault_cycle_id")
        if not cycle_id:
            return None
        return db.session.get(VaultCycle, int(cycle_id))

    def _signal_edge_payload(
        self,
        run: StrategyRun,
        signal: Any,
        feature_payload: dict[str, Any],
        mid: float,
        market_mode: str,
        previous_signal: dict[str, Any] | None = None,
        previous_heartbeat_at: datetime | None = None,
    ) -> dict[str, Any]:
        if signal.action not in {"buy", "sell"} or mid <= 0:
            return {}

        take_profit = self._safe_float(signal.take_profit)
        if take_profit > 0:
            expected_move_bps = abs(take_profit - mid) / mid * 10_000
        else:
            expected_move_bps = self._safe_float(run.parameters.get("take_profit_pct"), 0.0) * 10_000

        snapshot = self._market_snapshot(run.symbol, market_mode, run.timeframe)
        if "spread_bps" not in snapshot:
            snapshot["spread_bps"] = self._safe_spread_bps(run.symbol, market_mode, mid)
        snapshot["mid"] = snapshot.get("mid") or mid
        run_parameters = dict(run.parameters or {})
        if previous_signal:
            run_parameters["last_signal_action"] = previous_signal.get("action")
        if previous_heartbeat_at is not None:
            try:
                run_parameters["last_signal_age_seconds"] = max(
                    (datetime.utcnow() - previous_heartbeat_at).total_seconds(),
                    0.0,
                )
            except TypeError:
                run_parameters["last_signal_age_seconds"] = 0.0

        payload = self.signal_quality.evaluate(
            symbol=run.symbol,
            timeframe=run.timeframe,
            mode=market_mode,
            run_parameters=run_parameters,
            signal=signal,
            feature_payload=feature_payload,
            mid=mid,
            market_snapshot=snapshot,
        )
        if not payload:
            return {}
        payload["expected_move_bps"] = payload.get("expected_move_bps", expected_move_bps)
        return payload

    def _dynamic_leverage(
        self,
        run: StrategyRun,
        edge_payload: dict[str, Any],
        feature_payload: dict[str, Any],
    ) -> float:
        requested = max(1.0, self._safe_float(run.parameters.get("leverage"), 1.0))
        hard_cap = min(self._safe_float(self.config.get("MAX_LEVERAGE"), 3.0), 3.0)
        if not bool(self.config.get("LEVERAGE_OPTIMIZER_ENABLED", False)):
            return 1.0

        confidence = self._safe_float(edge_payload.get("confidence"), 0.0)
        volatility = self._safe_float(feature_payload.get("volatility"), 0.0)
        atr_pct = self._safe_float(feature_payload.get("atr_pct"), 0.0)
        adjusted = requested
        if confidence >= 0.75 and self._safe_float(edge_payload.get("edge_score"), 0.0) > 0:
            adjusted = max(adjusted, min(hard_cap, 1.0 + (confidence - 0.65) * 2.0))
        if volatility > 0.04 or atr_pct > 0.04:
            adjusted = min(adjusted, 1.5)
        return max(1.0, min(adjusted, hard_cap))

    def _record_no_trade(self, run: StrategyRun, edge_payload: dict[str, Any]) -> None:
        audit = AuditLog(
            category="strategy",
            action="no_trade",
            message="Strategy signal skipped because expected edge after costs was below threshold.",
            user_id=run.user_id,
            trading_connection_id=run.trading_connection_id,
        )
        audit.details = {
            "run_id": run.id,
            "symbol": run.symbol,
            "timeframe": run.timeframe,
            "strategy_name": run.strategy_name,
            "vault_cycle_id": run.parameters.get("vault_cycle_id"),
            "vault_leg_id": run.parameters.get("vault_leg_id"),
            "optimizer_profile": run.parameters.get("optimizer_profile"),
            **edge_payload,
        }
        db.session.add(audit)
        commit_with_retry()

    def _execution_order_shape(
        self,
        run: StrategyRun,
        signal: Any,
        side: str,
        mid: float,
        edge_payload: dict[str, Any],
    ) -> tuple[str, float | None]:
        if signal.action == "reduce" or mid <= 0:
            return "market", None
        if str(run.parameters.get("execution_style", "market")) != "maker_limit":
            suggested = str(edge_payload.get("suggested_execution_style", ""))
            if suggested != "maker_limit":
                return "market", None
        elif str(edge_payload.get("suggested_execution_style", "maker_limit")) == "market":
            return "market", None
        edge_score = self._safe_float(edge_payload.get("edge_score"), 0.0)
        profile = str(run.parameters.get("optimizer_profile", "")).lower()
        cost_drag = self._safe_float(edge_payload.get("cost_drag_bps"), 0.0)
        edge_after_cost = self._safe_float(edge_payload.get("edge_after_cost_bps"), edge_score - cost_drag)
        fill_quality = self._safe_float(edge_payload.get("expected_fill_quality"), 1.0)
        min_edge = max(self._safe_float(self.config.get("AGGRESSIVE_MIN_EDGE_BPS"), 4.0), 1.0)
        if profile == "aggressive_1h":
            min_edge = max(min_edge, cost_drag * 2.5, 18.0)
        post_cost_min = max(
            self._safe_float(self.config.get("AGGRESSIVE_MIN_EDGE_BPS"), 4.0),
            self._safe_float(self.config.get("NET_ROI_MIN_EDGE_BPS"), 4.0),
        )
        if edge_score < min_edge or edge_after_cost < post_cost_min or fill_quality < self._safe_float(self.config.get("NET_ROI_MIN_FILL_QUALITY"), 0.55):
            return "market", None
        offset = min(max(self._safe_float(edge_payload.get("spread_bps"), 1.0) / 20_000, 0.0001), 0.0015)
        limit_price = mid * (1 - offset if side == "buy" else 1 + offset)
        return "limit", round(limit_price, 8)

    def _pair_adjusted_exit_prices(
        self,
        run: StrategyRun,
        signal: Any,
        side: str,
        mid: float,
    ) -> tuple[float | None, float | None]:
        if signal.action == "reduce" or mid <= 0 or not run.parameters.get("pair_forced_side"):
            return signal.stop_loss, signal.take_profit
        stop_pct = self._safe_float(run.parameters.get("stop_loss_pct", run.parameters.get("fallback_stop_loss_pct")), 0.0)
        take_pct = self._safe_float(run.parameters.get("take_profit_pct", run.parameters.get("fallback_take_profit_pct")), 0.0)
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit
        if side == "buy":
            if self._safe_float(stop_loss) <= 0 or self._safe_float(stop_loss) >= mid:
                stop_loss = mid * (1 - max(stop_pct, 0.001))
            if self._safe_float(take_profit) <= 0 or self._safe_float(take_profit) <= mid:
                take_profit = mid * (1 + max(take_pct, 0.001))
        elif side == "sell":
            if self._safe_float(stop_loss) <= 0 or self._safe_float(stop_loss) <= mid:
                stop_loss = mid * (1 + max(stop_pct, 0.001))
            if self._safe_float(take_profit) <= 0 or self._safe_float(take_profit) >= mid:
                take_profit = mid * (1 - max(take_pct, 0.001))
        return stop_loss, take_profit

    def _market_snapshot(self, symbol: str, mode: str, timeframe: str) -> dict[str, Any]:
        if self.realtime_market is not None:
            try:
                return self.realtime_market.snapshot(symbol, mode, timeframe)
            except Exception:  # noqa: BLE001
                return {}
        return {}

    def _fallback_mode(self) -> str:
        return "live"

    def _entry_sizing_base(self, run: StrategyRun) -> float:
        sizing_base = max(0.0, self._safe_float(self.config.get("MAX_POSITION_NOTIONAL")))
        allocation_cap = self._safe_float(run.parameters.get("allocation_cap_usd"), 0.0)
        if allocation_cap > 0:
            sizing_base = min(sizing_base, allocation_cap)
        return sizing_base

    def _poll_seconds(self, run: StrategyRun) -> float:
        default = float(self.config.get("STRATEGY_POLL_SECONDS", 20))
        profile = str((run.parameters or {}).get("optimizer_profile", "")).lower()
        lock_seconds = int(run.lock_duration_seconds or (run.parameters or {}).get("lock_duration_seconds", 0) or 0)
        if profile in {"aggressive_1h", "extreme_roi_experimental"} or 0 < lock_seconds <= 3600:
            return max(1.0, min(default, float(self.config.get("AGGRESSIVE_1H_POLL_SECONDS", 10.0))))
        return max(1.0, default)

    def _strategy_order_key(self, run: StrategyRun, signal: Any, side: str) -> str:
        payload = getattr(signal, "metadata", {}) or {}
        candle_ts = payload.get("signal_timestamp") or payload.get("timestamp") or run.last_heartbeat_at
        cycle_id = run.parameters.get("vault_cycle_id", "manual")
        return f"strategy-{run.id}-{cycle_id}-{run.symbol}-{run.timeframe}-{side}-{candle_ts}"

    def _safe_mid(self, symbol: str, mode: str) -> float:
        try:
            return self.market_data.get_mid_price(symbol, mode)
        except Exception:  # noqa: BLE001
            return 0.0

    def _safe_spread_bps(self, symbol: str, mode: str, mid: float) -> float:
        try:
            order_book = self.market_data.get_order_book(symbol, mode)
        except Exception:  # noqa: BLE001
            return 0.0
        levels = order_book.get("levels") if isinstance(order_book, dict) else None
        if not isinstance(levels, list) or len(levels) < 2 or not levels[0] or not levels[1] or mid <= 0:
            return 0.0
        bid = float(levels[0][0].get("px", 0.0) if isinstance(levels[0][0], dict) else levels[0][0][0])
        ask = float(levels[1][0].get("px", 0.0) if isinstance(levels[1][0], dict) else levels[1][0][0])
        return ((ask - bid) / mid) * 10_000 if bid > 0 and ask > 0 else 0.0

    def _feature_payload(self, symbol: str, timeframe: str, candles: list[dict[str, Any]], signal: Any) -> dict[str, Any]:
        metadata = getattr(signal, "metadata", {}) or {}
        if isinstance(metadata.get("feature_snapshot"), dict):
            return metadata["feature_snapshot"]
        return self.feature_engine.snapshot(symbol=symbol, timeframe=timeframe, candles=candles).as_dict()

    @staticmethod
    def _market_mode(mode: str) -> str:
        return "live"

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
