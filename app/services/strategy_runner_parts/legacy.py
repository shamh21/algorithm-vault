"""In-process background runner for strategy execution."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from flask import Flask
from sqlalchemy import or_

from ...extensions import db
from ...ml.online_ranker import ONE_H10_HORIZON, OnlineRanker, extract_features, horizon_from_context, outcome_from_result
from ...features.engine import FeatureEngine
from ...models import AuditLog, Order, ShadowLiveObservation, StrategyRun, StrategyValidation, Setting, TradingConnection, VaultAllocationLeg, VaultCycle
from ...strategies.base import Signal
from ...strategies.registry import StrategyRegistry
from ..db_retry import commit_with_retry, is_database_locked, write_with_retry
from ..order_manager import OrderIntent, OrderManager
from ..market_data import MarketDataService
from ..signal_quality import SignalQualityEvaluator
from ..worker_lease import WorkerLeaseService


logger = logging.getLogger(__name__)
_UNSET = object()


@dataclass(slots=True)
class _LoopRuntimeState:
    last_market_fingerprint: str | None = None
    last_eval_at: float = 0.0
    last_persisted_heartbeat_at: float = 0.0
    last_signal_fingerprint: str | None = None
    ticks_total: int = 0
    ticks_skipped_unchanged: int = 0
    ticks_full_eval: int = 0
    full_eval_ms_sum: float = 0.0


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
        ml_signal_model: Any | None = None,
        ml_decision_engine: Any | None = None,
    ) -> None:
        self.app = app
        self.config = config
        self.registry = registry
        self.market_data = market_data
        self.order_manager = order_manager
        self.feature_engine = feature_engine or FeatureEngine()
        self.online_ranker = online_ranker or OnlineRanker(config)
        self.realtime_market = realtime_market
        self.ml_signal_model = ml_signal_model
        self.ml_decision_engine = ml_decision_engine
        self.signal_quality = SignalQualityEvaluator(config, self.online_ranker)
        self._lease_service = WorkerLeaseService(config)
        self._threads: dict[int, threading.Thread] = {}
        self._stop_events: dict[int, threading.Event] = {}
        self._provider_runtime_locks: dict[str, threading.Lock] = {}
        self._provider_runtime_locks_guard = threading.Lock()
        self._loop_state_by_run: dict[int, _LoopRuntimeState] = {}
        self._loop_state_guard = threading.Lock()
        self._loop_metrics = {
            "ticks_total": 0,
            "ticks_skipped_unchanged": 0,
            "ticks_full_eval": 0,
            "full_eval_ms_sum": 0.0,
            "db_writes_total": 0,
        }
        self._loop_metrics_guard = threading.Lock()
        self._no_trade_audit_state: dict[tuple[int, str, str], dict[str, Any]] = {}
        self._no_trade_audit_guard = threading.Lock()

    def start(self, run_id: int) -> bool:
        if run_id in self._threads and self._threads[run_id].is_alive():
            return True
        with self.app.app_context():
            lease = self._lease_service.acquire_strategy_run(
                run_id,
                metadata={"source": "strategy_manager.start"},
            )
            if lease is None:
                run = db.session.get(StrategyRun, int(run_id))
                if run is not None and run.status not in {"running", "starting"}:
                    run.status = "queued"
                    run.manual_enabled = True
                    commit_with_retry()
                return False
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
        return True

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
            self._lease_service.release_strategy_run(run_id, status="stopped")
        self._clear_loop_state(run_id)

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
            self._commit_loop_write()
            loop_state = self._loop_state(run_id)
            loop_state.last_persisted_heartbeat_at = time.time()
            if isinstance(run.last_signal, dict):
                loop_state.last_signal_fingerprint = self._signal_fingerprint(run.last_signal)

        while not stop_event.is_set():
            poll_seconds = float(self.config.get("STRATEGY_POLL_SECONDS", 20))
            with self.app.app_context():
                run = StrategyRun.query.get(run_id)
                if run is None:
                    self._clear_loop_state(run_id)
                    return
                if not self._lease_service.heartbeat_strategy_run(run_id):
                    run.status = "queued"
                    run.manual_enabled = True
                    run.last_error = "Strategy lease lost; queued for worker recovery."
                    run.last_heartbeat_at = datetime.utcnow()
                    self._commit_loop_write()
                    self._clear_loop_state(run_id)
                    return
                loop_state = self._loop_state(run_id)
                self._record_loop_tick(loop_state)
                if not run.manual_enabled or Setting.get_json("panic_lock", False):
                    run.status = "stopped"
                    run.last_heartbeat_at = datetime.utcnow()
                    self._commit_loop_write()
                    self._clear_loop_state(run_id)
                    return
                poll_seconds = self._poll_seconds(run)
                provider_lock: threading.Lock | None = None
                provider_lock_acquired = False

                try:
                    started_at = time.perf_counter()
                    tick_now = time.time()
                    market_mode = self._market_mode(run.mode)
                    backoff_remaining = self._runtime_provider_backoff_remaining(run)
                    if backoff_remaining > 0:
                        self._mark_runtime_provider_backoff_active(run, backoff_remaining)
                        self._persist_run_runtime_state(
                            run,
                            loop_state,
                            signal_payload=dict(run.last_signal or {}),
                            status="running",
                            last_error=run.last_error,
                            heartbeat_now=tick_now,
                            force=True,
                        )
                        stop_event.wait(max(poll_seconds, min(backoff_remaining, self._runtime_provider_wait_seconds())))
                        continue
                    provider_lock = self._provider_runtime_lock(run)
                    if provider_lock is not None:
                        provider_lock_acquired = provider_lock.acquire(blocking=False)
                        if not provider_lock_acquired:
                            self._mark_runtime_provider_backoff_active(run, poll_seconds, reason="Provider request already in progress.")
                            self._persist_run_runtime_state(
                                run,
                                loop_state,
                                signal_payload=dict(run.last_signal or {}),
                                status="running",
                                last_error=run.last_error,
                                heartbeat_now=tick_now,
                                force=True,
                            )
                            stop_event.wait(max(poll_seconds, 1.0))
                            continue
                    self._one_h10_rebalance_tick(run)
                    if run.mode not in {"shadow_live"}:
                        self.order_manager.enforce_protective_exit(
                            run.symbol,
                            run.mode,
                            run.user_id,
                            run.trading_connection_id,
                        )
                    candles = self._run_candles(run, market_mode, limit=200)
                    market_fingerprint = self._market_fingerprint(run.symbol, run.timeframe, candles)
                    if self._should_skip_full_eval(loop_state, market_fingerprint, tick_now):
                        self._record_loop_skip(loop_state)
                        self._persist_run_runtime_state(
                            run,
                            loop_state,
                            status="running",
                            last_error=None,
                            heartbeat_now=tick_now,
                        )
                        continue
                    strategy = self.registry.build(run.strategy_name, run.parameters)
                    current_position = self.order_manager.current_position(
                        run.symbol,
                        run.mode,
                        run.user_id,
                        run.trading_connection_id,
                    )
                    signal = strategy.generate_signal(
                        symbol=run.symbol,
                        timeframe=run.timeframe,
                        candles=candles,
                        position=current_position,
                    )
                    previous_signal = dict(run.last_signal or {})
                    previous_heartbeat_at = run.last_heartbeat_at
                    feature_payload = self._feature_payload(run.symbol, run.timeframe, candles, signal)
                    signal = self._high_upside_ml_signal(run, signal, candles, feature_payload, market_mode)
                    signal = self._one_h10_forecast_signal(run, signal, candles)
                    signal_payload = signal.as_dict()
                    signal_fingerprint = self._signal_fingerprint(signal_payload)
                    self._persist_run_runtime_state(
                        run,
                        loop_state,
                        signal_payload=signal_payload,
                        signal_fingerprint=signal_fingerprint,
                        status="running",
                        last_error=None,
                        heartbeat_now=tick_now,
                    )

                    if self._vault_live_gate_pending(run):
                        self._mark_full_eval_complete(loop_state, started_at=started_at, now_ts=tick_now, market_fingerprint=market_fingerprint)
                        self._record_shadow_live_observation(run, signal, started_at, feature_payload)
                        self._evaluate_vault_live_gate(run)
                        continue

                    if run.mode == "shadow_live":
                        self._mark_full_eval_complete(loop_state, started_at=started_at, now_ts=tick_now, market_fingerprint=market_fingerprint)
                        self._record_shadow_live_observation(run, signal, started_at, feature_payload)
                        continue

                    if signal.action != "hold":
                        edge_payload: dict[str, Any] = {}
                        if signal.action == "reduce":
                            current = current_position
                            current_qty = abs(float(current.get("quantity", 0.0)))
                            quantity = current_qty if current_qty > 0 else 0.0
                            side = "sell" if float(current.get("quantity", 0.0)) > 0 else "buy"
                        else:
                            side = "buy" if signal.action == "buy" else "sell"
                            mid = self._run_mid_price(run, market_mode)
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
                                signal_payload = dict(signal_payload or run.last_signal or signal.as_dict())
                                signal_payload.update(edge_payload)
                                signal_fingerprint = self._signal_fingerprint(signal_payload)
                                self._persist_run_runtime_state(
                                    run,
                                    loop_state,
                                    signal_payload=signal_payload,
                                    signal_fingerprint=signal_fingerprint,
                                    status="running",
                                    last_error=None,
                                    heartbeat_now=tick_now,
                                )
                                if edge_payload.get("no_trade_reason"):
                                    self._mark_full_eval_complete(
                                        loop_state,
                                        started_at=started_at,
                                        now_ts=tick_now,
                                        market_fingerprint=market_fingerprint,
                                    )
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
                                slippage_pct=0.0,
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
                                    "vault_cycle_name": run.parameters.get("vault_cycle_name"),
                                    "consumer_vault": run.parameters.get("consumer_vault"),
                                    "one_h10_vault": run.parameters.get("one_h10_vault"),
                                    "ml_horizon": run.parameters.get("ml_horizon"),
                                    "objective": run.parameters.get("objective"),
                                    "ml_objective": run.parameters.get("ml_objective"),
                                    "target_return_objective": run.parameters.get("target_return_objective"),
                                    "ml_policy_required": run.parameters.get("ml_policy_required"),
                                    "ml_governed_risk": run.parameters.get("ml_governed_risk"),
                                    "target_roi_pct": run.parameters.get("target_roi_pct"),
                                    "target_amount_usd": run.parameters.get("target_amount_usd"),
                                    "user_input_amount_usd": run.parameters.get("user_input_amount_usd"),
                                    "provider": run.parameters.get("provider"),
                                    "execution_venue": run.parameters.get("execution_venue"),
                                    "app_symbol": run.parameters.get("app_symbol") or run.symbol,
                                    "venue_symbol": run.parameters.get("venue_symbol"),
                                    "provider_symbol": run.parameters.get("provider_symbol") or run.parameters.get("venue_symbol"),
                                    "market_id": run.parameters.get("market_id"),
                                    "market_status": run.parameters.get("market_status"),
                                    "collateral_asset": run.parameters.get("collateral_asset"),
                                    "settlement_asset": run.parameters.get("settlement_asset"),
                                    "allocation_cap_usd": run.parameters.get("allocation_cap_usd"),
                                    "available_margin_usd": run.parameters.get("available_margin_usd"),
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
                                    "one_h10_forecast": run.parameters.get("one_h10_forecast"),
                                    "forecast_metadata": run.parameters.get("forecast_metadata", run.parameters.get("one_h10_forecast")),
                                    "forecast_blockers": run.parameters.get("forecast_blockers", []),
                                    "forecast_predicted_side": run.parameters.get("forecast_predicted_side"),
                                    "forecast_confidence": run.parameters.get("forecast_confidence"),
                                    "forecast_expected_return_bps": run.parameters.get("forecast_expected_return_bps"),
                                    "forecast_suggested_notional_usd": run.parameters.get("forecast_suggested_notional_usd"),
                                    "forecast_suggested_leverage": run.parameters.get("forecast_suggested_leverage"),
                                    "forecast_suggested_order_type": run.parameters.get("forecast_suggested_order_type"),
                                    "forecast_suggested_stop_loss_pct": run.parameters.get("forecast_suggested_stop_loss_pct"),
                                    "forecast_suggested_take_profit_pct": run.parameters.get("forecast_suggested_take_profit_pct"),
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
                                self._commit_loop_write()
                                self._mark_full_eval_complete(
                                    loop_state,
                                    started_at=started_at,
                                    now_ts=tick_now,
                                    market_fingerprint=market_fingerprint,
                                )
                                self._clear_loop_state(run_id)
                                return
                    self._mark_full_eval_complete(
                        loop_state,
                        started_at=started_at,
                        now_ts=tick_now,
                        market_fingerprint=market_fingerprint,
                    )
                except Exception as exc:  # noqa: BLE001
                    db.session.rollback()
                    if is_database_locked(exc):
                        logger.warning("Strategy loop hit SQLite write lock for run %s; retrying next tick.", run_id)
                        stop_event.wait(max(1.0, min(poll_seconds, 5.0)))
                        continue
                    run = db.session.get(StrategyRun, run_id)
                    if run and self._handle_provider_runtime_failure(run, exc):
                        logger.warning("Provider temporarily unavailable for run %s; backing off without limiting cycle: %s", run_id, exc)
                        audit = AuditLog(
                            category="strategy",
                            action="provider_runtime_backoff",
                            message=f"Strategy run {run_id} paused for provider runtime backoff.",
                            user_id=run.user_id,
                            trading_connection_id=run.trading_connection_id,
                        )
                        audit.details = {
                            "run_id": run_id,
                            "symbol": run.symbol,
                            "timeframe": run.timeframe,
                            "error": str(exc),
                            "blocker_category": self._provider_failure_category(exc),
                        }
                        db.session.add(audit)
                        self._commit_loop_write()
                    else:
                        logger.exception("Strategy loop failed for run %s", run_id)
                        if run:
                            run.status = "error"
                            run.last_error = str(exc)
                            run.last_heartbeat_at = datetime.utcnow()
                            self._mark_vault_limited(run, str(exc))
                            self._commit_loop_write()
                        audit = AuditLog(
                            category="strategy",
                            action="error",
                            message=f"Strategy run {run_id} failed: {exc}",
                            user_id=run.user_id if run else None,
                            trading_connection_id=run.trading_connection_id if run else None,
                        )
                        audit.details = {"run_id": run_id}
                        db.session.add(audit)
                        self._commit_loop_write()
                        self._clear_loop_state(run_id)
                        return
                finally:
                    if provider_lock is not None and provider_lock_acquired:
                        provider_lock.release()

            stop_event.wait(poll_seconds)
        self._clear_loop_state(run_id)

    def _commit_loop_write(self) -> None:
        commit_with_retry()
        with self._loop_metrics_guard:
            self._loop_metrics["db_writes_total"] = int(self._loop_metrics.get("db_writes_total", 0)) + 1

    def _loop_state(self, run_id: int) -> _LoopRuntimeState:
        with self._loop_state_guard:
            state = self._loop_state_by_run.get(run_id)
            if state is None:
                state = _LoopRuntimeState()
                self._loop_state_by_run[run_id] = state
            return state

    def _clear_loop_state(self, run_id: int) -> None:
        with self._loop_state_guard:
            self._loop_state_by_run.pop(run_id, None)
        try:
            self._lease_service.release_strategy_run(run_id)
        except RuntimeError:
            pass

    def _record_loop_tick(self, loop_state: _LoopRuntimeState) -> None:
        loop_state.ticks_total += 1
        with self._loop_metrics_guard:
            self._loop_metrics["ticks_total"] = int(self._loop_metrics.get("ticks_total", 0)) + 1

    def _record_loop_skip(self, loop_state: _LoopRuntimeState) -> None:
        loop_state.ticks_skipped_unchanged += 1
        with self._loop_metrics_guard:
            self._loop_metrics["ticks_skipped_unchanged"] = int(self._loop_metrics.get("ticks_skipped_unchanged", 0)) + 1

    def _mark_full_eval_complete(
        self,
        loop_state: _LoopRuntimeState,
        *,
        started_at: float,
        now_ts: float,
        market_fingerprint: str,
    ) -> None:
        elapsed_ms = max((time.perf_counter() - started_at) * 1000.0, 0.0)
        loop_state.last_market_fingerprint = market_fingerprint
        loop_state.last_eval_at = now_ts
        loop_state.ticks_full_eval += 1
        loop_state.full_eval_ms_sum += elapsed_ms
        with self._loop_metrics_guard:
            self._loop_metrics["ticks_full_eval"] = int(self._loop_metrics.get("ticks_full_eval", 0)) + 1
            self._loop_metrics["full_eval_ms_sum"] = float(self._loop_metrics.get("full_eval_ms_sum", 0.0)) + elapsed_ms

    def _persist_run_runtime_state(
        self,
        run: StrategyRun,
        loop_state: _LoopRuntimeState,
        *,
        signal_payload: dict[str, Any] | None = None,
        signal_fingerprint: str | None = None,
        status: str | None = None,
        last_error: Any = _UNSET,
        heartbeat_now: float | None = None,
        force: bool = False,
    ) -> bool:
        changed = False
        if signal_payload is not None:
            fingerprint = signal_fingerprint or self._signal_fingerprint(signal_payload)
            if force or fingerprint != loop_state.last_signal_fingerprint:
                run.last_signal = signal_payload
                loop_state.last_signal_fingerprint = fingerprint
                changed = True
        if status is not None and (force or run.status != status):
            run.status = status
            changed = True
        if last_error is not _UNSET and (force or run.last_error != last_error):
            run.last_error = last_error
            changed = True
        if heartbeat_now is not None and (force or self._should_persist_heartbeat(loop_state, heartbeat_now)):
            run.last_heartbeat_at = datetime.utcnow()
            loop_state.last_persisted_heartbeat_at = heartbeat_now
            changed = True
        if changed:
            self._commit_loop_write()
        return changed

    def _should_skip_full_eval(
        self,
        loop_state: _LoopRuntimeState,
        market_fingerprint: str,
        now_ts: float,
    ) -> bool:
        if not bool(self.config.get("STRATEGY_CHANGE_DRIVEN_LOOP_ENABLED", False)):
            return False
        if not loop_state.last_market_fingerprint or loop_state.last_market_fingerprint != market_fingerprint:
            return False
        idle_seconds = max(1.0, self._safe_float(self.config.get("STRATEGY_IDLE_REEVAL_SECONDS"), 15.0))
        return (now_ts - loop_state.last_eval_at) < idle_seconds

    def _should_persist_heartbeat(self, loop_state: _LoopRuntimeState, now_ts: float) -> bool:
        interval = max(1.0, self._safe_float(self.config.get("STRATEGY_HEARTBEAT_PERSIST_SECONDS"), 30.0))
        if loop_state.last_persisted_heartbeat_at <= 0:
            return True
        return (now_ts - loop_state.last_persisted_heartbeat_at) >= interval

    @staticmethod
    def _market_fingerprint(symbol: str, timeframe: str, candles: list[dict[str, Any]]) -> str:
        latest = candles[-1] if candles else {}
        timestamp = (
            latest.get("timestamp")
            or latest.get("t")
            or latest.get("time")
            or latest.get("ts")
            or latest.get("open_time")
        )
        close = latest.get("close")
        if close is None:
            close = latest.get("c")
        payload = {
            "symbol": str(symbol or "").upper(),
            "timeframe": str(timeframe or ""),
            "latest_candle_timestamp": timestamp,
            "latest_close": close,
            "candle_count": len(candles),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _signal_fingerprint(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload or {}, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()

    def get_loop_metrics(self) -> dict[str, Any]:
        with self._loop_metrics_guard:
            ticks_total = int(self._loop_metrics.get("ticks_total", 0))
            ticks_skipped = int(self._loop_metrics.get("ticks_skipped_unchanged", 0))
            ticks_full = int(self._loop_metrics.get("ticks_full_eval", 0))
            full_eval_ms_sum = float(self._loop_metrics.get("full_eval_ms_sum", 0.0))
            db_writes = int(self._loop_metrics.get("db_writes_total", 0))
        avg_full_eval_ms = full_eval_ms_sum / ticks_full if ticks_full else 0.0
        return {
            "ticks_total": ticks_total,
            "ticks_skipped_unchanged": ticks_skipped,
            "ticks_full_eval": ticks_full,
            "avg_full_eval_ms": avg_full_eval_ms,
            "db_writes_total": db_writes,
        }

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

    def _handle_one_h10_market_data_failure(self, run: StrategyRun, exc: Exception) -> bool:
        if not self._is_one_h10_run(run) or not self._is_market_data_failure(exc):
            return False
        now = datetime.utcnow()
        blocker = self._provider_failure_category(exc)
        backoff_seconds = self._runtime_provider_backoff_seconds(exc)
        backoff_until = now + timedelta(seconds=backoff_seconds)
        reason = self._sanitize_market_data_error(run, exc)
        params = dict(run.parameters or {})
        history = list(params.get("one_h10_market_data_failures", []) or [])
        history.append(
            {
                "timestamp": now.isoformat(),
                "symbol": run.symbol,
                "timeframe": run.timeframe,
                "reason": reason,
                "raw_error": str(exc),
                "blocker_category": blocker,
                "backoff_seconds": backoff_seconds,
            }
        )
        params["one_h10_market_data_failures"] = history[-20:]
        params["one_h10_market_data_status"] = "backoff"
        params["one_h10_market_data_error"] = reason
        params["one_h10_market_data_backoff_until"] = backoff_until.isoformat()
        params["one_h10_market_data_blocker"] = blocker
        run.parameters = params
        provider = str((run.parameters or {}).get("provider") or (run.parameters or {}).get("execution_venue") or "").strip().lower()
        connection_id = run.trading_connection_id or (run.parameters or {}).get("trading_connection_id")
        if provider:
            self._set_runtime_provider_backoff(
                provider=provider,
                connection_id=connection_id,
                run=run,
                reason=reason,
                blocker=blocker,
                retry_after=backoff_until,
                updated_at=now,
                include_one_h10=True,
            )
        run.status = "running"
        run.manual_enabled = True
        run.last_error = reason
        run.last_heartbeat_at = now
        run.last_signal = {
            "action": "hold",
            "rationale": "1H10 skipped order generation because live market data is unavailable.",
            "timeframe": run.timeframe,
            "stop_loss": None,
            "take_profit": None,
            "position_fraction": 0.0,
            "metadata": {
                "no_trade_reason": "market_data_unavailable",
                "blocker_category": blocker,
                "one_h10_vault": True,
                "ml_horizon": ONE_H10_HORIZON,
                "market_data_error": reason,
                "market_data_backoff_until": backoff_until.isoformat(),
            },
        }
        cycle = self._vault_cycle(run)
        if cycle is not None and cycle.status == "active":
            metadata = dict(cycle.selection_metadata or {})
            runtime = dict(metadata.get("one_h10_runtime_notice", {}) or {})
            runtime.update(
                {
                    "kind": "market_data_backoff",
                    "message": reason,
                    "provider": provider or (run.parameters or {}).get("provider"),
                    "symbol": run.symbol,
                    "venue_symbol": (run.parameters or {}).get("venue_symbol"),
                    "timeframe": run.timeframe,
                    "blocker_category": blocker,
                    "retry_after": backoff_until.isoformat(),
                }
            )
            metadata["one_h10_runtime_notice"] = runtime
            metadata["one_h10_market_data_backoff_until"] = backoff_until.isoformat()
            metadata["one_h10_market_data_error"] = reason
            metadata["one_h10_market_data_blocker"] = blocker
            blockers = list(metadata.get("risk_blockers", []) or [])
            blockers.append(blocker)
            metadata["risk_blockers"] = list(dict.fromkeys(str(item) for item in blockers if str(item)))
            categories = list(metadata.get("blocker_categories", []) or [])
            categories.append(blocker)
            metadata["blocker_categories"] = list(dict.fromkeys(str(item) for item in categories if str(item)))
            cycle.selection_metadata = metadata
            if cycle.execution_substatus in {"initializing", "validating_market"}:
                cycle.execution_substatus = "executing"
        return True

    def _handle_provider_runtime_failure(self, run: StrategyRun, exc: Exception) -> bool:
        if self._handle_one_h10_market_data_failure(run, exc):
            return True
        if run.mode != "live" or not self._is_provider_transient_failure(exc):
            return False
        provider = self._run_provider(run)
        if not provider:
            return False
        now = datetime.utcnow()
        blocker = self._provider_failure_category(exc)
        backoff_seconds = self._runtime_provider_backoff_seconds(exc)
        backoff_until = now + timedelta(seconds=backoff_seconds)
        reason = self._sanitize_provider_runtime_error(run, exc)
        self._set_runtime_provider_backoff(
            provider=provider,
            connection_id=run.trading_connection_id or (run.parameters or {}).get("trading_connection_id"),
            run=run,
            reason=reason,
            blocker=blocker,
            retry_after=backoff_until,
            updated_at=now,
            include_one_h10=self._is_one_h10_run(run),
        )
        run.status = "running"
        run.manual_enabled = True
        run.last_error = reason
        run.last_heartbeat_at = now
        run.last_signal = {
            "action": "hold",
            "rationale": "Strategy skipped order generation because the provider is temporarily unavailable.",
            "timeframe": run.timeframe,
            "stop_loss": None,
            "take_profit": None,
            "position_fraction": 0.0,
            "metadata": {
                "no_trade_reason": "provider_runtime_backoff",
                "blocker_category": blocker,
                "provider": provider,
                "provider_backoff_until": backoff_until.isoformat(),
            },
        }
        return True

    def _runtime_provider_backoff_remaining(self, run: StrategyRun) -> float:
        candidates: list[str] = []
        if self._is_one_h10_run(run):
            params = dict(run.parameters or {})
            candidates.append(str(params.get("one_h10_market_data_backoff_until") or ""))
        provider = self._run_provider(run)
        connection_id = run.trading_connection_id or (run.parameters or {}).get("trading_connection_id")
        if provider:
            for key in (
                self._provider_runtime_backoff_key(provider, connection_id),
                self._one_h10_provider_backoff_key(provider, connection_id),
            ):
                payload = Setting.get_json(key, {})
                if isinstance(payload, dict):
                    candidates.append(str(payload.get("retry_after") or payload.get("backoff_until") or ""))
        remaining = 0.0
        for raw in candidates:
            if not raw:
                continue
            try:
                until = datetime.fromisoformat(raw.replace("Z", ""))
            except ValueError:
                continue
            remaining = max(remaining, (until - datetime.utcnow()).total_seconds())
        return max(remaining, 0.0)

    def _mark_runtime_provider_backoff_active(self, run: StrategyRun, remaining_seconds: float, *, reason: str | None = None) -> None:
        message = reason or f"Provider runtime backoff active for {remaining_seconds:.0f}s."
        run.status = "running"
        run.last_error = message
        run.last_heartbeat_at = datetime.utcnow()
        run.last_signal = {
            "action": "hold",
            "rationale": "Provider runtime backoff is active.",
            "timeframe": run.timeframe,
            "stop_loss": None,
            "take_profit": None,
            "position_fraction": 0.0,
            "metadata": {
                "no_trade_reason": "provider_runtime_backoff_active",
                "blocker_category": "rate_limited",
                "provider": self._run_provider(run),
                "one_h10_vault": self._is_one_h10_run(run),
                "ml_horizon": ONE_H10_HORIZON if self._is_one_h10_run(run) else None,
            },
        }

    def _set_runtime_provider_backoff(
        self,
        *,
        provider: str,
        connection_id: int | str | None,
        run: StrategyRun,
        reason: str,
        blocker: str,
        retry_after: datetime,
        updated_at: datetime,
        include_one_h10: bool = False,
    ) -> None:
        payload = {
            "provider": provider,
            "trading_connection_id": connection_id,
            "symbol": run.symbol,
            "venue_symbol": (run.parameters or {}).get("venue_symbol"),
            "timeframe": run.timeframe,
            "reason": reason,
            "blocker_category": blocker,
            "retry_after": retry_after.isoformat(),
            "updated_at": updated_at.isoformat(),
            "source": "strategy_runner",
        }
        Setting.set_json(self._provider_runtime_backoff_key(provider, connection_id), payload)
        if include_one_h10:
            Setting.set_json(self._one_h10_provider_backoff_key(provider, connection_id), payload)

    def _provider_runtime_lock(self, run: StrategyRun) -> threading.Lock | None:
        provider = self._run_provider(run)
        if run.mode != "live" or not provider:
            return None
        key = self._provider_runtime_lock_key(provider, run.trading_connection_id or (run.parameters or {}).get("trading_connection_id"))
        with self._provider_runtime_locks_guard:
            lock = self._provider_runtime_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._provider_runtime_locks[key] = lock
            return lock

    @staticmethod
    def _provider_runtime_lock_key(provider: str, connection_id: int | str | None = None) -> str:
        return f"{str(provider or 'provider').strip().lower()}:{connection_id or 'global'}"

    def _one_h10_market_data_backoff_remaining(self, run: StrategyRun) -> float:
        return self._runtime_provider_backoff_remaining(run) if self._is_one_h10_run(run) else 0.0

    @staticmethod
    def _one_h10_provider_backoff_key(provider: str, connection_id: int | str | None = None) -> str:
        provider_key = str(provider or "provider").strip().lower() or "provider"
        suffix = str(connection_id or "global")
        return f"one_h10_market_data_backoff:{provider_key}:{suffix}"

    @staticmethod
    def _provider_runtime_backoff_key(provider: str, connection_id: int | str | None = None) -> str:
        provider_key = str(provider or "provider").strip().lower() or "provider"
        suffix = str(connection_id or "global")
        return f"strategy_provider_runtime_backoff:{provider_key}:{suffix}"

    def _one_h10_market_data_backoff_seconds(self) -> float:
        return max(5.0, self._safe_float(self.config.get("ONE_H10_MARKET_DATA_BACKOFF_SECONDS"), 30.0))

    def _runtime_provider_wait_seconds(self) -> float:
        return max(5.0, self._safe_float(self.config.get("LIVE_PROVIDER_BACKOFF_MAX_WAIT_SECONDS"), 60.0))

    def _runtime_provider_backoff_seconds(self, exc: Exception) -> float:
        category = self._provider_failure_category(exc)
        if category == "rate_limited":
            return max(60.0, self._safe_float(self.config.get("LIVE_PROVIDER_RATE_LIMIT_BACKOFF_SECONDS"), 300.0))
        if category in {"network_timeout", "dns_failure", "connection_reset"}:
            return max(30.0, self._safe_float(self.config.get("LIVE_PROVIDER_NETWORK_BACKOFF_SECONDS"), 120.0))
        return max(10.0, self._safe_float(self.config.get("LIVE_PROVIDER_FAILURE_BACKOFF_SECONDS"), 60.0))

    def _run_candles(self, run: StrategyRun, market_mode: str, *, limit: int) -> list[dict[str, Any]]:
        params = dict(run.parameters or {})
        provider = str(params.get("provider") or params.get("execution_venue") or "").strip().lower()
        symbol = self._provider_market_symbol(run)
        if self._is_one_h10_run(run) and provider and provider not in {"hyperliquid", "global"}:
            connector = self._run_connector(run)
            getter = getattr(connector, "get_candles", None) if connector is not None else None
            if callable(getter):
                return list(getter(symbol, run.timeframe, market_mode, limit))
            raise RuntimeError(f"provider_market_data_unavailable: {provider} candles {symbol} {run.timeframe}")
        return self.market_data.get_candles(
            symbol,
            run.timeframe,
            mode=market_mode,
            limit=limit,
            retry=run.mode != "live",
        )

    def _run_mid_price(self, run: StrategyRun, market_mode: str) -> float:
        params = dict(run.parameters or {})
        provider = str(params.get("provider") or params.get("execution_venue") or "").strip().lower()
        symbol = self._provider_market_symbol(run)
        if self._is_one_h10_run(run) and provider and provider not in {"hyperliquid", "global"}:
            connector = self._run_connector(run)
            getter = getattr(connector, "get_mid_price", None) if connector is not None else None
            if callable(getter):
                return float(getter(symbol, market_mode))
            raise RuntimeError(f"provider_market_data_unavailable: {provider} mid_price {symbol}")
        return float(self.market_data.get_mid_price(symbol, market_mode))

    def _run_connector(self, run: StrategyRun) -> Any | None:
        services = self.app.extensions.get("services", {})
        trading_connections = services.get("trading_connections")
        if trading_connections is None or run.user_id is None:
            return None
        try:
            return trading_connections.connector_for_user(run.user_id, run.trading_connection_id)
        except Exception:
            return None

    @staticmethod
    def _provider_market_symbol(run: StrategyRun) -> str:
        params = dict(run.parameters or {})
        provider = str(params.get("provider") or params.get("execution_venue") or "").strip().lower()
        if provider == "hyperliquid":
            return str(params.get("venue_symbol") or params.get("provider_symbol") or run.symbol).strip()
        if provider:
            return str(params.get("venue_symbol") or params.get("provider_symbol") or run.symbol).upper()
        return str(params.get("app_symbol") or run.symbol).upper()

    @staticmethod
    def _is_one_h10_run(run: StrategyRun) -> bool:
        params = dict(run.parameters or {})
        markers = {
            str(params.get("algorithm_profile") or "").strip().lower(),
            str(params.get("vault_cycle_name") or "").strip().lower(),
            str(params.get("ml_horizon") or "").strip().lower(),
            str(params.get("objective") or "").strip().lower(),
        }
        return bool(params.get("one_h10_vault")) or bool(markers & {"1h10", "one_h10", "one_hour_10x"})

    @staticmethod
    def _is_market_data_failure(exc: Exception) -> bool:
        text = repr(exc).lower()
        return any(
            marker in text
            for marker in (
                "candles_snapshot",
                "l2_snapshot",
                "all_mids",
                "positions",
                "account_snapshot",
                "provider_market_data_unavailable",
                "market data",
                "market_data",
                "mid price unavailable",
                "429",
                "rate limit",
                "too many requests",
            )
        )

    @staticmethod
    def _market_data_blocker_category(exc: Exception) -> str:
        return StrategyManager._provider_failure_category(exc)

    @staticmethod
    def _provider_failure_category(exc: Exception) -> str:
        text = repr(exc).lower()
        if "429" in text or "rate limit" in text or "too many requests" in text:
            return "rate_limited"
        if "400002" in text or "kc-api-timestamp" in text:
            return "timestamp_skew"
        if "read timed out" in text or "connecttimeout" in text or "timed out" in text or "timeout" in text:
            return "network_timeout"
        if "nameresolutionerror" in text or "failed to resolve" in text or "nodename nor servname" in text:
            return "dns_failure"
        if "connectionreseterror" in text or "connection reset" in text or "connection aborted" in text:
            return "connection_reset"
        if "provider_market_data_unavailable" in text:
            return "provider_market_data_unavailable"
        return "features_stale"

    @staticmethod
    def _is_provider_transient_failure(exc: Exception) -> bool:
        return StrategyManager._provider_failure_category(exc) in {
            "rate_limited",
            "network_timeout",
            "dns_failure",
            "connection_reset",
            "timestamp_skew",
            "provider_market_data_unavailable",
        }

    def _sanitize_market_data_error(self, run: StrategyRun, exc: Exception) -> str:
        text = str(exc)
        provider = str((run.parameters or {}).get("provider") or (run.parameters or {}).get("execution_venue") or "provider").strip()
        symbol = self._provider_market_symbol(run)
        timeframe = str(run.timeframe or "").strip()
        if "429" in text or "rate limit" in text.lower() or "too many requests" in text.lower():
            return f"{provider.title()} rate limited {symbol} {timeframe}; retrying after backoff."
        if "provider_market_data_unavailable" in text:
            return f"{provider.title()} market data unavailable for {symbol} {timeframe}; waiting for provider-specific data."
        if "positions" in text.lower():
            return f"{provider.title()} positions unavailable; retrying after backoff."
        if "candles_snapshot" in text or "market data" in text.lower() or "market_data" in text.lower():
            return f"{provider.title()} market data unavailable for {symbol} {timeframe}; retrying after backoff."
        return text[:240]

    def _sanitize_provider_runtime_error(self, run: StrategyRun, exc: Exception) -> str:
        provider = self._run_provider(run) or "provider"
        category = self._provider_failure_category(exc)
        if category == "rate_limited":
            return f"{provider.title()} rate limited live requests; retrying after shared provider backoff."
        if category == "network_timeout":
            return f"{provider.title()} live request timed out; retrying after shared provider backoff."
        if category == "dns_failure":
            return f"{provider.title()} DNS resolution failed; retrying after shared provider backoff."
        if category == "connection_reset":
            return f"{provider.title()} connection reset; retrying after shared provider backoff."
        if category == "timestamp_skew":
            return f"{provider.title()} rejected the request timestamp; synced server time will be retried after backoff."
        if category == "provider_market_data_unavailable":
            return f"{provider.title()} market data unavailable; retrying after shared provider backoff."
        return str(exc)[:240]

    def _run_provider(self, run: StrategyRun) -> str:
        params = dict(run.parameters or {})
        provider = str(params.get("provider") or params.get("execution_venue") or "").strip().lower()
        if provider:
            return provider
        if run.trading_connection_id:
            connection = db.session.get(TradingConnection, int(run.trading_connection_id))
            if connection is not None:
                return str(connection.provider or "").strip().lower()
        return ""

    @staticmethod
    def _order_vault_cycle_id(order: Order) -> int | None:
        raw = order.vault_cycle_id
        if raw is None:
            raw = order.details.get("vault_cycle_id")
        try:
            return int(raw) if raw is not None and str(raw).strip() else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _order_vault_leg_id(order: Order) -> int | None:
        raw = order.vault_leg_id
        if raw is None:
            raw = order.details.get("vault_leg_id")
        try:
            return int(raw) if raw is not None and str(raw).strip() else None
        except (TypeError, ValueError):
            return None

    def _refresh_vault_estimate(self, run: StrategyRun) -> None:
        cycle = self._vault_cycle(run)
        if cycle is None:
            return
        realized = 0.0
        has_trading_data = False
        leg_totals: dict[int, float] = {}
        query = Order.query.filter_by(mode=run.mode).filter(or_(Order.vault_cycle_id == cycle.id, Order.vault_cycle_id.is_(None)))
        if run.user_id is not None:
            query = query.filter_by(user_id=run.user_id)
        for order in query.order_by(Order.created_at.asc()).all():
            if self._order_vault_cycle_id(order) != cycle.id:
                continue
            has_trading_data = True
            order_pnl = sum(
                float(fill.pnl or 0.0) - float(fill.fee or 0.0) - float(getattr(fill, "funding_fee", 0.0) or 0.0)
                for fill in order.fills
            )
            realized += order_pnl
            leg_id = self._order_vault_leg_id(order)
            if leg_id:
                leg_totals[int(leg_id)] = leg_totals.get(int(leg_id), 0.0) + order_pnl
        unrealized = 0.0
        legs = VaultAllocationLeg.query.filter_by(vault_cycle_id=cycle.id).all()
        if self._vault_estimate_backoff_active(run, cycle, legs):
            for leg in legs:
                leg.realized_pnl_usd = leg_totals.get(leg.id, 0.0)
            unrealized = sum(float(leg.unrealized_pnl_usd or 0.0) for leg in legs)
            total = realized + unrealized
            metadata = cycle.selection_metadata
            metadata["realized_pnl_usd"] = realized
            metadata["unrealized_pnl_usd"] = unrealized
            metadata["total_pnl_usd"] = total
            cycle.selection_metadata = metadata
            commit_with_retry()
            return
        if str(cycle.algorithm_profile or "").upper() != "1H10":
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
            return
        positions_by_connection = self._vault_live_positions_by_connection(run, cycle, legs)
        counted_positions: set[tuple[int, str]] = set()
        if legs:
            for leg in legs:
                leg.realized_pnl_usd = leg_totals.get(leg.id, 0.0)
                position, connection_id, matched_key = self._vault_position_for_leg(leg, positions_by_connection, cycle.trading_connection_id)
                if position is None and has_trading_data:
                    try:
                        position = self.order_manager.current_position(
                            leg.symbol,
                            run.mode,
                            run.user_id,
                            connection_id,
                        )
                        matched_key = str(position.get("symbol") or leg.symbol or "").upper()
                    except Exception:  # noqa: BLE001
                        position = None
                if position is not None:
                    leg.unrealized_pnl_usd = float(position.get("unrealized_pnl", 0.0) or 0.0)
                    position_key = str(position.get("symbol") or matched_key or leg.symbol or "").upper()
                    if connection_id is not None and (connection_id, position_key) not in counted_positions:
                        unrealized += leg.unrealized_pnl_usd
                        counted_positions.add((connection_id, position_key))
                else:
                    leg.unrealized_pnl_usd = 0.0
        else:
            position = None
            if run.trading_connection_id is not None:
                positions = self._vault_live_positions_by_connection(run, cycle, [])
                for key in self._position_lookup_keys(
                    run.parameters.get("venue_symbol"),
                    run.parameters.get("provider_symbol"),
                    run.parameters.get("app_symbol"),
                    run.symbol,
                ):
                    position = positions.get((int(run.trading_connection_id), key))
                    if position is not None:
                        break
            unrealized = float(position.get("unrealized_pnl", 0.0) or 0.0) if position is not None else 0.0
        total = realized + unrealized
        if has_trading_data:
            cycle.current_estimated_value_usd = max(float(cycle.starting_value_usd or 0.0) + total, 0.0)
        metadata = cycle.selection_metadata
        metadata["realized_pnl_usd"] = realized
        metadata["unrealized_pnl_usd"] = unrealized
        metadata["total_pnl_usd"] = total
        cycle.selection_metadata = metadata
        commit_with_retry()

    def _vault_estimate_backoff_active(
        self,
        run: StrategyRun,
        cycle: VaultCycle,
        legs: list[VaultAllocationLeg],
    ) -> bool:
        if run.mode != "live" or str(cycle.algorithm_profile or "").upper() != "1H10":
            return False
        if self._runtime_provider_backoff_remaining(run) > 0:
            return True
        for leg in legs:
            provider = str((leg.details or {}).get("provider") or leg.provider or "").strip().lower()
            connection_id = leg.trading_connection_id or (leg.details or {}).get("trading_connection_id") or cycle.trading_connection_id
            if not provider:
                continue
            for key in (
                self._provider_runtime_backoff_key(provider, connection_id),
                self._one_h10_provider_backoff_key(provider, connection_id),
            ):
                payload = Setting.get_json(key, {})
                if not isinstance(payload, dict):
                    continue
                raw = str(payload.get("retry_after") or payload.get("backoff_until") or "")
                if not raw:
                    continue
                try:
                    if datetime.fromisoformat(raw.replace("Z", "")) > datetime.utcnow():
                        return True
                except ValueError:
                    continue
        return False

    def _vault_live_positions_by_connection(
        self,
        run: StrategyRun,
        cycle: VaultCycle,
        legs: list[VaultAllocationLeg],
    ) -> dict[tuple[int, str], dict[str, Any]]:
        positions: dict[tuple[int, str], dict[str, Any]] = {}
        if run.mode != "live" or run.user_id is None or self.order_manager.trading_connections is None:
            return positions
        connection_ids: set[int] = set()
        for leg in legs:
            connection_raw = leg.trading_connection_id or (leg.details or {}).get("trading_connection_id") or cycle.trading_connection_id
            try:
                if connection_raw is not None:
                    connection_ids.add(int(connection_raw))
            except (TypeError, ValueError):
                continue
        if not connection_ids and run.trading_connection_id is not None:
            connection_ids.add(int(run.trading_connection_id))
        for connection_id in connection_ids:
            try:
                snapshot = self.order_manager.trading_connections.account_snapshot(run.user_id, "live", connection_id)
            except Exception:  # noqa: BLE001
                continue
            if getattr(snapshot, "alerts", None) and not getattr(snapshot, "positions", None):
                continue
            for position in getattr(snapshot, "positions", []) or []:
                if not isinstance(position, dict):
                    continue
                for key in self._position_lookup_keys(
                    position.get("symbol"),
                    position.get("venue_symbol"),
                    position.get("coin"),
                ):
                    positions[(connection_id, key)] = position
        return positions

    def _vault_position_for_leg(
        self,
        leg: VaultAllocationLeg,
        positions_by_connection: dict[tuple[int, str], dict[str, Any]],
        fallback_connection_id: int | None,
    ) -> tuple[dict[str, Any] | None, int | None, str | None]:
        details = dict(leg.details or {})
        raw_connection_id = leg.trading_connection_id or details.get("trading_connection_id") or fallback_connection_id
        try:
            connection_id = int(raw_connection_id) if raw_connection_id is not None and str(raw_connection_id).strip() else None
        except (TypeError, ValueError):
            connection_id = None
        if connection_id is None:
            return None, None, None
        for key in self._position_lookup_keys(
            details.get("venue_symbol"),
            details.get("provider_symbol"),
            details.get("app_symbol"),
            leg.symbol,
        ):
            position = positions_by_connection.get((connection_id, key))
            if position is not None:
                return position, connection_id, key
        return None, connection_id, None

    @staticmethod
    def _position_lookup_keys(*values: Any) -> list[str]:
        keys: list[str] = []
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            keys.append(text)
            upper = text.upper()
            if upper != text:
                keys.append(upper)
        return list(dict.fromkeys(keys))

    def _learn_from_order_outcome(self, run: StrategyRun, order: Order) -> None:
        if not bool(self.config.get("ML_RANKER_ENABLED", False)):
            return
        if order.status != "filled" or not order.fills:
            return
        if order.mode != "live" and not self.online_ranker.should_update_from_mode(order.mode):
            return
        if order.reduce_only and any(not bool(getattr(fill, "realized_pnl_known", True)) for fill in order.fills):
            return
        realized = sum(
            float(fill.pnl or 0.0) - float(fill.fee or 0.0) - float(getattr(fill, "funding_fee", 0.0) or 0.0)
            for fill in order.fills
        )
        notional = abs(float(order.filled_quantity or order.quantity or 0.0) * float(order.average_fill_price or 0.0))
        if not order.reduce_only and abs(realized) < 1e-12:
            return
        details = order.details or {}
        duration_hours = 0
        cycle = self._vault_cycle(run)
        if cycle is not None:
            duration_hours = int(cycle.lock_duration_hours or 0)
        horizon = self._run_ml_horizon(run, duration_hours or details.get("lock_duration_hours") or 24)
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

    def _run_ml_horizon(self, run: StrategyRun, fallback_duration_hours: Any = 1) -> str:
        params = dict(run.parameters or {})
        explicit = str(params.get("ml_horizon") or "").strip().lower()
        if explicit:
            return explicit
        if bool(params.get("one_h10_vault")) or str(params.get("algorithm_profile") or "").upper() == "1H10":
            return ONE_H10_HORIZON
        return horizon_from_context(params, fallback_duration_hours or params.get("lock_duration_hours") or 1)

    def _vault_cycle(self, run: StrategyRun) -> VaultCycle | None:
        cycle_id = run.parameters.get("vault_cycle_id")
        if not cycle_id:
            return None
        with db.session.no_autoflush:
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
        ml_decision = self._execution_style_ml_decision(run, feature_payload, payload)
        if ml_decision:
            payload["ml_decision"] = ml_decision
            payload["ml_execution_style_decision"] = ml_decision
            suggested = str((ml_decision.get("raw") or {}).get("execution_style_suggestion") or "")
            if bool(ml_decision.get("ready", False)) and suggested:
                payload["suggested_execution_style"] = suggested
        return payload

    def _dynamic_leverage(
        self,
        run: StrategyRun,
        edge_payload: dict[str, Any],
        feature_payload: dict[str, Any],
    ) -> float:
        requested = max(1.0, self._safe_float(run.parameters.get("leverage"), 1.0))
        forecast = run.parameters.get("one_h10_forecast") if isinstance(run.parameters.get("one_h10_forecast"), dict) else {}
        if bool((run.parameters or {}).get("one_h10_vault")) and forecast:
            requested = max(requested, self._safe_float(forecast.get("suggested_leverage"), requested))
            hard_cap = min(
                self._safe_float(self.config.get("MAX_LEVERAGE"), requested),
                self._safe_float(self.config.get("ONE_H10_MAX_LEVERAGE"), requested),
                self._safe_float(run.parameters.get("max_leverage"), self._safe_float(run.parameters.get("market_max_leverage"), requested)),
            )
            return max(1.0, min(requested, hard_cap))
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

    def _execution_style_ml_decision(
        self,
        run: StrategyRun,
        feature_payload: dict[str, Any],
        edge_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if self.ml_decision_engine is None:
            return {}
        if not bool(self.config.get("ML_ALL_AREAS_ENABLED", False)):
            return {}
        if not bool(self.config.get("ML_ALLOW_EXECUTION_STYLE_SUGGESTIONS", True)):
            return {}
        horizon = self._run_ml_horizon(run, run.parameters.get("duration_hours") or run.parameters.get("lock_duration_hours") or 1)
        try:
            return dict(
                self.ml_decision_engine.decision(
                    "pytorch_allocator",
                    {
                        **dict(run.parameters or {}),
                        **dict(feature_payload or {}),
                        **dict(edge_payload or {}),
                        "strategy_name": run.strategy_name,
                        "symbol": run.symbol,
                        "timeframe": run.timeframe,
                        "mode": run.mode,
                        "horizon": horizon,
                    },
                    horizon=horizon,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "family": "pytorch_allocator",
                "ready": False,
                "action": "hold",
                "blockers": [str(exc)],
                "audit_metadata": {"status": "ml_execution_style_error"},
            }

    def _one_h10_rebalance_tick(self, run: StrategyRun) -> None:
        params = dict(run.parameters or {})
        if not bool(params.get("one_h10_vault")):
            return
        interval = max(5.0, self._safe_float(self.config.get("ONE_H10_REBALANCE_SECONDS"), 15.0))
        last_raw = str(params.get("one_h10_last_rebalance_at") or "")
        if last_raw:
            try:
                last_at = datetime.fromisoformat(last_raw)
                if (datetime.utcnow() - last_at).total_seconds() < interval:
                    return
            except ValueError:
                pass
        services = self.app.extensions.get("services", {})
        market_service = services.get("leveraged_markets")
        scanner = services.get("market_scanner")
        if market_service is None or scanner is None:
            return
        provider = str(params.get("provider") or params.get("execution_venue") or "")
        try:
            markets = market_service.active_markets(provider=provider, symbols=None)
            ranked = scanner.score_one_h10_markets(
                markets,
                provider=provider,
                limit=max(1, int(self.config.get("ONE_H10_MAX_PROVIDER_LEGS", 3) or 3) * 2),
            )
        except Exception as exc:  # noqa: BLE001
            params["one_h10_last_rebalance_at"] = datetime.utcnow().isoformat()
            params["one_h10_rebalance_error"] = str(exc)
            run.parameters = params
            commit_with_retry()
            return
        used_venues = self._one_h10_used_provider_venues(run, provider)
        current_venue_symbol = str(params.get("venue_symbol") or params.get("provider_symbol") or run.symbol).strip()
        if provider.lower() != "hyperliquid":
            current_venue_symbol = current_venue_symbol.upper()
        candidate = None
        for ranked_candidate in ranked:
            ranked_features = dict(getattr(ranked_candidate, "features", {}) or {})
            ranked_app_symbol = str(
                ranked_features.get("app_symbol")
                or ranked_features.get("symbol")
                or getattr(ranked_candidate, "symbol", "")
            ).strip().upper()
            ranked_venue_symbol = str(
                ranked_features.get("venue_symbol")
                or ranked_features.get("provider_symbol")
                or ranked_app_symbol
            ).strip()
            if provider.lower() != "hyperliquid":
                ranked_venue_symbol = ranked_venue_symbol.upper()
            if ranked_venue_symbol == current_venue_symbol or ranked_venue_symbol not in used_venues:
                candidate = ranked_candidate
                break
        params["one_h10_last_rebalance_at"] = datetime.utcnow().isoformat()
        params["one_h10_rebalance_candidate"] = self._one_h10_candidate_summary(candidate)
        switched = False
        candidate_features = dict(getattr(candidate, "features", {}) or {}) if candidate is not None else {}
        candidate_app_symbol = str(
            candidate_features.get("app_symbol")
            or candidate_features.get("symbol")
            or (getattr(candidate, "symbol", "") if candidate is not None else "")
            or run.symbol
        ).strip().upper()
        candidate_venue_symbol = str(
            candidate_features.get("venue_symbol")
            or candidate_features.get("provider_symbol")
            or candidate_app_symbol
        ).strip()
        if provider.lower() != "hyperliquid":
            candidate_venue_symbol = candidate_venue_symbol.upper()
        identity_changed = candidate is not None and (
            candidate_app_symbol != str(run.symbol or "").upper()
            or candidate_venue_symbol != current_venue_symbol
        )
        if identity_changed:
            try:
                position = self.order_manager.current_position(run.symbol, run.mode, run.user_id, run.trading_connection_id)
                flat = abs(self._safe_float(position.get("quantity"))) <= 1e-12
            except Exception:  # noqa: BLE001
                flat = False
            if flat:
                forecast = self._one_h10_rebalance_forecast(
                    services,
                    markets,
                    candidate_features,
                    provider=provider,
                    symbol=candidate_app_symbol,
                    allocation_cap_usd=self._safe_float(params.get("allocation_cap_usd"), 0.0),
                    available_margin_usd=self._safe_float(params.get("available_margin_usd"), 0.0),
                )
                params["one_h10_previous_symbol"] = run.symbol
                params["one_h10_previous_venue_symbol"] = current_venue_symbol
                params["one_h10_rebalance_reason"] = "higher_ranked_flat_provider_symbol"
                params["one_h10_scanner_score"] = candidate.score
                feature_summary = self._one_h10_feature_summary(candidate_features)
                if forecast:
                    feature_summary["one_h10_forecast"] = forecast
                feature_summary.update(
                    {
                        "symbol": candidate_app_symbol,
                        "app_symbol": candidate_app_symbol,
                        "venue_symbol": candidate_venue_symbol,
                        "provider_symbol": candidate_venue_symbol,
                        "market_id": candidate_features.get("market_id"),
                    }
                )
                params["scanner_features"] = feature_summary
                params["symbol"] = candidate_app_symbol
                params["app_symbol"] = candidate_app_symbol
                params["venue_symbol"] = candidate_venue_symbol
                params["provider_symbol"] = candidate_venue_symbol
                params["market_id"] = candidate_features.get("market_id")
                params["market_status"] = candidate_features.get("market_status", params.get("market_status"))
                if forecast:
                    params["one_h10_forecast"] = forecast
                    params["forecast_metadata"] = forecast
                    params["forecast_blockers"] = list(forecast.get("blockers", []) or [])
                    params["forecast_advisory_blockers"] = list(forecast.get("advisory_blockers", []) or [])
                    params["forecast_predicted_side"] = forecast.get("predicted_side")
                    params["forecast_confidence"] = forecast.get("confidence")
                    params["forecast_expected_return_bps"] = forecast.get("expected_return_bps")
                    params["forecast_suggested_notional_usd"] = forecast.get("suggested_notional_usd")
                    params["forecast_suggested_leverage"] = forecast.get("suggested_leverage")
                    params["forecast_suggested_order_type"] = forecast.get("suggested_order_type")
                    params["forecast_suggested_stop_loss_pct"] = forecast.get("suggested_stop_loss_pct")
                    params["forecast_suggested_take_profit_pct"] = forecast.get("suggested_take_profit_pct")
                run.symbol = candidate_app_symbol
                switched = True
                leg_id = params.get("vault_leg_id")
                if leg_id:
                    with db.session.no_autoflush:
                        leg = db.session.get(VaultAllocationLeg, int(leg_id))
                    if leg is not None:
                        leg.symbol = run.symbol
                        details = dict(leg.details or {})
                        details.update(
                            {
                                "one_h10_rebalanced": True,
                                "one_h10_scanner_score": candidate.score,
                                "scanner_features": feature_summary,
                                "symbol": candidate_app_symbol,
                                "app_symbol": candidate_app_symbol,
                                "venue_symbol": candidate_venue_symbol,
                                "provider_symbol": candidate_venue_symbol,
                                "market_id": candidate_features.get("market_id"),
                                "market_status": candidate_features.get("market_status", details.get("market_status")),
                            }
                        )
                        if forecast:
                            details.update(
                                {
                                    "one_h10_forecast": forecast,
                                    "forecast_metadata": forecast,
                                    "forecast_blockers": list(forecast.get("blockers", []) or []),
                                    "forecast_advisory_blockers": list(forecast.get("advisory_blockers", []) or []),
                                    "forecast_predicted_side": forecast.get("predicted_side"),
                                    "forecast_confidence": forecast.get("confidence"),
                                    "forecast_expected_return_bps": forecast.get("expected_return_bps"),
                                    "forecast_suggested_notional_usd": forecast.get("suggested_notional_usd"),
                                    "forecast_suggested_leverage": forecast.get("suggested_leverage"),
                                    "forecast_suggested_order_type": forecast.get("suggested_order_type"),
                                    "forecast_suggested_stop_loss_pct": forecast.get("suggested_stop_loss_pct"),
                                    "forecast_suggested_take_profit_pct": forecast.get("suggested_take_profit_pct"),
                                }
                            )
                        leg.details = details
        run.parameters = params
        cycle = self._vault_cycle(run)
        if cycle is not None:
            metadata = dict(cycle.selection_metadata or {})
            history = list(metadata.get("one_h10_rebalance_history", []) or [])
            history.append(
                {
                    "run_id": run.id,
                    "provider": provider,
                    "symbol": run.symbol,
                    "candidate": self._one_h10_candidate_summary(candidate),
                    "switched": switched,
                    "timestamp": params["one_h10_last_rebalance_at"],
                }
            )
            metadata["one_h10_rebalance_history"] = history[-20:]
            cycle.selection_metadata = metadata
        commit_with_retry()

    def _one_h10_candidate_summary(self, candidate: Any | None) -> dict[str, Any]:
        if candidate is None:
            return {}
        features = dict(getattr(candidate, "features", {}) or {})
        return {
            "symbol": getattr(candidate, "symbol", ""),
            "app_symbol": str(features.get("app_symbol") or features.get("symbol") or getattr(candidate, "symbol", "")).upper(),
            "venue_symbol": features.get("venue_symbol") or features.get("provider_symbol"),
            "market_id": features.get("market_id"),
            "score": self._safe_float(getattr(candidate, "score", 0.0), 0.0),
            "technical_score": self._safe_float(getattr(candidate, "technical_score", 0.0), 0.0),
            "ml_score": self._safe_float(getattr(candidate, "ml_score", 0.0), 0.0),
            "hot_score": self._safe_float(getattr(candidate, "hot_score", 0.0), 0.0),
            "source": getattr(candidate, "source", ""),
            "score_breakdown": dict(getattr(candidate, "score_breakdown", None) or {}),
            "rejection_reason": getattr(candidate, "rejection_reason", ""),
            "stale_data": bool(getattr(candidate, "stale_data", False)),
            "features": self._one_h10_feature_summary(features),
        }

    def _one_h10_used_provider_venues(self, run: StrategyRun, provider: str) -> set[str]:
        cycle_id = (run.parameters or {}).get("vault_cycle_id")
        if not cycle_id:
            return set()
        try:
            cycle_id_int = int(cycle_id)
        except (TypeError, ValueError):
            return set()
        used: set[str] = set()
        with db.session.no_autoflush:
            legs = VaultAllocationLeg.query.filter_by(vault_cycle_id=cycle_id_int, provider=provider).all()
        for leg in legs:
            if leg.strategy_run_id == run.id:
                continue
            sibling_run = db.session.get(StrategyRun, leg.strategy_run_id) if leg.strategy_run_id else None
            sibling_params = dict(getattr(sibling_run, "parameters", {}) or {}) if sibling_run is not None else {}
            details = dict(leg.details or {})
            venue = str(
                sibling_params.get("venue_symbol")
                or sibling_params.get("provider_symbol")
                or details.get("venue_symbol")
                or details.get("provider_symbol")
                or leg.symbol
                or ""
            ).strip()
            if not venue:
                continue
            if provider.lower() != "hyperliquid":
                venue = venue.upper()
            used.add(venue)
        return used

    def _one_h10_rebalance_forecast(
        self,
        services: dict[str, Any],
        markets: list[Any],
        features: dict[str, Any],
        *,
        provider: str,
        symbol: str,
        allocation_cap_usd: float,
        available_margin_usd: float,
    ) -> dict[str, Any]:
        forecast_service = services.get("one_h10_forecast")
        if forecast_service is None:
            return {}
        market = None
        market_id = features.get("market_id")
        if market_id is not None:
            for candidate_market in markets:
                try:
                    if int(getattr(candidate_market, "id", 0) or 0) == int(market_id):
                        market = candidate_market
                        break
                except (TypeError, ValueError):
                    continue
        try:
            return dict(
                forecast_service.forecast(
                    features,
                    provider=provider,
                    symbol=symbol,
                    allocation_cap_usd=allocation_cap_usd,
                    available_margin_usd=available_margin_usd,
                    market=market,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return {"blockers": ["one_h10_rebalance_forecast_error"], "advisory_blockers": [str(exc)]}

    def _one_h10_feature_summary(self, features: dict[str, Any]) -> dict[str, Any]:
        keys = {
            "fib_confluence_score",
            "fib_nearest_level_bps",
            "rsi",
            "ema_fast",
            "ema_slow",
            "sma_fast",
            "sma_slow",
            "macd_histogram",
            "atr_pct",
            "volatility_pct",
            "liquidity_usd",
            "volume",
            "funding_rate",
            "spread_bps",
            "cost_drag_bps",
            "expected_move_bps",
            "gross_expected_return_bps",
            "net_expected_return_bps",
            "edge_after_cost_bps",
            "expected_execution_quality",
            "slippage_bps",
            "order_book_imbalance",
            "book_imbalance",
        }
        summary: dict[str, Any] = {}
        for key in keys:
            if key in features:
                value = features.get(key)
                summary[key] = self._safe_float(value, 0.0) if isinstance(value, (int, float, str)) else value
        return summary

    def _record_no_trade(self, run: StrategyRun, edge_payload: dict[str, Any]) -> None:
        blocker_category = self._no_trade_blocker_category(edge_payload)
        reason = str(edge_payload.get("no_trade_reason") or edge_payload.get("reason") or "unknown").strip() or "unknown"
        suppressed_count = self._claim_no_trade_audit_slot(run.id, reason, blocker_category)
        if suppressed_count is None:
            return
        audit = AuditLog(
            category="strategy",
            action="no_trade",
            message=f"Strategy signal skipped: {reason}.",
            user_id=run.user_id,
            trading_connection_id=run.trading_connection_id,
        )
        base_details = {
            "run_id": run.id,
            "symbol": run.symbol,
            "timeframe": run.timeframe,
            "strategy_name": run.strategy_name,
            "vault_cycle_id": run.parameters.get("vault_cycle_id"),
            "vault_leg_id": run.parameters.get("vault_leg_id"),
            "optimizer_profile": run.parameters.get("optimizer_profile"),
            "one_h10_vault": run.parameters.get("one_h10_vault"),
            "ml_horizon": run.parameters.get("ml_horizon"),
            "provider": run.parameters.get("provider"),
            "blocker_category": blocker_category,
            "no_trade_reason": reason,
            "suppressed_since_last": suppressed_count,
        }
        if bool(self.config.get("NO_TRADE_AUDIT_COMPACT_ENABLED", True)):
            audit.details = {**base_details, **self._compact_no_trade_payload(edge_payload)}
        else:
            audit.details = {**base_details, **edge_payload}
        db.session.add(audit)
        commit_with_retry()

    def _claim_no_trade_audit_slot(self, run_id: int, reason: str, blocker_category: str) -> int | None:
        throttle = max(0.0, self._safe_float(self.config.get("NO_TRADE_AUDIT_THROTTLE_SECONDS"), 300.0))
        if throttle <= 0:
            return 0
        key = (int(run_id or 0), str(reason or ""), str(blocker_category or ""))
        now = time.time()
        with self._no_trade_audit_guard:
            state = self._no_trade_audit_state.get(key)
            if state is not None and now - float(state.get("last_at", 0.0) or 0.0) < throttle:
                state["suppressed"] = int(state.get("suppressed", 0) or 0) + 1
                return None
            suppressed = int(state.get("suppressed", 0) or 0) if state else 0
            self._no_trade_audit_state[key] = {"last_at": now, "suppressed": 0}
            return suppressed

    @staticmethod
    def _compact_no_trade_payload(edge_payload: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "edge_score",
            "net_roi_score",
            "net_roi_v2_score",
            "one_hour_edge_v2",
            "one_hour_edge_grade",
            "expected_move_bps",
            "cost_drag_bps",
            "gross_expected_return_bps",
            "net_expected_return_bps",
            "edge_after_cost_bps",
            "confidence",
            "expected_execution_quality",
            "slippage_bps",
            "spread_bps",
            "liquidity_usd",
            "signal_stability",
            "market_source",
            "forecast_predicted_side",
            "forecast_confidence",
            "forecast_expected_return_bps",
            "suggested_execution_style",
        )
        payload = {key: edge_payload.get(key) for key in keys if key in edge_payload}
        for key in ("quality_reasons", "forecast_blockers", "forecast_advisory_blockers"):
            values = edge_payload.get(key)
            if isinstance(values, list):
                payload[key] = [str(item) for item in values[:8]]
        for key in ("fibonacci_alignment", "feature_confluence", "ml_signal_quality", "ml_decision", "ml_execution_style_decision"):
            value = edge_payload.get(key)
            if isinstance(value, dict):
                payload[key] = StrategyManager._compact_mapping(value)
        return payload

    @staticmethod
    def _compact_mapping(value: dict[str, Any], *, limit: int = 12) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= limit:
                compact["_truncated"] = True
                break
            if isinstance(item, (str, int, float, bool)) or item is None:
                compact[str(key)] = item
            elif isinstance(item, list):
                compact[str(key)] = [str(entry) for entry in item[:6]]
            elif isinstance(item, dict):
                compact[str(key)] = StrategyManager._compact_mapping(item, limit=6)
            else:
                compact[str(key)] = str(item)
        return compact

    @staticmethod
    def _no_trade_blocker_category(edge_payload: dict[str, Any]) -> str:
        reason = str(edge_payload.get("no_trade_reason") or edge_payload.get("reason") or "").lower()
        if "ml" in reason and ("not_ready" in reason or "promoted" in reason or "missing" in reason):
            return "ml_not_ready"
        if "hold" in reason or "low_confidence" in reason or "zero_sizing" in reason:
            return "ml_hold"
        if "liquidity" in reason:
            return "liquidity_too_low"
        if "slippage" in reason or "spread" in reason:
            return "slippage_too_high"
        if "stale" in reason:
            return "features_stale"
        return "ml_hold" if reason else "risk_rejected"

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
        forecast = run.parameters.get("one_h10_forecast") if isinstance(run.parameters.get("one_h10_forecast"), dict) else {}
        if bool(run.parameters.get("one_h10_vault")) and str(forecast.get("suggested_order_type") or "").lower() == "limit":
            offset = min(max(self._safe_float(edge_payload.get("spread_bps", run.parameters.get("spread_bps", 1.0)), 1.0) / 20_000, 0.0001), 0.0015)
            return "limit", round(mid * (1 - offset if side == "buy" else 1 + offset), 8)
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
        if bool((run.parameters or {}).get("one_h10_vault")):
            forecast = run.parameters.get("one_h10_forecast") if isinstance(run.parameters.get("one_h10_forecast"), dict) else {}
            suggested = self._safe_float(forecast.get("suggested_notional_usd"), 0.0)
            allocation_cap = self._safe_float(run.parameters.get("allocation_cap_usd"), 0.0)
            available = self._safe_float(run.parameters.get("available_margin_usd"), 0.0)
            candidates = [value for value in (suggested, allocation_cap, available) if value > 0]
            return min(candidates) if candidates else 0.0
        allocation_cap = self._safe_float(run.parameters.get("allocation_cap_usd"), 0.0)
        if allocation_cap > 0:
            return allocation_cap
        available_margin = self._safe_float(run.parameters.get("available_margin_usd"), 0.0)
        if available_margin > 0:
            return available_margin
        return max(0.0, self._safe_float(self.config.get("FIXED_DOLLAR_SIZE"), 0.0))

    def _poll_seconds(self, run: StrategyRun) -> float:
        default = float(self.config.get("STRATEGY_POLL_SECONDS", 20))
        profile = str((run.parameters or {}).get("optimizer_profile", "")).lower()
        if bool((run.parameters or {}).get("one_h10_vault")) or str((run.parameters or {}).get("ml_horizon") or "").lower() == ONE_H10_HORIZON:
            return max(1.0, min(default, float(self.config.get("ONE_H10_POLL_SECONDS", 1.0))))
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

    def _high_upside_ml_signal(
        self,
        run: StrategyRun,
        signal: Any,
        candles: list[dict[str, Any]],
        feature_payload: dict[str, Any],
        market_mode: str,
    ) -> Any:
        high_upside = bool((run.parameters or {}).get("high_upside_profile", False))
        ml_first = bool(self.config.get("ML_FIRST_STRATEGIES_ENABLED", False)) and bool(
            self.config.get("ML_ALLOW_STRATEGY_SIGNAL_OVERRIDE", True)
        )
        if not high_upside and not ml_first:
            return signal
        required = (bool(self.config.get("HIGH_UPSIDE_REQUIRE_ML_SIGNAL", True)) and run.mode == "live" and high_upside) or ml_first
        enabled = bool(self.config.get("ML_SIGNAL_MODEL_ENABLED", False))
        if not required and not enabled:
            return signal

        horizon = self._run_ml_horizon(run, run.parameters.get("duration_hours") or run.parameters.get("lock_duration_hours") or 1)
        context = {
            **dict(run.parameters or {}),
            **dict(feature_payload or {}),
            "strategy_name": run.strategy_name,
            "symbol": run.symbol,
            "timeframe": run.timeframe,
            "mode": run.mode,
            "market_mode": market_mode,
            "base_signal_action": getattr(signal, "action", "hold"),
            "base_signal_confidence": (getattr(signal, "metadata", {}) or {}).get("confidence", 0.0),
            "horizon": horizon,
        }
        decision_payload: dict[str, Any] = {}
        if self.ml_decision_engine is not None:
            try:
                decision_payload = dict(
                    self.ml_decision_engine.decision(
                        "pytorch_gru_signal",
                        context,
                        horizon=horizon,
                        candles=candles,
                    )
                )
                payload = dict((decision_payload.get("raw") or {}) if isinstance(decision_payload.get("raw"), dict) else {})
            except Exception:  # noqa: BLE001
                decision_payload = {}
                payload = {}
        else:
            payload = {}
        if not payload and self.ml_signal_model is None:
            payload = {
                "status": "service_unavailable",
                "ready_for_live": False,
                "action": "hold",
                "confidence": 0.0,
                "blockers": ["ml_signal_model_service_unavailable"],
            }
        elif not payload:
            payload = self.ml_signal_model.score_payload(context, horizon, candles=candles)

        metadata = dict(getattr(signal, "metadata", {}) or {})
        metadata["feature_snapshot"] = feature_payload
        metadata["ml_feature_schema_version"] = "ml_feature_v1"
        if decision_payload:
            metadata["ml_decision"] = decision_payload
            metadata["ml_signal_decision"] = decision_payload
        metadata["ml_signal_model"] = payload
        metadata["ml_signal_quality"] = payload
        metadata["confidence"] = self._safe_float(payload.get("confidence"), metadata.get("confidence", 0.0))
        blockers = list(payload.get("blockers", []) or [])
        metadata["ml_safety_blockers"] = list(dict.fromkeys(blockers))
        one_h10_bootstrap = (
            bool((run.parameters or {}).get("one_h10_vault"))
            and bool(self.config.get("ONE_H10_BOOTSTRAP_LIVE_ENABLED", True))
            and run.mode == "live"
        )
        if required and not bool(payload.get("ready_for_live", False)) and one_h10_bootstrap:
            metadata["one_h10_bootstrap_live"] = True
            metadata["one_h10_bootstrap_signal_source"] = "deterministic_strategy"
            metadata["ml_signal_not_ready"] = True
            return Signal(
                signal.action,
                signal.rationale,
                signal.timeframe,
                signal.stop_loss,
                signal.take_profit,
                signal.position_fraction,
                metadata,
            )
        if required and not bool(payload.get("ready_for_live", False)):
            reason = "ml_signal_blocked:" + ",".join(blockers or [str(payload.get("status") or "not_ready")])
            metadata["no_trade_reason"] = reason
            return Signal("hold", reason, run.timeframe, None, None, 0.0, metadata)
        min_confidence = self._safe_float(
            self.config.get("ML_MIN_SIGNAL_CONFIDENCE", self.config.get("ML_SIGNAL_MIN_CONFIDENCE", 0.60)),
            0.60,
        )
        if required and self._safe_float(payload.get("confidence"), 0.0) < min_confidence:
            metadata["no_trade_reason"] = "ml_signal_blocked:low_confidence"
            metadata["ml_safety_blockers"] = list(dict.fromkeys([*metadata["ml_safety_blockers"], "low_confidence"]))
            return Signal("hold", "ML signal blocked because confidence is below threshold.", run.timeframe, None, None, 0.0, metadata)

        action = str(payload.get("action") or "hold").lower()
        if action not in {"buy", "sell"}:
            metadata["no_trade_reason"] = "ml_signal_hold"
            return Signal("hold", "ML signal selected hold.", run.timeframe, None, None, 0.0, metadata)

        mid = self._safe_float((candles[-1] if candles else {}).get("close"), 0.0)
        if mid <= 0:
            metadata["no_trade_reason"] = "ml_signal_mid_unavailable"
            return Signal("hold", "ML signal blocked because mid price is unavailable.", run.timeframe, None, None, 0.0, metadata)

        stop_pct = max(self._safe_float(payload.get("suggested_stop_loss_pct")), 0.0)
        take_pct = max(self._safe_float(payload.get("suggested_take_profit_pct")), 0.0)
        if stop_pct <= 0 or take_pct <= 0:
            metadata["no_trade_reason"] = "ml_signal_missing_exit"
            return Signal("hold", "ML signal blocked because stop loss or take profit is missing.", run.timeframe, None, None, 0.0, metadata)

        if action == "buy":
            stop_loss = mid * (1 - stop_pct)
            take_profit = mid * (1 + take_pct)
        else:
            stop_loss = mid * (1 + stop_pct)
            take_profit = mid * (1 - take_pct)
        base_fraction = self._safe_float(getattr(signal, "position_fraction", 0.0), 0.0)
        ml_fraction = self._safe_float(payload.get("position_fraction", payload.get("sizing_score")), 0.0)
        position_fraction = max(0.0, min(base_fraction if base_fraction > 0 else 1.0, ml_fraction, 1.0))
        if position_fraction <= 0:
            metadata["no_trade_reason"] = "ml_signal_zero_sizing"
            return Signal("hold", "ML signal blocked because sizing score is zero.", run.timeframe, None, None, 0.0, metadata)
        return Signal(
            action,
            f"Promoted ML signal selected {action}.",
            run.timeframe,
            stop_loss,
            take_profit,
            position_fraction,
            metadata,
        )

    def _one_h10_forecast_signal(self, run: StrategyRun, signal: Any, candles: list[dict[str, Any]]) -> Any:
        params = dict(run.parameters or {})
        if not bool(params.get("one_h10_vault")):
            return signal
        forecast = params.get("one_h10_forecast") if isinstance(params.get("one_h10_forecast"), dict) else {}
        if not forecast:
            return signal
        metadata = dict(getattr(signal, "metadata", {}) or {})
        metadata["one_h10_forecast"] = forecast
        metadata["forecast_metadata"] = forecast
        blockers = list(forecast.get("blockers", []) or [])
        advisory_blockers = list(forecast.get("advisory_blockers", []) or [])
        metadata["forecast_blockers"] = blockers
        metadata["forecast_advisory_blockers"] = advisory_blockers
        if bool(self.config.get("ONE_H10_REQUIRE_PROMOTED_ML", True)) and not bool(forecast.get("ml_ready", False)):
            metadata["no_trade_reason"] = metadata.get("no_trade_reason") or "one_h10_promoted_ml_not_ready"
            metadata["ml_signal_not_ready"] = True
            return Signal("hold", "1H10 opening signal blocked until promoted 1h10 ML is ready.", run.timeframe, None, None, 0.0, metadata)
        blocking_reasons = {"features_stale", "missing_fibonacci_features", "ml_not_ready"}
        if bool(self.config.get("ONE_H10_REQUIRE_PROMOTED_ML", True)) and blocking_reasons.intersection(str(item) for item in blockers):
            metadata["no_trade_reason"] = metadata.get("no_trade_reason") or "one_h10_forecast_blocked:" + ",".join(sorted(blocking_reasons.intersection(str(item) for item in blockers)))
            return Signal("hold", "1H10 opening signal blocked by required ML feature gates.", run.timeframe, None, None, 0.0, metadata)
        action = str(forecast.get("predicted_side") or forecast.get("action") or "hold").lower()
        if action not in {"buy", "sell"}:
            metadata["no_trade_reason"] = metadata.get("no_trade_reason") or "forecast_hold"
            return Signal(signal.action, signal.rationale, signal.timeframe, signal.stop_loss, signal.take_profit, signal.position_fraction, metadata)
        if "low_confidence" in blockers and bool(self.config.get("ONE_H10_REQUIRE_PROMOTED_ML", True)):
            metadata["no_trade_reason"] = metadata.get("no_trade_reason") or "forecast_low_confidence"
            return Signal(signal.action, signal.rationale, signal.timeframe, signal.stop_loss, signal.take_profit, signal.position_fraction, metadata)
        mid = self._safe_float((candles[-1] if candles else {}).get("close"), 0.0)
        if mid <= 0:
            metadata["no_trade_reason"] = metadata.get("no_trade_reason") or "forecast_mid_unavailable"
            return Signal(signal.action, signal.rationale, signal.timeframe, signal.stop_loss, signal.take_profit, signal.position_fraction, metadata)
        stop_pct = self._safe_float(forecast.get("suggested_stop_loss_pct"), 0.0)
        take_pct = self._safe_float(forecast.get("suggested_take_profit_pct"), 0.0)
        if stop_pct <= 0 or take_pct <= 0:
            metadata["no_trade_reason"] = metadata.get("no_trade_reason") or "forecast_missing_exit"
            return Signal(signal.action, signal.rationale, signal.timeframe, signal.stop_loss, signal.take_profit, signal.position_fraction, metadata)
        if action == "buy":
            stop_loss = mid * (1 - stop_pct)
            take_profit = mid * (1 + take_pct)
        else:
            stop_loss = mid * (1 + stop_pct)
            take_profit = mid * (1 - take_pct)
        forecast_fraction = min(max(self._safe_float(forecast.get("position_fraction"), 0.0), 0.0), 1.0)
        base_fraction = self._safe_float(getattr(signal, "position_fraction", 0.0), 0.0)
        position_fraction = forecast_fraction if forecast_fraction > 0 else base_fraction
        if position_fraction <= 0:
            metadata["no_trade_reason"] = metadata.get("no_trade_reason") or "forecast_zero_sizing"
            return Signal(signal.action, signal.rationale, signal.timeframe, signal.stop_loss, signal.take_profit, signal.position_fraction, metadata)
        metadata["forecast_signal_applied"] = True
        if blockers:
            metadata["ml_signal_not_ready"] = "ml_not_ready" in blockers
        return Signal(
            action,
            f"1H10 forecast selected {action}.",
            run.timeframe,
            stop_loss,
            take_profit,
            min(position_fraction, 1.0),
            metadata,
        )

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
