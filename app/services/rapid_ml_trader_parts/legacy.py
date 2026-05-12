"""Rapid dual-venue ML trading orchestration with hard live safety gates."""

from __future__ import annotations

import math
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Callable

from flask import current_app, has_app_context

from ...extensions import db
from ...models import AuditLog, Fill, LeveragedMarket, Order, RapidMLDecision, RapidMLSession, Setting, TradingConnection, utcnow
from ...services.live_provider_adapters import KUCOIN_CONTRACT_SPECS, KUCOIN_SYMBOLS
from ...services.order_manager import OrderIntent, OrderManager
from ...services.provider_assets import normalize_provider, provider_collateral_asset, provider_feature_context


REQUIRED_ML_FAMILIES = (
    "pytorch_gru_signal",
    "pytorch_allocator",
    "pytorch_risk_policy",
    "pytorch_execution_policy",
    "pytorch_exit_policy",
    "pytorch_cap_policy",
    "pytorch_ops_anomaly",
    "pytorch_roi_target",
)

PROVIDER_SYMBOLS = {
    "hyperliquid": ("BTC", "ETH", "SOL", "HYPE"),
    "kucoin": ("BTC", "ETH", "SOL", "HYPE"),
}

CONFIRMATION_PHRASE = "RAPID-ML-LIVE"


class RapidMLTraderService:
    """Runs rapid ML preview/submit sessions without weakening hard gates."""

    def __init__(
        self,
        config: dict[str, Any],
        trading_connections: Any,
        market_data: Any,
        ml_decision_engine: Any,
        offline_ranker: Any,
        order_manager: OrderManager,
        leveraged_markets: Any | None = None,
    ) -> None:
        self.config = config
        self.trading_connections = trading_connections
        self.market_data = market_data
        self.ml_decision_engine = ml_decision_engine
        self.offline_ranker = offline_ranker
        self.order_manager = order_manager
        self.leveraged_markets = leveraged_markets

    def run(
        self,
        *,
        user_id: int,
        capital_usd: float,
        provider: str = "both",
        duration_minutes: float = 0.0,
        decision_interval_ms: int | None = None,
        max_order_rate_per_provider: float | None = None,
        submit: bool = False,
        confirm: str = "",
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        provider_scope = self._provider_scope(provider)
        capital = self._positive_float(capital_usd)
        if capital <= 0:
            raise ValueError("--capital-usd must be greater than zero.")

        interval_ms = self._effective_interval_ms(decision_interval_ms)
        max_rate = self._effective_order_rate(max_order_rate_per_provider)
        daily_loss_cap = self._daily_loss_cap(capital)
        per_position_cap = self._per_position_cap(capital)
        submit_blockers = self._submit_blockers(submit=submit, confirm=confirm)
        effective_submit = bool(submit and not submit_blockers)

        session = RapidMLSession(
            user_id=int(user_id),
            provider_scope=provider_scope,
            capital_usd=capital,
            submit_requested=bool(submit),
            real_submit_enabled=effective_submit,
            status="running",
        )
        session.summary = self._json_safe({
            "duration_minutes": max(0.0, float(duration_minutes or 0.0)),
            "decision_interval_ms": interval_ms,
            "max_order_rate_per_provider": max_rate,
            "required_ml_families": list(REQUIRED_ML_FAMILIES),
            "confirmation_phrase": CONFIRMATION_PHRASE,
        })
        db.session.add(session)
        db.session.commit()
        session_id_value = int(session.id)

        global_blockers: list[str] = list(submit_blockers)
        connections = self._connections_for_scope(int(user_id), provider_scope)
        if not connections:
            global_blockers.append("verified_tradable_connection_missing")

        cycle_payloads: list[dict[str, Any]] = []
        submitted_count = 0
        preview_count = 0
        rejected_decision_count = 0
        last_submit_at: dict[str, float] = {}
        snapshot_cache: dict[int, tuple[float, Any]] = {}
        cycles = self._cycle_count(duration_minutes, interval_ms, submit=effective_submit)
        started = time.monotonic()
        duration_seconds = max(0.0, self._safe_float(duration_minutes) * 60.0)
        deadline = started + duration_seconds if duration_seconds > 0 else None

        for cycle_index in range(cycles):
            if cycle_index > 0 and deadline is not None and time.monotonic() >= deadline:
                break
            if cycle_index > 0:
                elapsed_ms = (time.monotonic() - started) * 1000.0
                target_ms = cycle_index * interval_ms
                if elapsed_ms < target_ms:
                    sleep_seconds = (target_ms - elapsed_ms) / 1000.0
                    if deadline is not None:
                        sleep_seconds = min(sleep_seconds, max(0.0, deadline - time.monotonic()))
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    if deadline is not None and time.monotonic() >= deadline:
                        break

            analyses = self._run_provider_workers(
                [int(connection.id) for connection in connections],
                lambda connection_id: self._analyze_provider(
                    user_id=int(user_id),
                    connection_id=int(connection_id),
                    session_id=session_id_value,
                    capital_usd=capital,
                    submit=effective_submit,
                    snapshot_cache=snapshot_cache if not effective_submit else None,
                ),
            )
            position_managements = self._run_provider_workers(
                [int(item["connection_id"]) for item in analyses if item.get("connection_id")],
                lambda connection_id: self._manage_provider_positions(
                    user_id=int(user_id),
                    session_id=session_id_value,
                    analysis=self._analysis_for_connection(analyses, int(connection_id)),
                    submit=effective_submit,
                ),
            )
            position_management_by_connection = {
                int(item.get("connection_id") or 0): item
                for item in position_managements
                if isinstance(item, dict)
            }
            allocations, allocation_blockers = self._allocate_capital(
                analyses,
                capital_usd=capital,
                per_position_cap=per_position_cap,
            )
            global_blockers.extend(allocation_blockers)
            self._apply_correlation_guard(analyses, allocations, capital_usd=capital)

            executions = self._run_provider_workers(
                [int(item["connection_id"]) for item in analyses if item.get("connection_id")],
                lambda connection_id: self._execute_provider(
                    user_id=int(user_id),
                    session_id=session_id_value,
                    analysis=self._analysis_for_connection(analyses, int(connection_id)),
                    allocation_usd=float(allocations.get(str(connection_id), 0.0) or 0.0),
                    per_position_cap=per_position_cap,
                    daily_loss_cap=daily_loss_cap,
                    submit=bool(submit),
                    submit_blockers=submit_blockers,
                    max_order_rate_per_provider=max_rate,
                    last_submit_at=last_submit_at,
                    position_management=position_management_by_connection.get(int(connection_id), {}),
                ),
            )
            position_submitted = sum(int(self._safe_float(item.get("submitted_count"))) for item in position_managements)
            position_preview = sum(1 for item in position_managements if item.get("preview_ready"))
            cycle_submitted = sum(1 for item in executions if item.get("submitted")) + position_submitted
            cycle_preview = sum(1 for item in executions if item.get("preview_ready")) + position_preview
            cycle_rejected = sum(1 for item in executions if item.get("blockers"))
            submitted_count += cycle_submitted
            preview_count += cycle_preview
            rejected_decision_count += cycle_rejected
            cycle_payloads.append(
                {
                    "cycle": cycle_index + 1,
                    "analyses": analyses,
                    "position_management": position_managements,
                    "allocations": allocations,
                    "executions": executions,
                    "submitted_count": cycle_submitted,
                    "preview_ready_count": cycle_preview,
                    "rejected_decision_count": cycle_rejected,
                }
            )
            if progress_callback is not None:
                try:
                    progress_callback(
                        {
                            "cycle": cycle_index + 1,
                            "planned_cycles": cycles,
                            "elapsed_seconds": time.monotonic() - started,
                            "submitted_count": cycle_submitted,
                            "preview_ready_count": cycle_preview,
                            "rejected_decision_count": cycle_rejected,
                            "position_submitted_count": position_submitted,
                            "position_preview_count": position_preview,
                        }
                    )
                except Exception:  # noqa: BLE001
                    pass

            if effective_submit and cycle_submitted == 0 and cycle_index == 0 and any(item.get("blockers") for item in executions):
                break

        latest_provider_states = {
            int(analysis.get("connection_id") or 0): analysis.get("snapshot")
            for analysis in (cycle_payloads[-1].get("analyses", []) if cycle_payloads else [])
            if isinstance(analysis, dict) and isinstance(analysis.get("snapshot"), dict)
        }
        performance = self._performance_payload(
            int(user_id),
            connections,
            session.started_at,
            provider_states_by_connection=latest_provider_states if not effective_submit else None,
        )
        global_blockers = list(dict.fromkeys(str(item) for item in global_blockers if str(item)))
        session.allocated_capital = self._json_safe(cycle_payloads[-1]["allocations"] if cycle_payloads else {})
        session.blockers = global_blockers
        session.summary = self._json_safe({
            **session.summary,
            "cycles": cycle_payloads,
            "performance": performance,
            "submitted_count": submitted_count,
            "preview_ready_count": preview_count,
            "rejected_decision_count": rejected_decision_count,
            "planned_cycle_count": cycles,
            "elapsed_seconds": time.monotonic() - started,
        })
        session.status = self._session_status(
            submit_requested=bool(submit),
            effective_submit=effective_submit,
            submitted_count=submitted_count,
            preview_count=preview_count,
            blockers=global_blockers,
            cycle_payloads=cycle_payloads,
        )
        session.completed_at = utcnow()
        db.session.commit()

        payload = {
            "ready": session.status in {"preview_ready", "submitted"},
            "mode": "live",
            "session_id": session.id,
            "status": session.status,
            "provider_request": provider_scope,
            "submit_requested": bool(submit),
            "real_submit_enabled": effective_submit,
            "submitted_count": submitted_count,
            "real_order_submitted": submitted_count > 0,
            "preview_ready_count": preview_count,
            "rejected_decision_count": rejected_decision_count,
            "capital": {
                "capital_usd": capital,
                "daily_loss_cap_usd": daily_loss_cap,
                "daily_loss_cap_pct": self._daily_loss_pct(),
                "per_position_cap_usd": per_position_cap,
                "per_position_cap_pct": self._position_pct(),
                "ml_sizing_enabled": self._ml_sizing_enabled(),
                "rapid_ml_fixed_hard_cap_usd": self._rapid_fixed_hard_cap_usd(),
            },
            "planned_cycle_count": cycles,
            "elapsed_seconds": time.monotonic() - started,
            "safety": {
                "rapid_ml_live_enabled": bool(self.config.get("RAPID_ML_LIVE_ENABLED", False)),
                "rapid_ml_preview_only": bool(self.config.get("RAPID_ML_PREVIEW_ONLY", True)),
                "canary_preview_only": bool(self.config.get("CANARY_PREVIEW_ONLY", True)),
                "required_ml_families": list(REQUIRED_ML_FAMILIES),
                "auto_futures_universe_enabled": self._auto_futures_universe_enabled(),
                "max_symbols_per_provider": int(self._safe_float(self.config.get("RAPID_ML_MAX_SYMBOLS_PER_PROVIDER"), 48)),
                "max_order_rate_per_provider": max_rate,
                "decision_interval_ms": interval_ms,
                "profitability_gate": self._profitability_gate_config(),
            },
            "blockers": global_blockers,
            "cycles": cycle_payloads,
            "performance": performance,
            "next_commands": self._next_commands(int(user_id), provider_scope, capital),
            "notes": [
                "Preview mode never submits orders.",
                "Rapid mode only submits when promoted ML edge clears fees, slippage, spread, and risk thresholds.",
                "Hard risk gates, daily loss cap, provider health, symbol sizing, and stop-loss requirements are non-disableable.",
            ],
        }
        self._audit_session(session, payload)
        return payload

    def _analyze_provider(
        self,
        *,
        user_id: int,
        connection_id: int,
        session_id: int,
        capital_usd: float,
        submit: bool,
        snapshot_cache: dict[int, tuple[float, Any]] | None = None,
    ) -> dict[str, Any]:
        connection = db.session.get(TradingConnection, int(connection_id))
        if connection is None or int(connection.user_id) != int(user_id):
            return {"connection_id": connection_id, "provider": "", "ready": False, "blockers": ["connection_not_found"]}
        provider = normalize_provider(connection.provider)
        blockers: list[str] = []
        can_trade = self.trading_connections.can_trade(user_id, "live", connection.id)
        if not can_trade:
            blockers.append("provider_can_trade_false")
        snapshot = self._account_snapshot_for_analysis(
            user_id,
            connection,
            use_cache=not submit,
            snapshot_cache=snapshot_cache,
        )
        provider_state = self._provider_state(provider, snapshot)
        blockers.extend(provider_state.get("blockers", []))
        blockers.extend(self._circuit_breakers(provider, connection.id))
        readiness = self._ml_readiness(provider)
        blockers.extend(readiness.get("blockers", []))

        candidates: list[dict[str, Any]] = []
        if not readiness.get("ready", False):
            return {
                "connection_id": connection.id,
                "provider": provider,
                "ready": False,
                "can_trade": bool(can_trade),
                "snapshot": provider_state,
                "ml_readiness": readiness,
                "candidates": candidates,
                "selected": None,
                "blockers": list(dict.fromkeys(blockers)),
            }

        symbols = self._symbols_for_provider(provider, user_id=user_id, connection_id=int(connection.id))
        if not symbols:
            blockers.append("futures_universe_empty")
        require_active_futures = self._auto_futures_universe_enabled() and not self._configured_symbols_for_provider(provider)
        for symbol in symbols:
            symbol_payload = self._symbol_metadata(provider, symbol, require_active_futures=require_active_futures)
            if symbol_payload.get("blockers"):
                candidates.append(
                    {
                        "symbol": symbol,
                        "provider": provider,
                        "action": "hold",
                        "opportunity_score": 0.0,
                        "blockers": list(symbol_payload["blockers"]),
                    }
                )
                continue
            price = self._reference_price(provider, symbol, connection.id, venue_symbol=symbol_payload.get("venue_symbol"))
            if price <= 0:
                candidates.append(
                    {
                        "symbol": symbol,
                        "provider": provider,
                        "action": "hold",
                        "opportunity_score": 0.0,
                        "blockers": ["price_unavailable"],
                    }
                )
                continue
            spread_bps = self._spread_bps(provider, symbol, connection.id, venue_symbol=symbol_payload.get("venue_symbol"))
            base_context = {
                **provider_feature_context(provider),
                "symbol": symbol,
                "provider_symbol": symbol_payload.get("venue_symbol") or symbol,
                "venue_symbol": symbol_payload.get("venue_symbol") or symbol,
                "mode": "live",
                "horizon": "1h",
                "capital_usd": capital_usd,
                "available_balance_usd": provider_state.get("available_balance_usd", 0.0),
                "allocation_amount_usd": min(capital_usd, self._safe_float(provider_state.get("available_balance_usd", 0.0))),
                "notional": min(capital_usd, self._safe_float(provider_state.get("available_balance_usd", 0.0))),
                "ml_live_hard_cap_usdc": self._rapid_fixed_hard_cap_usd(),
                "reference_price": price,
                "mid_price": price,
                "spread_bps": spread_bps,
                "expected_fill_quality": max(0.0, min(1.0, 1.0 - spread_bps / 100.0)),
                "score": 0.0,
                "expected_return": 0.0,
                "rapid_ml_all_futures_universe": bool(symbol_payload.get("active_futures_market")),
                "futures_market_id": symbol_payload.get("market_id"),
                "futures_market_liquidity_usd": symbol_payload.get("liquidity_usd"),
                "futures_market_max_leverage": symbol_payload.get("max_leverage"),
            }
            feature_context = self._live_feature_context(
                provider,
                symbol,
                connection.id,
                venue_symbol=symbol_payload.get("venue_symbol"),
                reference_price=price,
                spread_bps=spread_bps,
            )
            context = {**base_context, **feature_context}
            context.update(
                {
                    "symbol": symbol,
                    "provider_symbol": symbol_payload.get("venue_symbol") or symbol,
                    "venue_symbol": symbol_payload.get("venue_symbol") or symbol,
                    "provider": provider,
                    "execution_venue": provider,
                    "mode": "live",
                    "horizon": "1h",
                    "capital_usd": capital_usd,
                    "available_balance_usd": provider_state.get("available_balance_usd", 0.0),
                    "allocation_amount_usd": min(capital_usd, self._safe_float(provider_state.get("available_balance_usd", 0.0))),
                    "notional": min(capital_usd, self._safe_float(provider_state.get("available_balance_usd", 0.0))),
                    "ml_live_hard_cap_usdc": self._rapid_fixed_hard_cap_usd(),
                    "reference_price": price,
                    "mid_price": price,
                    "spread_bps": spread_bps,
                    "expected_fill_quality": max(0.0, min(1.0, 1.0 - spread_bps / 100.0)),
                    "rapid_ml_all_futures_universe": bool(symbol_payload.get("active_futures_market")),
                    "futures_market_id": symbol_payload.get("market_id"),
                    "futures_market_liquidity_usd": symbol_payload.get("liquidity_usd"),
                    "futures_market_max_leverage": symbol_payload.get("max_leverage"),
                }
            )
            context.setdefault("score", 0.0)
            context.setdefault("expected_return", 0.0)
            candidates.append(self._score_candidate(provider, context, readiness))

        selected = self._select_candidate(candidates)
        if selected is None:
            blockers.append("no_ml_actionable_candidate")
        return {
            "connection_id": connection.id,
            "provider": provider,
            "ready": bool(can_trade) and not provider_state["blockers"] and selected is not None,
            "can_trade": bool(can_trade),
            "snapshot": provider_state,
            "ml_readiness": readiness,
            "candidates": candidates,
            "selected": selected,
            "blockers": list(dict.fromkeys(blockers)),
        }

    def _execute_provider(
        self,
        *,
        user_id: int,
        session_id: int,
        analysis: dict[str, Any],
        allocation_usd: float,
        per_position_cap: float,
        daily_loss_cap: float,
        submit: bool,
        submit_blockers: list[str],
        max_order_rate_per_provider: float,
        last_submit_at: dict[str, float],
        position_management: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not analysis:
            return {"ready": False, "blockers": ["analysis_missing"]}
        provider = str(analysis.get("provider") or "")
        connection_id = int(analysis.get("connection_id") or 0)
        selected = analysis.get("selected") if isinstance(analysis.get("selected"), dict) else None
        blockers = list(analysis.get("blockers", []) or [])
        position_management = position_management if isinstance(position_management, dict) else {}
        if selected is None:
            blockers.append("no_selected_ml_candidate")
        if allocation_usd <= 0:
            blockers.append("allocation_zero")
        if submit_blockers:
            blockers.extend(submit_blockers)

        if submit and connection_id:
            snapshot = self.trading_connections.account_snapshot(user_id, "live", connection_id)
            provider_state = self._provider_state(provider, snapshot)
        else:
            provider_state = analysis.get("snapshot") if isinstance(analysis.get("snapshot"), dict) else {}
        blockers.extend(provider_state["blockers"])
        if provider_state.get("open_orders_count", 0) > 0:
            blockers.append("open_orders_block_rapid_submit")
        if provider_state.get("positions_count", 0) > 0:
            blockers.append("open_positions_block_rapid_opening")
        if int(self._safe_float(position_management.get("submitted_count"))) > 0:
            blockers.append("rapid_ml_position_close_submitted_wait_for_snapshot")
        elif bool(position_management.get("preview_ready")):
            blockers.append("rapid_ml_position_close_preview")

        intent_payload: dict[str, Any] = {}
        order_id: int | None = None
        submitted = False
        status = "blocked"
        order_payload: dict[str, Any] = {}
        if selected is not None and not blockers:
            intent, intent_blockers = self._build_intent(
                user_id=user_id,
                session_id=session_id,
                connection_id=connection_id,
                provider=provider,
                selected=selected,
                allocation_usd=min(allocation_usd, per_position_cap),
                daily_loss_cap=daily_loss_cap,
                provider_state=provider_state,
            )
            blockers.extend(intent_blockers)
            if intent is not None:
                intent_payload = self._intent_payload(intent)
                status = "preview"
                if submit and not blockers:
                    throttle_blocker = self._throttle_blocker(provider, max_order_rate_per_provider, last_submit_at)
                    if throttle_blocker:
                        blockers.append(throttle_blocker)
                    else:
                        order = self.order_manager.place_order(intent)
                        order_id = int(order.id)
                        submitted = str(order.status or "").lower() not in {"rejected", "failed"}
                        status = "submitted" if submitted else "rejected"
                        last_submit_at[provider] = time.monotonic()
                        order_payload = self._order_payload(order)
        decision = RapidMLDecision(
            session_id=session_id,
            user_id=user_id,
            trading_connection_id=connection_id or None,
            order_id=order_id,
            provider=provider,
            symbol=str((selected or {}).get("symbol") or ""),
            side=str((selected or {}).get("side") or ""),
            action=str((selected or {}).get("action") or "hold"),
            status=status if not blockers else ("submit_blocked" if submit else "blocked"),
            confidence=self._safe_float((selected or {}).get("confidence")),
            expected_return=self._safe_float((selected or {}).get("expected_return_after_costs")),
            opportunity_score=self._safe_float((selected or {}).get("opportunity_score")),
            allocation_usd=max(0.0, allocation_usd),
            notional_usd=self._safe_float(intent_payload.get("notional_usd")),
        )
        decision.order_intent = self._json_safe(intent_payload)
        decision.provider_state = self._json_safe(provider_state)
        decision.ml_decisions = self._json_safe(dict((selected or {}).get("ml_decisions") or {}))
        risk_payload = {
            "daily_loss_cap_usd": daily_loss_cap,
            "per_position_cap_usd": per_position_cap,
            "provider_blockers": analysis.get("blockers", []),
            "order": order_payload,
        }
        decision.risk = self._json_safe(risk_payload)
        decision.blockers = list(dict.fromkeys(str(item) for item in blockers if str(item)))
        db.session.add(decision)
        db.session.commit()

        return {
            "provider": provider,
            "connection_id": connection_id,
            "decision_id": decision.id,
            "ready": bool(not blockers and selected is not None),
            "preview_ready": bool(not submit and not blockers and selected is not None),
            "submitted": submitted,
            "status": decision.status,
            "selected": selected,
            "allocation_usd": max(0.0, allocation_usd),
            "order_intent": intent_payload,
            "order": order_payload,
            "position_management": position_management,
            "blockers": decision.blockers,
        }

    def _manage_provider_positions(
        self,
        *,
        user_id: int,
        session_id: int,
        analysis: dict[str, Any],
        submit: bool,
    ) -> dict[str, Any]:
        provider = str(analysis.get("provider") or "")
        connection_id = int(analysis.get("connection_id") or 0)
        result: dict[str, Any] = {
            "provider": provider,
            "connection_id": connection_id,
            "enabled": self._rapid_auto_close_enabled(),
            "status": "disabled" if not self._rapid_auto_close_enabled() else "flat",
            "positions": [],
            "submitted": False,
            "submitted_count": 0,
            "preview_ready": False,
            "blockers": [],
        }
        if not result["enabled"] or not analysis:
            return result

        provider_state = analysis.get("snapshot") if isinstance(analysis.get("snapshot"), dict) else {}
        positions = [item for item in list(provider_state.get("positions", []) or []) if isinstance(item, dict)]
        snapshot_unreliable = self._provider_snapshot_unreliable(provider_state)
        if not positions:
            if snapshot_unreliable:
                positions = self._local_rapid_ml_fallback_positions(
                    user_id=user_id,
                    connection_id=connection_id,
                    provider=provider,
                )
                if not positions:
                    result["status"] = "snapshot_unavailable"
                    result["blockers"] = list(provider_state.get("blockers", []) or [])[:8]
                    return result
                result["status"] = "snapshot_unavailable_local_fallback"
                for blocker in ["provider_snapshot_unavailable_rapid_ml_local_fallback", *list(provider_state.get("blockers", []) or [])]:
                    if blocker and blocker not in result["blockers"]:
                        result["blockers"].append(str(blocker))
            else:
                return result

        if result["status"] == "flat":
            result["status"] = "monitoring"
        for position in positions:
            row = self._manage_single_position(
                user_id=user_id,
                session_id=session_id,
                connection_id=connection_id,
                provider=provider,
                analysis=analysis,
                position=position,
                submit=submit,
            )
            result["positions"].append(row)
            if row.get("submitted"):
                result["submitted"] = True
                result["submitted_count"] = int(result["submitted_count"]) + 1
            if row.get("would_close") and not submit:
                result["preview_ready"] = True
            for blocker in row.get("blockers", []) or []:
                if blocker not in result["blockers"]:
                    result["blockers"].append(blocker)

        if result["submitted"]:
            result["status"] = "submitted"
        elif result["preview_ready"]:
            result["status"] = "would_close"
        elif result["blockers"]:
            result["status"] = "blocked"
        return result

    def _manage_single_position(
        self,
        *,
        user_id: int,
        session_id: int,
        connection_id: int,
        provider: str,
        analysis: dict[str, Any],
        position: dict[str, Any],
        submit: bool,
    ) -> dict[str, Any]:
        symbol = str(position.get("symbol") or "").upper()
        quantity = self._position_quantity(position)
        row: dict[str, Any] = {
            "symbol": symbol,
            "quantity": quantity,
            "side": "buy" if quantity > 0 else "sell" if quantity < 0 else "",
            "mark_price": 0.0,
            "source_order": None,
            "trigger": "",
            "would_close": False,
            "submitted": False,
            "close_order": {},
            "blockers": [],
        }
        if not symbol or abs(quantity) < 1e-9:
            row["blockers"].append("position_flat_or_symbol_missing")
            return row

        source_order = self._rapid_ml_source_order(
            user_id=user_id,
            trading_connection_id=connection_id,
            symbol=symbol,
        )
        if source_order is None:
            blocker = "position_not_rapid_ml_owned"
            if self._rapid_manage_manual_positions():
                blocker = "rapid_ml_source_order_missing"
            row["blockers"].append(blocker)
            return row

        row["source_order"] = self._source_order_payload(source_order)
        venue_symbol = str((source_order.details or {}).get("venue_symbol") or symbol)
        mark = self._position_mark_price(provider, symbol, connection_id, position, venue_symbol=venue_symbol)
        row["mark_price"] = mark
        if mark <= 0:
            row["blockers"].append("position_mark_price_unavailable")
            return row

        trigger, trigger_details = self._position_exit_trigger(
            position=position,
            source_order=source_order,
            analysis=analysis,
            mark=mark,
        )
        row.update(trigger_details)
        if not trigger:
            return row

        row["trigger"] = trigger
        row["would_close"] = True
        if not submit:
            return row

        intent = self._position_close_intent(
            source_order=source_order,
            position=position,
            provider=provider,
            connection_id=connection_id,
            mark=mark,
            trigger=trigger,
            session_id=session_id,
        )
        order = self.order_manager.place_order(intent)
        row["close_order"] = self._order_payload(order)
        row["submitted"] = str(order.status or "").lower() not in {"rejected", "failed"}
        if not row["submitted"]:
            row["blockers"].append("rapid_ml_position_close_rejected")
        return row

    def _position_exit_trigger(
        self,
        *,
        position: dict[str, Any],
        source_order: Order,
        analysis: dict[str, Any],
        mark: float,
    ) -> tuple[str, dict[str, Any]]:
        details: dict[str, Any] = {
            "current_candidate": None,
            "selected_candidate": self._rotation_candidate_payload(
                analysis.get("selected") if isinstance(analysis.get("selected"), dict) else None
            ),
            "rotation_score_delta": 0.0,
        }
        quantity = self._position_quantity(position)
        side = "buy" if quantity > 0 else "sell" if quantity < 0 else str(source_order.side or "").lower()
        if side == "buy":
            if self._safe_float(source_order.stop_loss) > 0 and mark <= self._safe_float(source_order.stop_loss):
                return "stop_loss", details
            if self._safe_float(source_order.take_profit) > 0 and mark >= self._safe_float(source_order.take_profit):
                return "take_profit", details
        if side == "sell":
            if self._safe_float(source_order.stop_loss) > 0 and mark >= self._safe_float(source_order.stop_loss):
                return "stop_loss", details
            if self._safe_float(source_order.take_profit) > 0 and mark <= self._safe_float(source_order.take_profit):
                return "take_profit", details
        if bool(position.get("rapid_ml_local_fallback")):
            details["blockers"] = ["provider_snapshot_unavailable"]
            return "provider_snapshot_unavailable_rapid_ml_close", details

        current = self._candidate_for_symbol(analysis, source_order.symbol)
        details["current_candidate"] = self._rotation_candidate_payload(current)
        if current is None:
            details["blockers"] = ["current_symbol_not_scored"]
            return "", details
        current_blockers = [str(item) for item in current.get("blockers", []) or [] if str(item)]
        if current_blockers:
            details["blockers"] = current_blockers[:8]
            return "ml_current_candidate_blocked", details
        current_side = str(current.get("side") or current.get("action") or "").lower()
        if current_side not in {"buy", "sell"}:
            return "ml_current_candidate_hold", details
        if current_side != side:
            return "ml_current_candidate_flipped", details

        selected = analysis.get("selected") if isinstance(analysis.get("selected"), dict) else None
        if not self._rapid_rotate_enabled() or not selected:
            return "", details
        selected_symbol = str(selected.get("symbol") or "").upper()
        if not selected_symbol or selected_symbol == str(source_order.symbol or "").upper():
            return "", details
        if selected.get("blockers"):
            return "", details
        score_delta = self._safe_float(selected.get("opportunity_score")) - self._safe_float(current.get("opportunity_score"))
        edge_current = self._safe_float(current.get("expected_edge_bps_after_costs"))
        edge_selected = self._safe_float(selected.get("expected_edge_bps_after_costs"))
        details["rotation_score_delta"] = score_delta
        if score_delta >= self._rapid_rotate_min_score_delta() and edge_selected + 1e-9 >= edge_current:
            return "ml_rotate_stronger_candidate", details
        return "", details

    def _position_close_intent(
        self,
        *,
        source_order: Order,
        position: dict[str, Any],
        provider: str,
        connection_id: int,
        mark: float,
        trigger: str,
        session_id: int,
    ) -> OrderIntent:
        quantity = abs(self._position_quantity(position))
        side = "sell" if self._position_quantity(position) > 0 else "buy"
        source_metadata = dict(source_order.details or {})
        metadata = {
            "rapid_ml": True,
            "rapid_ml_exit": True,
            "rapid_ml_exit_trigger": trigger,
            "rapid_ml_exit_source_order_id": source_order.id,
            "rapid_ml_source_session_id": source_metadata.get("rapid_ml_session_id"),
            "rapid_ml_session_id": session_id,
            "provider": provider,
            "execution_venue": provider,
            "collateral_asset": provider_collateral_asset(provider),
            "venue_symbol": source_metadata.get("venue_symbol") or source_order.symbol,
            "reference_price": mark,
            "source_order_side": source_order.side,
            "source_order_quantity": source_order.quantity,
            "source_order_stop_loss": source_order.stop_loss,
            "source_order_take_profit": source_order.take_profit,
            "source_order_reference_price": source_metadata.get("reference_price"),
            "provider_snapshot_fallback": bool(position.get("rapid_ml_local_fallback")),
            "ml_horizon": source_metadata.get("ml_horizon", "1h"),
            "strategy_name": "rapid_ml_trader",
            "reduce_only_exit": True,
        }
        return OrderIntent(
            symbol=source_order.symbol,
            side=side,
            quantity=quantity,
            mode="live",
            order_type="market",
            reduce_only=True,
            leverage=1.0,
            stop_loss=source_order.stop_loss,
            take_profit=source_order.take_profit,
            strategy_name="rapid_ml_trader",
            timeframe="1h",
            slippage_pct=0.0,
            user_id=source_order.user_id,
            trading_connection_id=connection_id,
            idempotency_key=f"rapid-ml-close-{source_order.id}-{trigger}-{uuid.uuid4().hex}",
            metadata=metadata,
        )

    def _rapid_ml_source_order(
        self,
        *,
        user_id: int,
        trading_connection_id: int,
        symbol: str,
    ) -> Order | None:
        query = (
            Order.query.filter_by(
                user_id=user_id,
                trading_connection_id=trading_connection_id,
                symbol=str(symbol or "").upper(),
                mode="live",
                reduce_only=False,
            )
            .filter(Order.status.in_(["filled", "open", "submitted"]))
            .order_by(Order.created_at.desc(), Order.id.desc())
        )
        for order in query.limit(25).all():
            details = order.details or {}
            if bool(details.get("rapid_ml")):
                return order
            if self._rapid_manage_manual_positions():
                return order
        return None

    def _local_rapid_ml_fallback_positions(
        self,
        *,
        user_id: int,
        connection_id: int,
        provider: str,
    ) -> list[dict[str, Any]]:
        if not user_id or not connection_id:
            return []
        query = (
            Order.query.filter_by(
                user_id=int(user_id),
                trading_connection_id=int(connection_id),
                mode="live",
                reduce_only=False,
            )
            .filter(Order.status.in_(["filled", "open", "submitted"]))
            .order_by(Order.created_at.desc(), Order.id.desc())
        )
        positions: list[dict[str, Any]] = []
        seen_symbols: set[str] = set()
        for order in query.limit(50).all():
            symbol = str(order.symbol or "").upper()
            if not symbol or symbol in seen_symbols:
                continue
            if not self._rapid_manage_manual_positions() and not bool((order.details or {}).get("rapid_ml")):
                continue
            if self._rapid_ml_source_has_successful_exit(order):
                continue
            quantity = self._safe_float(order.filled_quantity) or self._safe_float(order.quantity)
            if quantity <= 0:
                continue
            signed_quantity = quantity if str(order.side or "").lower() == "buy" else -quantity
            reference_price = self._source_order_reference_price(order)
            positions.append(
                {
                    "symbol": symbol,
                    "quantity": signed_quantity,
                    "entry_price": reference_price,
                    "mark_price": reference_price,
                    "unrealized_pnl": 0.0,
                    "leverage": self._safe_float(order.leverage, 1.0),
                    "provider": provider,
                    "rapid_ml_local_fallback": True,
                    "source_order_id": order.id,
                }
            )
            seen_symbols.add(symbol)
        return positions

    @staticmethod
    def _provider_snapshot_unreliable(provider_state: dict[str, Any]) -> bool:
        blockers = {str(item) for item in provider_state.get("blockers", []) or [] if str(item)}
        alerts = [str(item) for item in provider_state.get("alerts", []) or [] if str(item)]
        if alerts or "provider_snapshot_alerts" in blockers:
            return True
        if any(item.endswith("_balance_unavailable") for item in blockers) and not provider_state.get("balances"):
            return True
        return False

    @staticmethod
    def _rapid_ml_source_has_successful_exit(source_order: Order) -> bool:
        if source_order.id is None:
            return False
        exits = (
            Order.query.filter_by(
                user_id=source_order.user_id,
                trading_connection_id=source_order.trading_connection_id,
                symbol=source_order.symbol,
                mode="live",
                reduce_only=True,
            )
            .filter(Order.status.in_(["filled", "open", "submitted", "pending"]))
            .order_by(Order.created_at.desc(), Order.id.desc())
            .limit(25)
            .all()
        )
        for order in exits:
            details = order.details or {}
            if int(RapidMLTraderService._safe_float(details.get("rapid_ml_exit_source_order_id"))) == int(source_order.id):
                return True
        return False

    @staticmethod
    def _source_order_reference_price(order: Order) -> float:
        details = order.details or {}
        for value in (
            order.average_fill_price,
            details.get("fill_price"),
            details.get("reference_price"),
            details.get("submitted_price"),
            details.get("source_order_reference_price"),
        ):
            price = RapidMLTraderService._safe_float(value)
            if price > 0:
                return price
        exchange_response = details.get("exchange_response") if isinstance(details.get("exchange_response"), dict) else {}
        for value in (
            exchange_response.get("fill_price"),
            exchange_response.get("submitted_price"),
        ):
            price = RapidMLTraderService._safe_float(value)
            if price > 0:
                return price
        return 0.0

    def _position_mark_price(
        self,
        provider: str,
        symbol: str,
        connection_id: int,
        position: dict[str, Any],
        *,
        venue_symbol: str | None = None,
    ) -> float:
        for key in ("mark_price", "markPrice", "mid_price", "oracle_price", "oraclePrice"):
            value = self._safe_float(position.get(key))
            if value > 0:
                return value
        mark_value = self._safe_float(position.get("mark_value"))
        quantity = abs(self._position_quantity(position))
        if mark_value > 0 and quantity > 0:
            return mark_value / quantity
        return self._reference_price(provider, symbol, connection_id, venue_symbol=venue_symbol)

    @staticmethod
    def _position_quantity(position: dict[str, Any]) -> float:
        return RapidMLTraderService._safe_float(
            position.get("quantity", position.get("size", position.get("szi")))
        )

    @staticmethod
    def _candidate_for_symbol(analysis: dict[str, Any], symbol: str) -> dict[str, Any] | None:
        target = str(symbol or "").upper()
        for candidate in list(analysis.get("candidates", []) or []):
            if isinstance(candidate, dict) and str(candidate.get("symbol") or "").upper() == target:
                return candidate
        return None

    @staticmethod
    def _source_order_payload(order: Order) -> dict[str, Any]:
        return {
            "id": order.id,
            "status": order.status,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "stop_loss": order.stop_loss,
            "take_profit": order.take_profit,
            "created_at": order.created_at,
        }

    @staticmethod
    def _rotation_candidate_payload(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(candidate, dict):
            return None
        return {
            "symbol": candidate.get("symbol"),
            "action": candidate.get("action"),
            "side": candidate.get("side"),
            "confidence": candidate.get("confidence"),
            "opportunity_score": candidate.get("opportunity_score"),
            "expected_edge_bps_after_costs": candidate.get("expected_edge_bps_after_costs"),
            "blockers": list(candidate.get("blockers", []) or [])[:8],
        }

    def _score_candidate(self, provider: str, context: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
        ml_decisions: dict[str, Any] = {}
        blockers: list[str] = [str(item) for item in context.get("rapid_feature_blockers", []) or [] if str(item)]
        feature_candles = list(context.get("rapid_feature_candles") or [])
        for family in REQUIRED_ML_FAMILIES:
            try:
                decision = dict(
                    self.ml_decision_engine.decision(
                        family,
                        context,
                        horizon="1h",
                        candles=feature_candles if family == "pytorch_gru_signal" else None,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                decision = {"family": family, "action": "hold", "blockers": [str(exc)], "ready": False}
            ml_decisions[family] = decision
            blockers.extend(f"{family}:{item}" for item in decision.get("blockers", []) or [])
        try:
            offline = dict(self.offline_ranker.score_payload(context, "1h", base_score=0.0, rejected=False))
        except Exception as exc:  # noqa: BLE001
            offline = {"status": "error", "prediction": 0.0, "blockers": [str(exc)]}
        ml_decisions["offline_ranker"] = offline
        blockers.extend(f"offline_ranker:{item}" for item in offline.get("blockers", []) or [])

        signal = ml_decisions.get("pytorch_gru_signal", {})
        risk = ml_decisions.get("pytorch_risk_policy", {})
        allocator = ml_decisions.get("pytorch_allocator", {})
        execution = ml_decisions.get("pytorch_execution_policy", {})
        ops = ml_decisions.get("pytorch_ops_anomaly", {})
        roi = ml_decisions.get("pytorch_roi_target", {})
        action = str(signal.get("action") or "hold").lower()
        if action not in {"buy", "sell"}:
            blockers.append("ml_signal_not_actionable")
        if str(risk.get("action") or "").lower() != "approve":
            blockers.append("ml_risk_policy_not_approved")
        if str(allocator.get("action") or "").lower() not in {"allocate", "hold"}:
            blockers.append("ml_allocator_invalid_action")
        if str(execution.get("action") or "").lower() not in {"route", "hold"}:
            blockers.append("ml_execution_policy_invalid_action")
        ops_raw = ops.get("raw") if isinstance(ops.get("raw"), dict) else {}
        ops_score = self._safe_float(ops_raw.get("ops_anomaly_score"), self._safe_float(ops.get("confidence")))
        if str(ops.get("action") or "").lower() == "warn" or ops_score >= 0.70:
            blockers.append("ml_ops_anomaly_block")
        confidence = max(
            self._safe_float(signal.get("confidence")),
            self._safe_float(risk.get("confidence")),
            self._safe_float(allocator.get("confidence")),
        )
        spread_bps = self._safe_float(context.get("spread_bps"))
        profitability = self._profitability_model(
            provider,
            context,
            confidence=confidence,
            signal=signal,
            roi=roi,
            offline=offline,
        )
        blockers.extend(profitability["blockers"])
        gross_expected = self._safe_float(profitability.get("gross_expected_return"))
        edge_bps = self._safe_float(profitability.get("edge_bps_after_costs"))
        sizing_score = 0.0
        allocator_raw = allocator.get("raw") if isinstance(allocator.get("raw"), dict) else {}
        sizing_score = self._safe_float(allocator_raw.get("sizing_score"), self._safe_float(allocator.get("confidence")))
        offline_score = self._safe_float(offline.get("prediction"))
        opportunity = max(0.0, edge_bps / 100.0 + confidence + sizing_score + offline_score - ops_score)
        if blockers:
            opportunity = 0.0
        return {
            "provider": provider,
            "symbol": str(context.get("symbol") or "").upper(),
            "venue_symbol": context.get("venue_symbol") or context.get("symbol"),
            "action": action if action in {"buy", "sell"} else "hold",
            "side": action if action in {"buy", "sell"} else "",
            "reference_price": self._safe_float(context.get("reference_price")),
            "spread_bps": spread_bps,
            "confidence": confidence,
            "expected_return": gross_expected,
            "expected_return_after_costs": edge_bps / 10_000.0,
            "expected_edge_bps_after_costs": edge_bps,
            "profitability": profitability,
            "opportunity_score": opportunity,
            "rapid_ml_all_futures_universe": bool(context.get("rapid_ml_all_futures_universe")),
            "futures_market_id": context.get("futures_market_id"),
            "futures_market_liquidity_usd": self._safe_float(context.get("futures_market_liquidity_usd")),
            "futures_market_max_leverage": self._safe_float(context.get("futures_market_max_leverage")),
            "ml_readiness": readiness,
            "ml_decisions": self._compact_ml_decisions(ml_decisions),
            "blockers": list(dict.fromkeys(str(item) for item in blockers if str(item))),
        }

    def _build_intent(
        self,
        *,
        user_id: int,
        session_id: int,
        connection_id: int,
        provider: str,
        selected: dict[str, Any],
        allocation_usd: float,
        daily_loss_cap: float,
        provider_state: dict[str, Any],
    ) -> tuple[OrderIntent | None, list[str]]:
        blockers: list[str] = []
        symbol = str(selected.get("symbol") or "").upper()
        side = str(selected.get("side") or selected.get("action") or "").lower()
        price = self._safe_float(selected.get("reference_price"))
        if not symbol:
            blockers.append("symbol_missing")
        if side not in {"buy", "sell"}:
            blockers.append("side_missing")
        if price <= 0:
            blockers.append("price_unavailable")
        if allocation_usd <= 0:
            blockers.append("allocation_zero")
        if blockers:
            return None, blockers

        quantity, sizing = self._quantity_for(provider, symbol, allocation_usd, price, selected.get("venue_symbol"))
        blockers.extend(sizing.get("blockers", []))
        notional = quantity * price
        if notional <= 0:
            blockers.append("notional_zero")
        if notional > allocation_usd + 1e-9:
            blockers.append("notional_exceeds_allocation")
        if blockers:
            return None, blockers

        stop_loss, take_profit = self._stop_take_prices(selected, side, price)
        execution_style = self._execution_style(selected)
        order_type = "limit" if execution_style in {"maker_limit", "limit", "post_only_limit"} else "market"
        limit_price = None
        if order_type == "limit":
            limit_price = price * (0.999 if side == "buy" else 1.001)
        metadata = {
            "rapid_ml": True,
            "rapid_ml_session_id": session_id,
            "provider": provider,
            "execution_venue": provider,
            "collateral_asset": provider_collateral_asset(provider),
            "venue_symbol": selected.get("venue_symbol") or symbol,
            "allocation_cap_usd": allocation_usd,
            "available_margin_usd": provider_state.get("available_balance_usd", 0.0),
            "account_equity_usd": provider_state.get("equity_usd", 0.0),
            "daily_loss_cap_usd": daily_loss_cap,
            "reference_price": price,
            "expected_return_after_costs": selected.get("expected_return_after_costs"),
            "expected_edge_bps_after_costs": selected.get("expected_edge_bps_after_costs"),
            "profitability": selected.get("profitability", {}),
            "edge_score": selected.get("opportunity_score"),
            "ml_horizon": "1h",
            "ml_policy_required": True,
            "ml_governed_risk": True,
            "ml_policy_decisions": selected.get("ml_decisions", {}),
            "execution_style": execution_style,
            "ml_sizing": selected.get("ml_sizing", {}),
            "rapid_ml_all_futures_universe": bool(selected.get("rapid_ml_all_futures_universe")),
            "futures_market_id": selected.get("futures_market_id"),
            "futures_market_liquidity_usd": selected.get("futures_market_liquidity_usd"),
            "futures_market_max_leverage": selected.get("futures_market_max_leverage"),
            "ml_live_hard_cap_usdc": self._rapid_fixed_hard_cap_usd(),
            "provider_sizing": sizing,
            "stop_loss_required": True,
            "take_profit_required": True,
        }
        return (
            OrderIntent(
                symbol=symbol,
                side=side,
                quantity=quantity,
                mode="live",
                order_type=order_type,
                limit_price=limit_price,
                reduce_only=False,
                leverage=1.0,
                stop_loss=stop_loss,
                take_profit=take_profit,
                strategy_name="rapid_ml_trader",
                timeframe="1h",
                slippage_pct=0.0,
                user_id=user_id,
                trading_connection_id=connection_id,
                idempotency_key=f"rapid-ml-{uuid.uuid4().hex}",
                metadata=metadata,
            ),
            [],
        )

    def _ml_readiness(self, provider: str) -> dict[str, Any]:
        families: dict[str, Any] = {}
        blockers: list[str] = []
        for family in REQUIRED_ML_FAMILIES:
            try:
                readiness = dict(self.ml_decision_engine.family_readiness(family, "1h", provider=provider))
            except Exception as exc:  # noqa: BLE001
                readiness = {"ready": False, "blockers": [str(exc)], "provider": provider, "family": family}
            families[family] = self._compact_readiness_payload(readiness)
            blockers.extend(f"{family}:{item}" for item in readiness.get("blockers", []) or [])
        try:
            offline = dict(self.offline_ranker.readiness("1h", require_blend=False, provider=provider))
        except Exception as exc:  # noqa: BLE001
            offline = {"ready": False, "blockers": [str(exc)], "provider": provider}
        blockers.extend(f"offline_ranker:{item}" for item in offline.get("blockers", []) or [])
        return {
            "ready": not blockers,
            "provider": provider,
            "horizon": "1h",
            "required_families": list(REQUIRED_ML_FAMILIES),
            "families": families,
            "offline_ranker": self._compact_readiness_payload(offline),
            "blockers": list(dict.fromkeys(str(item) for item in blockers if str(item))),
        }

    def _provider_state(self, provider: str, snapshot: Any) -> dict[str, Any]:
        blockers: list[str] = []
        balances = list(getattr(snapshot, "balances", []) or []) if snapshot is not None else []
        positions = list(getattr(snapshot, "positions", []) or []) if snapshot is not None else []
        open_orders = list(getattr(snapshot, "open_orders", []) or []) if snapshot is not None else []
        fills = list(getattr(snapshot, "recent_fills", []) or []) if snapshot is not None else []
        alerts = [str(item) for item in (getattr(snapshot, "alerts", []) or []) if str(item)]
        if alerts:
            blockers.append("provider_snapshot_alerts")
        collateral = provider_collateral_asset(provider)
        available = self._balance_amount(balances, collateral, prefer_withdrawable=True)
        equity = self._balance_amount(balances, collateral, prefer_withdrawable=False)
        if available <= 0 and equity > 0:
            available = equity
        if available <= 0:
            blockers.append(f"{collateral.lower()}_balance_unavailable")
        return {
            "provider": provider,
            "collateral_asset": collateral,
            "balances": balances,
            "available_balance_usd": available,
            "equity_usd": equity,
            "positions": positions,
            "open_orders": open_orders,
            "positions_count": len(positions),
            "open_orders_count": len(open_orders),
            "recent_fills_count": len(fills),
            "alerts": alerts,
            "blockers": blockers,
        }

    def _account_snapshot_for_analysis(
        self,
        user_id: int,
        connection: TradingConnection,
        *,
        use_cache: bool,
        snapshot_cache: dict[int, tuple[float, Any]] | None,
    ) -> Any:
        if not use_cache or snapshot_cache is None:
            return self.trading_connections.account_snapshot(user_id, "live", connection.id)
        cache_key = int(connection.id)
        now = time.monotonic()
        ttl_seconds = self._preview_snapshot_ttl_seconds()
        cached = snapshot_cache.get(cache_key)
        if cached and now - cached[0] <= ttl_seconds:
            return cached[1]
        snapshot = self.trading_connections.account_snapshot(user_id, "live", connection.id)
        if self._snapshot_cacheable(snapshot):
            snapshot_cache[cache_key] = (now, snapshot)
        elif cached:
            return cached[1]
        return snapshot

    def _preview_snapshot_ttl_seconds(self) -> float:
        return max(15.0, min(300.0, self._safe_float(self.config.get("RAPID_ML_PREVIEW_SNAPSHOT_TTL_SECONDS"), 120.0)))

    @staticmethod
    def _snapshot_cacheable(snapshot: Any) -> bool:
        if snapshot is None:
            return False
        if getattr(snapshot, "alerts", None):
            return False
        return bool(getattr(snapshot, "balances", None))

    def _circuit_breakers(self, provider: str, connection_id: int) -> list[str]:
        blockers: list[str] = []
        threshold = max(1, int(self._safe_float(self.config.get("RAPID_ML_MAX_PROVIDER_FAILURES"), 3)))
        window_minutes = max(1.0, self._safe_float(self.config.get("RAPID_ML_CIRCUIT_BREAKER_WINDOW_MINUTES"), 5.0))
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_orders = (
            Order.query.filter_by(trading_connection_id=int(connection_id), mode="live")
            .filter(Order.created_at >= cutoff)
            .order_by(Order.created_at.desc(), Order.id.desc())
            .limit(20)
            .all()
        )
        rejected = [
            order
            for order in recent_orders
            if str(order.status or "").lower() in {"rejected", "failed"}
            and normalize_provider((order.details or {}).get("provider")) == provider
        ]
        if len(rejected) >= threshold:
            blockers.append("rejected_order_burst_circuit_breaker")
        slippage_breaches = 0
        for order in recent_orders:
            details = order.details or {}
            if normalize_provider(details.get("provider")) != provider:
                continue
            slippage = self._safe_float(details.get("reported_slippage"), self._safe_float(details.get("submitted_slippage_pct")))
            adaptive = details.get("adaptive_slippage") if isinstance(details.get("adaptive_slippage"), dict) else {}
            adaptive_limit = self._safe_float(adaptive.get("max_acceptable_pct"), 0.0)
            if adaptive_limit > 0 and slippage > adaptive_limit:
                slippage_breaches += 1
        if slippage_breaches >= threshold:
            blockers.append("adaptive_slippage_breach_circuit_breaker")
        recent_decisions = (
            RapidMLDecision.query.filter_by(provider=provider, trading_connection_id=int(connection_id))
            .filter(RapidMLDecision.created_at >= cutoff)
            .order_by(RapidMLDecision.created_at.desc(), RapidMLDecision.id.desc())
            .limit(20)
            .all()
        )
        provider_failures = 0
        failure_markers = (
            "provider_snapshot_alerts",
            "provider_can_trade_false",
            "price_unavailable",
            "timeout",
            "provider_request_failed",
        )
        for decision in recent_decisions:
            decision_blockers = [str(item) for item in decision.blockers]
            if any(marker in blocker for marker in failure_markers for blocker in decision_blockers):
                provider_failures += 1
        if provider_failures >= threshold:
            blockers.append("provider_failure_burst_circuit_breaker")
        return blockers

    def _allocate_capital(
        self,
        analyses: list[dict[str, Any]],
        *,
        capital_usd: float,
        per_position_cap: float,
    ) -> tuple[dict[str, float], list[str]]:
        blockers: list[str] = []
        eligible = [
            item
            for item in analyses
            if item.get("ready")
            and isinstance(item.get("selected"), dict)
            and self._safe_float(item["selected"].get("opportunity_score")) > 0
        ]
        if not eligible:
            return {}, ["no_positive_ml_opportunity"]
        score_sum = sum(self._safe_float(item["selected"].get("opportunity_score")) for item in eligible)
        if score_sum <= 0:
            return {}, ["opportunity_score_sum_zero"]
        allocations: dict[str, float] = {}
        for item in eligible:
            connection_id = str(item["connection_id"])
            provider_balance = self._safe_float((item.get("snapshot") or {}).get("available_balance_usd"))
            selected = item["selected"]
            if self._ml_sizing_enabled():
                allocation, sizing = self._ml_allocation_usd(
                    selected,
                    capital_usd=capital_usd,
                    provider_balance=provider_balance,
                    per_position_cap=per_position_cap,
                )
                selected["ml_sizing"] = sizing
                allocations[connection_id] = allocation
            else:
                score = self._safe_float(selected.get("opportunity_score"))
                raw = capital_usd * (score / score_sum)
                allocations[connection_id] = max(0.0, min(raw, provider_balance, per_position_cap))
            if allocations[connection_id] <= 0:
                blockers.append(f"{item.get('provider')}:allocation_zero_after_caps")
        if self._ml_sizing_enabled():
            self._scale_ml_allocations_to_capital(eligible, allocations, capital_usd=capital_usd, blockers=blockers)
        return allocations, blockers

    def _scale_ml_allocations_to_capital(
        self,
        eligible: list[dict[str, Any]],
        allocations: dict[str, float],
        *,
        capital_usd: float,
        blockers: list[str],
    ) -> None:
        total = sum(self._safe_float(value) for value in allocations.values())
        if capital_usd <= 0 or total <= capital_usd + 1e-9:
            return
        factor = capital_usd / total
        for item in eligible:
            key = str(item.get("connection_id"))
            selected = item.get("selected") if isinstance(item.get("selected"), dict) else {}
            sizing = selected.get("ml_sizing") if isinstance(selected.get("ml_sizing"), dict) else {}
            scaled = self._safe_float(allocations.get(key)) * factor
            min_required = self._safe_float(sizing.get("min_required_allocation_usd"))
            sizing["capital_scale_factor"] = factor
            sizing["allocation_before_capital_scale_usd"] = self._safe_float(allocations.get(key))
            sizing["allocation_usd"] = scaled
            sizing["capital_budget_usd"] = capital_usd
            selected["ml_sizing"] = sizing
            if min_required > 0 and scaled + 1e-9 < min_required:
                allocations[key] = 0.0
                blockers.append(f"{item.get('provider')}:ml_sizing_scaled_below_exchange_minimum")
                selected.setdefault("blockers", []).append("ml_sizing_scaled_below_exchange_minimum")
                continue
            allocations[key] = max(0.0, scaled)

    def _ml_allocation_usd(
        self,
        selected: dict[str, Any],
        *,
        capital_usd: float,
        provider_balance: float,
        per_position_cap: float,
    ) -> tuple[float, dict[str, Any]]:
        provider = normalize_provider(selected.get("provider"))
        min_notional = self._provider_min_notional_usd(provider)
        min_required = min_notional + self._min_notional_buffer_usd() if min_notional > 0 else 0.0
        allocation_ceiling = min(
            max(0.0, capital_usd),
            max(0.0, provider_balance),
            max(0.0, per_position_cap),
        )
        fixed_cap = self._rapid_fixed_hard_cap_usd()
        if fixed_cap > 0:
            allocation_ceiling = min(allocation_ceiling, fixed_cap)
        decisions = selected.get("ml_decisions") if isinstance(selected.get("ml_decisions"), dict) else {}
        allocator = decisions.get("pytorch_allocator") if isinstance(decisions.get("pytorch_allocator"), dict) else {}
        risk = decisions.get("pytorch_risk_policy") if isinstance(decisions.get("pytorch_risk_policy"), dict) else {}
        cap_policy = decisions.get("pytorch_cap_policy") if isinstance(decisions.get("pytorch_cap_policy"), dict) else {}
        roi = decisions.get("pytorch_roi_target") if isinstance(decisions.get("pytorch_roi_target"), dict) else {}
        allocator_raw = allocator.get("raw") if isinstance(allocator.get("raw"), dict) else {}
        cap_raw = cap_policy.get("raw") if isinstance(cap_policy.get("raw"), dict) else {}
        roi_raw = roi.get("raw") if isinstance(roi.get("raw"), dict) else {}

        edge_score = max(0.0, min(1.0, self._safe_float(selected.get("expected_edge_bps_after_costs")) / 300.0))
        confidence_score = max(0.0, min(1.0, self._safe_float(selected.get("confidence"))))
        risk_score = max(0.0, min(1.0, self._safe_float(risk.get("confidence"))))
        allocator_score = max(0.0, min(1.0, self._safe_float(allocator_raw.get("sizing_score"), self._safe_float(allocator.get("confidence")))))
        roi_score = max(0.0, min(1.0, self._safe_float(roi_raw.get("target_probability"), self._safe_float(roi.get("confidence")))))
        model_score = (
            edge_score * 0.30
            + confidence_score * 0.25
            + risk_score * 0.20
            + allocator_score * 0.15
            + roi_score * 0.10
        )
        suggested = max(0.0, self._safe_float(cap_raw.get("suggested_notional_usdc")))
        lower_bound = min_required if min_required > 0 else 0.01
        if allocation_ceiling <= 0:
            return 0.0, {
                "source": "ml_policy",
                "blockers": ["ml_sizing_allocation_ceiling_zero"],
                "allocation_ceiling_usd": allocation_ceiling,
                "fixed_hard_cap_usd": fixed_cap,
                "min_required_allocation_usd": lower_bound,
            }
        if allocation_ceiling + 1e-9 < lower_bound:
            return 0.0, {
                "source": "ml_policy",
                "blockers": ["ml_sizing_ceiling_below_exchange_minimum"],
                "allocation_ceiling_usd": allocation_ceiling,
                "fixed_hard_cap_usd": fixed_cap,
                "min_required_allocation_usd": lower_bound,
            }
        if suggested >= lower_bound:
            target = min(suggested, allocation_ceiling)
            source = "cap_policy_suggested_notional"
        else:
            target = lower_bound + (allocation_ceiling - lower_bound) * model_score
            source = "ml_edge_confidence_blend"
        target = max(lower_bound, min(target, allocation_ceiling))
        return target, {
            "source": source,
            "allocation_usd": target,
            "allocation_ceiling_usd": allocation_ceiling,
            "fixed_hard_cap_usd": fixed_cap,
            "provider_balance_usd": provider_balance,
            "capital_usd": capital_usd,
            "min_required_allocation_usd": lower_bound,
            "cap_policy_suggested_notional_usdc": suggested,
            "model_score": model_score,
            "edge_score": edge_score,
            "confidence_score": confidence_score,
            "risk_score": risk_score,
            "allocator_score": allocator_score,
            "roi_score": roi_score,
            "blockers": [],
        }

    def _apply_correlation_guard(
        self,
        analyses: list[dict[str, Any]],
        allocations: dict[str, float],
        *,
        capital_usd: float,
    ) -> None:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        same_direction: dict[str, list[dict[str, Any]]] = {}
        for item in analyses:
            selected = item.get("selected") if isinstance(item.get("selected"), dict) else None
            if not selected:
                continue
            side = str(selected.get("side") or "")
            symbol = str(selected.get("symbol") or "")
            entry = {
                "analysis": item,
                "selected": selected,
                "connection_id": item.get("connection_id"),
                "allocation": self._safe_float(allocations.get(str(item.get("connection_id")))),
            }
            grouped.setdefault((symbol, side), []).append(entry)
            same_direction.setdefault(side, []).append(entry)
        for entries in grouped.values():
            if len(entries) <= 1:
                continue
            entries.sort(key=lambda row: self._safe_float((row.get("selected") or {}).get("opportunity_score")), reverse=True)
            for entry in entries[1:]:
                allocations[str(entry.get("connection_id"))] = 0.0
                entry["analysis"].setdefault("blockers", []).append("global_correlation_same_symbol_side")
                entry["selected"].setdefault("blockers", []).append("global_correlation_same_symbol_side")
        max_directional = capital_usd * self._max_directional_exposure_pct()
        for entries in same_direction.values():
            if len(entries) <= 1:
                continue
            total = sum(self._safe_float(allocations.get(str(entry.get("connection_id")))) for entry in entries)
            if total <= max_directional:
                continue
            entries.sort(key=lambda row: self._safe_float((row.get("selected") or {}).get("opportunity_score")))
            for entry in entries:
                if total <= max_directional:
                    break
                key = str(entry.get("connection_id"))
                total -= self._safe_float(allocations.get(key))
                allocations[key] = 0.0
                entry["analysis"].setdefault("blockers", []).append("global_directional_exposure_guard")
                entry["selected"].setdefault("blockers", []).append("global_directional_exposure_guard")

    def _quantity_for(
        self,
        provider: str,
        symbol: str,
        allocation_usd: float,
        price: float,
        venue_symbol: Any,
    ) -> tuple[float, dict[str, Any]]:
        if price <= 0 or allocation_usd <= 0:
            return 0.0, {"blockers": ["invalid_quantity_inputs"]}
        raw_quantity = allocation_usd / price
        if provider == "hyperliquid":
            min_notional = self._provider_min_notional_usd(provider)
            min_buffer = self._min_notional_buffer_usd()
            min_allocation = min_notional + min_buffer
            if min_notional > 0 and allocation_usd + 1e-9 < min_allocation:
                return 0.0, {
                    "quantity": 0.0,
                    "notional_usd": 0.0,
                    "allocation_usd": allocation_usd,
                    "min_notional_usd": min_notional,
                    "min_notional_buffer_usd": min_buffer,
                    "min_required_allocation_usd": min_allocation,
                    "blockers": ["hyperliquid_min_order_value_exceeds_allocation"],
                }
            quantity = math.floor(raw_quantity * 1_000_000) / 1_000_000
            notional = quantity * price
            if min_notional > 0 and notional + 1e-9 < min_notional:
                return 0.0, {
                    "quantity": quantity,
                    "notional_usd": notional,
                    "allocation_usd": allocation_usd,
                    "min_notional_usd": min_notional,
                    "min_notional_buffer_usd": min_buffer,
                    "min_required_allocation_usd": min_allocation,
                    "blockers": ["hyperliquid_min_order_value_exceeds_allocation"],
                }
            return quantity, {"quantity": quantity, "notional_usd": notional, "blockers": []}
        if provider != "kucoin":
            quantity = math.floor(raw_quantity * 1_000_000) / 1_000_000
            return quantity, {"quantity": quantity, "notional_usd": quantity * price, "blockers": []}
        metadata = self._symbol_metadata(provider, symbol)
        if metadata.get("blockers"):
            return 0.0, {"blockers": list(metadata["blockers"])}
        specs = metadata["contract_spec"]
        contract_size = self._safe_float(specs.get("contract_size"))
        step = max(1, int(self._safe_float(specs.get("size_step"), 1.0)))
        min_size = max(step, int(self._safe_float(specs.get("min_size"), step)))
        if contract_size <= 0:
            return 0.0, {"blockers": ["kucoin_contract_size_invalid"]}
        raw_contracts = raw_quantity / contract_size
        contracts = int(math.floor(raw_contracts / step) * step)
        if contracts < min_size:
            min_notional = min_size * contract_size * price
            return 0.0, {
                "contract_size": contract_size,
                "contracts": contracts,
                "min_size": min_size,
                "min_notional_usd": min_notional,
                "blockers": ["kucoin_min_contract_size_exceeds_allocation"],
            }
        quantity = contracts * contract_size
        return (
            quantity,
            {
                "venue_symbol": metadata.get("venue_symbol") or venue_symbol,
                "contract_size": contract_size,
                "contracts": contracts,
                "size_step": step,
                "min_size": min_size,
                "quantity": quantity,
                "notional_usd": quantity * price,
                "blockers": [],
            },
        )

    def _provider_min_notional_usd(self, provider: str) -> float:
        if str(provider or "").lower() == "hyperliquid":
            return max(0.0, self._safe_float(self.config.get("HYPERLIQUID_MIN_ORDER_VALUE_USD"), 10.0))
        return 0.0

    def _min_notional_buffer_usd(self) -> float:
        return max(0.0, self._safe_float(self.config.get("RAPID_ML_MIN_NOTIONAL_BUFFER_USD"), 0.50))

    def _symbol_metadata(self, provider: str, symbol: str, *, require_active_futures: bool = False) -> dict[str, Any]:
        symbol_key = str(symbol or "").upper()
        if provider != "kucoin":
            blockers: list[str] = []
            market = self._active_market_row(provider, symbol_key)
            if require_active_futures and market is None:
                blockers.append(f"futures_market_not_active:{symbol_key}")
            allowed_symbols = {str(item).upper() for item in self.config.get("ALLOWED_SYMBOLS", []) if str(item)}
            if not require_active_futures and allowed_symbols and symbol_key not in allowed_symbols:
                blockers.append(f"symbol_not_allowed:{symbol_key}")
            return {
                "symbol": symbol_key,
                "venue_symbol": str(getattr(market, "venue_symbol", None) or symbol_key).strip(),
                "active_futures_market": market is not None,
                "market_id": int(market.id) if market is not None and market.id is not None else None,
                "liquidity_usd": self._safe_float(getattr(market, "liquidity_usd", 0.0)) if market is not None else 0.0,
                "max_leverage": self._safe_float(getattr(market, "max_leverage", 0.0)) if market is not None else 0.0,
                "blockers": blockers,
            }
        venue_symbol = self._kucoin_symbol_map().get(symbol_key)
        if not venue_symbol:
            return {"symbol": symbol_key, "blockers": [f"kucoin_symbol_mapping_missing:{symbol_key}"]}
        spec = self._kucoin_contract_specs().get(str(venue_symbol).upper())
        if not isinstance(spec, dict):
            return {
                "symbol": symbol_key,
                "venue_symbol": str(venue_symbol).upper(),
                "blockers": [f"kucoin_contract_specs_missing:{str(venue_symbol).upper()}"],
            }
        market = self._active_market_row(provider, symbol_key, venue_symbol=str(venue_symbol).upper())
        if require_active_futures and market is None:
            return {
                "symbol": symbol_key,
                "venue_symbol": str(venue_symbol).upper(),
                "blockers": [f"futures_market_not_active:{str(venue_symbol).upper()}"],
            }
        return {
            "symbol": symbol_key,
            "venue_symbol": str(venue_symbol).upper(),
            "contract_spec": spec,
            "active_futures_market": market is not None,
            "market_id": int(market.id) if market is not None and market.id is not None else None,
            "liquidity_usd": self._safe_float(getattr(market, "liquidity_usd", 0.0)) if market is not None else 0.0,
            "max_leverage": self._safe_float(getattr(market, "max_leverage", 0.0)) if market is not None else 0.0,
            "blockers": [],
        }

    def _reference_price(self, provider: str, symbol: str, connection_id: int, *, venue_symbol: Any = None) -> float:
        market_symbol = str(venue_symbol or symbol).strip() or symbol
        if provider == "kucoin":
            try:
                connector = self.trading_connections.connector_for_user(
                    int(db.session.get(TradingConnection, connection_id).user_id), connection_id
                )
                getter = getattr(connector, "get_mid_price", None)
                if callable(getter):
                    return max(0.0, float(getter(market_symbol, "live")))
            except Exception:  # noqa: BLE001
                return 0.0
        try:
            return max(0.0, float(self.market_data.get_mid_price(market_symbol, "live")))
        except Exception:  # noqa: BLE001
            return 0.0

    def _spread_bps(self, provider: str, symbol: str, connection_id: int, *, venue_symbol: Any = None) -> float:
        book: dict[str, Any] = {}
        market_symbol = str(venue_symbol or symbol).strip() or symbol
        try:
            if provider == "kucoin":
                connector = self.trading_connections.connector_for_user(
                    int(db.session.get(TradingConnection, connection_id).user_id), connection_id
                )
                getter = getattr(connector, "get_order_book", None)
                book = dict(getter(market_symbol, "live") or {}) if callable(getter) else {}
            else:
                book = dict(self.market_data.get_order_book(market_symbol, "live") or {})
        except Exception:  # noqa: BLE001
            return 0.0
        bid = self._best_price(book, "bid")
        ask = self._best_price(book, "ask")
        if bid <= 0 or ask <= 0 or ask < bid:
            return 0.0
        mid = (bid + ask) / 2.0
        return ((ask - bid) / mid) * 10_000.0 if mid > 0 else 0.0

    def _live_feature_context(
        self,
        provider: str,
        symbol: str,
        connection_id: int,
        *,
        venue_symbol: Any = None,
        reference_price: float = 0.0,
        spread_bps: float = 0.0,
    ) -> dict[str, Any]:
        factory = getattr(self.ml_decision_engine, "feature_factory", None)
        if factory is None:
            return {"rapid_feature_blockers": ["ml_feature_factory_unavailable"]}
        timeframe = str(self.config.get("RAPID_ML_FEATURE_TIMEFRAME") or "1m").lower()
        limit = max(40, min(500, int(self._safe_float(self.config.get("RAPID_ML_FEATURE_CANDLE_LIMIT"), 200))))
        market_symbol = str(venue_symbol or symbol).strip() or symbol
        try:
            if provider == "kucoin":
                connection = db.session.get(TradingConnection, int(connection_id))
                connector = self.trading_connections.connector_for_user(int(connection.user_id), int(connection_id))
                candles = list(connector.get_candles(market_symbol, timeframe, "live", limit))
            else:
                candles = list(self.market_data.get_candles(symbol, timeframe, "live", limit=limit, retry=False))
        except Exception as exc:  # noqa: BLE001
            return {
                "rapid_feature_timeframe": timeframe,
                "rapid_feature_candle_count": 0,
                "rapid_feature_blockers": [f"live_feature_candles_unavailable:{self._short_error(exc)}"],
            }
        if len(candles) < 40:
            return {
                "rapid_feature_timeframe": timeframe,
                "rapid_feature_candle_count": len(candles),
                "rapid_feature_blockers": ["live_feature_candles_insufficient"],
            }
        optimizer_context = {
            **provider_feature_context(provider),
            "provider": provider,
            "execution_venue": provider,
            "venue_symbol": market_symbol,
            "strategy_name": "rapid_ml_live",
            "profile": "rapid_ml",
            "optimizer_profile": "rapid_ml",
            "horizon": "1h",
            "reference_price": reference_price,
            "spread_bps": spread_bps,
            "fee_bps": self._provider_fee_map().get(provider, 0.0),
            "net_return_after_costs": 0.0,
            "total_return": 0.0,
            "recent_1h_return": 0.0,
            "profit_factor": 1.0,
            "trade_count": 0,
            "rejected": False,
            "rejection_reason": "",
        }
        try:
            payload = dict(
                factory.build(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=candles,
                    optimizer_context=optimizer_context,
                    provider_context=optimizer_context,
                    order_book={},
                )
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "rapid_feature_timeframe": timeframe,
                "rapid_feature_candle_count": len(candles),
                "rapid_feature_blockers": [f"ml_feature_build_failed:{self._short_error(exc)}"],
            }
        payload["rapid_feature_timeframe"] = timeframe
        payload["rapid_feature_candle_count"] = len(candles)
        payload["rapid_feature_candles"] = candles[-min(len(candles), 240) :]
        return payload

    def _connections_for_scope(self, user_id: int, provider_scope: str) -> list[TradingConnection]:
        providers = None if provider_scope == "both" else [provider_scope]
        connections = self.trading_connections.verified_tradable_connections(user_id, providers=providers)
        wanted = {"hyperliquid", "kucoin"} if provider_scope == "both" else {provider_scope}
        by_provider: dict[str, TradingConnection] = {}
        for connection in connections:
            provider = normalize_provider(connection.provider)
            if provider in wanted and provider not in by_provider:
                by_provider[provider] = connection
        return [by_provider[key] for key in ("hyperliquid", "kucoin") if key in by_provider]

    def _run_provider_workers(self, connection_ids: list[int], worker: Callable[[int], dict[str, Any]]) -> list[dict[str, Any]]:
        ids = [int(item) for item in connection_ids if item]
        if not ids:
            return []
        if len(ids) == 1 or bool(self.config.get("TESTING", False)) or not has_app_context():
            return [worker(connection_id) for connection_id in ids]
        app = current_app._get_current_object()
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(2, len(ids))) as executor:
            futures = {
                executor.submit(self._worker_with_app_context, app, worker, connection_id): connection_id
                for connection_id in ids
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    results.append({"connection_id": futures[future], "ready": False, "blockers": [str(exc)]})
        results.sort(key=lambda item: ids.index(int(item.get("connection_id") or 0)) if int(item.get("connection_id") or 0) in ids else 999)
        return results

    @staticmethod
    def _worker_with_app_context(app: Any, worker: Callable[[int], dict[str, Any]], connection_id: int) -> dict[str, Any]:
        with app.app_context():
            return worker(connection_id)

    def _submit_blockers(self, *, submit: bool, confirm: str) -> list[str]:
        if not submit:
            return []
        blockers: list[str] = []
        if str(confirm or "").strip() != CONFIRMATION_PHRASE:
            blockers.append("confirmation_required")
        if not bool(self.config.get("RAPID_ML_LIVE_ENABLED", False)):
            blockers.append("RAPID_ML_LIVE_ENABLED=false")
        if bool(self.config.get("RAPID_ML_PREVIEW_ONLY", True)):
            blockers.append("RAPID_ML_PREVIEW_ONLY=true")
        if bool(self.config.get("CANARY_PREVIEW_ONLY", True)):
            blockers.append("CANARY_PREVIEW_ONLY=true")
        if not bool(self.config.get("ENABLE_LIVE_TRADING", False)):
            blockers.append("ENABLE_LIVE_TRADING=false")
        if not bool(self.config.get("EXPLICIT_LIVE_CONFIRMED", False)) or not bool(Setting.get_json("explicit_live_confirmed", False)):
            blockers.append("explicit_live_confirmation_missing")
        if not bool(self.config.get("SECONDARY_CONFIRMATION", False)) or not bool(Setting.get_json("secondary_confirmation", False)):
            blockers.append("secondary_confirmation_missing")
        return blockers

    def _throttle_blocker(
        self,
        provider: str,
        max_order_rate_per_provider: float,
        last_submit_at: dict[str, float],
    ) -> str:
        if max_order_rate_per_provider <= 0:
            return "order_rate_zero"
        min_interval = 1.0 / max_order_rate_per_provider
        elapsed = time.monotonic() - float(last_submit_at.get(provider, 0.0) or 0.0)
        if provider in last_submit_at and elapsed < min_interval:
            return "order_rate_throttle"
        return ""

    def _performance_payload(
        self,
        user_id: int,
        connections: list[TradingConnection],
        started_at: datetime | None,
        *,
        provider_states_by_connection: dict[int, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        query = Order.query.filter_by(user_id=user_id, mode="live")
        if started_at is not None:
            query = query.filter(Order.created_at >= started_at)
        orders = [
            order
            for order in query.order_by(Order.created_at.asc(), Order.id.asc()).limit(500).all()
            if bool((order.details or {}).get("rapid_ml"))
        ]
        fills = Fill.query.filter(Fill.order_id.in_([order.id for order in orders])).all() if orders else []
        unrealized = 0.0
        open_orders = 0
        positions = 0
        for connection in connections:
            cached_state = (provider_states_by_connection or {}).get(int(connection.id))
            if isinstance(cached_state, dict):
                open_orders += int(self._safe_float(cached_state.get("open_orders_count")))
                positions += int(self._safe_float(cached_state.get("positions_count")))
                continue
            try:
                snapshot = self.trading_connections.account_snapshot(user_id, "live", connection.id)
                open_orders += len(getattr(snapshot, "open_orders", []) or [])
                rows = list(getattr(snapshot, "positions", []) or [])
                positions += len(rows)
                unrealized += sum(self._safe_float(row.get("unrealized_pnl")) for row in rows if isinstance(row, dict))
            except Exception:  # noqa: BLE001
                continue
        return {
            "orders": len(orders),
            "wins": sum(1 for fill in fills if self._fill_net_pnl(fill) > 0),
            "losses": sum(1 for fill in fills if self._fill_net_pnl(fill) < 0),
            "realized_pnl_usd": sum(self._fill_net_pnl(fill) for fill in fills),
            "unrealized_pnl_usd": unrealized,
            "fees_usd": sum(self._safe_float(fill.fee) + self._safe_float(getattr(fill, "funding_fee", 0.0)) for fill in fills),
            "open_orders_count": open_orders,
            "positions_count": positions,
        }

    def _fill_net_pnl(self, fill: Any) -> float:
        return self._safe_float(getattr(fill, "pnl", 0.0)) - self._safe_float(getattr(fill, "fee", 0.0)) - self._safe_float(getattr(fill, "funding_fee", 0.0))

    def _audit_session(self, session: RapidMLSession, payload: dict[str, Any]) -> None:
        action = "rapid_ml_submit" if session.submit_requested else "rapid_ml_preview"
        audit = AuditLog(
            user_id=session.user_id,
            category="rapid_ml",
            action=action,
            message=f"Rapid ML trader {session.status}",
        )
        audit.details = self._json_safe({
            "session_id": session.id,
            "status": session.status,
            "submitted_count": payload.get("submitted_count"),
            "blockers": payload.get("blockers", []),
            "capital": payload.get("capital", {}),
            "performance": payload.get("performance", {}),
        })
        db.session.add(audit)
        db.session.commit()

    def _session_status(
        self,
        *,
        submit_requested: bool,
        effective_submit: bool,
        submitted_count: int,
        preview_count: int,
        blockers: list[str],
        cycle_payloads: list[dict[str, Any]],
    ) -> str:
        if submit_requested and not effective_submit:
            return "submit_blocked"
        if submitted_count > 0:
            return "submitted"
        if preview_count > 0:
            return "preview_ready"
        if blockers:
            return "blocked"
        if cycle_payloads:
            return "no_trade"
        return "blocked"

    def _select_candidate(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        eligible = [item for item in candidates if self._safe_float(item.get("opportunity_score")) > 0 and not item.get("blockers")]
        if not eligible:
            return None
        eligible.sort(key=lambda item: self._safe_float(item.get("opportunity_score")), reverse=True)
        return eligible[0]

    @staticmethod
    def _analysis_for_connection(analyses: list[dict[str, Any]], connection_id: int) -> dict[str, Any]:
        for item in analyses:
            if int(item.get("connection_id") or 0) == int(connection_id):
                return item
        return {}

    @staticmethod
    def _balance_amount(balances: list[dict[str, Any]], asset: str, *, prefer_withdrawable: bool) -> float:
        keys = ("withdrawable", "available", "available_balance", "free", "value", "total") if prefer_withdrawable else (
            "value",
            "total",
            "withdrawable",
            "available",
        )
        best = 0.0
        for balance in balances:
            if not isinstance(balance, dict):
                continue
            if str(balance.get("asset") or "").upper() != str(asset or "").upper():
                continue
            for key in keys:
                value = RapidMLTraderService._safe_float(balance.get(key))
                if value > best:
                    best = value
        return best

    def _symbols_for_provider(self, provider: str, *, user_id: int | None = None, connection_id: int | None = None) -> tuple[str, ...]:
        provider_key = normalize_provider(provider)
        values = self._configured_symbols_for_provider(provider_key)
        if not values and self._auto_futures_universe_enabled():
            values = self._auto_futures_symbols(provider_key, user_id=user_id, connection_id=connection_id)
            if not values:
                return tuple()
        if not values:
            values = list(PROVIDER_SYMBOLS.get(provider_key, ("BTC", "ETH")))
        max_symbols = max(1, int(self._safe_float(self.config.get("RAPID_ML_MAX_SYMBOLS_PER_PROVIDER"), 12)))
        return tuple(values[:max_symbols])

    def _configured_symbols_for_provider(self, provider: str) -> list[str]:
        provider_key = normalize_provider(provider)
        configured = self.config.get(f"RAPID_ML_SYMBOLS_{provider_key.upper()}")
        if not configured:
            configured = self.config.get("RAPID_ML_SYMBOLS")
        return self._parse_symbols(configured)

    def _auto_futures_symbols(self, provider: str, *, user_id: int | None, connection_id: int | None) -> list[str]:
        provider_key = normalize_provider(provider)
        self._refresh_futures_universe(provider_key, user_id=user_id, connection_id=connection_id)
        markets = self._active_markets(provider_key)
        values: list[str] = []
        for market in markets:
            symbol = str(market.symbol or market.venue_symbol).upper()
            if symbol and symbol not in values:
                values.append(symbol)
        return values

    def _refresh_futures_universe(self, provider: str, *, user_id: int | None, connection_id: int | None) -> None:
        if self.leveraged_markets is None or user_id is None or connection_id is None:
            return
        refresh_seconds = self._universe_refresh_seconds()
        if refresh_seconds <= 0:
            return
        key = f"rapid_ml_universe_last_sync:{normalize_provider(provider)}:{int(connection_id)}"
        now = time.time()
        last = self._safe_float(Setting.get_json(key, 0.0))
        if last > 0 and now - last < refresh_seconds:
            return
        connection = db.session.get(TradingConnection, int(connection_id))
        if connection is None or int(connection.user_id) != int(user_id):
            return
        sync_for_connection = getattr(self.leveraged_markets, "sync_for_connection", None)
        if not callable(sync_for_connection):
            return
        try:
            sync_for_connection(connection, mode="live", feature_scope="all", persist_features=False)
            Setting.set_json(key, now)
        except Exception:  # noqa: BLE001
            return

    def _active_markets(self, provider: str) -> list[LeveragedMarket]:
        active_markets = getattr(self.leveraged_markets, "active_markets", None) if self.leveraged_markets is not None else None
        if callable(active_markets):
            try:
                return list(active_markets(provider))
            except Exception:  # noqa: BLE001
                pass
        return (
            LeveragedMarket.query.filter_by(provider=normalize_provider(provider), status="active")
            .order_by(LeveragedMarket.liquidity_usd.desc(), LeveragedMarket.max_leverage.desc(), LeveragedMarket.symbol.asc())
            .all()
        )

    def _active_market_row(self, provider: str, symbol: str, *, venue_symbol: str | None = None) -> LeveragedMarket | None:
        provider_key = normalize_provider(provider)
        symbol_key = str(symbol or "").upper()
        venue_key = str(venue_symbol or symbol_key).upper()
        if not symbol_key and not venue_key:
            return None
        return (
            LeveragedMarket.query.filter_by(provider=provider_key, status="active")
            .filter((LeveragedMarket.symbol == symbol_key) | (LeveragedMarket.venue_symbol == venue_key))
            .order_by(LeveragedMarket.liquidity_usd.desc(), LeveragedMarket.max_leverage.desc())
            .first()
        )

    @staticmethod
    def _parse_symbols(raw: Any) -> list[str]:
        if isinstance(raw, (list, tuple, set)):
            items = raw
        else:
            items = str(raw or "").replace(";", ",").split(",")
        values: list[str] = []
        for item in items:
            symbol = str(item or "").strip().upper()
            if symbol and symbol not in values:
                values.append(symbol)
        return values

    def _kucoin_symbol_map(self) -> dict[str, str]:
        raw = self.config.get("KUCOIN_SYMBOL_MAP_JSON")
        if isinstance(raw, dict):
            return {str(key).upper(): str(value).upper() for key, value in raw.items()}
        if isinstance(raw, str) and raw.strip():
            try:
                import json

                value = json.loads(raw)
                if isinstance(value, dict):
                    return {str(key).upper(): str(val).upper() for key, val in value.items()}
            except Exception:  # noqa: BLE001
                pass
        return dict(KUCOIN_SYMBOLS)

    def _kucoin_contract_specs(self) -> dict[str, dict[str, Any]]:
        raw = self.config.get("KUCOIN_CONTRACT_SPECS_JSON")
        if isinstance(raw, dict):
            return {str(key).upper(): dict(value) for key, value in raw.items() if isinstance(value, dict)}
        if isinstance(raw, str) and raw.strip():
            try:
                import json

                value = json.loads(raw)
                if isinstance(value, dict):
                    return {str(key).upper(): dict(val) for key, val in value.items() if isinstance(val, dict)}
            except Exception:  # noqa: BLE001
                pass
        return dict(KUCOIN_CONTRACT_SPECS)

    @staticmethod
    def _best_price(book: dict[str, Any], side: str) -> float:
        keys = ("bids", "levels", "bid") if side == "bid" else ("asks", "levels", "ask")
        for key in keys:
            value = book.get(key)
            if isinstance(value, (int, float, str)):
                return RapidMLTraderService._safe_float(value)
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, dict):
                    return RapidMLTraderService._safe_float(first.get("price") or first.get("px"))
                if isinstance(first, list) and first:
                    return RapidMLTraderService._safe_float(first[0])
        return 0.0

    def _profitability_model(
        self,
        provider: str,
        context: dict[str, Any],
        *,
        confidence: float,
        signal: dict[str, Any],
        roi: dict[str, Any],
        offline: dict[str, Any],
    ) -> dict[str, Any]:
        source_returns = {
            "signal": self._safe_float(signal.get("expected_return")),
            "roi_target": self._safe_float(roi.get("expected_return")),
            "offline_ranker": self._safe_float(offline.get("prediction")),
        }
        source_approved = {
            "signal": str(signal.get("action") or "").lower() in {"buy", "sell"} and not bool(signal.get("blockers")),
            "roi_target": str(roi.get("action") or "").lower() == "target_met_candidate" and not bool(roi.get("blockers")),
            "offline_ranker": str(offline.get("status") or "").lower() == "promoted" and not bool(offline.get("blockers")),
        }
        positive_sources = {
            key: value
            for key, value in source_returns.items()
            if value > 0 and math.isfinite(value) and bool(source_approved.get(key, False))
        }
        gross_expected = sum(positive_sources.values()) / len(positive_sources) if positive_sources else 0.0
        spread_bps = self._safe_float(context.get("spread_bps"))
        spread_estimated = False
        if spread_bps <= 0:
            spread_bps = self._unknown_spread_bps()
            spread_estimated = spread_bps > 0
        round_trip_fee_bps = self._provider_cost_bps(provider)
        slippage_bps = self._rapid_slippage_bps()
        reserve_bps = self._cost_reserve_bps()
        total_cost_bps = round_trip_fee_bps + slippage_bps + max(0.0, spread_bps) + reserve_bps
        edge_bps = gross_expected * 10_000.0 - total_cost_bps
        min_edge_bps = self._min_edge_bps()
        blockers: list[str] = []
        min_confidence = self._min_confidence()
        if confidence < min_confidence:
            blockers.append("ml_confidence_below_profitability_threshold")
        min_agreement = self._min_edge_agreement()
        if len(positive_sources) < min_agreement:
            blockers.append("ml_edge_agreement_below_threshold")
        if edge_bps < min_edge_bps:
            blockers.append("ml_edge_below_cost_threshold")
        return {
            "source_returns": source_returns,
            "source_approved": source_approved,
            "positive_edge_sources": sorted(positive_sources),
            "positive_edge_source_count": len(positive_sources),
            "required_positive_edge_sources": min_agreement,
            "gross_expected_return": gross_expected,
            "gross_expected_bps": gross_expected * 10_000.0,
            "round_trip_fee_bps": round_trip_fee_bps,
            "slippage_bps": slippage_bps,
            "spread_bps": spread_bps,
            "spread_estimated": spread_estimated,
            "cost_reserve_bps": reserve_bps,
            "total_cost_bps": total_cost_bps,
            "min_edge_bps": min_edge_bps,
            "min_confidence": min_confidence,
            "confidence": confidence,
            "edge_bps_after_costs": edge_bps,
            "expected_return_after_costs": edge_bps / 10_000.0,
            "blockers": blockers,
        }

    def _profitability_gate_config(self) -> dict[str, Any]:
        return {
            "min_edge_bps": self._min_edge_bps(),
            "min_confidence": self._min_confidence(),
            "min_edge_agreement": self._min_edge_agreement(),
            "cost_reserve_bps": self._cost_reserve_bps(),
            "slippage_bps": self._rapid_slippage_bps(),
            "unknown_spread_bps": self._unknown_spread_bps(),
            "fee_bps_by_provider": self._provider_fee_map(),
            "fee_model": "round_trip_entry_exit",
        }

    def _provider_cost_bps(self, provider: str) -> float:
        provider_key = normalize_provider(provider)
        fee_map = self._provider_fee_map()
        if provider_key in fee_map:
            return max(0.0, self._safe_float(fee_map[provider_key]))
        return 10.0 if provider_key == "kucoin" else 8.0

    def _provider_fee_map(self) -> dict[str, float]:
        raw = self.config.get("RAPID_ML_FEE_BPS_BY_PROVIDER_JSON")
        if isinstance(raw, dict):
            return {normalize_provider(key): max(0.0, self._safe_float(value)) for key, value in raw.items()}
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                return {normalize_provider(key): max(0.0, self._safe_float(value)) for key, value in parsed.items()}
        return {"hyperliquid": 8.0, "kucoin": 10.0}

    def _min_edge_bps(self) -> float:
        return max(0.0, self._safe_float(self.config.get("RAPID_ML_MIN_EDGE_BPS"), 5.0))

    def _min_confidence(self) -> float:
        return min(1.0, max(0.0, self._safe_float(self.config.get("RAPID_ML_MIN_CONFIDENCE"), 0.55)))

    def _min_edge_agreement(self) -> int:
        return max(1, int(self._safe_float(self.config.get("RAPID_ML_MIN_EDGE_AGREEMENT"), 2)))

    def _cost_reserve_bps(self) -> float:
        return max(0.0, self._safe_float(self.config.get("RAPID_ML_COST_RESERVE_BPS"), 2.0))

    def _rapid_slippage_bps(self) -> float:
        return max(0.0, self._safe_float(self.config.get("RAPID_ML_SLIPPAGE_BPS"), 4.0))

    def _unknown_spread_bps(self) -> float:
        return max(0.0, self._safe_float(self.config.get("RAPID_ML_UNKNOWN_SPREAD_BPS"), 5.0))

    @staticmethod
    def _execution_style(selected: dict[str, Any]) -> str:
        execution = (selected.get("ml_decisions") or {}).get("pytorch_execution_policy", {})
        raw = execution.get("raw") if isinstance(execution, dict) and isinstance(execution.get("raw"), dict) else {}
        return str(raw.get("execution_style_suggestion") or "risk_engine_default")

    def _stop_take_prices(self, selected: dict[str, Any], side: str, price: float) -> tuple[float, float]:
        ml_decisions = selected.get("ml_decisions") if isinstance(selected.get("ml_decisions"), dict) else {}
        raw_payloads = [
            payload.get("raw")
            for payload in ml_decisions.values()
            if isinstance(payload, dict) and isinstance(payload.get("raw"), dict)
        ]
        stop_pct = self._first_pct(
            raw_payloads,
            ("suggested_stop_loss_pct", "stop_loss_pct", "expected_stop_loss_pct", "max_loss_pct"),
            default=0.003,
            lower=0.001,
            upper=0.05,
        )
        take_pct = self._first_pct(
            raw_payloads,
            ("suggested_take_profit_pct", "take_profit_pct", "target_return_pct", "target_return"),
            default=max(0.006, stop_pct * 1.5),
            lower=0.001,
            upper=0.20,
        )
        if side == "buy":
            return price * (1.0 - stop_pct), price * (1.0 + take_pct)
        return price * (1.0 + stop_pct), price * (1.0 - take_pct)

    def _first_pct(
        self,
        payloads: list[Any],
        keys: tuple[str, ...],
        *,
        default: float,
        lower: float,
        upper: float,
    ) -> float:
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            for key in keys:
                if key not in payload:
                    continue
                value = self._safe_float(payload.get(key), math.nan)
                if not math.isfinite(value) or value <= 0:
                    continue
                if value > 1.0:
                    value /= 100.0
                return max(lower, min(value, upper))
        return max(lower, min(default, upper))

    def _effective_interval_ms(self, value: int | None) -> int:
        configured = int(value if value is not None else self.config.get("RAPID_ML_DECISION_INTERVAL_MS", 1000))
        return max(250, configured)

    def _effective_order_rate(self, value: float | None) -> float:
        configured = self._safe_float(value if value is not None else self.config.get("RAPID_ML_MAX_ORDER_RATE_PER_PROVIDER"), 1.0)
        return max(0.0, configured)

    def _cycle_count(self, duration_minutes: float, interval_ms: int, *, submit: bool) -> int:
        duration_seconds = max(0.0, self._safe_float(duration_minutes, 1.0) * 60.0)
        if duration_seconds <= 0:
            return 1
        return max(1, int(duration_seconds / max(interval_ms / 1000.0, 0.25)))

    def _daily_loss_pct(self) -> float:
        controls = Setting.get_json("risk_controls", {})
        if isinstance(controls, dict) and bool(controls.get("daily_loss_unlimited", False)):
            return 0.0
        return min(0.10, max(0.0001, self._safe_float(self.config.get("RAPID_ML_MAX_DAILY_LOSS_PCT"), 0.05)))

    def _position_pct(self) -> float:
        return min(1.0, max(0.0001, self._safe_float(self.config.get("RAPID_ML_MAX_POSITION_PCT"), 0.20)))

    def _daily_loss_cap(self, capital_usd: float) -> float:
        if self._daily_loss_pct() <= 0:
            return 0.0
        return max(0.01, capital_usd * self._daily_loss_pct())

    def _per_position_cap(self, capital_usd: float) -> float:
        if self._ml_sizing_enabled():
            return max(0.01, capital_usd)
        return max(0.01, capital_usd * self._position_pct())

    def _auto_futures_universe_enabled(self) -> bool:
        return bool(self.config.get("RAPID_ML_AUTO_FUTURES_UNIVERSE_ENABLED", True))

    def _ml_sizing_enabled(self) -> bool:
        return bool(self.config.get("RAPID_ML_ML_SIZING_ENABLED", True))

    def _rapid_auto_close_enabled(self) -> bool:
        return bool(self.config.get("RAPID_ML_AUTO_CLOSE_ENABLED", True))

    def _rapid_rotate_enabled(self) -> bool:
        return bool(self.config.get("RAPID_ML_ROTATE_ENABLED", True))

    def _rapid_manage_manual_positions(self) -> bool:
        return bool(self.config.get("RAPID_ML_MANAGE_MANUAL_POSITIONS", False))

    def _rapid_rotate_min_score_delta(self) -> float:
        return max(0.0, self._safe_float(self.config.get("RAPID_ML_ROTATE_MIN_SCORE_DELTA"), 0.10))

    def _rapid_fixed_hard_cap_usd(self) -> float:
        return max(0.0, self._safe_float(self.config.get("RAPID_ML_HARD_CAP_USDC"), 0.0))

    def _universe_refresh_seconds(self) -> float:
        return max(0.0, self._safe_float(self.config.get("RAPID_ML_UNIVERSE_REFRESH_SECONDS"), 300.0))

    def _max_directional_exposure_pct(self) -> float:
        return min(1.0, max(0.0001, self._safe_float(self.config.get("RAPID_ML_MAX_DIRECTIONAL_EXPOSURE_PCT"), 0.40)))

    @staticmethod
    def _provider_scope(provider: str) -> str:
        value = str(provider or "both").strip().lower()
        if value not in {"both", "hyperliquid", "kucoin"}:
            raise ValueError("--provider must be one of both, hyperliquid, or kucoin.")
        return value

    @staticmethod
    def _positive_float(value: Any) -> float:
        parsed = RapidMLTraderService._safe_float(value)
        return parsed if parsed > 0 and math.isfinite(parsed) else 0.0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): RapidMLTraderService._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [RapidMLTraderService._json_safe(item) for item in value]
        return str(value)

    @staticmethod
    def _short_error(exc: Exception) -> str:
        text = str(exc or exc.__class__.__name__).replace("\n", " ").strip()
        return text[:180] if text else exc.__class__.__name__

    @staticmethod
    def _compact_readiness_payload(payload: dict[str, Any]) -> dict[str, Any]:
        promoted = payload.get("promoted_model") if isinstance(payload.get("promoted_model"), dict) else None
        compact: dict[str, Any] = {
            "ready": bool(payload.get("ready", False)),
            "family": payload.get("family"),
            "horizon": payload.get("horizon"),
            "provider": payload.get("provider"),
            "model_type": payload.get("model_type"),
            "enabled": payload.get("enabled"),
            "source": payload.get("source"),
            "blockers": list(payload.get("blockers", []) or []),
        }
        if promoted:
            metrics = promoted.get("metrics") if isinstance(promoted.get("metrics"), dict) else {}
            compact["promoted_model"] = {
                "id": promoted.get("id") or promoted.get("model_id"),
                "status": promoted.get("status"),
                "model_type": promoted.get("model_type"),
                "validation_loss": promoted.get("validation_loss"),
                "negative_error_rate": promoted.get("negative_error_rate"),
                "drift": promoted.get("drift"),
                "false_positive_rate": metrics.get("false_positive_rate"),
                "expected_return": metrics.get("expected_return"),
                "artifact_exists": promoted.get("artifact_exists"),
                "promoted_at": promoted.get("promoted_at"),
            }
        for key in ("model_types", "blend_enabled", "require_blend"):
            if key in payload:
                compact[key] = payload.get(key)
        return RapidMLTraderService._json_safe(compact)

    @staticmethod
    def _compact_ml_decisions(decisions: dict[str, Any]) -> dict[str, Any]:
        compact: dict[str, Any] = {}
        raw_keys = {
            "execution_style_suggestion",
            "suggested_stop_loss_pct",
            "suggested_take_profit_pct",
            "stop_loss_pct",
            "take_profit_pct",
            "expected_stop_loss_pct",
            "target_return_pct",
            "target_return",
            "max_loss_pct",
            "ops_anomaly_score",
            "sizing_score",
            "policy",
            "order_type_suggestion",
            "maker_taker_preference",
            "limit_offset_bps",
            "slippage_tolerance_pct",
            "retry_policy",
            "projected_roi_pct",
            "target_probability",
            "suggested_notional_usdc",
            "suggested_leverage",
            "suggested_daily_loss_usdc",
            "risk_budget_usdc",
            "hard_cap_usdc",
            "notional_usdc",
            "probabilities",
            "signed_expected_return",
            "action_probability",
            "hold_probability",
            "directional_confidence",
            "directional_margin",
        }
        for family, payload in (decisions or {}).items():
            if not isinstance(payload, dict):
                compact[str(family)] = RapidMLTraderService._json_safe(payload)
                continue
            raw = payload.get("raw") if isinstance(payload.get("raw"), dict) else {}
            compact[str(family)] = {
                "family": payload.get("family") or family,
                "action": payload.get("action"),
                "confidence": payload.get("confidence"),
                "expected_return": payload.get("expected_return"),
                "ready": payload.get("ready"),
                "model_id": payload.get("model_id"),
                "blockers": list(payload.get("blockers", []) or []),
                "raw": {key: raw.get(key) for key in raw_keys if key in raw},
            }
        return RapidMLTraderService._json_safe(compact)

    @staticmethod
    def _intent_payload(intent: OrderIntent) -> dict[str, Any]:
        reference_price = RapidMLTraderService._safe_float(intent.limit_price or intent.metadata.get("reference_price"))
        if reference_price <= 0:
            reference_price = RapidMLTraderService._safe_float(intent.metadata.get("provider_sizing", {}).get("notional_usd")) / max(
                RapidMLTraderService._safe_float(intent.quantity), 1e-12
            )
        return {
            "symbol": intent.symbol,
            "side": intent.side,
            "quantity": intent.quantity,
            "mode": intent.mode,
            "order_type": intent.order_type,
            "reduce_only": intent.reduce_only,
            "leverage": intent.leverage,
            "stop_loss": intent.stop_loss,
            "take_profit": intent.take_profit,
            "slippage_pct": intent.slippage_pct,
            "trading_connection_id": intent.trading_connection_id,
            "idempotency_key": intent.idempotency_key,
            "notional_usd": RapidMLTraderService._safe_float(intent.metadata.get("provider_sizing", {}).get("notional_usd")),
            "metadata": dict(intent.metadata or {}),
        }

    @staticmethod
    def _order_payload(order: Order) -> dict[str, Any]:
        return {
            "id": order.id,
            "status": order.status,
            "exchange_order_id": order.exchange_order_id,
            "rejection_reason": order.rejection_reason,
            "details": order.details,
        }

    @staticmethod
    def _next_commands(user_id: int, provider: str, capital_usd: float) -> dict[str, str]:
        base = f"flask live-rapid-ml-trader --user-id {user_id} --capital-usd {capital_usd:g} --provider {provider}"
        return {
            "preview": base,
            "ml_quality_report": "flask ml-quality-report --horizon 1h --provider both --model-family all --candidate-limit 2 --compact",
            "submit": (
                "RAPID_ML_LIVE_ENABLED=true RAPID_ML_PREVIEW_ONLY=false CANARY_PREVIEW_ONLY=false "
                f"{base} --submit --confirm {CONFIRMATION_PHRASE}"
            ),
            "restore_preview_only": "Set RAPID_ML_PREVIEW_ONLY=true and CANARY_PREVIEW_ONLY=true immediately after a real session.",
        }
