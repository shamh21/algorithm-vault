"""Dashboard response caching and segment orchestration."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from flask import Flask, current_app

from ..models import AuditLog, Order, RiskEvent, ShadowLiveObservation, StrategyRanking, StrategyRun, StrategyValidation


@dataclass
class _CachedValue:
    expires_at: float
    stale_until: float
    value: dict[str, Any]


@dataclass
class _InflightFetch:
    event: threading.Event
    value: dict[str, Any] | None = None
    error: Exception | None = None


class DashboardPayloadService:
    """Caches dashboard fragments with short TTL and stale-while-revalidate behavior."""

    def __init__(self, app: Flask, config: dict[str, Any]) -> None:
        self.app = app
        self.config = config
        self._cache: dict[tuple[Any, ...], _CachedValue] = {}
        self._inflight: dict[tuple[Any, ...], _InflightFetch] = {}
        self._lock = threading.Lock()
        self.metrics = {
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "stale_serves": 0,
            "refresh_failures": 0,
            "segment_ms_sum": {},
            "last_assembly_ms": 0.0,
        }

    def get_shell_payload(
        self,
        *,
        user: Any,
        mode: str,
        market_mode: str,
        risk_engine: Any,
        trading_connections: Any,
        wallet_summary: Any,
    ) -> dict[str, Any]:
        """Return the lightweight authenticated dashboard shell payload."""

        self.metrics["requests"] += 1
        started_at = time.perf_counter()
        account_payload = self._get_cached_segment(
            self._account_segment_key(user, mode, False),
            "DASHBOARD_ACCOUNT_SEGMENT_TTL_SECONDS",
            "DASHBOARD_ACCOUNT_SEGMENT_STALE_SECONDS",
            lambda: self._cached_account_payload(user, trading_connections, wallet_summary, False),
            "dashboard-account",
        )
        active_connection = trading_connections.active_tradable_connection(user.id) if user else None
        risk_status = risk_engine.status(
            mode,
            user_id=user.id if user else None,
            trading_connection_id=active_connection.id if active_connection else None,
        )
        positions = self._limit_rows(account_payload.get("positions"), 30)
        recent_trades = self._limit_rows(account_payload.get("recent_trades"), 30)
        open_orders = self._limit_rows(account_payload.get("open_orders"), 30)
        strategy_rankings = (
            StrategyRanking.query.order_by(
                StrategyRanking.score.desc(),
                StrategyRanking.created_at.desc(),
            )
            .limit(6)
            .all()
        )
        strategy_runs = StrategyRun.query.order_by(StrategyRun.created_at.desc()).limit(6).all()
        payload = {
            "mode": mode,
            "modes": ["live", "paper", "shadow_live", "paper_shadow"],
            "balances": self._limit_rows(account_payload.get("balances"), 30),
            "account_synced_at": account_payload.get("synced_at"),
            "positions": positions,
            "open_orders": open_orders,
            "recent_trades": recent_trades,
            "pnl": _pnl(mode, None, positions, recent_trades),
            "paper_equity_curve": [],
            "risk_status": risk_status,
            "strategy_runs": [self._serialize_strategy_run(run) for run in strategy_runs],
            "strategy_definitions": [],
            "strategy_rankings": [self._serialize_ranking(row) for row in strategy_rankings],
            "latest_feature_snapshot": {},
            "external_adapter_status": {},
            "pattern_model_status": {},
            "shadow_observations": [],
            "validations": [],
            "local_orders": [],
            "audits": [],
            "alerts": self._limit_rows(account_payload.get("alerts"), 3),
            "account_snapshot": dict(account_payload.get("account_snapshot") or {"status": "unavailable"}),
            "recent_risk": [],
            "market_summary": [],
            "shell": True,
        }
        self.metrics["last_assembly_ms"] = (time.perf_counter() - started_at) * 1000
        return payload

    def get_payload(
        self,
        *,
        user: Any,
        mode: str,
        market_mode: str,
        market_data: Any,
        risk_engine: Any,
        order_manager: Any,
        trading_connections: Any,
        wallet_summary: Any,
        feature_engine: Any,
        registry: Any,
        refresh_exchange: bool,
    ) -> dict[str, Any]:
        self.metrics["requests"] += 1
        started_at = time.perf_counter()

        market_mode = str(market_mode or "testnet")
        account_payload = self._get_cached_segment(
            self._account_segment_key(user, mode, refresh_exchange),
            "DASHBOARD_ACCOUNT_SEGMENT_TTL_SECONDS",
            "DASHBOARD_ACCOUNT_SEGMENT_STALE_SECONDS",
            lambda: self._cached_account_payload(user, trading_connections, wallet_summary, refresh_exchange),
            "dashboard-account",
        )
        trade_payload = self._get_cached_segment(
            self._trade_segment_key(user, mode, refresh_exchange),
            "DASHBOARD_TRADE_LIST_SEGMENT_TTL_SECONDS",
            "DASHBOARD_TRADE_LIST_STALE_SECONDS",
            lambda: self._cached_trade_payload(account_payload),
            "dashboard-trades",
        )
        static_payload = self._get_cached_segment(
            self._static_segment_key(user, mode),
            "DASHBOARD_STATIC_SEGMENT_TTL_SECONDS",
            "DASHBOARD_STATIC_SEGMENT_STALE_SECONDS",
            lambda: self._cached_static_payload(mode, market_mode, registry, order_manager, market_data, feature_engine),
            "dashboard-static",
        )

        active_connection = trading_connections.active_tradable_connection(user.id) if user else None
        risk_status = risk_engine.status(
            mode,
            user_id=user.id if user else None,
            trading_connection_id=active_connection.id if active_connection else None,
        )
        pnl = _pnl(mode, order_manager, account_payload["positions"], account_payload["recent_trades"])

        payload = {
            "mode": mode,
            "modes": static_payload["modes"],
            "balances": self._limit_rows(account_payload["balances"], 150),
            "account_synced_at": account_payload.get("synced_at"),
            "positions": self._limit_rows(trade_payload["positions"], 150),
            "open_orders": self._limit_rows(trade_payload["open_orders"], 150),
            "recent_trades": self._limit_rows(trade_payload["recent_trades"], 150),
            "pnl": pnl,
            "paper_equity_curve": static_payload["paper_equity_curve"],
            "risk_status": risk_status,
            "strategy_runs": static_payload["strategy_runs"],
            "strategy_definitions": static_payload["strategy_definitions"],
            "strategy_rankings": static_payload["strategy_rankings"],
            "latest_feature_snapshot": static_payload["latest_feature_snapshot"],
            "external_adapter_status": static_payload["external_adapter_status"],
            "pattern_model_status": static_payload["pattern_model_status"],
            "shadow_observations": static_payload["shadow_observations"],
            "validations": static_payload["validations"],
            "local_orders": static_payload["local_orders"],
            "audits": static_payload["audits"],
            "alerts": account_payload["alerts"],
            "account_snapshot": dict(account_payload.get("account_snapshot") or {"status": "unavailable"}),
            "recent_risk": static_payload["recent_risk"],
            "market_summary": static_payload["market_summary"],
        }
        assembly_ms = (time.perf_counter() - started_at) * 1000
        self.metrics["last_assembly_ms"] = assembly_ms
        current_app.logger.debug(
            "Dashboard payload assembled mode=%s user=%s duration_ms=%.2f segments=%s",
            mode,
            getattr(user, "id", "anonymous"),
            assembly_ms,
            ",".join(("account", "trades", "static")),
        )
        return payload

    def activity_payload(self, *, limit: int = 30, cursor: str | None = None) -> dict[str, Any]:
        """Return a bounded mixed dashboard activity feed."""

        page_size = max(1, min(int(limit or 30), 150))
        try:
            offset = max(0, int(cursor or 0))
        except (TypeError, ValueError):
            offset = 0
        scan_limit = min(150, offset + page_size + 1)
        items: list[dict[str, Any]] = []
        audits = AuditLog.query.order_by(AuditLog.created_at.desc()).limit(scan_limit).all()
        orders = Order.query.order_by(Order.created_at.desc()).limit(scan_limit).all()
        risk_events = RiskEvent.query.order_by(RiskEvent.created_at.desc()).limit(scan_limit).all()
        for item in audits:
            items.append(self._serialize_activity_audit(item))
        for item in orders:
            items.append(self._serialize_activity_order(item))
        for item in risk_events:
            items.append(self._serialize_activity_risk(item))
        items.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
        items = items[:150]
        page = items[offset : offset + page_size]
        next_offset = offset + len(page)
        return {
            "items": page,
            "count": len(page),
            "next_cursor": str(next_offset) if next_offset < len(items) else None,
            "has_more": next_offset < len(items),
            "updated_at": time.time(),
        }

    def get_cache_stats(self) -> dict[str, Any]:
        with self._lock:
            total = max(1, self.metrics["hits"] + self.metrics["misses"])
            return {
                "entries": len(self._cache),
                "requests": self.metrics["requests"],
                "hits": self.metrics["hits"],
                "misses": self.metrics["misses"],
                "stale_serves": self.metrics["stale_serves"],
                "refresh_failures": self.metrics["refresh_failures"],
                "hit_rate": self.metrics["hits"] / total,
                "last_assembly_ms": self.metrics["last_assembly_ms"],
            }

    def _get_cached_segment(
        self,
        key: tuple[Any, ...],
        ttl_config_key: str,
        stale_config_key: str,
        builder: Any,
        segment_name: str,
    ) -> dict[str, Any]:
        ttl = self._config_float(ttl_config_key, 2.0)
        stale_ttl = max(ttl, self._config_float(stale_config_key, ttl or 2.0))
        now = time.time()
        start = time.perf_counter()

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and now < cached.expires_at:
                self.metrics["hits"] += 1
                return dict(cached.value)

            if cached is not None and now < cached.stale_until:
                self.metrics["stale_serves"] += 1
                self._ensure_refresh(key, builder, ttl, stale_ttl)
                self.metrics["hits"] += 1
                return dict(cached.value)

            inflight = self._inflight.get(key)
            if inflight is not None:
                event = inflight.event
            else:
                inflight = _InflightFetch(event=threading.Event())
                self._inflight[key] = inflight
                event = None

            if event is not None:
                self.metrics["misses"] += 1
            else:
                self.metrics["misses"] += 1

        if event is not None:
            event.wait()
            with self._lock:
                updated = self._cache.get(key)
                error = inflight.error
            if error is not None:
                raise RuntimeError(f"dashboard segment refresh failed for {segment_name}: {error}") from error
            if updated is not None:
                return dict(updated.value)
            raise RuntimeError(f"dashboard segment unavailable: {segment_name}")

        try:
            value = builder()
            if not isinstance(value, dict):
                raise TypeError("Dashboard segment builder must return dict payload")
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                entry = self._inflight.get(key)
                if entry is not None:
                    entry.error = exc
                    entry.event.set()
                    self._inflight.pop(key, None)
                    self.metrics["refresh_failures"] += 1
            raise

        with self._lock:
            now = time.time()
            self._cache[key] = _CachedValue(
                expires_at=now + ttl,
                stale_until=now + stale_ttl,
                value=dict(value),
            )
            inflight = self._inflight.pop(key, None)
            if inflight is not None:
                inflight.value = dict(value)
                inflight.error = None
                inflight.event.set()

        ms = (time.perf_counter() - start) * 1000
        self.metrics["segment_ms_sum"][segment_name] = ms
        return dict(value)

    def _ensure_refresh(
        self,
        key: tuple[Any, ...],
        builder: Any,
        ttl: float,
        stale_ttl: float,
    ) -> None:
        with self._lock:
            inflight = self._inflight.get(key)
            if inflight is not None:
                return
            self._inflight[key] = _InflightFetch(event=threading.Event())

        def _refresh() -> None:
            try:
                value = builder()
                if not isinstance(value, dict):
                    raise TypeError("Dashboard segment builder must return dict payload")
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    entry = self._inflight.get(key)
                    if entry is not None:
                        entry.error = exc
                        entry.event.set()
                        self._inflight.pop(key, None)
                        self.metrics["refresh_failures"] += 1
                return
            with self._lock:
                now = time.time()
                self._cache[key] = _CachedValue(
                    expires_at=now + ttl,
                    stale_until=now + stale_ttl,
                    value=dict(value),
                )
                entry = self._inflight.get(key)
                if entry is not None:
                    entry.value = dict(value)
                    entry.error = None
                    entry.event.set()
                    self._inflight.pop(key, None)

        threading.Thread(target=lambda: self._run_with_app_ctx(_refresh), daemon=True).start()

    def _run_with_app_ctx(self, fn: Any) -> None:
        try:
            with self.app.app_context():
                fn()
        except Exception:
            current_app.logger.exception("Dashboard async segment refresh failed")

    def _cached_account_payload(
        self,
        user: Any,
        trading_connections: Any,
        wallet_summary: Any,
        refresh_exchange: bool,
    ) -> dict[str, Any]:
        if user is None:
            return {
                "balances": [],
                "positions": [],
                "recent_trades": [],
                "open_orders": [],
                "alerts": [],
                "synced_at": None,
                "account_snapshot": {"status": "unavailable"},
            }

        alerts: list[str] = []
        active_connection = trading_connections.active_tradable_connection(user.id)
        active_provider = getattr(active_connection, "provider", None) if active_connection is not None else None
        if refresh_exchange:
            try:
                snapshot = trading_connections.account_snapshot(user.id, "live")
                refreshed = wallet_summary.refresh_exchange_snapshot(user, trading_connections, mode="live", snapshot=snapshot)
                synced_at = refreshed.get("synced_at") if isinstance(refreshed, dict) else None
                synced_at = synced_at or datetime.utcnow().isoformat() + "Z"
                return {
                    "balances": snapshot.balances,
                    "positions": snapshot.positions,
                    "recent_trades": snapshot.recent_fills,
                    "open_orders": snapshot.open_orders,
                    "alerts": [str(item) for item in snapshot.alerts],
                    "synced_at": synced_at,
                    "account_snapshot": {
                        "status": "live" if active_provider else "unavailable",
                        "provider": active_provider,
                        "synced_at": synced_at,
                    },
                }
            except Exception as exc:  # noqa: BLE001
                alerts.append(f"Live exchange snapshot refresh failed: {exc}")

        cached = wallet_summary.cached_exchange_snapshot(user.id)
        if not cached:
            alerts.append("Account data is cached to reduce exchange requests. Use refresh for a read-only live update.")
            return {
                "balances": [],
                "positions": [],
                "recent_trades": [],
                "open_orders": [],
                "alerts": alerts,
                "synced_at": None,
                "account_snapshot": {"status": "unavailable"},
            }

        if int(cached.get("positions_count", 0) or 0) > 0:
            alerts.append(f"Cached exchange snapshot reports {int(cached.get('positions_count', 0) or 0)} open position(s).")
        if int(cached.get("open_orders_count", 0) or 0) > 0:
            alerts.append(f"Cached exchange snapshot reports {int(cached.get('open_orders_count', 0) or 0)} open order(s).")
        alerts.extend([str(item) for item in (cached.get("alerts") or [])])

        return {
            "balances": self._limit_rows(cached.get("balances"), 150),
            "synced_at": cached.get("synced_at"),
            "positions": self._limit_rows(cached.get("positions"), 150),
            "recent_trades": self._limit_rows(cached.get("recent_fills"), 150),
            "open_orders": self._limit_rows(cached.get("open_orders"), 150),
            "alerts": self._limit_rows(alerts, 150),
            "account_snapshot": {
                "status": "cached",
                "provider": cached.get("provider"),
                "synced_at": cached.get("synced_at"),
            },
        }

    def _cached_trade_payload(self, account_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "positions": list(account_payload.get("positions") or []),
            "recent_trades": list(account_payload.get("recent_trades") or []),
            "open_orders": list(account_payload.get("open_orders") or []),
        }

    def _cached_static_payload(
        self,
        mode: str,
        market_mode: str,
        registry: Any,
        order_manager: Any,
        market_data: Any,
        feature_engine: Any,
    ) -> dict[str, Any]:
        strategy_runs = StrategyRun.query.order_by(StrategyRun.created_at.desc()).limit(10).all()
        strategy_rankings = (
            StrategyRanking.query.order_by(
                StrategyRanking.score.desc(),
                StrategyRanking.created_at.desc(),
            )
            .limit(10)
            .all()
        )
        shadow_observations = ShadowLiveObservation.query.order_by(ShadowLiveObservation.created_at.desc()).limit(10).all()
        validations = StrategyValidation.query.order_by(StrategyValidation.started_at.desc()).limit(10).all()
        local_orders = Order.query.order_by(Order.created_at.desc()).limit(10).all()
        audits = AuditLog.query.order_by(AuditLog.created_at.desc()).limit(10).all()
        recent_risk = RiskEvent.query.order_by(RiskEvent.created_at.desc()).limit(5).all()

        return {
            "strategy_runs": [self._serialize_strategy_run(run) for run in strategy_runs],
            "strategy_definitions": self._serialize_strategy_definitions(registry.definitions()),
            "strategy_rankings": [self._serialize_ranking(row) for row in strategy_rankings],
            "shadow_observations": [self._serialize_shadow_observation(item) for item in shadow_observations],
            "validations": [self._serialize_validation(item) for item in validations],
            "local_orders": [self._serialize_order(item) for item in local_orders],
            "audits": [self._serialize_audit(item) for item in audits],
            "recent_risk": [self._serialize_risk_event(item) for item in recent_risk],
            "market_summary": self._safe_market_summary(market_data, market_mode, self.config),
            "latest_feature_snapshot": self._latest_feature_snapshot(feature_engine, market_data, market_mode, self.config),
            "external_adapter_status": feature_engine.external_status,
            "pattern_model_status": feature_engine.pattern_status,
            "paper_equity_curve": [],
            "modes": ["live", "paper", "shadow_live", "paper_shadow"],
        }

    @staticmethod
    def _serialize_strategy_run(run: StrategyRun) -> dict[str, Any]:
        return {
            "id": run.id,
            "strategy_name": run.strategy_name,
            "symbol": run.symbol,
            "timeframe": run.timeframe,
            "status": run.status,
            "last_signal": dict(run.last_signal or {}),
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "mode": run.mode,
            "manual_enabled": bool(run.manual_enabled),
        }

    @staticmethod
    def _serialize_strategy_definitions(definitions: list[Any]) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for item in definitions:
            if isinstance(item, str):
                payload.append({"name": item})
                continue
            name = getattr(item, "name", None)
            if name is None and isinstance(item, dict):
                name = item.get("name")
            if not name:
                continue
            payload.append({"name": str(name)})
        payload.sort(key=lambda row: row.get("name", ""))
        return payload

    @staticmethod
    def _serialize_ranking(row: StrategyRanking) -> dict[str, Any]:
        return {
            "id": row.id,
            "strategy_name": row.strategy_name,
            "symbol": row.symbol,
            "timeframe": row.timeframe,
            "score": float(row.score or 0.0),
            "recent_performance_score": float(row.recent_performance_score or 0.0),
            "max_drawdown": float(row.max_drawdown or 0.0),
            "rejected": bool(row.rejected),
        }

    @staticmethod
    def _serialize_shadow_observation(item: ShadowLiveObservation) -> dict[str, Any]:
        return {
            "id": item.id,
            "strategy_name": item.strategy_name,
            "signal_action": item.signal_action,
            "live_mid": float(item.live_mid or 0.0),
            "observed_spread_bps": float(item.observed_spread_bps or 0.0),
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "status": item.status,
            "symbol": item.symbol,
            "timeframe": item.timeframe,
        }

    @staticmethod
    def _serialize_validation(item: StrategyValidation) -> dict[str, Any]:
        return {
            "id": item.id,
            "strategy_name": item.strategy_name,
            "symbol": item.symbol,
            "timeframe": item.timeframe,
            "stage": item.stage,
            "status": item.status,
            "metrics": dict(item.metrics or {}),
            "started_at": item.started_at.isoformat() if item.started_at else None,
        }

    @staticmethod
    def _serialize_order(item: Order) -> dict[str, Any]:
        return {
            "id": item.id,
            "symbol": item.symbol,
            "side": item.side,
            "order_type": item.order_type,
            "status": item.status,
            "quantity": float(item.quantity or 0.0),
            "price": float(item.limit_price or 0.0),
            "size": float(item.quantity or 0.0),
            "reduce_only": bool(item.reduce_only),
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }

    @staticmethod
    def _serialize_audit(item: AuditLog) -> dict[str, Any]:
        details = dict(item.details or {})
        return {
            "id": item.id,
            "category": item.category,
            "action": item.action,
            "message": item.message,
            "created_at": item.created_at.isoformat() if item.created_at else None,
            "rule_name": details.get("rule_name"),
            "reason": details.get("reason"),
        }

    @staticmethod
    def _serialize_risk_event(item: RiskEvent) -> dict[str, Any]:
        payload = dict(item.payload or {})
        return {
            "id": item.id,
            "rule_name": item.rule_name,
            "reason": item.reason,
            "severity": payload.get("severity", "warning"),
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }

    @staticmethod
    def _serialize_activity_audit(item: AuditLog) -> dict[str, Any]:
        return {
            "id": f"audit:{item.id}",
            "kind": "audit",
            "title": str(item.action or "Audit event").replace("_", " ").title(),
            "detail": item.message,
            "severity": "info",
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }

    @staticmethod
    def _serialize_activity_order(item: Order) -> dict[str, Any]:
        return {
            "id": f"order:{item.id}",
            "kind": "order",
            "title": f"{str(item.side or '').upper()} {item.symbol}",
            "detail": f"{item.status} {float(item.quantity or 0.0):.4f} @ {float(item.limit_price or 0.0):.4f}",
            "severity": "trade",
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }

    @staticmethod
    def _serialize_activity_risk(item: RiskEvent) -> dict[str, Any]:
        payload = dict(item.payload or {})
        return {
            "id": f"risk:{item.id}",
            "kind": "risk",
            "title": str(item.rule_name or "Risk event").replace("_", " ").title(),
            "detail": item.reason,
            "severity": str(payload.get("severity", "warning") or "warning"),
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }

    @staticmethod
    def _limit_rows(rows: Any, limit: int) -> list[Any]:
        if rows is None:
            return []
        if isinstance(rows, list):
            return rows[: max(0, int(limit or 0))]
        try:
            return list(rows)[: max(0, int(limit or 0))]
        except TypeError:
            return []

    @staticmethod
    def _safe_market_summary(market_data: Any, market_mode: str, config: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            return market_data.get_dashboard_market_summary(
                config.get("ALLOWED_SYMBOLS", ["BTC"]),
                config.get("DEFAULT_TIMEFRAME", "15m"),
                market_mode,
            )
        except Exception as exc:  # noqa: BLE001
            return [{"symbol": "N/A", "status": "error", "error": str(exc)}]

    @staticmethod
    def _latest_feature_snapshot(feature_engine: Any, market_data: Any, market_mode: str, config: dict[str, Any]) -> dict[str, Any]:
        symbols = config.get("ALLOWED_SYMBOLS", ["BTC"])
        symbol = symbols[0] if symbols else "BTC"
        timeframe = config.get("DEFAULT_TIMEFRAME", "15m")

        try:
            candles = market_data.get_candles(symbol, timeframe, mode=market_mode, limit=80)
            return feature_engine.snapshot(symbol=symbol, timeframe=timeframe, candles=candles).as_dict()
        except Exception as exc:  # noqa: BLE001
            return {"symbol": symbol, "timeframe": timeframe, "error": str(exc)}

    @staticmethod
    def _account_segment_key(user: Any, mode: str, refresh_exchange: bool) -> tuple[Any, ...]:
        return ("account", int(user.id) if user is not None else 0, str(mode), bool(refresh_exchange))

    @staticmethod
    def _trade_segment_key(user: Any, mode: str, refresh_exchange: bool) -> tuple[Any, ...]:
        return ("trade", int(user.id) if user is not None else 0, str(mode), bool(refresh_exchange))

    @staticmethod
    def _static_segment_key(user: Any, mode: str) -> tuple[Any, ...]:
        return ("static", int(user.id) if user is not None else 0, str(mode))

    def _config_float(self, key: str, default: float) -> float:
        try:
            return max(0.0, float(self.config.get(key, default) or 0.0))
        except (TypeError, ValueError):
            return max(0.0, default)


def _pnl(mode: str, order_manager: Any, positions: list[dict[str, Any]], recent_trades: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "realized": sum(float(trade.get("closed_pnl", 0.0) or 0.0) for trade in recent_trades),
        "unrealized": sum(float(position.get("unrealized_pnl", 0.0) or 0.0) for position in positions),
    }
