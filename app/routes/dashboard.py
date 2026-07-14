"""Dashboard monitoring routes."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, Response, current_app, jsonify, render_template, request, stream_with_context, url_for

from ..admin_auth import require_admin
from ..auth import current_user, require_authenticated_user
from ..models import StrategyRun, TradingConnection, VaultCycle, WalletTransaction
from ..runtime import get_current_mode, get_service, market_mode_for
from ..services.connection_health import latest_connection_health


dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/admin")


@dashboard_bp.record_once
def _register_user_trading_ops_routes(state) -> None:
    """Expose a read-only authenticated trading ops console for regular users."""

    app = state.app

    def _add(rule: str, endpoint: str, view_func: Any) -> None:
        if endpoint not in app.view_functions:
            app.add_url_rule(rule, endpoint=endpoint, view_func=view_func, methods=["GET"], strict_slashes=False)

    _add("/ops/", "trading_ops_index", user_ops_index)
    _add("/ops/providers/", "trading_ops_providers", user_ops_providers)
    _add("/ops/risk/", "trading_ops_risk", user_ops_risk)
    _add("/ops/activity/", "trading_ops_activity", user_ops_activity)
    _add("/ops/api/status", "trading_ops_status", user_ops_status)

    @app.after_request
    def _user_trading_ops_cache_headers(response: Response) -> Response:
        if request.path.rstrip("/") == "/ops" or request.path.startswith("/ops/api/") or request.path.startswith("/ops/providers") or request.path.startswith("/ops/risk") or request.path.startswith("/ops/activity"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


@dashboard_bp.before_request
def _protect_dashboard():
    return require_admin()


@dashboard_bp.after_request
def _dashboard_cache_headers(response: Response) -> Response:
    if request.path == "/admin/dashboard" or request.path.startswith("/admin/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@dashboard_bp.get("/dashboard")
def index():
    return render_template("advanced/dashboard.html", **_dashboard_shell_payload(refresh_exchange=True))


@dashboard_bp.get("/api/dashboard-data")
def dashboard_data():
    payload = _dashboard_payload()
    page_size = _page_size()
    cursor = _cursor()
    positions = _page_rows(payload["positions"], page_size=page_size, offset=cursor)
    open_orders = _page_rows(payload["open_orders"], page_size=page_size, offset=cursor)
    recent_trades = _page_rows(payload["recent_trades"], page_size=page_size, offset=cursor)
    rankings = _page_rows(payload["strategy_rankings"], page_size=page_size, offset=cursor)

    return jsonify(
        {
            "mode": payload["mode"],
            "balances": payload["balances"],
            "account_synced_at": payload.get("account_synced_at"),
            "account_snapshot": payload.get("account_snapshot", {"status": "unavailable"}),
            "provider_health": payload.get("provider_health", {}),
            "positions": positions["rows"],
            "open_orders": open_orders["rows"],
            "recent_trades": recent_trades["rows"],
            "risk_status": payload["risk_status"],
            "alerts": payload["alerts"],
            "market_summary": payload["market_summary"],
            "paper_equity_curve": payload["paper_equity_curve"],
            "latest_feature_snapshot": payload["latest_feature_snapshot"],
            "pnl": payload["pnl"],
            "strategy_rankings": rankings["rows"],
            "pagination": {
                "limit": page_size,
                "cursor": str(cursor),
                "positions": positions,
                "open_orders": open_orders,
                "recent_trades": recent_trades,
                "strategy_rankings": rankings,
            },
        }
    )


@dashboard_bp.get("/api/performance")
@dashboard_bp.get("/api/dashboard/performance")
def dashboard_performance():
    payload_service = get_service("dashboard_payload")
    strategy_manager = get_service("strategy_manager")
    return jsonify(
        {
            "dashboard_cache": payload_service.get_cache_stats(),
            "opportunity_scanner": get_service("dashboard_opportunities").health_payload(),
            "forecast_performance": get_service("forecast_performance").rolling_metrics(),
            "market_cache": get_service("market_data").cache_stats(),
            "strategy_loop": strategy_manager.get_loop_metrics(),
        }
    )


@dashboard_bp.get("/api/dashboard/opportunities")
def dashboard_opportunities():
    mode = get_current_mode()
    market_mode = market_mode_for(mode)
    service = get_service("dashboard_opportunities")
    return jsonify(
        service.opportunities(
            user=current_user(),
            mode=mode,
            market_mode=market_mode,
            limit=_int_arg("limit", current_app.config.get("DASHBOARD_PAGE_SIZE", 30)),
            cursor=str(request.args.get("cursor", "")),
            offset=_int_arg("offset", 0),
            refresh=_bool_arg("refresh"),
        )
    )


@dashboard_bp.get("/api/dashboard/chart")
def dashboard_chart():
    mode = get_current_mode()
    market_mode = market_mode_for(mode)
    service = get_service("dashboard_opportunities")
    return jsonify(
        service.chart_payload(
            provider=str(request.args.get("provider", "global")),
            symbol=str(request.args.get("symbol", "")),
            venue_symbol=str(request.args.get("venue_symbol", "")),
            timeframe=str(request.args.get("timeframe", "live")),
            market_mode=market_mode,
        )
    )


@dashboard_bp.get("/api/dashboard/activity")
def dashboard_activity():
    payload_service = get_service("dashboard_payload")
    return jsonify(
        payload_service.activity_payload(
            limit=_int_arg("limit", current_app.config.get("DASHBOARD_PAGE_SIZE", 30)),
            cursor=str(request.args.get("cursor", "")),
        )
    )


@dashboard_bp.get("/api/dashboard/stream")
def dashboard_stream():
    mode = get_current_mode()
    market_mode = market_mode_for(mode)
    service = get_service("chart_stream")
    events = service.events(
        user=current_user(),
        mode=mode,
        market_mode=market_mode,
        once=_bool_arg("once"),
        testing=bool(current_app.config.get("TESTING", False)),
    )
    headers = {
        "Cache-Control": "no-store, no-cache, no-transform, must-revalidate, max-age=0",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return Response(stream_with_context(events), mimetype="text/event-stream", headers=headers)


def user_ops_index():
    return _render_user_ops("overview")


def user_ops_providers():
    return _render_user_ops("providers")


def user_ops_risk():
    return _render_user_ops("risk")


def user_ops_activity():
    return _render_user_ops("activity")


def user_ops_status():
    guard = require_authenticated_user()
    if guard is not None:
        return guard
    payload = _user_ops_shell_payload(refresh_exchange=_refresh_exchange_requested(), active_tab="overview")
    return jsonify(
        {
            "ok": True,
            "mode": payload.get("mode"),
            "summary": payload.get("ops_summary", {}),
            "provider_count": len(payload.get("ops_provider_cards", [])),
            "active_cycles": payload.get("ops_summary", {}).get("active_cycles", 0),
            "risk_status": payload.get("risk_status", {}),
            "account_snapshot": payload.get("account_snapshot", {}),
        }
    )


def _render_user_ops(active_tab: str):
    guard = require_authenticated_user()
    if guard is not None:
        return guard
    return render_template(
        "trading_ops.html",
        **_user_ops_shell_payload(refresh_exchange=_refresh_exchange_requested(), active_tab=active_tab),
    )


def _dashboard_payload(refresh_exchange: bool | None = None) -> dict[str, Any]:
    mode = get_current_mode()
    market_mode = market_mode_for(mode)
    dashboard_payload = get_service("dashboard_payload")
    return dashboard_payload.get_payload(
        user=current_user(),
        mode=mode,
        market_mode=market_mode,
        market_data=get_service("market_data"),
        risk_engine=get_service("risk_engine"),
        order_manager=get_service("order_manager"),
        trading_connections=get_service("trading_connections"),
        wallet_summary=get_service("wallet_summary"),
        feature_engine=get_service("feature_engine"),
        registry=get_service("strategy_registry"),
        refresh_exchange=_refresh_exchange_requested() if refresh_exchange is None else bool(refresh_exchange),
    )


def _dashboard_shell_payload(*, refresh_exchange: bool = False) -> dict[str, Any]:
    mode = get_current_mode()
    market_mode = market_mode_for(mode)
    dashboard_payload = get_service("dashboard_payload")
    payload = dashboard_payload.get_shell_payload(
        user=current_user(),
        mode=mode,
        market_mode=market_mode,
        risk_engine=get_service("risk_engine"),
        trading_connections=get_service("trading_connections"),
        wallet_summary=get_service("wallet_summary"),
        refresh_exchange=refresh_exchange,
    )
    payload["refresh_exchange"] = bool(refresh_exchange)
    return payload


def _user_ops_shell_payload(*, refresh_exchange: bool = False, active_tab: str = "overview") -> dict[str, Any]:
    user = current_user()
    payload = _dashboard_shell_payload(refresh_exchange=refresh_exchange)
    provider_cards = _user_provider_cards(user)
    recent_cycles = _user_recent_cycles(user)
    strategy_runs = _user_strategy_runs(user)
    wallet_activity = _user_wallet_activity(user)
    links = {
        "overview": url_for("trading_ops_index"),
        "providers": url_for("trading_ops_providers"),
        "risk": url_for("trading_ops_risk"),
        "activity": url_for("trading_ops_activity"),
        "refresh": url_for("trading_ops_index", refresh_exchange=1),
        "status_api": url_for("trading_ops_status"),
        "vault": url_for("consumer.vault"),
        "settings": url_for("settings.index"),
        "connections": url_for("settings.connections"),
    }
    active_cycles = sum(1 for cycle in recent_cycles if cycle["status"] in {"active", "funding", "trading", "settling"})
    enabled_providers = sum(1 for card in provider_cards if card["enabled"])
    payload.update(
        {
            "ops_active_tab": active_tab if active_tab in {"overview", "providers", "risk", "activity"} else "overview",
            "ops_links": links,
            "ops_tabs": [
                {"key": "overview", "label": "Overview", "href": links["overview"]},
                {"key": "providers", "label": "Providers", "href": links["providers"]},
                {"key": "risk", "label": "Risk", "href": links["risk"]},
                {"key": "activity", "label": "Activity", "href": links["activity"]},
            ],
            "ops_provider_cards": provider_cards,
            "ops_recent_cycles": recent_cycles,
            "ops_strategy_runs": strategy_runs,
            "ops_wallet_activity": wallet_activity,
            "ops_summary": {
                "active_cycles": active_cycles,
                "recent_cycle_count": len(recent_cycles),
                "enabled_providers": enabled_providers,
                "provider_count": len(provider_cards),
                "recent_activity_count": len(wallet_activity),
                "strategy_run_count": len(strategy_runs),
            },
            "ops_title": "Trading Ops",
            "ops_kicker": f"{str(payload.get('mode', 'live')).replace('_', ' ').upper()} · USER OPS",
            "ops_badge": "SERVER GATED",
            "ops_subtitle": "Read-only execution posture, provider readiness, vault activity, and risk state for your account. Any trade, withdrawal, or cycle action still routes through the server-authoritative vault flow.",
        }
    )
    return payload


def _user_provider_cards(user: Any) -> list[dict[str, Any]]:
    if user is None:
        return []
    cards: list[dict[str, Any]] = []
    records = TradingConnection.query.filter_by(user_id=user.id).order_by(TradingConnection.updated_at.desc()).limit(12).all()
    for record in records:
        health = latest_connection_health(record.id) or {}
        status = health.get("status_label") or record.verification_status or "needs_verification"
        cards.append(
            {
                "provider": str(record.provider or "provider").title(),
                "connection_type": str(record.connection_type or "api").replace("_", " ").title(),
                "enabled": bool(record.is_active),
                "verification_status": str(record.verification_status or "needs_verification").replace("_", " ").title(),
                "status": str(status),
                "impact": str(health.get("impact") or record.last_verification_error or "No provider warning recorded."),
                "last_verified_at": _format_datetime(record.last_verified_at),
                "updated_at": _format_datetime(record.updated_at),
            }
        )
    return cards


def _user_recent_cycles(user: Any) -> list[dict[str, Any]]:
    if user is None:
        return []
    cycles = VaultCycle.query.filter_by(user_id=user.id).order_by(VaultCycle.created_at.desc()).limit(10).all()
    rows: list[dict[str, Any]] = []
    for cycle in cycles:
        rows.append(
            {
                "id": cycle.id,
                "public_id": cycle.public_id,
                "status": str(cycle.status or "unknown"),
                "substatus": str(cycle.execution_substatus or ""),
                "deposit_asset": cycle.deposit_asset,
                "settlement_asset": cycle.settlement_asset,
                "deposit_amount": float(cycle.deposit_amount or 0.0),
                "current_value": float(cycle.current_estimated_value_usd or 0.0),
                "final_amount": float(cycle.final_settlement_amount or 0.0) if cycle.final_settlement_amount is not None else None,
                "unlocks_at": _format_datetime(cycle.unlocks_at),
                "started_at": _format_datetime(cycle.started_at),
            }
        )
    return rows


def _user_strategy_runs(user: Any) -> list[dict[str, Any]]:
    if user is None:
        return []
    runs = StrategyRun.query.filter_by(user_id=user.id).order_by(StrategyRun.created_at.desc()).limit(10).all()
    rows: list[dict[str, Any]] = []
    for run in runs:
        signal = run.last_signal if isinstance(run.last_signal, dict) else {}
        rows.append(
            {
                "strategy_name": run.strategy_name,
                "symbol": run.symbol,
                "timeframe": run.timeframe,
                "status": str(run.status or "stopped"),
                "mode": str(run.mode or "live"),
                "last_signal": str(signal.get("action") or signal.get("side") or "n/a"),
                "updated_at": _format_datetime(run.updated_at),
            }
        )
    return rows


def _user_wallet_activity(user: Any) -> list[dict[str, Any]]:
    if user is None:
        return []
    items = WalletTransaction.query.filter_by(user_id=user.id).order_by(WalletTransaction.created_at.desc()).limit(12).all()
    rows: list[dict[str, Any]] = []
    for item in items:
        rows.append(
            {
                "asset": item.asset,
                "amount": float(item.amount or 0.0),
                "type": str(item.transaction_type or "activity").replace("_", " "),
                "status": str(item.status or "recorded").replace("_", " "),
                "network": item.network or "—",
                "note": item.note or "—",
                "created_at": _format_datetime(item.created_at),
            }
        )
    return rows


def _format_datetime(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return value.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:  # noqa: BLE001
        return str(value)


def _refresh_exchange_requested() -> bool:
    return str(request.args.get("refresh_exchange", "")).strip().lower() in {"1", "true", "yes", "on"}


def _bool_arg(name: str) -> bool:
    return str(request.args.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _int_arg(name: str, default: Any) -> int:
    try:
        return int(request.args.get(name, default))
    except (TypeError, ValueError):
        return int(default or 0)


def _page_size() -> int:
    return max(1, min(_int_arg("limit", current_app.config.get("DASHBOARD_PAGE_SIZE", 30)), 150))


def _cursor() -> int:
    return max(0, _int_arg("cursor", _int_arg("offset", 0)))


def _page_rows(rows: list[Any], *, page_size: int, offset: int) -> dict[str, Any]:
    bounded = list(rows or [])[:150]
    page = bounded[offset : offset + page_size]
    next_offset = offset + len(page)
    return {
        "rows": page,
        "count": len(page),
        "total": len(bounded),
        "next_cursor": str(next_offset) if next_offset < len(bounded) else None,
        "has_more": next_offset < len(bounded),
    }
