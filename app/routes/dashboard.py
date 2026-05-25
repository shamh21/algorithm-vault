"""Dashboard monitoring routes."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, Response, current_app, jsonify, render_template, request, stream_with_context

from ..admin_auth import require_admin
from ..auth import current_user
from ..runtime import get_current_mode, get_service, market_mode_for

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/admin")


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
            "market_data_deferred": payload.get("market_data_deferred", False),
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


@dashboard_bp.get("/api/dashboard/market")
def dashboard_market():
    mode = get_current_mode()
    market_mode = market_mode_for(mode)
    payload_service = get_service("dashboard_payload")
    return jsonify(
        payload_service.get_market_payload(
            mode=mode,
            market_mode=market_mode,
            market_data=get_service("market_data"),
            feature_engine=get_service("feature_engine"),
        )
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
