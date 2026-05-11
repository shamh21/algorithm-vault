"""Dashboard and strategy control routes."""

from __future__ import annotations

import json
from typing import Any

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, url_for

from ..auth import current_user
from ..admin_auth import require_admin
from ..extensions import db
from ..models import StrategyRun
from ..runtime import available_modes, get_current_mode, get_service, market_mode_for


dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/admin")


@dashboard_bp.before_request
def _protect_dashboard():
    return require_admin()


@dashboard_bp.get("/dashboard")
def index():
    return render_template("advanced/dashboard.html", **_dashboard_payload())


@dashboard_bp.get("/api/dashboard-data")
def dashboard_data():
    payload = _dashboard_payload()

    return jsonify(
        {
            "mode": payload["mode"],
            "balances": payload["balances"],
            "positions": payload["positions"],
            "open_orders": payload["open_orders"],
            "recent_trades": payload["recent_trades"],
            "risk_status": payload["risk_status"],
            "alerts": payload["alerts"],
            "market_summary": payload["market_summary"],
            "paper_equity_curve": payload["paper_equity_curve"],
            "latest_feature_snapshot": payload["latest_feature_snapshot"],
            "pnl": payload["pnl"],
            "strategy_rankings": payload["strategy_rankings"],
        }
    )


@dashboard_bp.get("/api/performance")
def dashboard_performance():
    payload_service = get_service("dashboard_payload")
    strategy_manager = get_service("strategy_manager")
    return jsonify(
        {
            "dashboard_cache": payload_service.get_cache_stats(),
            "market_cache": get_service("market_data").cache_stats(),
            "strategy_loop": strategy_manager.get_loop_metrics(),
        }
    )


@dashboard_bp.post("/strategies/start")
def start_strategy():
    registry = get_service("strategy_registry")
    manager = get_service("strategy_manager")

    strategy_name = str(request.form.get("strategy_name", "")).strip()
    symbol = str(request.form.get("symbol", "BTC")).upper().strip()
    timeframe = str(request.form.get("timeframe", current_app.config.get("DEFAULT_TIMEFRAME", "15m"))).strip()
    parameters_raw = str(request.form.get("parameters_json", "{}")).strip() or "{}"

    if strategy_name not in registry.names():
        flash("Unknown strategy selected.", "danger")
        return redirect(url_for("dashboard.index"))

    if not symbol:
        flash("Symbol is required.", "danger")
        return redirect(url_for("dashboard.index"))

    try:
        parameters = json.loads(parameters_raw)
    except json.JSONDecodeError:
        flash("Parameters must be valid JSON.", "danger")
        return redirect(url_for("dashboard.index"))

    if not isinstance(parameters, dict):
        flash("Strategy parameters must be a JSON object.", "danger")
        return redirect(url_for("dashboard.index"))

    user = current_user()
    connection = get_service("trading_connections").active_tradable_connection(user.id) if user is not None else None
    if get_current_mode() == "live" and connection is None:
        flash("Connect, verify, and activate a live-ready trading account before starting a live strategy.", "danger")
        return redirect(url_for("settings.connections"))

    run = StrategyRun(
        user_id=user.id if user is not None else None,
        trading_connection_id=connection.id if connection is not None else None,
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        mode=get_current_mode(),
        status="starting",
        manual_enabled=True,
    )
    run.parameters = parameters

    db.session.add(run)
    db.session.commit()

    manager.start(run.id)

    flash(f"Started {strategy_name} on {symbol} {timeframe}.", "success")
    return redirect(url_for("dashboard.index"))


@dashboard_bp.post("/strategies/<int:run_id>/stop")
def stop_strategy(run_id: int):
    get_service("strategy_manager").stop(run_id)
    flash("Strategy stopped.", "warning")
    return redirect(url_for("dashboard.index"))


def _dashboard_payload() -> dict[str, Any]:
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
        refresh_exchange=_refresh_exchange_requested(),
    )


def _refresh_exchange_requested() -> bool:
    return str(request.args.get("refresh_exchange", "")).strip().lower() in {"1", "true", "yes", "on"}
