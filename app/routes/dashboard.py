"""Dashboard and strategy control routes."""

from __future__ import annotations

import json
from typing import Any

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, url_for

from ..auth import current_user
from ..admin_auth import require_admin
from ..extensions import db
from ..models import AuditLog, Fill, Order, RiskEvent, ShadowLiveObservation, StrategyRanking, StrategyRun, StrategyValidation
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
            "strategy_rankings": [
                {
                    "strategy_name": ranking.strategy_name,
                    "symbol": ranking.symbol,
                    "timeframe": ranking.timeframe,
                    "score": float(ranking.score or 0.0),
                    "rejected": bool(ranking.rejected),
                }
                for ranking in payload["strategy_rankings"]
            ],
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
    user = current_user()

    client = get_service("hyperliquid_client")
    trading_connections = get_service("trading_connections")
    risk_engine = get_service("risk_engine")
    order_manager = get_service("order_manager")
    registry = get_service("strategy_registry")
    market_data = get_service("market_data")
    feature_engine = get_service("feature_engine")

    alerts: list[str] = []

    snapshot = trading_connections.account_snapshot(user.id if user else None, "live")
    balances = snapshot.balances
    positions = snapshot.positions
    recent_trades = snapshot.recent_fills
    open_orders = snapshot.open_orders
    alerts.extend(snapshot.alerts)
    if user is not None and trading_connections.active_tradable_connection(user.id) is None:
        alerts.append("Connect, verify, and activate a live-ready trading account before execution controls can run.")

    pnl = _pnl(mode, order_manager, positions, recent_trades)

    active_connection = trading_connections.active_tradable_connection(user.id) if user else None

    return {
        "mode": mode,
        "modes": available_modes(),
        "balances": balances,
        "positions": positions,
        "open_orders": open_orders,
        "recent_trades": recent_trades,
        "pnl": pnl,
        "paper_equity_curve": [],
        "risk_status": risk_engine.status(
            mode,
            user_id=user.id if user else None,
            trading_connection_id=active_connection.id if active_connection else None,
        ),
        "strategy_runs": StrategyRun.query.order_by(StrategyRun.created_at.desc()).limit(10).all(),
        "strategy_definitions": registry.definitions(),
        "strategy_rankings": StrategyRanking.query.order_by(
            StrategyRanking.score.desc(),
            StrategyRanking.created_at.desc(),
        ).limit(10).all(),
        "latest_feature_snapshot": _latest_feature_snapshot(feature_engine, market_data, market_mode),
        "external_adapter_status": feature_engine.external_status,
        "pattern_model_status": feature_engine.pattern_status,
        "shadow_observations": ShadowLiveObservation.query.order_by(ShadowLiveObservation.created_at.desc()).limit(10).all(),
        "validations": StrategyValidation.query.order_by(StrategyValidation.started_at.desc()).limit(10).all(),
        "local_orders": Order.query.order_by(Order.created_at.desc()).limit(10).all(),
        "audits": AuditLog.query.order_by(AuditLog.created_at.desc()).limit(10).all(),
        "alerts": alerts,
        "recent_risk": RiskEvent.query.order_by(RiskEvent.created_at.desc()).limit(5).all(),
        "market_summary": _safe_market_summary(market_data, market_mode),
    }


def _pnl(mode: str, order_manager: Any, positions: list[dict[str, Any]], recent_trades: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "realized": sum(float(trade.get("closed_pnl", 0.0) or 0.0) for trade in recent_trades),
        "unrealized": sum(float(position.get("unrealized_pnl", 0.0) or 0.0) for position in positions),
    }


def _safe_market_summary(market_data: Any, market_mode: str) -> list[dict[str, Any]]:
    try:
        return market_data.get_dashboard_market_summary(
            current_app.config.get("ALLOWED_SYMBOLS", ["BTC"]),
            current_app.config.get("DEFAULT_TIMEFRAME", "15m"),
            market_mode,
        )
    except Exception as exc:  # noqa: BLE001
        return [{"symbol": "N/A", "status": "error", "error": str(exc)}]


def _latest_feature_snapshot(feature_engine: Any, market_data: Any, market_mode: str) -> dict[str, Any]:
    symbols = current_app.config.get("ALLOWED_SYMBOLS", ["BTC"])
    symbol = symbols[0] if symbols else "BTC"
    timeframe = current_app.config.get("DEFAULT_TIMEFRAME", "15m")

    try:
        candles = market_data.get_candles(symbol, timeframe, mode=market_mode, limit=80)
        return feature_engine.snapshot(symbol=symbol, timeframe=timeframe, candles=candles).as_dict()
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symbol, "timeframe": timeframe, "error": str(exc)}
