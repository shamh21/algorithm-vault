"""Backtesting routes."""

from __future__ import annotations

import json
import math
from typing import Any

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, url_for

from ..admin_auth import require_admin
from ..auth import current_user
from ..backtesting.engine import BacktestConfig
from ..backtesting.optimizer import Profile
from ..extensions import db
from ..models import BacktestRun
from ..runtime import get_service

backtests_bp = Blueprint("backtests", __name__, url_prefix="/admin/backtests")

_TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
}

_CYCLE_DURATIONS = {
    "1h10": {"label": "1H10", "seconds": 70 * 60},
    "4h": {"label": "4H", "seconds": 4 * 60 * 60},
    "24h": {"label": "24H", "seconds": 24 * 60 * 60},
}


@backtests_bp.before_request
def _protect_backtests():
    return require_admin()


@backtests_bp.get("/", strict_slashes=False)
def index():
    simulator = get_service("backtest_vault_simulator")
    latest_run = BacktestRun.query.order_by(BacktestRun.created_at.desc()).first()

    return render_template(
        "backtests/index.html",
        latest_payload=_backtest_response_payload(latest_run) if latest_run else {},
        initial_symbols=simulator.symbol_payload(user=current_user(), limit=40),
        paper_balance_usd=simulator.allocation_cap_usd(),
        allocation_default_usd=simulator.allocation_default_usd(),
        allocation_cap_usd=simulator.allocation_cap_usd(),
        timeframes=simulator.timeframes(),
    )


@backtests_bp.get("/api/symbols")
def symbols_api():
    simulator = get_service("backtest_vault_simulator")
    return jsonify(
        simulator.symbol_payload(
            user=current_user(),
            query=str(request.args.get("q", "")),
            cursor=_arg_int("cursor", 0),
            limit=_arg_int("limit", 40),
            refresh=_arg_bool("refresh"),
        )
    )


@backtests_bp.get("/api/quote")
def quote_api():
    simulator = get_service("backtest_vault_simulator")
    return jsonify(
        simulator.quote_payload(
            provider=str(request.args.get("provider", "")),
            symbol=str(request.args.get("symbol", "")),
            venue_symbol=str(request.args.get("venue_symbol", "")),
            allocation_usd=_arg_float("allocation_usd", simulator.allocation_default_usd()),
        )
    )


@backtests_bp.post("/run")
def run():
    simulator = get_service("backtest_vault_simulator")
    wants_json = _wants_json_response()

    try:
        request_input = simulator.parse_input(request.form, user=current_user())
    except ValueError as exc:
        if wants_json:
            return jsonify({"ok": False, "error": str(exc)}), 400
        flash(str(exc), "danger")
        return redirect(url_for("backtests.index"))

    try:
        simulation = simulator.run(request_input)
    except Exception as exc:  # noqa: BLE001
        message = str(exc)
        if wants_json:
            return jsonify({"ok": False, "error": message}), 500
        flash(message, "danger")
        return redirect(url_for("backtests.index"))

    record = BacktestRun(
        strategy_name=simulation["record"]["strategy_name"],
        symbol=simulation["record"]["symbol"],
        timeframe=simulation["record"]["timeframe"],
    )
    record.parameters = simulation["parameters"]
    record.result = simulation["result"]

    db.session.add(record)
    db.session.commit()

    if wants_json:
        return jsonify(simulator.response_payload(record))

    flash("Vault simulation completed.", "success")
    return redirect(url_for("backtests.index"))


@backtests_bp.post("/optimize")
def optimize():
    optimizer = get_service("strategy_optimizer")
    registry = get_service("strategy_registry")
    manager = get_service("strategy_manager")

    symbols = _symbols()
    timeframes = _timeframes()
    selected_strategy = str(request.form.get("strategy_name", "all")).strip()
    strategy_names = registry.names() if selected_strategy == "all" else [selected_strategy]
    profile = str(request.form.get("profile", "short_term")).strip() or "short_term"

    if profile == Profile.AGGRESSIVE_1H.value and not current_app.config.get("AGGRESSIVE_1H_ENABLED", False):
        flash("Aggressive 1H Experimental optimization is disabled.", "danger")
        return redirect(url_for("backtests.index"))
    if profile == Profile.EXTREME_ROI_EXPERIMENTAL.value and not current_app.config.get("EXTREME_ROI_ENABLED", False):
        flash("Extreme ROI Experimental optimization is disabled.", "danger")
        return redirect(url_for("backtests.index"))

    invalid = [name for name in strategy_names if name not in registry.names()]
    if invalid:
        flash(f"Unknown strategy selected: {', '.join(invalid)}", "danger")
        return redirect(url_for("backtests.index"))

    try:
        config = optimizer.default_config(
            symbols=symbols,
            timeframes=timeframes,
            strategy_names=strategy_names,
            profile=profile,
            allocation_amount_usd=_form_float("allocation_amount_usd", 0.0),
            lock_duration_hours=_form_int("lock_duration_hours", 0),
            universe_mode=str(request.form.get("universe_mode", "configured")).strip() or "configured",
            max_parallel_legs=_form_int("max_parallel_legs", current_app.config.get("VAULT_MAX_PARALLEL_LEGS", 1)),
            allow_leverage_experiment=request.form.get("allow_leverage_experiment") == "on",
        )

        if profile not in {Profile.AGGRESSIVE_1H.value, Profile.EXTREME_ROI_EXPERIMENTAL.value, Profile.DYNAMIC_INTRADAY.value}:
            config.training_window_days = _form_int("training_window_days", config.training_window_days)
            config.testing_window_days = _form_int("testing_window_days", config.testing_window_days)
            config.step_days = _form_int("step_days", config.step_days)
        config.use_full_history = request.form.get("use_full_history") == "on"
        config.decay_factor = _form_float("decay_factor", config.decay_factor)
        config.min_trade_count = _form_int("min_trade_count", config.min_trade_count)
        config.max_parameter_sets = _form_int("max_parameter_sets", config.max_parameter_sets)
        config.auto_deploy_top_n = _form_int("auto_deploy_top_n", config.auto_deploy_top_n)
        config.high_upside_profile = request.form.get("high_upside_profile") == "on"
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("backtests.index"))

    try:
        result = optimizer.run(config)
    except Exception as exc:  # noqa: BLE001
        flash(f"Optimization failed: {exc}", "danger")
        return redirect(url_for("backtests.index"))

    _auto_deploy_rankings(result, config.auto_deploy_top_n, manager)
    flash("Optimization completed. Candidates are available for live-readiness review only.", "success")
    return redirect(url_for("backtests.index"))


def _backtest_config(parameters: dict[str, Any]) -> BacktestConfig:
    strategy_name = str(request.form.get("strategy_name", "ema_crossover")).strip()
    symbol = str(request.form.get("symbol", "BTC")).upper().strip()
    timeframe = str(request.form.get("timeframe", current_app.config.get("DEFAULT_TIMEFRAME", "15m"))).strip()
    cycle_key = _cycle_duration_key()
    cycle_seconds = int(_CYCLE_DURATIONS[cycle_key]["seconds"])
    allocation_amount = _form_float("allocation_amount_usd", _allocation_default_usd())
    paper_balance = _paper_balance_usd()

    if not strategy_name:
        raise ValueError("Strategy name is required.")
    if not symbol:
        raise ValueError("Symbol is required.")
    if not timeframe:
        raise ValueError("Timeframe is required.")
    if strategy_name not in get_service("strategy_registry").names():
        raise ValueError("Unknown strategy selected.")
    if symbol not in current_app.config.get("ALLOWED_SYMBOLS", ["BTC"]):
        raise ValueError("Unknown symbol selected.")
    if timeframe not in _TIMEFRAME_SECONDS:
        raise ValueError("Unsupported timeframe selected.")
    if allocation_amount <= 0:
        raise ValueError("Test allocation amount must be greater than zero.")
    if allocation_amount > paper_balance:
        raise ValueError(f"Test allocation amount cannot exceed ${paper_balance:,.2f} paper funds.")

    parameters = dict(parameters or {})
    parameters.update(
        {
            "sandbox_backtest": True,
            "simulated_capital_only": True,
            "paper_balance_usd": paper_balance,
            "vault_cycle_duration": cycle_key,
            "lock_duration_seconds": cycle_seconds,
            "lock_duration_hours": max(1, math.ceil(cycle_seconds / 3600)),
        }
    )
    return BacktestConfig(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        mode="testnet",
        initial_balance=allocation_amount,
        fee_bps=_form_float("fee_bps", current_app.config.get("FEE_BPS", 5.0)),
        slippage_bps=_form_float("slippage_bps", current_app.config.get("SIM_SLIPPAGE_BPS", 8.0)),
        stop_loss_pct=_form_float("stop_loss_pct", 0.01),
        take_profit_pct=_form_float("take_profit_pct", 0.02),
        position_size_fraction=1.0,
        parameters=parameters,
        sizing_mode=str(request.form.get("sizing_mode", "fixed_fraction")).strip(),
        fixed_dollar_size=allocation_amount,
        risk_per_trade_pct=_form_float("risk_per_trade_pct", current_app.config.get("RISK_PER_TRADE_PCT", 0.01)),
        max_daily_loss=_form_float("max_daily_loss", current_app.config.get("MAX_DAILY_LOSS_USDC", 100.0)),
        max_drawdown_pct=_form_float("max_drawdown_pct", current_app.config.get("MAX_BACKTEST_DRAWDOWN_PCT", 0.2)),
        loss_streak_cooldown=_form_int(
            "loss_streak_cooldown",
            current_app.config.get("LOSS_STREAK_COOLDOWN_THRESHOLD", 3),
        ),
        cooldown_minutes=_form_int("cooldown_minutes", current_app.config.get("LOSS_COOLDOWN_MINUTES", 30)),
        max_trades_per_window=_form_int("max_trades_per_window", current_app.config.get("MAX_TRADES_PER_WINDOW", 5)),
        trade_window_minutes=_form_int("trade_window_minutes", current_app.config.get("TRADE_WINDOW_MINUTES", 60)),
        intrabar_model=str(request.form.get("intrabar_model", "conservative")).strip() or "conservative",
        allocation_amount_usd=_form_float("allocation_amount_usd", 0.0),
        leverage=_form_float("leverage", 1.0),
        min_liquidation_buffer_pct=_form_float(
            "min_liquidation_buffer_pct",
            current_app.config.get("MIN_LIQUIDATION_BUFFER_PCT", 0.015),
        ),
        funding_cost_bps=_form_float("funding_cost_bps", 0.0),
    )


def _backtest_parameters_dict(config: BacktestConfig) -> dict[str, Any]:
    return {
        "initial_balance": config.initial_balance,
        "fee_bps": config.fee_bps,
        "slippage_bps": config.slippage_bps,
        "stop_loss_pct": config.stop_loss_pct,
        "take_profit_pct": config.take_profit_pct,
        "position_size_fraction": config.position_size_fraction,
        "sizing_mode": config.sizing_mode,
        "fixed_dollar_size": config.fixed_dollar_size,
        "risk_per_trade_pct": config.risk_per_trade_pct,
        "max_daily_loss": config.max_daily_loss,
        "max_drawdown_pct": config.max_drawdown_pct,
        "loss_streak_cooldown": config.loss_streak_cooldown,
        "cooldown_minutes": config.cooldown_minutes,
        "max_trades_per_window": config.max_trades_per_window,
        "trade_window_minutes": config.trade_window_minutes,
        "intrabar_model": config.intrabar_model,
        "allocation_amount_usd": config.allocation_amount_usd,
        "leverage": config.leverage,
        "min_liquidation_buffer_pct": config.min_liquidation_buffer_pct,
        "funding_cost_bps": config.funding_cost_bps,
        "parameters": config.parameters,
    }


def _backtest_candles(config: BacktestConfig) -> list[dict[str, Any]]:
    timeframe_seconds = _TIMEFRAME_SECONDS.get(config.timeframe, 15 * 60)
    duration_seconds = int((config.parameters or {}).get("lock_duration_seconds") or _CYCLE_DURATIONS["1h10"]["seconds"])
    warmup_candles = 30
    candle_limit = max(30, warmup_candles + math.ceil(duration_seconds / timeframe_seconds))
    return get_service("market_data").get_candles(config.symbol, config.timeframe, mode="testnet", limit=candle_limit)


def _backtest_response_payload(run: BacktestRun) -> dict[str, Any]:
    result = run.result if isinstance(run.result, dict) else {}
    if result.get("vault_simulation"):
        return get_service("backtest_vault_simulator").response_payload(run)
    params = run.parameters if isinstance(run.parameters, dict) else {}
    nested = params.get("parameters") if isinstance(params.get("parameters"), dict) else {}
    duration_key = str(nested.get("vault_cycle_duration") or "1h10")
    duration = _CYCLE_DURATIONS.get(duration_key, _CYCLE_DURATIONS["1h10"])
    initial_balance = _safe_float(params.get("initial_balance"), _safe_float(result.get("allocation_amount_usd"), 0.0))
    final_equity = _safe_float(result.get("final_equity"), initial_balance)
    pnl = final_equity - initial_balance
    trades = result.get("trades") if isinstance(result.get("trades"), list) else []
    wins = len([trade for trade in trades if _safe_float(trade.get("pnl")) > 0])
    losses = len([trade for trade in trades if _safe_float(trade.get("pnl")) < 0])
    flat = max(0, len(trades) - wins - losses)
    payload = {
        "ok": True,
        "run_id": run.id,
        "summary": {
            "strategy": run.strategy_name,
            "symbol": run.symbol,
            "timeframe": run.timeframe,
            "duration": duration["label"],
            "allocation": initial_balance,
            "leverage": _safe_float(params.get("leverage"), 1.0),
            "paper_balance": _safe_float(nested.get("paper_balance_usd"), _paper_balance_usd()),
        },
        "metrics": {
            "roi": _safe_float(result.get("total_return")),
            "pnl": pnl,
            "win_rate": _safe_float(result.get("win_rate")),
            "max_drawdown": _safe_float(result.get("max_drawdown")),
            "trades": int(result.get("trade_count") or 0),
            "fees": _safe_float(result.get("fees_paid")),
            "ending_balance": final_equity,
        },
        "charts": {
            "equity": _series_from_equity(result, "equity", initial_balance),
            "pnl": _pnl_series(result, initial_balance),
            "drawdown": _drawdown_series(result),
            "growth": _growth_series(result, initial_balance),
            "win_loss": {"wins": wins, "losses": losses, "flat": flat},
            "trade_distribution": _trade_distribution(trades),
        },
        "result": _json_safe(result),
    }
    return _json_safe(payload)


def _series_from_equity(result: dict[str, Any], key: str, default_value: float = 0.0) -> list[dict[str, float]]:
    points = result.get("equity_curve") if isinstance(result.get("equity_curve"), list) else []
    series = [
        {"x": _safe_float(point.get("timestamp")), "y": _safe_float(point.get(key), default_value)}
        for point in points
        if isinstance(point, dict)
    ]
    return _downsample_series(series)


def _pnl_series(result: dict[str, Any], initial_balance: float) -> list[dict[str, float]]:
    points = result.get("equity_curve") if isinstance(result.get("equity_curve"), list) else []
    series = [
        {"x": _safe_float(point.get("timestamp")), "y": _safe_float(point.get("equity"), initial_balance) - initial_balance}
        for point in points
        if isinstance(point, dict)
    ]
    return _downsample_series(series)


def _growth_series(result: dict[str, Any], initial_balance: float) -> list[dict[str, float]]:
    points = result.get("equity_curve") if isinstance(result.get("equity_curve"), list) else []
    base = max(initial_balance, 1e-9)
    series = [
        {"x": _safe_float(point.get("timestamp")), "y": (_safe_float(point.get("equity"), initial_balance) - initial_balance) / base}
        for point in points
        if isinstance(point, dict)
    ]
    return _downsample_series(series)


def _drawdown_series(result: dict[str, Any]) -> list[dict[str, float]]:
    points = result.get("drawdown_curve") if isinstance(result.get("drawdown_curve"), list) else []
    series = [
        {"x": _safe_float(point.get("timestamp")), "y": _safe_float(point.get("drawdown"))} for point in points if isinstance(point, dict)
    ]
    return _downsample_series(series)


def _trade_distribution(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = [
        {"label": "< -2%", "min": -math.inf, "max": -0.02, "count": 0},
        {"label": "-2% to 0", "min": -0.02, "max": 0.0, "count": 0},
        {"label": "0 to 2%", "min": 0.0, "max": 0.02, "count": 0},
        {"label": "> 2%", "min": 0.02, "max": math.inf, "count": 0},
    ]
    for trade in trades:
        value = _safe_float(trade.get("return"))
        for bucket in buckets:
            if bucket["min"] <= value < bucket["max"]:
                bucket["count"] += 1
                break
    return [{"label": bucket["label"], "count": bucket["count"]} for bucket in buckets]


def _downsample_series(series: list[dict[str, float]]) -> list[dict[str, float]]:
    max_points = max(12, int(current_app.config.get("BACKTEST_MAX_CHART_POINTS", 240) or 240))
    if len(series) <= max_points:
        return series
    step = math.ceil(len(series) / max_points)
    sampled = series[::step]
    if sampled[-1] != series[-1]:
        sampled.append(series[-1])
    return sampled


def _cycle_duration_key() -> str:
    raw = str(request.form.get("cycle_duration", "1h10")).strip().lower()
    return raw if raw in _CYCLE_DURATIONS else "1h10"


def _paper_balance_usd() -> float:
    return float(current_app.config.get("BACKTEST_PAPER_BALANCE_USD", 10_000.0) or 10_000.0)


def _allocation_default_usd() -> float:
    default = float(current_app.config.get("BACKTEST_ALLOCATION_DEFAULT_USD", 10_000.0) or 10_000.0)
    return min(max(default, 0.0), _paper_balance_usd())


def _wants_json_response() -> bool:
    if str(request.args.get("response", "")).strip().lower() == "json":
        return True
    return bool(request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _auto_deploy_rankings(result: dict[str, Any], limit: int, manager: Any) -> int:
    return 0


def _symbols() -> list[str]:
    values = request.form.getlist("symbols") or [request.form.get("symbol", "BTC")]
    return [str(symbol).upper().strip() for symbol in values if str(symbol).strip()]


def _timeframes() -> list[str]:
    values = request.form.getlist("timeframes") or [request.form.get("timeframe", current_app.config.get("DEFAULT_TIMEFRAME", "15m"))]
    return [str(timeframe).strip() for timeframe in values if str(timeframe).strip()]


def _parse_json_object(raw: str | None, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(raw or "{}").strip() or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")

    return parsed


def _form_float(key: str, default: Any) -> float:
    return float(_parse_number(request.form.get(key, default), key))


def _form_int(key: str, default: Any) -> int:
    return int(_parse_number(request.form.get(key, default), key))


def _parse_number(value: Any, key: str) -> float | int:
    try:
        text = str(value).strip()
        return float(text) if "." in text else int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric.") from exc


def _arg_int(key: str, default: int) -> int:
    try:
        return int(str(request.args.get(key, default)).strip())
    except (TypeError, ValueError):
        return int(default)


def _arg_float(key: str, default: float) -> float:
    try:
        return float(str(request.args.get(key, default)).strip())
    except (TypeError, ValueError):
        return float(default)


def _arg_bool(key: str) -> bool:
    return str(request.args.get(key, "")).strip().lower() in {"1", "true", "yes", "on"}
