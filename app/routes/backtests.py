"""Backtesting routes."""

from __future__ import annotations

import json
from typing import Any

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..admin_auth import require_admin
from ..backtesting.engine import BacktestConfig
from ..backtesting.optimizer import AGGRESSIVE_1H_WARNING, DYNAMIC_INTRADAY_WARNING, EXTREME_ROI_WARNING, Profile
from ..extensions import db
from ..models import BacktestRun, MLOfflineModel, MLModelState, OptimizerRun, StrategyRanking, StrategyRun
from ..runtime import get_service


backtests_bp = Blueprint("backtests", __name__, url_prefix="/admin/backtests")


@backtests_bp.before_request
def _protect_backtests():
    return require_admin()


@backtests_bp.get("/", strict_slashes=False)
def index():
    registry = get_service("strategy_registry")
    feature_engine = get_service("feature_engine")
    market_data = get_service("market_data")
    market_universe = get_service("market_universe")
    market_scanner = get_service("market_scanner")

    runs = BacktestRun.query.order_by(BacktestRun.created_at.desc()).limit(20).all()
    optimizer_runs = OptimizerRun.query.order_by(OptimizerRun.created_at.desc()).limit(5).all()
    latest_optimizer = optimizer_runs[0] if optimizer_runs else None

    rankings = []
    aggressive_comparisons = []
    if latest_optimizer is not None:
        rankings = (
            StrategyRanking.query.filter_by(optimizer_run_id=latest_optimizer.id)
            .order_by(StrategyRanking.rejected.asc(), StrategyRanking.score.desc())
            .limit(20)
            .all()
        )
        aggressive_comparisons = _aggressive_comparisons(latest_optimizer.id)

    return render_template(
        "backtests/index.html",
        strategies=registry.definitions(),
        runs=runs,
        latest_result=runs[0].result if runs else None,
        optimizer_runs=optimizer_runs,
        latest_optimizer=latest_optimizer,
        rankings=rankings,
        aggressive_comparisons=aggressive_comparisons,
        universe_candidates=_universe_candidates(market_universe),
        scanner_diagnostics=_scanner_diagnostics(market_scanner),
        high_upside_status=_high_upside_status(),
        latest_fibonacci=_latest_fibonacci(feature_engine, market_data),
        aggressive_enabled=bool(current_app.config.get("AGGRESSIVE_1H_ENABLED", False)),
        extreme_roi_enabled=bool(current_app.config.get("EXTREME_ROI_ENABLED", False)),
        aggressive_warning=AGGRESSIVE_1H_WARNING,
        dynamic_intraday_warning=DYNAMIC_INTRADAY_WARNING,
        extreme_roi_warning=EXTREME_ROI_WARNING,
        ml_ranker_enabled=bool(current_app.config.get("ML_RANKER_ENABLED", False)),
        offline_ml_enabled=bool(current_app.config.get("ML_OFFLINE_MODELS_ENABLED", False)),
        ml_model_states=MLModelState.query.order_by(MLModelState.horizon.asc()).all(),
        offline_ml_models=MLOfflineModel.query.order_by(MLOfflineModel.created_at.desc()).limit(5).all(),
        offline_ml_status=_offline_ml_status(),
    )


@backtests_bp.post("/run")
def run():
    engine = get_service("backtest_engine")

    try:
        parameters = _parse_json_object(request.form.get("parameters_json"), "Backtest parameters")
        parameters.update(_strategy_form_parameters())
        config = _backtest_config(parameters)
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("backtests.index"))

    try:
        result = engine.run(config)
    except Exception as exc:  # noqa: BLE001
        flash(f"Backtest failed: {exc}", "danger")
        return redirect(url_for("backtests.index"))

    record = BacktestRun(
        strategy_name=config.strategy_name,
        symbol=config.symbol,
        timeframe=config.timeframe,
    )
    record.parameters = _backtest_parameters_dict(config)
    record.result = result

    db.session.add(record)
    db.session.commit()

    flash("Backtest completed. Results are candle-based simulation only.", "success")
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

    if not strategy_name:
        raise ValueError("Strategy name is required.")
    if not symbol:
        raise ValueError("Symbol is required.")
    if not timeframe:
        raise ValueError("Timeframe is required.")

    return BacktestConfig(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        mode="testnet",
        initial_balance=_form_float("initial_balance", current_app.config.get("DEFAULT_PAPER_BALANCE", 10_000.0)),
        fee_bps=_form_float("fee_bps", current_app.config.get("FEE_BPS", 5.0)),
        slippage_bps=_form_float("slippage_bps", current_app.config.get("SIM_SLIPPAGE_BPS", 8.0)),
        stop_loss_pct=_form_float("stop_loss_pct", 0.01),
        take_profit_pct=_form_float("take_profit_pct", 0.02),
        position_size_fraction=_form_float("position_size_fraction", 0.1),
        parameters=parameters,
        sizing_mode=str(request.form.get("sizing_mode", "fixed_fraction")).strip(),
        fixed_dollar_size=_form_float("fixed_dollar_size", current_app.config.get("FIXED_DOLLAR_SIZE", 100.0)),
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


def _strategy_form_parameters() -> dict[str, Any]:
    keys = [
        "ema_fast_period",
        "ema_slow_period",
        "rsi_period",
        "rsi_oversold",
        "rsi_overbought",
        "atr_stop_multiplier",
        "atr_take_multiplier",
        "volume_spike_multiplier",
        "minimum_signal_score",
        "trend_weight",
        "rsi_weight",
        "volume_weight",
        "fibonacci_filter_weight",
        "external_weight",
    ]

    parsed: dict[str, Any] = {}

    for key in keys:
        value = request.form.get(key)
        if value in {None, ""}:
            continue

        parsed[key] = _parse_number(value, key)

    return parsed


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


def _auto_deploy_rankings(result: dict[str, Any], limit: int, manager: Any) -> int:
    return 0


def _aggressive_comparisons(optimizer_run_id: int) -> list[dict[str, Any]]:
    rankings = (
        StrategyRanking.query.filter_by(optimizer_run_id=optimizer_run_id, profile=Profile.AGGRESSIVE_1H.value)
        .order_by(StrategyRanking.score.desc())
        .all()
    )
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for ranking in rankings:
        key = (ranking.symbol, ranking.timeframe, ranking.strategy_name, ranking.profile)
        explanation = ranking.ml_explanation if isinstance(ranking.ml_explanation, dict) else {}
        net_roi_v2 = explanation.get("net_roi_v2") if isinstance(explanation.get("net_roi_v2"), dict) else {}
        row = grouped.setdefault(
            key,
            {
                "symbol": ranking.symbol,
                "timeframe": ranking.timeframe,
                "strategy_name": ranking.strategy_name,
                "profile": ranking.profile,
                "best_score": ranking.score,
                "net_roi_v2_score": net_roi_v2.get("net_roi_v2_score", 0.0),
                "roi_quality_grade": net_roi_v2.get("roi_quality_grade", "D"),
                "regime_support": net_roi_v2.get("regime_support", "regime-neutral"),
                "convex_edge_score": ranking.convex_edge_score or 0.0,
                "mfe_mae_ratio": ranking.mfe_mae_ratio or 0.0,
                "capacity_multiple": ranking.capacity_multiple or 0.0,
                "cost_adjusted_recent_1h_return": ranking.cost_adjusted_recent_1h_return or 0.0,
                "decay_penalty": ranking.decay_penalty or 0.0,
                "recent_1h_return": ranking.recent_1h_return or 0.0,
                "edge_score": ranking.edge_score or 0.0,
                "expectancy": ranking.expectancy or 0.0,
                "cost_drag_bps": ranking.cost_drag_bps or 0.0,
                "accepted": 0,
                "rejected": 0,
                "no_trade_reason": ranking.no_trade_reason or ranking.rejection_reason or "",
            },
        )
        row["best_score"] = max(float(row["best_score"] or 0.0), float(ranking.score or 0.0))
        row["net_roi_v2_score"] = max(float(row["net_roi_v2_score"] or 0.0), float(net_roi_v2.get("net_roi_v2_score", 0.0) or 0.0))
        if net_roi_v2.get("roi_quality_grade") in {"A", "B"}:
            row["roi_quality_grade"] = net_roi_v2.get("roi_quality_grade")
        if net_roi_v2.get("regime_support") == "regime-supported":
            row["regime_support"] = "regime-supported"
        row["convex_edge_score"] = max(float(row["convex_edge_score"] or 0.0), float(ranking.convex_edge_score or 0.0))
        row["mfe_mae_ratio"] = max(float(row["mfe_mae_ratio"] or 0.0), float(ranking.mfe_mae_ratio or 0.0))
        row["capacity_multiple"] = max(float(row["capacity_multiple"] or 0.0), float(ranking.capacity_multiple or 0.0))
        row["cost_adjusted_recent_1h_return"] = max(float(row["cost_adjusted_recent_1h_return"] or 0.0), float(ranking.cost_adjusted_recent_1h_return or 0.0))
        row["decay_penalty"] = max(float(row["decay_penalty"] or 0.0), float(ranking.decay_penalty or 0.0))
        row["recent_1h_return"] = max(float(row["recent_1h_return"] or 0.0), float(ranking.recent_1h_return or 0.0))
        row["edge_score"] = max(float(row["edge_score"] or 0.0), float(ranking.edge_score or 0.0))
        row["expectancy"] = max(float(row["expectancy"] or 0.0), float(ranking.expectancy or 0.0))
        row["cost_drag_bps"] = max(float(row["cost_drag_bps"] or 0.0), float(ranking.cost_drag_bps or 0.0))
        if ranking.rejected:
            row["rejected"] += 1
        else:
            row["accepted"] += 1
        if not row["no_trade_reason"]:
            row["no_trade_reason"] = ranking.no_trade_reason or ranking.rejection_reason or ""

    return sorted(grouped.values(), key=lambda item: float(item["best_score"]), reverse=True)[:20]


def _latest_fibonacci(feature_engine: Any, market_data: Any) -> dict[str, Any]:
    symbols = current_app.config.get("ALLOWED_SYMBOLS", ["BTC"])
    symbol = symbols[0] if symbols else "BTC"
    timeframe = current_app.config.get("DEFAULT_TIMEFRAME", "15m")

    try:
        candles = market_data.get_candles(symbol, timeframe, mode="testnet", limit=80)
        return feature_engine.snapshot(symbol=symbol, timeframe=timeframe, candles=candles).fibonacci_levels
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symbol, "timeframe": timeframe, "error": str(exc)}


def _universe_candidates(market_universe: Any) -> list[dict[str, Any]]:
    if not current_app.config.get("DYNAMIC_UNIVERSE_ENABLED", False):
        return []
    try:
        return [
            candidate.as_dict()
            for candidate in market_universe.liquid_universe("testnet", "5m")[:10]
        ]
    except Exception:  # noqa: BLE001
        return []


def _scanner_diagnostics(market_scanner: Any) -> dict[str, Any]:
    if not current_app.config.get("DYNAMIC_UNIVERSE_ENABLED", False):
        return {"accepted": [], "rejected": [], "rejection_breakdown": {}, "rejection_rate": 0.0}
    try:
        symbols = list(current_app.config.get("ALLOWED_SYMBOLS", ["BTC"]))
        market_scanner.score_candidates(
            symbols,
            mode="testnet",
            timeframe="5m",
            duration_seconds=3600,
            strategy_name="scalping",
            optimizer_profile="dynamic_intraday",
        )
        return dict(getattr(market_scanner, "last_scan_diagnostics", {}) or {})
    except Exception as exc:  # noqa: BLE001
        return {"accepted": [], "rejected": [], "rejection_breakdown": {}, "rejection_rate": 0.0, "error": str(exc)}


def _high_upside_status() -> dict[str, Any]:
    risk_status = get_service("risk_engine").status("live")
    return {
        "profile_enabled": bool(current_app.config.get("HIGH_UPSIDE_PROFILE_ENABLED", False)),
        "live_eligible": bool(current_app.config.get("HIGH_UPSIDE_LIVE_ELIGIBLE", False)),
        "auto_disabled": bool((risk_status.get("high_upside") or {}).get("auto_disabled", False)),
        "disabled_reason": (risk_status.get("high_upside") or {}).get("disabled_reason", {}),
        "max_scanner_rejection_rate": float(current_app.config.get("HIGH_UPSIDE_MAX_SCANNER_REJECTION_RATE", 0.65) or 0.65),
    }


def _offline_ml_status() -> dict[str, Any]:
    try:
        readiness = get_service("offline_ranker").readiness("1h")
    except Exception as exc:  # noqa: BLE001
        readiness = {"ready": False, "blockers": [str(exc)], "promoted_model": None}
    return {
        "enabled": bool(current_app.config.get("ML_OFFLINE_MODELS_ENABLED", False)),
        "blend_enabled": bool(current_app.config.get("ML_OFFLINE_BLEND_ENABLED", False)),
        "requires_for_high_upside": bool(current_app.config.get("HIGH_UPSIDE_REQUIRE_PROMOTED_ML", True)),
        "readiness": readiness,
    }


def _symbols() -> list[str]:
    values = request.form.getlist("symbols") or [request.form.get("symbol", "BTC")]
    return [str(symbol).upper().strip() for symbol in values if str(symbol).strip()]


def _timeframes() -> list[str]:
    values = request.form.getlist("timeframes") or [
        request.form.get("timeframe", current_app.config.get("DEFAULT_TIMEFRAME", "15m"))
    ]
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
