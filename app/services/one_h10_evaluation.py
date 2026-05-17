"""Minimal 1H10 evaluation report using existing backtest utilities."""

from __future__ import annotations

from typing import Any

from ..backtesting.engine import BacktestConfig, BacktestEngine
from .one_h10_quality import ONE_H10_HORIZON_SECONDS, one_h10_quality_thresholds

ONE_H10_EVALUATION_METRICS = (
    "total_return",
    "net_return_after_costs",
    "sharpe_like",
    "sortino_like",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "average_return_per_trade",
    "avg_loss",
    "trade_count",
    "fees_paid",
    "funding_cost_estimate",
    "cost_drag_bps",
)


def build_one_h10_evaluation_report(
    config: dict[str, Any],
    engine: BacktestEngine,
    *,
    symbol: str,
    timeframe: str,
    strategy_name: str = "scalping",
    candles: list[dict[str, Any]] | None = None,
    initial_balance: float | None = None,
) -> dict[str, Any]:
    """Run a cost-aware 1H10 backtest summary without fetching unsupported data."""

    resolved_symbol = str(symbol or "BTC").upper()
    resolved_timeframe = str(timeframe or config.get("DEFAULT_TIMEFRAME") or "1m")
    horizon_seconds = max(60, int(float(config.get("ONE_H10_HORIZON_SECONDS", ONE_H10_HORIZON_SECONDS) or ONE_H10_HORIZON_SECONDS)))
    eval_fraction = float(config.get("ONE_H10_EVAL_POSITION_FRACTION", 0.08) or 0.08)
    baseline_fraction = float(config.get("ONE_H10_EVAL_BASELINE_POSITION_FRACTION", max(eval_fraction * 0.5, 0.01)) or max(eval_fraction * 0.5, 0.01))
    base_parameters = {
        "one_h10_vault": True,
        "algorithm_profile": "1H10",
        "vault_cycle_duration": "1h10",
        "lock_duration_seconds": horizon_seconds,
        "lock_duration_hours": horizon_seconds / 3600.0,
        "ml_horizon": "1h10",
        "objective": "one_h10",
    }
    baseline_parameters = {
        **base_parameters,
        "one_h10_profit_optimizer_enabled": False,
        "optimizer_variant": "baseline",
    }
    optimized_parameters = {
        **base_parameters,
        "one_h10_profit_optimizer_enabled": bool(config.get("ONE_H10_PROFIT_OPTIMIZER_ENABLED", True)),
        "optimizer_variant": "net_expectancy",
        "min_profitability_score": float(config.get("ONE_H10_MIN_PROFITABILITY_SCORE", 0.35) or 0.35),
        "max_position_fraction": float(config.get("ONE_H10_MAX_POSITION_FRACTION", 0.75) or 0.75),
    }
    baseline_config = _backtest_config(
        config,
        symbol=resolved_symbol,
        timeframe=resolved_timeframe,
        strategy_name=strategy_name,
        parameters=baseline_parameters,
        position_size_fraction=baseline_fraction,
        initial_balance=initial_balance,
    )
    optimized_config = _backtest_config(
        config,
        symbol=resolved_symbol,
        timeframe=resolved_timeframe,
        strategy_name=strategy_name,
        parameters=optimized_parameters,
        position_size_fraction=eval_fraction,
        initial_balance=initial_balance,
    )
    baseline_result = engine.run(baseline_config, candles=candles)
    optimized_result = engine.run(optimized_config, candles=candles)
    baseline_metrics = _metric_summary(baseline_result)
    optimized_metrics = _metric_summary(optimized_result)
    min_trades = int(config.get("ONE_H10_EVAL_MIN_TRADES", 10) or 10)
    warnings = _evaluation_warnings(baseline_metrics, optimized_metrics, min_trades)
    return {
        "horizon": "1h10",
        "horizon_seconds": horizon_seconds,
        "symbol": resolved_symbol,
        "timeframe": resolved_timeframe,
        "strategy_name": strategy_name,
        "config": {
            **one_h10_quality_thresholds(config),
            "fee_bps": float(config.get("FEE_BPS", 5.0) or 5.0),
            "slippage_bps": float(config.get("SIM_SLIPPAGE_BPS", 8.0) or 8.0),
            "min_trades": min_trades,
            "baseline_position_fraction": baseline_fraction,
            "optimized_position_fraction": eval_fraction,
        },
        "data_period": _data_period(candles, optimized_result or baseline_result),
        "baseline": dict(baseline_metrics),
        "optimized": dict(optimized_metrics),
        "difference": _metric_difference(baseline_metrics, optimized_metrics),
        "warnings": list(dict.fromkeys(warnings)),
    }


def _backtest_config(
    config: dict[str, Any],
    *,
    symbol: str,
    timeframe: str,
    strategy_name: str,
    parameters: dict[str, Any],
    position_size_fraction: float,
    initial_balance: float | None,
) -> BacktestConfig:
    return BacktestConfig(
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        mode=str(config.get("ONE_H10_EVAL_MODE") or "testnet"),
        initial_balance=float(initial_balance or config.get("DEFAULT_PAPER_BALANCE", 1000.0) or 1000.0),
        fee_bps=float(config.get("FEE_BPS", 5.0) or 5.0),
        slippage_bps=float(config.get("SIM_SLIPPAGE_BPS", 8.0) or 8.0),
        stop_loss_pct=float(config.get("ONE_H10_EVAL_STOP_LOSS_PCT", 0.01) or 0.01),
        take_profit_pct=float(config.get("ONE_H10_EVAL_TAKE_PROFIT_PCT", 0.02) or 0.02),
        position_size_fraction=float(position_size_fraction or config.get("ONE_H10_EVAL_POSITION_FRACTION", 0.08) or 0.08),
        parameters=parameters,
        sizing_mode="risk_based",
        risk_per_trade_pct=float(config.get("RISK_PER_TRADE_PCT", 0.01) or 0.01),
        max_daily_loss=float(config.get("MAX_DAILY_LOSS_USDC", 0.0) or 0.0),
        max_drawdown_pct=float(config.get("MAX_BACKTEST_DRAWDOWN_PCT", 0.0) or 0.0),
        loss_streak_cooldown=int(config.get("LOSS_STREAK_COOLDOWN_THRESHOLD", 0) or 0),
        cooldown_minutes=int(config.get("LOSS_COOLDOWN_MINUTES", 0) or 0),
        max_trades_per_window=int(config.get("MAX_TRADES_PER_WINDOW", 0) or 0),
        trade_window_minutes=int(config.get("TRADE_WINDOW_MINUTES", 60) or 60),
        intrabar_model="conservative",
        allocation_amount_usd=float(config.get("ONE_H10_EVAL_ALLOCATION_USD", 0.0) or 0.0),
        leverage=float(config.get("ONE_H10_MAX_LEVERAGE", 1.0) or 1.0),
        funding_cost_bps=float(config.get("FUNDING_COST_BPS", 0.0) or 0.0),
        funding_interval_hours=float(config.get("FUNDING_INTERVAL_HOURS", 8.0) or 8.0),
    )


def _metric_summary(result: dict[str, Any]) -> dict[str, Any]:
    trades = list(result.get("trades", []) or [])
    long_count = sum(1 for trade in trades if str(trade.get("direction") or "").lower() == "long")
    short_count = sum(1 for trade in trades if str(trade.get("direction") or "").lower() == "short")
    durations = [float(trade.get("duration_minutes", 0.0) or 0.0) for trade in trades]
    return {
        **{key: result.get(key, 0.0) for key in ONE_H10_EVALUATION_METRICS},
        "average_loss": result.get("avg_loss", 0.0),
        "number_of_trades": result.get("trade_count", len(trades)),
        "long_trade_count": long_count,
        "short_trade_count": short_count,
        "hold_no_trade_frequency": result.get("no_trade_reason", "") or "",
        "exposure_minutes": sum(durations),
        "average_trade_duration_minutes": result.get("average_trade_duration_minutes", 0.0),
    }


def _metric_difference(baseline: dict[str, Any], optimized: dict[str, Any]) -> dict[str, float]:
    keys = set(baseline) | set(optimized)
    return {
        key: _safe_float(optimized.get(key)) - _safe_float(baseline.get(key))
        for key in sorted(keys)
        if isinstance(optimized.get(key, baseline.get(key)), (int, float)) or isinstance(baseline.get(key), (int, float))
    }


def _evaluation_warnings(baseline: dict[str, Any], optimized: dict[str, Any], min_trades: int) -> list[str]:
    warnings: list[str] = []
    optimized_trades = int(_safe_float(optimized.get("trade_count"), _safe_float(optimized.get("number_of_trades"))))
    if optimized_trades <= 0:
        warnings.append("no_trades")
    if optimized_trades < min_trades:
        warnings.append("sample_size_below_threshold")
    if _safe_float(optimized.get("net_return_after_costs")) < 0:
        warnings.append("optimized_net_return_negative")
    if _safe_float(optimized.get("profit_factor")) < _safe_float(baseline.get("profit_factor")):
        warnings.append("optimized_profit_factor_below_baseline")
    if _safe_float(optimized.get("max_drawdown")) < _safe_float(baseline.get("max_drawdown")):
        warnings.append("optimized_drawdown_worse_than_baseline")
    return warnings


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _data_period(candles: list[dict[str, Any]] | None, result: dict[str, Any]) -> dict[str, Any]:
    if not candles:
        equity_curve = list(result.get("equity_curve", []) or [])
        if not equity_curve:
            return {"start": None, "end": None, "candle_count": 0, "source": "none"}
        return {
            "start": equity_curve[0].get("timestamp"),
            "end": equity_curve[-1].get("timestamp"),
            "candle_count": None,
            "source": "backtest_equity_curve",
        }
    return {
        "start": candles[0].get("timestamp"),
        "end": candles[-1].get("timestamp"),
        "candle_count": len(candles),
        "source": "provided_candles",
    }
