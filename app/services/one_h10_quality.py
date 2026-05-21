"""Shared 1H10 signal-quality gates used before executable decisions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

ONE_H10_HORIZON_SECONDS = 60 * 60

ONE_H10_HARD_BLOCKERS = frozenset(
    {
        "forecast_unavailable",
        "forecast_edge_unavailable",
        "forecast_hold",
        "forecast_zero_sizing",
        "forecast_stale",
        "low_confidence",
        "low_edge_after_costs",
        "edge_below_fee_slippage_buffer",
        "cost_drag_above_threshold",
        "high_slippage",
        "poor_execution_quality",
        "poor_risk_reward",
        "low_liquidity_capacity",
        "low_profitability_score",
        "stale_market_data",
        "stale_candles",
        "missing_candles",
        "insufficient_candles",
        "features_stale",
        "missing_fibonacci_features",
        "ml_not_ready",
        "one_h10_rebalance_forecast_error",
    }
)

ONE_H10_REASON_CODES = {
    "forecast_unavailable": "PROVIDER_DEGRADED",
    "forecast_edge_unavailable": "BELOW_EDGE_THRESHOLD",
    "forecast_hold": "LOW_CONFIDENCE",
    "forecast_zero_sizing": "POOR_RISK_REWARD",
    "forecast_stale": "STALE_MARKET_DATA",
    "low_confidence": "LOW_CONFIDENCE",
    "low_edge_after_costs": "BELOW_EDGE_THRESHOLD",
    "edge_below_fee_slippage_buffer": "BELOW_EDGE_THRESHOLD",
    "cost_drag_above_threshold": "HIGH_SLIPPAGE",
    "high_slippage": "HIGH_SLIPPAGE",
    "poor_execution_quality": "HIGH_SLIPPAGE",
    "poor_risk_reward": "POOR_RISK_REWARD",
    "low_liquidity_capacity": "LOW_LIQUIDITY",
    "low_profitability_score": "BELOW_EDGE_THRESHOLD",
    "stale_market_data": "STALE_MARKET_DATA",
    "stale_candles": "STALE_MARKET_DATA",
    "missing_candles": "STALE_MARKET_DATA",
    "insufficient_candles": "STALE_MARKET_DATA",
    "features_stale": "STALE_MARKET_DATA",
    "missing_fibonacci_features": "STALE_MARKET_DATA",
    "ml_not_ready": "LOW_CONFIDENCE",
    "one_h10_rebalance_forecast_error": "PROVIDER_DEGRADED",
    "risk_engine_blocked": "RISK_ENGINE_BLOCKED",
}


def one_h10_quality_thresholds(config: dict[str, Any] | None) -> dict[str, float]:
    payload = config or {}
    horizon_seconds = max(60.0, _safe_float(payload.get("ONE_H10_HORIZON_SECONDS"), float(ONE_H10_HORIZON_SECONDS)))
    min_edge_default = _safe_float(payload.get("NET_ROI_MIN_EDGE_BPS"), 4.0)
    max_cost_default = _safe_float(payload.get("AGGRESSIVE_1H_MAX_COST_DRAG_BPS"), 18.0)
    min_rr_default = _safe_float(payload.get("MIN_REWARD_RISK"), 1.0)
    return {
        "min_edge_after_cost_bps": max(
            0.0,
            _safe_float(payload.get("ONE_H10_MIN_EDGE_AFTER_COST_BPS"), min_edge_default),
        ),
        "edge_buffer_bps": max(0.0, _safe_float(payload.get("ONE_H10_EDGE_BUFFER_BPS"), 2.0)),
        "max_cost_drag_bps": max(
            1.0,
            _safe_float(payload.get("ONE_H10_MAX_COST_DRAG_BPS"), max_cost_default),
        ),
        "min_risk_reward": max(
            0.0,
            _safe_float(payload.get("ONE_H10_MIN_RISK_REWARD"), min_rr_default),
        ),
        "max_signal_age_seconds": max(
            0.0,
            _safe_float(payload.get("ONE_H10_MAX_SIGNAL_AGE_SECONDS"), horizon_seconds),
        ),
        "min_execution_quality": max(
            0.0,
            min(
                _safe_float(
                    payload.get("ONE_H10_MIN_EXECUTION_QUALITY"),
                    _safe_float(payload.get("ONE_HOUR_MIN_EXECUTION_QUALITY"), 0.60),
                ),
                1.0,
            ),
        ),
        "min_confidence": max(
            0.0,
            min(_safe_float(payload.get("ONE_H10_MIN_FORECAST_CONFIDENCE"), 0.25), 1.0),
        ),
        "max_slippage_bps": max(
            0.0,
            _safe_float(
                payload.get("ONE_H10_MAX_SLIPPAGE_BPS"),
                _safe_float(payload.get("VAULT_MAX_SLIPPAGE_BPS"), 20.0),
            ),
        ),
        "min_profitability_score": max(
            0.0,
            min(_safe_float(payload.get("ONE_H10_MIN_PROFITABILITY_SCORE"), 0.35), 1.0),
        ),
    }


def one_h10_profitability_payload(payload: dict[str, Any] | None, config: dict[str, Any] | None = None) -> dict[str, float]:
    """Return cost-aware profitability and allocation scores for 1H10 candidates."""

    row = payload or {}
    thresholds = one_h10_quality_thresholds(config)
    min_edge = max(thresholds["min_edge_after_cost_bps"], 1.0)
    min_rr = max(thresholds["min_risk_reward"], 1.0)

    net_edge = _safe_float(
        row.get("execution_adjusted_net_return_bps"),
        _safe_float(row.get("net_expected_return_bps"), _safe_float(row.get("expected_return_bps"))),
    )
    raw_net_edge = _safe_float(row.get("net_expected_return_bps"), _safe_float(row.get("expected_return_bps"), net_edge))
    execution_quality = _bounded(
        _safe_float(row.get("expected_execution_quality"), _safe_float(row.get("execution_quality"), 0.0)),
        0.0,
        1.0,
    )
    risk_reward = _safe_float(row.get("risk_reward"), 0.0)
    capital_efficiency = _bounded(
        _safe_float(
            row.get("capital_efficiency_score"),
            _safe_float(row.get("capacity_multiple"), 0.0) / 10.0 if row.get("capacity_multiple") is not None else 1.0,
        ),
        0.0,
        1.0,
    )
    model_agreement = _bounded(
        _safe_float(
            row.get("ml_agreement_score"),
            _safe_float(row.get("ml_consensus_multiplier"), 1.0),
        ),
        0.0,
        1.0,
    )
    target_progress = _bounded(_safe_float(row.get("target_progress"), 0.0), 0.0, 1.0)

    edge_quality = _bounded(max(net_edge, 0.0) / max(min_edge * 8.0, 1.0), 0.0, 1.0)
    raw_edge_quality = _bounded(max(raw_net_edge, 0.0) / max(min_edge * 8.0, 1.0), 0.0, 1.0)
    risk_reward_quality = _bounded(risk_reward / max(min_rr * 3.0, 1.0), 0.0, 1.0)
    profitability_score = _bounded(
        edge_quality * 0.34
        + execution_quality * 0.24
        + risk_reward_quality * 0.16
        + capital_efficiency * 0.12
        + model_agreement * 0.08
        + target_progress * 0.06,
        0.0,
        1.0,
    )
    allocation_weight = _bounded(
        0.45 + edge_quality * 0.35 + execution_quality * 0.12 + risk_reward_quality * 0.08,
        0.0,
        1.0,
    )
    allocation_score = _bounded(profitability_score * allocation_weight * 100.0, 0.0, 100.0)
    return {
        "profitability_score": profitability_score,
        "allocation_score": allocation_score,
        "profitability_edge_quality": edge_quality,
        "profitability_raw_edge_quality": raw_edge_quality,
        "profitability_execution_quality": execution_quality,
        "profitability_risk_reward_quality": risk_reward_quality,
        "profitability_liquidity_quality": capital_efficiency,
        "profitability_model_agreement": model_agreement,
        "profitability_target_progress": target_progress,
        "min_profitability_score": thresholds["min_profitability_score"],
    }


def one_h10_forecast_live_blockers(
    forecast: dict[str, Any] | None,
    config: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> list[str]:
    """Return blockers that should prevent a 1H10 forecast from becoming executable."""

    if not isinstance(forecast, dict) or not forecast:
        return ["forecast_unavailable"]

    thresholds = one_h10_quality_thresholds(config)
    blockers: list[str] = []
    raw_blockers = [str(item) for item in (forecast.get("blockers", []) or []) if str(item)]
    advisory = [str(item) for item in (forecast.get("advisory_blockers", []) or []) if str(item)]
    require_promoted_ml = bool((config or {}).get("ONE_H10_REQUIRE_PROMOTED_ML", True))
    blockers.extend(item for item in raw_blockers if item in ONE_H10_HARD_BLOCKERS)
    blockers.extend(item for item in advisory if item in ONE_H10_HARD_BLOCKERS and (item != "ml_not_ready" or require_promoted_ml))

    action = str(forecast.get("predicted_side") or forecast.get("action") or "hold").strip().lower()
    actionable = action in {"buy", "sell"}
    if not actionable:
        blockers.append("forecast_hold")

    confidence = _optional_float(forecast.get("confidence"))
    if confidence is None or confidence < thresholds["min_confidence"]:
        blockers.append("low_confidence")

    net_edge = _optional_float(forecast.get("net_expected_return_bps"))
    if net_edge is None:
        net_edge = _optional_float(forecast.get("expected_return_bps"))
    if forecast.get("expected_net_edge_passed") is False:
        blockers.append("edge_below_fee_slippage_buffer")
    elif actionable and net_edge is None:
        blockers.append("forecast_edge_unavailable")
    elif net_edge is not None and net_edge < thresholds["min_edge_after_cost_bps"]:
        blockers.append("low_edge_after_costs")

    cost_drag = _optional_float(forecast.get("cost_drag_bps"))
    if cost_drag is not None and cost_drag > thresholds["max_cost_drag_bps"]:
        blockers.append("cost_drag_above_threshold")

    slippage = max(
        _safe_float(forecast.get("estimated_slippage_bps"), -1.0),
        _safe_float(forecast.get("slippage_bps"), -1.0),
        _safe_float(forecast.get("spread_bps"), -1.0),
    )
    if thresholds["max_slippage_bps"] > 0 and slippage > thresholds["max_slippage_bps"]:
        blockers.append("high_slippage")

    execution_quality = _optional_float(forecast.get("execution_quality"))
    if execution_quality is not None and execution_quality < thresholds["min_execution_quality"]:
        blockers.append("poor_execution_quality")

    risk_reward = _forecast_risk_reward(forecast)
    if actionable and thresholds["min_risk_reward"] > 0 and (risk_reward is None or risk_reward < thresholds["min_risk_reward"]):
        blockers.append("poor_risk_reward")

    if bool((config or {}).get("ONE_H10_PROFIT_OPTIMIZER_ENABLED", True)) and actionable:
        profitability_score = _optional_float(forecast.get("profitability_score"))
        if profitability_score is None:
            profitability_score = one_h10_profitability_payload(forecast, config)["profitability_score"]
        if profitability_score < thresholds["min_profitability_score"]:
            blockers.append("low_profitability_score")

    if _safe_float(forecast.get("suggested_notional_usd"), 0.0) <= 0:
        blockers.append("forecast_zero_sizing")

    if _forecast_is_stale(forecast, thresholds["max_signal_age_seconds"], now=now):
        blockers.append("forecast_stale")

    return list(dict.fromkeys(blockers))


def one_h10_reason_code(reason: str | None) -> str:
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return "RISK_ENGINE_BLOCKED"
    if normalized.startswith("one_h10_forecast_blocked:"):
        normalized = normalized.split(":", 1)[1].split(",", 1)[0].strip()
    if ":" in normalized:
        normalized = normalized.split(":", 1)[0].strip()
    return ONE_H10_REASON_CODES.get(normalized, "RISK_ENGINE_BLOCKED")


def first_one_h10_reason_code(blockers: list[str] | tuple[str, ...]) -> str:
    for blocker in blockers:
        code = one_h10_reason_code(str(blocker))
        if code:
            return code
    return "EXECUTABLE"


def _forecast_risk_reward(forecast: dict[str, Any]) -> float | None:
    explicit = _optional_float(forecast.get("risk_reward"))
    if explicit is not None:
        return explicit
    stop_pct = _optional_float(forecast.get("suggested_stop_loss_pct"))
    take_pct = _optional_float(forecast.get("suggested_take_profit_pct"))
    if stop_pct is None or take_pct is None or stop_pct <= 0:
        return None
    return take_pct / stop_pct


def _forecast_is_stale(forecast: dict[str, Any], max_age_seconds: float, *, now: datetime | None = None) -> bool:
    if max_age_seconds <= 0:
        return False
    raw = str(forecast.get("created_at") or forecast.get("updated_at") or "").strip()
    if not raw:
        return False
    try:
        created_at = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return True
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    return (reference.astimezone(UTC) - created_at.astimezone(UTC)).total_seconds() > max_age_seconds


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _safe_float(value: Any, default: float = 0.0) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _bounded(value: float, floor: float, ceiling: float) -> float:
    return max(floor, min(value, ceiling))
