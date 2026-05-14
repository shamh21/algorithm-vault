"""Shared 1H10 signal-quality gates used before executable decisions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

ONE_H10_HORIZON_SECONDS = 70 * 60

ONE_H10_HARD_BLOCKERS = frozenset(
    {
        "forecast_unavailable",
        "forecast_edge_unavailable",
        "forecast_hold",
        "forecast_zero_sizing",
        "forecast_stale",
        "low_confidence",
        "low_edge_after_costs",
        "cost_drag_above_threshold",
        "high_slippage",
        "poor_execution_quality",
        "poor_risk_reward",
        "low_liquidity_capacity",
        "stale_market_data",
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
    "cost_drag_above_threshold": "HIGH_SLIPPAGE",
    "high_slippage": "HIGH_SLIPPAGE",
    "poor_execution_quality": "HIGH_SLIPPAGE",
    "poor_risk_reward": "POOR_RISK_REWARD",
    "low_liquidity_capacity": "LOW_LIQUIDITY",
    "stale_market_data": "STALE_MARKET_DATA",
    "features_stale": "STALE_MARKET_DATA",
    "missing_fibonacci_features": "STALE_MARKET_DATA",
    "ml_not_ready": "LOW_CONFIDENCE",
    "one_h10_rebalance_forecast_error": "PROVIDER_DEGRADED",
    "risk_engine_blocked": "RISK_ENGINE_BLOCKED",
}


def one_h10_quality_thresholds(config: dict[str, Any] | None) -> dict[str, float]:
    payload = config or {}
    min_edge_default = _safe_float(payload.get("NET_ROI_MIN_EDGE_BPS"), 4.0)
    max_cost_default = _safe_float(payload.get("AGGRESSIVE_1H_MAX_COST_DRAG_BPS"), 18.0)
    min_rr_default = _safe_float(payload.get("MIN_REWARD_RISK"), 1.0)
    return {
        "min_edge_after_cost_bps": max(
            0.0,
            _safe_float(payload.get("ONE_H10_MIN_EDGE_AFTER_COST_BPS"), min_edge_default),
        ),
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
            _safe_float(payload.get("ONE_H10_MAX_SIGNAL_AGE_SECONDS"), float(ONE_H10_HORIZON_SECONDS)),
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
        blockers.append("low_edge_after_costs")
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
