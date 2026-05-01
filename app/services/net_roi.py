"""Shared cost-adjusted ROI scoring helpers for research and live gates."""

from __future__ import annotations

import math
from typing import Any, Mapping


def net_roi_diagnostics(payload: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return additive diagnostics for cost-adjusted candidate quality.

    The score is deliberately unit-light: it is a ranking signal, not a
    promised return. Inputs may come from optimizer rankings, scanner rows, or
    runtime signal snapshots, so all fields are optional and fail neutral.
    """

    config = config or {}
    net_return = _float(_first(payload, "net_return_after_costs", "total_return"))
    recent = max(
        _float(_first(payload, "recent_1h_return", "recent_performance_score")),
        _float(payload.get("cost_adjusted_recent_1h_return")),
        _float(payload.get("cost_adjusted_expected_move")) / 10_000.0,
    )
    expectancy = _float(payload.get("expectancy"))
    edge_bps = _float(_first(payload, "edge_score", "expected_move_bps"))
    cost_drag_bps = _float(payload.get("cost_drag_bps"))
    spread_bps = _float(payload.get("spread_bps"))
    slippage_bps = _float(_first(payload, "projected_slippage_bps", "estimated_slippage_bps"))
    liquidity_usd = max(_float(_first(payload, "liquidity_usd", "realtime_liquidity_usd")), 0.0)
    liquidity_capacity = max(_float(_first(payload, "liquidity_capacity_usd", "liquidity_capacity", "capacity_usd")), 0.0)
    allocation = max(_float(_first(payload, "allocation_amount_usd", "allocation_cap_usd")), 0.0)
    capacity_multiple = _float(payload.get("capacity_multiple"))
    if capacity_multiple <= 0 and allocation > 0 and liquidity_capacity > 0:
        capacity_multiple = liquidity_capacity / allocation
    if capacity_multiple <= 0 and allocation > 0 and liquidity_usd > 0:
        capacity_multiple = liquidity_usd * 0.05 / allocation

    favorable = max(_float(payload.get("max_favorable_excursion")), 0.0)
    adverse = abs(min(_float(payload.get("max_adverse_excursion")), 0.0))
    mfe_mae = _float(payload.get("mfe_mae_ratio"))
    if mfe_mae <= 0 and favorable > 0 and adverse > 0:
        mfe_mae = favorable / max(adverse, 1e-9)

    stability = max(
        _float(payload.get("window_stability")),
        _float(payload.get("accepted_window_ratio")),
        _float(payload.get("signal_stability")),
    )
    market_structure = _float(_first(payload, "market_structure_score", "market_structure_trend"))
    ml_signal = _float(payload.get("ml_score")) + _float(payload.get("offline_ml_prediction"))
    drawdown = _float(payload.get("max_drawdown"))
    turnover = max(_float(_first(payload, "turnover_after_fees", "turnover_rate")), 0.0)
    trades_per_day = max(_float(payload.get("trades_per_day")), 0.0)
    avg_trade_return = _float(payload.get("avg_trade_return"))
    decay_penalty = max(_float(payload.get("decay_penalty")), 0.0) + max(-_float(_first(payload, "recent_decay", "performance_decay_rate")), 0.0)
    stale_age = max(_float(payload.get("stale_data_age_seconds")), 0.0)
    stale = bool(payload.get("stale_data", False)) or stale_age > 3600.0
    volatility_regime = str(payload.get("volatility_regime", "") or "").lower()

    fill_quality = expected_fill_quality(
        spread_bps=spread_bps,
        cost_drag_bps=cost_drag_bps,
        slippage_bps=slippage_bps,
        liquidity_usd=liquidity_usd or liquidity_capacity,
        capacity_multiple=capacity_multiple,
        volatility_regime=volatility_regime,
        stale_data=stale,
    )
    churn = churn_penalty(
        turnover_after_fees=turnover,
        trades_per_day=trades_per_day,
        avg_trade_return=avg_trade_return,
        cost_drag_bps=cost_drag_bps,
    )
    edge_after_cost_bps = edge_bps - cost_drag_bps
    low_fill_penalty = max(0.0, _float(config.get("NET_ROI_MIN_FILL_QUALITY"), 0.55) - fill_quality)

    score = (
        net_return * 240.0
        + recent * 150.0
        + max(expectancy, 0.0) * 0.75
        + max(edge_after_cost_bps, 0.0) * 0.18
        + min(max(mfe_mae, 0.0), 8.0) * 0.65
        + min(max(capacity_multiple, 0.0), 6.0) * 0.85
        + max(stability, 0.0) * 4.0
        + max(market_structure, 0.0) * 5.0
        + max(ml_signal, 0.0) * 0.35
        + fill_quality * 6.0
        + drawdown * 20.0
        - max(cost_drag_bps - _float(config.get("NET_ROI_MIN_EDGE_BPS"), 4.0), 0.0) * 0.18
        - max(spread_bps - 12.0, 0.0) * 0.08
        - churn * 10.0
        - decay_penalty * 30.0
        - low_fill_penalty * 12.0
        - (2.0 if stale else 0.0)
    )

    return {
        "net_roi_score": float(score if math.isfinite(score) else 0.0),
        "expected_fill_quality": float(fill_quality),
        "churn_penalty": float(churn),
        "edge_after_cost_bps": float(edge_after_cost_bps),
        "capacity_multiple": float(capacity_multiple),
        "data_age_seconds": float(stale_age),
        "stale_data": stale,
        "net_roi_components": {
            "net_return": net_return,
            "recent": recent,
            "expectancy": expectancy,
            "mfe_mae_ratio": mfe_mae,
            "stability": stability,
            "market_structure": market_structure,
            "ml_signal": ml_signal,
            "drawdown": drawdown,
            "cost_drag_bps": cost_drag_bps,
            "spread_bps": spread_bps,
            "fill_quality": fill_quality,
            "churn_penalty": churn,
            "decay_penalty": decay_penalty,
        },
    }


def net_roi_v2_diagnostics(payload: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return second-generation research ROI diagnostics.

    V2 keeps the original score intact and adds regime-aware quality signals
    for research ranking. It intentionally does not imply live eligibility or
    alter any execution gate by itself.
    """

    config = config or {}
    base = net_roi_diagnostics(payload, config)
    components = base["net_roi_components"]
    net_return = _float(_first(payload, "net_return_after_costs", "total_return"))
    recent = max(
        _float(_first(payload, "recent_1h_return", "recent_performance_score")),
        _float(payload.get("cost_adjusted_recent_1h_return")),
        _float(payload.get("cost_adjusted_expected_move")) / 10_000.0,
    )
    expectancy = _float(payload.get("expectancy"))
    stability = max(
        _float(payload.get("window_stability")),
        _float(payload.get("accepted_window_ratio")),
        _float(payload.get("signal_stability")),
    )
    favorable = max(_float(payload.get("max_favorable_excursion")), 0.0)
    adverse = abs(min(_float(payload.get("max_adverse_excursion")), 0.0))
    mfe_mae = _float(payload.get("mfe_mae_ratio"))
    if mfe_mae <= 0 and favorable > 0 and adverse > 0:
        mfe_mae = favorable / max(adverse, 1e-9)
    drawdown_abs = abs(min(_float(_first(payload, "max_drawdown", "drawdown")), 0.0))
    avg_win = max(_float(payload.get("avg_win")), 0.0)
    avg_loss_abs = abs(min(_float(payload.get("avg_loss")), 0.0)) or max(-_float(payload.get("avg_loss")), 0.0)
    churn = _float(base.get("churn_penalty"))
    edge_after_cost_bps = _float(base.get("edge_after_cost_bps"))
    cost_adjusted_expected_move = _float(payload.get("cost_adjusted_expected_move"))
    breakout_proximity_bps = _float(payload.get("breakout_proximity_bps"), 250.0)
    breakout_proximity_score = max(0.0, 1.0 - max(breakout_proximity_bps, 0.0) / 250.0)
    cost_adjusted_breakout_potential = max(edge_after_cost_bps, cost_adjusted_expected_move, 0.0) * (0.45 + 0.55 * breakout_proximity_score)
    regime_bucket = _regime_bucket(payload, base)
    regime_adjustment = _regime_adjustment(regime_bucket)
    regime_support = _regime_support_label(regime_adjustment)
    regime_adjusted_expectancy = expectancy * max(0.35, 1.0 + regime_adjustment / 12.0)
    tail_loss_penalty = _tail_loss_penalty(
        drawdown_abs=drawdown_abs,
        adverse=adverse,
        volatility_regime=str(regime_bucket.get("volatility", "unknown")),
    )
    downside_asymmetry_penalty = _downside_asymmetry_penalty(
        mfe_mae=mfe_mae,
        avg_win=avg_win,
        avg_loss_abs=avg_loss_abs,
        drawdown_abs=drawdown_abs,
    )

    score = (
        _float(base.get("net_roi_score")) * 0.68
        + net_return * 220.0
        + recent * 120.0
        + max(regime_adjusted_expectancy, 0.0) * 0.65
        + min(max(mfe_mae, 0.0), 8.0) * 0.9
        + max(stability, 0.0) * 7.0
        + cost_adjusted_breakout_potential * 0.18
        + regime_adjustment * 2.4
        - tail_loss_penalty * 18.0
        - downside_asymmetry_penalty * 14.0
        - churn * 8.0
    )
    if not bool(config.get("NET_ROI_V2_ENABLED", True)):
        score = _float(base.get("net_roi_score"))

    score = float(score if math.isfinite(score) else 0.0)
    rejection_risk = _roi_rejection_risk(
        base=base,
        score=score,
        regime_support=regime_support,
        tail_loss_penalty=tail_loss_penalty,
        downside_asymmetry_penalty=downside_asymmetry_penalty,
        config=config,
    )
    grade = _roi_quality_grade(score, rejection_risk, _float(base.get("expected_fill_quality")), stability)
    return {
        "net_roi_v2_score": score,
        "roi_quality_grade": grade,
        "roi_rejection_risk": rejection_risk,
        "regime_bucket": regime_bucket,
        "regime_support": regime_support,
        "regime_adjustment": float(regime_adjustment),
        "regime_adjusted_expectancy": float(regime_adjusted_expectancy),
        "tail_loss_penalty": float(tail_loss_penalty),
        "downside_asymmetry_penalty": float(downside_asymmetry_penalty),
        "cost_adjusted_breakout_potential": float(cost_adjusted_breakout_potential),
        "net_roi_v2_components": {
            **dict(components),
            "regime_adjustment": float(regime_adjustment),
            "regime_adjusted_expectancy": float(regime_adjusted_expectancy),
            "tail_loss_penalty": float(tail_loss_penalty),
            "downside_asymmetry_penalty": float(downside_asymmetry_penalty),
            "cost_adjusted_breakout_potential": float(cost_adjusted_breakout_potential),
            "regime_bucket": dict(regime_bucket),
            "regime_support": regime_support,
        },
    }


def one_hour_edge_v2_diagnostics(payload: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return shared 1-hour profitability diagnostics for research ranking.

    This helper intentionally produces additive metadata only. It is suitable
    for scanner rows, optimizer rankings, runtime signal snapshots, and vault
    selection because all inputs are optional and missing values fail toward a
    conservative grade.
    """

    config = config or {}
    base = net_roi_diagnostics(payload, config)
    v2 = net_roi_v2_diagnostics(payload, config)

    raw_total_return_pct = _float(payload.get("raw_total_return_pct"))
    if raw_total_return_pct == 0.0:
        raw_total_return_pct = _float(_first(payload, "total_return", "gross_return")) * 100.0
    raw_net_return_pct = _float(payload.get("raw_net_return_pct"))
    if raw_net_return_pct == 0.0:
        raw_net_return_pct = _float(_first(payload, "net_return_after_costs", "total_return")) * 100.0
    raw_vs_net_gap = max(raw_total_return_pct - raw_net_return_pct, 0.0)

    raw_upside_score = max(_float(payload.get("raw_upside_score")), 0.0)
    recent = max(
        _float(_first(payload, "recent_1h_return", "recent_performance_score")),
        _float(payload.get("cost_adjusted_recent_1h_return")),
    )
    fill_quality = _float(base.get("expected_fill_quality"))
    churn = _float(base.get("churn_penalty"))
    edge_after_cost = _float(base.get("edge_after_cost_bps"))
    cost_drag_bps = _float(payload.get("cost_drag_bps"))
    spread_bps = _float(payload.get("spread_bps"))
    stability = max(
        _float(payload.get("window_stability")),
        _float(payload.get("accepted_window_ratio")),
        _float(payload.get("signal_stability")),
    )
    capacity_multiple = _float(base.get("capacity_multiple"))
    market_structure = _float(_first(payload, "market_structure_score", "market_structure_trend"))
    favorable = max(_float(payload.get("max_favorable_excursion")), 0.0)
    adverse = abs(min(_float(payload.get("max_adverse_excursion")), 0.0))
    mfe_mae = _float(payload.get("mfe_mae_ratio"))
    if mfe_mae <= 0 and favorable > 0 and adverse > 0:
        mfe_mae = favorable / max(adverse, 1e-9)
    drawdown_abs = abs(min(_float(_first(payload, "max_drawdown", "drawdown")), 0.0))
    decay = max(_float(payload.get("decay_penalty")), 0.0) + max(-_float(_first(payload, "recent_decay", "performance_decay_rate")), 0.0)
    stale = bool(base.get("stale_data")) or bool(payload.get("stale_data"))

    min_fill = _float(config.get("ONE_HOUR_MIN_EXECUTION_QUALITY"), 0.60)
    max_gap = _float(config.get("ONE_HOUR_MAX_RAW_NET_GAP_PCT"), 35.0)
    min_edge = _float(config.get("NET_ROI_MIN_EDGE_BPS"), 4.0)
    max_churn = _float(config.get("NET_ROI_MAX_CHURN_PENALTY"), 0.35)
    max_cost = _float(config.get("AGGRESSIVE_1H_MAX_COST_DRAG_BPS"), 18.0)
    min_stability = _float(config.get("AGGRESSIVE_1H_MIN_WINDOW_STABILITY"), 0.55)
    min_mfe_mae = _float(config.get("AGGRESSIVE_1H_MIN_MFE_MAE"), 1.5)
    min_capacity = _float(config.get("AGGRESSIVE_1H_MIN_CAPACITY_MULTIPLE"), 2.0)

    cost_quality = max(0.0, 1.0 - max(cost_drag_bps - 4.0, 0.0) / max(max_cost * 2.0, 1.0))
    spread_quality = max(0.0, 1.0 - max(spread_bps, 0.0) / 30.0)
    capacity_quality = min(max(capacity_multiple, 0.0) / max(min_capacity * 3.0, 1.0), 1.0)
    execution_quality = (
        fill_quality * 0.45
        + max(stability, 0.0) * 0.18
        + capacity_quality * 0.14
        + max(market_structure, 0.0) * 0.10
        + cost_quality * 0.08
        + spread_quality * 0.05
    )
    if stale:
        execution_quality -= 0.25
    expected_execution_quality = max(0.0, min(execution_quality, 1.0))

    blockers: list[str] = []
    if edge_after_cost < min_edge:
        blockers.append("low_edge_after_costs")
    if expected_execution_quality < min_fill:
        blockers.append("low_expected_execution_quality")
    if raw_vs_net_gap > max_gap:
        blockers.append("high_raw_vs_net_roi_gap")
    if churn > max_churn:
        blockers.append("excessive_churn")
    if cost_drag_bps > max_cost:
        blockers.append("cost_drag_above_threshold")
    if stability > 0 and stability < min_stability:
        blockers.append("low_window_stability")
    if capacity_multiple > 0 and capacity_multiple < min_capacity:
        blockers.append("insufficient_liquidity_capacity")
    if mfe_mae > 0 and mfe_mae < min_mfe_mae:
        blockers.append("weak_mfe_mae")
    if recent <= 0:
        blockers.append("non_positive_recent_1h_return")
    if str(v2.get("regime_support", "")).lower() == "regime-fragile":
        blockers.append("fragile_regime")
    if str(v2.get("roi_rejection_risk", "")).lower() == "high":
        blockers.append("high_roi_rejection_risk")
    if stale:
        blockers.append("stale_data")

    score = (
        _float(v2.get("net_roi_v2_score")) * 1.30
        + _float(base.get("net_roi_score")) * 0.35
        + raw_upside_score * 0.045
        + max(raw_net_return_pct, 0.0) * 0.025
        + max(recent, 0.0) * 190.0
        + min(max(mfe_mae, 0.0), 8.0) * 1.35
        + min(max(capacity_multiple, 0.0), 8.0) * 0.95
        + expected_execution_quality * 15.0
        + max(market_structure, 0.0) * 6.0
        - drawdown_abs * 95.0
        - churn * 16.0
        - max(cost_drag_bps - 4.0, 0.0) * 0.28
        - max(raw_vs_net_gap - max_gap, 0.0) * 0.22
        - max(decay, 0.0) * 20.0
    )
    if not bool(config.get("ONE_HOUR_EDGE_V2_ENABLED", True)):
        score = _float(v2.get("net_roi_v2_score"))
    score = float(score if math.isfinite(score) else 0.0)
    grade = _one_hour_edge_grade(score, blockers, expected_execution_quality)

    return {
        "one_hour_edge_v2": score,
        "one_hour_edge_grade": grade,
        "expected_execution_quality": float(expected_execution_quality),
        "profitability_blockers": list(dict.fromkeys(blockers)),
        "raw_vs_net_roi_gap": float(raw_vs_net_gap),
        "candidate_quality_breakdown": {
            "net_roi_v2_score": _float(v2.get("net_roi_v2_score")),
            "net_roi_score": _float(base.get("net_roi_score")),
            "raw_upside_score": raw_upside_score,
            "raw_total_return_pct": raw_total_return_pct,
            "raw_net_return_pct": raw_net_return_pct,
            "recent_1h_return": recent,
            "mfe_mae_ratio": mfe_mae,
            "drawdown_abs": drawdown_abs,
            "expected_fill_quality": fill_quality,
            "expected_execution_quality": expected_execution_quality,
            "capacity_multiple": capacity_multiple,
            "market_structure_score": market_structure,
            "cost_drag_bps": cost_drag_bps,
            "spread_bps": spread_bps,
            "churn_penalty": churn,
            "regime_support": v2.get("regime_support"),
            "roi_rejection_risk": v2.get("roi_rejection_risk"),
        },
    }


def expected_fill_quality(
    *,
    spread_bps: float,
    cost_drag_bps: float,
    slippage_bps: float,
    liquidity_usd: float,
    capacity_multiple: float,
    volatility_regime: str,
    stale_data: bool,
) -> float:
    quality = 1.0
    quality -= min(max(spread_bps, 0.0) / 80.0, 0.35)
    quality -= min(max(cost_drag_bps, 0.0) / 180.0, 0.30)
    quality -= min(max(slippage_bps, 0.0) / 100.0, 0.18)
    if liquidity_usd > 0:
        quality += min(liquidity_usd / 250_000.0, 0.12)
    if capacity_multiple > 0:
        quality += min(capacity_multiple / 25.0, 0.12)
    if volatility_regime == "elevated":
        quality -= 0.08
    elif volatility_regime == "dislocated":
        quality -= 0.28
    elif volatility_regime == "tradable":
        quality += 0.04
    if stale_data:
        quality -= 0.35
    return max(0.0, min(1.0, quality))


def churn_penalty(
    *,
    turnover_after_fees: float,
    trades_per_day: float,
    avg_trade_return: float,
    cost_drag_bps: float,
) -> float:
    turnover_penalty = max(turnover_after_fees - 3.0, 0.0) / 10.0
    frequency_penalty = max(trades_per_day - 18.0, 0.0) / 60.0
    low_edge_penalty = max(cost_drag_bps / 10_000.0 - max(avg_trade_return, 0.0), 0.0) * 25.0
    return max(0.0, min(turnover_penalty + frequency_penalty + low_edge_penalty, 1.0))


def _regime_bucket(payload: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, str]:
    volatility = str(payload.get("volatility_regime") or "").strip().lower()
    if not volatility or volatility == "unknown":
        volatility_pct_value = _float(_first(payload, "volatility_pct", "volatility"))
        if volatility_pct_value <= 0:
            volatility = "unknown"
        elif volatility_pct_value < 0.20:
            volatility = "compressed"
        elif volatility_pct_value < 1.50:
            volatility = "tradable"
        elif volatility_pct_value < 3.0:
            volatility = "elevated"
        else:
            volatility = "dislocated"

    spread_bps_value = _float(payload.get("spread_bps"))
    cost_drag_bps_value = _float(payload.get("cost_drag_bps"))
    cost_proxy = max(spread_bps_value, cost_drag_bps_value)
    if cost_proxy <= 0:
        spread_cost = "unknown_cost"
    elif cost_proxy <= 8.0:
        spread_cost = "low_cost"
    elif cost_proxy <= 18.0:
        spread_cost = "normal_cost"
    elif cost_proxy <= 35.0:
        spread_cost = "high_cost"
    else:
        spread_cost = "hostile_cost"

    capacity_multiple = _float(base.get("capacity_multiple"))
    if capacity_multiple <= 0:
        capacity_multiple = _float(payload.get("capacity_multiple"))
    if capacity_multiple >= 3.0:
        liquidity = "deep_liquidity"
    elif capacity_multiple >= 1.5:
        liquidity = "adequate_liquidity"
    elif capacity_multiple > 0:
        liquidity = "thin_liquidity"
    else:
        liquidity = "unknown_liquidity"

    momentum = _float(payload.get("momentum_acceleration"))
    structure = _float(_first(payload, "market_structure_score", "market_structure_trend"))
    breakout_proximity = _float(payload.get("breakout_proximity_bps"), 250.0)
    expected_move = max(_float(payload.get("cost_adjusted_expected_move")), _float(base.get("edge_after_cost_bps")))
    if expected_move > 0 and breakout_proximity <= 75.0 and structure >= 0.45:
        trend_breakout = "breakout_supported"
    elif expected_move > 0 and (momentum >= 0 or structure >= 0.35):
        trend_breakout = "trend_supported"
    elif expected_move > 0:
        trend_breakout = "neutral_breakout"
    else:
        trend_breakout = "fragile_breakout"

    return {
        "volatility": volatility,
        "spread_cost": spread_cost,
        "liquidity": liquidity,
        "trend_breakout": trend_breakout,
    }


def _regime_adjustment(bucket: Mapping[str, str]) -> float:
    score = 0.0
    score += {"compressed": 0.8, "tradable": 1.4, "elevated": -0.7, "dislocated": -2.2}.get(str(bucket.get("volatility")), 0.0)
    score += {"low_cost": 1.2, "normal_cost": 0.4, "high_cost": -1.0, "hostile_cost": -2.4}.get(str(bucket.get("spread_cost")), 0.0)
    score += {"deep_liquidity": 1.2, "adequate_liquidity": 0.5, "thin_liquidity": -1.0, "unknown_liquidity": -0.2}.get(str(bucket.get("liquidity")), 0.0)
    score += {
        "breakout_supported": 1.3,
        "trend_supported": 0.8,
        "neutral_breakout": 0.0,
        "fragile_breakout": -1.4,
    }.get(str(bucket.get("trend_breakout")), 0.0)
    return score


def _regime_support_label(regime_adjustment: float) -> str:
    if regime_adjustment >= 2.0:
        return "regime-supported"
    if regime_adjustment <= -1.8:
        return "regime-fragile"
    return "regime-neutral"


def _tail_loss_penalty(*, drawdown_abs: float, adverse: float, volatility_regime: str) -> float:
    penalty = max(drawdown_abs - 0.08, 0.0) * 4.0
    penalty += max(adverse - 0.012, 0.0) * 5.0
    if volatility_regime == "dislocated":
        penalty += 0.30
    elif volatility_regime == "elevated":
        penalty += 0.10
    return max(0.0, min(penalty, 1.5))


def _downside_asymmetry_penalty(*, mfe_mae: float, avg_win: float, avg_loss_abs: float, drawdown_abs: float) -> float:
    ratio_penalty = max(1.4 - max(mfe_mae, 0.0), 0.0) / 1.4
    trade_penalty = max(avg_loss_abs - avg_win, 0.0) / max(avg_loss_abs + avg_win, 1e-9) if (avg_loss_abs or avg_win) else 0.0
    drawdown_penalty = max(drawdown_abs - 0.12, 0.0) * 2.0
    return max(0.0, min(ratio_penalty * 0.55 + trade_penalty * 0.25 + drawdown_penalty, 1.3))


def _roi_rejection_risk(
    *,
    base: Mapping[str, Any],
    score: float,
    regime_support: str,
    tail_loss_penalty: float,
    downside_asymmetry_penalty: float,
    config: Mapping[str, Any],
) -> str:
    fill_quality = _float(base.get("expected_fill_quality"))
    min_fill = _float(config.get("NET_ROI_MIN_FILL_QUALITY"), 0.55)
    min_edge = _float(config.get("NET_ROI_MIN_EDGE_BPS"), 4.0)
    if (
        bool(base.get("stale_data"))
        or fill_quality < min_fill * 0.85
        or (regime_support == "regime-fragile" and score < 12.0)
        or tail_loss_penalty >= 0.45
        or downside_asymmetry_penalty >= 0.55
        or (_float(base.get("edge_after_cost_bps")) < min_edge and score <= 0.0)
    ):
        return "high"
    if (
        fill_quality < min_fill
        or regime_support == "regime-fragile"
        or tail_loss_penalty >= 0.20
        or downside_asymmetry_penalty >= 0.30
        or score < 10.0
    ):
        return "medium"
    return "low"


def _roi_quality_grade(score: float, rejection_risk: str, fill_quality: float, stability: float) -> str:
    if score >= 35.0 and rejection_risk == "low" and fill_quality >= 0.72 and stability >= 0.55:
        return "A"
    if score >= 18.0 and rejection_risk in {"low", "medium"} and fill_quality >= 0.55:
        return "B"
    if score >= 8.0 and rejection_risk != "high":
        return "C"
    return "D"


def _one_hour_edge_grade(score: float, blockers: list[str], execution_quality: float) -> str:
    blocker_count = len(blockers)
    critical = {
        "low_edge_after_costs",
        "low_expected_execution_quality",
        "fragile_regime",
        "high_roi_rejection_risk",
        "stale_data",
    }
    has_critical = any(item in critical for item in blockers)
    if score >= 55.0 and execution_quality >= 0.72 and blocker_count == 0:
        return "A"
    if score >= 32.0 and execution_quality >= 0.60 and blocker_count <= 1 and not has_critical:
        return "B"
    if score >= 16.0 and execution_quality >= 0.48 and not has_critical:
        return "C"
    return "D"


def _first(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return 0.0


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return default
    return candidate if math.isfinite(candidate) else default
