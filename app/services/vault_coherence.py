"""Explainable multi-horizon forecast scoring for the 1H10 Vault view."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


HORIZONS = ["1m", "5m", "15m", "45m", "1h", "4h", "1d", "7d"]
PHASES = [
    "idle",
    "collectingData",
    "analyzing",
    "scoringStrategies",
    "runningForecast",
    "checkingCoherence",
    "ready",
    "staleData",
    "error",
]


def get_available_horizons() -> list[str]:
    """Return the 1H10 forecast horizons exposed to services and UI."""

    return list(HORIZONS)


def build_horizon_forecasts(
    features: dict[str, Any] | None,
    *,
    config: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Build explainable forecast scoring outputs for each supported horizon."""

    resolved_now = _aware(now)
    horizon_features = _horizon_feature_map(features or {})
    rows: list[dict[str, Any]] = []
    for horizon in get_available_horizons():
        row = _feature_for_horizon(horizon, horizon_features)
        rows.append(_forecast_from_features(horizon, row, config=config or {}, now=resolved_now))
    return rows


def score_horizon_strategy(
    horizon: str,
    forecast: dict[str, Any] | None = None,
    features: dict[str, Any] | None = None,
    *,
    config: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the strategy-side horizon score used by coherence checks."""

    _ = features, config
    resolved_now = _aware(now)
    row = dict(forecast or {})
    direction = str(row.get("direction") or "neutral")
    bias = {"bullish": "long", "bearish": "short"}.get(direction, "neutral")
    confidence = _clamp(_safe_float(row.get("confidence")), 0.0, 100.0)
    strength = _clamp(_safe_float(row.get("strength")), 0.0, 100.0)
    risk = _risk_from_forecast(row)
    data_quality = str(row.get("dataQuality") or "insufficient")
    blockers: list[str] = []
    confirmations: list[str] = []

    if data_quality == "stale":
        blockers.append("Data stale - waiting for fresh market update")
    elif data_quality == "insufficient":
        blockers.append("Insufficient horizon data")
    else:
        confirmations.append("Fresh market features available")

    if risk == "high":
        blockers.append("Volatility elevated, confidence reduced")
    elif risk == "low":
        confirmations.append("Volatility within accepted range")

    if bias == "neutral":
        blockers.append("No clear directional strategy bias")
    else:
        confirmations.append(f"Strategy bias aligns {bias}")

    if confidence < 35:
        blockers.append("Weak signal confidence")
    elif confidence >= 60 and bias != "neutral":
        confirmations.append("Signal confidence supports strategy scoring")

    score = confidence * 0.55 + strength * 0.35
    if bias == "neutral":
        score = min(score, 48.0)
    if risk == "high":
        score *= 0.68
    if data_quality == "stale":
        score *= 0.70
    elif data_quality == "insufficient":
        score *= 0.35

    return {
        "horizon": str(horizon),
        "strategyBias": bias,
        "score": int(round(_clamp(score, 0.0, 100.0))),
        "confidence": int(round(_clamp(confidence, 0.0, 100.0))),
        "riskLevel": risk,
        "blockers": list(dict.fromkeys(blockers)),
        "confirmations": list(dict.fromkeys(confirmations)),
        "updatedAt": str(row.get("updatedAt") or _iso(resolved_now)),
    }


def calculate_coherence_summary(
    horizon_forecasts: list[dict[str, Any]] | None,
    horizon_strategy_scores: list[dict[str, Any]] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate horizon forecasts and strategy scores into a vault summary."""

    resolved_now = _aware(now)
    forecasts = [dict(item) for item in (horizon_forecasts or []) if isinstance(item, dict)]
    strategies = {
        str(item.get("horizon")): dict(item)
        for item in (horizon_strategy_scores or [])
        if isinstance(item, dict) and item.get("horizon")
    }
    if not forecasts:
        return {
            "overallDirection": "neutral",
            "overallConfidence": 0,
            "coherenceScore": 0,
            "automationReadiness": "notReady",
            "primaryHorizon": None,
            "conflictingHorizons": [],
            "riskNotes": ["Insufficient horizon data"],
            "summary": "Waiting for fresh market update before forming a next-move estimate.",
            "updatedAt": _iso(resolved_now),
        }

    usable = [item for item in forecasts if str(item.get("dataQuality")) != "insufficient"]
    insufficient = [item for item in forecasts if str(item.get("dataQuality")) == "insufficient"]
    stale = [item for item in forecasts if str(item.get("dataQuality")) == "stale"]
    high_vol = [item for item in forecasts if str(item.get("volatilityRisk")) == "high"]
    direction_weights = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
    for forecast in forecasts:
        direction = str(forecast.get("direction") or "neutral")
        if direction not in direction_weights:
            direction = "neutral"
        weight = max(_safe_float(forecast.get("confidence")), _safe_float(forecast.get("strength")) * 0.5)
        if str(forecast.get("dataQuality")) == "insufficient":
            weight *= 0.25
        elif str(forecast.get("dataQuality")) == "stale":
            weight *= 0.55
        direction_weights[direction] += _clamp(weight, 0.0, 100.0)

    directional_total = direction_weights["bullish"] + direction_weights["bearish"]
    leading_direction = "neutral"
    if directional_total > 0:
        leading_direction = "bullish" if direction_weights["bullish"] >= direction_weights["bearish"] else "bearish"
    opposing_direction = "bearish" if leading_direction == "bullish" else "bullish"
    leading_weight = direction_weights.get(leading_direction, 0.0)
    opposing_weight = direction_weights.get(opposing_direction, 0.0)
    conflict_ratio = opposing_weight / max(directional_total, 1.0)

    if not usable:
        overall_direction = "neutral"
    elif conflict_ratio >= 0.30 and opposing_weight >= 25:
        overall_direction = "mixed"
    elif leading_weight < 30 or leading_direction == "neutral":
        overall_direction = "neutral"
    else:
        overall_direction = leading_direction

    aligned = [
        item
        for item in forecasts
        if str(item.get("direction")) == overall_direction and str(item.get("dataQuality")) != "insufficient"
    ]
    if overall_direction == "mixed":
        aligned = [item for item in forecasts if str(item.get("direction")) in {"bullish", "bearish"}]
    primary = max(aligned or usable or forecasts, key=lambda item: _safe_float(item.get("confidence")), default={})
    primary_horizon = str(primary.get("horizon")) if primary.get("horizon") and str(primary.get("dataQuality")) != "insufficient" else None

    conflicting = []
    if leading_direction in {"bullish", "bearish"}:
        conflicting = [
            str(item.get("horizon"))
            for item in forecasts
            if str(item.get("direction")) == opposing_direction and _safe_float(item.get("confidence")) >= 35
        ]
    if overall_direction == "mixed" and not conflicting:
        conflicting = [
            str(item.get("horizon"))
            for item in forecasts
            if str(item.get("direction")) in {"bullish", "bearish"} and _safe_float(item.get("confidence")) >= 35
        ]

    strategy_disagreements = _strategy_disagreements(forecasts, strategies)
    risk_notes: list[str] = []
    if insufficient:
        risk_notes.append("Some horizons have insufficient data")
    if len(insufficient) >= max(1, len(forecasts) // 2):
        risk_notes.append("Majority horizon data is insufficient")
    if stale:
        risk_notes.append("Data stale - waiting for fresh market update")
    if high_vol:
        risk_notes.append("Forecast confidence reduced due to volatility")
    if conflicting:
        risk_notes.append("Higher-timeframe conflict detected" if _has_higher_timeframe_conflict(forecasts, conflicting) else "Mixed horizon signal")
    if strategy_disagreements:
        risk_notes.append("Strategy and forecast disagreement reduced readiness")

    direction_share = leading_weight / max(directional_total + direction_weights["neutral"], 1.0)
    coherence = direction_share * 100.0
    if overall_direction == "mixed":
        coherence = min(coherence, 55.0)
    if stale:
        coherence *= 0.78
    if high_vol:
        coherence *= 0.78
    if len(insufficient) >= max(1, len(forecasts) // 2):
        coherence *= 0.55
    elif insufficient:
        coherence *= 0.85
    if strategy_disagreements:
        coherence *= 0.75

    aligned_confidences = [_safe_float(item.get("confidence")) for item in aligned if _safe_float(item.get("confidence")) > 0]
    overall_confidence = (sum(aligned_confidences) / len(aligned_confidences)) if aligned_confidences else 0.0
    overall_confidence *= _clamp(coherence, 0.0, 100.0) / 100.0

    blockers = {
        "majority_insufficient": len(insufficient) >= max(1, len(forecasts) // 2),
        "strong_conflict": bool(conflicting) and conflict_ratio >= 0.35,
        "all_stale": bool(stale) and len(stale) >= max(1, len(forecasts) // 2),
        "error": any("error" in str(note).lower() for note in risk_notes),
    }
    if blockers["majority_insufficient"] or blockers["strong_conflict"] or blockers["all_stale"] or blockers["error"]:
        readiness = "notReady"
    elif (
        overall_direction in {"bullish", "bearish"}
        and _safe_float(coherence) >= 72
        and _safe_float(overall_confidence) >= 58
        and not stale
        and len(high_vol) <= 1
        and not strategy_disagreements
    ):
        readiness = "ready"
    else:
        readiness = "caution"

    if not risk_notes and readiness == "ready":
        risk_notes.append("Strategy coherence active")
    elif not risk_notes:
        risk_notes.append("Automation readiness: Caution")

    summary = _summary_text(overall_direction, readiness, bool(conflicting), bool(high_vol), bool(stale))
    return {
        "overallDirection": overall_direction,
        "overallConfidence": int(round(_clamp(overall_confidence, 0.0, 100.0))),
        "coherenceScore": int(round(_clamp(coherence, 0.0, 100.0))),
        "automationReadiness": readiness,
        "primaryHorizon": primary_horizon,
        "conflictingHorizons": list(dict.fromkeys(item for item in conflicting if item)),
        "riskNotes": list(dict.fromkeys(risk_notes)),
        "summary": summary,
        "updatedAt": _iso(resolved_now),
    }


def format_vault_cycle_status(
    context: dict[str, Any] | None = None,
    *,
    horizon_forecasts: list[dict[str, Any]] | None = None,
    horizon_strategy_scores: list[dict[str, Any]] | None = None,
    coherence_summary: dict[str, Any] | None = None,
    now: datetime | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Format the 70-minute 1H10 evaluation cadence for UI reporting."""

    resolved_now = _aware(now)
    context = dict(context or {})
    forecasts = [dict(item) for item in (horizon_forecasts or []) if isinstance(item, dict)]
    strategies = [dict(item) for item in (horizon_strategy_scores or []) if isinstance(item, dict)]
    coherence = dict(coherence_summary or {})
    last_data = _latest_timestamp([item.get("updatedAt") for item in forecasts] + [context.get("last_data_refresh")])
    last_strategy = _latest_timestamp([item.get("updatedAt") for item in strategies] + [context.get("last_strategy_score")])
    last_forecast = _latest_timestamp([item.get("updatedAt") for item in forecasts] + [context.get("last_ml_forecast")])
    last_completed = _latest_timestamp(
        [
            context.get("last_completed_vault_cycle"),
            context.get("completed_at"),
            context.get("created_at"),
            coherence.get("updatedAt"),
            last_forecast,
        ]
    )
    next_cycle = (last_completed or resolved_now) + timedelta(minutes=70)
    stale = any(str(item.get("dataQuality")) == "stale" for item in forecasts)
    insufficient_majority = bool(forecasts) and sum(1 for item in forecasts if str(item.get("dataQuality")) == "insufficient") >= max(1, len(forecasts) // 2)
    readiness = str(coherence.get("automationReadiness") or "")

    if error:
        phase = "error"
    elif stale:
        phase = "staleData"
    elif not forecasts:
        phase = "collectingData"
    elif insufficient_majority:
        phase = "analyzing"
    elif not strategies:
        phase = "scoringStrategies"
    elif not coherence:
        phase = "checkingCoherence"
    elif readiness in {"ready", "caution", "notReady"}:
        phase = "ready"
    else:
        phase = "idle"

    return {
        "phase": phase,
        "phaseLabel": _phase_label(phase),
        "phaseSequence": list(PHASES),
        "lastDataRefresh": _iso(last_data) if last_data else None,
        "lastStrategyScore": _iso(last_strategy) if last_strategy else None,
        "lastMlForecast": _iso(last_forecast) if last_forecast else None,
        "lastCompletedVaultCycle": _iso(last_completed) if last_completed else None,
        "nextScheduled1h10Cycle": _iso(next_cycle),
        "evaluationCadenceMinutes": 70,
        "stale": bool(stale),
        "error": str(error or ""),
        "statusMessage": _cycle_status_message(phase, coherence),
        "updatedAt": _iso(resolved_now),
    }


def cycle_coherence_payload_from_forecasts(
    forecasts: list[dict[str, Any]] | None,
    *,
    now: datetime | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose a cycle-level coherence payload from existing leg forecasts."""

    resolved_now = _aware(now)
    candidates = [dict(item) for item in (forecasts or []) if isinstance(item, dict) and item]
    selected = _select_cycle_forecast(candidates)
    if selected:
        horizon_forecasts = [dict(item) for item in selected.get("horizon_forecasts", []) if isinstance(item, dict)]
        strategy_scores = [dict(item) for item in selected.get("horizon_strategy_scores", []) if isinstance(item, dict)]
        coherence = dict(selected.get("coherence_summary") or {})
        if not coherence and horizon_forecasts:
            if not strategy_scores:
                strategy_scores = [score_horizon_strategy(str(row.get("horizon")), row, now=resolved_now) for row in horizon_forecasts]
            coherence = calculate_coherence_summary(horizon_forecasts, strategy_scores, now=resolved_now)
    else:
        horizon_forecasts = []
        strategy_scores = []
        coherence = calculate_coherence_summary([], [], now=resolved_now)

    cycle_context = {
        **(context or {}),
        "created_at": selected.get("created_at") if selected else None,
        "last_completed_vault_cycle": selected.get("created_at") if selected else None,
    }
    cycle_status = format_vault_cycle_status(
        cycle_context,
        horizon_forecasts=horizon_forecasts,
        horizon_strategy_scores=strategy_scores,
        coherence_summary=coherence,
        now=resolved_now,
    )
    reasoning = _cycle_reasoning(selected, horizon_forecasts, coherence)
    return {
        "cycle_status": cycle_status,
        "horizon_forecasts": horizon_forecasts,
        "horizon_strategy_scores": strategy_scores,
        "coherence_summary": coherence,
        "reasoning": reasoning,
    }


def extract_cycle_coherence_payload(cycle: Any) -> dict[str, Any]:
    """Read cycle-level coherence from metadata or rebuild it from leg forecasts."""

    metadata = dict(getattr(cycle, "selection_metadata", {}) or {})
    if all(metadata.get(key) is not None for key in ("cycle_status", "horizon_forecasts", "horizon_strategy_scores", "coherence_summary")):
        return {
            "cycle_status": metadata.get("cycle_status") or {},
            "horizon_forecasts": metadata.get("horizon_forecasts") or [],
            "horizon_strategy_scores": metadata.get("horizon_strategy_scores") or [],
            "coherence_summary": metadata.get("coherence_summary") or {},
            "reasoning": metadata.get("reasoning") or [],
        }

    forecasts: list[dict[str, Any]] = []
    for provider in metadata.get("provider_allocation_history", metadata.get("exchange_allocation_history", [])) or []:
        if not isinstance(provider, dict):
            continue
        for leg in provider.get("legs", []) or []:
            if isinstance(leg, dict) and isinstance(leg.get("forecast"), dict):
                forecasts.append(dict(leg["forecast"]))
    for leg in getattr(cycle, "allocation_legs", []) or []:
        details = dict(getattr(leg, "details", {}) or {})
        if isinstance(details.get("one_h10_forecast"), dict):
            forecasts.append(dict(details["one_h10_forecast"]))
        run = getattr(leg, "strategy_run", None)
        params = dict(getattr(run, "parameters", {}) or {}) if run is not None else {}
        if isinstance(params.get("one_h10_forecast"), dict):
            forecasts.append(dict(params["one_h10_forecast"]))

    context = {
        "created_at": _iso(getattr(cycle, "started_at", None)) if getattr(cycle, "started_at", None) else None,
        "completed_at": _iso(getattr(cycle, "settled_at", None)) if getattr(cycle, "settled_at", None) else None,
    }
    return cycle_coherence_payload_from_forecasts(forecasts, context=context)


class VaultCoherenceService:
    """Service wrapper for app registration and dependency injection."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def get_available_horizons(self) -> list[str]:
        return get_available_horizons()

    def build_horizon_forecasts(self, features: dict[str, Any] | None, *, now: datetime | None = None) -> list[dict[str, Any]]:
        return build_horizon_forecasts(features, config=self.config, now=now)

    def score_horizon_strategy(
        self,
        horizon: str,
        forecast: dict[str, Any] | None = None,
        features: dict[str, Any] | None = None,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        return score_horizon_strategy(horizon, forecast, features, config=self.config, now=now)

    def calculate_coherence_summary(
        self,
        horizon_forecasts: list[dict[str, Any]] | None,
        horizon_strategy_scores: list[dict[str, Any]] | None,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        return calculate_coherence_summary(horizon_forecasts, horizon_strategy_scores, now=now)

    def format_vault_cycle_status(
        self,
        context: dict[str, Any] | None = None,
        *,
        horizon_forecasts: list[dict[str, Any]] | None = None,
        horizon_strategy_scores: list[dict[str, Any]] | None = None,
        coherence_summary: dict[str, Any] | None = None,
        now: datetime | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        return format_vault_cycle_status(
            context,
            horizon_forecasts=horizon_forecasts,
            horizon_strategy_scores=horizon_strategy_scores,
            coherence_summary=coherence_summary,
            now=now,
            error=error,
        )

    def cycle_coherence_payload_from_forecasts(
        self,
        forecasts: list[dict[str, Any]] | None,
        *,
        now: datetime | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return cycle_coherence_payload_from_forecasts(forecasts, now=now, context=context)

    def extract_cycle_coherence_payload(self, cycle: Any) -> dict[str, Any]:
        return extract_cycle_coherence_payload(cycle)


def _forecast_from_features(
    horizon: str,
    row: dict[str, Any] | None,
    *,
    config: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    row = dict(row or {})
    updated_at = _parse_datetime(row.get("updated_at") or row.get("updatedAt"))
    derived = bool(row.get("derived", False))
    source_horizon = row.get("sourceHorizon") or row.get("source_horizon")
    has_data = _has_usable_features(row)
    data_quality = _data_quality(row, updated_at, config, now)
    volatility_raw = max(_safe_float(row.get("atr_pct")), _safe_float(row.get("volatility")))
    volatility_risk = _volatility_risk(volatility_raw, row)
    reasoning: list[str] = []

    if not has_data:
        return {
            "horizon": horizon,
            "direction": "neutral",
            "confidence": 0,
            "strength": 0,
            "volatilityRisk": "medium",
            "dataQuality": "insufficient",
            "reasoning": ["Insufficient horizon data"],
            "updatedAt": _iso(updated_at or now),
            "derived": derived,
            "sourceHorizon": source_horizon,
        }

    score, components = _directional_score(row)
    strength = _clamp(abs(score) * 100.0, 0.0, 100.0)
    confidence = strength * 0.62 + _component_breadth(components) * 22.0
    if volatility_risk == "high":
        confidence *= 0.68
        reasoning.append("Volatility elevated, confidence reduced")
    elif volatility_risk == "low":
        confidence += 6.0
    if data_quality == "stale":
        confidence *= 0.62
        reasoning.append("Data stale - waiting for fresh market update")
    elif data_quality == "insufficient":
        confidence *= 0.35
        reasoning.append("Insufficient horizon data")
    if derived:
        confidence *= 0.82 if horizon != "7d" or source_horizon == "1d" else 0.70
        reasoning.append(f"{horizon} horizon derived from {source_horizon} market structure")

    direction = "neutral"
    if abs(score) >= 0.13 and confidence >= 28:
        direction = "bullish" if score > 0 else "bearish"

    if direction == "bullish":
        if horizon in {"1m", "5m", "15m"}:
            reasoning.append("Momentum improving on shorter horizons")
        else:
            reasoning.append("Market structure leans bullish on this horizon")
    elif direction == "bearish":
        if horizon in {"1m", "5m", "15m"}:
            reasoning.append("Momentum weakening on shorter horizons")
        else:
            reasoning.append("Market structure leans bearish on this horizon")
    else:
        reasoning.append("Higher-timeframe trend remains neutral" if horizon in {"4h", "1d", "7d"} else "Directional signal remains neutral")

    if components.get("range") == "compressed":
        reasoning.append("Range compression detected")
    elif components.get("range") == "expanded":
        reasoning.append("Range expansion detected")
    if components.get("volume") == "rising":
        reasoning.append("Volume shift supports the move")
    if components.get("ma_alignment"):
        reasoning.append("Moving average alignment supports the direction")
    if components.get("candle"):
        reasoning.append("Recent candle structure supports the signal")

    if horizon == "7d" and source_horizon == "4h" and data_quality != "stale":
        data_quality = "insufficient"
        confidence *= 0.55
        reasoning.append("7d estimate needs daily data for stronger quality")

    return {
        "horizon": horizon,
        "direction": direction,
        "confidence": int(round(_clamp(confidence, 0.0, 100.0))),
        "strength": int(round(_clamp(strength, 0.0, 100.0))),
        "volatilityRisk": volatility_risk,
        "dataQuality": data_quality,
        "reasoning": list(dict.fromkeys(reasoning))[:5],
        "updatedAt": _iso(updated_at or now),
        "derived": derived,
        "sourceHorizon": source_horizon,
    }


def _horizon_feature_map(features: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    nested = features.get("one_h10_horizon_features")
    if isinstance(nested, dict):
        for horizon, row in nested.items():
            if isinstance(row, dict):
                rows[str(horizon)] = dict(row)

    for horizon in get_available_horizons():
        prefix = f"tf_{horizon.replace(' ', '_')}_"
        prefixed = {
            key.removeprefix(prefix): value
            for key, value in features.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        if prefixed:
            rows.setdefault(horizon, {}).update(prefixed)

    if not rows and _has_usable_features(features):
        rows["15m"] = dict(features)
    for horizon, row in rows.items():
        row.setdefault("timeframe", horizon)
        if features.get("one_h10_feature_updated_at") and not row.get("updated_at"):
            row["updated_at"] = features.get("one_h10_feature_updated_at")
    return rows


def _feature_for_horizon(horizon: str, rows: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if horizon in rows:
        return dict(rows[horizon])
    if horizon == "45m" and "15m" in rows:
        return {**rows["15m"], "derived": True, "sourceHorizon": "15m", "timeframe": "45m"}
    if horizon == "7d":
        if "1d" in rows:
            return {**rows["1d"], "derived": True, "sourceHorizon": "1d", "timeframe": "7d"}
        if "4h" in rows:
            return {**rows["4h"], "derived": True, "sourceHorizon": "4h", "timeframe": "7d"}
    return None


def _directional_score(row: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    close = max(_safe_float(row.get("close"), 1.0), 1e-9)
    trend = _normalize_component(_safe_float(row.get("trend_strength")) * 6.0)
    ema_trend = _safe_float(row.get("ema_trend"))
    if ema_trend == 0 and row.get("ema_fast") is not None and row.get("ema_slow") is not None:
        ema_trend = _safe_float(row.get("ema_fast")) - _safe_float(row.get("ema_slow"))
    ma_component = _normalize_component((ema_trend / close) * 120.0)
    if row.get("sma_fast") is not None and row.get("sma_slow") is not None:
        sma_component = _normalize_component(((_safe_float(row.get("sma_fast")) - _safe_float(row.get("sma_slow"))) / close) * 90.0)
        ma_component = (ma_component + sma_component) / 2.0
    macd = _normalize_component((_safe_float(row.get("macd_histogram")) / close) * 140.0)
    imbalance = _normalize_component(_safe_float(row.get("order_book_imbalance")) * 2.0)
    rsi = _safe_float(row.get("rsi"), 50.0)
    rsi_component = 0.0
    if rsi > 58:
        rsi_component = _normalize_component((rsi - 58.0) / 18.0)
    elif rsi < 42:
        rsi_component = -_normalize_component((42.0 - rsi) / 18.0)
    candle_component = _candle_component(row, close)
    range_component, range_state = _range_component(row)
    volume_component, volume_state = _volume_component(row)

    components = {
        "trend": trend,
        "ma": ma_component,
        "macd": macd,
        "imbalance": imbalance,
        "rsi": rsi_component,
        "candle": abs(candle_component) > 0.10,
        "range": range_state,
        "volume": volume_state,
        "ma_alignment": abs(ma_component) > 0.12,
    }
    score = (
        trend * 0.24
        + ma_component * 0.22
        + macd * 0.16
        + rsi_component * 0.12
        + imbalance * 0.08
        + candle_component * 0.08
        + range_component * 0.05
        + volume_component * 0.05
    )
    return _clamp(score, -1.0, 1.0), components


def _range_component(row: dict[str, Any]) -> tuple[float, str | None]:
    timing = row.get("fibonacci_timing") if isinstance(row.get("fibonacci_timing"), dict) else {}
    position = _safe_float(timing.get("range_position"), -1.0)
    if position >= 0:
        if position <= 0.28:
            return 0.35, "compressed"
        if position >= 0.78:
            return -0.30, "expanded"
    width = _safe_float(row.get("range_width_pct"), -1.0)
    if 0 <= width <= 0.012:
        return 0.10, "compressed"
    if width >= 0.05:
        return -0.08, "expanded"
    return 0.0, None


def _volume_component(row: dict[str, Any]) -> tuple[float, str | None]:
    volume = row.get("volume_spike") if isinstance(row.get("volume_spike"), dict) else {}
    ratio = _safe_float(volume.get("ratio"), _safe_float(row.get("volume_shift"), 1.0))
    if ratio >= 1.25:
        direction = 1.0 if _safe_float(row.get("trend_strength")) >= 0 else -1.0
        return 0.18 * direction, "rising"
    if 0 < ratio <= 0.70:
        return -0.05, "falling"
    return 0.0, None


def _candle_component(row: dict[str, Any], close: float) -> float:
    open_price = _safe_float(row.get("open"), close)
    high = _safe_float(row.get("high"), max(open_price, close))
    low = _safe_float(row.get("low"), min(open_price, close))
    span = max(high - low, close * 0.0001)
    body = (close - open_price) / span
    return _clamp(body, -1.0, 1.0)


def _data_quality(row: dict[str, Any], updated_at: datetime | None, config: dict[str, Any], now: datetime) -> str:
    if not _has_usable_features(row):
        return "insufficient"
    if updated_at is None:
        return "fresh"
    max_age = max(_safe_float(config.get("ONE_H10_FEATURE_REFRESH_SECONDS"), 3600.0) * 2.0, 70.0 * 60.0)
    age = (now - updated_at).total_seconds()
    return "stale" if age > max_age else "fresh"


def _has_usable_features(row: dict[str, Any] | None) -> bool:
    if not isinstance(row, dict) or not row:
        return False
    keys = {
        "close",
        "trend_strength",
        "ema_trend",
        "ema_fast",
        "ema_slow",
        "sma_fast",
        "sma_slow",
        "macd_histogram",
        "rsi",
    }
    return any(row.get(key) is not None for key in keys)


def _volatility_risk(value: float, row: dict[str, Any]) -> str:
    spread = _safe_float(row.get("spread_bps"))
    if value >= 0.045 or spread >= 25:
        return "high"
    if value >= 0.018 or spread >= 10:
        return "medium"
    return "low"


def _risk_from_forecast(row: dict[str, Any]) -> str:
    risk = str(row.get("volatilityRisk") or "medium")
    quality = str(row.get("dataQuality") or "")
    if quality == "insufficient":
        return "high"
    if quality == "stale" and risk == "low":
        return "medium"
    return risk if risk in {"low", "medium", "high"} else "medium"


def _strategy_disagreements(forecasts: list[dict[str, Any]], strategies: dict[str, dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    expected = {"bullish": "long", "bearish": "short", "neutral": "neutral"}
    for forecast in forecasts:
        horizon = str(forecast.get("horizon") or "")
        if not horizon or horizon not in strategies:
            continue
        direction = str(forecast.get("direction") or "neutral")
        if direction not in expected:
            continue
        strategy_bias = str(strategies[horizon].get("strategyBias") or "neutral")
        if strategy_bias != expected[direction] and _safe_float(forecast.get("confidence")) >= 35:
            rows.append(horizon)
    return rows


def _has_higher_timeframe_conflict(forecasts: list[dict[str, Any]], conflicts: list[str]) -> bool:
    high = {"4h", "1d", "7d"}
    return any(str(item) in high for item in conflicts) or any(str(item.get("horizon")) in high and str(item.get("direction")) in {"bullish", "bearish"} for item in forecasts)


def _component_breadth(components: dict[str, Any]) -> float:
    numeric = [abs(float(value)) for value in components.values() if isinstance(value, (int, float))]
    if not numeric:
        return 0.0
    active = sum(1 for value in numeric if value >= 0.08)
    return _clamp(active / max(len(numeric), 1), 0.0, 1.0)


def _summary_text(direction: str, readiness: str, conflict: bool, high_vol: bool, stale: bool) -> str:
    if stale:
        return "Data stale - waiting for fresh market update before improving automation readiness."
    if conflict:
        if readiness == "notReady":
            return "Mixed horizon signal detected; automation readiness is not ready."
        return "Mixed horizon signal detected; automation readiness remains cautious."
    if high_vol:
        return "Forecast confidence reduced due to volatility across active horizons."
    if readiness == "ready" and direction in {"bullish", "bearish"}:
        return f"Market structure is coherently {direction} across evaluated horizons."
    if direction == "neutral":
        return "Market structure remains neutral; no forced trade signal is being surfaced."
    return f"Market structure leans {direction}, with automation readiness set to caution."


def _cycle_reasoning(selected: dict[str, Any] | None, forecasts: list[dict[str, Any]], coherence: dict[str, Any]) -> list[str]:
    reasoning: list[str] = []
    if selected and isinstance(selected.get("reasoning"), list):
        reasoning.extend(str(item) for item in selected.get("reasoning", []) if str(item))
    reasoning.extend(str(item) for item in (coherence.get("riskNotes", []) or []) if str(item))
    primary = str(coherence.get("primaryHorizon") or "")
    for forecast in forecasts:
        if primary and str(forecast.get("horizon")) != primary:
            continue
        reasoning.extend(str(item) for item in (forecast.get("reasoning", []) or []) if str(item))
    if not reasoning and coherence.get("summary"):
        reasoning.append(str(coherence["summary"]))
    return list(dict.fromkeys(reasoning))[:8]


def _select_cycle_forecast(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {}

    def score(item: dict[str, Any]) -> float:
        coherence = item.get("coherence_summary") if isinstance(item.get("coherence_summary"), dict) else {}
        return (
            _safe_float(coherence.get("coherenceScore")) * 0.45
            + _safe_float(coherence.get("overallConfidence")) * 0.35
            + _safe_float(item.get("confidence")) * 100.0 * 0.20
        )

    return max(candidates, key=score)


def _latest_timestamp(values: list[Any]) -> datetime | None:
    parsed = [_parse_datetime(value) for value in values if value]
    rows = [item for item in parsed if item is not None]
    return max(rows) if rows else None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _aware(value)
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return _aware(datetime.fromisoformat(raw))
    except ValueError:
        return None


def _aware(value: datetime | None = None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        return resolved.replace(tzinfo=timezone.utc)
    return resolved.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return _aware(value).isoformat()


def _normalize_component(value: float) -> float:
    return _clamp(value, -1.0, 1.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed != parsed:
        return default
    return parsed


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _phase_label(phase: str) -> str:
    labels = {
        "idle": "Idle",
        "collectingData": "Collecting data",
        "analyzing": "Analyzing",
        "scoringStrategies": "Scoring strategies",
        "runningForecast": "Running forecast",
        "checkingCoherence": "Checking coherence",
        "ready": "Ready",
        "staleData": "Stale data",
        "error": "Error",
    }
    return labels.get(phase, phase)


def _cycle_status_message(phase: str, coherence: dict[str, Any]) -> str:
    if phase == "staleData":
        return "Data stale - waiting for fresh market update"
    if phase == "error":
        return "Cycle status update failed"
    if phase == "collectingData":
        return "Collecting latest market data"
    if phase == "analyzing":
        return "Analyzing market structure across horizons"
    if phase == "ready":
        readiness = str(coherence.get("automationReadiness") or "caution")
        return f"Automation readiness: {readiness[:1].upper()}{readiness[1:]}"
    return "1H10 Vault Cycle status updating"
