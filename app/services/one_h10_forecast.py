"""1H10-only forecast policy over persisted leveraged-market features."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..ml.online_ranker import ONE_H10_HORIZON
from .provider_assets import normalize_provider
from .vault_coherence import (
    build_horizon_forecasts,
    calculate_coherence_summary,
    format_vault_cycle_status,
    score_horizon_strategy,
)


class OneH10ForecastService:
    """Build auditable 1H10 forecast suggestions without owning execution."""

    def __init__(self, config: dict[str, Any], ml_decision_engine: Any | None = None, vault_coherence: Any | None = None) -> None:
        self.config = config
        self.ml_decision_engine = ml_decision_engine
        self.vault_coherence = vault_coherence

    def forecast(
        self,
        features: dict[str, Any],
        *,
        provider: str,
        symbol: str,
        allocation_cap_usd: float = 0.0,
        available_margin_usd: float = 0.0,
        market: Any | None = None,
    ) -> dict[str, Any]:
        """Return a 1h10 forecast envelope for scanner, strategy, and audit metadata."""

        provider_key = normalize_provider(provider)
        feature_payload = dict(features or {})
        context = {
            **feature_payload,
            "provider": provider_key,
            "execution_venue": provider_key,
            "symbol": str(symbol or feature_payload.get("symbol") or ""),
            "horizon": ONE_H10_HORIZON,
            "ml_horizon": ONE_H10_HORIZON,
            "objective": "one_h10",
            "one_h10_vault": True,
            "allocation_cap_usd": self._safe_float(allocation_cap_usd),
            "available_margin_usd": self._safe_float(available_margin_usd),
            "market_max_leverage": self._safe_float(getattr(market, "max_leverage", feature_payload.get("max_leverage", 1.0)), 1.0),
            "settlement_asset": getattr(market, "settlement_asset", feature_payload.get("settlement_asset", "")),
            "forecast_source": "one_h10_forecast",
        }
        ml_suite = self._ml_suite_decisions(context, provider_key)
        ml_decision = ml_suite.get("pytorch_fibonacci") or self._ml_decision(context, provider_key)
        fallback = self._deterministic_forecast(context)
        feature_blockers = self._feature_blockers(context)
        raw = ml_decision.get("raw") if isinstance(ml_decision.get("raw"), dict) else {}
        ready = self._suite_has_ready_decision(ml_suite) or bool(ml_decision.get("ready", False))
        advisory_blockers: list[str] = []
        if self._feature_blockers_are_advisory():
            advisory_blockers.extend(feature_blockers)
            base_blockers = list(dict.fromkeys(fallback.get("blockers", [])))
        else:
            base_blockers = list(dict.fromkeys([*fallback.get("blockers", []), *feature_blockers]))
        forecast = {
            **fallback,
            "ml_namespace": ONE_H10_HORIZON,
            "ml_horizon": ONE_H10_HORIZON,
            "objective": "one_h10",
            "provider": provider_key,
            "symbol": context["symbol"],
            "source": "one_h10_ml_profit_suite" if ready else "one_h10_bootstrap_forecast",
            "ml_ready": ready,
            "ml_decision": ml_decision,
            "ml_policy_decisions": ml_suite,
            "blockers": base_blockers,
            "advisory_blockers": advisory_blockers,
            "created_at": datetime.utcnow().isoformat(),
        }
        if ready:
            coerced = self._coerce_ml_suite(ml_suite, fallback, context)
            advisory_blockers.extend(coerced.pop("advisory_blockers", []) or [])
            raw_blockers = list(coerced.pop("blockers", []) or [])
            forecast.update(coerced)
            forecast["blockers"] = list(dict.fromkeys([*base_blockers, *raw_blockers]))
        else:
            if self._bootstrap_live_enabled():
                advisory_blockers.append("ml_not_ready")
            else:
                forecast["blockers"] = list(dict.fromkeys([*forecast["blockers"], "ml_not_ready"]))
        strict_ml = bool(self.config.get("ONE_H10_REQUIRE_PROMOTED_ML", True))
        if forecast["predicted_side"] == "hold" and "forecast_hold" not in forecast["blockers"]:
            if strict_ml or not self._bootstrap_live_enabled():
                forecast["blockers"].append("forecast_hold")
            else:
                advisory_blockers.append("forecast_hold")
        if self._safe_float(forecast.get("confidence")) < self._safe_float(self.config.get("ONE_H10_MIN_FORECAST_CONFIDENCE"), 0.25):
            if strict_ml:
                forecast["blockers"].append("low_confidence")
            else:
                advisory_blockers.append("low_confidence")
        forecast["blockers"] = list(dict.fromkeys(str(item) for item in forecast["blockers"] if str(item)))
        forecast["advisory_blockers"] = list(dict.fromkeys(str(item) for item in advisory_blockers if str(item)))
        forecast.update(self._coherence_payload(context, forecast))
        return forecast

    def _feature_blockers(self, context: dict[str, Any]) -> list[str]:
        blockers: list[str] = []
        required = self.config.get("ONE_H10_REQUIRED_FEATURE_TIMEFRAMES", ["15m", "1h", "4h"])
        if isinstance(required, str):
            required = [item.strip() for item in required.split(",") if item.strip()]
        required_set = {str(item).strip() for item in (required or []) if str(item).strip()}
        present = {
            str(item).strip()
            for item in (context.get("one_h10_feature_timeframes", []) or [])
            if str(item).strip()
        }
        if required_set and not required_set.issubset(present):
            blockers.append("features_stale")
        if self._features_are_stale(context):
            blockers.append("features_stale")
        fib = context.get("fibonacci_confluence") if isinstance(context.get("fibonacci_confluence"), dict) else {}
        levels = context.get("fibonacci_levels") if isinstance(context.get("fibonacci_levels"), dict) else {}
        timing = context.get("fibonacci_timing") if isinstance(context.get("fibonacci_timing"), dict) else {}
        if not fib or not levels or not timing:
            blockers.append("missing_fibonacci_features")
        return blockers

    def _coherence_payload(self, context: dict[str, Any], forecast: dict[str, Any]) -> dict[str, Any]:
        try:
            if self.vault_coherence is not None:
                horizon_forecasts = self.vault_coherence.build_horizon_forecasts(context)
                horizon_strategy_scores = [
                    self.vault_coherence.score_horizon_strategy(str(row.get("horizon")), row, context)
                    for row in horizon_forecasts
                ]
                coherence_summary = self.vault_coherence.calculate_coherence_summary(horizon_forecasts, horizon_strategy_scores)
                cycle_status = self.vault_coherence.format_vault_cycle_status(
                    {
                        "created_at": forecast.get("created_at"),
                        "last_completed_vault_cycle": forecast.get("created_at"),
                    },
                    horizon_forecasts=horizon_forecasts,
                    horizon_strategy_scores=horizon_strategy_scores,
                    coherence_summary=coherence_summary,
                )
            else:
                horizon_forecasts = build_horizon_forecasts(context, config=self.config)
                horizon_strategy_scores = [
                    score_horizon_strategy(str(row.get("horizon")), row, context, config=self.config)
                    for row in horizon_forecasts
                ]
                coherence_summary = calculate_coherence_summary(horizon_forecasts, horizon_strategy_scores)
                cycle_status = format_vault_cycle_status(
                    {
                        "created_at": forecast.get("created_at"),
                        "last_completed_vault_cycle": forecast.get("created_at"),
                    },
                    horizon_forecasts=horizon_forecasts,
                    horizon_strategy_scores=horizon_strategy_scores,
                    coherence_summary=coherence_summary,
                )
        except Exception as exc:  # noqa: BLE001
            horizon_forecasts = []
            horizon_strategy_scores = []
            coherence_summary = {
                "overallDirection": "neutral",
                "overallConfidence": 0,
                "coherenceScore": 0,
                "automationReadiness": "notReady",
                "primaryHorizon": None,
                "conflictingHorizons": [],
                "riskNotes": ["Forecast scoring unavailable"],
                "summary": "Forecast scoring unavailable; automation readiness is not ready.",
                "updatedAt": datetime.utcnow().isoformat(),
            }
            cycle_status = format_vault_cycle_status({}, error=str(exc))

        reasoning: list[str] = []
        reasoning.extend(str(item) for item in coherence_summary.get("riskNotes", []) or [] if str(item))
        primary = str(coherence_summary.get("primaryHorizon") or "")
        for row in horizon_forecasts:
            if primary and str(row.get("horizon")) != primary:
                continue
            reasoning.extend(str(item) for item in row.get("reasoning", []) or [] if str(item))
        if not reasoning and coherence_summary.get("summary"):
            reasoning.append(str(coherence_summary["summary"]))
        return {
            "horizon_forecasts": horizon_forecasts,
            "horizon_strategy_scores": horizon_strategy_scores,
            "coherence_summary": coherence_summary,
            "cycle_status": cycle_status,
            "reasoning": list(dict.fromkeys(reasoning))[:8],
        }

    def _ml_decision(self, context: dict[str, Any], provider: str) -> dict[str, Any]:
        if self.ml_decision_engine is None:
            return {
                "ready": False,
                "family": "pytorch_fibonacci",
                "horizon": ONE_H10_HORIZON,
                "provider": provider,
                "blockers": ["ml_decision_engine_unavailable"],
                "raw": {},
            }
        try:
            return dict(self.ml_decision_engine.decision("pytorch_fibonacci", context, horizon=ONE_H10_HORIZON))
        except Exception as exc:  # noqa: BLE001
            return {
                "ready": False,
                "family": "pytorch_fibonacci",
                "horizon": ONE_H10_HORIZON,
                "provider": provider,
                "blockers": [str(exc)],
                "raw": {},
            }

    def _ml_suite_decisions(self, context: dict[str, Any], provider: str) -> dict[str, dict[str, Any]]:
        families = self.config.get("ONE_H10_ML_FORECAST_FAMILIES", [])
        if isinstance(families, str):
            families = [item.strip() for item in families.split(",") if item.strip()]
        resolved = [str(item).strip() for item in (families or []) if str(item).strip()]
        if not resolved:
            resolved = ["pytorch_fibonacci"]
        suite: dict[str, dict[str, Any]] = {}
        for family in dict.fromkeys(resolved):
            if self.ml_decision_engine is None:
                suite[family] = {
                    "ready": False,
                    "family": family,
                    "horizon": ONE_H10_HORIZON,
                    "provider": provider,
                    "blockers": ["ml_decision_engine_unavailable"],
                    "raw": {},
                }
                continue
            try:
                suite[family] = dict(self.ml_decision_engine.decision(family, context, horizon=ONE_H10_HORIZON))
            except Exception as exc:  # noqa: BLE001
                suite[family] = {
                    "ready": False,
                    "family": family,
                    "horizon": ONE_H10_HORIZON,
                    "provider": provider,
                    "blockers": [str(exc)],
                    "raw": {},
                }
        return suite

    @staticmethod
    def _suite_has_ready_decision(suite: dict[str, dict[str, Any]]) -> bool:
        return any(bool(decision.get("ready", False)) and isinstance(decision.get("raw"), dict) for decision in suite.values())

    def _deterministic_forecast(self, context: dict[str, Any]) -> dict[str, Any]:
        close = self._safe_float(context.get("close"), 1.0)
        trend = self._safe_float(context.get("trend_strength"))
        ema_trend = self._safe_float(context.get("ema_trend")) / max(close, 1.0)
        macd = self._safe_float(context.get("macd_histogram")) / max(close, 1.0)
        imbalance = self._safe_float(context.get("order_book_imbalance"))
        rsi = self._safe_float(context.get("rsi"), 50.0)
        fib = context.get("fibonacci_confluence") if isinstance(context.get("fibonacci_confluence"), dict) else {}
        fib_score = self._safe_float(fib.get("score"))
        range_position = 0.5
        timing = context.get("fibonacci_timing") if isinstance(context.get("fibonacci_timing"), dict) else {}
        if timing:
            range_position = self._safe_float(timing.get("range_position"), 0.5)

        directional = (trend * 4.0) + (ema_trend * 40.0) + (macd * 30.0) + (imbalance * 0.8)
        if rsi < 35:
            directional += 0.4
        elif rsi > 72:
            directional -= 0.4
        if range_position < 0.382:
            directional += fib_score * 0.25
        elif range_position > 0.786:
            directional -= fib_score * 0.25

        confidence = min(max(abs(directional) / 3.0 + min(fib_score, 2.0) / 10.0, 0.0), 0.95)
        min_confidence = self._safe_float(self.config.get("ONE_H10_MIN_BOOTSTRAP_CONFIDENCE"), 0.03)
        directional_threshold = self._safe_float(self.config.get("ONE_H10_DIRECTIONAL_THRESHOLD"), 0.04)
        side = "hold"
        if confidence >= min_confidence and abs(directional) >= directional_threshold:
            side = "buy" if directional > 0 else "sell"

        volatility = max(self._safe_float(context.get("atr_pct")), self._safe_float(context.get("volatility")), 0.002)
        stop_pct = min(max(volatility * 1.5, 0.002), 0.035)
        take_pct = min(max(stop_pct * (1.8 + confidence), 0.004), 0.12)
        gross_expected_bps = max(abs(directional) * 75.0, take_pct * 10_000 * confidence)
        quality = self._market_quality(context, gross_expected_bps)
        confidence = max(0.0, min(confidence * quality["confidence_multiplier"], 1.0))
        budget = self._capital_budget(context)
        max_leverage = min(
            self._safe_float(self.config.get("MAX_LEVERAGE"), 1.0),
            self._safe_float(self.config.get("ONE_H10_MAX_LEVERAGE"), self._safe_float(self.config.get("MAX_LEVERAGE"), 1.0)),
            self._safe_float(context.get("market_max_leverage"), self._safe_float(self.config.get("MAX_LEVERAGE"), 1.0)),
        )
        aggressive_confidence = max(confidence, self._safe_float(self.config.get("ONE_H10_MIN_POSITION_FRACTION"), 0.20) if side != "hold" else 0.0)
        suggested_leverage = max(1.0, min(max_leverage, 1.0 + aggressive_confidence * max(max_leverage - 1.0, 0.0)))
        sizing_fraction = min(max(aggressive_confidence * quality["sizing_multiplier"], 0.0), 1.0)
        blockers = list(quality["blockers"])
        return {
            "predicted_side": side,
            "action": side,
            "horizon_seconds": 3600,
            "gross_expected_return_bps": gross_expected_bps,
            "expected_return_bps": quality["net_expected_return_bps"],
            "net_expected_return_bps": quality["net_expected_return_bps"],
            "cost_drag_bps": quality["cost_drag_bps"],
            "spread_bps": quality["spread_bps"],
            "execution_quality": quality["execution_quality"],
            "capital_efficiency_score": quality["capital_efficiency_score"],
            "expected_net_edge_passed": quality["net_expected_return_bps"] >= quality["min_edge_bps"],
            "confidence": confidence,
            "position_fraction": sizing_fraction,
            "suggested_notional_usd": budget * sizing_fraction,
            "suggested_leverage": suggested_leverage,
            "suggested_order_type": "limit" if quality["spread_bps"] > 4.0 or quality["cost_drag_bps"] > quality["max_cost_drag_bps"] else "market",
            "suggested_stop_loss_pct": stop_pct,
            "suggested_take_profit_pct": take_pct,
            "directional_score": directional,
            "blockers": blockers,
        }

    def _coerce_ml_suite(self, suite: dict[str, dict[str, Any]], fallback: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        raw_by_family = {
            family: dict(decision.get("raw") or {})
            for family, decision in (suite or {}).items()
            if isinstance(decision.get("raw"), dict)
        }
        ready_decisions = [decision for decision in suite.values() if bool(decision.get("ready", False))]
        advisory_blockers: list[str] = []
        hard_blockers: list[str] = []
        for family, decision in (suite or {}).items():
            blockers = [str(item) for item in (decision.get("blockers", []) or []) if str(item)]
            raw_blockers = [str(item) for item in (raw_by_family.get(family, {}).get("blockers", []) or []) if str(item)]
            if not blockers and not raw_blockers:
                continue
            if self._bootstrap_live_enabled():
                advisory_blockers.extend([*blockers, *raw_blockers])
            else:
                hard_blockers.extend([*blockers, *raw_blockers])

        fib_raw = raw_by_family.get("pytorch_fibonacci", {})
        roi_raw = raw_by_family.get("pytorch_roi_target", {})
        upside_raw = raw_by_family.get("pytorch_extreme_upside", {})
        cap_raw = raw_by_family.get("pytorch_cap_policy", {})
        exit_raw = raw_by_family.get("pytorch_exit_policy", {})
        execution_raw = raw_by_family.get("pytorch_execution_policy", {})
        risk_raw = raw_by_family.get("pytorch_risk_policy", {})
        optimizer_raw = raw_by_family.get("pytorch_optimizer_policy", {})

        fallback_confidence = self._safe_float(fallback.get("confidence"), 0.0)
        base_expected_bps = self._safe_float(fallback.get("gross_expected_return_bps"), fallback.get("expected_return_bps", 0.0))
        projected_roi_pct = max(
            self._safe_float(roi_raw.get("projected_roi_pct"), 0.0),
            self._safe_float(upside_raw.get("projected_roi_pct"), 0.0),
        )
        target_probability = max(
            0.0,
            min(self._safe_float(roi_raw.get("target_probability"), fallback_confidence), 1.0),
        )
        upside_probability = max(
            0.0,
            min(self._safe_float(upside_raw.get("extreme_upside_probability"), fallback_confidence), 1.0),
        )
        optimizer_score = max(-1.0, min(self._safe_float(optimizer_raw.get("optimizer_policy_score"), 0.0), 1.0))
        fib_quality = max(0.0, min(self._safe_float(fib_raw.get("target_zone_quality"), fallback_confidence), 1.0))
        projected_probability = max(target_probability, upside_probability, fib_quality, fallback_confidence)
        projected_edge_bps = projected_roi_pct * 100.0 * projected_probability
        edge_cap = self._safe_float(self.config.get("ONE_H10_ML_EXPECTED_EDGE_CAP_BPS"), 400.0)
        if edge_cap > 0:
            projected_edge_bps = min(projected_edge_bps, edge_cap)
        gross_expected_bps = max(base_expected_bps, projected_edge_bps)
        quality = self._market_quality(context, gross_expected_bps)
        confidence_values = [fallback_confidence]
        confidence_values.extend(self._safe_float(item.get("confidence"), -1.0) for item in ready_decisions)
        confidence_values.extend(
            [
                target_probability,
                upside_probability,
                fib_quality,
            ]
        )
        confidence = max(0.0, min(max(value for value in confidence_values if value >= 0), 1.0))
        confidence = max(0.0, min(confidence * quality["confidence_multiplier"], 1.0))
        expected_component = max(0.0, min(max(projected_roi_pct / 1000.0, gross_expected_bps / 10_000.0), 1.0))
        consensus = self._ml_consensus_metrics(
            suite,
            raw_by_family,
            fallback,
            confidence_values=[value for value in confidence_values if value >= 0],
            probability_values=[target_probability, upside_probability, fib_quality],
        )
        consensus_multiplier = self._safe_float(consensus.get("consensus_multiplier"), 1.0)
        confidence = max(0.0, min(confidence * consensus_multiplier, 1.0))
        profit_weight = max(0.0, min(self._safe_float(self.config.get("ONE_H10_ML_PROFIT_WEIGHT"), 0.65), 1.0))
        model_profit_score = max(
            0.0,
            min(
                (
                    target_probability * 0.30
                    + upside_probability * 0.25
                    + confidence * 0.20
                    + fib_quality * 0.10
                    + max(optimizer_score, 0.0) * 0.05
                    + expected_component * 0.10
                ),
                1.0,
            ),
        )
        profit_score = max(fallback_confidence * (1.0 - profit_weight), model_profit_score * profit_weight)
        profit_score = max(0.0, min(profit_score * quality["profit_multiplier"] * consensus_multiplier, 1.0))

        side = str(fallback.get("predicted_side") or fallback.get("action") or "hold").lower()
        if side not in {"buy", "sell"} and profit_score >= self._safe_float(self.config.get("ONE_H10_MIN_BOOTSTRAP_CONFIDENCE"), 0.03):
            directional = self._safe_float(fallback.get("directional_score"), 0.0)
            if directional != 0:
                side = "buy" if directional > 0 else "sell"
        if side not in {"buy", "sell"}:
            side = "hold"

        budget = self._capital_budget(context)
        leverage_cap = min(
            self._safe_float(self.config.get("MAX_LEVERAGE"), 1.0),
            self._safe_float(self.config.get("ONE_H10_MAX_LEVERAGE"), self._safe_float(self.config.get("MAX_LEVERAGE"), 1.0)),
            self._safe_float(context.get("market_max_leverage"), self._safe_float(self.config.get("MAX_LEVERAGE"), 1.0)),
        )
        suggested_leverage = max(
            self._safe_float(fallback.get("suggested_leverage"), 1.0),
            self._safe_float(cap_raw.get("suggested_leverage"), 1.0),
            self._safe_float(upside_raw.get("suggested_leverage"), 1.0),
            1.0 + profit_score * max(leverage_cap - 1.0, 0.0),
        )
        suggested_leverage = max(1.0, min(suggested_leverage, leverage_cap))
        suggested_leverage = 1.0 + (suggested_leverage - 1.0) * consensus_multiplier
        min_fraction = self._safe_float(self.config.get("ONE_H10_MIN_POSITION_FRACTION"), 0.20) if side != "hold" else 0.0
        position_fraction = max(self._safe_float(fallback.get("position_fraction"), 0.0), min_fraction, profit_score)
        position_fraction = max(0.0, min(position_fraction * quality["sizing_multiplier"], 1.0))
        position_fraction = max(0.0, min(position_fraction * consensus_multiplier, 1.0))
        notional_candidates = [
            self._safe_float(fallback.get("suggested_notional_usd"), 0.0),
            self._safe_float(cap_raw.get("suggested_notional_usdc"), 0.0),
            self._safe_float(upside_raw.get("suggested_notional_usdc"), 0.0),
            budget * position_fraction,
        ]
        suggested_notional = min(max(notional_candidates), budget) if budget > 0 else max(notional_candidates)
        suggested_notional *= quality["sizing_multiplier"] * consensus_multiplier

        stop_candidates = [
            self._safe_float(fallback.get("suggested_stop_loss_pct"), 0.0),
            self._safe_float(fib_raw.get("suggested_stop_loss_pct"), 0.0),
            self._safe_float(exit_raw.get("suggested_stop_loss_pct"), 0.0),
        ]
        stop_pct = max(value for value in stop_candidates if value > 0) if any(value > 0 for value in stop_candidates) else 0.0
        stop_pct = min(stop_pct, self._safe_float(self.config.get("ONE_H10_MAX_STOP_LOSS_PCT"), 0.08))
        take_candidates = [
            self._safe_float(fallback.get("suggested_take_profit_pct"), 0.0),
            self._safe_float(fib_raw.get("suggested_take_profit_pct"), 0.0),
            self._safe_float(exit_raw.get("suggested_take_profit_pct"), 0.0),
            stop_pct * (2.0 + profit_score * 4.0),
        ]
        take_pct = max(value for value in take_candidates if value > 0) if any(value > 0 for value in take_candidates) else 0.0
        take_pct = min(take_pct, self._safe_float(self.config.get("ONE_H10_MAX_TAKE_PROFIT_PCT"), 0.35))
        order_type = str(execution_raw.get("order_type_suggestion") or fallback.get("suggested_order_type") or "market").lower()
        if order_type not in {"market", "limit"}:
            order_type = "market"
        if quality["cost_drag_bps"] > quality["max_cost_drag_bps"] or quality["spread_bps"] > 4.0:
            order_type = "limit"
        if bool(optimizer_raw.get("skip_candidate")):
            advisory_blockers.append(str(optimizer_raw.get("skip_reason") or "ml_optimizer_policy_low_edge"))
        if risk_raw and risk_raw.get("approve") is False:
            advisory_blockers.append("ml_risk_policy_cautious")
        if self._safe_float(consensus.get("agreement_score"), 1.0) < self._safe_float(self.config.get("ONE_H10_MIN_MODEL_AGREEMENT"), 0.55):
            advisory_blockers.append("ml_model_disagreement")
        if self._safe_float(consensus.get("calibration_score"), 1.0) < 0.50:
            advisory_blockers.append("ml_calibration_weak")
        advisory_blockers.extend(quality["blockers"])

        return {
            "predicted_side": side,
            "action": side,
            "horizon_seconds": int(self._safe_float(fallback.get("horizon_seconds"), 3600)),
            "gross_expected_return_bps": gross_expected_bps,
            "expected_return_bps": quality["net_expected_return_bps"],
            "net_expected_return_bps": quality["net_expected_return_bps"],
            "cost_drag_bps": quality["cost_drag_bps"],
            "spread_bps": quality["spread_bps"],
            "execution_quality": quality["execution_quality"],
            "capital_efficiency_score": quality["capital_efficiency_score"],
            "expected_net_edge_passed": quality["net_expected_return_bps"] >= quality["min_edge_bps"],
            "confidence": confidence,
            "position_fraction": position_fraction,
            "suggested_notional_usd": max(suggested_notional, 0.0),
            "suggested_leverage": suggested_leverage,
            "suggested_order_type": order_type,
            "suggested_stop_loss_pct": max(stop_pct, 0.0),
            "suggested_take_profit_pct": max(take_pct, 0.0),
            "directional_score": fallback.get("directional_score", 0.0),
            "ml_profit_score": profit_score,
            "ml_model_profit_score": model_profit_score,
            "ml_cost_adjusted_profit_score": profit_score,
            "ml_agreement_score": consensus.get("agreement_score"),
            "ml_consensus_multiplier": consensus_multiplier,
            "ml_direction_agreement": consensus.get("direction_agreement"),
            "ml_probability_consistency": consensus.get("probability_consistency"),
            "ml_probability_range": consensus.get("probability_range"),
            "ml_calibration_score": consensus.get("calibration_score"),
            "ml_ready_family_count": consensus.get("ready_family_count"),
            "ml_total_family_count": consensus.get("total_family_count"),
            "ml_expected_edge_cap_bps": edge_cap,
            "roi_target_probability": target_probability,
            "extreme_upside_probability": upside_probability,
            "projected_roi_pct": projected_roi_pct,
            "target_roi_pct": self._safe_float(self.config.get("ONE_H10_TARGET_ROI_PCT"), 1000.0),
            "ml_policy_summary": {
                family: {
                    "ready": bool(decision.get("ready", False)),
                    "action": decision.get("action"),
                    "confidence": self._safe_float(decision.get("confidence"), 0.0),
                    "blockers": list(decision.get("blockers", []) or []),
                }
                for family, decision in (suite or {}).items()
            },
            "blockers": hard_blockers,
            "advisory_blockers": advisory_blockers,
        }

    def _ml_consensus_metrics(
        self,
        suite: dict[str, dict[str, Any]],
        raw_by_family: dict[str, dict[str, Any]],
        fallback: dict[str, Any],
        *,
        confidence_values: list[float],
        probability_values: list[float],
    ) -> dict[str, Any]:
        total_family_count = max(len(suite or {}), 1)
        ready_family_count = sum(1 for decision in (suite or {}).values() if bool(decision.get("ready", False)))
        readiness_score = max(0.0, min(ready_family_count / total_family_count, 1.0))

        direction_votes = [
            side
            for side in [
                self._side_vote(fallback.get("predicted_side") or fallback.get("action")),
                *[
                    self._side_vote(
                        raw.get("predicted_side")
                        or raw.get("action")
                        or raw.get("side")
                        or raw.get("direction")
                        or raw.get("order_side")
                    )
                    for raw in raw_by_family.values()
                ],
            ]
            if side in {"buy", "sell"}
        ]
        if len(direction_votes) <= 1:
            direction_agreement = 1.0
        else:
            direction_agreement = max(direction_votes.count("buy"), direction_votes.count("sell")) / len(direction_votes)

        bounded_probabilities = [
            max(0.0, min(self._safe_float(value), 1.0))
            for value in [*confidence_values, *probability_values]
            if 0.0 <= self._safe_float(value, -1.0) <= 1.0
        ]
        if len(bounded_probabilities) <= 1:
            probability_range = 0.0
            probability_consistency = 1.0
        else:
            probability_range = max(bounded_probabilities) - min(bounded_probabilities)
            probability_consistency = max(0.0, min(1.0 - probability_range, 1.0))

        calibration_risks: list[float] = []
        for family, decision in (suite or {}).items():
            raw = raw_by_family.get(family, {})
            calibration_risks.extend(
                self._safe_float(raw.get(key), -1.0)
                for key in (
                    "uncertainty",
                    "calibration_error",
                    "drift",
                    "drift_score",
                    "false_positive_rate",
                    "negative_error_rate",
                )
            )
            calibration_risks.append(self._safe_float(decision.get("uncertainty"), -1.0))
        bounded_risks = [max(0.0, min(value, 1.0)) for value in calibration_risks if value >= 0.0]
        calibration_risk = max(bounded_risks) if bounded_risks else 0.0
        calibration_score = max(0.0, min(1.0 - calibration_risk, 1.0))

        agreement_score = max(
            0.0,
            min(
                direction_agreement * 0.30
                + probability_consistency * 0.25
                + readiness_score * 0.20
                + calibration_score * 0.25,
                1.0,
            ),
        )
        agreement_weight = max(0.0, min(self._safe_float(self.config.get("ONE_H10_MODEL_AGREEMENT_WEIGHT"), 0.35), 1.0))
        consensus_multiplier = max(0.0, min(1.0 - (1.0 - agreement_score) * agreement_weight, 1.0))
        return {
            "agreement_score": agreement_score,
            "consensus_multiplier": consensus_multiplier,
            "direction_agreement": direction_agreement,
            "probability_consistency": probability_consistency,
            "probability_range": probability_range,
            "calibration_score": calibration_score,
            "calibration_risk": calibration_risk,
            "readiness_score": readiness_score,
            "ready_family_count": ready_family_count,
            "total_family_count": total_family_count,
        }

    @staticmethod
    def _side_vote(value: Any) -> str:
        side = str(value or "").strip().lower()
        if side in {"long", "open_long"}:
            return "buy"
        if side in {"short", "open_short"}:
            return "sell"
        return side if side in {"buy", "sell"} else ""

    def _market_quality(self, context: dict[str, Any], gross_expected_bps: float) -> dict[str, Any]:
        spread = max(self._safe_float(context.get("spread_bps")), 0.0)
        fee = self._safe_float(context.get("fee_bps"), -1.0)
        if fee <= 0:
            fee = self._safe_float(self.config.get("FEE_BPS"), 5.0)
        fee = max(fee, 0.0)
        slippage = max(
            self._safe_float(
                context.get("projected_slippage_bps"),
                self._safe_float(context.get("slippage_bps"), self.config.get("SIM_SLIPPAGE_BPS", 8.0)),
            ),
            0.0,
        )
        if slippage <= 0:
            slippage = max(self._safe_float(self.config.get("SIM_SLIPPAGE_BPS"), 8.0), 0.0)
        configured_cost = self._safe_float(context.get("cost_drag_bps"), -1.0)
        implied_cost = spread + fee * 2.0 + slippage
        cost_drag = max(configured_cost, implied_cost) if configured_cost >= 0 else implied_cost
        max_cost = max(1.0, self._safe_float(self.config.get("AGGRESSIVE_1H_MAX_COST_DRAG_BPS"), 18.0))
        min_edge = max(0.0, self._safe_float(self.config.get("NET_ROI_MIN_EDGE_BPS"), 4.0))
        net_expected = self._safe_float(gross_expected_bps) - cost_drag

        liquidity = max(
            self._safe_float(context.get("liquidity_capacity_usd")),
            self._safe_float(context.get("liquidity_usd")),
            0.0,
        )
        budget = self._capital_budget(context)
        min_liquidity = max(1.0, self._safe_float(self.config.get("ONE_H10_MIN_LIQUIDITY_USD"), self.config.get("VAULT_MIN_LIQUIDITY_USD", 1_000.0)))
        capacity_multiple = liquidity / max(budget, min_liquidity, 1.0) if liquidity > 0 else 1.0
        capital_efficiency = max(0.0, min(capacity_multiple / 10.0, 1.0))

        max_spread = max(1.0, self._safe_float(self.config.get("ONE_H10_MAX_SLIPPAGE_BPS"), self.config.get("VAULT_MAX_SLIPPAGE_BPS", 20.0)))
        spread_quality = max(0.0, min(1.0 - spread / max(max_spread, 1.0), 1.0))
        cost_quality = max(0.0, min(1.0 - max(cost_drag - min_edge, 0.0) / max(max_cost * 2.0, 1.0), 1.0))
        edge_quality = max(0.0, min(net_expected / max(min_edge * 8.0, 1.0), 1.0))
        execution_quality = max(0.0, min(cost_quality * 0.45 + spread_quality * 0.20 + capital_efficiency * 0.20 + edge_quality * 0.15, 1.0))

        blockers: list[str] = []
        if cost_drag > max_cost:
            blockers.append("cost_drag_above_threshold")
        if net_expected < min_edge:
            blockers.append("low_edge_after_costs")
        if liquidity > 0 and capital_efficiency < 0.25:
            blockers.append("low_liquidity_capacity")
        if bool(context.get("stale_data", False)):
            blockers.append("stale_market_data")

        confidence_multiplier = max(0.15, min(0.55 + execution_quality * 0.45, 1.0))
        sizing_multiplier = max(0.10, min(0.35 + execution_quality * 0.45 + edge_quality * 0.20, 1.0))
        if net_expected < min_edge:
            sizing_multiplier *= 0.50
        if cost_drag > max_cost:
            sizing_multiplier *= 0.75
        profit_multiplier = max(0.10, min(0.50 + execution_quality * 0.30 + edge_quality * 0.20, 1.0))
        return {
            "spread_bps": spread,
            "fee_bps": fee,
            "slippage_bps": slippage,
            "cost_drag_bps": cost_drag,
            "max_cost_drag_bps": max_cost,
            "min_edge_bps": min_edge,
            "gross_expected_return_bps": self._safe_float(gross_expected_bps),
            "net_expected_return_bps": net_expected,
            "execution_quality": execution_quality,
            "capital_efficiency_score": capital_efficiency,
            "capacity_multiple": capacity_multiple,
            "confidence_multiplier": confidence_multiplier,
            "sizing_multiplier": sizing_multiplier,
            "profit_multiplier": profit_multiplier,
            "blockers": blockers,
        }

    def _capital_budget(self, context: dict[str, Any]) -> float:
        allocation = self._safe_float(context.get("allocation_cap_usd"))
        margin = self._safe_float(context.get("available_margin_usd"))
        if allocation > 0 and margin > 0:
            return min(allocation, margin)
        return max(allocation, margin, 0.0)

    def _features_are_stale(self, context: dict[str, Any]) -> bool:
        raw_updated_at = str(context.get("one_h10_feature_updated_at") or context.get("updated_at") or "").strip()
        if not raw_updated_at:
            return False
        try:
            updated_at = datetime.fromisoformat(raw_updated_at.replace("Z", "+00:00"))
        except ValueError:
            return True
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        max_age = self._safe_float(
            self.config.get(
                "ONE_H10_MAX_FEATURE_AGE_SECONDS",
                self._safe_float(self.config.get("ONE_H10_FEATURE_REFRESH_SECONDS"), 3600.0) * 2.0,
            ),
            7200.0,
        )
        if max_age <= 0:
            return False
        return (datetime.now(timezone.utc) - updated_at.astimezone(timezone.utc)).total_seconds() > max_age

    def _coerce_ml_raw(self, raw: dict[str, Any], fallback: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        advisory_blockers = [str(item) for item in raw.get("blockers", [])] if isinstance(raw.get("blockers"), list) else []
        side = str(raw.get("predicted_side") or raw.get("action") or fallback.get("predicted_side") or "hold").lower()
        if side not in {"buy", "sell"}:
            side = str(fallback.get("predicted_side") or "hold").lower() if self._bootstrap_live_enabled() else "hold"
            if side not in {"buy", "sell"}:
                side = "hold"
        leverage_cap = min(
            self._safe_float(self.config.get("MAX_LEVERAGE"), 1.0),
            self._safe_float(self.config.get("ONE_H10_MAX_LEVERAGE"), self._safe_float(self.config.get("MAX_LEVERAGE"), 1.0)),
            self._safe_float(context.get("market_max_leverage"), self._safe_float(self.config.get("MAX_LEVERAGE"), 1.0)),
        )
        return {
            "predicted_side": side,
            "action": side,
            "horizon_seconds": int(self._safe_float(raw.get("horizon_seconds"), fallback.get("horizon_seconds", 3600))),
            "expected_return_bps": self._safe_float(raw.get("expected_return_bps"), fallback.get("expected_return_bps", 0.0)),
            "confidence": min(max(self._safe_float(raw.get("confidence"), fallback.get("confidence", 0.0)), 0.0), 1.0),
            "position_fraction": min(max(self._safe_float(raw.get("position_fraction"), fallback.get("position_fraction", 0.0)), 0.0), 1.0),
            "suggested_notional_usd": max(self._safe_float(raw.get("suggested_notional_usd"), fallback.get("suggested_notional_usd", 0.0)), 0.0),
            "suggested_leverage": max(1.0, min(self._safe_float(raw.get("suggested_leverage"), fallback.get("suggested_leverage", 1.0)), leverage_cap)),
            "suggested_order_type": "limit" if str(raw.get("suggested_order_type") or "").lower() == "limit" else "market",
            "suggested_stop_loss_pct": max(self._safe_float(raw.get("suggested_stop_loss_pct"), fallback.get("suggested_stop_loss_pct", 0.0)), 0.0),
            "suggested_take_profit_pct": max(self._safe_float(raw.get("suggested_take_profit_pct"), fallback.get("suggested_take_profit_pct", 0.0)), 0.0),
            "blockers": [] if self._bootstrap_live_enabled() else advisory_blockers,
            "advisory_blockers": advisory_blockers,
        }

    def _bootstrap_live_enabled(self) -> bool:
        return bool(self.config.get("ONE_H10_BOOTSTRAP_LIVE_ENABLED", True)) and not bool(
            self.config.get("ONE_H10_REQUIRE_PROMOTED_ML", True)
        )

    def _feature_blockers_are_advisory(self) -> bool:
        return self._bootstrap_live_enabled() and bool(
            self.config.get("ONE_H10_BOOTSTRAP_FEATURE_BLOCKERS_ADVISORY", True)
        )

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
