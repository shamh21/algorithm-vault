"""Explainable dashboard signal confidence and regime scoring."""

from __future__ import annotations

from datetime import datetime
from typing import Any


class DashboardPredictionService:
    """Builds probabilistic, explainable signal metadata for dashboard payloads."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def enrich_forecast(
        self,
        *,
        features: dict[str, Any] | None,
        forecast: dict[str, Any] | None,
        data_quality: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        features = dict(features or {})
        forecast = dict(forecast or {})
        data_quality = dict(data_quality or {})
        weighted = self.weighted_signal_score(features, forecast, data_quality=data_quality)
        regime = self.market_regime_classifier(features, data_quality=data_quality)
        decay = self.confidence_decay(forecast)
        liquidity_penalty = self.liquidity_risk_penalty(features)
        volatility_adjustment = self.volatility_risk_adjustment(features, data_quality=data_quality)
        quality_multiplier = max(0.20, min(self._safe_float(data_quality.get("score"), 70.0) / 100.0, 1.0))
        base_confidence = max(
            self._safe_float(forecast.get("confidence")),
            self._safe_float(weighted.get("raw_confidence")),
        )
        adjusted = (
            base_confidence
            * quality_multiplier
            * decay["multiplier"]
            * liquidity_penalty["multiplier"]
            * volatility_adjustment["multiplier"]
        )
        adjusted = max(0.0, min(adjusted, 1.0))
        suppressed = (
            adjusted < self._safe_float(self.config.get("ONE_H10_MIN_FORECAST_CONFIDENCE"), 0.25)
            or liquidity_penalty["suppressed"]
            or str(data_quality.get("state")) in {"poor", "stale", "insufficient"}
        )
        side = str(forecast.get("predicted_side") or forecast.get("action") or "hold").lower()
        if side not in {"buy", "sell"} or suppressed:
            side = "hold"
        confidence_score = int(round(adjusted * 100))
        explanation = self._explanation(
            features,
            forecast,
            weighted=weighted,
            regime=regime,
            data_quality=data_quality,
            decay=decay,
            liquidity_penalty=liquidity_penalty,
            volatility_adjustment=volatility_adjustment,
        )
        return {
            "predicted_side": side,
            "action": side,
            "confidence": adjusted,
            "confidence_score": confidence_score,
            "confidence_original": base_confidence,
            "confidence_kind": forecast.get("confidence_kind") or "weighted_dashboard_confidence",
            "confidence_calibrated": bool(forecast.get("confidence_calibrated", False)),
            "weighted_signal_score": weighted["score"],
            "weighted_signal_components": weighted["components"],
            "signal_quality": self._signal_quality(confidence_score, suppressed=suppressed),
            "market_regime": regime,
            "data_quality": data_quality,
            "explanation": explanation,
            "bullish_factors": explanation["bullish_factors"],
            "bearish_factors": explanation["bearish_factors"],
            "neutralizing_factors": explanation["neutralizing_factors"],
            "risk_penalties": explanation["risk_penalties"],
            "trend_continuation_probability": max(0, min(int(round(weighted["trend_probability"] * 100)), 100)),
            "no_trade_probability": max(0, min(int(round((1.0 - adjusted) * 100)), 100)),
            "forecast_expiry_seconds": int(
                self._safe_float(forecast.get("horizon_seconds"), self.config.get("ONE_H10_HORIZON_SECONDS", 3600))
            ),
        }

    def weighted_signal_score(
        self,
        features: dict[str, Any],
        forecast: dict[str, Any],
        *,
        data_quality: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data_quality = data_quality or {}
        trend = self._bounded(abs(self._safe_float(features.get("trend_strength"))) * 12.0)
        volatility = self._volatility_component(features, data_quality)
        volume = self._bounded(self._safe_float(features.get("volume_confirmation"), features.get("volume_zscore", 0.0)) / 2.0)
        momentum = self._momentum_alignment(features)
        timeframe = self._timeframe_confirmation(features, forecast)
        indicator = self._indicator_agreement(features)
        liquidity = self._liquidity_component(features)
        spread = self._spread_quality(features, forecast)
        drawdown = max(0.0, 1.0 - self._safe_float(features.get("max_drawdown"), features.get("drawdown_risk", 0.0)) * 4.0)
        freshness = self._bounded(self._safe_float(data_quality.get("score"), 70.0) / 100.0)
        weights = {
            "trend_strength": 0.15,
            "volatility_regime": 0.10,
            "volume_confirmation": 0.08,
            "momentum_alignment": 0.13,
            "timeframe_confirmation": 0.12,
            "rsi_macd_agreement": 0.10,
            "liquidity_score": 0.12,
            "spread_quality": 0.08,
            "drawdown_risk": 0.06,
            "freshness": 0.06,
        }
        values = {
            "trend_strength": trend,
            "volatility_regime": volatility,
            "volume_confirmation": volume,
            "momentum_alignment": momentum,
            "timeframe_confirmation": timeframe,
            "rsi_macd_agreement": indicator,
            "liquidity_score": liquidity,
            "spread_quality": spread,
            "drawdown_risk": drawdown,
            "freshness": freshness,
        }
        score = sum(values[key] * weights[key] for key in weights)
        confidence = max(self._safe_float(forecast.get("confidence")), score)
        return {
            "score": int(round(max(0.0, min(score, 1.0)) * 100)),
            "raw_confidence": max(0.0, min(confidence, 1.0)),
            "trend_probability": max(0.0, min((trend + momentum + timeframe) / 3.0, 1.0)),
            "components": [{"name": key, "score": int(round(values[key] * 100)), "weight": weights[key]} for key in weights],
        }

    def market_regime_classifier(self, features: dict[str, Any], *, data_quality: dict[str, Any] | None = None) -> dict[str, Any]:
        data_quality = data_quality or {}
        liquidity = self._liquidity_component(features)
        volatility = max(
            self._safe_float(features.get("atr_pct")),
            self._safe_float(features.get("volatility")),
        )
        trend = abs(self._safe_float(features.get("trend_strength"))) + abs(self._safe_float(features.get("ema_trend"))) / max(
            self._safe_float(features.get("close"), 1.0),
            1.0,
        )
        spread = max(self._safe_float(features.get("spread_bps")), self._safe_float(features.get("projected_slippage_bps")))
        state = "ranging"
        reasons: list[str] = []
        if str(data_quality.get("state")) in {"poor", "stale", "insufficient"}:
            state = "unstable_feed"
            reasons.append("market data quality is degraded")
        elif liquidity < 0.35:
            state = "low_liquidity"
            reasons.append("liquidity capacity is below the configured floor")
        elif volatility >= 0.035:
            state = "high_volatility"
            reasons.append("realized volatility is elevated")
        elif trend >= 0.035:
            state = "trending"
            reasons.append("trend and moving-average alignment are directional")
        elif spread >= self._safe_float(self.config.get("ONE_H10_MAX_SLIPPAGE_BPS"), 20.0) * 0.75:
            state = "news_sensitive"
            reasons.append("spread/slippage conditions are unstable")
        elif self._safe_float(features.get("breakout_score")) >= 0.65:
            state = "breakout"
            reasons.append("breakout features are elevated")
        else:
            reasons.append("market is range-bound or mixed")
        confidence_modifier = {
            "trending": 1.0,
            "breakout": 0.92,
            "ranging": 0.84,
            "news_sensitive": 0.72,
            "high_volatility": 0.68,
            "low_liquidity": 0.55,
            "unstable_feed": 0.38,
        }.get(state, 0.75)
        return {
            "state": state,
            "label": state.replace("_", " ").title(),
            "confidence_modifier": confidence_modifier,
            "reasons": reasons,
        }

    def confidence_decay(self, forecast: dict[str, Any]) -> dict[str, Any]:
        created_at = self._timestamp(forecast.get("created_at") or forecast.get("updated_at"))
        horizon = max(60.0, self._safe_float(forecast.get("horizon_seconds"), self.config.get("ONE_H10_HORIZON_SECONDS", 3600)))
        if created_at <= 0:
            return {"age_seconds": None, "multiplier": 0.82, "label": "unknown age"}
        age = max(0.0, datetime.utcnow().timestamp() - created_at)
        multiplier = max(0.35, min(1.0 - max(age - horizon * 0.25, 0.0) / max(horizon * 1.25, 1.0), 1.0))
        return {"age_seconds": age, "multiplier": multiplier, "label": "fresh" if multiplier >= 0.9 else "aging"}

    def liquidity_risk_penalty(self, features: dict[str, Any]) -> dict[str, Any]:
        score = self._liquidity_component(features)
        suppressed = score < 0.25
        return {
            "score": int(round(score * 100)),
            "multiplier": max(0.35, min(0.70 + score * 0.30, 1.0)),
            "suppressed": suppressed,
            "reason": "low liquidity" if suppressed else "liquidity acceptable",
        }

    def volatility_risk_adjustment(self, features: dict[str, Any], *, data_quality: dict[str, Any] | None = None) -> dict[str, Any]:
        state = str((data_quality or {}).get("market_volatility_state") or "")
        volatility = max(self._safe_float(features.get("atr_pct")), self._safe_float(features.get("volatility")))
        high = state == "high_volatility" or volatility >= 0.035
        low = state == "low_volatility" or (0 < volatility <= 0.0025)
        multiplier = 0.72 if high else 0.88 if low else 1.0
        return {
            "state": "high_volatility" if high else "low_volatility" if low else "normal",
            "multiplier": multiplier,
            "reason": "high volatility risk dampening" if high else "low volatility reduces follow-through" if low else "volatility normal",
        }

    def _explanation(
        self,
        features: dict[str, Any],
        forecast: dict[str, Any],
        *,
        weighted: dict[str, Any],
        regime: dict[str, Any],
        data_quality: dict[str, Any],
        decay: dict[str, Any],
        liquidity_penalty: dict[str, Any],
        volatility_adjustment: dict[str, Any],
    ) -> dict[str, Any]:
        side = str(forecast.get("predicted_side") or forecast.get("action") or "hold")
        bullish: list[str] = []
        bearish: list[str] = []
        neutral: list[str] = []
        penalties: list[str] = []
        directional = self._safe_float(forecast.get("directional_score"))
        if directional > 0:
            bullish.append("Directional model bias is positive")
        elif directional < 0:
            bearish.append("Directional model bias is negative")
        else:
            neutral.append("Directional bias is mixed")
        if self._indicator_agreement(features) >= 0.65:
            bullish.append("RSI and MACD are aligned")
        else:
            neutral.append("RSI/MACD agreement is limited")
        if self._timeframe_confirmation(features, forecast) >= 0.65:
            bullish.append("Timeframe confirmation is supportive")
        else:
            neutral.append("Timeframe confirmation is incomplete")
        if self._liquidity_component(features) < 0.35:
            penalties.append("Liquidity reduced confidence")
        if str(data_quality.get("state")) != "good":
            penalties.append(f"Data quality is {data_quality.get('state', 'degraded')}")
        if volatility_adjustment["multiplier"] < 1.0:
            penalties.append(volatility_adjustment["reason"])
        if decay["multiplier"] < 0.9:
            penalties.append("Signal age reduced confidence")
        return {
            "summary": f"{side.upper()} confidence reflects weighted market, liquidity, model, and data-quality inputs.",
            "bullish_factors": bullish[:4],
            "bearish_factors": bearish[:4],
            "neutralizing_factors": neutral[:5],
            "risk_penalties": list(dict.fromkeys(penalties))[:5],
            "confidence_contributors": weighted.get("components", []),
            "timeframe_agreement": self._timeframe_confirmation(features, forecast),
            "data_freshness": data_quality.get("signal_freshness", "unknown"),
            "provider_reliability": data_quality.get("provider_reliability", "unknown"),
            "volatility_condition": regime.get("label", "Unknown"),
            "expected_risk_reward_range": {
                "base": self._safe_float(forecast.get("risk_reward")),
                "low": max(0.0, self._safe_float(forecast.get("risk_reward")) * 0.75),
                "high": max(0.0, self._safe_float(forecast.get("risk_reward")) * 1.20),
            },
            "market_regime_reasons": regime.get("reasons", []),
            "liquidity_penalty": liquidity_penalty,
            "confidence_decay": decay,
        }

    @staticmethod
    def _signal_quality(score: int, *, suppressed: bool) -> dict[str, Any]:
        if suppressed:
            return {"grade": "No Trade", "tone": "neutral", "score": score}
        if score >= 80:
            return {"grade": "High", "tone": "positive", "score": score}
        if score >= 60:
            return {"grade": "Moderate", "tone": "watch", "score": score}
        return {"grade": "Low", "tone": "neutral", "score": score}

    def _timeframe_confirmation(self, features: dict[str, Any], forecast: dict[str, Any]) -> float:
        values: list[float] = []
        if isinstance(forecast.get("coherence_summary"), dict):
            values.append(self._safe_float(forecast["coherence_summary"].get("coherenceScore")) / 100.0)
            values.append(self._safe_float(forecast["coherence_summary"].get("overallConfidence")) / 100.0)
        values.append(self._safe_float(features.get("multi_timeframe_confirmation"), -1.0))
        values.append(
            self._safe_float(
                features.get("fibonacci_confluence", {}).get("score") if isinstance(features.get("fibonacci_confluence"), dict) else -1.0
            )
        )
        usable = [max(0.0, min(value, 1.0)) for value in values if value >= 0]
        return sum(usable) / len(usable) if usable else 0.5

    def _indicator_agreement(self, features: dict[str, Any]) -> float:
        rsi = self._safe_float(features.get("rsi"), 50.0)
        macd = self._safe_float(features.get("macd_histogram"))
        rsi_score = 1.0 - min(abs(rsi - 55.0) / 55.0, 1.0)
        macd_score = 0.65 if macd == 0 else min(abs(macd) / max(abs(self._safe_float(features.get("close"), 1.0)) * 0.004, 1e-9), 1.0)
        return max(0.0, min((rsi_score + macd_score) / 2.0, 1.0))

    def _momentum_alignment(self, features: dict[str, Any]) -> float:
        values = [
            abs(self._safe_float(features.get("momentum"))),
            abs(self._safe_float(features.get("trend_strength"))) * 8.0,
            abs(self._safe_float(features.get("ema_trend"))) / max(self._safe_float(features.get("close"), 1.0), 1.0) * 20.0,
        ]
        return max(0.0, min(max(values), 1.0))

    def _volatility_component(self, features: dict[str, Any], data_quality: dict[str, Any]) -> float:
        state = str(data_quality.get("market_volatility_state") or "")
        volatility = max(self._safe_float(features.get("atr_pct")), self._safe_float(features.get("volatility")))
        if state == "high_volatility" or volatility >= 0.035:
            return 0.45
        if state == "low_volatility" or (0 < volatility <= 0.0025):
            return 0.55
        return 0.80

    def _liquidity_component(self, features: dict[str, Any]) -> float:
        liquidity = max(self._safe_float(features.get("liquidity_capacity_usd")), self._safe_float(features.get("liquidity_usd")))
        floor = max(
            1.0, self._safe_float(self.config.get("ONE_H10_MIN_LIQUIDITY_USD"), self.config.get("VAULT_MIN_LIQUIDITY_USD", 50_000.0))
        )
        return max(0.0, min(liquidity / floor, 1.0))

    def _spread_quality(self, features: dict[str, Any], forecast: dict[str, Any]) -> float:
        spread = max(self._safe_float(features.get("spread_bps")), self._safe_float(forecast.get("spread_bps")))
        max_spread = max(1.0, self._safe_float(self.config.get("ONE_H10_MAX_SLIPPAGE_BPS"), 20.0))
        return max(0.0, min(1.0 - spread / max_spread, 1.0))

    @staticmethod
    def _bounded(value: float) -> float:
        return max(0.0, min(value, 1.0))

    @staticmethod
    def _timestamp(value: Any) -> float:
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, str) and value.strip():
            try:
                return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
            except ValueError:
                return 0.0
        return DashboardPredictionService._safe_float(value)

    @staticmethod
    def _safe_float(value: Any, default: Any = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            try:
                parsed = float(default)
            except (TypeError, ValueError):
                return 0.0
        return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else 0.0
