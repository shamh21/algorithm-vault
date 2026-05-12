"""Deterministic signal rule evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SignalDecision:
    """Rule evaluation result consumed by rule-based strategies."""

    action: str
    score: float
    long_score: float
    short_score: float
    rationale: str
    components: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "score": self.score,
            "long_score": self.long_score,
            "short_score": self.short_score,
            "rationale": self.rationale,
            "components": dict(self.components),
        }


class SignalRule:
    """Combines deterministic feature values into one entry/exit decision."""

    def __init__(self, parameters: dict[str, Any]) -> None:
        self.parameters = dict(parameters or {})

    def evaluate(self, features: dict[str, Any], position: dict[str, Any]) -> SignalDecision:
        close = self._safe_float(features.get("close"))
        ema_trend = self._safe_float(features.get("ema_trend"))
        rsi = self._safe_float(features.get("rsi"), 50.0)
        atr_pct = self._safe_float(features.get("atr_pct"))
        qty = self._safe_float(position.get("quantity"))

        if close <= 0:
            return self._decision("hold", 0.0, 0.0, "Invalid or missing close price.", features, 0.0)

        min_score = self._param("minimum_signal_score", 1.0)
        trend_weight = self._param("trend_weight", 0.55)
        rsi_weight = self._param("rsi_weight", 0.3)
        volume_weight = self._param("volume_weight", 0.15)
        external_weight = self._param("external_weight", 0.0)
        pattern_weight = self._param("pattern_weight", 0.1)
        fibonacci_weight = self._param("fibonacci_filter_weight", 0.05)

        max_atr_pct = self._param("max_atr_pct", 0.08)
        oversold = self._param("rsi_oversold", 35.0)
        overbought = self._param("rsi_overbought", 65.0)
        exit_rsi = self._param("exit_rsi", 50.0)

        trend_component = self._trend_component(ema_trend)
        rsi_component = self._rsi_component(rsi, oversold, overbought)
        volume_component = self._volume_component(features.get("volume_spike", {}))
        external_component = self._external_component(features.get("external_scores", {}))
        pattern_component = self._pattern_component(features.get("pattern_prediction", {}))
        fibonacci_component = self._fibonacci_component(features, close)

        long_score = 0.0
        short_score = 0.0

        if trend_component > 0:
            long_score += trend_weight
        elif trend_component < 0:
            short_score += trend_weight

        if rsi_component > 0:
            long_score += rsi_weight
        elif rsi_component < 0:
            short_score += rsi_weight

        if volume_component > 0:
            if trend_component > 0:
                long_score += volume_weight
            elif trend_component < 0:
                short_score += volume_weight

        if external_component > 0:
            long_score += external_weight * external_component
        elif external_component < 0:
            short_score += external_weight * abs(external_component)

        if pattern_component > 0:
            long_score += pattern_weight * pattern_component
        elif pattern_component < 0:
            short_score += pattern_weight * abs(pattern_component)

        if fibonacci_component > 0 and (long_score > 0 or short_score > 0):
            if trend_component >= 0:
                long_score += fibonacci_weight
            if trend_component <= 0:
                short_score += fibonacci_weight

        volatility_penalty = 0.35 if max_atr_pct > 0 and atr_pct > max_atr_pct else 0.0
        long_score = max(0.0, long_score - volatility_penalty)
        short_score = max(0.0, short_score - volatility_penalty)

        if qty > 0 and (ema_trend < 0 or rsi >= exit_rsi + 12):
            return self._decision("reduce", long_score, short_score, "Long exit rule triggered.", features, fibonacci_component)

        if qty < 0 and (ema_trend > 0 or rsi <= exit_rsi - 12):
            return self._decision("reduce", long_score, short_score, "Short exit rule triggered.", features, fibonacci_component)

        # Avoid adding to an existing same-direction position.
        if qty <= 0 and long_score >= min_score and long_score > short_score:
            return self._decision("buy", long_score, short_score, "Rule score favors long entry.", features, fibonacci_component)

        if qty >= 0 and short_score >= min_score and short_score > long_score:
            return self._decision("sell", long_score, short_score, "Rule score favors short entry.", features, fibonacci_component)

        return self._decision("hold", long_score, short_score, "No deterministic rule threshold met.", features, fibonacci_component)

    def _decision(
        self,
        action: str,
        long_score: float,
        short_score: float,
        rationale: str,
        features: dict[str, Any],
        fibonacci_component: float,
    ) -> SignalDecision:
        external = self._external_component(features.get("external_scores", {}))
        pattern = self._pattern_component(features.get("pattern_prediction", {}))

        return SignalDecision(
            action=action,
            score=round(max(long_score, short_score), 8),
            long_score=round(long_score, 8),
            short_score=round(short_score, 8),
            rationale=rationale,
            components={
                "ema_trend": round(self._safe_float(features.get("ema_trend")), 8),
                "rsi": round(self._safe_float(features.get("rsi"), 50.0), 8),
                "atr_pct": round(self._safe_float(features.get("atr_pct")), 8),
                "volume_spike": self._volume_component(features.get("volume_spike", {})),
                "external": round(external, 8),
                "pattern": round(pattern, 8),
                "fibonacci_filter": round(fibonacci_component, 8),
            },
        )

    def _param(self, key: str, default: float) -> float:
        return self._safe_float(self.parameters.get(key), default)

    @staticmethod
    def _trend_component(ema_trend: float) -> float:
        if ema_trend > 0:
            return 1.0
        if ema_trend < 0:
            return -1.0
        return 0.0

    @staticmethod
    def _rsi_component(rsi: float, oversold: float, overbought: float) -> float:
        if oversold >= overbought:
            return 0.0
        if rsi <= oversold:
            return 1.0
        if rsi >= overbought:
            return -1.0
        return 0.0

    @staticmethod
    def _volume_component(volume: Any) -> float:
        return 1.0 if isinstance(volume, dict) and volume.get("is_spike") else 0.0

    @staticmethod
    def _external_component(external_scores: Any) -> float:
        if not isinstance(external_scores, dict) or not external_scores:
            return 0.0

        weighted_scores: list[float] = []

        for payload in external_scores.values():
            if not isinstance(payload, dict):
                continue

            score = SignalRule._safe_float(payload.get("score"))
            confidence = SignalRule._safe_float(payload.get("confidence"), 1.0)
            weighted_scores.append(score * max(0.0, min(confidence, 1.0)))

        if not weighted_scores:
            return 0.0

        return max(-1.0, min(sum(weighted_scores) / len(weighted_scores), 1.0))

    @staticmethod
    def _pattern_component(pattern: Any) -> float:
        if not isinstance(pattern, dict):
            return 0.0

        probability = SignalRule._safe_float(pattern.get("probability"), 0.5)
        confidence = SignalRule._safe_float(pattern.get("confidence"), 0.0)

        directional_score = (probability - 0.5) * 2
        return max(-1.0, min(directional_score * max(0.0, min(confidence, 1.0)), 1.0))

    @staticmethod
    def _fibonacci_component(features: dict[str, Any], close: float) -> float:
        fibonacci = features.get("fibonacci_levels", {})
        zone = fibonacci.get("golden_zone", {}) if isinstance(fibonacci, dict) else {}

        lower = SignalRule._safe_float(zone.get("lower"))
        upper = SignalRule._safe_float(zone.get("upper"))

        if close > 0 and lower > 0 and upper > 0 and lower <= close <= upper:
            return 1.0

        return 0.0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default