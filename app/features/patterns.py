"""Pattern-model interface with deterministic candle-pattern implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class PatternPrediction:
    """Model prediction payload used for logging and future risk modifiers."""

    probability: float
    confidence: float
    label: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "probability": self.probability,
            "confidence": self.confidence,
            "label": self.label,
        }


class PatternModel(Protocol):
    """Protocol for future ML pattern models."""

    name: str

    def predict(self, candles: list[dict[str, Any]]) -> PatternPrediction:
        ...


class StubPatternModel:
    """Deterministic candle-pattern model used until an ML model is configured."""

    name = "StubPatternModel"

    def predict(self, candles: list[dict[str, Any]]) -> PatternPrediction:
        window = [self._normalize_candle(candle) for candle in candles[-8:]]
        window = [candle for candle in window if candle is not None]

        if len(window) < 5:
            return PatternPrediction(probability=0.5, confidence=0.0, label="neutral")

        first = window[0]["close"]
        last = window[-1]
        previous = window[-2]["close"]
        if first <= 0 or previous <= 0:
            return PatternPrediction(probability=0.5, confidence=0.0, label="neutral")

        ranges = [max(candle["high"] - candle["low"], 0.0) for candle in window]
        average_range = sum(ranges) / len(ranges) if ranges else 0.0
        body = last["close"] - last["open"]
        body_strength = abs(body) / max(average_range, 1e-9)
        momentum = (last["close"] - first) / first
        one_bar_change = (last["close"] - previous) / previous

        confidence = self._clamp((abs(momentum) * 40) + (body_strength * 0.25), 0.0, 1.0)
        if confidence < 0.15 or abs(momentum) < 0.001:
            return PatternPrediction(probability=0.5, confidence=0.0, label="neutral")

        if momentum > 0 and body > 0 and one_bar_change >= -0.001:
            return PatternPrediction(
                probability=round(self._clamp(0.5 + confidence * 0.25, 0.5, 0.75), 8),
                confidence=round(confidence, 8),
                label="bullish",
            )
        if momentum < 0 and body < 0 and one_bar_change <= 0.001:
            return PatternPrediction(
                probability=round(self._clamp(0.5 - confidence * 0.25, 0.25, 0.5), 8),
                confidence=round(confidence, 8),
                label="bearish",
            )

        return PatternPrediction(probability=0.5, confidence=0.0, label="neutral")

    @staticmethod
    def _normalize_candle(candle: dict[str, Any]) -> dict[str, float] | None:
        open_price = StubPatternModel._safe_float(candle.get("open"))
        high = StubPatternModel._safe_float(candle.get("high"))
        low = StubPatternModel._safe_float(candle.get("low"))
        close = StubPatternModel._safe_float(candle.get("close"))

        if close <= 0:
            return None

        if open_price <= 0:
            open_price = close

        if high <= 0:
            high = max(open_price, close)

        if low <= 0:
            low = min(open_price, close)

        if low > high:
            low, high = high, low

        return {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        }

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(value, upper))
