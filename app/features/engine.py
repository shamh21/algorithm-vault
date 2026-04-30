"""Deterministic feature snapshot builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .external import ExternalDataAdapter, default_external_adapters
from .fibonacci import FibonacciService
from .indicators import atr, bollinger_bands, ema, macd, rsi, sma, trend_strength, volatility, volume_spike
from .patterns import PatternModel, StubPatternModel


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Configuration for deterministic feature generation."""

    ema_fast_period: int = 8
    ema_slow_period: int = 21
    rsi_period: int = 7
    atr_period: int = 14
    volume_lookback: int = 20
    volume_spike_multiplier: float = 1.5
    fibonacci_lookback: int = 50
    fibonacci_confluence_lookbacks: tuple[int, ...] = (20, 50, 100)
    fibonacci_confluence_tolerance_bps: float = 18.0


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """Feature values for one symbol/timeframe/candle."""

    symbol: str
    timeframe: str
    timestamp: int
    close: float
    ema_fast: float
    ema_slow: float
    sma_fast: float
    sma_slow: float
    ema_trend: float
    trend_strength: float
    rsi: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    bollinger_bands: dict[str, Any]
    atr: float
    atr_pct: float
    volatility: float
    volume_spike: dict[str, Any]
    external_scores: dict[str, dict[str, Any]]
    pattern_prediction: dict[str, Any]
    fibonacci_levels: dict[str, Any]
    fibonacci_confluence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp,
            "close": self.close,
            "ema_fast": self.ema_fast,
            "ema_slow": self.ema_slow,
            "sma_fast": self.sma_fast,
            "sma_slow": self.sma_slow,
            "ema_trend": self.ema_trend,
            "trend_strength": self.trend_strength,
            "rsi": self.rsi,
            "macd_line": self.macd_line,
            "macd_signal": self.macd_signal,
            "macd_histogram": self.macd_histogram,
            "bollinger_bands": dict(self.bollinger_bands),
            "atr": self.atr,
            "atr_pct": self.atr_pct,
            "volatility": self.volatility,
            "volume_spike": dict(self.volume_spike),
            "external_scores": dict(self.external_scores),
            "pattern_prediction": dict(self.pattern_prediction),
            "fibonacci_levels": dict(self.fibonacci_levels),
            "fibonacci_confluence": dict(self.fibonacci_confluence),
        }


class FeatureEngine:
    """Builds deterministic feature snapshots from candle history."""

    def __init__(
        self,
        external_adapters: list[ExternalDataAdapter] | None = None,
        pattern_model: PatternModel | None = None,
        fibonacci_service: FibonacciService | None = None,
    ) -> None:
        self.external_adapters = external_adapters if external_adapters is not None else default_external_adapters()
        self.pattern_model = pattern_model or StubPatternModel()
        self.fibonacci_service = fibonacci_service or FibonacciService()

    def snapshot(
        self,
        *,
        symbol: str,
        timeframe: str,
        candles: list[dict[str, Any]],
        config: FeatureConfig | None = None,
    ) -> FeatureSnapshot:
        config = self._normalize_config(config or FeatureConfig())
        candles = self._valid_candles(candles)

        if not candles:
            return self._empty_snapshot(symbol, timeframe)

        closes = [candle["close"] for candle in candles]
        last = candles[-1]
        close = last["close"]

        fast = ema(closes, config.ema_fast_period)
        slow = ema(closes, config.ema_slow_period)
        fast_sma = sma(closes, config.ema_fast_period)
        slow_sma = sma(closes, config.ema_slow_period)
        current_atr = atr(candles, config.atr_period)
        current_volatility = volatility(closes, config.ema_slow_period)
        current_trend_strength = trend_strength(closes, config.ema_fast_period, config.ema_slow_period)
        current_macd = macd(closes)
        current_bollinger = bollinger_bands(closes)

        return FeatureSnapshot(
            symbol=str(symbol).upper(),
            timeframe=timeframe,
            timestamp=int(last.get("timestamp", 0)),
            close=round(close, 8),
            ema_fast=round(fast, 8),
            ema_slow=round(slow, 8),
            sma_fast=round(fast_sma, 8),
            sma_slow=round(slow_sma, 8),
            ema_trend=round(fast - slow, 8),
            trend_strength=round(current_trend_strength, 8),
            rsi=round(rsi(closes, config.rsi_period), 8),
            macd_line=round(current_macd["macd"], 8),
            macd_signal=round(current_macd["signal"], 8),
            macd_histogram=round(current_macd["histogram"], 8),
            bollinger_bands=self._round_bollinger(current_bollinger),
            atr=round(current_atr, 8),
            atr_pct=round(current_atr / close, 8) if close > 0 else 0.0,
            volatility=round(current_volatility, 8),
            volume_spike=self._round_volume_spike(
                volume_spike(candles, config.volume_lookback, config.volume_spike_multiplier)
            ),
            external_scores=self._external_scores(symbol),
            pattern_prediction=self._pattern_prediction(candles),
            fibonacci_levels=self._fibonacci_levels(candles, config.fibonacci_lookback),
            fibonacci_confluence=self._fibonacci_confluence(
                candles,
                close,
                config.fibonacci_confluence_lookbacks,
                config.fibonacci_confluence_tolerance_bps,
            ),
        )

    @property
    def external_status(self) -> list[dict[str, str]]:
        return [
            {
                "name": adapter.name,
                "status": getattr(adapter, "status", "stubbed_neutral"),
            }
            for adapter in self.external_adapters
        ]

    @property
    def pattern_status(self) -> dict[str, str]:
        return {
            "name": self.pattern_model.name,
            "status": getattr(self.pattern_model, "status", "stubbed_neutral"),
        }

    @staticmethod
    def config_from_parameters(parameters: dict[str, Any] | None = None) -> FeatureConfig:
        parameters = parameters or {}

        return FeatureEngine._normalize_config(
            FeatureConfig(
                ema_fast_period=FeatureEngine._safe_int(parameters.get("ema_fast_period", parameters.get("fast_period")), 8),
                ema_slow_period=FeatureEngine._safe_int(parameters.get("ema_slow_period", parameters.get("slow_period")), 21),
                rsi_period=FeatureEngine._safe_int(parameters.get("rsi_period", parameters.get("period")), 7),
                atr_period=FeatureEngine._safe_int(parameters.get("atr_period"), 14),
                volume_lookback=FeatureEngine._safe_int(parameters.get("volume_lookback"), 20),
                volume_spike_multiplier=FeatureEngine._safe_float(parameters.get("volume_spike_multiplier"), 1.5),
                fibonacci_lookback=FeatureEngine._safe_int(parameters.get("fibonacci_lookback"), 50),
                fibonacci_confluence_lookbacks=FeatureEngine._parse_int_tuple(
                    parameters.get("fibonacci_confluence_lookbacks"),
                    (20, 50, 100),
                ),
                fibonacci_confluence_tolerance_bps=FeatureEngine._safe_float(
                    parameters.get("fibonacci_confluence_tolerance_bps"),
                    18.0,
                ),
            )
        )

    def _external_scores(self, symbol: str) -> dict[str, dict[str, Any]]:
        scores: dict[str, dict[str, Any]] = {}

        for adapter in self.external_adapters:
            try:
                scores[adapter.name] = adapter.get_signal(symbol).as_dict()
            except Exception as exc:  # noqa: BLE001
                scores[adapter.name] = {
                    "score": 0.0,
                    "label": "neutral",
                    "error": str(exc),
                }

        return scores

    def _pattern_prediction(self, candles: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            return self.pattern_model.predict(candles).as_dict()
        except Exception as exc:  # noqa: BLE001
            return {
                "probability": 0.5,
                "confidence": 0.0,
                "label": "neutral",
                "error": str(exc),
            }

    def _fibonacci_levels(self, candles: list[dict[str, Any]], lookback: int) -> dict[str, Any]:
        try:
            return self.fibonacci_service.compute(candles, lookback).as_dict()
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def _fibonacci_confluence(
        self,
        candles: list[dict[str, Any]],
        price: float,
        lookbacks: tuple[int, ...],
        tolerance_bps: float,
    ) -> dict[str, Any]:
        try:
            return self.fibonacci_service.confluence(
                candles,
                price,
                lookbacks=list(lookbacks),
                tolerance_bps=tolerance_bps,
            ).as_dict()
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "score": 0.0}

    @staticmethod
    def _valid_candles(candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        valid: list[dict[str, Any]] = []

        for candle in candles:
            close = FeatureEngine._safe_float(candle.get("close"))

            if close <= 0:
                continue

            valid.append(
                {
                    **candle,
                    "open": FeatureEngine._safe_float(candle.get("open"), close),
                    "high": FeatureEngine._safe_float(candle.get("high"), close),
                    "low": FeatureEngine._safe_float(candle.get("low"), close),
                    "close": close,
                    "volume": FeatureEngine._safe_float(candle.get("volume")),
                    "timestamp": FeatureEngine._safe_int(candle.get("timestamp")),
                }
            )

        return valid

    @staticmethod
    def _normalize_config(config: FeatureConfig) -> FeatureConfig:
        fast = max(2, int(config.ema_fast_period))
        slow = max(fast + 1, int(config.ema_slow_period))

        return FeatureConfig(
            ema_fast_period=fast,
            ema_slow_period=slow,
            rsi_period=max(2, int(config.rsi_period)),
            atr_period=max(2, int(config.atr_period)),
            volume_lookback=max(2, int(config.volume_lookback)),
            volume_spike_multiplier=max(0.0, float(config.volume_spike_multiplier)),
            fibonacci_lookback=max(2, int(config.fibonacci_lookback)),
            fibonacci_confluence_lookbacks=tuple(
                sorted({max(2, int(value)) for value in config.fibonacci_confluence_lookbacks})
            ),
            fibonacci_confluence_tolerance_bps=max(1.0, float(config.fibonacci_confluence_tolerance_bps)),
        )

    @staticmethod
    def _round_volume_spike(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "is_spike": bool(payload.get("is_spike", False)),
            "ratio": round(FeatureEngine._safe_float(payload.get("ratio")), 8),
            "current": round(FeatureEngine._safe_float(payload.get("current")), 8),
            "average": round(FeatureEngine._safe_float(payload.get("average")), 8),
        }

    @staticmethod
    def _round_bollinger(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "middle": round(FeatureEngine._safe_float(payload.get("middle")), 8),
            "upper": round(FeatureEngine._safe_float(payload.get("upper")), 8),
            "lower": round(FeatureEngine._safe_float(payload.get("lower")), 8),
            "bandwidth": round(FeatureEngine._safe_float(payload.get("bandwidth")), 8),
            "percent_b": round(FeatureEngine._safe_float(payload.get("percent_b"), 0.5), 8),
        }

    @staticmethod
    def _empty_snapshot(symbol: str, timeframe: str) -> FeatureSnapshot:
        return FeatureSnapshot(
            symbol=str(symbol).upper(),
            timeframe=timeframe,
            timestamp=0,
            close=0.0,
            ema_fast=0.0,
            ema_slow=0.0,
            sma_fast=0.0,
            sma_slow=0.0,
            ema_trend=0.0,
            trend_strength=0.0,
            rsi=50.0,
            macd_line=0.0,
            macd_signal=0.0,
            macd_histogram=0.0,
            bollinger_bands={"middle": 0.0, "upper": 0.0, "lower": 0.0, "bandwidth": 0.0, "percent_b": 0.5},
            atr=0.0,
            atr_pct=0.0,
            volatility=0.0,
            volume_spike={"is_spike": False, "ratio": 0.0, "current": 0.0, "average": 0.0},
            external_scores={},
            pattern_prediction={"probability": 0.5, "confidence": 0.0, "label": "neutral"},
            fibonacci_levels={},
            fibonacci_confluence={
                "score": 0.0,
                "cluster_count": 0,
                "golden_zone_count": 0,
                "trend_bias": "flat",
                "levels": [],
            },
        )

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_int_tuple(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
        if value is None or value == "":
            return default
        if isinstance(value, str):
            raw_items = value.split(",")
        elif isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            raw_items = [value]
        parsed: list[int] = []
        for item in raw_items:
            try:
                parsed.append(int(item))
            except (TypeError, ValueError):
                continue
        return tuple(parsed) or default
