"""Base strategy interfaces and signal types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar


SUPPORTED_TIMEFRAMES = ("1m", "5m", "15m", "1h")
SUPPORTED_ACTIONS = ("buy", "sell", "hold", "reduce")


@dataclass(slots=True)
class Signal:
    """Structured strategy output consumed by the order manager."""

    action: str
    rationale: str
    timeframe: str
    stop_loss: float | None
    take_profit: float | None
    position_fraction: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported signal action: {self.action}")

        if self.timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {self.timeframe}")

        if not 0.0 <= self.position_fraction <= 1.0:
            raise ValueError("position_fraction must be between 0.0 and 1.0")

        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("stop_loss must be positive when provided")

        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError("take_profit must be positive when provided")

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "rationale": self.rationale,
            "timeframe": self.timeframe,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_fraction": self.position_fraction,
            "metadata": dict(self.metadata),
        }


class BaseStrategy:
    """Base class for plugin strategies."""

    name: ClassVar[str] = "base"
    description: ClassVar[str] = "Abstract strategy"
    default_parameters: ClassVar[dict[str, Any]] = {}

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        self.parameters = self.default_parameters.copy()
        self.parameters.update(parameters or {})

    @classmethod
    def parameter_schema(cls) -> dict[str, Any]:
        return cls.default_parameters.copy()

    @classmethod
    def parameter_grid(cls) -> dict[str, list[Any]]:
        """Return optimizer-ready parameter candidates."""

        return {key: [value] for key, value in cls.default_parameters.items()}

    @staticmethod
    def validate_timeframe(timeframe: str) -> None:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            supported = ", ".join(SUPPORTED_TIMEFRAMES)
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported: {supported}")

    def signal_metadata(
        self,
        *,
        symbol: str,
        timeframe: str,
        candles: list[dict[str, Any]],
        indicators: dict[str, Any] | None = None,
        thresholds: dict[str, Any] | None = None,
        risk: dict[str, Any] | None = None,
        components: dict[str, Any] | None = None,
        confidence: float = 0.0,
        no_trade_reason: str | None = None,
    ) -> dict[str, Any]:
        """Build consistent signal metadata without changing the Signal API."""

        latest = self._latest_candle(candles)
        metadata: dict[str, Any] = {
            "symbol": str(symbol).upper(),
            "timeframe": timeframe,
            "strategy": self.name,
            "signal_timestamp": latest.get("timestamp", 0),
            "timestamp": latest.get("timestamp", 0),
            "price": latest.get("close", 0.0),
            "confidence": max(0.0, min(self._safe_float(confidence), 1.0)),
            "indicators": indicators or {},
            "thresholds": thresholds or {},
            "risk": risk or {},
            "components": components or {},
        }
        if no_trade_reason:
            metadata["no_trade_reason"] = no_trade_reason
        return metadata

    @classmethod
    def _latest_candle(cls, candles: list[dict[str, Any]]) -> dict[str, Any]:
        if not candles:
            return {"timestamp": 0, "close": 0.0}
        last = candles[-1] if isinstance(candles[-1], dict) else {}
        close = cls._safe_float(last.get("close"))
        return {
            "timestamp": int(cls._safe_float(last.get("timestamp"))),
            "open": cls._safe_float(last.get("open"), close),
            "high": cls._safe_float(last.get("high"), close),
            "low": cls._safe_float(last.get("low"), close),
            "close": close,
            "volume": cls._safe_float(last.get("volume")),
        }

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def generate_signal(
        self,
        *,
        symbol: str,
        timeframe: str,
        candles: list[dict[str, Any]],
        position: dict[str, Any],
    ) -> Signal:
        raise NotImplementedError(f"{self.__class__.__name__} must implement generate_signal().")
