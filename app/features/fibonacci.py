"""Deterministic Fibonacci level calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RETRACEMENT_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
EXTENSION_RATIOS = (1.272, 1.618, 2.618)


@dataclass(frozen=True, slots=True)
class FibonacciLevels:
    """Fibonacci levels based on deterministic swing high/low selection."""

    swing_high: float
    swing_low: float
    trend: str
    lookback: int
    retracements: dict[str, float]
    extensions: dict[str, float]
    golden_zone: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "swing_high": self.swing_high,
            "swing_low": self.swing_low,
            "trend": self.trend,
            "lookback": self.lookback,
            "retracements": dict(self.retracements),
            "extensions": dict(self.extensions),
            "golden_zone": dict(self.golden_zone),
        }


@dataclass(frozen=True, slots=True)
class FibonacciConfluence:
    """Multi-lookback Fibonacci cluster score around the current price."""

    score: float
    price: float
    lookbacks: list[int]
    cluster_count: int
    golden_zone_count: int
    nearest_support: float | None
    nearest_resistance: float | None
    support_distance_bps: float
    resistance_distance_bps: float
    trend_bias: str
    levels: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "price": self.price,
            "lookbacks": list(self.lookbacks),
            "cluster_count": self.cluster_count,
            "golden_zone_count": self.golden_zone_count,
            "nearest_support": self.nearest_support,
            "nearest_resistance": self.nearest_resistance,
            "support_distance_bps": self.support_distance_bps,
            "resistance_distance_bps": self.resistance_distance_bps,
            "trend_bias": self.trend_bias,
            "levels": [dict(level) for level in self.levels],
        }


class FibonacciService:
    """Computes deterministic Fibonacci levels from recent candles."""

    def compute(self, candles: list[dict[str, Any]], lookback: int = 50) -> FibonacciLevels:
        lookback = max(self._safe_int(lookback, 50), 2)
        window = self._valid_window(candles, lookback)

        if not window:
            return self._empty_levels(lookback)

        high_index, high = max(
            enumerate(window),
            key=lambda item: (item[1]["high"], item[0]),
        )
        low_index, low = min(
            enumerate(window),
            key=lambda item: (item[1]["low"], item[0]),
        )

        swing_high = high["high"]
        swing_low = low["low"]
        trend = self._trend(high_index, low_index)
        distance = swing_high - swing_low

        if distance <= 0:
            return self._flat_levels(swing_high, swing_low, lookback)

        retracements = self._retracements(swing_high, swing_low, distance, trend)
        extensions = self._extensions(swing_high, swing_low, distance, trend)

        golden_values = sorted(
            [
                retracements.get("50.0", 0.0),
                retracements.get("61.8", 0.0),
            ]
        )

        return FibonacciLevels(
            swing_high=round(swing_high, 8),
            swing_low=round(swing_low, 8),
            trend=trend,
            lookback=lookback,
            retracements=retracements,
            extensions=extensions,
            golden_zone={
                "lower": golden_values[0],
                "upper": golden_values[1],
            },
        )

    def confluence(
        self,
        candles: list[dict[str, Any]],
        price: float,
        *,
        lookbacks: list[int] | tuple[int, ...] | None = None,
        tolerance_bps: float = 18.0,
    ) -> FibonacciConfluence:
        price = self._safe_float(price)
        normalized_lookbacks = self._normalize_lookbacks(lookbacks or (20, 50, 100))
        tolerance_bps = max(self._safe_float(tolerance_bps, 18.0), 1.0)

        if price <= 0:
            return self._empty_confluence(price, normalized_lookbacks)

        computed = [self.compute(candles, lookback) for lookback in normalized_lookbacks]
        levels: list[dict[str, Any]] = []
        trends: list[str] = []
        golden_zone_count = 0

        for fib in computed:
            if fib.swing_high <= 0 or fib.swing_low <= 0:
                continue
            trends.append(fib.trend)
            if self.price_in_golden_zone(price, fib):
                golden_zone_count += 1
            for label, level in {**fib.retracements, **fib.extensions}.items():
                parsed = self._safe_float(level)
                if parsed <= 0:
                    continue
                distance_bps = abs(parsed - price) / price * 10_000
                levels.append(
                    {
                        "lookback": fib.lookback,
                        "label": label,
                        "price": parsed,
                        "distance_bps": round(distance_bps, 4),
                        "side": "support" if parsed <= price else "resistance",
                    }
                )

        if not levels:
            return self._empty_confluence(price, normalized_lookbacks)

        cluster_count = len([level for level in levels if self._safe_float(level.get("distance_bps")) <= tolerance_bps])
        supports = [level for level in levels if self._safe_float(level.get("price")) <= price]
        resistances = [level for level in levels if self._safe_float(level.get("price")) >= price]
        nearest_support = max(supports, key=lambda item: self._safe_float(item.get("price")), default=None)
        nearest_resistance = min(resistances, key=lambda item: self._safe_float(item.get("price")), default=None)
        support_price = self._safe_float(nearest_support.get("price")) if nearest_support else 0.0
        resistance_price = self._safe_float(nearest_resistance.get("price")) if nearest_resistance else 0.0
        support_distance = abs(price - support_price) / price * 10_000 if support_price > 0 else 0.0
        resistance_distance = abs(resistance_price - price) / price * 10_000 if resistance_price > 0 else 0.0
        trend_bias = self._trend_bias(trends)
        trend_bonus = 0.15 if trend_bias in {"up", "down"} else 0.0
        score = min(
            1.0,
            cluster_count / max(len(normalized_lookbacks) * 2, 1)
            + golden_zone_count / max(len(normalized_lookbacks), 1) * 0.25
            + trend_bonus,
        )

        return FibonacciConfluence(
            score=round(score, 6),
            price=round(price, 8),
            lookbacks=normalized_lookbacks,
            cluster_count=cluster_count,
            golden_zone_count=golden_zone_count,
            nearest_support=round(support_price, 8) if support_price > 0 else None,
            nearest_resistance=round(resistance_price, 8) if resistance_price > 0 else None,
            support_distance_bps=round(support_distance, 4),
            resistance_distance_bps=round(resistance_distance, 4),
            trend_bias=trend_bias,
            levels=sorted(levels, key=lambda item: self._safe_float(item.get("distance_bps")))[:12],
        )

    @staticmethod
    def price_in_golden_zone(price: float, levels: FibonacciLevels) -> bool:
        price = FibonacciService._safe_float(price)
        lower = FibonacciService._safe_float(levels.golden_zone.get("lower"))
        upper = FibonacciService._safe_float(levels.golden_zone.get("upper"))

        return price > 0 and lower > 0 and upper > 0 and lower <= price <= upper

    @staticmethod
    def nearest_level(price: float, levels: FibonacciLevels, direction: str = "any") -> float | None:
        price = FibonacciService._safe_float(price)

        if price <= 0:
            return None

        values = [
            FibonacciService._safe_float(value)
            for value in [*levels.retracements.values(), *levels.extensions.values()]
        ]
        values = [value for value in values if value > 0]

        if not values:
            return None

        direction = str(direction).lower()

        if direction == "above":
            candidates = [value for value in values if value > price]
        elif direction == "below":
            candidates = [value for value in values if value < price]
        else:
            candidates = values

        if not candidates:
            return None

        return min(candidates, key=lambda value: abs(value - price))

    @classmethod
    def _retracements(
        cls,
        swing_high: float,
        swing_low: float,
        distance: float,
        trend: str,
    ) -> dict[str, float]:
        levels: dict[str, float] = {}

        for ratio in RETRACEMENT_RATIOS:
            if trend == "up":
                level = swing_high - distance * ratio
            elif trend == "down":
                level = swing_low + distance * ratio
            else:
                level = (swing_high + swing_low) / 2

            levels[cls._label(ratio)] = round(level, 8)

        return levels

    @classmethod
    def _extensions(
        cls,
        swing_high: float,
        swing_low: float,
        distance: float,
        trend: str,
    ) -> dict[str, float]:
        levels: dict[str, float] = {}

        for ratio in EXTENSION_RATIOS:
            if trend == "up":
                level = swing_high + distance * (ratio - 1)
            elif trend == "down":
                level = swing_low - distance * (ratio - 1)
            else:
                level = (swing_high + swing_low) / 2

            levels[cls._label(ratio)] = round(level, 8)

        return levels

    @staticmethod
    def _valid_window(candles: list[dict[str, Any]], lookback: int) -> list[dict[str, float]]:
        window = candles[-lookback:] if len(candles) >= lookback else candles[:]
        valid: list[dict[str, float]] = []

        for candle in window:
            high = FibonacciService._safe_float(candle.get("high"))
            low = FibonacciService._safe_float(candle.get("low"))

            if high <= 0 or low <= 0:
                continue

            if low > high:
                low, high = high, low

            valid.append({"high": high, "low": low})

        return valid

    @staticmethod
    def _trend(high_index: int, low_index: int) -> str:
        if low_index < high_index:
            return "up"

        if high_index < low_index:
            return "down"

        return "flat"

    @classmethod
    def _flat_levels(cls, swing_high: float, swing_low: float, lookback: int) -> FibonacciLevels:
        price = round(max(swing_high, swing_low, 0.0), 8)

        return FibonacciLevels(
            swing_high=price,
            swing_low=price,
            trend="flat",
            lookback=lookback,
            retracements={cls._label(ratio): price for ratio in RETRACEMENT_RATIOS},
            extensions={cls._label(ratio): price for ratio in EXTENSION_RATIOS},
            golden_zone={"lower": price, "upper": price},
        )

    @staticmethod
    def _empty_levels(lookback: int) -> FibonacciLevels:
        return FibonacciLevels(
            swing_high=0.0,
            swing_low=0.0,
            trend="flat",
            lookback=lookback,
            retracements={},
            extensions={},
            golden_zone={"lower": 0.0, "upper": 0.0},
        )

    @staticmethod
    def _empty_confluence(price: float, lookbacks: list[int]) -> FibonacciConfluence:
        return FibonacciConfluence(
            score=0.0,
            price=round(max(price, 0.0), 8),
            lookbacks=list(lookbacks),
            cluster_count=0,
            golden_zone_count=0,
            nearest_support=None,
            nearest_resistance=None,
            support_distance_bps=0.0,
            resistance_distance_bps=0.0,
            trend_bias="flat",
            levels=[],
        )

    @staticmethod
    def _normalize_lookbacks(lookbacks: list[int] | tuple[int, ...]) -> list[int]:
        normalized = sorted({max(int(FibonacciService._safe_float(value, 0)), 2) for value in lookbacks})
        return normalized or [20, 50, 100]

    @staticmethod
    def _trend_bias(trends: list[str]) -> str:
        up = len([trend for trend in trends if trend == "up"])
        down = len([trend for trend in trends if trend == "down"])
        if up > down:
            return "up"
        if down > up:
            return "down"
        return "flat"

    @staticmethod
    def _label(ratio: float) -> str:
        return f"{ratio * 100:.1f}"

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
