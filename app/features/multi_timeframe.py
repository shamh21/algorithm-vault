"""Multi-timeframe confluence scoring for short-horizon ensemble filters."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from .fibonacci import FibonacciService
from .indicators import rsi


@dataclass(frozen=True, slots=True)
class MultiTimeframeConfluence:
    score: float
    passed: bool
    threshold: float
    timeframes: list[str]
    cluster_count: int
    volume_confirmation: bool
    rsi_confirmation: bool
    trend_regime: str
    momentum_exhaustion: bool
    invalidation_distance_bps: float
    timeframe_scores: list[dict[str, Any]]
    skip_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "passed": self.passed,
            "threshold": self.threshold,
            "timeframes": list(self.timeframes),
            "cluster_count": self.cluster_count,
            "volume_confirmation": self.volume_confirmation,
            "rsi_confirmation": self.rsi_confirmation,
            "trend_regime": self.trend_regime,
            "momentum_exhaustion": self.momentum_exhaustion,
            "invalidation_distance_bps": self.invalidation_distance_bps,
            "timeframe_scores": [dict(item) for item in self.timeframe_scores],
            "skip_reason": self.skip_reason,
        }


class MultiTimeframeConfluenceService:
    """Scores Fibonacci, trend, volume, RSI, and exhaustion across higher timeframes."""

    def __init__(self, market_data: Any, config: dict[str, Any], fibonacci_service: FibonacciService | None = None) -> None:
        self.market_data = market_data
        self.config = config
        self.fibonacci_service = fibonacci_service or FibonacciService()

    def score(
        self,
        *,
        symbol: str,
        entry_timeframe: str,
        mode: str,
        price: float,
        side: str = "buy",
    ) -> MultiTimeframeConfluence:
        threshold = self._float(self.config.get("FIB_CONFLUENCE_THRESHOLD"), 0.55)
        timeframes = self._timeframes(entry_timeframe)
        lookbacks = self._lookbacks()
        price = self._float(price)
        if price <= 0:
            return self._empty(timeframes, threshold, "price_unavailable")

        timeframe_scores: list[dict[str, Any]] = []
        cluster_count = 0
        invalidation_distances: list[float] = []
        volume_confirmations = 0
        rsi_confirmations = 0
        trend_votes: list[str] = []
        exhaustion_votes = 0

        for timeframe in timeframes:
            candles = self._candles(symbol, timeframe, mode, max(max(lookbacks) + 5, 120))
            if len(candles) < min(max(lookbacks), 20):
                timeframe_scores.append({"timeframe": timeframe, "score": 0.0, "skip_reason": "insufficient_history"})
                continue

            confluence = self.fibonacci_service.confluence(
                candles,
                price,
                lookbacks=lookbacks,
                tolerance_bps=self._float(self.config.get("FIB_CONFLUENCE_CLUSTER_TOLERANCE_BPS"), 18.0),
            ).as_dict()
            closes = [self._float(row.get("close")) for row in candles if self._float(row.get("close")) > 0]
            volumes = [self._float(row.get("volume")) for row in candles if self._float(row.get("volume")) >= 0]
            current_rsi = rsi(closes, 14) if len(closes) >= 15 else 50.0
            trend = self._trend(closes)
            volume_ok = self._volume_confirmation(volumes)
            rsi_ok = self._rsi_confirmation(current_rsi, side)
            exhausted = self._momentum_exhausted(current_rsi, side)
            invalidation = self._invalidation_distance(confluence, side)

            cluster = int(self._float(confluence.get("cluster_count")))
            cluster_count += cluster
            if invalidation > 0:
                invalidation_distances.append(invalidation)
            if volume_ok:
                volume_confirmations += 1
            if rsi_ok:
                rsi_confirmations += 1
            if exhausted:
                exhaustion_votes += 1
            trend_votes.append(trend)

            directional_trend = (side == "buy" and trend == "up") or (side == "sell" and trend == "down")
            tf_score = min(
                1.0,
                self._float(confluence.get("score")) * 0.55
                + min(cluster / max(len(lookbacks) * 2, 1), 1.0) * 0.18
                + (0.10 if volume_ok else 0.0)
                + (0.10 if rsi_ok else 0.0)
                + (0.10 if directional_trend else 0.0)
                - (0.18 if exhausted else 0.0),
            )
            timeframe_scores.append(
                {
                    "timeframe": timeframe,
                    "score": round(max(0.0, tf_score), 6),
                    "fib_score": confluence.get("score", 0.0),
                    "cluster_count": cluster,
                    "rsi": round(current_rsi, 4),
                    "volume_confirmation": volume_ok,
                    "rsi_confirmation": rsi_ok,
                    "trend_regime": trend,
                    "momentum_exhaustion": exhausted,
                    "invalidation_distance_bps": round(invalidation, 4),
                }
            )

        valid_scores = [self._float(item.get("score")) for item in timeframe_scores if not item.get("skip_reason")]
        if not valid_scores:
            return self._empty(timeframes, threshold, "multi_timeframe_data_unavailable")

        dominant_trend = self._dominant_trend(trend_votes)
        score = min(
            1.0,
            mean(valid_scores) * 0.72
            + min(cluster_count / max(len(timeframes) * len(lookbacks) * 2, 1), 1.0) * 0.16
            + min(volume_confirmations / max(len(valid_scores), 1), 1.0) * 0.06
            + min(rsi_confirmations / max(len(valid_scores), 1), 1.0) * 0.06
            - min(exhaustion_votes / max(len(valid_scores), 1), 1.0) * 0.12,
        )
        score = round(max(0.0, score), 6)
        passed = score >= threshold and exhaustion_votes < len(valid_scores)
        skip_reason = "" if passed else "fib_confluence_below_threshold"
        return MultiTimeframeConfluence(
            score=score,
            passed=passed,
            threshold=threshold,
            timeframes=timeframes,
            cluster_count=cluster_count,
            volume_confirmation=volume_confirmations > 0,
            rsi_confirmation=rsi_confirmations > 0,
            trend_regime=dominant_trend,
            momentum_exhaustion=exhaustion_votes > 0,
            invalidation_distance_bps=round(min(invalidation_distances) if invalidation_distances else 0.0, 4),
            timeframe_scores=timeframe_scores,
            skip_reason=skip_reason,
        )

    def _empty(self, timeframes: list[str], threshold: float, reason: str) -> MultiTimeframeConfluence:
        return MultiTimeframeConfluence(
            score=0.0,
            passed=False,
            threshold=threshold,
            timeframes=timeframes,
            cluster_count=0,
            volume_confirmation=False,
            rsi_confirmation=False,
            trend_regime="unknown",
            momentum_exhaustion=False,
            invalidation_distance_bps=0.0,
            timeframe_scores=[],
            skip_reason=reason,
        )

    def _candles(self, symbol: str, timeframe: str, mode: str, limit: int) -> list[dict[str, Any]]:
        try:
            candles = self.market_data.get_candles(symbol, timeframe, mode=mode, limit=limit)
        except Exception:  # noqa: BLE001
            return []
        return candles if isinstance(candles, list) else []

    def _timeframes(self, entry_timeframe: str) -> list[str]:
        raw = self.config.get("FIB_MULTI_TIMEFRAMES", ["1h", "4h", "1d"])
        if isinstance(raw, str):
            raw = raw.split(",")
        values = [str(item).strip() for item in raw if str(item).strip()]
        return list(dict.fromkeys([entry_timeframe, *values]))

    def _lookbacks(self) -> list[int]:
        raw = self.config.get("FIB_MULTI_LOOKBACKS", self.config.get("FIB_CONFLUENCE_LOOKBACKS", [20, 50, 100]))
        if isinstance(raw, str):
            raw = raw.split(",")
        parsed: list[int] = []
        for value in raw:
            try:
                parsed.append(max(2, int(value)))
            except (TypeError, ValueError):
                continue
        return sorted(set(parsed or [20, 50, 100]))

    @staticmethod
    def _trend(closes: list[float]) -> str:
        if len(closes) < 6:
            return "unknown"
        move = (closes[-1] - closes[-6]) / max(closes[-6], 1e-9)
        if move > 0.002:
            return "up"
        if move < -0.002:
            return "down"
        return "range"

    @staticmethod
    def _dominant_trend(trends: list[str]) -> str:
        counts = {trend: trends.count(trend) for trend in {"up", "down", "range"}}
        return max(counts, key=counts.get) if any(counts.values()) else "unknown"

    @staticmethod
    def _volume_confirmation(volumes: list[float]) -> bool:
        if len(volumes) < 20:
            return False
        baseline = mean(volumes[-20:-1]) if len(volumes[-20:-1]) else 0.0
        return baseline > 0 and volumes[-1] >= baseline * 1.15

    @staticmethod
    def _rsi_confirmation(current_rsi: float, side: str) -> bool:
        if side == "buy":
            return 35.0 <= current_rsi <= 68.0
        return 32.0 <= current_rsi <= 65.0

    @staticmethod
    def _momentum_exhausted(current_rsi: float, side: str) -> bool:
        return current_rsi >= 78.0 if side == "buy" else current_rsi <= 22.0

    @staticmethod
    def _invalidation_distance(confluence: dict[str, Any], side: str) -> float:
        key = "support_distance_bps" if side == "buy" else "resistance_distance_bps"
        try:
            return float(confluence.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
