"""Dashboard market data validation and quality scoring helpers."""

from __future__ import annotations

import math
import statistics
import time
from datetime import UTC, datetime
from typing import Any

TIMEFRAME_SECONDS = {
    "live": 60,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "45m": 2700,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


class MarketDataQualityService:
    """Normalizes chart data and produces dashboard-safe quality metadata."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def validate_candles(
        self,
        rows: list[dict[str, Any]] | None,
        *,
        timeframe: str = "1m",
        now: float | None = None,
    ) -> dict[str, Any]:
        expected = self._interval_seconds(timeframe)
        resolved_now = float(now if now is not None else time.time())
        duplicate_count = 0
        invalid_count = 0
        outlier_count = 0
        by_time: dict[int, dict[str, float]] = {}
        max_jump = max(0.05, self._safe_float(self.config.get("DASHBOARD_CANDLE_OUTLIER_MAX_MOVE_PCT"), 0.35))

        previous_close = 0.0
        for index, row in enumerate(rows or []):
            if not isinstance(row, dict):
                invalid_count += 1
                continue
            candle = self._normalized_candle(row, index=index, now=resolved_now, expected_interval=expected)
            if candle is None:
                invalid_count += 1
                continue
            if previous_close > 0:
                jump = abs(candle["close"] / previous_close - 1.0)
                if jump > max_jump:
                    outlier_count += 1
                    continue
            previous_close = candle["close"]
            candle_time = int(candle["time"])
            if candle_time in by_time:
                duplicate_count += 1
            by_time[candle_time] = candle

        candles = [by_time[key] for key in sorted(by_time)]
        gaps = self.detect_gaps(candles, timeframe=timeframe)
        stale = self.detect_stale_feed(candles=candles, timeframe=timeframe, now=resolved_now)
        completeness = self._completeness(candles, gaps=gaps, expected_interval=expected)
        returns = self._returns(candles)
        volatility = statistics.fmean(abs(value) for value in returns) if returns else 0.0
        volatility_state = self._volatility_state(volatility)
        score = self._quality_score(
            candles=candles,
            completeness=completeness,
            stale=bool(stale["stale"]),
            invalid_count=invalid_count,
            duplicate_count=duplicate_count,
            outlier_count=outlier_count,
            gap_count=len(gaps),
        )
        state = self._quality_state(score, stale=bool(stale["stale"]), candles=candles)
        issues = []
        if invalid_count:
            issues.append("invalid_candles_removed")
        if duplicate_count:
            issues.append("duplicate_candles_removed")
        if outlier_count:
            issues.append("outlier_candles_rejected")
        if gaps:
            issues.append("candle_gaps_detected")
        if stale["stale"]:
            issues.append("stale_feed")
        if not candles:
            issues.append("insufficient_candles")
        return {
            "candles": candles,
            "quality": {
                "score": score,
                "state": state,
                "issues": issues,
                "duplicate_count": duplicate_count,
                "invalid_count": invalid_count,
                "outlier_count": outlier_count,
                "gap_count": len(gaps),
                "gaps": gaps[:12],
                "expected_interval_seconds": expected,
                "candle_count": len(candles),
                "candle_completeness": completeness,
                "last_sync_age_seconds": stale["age_seconds"],
                "stale": bool(stale["stale"]),
                "market_volatility_state": volatility_state,
                "signal_freshness": "stale" if stale["stale"] else "fresh" if score >= 80 else "degraded",
            },
        }

    def detect_gaps(self, candles: list[dict[str, Any]], *, timeframe: str = "1m") -> list[dict[str, float]]:
        expected = self._interval_seconds(timeframe)
        gaps: list[dict[str, float]] = []
        previous: dict[str, Any] | None = None
        for candle in candles:
            if previous is None:
                previous = candle
                continue
            delta = self._safe_float(candle.get("time")) - self._safe_float(previous.get("time"))
            if delta > expected * 1.5:
                gaps.append(
                    {
                        "from": self._safe_float(previous.get("time")),
                        "to": self._safe_float(candle.get("time")),
                        "missing_intervals": max(1, int(round(delta / expected)) - 1),
                    }
                )
            previous = candle
        return gaps

    def score_data_quality(
        self,
        candles: list[dict[str, Any]] | None = None,
        *,
        features: dict[str, Any] | None = None,
        forecast: dict[str, Any] | None = None,
        timeframe: str = "1m",
        provider_latency_ms: float | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        payload = self.validate_candles(candles or [], timeframe=timeframe, now=now)
        quality = dict(payload["quality"])
        features = features or {}
        forecast = forecast or {}
        if not candles:
            freshness = self._feature_freshness(features, forecast)
            quality.update(
                {
                    "score": min(quality["score"], freshness["score"]),
                    "state": self._quality_state(freshness["score"], stale=freshness["stale"], candles=[{"time": time.time()}]),
                    "last_sync_age_seconds": freshness["age_seconds"],
                    "stale": freshness["stale"],
                    "signal_freshness": freshness["signal_freshness"],
                }
            )
            if freshness["stale"]:
                quality.setdefault("issues", []).append("stale_features")
        if provider_latency_ms is not None:
            quality["provider_latency_ms"] = max(0.0, self._safe_float(provider_latency_ms))
        else:
            quality.setdefault("provider_latency_ms", None)
        quality["provider_reliability"] = self._provider_reliability(quality)
        return quality

    def sanitize_forecast_payload(self, forecast: dict[str, Any] | None) -> dict[str, Any]:
        row = dict(forecast or {})
        side = str(row.get("predicted_side") or row.get("action") or "hold").lower()
        if side not in {"buy", "sell", "hold"}:
            side = "hold"
        row["predicted_side"] = side
        row["action"] = side
        row["confidence"] = max(0.0, min(self._safe_float(row.get("confidence")), 1.0))
        for key in (
            "expected_return_bps",
            "net_expected_return_bps",
            "gross_expected_return_bps",
            "suggested_stop_loss_pct",
            "suggested_take_profit_pct",
            "spread_bps",
            "estimated_slippage_bps",
            "risk_reward",
        ):
            if key in row:
                row[key] = self._safe_float(row.get(key))
        for key in ("blockers", "advisory_blockers", "decision_blockers", "reasoning"):
            value = row.get(key)
            row[key] = [str(item)[:180] for item in (value if isinstance(value, list) else []) if str(item)]
        return row

    def detect_stale_feed(
        self,
        *,
        candles: list[dict[str, Any]] | None = None,
        updated_at: Any | None = None,
        timeframe: str = "1m",
        now: float | None = None,
    ) -> dict[str, Any]:
        resolved_now = float(now if now is not None else time.time())
        expected = self._interval_seconds(timeframe)
        timestamp = self._timestamp(updated_at)
        if timestamp <= 0 and candles:
            timestamp = self._safe_float(candles[-1].get("time"))
        age = max(0.0, resolved_now - timestamp) if timestamp > 0 else None
        max_age = max(expected * 3.0, self._safe_float(self.config.get("DASHBOARD_STALE_FEED_SECONDS"), 180.0))
        return {
            "stale": age is None or age > max_age,
            "age_seconds": age,
            "max_age_seconds": max_age,
        }

    def _normalized_candle(
        self,
        row: dict[str, Any],
        *,
        index: int,
        now: float,
        expected_interval: int,
    ) -> dict[str, float] | None:
        timestamp = self._timestamp(row.get("time", row.get("timestamp", row.get("t"))))
        if timestamp <= 0:
            timestamp = now - max(1, 150 - index) * expected_interval
        close = self._finite(row.get("close", row.get("c", row.get("price"))))
        if close is None or close <= 0:
            return None
        open_value = self._finite(row.get("open", row.get("o"))) or close
        high = self._finite(row.get("high", row.get("h"))) or max(open_value, close)
        low = self._finite(row.get("low", row.get("l"))) or min(open_value, close)
        volume = self._finite(row.get("volume", row.get("v"))) or 0.0
        high = max(high, open_value, close)
        low = min(low, open_value, close)
        if low <= 0 or high <= 0:
            return None
        return {
            "time": float(timestamp),
            "open": float(open_value),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": max(float(volume), 0.0),
        }

    def _interval_seconds(self, timeframe: str) -> int:
        return TIMEFRAME_SECONDS.get(str(timeframe or "1m").lower(), 60)

    def _completeness(self, candles: list[dict[str, Any]], *, gaps: list[dict[str, float]], expected_interval: int) -> float:
        if not candles:
            return 0.0
        span = max(0.0, self._safe_float(candles[-1].get("time")) - self._safe_float(candles[0].get("time")))
        expected_count = max(1, int(round(span / max(expected_interval, 1))) + 1)
        missing = sum(int(row.get("missing_intervals", 0) or 0) for row in gaps)
        return max(0.0, min((len(candles) - missing * 0.35) / expected_count, 1.0))

    def _quality_score(
        self,
        *,
        candles: list[dict[str, Any]],
        completeness: float,
        stale: bool,
        invalid_count: int,
        duplicate_count: int,
        outlier_count: int,
        gap_count: int,
    ) -> int:
        if not candles:
            return 0
        score = 100.0 * max(0.0, min(completeness, 1.0))
        score -= min(28.0, gap_count * 7.0)
        score -= min(20.0, duplicate_count * 4.0)
        score -= min(24.0, invalid_count * 4.0)
        score -= min(30.0, outlier_count * 10.0)
        if stale:
            score -= 34.0
        return int(round(max(0.0, min(score, 100.0))))

    @staticmethod
    def _quality_state(score: int | float, *, stale: bool, candles: list[dict[str, Any]]) -> str:
        if not candles:
            return "insufficient"
        if stale:
            return "stale"
        if score >= 85:
            return "good"
        if score >= 65:
            return "degraded"
        return "poor"

    def _feature_freshness(self, features: dict[str, Any], forecast: dict[str, Any]) -> dict[str, Any]:
        raw = (
            features.get("one_h10_feature_updated_at")
            or features.get("updated_at")
            or forecast.get("created_at")
            or forecast.get("updated_at")
        )
        stale = self.detect_stale_feed(
            updated_at=raw,
            timeframe="1m",
            now=time.time(),
        )
        age = stale["age_seconds"]
        max_age = max(60.0, self._safe_float(self.config.get("ONE_H10_MAX_SIGNAL_AGE_SECONDS"), 3600.0))
        is_stale = age is None or age > max_age
        score = 58 if age is None else int(round(max(0.0, min(100.0 * (1.0 - age / max(max_age * 1.35, 1.0)), 100.0))))
        return {
            "stale": is_stale,
            "age_seconds": age,
            "score": score,
            "signal_freshness": "stale" if is_stale else "fresh" if score >= 80 else "degraded",
        }

    @staticmethod
    def _provider_reliability(quality: dict[str, Any]) -> str:
        score = int(quality.get("score", 0) or 0)
        if quality.get("stale"):
            return "stale"
        if score >= 85:
            return "strong"
        if score >= 65:
            return "degraded"
        return "weak"

    @staticmethod
    def _returns(candles: list[dict[str, Any]]) -> list[float]:
        values: list[float] = []
        previous = 0.0
        for candle in candles:
            close = MarketDataQualityService._safe_float(candle.get("close"))
            if previous > 0 and close > 0:
                values.append(close / previous - 1.0)
            previous = close
        return values

    @staticmethod
    def _volatility_state(volatility: float) -> str:
        if volatility >= 0.035:
            return "high_volatility"
        if volatility <= 0.0025:
            return "low_volatility"
        return "normal"

    @staticmethod
    def _timestamp(value: Any) -> float:
        if isinstance(value, datetime):
            resolved = value if value.tzinfo else value.replace(tzinfo=UTC)
            return resolved.timestamp()
        if isinstance(value, str) and value.strip():
            try:
                return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
        parsed = MarketDataQualityService._safe_float(value)
        if parsed > 10_000_000_000:
            parsed /= 1000.0
        return parsed

    @staticmethod
    def _finite(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    @staticmethod
    def _safe_float(value: Any, default: Any = 0.0) -> float:
        parsed = MarketDataQualityService._finite(value)
        if parsed is not None:
            return parsed
        fallback = MarketDataQualityService._finite(default)
        return fallback if fallback is not None else 0.0
