"""Dashboard chart and forecast projection helpers."""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import Any


class MLProjectionEngine:
    """Normalizes multi-timeframe chart data and builds lightweight overlays."""

    PUBLIC_TIMEFRAMES = ("live", "1m", "5m", "15m", "45m", "4h", "1d")

    def __init__(
        self,
        config: dict[str, Any],
        market_data: Any,
        feature_engine: Any,
        forecast_service: Any | None = None,
        data_quality: Any | None = None,
        prediction_quality: Any | None = None,
    ) -> None:
        self.config = config
        self.market_data = market_data
        self.feature_engine = feature_engine
        self.forecast_service = forecast_service
        self.data_quality = data_quality
        self.prediction_quality = prediction_quality

    def normalize_timeframe(self, timeframe: str | None) -> str:
        value = str(timeframe or "live").strip().lower()
        aliases = {
            "": "live",
            "realtime": "live",
            "rt": "live",
            "4hr": "4h",
            "4hour": "4h",
            "4hours": "4h",
            "240m": "4h",
            "1day": "1d",
            "24h": "1d",
        }
        value = aliases.get(value, value)
        return value if value in self.PUBLIC_TIMEFRAMES else "live"

    def candle_timeframe(self, timeframe: str | None) -> str:
        public = self.normalize_timeframe(timeframe)
        if public == "live":
            return "1m"
        if public == "45m":
            return "15m"
        if public == "1d":
            return "4h"
        return public

    def chart_payload(
        self,
        *,
        provider: str,
        symbol: str,
        venue_symbol: str = "",
        mode: str = "live",
        timeframe: str = "live",
        forecast: dict[str, Any] | None = None,
        features: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        public_timeframe = self.normalize_timeframe(timeframe)
        candle_payload = self._candles_payload(
            venue_symbol or symbol,
            public_timeframe,
            mode=mode,
            limit=limit or int(self.config.get("DASHBOARD_CHART_CANDLE_LIMIT", 150) or 150),
        )
        candles = candle_payload["candles"]
        enriched_forecast = self.enrich_forecast(
            forecast or {},
            features or {},
            candles=candles,
            timeframe=public_timeframe,
            provider=provider,
            symbol=symbol,
            data_quality=candle_payload["data_quality"],
        )
        overlays = self.overlays(candles, enriched_forecast, features or {})
        expires_at = datetime.utcnow() + timedelta(seconds=max(60, int(self._safe_float(enriched_forecast.get("horizon_seconds"), 3600.0))))
        return {
            "provider": str(provider or "global"),
            "symbol": str(symbol or "").upper(),
            "venue_symbol": str(venue_symbol or symbol or "").upper(),
            "timeframe": public_timeframe,
            "source_timeframe": self.candle_timeframe(public_timeframe),
            "candles": candles[-150:],
            "forecast": enriched_forecast,
            "overlays": overlays,
            "data_quality": candle_payload["data_quality"],
            "provider_quality": {
                "provider": str(provider or "global"),
                "latency_ms": candle_payload["data_quality"].get("provider_latency_ms"),
                "last_sync_age_seconds": candle_payload["data_quality"].get("last_sync_age_seconds"),
                "candle_completeness": candle_payload["data_quality"].get("candle_completeness"),
                "market_volatility_state": candle_payload["data_quality"].get("market_volatility_state"),
                "signal_freshness": candle_payload["data_quality"].get("signal_freshness"),
            },
            "forecast_explanation": enriched_forecast.get("explanation", {}),
            "expiry": {
                "expires_at": expires_at.isoformat(),
                "seconds": max(60, int(self._safe_float(enriched_forecast.get("horizon_seconds"), 3600.0))),
            },
            "updated_at": datetime.utcnow().isoformat(),
        }

    def candles(self, symbol: str, timeframe: str, *, mode: str, limit: int = 150) -> list[dict[str, Any]]:
        return self._candles_payload(symbol, timeframe, mode=mode, limit=limit)["candles"]

    def _candles_payload(self, symbol: str, timeframe: str, *, mode: str, limit: int = 150) -> dict[str, Any]:
        public_timeframe = self.normalize_timeframe(timeframe)
        source_timeframe = self.candle_timeframe(public_timeframe)
        source_limit = min(max(int(limit or 150), 1), 150)
        if public_timeframe == "45m":
            fetch_limit = source_limit * 3
        elif public_timeframe == "1d":
            fetch_limit = source_limit * 6
        else:
            fetch_limit = source_limit
        rows = self._safe_candles(symbol, source_timeframe, mode, fetch_limit)
        normalized = [self._normalize_candle(row, index) for index, row in enumerate(rows or []) if isinstance(row, dict)]
        normalized = [row for row in normalized if row["close"] > 0]
        if public_timeframe == "45m":
            normalized = self._aggregate_candles(normalized, group_size=3)
        if public_timeframe == "1d":
            normalized = self._aggregate_candles(normalized, group_size=6)
        normalized = normalized[-source_limit:]
        if self.data_quality is None:
            return {
                "candles": normalized,
                "data_quality": {
                    "score": 100 if normalized else 0,
                    "state": "good" if normalized else "insufficient",
                    "issues": [] if normalized else ["insufficient_candles"],
                    "candle_count": len(normalized),
                    "candle_completeness": 1.0 if normalized else 0.0,
                    "market_volatility_state": "normal",
                    "signal_freshness": "fresh" if normalized else "stale",
                    "last_sync_age_seconds": None,
                },
            }
        checked = self.data_quality.validate_candles(normalized, timeframe=public_timeframe)
        return {"candles": checked["candles"][-source_limit:], "data_quality": checked["quality"]}

    def forecast_from_features(
        self,
        features: dict[str, Any],
        *,
        provider: str,
        symbol: str,
        allocation_cap_usd: float = 0.0,
        available_margin_usd: float = 0.0,
        market: Any | None = None,
    ) -> dict[str, Any]:
        if self.forecast_service is None:
            return {}
        try:
            return dict(
                self.forecast_service.forecast(
                    dict(features or {}),
                    provider=provider,
                    symbol=symbol,
                    allocation_cap_usd=allocation_cap_usd,
                    available_margin_usd=available_margin_usd,
                    market=market,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "predicted_side": "hold",
                "action": "hold",
                "confidence": 0.0,
                "expected_return_bps": 0.0,
                "blockers": ["forecast_unavailable"],
                "advisory_blockers": [str(exc)[:240]],
                "source": "dashboard_forecast_error",
            }

    def enrich_forecast(
        self,
        forecast: dict[str, Any],
        features: dict[str, Any] | None = None,
        *,
        candles: list[dict[str, Any]] | None = None,
        timeframe: str = "1m",
        provider: str = "global",
        symbol: str = "",
        data_quality: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        features = features or {}
        if self.data_quality is not None:
            safe_forecast = self.data_quality.sanitize_forecast_payload(forecast)
            quality = data_quality or self.data_quality.score_data_quality(
                candles or [],
                features=features,
                forecast=safe_forecast,
                timeframe=timeframe,
            )
        else:
            safe_forecast = dict(forecast or {})
            quality = data_quality or {"score": 70, "state": "degraded", "issues": []}
        if self.prediction_quality is None:
            return {**safe_forecast, "data_quality": quality}
        enriched = self.prediction_quality.enrich_forecast(features=features, forecast=safe_forecast, data_quality=quality)
        blockers = [str(item) for item in (safe_forecast.get("blockers", []) or []) if str(item)]
        advisory = [str(item) for item in (safe_forecast.get("advisory_blockers", []) or []) if str(item)]
        if quality.get("state") not in {"good", None}:
            advisory.append(f"data_quality_{quality.get('state')}")
        if enriched.get("confidence", 0.0) < self._safe_float(safe_forecast.get("confidence"), 0.0):
            advisory.append("confidence_degraded")
        return {
            **safe_forecast,
            **enriched,
            "provider": str(provider or safe_forecast.get("provider") or "global"),
            "symbol": str(symbol or safe_forecast.get("symbol") or "").upper(),
            "blockers": list(dict.fromkeys(blockers)),
            "advisory_blockers": list(dict.fromkeys(item for item in advisory if item)),
        }

    def price_levels(
        self, candles: list[dict[str, Any]], forecast: dict[str, Any], features: dict[str, Any] | None = None
    ) -> dict[str, float]:
        latest = self._latest_close(candles, features or {})
        if latest <= 0:
            latest = 1.0
        side = str(forecast.get("predicted_side") or forecast.get("action") or "hold").lower()
        direction = -1.0 if side == "sell" else 1.0
        if side not in {"buy", "sell"}:
            direction = 0.0
        stop_pct = self._safe_float(forecast.get("suggested_stop_loss_pct"), 0.0)
        take_pct = self._safe_float(forecast.get("suggested_take_profit_pct"), 0.0)
        expected_bps = self._safe_float(forecast.get("expected_return_bps"), forecast.get("net_expected_return_bps", 0.0))
        if take_pct <= 0 and expected_bps:
            take_pct = min(max(abs(expected_bps) / 10_000.0, 0.001), 0.35)
        if stop_pct <= 0:
            volatility = max(
                self._safe_float((features or {}).get("atr_pct")),
                self._safe_float((features or {}).get("volatility")),
                0.002,
            )
            stop_pct = min(max(volatility * 1.35, 0.002), 0.08)
        entry = self._safe_float(forecast.get("entry_price"), latest) or latest
        exit_price = self._safe_float(forecast.get("exit_price"), 0.0)
        stop_price = self._safe_float(forecast.get("stop_loss_price"), 0.0)
        if exit_price <= 0:
            exit_price = entry * (1.0 + direction * take_pct) if direction else entry
        if stop_price <= 0:
            stop_price = entry * (1.0 - direction * stop_pct) if direction else entry
        risk = abs(entry - stop_price)
        reward = abs(exit_price - entry)
        risk_reward = reward / risk if risk > 0 else 0.0
        return {
            "entry": entry,
            "exit": exit_price,
            "stop_loss": stop_price,
            "risk_reward": risk_reward,
        }

    def overlays(self, candles: list[dict[str, Any]], forecast: dict[str, Any], features: dict[str, Any] | None = None) -> dict[str, Any]:
        features = features or {}
        levels = self.price_levels(candles, forecast, features)
        entry = levels["entry"]
        side = str(forecast.get("predicted_side") or forecast.get("action") or "hold").lower()
        confidence = max(0.0, min(self._safe_float(forecast.get("confidence")), 1.0))
        direction = -1.0 if side == "sell" else 1.0 if side == "buy" else 0.0
        horizon_seconds = max(60, int(self._safe_float(forecast.get("horizon_seconds"), 3600.0)))
        last_time = self._last_time(candles)
        step = max(60, horizon_seconds // 8)
        path: list[dict[str, float]] = []
        upper: list[dict[str, float]] = []
        lower: list[dict[str, float]] = []
        cone: list[dict[str, float]] = []
        target_delta = levels["exit"] - entry
        volatility = max(self._safe_float(features.get("atr_pct")), self._safe_float(features.get("volatility")), 0.002)
        for index in range(1, 9):
            progress = index / 8
            easing = 1 - (1 - progress) ** 2
            wave = math.sin(progress * math.pi) * volatility * entry * 0.18 * direction
            value = entry + target_delta * easing + wave
            band = entry * volatility * (0.7 + progress * 2.6) * (1.0 - confidence * 0.35)
            point_time = float(last_time + step * index)
            path.append({"time": point_time, "value": value})
            upper.append({"time": point_time, "value": value + band})
            lower.append({"time": point_time, "value": max(value - band, 0.0)})
            cone.append({"time": point_time, "upper": value + band, "lower": max(value - band, 0.0)})
        fib_markers = [
            {"time": float(last_time + max(60, int(horizon_seconds * ratio))), "ratio": ratio}
            for ratio in (0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
        ]
        reversal = path[min(max(int(round((1.0 - confidence) * 7)), 0), len(path) - 1)] if path else {}
        stop_band_width = max(entry * volatility * 0.75, abs(entry - levels["stop_loss"]) * 0.18)
        invalidation_price = levels["stop_loss"] if side in {"buy", "sell"} else entry
        uncertainty = max(0.0, min(1.0 - confidence, 1.0))
        return {
            "path": path,
            "confidence_band": {"upper": upper, "lower": lower},
            "volatility_cone": cone,
            "projected_range": {
                "upper": upper,
                "lower": lower,
                "label": "Projected range",
            },
            "uncertainty_shading": {
                "intensity": uncertainty,
                "quality": forecast.get("data_quality", {}).get("state") if isinstance(forecast.get("data_quality"), dict) else None,
            },
            "zones": {
                "entry": {"price": levels["entry"]},
                "exit": {"price": levels["exit"]},
                "stop_loss": {"price": levels["stop_loss"]},
            },
            "invalidation_zone": {
                "price": invalidation_price,
                "label": "Invalidation",
            },
            "stop_loss_band": {
                "upper": levels["stop_loss"] + stop_band_width,
                "lower": max(levels["stop_loss"] - stop_band_width, 0.0),
                "label": "Stop risk band",
            },
            "volatility_expansion": {
                "state": forecast.get("market_regime", {}).get("state") if isinstance(forecast.get("market_regime"), dict) else "normal",
                "volatility": volatility,
            },
            "markers": [
                {"time": float(last_time), "price": entry, "type": "buy" if side == "buy" else "sell" if side == "sell" else "hold"},
            ],
            "fibonacci_time_zones": fib_markers,
            "reversal_points": [reversal] if reversal else [],
            "forecast_horizon": {"start": float(last_time), "end": float(last_time + horizon_seconds), "seconds": horizon_seconds},
            "trend_continuation_probability": self._safe_float(forecast.get("trend_continuation_probability"), confidence * 100.0),
            "forecast_expiry": {
                "seconds": horizon_seconds,
                "expires_at": (datetime.utcnow() + timedelta(seconds=horizon_seconds)).isoformat(),
            },
            "strategy_agreement_score": self._safe_float(
                forecast.get("ml_agreement_score"),
                forecast.get("strategy_consensus", forecast.get("confidence", 0.0)),
            ),
            "risk_reward": levels["risk_reward"],
        }

    def _safe_candles(self, symbol: str, timeframe: str, mode: str, limit: int) -> list[dict[str, Any]]:
        try:
            return list(self.market_data.get_candles(symbol, timeframe, mode=mode, limit=limit) or [])
        except Exception:  # noqa: BLE001
            return []

    def _aggregate_candles(self, candles: list[dict[str, Any]], *, group_size: int) -> list[dict[str, Any]]:
        aggregated: list[dict[str, Any]] = []
        for start in range(0, len(candles), group_size):
            group = candles[start : start + group_size]
            if not group:
                continue
            aggregated.append(
                {
                    "time": group[-1]["time"],
                    "open": group[0]["open"],
                    "high": max(row["high"] for row in group),
                    "low": min(row["low"] for row in group),
                    "close": group[-1]["close"],
                    "volume": sum(row.get("volume", 0.0) for row in group),
                }
            )
        return aggregated

    def _normalize_candle(self, row: dict[str, Any], index: int) -> dict[str, float]:
        close = self._safe_float(row.get("close", row.get("c", row.get("price", 0.0))))
        timestamp = row.get("time", row.get("timestamp", row.get("t", 0)))
        return {
            "time": self._time_value(timestamp, index),
            "open": self._safe_float(row.get("open", row.get("o", close)), close),
            "high": self._safe_float(row.get("high", row.get("h", close)), close),
            "low": self._safe_float(row.get("low", row.get("l", close)), close),
            "close": close,
            "volume": self._safe_float(row.get("volume", row.get("v", 0.0))),
        }

    def _time_value(self, value: Any, index: int) -> float:
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return parsed.timestamp()
            except ValueError:
                pass
        parsed = self._safe_float(value, 0.0)
        if parsed <= 0:
            return float(int(time.time()) - (150 - index) * 60)
        if parsed > 10_000_000_000:
            parsed /= 1000.0
        return float(parsed)

    def _last_time(self, candles: list[dict[str, Any]]) -> int:
        if not candles:
            return int(time.time())
        return int(self._safe_float(candles[-1].get("time"), time.time()))

    def _latest_close(self, candles: list[dict[str, Any]], features: dict[str, Any]) -> float:
        if candles:
            return self._safe_float(candles[-1].get("close"))
        return self._safe_float(features.get("close"), 1.0)

    @staticmethod
    def _safe_float(value: Any, default: Any = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return float(default)
            except (TypeError, ValueError):
                return 0.0
