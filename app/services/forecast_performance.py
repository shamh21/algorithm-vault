"""Rolling dashboard forecast performance tracking."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..models import Setting


class ForecastPerformanceService:
    """Persists compact prediction outcome history in the runtime settings store."""

    SETTING_KEY = "dashboard_forecast_performance_v1"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def record_forecast(self, row: dict[str, Any], forecast: dict[str, Any], features: dict[str, Any]) -> None:
        state = self._state()
        records = list(state.get("records") or [])
        now = datetime.utcnow()
        current_price = self._current_price(row, features)
        self._settle_expired(
            records, now=now, provider=str(row.get("provider") or ""), symbol=str(row.get("symbol") or ""), price=current_price
        )
        record = self._snapshot(row, forecast, features, now=now, price=current_price)
        key = record["key"]
        records = [item for item in records if item.get("key") != key]
        records.append(record)
        cutoff = now - timedelta(days=35)
        records = [item for item in records if self._timestamp(item.get("created_at")) >= cutoff.timestamp()][
            -int(self.config.get("DASHBOARD_FORECAST_PERFORMANCE_MAX_ROWS", 500) or 500) :
        ]
        Setting.set_json(self.SETTING_KEY, {"records": records, "updated_at": now.isoformat()})

    def rolling_metrics(self) -> dict[str, Any]:
        records = list(self._state().get("records") or [])
        return {
            "updated_at": datetime.utcnow().isoformat(),
            "windows": {
                "24h": self._metrics(records, hours=24),
                "7d": self._metrics(records, hours=24 * 7),
                "30d": self._metrics(records, hours=24 * 30),
            },
            "pending": sum(1 for item in records if not item.get("settled_at")),
        }

    def _snapshot(
        self,
        row: dict[str, Any],
        forecast: dict[str, Any],
        features: dict[str, Any],
        *,
        now: datetime,
        price: float,
    ) -> dict[str, Any]:
        horizon_seconds = max(60, int(self._safe_float(forecast.get("horizon_seconds"), self.config.get("ONE_H10_HORIZON_SECONDS", 3600))))
        provider = str(row.get("provider") or "").lower()
        symbol = str(row.get("symbol") or "").upper()
        side = str(row.get("direction") or forecast.get("predicted_side") or "hold").lower()
        return {
            "key": f"{provider}:{symbol}:{int(now.timestamp() // 60)}",
            "provider": provider,
            "symbol": symbol,
            "strategy": str(row.get("forecast_source") or forecast.get("source") or row.get("source") or "dashboard"),
            "side": side if side in {"buy", "sell", "hold"} else "hold",
            "confidence": max(0.0, min(self._safe_float(row.get("confidence")), 1.0)),
            "confidence_score": int(
                row.get("confidence_score") or round(max(0.0, min(self._safe_float(row.get("confidence")), 1.0)) * 100)
            ),
            "expected_return_bps": self._safe_float(row.get("expected_return_bps"), forecast.get("expected_return_bps", 0.0)),
            "entry_price": price,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=horizon_seconds)).isoformat(),
            "market_regime": row.get("market_regime", {}),
            "data_quality": row.get("data_quality", {}),
            "settled_at": None,
        }

    def _settle_expired(self, records: list[dict[str, Any]], *, now: datetime, provider: str, symbol: str, price: float) -> None:
        if price <= 0:
            return
        for item in records:
            if item.get("settled_at"):
                continue
            if str(item.get("provider") or "") != provider or str(item.get("symbol") or "") != symbol:
                continue
            expires_at = self._timestamp(item.get("expires_at"))
            if expires_at <= 0 or expires_at > now.timestamp():
                continue
            entry = self._safe_float(item.get("entry_price"))
            if entry <= 0:
                continue
            side = str(item.get("side") or "hold")
            move_bps = ((price / entry) - 1.0) * 10_000.0
            expected = self._safe_float(item.get("expected_return_bps"))
            directional_move = -move_bps if side == "sell" else move_bps if side == "buy" else -abs(move_bps)
            hit = directional_move >= max(0.0, min(abs(expected) * 0.25, 15.0)) if side in {"buy", "sell"} else abs(move_bps) < 10.0
            item.update(
                {
                    "settled_at": now.isoformat(),
                    "exit_price": price,
                    "realized_move_bps": move_bps,
                    "realized_directional_bps": directional_move,
                    "hit": bool(hit),
                    "false_positive": bool(side in {"buy", "sell"} and not hit),
                    "absolute_error_bps": abs(expected - directional_move),
                }
            )

    def _metrics(self, records: list[dict[str, Any]], *, hours: int) -> dict[str, Any]:
        cutoff = datetime.utcnow().timestamp() - hours * 3600
        settled = [item for item in records if item.get("settled_at") and self._timestamp(item.get("created_at")) >= cutoff]
        if not settled:
            return {
                "count": 0,
                "hit_rate": None,
                "average_confidence_accuracy": None,
                "realized_vs_predicted_bps": None,
                "false_positive_rate": None,
                "confidence_calibration_quality": None,
                "strategy_accuracy": {},
            }
        hits = sum(1 for item in settled if item.get("hit"))
        false_positives = sum(1 for item in settled if item.get("false_positive"))
        avg_error = sum(self._safe_float(item.get("absolute_error_bps")) for item in settled) / len(settled)
        avg_confidence = sum(self._safe_float(item.get("confidence")) for item in settled) / len(settled)
        hit_rate = hits / len(settled)
        strategy_accuracy: dict[str, dict[str, Any]] = {}
        for item in settled:
            key = str(item.get("strategy") or "dashboard")
            bucket = strategy_accuracy.setdefault(key, {"count": 0, "hits": 0})
            bucket["count"] += 1
            bucket["hits"] += 1 if item.get("hit") else 0
        for bucket in strategy_accuracy.values():
            bucket["hit_rate"] = bucket["hits"] / max(bucket["count"], 1)
        return {
            "count": len(settled),
            "hit_rate": hit_rate,
            "average_confidence_accuracy": 1.0 - min(abs(avg_confidence - hit_rate), 1.0),
            "realized_vs_predicted_bps": -avg_error,
            "false_positive_rate": false_positives / len(settled),
            "confidence_calibration_quality": max(0.0, 1.0 - min(abs(avg_confidence - hit_rate), 1.0)),
            "prediction_expiry_hours": hours,
            "strategy_accuracy": strategy_accuracy,
        }

    def _state(self) -> dict[str, Any]:
        value = Setting.get_json(self.SETTING_KEY, {"records": []})
        return value if isinstance(value, dict) else {"records": []}

    @staticmethod
    def _current_price(row: dict[str, Any], features: dict[str, Any]) -> float:
        for value in (features.get("close"), row.get("entry"), features.get("mid"), features.get("price")):
            parsed = ForecastPerformanceService._safe_float(value)
            if parsed > 0:
                return parsed
        return 0.0

    @staticmethod
    def _timestamp(value: Any) -> float:
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, str) and value.strip():
            try:
                return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
            except ValueError:
                return 0.0
        return ForecastPerformanceService._safe_float(value)

    @staticmethod
    def _safe_float(value: Any, default: Any = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return float(default)
            except (TypeError, ValueError):
                return 0.0
