"""Server-sent event stream for the mobile dashboard."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, Iterable


class ChartStreamService:
    """Builds bounded SSE event batches with polling-compatible payloads."""

    def __init__(self, config: dict[str, Any], opportunities: Any, activity: Any | None = None) -> None:
        self.config = config
        self.opportunities = opportunities
        self.activity = activity

    def events(
        self,
        *,
        user: Any,
        mode: str,
        market_mode: str,
        once: bool = False,
        testing: bool = False,
    ) -> Iterable[str]:
        interval = max(2.0, float(self.config.get("DASHBOARD_STREAM_INTERVAL_SECONDS", 10.0) or 10.0))
        yield self._event("heartbeat", {"at": datetime.utcnow().isoformat(), "status": "connected"})
        payload = self.opportunities.opportunities(user=user, mode=mode, market_mode=market_mode, refresh=False)
        yield self._event("opportunities", payload)
        yield self._event("health", self.opportunities.health_payload())
        yield self._event("market_tick", self._market_tick(payload))
        yield self._event("activity", self._activity_payload())
        yield self._event("chart_delta", self._chart_delta(payload, market_mode=market_mode))
        if once or testing:
            return
        while True:
            time.sleep(interval)
            payload = self.opportunities.opportunities(user=user, mode=mode, market_mode=market_mode, refresh=False)
            yield self._event("opportunities", payload)
            yield self._event("health", self.opportunities.health_payload())
            yield self._event("activity", self._activity_payload())
            yield self._event("heartbeat", {"at": datetime.utcnow().isoformat(), "status": "ok"})

    def _chart_delta(self, payload: dict[str, Any], *, market_mode: str) -> dict[str, Any]:
        first = self._first_opportunity(payload)
        if not first:
            return {"chart": {}, "updated_at": datetime.utcnow().isoformat()}
        try:
            chart = self.opportunities.chart_payload(
                provider=str(first.get("provider") or ""),
                symbol=str(first.get("symbol") or ""),
                venue_symbol=str(first.get("venue_symbol") or ""),
                timeframe="live",
                market_mode=market_mode,
            )
        except Exception as exc:  # noqa: BLE001
            chart = {"error": str(exc)}
        return {"chart": chart, "updated_at": datetime.utcnow().isoformat()}

    @staticmethod
    def _market_tick(payload: dict[str, Any]) -> dict[str, Any]:
        first = ChartStreamService._first_opportunity(payload)
        if not first:
            return {"ticks": [], "updated_at": datetime.utcnow().isoformat()}
        return {
            "ticks": [
                {
                    "provider": first.get("provider"),
                    "symbol": first.get("symbol"),
                    "direction": first.get("direction"),
                    "score": first.get("score"),
                    "confidence": first.get("confidence"),
                }
            ],
            "updated_at": datetime.utcnow().isoformat(),
        }

    def _activity_payload(self) -> dict[str, Any]:
        if self.activity is None:
            return {"items": [], "count": 0, "next_cursor": None, "has_more": False, "updated_at": datetime.utcnow().isoformat()}
        try:
            return dict(self.activity.activity_payload(limit=int(self.config.get("DASHBOARD_PAGE_SIZE", 30) or 30)))
        except Exception as exc:  # noqa: BLE001
            return {"items": [], "count": 0, "error": str(exc), "updated_at": datetime.utcnow().isoformat()}

    @staticmethod
    def _first_opportunity(payload: dict[str, Any]) -> dict[str, Any]:
        rows = payload.get("opportunities") if isinstance(payload, dict) else []
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return rows[0]
        return {}

    @staticmethod
    def _event(event: str, payload: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"
