from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import app.services.live_provider_adapters as adapters
from app.routes.consumer import _cycle_one_h10_runtime_notice
from app.services.live_provider_adapters import KucoinFuturesConnector
from app.models import VaultCycle


def _credentials() -> SimpleNamespace:
    return SimpleNamespace(api_key="key", api_secret="secret", passphrase="pass", wallet_address="")


def test_kucoin_signed_requests_sync_server_time_and_retry_timestamp_rejection(monkeypatch) -> None:
    monkeypatch.setattr(adapters.time, "time", lambda: 1000.0)
    calls: list[dict[str, object]] = []

    def fake_request(session, method, url, *, provider, attempts, sleep_seconds, timeout, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        if url.endswith("/api/v1/timestamp"):
            return {"code": "200000", "data": 1005000}
        signed_calls = [call for call in calls if not str(call["url"]).endswith("/api/v1/timestamp")]
        if len(signed_calls) == 1:
            return {"code": "400002", "msg": "Invalid KC-API-TIMESTAMP"}
        return {"code": "200000", "data": []}

    monkeypatch.setattr(adapters, "_request_with_retries", fake_request)
    connector = KucoinFuturesConnector(
        {
            "ENABLE_LIVE_TRADING": True,
            "KUCOIN_FUTURES_BASE_URL": "https://example.test",
            "KUCOIN_TIME_SYNC_ENABLED": True,
            "KUCOIN_TIME_SYNC_TTL_SECONDS": 300,
        },
        _credentials(),
    )

    result = connector.get_positions("live")

    signed_calls = [call for call in calls if not str(call["url"]).endswith("/api/v1/timestamp")]
    time_calls = [call for call in calls if str(call["url"]).endswith("/api/v1/timestamp")]
    assert result == []
    assert len(time_calls) == 2
    assert [call["headers"]["KC-API-TIMESTAMP"] for call in signed_calls] == ["1005000", "1005000"]


def test_expired_one_h10_runtime_notice_is_hidden() -> None:
    cycle = VaultCycle(
        algorithm_profile="1H10",
        deposit_asset="USDC",
        deposit_amount=1.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        unlocks_at=datetime.utcnow(),
    )
    cycle.selection_metadata = {
        "one_h10_runtime_notice": {
            "kind": "market_data_backoff",
            "message": "Provider rate limited market data.",
            "retry_after": "2000-01-01T00:00:00",
        }
    }

    assert _cycle_one_h10_runtime_notice(cycle) == {}
