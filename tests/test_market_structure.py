from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.services.market_structure import MarketStructureService


def _candles(count: int = 40) -> list[dict[str, Any]]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, Any]] = []
    price = 100.0
    for index in range(count):
        price += 0.18 if index % 3 else -0.05
        rows.append(
            {
                "timestamp": int((start + timedelta(minutes=15 * index)).timestamp() * 1000),
                "open": price - 0.05,
                "high": price + 0.25,
                "low": price - 0.20,
                "close": price,
                "volume": 1_000.0 if index < count - 1 else 3_500.0,
            }
        )
    return rows


def _book() -> dict[str, Any]:
    return {
        "levels": [
            [{"px": "99.95", "sz": "600"}, {"px": "99.90", "sz": "400"}],
            [{"px": "100.05", "sz": "550"}, {"px": "100.10", "sz": "450"}],
        ]
    }


class _MarketData:
    def get_candles(self, symbol: str, timeframe: str, mode: str, limit: int):
        return _candles(limit)

    def get_order_book(self, symbol: str, mode: str):
        return _book()


def test_market_structure_extracts_existing_candle_and_book_features() -> None:
    service = MarketStructureService(
        {
            "MARKET_STRUCTURE_FEATURES_ENABLED": True,
            "MARKET_STRUCTURE_PROVIDER": "existing",
            "MARKET_STRUCTURE_FAIL_CLOSED": True,
            "MARKET_STRUCTURE_DEPTH_USD_SCALE": 100_000.0,
            "VAULT_BOOK_DEPTH_LEVELS": 2,
        },
        _MarketData(),
    )

    snapshot = service.snapshot("BTC", "15m", mode="testnet")

    assert snapshot["enabled"] is True
    assert snapshot["fail_closed"] is False
    assert snapshot["book_depth_usd"] > 100_000.0
    assert snapshot["book_depth_score"] == 1.0
    assert snapshot["volume_impulse"] > 0.0
    assert 0.0 < snapshot["score"] <= 1.0
    assert snapshot["coverage"] > 0.0


def test_market_structure_fails_closed_to_neutral_on_provider_errors() -> None:
    class BrokenMarketData:
        def get_candles(self, symbol: str, timeframe: str, mode: str, limit: int):
            raise RuntimeError("provider unavailable")

        def get_order_book(self, symbol: str, mode: str):
            raise RuntimeError("provider unavailable")

    service = MarketStructureService(
        {
            "MARKET_STRUCTURE_FEATURES_ENABLED": True,
            "MARKET_STRUCTURE_PROVIDER": "existing",
            "MARKET_STRUCTURE_FAIL_CLOSED": True,
        },
        BrokenMarketData(),
    )

    snapshot = service.snapshot("BTC", "15m", mode="testnet")

    assert snapshot["enabled"] is True
    assert snapshot["fail_closed"] is True
    assert snapshot["score"] == 0.0
    assert snapshot["coverage"] == 0.0
    assert "provider unavailable" in snapshot["error"]
