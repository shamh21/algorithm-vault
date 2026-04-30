from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from app.services.pair_screening import PairScreeningService


def _pair_candles(symbol: str, count: int = 120) -> list[dict[str, Any]]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        base = 100.0 + index * 0.08 + math.sin(index / 5) * 0.4
        if symbol == "ETH":
            price = base * 0.95
        elif symbol == "SOL":
            price = 70.0 + math.cos(index / 3) * 3.0
        else:
            price = base
        rows.append(
            {
                "timestamp": int((start + timedelta(minutes=index)).timestamp() * 1000),
                "open": price - 0.04,
                "high": price + 0.10,
                "low": price - 0.10,
                "close": price,
                "volume": 2_500,
            }
        )
    return rows


def _book(price: float = 100.0, spread: float = 0.02, size: float = 900.0) -> dict[str, Any]:
    return {
        "levels": [
            [{"px": str(price - spread / 2), "sz": str(size)}],
            [{"px": str(price + spread / 2), "sz": str(size)}],
        ]
    }


class _MarketData:
    def get_candles(self, symbol: str, timeframe: str, mode: str, limit: int):
        return _pair_candles(symbol, limit)

    def get_order_book(self, symbol: str, mode: str):
        return _book(100.0 if symbol != "SOL" else 70.0)


def test_pair_screening_scores_stat_arb_and_relative_strength_candidates() -> None:
    config = {
        "PAIR_SCREENING_ENABLED": True,
        "PAIR_MIN_CORRELATION": 0.75,
        "PAIR_MAX_SPREAD_ZSCORE": 2.5,
        "PAIR_MIN_LIQUIDITY_USD": 25_000.0,
        "PAIR_MAX_SPREAD_BPS": 20.0,
    }
    service = PairScreeningService(config, _MarketData())

    candidates = service.screen(["BTC", "ETH", "SOL"], mode="testnet", timeframe="5m", pair_mode="both")

    stat_arb = next(candidate for candidate in candidates if candidate.pair_mode == "stat_arb")
    relative = next(candidate for candidate in candidates if candidate.pair_mode == "relative_strength")
    assert stat_arb.base_symbol == "BTC"
    assert stat_arb.pair_symbol == "ETH"
    assert stat_arb.correlation >= 0.75
    assert abs(stat_arb.spread_zscore) <= 2.5
    assert stat_arb.hedge_ratio > 0
    assert stat_arb.liquidity_balance > 0.9
    assert stat_arb.spread_cost_bps <= 20.0
    assert stat_arb.pair_signal["long_symbol"] in {"BTC", "ETH"}
    assert relative.leader_symbol in {"BTC", "ETH"}


def test_pair_screening_fails_closed_on_missing_or_illiquid_data() -> None:
    class MissingData:
        def get_candles(self, *args, **kwargs):
            raise RuntimeError("provider down")

        def get_order_book(self, *args, **kwargs):
            return _book()

    service = PairScreeningService({"PAIR_SCREENING_ENABLED": True}, MissingData())

    assert service.screen(["BTC", "ETH"]) == []
    assert service.last_rejections["market_data_unavailable"] == 2
