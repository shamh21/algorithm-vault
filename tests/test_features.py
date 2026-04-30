from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.features.engine import FeatureEngine
from app.features.external import default_external_adapters
from app.features.fibonacci import FibonacciService
from app.features.indicators import atr, bollinger_bands, ema, macd, rsi, sma, trend_strength, volatility, volume_spike
from app.features.multi_timeframe import MultiTimeframeConfluenceService
from app.features.patterns import StubPatternModel
from app.features.rules import SignalRule
from app.strategies.rule_based import RuleBasedSignalStrategy


def candles(count: int = 80) -> list[dict[str, Any]]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 100.0
    for index in range(count):
        price += 0.2 if index < count // 2 else -0.05
        volume = 1000 + index
        if index % 12 == 0:
            volume *= 2
        rows.append(
            {
                "timestamp": int((start + timedelta(minutes=15 * index)).timestamp() * 1000),
                "open": price - 0.1,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": volume,
            }
        )
    return rows


def test_indicators_are_deterministic() -> None:
    rows = candles()
    closes = [row["close"] for row in rows]

    first = (
        ema(closes, 8),
        rsi(closes, 7),
        atr(rows, 14),
        volume_spike(rows, 20, 1.5),
        macd(closes),
        bollinger_bands(closes),
    )
    second = (
        ema(closes, 8),
        rsi(closes, 7),
        atr(rows, 14),
        volume_spike(rows, 20, 1.5),
        macd(closes),
        bollinger_bands(closes),
    )

    assert first == second


def test_feature_snapshot_is_deterministic_with_neutral_stubs() -> None:
    engine = FeatureEngine()
    rows = candles()

    first = engine.snapshot(symbol="BTC", timeframe="15m", candles=rows).as_dict()
    second = engine.snapshot(symbol="BTC", timeframe="15m", candles=rows).as_dict()

    assert first == second
    assert first["pattern_prediction"]["label"] in {"neutral", "bullish", "bearish"}
    assert 0.0 <= first["pattern_prediction"]["confidence"] <= 1.0
    assert set(first["external_scores"]) == {"coingecko", "sentiment", "dune", "nansen"}
    assert "sma_fast" in first
    assert "sma_slow" in first
    assert "volatility" in first
    assert "trend_strength" in first
    assert "macd_histogram" in first
    assert "bollinger_bands" in first
    assert 0.0 <= first["bollinger_bands"]["percent_b"] <= 1.5


def test_fibonacci_levels_are_deterministic() -> None:
    service = FibonacciService()
    rows = candles()

    first = service.compute(rows, lookback=50).as_dict()
    second = service.compute(rows, lookback=50).as_dict()

    assert first == second
    assert "61.8" in first["retracements"]
    assert "161.8" in first["extensions"]


def test_fibonacci_confluence_scores_multi_lookback_clusters() -> None:
    service = FibonacciService()
    rows = candles(120)
    price = rows[-1]["close"]

    confluence = service.confluence(rows, price, lookbacks=[20, 50, 100], tolerance_bps=250).as_dict()

    assert confluence["lookbacks"] == [20, 50, 100]
    assert 0.0 <= confluence["score"] <= 1.0
    assert confluence["cluster_count"] >= 1
    assert confluence["nearest_support"] is not None or confluence["nearest_resistance"] is not None


def test_feature_snapshot_includes_fibonacci_confluence() -> None:
    snapshot = FeatureEngine().snapshot(symbol="BTC", timeframe="15m", candles=candles(120)).as_dict()

    assert "fibonacci_confluence" in snapshot
    assert "score" in snapshot["fibonacci_confluence"]
    assert snapshot["fibonacci_confluence"]["lookbacks"] == [20, 50, 100]


def test_multi_timeframe_confluence_scores_clusters_volume_and_rsi() -> None:
    rows = candles(140)
    rows[-1]["volume"] = rows[-2]["volume"] * 2

    class MarketData:
        def get_candles(self, symbol: str, timeframe: str, mode: str, limit: int):
            return rows[-limit:]

    service = MultiTimeframeConfluenceService(
        MarketData(),
        {
            "FIB_CONFLUENCE_THRESHOLD": 0.05,
            "FIB_MULTI_LOOKBACKS": [20, 50, 100],
            "FIB_MULTI_TIMEFRAMES": ["1h", "4h", "1d"],
            "FIB_CONFLUENCE_CLUSTER_TOLERANCE_BPS": 250.0,
        },
    )

    result = service.score(symbol="BTC", entry_timeframe="1m", mode="testnet", price=rows[-1]["close"]).as_dict()

    assert result["passed"] is True
    assert result["score"] >= 0.05
    assert result["cluster_count"] >= 1
    assert result["volume_confirmation"] is True
    assert result["invalidation_distance_bps"] >= 0


def test_multi_timeframe_confluence_fails_closed_below_threshold() -> None:
    class MarketData:
        def get_candles(self, symbol: str, timeframe: str, mode: str, limit: int):
            return []

    service = MultiTimeframeConfluenceService(MarketData(), {"FIB_CONFLUENCE_THRESHOLD": 0.5})

    result = service.score(symbol="BTC", entry_timeframe="1m", mode="testnet", price=100.0).as_dict()

    assert result["passed"] is False
    assert result["skip_reason"] == "multi_timeframe_data_unavailable"


def test_neutral_external_and_pattern_outputs_do_not_change_rule_decision() -> None:
    rows = candles()
    engine_with_stubs = FeatureEngine(default_external_adapters(), StubPatternModel())
    engine_without_stubs = FeatureEngine([], StubPatternModel())
    parameters = {
        "minimum_signal_score": 0.5,
        "external_weight": 0.0,
        "fibonacci_filter_weight": 0.05,
    }
    rule = SignalRule(parameters)

    with_stubs = engine_with_stubs.snapshot(symbol="BTC", timeframe="15m", candles=rows).as_dict()
    without_stubs = engine_without_stubs.snapshot(symbol="BTC", timeframe="15m", candles=rows).as_dict()

    assert rule.evaluate(with_stubs, {"quantity": 0.0}).as_dict() == rule.evaluate(
        without_stubs,
        {"quantity": 0.0},
    ).as_dict()


def test_fibonacci_cannot_independently_trigger_trade() -> None:
    features = {
        "close": 100.0,
        "ema_trend": 0.0,
        "rsi": 50.0,
        "atr_pct": 0.01,
        "volume_spike": {"is_spike": False},
        "external_scores": {},
        "fibonacci_levels": {"golden_zone": {"lower": 99.0, "upper": 101.0}},
    }
    decision = SignalRule({"minimum_signal_score": 0.01, "fibonacci_filter_weight": 100.0}).evaluate(
        features,
        {"quantity": 0.0},
    )

    assert decision.action == "hold"


def test_rule_based_strategy_signal_is_deterministic() -> None:
    parameters = {"minimum_signal_score": 0.5, "external_weight": 0.0}
    strategy = RuleBasedSignalStrategy(parameters)
    rows = candles()

    first = strategy.generate_signal(symbol="BTC", timeframe="15m", candles=rows, position={"quantity": 0.0}).as_dict()
    second = strategy.generate_signal(symbol="BTC", timeframe="15m", candles=rows, position={"quantity": 0.0}).as_dict()

    assert first == second
    assert first["metadata"]["pattern_prediction"]["label"] in {"neutral", "bullish", "bearish"}


def test_indicator_edge_cases_are_safe() -> None:
    malformed = [{}, {"close": None}, {"close": "bad"}, {"close": 100, "high": 99, "low": 101}]
    flat = [100.0] * 20
    extreme = [100.0, 150.0, 75.0, 160.0, 80.0]

    assert sma([], 5) == 0.0
    assert ema([], 5) == 0.0
    assert rsi(flat, 14) == 50.0
    assert atr(malformed, 14) >= 0.0
    assert volume_spike([{"close": 100}], 20, 1.5)["is_spike"] is False
    assert macd([])["histogram"] == 0.0
    assert bollinger_bands([])["percent_b"] == 0.5
    assert volatility(flat, 14) == 0.0
    assert volatility(extreme, 4) > 0.0
    assert trend_strength(flat, 8, 21) == 0.0


def test_feature_snapshot_handles_empty_and_malformed_candles() -> None:
    engine = FeatureEngine()

    empty = engine.snapshot(symbol="ETH", timeframe="1m", candles=[]).as_dict()
    malformed = engine.snapshot(symbol="ETH", timeframe="1m", candles=[{}, {"close": "bad"}, {"close": 100}]).as_dict()

    assert empty["close"] == 0.0
    assert empty["rsi"] == 50.0
    assert empty["pattern_prediction"]["label"] == "neutral"
    assert malformed["close"] == 100.0
    assert malformed["volume_spike"]["is_spike"] is False


def test_pattern_prediction_is_directional_without_fabricating_confidence() -> None:
    model = StubPatternModel()
    bullish = [
        {"open": 100 + index, "high": 101 + index, "low": 99 + index, "close": 100.8 + index}
        for index in range(8)
    ]

    assert model.predict([]).label == "neutral"
    prediction = model.predict(bullish)
    assert prediction.label in {"bullish", "neutral"}
    assert 0.0 <= prediction.confidence <= 1.0
