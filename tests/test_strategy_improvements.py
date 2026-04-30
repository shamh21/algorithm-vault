from __future__ import annotations

import inspect

from app.extensions import db
from app.models import AuditLog, OptimizerRun, Order, StrategyRanking, StrategyRun
from app.services.signal_quality import SignalQualityEvaluator
from app.services.strategy_runner import StrategyManager
from app.strategies.base import Signal
from app.strategies.registry import StrategyRegistry
from app.strategies.rule_based import RuleBasedSignalStrategy
from app.strategies.scalping import ScalpingStrategy


def test_strategy_runner_limited_helper_and_error_message_are_not_duplicated() -> None:
    source = inspect.getsource(StrategyManager)

    assert source.count("def _mark_vault_limited") == 1
    assert source.count("Strategy run {run_id} failed") == 1


def test_aggressive_signal_edge_below_cost_threshold_records_no_trade(app) -> None:
    manager = app.extensions["services"]["strategy_manager"]
    manager.market_data.get_order_book = lambda symbol, mode: {
        "levels": [[{"px": "99.99", "sz": "10"}], [{"px": "100.01", "sz": "10"}]]
    }
    run = StrategyRun(
        strategy_name="scalping",
        symbol="BTC",
        timeframe="1m",
        mode="paper",
        status="running",
    )
    run.parameters = {"optimizer_profile": "aggressive_1h"}
    db.session.add(run)
    db.session.commit()

    signal = Signal("buy", "small expected move", "1m", 99.0, 100.05, 0.1)
    payload = manager._signal_edge_payload(run, signal, {"atr_pct": 0.0}, 100.0, "testnet")
    manager._record_no_trade(run, payload)

    audit = AuditLog.query.filter_by(action="no_trade").one()
    assert payload["no_trade_reason"] == "low_edge_after_costs"
    assert audit.details["optimizer_profile"] == "aggressive_1h"
    assert audit.details["edge_score"] < app.config["AGGRESSIVE_MIN_EDGE_BPS"]


def test_strategy_entry_sizing_uses_live_caps_without_simulated_portfolio(app) -> None:
    manager = app.extensions["services"]["strategy_manager"]
    run = StrategyRun(strategy_name="scalping", symbol="BTC", timeframe="1m", mode="live")
    run.parameters = {"allocation_cap_usd": 250.0}

    assert manager._entry_sizing_base(run) == 250.0
    run.parameters = {"allocation_cap_usd": app.config["MAX_POSITION_NOTIONAL"] * 2}
    assert manager._entry_sizing_base(run) == app.config["MAX_POSITION_NOTIONAL"]
    run.parameters = {}
    assert manager._entry_sizing_base(run) == app.config["MAX_POSITION_NOTIONAL"]


def test_strategy_and_vault_fallbacks_are_live_only(app) -> None:
    manager = app.extensions["services"]["strategy_manager"]
    selector = app.extensions["services"]["vault_strategy_selector"]
    metadata: dict[str, object] = {}

    assert manager._fallback_mode() == "live"
    assert manager._market_mode("paper") == "live"
    assert manager._market_mode("testnet") == "live"
    assert manager._market_mode("shadow_live") == "live"
    assert manager._market_mode("live") == "live"
    assert selector._fallback_mode() == "live"
    assert selector._execution_state("paper", [], metadata) == ("live", "live", "failed", "limited")


def test_signal_quality_combines_features_fibonacci_market_and_ml(app) -> None:
    app.config["ML_RANKER_ENABLED"] = False
    evaluator = SignalQualityEvaluator(app.config)
    signal = Signal("buy", "breakout", "1m", 99.0, 101.6, 0.1)
    feature_payload = {
        "ema_trend": 1.2,
        "trend_strength": 0.8,
        "atr_pct": 0.001,
        "volume_spike": {"is_spike": True, "ratio": 2.5},
        "pattern_prediction": {"label": "bullish", "confidence": 0.8, "probability": 0.7},
        "fibonacci_levels": {
            "swing_high": 101.0,
            "swing_low": 95.0,
            "trend": "up",
            "lookback": 50,
            "retracements": {"50.0": 98.0, "61.8": 97.292},
            "extensions": {"127.2": 102.632, "161.8": 104.708},
            "golden_zone": {"lower": 97.292, "upper": 98.0},
        },
    }
    payload = evaluator.evaluate(
        symbol="BTC",
        timeframe="1m",
        mode="testnet",
        run_parameters={"optimizer_profile": "extreme_roi_experimental", "take_profit_pct": 0.016},
        signal=signal,
        feature_payload=feature_payload,
        mid=100.0,
        market_snapshot={
            "source": "websocket",
            "spread_bps": 1.0,
            "liquidity_usd": 50_000.0,
            "volatility_pct": 0.6,
            "signal_stability": 0.9,
            "recent_trades": [{"px": "100.0"}, {"px": "100.8"}],
        },
    )

    assert payload["edge_score"] > app.config["EXTREME_ROI_MIN_EDGE_BPS"]
    assert payload["confidence"] > app.config["EXTREME_ROI_MIN_CONFIDENCE"]
    assert payload["suggested_execution_style"] == "maker_limit"
    assert payload["no_trade_reason"] == ""
    assert payload["fibonacci_alignment"]["bonus_bps"] > 0
    assert payload["net_roi_v2_score"] > 0
    assert payload["roi_quality_grade"] in {"A", "B", "C", "D"}
    assert payload["regime_support"] in {"regime-supported", "regime-neutral", "regime-fragile"}
    assert payload["signal_quality_breakdown"]["raw_strategy"]["action"] == "buy"
    assert payload["signal_quality_breakdown"]["cost_drag"]["spread_bps"] == 1.0


def test_signal_quality_scores_1h_confluence_and_fibonacci_clusters(app) -> None:
    app.config["ML_RANKER_ENABLED"] = False
    evaluator = SignalQualityEvaluator(app.config)
    signal = Signal("buy", "breakout", "1m", 99.5, 102.0, 0.1)
    feature_payload = {
        "ema_trend": 0.4,
        "trend_strength": 0.01,
        "atr_pct": 0.002,
        "rsi": 58.0,
        "macd_histogram": 0.05,
        "bollinger_bands": {"percent_b": 0.62},
        "volume_spike": {"is_spike": True, "ratio": 2.0},
        "fibonacci_levels": {
            "swing_high": 101.0,
            "swing_low": 96.0,
            "trend": "up",
            "lookback": 50,
            "retracements": {"50.0": 98.5, "61.8": 97.91},
            "extensions": {"127.2": 102.36, "161.8": 104.09},
            "golden_zone": {"lower": 97.91, "upper": 98.5},
        },
        "fibonacci_confluence": {
            "score": 0.72,
            "cluster_count": 3,
            "golden_zone_count": 2,
            "trend_bias": "up",
            "support_distance_bps": 10.0,
            "resistance_distance_bps": 180.0,
        },
    }

    payload = evaluator.evaluate(
        symbol="BTC",
        timeframe="1m",
        mode="testnet",
        run_parameters={"optimizer_profile": "aggressive_1h", "lock_duration_hours": 1, "take_profit_pct": 0.02},
        signal=signal,
        feature_payload=feature_payload,
        mid=100.0,
        market_snapshot={
            "source": "websocket",
            "spread_bps": 0.0,
            "liquidity_usd": 75_000.0,
            "volatility_pct": 0.8,
            "signal_stability": 0.9,
        },
    )

    assert payload["fibonacci_alignment"]["confluence"]["score"] == 0.72
    assert payload["fibonacci_alignment"]["bonus_bps"] > 10.0
    assert payload["one_hour_confluence"]["score"] > 0.8
    assert payload["no_trade_reason"] == ""
    assert "market source: websocket" in payload["quality_reasons"]


def test_aggressive_1h_signal_blocks_cost_drag_above_threshold(app) -> None:
    app.config["ML_RANKER_ENABLED"] = False
    evaluator = SignalQualityEvaluator(app.config)
    signal = Signal("buy", "wide spread breakout", "1m", 99.0, 103.0, 0.1)

    payload = evaluator.evaluate(
        symbol="BTC",
        timeframe="1m",
        mode="live",
        run_parameters={"optimizer_profile": "aggressive_1h", "take_profit_pct": 0.03},
        signal=signal,
        feature_payload={"atr_pct": 0.001, "trend_strength": 0.5},
        mid=100.0,
        market_snapshot={
            "source": "websocket",
            "spread_bps": 30.0,
            "liquidity_usd": 100_000.0,
            "volatility_pct": 0.5,
            "signal_stability": 0.9,
        },
    )

    assert payload["edge_score"] > app.config["AGGRESSIVE_MIN_EDGE_BPS"]
    assert payload["cost_drag_bps"] > app.config["AGGRESSIVE_1H_MAX_COST_DRAG_BPS"]
    assert payload["no_trade_reason"] == "cost_drag_above_threshold"


def test_live_signal_quality_blocks_low_fill_quality_and_reports_net_roi(app) -> None:
    app.config["ML_RANKER_ENABLED"] = False
    evaluator = SignalQualityEvaluator(app.config)
    signal = Signal("buy", "wide unstable book", "1m", 99.0, 105.0, 0.1)

    payload = evaluator.evaluate(
        symbol="BTC",
        timeframe="1m",
        mode="live",
        run_parameters={"take_profit_pct": 0.05},
        signal=signal,
        feature_payload={"atr_pct": 0.001, "trend_strength": 0.6},
        mid=100.0,
        market_snapshot={
            "source": "websocket",
            "spread_bps": 90.0,
            "liquidity_usd": 1.0,
            "volatility_pct": 3.0,
            "volatility_regime": "dislocated",
            "signal_stability": 0.6,
        },
    )

    assert payload["no_trade_reason"] == "low_expected_fill_quality"
    assert payload["expected_fill_quality"] < app.config["NET_ROI_MIN_FILL_QUALITY"]
    assert "net_roi_score" in payload
    assert "churn_penalty" in payload


def test_live_signal_quality_debounces_repeated_signal(app) -> None:
    app.config["ML_RANKER_ENABLED"] = False
    evaluator = SignalQualityEvaluator(app.config)
    signal = Signal("buy", "same candle", "1m", 99.0, 102.0, 0.1)

    payload = evaluator.evaluate(
        symbol="BTC",
        timeframe="1m",
        mode="live",
        run_parameters={"take_profit_pct": 0.02, "last_signal_action": "buy", "last_signal_age_seconds": 5.0},
        signal=signal,
        feature_payload={"atr_pct": 0.001, "trend_strength": 0.6},
        mid=100.0,
        market_snapshot={
            "source": "websocket",
            "spread_bps": 1.0,
            "liquidity_usd": 100_000.0,
            "volatility_pct": 0.4,
            "signal_stability": 0.95,
        },
    )

    assert payload["no_trade_reason"] == "signal_debounce_active"


def test_live_signal_quality_blocks_stale_v2_market_data(app) -> None:
    app.config["ML_RANKER_ENABLED"] = False
    app.config["NET_ROI_MIN_FILL_QUALITY"] = 0.0
    evaluator = SignalQualityEvaluator(app.config)
    signal = Signal("buy", "stale feed", "1m", 99.0, 104.0, 0.1)

    payload = evaluator.evaluate(
        symbol="BTC",
        timeframe="1m",
        mode="live",
        run_parameters={"take_profit_pct": 0.04},
        signal=signal,
        feature_payload={"atr_pct": 0.001, "trend_strength": 0.7},
        mid=100.0,
        market_snapshot={
            "source": "websocket",
            "spread_bps": 1.0,
            "liquidity_usd": 100_000.0,
            "volatility_pct": 0.4,
            "signal_stability": 0.95,
            "stale_data": True,
            "stale_data_age_seconds": 7200.0,
        },
    )

    assert payload["no_trade_reason"] == "stale_signal_market_data"
    assert payload["roi_rejection_risk"] == "high"


def test_live_signal_quality_blocks_fragile_v2_regime(app) -> None:
    app.config["ML_RANKER_ENABLED"] = False
    app.config["NET_ROI_MIN_FILL_QUALITY"] = 0.0
    evaluator = SignalQualityEvaluator(app.config)
    signal = Signal("buy", "hostile regime", "1m", 99.0, 110.0, 0.1)

    payload = evaluator.evaluate(
        symbol="BTC",
        timeframe="1m",
        mode="live",
        run_parameters={"take_profit_pct": 0.10},
        signal=signal,
        feature_payload={"atr_pct": 0.001, "trend_strength": 0.7},
        mid=100.0,
        market_snapshot={
            "source": "websocket",
            "spread_bps": 42.0,
            "liquidity_usd": 100_000.0,
            "volatility_regime": "dislocated",
            "volatility_pct": 3.2,
            "signal_stability": 0.95,
        },
    )

    assert payload["regime_support"] == "regime-fragile"
    assert payload["no_trade_reason"] == "fragile_roi_v2_regime"


def test_aggressive_1h_maker_limit_requires_edge_after_costs(app) -> None:
    manager = app.extensions["services"]["strategy_manager"]
    run = StrategyRun(strategy_name="scalping", symbol="BTC", timeframe="1m", mode="live")
    run.parameters = {"optimizer_profile": "aggressive_1h", "execution_style": "maker_limit"}
    signal = Signal("buy", "entry", "1m", 99.0, 101.0, 0.1)

    order_type, limit_price = manager._execution_order_shape(
        run,
        signal,
        "buy",
        100.0,
        {
            "edge_score": 20.0,
            "cost_drag_bps": 10.0,
            "spread_bps": 2.0,
            "suggested_execution_style": "maker_limit",
        },
    )
    assert (order_type, limit_price) == ("market", None)

    order_type, limit_price = manager._execution_order_shape(
        run,
        signal,
        "buy",
        100.0,
        {
            "edge_score": 30.0,
            "cost_drag_bps": 10.0,
            "spread_bps": 2.0,
            "suggested_execution_style": "maker_limit",
        },
    )
    assert order_type == "limit"
    assert limit_price is not None
    assert limit_price < 100.0


def test_protective_exit_preserves_vault_and_optimizer_metadata(app, monkeypatch) -> None:
    order_manager = app.extensions["services"]["order_manager"]
    source = Order(
        client_order_id="source-order",
        mode="paper",
        symbol="BTC",
        side="buy",
        order_type="market",
        status="filled",
        strategy_name="scalping",
        quantity=1.0,
        stop_loss=95.0,
        take_profit=110.0,
    )
    source.details = {
        "vault_cycle_id": 42,
        "execution_mode": "paper",
        "optimizer_profile": "aggressive_1h",
        "experimental": True,
        "risk_label": "Very High Risk",
        "algorithm_profile": "Aggressive",
        "consumer_vault": True,
        "edge_score": 7.5,
        "cost_drag_bps": 4.0,
    }
    db.session.add(source)
    db.session.commit()

    captured = {}
    monkeypatch.setattr(order_manager, "current_position", lambda *args, **kwargs: {"quantity": 1.0})
    monkeypatch.setattr(order_manager, "_safe_market_price", lambda symbol, mode: 94.0)

    def fake_place_order(intent):
        captured.update(intent.metadata)
        return Order(client_order_id=intent.idempotency_key, mode=intent.mode, symbol=intent.symbol)

    monkeypatch.setattr(order_manager, "place_order", fake_place_order)

    order_manager.enforce_protective_exit("BTC", "paper")

    assert captured["vault_cycle_id"] == 42
    assert captured["optimizer_profile"] == "aggressive_1h"
    assert captured["experimental"] is True
    assert captured["risk_label"] == "Very High Risk"
    assert captured["algorithm_profile"] == "Aggressive"
    assert captured["edge_score"] == 7.5


def test_scalping_strategy_reduces_when_momentum_fades_after_favorable_move() -> None:
    strategy = ScalpingStrategy(
        {
            "momentum_lookback": 3,
            "breakeven_trigger_pct": 0.001,
            "trailing_stop_pct": 0.001,
            "fast_fade_exit_pct": 0.0005,
        }
    )
    candles = [
        {"close": 100.0},
        {"close": 100.5},
        {"close": 100.9},
        {"close": 101.2},
        {"close": 100.4},
    ]

    signal = strategy.generate_signal(
        symbol="BTC",
        timeframe="1m",
        candles=candles,
        position={"quantity": 1.0, "entry_price": 100.0},
    )

    assert signal.action == "reduce"
    assert "faded" in signal.rationale


def test_vault_selector_prefers_regime_matched_aggressive_rankings(app) -> None:
    selector = app.extensions["services"]["vault_strategy_selector"]
    market_data = app.extensions["services"]["market_data"]
    market_data.get_order_book = lambda symbol, mode: {
        "levels": [[{"px": "99.95", "sz": "1000"}], [{"px": "100.05", "sz": "1000"}]]
    }
    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    for name in ["scalping", "rsi_mean_reversion", "volatility_breakout"]:
        ranking = StrategyRanking(
            optimizer_run_id=optimizer_run.id,
            strategy_name=name,
            symbol="BTC",
            timeframe="1m",
            profile="aggressive_1h",
            experimental=True,
            risk_label="Very High Risk",
            score=1.0,
            recent_1h_return=0.02,
            max_drawdown=-0.03,
            profit_factor=1.5,
            trade_count=20,
            edge_score=8.0,
            rejected=False,
        )
        ranking.parameters = {"risk_fraction": 0.02}
        db.session.add(ranking)
    db.session.commit()

    def choose(closes: list[float]) -> str:
        market_data.get_candles = lambda symbol, timeframe, mode, limit: [
            {"close": close, "timestamp": index} for index, close in enumerate(closes)
        ]
        return selector.select("BTC", 1, "paper").strategy_name

    assert choose([100 + index * 0.2 for index in range(40)]) == "scalping"
    assert choose(([100.0, 100.8] * 14) + ([100.0, 100.6] * 5) + [100.6, 99.9]) == "rsi_mean_reversion"
    assert choose(([100.0, 100.1] * 14) + [99.0, 99.4, 100.0, 101.2, 102.0, 101.4, 100.6, 102.2, 101.0, 102.5, 101.8, 102.8]) == "volatility_breakout"


def _strategy_candles(count: int = 90) -> list[dict[str, float]]:
    price = 100.0
    rows: list[dict[str, float]] = []
    for index in range(count):
        price += 0.12 if index % 5 else -0.04
        rows.append(
            {
                "timestamp": float(index + 1),
                "open": price - 0.05,
                "high": price + 0.35,
                "low": price - 0.35,
                "close": price,
                "volume": 1000.0 + index,
            }
        )
    return rows


def test_all_strategies_emit_consistent_metadata_without_lookahead() -> None:
    registry = StrategyRegistry()
    rows = _strategy_candles()

    for strategy_name in registry.names():
        strategy = registry.build(strategy_name, {})
        for timeframe in ["1m", "5m", "15m", "1h"]:
            signal = strategy.generate_signal(
                symbol="BTC",
                timeframe=timeframe,
                candles=rows,
                position={"quantity": 0.0, "entry_price": 0.0},
            )
            assert signal.action in {"buy", "sell", "reduce", "hold"}
            assert signal.metadata["symbol"] == "BTC"
            assert signal.metadata["timeframe"] == timeframe
            assert signal.metadata["strategy"] == strategy_name
            assert signal.metadata["signal_timestamp"] == rows[-1]["timestamp"]
            assert "indicators" in signal.metadata
            assert "thresholds" in signal.metadata
            assert "risk" in signal.metadata


def test_strategies_handle_insufficient_and_malformed_history_safely() -> None:
    registry = StrategyRegistry()
    malformed = [{"timestamp": 1, "close": "bad"}, {"timestamp": 2}, object()]

    for strategy_name in registry.names():
        strategy = registry.build(strategy_name, {})
        short_signal = strategy.generate_signal(symbol="BTC", timeframe="1m", candles=[], position={"quantity": 0.0})
        assert short_signal.action == "hold"
        assert short_signal.metadata.get("no_trade_reason") in {
            "insufficient_history",
            "invalid_candle_data",
            "invalid_price",
            None,
        }

        malformed_signal = strategy.generate_signal(
            symbol="BTC",
            timeframe="1m",
            candles=malformed,
            position={"quantity": 0.0},
        )
        assert malformed_signal.action == "hold"
        assert malformed_signal.metadata["strategy"] == strategy_name


def test_rule_based_low_reward_risk_downgrades_to_hold() -> None:
    strategy = RuleBasedSignalStrategy(
        {
            "minimum_signal_score": 0.1,
            "trend_weight": 1.0,
            "rsi_weight": 0.0,
            "volume_weight": 0.0,
            "fibonacci_filter_weight": 0.0,
            "atr_stop_multiplier": 2.0,
            "atr_take_multiplier": 0.1,
        }
    )
    strategy.feature_engine.snapshot = lambda **kwargs: type(
        "Snapshot",
        (),
        {
            "as_dict": lambda self: {
                "symbol": "BTC",
                "timeframe": "15m",
                "timestamp": 90,
                "close": 100.0,
                "ema_fast": 101.0,
                "ema_slow": 100.0,
                "ema_trend": 1.0,
                "trend_strength": 0.01,
                "rsi": 50.0,
                "atr": 1.0,
                "atr_pct": 0.01,
                "volatility": 0.001,
                "volume_spike": {"is_spike": False},
                "external_scores": {},
                "pattern_prediction": {"probability": 0.5, "confidence": 0.0, "label": "neutral"},
                "fibonacci_levels": {},
            }
        },
    )()

    signal = strategy.generate_signal(
        symbol="BTC",
        timeframe="15m",
        candles=_strategy_candles(),
        position={"quantity": 0.0},
    )

    assert signal.action == "hold"
    assert signal.stop_loss is None
    assert signal.take_profit is None
    assert signal.metadata["no_trade_reason"] == "reward_risk_below_minimum"
