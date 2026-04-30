from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.backtesting.engine import BacktestConfig
from app.extensions import db
from app.models import Order
from app.services.order_manager import OrderIntent


def candles(count: int = 100) -> list[dict[str, Any]]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 100.0
    for index in range(count):
        price += 0.3
        rows.append(
            {
                "timestamp": int((start + timedelta(minutes=15 * index)).timestamp() * 1000),
                "open": price - 0.1,
                "high": price + 0.5,
                "low": price - 0.4,
                "close": price,
                "volume": 1000 * (2 if index % 4 == 0 else 1),
            }
        )
    return rows


def test_backtest_trades_include_feature_logs(app) -> None:
    engine = app.extensions["services"]["backtest_engine"]
    result = engine.run(
        BacktestConfig(
            strategy_name="rule_based_signal",
            symbol="BTC",
            timeframe="15m",
            mode="testnet",
            initial_balance=1_000.0,
            fee_bps=5.0,
            slippage_bps=5.0,
            stop_loss_pct=0.01,
            take_profit_pct=0.02,
            position_size_fraction=0.1,
            parameters={
                "minimum_signal_score": 0.5,
                "trend_weight": 0.6,
                "external_weight": 0.0,
                "volume_spike_multiplier": 1.1,
            },
        ),
        candles=candles(),
    )

    assert result["trade_count"] > 0
    assert result["feature_audit_summary"]["feature_logged_trades"] == result["trade_count"]
    assert "entry_features" in result["trades"][0]
    assert "pattern_prediction" in result["trades"][0]


def test_live_order_metadata_preserves_feature_snapshot_on_rejection(app) -> None:
    order_manager = app.extensions["services"]["order_manager"]
    order_manager.market_data.get_mid_price = lambda symbol, mode: 100.0

    order = order_manager.place_order(
        OrderIntent(
            symbol="BTC",
            side="buy",
            quantity=1.0,
            mode="live",
            stop_loss=95.0,
            take_profit=110.0,
            slippage_pct=0.0,
            metadata={
                "feature_snapshot": {"ema_trend": 1.0},
                "rule_based_signal": True,
                "risk_reward": 2.0,
                "pattern_prediction": {"label": "neutral", "probability": 0.5, "confidence": 0.0},
            },
        )
    )

    assert order.status == "rejected"
    assert db.session.get(Order, order.id).details["feature_snapshot"] == {"ema_trend": 1.0}


def test_rule_based_invalid_reward_risk_is_rejected(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = OrderIntent(
        symbol="BTC",
        side="buy",
        quantity=1.0,
        mode="live",
        stop_loss=99.0,
        take_profit=100.5,
        metadata={"rule_based_signal": True, "risk_reward": 0.5},
    )

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "invalid_reward_risk"


def test_pattern_prediction_cannot_approve_trade_without_stop(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = OrderIntent(
        symbol="BTC",
        side="buy",
        quantity=1.0,
        mode="live",
        metadata={"pattern_prediction": {"label": "bullish", "probability": 0.99, "confidence": 0.99}},
    )

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "stop_loss_required"
