from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.backtesting.engine import BacktestConfig, BacktestEngine
from app.strategies.base import Signal


class _BuyThenReduceStrategy:
    parameters: dict[str, Any] = {}

    def generate_signal(self, *, symbol: str, timeframe: str, candles: list[dict[str, Any]], position: dict[str, Any]) -> Signal:
        if position.get("quantity", 0.0):
            return Signal("reduce", "close", timeframe, None, None, 0.0)
        return Signal("buy", "open", timeframe, candles[-1]["close"] * 0.99, candles[-1]["close"] * 1.02, 0.2)


class _Registry:
    def build(self, name: str, parameters: dict[str, Any] | None = None):
        return _BuyThenReduceStrategy()


class _BuyHoldWithSignalStopStrategy:
    parameters: dict[str, Any] = {}

    def __init__(self) -> None:
        self.opened = False

    def generate_signal(self, *, symbol: str, timeframe: str, candles: list[dict[str, Any]], position: dict[str, Any]) -> Signal:
        if position.get("quantity", 0.0):
            return Signal("hold", "hold", timeframe, None, None, 0.0, {"signal_timestamp": candles[-1]["timestamp"]})
        if self.opened:
            return Signal("hold", "already tested", timeframe, None, None, 0.0, {"signal_timestamp": candles[-1]["timestamp"]})
        self.opened = True
        return Signal("buy", "open", timeframe, 99.5, 101.0, 0.2, {"signal_timestamp": candles[-1]["timestamp"]})


class _SellThenReduceStrategy:
    parameters: dict[str, Any] = {}

    def generate_signal(self, *, symbol: str, timeframe: str, candles: list[dict[str, Any]], position: dict[str, Any]) -> Signal:
        if position.get("quantity", 0.0):
            return Signal("reduce", "close", timeframe, None, None, 0.0)
        return Signal("sell", "open short", timeframe, candles[-1]["close"] * 1.01, candles[-1]["close"] * 0.98, 0.2)


class _CustomRegistry:
    def __init__(self, strategy):
        self.strategy = strategy

    def build(self, name: str, parameters: dict[str, Any] | None = None):
        return self.strategy


class _MarketData:
    def get_candles(self, *args, **kwargs):
        return []


def _candles(count: int = 80) -> list[dict[str, Any]]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 100.0
    for index in range(count):
        price += 0.25
        timestamp = int((start + timedelta(minutes=15 * index)).timestamp() * 1000)
        rows.append(
            {
                "timestamp": timestamp,
                "open": price - 0.1,
                "high": price + 0.4,
                "low": price - 0.4,
                "close": price,
                "volume": 1000,
            }
        )
    return rows


def _config(**overrides: Any) -> BacktestConfig:
    payload = {
        "strategy_name": "test",
        "symbol": "BTC",
        "timeframe": "15m",
        "mode": "testnet",
        "initial_balance": 1_000.0,
        "fee_bps": 5.0,
        "slippage_bps": 5.0,
        "stop_loss_pct": 0.01,
        "take_profit_pct": 0.02,
        "position_size_fraction": 0.2,
        "parameters": {},
        "max_daily_loss": 100.0,
        "max_drawdown_pct": 0.5,
        "intrabar_model": "conservative",
    }
    payload.update(overrides)
    return BacktestConfig(**payload)


def test_backtest_tracks_short_term_metrics() -> None:
    engine = BacktestEngine({}, _Registry(), _MarketData())
    result = engine.run(_config(), _candles())

    assert result["trade_count"] > 0
    assert result["fees_paid"] > 0
    assert result["trades_per_day"] > 0
    assert result["average_trade_duration_minutes"] >= 0
    assert result["capital_turnover_rate"] > 0
    assert "edge_score" in result
    assert "expectancy" in result
    assert "cost_drag_bps" in result
    assert "net_return_after_costs" in result
    assert "turnover_after_fees" in result
    assert "risk_event_count" in result
    assert result["cost_drag_bps"] >= 0
    assert result["max_favorable_excursion"] >= result["max_adverse_excursion"]
    assert result["equity_curve"]
    assert result["drawdown_curve"]


def test_conservative_intrabar_uses_stop_when_stop_and_take_hit() -> None:
    engine = BacktestEngine({}, _Registry(), _MarketData())
    exit_price, reason = engine._intrabar_exit(
        {"high": 103.0, "low": 98.0},
        entry_price=100.0,
        quantity=1.0,
        stop_pct=0.01,
        take_pct=0.02,
        model="conservative",
    )

    assert exit_price == 99.0
    assert reason == "stop_loss"


def test_risk_based_sizing_uses_stop_distance() -> None:
    engine = BacktestEngine({}, _Registry(), _MarketData())
    signal = Signal("buy", "risk", "15m", 99.0, 102.0, 0.5)
    qty = engine._position_quantity(
        _config(sizing_mode="risk_based", risk_per_trade_pct=0.01, position_size_fraction=0.5),
        signal,
        price=100.0,
        cash=1_000.0,
        equity=1_000.0,
    )

    assert qty == 5.0


def test_backtest_uses_signal_stop_take_for_intrabar_exit() -> None:
    rows = _candles(32)
    rows[25]["close"] = 100.0
    rows[25]["high"] = 100.2
    rows[25]["low"] = 99.9
    rows[26]["close"] = 99.8
    rows[26]["high"] = 100.1
    rows[26]["low"] = 99.4
    engine = BacktestEngine({}, _CustomRegistry(_BuyHoldWithSignalStopStrategy()), _MarketData())

    result = engine.run(
        _config(stop_loss_pct=0.5, take_profit_pct=0.5, slippage_bps=0.0, fee_bps=0.0),
        rows,
    )

    assert result["trade_count"] == 1
    assert result["trades"][0]["reason"] == "stop_loss"
    assert result["trades"][0]["exit_price"] == 99.5


def test_backtest_handles_short_pnl_and_signal_reduce() -> None:
    engine = BacktestEngine({}, _CustomRegistry(_SellThenReduceStrategy()), _MarketData())
    rows = _candles(40)
    for index, row in enumerate(rows):
        row["close"] = 120.0 - index * 0.2
        row["high"] = row["close"] + 0.1
        row["low"] = row["close"] - 0.1

    result = engine.run(_config(slippage_bps=0.0, fee_bps=0.0), rows)

    assert result["trade_count"] > 0
    assert any(trade["direction"] == "short" for trade in result["trades"])
    assert result["realized_pnl"] > 0
