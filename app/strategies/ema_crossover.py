"""EMA crossover strategy."""

from __future__ import annotations

from typing import Any

from ..features.indicators import ema_series
from .base import BaseStrategy, Signal


def _ema(values: list[float], period: int) -> list[float]:
    """Return EMA values for a price series."""
    if period <= 0 or not values:
        return []

    multiplier = 2 / (period + 1)
    ema_values = [values[0]]

    for value in values[1:]:
        ema_values.append((value - ema_values[-1]) * multiplier + ema_values[-1])

    return ema_values


class EmaCrossoverStrategy(BaseStrategy):
    """Uses fast and slow EMAs to generate trend signals."""

    name = "ema_crossover"
    description = "Trend-following EMA crossover with spread compression exits."

    default_parameters = {
        "fast_period": 9,
        "slow_period": 21,
        "risk_fraction": 0.1,
        "stop_loss_pct": 0.01,
        "take_profit_pct": 0.02,
        "compression_exit_pct": 0.0001,
        "leverage": 1.0,
    }

    @classmethod
    def parameter_grid(cls) -> dict[str, list[Any]]:
        return {
            "fast_period": [5, 8, 9, 12],
            "slow_period": [13, 21, 34],
            "risk_fraction": [0.05, 0.08, 0.1],
            "stop_loss_pct": [0.004, 0.008, 0.01],
            "take_profit_pct": [0.008, 0.015, 0.02],
            "compression_exit_pct": [0.00005, 0.0001, 0.0002],
            "leverage": [1.0],
        }

    def generate_signal(
        self,
        *,
        symbol: str,
        timeframe: str,
        candles: list[dict[str, Any]],
        position: dict[str, Any],
    ) -> Signal:
        self.validate_timeframe(timeframe)

        fast_period = max(2, int(self.parameters["fast_period"]))
        slow_period = max(3, int(self.parameters["slow_period"]))

        thresholds = {
            "fast_period": fast_period,
            "slow_period": slow_period,
            "compression_exit_pct": self.parameters.get("compression_exit_pct", 0.0001),
        }

        if fast_period >= slow_period:
            return Signal(
                "hold",
                "Fast EMA period must be below slow EMA period.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="invalid_ema_periods"),
            )

        try:
            closes = [float(row["close"]) for row in candles if row.get("close") is not None]
        except (AttributeError, TypeError, ValueError):
            return Signal(
                "hold",
                "Invalid candle close data.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="invalid_candle_data"),
            )

        if len(closes) < slow_period + 2:
            return Signal(
                "hold",
                "Waiting for more candle history.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="insufficient_history"),
            )

        last_price = closes[-1]

        if last_price <= 0:
            return Signal(
                "hold",
                "Invalid price data.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="invalid_price"),
            )

        fast = ema_series(closes, fast_period)
        slow = ema_series(closes, slow_period)

        previous_diff = fast[-2] - slow[-2]
        current_diff = fast[-1] - slow[-1]

        qty = float(position.get("quantity", 0.0))
        stop_loss_pct = float(self.parameters["stop_loss_pct"])
        take_profit_pct = float(self.parameters["take_profit_pct"])
        size = float(self.parameters["risk_fraction"])
        compression_exit_pct = float(self.parameters.get("compression_exit_pct", 0.0001))

        bullish_cross = previous_diff <= 0 < current_diff
        bearish_cross = previous_diff >= 0 > current_diff
        indicators = {
            "fast_ema": fast[-1],
            "slow_ema": slow[-1],
            "previous_ema_diff": previous_diff,
            "current_ema_diff": current_diff,
        }
        risk = {
            "risk_fraction": size,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        }
        confidence = min(abs(current_diff) / max(last_price * compression_exit_pct, 1e-9), 1.0)

        if bullish_cross and qty <= 0:
            stop_loss = last_price * (1 - stop_loss_pct)
            take_profit = last_price * (1 + take_profit_pct)
            return Signal(
                "buy",
                f"Fast EMA crossed above slow EMA on {symbol}.",
                timeframe,
                stop_loss,
                take_profit,
                size,
                self.signal_metadata(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=candles,
                    indicators=indicators,
                    thresholds=thresholds,
                    risk={**risk, "stop_loss": stop_loss, "take_profit": take_profit},
                    confidence=confidence,
                ),
            )

        if bearish_cross and qty >= 0:
            stop_loss = last_price * (1 + stop_loss_pct)
            take_profit = last_price * (1 - take_profit_pct)
            return Signal(
                "sell",
                f"Fast EMA crossed below slow EMA on {symbol}.",
                timeframe,
                stop_loss,
                take_profit,
                size,
                self.signal_metadata(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=candles,
                    indicators=indicators,
                    thresholds=thresholds,
                    risk={**risk, "stop_loss": stop_loss, "take_profit": take_profit},
                    confidence=confidence,
                ),
            )

        if qty != 0 and abs(current_diff) < last_price * compression_exit_pct:
            return Signal(
                "reduce",
                "EMA spread compressed while a position is open.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=candles,
                    indicators=indicators,
                    thresholds=thresholds,
                    risk=risk,
                    components={"exit_reason": "ema_spread_compression"},
                    confidence=confidence,
                ),
            )

        return Signal(
            "hold",
            "No crossover signal.",
            timeframe,
            None,
            None,
            0.0,
            self.signal_metadata(
                symbol=symbol,
                timeframe=timeframe,
                candles=candles,
                indicators=indicators,
                thresholds=thresholds,
                risk=risk,
                confidence=confidence,
                no_trade_reason="no_crossover",
            ),
        )
