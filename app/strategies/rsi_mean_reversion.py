"""Short-window RSI mean reversion strategy."""

from __future__ import annotations

from typing import Any

from ..features.indicators import rsi
from .base import BaseStrategy, Signal


def _rsi(closes: list[float], period: int) -> float:
    """Calculate simple RSI over the latest period."""
    if period <= 0 or len(closes) < period + 1:
        return 50.0

    gains = 0.0
    losses = 0.0

    for index in range(-period, 0):
        change = closes[index] - closes[index - 1]
        if change > 0:
            gains += change
        elif change < 0:
            losses += abs(change)

    if gains == 0 and losses == 0:
        return 50.0
    if losses == 0:
        return 100.0
    if gains == 0:
        return 0.0

    relative_strength = gains / losses
    return 100 - (100 / (1 + relative_strength))


class RsiMeanReversionStrategy(BaseStrategy):
    """Trades short-term RSI extremes back toward neutral."""

    name = "rsi_mean_reversion"
    description = "Short-window RSI mean reversion for intraday oversold/overbought moves."

    default_parameters = {
        "period": 7,
        "oversold": 28,
        "overbought": 72,
        "exit_rsi": 50,
        "risk_fraction": 0.06,
        "stop_loss_pct": 0.006,
        "take_profit_pct": 0.01,
        "reentry_buffer": 3,
        "leverage": 1.0,
    }

    @classmethod
    def parameter_grid(cls) -> dict[str, list[Any]]:
        return {
            "period": [5, 7, 9],
            "oversold": [25, 30, 35],
            "overbought": [65, 70, 75],
            "exit_rsi": [48, 50, 52],
            "risk_fraction": [0.04, 0.06, 0.08],
            "stop_loss_pct": [0.004, 0.006, 0.01],
            "take_profit_pct": [0.006, 0.01, 0.014],
            "reentry_buffer": [2, 3, 5],
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

        period = max(2, int(self.parameters["period"]))
        thresholds = {
            "period": period,
            "oversold": self.parameters.get("oversold"),
            "overbought": self.parameters.get("overbought"),
            "exit_rsi": self.parameters.get("exit_rsi"),
            "reentry_buffer": self.parameters.get("reentry_buffer", 0.0),
        }

        try:
            closes = [float(row["close"]) for row in candles if row.get("close") is not None]
        except (AttributeError, TypeError, ValueError):
            return Signal(
                "hold",
                "Invalid RSI candle data.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="invalid_candle_data"),
            )

        if len(closes) < period + 2:
            return Signal(
                "hold",
                "Waiting for RSI history.",
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

        value = rsi(closes, period)

        qty = float(position.get("quantity", 0.0))
        oversold = float(self.parameters["oversold"])
        overbought = float(self.parameters["overbought"])
        exit_rsi = float(self.parameters["exit_rsi"])
        reentry_buffer = float(self.parameters.get("reentry_buffer", 0.0))

        stop_loss_pct = float(self.parameters["stop_loss_pct"])
        take_profit_pct = float(self.parameters["take_profit_pct"])
        risk_fraction = float(self.parameters["risk_fraction"])
        indicators = {"rsi": value, "last_price": last_price}
        risk = {
            "risk_fraction": risk_fraction,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        }
        confidence = min(max(abs(value - exit_rsi) / max(abs(overbought - oversold), 1.0), 0.0), 1.0)

        if oversold >= overbought:
            return Signal(
                "hold",
                "Invalid RSI thresholds.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, no_trade_reason="invalid_thresholds"),
            )

        if qty > 0 and value >= exit_rsi:
            return Signal(
                "reduce",
                "RSI reverted toward neutral while long.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, components={"exit_reason": "rsi_reverted_long"}, confidence=confidence),
            )

        if qty < 0 and value <= exit_rsi:
            return Signal(
                "reduce",
                "RSI reverted toward neutral while short.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, components={"exit_reason": "rsi_reverted_short"}, confidence=confidence),
            )

        # Avoid repeatedly adding to the same direction unless RSI reaches a deeper extreme.
        if value <= oversold and qty <= 0:
            return Signal(
                "buy",
                f"{symbol} RSI {value:.1f} is oversold.",
                timeframe,
                last_price * (1 - stop_loss_pct),
                last_price * (1 + take_profit_pct),
                risk_fraction,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk={**risk, "stop_loss": last_price * (1 - stop_loss_pct), "take_profit": last_price * (1 + take_profit_pct)}, confidence=confidence),
            )

        if value <= oversold - reentry_buffer and qty > 0:
            return Signal(
                "buy",
                f"{symbol} RSI {value:.1f} reached a deeper oversold extreme.",
                timeframe,
                last_price * (1 - stop_loss_pct),
                last_price * (1 + take_profit_pct),
                risk_fraction * 0.5,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk={**risk, "risk_fraction": risk_fraction * 0.5, "stop_loss": last_price * (1 - stop_loss_pct), "take_profit": last_price * (1 + take_profit_pct)}, confidence=confidence),
            )

        if value >= overbought and qty >= 0:
            return Signal(
                "sell",
                f"{symbol} RSI {value:.1f} is overbought.",
                timeframe,
                last_price * (1 + stop_loss_pct),
                last_price * (1 - take_profit_pct),
                risk_fraction,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk={**risk, "stop_loss": last_price * (1 + stop_loss_pct), "take_profit": last_price * (1 - take_profit_pct)}, confidence=confidence),
            )

        if value >= overbought + reentry_buffer and qty < 0:
            return Signal(
                "sell",
                f"{symbol} RSI {value:.1f} reached a deeper overbought extreme.",
                timeframe,
                last_price * (1 + stop_loss_pct),
                last_price * (1 - take_profit_pct),
                risk_fraction * 0.5,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk={**risk, "risk_fraction": risk_fraction * 0.5, "stop_loss": last_price * (1 + stop_loss_pct), "take_profit": last_price * (1 - take_profit_pct)}, confidence=confidence),
            )

        return Signal(
            "hold",
            "RSI is not at an actionable extreme.",
            timeframe,
            None,
            None,
            0.0,
            self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, confidence=confidence, no_trade_reason="rsi_not_extreme"),
        )
