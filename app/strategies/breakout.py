"""Breakout strategy using recent highs and lows."""

from __future__ import annotations

from typing import Any

from .base import BaseStrategy, Signal


class BreakoutStrategy(BaseStrategy):
    """Trades confirmed breaks above resistance or below support."""

    name = "breakout"
    description = "Channel breakout strategy with support and resistance bands."

    default_parameters = {
        "lookback": 20,
        "confirmation_buffer_pct": 0.002,
        "risk_fraction": 0.08,
        "stop_loss_pct": 0.009,
        "take_profit_pct": 0.02,
        "exit_buffer_pct": 0.001,
        "leverage": 1.0,
    }

    @classmethod
    def parameter_grid(cls) -> dict[str, list[Any]]:
        return {
            "lookback": [10, 20, 30],
            "confirmation_buffer_pct": [0.001, 0.002, 0.004],
            "risk_fraction": [0.04, 0.08, 0.1],
            "stop_loss_pct": [0.004, 0.009, 0.014],
            "take_profit_pct": [0.01, 0.02, 0.03],
            "exit_buffer_pct": [0.0, 0.001, 0.002],
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

        lookback = max(2, int(self.parameters["lookback"]))
        thresholds = {
            "lookback": lookback,
            "confirmation_buffer_pct": self.parameters.get("confirmation_buffer_pct"),
            "exit_buffer_pct": self.parameters.get("exit_buffer_pct", 0.0),
        }

        if len(candles) < lookback + 1:
            return Signal(
                "hold",
                "Waiting for breakout lookback window.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="insufficient_history"),
            )

        recent = candles[-lookback - 1 : -1]

        try:
            highs = [float(row["high"]) for row in recent]
            lows = [float(row["low"]) for row in recent]
            last_price = float(candles[-1]["close"])
        except (KeyError, TypeError, ValueError):
            return Signal(
                "hold",
                "Invalid candle data.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="invalid_candle_data"),
            )

        if last_price <= 0 or not highs or not lows:
            return Signal(
                "hold",
                "Invalid price data.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="invalid_price"),
            )

        resistance = max(highs)
        support = min(lows)

        if support <= 0 or resistance <= 0 or support >= resistance:
            return Signal(
                "hold",
                "Invalid breakout channel.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="invalid_channel"),
            )

        buffer_pct = float(self.parameters["confirmation_buffer_pct"])
        exit_buffer_pct = float(self.parameters.get("exit_buffer_pct", 0.0))
        stop_loss_pct = float(self.parameters["stop_loss_pct"])
        take_profit_pct = float(self.parameters["take_profit_pct"])
        size = float(self.parameters["risk_fraction"])
        qty = float(position.get("quantity", 0.0))

        long_breakout = last_price > resistance * (1 + buffer_pct)
        short_breakout = last_price < support * (1 - buffer_pct)
        indicators = {
            "resistance": resistance,
            "support": support,
            "channel_width_pct": (resistance - support) / last_price,
            "last_price": last_price,
        }
        risk = {
            "risk_fraction": size,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        }
        breakout_distance = max(
            abs(last_price - resistance) / max(resistance, 1e-9),
            abs(last_price - support) / max(support, 1e-9),
        )
        confidence = min(breakout_distance / max(buffer_pct, 1e-9), 1.0)

        # Exit failed breakouts when price returns inside the channel.
        if qty > 0 and last_price < resistance * (1 - exit_buffer_pct):
            return Signal(
                "reduce",
                "Long breakout failed and price returned inside the channel.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, components={"exit_reason": "long_breakout_failed"}, confidence=confidence),
            )

        if qty < 0 and last_price > support * (1 + exit_buffer_pct):
            return Signal(
                "reduce",
                "Short breakout failed and price returned inside the channel.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, components={"exit_reason": "short_breakout_failed"}, confidence=confidence),
            )

        # Avoid adding to an existing long.
        if long_breakout and qty <= 0:
            return Signal(
                "buy",
                f"Price broke above recent resistance at {resistance:.2f}.",
                timeframe,
                last_price * (1 - stop_loss_pct),
                last_price * (1 + take_profit_pct),
                size,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk={**risk, "stop_loss": last_price * (1 - stop_loss_pct), "take_profit": last_price * (1 + take_profit_pct)}, confidence=confidence),
            )

        # Avoid adding to an existing short.
        if short_breakout and qty >= 0:
            return Signal(
                "sell",
                f"Price broke below recent support at {support:.2f}.",
                timeframe,
                last_price * (1 + stop_loss_pct),
                last_price * (1 - take_profit_pct),
                size,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk={**risk, "stop_loss": last_price * (1 + stop_loss_pct), "take_profit": last_price * (1 - take_profit_pct)}, confidence=confidence),
            )

        return Signal(
            "hold",
            "No confirmed breakout.",
            timeframe,
            None,
            None,
            0.0,
            self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, confidence=confidence, no_trade_reason="no_confirmed_breakout"),
        )
