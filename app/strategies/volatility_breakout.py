"""Volatility breakout strategy for short-term range expansion."""

from __future__ import annotations

from statistics import mean
from typing import Any

from .base import BaseStrategy, Signal


class VolatilityBreakoutStrategy(BaseStrategy):
    """Trades expansion beyond a recent volatility-adjusted channel."""

    name = "volatility_breakout"
    description = "Range expansion breakout using average candle range as a volatility filter."

    default_parameters = {
        "lookback": 16,
        "range_multiplier": 1.5,
        "risk_fraction": 0.06,
        "stop_loss_pct": 0.007,
        "take_profit_pct": 0.014,
        "fade_exit_ratio": 0.25,
        "leverage": 1.0,
    }

    @classmethod
    def parameter_grid(cls) -> dict[str, list[Any]]:
        return {
            "lookback": [8, 16, 24],
            "range_multiplier": [1.2, 1.5, 2.0],
            "risk_fraction": [0.04, 0.06, 0.08],
            "stop_loss_pct": [0.004, 0.007, 0.011],
            "take_profit_pct": [0.008, 0.014, 0.02],
            "fade_exit_ratio": [0.2, 0.25, 0.35],
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
            "range_multiplier": self.parameters.get("range_multiplier"),
            "fade_exit_ratio": self.parameters.get("fade_exit_ratio", 0.25),
        }

        if len(candles) < lookback + 2:
            return Signal(
                "hold",
                "Waiting for volatility breakout history.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="insufficient_history"),
            )

        recent = candles[-lookback - 1 : -1]
        last = candles[-1]

        try:
            ranges = [float(row["high"]) - float(row["low"]) for row in recent]
            closes = [float(row["close"]) for row in recent]
            last_price = float(last["close"])
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

        if last_price <= 0 or not ranges or not closes:
            return Signal(
                "hold",
                "Invalid price data.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="invalid_price"),
            )

        avg_range = mean(ranges)

        if avg_range <= 0:
            return Signal(
                "hold",
                "Insufficient volatility for breakout signal.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="insufficient_volatility"),
            )

        channel_mid = mean(closes)
        threshold = avg_range * float(self.parameters["range_multiplier"])

        qty = float(position.get("quantity", 0.0))
        stop_loss_pct = float(self.parameters["stop_loss_pct"])
        take_profit_pct = float(self.parameters["take_profit_pct"])
        risk_fraction = float(self.parameters["risk_fraction"])
        fade_exit_ratio = float(self.parameters.get("fade_exit_ratio", 0.25))

        upper_breakout = channel_mid + threshold
        lower_breakout = channel_mid - threshold
        fade_zone = avg_range * fade_exit_ratio
        indicators = {
            "avg_range": avg_range,
            "channel_mid": channel_mid,
            "upper_breakout": upper_breakout,
            "lower_breakout": lower_breakout,
            "last_price": last_price,
        }
        risk = {
            "risk_fraction": risk_fraction,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        }
        confidence = min(abs(last_price - channel_mid) / max(threshold, 1e-9), 1.0)

        if qty > 0 and last_price < channel_mid + fade_zone:
            return Signal(
                "reduce",
                "Long volatility breakout faded back toward the channel.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, components={"exit_reason": "long_volatility_breakout_faded"}, confidence=confidence),
            )

        if qty < 0 and last_price > channel_mid - fade_zone:
            return Signal(
                "reduce",
                "Short volatility breakout faded back toward the channel.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, components={"exit_reason": "short_volatility_breakout_faded"}, confidence=confidence),
            )

        if last_price > upper_breakout and qty <= 0:
            return Signal(
                "buy",
                f"{symbol} expanded above volatility channel.",
                timeframe,
                last_price * (1 - stop_loss_pct),
                last_price * (1 + take_profit_pct),
                risk_fraction,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk={**risk, "stop_loss": last_price * (1 - stop_loss_pct), "take_profit": last_price * (1 + take_profit_pct)}, confidence=confidence),
            )

        if last_price < lower_breakout and qty >= 0:
            return Signal(
                "sell",
                f"{symbol} expanded below volatility channel.",
                timeframe,
                last_price * (1 + stop_loss_pct),
                last_price * (1 - take_profit_pct),
                risk_fraction,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk={**risk, "stop_loss": last_price * (1 + stop_loss_pct), "take_profit": last_price * (1 - take_profit_pct)}, confidence=confidence),
            )

        return Signal(
            "hold",
            "No volatility expansion.",
            timeframe,
            None,
            None,
            0.0,
            self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, confidence=confidence, no_trade_reason="no_volatility_expansion"),
        )
