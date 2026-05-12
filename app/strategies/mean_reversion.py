"""Mean reversion strategy using SMA deviation with volatility-aware exits."""

from __future__ import annotations

from statistics import mean
from typing import Any

from .base import BaseStrategy, Signal


class MeanReversionStrategy(BaseStrategy):
    """Trades price deviations from SMA when no same-direction position exists."""

    name = "mean_reversion"
    description = "Mean reversion around SMA with volatility-aware risk controls."

    default_parameters = {
        "lookback": 20,
        "entry_threshold_pct": 0.012,
        "exit_threshold_ratio": 0.33,
        "risk_fraction": 0.08,
        "stop_loss_pct": 0.008,
        "take_profit_pct": 0.015,
        "leverage": 1.0,
    }

    @classmethod
    def parameter_grid(cls) -> dict[str, list[Any]]:
        return {
            "lookback": [10, 20, 30],
            "entry_threshold_pct": [0.006, 0.01, 0.014],
            "exit_threshold_ratio": [0.25, 0.33, 0.5],
            "risk_fraction": [0.04, 0.08, 0.1],
            "stop_loss_pct": [0.004, 0.008, 0.012],
            "take_profit_pct": [0.008, 0.015, 0.022],
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
            "entry_threshold_pct": self.parameters.get("entry_threshold_pct"),
            "exit_threshold_ratio": self.parameters.get("exit_threshold_ratio", 0.33),
        }
        try:
            closes = [float(row["close"]) for row in candles if row.get("close") is not None]
        except (AttributeError, TypeError, ValueError):
            return Signal(
                "hold",
                "Invalid mean reversion candle data.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="invalid_candle_data"),
            )

        if len(closes) < lookback:
            return Signal(
                "hold",
                "Waiting for mean reversion lookback window.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="insufficient_history"),
            )

        avg_price = mean(closes[-lookback:])
        last_price = closes[-1]

        if avg_price <= 0 or last_price <= 0:
            return Signal(
                "hold",
                "Invalid price data.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="invalid_price"),
            )

        deviation = (last_price - avg_price) / avg_price

        qty = float(position.get("quantity", 0.0))
        threshold = float(self.parameters["entry_threshold_pct"])
        exit_threshold = threshold * float(self.parameters.get("exit_threshold_ratio", 0.33))

        stop_loss_pct = float(self.parameters["stop_loss_pct"])
        take_profit_pct = float(self.parameters["take_profit_pct"])
        size = float(self.parameters["risk_fraction"])
        indicators = {
            "sma": avg_price,
            "deviation": deviation,
            "last_price": last_price,
        }
        risk = {
            "risk_fraction": size,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        }
        confidence = min(abs(deviation) / max(threshold, 1e-9), 1.0)

        # Exit when price has reverted close to the moving average.
        if qty != 0 and abs(deviation) <= exit_threshold:
            return Signal(
                "reduce",
                "Price has reverted near the moving average.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, components={"exit_reason": "mean_reversion_complete"}, confidence=confidence),
            )

        # Avoid adding to an existing long.
        if deviation <= -threshold and qty <= 0:
            stop_loss = last_price * (1 - stop_loss_pct)
            take_profit = min(avg_price, last_price * (1 + take_profit_pct))

            return Signal(
                "buy",
                f"Price is {abs(deviation) * 100:.2f}% below its moving average.",
                timeframe,
                stop_loss,
                take_profit,
                size,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk={**risk, "stop_loss": stop_loss, "take_profit": take_profit}, confidence=confidence),
            )

        # Avoid adding to an existing short.
        if deviation >= threshold and qty >= 0:
            stop_loss = last_price * (1 + stop_loss_pct)
            take_profit = max(avg_price, last_price * (1 - take_profit_pct))

            return Signal(
                "sell",
                f"Price is {deviation * 100:.2f}% above its moving average.",
                timeframe,
                stop_loss,
                take_profit,
                size,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk={**risk, "stop_loss": stop_loss, "take_profit": take_profit}, confidence=confidence),
            )

        return Signal(
            "hold",
            "No mean reversion edge detected.",
            timeframe,
            None,
            None,
            0.0,
            self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, confidence=confidence, no_trade_reason="no_mean_reversion_edge"),
        )
