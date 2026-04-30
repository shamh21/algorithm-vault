"""Tight stop/take-profit scalping strategy."""

from __future__ import annotations

from statistics import mean
from typing import Any

from .base import BaseStrategy, Signal


class ScalpingStrategy(BaseStrategy):
    """Uses short momentum bursts with tight risk controls."""

    name = "scalping"
    description = "Short-horizon momentum scalping with trend and fade protection."

    default_parameters = {
        "momentum_lookback": 4,
        "minimum_move_pct": 0.0015,
        "risk_fraction": 0.03,
        "stop_loss_pct": 0.003,
        "take_profit_pct": 0.0045,
        "fade_exit_buffer_pct": 0.0005,
        "breakeven_trigger_pct": 0.0025,
        "trailing_stop_pct": 0.002,
        "fast_fade_exit_pct": 0.001,
        "leverage": 1.0,
    }

    @classmethod
    def parameter_grid(cls) -> dict[str, list[Any]]:
        return {
            "momentum_lookback": [3, 4, 6],
            "minimum_move_pct": [0.001, 0.0015, 0.0025],
            "risk_fraction": [0.02, 0.03, 0.05],
            "stop_loss_pct": [0.002, 0.003, 0.005],
            "take_profit_pct": [0.003, 0.0045, 0.007],
            "fade_exit_buffer_pct": [0.0, 0.0005, 0.001],
            "breakeven_trigger_pct": [0.0015, 0.0025, 0.004],
            "trailing_stop_pct": [0.0015, 0.002, 0.003],
            "fast_fade_exit_pct": [0.0005, 0.001, 0.0015],
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

        lookback = max(2, int(self.parameters["momentum_lookback"]))
        thresholds = {
            "momentum_lookback": lookback,
            "minimum_move_pct": self.parameters.get("minimum_move_pct"),
            "fade_exit_buffer_pct": self.parameters.get("fade_exit_buffer_pct", 0.0),
            "breakeven_trigger_pct": self.parameters.get("breakeven_trigger_pct", 0.0025),
            "trailing_stop_pct": self.parameters.get("trailing_stop_pct", 0.002),
            "fast_fade_exit_pct": self.parameters.get("fast_fade_exit_pct", 0.001),
        }

        if len(candles) < lookback + 2:
            return Signal(
                "hold",
                "Waiting for scalping momentum history.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="insufficient_history"),
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

        if len(closes) < lookback + 2:
            return Signal(
                "hold",
                "Waiting for valid scalping momentum history.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="insufficient_valid_history"),
            )

        last_price = closes[-1]
        previous_price = closes[-lookback - 1]

        if last_price <= 0 or previous_price <= 0:
            return Signal(
                "hold",
                "Invalid price data.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, thresholds=thresholds, no_trade_reason="invalid_price"),
            )

        move = (last_price - previous_price) / previous_price
        local_average = mean(closes[-lookback:])

        qty = float(position.get("quantity", 0.0))
        minimum_move = float(self.parameters["minimum_move_pct"])
        stop_loss_pct = float(self.parameters["stop_loss_pct"])
        take_profit_pct = float(self.parameters["take_profit_pct"])
        risk_fraction = float(self.parameters["risk_fraction"])
        fade_exit_buffer_pct = float(self.parameters.get("fade_exit_buffer_pct", 0.0))
        breakeven_trigger_pct = float(self.parameters.get("breakeven_trigger_pct", 0.0025))
        trailing_stop_pct = float(self.parameters.get("trailing_stop_pct", 0.002))
        fast_fade_exit_pct = float(self.parameters.get("fast_fade_exit_pct", 0.001))
        entry_price = float(position.get("entry_price", 0.0) or 0.0)
        indicators = {
            "move": move,
            "local_average": local_average,
            "last_price": last_price,
            "previous_price": previous_price,
        }
        risk = {
            "risk_fraction": risk_fraction,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        }
        confidence = min(abs(move) / max(minimum_move, 1e-9), 1.0)

        if qty > 0 and entry_price > 0:
            favorable_move = (last_price - entry_price) / entry_price
            if favorable_move >= breakeven_trigger_pct and last_price <= entry_price * (1 + 0.0001):
                return Signal(
                    "reduce",
                    "Long scalp moved back to break-even after favorable movement.",
                    timeframe,
                    None,
                    None,
                    0.0,
                    self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, components={"exit_reason": "long_breakeven_reversal"}, confidence=confidence),
                )
            if favorable_move >= trailing_stop_pct and move <= -fast_fade_exit_pct:
                return Signal(
                    "reduce",
                    "Long scalp momentum faded after a favorable move.",
                    timeframe,
                    None,
                    None,
                    0.0,
                    self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, components={"exit_reason": "long_momentum_fade"}, confidence=confidence),
                )

        if qty < 0 and entry_price > 0:
            favorable_move = (entry_price - last_price) / entry_price
            if favorable_move >= breakeven_trigger_pct and last_price >= entry_price * (1 - 0.0001):
                return Signal(
                    "reduce",
                    "Short scalp moved back to break-even after favorable movement.",
                    timeframe,
                    None,
                    None,
                    0.0,
                    self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, components={"exit_reason": "short_breakeven_reversal"}, confidence=confidence),
                )
            if favorable_move >= trailing_stop_pct and move >= fast_fade_exit_pct:
                return Signal(
                    "reduce",
                    "Short scalp momentum faded after a favorable move.",
                    timeframe,
                    None,
                    None,
                    0.0,
                    self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, components={"exit_reason": "short_momentum_fade"}, confidence=confidence),
                )

        # Exit stale scalps when momentum fades back through the local average.
        if qty > 0 and last_price < local_average * (1 - fade_exit_buffer_pct):
            return Signal(
                "reduce",
                "Long scalp momentum faded below the local average.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, components={"exit_reason": "long_local_average_fade"}, confidence=confidence),
            )

        if qty < 0 and last_price > local_average * (1 + fade_exit_buffer_pct):
            return Signal(
                "reduce",
                "Short scalp momentum faded above the local average.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, components={"exit_reason": "short_local_average_fade"}, confidence=confidence),
            )

        # Avoid repeatedly adding to the same direction.
        if move >= minimum_move and qty <= 0:
            return Signal(
                "buy",
                f"{symbol} short-term momentum rose {move * 100:.2f}%.",
                timeframe,
                last_price * (1 - stop_loss_pct),
                last_price * (1 + take_profit_pct),
                risk_fraction,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk={**risk, "stop_loss": last_price * (1 - stop_loss_pct), "take_profit": last_price * (1 + take_profit_pct)}, confidence=confidence),
            )

        if move <= -minimum_move and qty >= 0:
            return Signal(
                "sell",
                f"{symbol} short-term momentum fell {abs(move) * 100:.2f}%.",
                timeframe,
                last_price * (1 + stop_loss_pct),
                last_price * (1 - take_profit_pct),
                risk_fraction,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk={**risk, "stop_loss": last_price * (1 + stop_loss_pct), "take_profit": last_price * (1 - take_profit_pct)}, confidence=confidence),
            )

        return Signal(
            "hold",
            "No scalp signal.",
            timeframe,
            None,
            None,
            0.0,
            self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, indicators=indicators, thresholds=thresholds, risk=risk, confidence=confidence, no_trade_reason="no_scalp_signal"),
        )
