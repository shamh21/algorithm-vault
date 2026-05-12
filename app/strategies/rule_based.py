"""Deterministic rule-based signal strategy."""

from __future__ import annotations

from typing import Any

from ..features.engine import FeatureEngine
from ..features.rules import SignalRule
from .base import BaseStrategy, Signal


class RuleBasedSignalStrategy(BaseStrategy):
    """Combines deterministic feature scores into bounded trade signals."""

    name = "rule_based_signal"
    description = "Deterministic EMA/RSI/ATR/volume rule strategy with Fibonacci risk context."
    default_parameters = {
        "ema_fast_period": 8,
        "ema_slow_period": 21,
        "rsi_period": 7,
        "rsi_oversold": 35.0,
        "rsi_overbought": 65.0,
        "exit_rsi": 50.0,
        "atr_period": 14,
        "atr_stop_multiplier": 1.5,
        "atr_take_multiplier": 2.25,
        "max_atr_pct": 0.08,
        "volume_lookback": 20,
        "volume_spike_multiplier": 1.5,
        "fibonacci_lookback": 50,
        "fibonacci_filter_weight": 0.05,
        "trend_weight": 0.55,
        "rsi_weight": 0.3,
        "volume_weight": 0.15,
        "external_weight": 0.0,
        "minimum_signal_score": 0.88,
        "risk_fraction": 0.05,
        "fallback_stop_loss_pct": 0.006,
        "fallback_take_profit_pct": 0.016,
        "leverage": 1.0,
    }

    def __init__(self, parameters: dict[str, Any] | None = None) -> None:
        super().__init__(parameters)
        self.feature_engine = FeatureEngine()
        self.rule = SignalRule(self.parameters)

    @classmethod
    def parameter_grid(cls) -> dict[str, list[Any]]:
        return {
            "ema_fast_period": [5, 8, 10],
            "ema_slow_period": [21, 34],
            "rsi_period": [7, 9],
            "rsi_oversold": [30.0, 35.0],
            "rsi_overbought": [65.0, 70.0],
            "atr_stop_multiplier": [1.2, 1.5, 2.0],
            "atr_take_multiplier": [1.8, 2.25, 3.0],
            "volume_spike_multiplier": [1.2, 1.5, 2.0],
            "minimum_signal_score": [0.8, 0.85, 1.0],
            "risk_fraction": [0.04, 0.06, 0.08],
            "external_weight": [0.0],
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
        if len(candles) < max(int(self.parameters["ema_slow_period"]), int(self.parameters["atr_period"])) + 2:
            return Signal(
                "hold",
                "Waiting for deterministic feature history.",
                timeframe,
                None,
                None,
                0.0,
                self.signal_metadata(symbol=symbol, timeframe=timeframe, candles=candles, no_trade_reason="insufficient_history"),
            )

        feature_config = self.feature_engine.config_from_parameters(self.parameters)
        snapshot = self.feature_engine.snapshot(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            config=feature_config,
        )
        features = snapshot.as_dict()

        # Dynamic risk adjustment based on volatility (ATR)
        atr_value = float(features.get("atr", 0.0))
        max_atr = float(self.parameters.get("max_atr_pct", 0.08))

        adjusted_risk_fraction = float(self.parameters["risk_fraction"])
        if atr_value > 0 and atr_value > max_atr:
            adjusted_risk_fraction *= 0.6
        elif atr_value > max_atr * 0.7:
            adjusted_risk_fraction *= 0.8

        decision = self.rule.evaluate(features, position)
        metadata = self._metadata(features, decision.as_dict())

        if decision.action == "reduce":
            metadata.update(
                self.signal_metadata(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=candles,
                    indicators=self._indicator_metadata(features),
                    thresholds=self._threshold_metadata(),
                    components={"rule_decision": decision.as_dict(), "exit_reason": "rule_reduce"},
                    confidence=decision.score,
                )
            )
            return Signal("reduce", decision.rationale, timeframe, None, None, 0.0, metadata)

        # Reject low-quality signals for profitability protection
        if decision.score < float(self.parameters["minimum_signal_score"]):
            metadata.update(
                self.signal_metadata(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=candles,
                    indicators=self._indicator_metadata(features),
                    thresholds=self._threshold_metadata(),
                    components={"rule_decision": decision.as_dict()},
                    confidence=decision.score,
                    no_trade_reason="low_confidence",
                )
            )
            return Signal("hold", f"Low confidence signal ({decision.score:.2f})", timeframe, None, None, 0.0, metadata)

        if decision.action not in {"buy", "sell"}:
            metadata.update(
                self.signal_metadata(
                    symbol=symbol,
                    timeframe=timeframe,
                    candles=candles,
                    indicators=self._indicator_metadata(features),
                    thresholds=self._threshold_metadata(),
                    components={"rule_decision": decision.as_dict()},
                    confidence=decision.score,
                    no_trade_reason="rule_hold",
                )
            )
            return Signal("hold", decision.rationale, timeframe, None, None, 0.0, metadata)

        last_price = float(candles[-1]["close"])
        stop_loss, take_profit, reward_risk = self._risk_levels(last_price, decision.action, features)
        metadata["risk_reward"] = reward_risk
        metadata["rule_based_signal"] = True
        metadata["fibonacci_alignment"] = self._fibonacci_alignment(stop_loss, take_profit, features)
        metadata.update(
            self.signal_metadata(
                symbol=symbol,
                timeframe=timeframe,
                candles=candles,
                indicators=self._indicator_metadata(features),
                thresholds=self._threshold_metadata(),
                risk={
                    "risk_fraction": adjusted_risk_fraction,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "reward_risk": reward_risk,
                },
                components={"rule_decision": decision.as_dict()},
                confidence=decision.score,
            )
        )
        if reward_risk < 1.2:
            metadata["no_trade_reason"] = "reward_risk_below_minimum"
            return Signal(
                "hold",
                f"Reward/risk {reward_risk:.2f} is below the rule threshold.",
                timeframe,
                None,
                None,
                0.0,
                metadata,
            )
        return Signal(
            decision.action,
            decision.rationale,
            timeframe,
            stop_loss,
            take_profit,
            adjusted_risk_fraction,
            metadata,
        )

    def _risk_levels(self, price: float, action: str, features: dict[str, Any]) -> tuple[float, float, float]:
        atr_value = float(features.get("atr", 0.0))
        stop_distance = atr_value * float(self.parameters["atr_stop_multiplier"])
        take_distance = atr_value * float(self.parameters["atr_take_multiplier"])
        if stop_distance <= 0:
            stop_distance = price * float(self.parameters["fallback_stop_loss_pct"])
        if take_distance <= 0:
            take_distance = price * float(self.parameters["fallback_take_profit_pct"])

        fibonacci = features.get("fibonacci_levels", {})
        retracements = list((fibonacci.get("retracements") or {}).values())
        extensions = list((fibonacci.get("extensions") or {}).values())
        if action == "buy":
            stop_loss = price - stop_distance
            take_profit = price + take_distance
            lower_fib = [float(value) for value in retracements if float(value) < price]
            upper_fib = [float(value) for value in extensions + retracements if float(value) > price]
            if lower_fib:
                stop_loss = min(stop_loss, max(lower_fib))
            if upper_fib:
                take_profit = max(take_profit, min(upper_fib))
        else:
            stop_loss = price + stop_distance
            take_profit = price - take_distance
            upper_fib = [float(value) for value in retracements if float(value) > price]
            lower_fib = [float(value) for value in extensions + retracements if float(value) < price]
            if upper_fib:
                stop_loss = max(stop_loss, min(upper_fib))
            if lower_fib:
                take_profit = min(take_profit, max(lower_fib))

        risk = abs(price - stop_loss)
        reward = abs(take_profit - price)
        reward_risk = reward / risk if risk > 0 else 0.0

        return round(stop_loss, 8), round(take_profit, 8), round(reward_risk, 8)

    @staticmethod
    def _metadata(features: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        return {
            "feature_snapshot": features,
            "rule_decision": decision,
            "fibonacci_levels": features.get("fibonacci_levels", {}),
            "external_scores": features.get("external_scores", {}),
            "pattern_prediction": features.get("pattern_prediction", {}),
            "risk_adjusted": True,
        }

    def _indicator_metadata(self, features: dict[str, Any]) -> dict[str, Any]:
        return {
            "ema_fast": features.get("ema_fast", 0.0),
            "ema_slow": features.get("ema_slow", 0.0),
            "ema_trend": features.get("ema_trend", 0.0),
            "trend_strength": features.get("trend_strength", 0.0),
            "rsi": features.get("rsi", 50.0),
            "atr": features.get("atr", 0.0),
            "atr_pct": features.get("atr_pct", 0.0),
            "volatility": features.get("volatility", 0.0),
            "volume_spike": features.get("volume_spike", {}),
        }

    def _threshold_metadata(self) -> dict[str, Any]:
        return {
            "minimum_signal_score": self.parameters.get("minimum_signal_score"),
            "max_atr_pct": self.parameters.get("max_atr_pct"),
            "rsi_oversold": self.parameters.get("rsi_oversold"),
            "rsi_overbought": self.parameters.get("rsi_overbought"),
            "atr_stop_multiplier": self.parameters.get("atr_stop_multiplier"),
            "atr_take_multiplier": self.parameters.get("atr_take_multiplier"),
        }

    @staticmethod
    def _fibonacci_alignment(stop_loss: float, take_profit: float, features: dict[str, Any]) -> dict[str, Any]:
        fibonacci = features.get("fibonacci_levels", {})
        values = list((fibonacci.get("retracements") or {}).values()) + list((fibonacci.get("extensions") or {}).values())
        if not values:
            return {"stop_aligned": False, "take_aligned": False}
        stop_aligned = any(abs(float(value) - stop_loss) / max(abs(stop_loss), 1e-9) <= 0.002 for value in values)
        take_aligned = any(abs(float(value) - take_profit) / max(abs(take_profit), 1e-9) <= 0.002 for value in values)
        return {"stop_aligned": stop_aligned, "take_aligned": take_aligned}
