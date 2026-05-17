"""Strategy plugin registry."""

from __future__ import annotations

from typing import Any

from .base import BaseStrategy
from .breakout import BreakoutStrategy
from .ema_crossover import EmaCrossoverStrategy
from .mean_reversion import MeanReversionStrategy
from .rsi_mean_reversion import RsiMeanReversionStrategy
from .rule_based import RuleBasedSignalStrategy
from .scalping import ScalpingStrategy
from .volatility_breakout import VolatilityBreakoutStrategy

_BUILT_IN_STRATEGIES: tuple[type[BaseStrategy], ...] = (
    EmaCrossoverStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    RsiMeanReversionStrategy,
    RuleBasedSignalStrategy,
    VolatilityBreakoutStrategy,
    ScalpingStrategy,
)


class StrategyRegistry:
    """Loads and instantiates built-in strategies by name."""

    def __init__(
        self,
        strategies: tuple[type[BaseStrategy], ...] = _BUILT_IN_STRATEGIES,
    ) -> None:
        self._strategies: dict[str, type[BaseStrategy]] = {}

        for strategy in strategies:
            self.register(strategy)

    def register(self, strategy: type[BaseStrategy]) -> None:
        """Register a strategy class."""

        name = getattr(strategy, "name", "")

        if not name:
            raise ValueError(f"Strategy {strategy.__name__} must define a non-empty name.")

        if name in self._strategies:
            raise ValueError(f"Duplicate strategy name registered: {name}")

        self._strategies[name] = strategy

    def names(self) -> list[str]:
        return sorted(self._strategies)

    def definitions(self) -> list[dict[str, Any]]:
        return [self.definition(name) for name in self.names()]

    def definition(self, name: str) -> dict[str, Any]:
        strategy = self._get_strategy(name)

        return {
            "name": name,
            "description": strategy.description,
            "parameters": strategy.parameter_schema(),
            "parameter_grid": strategy.parameter_grid(),
        }

    def parameter_grid(self, name: str) -> dict[str, list[Any]]:
        return self._get_strategy(name).parameter_grid()

    def build(self, name: str, parameters: dict[str, Any] | None = None) -> BaseStrategy:
        return self._get_strategy(name)(parameters)

    def _get_strategy(self, name: str) -> type[BaseStrategy]:
        strategy = self._strategies.get(name)

        if strategy is None:
            available = ", ".join(self.names())
            raise KeyError(f"Unknown strategy '{name}'. Available strategies: {available}")

        return strategy
