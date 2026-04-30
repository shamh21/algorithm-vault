"""Walk‑forward strategy optimisation and ranking.

This module provides classes to perform rolling walk‑forward optimisations
across multiple symbols, timeframes and strategy parameter sets.  It
evaluates each candidate over a series of train/test windows, aggregates
their performance using recency weighting, and persists both the raw
backtest results and aggregated rankings to the database.  At the end of
an optimisation run, the top performing strategies can optionally be
validated for deployment.

The refactored implementation introduces clearer abstractions and type
annotations, extracts shared logic into dedicated helper methods, and
supports explicit enumerations for the optimisation profile and its
associated window lengths.  Extensive docstrings make the behaviour
explicit and improve maintainability.  Error handling is also improved
around database operations to ensure runs are correctly marked as failed
if an exception occurs during evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from itertools import product
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..extensions import db
from ..ml.offline_ranker import OfflineRanker
from ..ml.online_ranker import OnlineRanker, extract_features, horizon_from_duration, outcome_from_result
from ..models import OptimizerRun, Setting, StrategyRanking, StrategyValidation
from ..services.ensemble_allocator import EnhancedEnsembleAllocator
from ..services.market_data import MarketDataService
from ..services.net_roi import net_roi_diagnostics, net_roi_v2_diagnostics
from ..strategies.registry import StrategyRegistry
from .engine import BacktestConfig, BacktestEngine


class Profile(Enum):
    """Enumeration of optimisation profiles.

    Each profile defines the length of the training window, testing window
    and the step size in days for the rolling walk‑forward process.  These
    values roughly correspond to short, medium and long term horizons.
    """

    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    AGGRESSIVE_1H = "aggressive_1h"
    AGGRESSIVE_RISK_ADJUSTED = "aggressive_risk_adjusted"
    EXTREME_ROI_EXPERIMENTAL = "extreme_roi_experimental"
    DYNAMIC_INTRADAY = "dynamic_intraday"

    @property
    def windows(self) -> Tuple[int, int, int]:
        """Return the (training_days, testing_days, step_days) tuple for the profile."""
        return {
            Profile.SHORT_TERM: (14, 2, 1),
            Profile.MEDIUM_TERM: (45, 5, 2),
            Profile.LONG_TERM: (120, 14, 7),
            Profile.AGGRESSIVE_1H: (3, 1, 1),
            Profile.AGGRESSIVE_RISK_ADJUSTED: (14, 2, 1),
            Profile.EXTREME_ROI_EXPERIMENTAL: (3, 1, 1),
            Profile.DYNAMIC_INTRADAY: (3, 1, 1),
        }[self]

    @classmethod
    def from_str(cls, value: str) -> "Profile":
        """Parse a profile name into a ``Profile`` enumeration member.

        Falls back to ``SHORT_TERM`` for unknown values.
        """
        try:
            return cls(value)
        except Exception:
            return cls.SHORT_TERM


@dataclass(slots=True)
class OptimizerConfig:
    """Configuration for a rolling walk‑forward optimisation run.

    Attributes:
        symbols: List of trading symbols to evaluate.
        timeframes: List of candle timeframes.
        strategy_names: Names of strategies to include; if empty then all
            registered strategies are used.
        profile: Optimisation profile controlling training and testing
            window lengths and the rolling step size.
        mode: Market data mode (e.g. ``"testnet"`` or ``"live"``).
        initial_balance: Starting capital for each backtest.
        fee_bps: Trading fee expressed in basis points.
        slippage_bps: Slippage expressed in basis points.
        training_window_days: Override for training window length; used when
            constructing the config via the registry, otherwise derived from
            the profile.
        testing_window_days: Override for testing window length; used when
            constructing the config via the registry, otherwise derived from
            the profile.
        step_days: Override for the rolling step size; used when constructing
            the config via the registry, otherwise derived from the profile.
        use_full_history: Whether to fetch the entire candle history instead
            of just enough to cover the rolling windows.  This can yield
            smoother results but may reduce responsiveness to recent market
            conditions.
        recency_weighting_enabled: Whether to weight window results by
            recency when aggregating.  More recent windows receive higher
            weights according to the decay factor.
        decay_factor: Geometric decay factor applied for recency weighting.
        min_trade_count: Minimum number of trades required for a candidate to
            be considered; below this threshold the candidate is rejected.
        max_drawdown_pct: Maximum tolerable drawdown across windows; more
            negative drawdowns cause rejection.
        max_parameter_sets: Maximum number of parameter sets sampled from
            the strategy’s parameter grid.
        auto_deploy_top_n: Number of top ranked strategies to automatically
            validate for deployment after optimisation.
    """

    symbols: List[str]
    timeframes: List[str]
    strategy_names: List[str] = field(default_factory=list)
    profile: str = "short_term"
    mode: str = "testnet"
    initial_balance: float = 1_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 8.0
    training_window_days: int = 14
    testing_window_days: int = 2
    step_days: int = 1
    training_window_hours: int | None = None
    testing_window_hours: int | None = None
    step_hours: int | None = None
    use_full_history: bool = False
    recency_weighting_enabled: bool = True
    decay_factor: float = 0.9
    min_trade_count: int = 5
    max_drawdown_pct: float = 0.25
    max_parameter_sets: int = 8
    auto_deploy_top_n: int = 1
    allocation_amount_usd: float = 0.0
    lock_duration_hours: int = 0
    universe_mode: str = "configured"
    max_parallel_legs: int = 1
    allow_leverage_experiment: bool = False
    min_edge_bps: float = 0.0
    fib_confluence_threshold: float = 0.0
    require_shadow_validation: bool = True
    enhanced_ensemble_enabled: bool = False
    ensemble_max_legs: int = 5
    ensemble_min_sharpe: float = 0.5
    ensemble_learning_decay: float = 0.8
    experimental_duration_ensemble_enabled: bool = False
    experimental_live_eligible: bool = False
    ensemble_primary_metric: str = "net_return_after_costs"
    duration_live_cap_usdc_by_duration: dict[str, float] = field(default_factory=dict)
    duration_live_cap_pct_by_duration: dict[str, float] = field(default_factory=dict)
    max_return_optimizer_enabled: bool = False
    max_return_live_eligible: bool = False
    max_return_min_net_return: float = 0.0
    max_return_max_drawdown_pct: float = 0.0
    market_structure_features_enabled: bool = False
    market_structure_provider: str = "existing"
    pair_screening_enabled: bool = False
    pair_trading_enabled: bool = False
    pair_live_eligible: bool = False
    pair_min_correlation: float = 0.75
    pair_max_spread_zscore: float = 2.5
    dynamic_intraday_live_eligible: bool = False
    high_upside_profile: bool = False


AGGRESSIVE_1H_WARNING = (
    "Aggressive 1H mode is experimental and can lose capital quickly. "
    "Past backtests do not guarantee future returns."
)
EXTREME_ROI_WARNING = (
    "Extreme ROI Experimental is historical simulation research only. It seeks unusually high upside, "
    "but losses and rapid drawdowns remain possible."
)
DYNAMIC_INTRADAY_WARNING = (
    "Dynamic Intraday mode is production-oriented candidate discovery; "
    "live use requires explicit eligibility, validation, and all risk gates."
)


class BacktestRunner:
    """Thin batch runner for single‑strategy backtests.

    This class exists primarily to decouple the optimisation logic from the
    underlying backtesting engine.  It accepts a pre‑configured
    ``BacktestEngine`` and exposes a simple ``run`` method that forwards
    arguments to the engine.  In testing contexts this can be mocked or
    substituted to provide synthetic results.
    """

    def __init__(self, engine: BacktestEngine) -> None:
        self.engine = engine

    def run(self, config: BacktestConfig, candles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run a backtest using the provided engine and return the result."""
        return self.engine.run(config, candles=candles)


class StrategyOptimizer:
    """Ranks strategies using rolling out‑of‑sample short‑term performance.

    The optimiser iterates over the Cartesian product of symbols, timeframes
    and strategy parameter sets.  For each candidate it builds training and
    testing windows, executes backtests using the supplied runner, aggregates
    the window performance using recency weighting, and produces a ranking
    score based on a mixture of return, drawdown, consistency and
    risk‑adjusted metrics.  Results and warnings are persisted to the
    database via the provided SQLAlchemy session.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        registry: StrategyRegistry,
        market_data: MarketDataService,
        backtest_engine: BacktestEngine,
        universe_service: Any | None = None,
        online_ranker: OnlineRanker | None = None,
        market_structure: Any | None = None,
        pair_screening: Any | None = None,
        offline_ranker: OfflineRanker | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.market_data = market_data
        self.universe_service = universe_service
        self.online_ranker = online_ranker or OnlineRanker(config)
        self.market_structure = market_structure
        self.pair_screening = pair_screening
        self.offline_ranker = offline_ranker or OfflineRanker(config)
        self.runner = BacktestRunner(backtest_engine)

    # ------------------------------------------------------------------
    # Configuration helpers
    #
    def default_config(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        strategy_names: Optional[List[str]] = None,
        profile: str = "short_term",
        allocation_amount_usd: float = 0.0,
        lock_duration_hours: int = 0,
        universe_mode: str = "configured",
        max_parallel_legs: int = 1,
        allow_leverage_experiment: bool = False,
    ) -> OptimizerConfig:
        """Produce a default ``OptimizerConfig`` based on stored settings.

        This helper uses the registry and internal configuration to supply
        sensible defaults for a new optimisation run.  If a profile name
        does not map to a known profile enumeration, ``SHORT_TERM`` is used.

        Args:
            symbols: Optional list of symbols to override the default.
            timeframes: Optional list of timeframes to override the default.
            strategy_names: Optional list of strategy names to override the
                registry default of all registered strategies.
            profile: Name of the optimisation profile.

        Returns:
            A fully populated ``OptimizerConfig`` instance.
        """
        profile_enum = Profile.from_str(profile)
        train_days, test_days, step_days = profile_enum.windows
        if profile_enum in {Profile.AGGRESSIVE_1H, Profile.EXTREME_ROI_EXPERIMENTAL}:
            train_hours, test_hours, step_hours = self._horizon_hours(lock_duration_hours)
            if profile_enum == Profile.EXTREME_ROI_EXPERIMENTAL:
                timeframes = timeframes or list(self.config.get("EXTREME_ROI_TIMEFRAMES", ["1m", "5m"]))
                aggressive_names = [
                    "scalping",
                    "volatility_breakout",
                    "rule_based_signal",
                    "breakout",
                    "ema_crossover",
                    "rsi_mean_reversion",
                ]
                min_trades = int(self.config.get("EXTREME_ROI_MIN_TRADES", 10))
                max_drawdown = float(self.config.get("EXTREME_ROI_MAX_DRAWDOWN_PCT", 0.45))
                max_parameter_sets = int(self.config.get("EXTREME_ROI_MAX_PARAMETER_SETS", 40))
                leverage_experiment = True
            else:
                timeframes = timeframes or list(self.config.get("AGGRESSIVE_1H_TIMEFRAMES", ["1m", "5m"]))
                aggressive_names = ["scalping", "rsi_mean_reversion", "volatility_breakout", "ema_crossover", "breakout"]
                min_trades = int(self.config.get("AGGRESSIVE_1H_MIN_TRADES", 8))
                max_drawdown = float(self.config.get("AGGRESSIVE_1H_MAX_DRAWDOWN_PCT", 0.35))
                max_parameter_sets = int(self.config.get("AGGRESSIVE_1H_MAX_PARAMETER_SETS", 25))
                leverage_experiment = bool(allow_leverage_experiment)
            default_aggressive_names = [name for name in aggressive_names if name in self.registry.names()]
            return OptimizerConfig(
                symbols=symbols or list(self.config.get("ALLOWED_SYMBOLS", ["BTC"])),
                timeframes=timeframes,
                strategy_names=strategy_names or default_aggressive_names or self.registry.names(),
                profile=profile_enum.value,
                mode="testnet",
                initial_balance=float(self.config.get("DEFAULT_PAPER_BALANCE", 1_000.0)),
                fee_bps=float(self.config.get("FEE_BPS", 5.0)),
                slippage_bps=float(self.config.get("SIM_SLIPPAGE_BPS", 8.0)),
                training_window_days=train_days,
                testing_window_days=test_days,
                step_days=step_days,
                training_window_hours=train_hours,
                testing_window_hours=test_hours,
                step_hours=step_hours,
                use_full_history=bool(self.config.get("OPTIMIZER_USE_FULL_HISTORY", False)),
                recency_weighting_enabled=bool(self.config.get("OPTIMIZER_RECENCY_WEIGHTING_ENABLED", True)),
                decay_factor=float(self.config.get("OPTIMIZER_DECAY_FACTOR", 0.9)),
                min_trade_count=min_trades,
                max_drawdown_pct=max_drawdown,
                max_parameter_sets=max_parameter_sets,
                allocation_amount_usd=float(allocation_amount_usd or 0.0),
                lock_duration_hours=int(lock_duration_hours or 0),
                universe_mode=universe_mode,
                max_parallel_legs=max(1, int(max_parallel_legs or 1)),
                allow_leverage_experiment=leverage_experiment,
                min_edge_bps=float(
                    self.config.get(
                        "EXTREME_ROI_MIN_EDGE_BPS" if profile_enum == Profile.EXTREME_ROI_EXPERIMENTAL else "ENSEMBLE_1H_MIN_EDGE_BPS",
                        8.0 if profile_enum == Profile.EXTREME_ROI_EXPERIMENTAL else 4.0,
                    )
                ),
                fib_confluence_threshold=float(self.config.get("ENSEMBLE_1H_MIN_CONFLUENCE", 0.0)),
                require_shadow_validation=bool(self.config.get("ENSEMBLE_1H_REQUIRE_SHADOW_VALIDATION", True)),
                enhanced_ensemble_enabled=bool(self.config.get("ENSEMBLE_ENHANCED_ENABLED", False)),
                ensemble_max_legs=max(2, min(int(self.config.get("ENSEMBLE_MAX_LEGS", 5)), 5)),
                ensemble_min_sharpe=float(self.config.get("ENSEMBLE_MIN_SHARPE", 0.5)),
                ensemble_learning_decay=float(self.config.get("ENSEMBLE_LEARNING_DECAY", 0.8)),
                experimental_duration_ensemble_enabled=bool(self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_ENABLED", False)),
                experimental_live_eligible=bool(self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_LIVE_ELIGIBLE", False)),
                ensemble_primary_metric=str(self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_PRIMARY_METRIC", "net_return_after_costs")),
                duration_live_cap_usdc_by_duration=dict(self.config.get("EXPERIMENTAL_LIVE_CAP_USDC_BY_DURATION", {}) or {}),
                duration_live_cap_pct_by_duration=dict(self.config.get("EXPERIMENTAL_LIVE_CAP_PCT_BY_DURATION", {}) or {}),
                **self._max_return_config_kwargs(),
            )
        if profile_enum == Profile.DYNAMIC_INTRADAY:
            train_hours, test_hours, step_hours = self._horizon_hours(lock_duration_hours or 1)
            preferred_names = [
                "scalping",
                "volatility_breakout",
                "rsi_mean_reversion",
                "ema_crossover",
                "breakout",
                "rule_based_signal",
            ]
            default_names = [name for name in preferred_names if name in self.registry.names()]
            return OptimizerConfig(
                symbols=symbols or list(self.config.get("ALLOWED_SYMBOLS", ["BTC"])),
                timeframes=timeframes or list(self.config.get("DYNAMIC_INTRADAY_TIMEFRAMES", ["1m", "5m", "15m"])),
                strategy_names=strategy_names or default_names or self.registry.names(),
                profile=profile_enum.value,
                mode="testnet",
                initial_balance=float(self.config.get("DEFAULT_PAPER_BALANCE", 1_000.0)),
                fee_bps=float(self.config.get("FEE_BPS", 5.0)),
                slippage_bps=float(self.config.get("SIM_SLIPPAGE_BPS", 8.0)),
                training_window_days=train_days,
                testing_window_days=test_days,
                step_days=step_days,
                training_window_hours=train_hours,
                testing_window_hours=test_hours,
                step_hours=step_hours,
                use_full_history=bool(self.config.get("OPTIMIZER_USE_FULL_HISTORY", False)),
                recency_weighting_enabled=bool(self.config.get("OPTIMIZER_RECENCY_WEIGHTING_ENABLED", True)),
                decay_factor=float(self.config.get("OPTIMIZER_DECAY_FACTOR", 0.9)),
                min_trade_count=int(self.config.get("DYNAMIC_INTRADAY_MIN_TRADES", 8)),
                max_drawdown_pct=float(self.config.get("DYNAMIC_INTRADAY_MAX_DRAWDOWN_PCT", 0.25)),
                max_parameter_sets=int(self.config.get("DYNAMIC_INTRADAY_MAX_PARAMETER_SETS", 30)),
                allocation_amount_usd=float(allocation_amount_usd or 0.0),
                lock_duration_hours=max(1, int(lock_duration_hours or 1)),
                universe_mode=universe_mode if universe_mode != "configured" else "dynamic_liquid",
                max_parallel_legs=max(1, int(max_parallel_legs or 1)),
                allow_leverage_experiment=bool(allow_leverage_experiment),
                min_edge_bps=float(self.config.get("DYNAMIC_INTRADAY_MIN_EDGE_BPS", 6.0)),
                fib_confluence_threshold=float(self.config.get("ENSEMBLE_1H_MIN_CONFLUENCE", 0.0)),
                require_shadow_validation=bool(self.config.get("DYNAMIC_INTRADAY_REQUIRE_SHADOW_VALIDATION", True)),
                enhanced_ensemble_enabled=bool(self.config.get("ENSEMBLE_ENHANCED_ENABLED", False)),
                ensemble_max_legs=max(2, min(int(self.config.get("ENSEMBLE_MAX_LEGS", 5)), 5)),
                ensemble_min_sharpe=float(self.config.get("ENSEMBLE_MIN_SHARPE", 0.5)),
                ensemble_learning_decay=float(self.config.get("ENSEMBLE_LEARNING_DECAY", 0.8)),
                experimental_duration_ensemble_enabled=bool(self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_ENABLED", False)),
                experimental_live_eligible=bool(self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_LIVE_ELIGIBLE", False)),
                ensemble_primary_metric=str(self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_PRIMARY_METRIC", "net_return_after_costs")),
                duration_live_cap_usdc_by_duration=dict(self.config.get("EXPERIMENTAL_LIVE_CAP_USDC_BY_DURATION", {}) or {}),
                duration_live_cap_pct_by_duration=dict(self.config.get("EXPERIMENTAL_LIVE_CAP_PCT_BY_DURATION", {}) or {}),
                market_structure_features_enabled=True,
                market_structure_provider=str(self.config.get("MARKET_STRUCTURE_PROVIDER", "existing") or "existing"),
                dynamic_intraday_live_eligible=bool(self.config.get("DYNAMIC_INTRADAY_LIVE_ELIGIBLE", False)),
                **{
                    key: value
                    for key, value in self._max_return_config_kwargs().items()
                    if key not in {"market_structure_features_enabled", "market_structure_provider"}
                },
            )
        if profile_enum == Profile.AGGRESSIVE_RISK_ADJUSTED:
            train_hours, test_hours, step_hours = self._horizon_hours(lock_duration_hours or 24)
            duration = max(1, int(lock_duration_hours or 24))
            if duration <= 1:
                default_timeframes = ["1m", "5m"]
                preferred_names = ["scalping", "rsi_mean_reversion", "volatility_breakout", "ema_crossover", "breakout"]
            elif duration <= 24:
                default_timeframes = ["5m", "15m", "1h"]
                preferred_names = [
                    "ema_crossover",
                    "breakout",
                    "mean_reversion",
                    "rsi_mean_reversion",
                    "rule_based_signal",
                    "volatility_breakout",
                ]
            elif duration <= 48:
                default_timeframes = ["15m", "1h"]
                preferred_names = [
                    "ema_crossover",
                    "breakout",
                    "rule_based_signal",
                    "volatility_breakout",
                    "mean_reversion",
                ]
            else:
                default_timeframes = ["1h", "15m"]
                preferred_names = ["volatility_breakout", "ema_crossover", "breakout", "rule_based_signal"]
            default_names = [name for name in preferred_names if name in self.registry.names()]
            return OptimizerConfig(
                symbols=symbols or list(self.config.get("ALLOWED_SYMBOLS", ["BTC"])),
                timeframes=timeframes or default_timeframes,
                strategy_names=strategy_names or default_names or self.registry.names(),
                profile=profile_enum.value,
                mode="testnet",
                initial_balance=float(self.config.get("DEFAULT_PAPER_BALANCE", 1_000.0)),
                fee_bps=float(self.config.get("FEE_BPS", 5.0)),
                slippage_bps=float(self.config.get("SIM_SLIPPAGE_BPS", 8.0)),
                training_window_days=train_days,
                testing_window_days=test_days,
                step_days=step_days,
                training_window_hours=train_hours,
                testing_window_hours=test_hours,
                step_hours=step_hours,
                use_full_history=bool(self.config.get("OPTIMIZER_USE_FULL_HISTORY", False)),
                recency_weighting_enabled=bool(self.config.get("OPTIMIZER_RECENCY_WEIGHTING_ENABLED", True)),
                decay_factor=float(self.config.get("OPTIMIZER_DECAY_FACTOR", 0.9)),
                min_trade_count=int(self.config.get("AGGRESSIVE_RISK_ADJUSTED_MIN_TRADES", 6)),
                max_drawdown_pct=float(self.config.get("AGGRESSIVE_RISK_ADJUSTED_MAX_DRAWDOWN_PCT", 0.25)),
                max_parameter_sets=int(self.config.get("AGGRESSIVE_RISK_ADJUSTED_MAX_PARAMETER_SETS", 30)),
                allocation_amount_usd=float(allocation_amount_usd or 0.0),
                lock_duration_hours=duration,
                universe_mode=universe_mode,
                max_parallel_legs=max(1, int(max_parallel_legs or 1)),
                allow_leverage_experiment=bool(allow_leverage_experiment),
                min_edge_bps=float(self.config.get("ENSEMBLE_1H_MIN_EDGE_BPS", self.config.get("AGGRESSIVE_MIN_EDGE_BPS", 4.0))) if duration <= 1 else float(self.config.get("AGGRESSIVE_MIN_EDGE_BPS", 4.0)),
                fib_confluence_threshold=float(self.config.get("ENSEMBLE_1H_MIN_CONFLUENCE", 0.0)) if duration <= 1 else 0.0,
                require_shadow_validation=bool(self.config.get("ENSEMBLE_1H_REQUIRE_SHADOW_VALIDATION", True)) if duration <= 1 else False,
                enhanced_ensemble_enabled=bool(self.config.get("ENSEMBLE_ENHANCED_ENABLED", False)) if duration <= 1 else False,
                ensemble_max_legs=max(2, min(int(self.config.get("ENSEMBLE_MAX_LEGS", 5)), 5)),
                ensemble_min_sharpe=float(self.config.get("ENSEMBLE_MIN_SHARPE", 0.5)),
                ensemble_learning_decay=float(self.config.get("ENSEMBLE_LEARNING_DECAY", 0.8)),
                experimental_duration_ensemble_enabled=bool(self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_ENABLED", False)),
                experimental_live_eligible=bool(self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_LIVE_ELIGIBLE", False)),
                ensemble_primary_metric=str(self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_PRIMARY_METRIC", "net_return_after_costs")),
                duration_live_cap_usdc_by_duration=dict(self.config.get("EXPERIMENTAL_LIVE_CAP_USDC_BY_DURATION", {}) or {}),
                duration_live_cap_pct_by_duration=dict(self.config.get("EXPERIMENTAL_LIVE_CAP_PCT_BY_DURATION", {}) or {}),
                **self._max_return_config_kwargs(),
            )
        return OptimizerConfig(
            symbols=symbols or list(self.config.get("ALLOWED_SYMBOLS", ["BTC"])),
            timeframes=timeframes or [self.config.get("DEFAULT_TIMEFRAME", "15m")],
            strategy_names=strategy_names or self.registry.names(),
            profile=profile_enum.value,
            mode="testnet",
            initial_balance=float(self.config.get("DEFAULT_PAPER_BALANCE", 1_000.0)),
            fee_bps=float(self.config.get("FEE_BPS", 5.0)),
            slippage_bps=float(self.config.get("SIM_SLIPPAGE_BPS", 8.0)),
            training_window_days=int(self.config.get("OPTIMIZER_TRAINING_WINDOW_DAYS", train_days)),
            testing_window_days=int(self.config.get("OPTIMIZER_TESTING_WINDOW_DAYS", test_days)),
            step_days=int(self.config.get("OPTIMIZER_STEP_DAYS", step_days)),
            use_full_history=bool(self.config.get("OPTIMIZER_USE_FULL_HISTORY", False)),
            recency_weighting_enabled=bool(self.config.get("OPTIMIZER_RECENCY_WEIGHTING_ENABLED", True)),
            decay_factor=float(self.config.get("OPTIMIZER_DECAY_FACTOR", 0.9)),
            min_trade_count=5,
            max_drawdown_pct=float(self.config.get("MAX_BACKTEST_DRAWDOWN_PCT", 0.25)),
            allocation_amount_usd=float(allocation_amount_usd or 0.0),
            lock_duration_hours=int(lock_duration_hours or 0),
            universe_mode=universe_mode,
            max_parallel_legs=max(1, int(max_parallel_legs or 1)),
            allow_leverage_experiment=bool(allow_leverage_experiment),
            min_edge_bps=float(self.config.get("AGGRESSIVE_MIN_EDGE_BPS", 4.0)),
            fib_confluence_threshold=0.0,
            require_shadow_validation=False,
            enhanced_ensemble_enabled=False,
            ensemble_max_legs=max(2, min(int(self.config.get("ENSEMBLE_MAX_LEGS", 5)), 5)),
            ensemble_min_sharpe=float(self.config.get("ENSEMBLE_MIN_SHARPE", 0.5)),
            ensemble_learning_decay=float(self.config.get("ENSEMBLE_LEARNING_DECAY", 0.8)),
            experimental_duration_ensemble_enabled=bool(self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_ENABLED", False)),
            experimental_live_eligible=bool(self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_LIVE_ELIGIBLE", False)),
            ensemble_primary_metric=str(self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_PRIMARY_METRIC", "net_return_after_costs")),
            duration_live_cap_usdc_by_duration=dict(self.config.get("EXPERIMENTAL_LIVE_CAP_USDC_BY_DURATION", {}) or {}),
            duration_live_cap_pct_by_duration=dict(self.config.get("EXPERIMENTAL_LIVE_CAP_PCT_BY_DURATION", {}) or {}),
            **self._max_return_config_kwargs(),
        )

    def _max_return_config_kwargs(self) -> dict[str, Any]:
        return {
            "max_return_optimizer_enabled": bool(self.config.get("MAX_RETURN_OPTIMIZER_ENABLED", False)),
            "max_return_live_eligible": bool(self.config.get("MAX_RETURN_LIVE_ELIGIBLE", False)),
            "max_return_min_net_return": float(self.config.get("MAX_RETURN_MIN_NET_RETURN", 0.0) or 0.0),
            "max_return_max_drawdown_pct": float(self.config.get("MAX_RETURN_MAX_DRAWDOWN_PCT", 0.0) or 0.0),
            "market_structure_features_enabled": bool(self.config.get("MARKET_STRUCTURE_FEATURES_ENABLED", False)),
            "market_structure_provider": str(self.config.get("MARKET_STRUCTURE_PROVIDER", "existing") or "existing"),
            "pair_screening_enabled": bool(self.config.get("PAIR_SCREENING_ENABLED", False)),
            "pair_trading_enabled": bool(self.config.get("PAIR_TRADING_ENABLED", False)),
            "pair_live_eligible": bool(self.config.get("PAIR_LIVE_ELIGIBLE", False)),
            "pair_min_correlation": float(self.config.get("PAIR_MIN_CORRELATION", 0.75) or 0.75),
            "pair_max_spread_zscore": float(self.config.get("PAIR_MAX_SPREAD_ZSCORE", 2.5) or 2.5),
        }

    # ------------------------------------------------------------------
    # Entry point
    #
    def run(self, optimizer_config: OptimizerConfig) -> Dict[str, Any]:
        """Execute an optimisation run and return the summary result.

        A new ``OptimizerRun`` record is created at the start with status
        ``running``.  The optimiser iterates through all combinations of
        symbols, timeframes, strategies and parameter sets, evaluating each
        candidate over all rolling windows.  After evaluation the run
        record is marked as ``completed`` or ``failed`` depending on
        whether an exception occurred.  Rankings and optional backtest
        validations are persisted to the database within the run.

        Args:
            optimizer_config: Configuration controlling the optimisation run.

        Returns:
            A dictionary summarising the optimisation result including the
            number of ranked candidates, accepted candidates, the top
            performers and any warnings.
        """
        # Create a run record and persist initial state.
        run = OptimizerRun(profile=optimizer_config.profile, status="running")
        run.symbols = optimizer_config.symbols
        run.timeframes = optimizer_config.timeframes
        run.config_payload = self._config_payload(optimizer_config)
        warnings: List[str] = []
        if optimizer_config.use_full_history:
            warnings.append(
                "Using all data may reduce short‑term edge due to changing market conditions."
            )
        if optimizer_config.profile in {Profile.AGGRESSIVE_1H.value, Profile.EXTREME_ROI_EXPERIMENTAL.value, Profile.DYNAMIC_INTRADAY.value}:
            warnings.append(self._profile_warning(optimizer_config.profile))
        run.warnings = warnings
        db.session.add(run)
        db.session.commit()

        rankings: List[Dict[str, Any]] = []
        try:
            strategy_names = optimizer_config.strategy_names or self.registry.names()
            symbols = self._optimizer_symbols(optimizer_config)
            run.symbols = symbols
            for symbol in symbols:
                for timeframe in optimizer_config.timeframes:
                    # Fetch all required candles up front.
                    candles = self._fetch_candles(symbol, timeframe, optimizer_config)
                    # Precompute rolling windows.
                    windows = self._rolling_windows(candles, optimizer_config)
                    for strategy_name in strategy_names:
                        # Determine parameter sets to evaluate, capped by max_parameter_sets.
                        for parameters in self._parameter_sets(
                            strategy_name,
                            optimizer_config.max_parameter_sets,
                            optimizer_config,
                        ):
                            if optimizer_config.high_upside_profile:
                                parameters = {
                                    **parameters,
                                    "high_upside_profile": True,
                                    "duration_hours": int(optimizer_config.lock_duration_hours or 0),
                                }
                            result = self._evaluate_candidate(
                                symbol=symbol,
                                timeframe=timeframe,
                                strategy_name=strategy_name,
                                parameters=parameters,
                                candles=candles,
                                windows=windows,
                                optimizer_config=optimizer_config,
                            )
                            rankings.append(result)
                            # Persist each ranking incrementally to avoid large transactions.
                            self._persist_ranking(run, result)

            self._train_ranker_from_results(run, rankings, optimizer_config)
            # Sort by rejection status then by descending score.
            rankings.sort(key=lambda item: self._ranking_sort_key(item, optimizer_config))
            # Persist optional validations for the top N non‑rejected strategies.
            self._persist_backtest_validations(rankings[: optimizer_config.auto_deploy_top_n])
            run.status = "completed"
            ensemble_backtest = self._ensemble_backtest_summary(rankings, optimizer_config)
            duration_ensemble_backtest = self._duration_ensemble_backtest_summary(rankings, optimizer_config)
            rejection_breakdown = self._rejection_breakdown(rankings)
            raw_upside_report = self._raw_upside_report(rankings, optimizer_config)
            max_return_summary = self._max_return_summary(rankings, optimizer_config, duration_ensemble_backtest)
            pair_screening_summary = self._pair_screening_summary(rankings, optimizer_config, duration_ensemble_backtest)
            one_hour_diagnostics = self._one_hour_diagnostics(rankings, optimizer_config)
            run.result = {
                "ranking_count": len(rankings),
                "accepted_count": len([item for item in rankings if not item["rejected"]]),
                "top": rankings[:10],
                "raw_upside_report": raw_upside_report,
                "raw_upside_leaderboard": raw_upside_report.get("top_by_raw_upside", []),
                "max_return_candidate": raw_upside_report.get("max_return_candidate", {}),
                "target_roi_pct": raw_upside_report.get("target_roi_pct", 1000.0),
                "target_roi_hit": raw_upside_report.get("target_roi_hit", False),
                "one_hour_diagnostics": one_hour_diagnostics,
                "one_hour_rejection_breakdown": one_hour_diagnostics.get("rejection_breakdown", {}),
                "net_roi_v2_enabled": bool(self.config.get("NET_ROI_V2_ENABLED", True)),
                "net_roi_v2_summary": self._net_roi_v2_summary(rankings),
                "ensemble_backtest": ensemble_backtest,
                "duration_ensemble_backtest": duration_ensemble_backtest,
                "baseline_single_strategy": duration_ensemble_backtest.get("baseline_single_strategy", ensemble_backtest.get("baseline_best", {})),
                "baseline_current_basket": duration_ensemble_backtest.get("baseline_current_basket", {}),
                "allocation_conservation": duration_ensemble_backtest.get("allocation_conservation", ensemble_backtest.get("allocation_conservation", {})),
                "overfit_rejections": duration_ensemble_backtest.get("overfit_rejections", {}),
                "cap_rejections": duration_ensemble_backtest.get("cap_rejections", {}),
                "max_return_summary": max_return_summary,
                "market_structure_feature_coverage": self._market_structure_feature_coverage(rankings),
                "duration_return_leaders": self._duration_return_leaders(rankings, optimizer_config),
                "rejection_breakdown": rejection_breakdown,
                "live_eligibility_summary": self._live_eligibility_summary(optimizer_config),
                "dynamic_intraday_diagnostics": self._dynamic_intraday_diagnostics(rankings, optimizer_config),
                "high_upside_readiness": self._high_upside_readiness(optimizer_config),
                "offline_ml_readiness": self.offline_ranker.readiness(horizon_from_duration(optimizer_config.lock_duration_hours or 1)),
                "pair_screening_summary": pair_screening_summary,
                "pair_candidates": pair_screening_summary.get("pair_candidates", []),
                "pair_rejection_breakdown": pair_screening_summary.get("pair_rejection_breakdown", {}),
                "pair_baseline_comparison": pair_screening_summary.get("pair_baseline_comparison", {}),
            }
            run.completed_at = datetime.utcnow()
            db.session.commit()
            return run.result | {"optimizer_run_id": run.id, "warnings": warnings}
        except Exception as exc:
            # Mark the run as failed and propagate the exception after recording.
            run.status = "failed"
            run.result = {"error": str(exc)}
            run.completed_at = datetime.utcnow()
            db.session.commit()
            raise

    # ------------------------------------------------------------------
    # Candidate evaluation
    #
    def _evaluate_candidate(
        self,
        *,
        symbol: str,
        timeframe: str,
        strategy_name: str,
        parameters: Dict[str, Any],
        candles: List[Dict[str, Any]],
        windows: List[Tuple[int, int, int]],
        optimizer_config: OptimizerConfig,
    ) -> Dict[str, Any]:
        """Evaluate a single strategy candidate across all rolling windows.

        For each window defined by ``windows`` the method runs a backtest on
        the subset of candle data covering both the training and test period.
        The evaluation start timestamp is set to the beginning of the test
        segment to ensure only out‑of‑sample data contributes to the results.
        Results are aggregated via ``_aggregate_candidate`` to produce a
        single dictionary containing weighted metrics and rejection flags.

        Args:
            symbol: Trading symbol.
            timeframe: Candle timeframe.
            strategy_name: Name of the strategy under test.
            parameters: Parameter dictionary for the strategy.
            candles: All candle data used for the entire optimisation period.
            windows: List of tuples ``(train_start, test_start, test_end)``
                specifying window boundaries as Unix timestamps in milliseconds.
            optimizer_config: Optimiser configuration containing risk limits
                and weighting preferences.

        Returns:
            A dictionary summarising performance and metrics for the candidate.
        """
        window_results: List[Dict[str, Any]] = []
        for train_start, test_start, test_end in windows:
            # Extract only the candles within the current train+test window.
            window_candles = [row for row in candles if train_start <= int(row["timestamp"]) <= test_end]
            # Ensure enough candles for a reliable backtest.
            if len(window_candles) < 30:
                continue
            backtest_config = BacktestConfig(
                strategy_name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                mode=optimizer_config.mode,
                initial_balance=optimizer_config.initial_balance,
                fee_bps=optimizer_config.fee_bps,
                slippage_bps=optimizer_config.slippage_bps,
                stop_loss_pct=float(parameters.get("stop_loss_pct", 0.01)),
                take_profit_pct=float(parameters.get("take_profit_pct", 0.02)),
                position_size_fraction=self._position_size_fraction(parameters, optimizer_config),
                parameters=parameters,
                sizing_mode="risk_based",
                risk_per_trade_pct=self._risk_per_trade_pct(optimizer_config),
                max_daily_loss=float(self.config.get("MAX_DAILY_LOSS_USDC", 0.0)),
                max_drawdown_pct=optimizer_config.max_drawdown_pct,
                loss_streak_cooldown=int(self.config.get("LOSS_STREAK_COOLDOWN_THRESHOLD", 3)),
                cooldown_minutes=int(self.config.get("LOSS_COOLDOWN_MINUTES", 30)),
                max_trades_per_window=int(self.config.get("MAX_TRADES_PER_WINDOW", 0)),
                trade_window_minutes=int(self.config.get("TRADE_WINDOW_MINUTES", 60)),
                intrabar_model="conservative",
                evaluation_start_timestamp=test_start,
                allocation_amount_usd=float(optimizer_config.allocation_amount_usd or 0.0),
                leverage=self._candidate_leverage(parameters, optimizer_config),
                min_liquidation_buffer_pct=float(self.config.get("MIN_LIQUIDATION_BUFFER_PCT", 0.015)),
                funding_cost_bps=float(parameters.get("funding_cost_bps", self.config.get("FUNDING_COST_BPS", 0.0))),
            )
            result = self.runner.run(backtest_config, window_candles)
            result["window_start"] = test_start
            result["window_end"] = test_end
            window_results.append(result)
        market_structure = self._market_structure_features(symbol, timeframe, optimizer_config, candles=candles)
        return self._aggregate_candidate(
            symbol,
            timeframe,
            strategy_name,
            parameters,
            window_results,
            optimizer_config,
            market_structure=market_structure,
        )

    def _aggregate_candidate(
        self,
        symbol: str,
        timeframe: str,
        strategy_name: str,
        parameters: Dict[str, Any],
        window_results: List[Dict[str, Any]],
        optimizer_config: OptimizerConfig,
        market_structure: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Aggregate performance metrics across multiple windows and apply rejection rules."""
        if not window_results:
            reason = "no_test_window_data" if self._is_hourly_experimental(optimizer_config.profile) else "not_enough_window_data"
            return self._rejected_candidate(
                symbol,
                timeframe,
                strategy_name,
                parameters,
                reason,
                optimizer_config,
            )
        # Compute recency weights.
        weights: List[float] = self._weights(len(window_results), optimizer_config)
        # Weighted average metrics across windows.
        weighted_return = self._weighted_mean([row["total_return"] for row in window_results], weights)
        net_return_after_costs = self._weighted_mean([
            float(row.get("net_return_after_costs", row["total_return"]) or 0.0)
            for row in window_results
        ], weights)
        weighted_drawdown = self._weighted_mean([row["max_drawdown"] for row in window_results], weights)
        weighted_profit_factor = min(
            10.0,
            self._weighted_mean([
                self._finite(row["profit_factor"]) for row in window_results
            ], weights),
        )
        weighted_sharpe = self._weighted_mean([row["sharpe_like"] for row in window_results], weights)
        weighted_sortino = self._weighted_mean([row["sortino_like"] for row in window_results], weights)
        weighted_win_rate = self._weighted_mean([float(row.get("win_rate", 0.0) or 0.0) for row in window_results], weights)
        trades = sum(int(row["trade_count"]) for row in window_results)
        trades_per_day = self._weighted_mean([row["trades_per_day"] for row in window_results], weights)
        avg_trade_return = self._weighted_mean([row["average_return_per_trade"] for row in window_results], weights)
        turnover = self._weighted_mean([row["capital_turnover_rate"] for row in window_results], weights)
        turnover_after_fees = self._weighted_mean([
            float(row.get("turnover_after_fees", row.get("capital_turnover_rate", 0.0)) or 0.0)
            for row in window_results
        ], weights)
        profitable = [row for row in window_results if row["total_return"] > 0]
        consistency = len(profitable) / len(window_results)
        returns = [row["total_return"] for row in window_results]
        instability = pstdev(returns) if len(returns) > 1 else 0.0
        window_stability = max(0.0, 1.0 - min(instability * 10, 1.0))
        accepted_window_ratio = consistency
        recent_performance_score = self._recent_score(window_results)
        recent_1h_return = self._recent_return(window_results, hours=1)
        estimated_fees = sum(float(row.get("fees_paid", 0.0) or 0.0) for row in window_results)
        cost_drag_bps = self._weighted_mean([
            self._window_cost_drag_bps(row, optimizer_config) for row in window_results
        ], weights)
        edge_score = self._weighted_mean([
            self._window_edge_score(row, optimizer_config) for row in window_results
        ], weights)
        expectancy = self._weighted_mean([float(row.get("expectancy", 0.0) or 0.0) for row in window_results], weights)
        avg_win = self._weighted_mean([float(row.get("avg_win", 0.0) or 0.0) for row in window_results], weights)
        avg_loss = self._weighted_mean([float(row.get("avg_loss", 0.0) or 0.0) for row in window_results], weights)
        win_loss_ratio = self._weighted_mean([float(row.get("win_loss_ratio", 0.0) or 0.0) for row in window_results], weights)
        max_adverse_excursion = min(float(row.get("max_adverse_excursion", 0.0) or 0.0) for row in window_results)
        max_favorable_excursion = max(float(row.get("max_favorable_excursion", 0.0) or 0.0) for row in window_results)
        no_trade_reason = self._no_trade_reason(window_results)
        leverage = self._weighted_mean([float(row.get("leverage", 1.0) or 1.0) for row in window_results], weights)
        liquidation_buffer_pct = min(float(row.get("liquidation_buffer_pct", 1.0) or 0.0) for row in window_results)
        funding_cost_estimate = sum(float(row.get("funding_cost_estimate", 0.0) or 0.0) for row in window_results)
        capacity_usd = self._capacity_usd(symbol, optimizer_config)
        universe_source = self._universe_source(symbol, optimizer_config)
        execution_style = self._execution_style(edge_score, cost_drag_bps)
        market_structure = self._market_structure_features(
            symbol,
            timeframe,
            optimizer_config,
            market_structure=market_structure,
        )
        market_structure_score = self._safe_market_structure_score(market_structure)
        spread_bps = float(market_structure.get("spread_bps", market_structure.get("spread_trend_bps", 0.0)) or 0.0)
        liquidity_capacity_usd = max(
            float(market_structure.get("book_depth_usd", 0.0) or 0.0),
            capacity_usd / 0.05 if capacity_usd > 0 else 0.0,
        )
        volatility_regime = str(market_structure.get("volatility_regime", "unknown") or "unknown")
        if optimizer_config.profile == Profile.DYNAMIC_INTRADAY.value:
            cost_drag_bps = max(cost_drag_bps, self._market_cost_drag_bps(market_structure, optimizer_config))
        performance_last_3_days = self._recent_return(window_results, days=3)
        performance_last_7_days = self._recent_return(window_results, days=7)
        performance_decay_rate = returns[-1] - returns[0] if len(returns) > 1 else 0.0
        convex_metrics = self._one_hour_convex_metrics(
            net_return_after_costs=net_return_after_costs,
            recent_1h_return=recent_1h_return,
            edge_score=edge_score,
            cost_drag_bps=cost_drag_bps,
            expectancy=expectancy,
            max_adverse_excursion=max_adverse_excursion,
            max_favorable_excursion=max_favorable_excursion,
            liquidity_capacity_usd=liquidity_capacity_usd,
            allocation_amount_usd=float(optimizer_config.allocation_amount_usd or optimizer_config.initial_balance or 0.0),
            window_stability=window_stability,
            weighted_drawdown=weighted_drawdown,
            turnover_after_fees=turnover_after_fees,
            market_structure_score=market_structure_score,
            performance_decay_rate=performance_decay_rate,
        )
        # Compute an overall score using a weighted combination of metrics.
        turnover_penalty = max(0.0, turnover - 3.0) * max(0.0, 0.01 - net_return_after_costs)
        score = (
            recent_performance_score * 2.0
            + net_return_after_costs * 1.4
            + weighted_return * 0.4
            + max(edge_score, 0.0) / 10_000
            + max(weighted_profit_factor - 1.0, 0.0) * 0.15
            + consistency * 0.25
            + window_stability * 0.15
            + min(trades_per_day / 5, 1.0) * 0.1
            + weighted_sharpe * 0.03
            + weighted_sortino * 0.03
            + weighted_drawdown * 1.5
            - instability * 0.5
            - turnover_penalty
        )
        if optimizer_config.market_structure_features_enabled:
            score += market_structure_score * 0.18
            score += min(float(market_structure.get("volume_impulse", 0.0) or 0.0), 5.0) * 0.01
            score -= max(float(market_structure.get("spread_trend_bps", 0.0) or 0.0) - 12.0, 0.0) / 10_000
        rejected = False
        rejection_reason = ""
        warnings: List[str] = []
        # Apply rejection criteria.
        if trades < optimizer_config.min_trade_count:
            rejected = True
            rejection_reason = "low_trade_count"
            warnings.append("Low trade sample; result is statistically weak.")
        if weighted_drawdown <= -abs(optimizer_config.max_drawdown_pct):
            rejected = True
            rejection_reason = rejection_reason or "high_drawdown"
            warnings.append("Drawdown exceeded optimiser threshold.")
        if weighted_profit_factor < 1.0:
            rejected = True
            rejection_reason = rejection_reason or "profit_factor_below_one"
            warnings.append("Losses outweighed profits after costs.")
        if consistency < 0.4:
            rejected = True
            rejection_reason = rejection_reason or "unstable_consistency"
            warnings.append("Performance was not consistent across windows.")
        if recent_performance_score < 0:
            rejected = True
            rejection_reason = rejection_reason or "recent_performance_deterioration"
            warnings.append("Recent out‑of‑sample performance deteriorated.")
        if turnover > 4.0 and net_return_after_costs < max(cost_drag_bps / 10_000, 0.005):
            rejected = True
            rejection_reason = rejection_reason or "high_turnover_low_net_return"
            warnings.append("Turnover was high relative to net return after costs.")
        if len(window_results) >= 3 and len(profitable) <= 1:
            rejected = True
            if not rejection_reason or rejection_reason == "unstable_consistency":
                rejection_reason = "one_window_winner"
            warnings.append("Candidate only worked in one walk-forward window.")
        if liquidation_buffer_pct < float(self.config.get("MIN_LIQUIDATION_BUFFER_PCT", 0.015)):
            rejected = True
            rejection_reason = rejection_reason or "liquidation_buffer_too_tight"
            warnings.append("Leverage left too little liquidation buffer.")
        if optimizer_config.allocation_amount_usd > 0 and capacity_usd > 0 and capacity_usd < optimizer_config.allocation_amount_usd * 0.25:
            rejected = True
            rejection_reason = rejection_reason or "insufficient_liquidity_capacity"
            warnings.append("Liquidity capacity was too low for the requested allocation.")
        if optimizer_config.profile == Profile.AGGRESSIVE_1H.value:
            score, rejected, rejection_reason, warnings = self._aggressive_score_and_rejections(
                score=score,
                net_return_after_costs=net_return_after_costs,
                weighted_return=weighted_return,
                weighted_drawdown=weighted_drawdown,
                weighted_profit_factor=weighted_profit_factor,
                trades=trades,
                trades_per_day=trades_per_day,
                recent_1h_return=recent_1h_return,
                estimated_fees=estimated_fees,
                edge_score=edge_score,
                expectancy=expectancy,
                cost_drag_bps=cost_drag_bps,
                turnover_after_fees=turnover_after_fees,
                max_adverse_excursion=max_adverse_excursion,
                max_favorable_excursion=max_favorable_excursion,
                convex_metrics=convex_metrics,
                window_stability=window_stability,
                instability=instability,
                capacity_usd=capacity_usd,
                liquidity_capacity_usd=liquidity_capacity_usd,
                market_structure_score=market_structure_score,
                no_trade_reason=no_trade_reason,
                optimizer_config=optimizer_config,
                rejected=rejected,
                rejection_reason=rejection_reason,
                warnings=warnings,
            )
            if rejection_reason == "low_edge_after_costs" and not no_trade_reason:
                no_trade_reason = "low_edge_after_costs"
        if optimizer_config.profile == Profile.EXTREME_ROI_EXPERIMENTAL.value:
            score, rejected, rejection_reason, warnings = self._extreme_roi_score_and_rejections(
                weighted_return=weighted_return,
                net_return_after_costs=net_return_after_costs,
                weighted_drawdown=weighted_drawdown,
                weighted_profit_factor=weighted_profit_factor,
                weighted_sortino=weighted_sortino,
                trades=trades,
                recent_1h_return=recent_1h_return,
                edge_score=edge_score,
                expectancy=expectancy,
                win_loss_ratio=win_loss_ratio,
                max_adverse_excursion=max_adverse_excursion,
                max_favorable_excursion=max_favorable_excursion,
                instability=instability,
                cost_drag_bps=cost_drag_bps,
                turnover_after_fees=turnover_after_fees,
                leverage=leverage,
                liquidation_buffer_pct=liquidation_buffer_pct,
                capacity_usd=capacity_usd,
                allocation_amount_usd=float(optimizer_config.allocation_amount_usd or 0.0),
                no_trade_reason=no_trade_reason,
                optimizer_config=optimizer_config,
                warnings=warnings,
            )
            if rejected and rejection_reason == "low_edge_after_costs" and not no_trade_reason:
                no_trade_reason = "low_edge_after_costs"
        if optimizer_config.profile == Profile.AGGRESSIVE_RISK_ADJUSTED.value:
            score = self._aggressive_risk_adjusted_score(
                net_return_after_costs=net_return_after_costs,
                recent_performance_score=recent_performance_score,
                recent_1h_return=recent_1h_return,
                weighted_profit_factor=weighted_profit_factor,
                weighted_sortino=weighted_sortino,
                weighted_sharpe=weighted_sharpe,
                consistency=consistency,
                window_stability=window_stability,
                weighted_drawdown=weighted_drawdown,
                edge_score=edge_score,
                cost_drag_bps=cost_drag_bps,
                turnover_after_fees=turnover_after_fees,
                capacity_usd=capacity_usd,
                allocation_amount_usd=float(optimizer_config.allocation_amount_usd or 0.0),
                trades=trades,
                trades_per_day=trades_per_day,
                optimizer_config=optimizer_config,
            )
            if net_return_after_costs <= 0:
                rejected = True
                rejection_reason = rejection_reason or "negative_net_return_after_costs"
                warnings.append("Net return after fees and slippage was not positive.")

        if optimizer_config.profile == Profile.DYNAMIC_INTRADAY.value:
            score, rejected, rejection_reason, warnings = self._dynamic_intraday_score_and_rejections(
                score=score,
                net_return_after_costs=net_return_after_costs,
                recent_performance_score=recent_performance_score,
                recent_1h_return=recent_1h_return,
                weighted_drawdown=weighted_drawdown,
                weighted_profit_factor=weighted_profit_factor,
                trades=trades,
                trades_per_day=trades_per_day,
                edge_score=edge_score,
                cost_drag_bps=cost_drag_bps,
                turnover_after_fees=turnover_after_fees,
                window_stability=window_stability,
                capacity_usd=capacity_usd,
                market_structure=market_structure,
                optimizer_config=optimizer_config,
                rejected=rejected,
                rejection_reason=rejection_reason,
                warnings=warnings,
            )
            if rejected and rejection_reason == "low_edge_after_costs" and not no_trade_reason:
                no_trade_reason = "low_edge_after_costs"

        if optimizer_config.max_return_optimizer_enabled:
            score = self._max_return_score(
                net_return_after_costs=net_return_after_costs,
                recent_performance_score=recent_performance_score,
                recent_1h_return=recent_1h_return,
                edge_score=edge_score,
                expectancy=expectancy,
                max_adverse_excursion=max_adverse_excursion,
                max_favorable_excursion=max_favorable_excursion,
                capacity_usd=capacity_usd,
                allocation_amount_usd=float(optimizer_config.allocation_amount_usd or 0.0),
                cost_drag_bps=cost_drag_bps,
                weighted_drawdown=weighted_drawdown,
                turnover_after_fees=turnover_after_fees,
                market_structure_score=market_structure_score,
            )
            min_net_return = float(optimizer_config.max_return_min_net_return or 0.0)
            if net_return_after_costs <= min_net_return:
                rejected = True
                rejection_reason = rejection_reason or "negative_net_return_after_costs"
                warnings.append("Max-return objective requires positive net return after costs.")
            drawdown_cap = float(optimizer_config.max_return_max_drawdown_pct or optimizer_config.max_drawdown_pct or 0.0)
            if drawdown_cap > 0 and weighted_drawdown <= -abs(drawdown_cap):
                rejected = True
                rejection_reason = rejection_reason or "high_drawdown"
                warnings.append("Max-return objective rejected excessive drawdown.")

        roi_context = {
            "net_return_after_costs": net_return_after_costs,
            "total_return": weighted_return,
            "recent_performance_score": recent_performance_score,
            "recent_1h_return": recent_1h_return,
            "cost_adjusted_recent_1h_return": convex_metrics.get("cost_adjusted_recent_1h_return", 0.0),
            "max_drawdown": weighted_drawdown,
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "edge_score": edge_score + cost_drag_bps,
            "cost_drag_bps": cost_drag_bps,
            "spread_bps": spread_bps,
            "projected_slippage_bps": optimizer_config.slippage_bps,
            "liquidity_capacity_usd": liquidity_capacity_usd,
            "capacity_usd": capacity_usd,
            "allocation_amount_usd": float(optimizer_config.allocation_amount_usd or optimizer_config.initial_balance or 0.0),
            "capacity_multiple": convex_metrics.get("capacity_multiple", 0.0),
            "mfe_mae_ratio": convex_metrics.get("mfe_mae_ratio", 0.0),
            "max_adverse_excursion": max_adverse_excursion,
            "max_favorable_excursion": max_favorable_excursion,
            "window_stability": window_stability,
            "accepted_window_ratio": accepted_window_ratio,
            "market_structure_score": market_structure_score,
            "ml_score": 0.0,
            "turnover_after_fees": turnover_after_fees,
            "trades_per_day": trades_per_day,
            "avg_trade_return": avg_trade_return,
            "decay_penalty": convex_metrics.get("decay_penalty", 0.0),
            "performance_decay_rate": performance_decay_rate,
            "volatility_regime": volatility_regime,
            "volatility_pct": market_structure.get("volatility_pct", 0.0),
            "breakout_proximity_bps": market_structure.get("breakout_proximity_bps", 250.0),
            "cost_adjusted_expected_move": edge_score - cost_drag_bps,
        }
        roi_payload = net_roi_diagnostics(roi_context, self.config)
        roi_v2_payload = net_roi_v2_diagnostics(roi_context, self.config)
        if bool(self.config.get("OPTIMIZER_PREFILTER_ENABLED", True)) and (
            optimizer_config.profile
            in {
                Profile.AGGRESSIVE_1H.value,
                Profile.AGGRESSIVE_RISK_ADJUSTED.value,
                Profile.EXTREME_ROI_EXPERIMENTAL.value,
                Profile.DYNAMIC_INTRADAY.value,
            }
            or bool(optimizer_config.max_return_optimizer_enabled)
        ):
            if float(roi_payload["expected_fill_quality"]) < float(self.config.get("NET_ROI_MIN_FILL_QUALITY", 0.55) or 0.55):
                rejected = True
                rejection_reason = rejection_reason or "low_expected_fill_quality"
                warnings.append("Expected fill quality was too low after spread, cost, liquidity, and volatility checks.")
            if (
                float(roi_payload["churn_penalty"]) > float(self.config.get("NET_ROI_MAX_CHURN_PENALTY", 0.35) or 0.35)
                and float(roi_payload["edge_after_cost_bps"]) < float(self.config.get("NET_ROI_MIN_EDGE_BPS", 4.0) or 4.0)
            ):
                rejected = True
                rejection_reason = rejection_reason or "excessive_churn"
                warnings.append("Turnover and trade frequency were too high for the expected edge.")
            if float(roi_payload["edge_after_cost_bps"]) < float(self.config.get("NET_ROI_MIN_EDGE_BPS", 4.0) or 4.0):
                rejected = True
                rejection_reason = rejection_reason or "low_net_roi_edge"
                warnings.append("Expected edge after costs was below the net ROI threshold.")

        raw_upside_payload = self._raw_upside_candidate_metrics(
            total_return=weighted_return,
            net_return_after_costs=net_return_after_costs,
            recent_1h_return=recent_1h_return,
            max_favorable_excursion=max_favorable_excursion,
            leverage=leverage,
            rejected=rejected,
            rejection_reason=rejection_reason,
            weighted_drawdown=weighted_drawdown,
            expected_fill_quality=float(roi_payload["expected_fill_quality"]),
            window_stability=window_stability,
            churn_penalty=float(roi_payload["churn_penalty"]),
            regime_support=str(roi_v2_payload["regime_support"]),
            data_age_seconds=float(roi_payload["data_age_seconds"]),
            optimizer_config=optimizer_config,
        )

        ml_payload = self._ml_score_payload(
            score=score,
            rejected=rejected,
            optimizer_config=optimizer_config,
            result_context={
                "strategy_name": strategy_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "profile": optimizer_config.profile,
                "optimizer_profile": optimizer_config.profile,
                "net_return_after_costs": net_return_after_costs,
                "total_return": weighted_return,
                "recent_performance_score": recent_performance_score,
                "recent_1h_return": recent_1h_return,
                "max_drawdown": weighted_drawdown,
                "profit_factor": weighted_profit_factor,
                "sortino_like": weighted_sortino,
                "sharpe_like": weighted_sharpe,
                "consistency": consistency,
                "window_stability": window_stability,
                "accepted_window_ratio": accepted_window_ratio,
                "win_rate": weighted_win_rate,
                "trade_count": trades,
                "trades_per_day": trades_per_day,
                "avg_trade_return": avg_trade_return,
                "edge_score": edge_score,
                "expectancy": expectancy,
                "cost_drag_bps": cost_drag_bps,
                "turnover_after_fees": turnover_after_fees,
                "turnover_rate": turnover,
                "estimated_fees": estimated_fees,
                "allocation_amount_usd": float(optimizer_config.allocation_amount_usd or 0.0),
                "lock_duration_hours": int(optimizer_config.lock_duration_hours or 0),
                "leverage": leverage,
                "liquidation_buffer_pct": liquidation_buffer_pct,
                "capacity_usd": capacity_usd,
                "market_structure": market_structure,
                "market_structure_score": market_structure_score,
                "net_roi_score": roi_payload["net_roi_score"],
                "expected_fill_quality": roi_payload["expected_fill_quality"],
                "churn_penalty": roi_payload["churn_penalty"],
                "net_roi_v2_score": roi_v2_payload["net_roi_v2_score"],
                "roi_quality_grade": roi_v2_payload["roi_quality_grade"],
                "roi_rejection_risk": roi_v2_payload["roi_rejection_risk"],
                "regime_bucket": roi_v2_payload["regime_bucket"],
                "regime_support": roi_v2_payload["regime_support"],
                "tail_loss_penalty": roi_v2_payload["tail_loss_penalty"],
                "downside_asymmetry_penalty": roi_v2_payload["downside_asymmetry_penalty"],
                "cost_adjusted_breakout_potential": roi_v2_payload["cost_adjusted_breakout_potential"],
                "profit_objective_version": "max_return_v3" if optimizer_config.max_return_optimizer_enabled else "",
            },
        )
        if not rejected:
            score = ml_payload["ml_adjusted_score"]
            score += float(roi_payload["net_roi_score"]) * 0.05
            if bool(self.config.get("NET_ROI_V2_ENABLED", True)):
                score += float(roi_v2_payload["net_roi_v2_score"]) * 0.03
        return {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "profile": optimizer_config.profile,
            "experimental": self._is_experimental_profile(optimizer_config.profile),
            "risk_label": self._risk_label(optimizer_config.profile),
            "warning": self._profile_warning(optimizer_config.profile),
            "parameters": parameters,
            "score": score,
            "base_score": ml_payload["base_score"],
            "ml_score": ml_payload["ml_score"],
            "ml_adjusted_score": ml_payload["ml_adjusted_score"],
            "ml_warmup": ml_payload["ml_warmup"],
            "ml_explanation": ml_payload["ml_explanation"],
            "offline_ml_prediction": ml_payload["offline_ml_prediction"],
            "offline_ml_status": ml_payload["offline_ml_status"],
            "offline_ml_blend_applied": ml_payload["offline_ml_blend_applied"],
            "offline_ml_model_id": ml_payload["offline_ml_model_id"],
            "total_return": weighted_return,
            "net_return_after_costs": net_return_after_costs,
            "raw_upside_score": raw_upside_payload["raw_upside_score"],
            "raw_total_return_pct": raw_upside_payload["raw_total_return_pct"],
            "raw_net_return_pct": raw_upside_payload["raw_net_return_pct"],
            "target_roi_pct": raw_upside_payload["target_roi_pct"],
            "target_roi_hit": raw_upside_payload["target_roi_hit"],
            "live_blockers": raw_upside_payload["live_blockers"],
            "recent_performance_score": recent_performance_score,
            "recent_1h_return": recent_1h_return,
            "estimated_fees": estimated_fees,
            "edge_score": edge_score,
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "cost_drag_bps": cost_drag_bps,
            "net_roi_score": roi_payload["net_roi_score"],
            "net_roi_v2_score": roi_v2_payload["net_roi_v2_score"],
            "roi_quality_grade": roi_v2_payload["roi_quality_grade"],
            "roi_rejection_risk": roi_v2_payload["roi_rejection_risk"],
            "regime_bucket": roi_v2_payload["regime_bucket"],
            "regime_support": roi_v2_payload["regime_support"],
            "regime_adjustment": roi_v2_payload["regime_adjustment"],
            "regime_adjusted_expectancy": roi_v2_payload["regime_adjusted_expectancy"],
            "tail_loss_penalty": roi_v2_payload["tail_loss_penalty"],
            "downside_asymmetry_penalty": roi_v2_payload["downside_asymmetry_penalty"],
            "cost_adjusted_breakout_potential": roi_v2_payload["cost_adjusted_breakout_potential"],
            "expected_fill_quality": roi_payload["expected_fill_quality"],
            "churn_penalty": roi_payload["churn_penalty"],
            "edge_after_cost_bps": roi_payload["edge_after_cost_bps"],
            "data_age_seconds": roi_payload["data_age_seconds"],
            "net_roi_components": roi_payload["net_roi_components"],
            "net_roi_v2_components": roi_v2_payload["net_roi_v2_components"],
            "convex_edge_score": convex_metrics["convex_edge_score"],
            "mfe_mae_ratio": convex_metrics["mfe_mae_ratio"],
            "capacity_multiple": convex_metrics["capacity_multiple"],
            "cost_adjusted_recent_1h_return": convex_metrics["cost_adjusted_recent_1h_return"],
            "decay_penalty": convex_metrics["decay_penalty"],
            "liquidity_capacity_usd": liquidity_capacity_usd,
            "spread_bps": spread_bps,
            "volatility_regime": volatility_regime,
            "recent_decay": performance_decay_rate,
            "max_adverse_excursion": max_adverse_excursion,
            "max_favorable_excursion": max_favorable_excursion,
            "no_trade_reason": no_trade_reason,
            "allocation_amount_usd": float(optimizer_config.allocation_amount_usd or 0.0),
            "lock_duration_hours": int(optimizer_config.lock_duration_hours or 0),
            "leverage": leverage,
            "liquidation_buffer_pct": liquidation_buffer_pct,
            "capacity_usd": capacity_usd,
            "universe_source": universe_source,
            "vault_leg_count": max(1, int(optimizer_config.max_parallel_legs or 1)),
            "execution_style": execution_style,
            "profit_objective_version": "max_return_v3" if optimizer_config.max_return_optimizer_enabled else "",
            "high_upside_profile": bool(optimizer_config.high_upside_profile),
            "market_structure": market_structure,
            "market_structure_score": market_structure_score,
            "funding_cost_estimate": funding_cost_estimate,
            "performance_last_3_days": performance_last_3_days,
            "performance_last_7_days": performance_last_7_days,
            "performance_decay_rate": performance_decay_rate,
            "max_drawdown": weighted_drawdown,
            "profit_factor": weighted_profit_factor,
            "sharpe_like": weighted_sharpe,
            "sortino_like": weighted_sortino,
            "trades_per_day": trades_per_day,
            "avg_trade_return": avg_trade_return,
            "turnover_rate": turnover,
            "turnover_after_fees": turnover_after_fees,
            "consistency": consistency,
            "window_stability": window_stability,
            "accepted_window_ratio": accepted_window_ratio,
            "win_rate": weighted_win_rate,
            "trade_count": trades,
            "window_count": len(window_results),
            "rejected": rejected,
            "rejection_reason": rejection_reason,
            "warnings": warnings,
            "live_eligibility": self._candidate_live_eligibility(rejected, optimizer_config),
        }

    # ------------------------------------------------------------------
    # Candle and window handling
    #
    def _one_hour_convex_metrics(
        self,
        *,
        net_return_after_costs: float,
        recent_1h_return: float,
        edge_score: float,
        cost_drag_bps: float,
        expectancy: float,
        max_adverse_excursion: float,
        max_favorable_excursion: float,
        liquidity_capacity_usd: float,
        allocation_amount_usd: float,
        window_stability: float,
        weighted_drawdown: float,
        turnover_after_fees: float,
        market_structure_score: float,
        performance_decay_rate: float,
    ) -> dict[str, float]:
        """Score 1H candidates for asymmetric upside after execution costs."""

        adverse = abs(min(float(max_adverse_excursion or 0.0), 0.0))
        favorable = max(float(max_favorable_excursion or 0.0), 0.0)
        mfe_mae_ratio = favorable / max(adverse, 1e-9) if favorable > 0 else 0.0
        allocation = max(float(allocation_amount_usd or 0.0), 0.0)
        capacity = max(float(liquidity_capacity_usd or 0.0), 0.0)
        capacity_multiple = capacity / allocation if allocation > 0 and capacity > 0 else 0.0
        cost_adjusted_recent = float(recent_1h_return or 0.0) - max(float(cost_drag_bps or 0.0), 0.0) / 10_000
        half_life = max(float(self.config.get("AGGRESSIVE_1H_RECENCY_HALF_LIFE_HOURS", 12.0) or 12.0), 1.0)
        decay_penalty = max(0.0, -float(performance_decay_rate or 0.0)) * min(max(24.0 / half_life, 0.25), 4.0)
        edge_after_cost = max(float(edge_score or 0.0) - float(cost_drag_bps or 0.0), 0.0)
        max_cost_drag = float(self.config.get("AGGRESSIVE_1H_MAX_COST_DRAG_BPS", 18.0) or 18.0)
        convex_edge_score = (
            float(net_return_after_costs or 0.0) * 240.0
            + cost_adjusted_recent * 180.0
            + edge_after_cost * 0.35
            + max(float(expectancy or 0.0), 0.0) * 0.8
            + min(mfe_mae_ratio, 8.0) * 0.55
            + min(capacity_multiple, 5.0) * 0.8
            + max(float(market_structure_score or 0.0), 0.0) * 6.0
            + max(float(window_stability or 0.0), 0.0) * 1.5
            + float(weighted_drawdown or 0.0) * 20.0
            - max(float(cost_drag_bps or 0.0) - max_cost_drag, 0.0) * 0.25
            - max(float(turnover_after_fees or 0.0) - 4.0, 0.0) * 0.25
            - decay_penalty * 50.0
        )
        return {
            "convex_edge_score": convex_edge_score,
            "mfe_mae_ratio": mfe_mae_ratio,
            "capacity_multiple": capacity_multiple,
            "cost_adjusted_recent_1h_return": cost_adjusted_recent,
            "decay_penalty": decay_penalty,
        }

    def _aggressive_score_and_rejections(
        self,
        *,
        score: float,
        net_return_after_costs: float,
        weighted_return: float,
        weighted_drawdown: float,
        weighted_profit_factor: float,
        trades: int,
        trades_per_day: float,
        recent_1h_return: float,
        estimated_fees: float,
        edge_score: float,
        expectancy: float,
        cost_drag_bps: float,
        turnover_after_fees: float,
        max_adverse_excursion: float,
        max_favorable_excursion: float,
        convex_metrics: dict[str, float],
        window_stability: float,
        instability: float,
        capacity_usd: float,
        liquidity_capacity_usd: float,
        market_structure_score: float,
        no_trade_reason: str,
        optimizer_config: OptimizerConfig,
        rejected: bool,
        rejection_reason: str,
        warnings: List[str],
    ) -> tuple[float, bool, str, List[str]]:
        """Apply the experimental 1H score and hard rejection rules."""
        productive_frequency = 0.0
        if net_return_after_costs > 0 and edge_score > cost_drag_bps:
            productive_frequency = min(trades_per_day / 12.0, 1.0) * 0.12
        liquidity_bonus = 0.0
        capacity_multiple = float(convex_metrics.get("capacity_multiple", 0.0) or 0.0)
        if capacity_multiple > 0:
            liquidity_bonus = min(capacity_multiple, 4.0) * 0.025
        churn_penalty = max(0.0, turnover_after_fees - 4.0) * max(0.0, 0.012 - net_return_after_costs) * 0.7
        score = (
            float(convex_metrics.get("convex_edge_score", score) or score)
            + recent_1h_return * 1.2
            + net_return_after_costs * 0.8
            + weighted_return * 0.4
            + max(weighted_profit_factor - 1.2, 0.0) * 0.20
            + min(trades / max(optimizer_config.min_trade_count, 1), 2.0) * 0.05
            + productive_frequency
            + liquidity_bonus
            + market_structure_score * 0.8
            - instability * 0.5
            - min(estimated_fees / max(optimizer_config.initial_balance, 1.0), 1.0) * 0.03
            - churn_penalty
        )
        if AGGRESSIVE_1H_WARNING not in warnings:
            warnings.append(AGGRESSIVE_1H_WARNING)
        rejected = False
        rejection_reason = ""

        if trades < optimizer_config.min_trade_count:
            rejected = True
            rejection_reason = "low_trade_count"
        if weighted_drawdown <= -abs(optimizer_config.max_drawdown_pct):
            rejected = True
            rejection_reason = rejection_reason or "high_drawdown"
        if recent_1h_return < 0:
            rejected = True
            rejection_reason = rejection_reason or "negative_recent_1h_return"
        if weighted_profit_factor < 1.0:
            rejected = True
            rejection_reason = rejection_reason or "profit_factor_below_one"
        if weighted_return <= 0:
            rejected = True
            rejection_reason = rejection_reason or "negative_net_pnl_after_costs"
        min_edge = float(optimizer_config.min_edge_bps or self.config.get("AGGRESSIVE_MIN_EDGE_BPS", 4.0))
        if edge_score < min_edge:
            rejected = True
            rejection_reason = rejection_reason or "low_edge_after_costs"
            if no_trade_reason != "low_edge_after_costs":
                warnings.append("Expected edge after fees and slippage was below the aggressive threshold.")
        max_cost_drag = float(self.config.get("AGGRESSIVE_1H_MAX_COST_DRAG_BPS", 18.0) or 18.0)
        if cost_drag_bps > max_cost_drag:
            rejected = True
            rejection_reason = rejection_reason or "cost_drag_above_threshold"
            warnings.append("1H candidate execution costs exceeded the configured cost-drag cap.")
        min_capacity_multiple = float(self.config.get("AGGRESSIVE_1H_MIN_CAPACITY_MULTIPLE", 2.0) or 2.0)
        if optimizer_config.allocation_amount_usd > 0 and liquidity_capacity_usd > 0 and capacity_multiple < min_capacity_multiple:
            rejected = True
            rejection_reason = rejection_reason or "insufficient_liquidity_capacity"
            warnings.append("1H candidate liquidity capacity was too small for the requested allocation.")
        min_mfe_mae = float(self.config.get("AGGRESSIVE_1H_MIN_MFE_MAE", 1.5) or 1.5)
        mfe_mae_ratio = float(convex_metrics.get("mfe_mae_ratio", 0.0) or 0.0)
        if max_favorable_excursion > 0 and abs(min(max_adverse_excursion, 0.0)) > 0 and mfe_mae_ratio < min_mfe_mae:
            rejected = True
            rejection_reason = rejection_reason or "weak_convexity"
            warnings.append("1H candidate did not show enough favorable excursion versus adverse excursion.")
        min_stability = float(self.config.get("AGGRESSIVE_1H_MIN_WINDOW_STABILITY", 0.55) or 0.55)
        if window_stability > 0 and window_stability < min_stability:
            rejected = True
            rejection_reason = rejection_reason or "low_window_stability"
            warnings.append("1H candidate was not stable enough across walk-forward windows.")

        return score, rejected, rejection_reason, warnings

    def _aggressive_risk_adjusted_score(
        self,
        *,
        net_return_after_costs: float,
        recent_performance_score: float,
        recent_1h_return: float,
        weighted_profit_factor: float,
        weighted_sortino: float,
        weighted_sharpe: float,
        consistency: float,
        window_stability: float,
        weighted_drawdown: float,
        edge_score: float,
        cost_drag_bps: float,
        turnover_after_fees: float,
        capacity_usd: float,
        allocation_amount_usd: float,
        trades: int,
        trades_per_day: float,
        optimizer_config: OptimizerConfig,
    ) -> float:
        """Score for aggressive growth after risk and execution constraints."""
        liquidity_bonus = 0.0
        if allocation_amount_usd > 0 and capacity_usd > 0:
            liquidity_bonus = min(capacity_usd / max(allocation_amount_usd, 1.0), 4.0) * 0.03
        trade_sample = min(trades / max(optimizer_config.min_trade_count, 1), 2.0) * 0.06
        duration = max(int(optimizer_config.lock_duration_hours or 24), 1)
        recent_component = recent_1h_return if duration <= 1 else recent_performance_score
        upside_impulse = max(recent_1h_return, 0.0) * (1.2 if duration <= 1 else 0.55)
        productive_frequency = 0.0
        if net_return_after_costs > 0 and edge_score > cost_drag_bps:
            productive_frequency = min(trades_per_day / max(6.0 if duration <= 1 else 3.0, 1.0), 1.0) * 0.09
        turnover_penalty = max(0.0, turnover_after_fees - 3.5) * max(0.0, 0.012 - net_return_after_costs) * 0.6
        cost_penalty = max(0.0, cost_drag_bps - 20.0) / 10_000 * 0.7

        return (
            net_return_after_costs * 2.4
            + recent_component * 1.6
            + upside_impulse
            + max(weighted_profit_factor - 1.0, 0.0) * 0.22
            + weighted_sortino * 0.055
            + weighted_sharpe * 0.025
            + consistency * 0.34
            + window_stability * 0.22
            + max(edge_score, 0.0) / 10_000 * 0.65
            + liquidity_bonus
            + trade_sample
            + productive_frequency
            + weighted_drawdown * 2.0
            - turnover_penalty
            - cost_penalty
        )

    def _dynamic_intraday_score_and_rejections(
        self,
        *,
        score: float,
        net_return_after_costs: float,
        recent_performance_score: float,
        recent_1h_return: float,
        weighted_drawdown: float,
        weighted_profit_factor: float,
        trades: int,
        trades_per_day: float,
        edge_score: float,
        cost_drag_bps: float,
        turnover_after_fees: float,
        window_stability: float,
        capacity_usd: float,
        market_structure: dict[str, Any],
        optimizer_config: OptimizerConfig,
        rejected: bool,
        rejection_reason: str,
        warnings: List[str],
    ) -> tuple[float, bool, str, List[str]]:
        """Score dynamic intraday candidates by tradable edge after execution costs."""
        if DYNAMIC_INTRADAY_WARNING not in warnings:
            warnings.append(DYNAMIC_INTRADAY_WARNING)

        spread_bps = float(market_structure.get("spread_bps", market_structure.get("spread_trend_bps", 0.0)) or 0.0)
        liquidity_usd = max(float(market_structure.get("book_depth_usd", 0.0) or 0.0), capacity_usd / 0.05 if capacity_usd > 0 else 0.0)
        market_score = self._safe_market_structure_score(market_structure)
        volume_impulse = min(float(market_structure.get("volume_impulse", 0.0) or 0.0), 5.0)
        volatility_regime = str(market_structure.get("volatility_regime", "unknown") or "unknown")
        liquidity_bonus = min(liquidity_usd / max(float(self.config.get("DYNAMIC_INTRADAY_MIN_LIQUIDITY_USD", 50_000.0)), 1.0), 4.0) * 0.08
        productive_frequency = 0.0
        if net_return_after_costs > 0 and edge_score > cost_drag_bps:
            productive_frequency = min(trades_per_day / 18.0, 1.0) * 0.10
        churn_penalty = max(0.0, turnover_after_fees - 4.0) * max(0.0, 0.01 - net_return_after_costs) * 0.8
        spread_penalty = max(0.0, spread_bps - float(self.config.get("DYNAMIC_INTRADAY_MAX_SPREAD_BPS", 12.0))) / 10_000
        cost_penalty = max(0.0, cost_drag_bps - float(optimizer_config.min_edge_bps or 0.0)) / 10_000

        score = (
            score
            + max(recent_1h_return, recent_performance_score) * 1.4
            + max(edge_score - cost_drag_bps, 0.0) / 10_000 * 1.2
            + market_score * 0.25
            + volume_impulse * 0.025
            + liquidity_bonus
            + productive_frequency
            + window_stability * 0.12
            + weighted_drawdown * 0.7
            - churn_penalty
            - spread_penalty
            - cost_penalty
        )

        min_edge = float(optimizer_config.min_edge_bps or self.config.get("DYNAMIC_INTRADAY_MIN_EDGE_BPS", 6.0))
        min_liquidity = float(self.config.get("DYNAMIC_INTRADAY_MIN_LIQUIDITY_USD", 50_000.0) or 50_000.0)
        max_spread = float(self.config.get("DYNAMIC_INTRADAY_MAX_SPREAD_BPS", 12.0) or 12.0)

        if trades < optimizer_config.min_trade_count:
            rejected = True
            rejection_reason = rejection_reason or "low_trade_count"
        if net_return_after_costs <= 0:
            rejected = True
            rejection_reason = rejection_reason or "negative_net_return_after_costs"
        if weighted_drawdown <= -abs(optimizer_config.max_drawdown_pct):
            rejected = True
            rejection_reason = rejection_reason or "high_drawdown"
        if weighted_profit_factor < 1.0:
            rejected = True
            rejection_reason = rejection_reason or "profit_factor_below_one"
        if edge_score < min_edge:
            rejected = True
            rejection_reason = rejection_reason or "low_edge_after_costs"
            warnings.append("Dynamic intraday edge after execution costs was below threshold.")
        if liquidity_usd > 0 and liquidity_usd < min_liquidity:
            rejected = True
            rejection_reason = rejection_reason or "insufficient_liquidity_capacity"
            warnings.append("Dynamic intraday liquidity was below the configured production floor.")
        if spread_bps > max_spread:
            rejected = True
            rejection_reason = rejection_reason or "spread_above_threshold"
            warnings.append("Dynamic intraday spread exceeded the configured production cap.")
        if volatility_regime == "dislocated":
            rejected = True
            rejection_reason = rejection_reason or "dislocated_volatility_regime"
            warnings.append("Dynamic intraday volatility regime was dislocated.")

        return score, rejected, rejection_reason, warnings

    def _extreme_roi_score_and_rejections(
        self,
        *,
        weighted_return: float,
        net_return_after_costs: float,
        weighted_drawdown: float,
        weighted_profit_factor: float,
        weighted_sortino: float,
        trades: int,
        recent_1h_return: float,
        edge_score: float,
        expectancy: float,
        win_loss_ratio: float,
        max_adverse_excursion: float,
        max_favorable_excursion: float,
        instability: float,
        cost_drag_bps: float,
        turnover_after_fees: float,
        leverage: float,
        liquidation_buffer_pct: float,
        capacity_usd: float,
        allocation_amount_usd: float,
        no_trade_reason: str,
        optimizer_config: OptimizerConfig,
        warnings: List[str],
    ) -> tuple[float, bool, str, List[str]]:
        """Score high-upside 1H candidates without relaxing hard risk gates."""

        if EXTREME_ROI_WARNING not in warnings:
            warnings.append(EXTREME_ROI_WARNING)

        mfe_mae_ratio = max_favorable_excursion / max(abs(max_adverse_excursion), 1e-9)
        liquidity_bonus = 0.0
        if allocation_amount_usd > 0 and capacity_usd > 0:
            liquidity_bonus = min(capacity_usd / max(allocation_amount_usd, 1.0), 5.0) * 0.04
        convexity = max(recent_1h_return, 0.0) * max(leverage, 1.0)
        payoff_asymmetry = min(mfe_mae_ratio, 6.0) * 0.04
        leverage_bonus = max(leverage - 1.0, 0.0) * 0.025
        cost_penalty = max(0.0, cost_drag_bps - 18.0) / 10_000 * 1.4
        churn_penalty = max(0.0, turnover_after_fees - 5.0) * max(0.0, 0.015 - net_return_after_costs) * 0.8

        score = (
            convexity * 4.2
            + net_return_after_costs * 2.2
            + weighted_return * 1.1
            + max(weighted_profit_factor - 1.0, 0.0) * 0.30
            + weighted_sortino * 0.06
            + max(edge_score, 0.0) / 10_000 * 1.25
            + max(expectancy, 0.0) * 0.02
            + max(win_loss_ratio - 1.0, 0.0) * 0.035
            + payoff_asymmetry
            + min(trades / max(optimizer_config.min_trade_count, 1), 2.5) * 0.08
            + leverage_bonus
            + liquidity_bonus
            + weighted_drawdown * 1.7
            - instability * 0.65
            - cost_penalty
            - churn_penalty
        )

        rejected = False
        rejection_reason = ""
        if trades < optimizer_config.min_trade_count:
            rejected = True
            rejection_reason = "low_trade_count"
        if weighted_drawdown <= -abs(optimizer_config.max_drawdown_pct):
            rejected = True
            rejection_reason = rejection_reason or "high_drawdown"
        if recent_1h_return <= 0:
            rejected = True
            rejection_reason = rejection_reason or "negative_recent_1h_return"
        if net_return_after_costs <= 0:
            rejected = True
            rejection_reason = rejection_reason or "negative_net_return_after_costs"
        if weighted_profit_factor < 1.0:
            rejected = True
            rejection_reason = rejection_reason or "profit_factor_below_one"
        min_edge = float(optimizer_config.min_edge_bps or self.config.get("EXTREME_ROI_MIN_EDGE_BPS", 8.0))
        if edge_score < min_edge:
            rejected = True
            rejection_reason = rejection_reason or "low_edge_after_costs"
            if no_trade_reason != "low_edge_after_costs":
                warnings.append("Extreme ROI candidate edge after costs was below threshold.")
        if liquidation_buffer_pct < float(self.config.get("MIN_LIQUIDATION_BUFFER_PCT", 0.015)):
            rejected = True
            rejection_reason = rejection_reason or "liquidation_buffer_too_tight"
        if allocation_amount_usd > 0 and capacity_usd > 0 and capacity_usd < allocation_amount_usd * 0.25:
            rejected = True
            rejection_reason = rejection_reason or "insufficient_liquidity_capacity"

        return score, rejected, rejection_reason, warnings

    def _ml_score_payload(
        self,
        *,
        score: float,
        rejected: bool,
        optimizer_config: OptimizerConfig,
        result_context: dict[str, Any],
    ) -> dict[str, Any]:
        horizon = horizon_from_duration(optimizer_config.lock_duration_hours or 0)
        features = extract_features({**result_context, "horizon": horizon})
        explanation: dict[str, Any] = {}
        ml_score = 0.0
        warmup = True
        adjusted = score
        if bool(self.config.get("ML_RANKER_ENABLED", False)):
            explanation = self.online_ranker.explain(features, horizon)
            ml_score = float(explanation.get("prediction", 0.0) or 0.0)
            warmup = bool(explanation.get("warmup", True))
            if not warmup and not rejected:
                adjusted += float(self.config.get("ML_SCORE_WEIGHT", 0.15)) * ml_score
        offline_payload = self._offline_ml_payload(
            {**result_context, "horizon": horizon},
            horizon,
            base_score=adjusted,
            rejected=rejected,
        )
        if bool(offline_payload.get("blend_applied", False)):
            adjusted = float(offline_payload.get("blended_score", adjusted) or adjusted)
        explanation = {
            **explanation,
            "offline_model": offline_payload,
        }
        return {
            "base_score": score,
            "ml_score": ml_score,
            "ml_adjusted_score": adjusted,
            "ml_warmup": warmup,
            "ml_explanation": explanation,
            "offline_ml_prediction": float(offline_payload.get("prediction", 0.0) or 0.0),
            "offline_ml_status": offline_payload.get("status", "no_promoted_model"),
            "offline_ml_blend_applied": bool(offline_payload.get("blend_applied", False)),
            "offline_ml_model_id": offline_payload.get("model_id"),
        }

    def _offline_ml_payload(
        self,
        context: dict[str, Any],
        horizon: str,
        *,
        base_score: float,
        rejected: bool,
    ) -> dict[str, Any]:
        try:
            return dict(self.offline_ranker.score_payload(context, horizon, base_score=base_score, rejected=rejected))
        except Exception as exc:  # noqa: BLE001
            return {
                "enabled": bool(self.config.get("ML_OFFLINE_MODELS_ENABLED", False)),
                "blend_enabled": bool(self.config.get("ML_OFFLINE_BLEND_ENABLED", False)),
                "blend_applied": False,
                "status": "offline_ranker_error",
                "horizon": str(horizon or "global"),
                "prediction": 0.0,
                "blended_score": base_score,
                "blockers": [str(exc)],
            }

    def _market_structure_features(
        self,
        symbol: str,
        timeframe: str,
        optimizer_config: OptimizerConfig,
        *,
        candles: list[dict[str, Any]] | None = None,
        market_structure: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(market_structure, dict):
            return market_structure
        if self.market_structure is None:
            return self._neutral_market_structure(optimizer_config)
        service_config = getattr(self.market_structure, "config", None)
        previous: dict[str, Any] = {}
        overrides = {
            "MARKET_STRUCTURE_FEATURES_ENABLED": optimizer_config.market_structure_features_enabled,
            "MARKET_STRUCTURE_PROVIDER": optimizer_config.market_structure_provider,
        }
        try:
            if isinstance(service_config, dict):
                for key, value in overrides.items():
                    previous[key] = service_config.get(key)
                    service_config[key] = value
            return self.market_structure.snapshot(
                symbol,
                timeframe,
                mode=optimizer_config.mode,
                candles=candles,
            )
        except Exception as exc:  # noqa: BLE001
            if bool(self.config.get("MARKET_STRUCTURE_FAIL_CLOSED", True)):
                return self._neutral_market_structure(optimizer_config, error=str(exc))
            raise
        finally:
            if isinstance(service_config, dict):
                for key, value in previous.items():
                    service_config[key] = value

    @staticmethod
    def _neutral_market_structure(optimizer_config: OptimizerConfig, error: str = "") -> dict[str, Any]:
        return {
            "enabled": bool(optimizer_config.market_structure_features_enabled),
            "provider": optimizer_config.market_structure_provider,
            "score": 0.0,
            "coverage": 0.0,
            "funding_rate": 0.0,
            "open_interest_change_pct": 0.0,
            "book_depth_score": 0.0,
            "spread_trend_bps": 0.0,
            "liquidation_proxy": 0.0,
            "volume_impulse": 0.0,
            "volatility_regime": "unknown",
            "volatility_regime_score": 0.0,
            "fail_closed": bool(error),
            "error": error,
        }

    @staticmethod
    def _safe_market_structure_score(market_structure: dict[str, Any]) -> float:
        try:
            return max(0.0, min(float(market_structure.get("score", 0.0) or 0.0), 1.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _market_cost_drag_bps(market_structure: dict[str, Any], optimizer_config: OptimizerConfig) -> float:
        spread = float(market_structure.get("spread_bps", market_structure.get("spread_trend_bps", 0.0)) or 0.0)
        fee_round_trip = float(optimizer_config.fee_bps or 0.0) * 2
        slippage = float(optimizer_config.slippage_bps or 0.0)
        return max(0.0, spread + fee_round_trip + slippage)

    @staticmethod
    def _candidate_live_eligibility(rejected: bool, optimizer_config: OptimizerConfig) -> dict[str, Any]:
        if rejected:
            status = "paper_only"
        elif optimizer_config.high_upside_profile:
            status = "high_upside_live_eligible" if bool(optimizer_config.dynamic_intraday_live_eligible or optimizer_config.max_return_live_eligible or optimizer_config.experimental_live_eligible) else "pre_live_validation_required"
        elif optimizer_config.profile == Profile.DYNAMIC_INTRADAY.value:
            status = "live_eligible" if optimizer_config.dynamic_intraday_live_eligible else "shadow_live_eligible"
        elif optimizer_config.require_shadow_validation:
            status = "shadow_live_eligible"
        else:
            status = "paper_eligible"
        return {
            "status": status,
            "paper_eligible": not rejected,
            "shadow_live_required": bool(optimizer_config.require_shadow_validation),
            "shadow_live_eligible": not rejected and bool(optimizer_config.require_shadow_validation),
            "live_eligible": (
                not rejected
                and (
                    (
                        optimizer_config.profile == Profile.DYNAMIC_INTRADAY.value
                        and bool(optimizer_config.dynamic_intraday_live_eligible)
                    )
                    or (
                        bool(optimizer_config.high_upside_profile)
                        and bool(
                            optimizer_config.dynamic_intraday_live_eligible
                            or optimizer_config.max_return_live_eligible
                            or optimizer_config.experimental_live_eligible
                        )
                    )
                )
            ),
        }

    @staticmethod
    def _max_return_score(
        *,
        net_return_after_costs: float,
        recent_performance_score: float,
        recent_1h_return: float,
        edge_score: float,
        expectancy: float,
        max_adverse_excursion: float,
        max_favorable_excursion: float,
        capacity_usd: float,
        allocation_amount_usd: float,
        cost_drag_bps: float,
        weighted_drawdown: float,
        turnover_after_fees: float,
        market_structure_score: float,
    ) -> float:
        adverse = abs(min(float(max_adverse_excursion or 0.0), 0.0))
        favorable = max(float(max_favorable_excursion or 0.0), 0.0)
        mfe_mae = favorable / max(adverse, 1e-9) if favorable > 0 else 0.0
        capacity_ratio = min(float(capacity_usd or 0.0) / max(float(allocation_amount_usd or 0.0), 1.0), 5.0)
        recent = max(float(recent_performance_score or 0.0), float(recent_1h_return or 0.0))
        return (
            float(net_return_after_costs or 0.0) * 260.0
            + recent * 95.0
            + max(float(edge_score or 0.0), 0.0) * 0.45
            + max(float(expectancy or 0.0), 0.0) * 0.8
            + min(mfe_mae, 8.0) * 0.65
            + capacity_ratio * 1.2
            + max(float(market_structure_score or 0.0), 0.0) * 8.0
            + float(weighted_drawdown or 0.0) * 25.0
            - max(float(cost_drag_bps or 0.0) - 12.0, 0.0) * 0.18
            - max(float(turnover_after_fees or 0.0) - 5.0, 0.0) * 0.20
        )

    @staticmethod
    def _ranking_sort_key(item: dict[str, Any], optimizer_config: OptimizerConfig) -> tuple[Any, ...]:
        if item.get("rejected"):
            return (1, 0.0, 0.0, 0.0)
        if optimizer_config.profile == Profile.AGGRESSIVE_1H.value:
            return (
                0,
                -float(item.get("net_roi_v2_score", item.get("net_roi_score", 0.0)) or 0.0),
                -float(item.get("net_roi_score", 0.0) or 0.0),
                -float(item.get("convex_edge_score", item.get("score", 0.0)) or 0.0),
                -float(item.get("cost_adjusted_recent_1h_return", 0.0) or 0.0),
                -float(item.get("mfe_mae_ratio", 0.0) or 0.0),
                -float(item.get("capacity_multiple", 0.0) or 0.0),
                float(item.get("cost_drag_bps", 0.0) or 0.0),
                -float(item.get("score", 0.0) or 0.0),
            )
        if optimizer_config.max_return_optimizer_enabled:
            favorable = max(float(item.get("max_favorable_excursion", 0.0) or 0.0), 0.0)
            adverse = abs(min(float(item.get("max_adverse_excursion", 0.0) or 0.0), 0.0))
            mfe_mae = favorable / max(adverse, 1e-9) if favorable > 0 else 0.0
            return (
                0,
                -float(item.get("net_roi_v2_score", item.get("net_roi_score", 0.0)) or 0.0),
                -float(item.get("net_roi_score", 0.0) or 0.0),
                -float(item.get("net_return_after_costs", 0.0) or 0.0),
                -max(float(item.get("recent_performance_score", 0.0) or 0.0), float(item.get("recent_1h_return", 0.0) or 0.0)),
                -float(item.get("expectancy", 0.0) or 0.0),
                -min(mfe_mae, 8.0),
                -float(item.get("capacity_usd", 0.0) or 0.0),
                float(item.get("cost_drag_bps", 0.0) or 0.0),
                -float(item.get("score", 0.0) or 0.0),
            )
        return (
            0,
            -float(item.get("net_roi_v2_score", item.get("net_roi_score", item.get("score", 0.0))) or 0.0),
            -float(item.get("net_roi_score", item.get("score", 0.0)) or 0.0),
            -float(item.get("score", 0.0) or 0.0),
            -float(item.get("net_return_after_costs", 0.0) or 0.0),
            0.0,
        )

    def _train_ranker_from_results(
        self,
        run: OptimizerRun,
        rankings: list[dict[str, Any]],
        optimizer_config: OptimizerConfig,
    ) -> None:
        if not bool(self.config.get("ML_RANKER_ENABLED", False)):
            return
        horizon = horizon_from_duration(optimizer_config.lock_duration_hours or 0)
        for ranking in rankings:
            if int(ranking.get("window_count", 0) or 0) <= 0:
                continue
            features = extract_features(
                {
                    **ranking,
                    "optimizer_profile": ranking.get("profile"),
                    "lock_duration_hours": optimizer_config.lock_duration_hours,
                    "horizon": horizon,
                    "initial_balance": optimizer_config.initial_balance,
                }
            )
            self.online_ranker.update(
                features,
                outcome_from_result(ranking),
                horizon=horizon,
                source="optimizer",
                source_id=run.id,
                mode=optimizer_config.mode,
                metadata={
                    "strategy_name": ranking.get("strategy_name"),
                    "symbol": ranking.get("symbol"),
                    "timeframe": ranking.get("timeframe"),
                    "profile": ranking.get("profile"),
                    "rejected": bool(ranking.get("rejected", False)),
                    "rejection_reason": ranking.get("rejection_reason", ""),
                },
            )

    def _fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        optimizer_config: OptimizerConfig,
    ) -> List[Dict[str, Any]]:
        """Retrieve candle data sufficient to cover all train and test windows."""
        total_hours = self._duration_hours(optimizer_config, include_buffer=True)
        if optimizer_config.training_window_hours is not None or optimizer_config.testing_window_hours is not None:
            days = max(total_hours / 24, 1)
            limit = int(days * self._candles_per_day(timeframe))
            return self.market_data.get_candles(
                symbol, timeframe, mode=optimizer_config.mode, limit=max(limit, 250)
            )

        if optimizer_config.use_full_history:
            # Fetch at least 90 days of data or enough to cover the windows plus some buffer.
            days = max(
                optimizer_config.training_window_days
                + optimizer_config.testing_window_days
                + 30,
                90,
            )
        else:
            # Fetch just enough data for all windows plus a small buffer.
            days = (
                optimizer_config.training_window_days
                + optimizer_config.testing_window_days
                + max(7, optimizer_config.step_days * 8)
            )
        limit = int(days * self._candles_per_day(timeframe))
        return self.market_data.get_candles(
            symbol, timeframe, mode=optimizer_config.mode, limit=max(limit, 250)
        )

    def _rolling_windows(
        self,
        candles: List[Dict[str, Any]],
        optimizer_config: OptimizerConfig,
    ) -> List[Tuple[int, int, int]]:
        """Generate a list of rolling (train_start, test_start, test_end) timestamp tuples."""
        if len(candles) < 30:
            return []
        start_dt = self._to_datetime(int(candles[0]["timestamp"]))
        end_dt = self._to_datetime(int(candles[-1]["timestamp"]))
        train, test, step = self._window_durations(optimizer_config)
        windows: List[Tuple[int, int, int]] = []
        cursor = start_dt
        # Slide the window until the end of the dataset is reached.
        while cursor + train + test <= end_dt:
            train_start = cursor
            test_start = cursor + train
            test_end = test_start + test
            windows.append(
                (
                    self._to_timestamp(train_start),
                    self._to_timestamp(test_start),
                    self._to_timestamp(test_end),
                )
            )
            cursor += step
        # If no windows were generated (e.g. too little data), create a single window covering 70% of the range.
        if not windows and not self._is_hourly_experimental(optimizer_config.profile):
            midpoint = start_dt + ((end_dt - start_dt) * 0.7)
            windows.append(
                (
                    self._to_timestamp(start_dt),
                    self._to_timestamp(midpoint),
                    self._to_timestamp(end_dt),
                )
            )
        return windows

    # ------------------------------------------------------------------
    # Parameter grid handling
    #
    def _window_durations(self, optimizer_config: OptimizerConfig) -> tuple[timedelta, timedelta, timedelta]:
        if optimizer_config.training_window_hours is not None or optimizer_config.testing_window_hours is not None:
            return (
                timedelta(hours=max(int(optimizer_config.training_window_hours or 72), 1)),
                timedelta(hours=max(int(optimizer_config.testing_window_hours or 1), 1)),
                timedelta(hours=max(int(optimizer_config.step_hours or 1), 1)),
            )
        return (
            timedelta(days=max(int(optimizer_config.training_window_days), 1)),
            timedelta(days=max(int(optimizer_config.testing_window_days), 1)),
            timedelta(days=max(int(optimizer_config.step_days), 1)),
        )

    def _duration_hours(self, optimizer_config: OptimizerConfig, *, include_buffer: bool = False) -> float:
        train, test, step = self._window_durations(optimizer_config)
        buffer_hours = max(7 * 24, step.total_seconds() / 3600 * 8) if include_buffer else 0
        if optimizer_config.use_full_history:
            buffer_hours = max(buffer_hours, 30 * 24)
        return (train + test).total_seconds() / 3600 + buffer_hours

    def _risk_per_trade_pct(self, optimizer_config: OptimizerConfig) -> float:
        if optimizer_config.profile == Profile.EXTREME_ROI_EXPERIMENTAL.value:
            return float(self.config.get("EXTREME_ROI_RISK_PER_TRADE_PCT", 0.006))
        if optimizer_config.profile == Profile.AGGRESSIVE_1H.value:
            return float(self.config.get("AGGRESSIVE_1H_RISK_PER_TRADE_PCT", 0.005))
        if optimizer_config.profile == Profile.DYNAMIC_INTRADAY.value:
            return min(float(self.config.get("RISK_PER_TRADE_PCT", 0.01)), 0.006)
        if optimizer_config.profile == Profile.AGGRESSIVE_RISK_ADJUSTED.value:
            duration = max(int(optimizer_config.lock_duration_hours or 24), 1)
            if duration <= 1:
                return min(float(self.config.get("RISK_PER_TRADE_PCT", 0.01)), 0.006)
            if duration >= 168:
                return min(float(self.config.get("RISK_PER_TRADE_PCT", 0.01)), 0.008)
        return float(self.config.get("RISK_PER_TRADE_PCT", 0.01))

    def _position_size_fraction(self, parameters: Dict[str, Any], optimizer_config: OptimizerConfig) -> float:
        if optimizer_config.profile == Profile.EXTREME_ROI_EXPERIMENTAL.value:
            return float(
                parameters.get(
                    "risk_fraction",
                    self.config.get("EXTREME_ROI_POSITION_SIZE_FRACTION", 0.16),
                )
            )
        if optimizer_config.profile == Profile.AGGRESSIVE_1H.value:
            return float(
                parameters.get(
                    "risk_fraction",
                    self.config.get("AGGRESSIVE_1H_POSITION_SIZE_FRACTION", 0.12),
                )
            )
        if optimizer_config.profile == Profile.DYNAMIC_INTRADAY.value:
            return min(float(parameters.get("risk_fraction", 0.10)), 0.10)
        if optimizer_config.profile == Profile.AGGRESSIVE_RISK_ADJUSTED.value:
            duration = max(int(optimizer_config.lock_duration_hours or 24), 1)
            cap = 0.10 if duration <= 1 else 0.18 if duration <= 48 else 0.14
            return min(float(parameters.get("risk_fraction", cap)), cap)
        return float(parameters.get("risk_fraction", 0.08))

    def _parameter_sets(
        self,
        strategy_name: str,
        max_parameter_sets: int,
        optimizer_config: OptimizerConfig | None = None,
    ) -> List[Dict[str, Any]]:
        """Generate a list of candidate parameter dictionaries for a strategy."""
        # Start with the strategy’s default parameters.
        defaults: Dict[str, Any] = self.registry.build(strategy_name, {}).parameters
        grid = self.registry.parameter_grid(strategy_name)
        keys = list(grid)
        candidates: List[Dict[str, Any]] = [defaults]
        for values in product(*[grid[key] for key in keys]):
            candidate = dict(zip(keys, values))
            if candidate not in candidates:
                candidates.append(candidate)
            if len(candidates) >= max_parameter_sets:
                break
        if optimizer_config is not None and optimizer_config.max_return_optimizer_enabled:
            candidates = self._max_return_parameter_sets(strategy_name, candidates, max_parameter_sets, optimizer_config)
        if optimizer_config is not None and (
            optimizer_config.profile == Profile.AGGRESSIVE_1H.value
            or (
                optimizer_config.profile == Profile.AGGRESSIVE_RISK_ADJUSTED.value
                and max(int(optimizer_config.lock_duration_hours or 24), 1) <= 1
            )
        ):
            candidates = self._aggressive_1h_parameter_sets(strategy_name, candidates, max_parameter_sets)
        if optimizer_config is not None and optimizer_config.allow_leverage_experiment:
            if optimizer_config.profile == Profile.EXTREME_ROI_EXPERIMENTAL.value:
                candidates = self._extreme_roi_parameter_sets(strategy_name, candidates, max_parameter_sets)
            leveraged: list[Dict[str, Any]] = []
            for candidate in candidates:
                for leverage in self._leverage_values(optimizer_config):
                    item = dict(candidate)
                    item["leverage"] = leverage
                    if item not in leveraged:
                        leveraged.append(item)
                    if len(leveraged) >= max_parameter_sets:
                        return leveraged
            return leveraged

        return candidates

    def _max_return_parameter_sets(
        self,
        strategy_name: str,
        candidates: list[dict[str, Any]],
        max_parameter_sets: int,
        optimizer_config: OptimizerConfig,
    ) -> list[dict[str, Any]]:
        duration = max(int(optimizer_config.lock_duration_hours or optimizer_config.testing_window_hours or 24), 1)
        if duration <= 1:
            stop_values = (0.003, 0.005, 0.007)
            reward_multipliers = (1.4, 1.8, 2.2)
        elif duration <= 24:
            stop_values = (0.006, 0.009, 0.012)
            reward_multipliers = (1.6, 2.0, 2.6)
        elif duration <= 72:
            stop_values = (0.008, 0.012, 0.018)
            reward_multipliers = (1.8, 2.4, 3.0)
        else:
            stop_values = (0.012, 0.018, 0.026)
            reward_multipliers = (2.0, 2.8, 3.6)

        expanded: list[dict[str, Any]] = []
        for candidate in candidates:
            if candidate not in expanded:
                expanded.append(candidate)
            for stop_loss in stop_values:
                for reward_multiplier in reward_multipliers:
                    item = dict(candidate)
                    item["stop_loss_pct"] = stop_loss
                    item["take_profit_pct"] = round(stop_loss * reward_multiplier, 6)
                    if strategy_name in {"volatility_breakout", "breakout"}:
                        item["range_multiplier"] = item.get("range_multiplier", 1.4 if duration <= 24 else 1.8)
                    if item not in expanded:
                        expanded.append(item)
                    if len(expanded) >= max_parameter_sets:
                        return expanded
        return expanded[:max_parameter_sets]

    def _horizon_hours(self, lock_duration_hours: int) -> tuple[int, int, int]:
        duration = max(1, int(lock_duration_hours or 1))
        if duration <= 1:
            return 72, 1, 1
        if duration <= 4:
            return 120, 4, 1
        if duration <= 24:
            return 336, 24, 4
        if duration <= 48:
            return 720, 48, 8
        if duration <= 168:
            return 1_440, 168, 24
        return 1_440, min(duration, 336), 24

    def _optimizer_symbols(self, optimizer_config: OptimizerConfig) -> list[str]:
        if (
            optimizer_config.universe_mode == "dynamic_liquid"
            and self.universe_service is not None
            and bool(self.config.get("DYNAMIC_UNIVERSE_ENABLED", False))
        ):
            timeframe = optimizer_config.timeframes[0] if optimizer_config.timeframes else "5m"
            symbols = self.universe_service.symbols(optimizer_config.mode, timeframe)
            if symbols:
                return symbols
        return list(optimizer_config.symbols)

    def _candidate_leverage(self, parameters: Dict[str, Any], optimizer_config: OptimizerConfig) -> float:
        requested = float(parameters.get("leverage", 1.0) or 1.0)
        if not optimizer_config.allow_leverage_experiment:
            return 1.0
        max_leverage = float(self.config.get("AGGRESSIVE_MAX_TEST_LEVERAGE", 5.0))
        return max(1.0, min(requested, max_leverage, float(self.config.get("MAX_LEVERAGE", max_leverage))))

    def _leverage_values(self, optimizer_config: OptimizerConfig) -> list[float]:
        if not bool(self.config.get("LEVERAGE_OPTIMIZER_ENABLED", False)) and optimizer_config.profile != Profile.EXTREME_ROI_EXPERIMENTAL.value:
            return [1.0]
        max_leverage = min(
            float(self.config.get("AGGRESSIVE_MAX_TEST_LEVERAGE", 5.0)),
            float(self.config.get("MAX_LEVERAGE", 5.0)),
        )
        if max_leverage <= 1:
            return [1.0]
        midpoint = max(1.0, round((1.0 + max_leverage) / 2, 2))
        return sorted({1.0, midpoint, round(max_leverage, 2)})

    def _extreme_roi_parameter_sets(
        self,
        strategy_name: str,
        candidates: list[dict[str, Any]],
        max_parameter_sets: int,
    ) -> list[dict[str, Any]]:
        overlays_by_strategy: dict[str, list[dict[str, Any]]] = {
            "scalping": [
                {"momentum_lookback": 3, "minimum_move_pct": 0.0012, "stop_loss_pct": 0.0025, "take_profit_pct": 0.0075, "trailing_stop_pct": 0.0015, "breakeven_trigger_pct": 0.002},
                {"momentum_lookback": 5, "minimum_move_pct": 0.0018, "stop_loss_pct": 0.003, "take_profit_pct": 0.010, "fast_fade_exit_pct": 0.0006},
            ],
            "volatility_breakout": [
                {"lookback": 12, "range_multiplier": 1.15, "stop_loss_pct": 0.004, "take_profit_pct": 0.014},
                {"lookback": 18, "range_multiplier": 1.35, "stop_loss_pct": 0.005, "take_profit_pct": 0.018},
            ],
            "rule_based_signal": [
                {"minimum_signal_score": 0.78, "fibonacci_filter_weight": 0.15, "pattern_weight": 0.10, "atr_stop_multiplier": 1.1, "atr_take_multiplier": 3.0},
                {"minimum_signal_score": 0.84, "fibonacci_filter_weight": 0.20, "volume_weight": 0.2, "atr_stop_multiplier": 1.0, "atr_take_multiplier": 3.5},
            ],
            "breakout": [
                {"lookback": 14, "breakout_buffer_pct": 0.001, "stop_loss_pct": 0.004, "take_profit_pct": 0.015},
                {"lookback": 20, "breakout_buffer_pct": 0.0015, "stop_loss_pct": 0.005, "take_profit_pct": 0.020},
            ],
            "ema_crossover": [
                {"fast_period": 5, "slow_period": 13, "stop_loss_pct": 0.0035, "take_profit_pct": 0.012},
                {"fast_period": 8, "slow_period": 21, "stop_loss_pct": 0.0045, "take_profit_pct": 0.016},
            ],
        }
        overlays = overlays_by_strategy.get(strategy_name, [])
        expanded = list(candidates)
        for base in candidates:
            for overlay in overlays:
                item = dict(base)
                item.update(overlay)
                item.setdefault("risk_fraction", self.config.get("EXTREME_ROI_POSITION_SIZE_FRACTION", 0.16))
                item["extreme_roi"] = True
                if item not in expanded:
                    expanded.append(item)
                if len(expanded) >= max_parameter_sets:
                    return expanded
        return expanded[:max_parameter_sets]

    def _aggressive_1h_parameter_sets(
        self,
        strategy_name: str,
        candidates: list[dict[str, Any]],
        max_parameter_sets: int,
    ) -> list[dict[str, Any]]:
        overlays_by_strategy: dict[str, list[dict[str, Any]]] = {
            "scalping": [
                {"momentum_lookback": 3, "minimum_move_pct": 0.0012, "stop_loss_pct": 0.0025, "take_profit_pct": 0.0065, "trailing_stop_pct": 0.0015, "breakeven_trigger_pct": 0.002},
                {"momentum_lookback": 5, "minimum_move_pct": 0.0018, "stop_loss_pct": 0.0035, "take_profit_pct": 0.009, "fast_fade_exit_pct": 0.0008},
            ],
            "volatility_breakout": [
                {"lookback": 10, "range_multiplier": 1.15, "stop_loss_pct": 0.004, "take_profit_pct": 0.012, "fade_exit_ratio": 0.20},
                {"lookback": 18, "range_multiplier": 1.35, "stop_loss_pct": 0.0055, "take_profit_pct": 0.0165, "fade_exit_ratio": 0.25},
            ],
            "breakout": [
                {"lookback": 12, "confirmation_buffer_pct": 0.001, "stop_loss_pct": 0.004, "take_profit_pct": 0.012, "exit_buffer_pct": 0.0005},
                {"lookback": 20, "confirmation_buffer_pct": 0.0015, "stop_loss_pct": 0.0055, "take_profit_pct": 0.017, "exit_buffer_pct": 0.001},
            ],
            "rsi_mean_reversion": [
                {"period": 5, "oversold": 24, "overbought": 76, "exit_rsi": 50, "stop_loss_pct": 0.004, "take_profit_pct": 0.0085, "reentry_buffer": 2},
                {"period": 7, "oversold": 28, "overbought": 72, "exit_rsi": 50, "stop_loss_pct": 0.005, "take_profit_pct": 0.011, "reentry_buffer": 3},
            ],
            "ema_crossover": [
                {"fast_period": 5, "slow_period": 13, "stop_loss_pct": 0.0035, "take_profit_pct": 0.0105, "compression_exit_pct": 0.00005},
                {"fast_period": 8, "slow_period": 21, "stop_loss_pct": 0.0045, "take_profit_pct": 0.014, "compression_exit_pct": 0.0001},
            ],
            "rule_based_signal": [
                {"minimum_signal_score": 0.78, "fibonacci_filter_weight": 0.15, "volume_weight": 0.2, "atr_stop_multiplier": 1.05, "atr_take_multiplier": 2.8, "fallback_stop_loss_pct": 0.0045, "fallback_take_profit_pct": 0.013},
                {"minimum_signal_score": 0.84, "fibonacci_filter_weight": 0.20, "trend_weight": 0.50, "rsi_weight": 0.25, "atr_stop_multiplier": 1.1, "atr_take_multiplier": 3.2, "fallback_stop_loss_pct": 0.005, "fallback_take_profit_pct": 0.016},
            ],
        }
        overlays = overlays_by_strategy.get(strategy_name, [])
        expanded: list[dict[str, Any]] = []
        for base in candidates:
            for overlay in overlays:
                item = dict(base)
                item.update(overlay)
                item.setdefault("risk_fraction", self.config.get("AGGRESSIVE_1H_POSITION_SIZE_FRACTION", 0.12))
                if item not in expanded:
                    expanded.append(item)
                if len(expanded) >= max_parameter_sets:
                    return expanded
        for candidate in candidates:
            if candidate not in expanded:
                expanded.append(candidate)
            if len(expanded) >= max_parameter_sets:
                return expanded
        return expanded

    def _capacity_usd(self, symbol: str, optimizer_config: OptimizerConfig) -> float:
        if self.universe_service is None:
            return float(optimizer_config.allocation_amount_usd or 0.0)
        timeframe = optimizer_config.timeframes[0] if optimizer_config.timeframes else "5m"
        for candidate in self.universe_service.liquid_universe(optimizer_config.mode, timeframe):
            if candidate.symbol == symbol:
                return candidate.liquidity_usd * 0.05
        return float(optimizer_config.allocation_amount_usd or 0.0)

    def _universe_source(self, symbol: str, optimizer_config: OptimizerConfig) -> str:
        if self.universe_service is None:
            return "configured"
        timeframe = optimizer_config.timeframes[0] if optimizer_config.timeframes else "5m"
        for candidate in self.universe_service.liquid_universe(optimizer_config.mode, timeframe):
            if candidate.symbol == symbol:
                return candidate.source
        return "configured"

    @staticmethod
    def _execution_style(edge_score: float, cost_drag_bps: float) -> str:
        return "maker_limit" if edge_score > max(cost_drag_bps * 2, 12.0) else "market"

    # ------------------------------------------------------------------
    # Persistence helpers
    #
    def _persist_ranking(self, run: OptimizerRun, result: Dict[str, Any]) -> None:
        """Persist a single ranking result to the database."""
        ranking = StrategyRanking(
            optimizer_run_id=run.id,
            strategy_name=result["strategy_name"],
            symbol=result["symbol"],
            timeframe=result["timeframe"],
            profile=result.get("profile") or run.profile,
            experimental=bool(result.get("experimental", False)),
            risk_label=result.get("risk_label") or "",
            score=result["score"],
            total_return=result["total_return"],
            net_return_after_costs=result.get("net_return_after_costs", result["total_return"]),
            recent_performance_score=result["recent_performance_score"],
            recent_1h_return=result.get("recent_1h_return", 0.0),
            estimated_fees=result.get("estimated_fees", 0.0),
            edge_score=result.get("edge_score", 0.0),
            expectancy=result.get("expectancy", 0.0),
            avg_win=result.get("avg_win", 0.0),
            avg_loss=result.get("avg_loss", 0.0),
            win_loss_ratio=result.get("win_loss_ratio", 0.0),
            cost_drag_bps=result.get("cost_drag_bps", 0.0),
            convex_edge_score=result.get("convex_edge_score", 0.0),
            mfe_mae_ratio=result.get("mfe_mae_ratio", 0.0),
            capacity_multiple=result.get("capacity_multiple", 0.0),
            cost_adjusted_recent_1h_return=result.get("cost_adjusted_recent_1h_return", 0.0),
            decay_penalty=result.get("decay_penalty", 0.0),
            max_adverse_excursion=result.get("max_adverse_excursion", 0.0),
            max_favorable_excursion=result.get("max_favorable_excursion", 0.0),
            no_trade_reason=result.get("no_trade_reason") or None,
            allocation_amount_usd=result.get("allocation_amount_usd", 0.0),
            lock_duration_hours=result.get("lock_duration_hours", 0),
            leverage=result.get("leverage", 1.0),
            liquidation_buffer_pct=result.get("liquidation_buffer_pct", 1.0),
            capacity_usd=result.get("capacity_usd", 0.0),
            universe_source=result.get("universe_source") or "configured",
            vault_leg_count=result.get("vault_leg_count", 1),
            execution_style=result.get("execution_style") or "market",
            funding_cost_estimate=result.get("funding_cost_estimate", 0.0),
            max_drawdown=result["max_drawdown"],
            profit_factor=result["profit_factor"],
            sharpe_like=result["sharpe_like"],
            sortino_like=result["sortino_like"],
            trades_per_day=result["trades_per_day"],
            avg_trade_return=result["avg_trade_return"],
            turnover_rate=result["turnover_rate"],
            turnover_after_fees=result.get("turnover_after_fees", result["turnover_rate"]),
            consistency=result["consistency"],
            window_stability=result.get("window_stability", 0.0),
            accepted_window_ratio=result.get("accepted_window_ratio", result["consistency"]),
            win_rate=result.get("win_rate", 0.0),
            trade_count=result.get("trade_count", 0),
            ml_score=result.get("ml_score", 0.0),
            ml_adjusted_score=result.get("ml_adjusted_score", result["score"]),
            ml_warmup=bool(result.get("ml_warmup", True)),
            rejected=result["rejected"],
            rejection_reason=result["rejection_reason"],
        )
        ranking.parameters = result["parameters"]
        ranking.warnings = result["warnings"]
        ranking.ml_explanation = {
            **dict(result.get("ml_explanation", {}) or {}),
            "net_roi": {
                "net_roi_score": result.get("net_roi_score", 0.0),
                "expected_fill_quality": result.get("expected_fill_quality", 0.0),
                "churn_penalty": result.get("churn_penalty", 0.0),
                "edge_after_cost_bps": result.get("edge_after_cost_bps", 0.0),
                "data_age_seconds": result.get("data_age_seconds", 0.0),
                "components": result.get("net_roi_components", {}),
            },
            "net_roi_v2": {
                "net_roi_v2_score": result.get("net_roi_v2_score", 0.0),
                "roi_quality_grade": result.get("roi_quality_grade", "D"),
                "roi_rejection_risk": result.get("roi_rejection_risk", "high"),
                "regime_bucket": result.get("regime_bucket", {}),
                "regime_support": result.get("regime_support", "regime-neutral"),
                "regime_adjustment": result.get("regime_adjustment", 0.0),
                "regime_adjusted_expectancy": result.get("regime_adjusted_expectancy", 0.0),
                "tail_loss_penalty": result.get("tail_loss_penalty", 0.0),
                "downside_asymmetry_penalty": result.get("downside_asymmetry_penalty", 0.0),
                "cost_adjusted_breakout_potential": result.get("cost_adjusted_breakout_potential", 0.0),
                "components": result.get("net_roi_v2_components", {}),
            },
            "raw_upside": {
                "raw_upside_score": result.get("raw_upside_score", 0.0),
                "raw_total_return_pct": result.get("raw_total_return_pct", 0.0),
                "raw_net_return_pct": result.get("raw_net_return_pct", 0.0),
                "target_roi_pct": result.get("target_roi_pct", 1000.0),
                "target_roi_hit": bool(result.get("target_roi_hit", False)),
                "live_blockers": result.get("live_blockers", []),
            },
        }
        db.session.add(ranking)
        # Flush rather than commit to keep the transaction open; commit happens in run().
        db.session.flush()

    def _persist_backtest_validations(self, rankings: List[Dict[str, Any]]) -> None:
        """Persist backtest validation records for the top N non‑rejected rankings."""
        for ranking in rankings:
            if ranking["rejected"]:
                continue
            validation = StrategyValidation(
                strategy_name=ranking["strategy_name"],
                symbol=ranking["symbol"],
                timeframe=ranking["timeframe"],
                stage="backtest",
                status="passed",
                completed_at=datetime.utcnow(),
            )
            validation.parameters = ranking["parameters"]
            validation.metrics = {
                "score": ranking["score"],
                "total_return": ranking["total_return"],
                "net_return_after_costs": ranking.get("net_return_after_costs", ranking["total_return"]),
                "recent_performance_score": ranking["recent_performance_score"],
                "max_drawdown": ranking["max_drawdown"],
                "profit_factor": ranking["profit_factor"],
                "trade_count": ranking["trade_count"],
                "profile": ranking.get("profile", ""),
                "experimental": ranking.get("experimental", False),
                "risk_label": ranking.get("risk_label", ""),
                "recent_1h_return": ranking.get("recent_1h_return", 0.0),
                "estimated_fees": ranking.get("estimated_fees", 0.0),
                "edge_score": ranking.get("edge_score", 0.0),
                "expectancy": ranking.get("expectancy", 0.0),
                "cost_drag_bps": ranking.get("cost_drag_bps", 0.0),
                "net_roi_score": ranking.get("net_roi_score", 0.0),
                "net_roi_v2_score": ranking.get("net_roi_v2_score", 0.0),
                "roi_quality_grade": ranking.get("roi_quality_grade", "D"),
                "roi_rejection_risk": ranking.get("roi_rejection_risk", "high"),
                "regime_bucket": ranking.get("regime_bucket", {}),
                "regime_support": ranking.get("regime_support", "regime-neutral"),
                "tail_loss_penalty": ranking.get("tail_loss_penalty", 0.0),
                "downside_asymmetry_penalty": ranking.get("downside_asymmetry_penalty", 0.0),
                "cost_adjusted_breakout_potential": ranking.get("cost_adjusted_breakout_potential", 0.0),
                "expected_fill_quality": ranking.get("expected_fill_quality", 0.0),
                "churn_penalty": ranking.get("churn_penalty", 0.0),
                "edge_after_cost_bps": ranking.get("edge_after_cost_bps", 0.0),
                "data_age_seconds": ranking.get("data_age_seconds", 0.0),
                "net_roi_components": ranking.get("net_roi_components", {}),
                "net_roi_v2_components": ranking.get("net_roi_v2_components", {}),
                "convex_edge_score": ranking.get("convex_edge_score", 0.0),
                "mfe_mae_ratio": ranking.get("mfe_mae_ratio", 0.0),
                "capacity_multiple": ranking.get("capacity_multiple", 0.0),
                "cost_adjusted_recent_1h_return": ranking.get("cost_adjusted_recent_1h_return", 0.0),
                "decay_penalty": ranking.get("decay_penalty", 0.0),
                "turnover_after_fees": ranking.get("turnover_after_fees", 0.0),
                "window_stability": ranking.get("window_stability", 0.0),
                "accepted_window_ratio": ranking.get("accepted_window_ratio", 0.0),
                "win_rate": ranking.get("win_rate", 0.0),
                "no_trade_reason": ranking.get("no_trade_reason", ""),
                "allocation_amount_usd": ranking.get("allocation_amount_usd", 0.0),
                "lock_duration_hours": ranking.get("lock_duration_hours", 0),
                "leverage": ranking.get("leverage", 1.0),
                "liquidation_buffer_pct": ranking.get("liquidation_buffer_pct", 1.0),
                "capacity_usd": ranking.get("capacity_usd", 0.0),
                "universe_source": ranking.get("universe_source", "configured"),
                "execution_style": ranking.get("execution_style", "market"),
                "profit_objective_version": ranking.get("profit_objective_version", ""),
                "high_upside_profile": bool(ranking.get("high_upside_profile", False)),
                "market_structure": ranking.get("market_structure", {}),
                "market_structure_score": ranking.get("market_structure_score", 0.0),
                "funding_cost_estimate": ranking.get("funding_cost_estimate", 0.0),
                "ml_score": ranking.get("ml_score", 0.0),
                "ml_adjusted_score": ranking.get("ml_adjusted_score", ranking.get("score", 0.0)),
                "ml_warmup": ranking.get("ml_warmup", True),
            }
            db.session.add(validation)

    def _ensemble_backtest_summary(
        self,
        rankings: list[dict[str, Any]],
        optimizer_config: OptimizerConfig,
    ) -> dict[str, Any]:
        if not optimizer_config.enhanced_ensemble_enabled:
            return {"enabled": False}
        accepted = [
            item
            for item in rankings
            if not item.get("rejected")
            and float(item.get("edge_score", 0.0) or 0.0) >= float(optimizer_config.min_edge_bps or 0.0)
            and max(float(item.get("sharpe_like", 0.0) or 0.0), float(item.get("sortino_like", 0.0) or 0.0)) >= float(optimizer_config.ensemble_min_sharpe)
        ]
        accepted.sort(
            key=lambda item: (
                float(item.get("score", 0.0) or 0.0),
                float(item.get("recent_1h_return", 0.0) or 0.0),
                float(item.get("edge_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        selected = accepted[: max(2, min(int(optimizer_config.ensemble_max_legs or 5), 5))]
        if len(selected) < 2:
            return {
                "enabled": True,
                "accepted": False,
                "skip_reason": "not_enough_accepted_ensemble_candidates",
                "candidate_count": len(accepted),
            }
        weights = [max(float(item.get("score", 0.0) or 0.0), float(item.get("edge_score", 0.0) or 0.0), 0.01) for item in selected]
        total_weight = sum(weights) or 1.0
        normalized = [weight / total_weight for weight in weights]

        def weighted(key: str) -> float:
            return sum(float(item.get(key, 0.0) or 0.0) * weight for item, weight in zip(selected, normalized))

        baseline = max(accepted, key=lambda item: float(item.get("score", 0.0) or 0.0)) if accepted else {}
        ensemble_return = weighted("net_return_after_costs")
        baseline_return = float(baseline.get("net_return_after_costs", 0.0) or 0.0)
        return {
            "enabled": True,
            "accepted": True,
            "selected_legs": [
                {
                    "strategy_name": item.get("strategy_name"),
                    "symbol": item.get("symbol"),
                    "timeframe": item.get("timeframe"),
                    "weight": round(weight, 6),
                    "score": item.get("score"),
                    "edge_score": item.get("edge_score"),
                }
                for item, weight in zip(selected, normalized)
            ],
            "net_return_after_costs": ensemble_return,
            "sharpe_like": weighted("sharpe_like"),
            "sortino_like": weighted("sortino_like"),
            "max_drawdown": weighted("max_drawdown"),
            "win_rate": weighted("win_rate"),
            "expectancy": weighted("expectancy"),
            "cost_drag_bps": weighted("cost_drag_bps"),
            "baseline_best": {
                "strategy_name": baseline.get("strategy_name"),
                "symbol": baseline.get("symbol"),
                "net_return_after_costs": baseline_return,
                "sharpe_like": baseline.get("sharpe_like", 0.0),
                "max_drawdown": baseline.get("max_drawdown", 0.0),
            },
            "improvement_vs_baseline_return": ensemble_return - baseline_return,
        }

    def _duration_ensemble_backtest_summary(
        self,
        rankings: list[dict[str, Any]],
        optimizer_config: OptimizerConfig,
    ) -> dict[str, Any]:
        if not optimizer_config.experimental_duration_ensemble_enabled:
            return {"enabled": False}

        duration = int(optimizer_config.lock_duration_hours or optimizer_config.testing_window_hours or 1)
        bucket = EnhancedEnsembleAllocator.duration_bucket(duration)
        library = EnhancedEnsembleAllocator.strategy_library(duration)
        rejected_reasons: dict[str, int] = {}
        for item in rankings:
            if item.get("rejected"):
                reason = str(item.get("rejection_reason") or item.get("no_trade_reason") or "rejected")
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

        accepted = [
            item
            for item in rankings
            if not item.get("rejected")
            and str(item.get("strategy_name") or "") in library
            and not str(item.get("no_trade_reason") or "").strip()
            and float(item.get("net_return_after_costs", 0.0) or 0.0) > 0
            and float(item.get("edge_score", 0.0) or 0.0) >= float(optimizer_config.min_edge_bps or 0.0)
            and max(float(item.get("sharpe_like", 0.0) or 0.0), float(item.get("sortino_like", 0.0) or 0.0)) >= float(optimizer_config.ensemble_min_sharpe)
            and int(item.get("trade_count", 0) or 0) >= int(optimizer_config.min_trade_count or 0)
        ]
        accepted.sort(
            key=lambda item: (
                float(item.get("net_return_after_costs", 0.0) or 0.0),
                max(float(item.get("recent_performance_score", 0.0) or 0.0), float(item.get("recent_1h_return", 0.0) or 0.0)),
                float(item.get("expectancy", 0.0) or 0.0),
                float(item.get("score", 0.0) or 0.0),
                float(item.get("edge_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        baseline_single = accepted[0] if accepted else {}
        baseline_current_basket = self._basket_baseline_summary(
            sorted(accepted, key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)[
                : max(2, min(int(optimizer_config.ensemble_max_legs or 5), 5))
            ]
        )
        if len(accepted) < 2:
            return {
                "enabled": True,
                "accepted": False,
                "duration_bucket": bucket,
                "skip_reason": "not_enough_accepted_duration_ensemble_candidates",
                "candidate_count": len(accepted),
                "baseline_single_strategy": self._baseline_payload(baseline_single),
                "baseline_current_basket": baseline_current_basket,
                "overfit_rejections": rejected_reasons,
                "cap_rejections": {"cap_blocked_count": 0},
                "allocation_conservation": {},
            }

        allocator_config = dict(self.config)
        allocator_config["EXPERIMENTAL_DURATION_ENSEMBLE_MAX_LEGS"] = max(2, min(int(optimizer_config.ensemble_max_legs or 5), 5))
        allocator_config["ENSEMBLE_MIN_EDGE_BPS"] = float(optimizer_config.min_edge_bps or 0.0)
        allocator_config["ENSEMBLE_MIN_SHARPE"] = float(optimizer_config.ensemble_min_sharpe or 0.0)
        allocator = EnhancedEnsembleAllocator(allocator_config, self.online_ranker)
        metadata = {
            "duration_bucket": bucket,
            "duration_ensemble_enabled": True,
            "ensemble_primary_metric": optimizer_config.ensemble_primary_metric,
            "ensemble_id": f"optimizer-duration-{bucket}",
            "multi_timeframe_confluence": {"score": 1.0, "passed": True},
        }
        if optimizer_config.max_return_optimizer_enabled:
            metadata["profit_objective_version"] = "max_return_v3"
        market_structure_summary = self._aggregate_market_structure(accepted)
        if market_structure_summary:
            metadata["market_structure"] = market_structure_summary
            metadata["market_structure_score"] = float(market_structure_summary.get("score", 0.0) or 0.0)
        namespaces = [self._ranking_namespace(item, index, optimizer_config) for index, item in enumerate(accepted)]
        ranked, skipped = allocator.rank(namespaces, duration_hours=duration, metadata=metadata)
        allocation_amount = float(optimizer_config.allocation_amount_usd or optimizer_config.initial_balance or 0.0)
        min_leg_usd = 0.0 if allocation_amount <= 0 else min(float(self.config.get("VAULT_MIN_LEG_USD", 10.0)), allocation_amount / 2)
        legs = allocator.allocate(
            ranked,
            base_parameters={},
            allocation_amount_usd=allocation_amount,
            profile="Aggressive",
            metadata=metadata,
            adaptive_leverage=lambda params, _profile, _volatility, _mode: float(params.get("leverage", 1.0) or 1.0),
            min_leg_usd=min_leg_usd,
            mode="paper",
        )
        if len(legs) < 2:
            return {
                "enabled": True,
                "accepted": False,
                "duration_bucket": bucket,
                "skip_reason": "duration_ensemble_caps_prevented_allocation",
                "candidate_count": len(accepted),
                "skipped": skipped,
                "baseline_single_strategy": self._baseline_payload(baseline_single),
                "baseline_current_basket": baseline_current_basket,
                "overfit_rejections": rejected_reasons,
                "cap_rejections": {"cap_blocked_count": int(metadata.get("cap_blocked_count", 0) or 0)},
                "allocation_conservation": metadata.get("allocation_conservation", {}),
            }

        by_id = {int(item.id): item for item in namespaces}

        def weighted(key: str) -> float:
            return sum(
                float(getattr(by_id.get(int(leg.get("optimizer_ranking_id") or -1)), key, 0.0) or 0.0)
                * float(leg.get("effective_allocation_weight", 0.0) or 0.0)
                for leg in legs
            )

        ensemble_return = weighted("net_return_after_costs")
        baseline_return = float(baseline_single.get("net_return_after_costs", 0.0) or 0.0)
        return {
            "enabled": True,
            "accepted": True,
            "duration_bucket": bucket,
            "primary_metric": optimizer_config.ensemble_primary_metric,
            "selected_legs": [
                {
                    "strategy_name": leg.get("strategy_name"),
                    "symbol": leg.get("symbol"),
                    "timeframe": leg.get("timeframe"),
                    "target_ensemble_weight": leg.get("target_ensemble_weight"),
                    "effective_allocation_weight": leg.get("effective_allocation_weight"),
                    "allocation_cap_usd": leg.get("allocation_cap_usd"),
                    "cap_limited": leg.get("cap_limited"),
                    "cap_limit_reason": leg.get("cap_limit_reason"),
                    "profit_objective_version": leg.get("profit_objective_version", ""),
                    "market_structure_score": leg.get("market_structure_score", 0.0),
                }
                for leg in legs
            ],
            "net_return_after_costs": ensemble_return,
            "sharpe_like": weighted("sharpe_like"),
            "sortino_like": weighted("sortino_like"),
            "max_drawdown": weighted("max_drawdown"),
            "win_rate": weighted("win_rate"),
            "expectancy": weighted("expectancy"),
            "cost_drag_bps": weighted("cost_drag_bps"),
            "baseline_single_strategy": self._baseline_payload(baseline_single),
            "baseline_current_basket": baseline_current_basket,
            "improvement_vs_baseline_return": ensemble_return - baseline_return,
            "allocation_conservation": metadata.get("allocation_conservation", {}),
            "overfit_rejections": rejected_reasons,
            "cap_rejections": {"cap_blocked_count": int(metadata.get("cap_blocked_count", 0) or 0)},
            "skipped": skipped,
        }

    @staticmethod
    def _aggregate_market_structure(items: list[dict[str, Any]]) -> dict[str, Any]:
        snapshots = [item.get("market_structure") for item in items if isinstance(item.get("market_structure"), dict)]
        if not snapshots:
            return {}
        def average(key: str) -> float:
            values = [float(snapshot.get(key, 0.0) or 0.0) for snapshot in snapshots]
            return sum(values) / len(values) if values else 0.0
        return {
            "enabled": any(bool(snapshot.get("enabled", False)) for snapshot in snapshots),
            "provider": str(snapshots[-1].get("provider", "existing")),
            "score": average("score"),
            "coverage": average("coverage"),
            "book_depth_score": average("book_depth_score"),
            "spread_trend_bps": average("spread_trend_bps"),
            "volume_impulse": average("volume_impulse"),
            "volatility_regime_score": average("volatility_regime_score"),
        }

    @staticmethod
    def _rejection_breakdown(rankings: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in rankings:
            if not item.get("rejected"):
                continue
            reason = str(item.get("rejection_reason") or item.get("no_trade_reason") or "rejected")
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def _duration_return_leaders(
        self,
        rankings: list[dict[str, Any]],
        optimizer_config: OptimizerConfig,
    ) -> dict[str, dict[str, Any]]:
        leaders: dict[str, dict[str, Any]] = {}
        for item in rankings:
            if item.get("rejected"):
                continue
            duration = int(item.get("lock_duration_hours") or optimizer_config.lock_duration_hours or 1)
            bucket = EnhancedEnsembleAllocator.duration_bucket(duration)
            current = leaders.get(bucket)
            if current is None or float(item.get("net_return_after_costs", 0.0) or 0.0) > float(current.get("net_return_after_costs", 0.0) or 0.0):
                leaders[bucket] = self._baseline_payload(item)
        return leaders

    @staticmethod
    def _market_structure_feature_coverage(rankings: list[dict[str, Any]]) -> dict[str, Any]:
        snapshots = [item.get("market_structure") for item in rankings if isinstance(item.get("market_structure"), dict)]
        if not snapshots:
            return {"enabled": False, "candidate_count": 0, "average_coverage": 0.0}
        enabled = [snapshot for snapshot in snapshots if bool(snapshot.get("enabled", False))]
        coverage = sum(float(snapshot.get("coverage", 0.0) or 0.0) for snapshot in snapshots) / len(snapshots)
        return {
            "enabled": bool(enabled),
            "candidate_count": len(snapshots),
            "enabled_candidate_count": len(enabled),
            "average_coverage": coverage,
            "provider": str(snapshots[-1].get("provider", "existing")),
        }

    def _max_return_summary(
        self,
        rankings: list[dict[str, Any]],
        optimizer_config: OptimizerConfig,
        duration_ensemble_backtest: dict[str, Any],
    ) -> dict[str, Any]:
        accepted = [item for item in rankings if not item.get("rejected")]
        accepted.sort(key=lambda item: float(item.get("net_return_after_costs", 0.0) or 0.0), reverse=True)
        top = self._baseline_payload(accepted[0]) if accepted else {}
        return {
            "enabled": bool(optimizer_config.max_return_optimizer_enabled),
            "profit_objective_version": "max_return_v3" if optimizer_config.max_return_optimizer_enabled else "",
            "primary_sort": "net_return_after_costs",
            "min_net_return": float(optimizer_config.max_return_min_net_return or 0.0),
            "max_drawdown_pct": float(optimizer_config.max_return_max_drawdown_pct or optimizer_config.max_drawdown_pct or 0.0),
            "accepted_count": len(accepted),
            "rejected_count": len(rankings) - len(accepted),
            "top_candidate": top,
            "duration_ensemble_return": duration_ensemble_backtest.get("net_return_after_costs"),
        }

    def _pair_screening_summary(
        self,
        rankings: list[dict[str, Any]],
        optimizer_config: OptimizerConfig,
        duration_ensemble_backtest: dict[str, Any],
    ) -> dict[str, Any]:
        if not bool(optimizer_config.pair_screening_enabled) or self.pair_screening is None:
            return {
                "enabled": bool(optimizer_config.pair_screening_enabled),
                "trading_enabled": bool(optimizer_config.pair_trading_enabled),
                "candidate_count": 0,
                "pair_candidates": [],
                "pair_rejection_breakdown": {},
                "pair_baseline_comparison": {},
            }
        service_config = getattr(self.pair_screening, "config", None)
        previous_values: dict[str, Any] = {}
        if isinstance(service_config, dict):
            for key, value in {
                "PAIR_SCREENING_ENABLED": optimizer_config.pair_screening_enabled,
                "PAIR_TRADING_ENABLED": optimizer_config.pair_trading_enabled,
                "PAIR_LIVE_ELIGIBLE": optimizer_config.pair_live_eligible,
                "PAIR_MIN_CORRELATION": optimizer_config.pair_min_correlation,
                "PAIR_MAX_SPREAD_ZSCORE": optimizer_config.pair_max_spread_zscore,
            }.items():
                previous_values[key] = service_config.get(key)
                service_config[key] = value
        try:
            symbols = self._optimizer_symbols(optimizer_config)
            timeframe = optimizer_config.timeframes[0] if optimizer_config.timeframes else "5m"
            pair_candidates = self.pair_screening.screen(
                symbols,
                mode=optimizer_config.mode,
                timeframe=timeframe,
                duration_hours=optimizer_config.lock_duration_hours or 24,
                pair_mode="both",
            )
        finally:
            if isinstance(service_config, dict):
                for key, value in previous_values.items():
                    service_config[key] = value

        payloads = [candidate.as_dict() if hasattr(candidate, "as_dict") else dict(candidate) for candidate in pair_candidates]
        accepted = [item for item in rankings if not item.get("rejected")]
        accepted.sort(key=lambda item: float(item.get("net_return_after_costs", 0.0) or 0.0), reverse=True)
        best_single = self._baseline_payload(accepted[0]) if accepted else {}
        top_pair = payloads[0] if payloads else {}
        pair_score = float(top_pair.get("pair_score", top_pair.get("score", 0.0)) or 0.0) if top_pair else 0.0
        baseline_return = float(best_single.get("net_return_after_costs", 0.0) or 0.0)
        duration_return = float(duration_ensemble_backtest.get("net_return_after_costs", 0.0) or 0.0)
        return {
            "enabled": True,
            "trading_enabled": bool(optimizer_config.pair_trading_enabled),
            "live_eligible": bool(optimizer_config.pair_live_eligible),
            "candidate_count": len(payloads),
            "pair_candidates": payloads[:10],
            "pair_rejection_breakdown": dict(getattr(self.pair_screening, "last_rejections", {}) or {}),
            "pair_baseline_comparison": {
                "baseline_single_strategy": best_single,
                "baseline_current_basket": duration_ensemble_backtest.get("baseline_current_basket", {}),
                "duration_ensemble_net_return_after_costs": duration_return,
                "top_pair_score": pair_score,
                "pair_score_vs_baseline_return": pair_score - baseline_return,
            },
        }

    @staticmethod
    def _live_eligibility_summary(optimizer_config: OptimizerConfig) -> dict[str, Any]:
        return {
            "dynamic_intraday_live_eligible": bool(optimizer_config.dynamic_intraday_live_eligible),
            "max_return_live_eligible": bool(optimizer_config.max_return_live_eligible),
            "experimental_duration_live_eligible": bool(optimizer_config.experimental_live_eligible),
            "pair_live_eligible": bool(optimizer_config.pair_live_eligible),
            "high_upside_profile": bool(optimizer_config.high_upside_profile),
            "no_default_live_cap_increase": True,
            "duration_live_cap_usdc_by_duration": optimizer_config.duration_live_cap_usdc_by_duration,
            "duration_live_cap_pct_by_duration": optimizer_config.duration_live_cap_pct_by_duration,
        }

    def _high_upside_readiness(self, optimizer_config: OptimizerConfig) -> dict[str, Any]:
        duration_bucket = EnhancedEnsembleAllocator.duration_bucket(optimizer_config.lock_duration_hours or 1)
        cap_usdc = self._duration_config_cap("HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION", duration_bucket)
        cap_pct = self._duration_config_cap("HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION", duration_bucket)
        max_notional = float(self.config.get("HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD", 0.0) or 0.0)
        max_daily_loss = float(self.config.get("HIGH_UPSIDE_MAX_DAILY_LOSS_USDC", 0.0) or 0.0)
        blockers: list[str] = []
        if not bool(optimizer_config.high_upside_profile):
            blockers.append("high_upside_profile_not_requested")
        if not bool(self.config.get("HIGH_UPSIDE_PROFILE_ENABLED", False)):
            blockers.append("HIGH_UPSIDE_PROFILE_ENABLED=false")
        if not bool(self.config.get("HIGH_UPSIDE_LIVE_ELIGIBLE", False)):
            blockers.append("HIGH_UPSIDE_LIVE_ELIGIBLE=false")
        if bool(Setting.get_json("high_upside_live_disabled", False)):
            blockers.append("high_upside_auto_disabled")
        if max_notional <= 0:
            blockers.append("HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD_missing")
        if max_daily_loss <= 0:
            blockers.append("HIGH_UPSIDE_MAX_DAILY_LOSS_USDC_missing")
        if cap_usdc <= 0:
            blockers.append("HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION_JSON_missing")
        if cap_pct <= 0:
            blockers.append("HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION_JSON_missing")
        ml_readiness = self.offline_ranker.readiness(duration_bucket, require_blend=False)
        if bool(self.config.get("HIGH_UPSIDE_REQUIRE_PROMOTED_ML", True)) and not bool(ml_readiness.get("ready", False)):
            blockers.append("promoted_offline_ml_required")

        return {
            "requested": bool(optimizer_config.high_upside_profile),
            "profile_enabled": bool(self.config.get("HIGH_UPSIDE_PROFILE_ENABLED", False)),
            "live_eligible": bool(self.config.get("HIGH_UPSIDE_LIVE_ELIGIBLE", False)),
            "auto_disabled": bool(Setting.get_json("high_upside_live_disabled", False)),
            "disabled_reason": Setting.get_json("high_upside_live_disabled_reason", {}),
            "ready": not blockers,
            "blockers": blockers,
            "duration_bucket": duration_bucket,
            "caps": {
                "max_position_notional_usd": max_notional,
                "max_daily_loss_usdc": max_daily_loss,
                "duration_cap_usdc": cap_usdc,
                "duration_cap_pct": cap_pct,
            },
            "requires_candidate_tag": True,
            "requires_shadow_or_pre_live_validation": bool(optimizer_config.require_shadow_validation),
            "requires_promoted_offline_ml": bool(self.config.get("HIGH_UPSIDE_REQUIRE_PROMOTED_ML", True)),
            "offline_ml_readiness": ml_readiness,
            "no_default_live_cap_increase": True,
        }

    def _duration_config_cap(self, key: str, bucket: str) -> float:
        mapping = self.config.get(key, {}) or {}
        if not isinstance(mapping, dict):
            return 0.0
        for lookup in (bucket, str(bucket).lower(), str(bucket).upper()):
            if lookup in mapping:
                try:
                    return float(mapping[lookup] or 0.0)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def _raw_upside_candidate_metrics(
        self,
        *,
        total_return: float,
        net_return_after_costs: float,
        recent_1h_return: float,
        max_favorable_excursion: float,
        leverage: float,
        rejected: bool,
        rejection_reason: str,
        weighted_drawdown: float,
        expected_fill_quality: float,
        window_stability: float,
        churn_penalty: float,
        regime_support: str,
        data_age_seconds: float,
        optimizer_config: OptimizerConfig,
    ) -> dict[str, Any]:
        """Score historical raw upside while keeping live-readiness blockers explicit."""

        target_roi_pct = float(self.config.get("RAW_UPSIDE_TARGET_ROI_PCT", 1000.0) or 1000.0)
        raw_total_return_pct = float(total_return or 0.0) * 100.0
        raw_net_return_pct = float(net_return_after_costs or 0.0) * 100.0
        raw_upside_score = (
            raw_total_return_pct
            + max(float(recent_1h_return or 0.0), 0.0) * 100.0
            + max(float(max_favorable_excursion or 0.0), 0.0) * 100.0
            + max(float(leverage or 1.0) - 1.0, 0.0) * 5.0
        )
        blockers: list[str] = []
        if rejected:
            blockers.append(f"candidate_rejected:{rejection_reason or 'unknown'}")
        if float(weighted_drawdown or 0.0) <= -abs(float(optimizer_config.max_drawdown_pct or 0.0)):
            blockers.append("excessive_drawdown")
        if float(expected_fill_quality or 0.0) < float(self.config.get("NET_ROI_MIN_FILL_QUALITY", 0.55) or 0.55):
            blockers.append("low_fill_quality")
        if float(window_stability or 0.0) < float(self.config.get("AGGRESSIVE_1H_MIN_WINDOW_STABILITY", 0.55) or 0.55):
            blockers.append("weak_signal_stability")
        if float(churn_penalty or 0.0) > float(self.config.get("NET_ROI_MAX_CHURN_PENALTY", 0.35) or 0.35):
            blockers.append("high_churn")
        if str(regime_support or "").lower() == "regime-fragile":
            blockers.append("fragile_regime")
        if float(data_age_seconds or 0.0) > 3600.0:
            blockers.append("stale_data")
        if bool(self.config.get("HIGH_UPSIDE_REQUIRE_PROMOTED_ML", True)):
            blockers.append("promoted_ml_required_for_high_upside")
        if bool(self.config.get("CANARY_PREVIEW_ONLY", True)):
            blockers.append("canary_preview_only_enabled")
        blockers.extend(
            [
                "strict_readiness_required",
                "funded_account_required",
                "active_verified_live_connection_required",
            ]
        )
        return {
            "raw_upside_score": float(raw_upside_score),
            "raw_total_return_pct": float(raw_total_return_pct),
            "raw_net_return_pct": float(raw_net_return_pct),
            "target_roi_pct": float(target_roi_pct),
            "target_roi_hit": bool(raw_total_return_pct >= target_roi_pct),
            "live_blockers": list(dict.fromkeys(blockers)),
        }

    def _raw_upside_report(self, rankings: list[dict[str, Any]], optimizer_config: OptimizerConfig) -> dict[str, Any]:
        target_roi_pct = float(self.config.get("RAW_UPSIDE_TARGET_ROI_PCT", 1000.0) or 1000.0)
        sorted_raw = sorted(
            rankings,
            key=lambda item: float(item.get("raw_upside_score", item.get("raw_total_return_pct", 0.0)) or 0.0),
            reverse=True,
        )
        sorted_net = sorted(
            rankings,
            key=lambda item: float(item.get("net_roi_v2_score", item.get("net_roi_score", 0.0)) or 0.0),
            reverse=True,
        )
        target_hits = [
            item for item in sorted_raw if float(item.get("raw_total_return_pct", 0.0) or 0.0) >= target_roi_pct
        ]
        approaching = [
            item for item in sorted_raw if float(item.get("raw_total_return_pct", 0.0) or 0.0) >= target_roi_pct * 0.5
        ]
        if not approaching:
            approaching = sorted_raw[:3]
        accepted = [item for item in sorted_net if not item.get("rejected")]
        rejected_high_raw = [item for item in sorted_raw if item.get("rejected")]
        return {
            "enabled": True,
            "research_only": True,
            "profile": optimizer_config.profile,
            "target_roi_pct": target_roi_pct,
            "target_roi_hit": bool(target_hits),
            "target_roi_hit_count": len(target_hits),
            "candidate_count": len(rankings),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected_high_raw),
            "max_return_candidate": self._raw_upside_payload(sorted_raw[0]) if sorted_raw else {},
            "top_by_raw_upside": [self._raw_upside_payload(item) for item in sorted_raw[:10]],
            "top_by_net_roi_v2": [self._raw_upside_payload(item) for item in sorted_net[:10]],
            "target_or_near_target_candidates": [self._raw_upside_payload(item) for item in approaching[:10]],
            "rejected_high_raw_upside": [self._raw_upside_payload(item) for item in rejected_high_raw[:10]],
            "best_preview_candidate": self._raw_upside_payload(accepted[0]) if accepted else {},
            "live_orders_created": False,
            "canary_preview_only_required": True,
        }

    @staticmethod
    def _raw_upside_payload(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "strategy_name": item.get("strategy_name"),
            "symbol": item.get("symbol"),
            "timeframe": item.get("timeframe"),
            "profile": item.get("profile"),
            "rejected": bool(item.get("rejected")),
            "rejection_reason": item.get("rejection_reason") or "",
            "raw_upside_score": float(item.get("raw_upside_score", 0.0) or 0.0),
            "raw_total_return_pct": float(item.get("raw_total_return_pct", 0.0) or 0.0),
            "raw_net_return_pct": float(item.get("raw_net_return_pct", 0.0) or 0.0),
            "target_roi_pct": float(item.get("target_roi_pct", 1000.0) or 1000.0),
            "target_roi_hit": bool(item.get("target_roi_hit", False)),
            "live_blockers": list(item.get("live_blockers", []) or []),
            "total_return": float(item.get("total_return", 0.0) or 0.0),
            "net_return_after_costs": float(item.get("net_return_after_costs", 0.0) or 0.0),
            "net_roi_v2_score": float(item.get("net_roi_v2_score", 0.0) or 0.0),
            "net_roi_score": float(item.get("net_roi_score", 0.0) or 0.0),
            "roi_quality_grade": item.get("roi_quality_grade", "D"),
            "roi_rejection_risk": item.get("roi_rejection_risk", "high"),
            "regime_support": item.get("regime_support", "regime-neutral"),
            "max_drawdown": float(item.get("max_drawdown", 0.0) or 0.0),
            "expected_fill_quality": float(item.get("expected_fill_quality", 0.0) or 0.0),
            "churn_penalty": float(item.get("churn_penalty", 0.0) or 0.0),
            "cost_drag_bps": float(item.get("cost_drag_bps", 0.0) or 0.0),
            "spread_bps": float(item.get("spread_bps", 0.0) or 0.0),
            "window_stability": float(item.get("window_stability", 0.0) or 0.0),
            "mfe_mae_ratio": float(item.get("mfe_mae_ratio", 0.0) or 0.0),
            "max_favorable_excursion": float(item.get("max_favorable_excursion", 0.0) or 0.0),
            "max_adverse_excursion": float(item.get("max_adverse_excursion", 0.0) or 0.0),
            "trade_count": int(item.get("trade_count", 0) or 0),
            "parameters": dict(item.get("parameters", {}) or {}),
        }

    def _one_hour_diagnostics(self, rankings: list[dict[str, Any]], optimizer_config: OptimizerConfig) -> dict[str, Any]:
        if optimizer_config.profile != Profile.AGGRESSIVE_1H.value:
            return {"enabled": False}

        accepted = [item for item in rankings if not item.get("rejected")]
        rejected = [item for item in rankings if item.get("rejected")]
        sort_field = "net_roi_v2_score" if bool(self.config.get("NET_ROI_V2_ENABLED", True)) else "net_roi_score"
        accepted.sort(key=lambda item: float(item.get(sort_field, item.get("net_roi_score", item.get("convex_edge_score", item.get("score", 0.0)))) or 0.0), reverse=True)
        rejected.sort(key=lambda item: float(item.get(sort_field, item.get("net_roi_score", item.get("convex_edge_score", item.get("score", 0.0)))) or 0.0), reverse=True)

        def payload(item: dict[str, Any]) -> dict[str, Any]:
            return {
                "strategy_name": item.get("strategy_name"),
                "symbol": item.get("symbol"),
                "timeframe": item.get("timeframe"),
                "rejected": bool(item.get("rejected")),
                "rejection_reason": item.get("rejection_reason") or "",
                "score": float(item.get("score", 0.0) or 0.0),
                "net_roi_score": float(item.get("net_roi_score", 0.0) or 0.0),
                "net_roi_v2_score": float(item.get("net_roi_v2_score", 0.0) or 0.0),
                "roi_quality_grade": item.get("roi_quality_grade", "D"),
                "roi_rejection_risk": item.get("roi_rejection_risk", "high"),
                "regime_bucket": item.get("regime_bucket", {}),
                "regime_support": item.get("regime_support", "regime-neutral"),
                "tail_loss_penalty": float(item.get("tail_loss_penalty", 0.0) or 0.0),
                "downside_asymmetry_penalty": float(item.get("downside_asymmetry_penalty", 0.0) or 0.0),
                "cost_adjusted_breakout_potential": float(item.get("cost_adjusted_breakout_potential", 0.0) or 0.0),
                "expected_fill_quality": float(item.get("expected_fill_quality", 0.0) or 0.0),
                "churn_penalty": float(item.get("churn_penalty", 0.0) or 0.0),
                "convex_edge_score": float(item.get("convex_edge_score", 0.0) or 0.0),
                "mfe_mae_ratio": float(item.get("mfe_mae_ratio", 0.0) or 0.0),
                "capacity_multiple": float(item.get("capacity_multiple", 0.0) or 0.0),
                "cost_adjusted_recent_1h_return": float(item.get("cost_adjusted_recent_1h_return", 0.0) or 0.0),
                "decay_penalty": float(item.get("decay_penalty", 0.0) or 0.0),
                "net_return_after_costs": float(item.get("net_return_after_costs", 0.0) or 0.0),
                "recent_1h_return": float(item.get("recent_1h_return", 0.0) or 0.0),
                "edge_score": float(item.get("edge_score", 0.0) or 0.0),
                "cost_drag_bps": float(item.get("cost_drag_bps", 0.0) or 0.0),
                "window_stability": float(item.get("window_stability", 0.0) or 0.0),
                "liquidity_capacity_usd": float(item.get("liquidity_capacity_usd", item.get("capacity_usd", 0.0)) or 0.0),
            }

        return {
            "enabled": True,
            "primary_sort": sort_field,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "rejection_breakdown": self._rejection_breakdown(rankings),
            "top_accepted": [payload(item) for item in accepted[:5]],
            "top_rejected": [payload(item) for item in rejected[:5]],
            "thresholds": {
                "max_cost_drag_bps": float(self.config.get("AGGRESSIVE_1H_MAX_COST_DRAG_BPS", 18.0) or 18.0),
                "min_capacity_multiple": float(self.config.get("AGGRESSIVE_1H_MIN_CAPACITY_MULTIPLE", 2.0) or 2.0),
                "min_mfe_mae": float(self.config.get("AGGRESSIVE_1H_MIN_MFE_MAE", 1.5) or 1.5),
                "min_window_stability": float(self.config.get("AGGRESSIVE_1H_MIN_WINDOW_STABILITY", 0.55) or 0.55),
            },
        }

    def _dynamic_intraday_diagnostics(self, rankings: list[dict[str, Any]], optimizer_config: OptimizerConfig) -> dict[str, Any]:
        if optimizer_config.profile != Profile.DYNAMIC_INTRADAY.value:
            return {"enabled": False}

        status_counts: dict[str, int] = {}
        top_candidates: list[dict[str, Any]] = []
        for item in rankings[:10]:
            eligibility = item.get("live_eligibility") if isinstance(item.get("live_eligibility"), dict) else {}
            status = str(eligibility.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            top_candidates.append(
                {
                    "strategy_name": item.get("strategy_name"),
                    "symbol": item.get("symbol"),
                    "timeframe": item.get("timeframe"),
                    "rejected": bool(item.get("rejected")),
                    "rejection_reason": item.get("rejection_reason") or "",
                    "eligibility_status": status,
                    "cost_drag_bps": float(item.get("cost_drag_bps", 0.0) or 0.0),
                    "liquidity_capacity_usd": float(item.get("liquidity_capacity_usd", item.get("capacity_usd", 0.0)) or 0.0),
                    "spread_bps": float(item.get("spread_bps", 0.0) or 0.0),
                    "volatility_regime": item.get("volatility_regime", "unknown"),
                    "recent_decay": float(item.get("recent_decay", item.get("performance_decay_rate", 0.0)) or 0.0),
                    "edge_score": float(item.get("edge_score", 0.0) or 0.0),
                    "net_roi_score": float(item.get("net_roi_score", 0.0) or 0.0),
                    "net_roi_v2_score": float(item.get("net_roi_v2_score", 0.0) or 0.0),
                    "roi_quality_grade": item.get("roi_quality_grade", "D"),
                    "roi_rejection_risk": item.get("roi_rejection_risk", "high"),
                    "regime_bucket": item.get("regime_bucket", {}),
                    "regime_support": item.get("regime_support", "regime-neutral"),
                    "expected_fill_quality": float(item.get("expected_fill_quality", 0.0) or 0.0),
                    "churn_penalty": float(item.get("churn_penalty", 0.0) or 0.0),
                    "score": float(item.get("score", 0.0) or 0.0),
                }
            )

        return {
            "enabled": True,
            "require_shadow_validation": bool(optimizer_config.require_shadow_validation),
            "dynamic_intraday_live_eligible": bool(optimizer_config.dynamic_intraday_live_eligible),
            "candidate_status_counts": status_counts,
            "universe_rejection_breakdown": dict(getattr(self.universe_service, "last_rejections", {}) or {}),
            "top_candidates": top_candidates,
        }

    def _net_roi_v2_summary(self, rankings: list[dict[str, Any]]) -> dict[str, Any]:
        accepted = [item for item in rankings if not item.get("rejected")]
        rejected = [item for item in rankings if item.get("rejected")]

        def ordered(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            sorted_items = sorted(
                items,
                key=lambda item: float(item.get("net_roi_v2_score", item.get("net_roi_score", 0.0)) or 0.0),
                reverse=True,
            )
            return [
                {
                    "strategy_name": item.get("strategy_name"),
                    "symbol": item.get("symbol"),
                    "timeframe": item.get("timeframe"),
                    "rejected": bool(item.get("rejected")),
                    "rejection_reason": item.get("rejection_reason", ""),
                    "net_roi_v2_score": float(item.get("net_roi_v2_score", 0.0) or 0.0),
                    "net_roi_score": float(item.get("net_roi_score", 0.0) or 0.0),
                    "roi_quality_grade": item.get("roi_quality_grade", "D"),
                    "roi_rejection_risk": item.get("roi_rejection_risk", "high"),
                    "regime_support": item.get("regime_support", "regime-neutral"),
                    "regime_bucket": item.get("regime_bucket", {}),
                    "tail_loss_penalty": float(item.get("tail_loss_penalty", 0.0) or 0.0),
                    "downside_asymmetry_penalty": float(item.get("downside_asymmetry_penalty", 0.0) or 0.0),
                    "cost_adjusted_breakout_potential": float(item.get("cost_adjusted_breakout_potential", 0.0) or 0.0),
                }
                for item in sorted_items[:5]
            ]

        grade_counts: dict[str, int] = {}
        risk_counts: dict[str, int] = {}
        regime_counts: dict[str, int] = {}
        for item in rankings:
            grade = str(item.get("roi_quality_grade") or "D")
            risk = str(item.get("roi_rejection_risk") or "high")
            regime = str(item.get("regime_support") or "regime-neutral")
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        return {
            "enabled": bool(self.config.get("NET_ROI_V2_ENABLED", True)),
            "min_quality_grade": self.config.get("NET_ROI_V2_MIN_QUALITY_GRADE", "B"),
            "grade_counts": grade_counts,
            "risk_counts": risk_counts,
            "regime_support_counts": regime_counts,
            "top_accepted_by_research_roi": ordered(accepted),
            "top_rejected_by_research_roi": ordered(rejected),
        }

    @staticmethod
    def _ranking_namespace(item: dict[str, Any], index: int, optimizer_config: OptimizerConfig) -> SimpleNamespace:
        return SimpleNamespace(
            id=index + 1,
            strategy_name=item.get("strategy_name"),
            symbol=item.get("symbol"),
            timeframe=item.get("timeframe"),
            profile=item.get("profile") or optimizer_config.profile,
            parameters=item.get("parameters", {}),
            score=float(item.get("score", 0.0) or 0.0),
            total_return=float(item.get("total_return", 0.0) or 0.0),
            net_return_after_costs=float(item.get("net_return_after_costs", 0.0) or 0.0),
            recent_performance_score=float(item.get("recent_performance_score", 0.0) or 0.0),
            recent_1h_return=float(item.get("recent_1h_return", 0.0) or 0.0),
            edge_score=float(item.get("edge_score", 0.0) or 0.0),
            expectancy=float(item.get("expectancy", 0.0) or 0.0),
            cost_drag_bps=float(item.get("cost_drag_bps", 0.0) or 0.0),
            net_roi_v2_score=float(item.get("net_roi_v2_score", 0.0) or 0.0),
            roi_quality_grade=item.get("roi_quality_grade", "D"),
            roi_rejection_risk=item.get("roi_rejection_risk", "high"),
            regime_bucket=item.get("regime_bucket", {}),
            regime_support=item.get("regime_support", "regime-neutral"),
            tail_loss_penalty=float(item.get("tail_loss_penalty", 0.0) or 0.0),
            downside_asymmetry_penalty=float(item.get("downside_asymmetry_penalty", 0.0) or 0.0),
            cost_adjusted_breakout_potential=float(item.get("cost_adjusted_breakout_potential", 0.0) or 0.0),
            max_adverse_excursion=float(item.get("max_adverse_excursion", 0.0) or 0.0),
            max_favorable_excursion=float(item.get("max_favorable_excursion", 0.0) or 0.0),
            no_trade_reason=item.get("no_trade_reason", ""),
            allocation_amount_usd=float(optimizer_config.allocation_amount_usd or optimizer_config.initial_balance or 0.0),
            lock_duration_hours=int(optimizer_config.lock_duration_hours or 0),
            leverage=float(item.get("leverage", 1.0) or 1.0),
            capacity_usd=float(item.get("capacity_usd", 0.0) or 0.0),
            convex_edge_score=float(item.get("convex_edge_score", 0.0) or 0.0),
            mfe_mae_ratio=float(item.get("mfe_mae_ratio", 0.0) or 0.0),
            capacity_multiple=float(item.get("capacity_multiple", 0.0) or 0.0),
            cost_adjusted_recent_1h_return=float(item.get("cost_adjusted_recent_1h_return", 0.0) or 0.0),
            decay_penalty=float(item.get("decay_penalty", 0.0) or 0.0),
            high_upside_profile=bool(item.get("high_upside_profile", False)),
            universe_source=item.get("universe_source", "optimizer"),
            execution_style=item.get("execution_style", "market"),
            profit_objective_version=item.get("profit_objective_version", ""),
            market_structure=item.get("market_structure", {}),
            market_structure_score=float(item.get("market_structure_score", 0.0) or 0.0),
            max_drawdown=float(item.get("max_drawdown", 0.0) or 0.0),
            profit_factor=float(item.get("profit_factor", 0.0) or 0.0),
            sharpe_like=float(item.get("sharpe_like", 0.0) or 0.0),
            sortino_like=float(item.get("sortino_like", 0.0) or 0.0),
            trades_per_day=float(item.get("trades_per_day", 0.0) or 0.0),
            trade_count=int(item.get("trade_count", 0) or 0),
            win_rate=float(item.get("win_rate", 0.0) or 0.0),
            created_at=datetime.fromtimestamp(index, tz=timezone.utc),
        )

    @staticmethod
    def _baseline_payload(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "strategy_name": item.get("strategy_name"),
            "symbol": item.get("symbol"),
            "timeframe": item.get("timeframe"),
            "net_return_after_costs": float(item.get("net_return_after_costs", 0.0) or 0.0),
            "score": float(item.get("score", 0.0) or 0.0),
            "net_roi_v2_score": float(item.get("net_roi_v2_score", 0.0) or 0.0),
            "roi_quality_grade": item.get("roi_quality_grade", "D"),
            "regime_support": item.get("regime_support", "regime-neutral"),
            "convex_edge_score": float(item.get("convex_edge_score", 0.0) or 0.0),
            "mfe_mae_ratio": float(item.get("mfe_mae_ratio", 0.0) or 0.0),
            "capacity_multiple": float(item.get("capacity_multiple", 0.0) or 0.0),
            "sharpe_like": float(item.get("sharpe_like", 0.0) or 0.0),
            "max_drawdown": float(item.get("max_drawdown", 0.0) or 0.0),
            "profit_objective_version": item.get("profit_objective_version", ""),
            "market_structure_score": float(item.get("market_structure_score", 0.0) or 0.0),
        }

    @staticmethod
    def _basket_baseline_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"selected_legs": [], "net_return_after_costs": 0.0}
        weights = [max(float(item.get("score", 0.0) or 0.0), float(item.get("edge_score", 0.0) or 0.0), 0.01) for item in items]
        total = sum(weights) or 1.0
        normalized = [weight / total for weight in weights]

        def weighted(key: str) -> float:
            return sum(float(item.get(key, 0.0) or 0.0) * weight for item, weight in zip(items, normalized))

        return {
            "selected_legs": [
                {
                    "strategy_name": item.get("strategy_name"),
                    "symbol": item.get("symbol"),
                    "timeframe": item.get("timeframe"),
                    "weight": round(weight, 6),
                }
                for item, weight in zip(items, normalized)
            ],
            "net_return_after_costs": weighted("net_return_after_costs"),
            "sharpe_like": weighted("sharpe_like"),
            "max_drawdown": weighted("max_drawdown"),
        }

    # ------------------------------------------------------------------
    # Static utilities
    #
    @staticmethod
    def _is_hourly_experimental(profile: str | None) -> bool:
        return str(profile or "") in {Profile.AGGRESSIVE_1H.value, Profile.EXTREME_ROI_EXPERIMENTAL.value, Profile.DYNAMIC_INTRADAY.value}

    @staticmethod
    def _is_experimental_profile(profile: str | None) -> bool:
        return str(profile or "") in {Profile.AGGRESSIVE_1H.value, Profile.EXTREME_ROI_EXPERIMENTAL.value}

    @staticmethod
    def _risk_label(profile: str | None) -> str:
        if str(profile or "") == Profile.EXTREME_ROI_EXPERIMENTAL.value:
            return "Extreme Experimental Risk"
        if str(profile or "") == Profile.AGGRESSIVE_1H.value:
            return "Very High Risk"
        if str(profile or "") == Profile.DYNAMIC_INTRADAY.value:
            return "Dynamic Intraday Production"
        return ""

    @staticmethod
    def _profile_warning(profile: str | None) -> str:
        if str(profile or "") == Profile.EXTREME_ROI_EXPERIMENTAL.value:
            return EXTREME_ROI_WARNING
        if str(profile or "") == Profile.AGGRESSIVE_1H.value:
            return AGGRESSIVE_1H_WARNING
        if str(profile or "") == Profile.DYNAMIC_INTRADAY.value:
            return DYNAMIC_INTRADAY_WARNING
        return ""

    @staticmethod
    def _rejected_candidate(
        symbol: str,
        timeframe: str,
        strategy_name: str,
        parameters: Dict[str, Any],
        reason: str,
        optimizer_config: OptimizerConfig | None = None,
    ) -> Dict[str, Any]:
        """Return a default result for a candidate that could not be evaluated."""
        profile = optimizer_config.profile if optimizer_config is not None else ""
        experimental = StrategyOptimizer._is_experimental_profile(profile)
        warnings = [reason]
        if StrategyOptimizer._profile_warning(profile):
            warnings.append(StrategyOptimizer._profile_warning(profile))
        return {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "profile": profile,
            "experimental": experimental,
            "risk_label": StrategyOptimizer._risk_label(profile),
            "warning": StrategyOptimizer._profile_warning(profile),
            "parameters": parameters,
            "score": -999.0,
            "base_score": -999.0,
            "ml_score": 0.0,
            "ml_adjusted_score": -999.0,
            "ml_warmup": True,
            "ml_explanation": {},
            "total_return": 0.0,
            "net_return_after_costs": 0.0,
            "raw_upside_score": 0.0,
            "raw_total_return_pct": 0.0,
            "raw_net_return_pct": 0.0,
            "target_roi_pct": 1000.0,
            "target_roi_hit": False,
            "live_blockers": [
                f"candidate_rejected:{reason}",
                "strict_readiness_required",
                "canary_preview_only_enabled",
            ],
            "recent_performance_score": 0.0,
            "recent_1h_return": 0.0,
            "estimated_fees": 0.0,
            "edge_score": 0.0,
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_loss_ratio": 0.0,
            "cost_drag_bps": 0.0,
            "net_roi_score": 0.0,
            "net_roi_v2_score": 0.0,
            "roi_quality_grade": "D",
            "roi_rejection_risk": "high",
            "regime_bucket": {},
            "regime_support": "regime-neutral",
            "regime_adjustment": 0.0,
            "regime_adjusted_expectancy": 0.0,
            "tail_loss_penalty": 0.0,
            "downside_asymmetry_penalty": 0.0,
            "cost_adjusted_breakout_potential": 0.0,
            "expected_fill_quality": 0.0,
            "churn_penalty": 0.0,
            "edge_after_cost_bps": 0.0,
            "data_age_seconds": 0.0,
            "net_roi_components": {},
            "net_roi_v2_components": {},
            "convex_edge_score": 0.0,
            "mfe_mae_ratio": 0.0,
            "capacity_multiple": 0.0,
            "cost_adjusted_recent_1h_return": 0.0,
            "decay_penalty": 0.0,
            "liquidity_capacity_usd": 0.0,
            "spread_bps": 0.0,
            "volatility_regime": "unknown",
            "recent_decay": 0.0,
            "max_adverse_excursion": 0.0,
            "max_favorable_excursion": 0.0,
            "no_trade_reason": reason,
            "allocation_amount_usd": float(optimizer_config.allocation_amount_usd or 0.0) if optimizer_config is not None else 0.0,
            "lock_duration_hours": int(optimizer_config.lock_duration_hours or 0) if optimizer_config is not None else 0,
            "leverage": 1.0,
            "liquidation_buffer_pct": 1.0,
            "capacity_usd": 0.0,
            "universe_source": "configured",
            "vault_leg_count": 1,
            "execution_style": "market",
            "profit_objective_version": "max_return_v3" if optimizer_config is not None and optimizer_config.max_return_optimizer_enabled else "",
            "high_upside_profile": bool(optimizer_config.high_upside_profile) if optimizer_config is not None else False,
            "market_structure": StrategyOptimizer._neutral_market_structure(optimizer_config) if optimizer_config is not None else {},
            "market_structure_score": 0.0,
            "funding_cost_estimate": 0.0,
            "performance_last_3_days": 0.0,
            "performance_last_7_days": 0.0,
            "performance_decay_rate": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "sharpe_like": 0.0,
            "sortino_like": 0.0,
            "trades_per_day": 0.0,
            "avg_trade_return": 0.0,
            "turnover_rate": 0.0,
            "turnover_after_fees": 0.0,
            "consistency": 0.0,
            "window_stability": 0.0,
            "accepted_window_ratio": 0.0,
            "win_rate": 0.0,
            "trade_count": 0,
            "window_count": 0,
            "rejected": True,
            "rejection_reason": reason,
            "warnings": warnings,
            "live_eligibility": StrategyOptimizer._candidate_live_eligibility(True, optimizer_config) if optimizer_config is not None else {},
        }

    @staticmethod
    def _weights(count: int, optimizer_config: OptimizerConfig) -> List[float]:
        """Compute a list of recency weights for a given number of windows."""
        if not optimizer_config.recency_weighting_enabled:
            return [1.0] * count
        decay = min(max(optimizer_config.decay_factor, 0.01), 1.0)
        return [decay ** (count - index - 1) for index in range(count)]

    @staticmethod
    def _weighted_mean(values: List[float], weights: List[float]) -> float:
        """Compute the weighted mean of a list of values."""
        if not values:
            return 0.0
        total_weight = sum(weights) or 1.0
        return sum(value * weight for value, weight in zip(values, weights)) / total_weight

    @staticmethod
    def _window_cost_drag_bps(row: Dict[str, Any], optimizer_config: OptimizerConfig) -> float:
        if row.get("cost_drag_bps") is not None:
            return float(row.get("cost_drag_bps") or 0.0)
        fees = float(row.get("fees_paid", 0.0) or 0.0)
        turnover = float(row.get("capital_turnover_rate", 0.0) or 0.0)
        notional = max(turnover * float(optimizer_config.initial_balance or 0.0), 0.0)
        if notional <= 0:
            return (fees / max(float(optimizer_config.initial_balance or 1.0), 1.0)) * 10_000
        return (fees / notional) * 10_000

    @staticmethod
    def _window_edge_score(row: Dict[str, Any], optimizer_config: OptimizerConfig) -> float:
        if row.get("edge_score") is not None:
            return float(row.get("edge_score") or 0.0)
        avg_return_bps = float(row.get("average_return_per_trade", 0.0) or 0.0) * 10_000
        return avg_return_bps - StrategyOptimizer._window_cost_drag_bps(row, optimizer_config)

    @staticmethod
    def _no_trade_reason(window_results: List[Dict[str, Any]]) -> str:
        reasons = [
            str(row.get("no_trade_reason") or "").strip()
            for row in window_results
            if str(row.get("no_trade_reason") or "").strip()
        ]
        return reasons[-1] if reasons else ""

    @staticmethod
    def _recent_score(window_results: List[Dict[str, Any]]) -> float:
        """Compute a recency weighted performance score for the most recent three windows."""
        if not window_results:
            return 0.0
        recent = window_results[-min(3, len(window_results)) :]
        return mean([
            row["total_return"] - abs(min(row["max_drawdown"], 0.0)) * 0.5 for row in recent
        ])

    @staticmethod
    def _recent_return(
        window_results: List[Dict[str, Any]],
        days: int | None = None,
        hours: int | None = None,
    ) -> float:
        """Calculate average return over recent windows."""
        if not window_results:
            return 0.0
        latest_end = StrategyOptimizer._to_datetime(int(window_results[-1]["window_end"]))
        cutoff = latest_end - (timedelta(hours=hours) if hours is not None else timedelta(days=days or 0))
        values = [
            row["total_return"]
            for row in window_results
            if StrategyOptimizer._to_datetime(int(row["window_end"])) >= cutoff
        ]
        return mean(values) if values else 0.0

    @staticmethod
    def _finite(value: float) -> float:
        """Clamp infinite values to a finite range."""
        if value == float("inf"):
            return 10.0
        if value == float("-inf"):
            return -10.0
        return float(value)

    @staticmethod
    def _candles_per_day(timeframe: str) -> int:
        """Return the number of candles per day for a given timeframe string."""
        return {
            "1m": 1_440,
            "5m": 288,
            "15m": 96,
            "1h": 24,
        }.get(timeframe, 96)

    @staticmethod
    def _to_datetime(timestamp: int) -> datetime:
        """Convert an integer timestamp (seconds or milliseconds) into a timezone aware datetime."""
        if timestamp > 10_000_000_000:
            return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    @staticmethod
    def _to_timestamp(value: datetime) -> int:
        """Convert a ``datetime`` into a Unix timestamp in milliseconds."""
        return int(value.timestamp() * 1000)

    @staticmethod
    def _config_payload(optimizer_config: OptimizerConfig) -> Dict[str, Any]:
        """Prepare a serialisable configuration payload for storage on an ``OptimizerRun`` record."""
        return {
            "profile": optimizer_config.profile,
            "training_window_days": optimizer_config.training_window_days,
            "testing_window_days": optimizer_config.testing_window_days,
            "step_days": optimizer_config.step_days,
            "training_window_hours": optimizer_config.training_window_hours,
            "testing_window_hours": optimizer_config.testing_window_hours,
            "step_hours": optimizer_config.step_hours,
            "use_full_history": optimizer_config.use_full_history,
            "recency_weighting_enabled": optimizer_config.recency_weighting_enabled,
            "decay_factor": optimizer_config.decay_factor,
            "min_trade_count": optimizer_config.min_trade_count,
            "max_drawdown_pct": optimizer_config.max_drawdown_pct,
            "max_parameter_sets": optimizer_config.max_parameter_sets,
            "allocation_amount_usd": optimizer_config.allocation_amount_usd,
            "lock_duration_hours": optimizer_config.lock_duration_hours,
            "universe_mode": optimizer_config.universe_mode,
            "max_parallel_legs": optimizer_config.max_parallel_legs,
            "allow_leverage_experiment": optimizer_config.allow_leverage_experiment,
            "min_edge_bps": optimizer_config.min_edge_bps,
            "fib_confluence_threshold": optimizer_config.fib_confluence_threshold,
            "require_shadow_validation": optimizer_config.require_shadow_validation,
            "enhanced_ensemble_enabled": optimizer_config.enhanced_ensemble_enabled,
            "ensemble_max_legs": optimizer_config.ensemble_max_legs,
            "ensemble_min_sharpe": optimizer_config.ensemble_min_sharpe,
            "ensemble_learning_decay": optimizer_config.ensemble_learning_decay,
            "experimental_duration_ensemble_enabled": optimizer_config.experimental_duration_ensemble_enabled,
            "experimental_live_eligible": optimizer_config.experimental_live_eligible,
            "ensemble_primary_metric": optimizer_config.ensemble_primary_metric,
            "duration_live_cap_usdc_by_duration": optimizer_config.duration_live_cap_usdc_by_duration,
            "duration_live_cap_pct_by_duration": optimizer_config.duration_live_cap_pct_by_duration,
            "max_return_optimizer_enabled": optimizer_config.max_return_optimizer_enabled,
            "max_return_live_eligible": optimizer_config.max_return_live_eligible,
            "max_return_min_net_return": optimizer_config.max_return_min_net_return,
            "max_return_max_drawdown_pct": optimizer_config.max_return_max_drawdown_pct,
            "market_structure_features_enabled": optimizer_config.market_structure_features_enabled,
            "market_structure_provider": optimizer_config.market_structure_provider,
            "pair_screening_enabled": optimizer_config.pair_screening_enabled,
            "pair_trading_enabled": optimizer_config.pair_trading_enabled,
            "pair_live_eligible": optimizer_config.pair_live_eligible,
            "pair_min_correlation": optimizer_config.pair_min_correlation,
            "pair_max_spread_zscore": optimizer_config.pair_max_spread_zscore,
            "dynamic_intraday_live_eligible": optimizer_config.dynamic_intraday_live_eligible,
            "high_upside_profile": optimizer_config.high_upside_profile,
        }
