"""Backtesting engine for trading strategies.

This module contains a candle‑driven backtesting engine designed to simulate
the behaviour of trading strategies under a variety of risk constraints.  It
supports multiple sizing modes, conservative or optimistic intrabar exit
models, and optional cooldown and drawdown limits.  The core loop walks
through each candle, asks a strategy for a signal, and then decides
whether to open or close a position based on both the signal and the
configured risk controls.  The engine records detailed trade objects
alongside an equity and drawdown curve to facilitate later analysis.

The original implementation bundled all of the logic into a single large
``run`` method.  This refactored version extracts common logic into
helper functions, introduces enumerations for sizing modes and intrabar
models, adds thorough type annotations, and includes docstrings for all
public methods.  These changes improve readability and ease future
maintenance without altering the external API or core functionality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from math import inf, isfinite, sqrt
from statistics import mean
from typing import Any

from ..features.engine import FeatureEngine
from ..ml.features import ML_FEATURE_SCHEMA_VERSION, MLFeatureFactory
from ..ml.online_ranker import ONE_H10_HORIZON, horizon_from_duration
from ..services.market_data import MarketDataService
from ..strategies.base import Signal
from ..strategies.registry import StrategyRegistry


class SizingMode(StrEnum):
    """Enumeration of supported position sizing modes.

    - ``FIXED_FRACTION``: Position size is a fixed fraction of current equity.
    - ``FIXED_DOLLAR``: Position size is a fixed notional amount.
    - ``RISK_BASED``: Position size is computed based on a risk per trade and
      the distance to the stop loss.
    """

    FIXED_FRACTION = "fixed_fraction"
    FIXED_DOLLAR = "fixed_dollar"
    RISK_BASED = "risk_based"

    @classmethod
    def from_str(cls, value: str) -> SizingMode:
        """Parse a sizing mode string into its corresponding enumeration.

        Args:
            value: A sizing mode string.  If empty or None, defaults to
                ``FIXED_FRACTION``.

        Returns:
            A ``SizingMode`` enumeration member.
        """
        if not value:
            return cls.FIXED_FRACTION
        try:
            return cls(value.lower())
        except ValueError:
            # Fallback to fixed fraction for unrecognised values.
            return cls.FIXED_FRACTION


class IntrabarModel(StrEnum):
    """Enumeration of intrabar exit models.

    These models dictate how the engine resolves simultaneous stop loss and
    take profit hits within the same candle when both could plausibly occur.

    - ``CONSERVATIVE``: Always assume the stop loss was hit first.
    - ``OPTIMISTIC``: Always assume the take profit was hit first.
    - ``OPEN_HIGH_LOW_CLOSE``: Mimic an OHLC reading where long positions
      favour the take profit and short positions favour the stop loss.
    - ``OPEN_LOW_HIGH_CLOSE``: The inverse of ``OPEN_HIGH_LOW_CLOSE``.
    """

    CONSERVATIVE = "conservative"
    OPTIMISTIC = "optimistic"
    OPEN_HIGH_LOW_CLOSE = "open_high_low_close"
    OPEN_LOW_HIGH_CLOSE = "open_low_high_close"

    @classmethod
    def from_str(cls, value: str) -> IntrabarModel:
        """Parse an intrabar model string into its corresponding enumeration.

        Args:
            value: A model string.  If empty or None, defaults to
                ``CONSERVATIVE``.

        Returns:
            An ``IntrabarModel`` enumeration member.
        """
        if not value:
            return cls.CONSERVATIVE
        try:
            return cls(value.lower())
        except ValueError:
            return cls.CONSERVATIVE


@dataclass(slots=True)
class BacktestConfig:
    """Input parameters required to execute a backtest.

    The fields largely mirror those in the original implementation.  See
    ``app/backtesting/engine.py`` in the prior version for descriptions of
    individual parameters.
    """

    strategy_name: str
    symbol: str
    timeframe: str
    mode: str
    initial_balance: float
    fee_bps: float
    slippage_bps: float
    stop_loss_pct: float
    take_profit_pct: float
    position_size_fraction: float
    parameters: dict[str, Any]
    sizing_mode: str = "fixed_fraction"
    fixed_dollar_size: float = 0.0
    risk_per_trade_pct: float = 0.01
    max_daily_loss: float = 0.0
    max_drawdown_pct: float = 0.0
    loss_streak_cooldown: int = 0
    cooldown_minutes: int = 0
    max_trades_per_window: int = 0
    trade_window_minutes: int = 60
    intrabar_model: str = "conservative"
    evaluation_start_timestamp: int | None = None
    runtime_deadline_monotonic: float = 0.0
    signal_history_limit: int = 0
    allocation_amount_usd: float = 0.0
    leverage: float = 1.0
    min_liquidation_buffer_pct: float = 0.0
    funding_cost_bps: float = 0.0
    funding_interval_hours: float = 8.0


@dataclass(slots=True)
class _Position:
    """Internal record of an open position.

    Attributes:
        quantity: The signed quantity of the position (positive for long,
            negative for short, zero if flat).
        entry_price: The fill price of the position after slippage and fees.
        entry_timestamp: Unix timestamp when the position was opened.
        entry_equity: Equity at entry time, used for return calculations.
        entry_features: Snapshot of features at entry for audit purposes.
        entry_signal_metadata: Metadata produced by the signal that opened
            this position.
    """

    quantity: float = 0.0
    entry_price: float = 0.0
    entry_timestamp: int = 0
    entry_equity: float = 0.0
    entry_fee: float = 0.0
    entry_notional: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    entry_features: dict[str, Any] | None = None
    entry_signal_metadata: dict[str, Any] | None = None


class BacktestEngine:
    """Simulates a trading strategy on candle data one candle at a time.

    This engine is constructed with a ``StrategyRegistry`` from which it
    instantiates strategies, a ``MarketDataService`` to fetch candle data,
    and a ``FeatureEngine`` to compute feature snapshots for audit logging.

    The primary entry point is the ``run`` method which returns a dictionary
    describing the backtest outcome.  Users can configure risk rules such
    as maximum drawdown, maximum daily loss, cooldown between trades, and
    limits on trades per time window.
    """

    #: Constant divisor used to convert basis points to decimal rates.
    BPS_DIVISOR: float = 10_000.0

    def __init__(
        self,
        config: dict[str, Any],
        registry: StrategyRegistry,
        market_data: MarketDataService,
        *,
        ml_decision_engine: Any | None = None,
        ml_feature_factory: MLFeatureFactory | None = None,
    ) -> None:
        """Create a new backtesting engine.

        Args:
            config: A dictionary of engine configuration.  Currently used for
                dashboard limits and default fixed dollar size.
            registry: A strategy registry used to instantiate strategies by
                name.
            market_data: Service responsible for fetching candle data.
        """
        self.config = config
        self.registry = registry
        self.market_data = market_data
        self.feature_engine = FeatureEngine()
        self.ml_decision_engine = ml_decision_engine
        self.ml_feature_factory = ml_feature_factory or MLFeatureFactory(config, self.feature_engine)

    # -------------------------------------------------------------------
    # Public API
    #
    def run(self, backtest: BacktestConfig, candles: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Execute a backtest for a given strategy and configuration.

        This method will fetch candle data if not provided, iterate over each
        candle (skipping the first 25 to allow indicator warm‑up), generate
        signals using the registered strategy, manage an open position subject
        to intrabar exits and risk controls, and record trade information.

        Args:
            backtest: Configuration specifying how the backtest should be run.
            candles: Optional pre‑loaded list of candle dictionaries.  Each
                candle must provide at least ``timestamp``, ``open``, ``high``,
                ``low`` and ``close`` keys.

        Returns:
            A dictionary containing summary metrics, equity curves, trade
            details and risk events.  See ``_result`` for the full schema.
        """
        self._validate_config(backtest)
        # Acquire candle data if not provided by the caller.  Honour a dashboard
        # candle limit if configured; default to a modest number to avoid
        # overwhelming the UI or memory usage.
        candles = candles or self.market_data.get_candles(
            backtest.symbol,
            backtest.timeframe,
            mode=backtest.mode,
            limit=int(self.config.get("DASHBOARD_CANDLE_LIMIT", 250)),
        )
        candles = self._prepare_candles(candles)
        # Need at least 30 candles for meaningful results; otherwise return an
        # empty result structure.
        if not candles or len(candles) < 30:
            return self._empty_result()

        # Instantiate the strategy with its provided parameters.  The registry
        # is responsible for returning a strategy instance that conforms to
        # the expected interface (has ``generate_signal`` method).
        strategy = self.registry.build(backtest.strategy_name, backtest.parameters)

        # Initial capital and rates derived from backtest configuration.
        cash: float = float(backtest.initial_balance)
        fee_rate: float = backtest.fee_bps / self.BPS_DIVISOR
        slippage_rate: float = backtest.slippage_bps / self.BPS_DIVISOR

        # Track the current open position, trades, equity and drawdown curves.
        position = _Position()
        trades: list[dict[str, Any]] = []
        equity_curve: list[dict[str, Any]] = []
        drawdown_curve: list[dict[str, Any]] = []
        returns: list[float] = []
        risk_events: list[dict[str, Any]] = []
        daily_realized: dict[str, float] = {}
        entry_timestamps: list[int] = []

        # Risk management state variables.
        loss_streak: int = 0
        cooldown_until: datetime | None = None
        total_fees: float = 0.0
        total_funding_cost: float = 0.0
        total_traded_notional: float = 0.0
        peak_equity: float = float(backtest.initial_balance)
        evaluation_started: bool = backtest.evaluation_start_timestamp is None
        last_equity: float = float(backtest.initial_balance)

        # Pre‑compute sizing mode and intrabar model as enums.  This avoids
        # repeatedly calling str.lower() on every iteration of the loop.
        sizing_mode: SizingMode = SizingMode.from_str(backtest.sizing_mode)
        intrabar_model: IntrabarModel = IntrabarModel.from_str(backtest.intrabar_model)

        # Extract frequently accessed risk parameters to local variables.
        stop_loss_pct: float = backtest.stop_loss_pct
        take_profit_pct: float = backtest.take_profit_pct
        position_size_fraction: float = backtest.position_size_fraction
        risk_per_trade_pct: float = backtest.risk_per_trade_pct
        max_daily_loss: float = backtest.max_daily_loss
        max_drawdown_pct: float = backtest.max_drawdown_pct
        loss_streak_cooldown: int = backtest.loss_streak_cooldown
        cooldown_minutes: int = backtest.cooldown_minutes
        max_trades_per_window: int = backtest.max_trades_per_window
        trade_window_minutes: int = backtest.trade_window_minutes
        fixed_dollar_size: float = backtest.fixed_dollar_size or float(self.config.get("FIXED_DOLLAR_SIZE", 0.0))
        leverage: float = max(1.0, float(backtest.leverage or 1.0))
        min_liquidation_buffer_pct: float = max(0.0, float(backtest.min_liquidation_buffer_pct or 0.0))
        runtime_deadline: float = max(0.0, float(backtest.runtime_deadline_monotonic or 0.0))
        signal_history_limit: int = max(0, int(backtest.signal_history_limit or 0))

        # Main simulation loop: start after 25th candle to allow indicator
        # values to populate for typical technical indicators.
        for index in range(25, len(candles)):
            if runtime_deadline > 0 and index % 25 == 0 and time.monotonic() >= runtime_deadline:
                risk_events.append(
                    {
                        "timestamp": int(candles[index].get("timestamp", 0)),
                        "rule": "optimizer_deadline_reached",
                        "message": "Backtest stopped early because the optimizer deadline was reached.",
                    }
                )
                break
            candle = candles[index]
            price: float = float(candle.get("close", 0.0))
            timestamp: int = int(candle.get("timestamp", 0))
            candle_time: datetime = self._timestamp_to_datetime(timestamp)

            # Determine when evaluation starts based on optional start timestamp.
            if not evaluation_started and backtest.evaluation_start_timestamp is not None:
                evaluation_started = timestamp >= backtest.evaluation_start_timestamp

            # Always update equity and drawdown using the latest data.
            equity: float = self._equity(cash, position.quantity, position.entry_price, price)
            peak_equity = max(peak_equity, equity)
            drawdown: float = ((equity - peak_equity) / peak_equity) if peak_equity else 0.0

            # Check maximum drawdown rule: exit everything and stop the simulation.
            if max_drawdown_pct > 0 and drawdown <= -abs(max_drawdown_pct):
                risk_events.append(
                    {
                        "timestamp": timestamp,
                        "rule": "max_drawdown_kill_switch",
                        "message": "Max drawdown threshold reached; simulation stopped.",
                        "drawdown": drawdown,
                    }
                )
                if position.quantity != 0:
                    # Close open position at current price when drawdown threshold is hit.
                    pnl, exit_fee, adjusted_exit = self._close_position(
                        position.quantity,
                        position.entry_price,
                        price,
                        fee_rate,
                        slippage_rate,
                    )
                    funding_fee = self._funding_cost(position, timestamp, backtest)
                    settlement = self._settle_close(position, cash, pnl, exit_fee, funding_fee)
                    cash += settlement["cash_delta"]
                    total_fees += settlement["exit_fee"]
                    total_funding_cost += funding_fee
                    total_traded_notional += abs(position.quantity * adjusted_exit)
                    if settlement["balance_floor_applied"]:
                        risk_events.append(self._balance_floor_event(timestamp, settlement))
                    if evaluation_started:
                        duration_minutes = self._duration_minutes(position.entry_timestamp, timestamp)
                        trades.append(
                            {
                                "direction": "long" if position.quantity > 0 else "short",
                                "entry_price": position.entry_price,
                                "exit_price": adjusted_exit,
                                "pnl": settlement["pnl"],
                                "uncapped_pnl": settlement["uncapped_pnl"],
                                "gross_pnl": pnl,
                                "fee": settlement["fee"],
                                "entry_fee": settlement["entry_fee"],
                                "exit_fee": settlement["exit_fee"],
                                "funding_fee": settlement["funding_fee"],
                                "balance_floor_applied": settlement["balance_floor_applied"],
                                "cash_floor_adjustment": settlement["cash_floor_adjustment"],
                                "return": settlement["pnl"] / max(position.entry_equity, 1e-9),
                                "reason": "max_drawdown_kill_switch",
                                "entry_timestamp": position.entry_timestamp,
                                "timestamp": timestamp,
                                "duration_minutes": duration_minutes,
                                "notional": abs(position.quantity * position.entry_price),
                                "entry_features": position.entry_features or {},
                                "exit_features": {},
                                "fibonacci_levels": (position.entry_features or {}).get("fibonacci_levels", {}),
                                "external_scores": (position.entry_features or {}).get("external_scores", {}),
                                "pattern_prediction": (position.entry_features or {}).get("pattern_prediction", {}),
                                "rule_decision": (position.entry_signal_metadata or {}).get("rule_decision", {}),
                                "rule_score": (position.entry_signal_metadata or {}).get("rule_decision", {}).get("score", 0.0),
                                "ml_signal_decision": (position.entry_signal_metadata or {}).get("ml_signal_decision", {}),
                                "ml_feature_schema_version": (position.entry_signal_metadata or {}).get("ml_feature_schema_version", ""),
                                "leverage": (position.entry_signal_metadata or {}).get("leverage", 1.0),
                                "liquidation_buffer_pct": (position.entry_signal_metadata or {}).get("liquidation_buffer_pct", 1.0),
                            }
                        )
                    position = _Position()
                break

            # Optimizer runs can cap history to keep short-interval research bounded.
            history_start = max(0, index + 1 - signal_history_limit) if signal_history_limit else 0
            signal_history = candles[history_start : index + 1]

            # Generate a signal from the strategy using candle history up to the current one.
            signal = strategy.generate_signal(
                symbol=backtest.symbol,
                timeframe=backtest.timeframe,
                candles=signal_history,
                position={"quantity": position.quantity, "entry_price": position.entry_price},
            )
            feature_payload = self._feature_payload(backtest.symbol, backtest.timeframe, signal_history, signal)
            signal = self._ml_first_signal(backtest, signal, signal_history, feature_payload)
            feature_payload = self._feature_payload(backtest.symbol, backtest.timeframe, signal_history, signal)
            signal_metadata: dict[str, Any] = getattr(signal, "metadata", {}) or {}

            # Handle exiting an open position before considering new entries.
            if position.quantity != 0:
                exit_price, exit_reason = self._intrabar_exit(
                    candle,
                    position.entry_price,
                    position.quantity,
                    stop_loss_pct,
                    take_profit_pct,
                    intrabar_model,
                    stop_loss_price=position.stop_loss,
                    take_profit_price=position.take_profit,
                )
                # Honour signal reductions or flips if no stop/take levels were hit.
                if exit_price is None and signal.action == "reduce":
                    exit_price, exit_reason = price, "signal_reduce"
                elif exit_price is None and (
                    (position.quantity > 0 and signal.action == "sell") or (position.quantity < 0 and signal.action == "buy")
                ):
                    exit_price, exit_reason = price, "signal_flip"

                # Close the position if an exit condition was triggered.
                if exit_price is not None:
                    pnl, exit_fee, adjusted_exit = self._close_position(
                        position.quantity,
                        position.entry_price,
                        exit_price,
                        fee_rate,
                        slippage_rate,
                    )
                    funding_fee = self._funding_cost(position, timestamp, backtest)
                    settlement = self._settle_close(position, cash, pnl, exit_fee, funding_fee)
                    cash += settlement["cash_delta"]
                    total_fees += settlement["exit_fee"]
                    total_funding_cost += funding_fee
                    total_traded_notional += abs(position.quantity * adjusted_exit)
                    if settlement["balance_floor_applied"]:
                        risk_events.append(self._balance_floor_event(timestamp, settlement))
                    day = candle_time.date().isoformat()
                    daily_realized[day] = daily_realized.get(day, 0.0) + settlement["pnl"]
                    duration_minutes = self._duration_minutes(position.entry_timestamp, timestamp)
                    trade_return = settlement["pnl"] / max(position.entry_equity, 1e-9)
                    if evaluation_started:
                        trades.append(
                            {
                                "direction": "long" if position.quantity > 0 else "short",
                                "entry_price": position.entry_price,
                                "exit_price": adjusted_exit,
                                "pnl": settlement["pnl"],
                                "uncapped_pnl": settlement["uncapped_pnl"],
                                "gross_pnl": pnl,
                                "fee": settlement["fee"],
                                "entry_fee": settlement["entry_fee"],
                                "exit_fee": settlement["exit_fee"],
                                "funding_fee": settlement["funding_fee"],
                                "balance_floor_applied": settlement["balance_floor_applied"],
                                "cash_floor_adjustment": settlement["cash_floor_adjustment"],
                                "return": trade_return,
                                "reason": exit_reason,
                                "entry_timestamp": position.entry_timestamp,
                                "timestamp": timestamp,
                                "duration_minutes": duration_minutes,
                                "notional": abs(position.quantity * position.entry_price),
                                "entry_features": position.entry_features or {},
                                "exit_features": feature_payload,
                                "fibonacci_levels": (position.entry_features or {}).get("fibonacci_levels", {}),
                                "external_scores": (position.entry_features or {}).get("external_scores", {}),
                                "pattern_prediction": (position.entry_features or {}).get("pattern_prediction", {}),
                                "rule_decision": (position.entry_signal_metadata or {}).get("rule_decision", {}),
                                "rule_score": (position.entry_signal_metadata or {}).get("rule_decision", {}).get("score", 0.0),
                                "ml_signal_decision": (position.entry_signal_metadata or {}).get("ml_signal_decision", {}),
                                "ml_feature_schema_version": (position.entry_signal_metadata or {}).get("ml_feature_schema_version", ""),
                                "leverage": (position.entry_signal_metadata or {}).get("leverage", 1.0),
                                "liquidation_buffer_pct": (position.entry_signal_metadata or {}).get("liquidation_buffer_pct", 1.0),
                            }
                        )
                    # Update loss streak and handle loss streak cooldown logic.
                    loss_streak = loss_streak + 1 if settlement["pnl"] < 0 else 0
                    if loss_streak_cooldown > 0 and loss_streak >= loss_streak_cooldown and cooldown_minutes > 0:
                        cooldown_until = candle_time + timedelta(minutes=cooldown_minutes)
                        risk_events.append(
                            {
                                "timestamp": timestamp,
                                "rule": "loss_streak_cooldown",
                                "message": f"Loss streak reached {loss_streak}; entries paused.",
                            }
                        )
                    # Reset position after closing.
                    position = _Position()

            # Evaluate the possibility of entering a new position if we are flat.
            can_enter: bool = (
                evaluation_started
                and position.quantity == 0
                and signal.action in {"buy", "sell"}
                and not self._daily_loss_reached(daily_realized, candle_time, max_daily_loss)
                and not self._cooling_down(cooldown_until, candle_time)
                and not self._trade_window_full(entry_timestamps, timestamp, max_trades_per_window, trade_window_minutes)
            )
            if can_enter:
                qty = self._position_quantity(
                    sizing_mode,
                    price,
                    cash,
                    equity,
                    stop_loss_pct,
                    risk_per_trade_pct,
                    position_size_fraction,
                    fixed_dollar_size,
                    signal,
                    leverage,
                )
                if signal.action == "sell":
                    qty *= -1
                if abs(qty) > 1e-12:
                    entry_price: float = price * (1 + slippage_rate if qty > 0 else 1 - slippage_rate)
                    liquidation_buffer = self._liquidation_buffer_pct(entry_price, qty, signal.stop_loss, stop_loss_pct, leverage)
                    if liquidation_buffer < min_liquidation_buffer_pct:
                        risk_events.append(
                            {
                                "timestamp": timestamp,
                                "rule": "liquidation_buffer_too_tight",
                                "message": "Entry blocked because leverage left too little liquidation buffer.",
                                "liquidation_buffer_pct": liquidation_buffer,
                                "minimum_buffer_pct": min_liquidation_buffer_pct,
                                "leverage": leverage,
                            }
                        )
                        qty = 0.0
                    if abs(qty) <= 1e-12:
                        pass
                    else:
                        entry_fee: float = abs(qty * entry_price) * fee_rate
                        if cash - entry_fee < -1e-9:
                            risk_events.append(
                                {
                                    "timestamp": timestamp,
                                    "rule": "entry_fee_exceeds_cash",
                                    "message": "Entry blocked because fees would make simulated cash negative.",
                                    "entry_fee": entry_fee,
                                    "cash": cash,
                                }
                            )
                            continue
                        cash -= entry_fee
                        total_fees += entry_fee
                        total_traded_notional += abs(qty * entry_price)
                        entry_metadata = dict(signal_metadata)
                        entry_metadata["liquidation_buffer_pct"] = liquidation_buffer
                        entry_metadata["leverage"] = leverage
                        entry_metadata["expected_stop_loss"] = signal.stop_loss
                        entry_metadata["expected_take_profit"] = signal.take_profit
                        position = _Position(
                            quantity=qty,
                            entry_price=entry_price,
                            entry_timestamp=timestamp,
                            entry_equity=max(equity, 1e-9),
                            entry_fee=entry_fee,
                            entry_notional=abs(qty * entry_price),
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            entry_features=feature_payload,
                            entry_signal_metadata=entry_metadata,
                        )
                        entry_timestamps.append(timestamp)
            elif evaluation_started and position.quantity == 0 and signal.action in {"buy", "sell"}:
                # Record why an entry was blocked for audit purposes.
                reason = self._blocked_entry_reason(
                    daily_realized,
                    candle_time,
                    cooldown_until,
                    entry_timestamps,
                    timestamp,
                    max_trades_per_window,
                    trade_window_minutes,
                    max_daily_loss,
                )
                if reason:
                    risk_events.append({"timestamp": timestamp, "rule": reason, "message": "Entry blocked by risk control."})

            # Append equity and drawdown data for this candle if evaluation has begun.
            equity = self._equity(cash, position.quantity, position.entry_price, price)
            if evaluation_started:
                equity_point = {
                    "timestamp": timestamp,
                    "cash": round(cash, 6),
                    "equity": round(equity, 6),
                    "position_value": round(abs(position.quantity * price), 6),
                    "realized_pnl": round(cash - backtest.initial_balance, 6),
                    "unrealized_pnl": round(equity - cash, 6),
                }
                equity_curve.append(equity_point)
                # Compute drawdown relative to the peak observed on the equity curve so far.
                curve_peak = max(point["equity"] for point in equity_curve) if equity_curve else equity
                curve_drawdown = ((equity - curve_peak) / curve_peak) if curve_peak else 0.0
                drawdown_curve.append({"timestamp": timestamp, "drawdown": round(curve_drawdown, 6)})
                if len(equity_curve) > 1:
                    returns.append((equity - last_equity) / last_equity if last_equity else 0.0)
                last_equity = equity

        # If nothing was recorded in the equity curve then bail out early.
        if not equity_curve:
            return self._empty_result()

        # Liquidate any remaining position at the final candle so exits include
        # realistic closing fees and held-position funding cost.
        final_price: float = float(candles[-1]["close"])
        if position.quantity != 0 and equity_curve:
            final_timestamp = int(candles[-1].get("timestamp", 0))
            pnl, exit_fee, adjusted_exit = self._close_position(
                position.quantity,
                position.entry_price,
                final_price,
                fee_rate,
                slippage_rate,
            )
            funding_fee = self._funding_cost(position, final_timestamp, backtest)
            settlement = self._settle_close(position, cash, pnl, exit_fee, funding_fee)
            cash += settlement["cash_delta"]
            total_fees += settlement["exit_fee"]
            total_funding_cost += funding_fee
            total_traded_notional += abs(position.quantity * adjusted_exit)
            if settlement["balance_floor_applied"]:
                risk_events.append(self._balance_floor_event(final_timestamp, settlement))
            if evaluation_started:
                duration_minutes = self._duration_minutes(position.entry_timestamp, final_timestamp)
                trades.append(
                    {
                        "direction": "long" if position.quantity > 0 else "short",
                        "entry_price": position.entry_price,
                        "exit_price": adjusted_exit,
                        "pnl": settlement["pnl"],
                        "uncapped_pnl": settlement["uncapped_pnl"],
                        "gross_pnl": pnl,
                        "fee": settlement["fee"],
                        "entry_fee": settlement["entry_fee"],
                        "exit_fee": settlement["exit_fee"],
                        "funding_fee": settlement["funding_fee"],
                        "balance_floor_applied": settlement["balance_floor_applied"],
                        "cash_floor_adjustment": settlement["cash_floor_adjustment"],
                        "return": settlement["pnl"] / max(position.entry_equity, 1e-9),
                        "reason": "final_position_liquidation",
                        "entry_timestamp": position.entry_timestamp,
                        "timestamp": final_timestamp,
                        "duration_minutes": duration_minutes,
                        "notional": abs(position.quantity * position.entry_price),
                        "entry_features": position.entry_features or {},
                        "exit_features": {},
                        "fibonacci_levels": (position.entry_features or {}).get("fibonacci_levels", {}),
                        "external_scores": (position.entry_features or {}).get("external_scores", {}),
                        "pattern_prediction": (position.entry_features or {}).get("pattern_prediction", {}),
                        "rule_decision": (position.entry_signal_metadata or {}).get("rule_decision", {}),
                        "rule_score": (position.entry_signal_metadata or {}).get("rule_decision", {}).get("score", 0.0),
                        "ml_signal_decision": (position.entry_signal_metadata or {}).get("ml_signal_decision", {}),
                        "ml_feature_schema_version": (position.entry_signal_metadata or {}).get("ml_feature_schema_version", ""),
                        "leverage": (position.entry_signal_metadata or {}).get("leverage", 1.0),
                        "liquidation_buffer_pct": (position.entry_signal_metadata or {}).get("liquidation_buffer_pct", 1.0),
                    }
                )
            risk_events.append(
                {
                    "timestamp": final_timestamp,
                    "rule": "final_position_liquidation",
                    "message": "Open position was liquidated at the final candle for after-cost accounting.",
                }
            )
            position = _Position()
            equity_curve[-1]["cash"] = round(cash, 6)
            equity_curve[-1]["equity"] = round(cash, 6)
            equity_curve[-1]["realized_pnl"] = round(cash - backtest.initial_balance, 6)
            equity_curve[-1]["unrealized_pnl"] = 0.0
            equity_curve[-1]["position_value"] = 0.0
        final_equity: float = self._equity(cash, position.quantity, position.entry_price, final_price)

        return self._result(
            backtest=backtest,
            cash=cash,
            final_equity=final_equity,
            total_fees=total_fees,
            total_funding_cost=total_funding_cost,
            total_traded_notional=total_traded_notional,
            trades=trades,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            returns=returns,
            risk_events=risk_events,
        )

    # -------------------------------------------------------------------
    # Internal helpers
    #
    @staticmethod
    def _validate_config(backtest: BacktestConfig) -> None:
        """Reject invalid backtest inputs before the simulation mutates state."""
        if not str(backtest.strategy_name or "").strip():
            raise ValueError("Strategy name is required.")
        if not str(backtest.symbol or "").strip():
            raise ValueError("Symbol is required.")
        if not str(backtest.timeframe or "").strip():
            raise ValueError("Timeframe is required.")
        numeric_fields = {
            "initial_balance": backtest.initial_balance,
            "fee_bps": backtest.fee_bps,
            "slippage_bps": backtest.slippage_bps,
            "stop_loss_pct": backtest.stop_loss_pct,
            "take_profit_pct": backtest.take_profit_pct,
            "position_size_fraction": backtest.position_size_fraction,
            "risk_per_trade_pct": backtest.risk_per_trade_pct,
            "max_daily_loss": backtest.max_daily_loss,
            "max_drawdown_pct": backtest.max_drawdown_pct,
            "leverage": backtest.leverage,
            "funding_cost_bps": backtest.funding_cost_bps,
            "funding_interval_hours": backtest.funding_interval_hours,
        }
        for name, value in numeric_fields.items():
            parsed = float(value or 0.0)
            if not isfinite(parsed):
                raise ValueError(f"{name} must be finite.")
        if float(backtest.initial_balance) <= 0:
            raise ValueError("Starting balance must be greater than zero.")
        if float(backtest.fee_bps) < 0 or float(backtest.slippage_bps) < 0:
            raise ValueError("Fees and slippage cannot be negative.")
        if float(backtest.stop_loss_pct) < 0 or float(backtest.take_profit_pct) < 0:
            raise ValueError("Stop loss and take profit percentages cannot be negative.")
        if float(backtest.position_size_fraction) < 0 or float(backtest.risk_per_trade_pct) < 0:
            raise ValueError("Position sizing and risk settings cannot be negative.")
        if float(backtest.leverage or 1.0) < 1.0:
            raise ValueError("Leverage must be at least 1x.")
        if float(backtest.funding_cost_bps or 0.0) < 0 or float(backtest.funding_interval_hours) <= 0:
            raise ValueError("Funding costs and intervals must be positive.")

    @staticmethod
    def _prepare_candles(candles: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Normalize candle inputs and discard rows that cannot be simulated."""
        prepared: list[dict[str, Any]] = []
        for index, candle in enumerate(candles or []):
            if not isinstance(candle, dict):
                continue
            try:
                timestamp = int(candle.get("timestamp", index))
                open_price = float(candle.get("open", candle.get("close", 0.0)))
                high = float(candle.get("high", open_price))
                low = float(candle.get("low", open_price))
                close = float(candle.get("close", open_price))
            except (TypeError, ValueError):
                continue
            if not all(isfinite(value) and value > 0 for value in (open_price, high, low, close)):
                continue
            normalized_high = max(high, open_price, low, close)
            normalized_low = min(low, open_price, high, close)
            prepared.append(
                {
                    **candle,
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": normalized_high,
                    "low": normalized_low,
                    "close": close,
                }
            )
        prepared.sort(key=lambda row: int(row.get("timestamp", 0)))
        return prepared

    def _result(
        self,
        *,
        backtest: BacktestConfig,
        cash: float,
        final_equity: float,
        total_fees: float,
        total_funding_cost: float,
        total_traded_notional: float,
        trades: list[dict[str, Any]],
        equity_curve: list[dict[str, Any]],
        drawdown_curve: list[dict[str, Any]],
        returns: list[float],
        risk_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble the final result dictionary from backtest components."""
        funding_cost_estimate = max(float(total_funding_cost or 0.0), 0.0)
        adjusted_final_equity = final_equity
        total_return = ((adjusted_final_equity - backtest.initial_balance) / backtest.initial_balance) if backtest.initial_balance else 0.0
        wins = [trade for trade in trades if trade["pnl"] > 0]
        losses = [trade for trade in trades if trade["pnl"] < 0]
        gross_profit = sum(trade["pnl"] for trade in wins)
        gross_loss = abs(sum(trade["pnl"] for trade in losses))
        net_pnl = adjusted_final_equity - backtest.initial_balance
        gross_pnl = sum(float(trade.get("gross_pnl", 0.0) or 0.0) for trade in trades)
        average_trade = mean([trade["pnl"] for trade in trades]) if trades else 0.0
        durations = [trade["duration_minutes"] for trade in trades if trade.get("duration_minutes") is not None]
        trade_returns = [trade["return"] for trade in trades]
        avg_win = mean([trade["pnl"] for trade in wins]) if wins else 0.0
        avg_loss = abs(mean([trade["pnl"] for trade in losses])) if losses else 0.0
        win_rate = (len(wins) / len(trades)) if trades else 0.0
        loss_rate = 1.0 - win_rate if trades else 0.0
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss) if trades else 0.0
        cost_drag_bps = (total_fees / total_traded_notional * self.BPS_DIVISOR) if total_traded_notional > 0 else 0.0
        avg_return_bps = (mean(trade_returns) * self.BPS_DIVISOR) if trade_returns else 0.0
        edge_score = avg_return_bps - cost_drag_bps
        elapsed_days = self._elapsed_days(equity_curve)
        avg_equity = mean([point["equity"] for point in equity_curve]) if equity_curve else backtest.initial_balance
        turnover_after_fees = max(total_traded_notional - total_fees - funding_cost_estimate, 0.0) / max(avg_equity, 1e-9)

        return {
            "total_return": total_return,
            "net_return_after_costs": total_return,
            "max_drawdown": self._max_drawdown(equity_curve),
            "win_rate": win_rate,
            "sharpe_like": self._sharpe_like(returns),
            "sortino_like": self._sortino_like(returns),
            "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else (inf if gross_profit > 0 else 0.0),
            "trade_count": len(trades),
            "closed_trade_count": len(trades),
            "open_trade_count": 0,
            "trades_per_day": (len(trades) / elapsed_days) if elapsed_days > 0 else 0.0,
            "average_trade_duration_minutes": mean(durations) if durations else 0.0,
            "average_trade": average_trade,
            "capital_turnover_rate": total_traded_notional / max(avg_equity, 1e-9),
            "turnover_after_fees": turnover_after_fees,
            "average_return_per_trade": mean(trade_returns) if trade_returns else 0.0,
            "edge_score": edge_score,
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": (avg_win / avg_loss) if avg_loss > 0 else 0.0,
            "cost_drag_bps": cost_drag_bps,
            "max_adverse_excursion": min(trade_returns) if trade_returns else 0.0,
            "max_favorable_excursion": max(trade_returns) if trade_returns else 0.0,
            "no_trade_reason": "" if trades else "no_trades_after_costs",
            "net_pnl": net_pnl,
            "gross_pnl": gross_pnl,
            "realized_pnl": cash - backtest.initial_balance,
            "unrealized_pnl": final_equity - cash,
            "fees_paid": total_fees,
            "funding_cost_estimate": funding_cost_estimate,
            "final_cash": cash,
            "final_equity": adjusted_final_equity,
            "raw_final_equity": final_equity,
            "leverage": max(1.0, float(backtest.leverage or 1.0)),
            "allocation_amount_usd": float(backtest.allocation_amount_usd or 0.0),
            "liquidation_buffer_pct": self._minimum_trade_liquidation_buffer(trades),
            "equity_curve": equity_curve,
            "drawdown_curve": drawdown_curve,
            "trades": trades,
            "risk_events": risk_events,
            "risk_event_count": len(risk_events),
            "feature_audit_summary": self._feature_audit_summary(trades),
            "ml_feature_schema_version": ML_FEATURE_SCHEMA_VERSION,
            "ml_first_enabled": bool(self.config.get("ML_FIRST_STRATEGIES_ENABLED", False)),
            "ml_decision_trade_count": len([trade for trade in trades if trade.get("ml_signal_decision")]),
        }

    def _ml_first_signal(
        self,
        backtest: BacktestConfig,
        signal: Signal,
        candles: list[dict[str, Any]],
        feature_payload: dict[str, Any],
    ) -> Signal:
        """Use promoted ML as the final signal source when ML-first mode is enabled.

        The deterministic strategy still provides baseline features. If ML is
        unavailable, stale, low confidence, or missing exits, the backtest emits
        ``hold`` instead of falling back into an automatic trade.
        """

        if not bool(self.config.get("ML_FIRST_STRATEGIES_ENABLED", False)):
            return signal
        metadata = dict(getattr(signal, "metadata", {}) or {})
        parameters = dict(backtest.parameters or {})
        explicit_horizon = str(parameters.get("ml_horizon") or parameters.get("horizon") or "").strip().lower()
        if explicit_horizon:
            horizon = explicit_horizon
        elif bool(parameters.get("one_h10_vault")) or str(parameters.get("vault_cycle_duration") or "").lower() == ONE_H10_HORIZON:
            horizon = ONE_H10_HORIZON
        else:
            horizon = horizon_from_duration(parameters.get("lock_duration_hours") or 1)
        if self.ml_decision_engine is None:
            metadata.update(
                {
                    "ml_feature_schema_version": ML_FEATURE_SCHEMA_VERSION,
                    "ml_safety_blockers": ["ml_decision_engine_unavailable"],
                    "no_trade_reason": "ml_first_signal_blocked:ml_decision_engine_unavailable",
                }
            )
            return Signal("hold", metadata["no_trade_reason"], backtest.timeframe, None, None, 0.0, metadata)

        context = self.ml_feature_factory.build(
            symbol=backtest.symbol,
            timeframe=backtest.timeframe,
            candles=candles,
            deterministic_signal=signal,
            optimizer_context={
                **dict(backtest.parameters or {}),
                **dict(feature_payload or {}),
                "strategy_name": backtest.strategy_name,
                "mode": backtest.mode,
                "horizon": horizon,
            },
            cutoff_timestamp=(candles[-1] if candles else {}).get("timestamp"),
        )
        try:
            decision = dict(
                self.ml_decision_engine.decision(
                    "pytorch_gru_signal",
                    context,
                    horizon=horizon,
                    candles=candles,
                )
            )
        except Exception as exc:  # noqa: BLE001
            decision = {
                "ready": False,
                "action": "hold",
                "blockers": [f"ml_signal_error:{exc}"],
                "raw": {"ready_for_live": False, "blockers": [str(exc)]},
            }
        raw = dict(decision.get("raw") or {}) if isinstance(decision.get("raw"), dict) else {}
        blockers = list(decision.get("blockers", []) or []) + list(raw.get("blockers", []) or [])
        metadata.update(
            {
                "feature_snapshot": feature_payload,
                "ml_feature_schema_version": ML_FEATURE_SCHEMA_VERSION,
                "ml_signal_decision": decision,
                "ml_signal_model": raw,
                "ml_signal_quality": raw,
                "ml_safety_blockers": list(dict.fromkeys(blockers)),
                "ml_first_strategy_enabled": True,
            }
        )
        min_confidence = float(self.config.get("ML_MIN_SIGNAL_CONFIDENCE", self.config.get("ML_SIGNAL_MIN_CONFIDENCE", 0.60)) or 0.60)
        confidence = self._safe_float(raw.get("confidence", decision.get("confidence", 0.0)))
        action = str(raw.get("action") or decision.get("action") or "hold").lower()
        if not bool(raw.get("ready_for_live", False)) or not bool(decision.get("ready", False)):
            reason = "ml_first_signal_blocked:" + ",".join(list(dict.fromkeys(blockers)) or ["not_ready"])
            metadata["no_trade_reason"] = reason
            return Signal("hold", reason, backtest.timeframe, None, None, 0.0, metadata)
        if confidence < min_confidence:
            reason = "ml_first_signal_blocked:low_confidence"
            metadata["no_trade_reason"] = reason
            metadata["ml_safety_blockers"] = list(dict.fromkeys([*metadata["ml_safety_blockers"], "low_confidence"]))
            return Signal("hold", reason, backtest.timeframe, None, None, 0.0, metadata)
        if action not in {"buy", "sell"}:
            metadata["no_trade_reason"] = "ml_signal_hold"
            return Signal("hold", "ML signal selected hold.", backtest.timeframe, None, None, 0.0, metadata)
        mid = self._safe_float((candles[-1] if candles else {}).get("close"), 0.0)
        stop_pct = max(self._safe_float(raw.get("suggested_stop_loss_pct")), 0.0)
        take_pct = max(self._safe_float(raw.get("suggested_take_profit_pct")), 0.0)
        if mid <= 0 or stop_pct <= 0 or take_pct <= 0:
            reason = "ml_first_signal_blocked:missing_price_or_exits"
            metadata["no_trade_reason"] = reason
            metadata["ml_safety_blockers"] = list(dict.fromkeys([*metadata["ml_safety_blockers"], "missing_price_or_exits"]))
            return Signal("hold", reason, backtest.timeframe, None, None, 0.0, metadata)
        stop_loss = mid * (1 - stop_pct) if action == "buy" else mid * (1 + stop_pct)
        take_profit = mid * (1 + take_pct) if action == "buy" else mid * (1 - take_pct)
        base_fraction = self._safe_float(getattr(signal, "position_fraction", 0.0), 0.0)
        ml_fraction = self._safe_float(raw.get("position_fraction", raw.get("sizing_score")), 0.0)
        position_fraction = max(0.0, min(base_fraction if base_fraction > 0 else 1.0, ml_fraction, 1.0))
        if position_fraction <= 0:
            reason = "ml_first_signal_blocked:zero_sizing"
            metadata["no_trade_reason"] = reason
            return Signal("hold", reason, backtest.timeframe, None, None, 0.0, metadata)
        return Signal(
            action,
            f"ML-first signal selected {action}.",
            backtest.timeframe,
            stop_loss,
            take_profit,
            position_fraction,
            metadata,
        )

    def _position_quantity(self, sizing_mode: SizingMode | BacktestConfig, *args: Any, **kwargs: Any) -> float:
        """Calculate the quantity to trade given the sizing mode and risk parameters."""
        if isinstance(sizing_mode, BacktestConfig):
            backtest = sizing_mode
            signal = args[0] if args else kwargs.get("signal")
            sizing_mode = SizingMode.from_str(backtest.sizing_mode)
            price = float(kwargs["price"])
            cash = float(kwargs["cash"])
            equity = float(kwargs["equity"])
            stop_loss_pct = backtest.stop_loss_pct
            risk_per_trade_pct = backtest.risk_per_trade_pct
            position_size_fraction = backtest.position_size_fraction
            fixed_dollar_size = backtest.fixed_dollar_size
            leverage = float(backtest.leverage or 1.0)
        else:
            (
                price,
                cash,
                equity,
                stop_loss_pct,
                risk_per_trade_pct,
                position_size_fraction,
                fixed_dollar_size,
                signal,
                leverage,
            ) = args
        if sizing_mode is SizingMode.FIXED_DOLLAR:
            notional = fixed_dollar_size * max(float(leverage or 1.0), 1.0)
        elif sizing_mode is SizingMode.RISK_BASED:
            # Determine stop distance from either explicit signal stop or the configured percentage.
            stop_price = getattr(signal, "stop_loss", None) or self._position_stop(price, 1, stop_loss_pct)
            stop_distance = abs(price - float(stop_price))
            risk_amount = equity * max(risk_per_trade_pct, 0.0)
            qty = risk_amount / max(stop_distance, 1e-9)
            cap_notional = equity * max(position_size_fraction, 0.0) * max(float(leverage or 1.0), 1.0)
            return min(qty, cap_notional / max(price, 1e-9))
        else:
            notional = equity * max(position_size_fraction, 0.0) * max(float(leverage or 1.0), 1.0)
        # Clamp the notional so we never buy more than our cash or equity allows.  Use a
        # small epsilon to avoid division by zero when price is zero.
        return max(min(notional, max(cash, equity, 0.0) * max(float(leverage or 1.0), 1.0)) / max(price, 1e-9), 0.0)

    def _intrabar_exit(
        self,
        candle: dict[str, Any],
        entry_price: float,
        quantity: float,
        stop_pct: float,
        take_pct: float,
        model: IntrabarModel,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> tuple[float | None, str | None]:
        """Determine intrabar exit price and reason if stop loss or take profit is hit.

        Args:
            candle: The current candle containing ``high`` and ``low`` values.
            entry_price: Entry price of the open position.
            quantity: Signed quantity; positive for long, negative for short.
            stop_pct: Stop loss percentage expressed as a fraction of entry price.
            take_pct: Take profit percentage expressed as a fraction of entry price.
            model: Which intrabar model to apply when both stop and take would be hit.

        Returns:
            A tuple ``(exit_price, reason)``.  If neither level was hit, both
            values will be ``None``.  Possible reasons include ``"stop_loss"``,
            ``"take_profit"``, and behaviour from the model if both are hit.
        """
        stop_loss = self._positive_or_default(stop_loss_price, self._position_stop(entry_price, quantity, stop_pct))
        take_profit = self._positive_or_default(take_profit_price, self._position_take(entry_price, quantity, take_pct))
        high = float(candle.get("high", 0.0))
        low = float(candle.get("low", 0.0))
        stop_hit = low <= stop_loss if quantity > 0 else high >= stop_loss
        take_hit = high >= take_profit if quantity > 0 else low <= take_profit
        if not stop_hit and not take_hit:
            return None, None
        if stop_hit and not take_hit:
            return stop_loss, "stop_loss"
        if take_hit and not stop_hit:
            return take_profit, "take_profit"
        # If both are hit on the same candle, the model dictates the outcome.
        if model is IntrabarModel.OPTIMISTIC:
            return take_profit, "take_profit"
        if model is IntrabarModel.OPEN_HIGH_LOW_CLOSE:
            return (take_profit, "take_profit") if quantity > 0 else (stop_loss, "stop_loss")
        if model is IntrabarModel.OPEN_LOW_HIGH_CLOSE:
            return (stop_loss, "stop_loss") if quantity > 0 else (take_profit, "take_profit")
        # Default conservative behaviour: assume the stop loss was hit first.
        return stop_loss, "stop_loss"

    @staticmethod
    def _position_stop(entry_price: float, quantity: float, stop_pct: float) -> float:
        """Calculate the stop loss price for a given entry and quantity."""
        return entry_price * (1 - stop_pct) if quantity > 0 else entry_price * (1 + stop_pct)

    @staticmethod
    def _position_take(entry_price: float, quantity: float, take_pct: float) -> float:
        """Calculate the take profit price for a given entry and quantity."""
        return entry_price * (1 + take_pct) if quantity > 0 else entry_price * (1 - take_pct)

    @staticmethod
    def _positive_or_default(value: float | None, default: float) -> float:
        try:
            parsed = float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            parsed = 0.0
        return parsed if parsed > 0 else default

    def _liquidation_buffer_pct(
        self,
        entry_price: float,
        quantity: float,
        stop_loss: float | None,
        stop_pct: float,
        leverage: float,
    ) -> float:
        leverage = max(float(leverage or 1.0), 1.0)
        if leverage <= 1.0 or entry_price <= 0:
            return 1.0
        liquidation_price = entry_price * (1 - (1 / leverage)) if quantity > 0 else entry_price * (1 + (1 / leverage))
        stop_price = float(stop_loss or self._position_stop(entry_price, quantity, stop_pct))
        return max(0.0, abs(stop_price - liquidation_price) / entry_price)

    @staticmethod
    def _minimum_trade_liquidation_buffer(trades: list[dict[str, Any]]) -> float:
        values = [float(trade.get("liquidation_buffer_pct", 1.0) or 0.0) for trade in trades]
        return min(values) if values else 1.0

    @staticmethod
    def _close_position(
        quantity: float,
        entry_price: float,
        exit_price: float,
        fee_rate: float,
        slippage_rate: float,
    ) -> tuple[float, float, float]:
        """Compute gross PnL, fees and slippage for closing a position."""
        adjusted_exit = exit_price * (1 - slippage_rate if quantity > 0 else 1 + slippage_rate)
        gross = quantity * (adjusted_exit - entry_price) if quantity > 0 else abs(quantity) * (entry_price - adjusted_exit)
        fee = abs(quantity * adjusted_exit) * fee_rate
        return gross, fee, adjusted_exit

    @staticmethod
    def _settle_close(
        position: _Position,
        cash: float,
        gross_pnl: float,
        exit_fee: float,
        funding_fee: float,
    ) -> dict[str, Any]:
        """Return cash settlement and trade-ledger values for a closing fill."""
        entry_fee = max(float(position.entry_fee or 0.0), 0.0)
        fee = entry_fee + max(float(exit_fee or 0.0), 0.0)
        funding = max(float(funding_fee or 0.0), 0.0)
        uncapped_cash_delta = float(gross_pnl) - max(float(exit_fee or 0.0), 0.0) - funding
        cash_delta = uncapped_cash_delta
        projected_cash = float(cash) + cash_delta
        balance_floor_applied = projected_cash < 0.0
        cash_floor_adjustment = abs(projected_cash) if balance_floor_applied else 0.0
        if balance_floor_applied:
            cash_delta += cash_floor_adjustment
        return {
            "cash_delta": cash_delta,
            "pnl": cash_delta - entry_fee,
            "uncapped_pnl": uncapped_cash_delta - entry_fee,
            "fee": fee,
            "entry_fee": entry_fee,
            "exit_fee": max(float(exit_fee or 0.0), 0.0),
            "funding_fee": funding,
            "balance_floor_applied": balance_floor_applied,
            "cash_floor_adjustment": cash_floor_adjustment,
        }

    @staticmethod
    def _balance_floor_event(timestamp: int, settlement: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": timestamp,
            "rule": "balance_floor_applied",
            "message": "Close settlement was capped at zero cash because margin is not enabled for this backtest.",
            "uncapped_pnl": settlement.get("uncapped_pnl", 0.0),
            "cash_floor_adjustment": settlement.get("cash_floor_adjustment", 0.0),
        }

    def _funding_cost(self, position: _Position, exit_timestamp: int, backtest: BacktestConfig) -> float:
        funding_bps = max(float(backtest.funding_cost_bps or 0.0), 0.0)
        if funding_bps <= 0 or position.quantity == 0:
            return 0.0
        interval_hours = max(float(backtest.funding_interval_hours or 8.0), 1.0)
        held_hours = max(self._duration_minutes(position.entry_timestamp, exit_timestamp), 0.0) / 60.0
        intervals = held_hours / interval_hours
        notional = abs(position.quantity * position.entry_price)
        return notional * funding_bps / self.BPS_DIVISOR * intervals

    @staticmethod
    def _equity(cash: float, quantity: float, entry_price: float, price: float) -> float:
        """Calculate total equity including unrealized profit or loss on an open position."""
        if quantity == 0:
            return cash
        unrealized = quantity * (price - entry_price) if quantity > 0 else abs(quantity) * (entry_price - price)
        return cash + unrealized

    @staticmethod
    def _max_drawdown(equity_curve: list[dict[str, Any]]) -> float:
        """Calculate the maximum drawdown of the equity curve as a fraction."""
        peak = 0.0
        drawdown = 0.0
        for point in equity_curve:
            peak = max(peak, point["equity"])
            if peak > 0:
                drawdown = min(drawdown, (point["equity"] - peak) / peak)
        return drawdown

    @staticmethod
    def _sharpe_like(returns: list[float]) -> float:
        """Compute a Sharpe ratio analogue for the series of returns."""
        if len(returns) < 2:
            return 0.0
        avg = mean(returns)
        variance = sum((value - avg) ** 2 for value in returns) / (len(returns) - 1)
        stdev = variance**0.5
        if stdev == 0:
            return 0.0
        return (avg / stdev) * sqrt(len(returns))

    @staticmethod
    def _sortino_like(returns: list[float]) -> float:
        """Compute a Sortino ratio analogue for the series of returns."""
        if len(returns) < 2:
            return 0.0
        downside = [value for value in returns if value < 0]
        if not downside:
            return 0.0
        avg = mean(returns)
        downside_deviation = (sum(value**2 for value in downside) / len(downside)) ** 0.5
        if downside_deviation == 0:
            return 0.0
        return (avg / downside_deviation) * sqrt(len(returns))

    @staticmethod
    def _duration_minutes(start_timestamp: int, end_timestamp: int) -> float:
        """Compute the duration between two timestamps in minutes."""
        start = BacktestEngine._timestamp_to_datetime(start_timestamp)
        end = BacktestEngine._timestamp_to_datetime(end_timestamp)
        return max(0.0, (end - start).total_seconds() / 60)

    @staticmethod
    def _timestamp_to_datetime(timestamp: int) -> datetime:
        """Convert either seconds or milliseconds since epoch into a timezone aware datetime."""
        value = int(timestamp)
        if value > 10_000_000_000:
            # Millisecond timestamp
            return datetime.fromtimestamp(value / 1000, tz=UTC)
        return datetime.fromtimestamp(value, tz=UTC)

    @staticmethod
    def _elapsed_days(equity_curve: list[dict[str, Any]]) -> float:
        """Compute the total elapsed time of the equity curve in days."""
        if len(equity_curve) < 2:
            return 1.0
        start = BacktestEngine._timestamp_to_datetime(int(equity_curve[0]["timestamp"]))
        end = BacktestEngine._timestamp_to_datetime(int(equity_curve[-1]["timestamp"]))
        return max((end - start).total_seconds() / 86_400, 1 / 24)

    @staticmethod
    def _daily_loss_reached(daily_realized: dict[str, float], candle_time: datetime, max_daily_loss: float) -> bool:
        """Check whether the maximum daily loss has been exceeded for the current day."""
        if max_daily_loss <= 0:
            return False
        return daily_realized.get(candle_time.date().isoformat(), 0.0) <= -abs(max_daily_loss)

    @staticmethod
    def _cooling_down(cooldown_until: datetime | None, candle_time: datetime) -> bool:
        """Determine whether entries are currently cooling down due to a loss streak."""
        return cooldown_until is not None and candle_time < cooldown_until

    @staticmethod
    def _trade_window_full(
        entry_timestamps: list[int],
        timestamp: int,
        max_trades_per_window: int,
        trade_window_minutes: int,
    ) -> bool:
        """Check whether the number of trades in the sliding window exceeds the allowed limit."""
        if max_trades_per_window <= 0:
            return False
        current = BacktestEngine._timestamp_to_datetime(timestamp)
        window_start = current - timedelta(minutes=max(trade_window_minutes, 1))
        recent = [value for value in entry_timestamps if BacktestEngine._timestamp_to_datetime(value) >= window_start]
        return len(recent) >= max_trades_per_window

    def _blocked_entry_reason(
        self,
        daily_realized: dict[str, float],
        candle_time: datetime,
        cooldown_until: datetime | None,
        entry_timestamps: list[int],
        timestamp: int,
        max_trades_per_window: int,
        trade_window_minutes: int,
        max_daily_loss: float,
    ) -> str | None:
        """Determine which risk rule blocked a new entry if any."""
        if self._daily_loss_reached(daily_realized, candle_time, max_daily_loss):
            return "max_daily_loss"
        if self._cooling_down(cooldown_until, candle_time):
            return "loss_streak_cooldown"
        if self._trade_window_full(entry_timestamps, timestamp, max_trades_per_window, trade_window_minutes):
            return "max_trades_per_window"
        return None

    def _feature_payload(
        self,
        symbol: str,
        timeframe: str,
        candles: list[dict[str, Any]],
        signal: Any | None = None,
    ) -> dict[str, Any]:
        """Extract a feature snapshot for audit purposes.

        The feature payload is determined from either metadata contained on the
        signal or by using the ``FeatureEngine`` to compute a snapshot based on
        the provided candles and optional configuration contained within the
        signal metadata.

        Args:
            symbol: Trading symbol.
            timeframe: Candle timeframe.
            candles: List of candle dictionaries up to the current time.
            signal: The current trading signal whose metadata may include a
                feature snapshot or configuration parameters.

        Returns:
            A dictionary of feature values used for audit and analytics.
        """
        metadata: dict[str, Any] = getattr(signal, "metadata", {}) if signal is not None else {}
        if isinstance(metadata, dict) and isinstance(metadata.get("feature_snapshot"), dict):
            return metadata["feature_snapshot"]
        config = self.feature_engine.config_from_parameters(metadata if isinstance(metadata, dict) else {})
        return self.feature_engine.snapshot(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            config=config,
        ).as_dict()

    @staticmethod
    def _feature_audit_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarise feature usage across all trades for auditing."""
        if not trades:
            return {
                "feature_logged_trades": 0,
                "volume_confirmed_trades": 0,
                "average_rule_score": 0.0,
                "average_atr_pct": 0.0,
                "fibonacci_level_trades": 0,
                "pattern_logged_trades": 0,
            }
        feature_logged = [trade for trade in trades if trade.get("entry_features")]
        volume_confirmed = [trade for trade in feature_logged if trade.get("entry_features", {}).get("volume_spike", {}).get("is_spike")]
        rule_scores = [float(trade.get("rule_score", 0.0) or 0.0) for trade in trades]
        atr_values = [float(trade.get("entry_features", {}).get("atr_pct", 0.0) or 0.0) for trade in feature_logged]
        fibonacci_level_trades = [trade for trade in trades if trade.get("fibonacci_levels")]
        pattern_logged = [trade for trade in trades if trade.get("pattern_prediction")]
        return {
            "feature_logged_trades": len(feature_logged),
            "volume_confirmed_trades": len(volume_confirmed),
            "average_rule_score": mean(rule_scores) if rule_scores else 0.0,
            "average_atr_pct": mean(atr_values) if atr_values else 0.0,
            "fibonacci_level_trades": len(fibonacci_level_trades),
            "pattern_logged_trades": len(pattern_logged),
        }

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return default
        return candidate if candidate == candidate and abs(candidate) != float("inf") else default

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        """Return an empty result structure used when no backtest can be run."""
        return {
            "total_return": 0.0,
            "net_return_after_costs": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "sharpe_like": 0.0,
            "sortino_like": 0.0,
            "profit_factor": 0.0,
            "trade_count": 0,
            "closed_trade_count": 0,
            "open_trade_count": 0,
            "trades_per_day": 0.0,
            "average_trade_duration_minutes": 0.0,
            "average_trade": 0.0,
            "capital_turnover_rate": 0.0,
            "turnover_after_fees": 0.0,
            "average_return_per_trade": 0.0,
            "edge_score": 0.0,
            "expectancy": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_loss_ratio": 0.0,
            "cost_drag_bps": 0.0,
            "max_adverse_excursion": 0.0,
            "max_favorable_excursion": 0.0,
            "no_trade_reason": "no_test_window_data",
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "fees_paid": 0.0,
            "funding_cost_estimate": 0.0,
            "final_cash": 0.0,
            "final_equity": 0.0,
            "raw_final_equity": 0.0,
            "leverage": 1.0,
            "allocation_amount_usd": 0.0,
            "liquidation_buffer_pct": 1.0,
            "equity_curve": [],
            "drawdown_curve": [],
            "trades": [],
            "risk_events": [],
            "risk_event_count": 0,
            "feature_audit_summary": {},
            "ml_feature_schema_version": ML_FEATURE_SCHEMA_VERSION,
            "ml_first_enabled": False,
            "ml_decision_trade_count": 0,
        }
