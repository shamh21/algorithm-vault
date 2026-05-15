"""Flask CLI commands for backtesting and optimization."""

from __future__ import annotations

import importlib.util
import inspect
import json
import signal
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import click
from cryptography.fernet import Fernet
from flask import Flask, current_app, has_app_context
from flask.cli import with_appcontext
from sqlalchemy import func, or_
from sqlalchemy import inspect as sa_inspect

from ..backtesting.engine import BacktestConfig
from ..config import public_origin_violations
from ..extensions import db
from ..ml.decision_engine import MODEL_FAMILIES as ML_DECISION_MODEL_FAMILIES
from ..ml.decision_engine import SIGNAL_FAMILY as ML_SIGNAL_FAMILY
from ..ml.features import MLFeatureFactory
from ..ml.online_ranker import OnlineRanker, extract_features, horizon_from_duration, outcome_from_result
from ..ml.signal_model import MLSignalModel
from ..models import (
    AuditLog,
    BacktestRun,
    Fill,
    MLMarketHistory,
    MLOfflineModel,
    MLTrainingEvent,
    OptimizerRun,
    Order,
    RiskEvent,
    Setting,
    StrategyRanking,
    StrategyRun,
    TradingConnection,
    User,
    VaultAllocationLeg,
    VaultCycle,
    WalletAddress,
    WalletAuditLog,
    WalletBalance,
    WalletTransaction,
    WalletWithdrawal,
)
from ..runtime import get_service
from ..services.connection_health import (
    active_connection_health,
    build_connection_health,
    operator_connection_message,
    parse_exchange_failure,
    store_connection_health,
)
from ..services.live_provider_adapters import KUCOIN_CONTRACT_SPECS, KUCOIN_SYMBOLS
from ..services.order_manager import OrderIntent
from ..services.provider_assets import normalize_provider as normalize_exchange_provider
from ..services.provider_assets import provider_collateral_asset, provider_feature_context
from ..services.signal_quality import SignalQualityEvaluator


class _OptimizerCliTimeout(TimeoutError):
    """Raised when the operator-facing optimizer command exceeds its deadline."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = float(timeout_seconds)
        super().__init__(
            f"run-optimization exceeded OPTIMIZER_MARKET_DATA_TIMEOUT_SECONDS={self.timeout_seconds:.3f}"
        )


class _FindCanaryRankingTimeout(TimeoutError):
    """Raised when find-live-canary-ranking exceeds its command deadline."""

    def __init__(self, timeout_seconds: float, failed_phase: str | None = None) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.failed_phase = str(failed_phase or "unknown")
        super().__init__(
            "find-live-canary-ranking exceeded "
            f"FIND_CANARY_RANKING_TIMEOUT_SECONDS={self.timeout_seconds:.3f} "
            f"during {self.failed_phase}"
        )


FIND_CANARY_PROGRESS_PREFIX = "[find-live-canary-ranking]"


class _HighUpsideDiscoveryTimeout(TimeoutError):
    """Raised when high-upside discovery exceeds its command deadline."""

    def __init__(self, timeout_seconds: float, failed_phase: str | None = None) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.failed_phase = str(failed_phase or "unknown")
        super().__init__(
            "discover-high-upside-vault-candidates exceeded "
            f"HIGH_UPSIDE_COMMAND_TIMEOUT_SECONDS={self.timeout_seconds:.3f} "
            f"during {self.failed_phase}"
        )


HIGH_UPSIDE_DISCOVERY_PROGRESS_PREFIX = "[discover-high-upside-vault-candidates]"


def _call_with_supported_kwargs(func: Callable[..., object], *args: object, **kwargs: object) -> object:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return func(*args, **kwargs)
    supported = {key: value for key, value in kwargs.items() if key in parameters}
    return func(*args, **supported)


def _production_readiness_payload_for(*, provider: str = "global", horizon: str = "1h") -> dict[str, object]:
    return dict(_call_with_supported_kwargs(_production_readiness_payload, provider=provider, horizon=horizon))


def register_cli(app: Flask) -> None:
    """Register operational commands on the Flask app."""

    @app.cli.group("worker")
    def worker_cli() -> None:
        """Dedicated worker process commands."""

    @worker_cli.command("start")
    @click.option("--once", is_flag=True, help="Run due jobs once and exit.")
    @click.option("--interval", default=None, type=int, help="Polling interval in seconds.")
    @click.option("--owner-id", default="", help="Stable worker owner id for lease diagnostics.")
    @click.option(
        "--job",
        "jobs",
        multiple=True,
        type=click.Choice(["strategy_starter", "vault_cycle_enforcement", "treasury_solvency"]),
        help="Limit execution to one or more worker job names.",
    )
    def worker_start(once: bool, interval: int | None, owner_id: str, jobs: tuple[str, ...]) -> None:
        """Run the dedicated DB-lease worker."""

        from ..workers.runner import run_worker

        results = run_worker(
            once=once,
            interval_seconds=interval,
            owner_id=owner_id or None,
            job_filter=set(jobs) if jobs else None,
        )
        click.echo(json.dumps({"ok": True, "results": results}, indent=2, default=str))

    @app.cli.command("run-backtest")
    @click.option("--strategy", "strategy_name", default="ema_crossover", show_default=True)
    @click.option("--symbol", default="BTC", show_default=True)
    @click.option("--timeframe", default=None)
    @click.option("--parameters-json", default="{}", show_default=True)
    @click.option("--initial-balance", default=None, type=float)
    @with_appcontext
    def run_backtest(
        strategy_name: str,
        symbol: str,
        timeframe: str | None,
        parameters_json: str,
        initial_balance: float | None,
    ) -> None:
        """Run one historical candle backtest and store the result."""

        from flask import current_app

        parameters = json.loads(parameters_json or "{}")
        config = BacktestConfig(
            strategy_name=strategy_name,
            symbol=symbol.upper(),
            timeframe=timeframe or current_app.config["DEFAULT_TIMEFRAME"],
            mode="testnet",
            initial_balance=initial_balance or float(current_app.config["DEFAULT_PAPER_BALANCE"]),
            fee_bps=float(current_app.config["FEE_BPS"]),
            slippage_bps=float(current_app.config["SIM_SLIPPAGE_BPS"]),
            stop_loss_pct=float(parameters.get("stop_loss_pct", 0.01)),
            take_profit_pct=float(parameters.get("take_profit_pct", 0.02)),
            position_size_fraction=float(parameters.get("risk_fraction", 0.08)),
            parameters=parameters,
            sizing_mode="risk_based",
            risk_per_trade_pct=float(current_app.config["RISK_PER_TRADE_PCT"]),
            max_daily_loss=float(current_app.config["MAX_DAILY_LOSS_USDC"]),
            max_drawdown_pct=float(current_app.config["MAX_BACKTEST_DRAWDOWN_PCT"]),
            loss_streak_cooldown=int(current_app.config["LOSS_STREAK_COOLDOWN_THRESHOLD"]),
            cooldown_minutes=int(current_app.config["LOSS_COOLDOWN_MINUTES"]),
            max_trades_per_window=int(current_app.config["MAX_TRADES_PER_WINDOW"]),
            trade_window_minutes=int(current_app.config["TRADE_WINDOW_MINUTES"]),
            intrabar_model="conservative",
        )
        result = get_service("backtest_engine").run(config)
        record = BacktestRun(strategy_name=config.strategy_name, symbol=config.symbol, timeframe=config.timeframe)
        record.parameters = {"parameters": parameters, "cli": True}
        record.result = result
        db.session.add(record)
        db.session.commit()
        click.echo(
            json.dumps(
                {
                    "backtest_run_id": record.id,
                    "total_return": result["total_return"],
                    "max_drawdown": result["max_drawdown"],
                    "trade_count": result["trade_count"],
                    "trades_per_day": result["trades_per_day"],
                },
                indent=2,
            )
        )

    @app.cli.command("one-h10-evaluation-report")
    @click.option("--symbol", default="BTC", show_default=True)
    @click.option("--timeframe", default="1m", show_default=True)
    @click.option("--strategy", "strategy_name", default="scalping", show_default=True)
    @click.option("--initial-balance", default=None, type=float)
    @with_appcontext
    def one_h10_evaluation_report(symbol: str, timeframe: str, strategy_name: str, initial_balance: float | None) -> None:
        """Run a minimal cost-aware 1H10 validation report."""

        from ..services.one_h10_evaluation import build_one_h10_evaluation_report

        report = build_one_h10_evaluation_report(
            current_app.config,
            get_service("backtest_engine"),
            symbol=symbol,
            timeframe=timeframe,
            strategy_name=strategy_name,
            initial_balance=initial_balance,
        )
        click.echo(json.dumps(report, indent=2, default=str))

    @app.cli.command("one-h10-backfill")
    @click.option("--user-id", default=1, show_default=True, type=int)
    @click.option("--mode", default="live", show_default=True, type=click.Choice(["live"]))
    @click.option("--confirm", default="", help="Must be BACKFILL-1H10-FEATURES.")
    @with_appcontext
    def one_h10_backfill(user_id: int, mode: str, confirm: str) -> None:
        """Sync all 1H10 leveraged markets and backfill a rate-limited feature batch."""

        if confirm != "BACKFILL-1H10-FEATURES":
            raise click.ClickException("Refusing to run 1H10 backfill without BACKFILL-1H10-FEATURES confirmation.")
        results = get_service("leveraged_markets").sync_one_h10_backfill_for_user(user_id, mode=mode)
        db.session.commit()
        click.echo(json.dumps({"user_id": user_id, "mode": mode, "results": results}, indent=2, sort_keys=True))

    @app.cli.command("run-optimization")
    @click.option("--profile", default="short_term", show_default=True)
    @click.option("--symbol", "symbols", multiple=True)
    @click.option("--timeframe", "timeframes", multiple=True)
    @click.option("--strategy", "strategy_names", multiple=True)
    @click.option("--max-parameter-sets", default=8, show_default=True, type=int)
    @click.option("--auto-deploy-top-n", default=1, show_default=True, type=int)
    @click.option("--allocation-amount-usd", default=0.0, show_default=True, type=float)
    @click.option("--lock-duration-hours", default=0, show_default=True, type=int)
    @click.option("--universe-mode", type=click.Choice(["configured", "dynamic_liquid"]), default="configured", show_default=True)
    @click.option("--max-parallel-legs", default=None, type=int)
    @click.option("--allow-leverage-experiment", is_flag=True, default=False)
    @click.option("--use-full-history", is_flag=True, default=False)
    @click.option("--decay-factor", default=None, type=float)
    @click.option("--min-trade-count", default=None, type=int)
    @click.option("--min-edge-bps", default=None, type=float)
    @click.option("--fib-confluence-threshold", "--confluence-threshold", default=None, type=float)
    @click.option("--require-shadow-validation/--no-require-shadow-validation", default=None)
    @click.option("--enhanced-ensemble-enabled/--no-enhanced-ensemble-enabled", default=None)
    @click.option("--ensemble-max-legs", default=None, type=int)
    @click.option("--ensemble-min-sharpe", default=None, type=float)
    @click.option("--ensemble-learning-decay", default=None, type=float)
    @click.option("--experimental-duration-ensemble-enabled/--no-experimental-duration-ensemble-enabled", default=None)
    @click.option("--experimental-live-eligible/--no-experimental-live-eligible", default=None)
    @click.option("--ensemble-primary-metric", default=None)
    @click.option("--duration-live-cap-usdc-json", default=None)
    @click.option("--duration-live-cap-pct-json", default=None)
    @click.option("--max-return-optimizer-enabled/--no-max-return-optimizer-enabled", default=None)
    @click.option("--max-return-live-eligible/--no-max-return-live-eligible", default=None)
    @click.option("--dynamic-intraday-live-eligible/--no-dynamic-intraday-live-eligible", default=None)
    @click.option("--high-upside-profile/--no-high-upside-profile", default=None)
    @click.option("--market-structure-features-enabled/--no-market-structure-features-enabled", default=None)
    @click.option("--market-structure-provider", default=None)
    @click.option("--pair-screening-enabled/--no-pair-screening-enabled", default=None)
    @click.option("--pair-trading-enabled/--no-pair-trading-enabled", default=None)
    @click.option("--pair-live-eligible/--no-pair-live-eligible", default=None)
    @click.option("--pair-min-correlation", default=None, type=float)
    @click.option("--pair-max-spread-zscore", default=None, type=float)
    @with_appcontext
    def run_optimization(
        profile: str,
        symbols: tuple[str, ...],
        timeframes: tuple[str, ...],
        strategy_names: tuple[str, ...],
        max_parameter_sets: int,
        auto_deploy_top_n: int,
        allocation_amount_usd: float,
        lock_duration_hours: int,
        universe_mode: str,
        max_parallel_legs: int | None,
        allow_leverage_experiment: bool,
        use_full_history: bool,
        decay_factor: float | None,
        min_trade_count: int | None,
        min_edge_bps: float | None,
        fib_confluence_threshold: float | None,
        require_shadow_validation: bool | None,
        enhanced_ensemble_enabled: bool | None,
        ensemble_max_legs: int | None,
        ensemble_min_sharpe: float | None,
        ensemble_learning_decay: float | None,
        experimental_duration_ensemble_enabled: bool | None,
        experimental_live_eligible: bool | None,
        ensemble_primary_metric: str | None,
        duration_live_cap_usdc_json: str | None,
        duration_live_cap_pct_json: str | None,
        max_return_optimizer_enabled: bool | None,
        max_return_live_eligible: bool | None,
        dynamic_intraday_live_eligible: bool | None,
        high_upside_profile: bool | None,
        market_structure_features_enabled: bool | None,
        market_structure_provider: str | None,
        pair_screening_enabled: bool | None,
        pair_trading_enabled: bool | None,
        pair_live_eligible: bool | None,
        pair_min_correlation: float | None,
        pair_max_spread_zscore: float | None,
    ) -> None:
        """Run short-term walk-forward optimization and persist rankings."""

        optimizer = get_service("strategy_optimizer")
        config = optimizer.default_config(
            symbols=[symbol.upper() for symbol in symbols] or None,
            timeframes=list(timeframes) or None,
            strategy_names=list(strategy_names) or None,
            profile=profile,
            allocation_amount_usd=allocation_amount_usd,
            lock_duration_hours=lock_duration_hours,
            universe_mode=universe_mode,
            max_parallel_legs=max_parallel_legs or int(current_app.config.get("VAULT_MAX_PARALLEL_LEGS", 1)),
            allow_leverage_experiment=allow_leverage_experiment,
        )
        config.max_parameter_sets = max_parameter_sets
        config.auto_deploy_top_n = auto_deploy_top_n
        config.use_full_history = use_full_history
        if decay_factor is not None:
            config.decay_factor = decay_factor
        if min_trade_count is not None:
            config.min_trade_count = min_trade_count
        if min_edge_bps is not None:
            config.min_edge_bps = min_edge_bps
        if fib_confluence_threshold is not None:
            config.fib_confluence_threshold = fib_confluence_threshold
        if require_shadow_validation is not None:
            config.require_shadow_validation = require_shadow_validation
        if enhanced_ensemble_enabled is not None:
            config.enhanced_ensemble_enabled = enhanced_ensemble_enabled
        if ensemble_max_legs is not None:
            config.ensemble_max_legs = ensemble_max_legs
        if ensemble_min_sharpe is not None:
            config.ensemble_min_sharpe = ensemble_min_sharpe
        if ensemble_learning_decay is not None:
            config.ensemble_learning_decay = ensemble_learning_decay
        if experimental_duration_ensemble_enabled is not None:
            config.experimental_duration_ensemble_enabled = experimental_duration_ensemble_enabled
        if experimental_live_eligible is not None:
            config.experimental_live_eligible = experimental_live_eligible
        if ensemble_primary_metric is not None:
            config.ensemble_primary_metric = ensemble_primary_metric
        if duration_live_cap_usdc_json is not None:
            config.duration_live_cap_usdc_by_duration = _json_float_map(duration_live_cap_usdc_json)
        if duration_live_cap_pct_json is not None:
            config.duration_live_cap_pct_by_duration = _json_float_map(duration_live_cap_pct_json)
        if max_return_optimizer_enabled is not None:
            config.max_return_optimizer_enabled = max_return_optimizer_enabled
        if max_return_live_eligible is not None:
            config.max_return_live_eligible = max_return_live_eligible
        if dynamic_intraday_live_eligible is not None:
            config.dynamic_intraday_live_eligible = dynamic_intraday_live_eligible
        if high_upside_profile is not None:
            config.high_upside_profile = high_upside_profile
        if market_structure_features_enabled is not None:
            config.market_structure_features_enabled = market_structure_features_enabled
        if market_structure_provider is not None:
            config.market_structure_provider = market_structure_provider
        if pair_screening_enabled is not None:
            config.pair_screening_enabled = pair_screening_enabled
        if pair_trading_enabled is not None:
            config.pair_trading_enabled = pair_trading_enabled
        if pair_live_eligible is not None:
            config.pair_live_eligible = pair_live_eligible
        if pair_min_correlation is not None:
            config.pair_min_correlation = pair_min_correlation
        if pair_max_spread_zscore is not None:
            config.pair_max_spread_zscore = pair_max_spread_zscore
        timeout_seconds = _optimizer_timeout_seconds()
        started_at = time.monotonic()
        config.runtime_deadline_monotonic = _optimizer_cooperative_deadline(started_at, timeout_seconds)
        try:
            with _optimizer_deadline(timeout_seconds):
                result = optimizer.run(config)
            result = _optimization_result_with_diagnostics(
                config,
                result,
                started_at=started_at,
                timeout_seconds=timeout_seconds,
                timed_out=False,
            )
        except _OptimizerCliTimeout as exc:
            db.session.rollback()
            result = _optimization_timeout_result(
                config,
                started_at=started_at,
                timeout_seconds=timeout_seconds,
                error=str(exc),
            )
        click.echo(json.dumps(result, indent=2, default=str))

    @app.cli.command("discover-high-upside-vault-candidates")
    @click.option("--user-id", default=1, show_default=True, type=int)
    @click.option(
        "--provider-scope",
        default=None,
        type=click.Choice(["active_provider", "multi_provider", "hyperliquid", "binance", "kucoin", "dydx", "uniswap"]),
        help="Provider discovery scope. Defaults to HIGH_UPSIDE_PROVIDER_SCOPE.",
    )
    @click.option("--provider", "providers", multiple=True, help="Explicit provider(s) to inspect.")
    @click.option("--symbol", "symbols", multiple=True, help="Explicit symbol(s), preserving order.")
    @click.option("--timeframe", "timeframes", multiple=True)
    @click.option("--profile", "profiles", multiple=True)
    @click.option("--strategy", "strategy_names", multiple=True)
    @click.option("--max-symbols", default=None, type=int)
    @click.option("--max-sweeps", default=None, type=int)
    @click.option("--max-parameter-sets", default=8, show_default=True, type=int)
    @click.option("--allocation-amount-usd", default=10.0, show_default=True, type=float)
    @click.option("--lock-duration-hours", default=1, show_default=True, type=int)
    @click.option(
        "--objective",
        default="risk_adjusted",
        show_default=True,
        type=click.Choice(["risk_adjusted", "extreme_upside", "extreme_roi_1h", "one_h10", "consistent_roi_1w"]),
    )
    @click.option("--target-roi-pct", default=None, type=float)
    @click.option("--timeout-seconds", default=None, type=float)
    @click.option("--run-backtests/--no-run-backtests", default=True, show_default=True)
    @with_appcontext
    def discover_high_upside_vault_candidates(
        user_id: int,
        provider_scope: str | None,
        providers: tuple[str, ...],
        symbols: tuple[str, ...],
        timeframes: tuple[str, ...],
        profiles: tuple[str, ...],
        strategy_names: tuple[str, ...],
        max_symbols: int | None,
        max_sweeps: int | None,
        max_parameter_sets: int,
        allocation_amount_usd: float,
        lock_duration_hours: int,
        objective: str,
        target_roi_pct: float | None,
        timeout_seconds: float | None,
        run_backtests: bool,
    ) -> None:
        """Research hot multi-provider high-upside vault candidates without placing orders."""

        started_at = time.monotonic()
        timeout = _high_upside_timeout_seconds(timeout_seconds)
        progress_state: dict[str, object] = {"current_phase": "starting", "failed_phase": ""}
        try:
            with _high_upside_deadline(timeout, progress_state):
                payload = _discover_high_upside_vault_candidates_payload(
                    user_id=user_id,
                    provider_scope=provider_scope,
                    providers=list(providers),
                    symbols=list(symbols),
                    timeframes=list(timeframes),
                    profiles=list(profiles),
                    strategy_names=list(strategy_names),
                    max_symbols=max_symbols,
                    max_sweeps=max_sweeps,
                    max_parameter_sets=max_parameter_sets,
                    allocation_amount_usd=allocation_amount_usd,
                    lock_duration_hours=lock_duration_hours,
                    objective=objective,
                    target_roi_pct=target_roi_pct,
                    timeout_seconds=timeout,
                    started_at=started_at,
                    progress_state=progress_state,
                    run_backtests=run_backtests,
                )
        except _HighUpsideDiscoveryTimeout as exc:
            db.session.rollback()
            payload = _high_upside_timeout_payload(
                user_id=user_id,
                provider_scope=provider_scope,
                providers=list(providers),
                symbols=list(symbols),
                timeframes=list(timeframes),
                profiles=list(profiles),
                objective=objective,
                target_roi_pct=target_roi_pct,
                timeout_seconds=timeout,
                started_at=started_at,
                failed_phase=exc.failed_phase,
            )
        click.echo(json.dumps(payload, indent=2, default=str))

    @app.cli.command("ml-history-backfill")
    @click.option(
        "--provider-scope",
        default="all",
        show_default=True,
        type=click.Choice(["all", "active_provider", "multi_provider", "hyperliquid", "binance", "kucoin", "dydx", "uniswap"]),
    )
    @click.option("--provider", "providers", multiple=True)
    @click.option("--symbol", "symbols", multiple=True)
    @click.option("--timeframe", "timeframes", multiple=True, default=("5m", "15m", "1h"))
    @click.option("--max-symbols", default=None, type=int)
    @click.option("--lookback-days", default=None, type=int)
    @click.option("--user-id", default=1, show_default=True, type=int)
    @click.option("--all-pairs", is_flag=True, default=False, help="Backfill every tradable Hyperliquid perp when no explicit symbols are supplied.")
    @click.option("--confirm", default="", help="Must be BACKFILL-ML-HISTORY.")
    @with_appcontext
    def ml_history_backfill(
        provider_scope: str,
        providers: tuple[str, ...],
        symbols: tuple[str, ...],
        timeframes: tuple[str, ...],
        max_symbols: int | None,
        lookback_days: int | None,
        user_id: int,
        all_pairs: bool,
        confirm: str,
    ) -> None:
        """Collect bounded market-history rows for ML training; research-only."""

        if confirm != "BACKFILL-ML-HISTORY":
            raise click.ClickException("Refusing ML history backfill. Pass --confirm BACKFILL-ML-HISTORY.")
        payload = _ml_history_backfill_payload(
            provider_scope=provider_scope,
            providers=list(providers),
            symbols=list(symbols),
            timeframes=list(timeframes),
            max_symbols=max_symbols,
            lookback_days=lookback_days,
            user_id=user_id,
            all_pairs=all_pairs,
        )
        click.echo(json.dumps(payload, indent=2, default=str))

    @app.cli.command("find-live-canary-ranking")
    @click.option("--user-id", default=1, show_default=True, type=int)
    @click.option("--provider", type=click.Choice(["active", "hyperliquid", "kucoin"]), default="active", show_default=True)
    @click.option("--connection-id", default=None, type=int, help="Use one verified connection, even if it is not active.")
    @click.option("--symbol", "symbols", multiple=True)
    @click.option("--timeframe", "timeframes", multiple=True)
    @click.option("--profile", "profiles", multiple=True)
    @click.option("--max-parameter-sets", default=8, show_default=True, type=int)
    @click.option("--allocation-amount-usd", default=10.0, show_default=True, type=float)
    @click.option("--lock-duration-hours", default=1, show_default=True, type=int)
    @click.option("--auto-deploy-top-n", default=1, show_default=True, type=int)
    @click.option("--strategy", "strategy_names", multiple=True)
    @click.option(
        "--timeout-seconds",
        default=None,
        type=float,
        help="Outer command timeout. Defaults to FIND_CANARY_RANKING_TIMEOUT_SECONDS.",
    )
    @click.option(
        "--research-depth",
        type=click.Choice(["quick", "standard", "deep", "ml"]),
        default="standard",
        show_default=True,
        help="Use quick for only the requested sweep, standard for focused follow-up sweeps, deep for additional per-symbol coverage, or ml for ML-prioritized coverage.",
    )
    @with_appcontext
    def find_live_canary_ranking(
        user_id: int,
        provider: str,
        connection_id: int | None,
        symbols: tuple[str, ...],
        timeframes: tuple[str, ...],
        profiles: tuple[str, ...],
        max_parameter_sets: int,
        allocation_amount_usd: float,
        lock_duration_hours: int,
        auto_deploy_top_n: int,
        strategy_names: tuple[str, ...],
        timeout_seconds: float | None,
        research_depth: str,
    ) -> None:
        """Run staged optimizer sweeps until an accepted live canary ranking exists."""

        started_at = time.monotonic()
        timeout = _find_canary_ranking_timeout_seconds(timeout_seconds)
        max_symbols = _find_canary_max_symbols()
        max_rankings = _find_canary_max_rankings()
        explicit_symbols = bool(symbols)
        requested_symbols = [symbol.upper() for symbol in symbols]
        research_depth_value = str(research_depth or "standard").lower()
        candidate_symbols = _find_canary_candidate_symbols(
            provider=provider,
            requested_symbols=requested_symbols,
            explicit_symbols=explicit_symbols,
            max_symbols=max_symbols,
        )
        selected_symbols = list(candidate_symbols["symbols"])
        omitted_symbols = list(candidate_symbols["omitted_symbols"])
        selected_profiles = list(profiles) or (
            ["short_term"] if research_depth_value == "quick" else ["short_term", "aggressive_1h"]
        )
        selected_timeframes = list(timeframes) or (
            ["15m"] if research_depth_value == "quick" else ["5m", "15m", "1h"]
        )
        progress = _find_canary_progress_reporter()
        progress_state = _find_canary_progress_state(
            started_at=started_at,
            timeout_seconds=timeout,
            symbols=selected_symbols,
            omitted_symbols=omitted_symbols,
            profiles=selected_profiles,
            max_symbols=max_symbols,
            max_rankings=max_rankings,
        )
        deadline_monotonic = started_at + timeout
        result: dict[str, object]
        try:
            with _find_canary_deadline(timeout, progress_state):
                result = _find_live_canary_ranking_payload(
                    user_id=user_id,
                    provider=provider,
                    connection_id=connection_id,
                    symbols=selected_symbols,
                    omitted_symbols=omitted_symbols,
                    timeframes=selected_timeframes,
                    profiles=selected_profiles,
                    max_parameter_sets=max_parameter_sets,
                    allocation_amount_usd=allocation_amount_usd,
                    lock_duration_hours=lock_duration_hours,
                    auto_deploy_top_n=min(max(0, int(auto_deploy_top_n or 0)), max_rankings),
                    strategy_names=list(strategy_names) or None,
                    research_depth=research_depth,
                    adaptive_research=not profiles and not strategy_names,
                    deadline_monotonic=deadline_monotonic,
                    timeout_seconds=timeout,
                    timeout_blocker="find_canary_ranking_timeout",
                    explicit_symbols=explicit_symbols,
                    fallback_symbols_used=bool(candidate_symbols["fallback_symbols_used"]),
                    candidate_order=list(candidate_symbols["candidate_order"]),
                    max_symbols=max_symbols,
                    max_rankings=max_rankings,
                    progress=progress,
                    progress_state=progress_state,
                    market_data_mode="live",
                )
        except _FindCanaryRankingTimeout:
            db.session.rollback()
            failed_phase = str(progress_state.get("failed_phase") or progress_state.get("current_phase") or "unknown")
            _find_canary_phase(progress_state, progress, "timeout", failed_phase=failed_phase)
            progress_state["failed_phase"] = failed_phase
            result = _find_canary_timeout_payload(
                user_id=user_id,
                provider=provider,
                symbols=selected_symbols,
                omitted_symbols=omitted_symbols,
                timeframes=selected_timeframes,
                profiles=selected_profiles,
                research_depth=research_depth,
                market_data_mode="live",
                timeout_seconds=timeout,
                started_at=started_at,
                progress_state=progress_state,
            )
            audit_id = _record_find_canary_timeout_audit(result)
            result["timeout_audit_id"] = audit_id
        if (
            bool(result.get("timed_out", False))
            and "find_canary_ranking_timeout" in list(result.get("blockers", []) or [])
            and not result.get("timeout_audit_id")
        ):
            result["timeout_audit_id"] = _record_find_canary_timeout_audit(result)
        click.echo(json.dumps(result, indent=2, default=str))

    @app.cli.command("live-auto-canary")
    @click.option("--user-id", default=1, show_default=True, type=int)
    @click.option("--provider", type=click.Choice(["hyperliquid", "kucoin", "both"]), default="both", show_default=True)
    @click.option("--connection-id", default=None, type=int, help="Use one verified provider connection for a single-provider run.")
    @click.option("--research-budget-minutes", default=10.0, show_default=True, type=float)
    @click.option("--max-submissions", default=1, show_default=True, type=int)
    @click.option("--submit-interval-seconds", default=0.0, show_default=True, type=float)
    @click.option("--submit", is_flag=True, default=False)
    @click.option("--confirm", default="", help="Must be LIVE-CANARY-TRADE when --submit is used.")
    @with_appcontext
    def live_auto_canary(
        user_id: int,
        provider: str,
        connection_id: int | None,
        research_budget_minutes: float,
        max_submissions: int,
        submit_interval_seconds: float,
        submit: bool,
        confirm: str,
    ) -> None:
        """Find an accepted ranking, preview providers, and optionally submit guarded canaries."""

        if submit and confirm != "LIVE-CANARY-TRADE":
            raise click.ClickException("Refusing live auto canary submission. Pass --confirm LIVE-CANARY-TRADE.")

        result = _live_auto_canary_payload(
            user_id=user_id,
            provider=provider,
            connection_id=connection_id,
            research_budget_minutes=research_budget_minutes,
            max_submissions=max_submissions,
            submit_interval_seconds=submit_interval_seconds,
            submit=submit,
        )
        click.echo(json.dumps(result, indent=2, default=str))

    @app.cli.command("live-rapid-ml-trader")
    @click.option("--user-id", default=1, show_default=True, type=int)
    @click.option("--capital-usd", required=True, type=float, help="Total account-level capital cap for this rapid session.")
    @click.option("--provider", type=click.Choice(["both", "hyperliquid", "kucoin"]), default="both", show_default=True)
    @click.option("--duration-minutes", default=0.0, show_default=True, type=float)
    @click.option("--decision-interval-ms", default=None, type=int, help="Decision-loop interval; default comes from RAPID_ML_DECISION_INTERVAL_MS.")
    @click.option(
        "--max-order-rate-per-provider",
        default=None,
        type=float,
        help="Maximum opening order rate per provider; default comes from RAPID_ML_MAX_ORDER_RATE_PER_PROVIDER.",
    )
    @click.option("--submit", is_flag=True, default=False)
    @click.option("--confirm", default="", help="Must be RAPID-ML-LIVE when --submit is used.")
    @click.option("--compact", is_flag=True, default=False, help="Print an operator summary without deep ML model diagnostics.")
    @with_appcontext
    def live_rapid_ml_trader(
        user_id: int,
        capital_usd: float,
        provider: str,
        duration_minutes: float,
        decision_interval_ms: int | None,
        max_order_rate_per_provider: float | None,
        submit: bool,
        confirm: str,
        compact: bool,
    ) -> None:
        """Run rapid ML dual-venue preview or exact-confirmed live execution."""

        if submit and confirm != "RAPID-ML-LIVE":
            raise click.ClickException("Refusing rapid ML live submission. Pass --confirm RAPID-ML-LIVE.")

        progress_callback = None
        if duration_minutes > 0 and not current_app.testing:
            def progress_callback(event: dict[str, object]) -> None:
                click.echo(
                    "[rapid-ml] "
                    f"cycle={event.get('cycle')}/{event.get('planned_cycles')} "
                    f"elapsed={float(event.get('elapsed_seconds') or 0.0):.1f}s "
                    f"submitted={event.get('submitted_count')} "
                    f"position_submitted={event.get('position_submitted_count')} "
                    f"preview={event.get('preview_ready_count')} "
                    f"rejected={event.get('rejected_decision_count')}",
                    err=True,
                )

        result = get_service("rapid_ml_trader").run(
            user_id=user_id,
            capital_usd=capital_usd,
            provider=provider,
            duration_minutes=duration_minutes,
            decision_interval_ms=decision_interval_ms,
            max_order_rate_per_provider=max_order_rate_per_provider,
            submit=submit,
            confirm=confirm,
            progress_callback=progress_callback,
        )
        if compact:
            result = _compact_rapid_ml_payload(result)
        click.echo(json.dumps(result, indent=2, default=str))

    @app.cli.command("profile-wallet-check")
    @click.option("--username", required=True, help="Application username to inspect.")
    @click.option("--user-id", default=None, type=int, help="Optional user id cross-check.")
    @with_appcontext
    def profile_wallet_check(username: str, user_id: int | None) -> None:
        """Print a read-only local wallet/profile reconciliation summary."""

        result = get_service("wallet_summary").profile_wallet_check(username=username, user_id=user_id)
        click.echo(json.dumps(result, indent=2, default=str))
        if not bool(result.get("exists", False)):
            raise click.exceptions.Exit(1)

    @app.cli.command("production-account-readiness")
    @click.option("--username", default="sufyanh", show_default=True, help="Protected application username to verify.")
    @click.option("--user-id", default=None, type=int, help="Optional user id cross-check.")
    @click.option("--expected-origin", default="https://app.algvault.com", show_default=True, help="Production origin expected for browser traffic.")
    @click.option("--expected-wallet-snapshot", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
    @click.option("--allow-no-snapshot", is_flag=True, default=False, help="Only verify the account exists and has nonzero local app funds.")
    @with_appcontext
    def production_account_readiness(
        username: str,
        user_id: int | None,
        expected_origin: str,
        expected_wallet_snapshot: Path | None,
        allow_no_snapshot: bool,
    ) -> None:
        """Verify production URL config and protected account wallet readiness."""

        snapshot = _load_json_file(expected_wallet_snapshot) if expected_wallet_snapshot is not None else None
        result = _production_account_readiness_payload(
            username=username,
            user_id=user_id,
            expected_origin=expected_origin,
            expected_wallet_snapshot=snapshot,
            require_expected_snapshot=not allow_no_snapshot,
        )
        click.echo(json.dumps(result, indent=2, default=str))
        if result.get("blockers"):
            raise click.exceptions.Exit(1)

    @app.cli.command("reset-local-state")
    @click.option("--confirm", default="", help="Must be FULL-LIVE-RESET.")
    @with_appcontext
    def reset_local_state(confirm: str) -> None:
        """Delete all local app data and reseed live defaults."""

        if confirm != "FULL-LIVE-RESET":
            raise click.ClickException("Refusing reset. Pass --confirm FULL-LIVE-RESET.")

        backup_path = _sqlite_backup_path()
        db.session.remove()
        db.drop_all()
        _drop_legacy_tables()
        db.create_all()

        live_enabled = bool(current_app.config.get("ENABLE_LIVE_TRADING", False))
        Setting.set_json("current_mode", "live")
        Setting.set_json("panic_lock", False)
        Setting.set_json("explicit_live_confirmed", True)
        Setting.set_json("secondary_confirmation", True)
        Setting.set_json("live_trading_blocked", False)
        Setting.set_json("use_real_addresses", bool(current_app.config.get("USE_REAL_ADDRESSES", False)))
        db.session.commit()
        click.echo(
            json.dumps(
                {
                    "backup": str(backup_path) if backup_path else None,
                    "current_mode": Setting.get_json("current_mode"),
                    "message": "Local state reset. Current mode is live. Register new users and add per-user trading connections.",
                },
                indent=2,
            )
        )
        if live_enabled:
            click.echo(
                "WARNING: ENABLE_LIVE_TRADING is true, so reset-local-state seeded live-oriented confirmation state. "
                "Live orders still require an active verified connection plus all risk gates."
            )

    @app.cli.command("live-only-clean-slate")
    @click.option("--confirm", default="", help="Must be LIVE-ONLY-RESET.")
    @with_appcontext
    def live_only_clean_slate(confirm: str) -> None:
        """Back up SQLite and purge non-live execution records."""

        if confirm != "LIVE-ONLY-RESET":
            raise click.ClickException("Refusing cleanup. Pass --confirm LIVE-ONLY-RESET.")

        backup_path = _sqlite_backup_path()
        counts_before = {
            "non_live_orders": _scalar_count("SELECT COUNT(*) FROM \"order\" WHERE mode != 'live'"),
            "non_live_strategy_runs": _scalar_count("SELECT COUNT(*) FROM strategy_run WHERE mode != 'live'"),
            "non_live_vault_cycles": _scalar_count("SELECT COUNT(*) FROM vault_cycle WHERE execution_mode != 'live'"),
            "non_live_position_snapshots": _scalar_count("SELECT COUNT(*) FROM position_snapshot WHERE mode != 'live'"),
            "legacy_paper_accounts": _table_count("paper_account"),
            "legacy_paper_equity_snapshots": _table_count("paper_equity_snapshot"),
        }

        db.session.execute(db.text("DELETE FROM fill WHERE order_id IN (SELECT id FROM \"order\" WHERE mode != 'live')"))
        db.session.execute(
            db.text(
                "DELETE FROM vault_allocation_leg WHERE vault_cycle_id IN "
                "(SELECT id FROM vault_cycle WHERE execution_mode != 'live')"
            )
        )
        db.session.execute(
            db.text(
                "DELETE FROM vault_allocation_leg WHERE strategy_run_id IN "
                "(SELECT id FROM strategy_run WHERE mode != 'live')"
            )
        )
        db.session.execute(db.text("DELETE FROM \"order\" WHERE mode != 'live'"))
        db.session.execute(db.text("DELETE FROM position_snapshot WHERE mode != 'live'"))
        db.session.execute(db.text("DELETE FROM vault_cycle WHERE execution_mode != 'live'"))
        db.session.execute(db.text("DELETE FROM strategy_run WHERE mode != 'live'"))
        _drop_legacy_tables()
        Setting.set_json("current_mode", "live")
        db.session.commit()

        click.echo(
            json.dumps(
                {
                    "backup": str(backup_path) if backup_path else None,
                    "purged": counts_before,
                    "current_mode": Setting.get_json("current_mode"),
                    "remaining_non_live_orders": _scalar_count("SELECT COUNT(*) FROM \"order\" WHERE mode != 'live'"),
                    "remaining_non_live_strategy_runs": _scalar_count("SELECT COUNT(*) FROM strategy_run WHERE mode != 'live'"),
                },
                indent=2,
            )
        )

    @app.cli.command("prune-efficiency-data")
    @click.option("--protect-username", required=True, help="Must be the account to preserve, currently sufyanh.")
    @click.option("--confirm", default="", help="Pass PRUNE-EFFICIENCY-DATA to delete rows. Omit for dry-run.")
    @click.option("--no-trade-retention-hours", default=None, type=float)
    @click.option("--transient-retention-hours", default=None, type=float)
    @click.option("--vacuum/--no-vacuum", default=False, help="Compact SQLite after confirmed deletion.")
    @with_appcontext
    def prune_efficiency_data(
        protect_username: str,
        confirm: str,
        no_trade_retention_hours: float | None,
        transient_retention_hours: float | None,
        vacuum: bool,
    ) -> None:
        """Dry-run or prune noisy efficiency diagnostics while preserving protected account records."""

        result = _prune_efficiency_data_payload(
            protect_username=protect_username,
            confirmed=confirm == "PRUNE-EFFICIENCY-DATA",
            confirmation_value=confirm,
            no_trade_retention_hours=no_trade_retention_hours,
            transient_retention_hours=transient_retention_hours,
            vacuum=vacuum,
        )
        click.echo(json.dumps(result, indent=2, default=str))
        if result.get("blockers"):
            raise click.exceptions.Exit(1)

    @app.cli.command("wallet-readiness")
    @with_appcontext
    def wallet_readiness() -> None:
        """Print live generated-wallet readiness diagnostics."""

        click.echo(json.dumps(_wallet_readiness_payload(), indent=2, default=str))

    @app.cli.group("platform-treasury")
    def platform_treasury_cli() -> None:
        """Manage the platform gas treasury."""

    @platform_treasury_cli.command("status")
    @click.option("--network", default="Ethereum", show_default=True)
    @with_appcontext
    def platform_treasury_status(network: str) -> None:
        """Print platform gas treasury status without exposing secrets."""

        click.echo(json.dumps(get_service("platform_treasury").status(network=network), indent=2, default=str))

    @platform_treasury_cli.command("create")
    @click.option("--network", default="Ethereum", show_default=True)
    @click.option("--confirm", default="", help="Must be CREATE-PLATFORM-TREASURY.")
    @with_appcontext
    def platform_treasury_create(network: str, confirm: str) -> None:
        """Create the active platform gas treasury wallet."""

        if confirm != "CREATE-PLATFORM-TREASURY":
            raise click.ClickException("Refusing to create treasury without CREATE-PLATFORM-TREASURY confirmation.")
        wallet = get_service("platform_treasury").create_wallet(network=network)
        db.session.commit()
        click.echo(json.dumps({"created": True, "wallet_id": wallet.id, "network": wallet.network, "address": wallet.address}, indent=2))

    @platform_treasury_cli.command("rotate")
    @click.option("--network", default="Ethereum", show_default=True)
    @click.option("--confirm", default="", help="Must be ROTATE-PLATFORM-TREASURY.")
    @with_appcontext
    def platform_treasury_rotate(network: str, confirm: str) -> None:
        """Rotate the active platform gas treasury wallet."""

        if confirm != "ROTATE-PLATFORM-TREASURY":
            raise click.ClickException("Refusing to rotate treasury without ROTATE-PLATFORM-TREASURY confirmation.")
        wallet = get_service("platform_treasury").rotate_wallet(network=network)
        db.session.commit()
        click.echo(json.dumps({"rotated": True, "wallet_id": wallet.id, "network": wallet.network, "address": wallet.address}, indent=2))

    @platform_treasury_cli.command("top-up-withdrawal")
    @click.option("--withdrawal-id", required=True, type=int)
    @click.option("--confirm", default="", help="Must be TOP-UP-WITHDRAWAL-GAS.")
    @with_appcontext
    def platform_treasury_top_up_withdrawal(withdrawal_id: int, confirm: str) -> None:
        """Submit an ETH gas top-up for a pending EVM token withdrawal."""

        if confirm != "TOP-UP-WITHDRAWAL-GAS":
            raise click.ClickException("Refusing to top up withdrawal gas without TOP-UP-WITHDRAWAL-GAS confirmation.")
        withdrawal = db.session.get(WalletWithdrawal, int(withdrawal_id))
        if withdrawal is None:
            raise click.ClickException("Withdrawal was not found.")
        result = get_service("platform_treasury").top_up_withdrawal_gas(withdrawal)
        db.session.commit()
        click.echo(json.dumps(result, indent=2, default=str))

    @platform_treasury_cli.command("process")
    @click.option("--reserve-limit", default=25, show_default=True, type=int)
    @click.option("--withdrawal-limit", default=25, show_default=True, type=int)
    @with_appcontext
    def platform_treasury_process(reserve_limit: int, withdrawal_limit: int) -> None:
        """Process queued treasury reserve jobs and gas-sponsored withdrawals."""

        treasury = get_service("platform_treasury")
        result = treasury.process_solvency_cycle(reserve_limit=reserve_limit, withdrawal_limit=withdrawal_limit)
        db.session.commit()
        click.echo(json.dumps(result, indent=2, default=str))

    @platform_treasury_cli.command("recalculate")
    @click.option("--network", default="Ethereum", show_default=True)
    @with_appcontext
    def platform_treasury_recalculate(network: str) -> None:
        """Recalculate global treasury solvency and persist a fresh snapshot."""

        state = get_service("treasury_solvency").recalculate(network=network, persist=True, create_alerts=True)
        db.session.commit()
        click.echo(json.dumps(get_service("treasury_solvency").solvency_payload(network=state.network), indent=2, default=str))

    @platform_treasury_cli.command("forecast")
    @click.option("--network", default="Ethereum", show_default=True)
    @with_appcontext
    def platform_treasury_forecast(network: str) -> None:
        """Print the latest treasury reserve forecasts."""

        solvency = get_service("treasury_solvency")
        solvency.recalculate(network=network, persist=True, create_alerts=True)
        db.session.commit()
        click.echo(json.dumps({"network": network, "forecasts": solvency.solvency_payload(network=network).get("forecasts", [])}, indent=2, default=str))

    @platform_treasury_cli.command("rebalance")
    @click.option("--network", default="Ethereum", show_default=True)
    @click.option("--force", is_flag=True, default=False)
    @with_appcontext
    def platform_treasury_rebalance(network: str, force: bool) -> None:
        """Queue ETH reserve acquisition when solvency is below target."""

        result = get_service("treasury_solvency").rebalance_if_needed(network=network, force=force)
        db.session.commit()
        click.echo(json.dumps(result, indent=2, default=str))

    @app.cli.command("reconcile-wallet-withdrawals")
    @click.option("--username", default="", help="Optional username scope.")
    @click.option("--withdrawal-id", default=None, type=int, help="Optional withdrawal id scope.")
    @click.option("--confirm", default="", help="Must be RECONCILE-WALLET-WITHDRAWALS to write changes.")
    @with_appcontext
    def reconcile_wallet_withdrawals(username: str, withdrawal_id: int | None, confirm: str) -> None:
        """Dry-run or apply on-chain reconciliation for submitted wallet withdrawals."""

        confirmed = confirm == "RECONCILE-WALLET-WITHDRAWALS"
        result = _reconcile_wallet_withdrawals_payload(
            username=username,
            withdrawal_id=withdrawal_id,
            confirmed=confirmed,
            confirmation_value=confirm,
        )
        click.echo(json.dumps(result, indent=2, default=str))
        if result.get("blockers"):
            raise click.exceptions.Exit(1)

    @app.cli.command("recover-evm-token-deposit")
    @click.option("--user-id", required=True, type=int)
    @click.option("--asset", required=True, help="Supported ERC-20 token asset, such as USDT.")
    @click.option("--address", required=True, help="Existing generated EVM address that received the token.")
    @click.option("--tx-hash", required=True, help="On-chain ERC-20 transfer transaction hash.")
    @click.option("--confirm", default="", help="Must be RECOVER-EVM-TOKEN to write recovery records.")
    @with_appcontext
    def recover_evm_token_deposit(user_id: int, asset: str, address: str, tx_hash: str, confirm: str) -> None:
        """Recover supported ERC-20 tokens sent to an existing generated EVM address."""

        confirmed = confirm == "RECOVER-EVM-TOKEN"
        result = get_service("wallet_custody").recover_evm_token_deposit(
            user_id=user_id,
            asset=asset,
            address=address,
            tx_hash=tx_hash,
            confirm=confirmed,
        )
        db.session.commit()
        click.echo(json.dumps(result, indent=2, default=str))
        if confirmed and result.get("blockers"):
            raise click.exceptions.Exit(1)

    @app.cli.command("repair-limited-cycle")
    @click.option("--cycle-id", required=True, type=int)
    @click.option("--confirm", default="", help="Must be REPAIR-LIMITED-CYCLE to release funds.")
    @with_appcontext
    def repair_limited_cycle(cycle_id: int, confirm: str) -> None:
        """Release funds from a live no-order cycle that failed before execution."""

        confirmed = confirm == "REPAIR-LIMITED-CYCLE"
        result = _repair_limited_cycle_payload(cycle_id, confirmed=confirmed, confirmation_value=confirm)
        click.echo(json.dumps(result, indent=2, default=str))
        if result.get("blockers"):
            raise click.exceptions.Exit(1)

    @app.cli.command("production-readiness")
    @click.option("--strict", is_flag=True, default=False, help="Exit nonzero when production blockers are present.")
    @click.option("--provider", default="global", show_default=True, type=click.Choice(["global", "hyperliquid", "kucoin"]))
    @click.option("--horizon", default="1h", show_default=True)
    @with_appcontext
    def production_readiness(strict: bool, provider: str, horizon: str) -> None:
        """Print local live-production readiness diagnostics."""

        payload = _production_readiness_payload_for(provider=provider, horizon=horizon)
        click.echo(json.dumps(payload, indent=2, default=str))
        if strict and payload["blockers"]:
            raise click.exceptions.Exit(1)

    @app.cli.command("activate-trading-connection")
    @click.option("--user-id", required=True, type=int)
    @click.option("--connection-id", required=True, type=int)
    @click.option("--confirm", default="", help="Must be ACTIVATE-LIVE-CONNECTION.")
    @with_appcontext
    def activate_trading_connection(user_id: int, connection_id: int, confirm: str) -> None:
        """Enable one verified live trading connection without disabling others."""

        if confirm != "ACTIVATE-LIVE-CONNECTION":
            raise click.ClickException("Refusing activation. Pass --confirm ACTIVATE-LIVE-CONNECTION.")

        service = get_service("trading_connections")
        previous_active = TradingConnection.query.filter_by(user_id=user_id, is_active=True).order_by(TradingConnection.id.asc()).all()
        previous_payload = [
            {
                "connection_id": connection.id,
                "provider": connection.provider,
                "verification_status": connection.verification_status,
            }
            for connection in previous_active
        ]
        try:
            connection = service.activate_verified(user_id, connection_id)
        except PermissionError as exc:
            raise click.ClickException("Trading connection was not found for this user.") from exc
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        enabled_connections = TradingConnection.query.filter_by(
            user_id=user_id,
            is_active=True,
            verification_status="verified",
        ).order_by(TradingConnection.id.asc()).all()
        audit = AuditLog(
            category="connections",
            action="trading_connection_activated",
            message=f"{connection.provider.title()} trading connection enabled by operator CLI.",
            user_id=user_id,
            trading_connection_id=connection.id,
        )
        audit.details = {
            "provider": connection.provider,
            "connection_id": connection.id,
            "previous_active_connections": previous_payload,
            "enabled_connection_ids": [item.id for item in enabled_connections],
        }
        db.session.add(audit)
        db.session.commit()

        click.echo(
            json.dumps(
                {
                    "activated": True,
                    "enabled": True,
                    "user_id": user_id,
                    "active_connection_id": connection.id,
                    "provider": connection.provider,
                    "verification_status": connection.verification_status,
                    "enabled_connection_ids": [item.id for item in enabled_connections],
                },
                indent=2,
                default=str,
            )
        )

    @app.cli.command("promote-live-ranker")
    @click.option("--horizon", default="1h", show_default=True)
    @click.option("--confirm", default="", help="Must be PROMOTE-LIVE-RANKER.")
    @with_appcontext
    def promote_live_ranker(horizon: str, confirm: str) -> None:
        """Promote quarantined live ranker events after guardrail checks."""

        if confirm != "PROMOTE-LIVE-RANKER":
            raise click.ClickException("Refusing promotion. Pass --confirm PROMOTE-LIVE-RANKER.")

        result = get_service("online_ranker").promote_quarantined_events(horizon)
        if not bool(result.get("promoted", False)):
            Setting.set_json("high_upside_live_disabled", True)
            Setting.set_json(
                "high_upside_live_disabled_reason",
                {
                    "reason": "model_promotion_diagnostic_failure",
                    "horizon": horizon,
                    "blockers": result.get("blockers", []),
                },
            )
            db.session.add(
                AuditLog(
                    category="risk",
                    action="high_upside_auto_disabled",
                    message="High-upside live profile auto-disabled after live-ranker promotion diagnostics failed.",
                )
            )
        db.session.commit()
        click.echo(json.dumps(result, indent=2, default=str))

    @app.cli.command("train-offline-ranker")
    @click.option("--horizon", default="1h", show_default=True)
    @click.option("--provider", default="global", show_default=True, type=click.Choice(["global", "hyperliquid", "kucoin"]))
    @click.option("--model", "model_types", default="both", show_default=True, type=click.Choice(["sklearn", "xgboost", "both"]))
    @click.option("--use-market-history/--no-use-market-history", default=False, show_default=True)
    @click.option("--confirm", default="", help="Must be TRAIN-OFFLINE-RANKER.")
    @with_appcontext
    def train_offline_ranker(horizon: str, provider: str, model_types: str, use_market_history: bool, confirm: str) -> None:
        """Train candidate offline ranker artifacts from historical and quarantined outcomes."""

        if confirm != "TRAIN-OFFLINE-RANKER":
            raise click.ClickException("Refusing offline training. Pass --confirm TRAIN-OFFLINE-RANKER.")

        result = _call_with_supported_kwargs(
            get_service("offline_ranker").train,
            horizon,
            model_types=model_types,
            provider=provider,
            use_market_history=use_market_history,
        )
        click.echo(json.dumps(result, indent=2, default=str))

    @app.cli.command("promote-offline-ranker")
    @click.option("--horizon", default="1h", show_default=True)
    @click.option("--provider", default="global", show_default=True, type=click.Choice(["global", "hyperliquid", "kucoin"]))
    @click.option("--model-id", required=True, type=int)
    @click.option("--confirm", default="", help="Must be PROMOTE-OFFLINE-RANKER.")
    @with_appcontext
    def promote_offline_ranker(horizon: str, provider: str, model_id: int, confirm: str) -> None:
        """Promote one candidate offline ranker artifact after guardrail checks."""

        if confirm != "PROMOTE-OFFLINE-RANKER":
            raise click.ClickException("Refusing offline promotion. Pass --confirm PROMOTE-OFFLINE-RANKER.")

        result = get_service("offline_ranker").promote(horizon, model_id=model_id, provider=provider)
        if not bool(result.get("promoted", False)):
            Setting.set_json("high_upside_live_disabled", True)
            Setting.set_json(
                "high_upside_live_disabled_reason",
                {
                    "reason": "offline_model_promotion_diagnostic_failure",
                    "horizon": horizon,
                    "model_id": model_id,
                    "blockers": result.get("blockers", []),
                },
            )
            db.session.add(
                AuditLog(
                    category="risk",
                    action="high_upside_auto_disabled",
                    message="High-upside live profile auto-disabled after offline-ranker promotion diagnostics failed.",
                )
            )
            db.session.commit()
        click.echo(json.dumps(result, indent=2, default=str))

    @app.cli.command("train-ml-signal-model")
    @click.option("--horizon", default="1h", show_default=True)
    @click.option("--provider", default="global", show_default=True, type=click.Choice(["global", "hyperliquid", "kucoin"]))
    @click.option("--model", "model_type", default="pytorch_gru", show_default=True, type=click.Choice(["pytorch_gru"]))
    @click.option(
        "--objective",
        default="risk_adjusted",
        show_default=True,
        type=click.Choice(["risk_adjusted", "extreme_upside", "extreme_roi_1h", "one_h10", "consistent_roi_1w"]),
    )
    @click.option("--use-market-history/--no-use-market-history", default=False, show_default=True)
    @click.option("--confirm", default="", help="Must be TRAIN-ML-SIGNAL-MODEL.")
    @with_appcontext
    def train_ml_signal_model(
        horizon: str,
        provider: str,
        model_type: str,
        objective: str,
        use_market_history: bool,
        confirm: str,
    ) -> None:
        """Train a candidate promoted-signal model for high-upside vault runs."""

        if confirm != "TRAIN-ML-SIGNAL-MODEL":
            raise click.ClickException("Refusing signal-model training. Pass --confirm TRAIN-ML-SIGNAL-MODEL.")

        result = _call_with_supported_kwargs(
            get_service("ml_signal_model").train,
            horizon,
            model_type=model_type,
            objective=objective,
            use_market_history=use_market_history,
            provider=provider,
        )
        click.echo(json.dumps(result, indent=2, default=str))

    @app.cli.command("sweep-ml-signal-model")
    @click.option("--horizon", default="1h", show_default=True)
    @click.option("--provider", default="both", show_default=True, type=click.Choice(["global", "hyperliquid", "kucoin", "both"]))
    @click.option("--model", "model_type", default="pytorch_gru", show_default=True, type=click.Choice(["pytorch_gru"]))
    @click.option(
        "--objective",
        default="risk_adjusted",
        show_default=True,
        type=click.Choice(["risk_adjusted", "extreme_upside", "extreme_roi_1h", "one_h10", "consistent_roi_1w"]),
    )
    @click.option("--use-market-history/--no-use-market-history", default=True, show_default=True)
    @click.option("--thresholds", default="0.0005,0.001,0.002,0.003,0.005", show_default=True)
    @click.option("--epochs", default="16,32", show_default=True)
    @click.option("--min-confidences", default="0.55,0.60", show_default=True)
    @click.option("--max-runs-per-provider", default=6, show_default=True, type=int)
    @click.option("--confirm", default="", help="Must be SWEEP-ML-SIGNAL-MODEL.")
    @with_appcontext
    def sweep_ml_signal_model(
        horizon: str,
        provider: str,
        model_type: str,
        objective: str,
        use_market_history: bool,
        thresholds: str,
        epochs: str,
        min_confidences: str,
        max_runs_per_provider: int,
        confirm: str,
    ) -> None:
        """Train bounded signal-model candidates across conservative label/confidence overlays."""

        if confirm != "SWEEP-ML-SIGNAL-MODEL":
            raise click.ClickException("Refusing signal-model sweep. Pass --confirm SWEEP-ML-SIGNAL-MODEL.")
        payload = _sweep_ml_signal_payload(
            horizon=horizon,
            provider=provider,
            model_type=model_type,
            objective=objective,
            use_market_history=use_market_history,
            thresholds=thresholds,
            epochs=epochs,
            min_confidences=min_confidences,
            max_runs_per_provider=max_runs_per_provider,
        )
        click.echo(json.dumps(payload, indent=2, default=str))

    @app.cli.command("promote-ml-signal-model")
    @click.option("--horizon", default="1h", show_default=True)
    @click.option("--provider", default="global", show_default=True, type=click.Choice(["global", "hyperliquid", "kucoin"]))
    @click.option("--model-id", required=True, type=int)
    @click.option("--confirm", default="", help="Must be PROMOTE-ML-SIGNAL-MODEL.")
    @with_appcontext
    def promote_ml_signal_model(horizon: str, provider: str, model_id: int, confirm: str) -> None:
        """Promote one candidate signal model after guardrail checks."""

        if confirm != "PROMOTE-ML-SIGNAL-MODEL":
            raise click.ClickException("Refusing signal-model promotion. Pass --confirm PROMOTE-ML-SIGNAL-MODEL.")

        result = get_service("ml_signal_model").promote(horizon, model_id=model_id, provider=provider)
        if not bool(result.get("promoted", False)):
            Setting.set_json("high_upside_live_disabled", True)
            Setting.set_json(
                "high_upside_live_disabled_reason",
                {
                    "reason": "ml_signal_model_promotion_diagnostic_failure",
                    "horizon": horizon,
                    "provider": provider,
                    "model_id": model_id,
                    "blockers": result.get("blockers", []),
                },
            )
            db.session.add(
                AuditLog(
                    category="risk",
                    action="high_upside_auto_disabled",
                    message="High-upside live profile auto-disabled after signal-model promotion diagnostics failed.",
                )
            )
            db.session.commit()
        click.echo(json.dumps(result, indent=2, default=str))

    @app.cli.command("train-ml-suite")
    @click.option("--horizon", default="1h", show_default=True)
    @click.option("--provider", default="global", show_default=True, type=click.Choice(["global", "hyperliquid", "kucoin"]))
    @click.option(
        "--model-family",
        default="all",
        show_default=True,
        type=click.Choice([
            "all",
            "pytorch_gru_signal",
            "pytorch_extreme_upside",
            "pytorch_fibonacci",
            "pytorch_backtest_scorer",
            "pytorch_optimizer_policy",
            "pytorch_allocator",
            "pytorch_universe",
            "pytorch_ops_anomaly",
            "pytorch_risk_policy",
            "pytorch_exit_policy",
            "pytorch_cap_policy",
            "pytorch_execution_policy",
            "pytorch_roi_target",
        ]),
    )
    @click.option(
        "--objective",
        default="risk_adjusted",
        show_default=True,
        type=click.Choice(["risk_adjusted", "extreme_upside", "extreme_roi_1h", "one_h10", "consistent_roi_1w"]),
    )
    @click.option("--use-market-history/--no-use-market-history", default=False, show_default=True)
    @click.option("--confirm", default="", help="Must be TRAIN-ML-SUITE.")
    @with_appcontext
    def train_ml_suite(
        horizon: str,
        provider: str,
        model_family: str,
        objective: str,
        use_market_history: bool,
        confirm: str,
    ) -> None:
        """Train app-wide ML model families where implemented; research-only."""

        if confirm != "TRAIN-ML-SUITE":
            raise click.ClickException("Refusing ML suite training. Pass --confirm TRAIN-ML-SUITE.")

        result = _call_with_supported_kwargs(
            get_service("ml_decision_engine").train_suite,
            horizon,
            family=model_family,
            objective=objective,
            use_market_history=use_market_history,
            provider=provider,
        )
        click.echo(json.dumps(result, indent=2, default=str))

    @app.cli.command("ml-feedback-sync")
    @click.option("--horizon", default="all", show_default=True)
    @click.option("--provider", default="all", show_default=True, help="Filter feedback rows by provider, or all.")
    @click.option(
        "--source",
        "sources",
        multiple=True,
        type=click.Choice(["all", "rankings", "backtests", "orders", "vault_cycles", "ops", "market_history"]),
    )
    @click.option("--max-rows", default=None, type=int)
    @click.option("--confirm", default="", help="Must be SYNC-ML-FEEDBACK.")
    @with_appcontext
    def ml_feedback_sync(horizon: str, provider: str, sources: tuple[str, ...], max_rows: int | None, confirm: str) -> None:
        """Convert historical app outcomes into ML training feedback; research-only."""

        if confirm != "SYNC-ML-FEEDBACK":
            raise click.ClickException("Refusing ML feedback sync. Pass --confirm SYNC-ML-FEEDBACK.")

        payload = _ml_feedback_sync_payload(horizon=horizon, provider=provider, sources=list(sources), max_rows=max_rows)
        click.echo(json.dumps(payload, indent=2, default=str))

    @app.cli.command("promote-ml-suite")
    @click.option("--horizon", default="1h", show_default=True)
    @click.option("--provider", default="global", show_default=True, type=click.Choice(["global", "hyperliquid", "kucoin"]))
    @click.option("--model-id", required=True, type=int)
    @click.option("--confirm", default="", help="Must be PROMOTE-ML-SUITE.")
    @with_appcontext
    def promote_ml_suite(horizon: str, provider: str, model_id: int, confirm: str) -> None:
        """Promote one ML suite artifact after family-specific diagnostics."""

        if confirm != "PROMOTE-ML-SUITE":
            raise click.ClickException("Refusing ML suite promotion. Pass --confirm PROMOTE-ML-SUITE.")

        result = get_service("ml_decision_engine").promote_suite(horizon, model_id=model_id, provider=provider)
        if not bool(result.get("promoted", False)):
            Setting.set_json("high_upside_live_disabled", True)
            Setting.set_json(
                "high_upside_live_disabled_reason",
                {
                    "reason": "ml_suite_promotion_diagnostic_failure",
                    "horizon": horizon,
                    "provider": provider,
                    "model_id": model_id,
                    "blockers": result.get("blockers", []),
                },
            )
            db.session.add(
                AuditLog(
                    category="risk",
                    action="high_upside_auto_disabled",
                    message="High-upside live profile auto-disabled after ML suite promotion diagnostics failed.",
                )
            )
            db.session.commit()
        click.echo(json.dumps(result, indent=2, default=str))

    @app.cli.command("ml-readiness")
    @click.option("--horizon", default="1h", show_default=True)
    @click.option("--provider", default="global", show_default=True, type=click.Choice(["global", "hyperliquid", "kucoin"]))
    @click.option(
        "--model-family",
        default="all",
        show_default=True,
        type=click.Choice([
            "all",
            "pytorch_gru_signal",
            "pytorch_extreme_upside",
            "pytorch_fibonacci",
            "pytorch_backtest_scorer",
            "pytorch_optimizer_policy",
            "pytorch_allocator",
            "pytorch_universe",
            "pytorch_ops_anomaly",
            "pytorch_risk_policy",
            "pytorch_exit_policy",
            "pytorch_cap_policy",
            "pytorch_execution_policy",
            "pytorch_roi_target",
        ]),
    )
    @with_appcontext
    def ml_readiness(horizon: str, provider: str, model_family: str) -> None:
        """Print app-wide ML readiness diagnostics without placing orders."""

        click.echo(json.dumps(_ml_suite_readiness(horizon, family=model_family, provider=provider), indent=2, default=str))

    @app.cli.command("ml-quality-report")
    @click.option("--horizon", default="1h", show_default=True)
    @click.option(
        "--provider",
        default="both",
        show_default=True,
        type=click.Choice(["global", "hyperliquid", "kucoin", "both"]),
    )
    @click.option(
        "--model-family",
        default="all",
        show_default=True,
        type=click.Choice([
            "all",
            "pytorch_gru_signal",
            "pytorch_extreme_upside",
            "pytorch_fibonacci",
            "pytorch_backtest_scorer",
            "pytorch_optimizer_policy",
            "pytorch_allocator",
            "pytorch_universe",
            "pytorch_ops_anomaly",
            "pytorch_risk_policy",
            "pytorch_exit_policy",
            "pytorch_cap_policy",
            "pytorch_execution_policy",
            "pytorch_roi_target",
        ]),
    )
    @click.option("--candidate-limit", default=3, show_default=True, type=int)
    @click.option("--compact", is_flag=True, help="Print a short operator summary instead of full model diagnostics.")
    @with_appcontext
    def ml_quality_report(horizon: str, provider: str, model_family: str, candidate_limit: int, compact: bool) -> None:
        """Summarize promoted/candidate ML quality blockers without changing models or orders."""

        payload = _ml_quality_report_payload(
            horizon=horizon,
            provider=provider,
            family=model_family,
            candidate_limit=max(1, int(candidate_limit or 1)),
        )
        if compact:
            payload = _ml_quality_compact_payload(payload)
        click.echo(json.dumps(payload, indent=2, default=str))

    @app.cli.command("ml-risk-preview")
    @click.option("--ranking-id", required=True, type=int)
    @click.option("--horizon", default="1h", show_default=True, type=click.Choice(["1h", "1h10", "1w", "7d"]))
    @click.option("--user-id", default=1, show_default=True, type=int)
    @click.option("--connection-id", default=None, type=int)
    @click.option("--side", default="buy", show_default=True, type=click.Choice(["buy", "sell"]))
    @with_appcontext
    def ml_risk_preview(ranking_id: int, horizon: str, user_id: int, connection_id: int | None, side: str) -> None:
        """Preview ML risk/cap/exit/execution policy and SafetyEnvelope without submitting."""

        payload = _ml_risk_preview_payload(
            ranking_id=ranking_id,
            horizon=horizon,
            user_id=user_id,
            connection_id=connection_id,
            side=side,
        )
        click.echo(json.dumps(payload, indent=2, default=str))

    @app.cli.command("ml-live-vault-preview")
    @click.option("--user-id", default=1, show_default=True, type=int)
    @click.option("--provider", default="active", show_default=True)
    @click.option("--connection-id", default=None, type=int)
    @click.option("--cap-usdc", default="10.0", show_default=True, help="USD allocation budget, or all to use available collateral clipped by risk budgets.")
    @click.option("--horizon", default="1h", show_default=True)
    @click.option(
        "--objective",
        default="risk_adjusted",
        show_default=True,
        type=click.Choice(["risk_adjusted", "extreme_upside", "extreme_roi_1h", "consistent_roi_1w"]),
    )
    @click.option("--target-roi-pct", default=None, type=float)
    @click.option("--symbol", "symbols", multiple=True)
    @click.option("--timeout-seconds", default=None, type=float)
    @with_appcontext
    def ml_live_vault_preview(
        user_id: int,
        provider: str,
        connection_id: int | None,
        cap_usdc: str,
        horizon: str,
        objective: str,
        target_roi_pct: float | None,
        symbols: tuple[str, ...],
        timeout_seconds: float | None,
    ) -> None:
        """Preview one manually confirmed ML live vault cycle without placing orders."""

        payload = _ml_live_vault_preview_payload(
            user_id=user_id,
            provider=provider,
            connection_id=connection_id,
            cap_usdc=cap_usdc,
            horizon=horizon,
            objective=objective,
            target_roi_pct=target_roi_pct,
            symbols=list(symbols),
            timeout_seconds=timeout_seconds,
            record_audit=True,
        )
        click.echo(json.dumps(payload, indent=2, default=str))

    @app.cli.command("ml-auto-vault-cycle")
    @click.option("--user-id", default=1, show_default=True, type=int)
    @click.option("--provider", default="active", show_default=True)
    @click.option("--connection-id", default=None, type=int)
    @click.option("--cap-usdc", default="10.0", show_default=True, help="USD allocation budget, or all to use available collateral clipped by risk budgets.")
    @click.option("--horizon", default="1h", show_default=True)
    @click.option(
        "--objective",
        default="extreme_upside",
        show_default=True,
        type=click.Choice(["risk_adjusted", "extreme_upside", "extreme_roi_1h", "consistent_roi_1w"]),
    )
    @click.option("--target-roi-pct", default=None, type=float)
    @click.option("--symbol", "symbols", multiple=True)
    @click.option("--timeout-seconds", default=None, type=float)
    @click.option("--confirm", default="", help="Must match ML_AUTO_VAULT_EXACT_CONFIRMATION.")
    @with_appcontext
    def ml_auto_vault_cycle(
        user_id: int,
        provider: str,
        connection_id: int | None,
        cap_usdc: str,
        horizon: str,
        objective: str,
        target_roi_pct: float | None,
        symbols: tuple[str, ...],
        timeout_seconds: float | None,
        confirm: str,
    ) -> None:
        """Start one ML auto-vault cycle only after a fresh clean preview and exact confirmation."""

        payload = _ml_auto_vault_cycle_payload(
            user_id=user_id,
            provider=provider,
            connection_id=connection_id,
            cap_usdc=cap_usdc,
            horizon=horizon,
            objective=objective,
            target_roi_pct=target_roi_pct,
            symbols=list(symbols),
            timeout_seconds=timeout_seconds,
            confirm=confirm,
        )
        click.echo(json.dumps(payload, indent=2, default=str))

    @app.cli.command("ml-vault-tick")
    @click.option("--user-id", default=1, show_default=True, type=int)
    @click.option(
        "--provider-scope",
        default=None,
        type=click.Choice(["all", "active_provider", "multi_provider", "hyperliquid", "binance", "kucoin", "dydx", "uniswap"]),
    )
    @click.option("--cap-usdc", default=10.0, show_default=True, type=float)
    @click.option(
        "--objective",
        default="extreme_upside",
        show_default=True,
        type=click.Choice(["risk_adjusted", "extreme_upside", "extreme_roi_1h", "consistent_roi_1w"]),
    )
    @click.option("--horizon", default="1h", show_default=True)
    @click.option("--timeout-seconds", default=None, type=float)
    @click.option("--confirm", default="", help="Must be ML-VAULT-TICK.")
    @with_appcontext
    def ml_vault_tick(
        user_id: int,
        provider_scope: str | None,
        cap_usdc: float,
        objective: str,
        horizon: str,
        timeout_seconds: float | None,
        confirm: str,
    ) -> None:
        """Run one continuous ML vault decision tick; fail-closed by default."""

        payload = _ml_vault_tick_payload(
            user_id=user_id,
            provider_scope=provider_scope,
            cap_usdc=cap_usdc,
            objective=objective,
            horizon=horizon,
            timeout_seconds=timeout_seconds,
            confirm=confirm,
        )
        click.echo(json.dumps(payload, indent=2, default=str))

    @app.cli.command("vault-cycle-enforcement-tick")
    @click.option("--user-id", default=None, type=int, help="Limit enforcement to one user.")
    @with_appcontext
    def vault_cycle_enforcement_tick(user_id: int | None) -> None:
        """Run one active-trading enforcement tick for Vault Cycle engine cycles."""

        results = get_service("vault_cycle_trading_enforcer").enforce_active_cycles(user_id)
        click.echo(json.dumps({"ok": True, "cycle_count": len(results), "cycles": results}, indent=2, default=str))

    @app.cli.command("ml-live-vault-one-shot")
    @click.option("--preview-audit-id", required=True, type=int)
    @click.option("--user-id", default=1, show_default=True, type=int)
    @click.option("--confirm", default="", help="Must be ML-LIVE-VAULT-10USDC.")
    @with_appcontext
    def ml_live_vault_one_shot(preview_audit_id: int, user_id: int, confirm: str) -> None:
        """Start exactly one guarded live ML vault cycle from a fresh preview."""

        payload = _ml_live_vault_one_shot_payload(
            preview_audit_id=preview_audit_id,
            user_id=user_id,
            confirm=confirm,
        )
        click.echo(json.dumps(payload, indent=2, default=str))

    @app.cli.command("ml-shadow-evaluate")
    @click.option("--horizon", default="1h", show_default=True)
    @click.option("--provider", default="hyperliquid", show_default=True)
    @with_appcontext
    def ml_shadow_evaluate(horizon: str, provider: str) -> None:
        """Run a research-only ML decision shadow pass for provider operations."""

        readiness = _ml_suite_readiness(horizon, provider=provider)
        decision = get_service("ml_decision_engine").decision(
            "pytorch_ops_anomaly",
            {
                "provider": provider,
                "horizon": horizon,
                "rate_limited": _high_upside_rate_limit_backoff_remaining_seconds() > 0,
                "error_rate": 0.0,
                "latency_ms": 0.0,
            },
            horizon=horizon,
        )
        payload = {
            "ready": bool(readiness.get("ready", False)) and bool(decision.get("ready", False)),
            "research_only": True,
            "submitted": False,
            "live_orders_created": False,
            "provider": provider,
            "horizon": horizon,
            "ml_readiness": readiness,
            "ml_decision": decision,
            "ml_blockers": list(dict.fromkeys([*list(readiness.get("blockers", []) or []), *list(decision.get("blockers", []) or [])])),
        }
        click.echo(json.dumps(payload, indent=2, default=str))

    @app.cli.command("live-canary-trade")
    @click.option("--ranking-id", required=True, type=int)
    @click.option("--user-id", required=True, type=int)
    @click.option("--connection-id", default=None, type=int)
    @click.option("--submit", is_flag=True, default=False)
    @click.option("--confirm", default="", help="Must be LIVE-CANARY-TRADE when --submit is used.")
    @with_appcontext
    def live_canary_trade(
        ranking_id: int,
        user_id: int,
        connection_id: int | None,
        submit: bool,
        confirm: str,
    ) -> None:
        """Preview or submit one operator-approved live canary order from a ranking."""

        expected_confirmation = _live_micro_canary_confirmation_phrase()
        if submit and confirm != expected_confirmation:
            raise click.ClickException(f"Refusing live canary submission. Pass --confirm {expected_confirmation}.")

        result = _live_canary_trade_payload(
            ranking_id=ranking_id,
            user_id=user_id,
            connection_id=connection_id,
            submit=submit,
        )
        click.echo(json.dumps(result, indent=2, default=str))

    @app.cli.command("submit-live-canary")
    @click.option("--ranking-id", required=True, type=int)
    @click.option("--user-id", default=1, show_default=True, type=int)
    @click.option("--connection-id", default=None, type=int)
    @click.option("--confirm", default="", help="Must match LIVE_MICRO_CANARY_EXACT_CONFIRMATION.")
    @with_appcontext
    def submit_live_canary(
        ranking_id: int,
        user_id: int,
        connection_id: int | None,
        confirm: str,
    ) -> None:
        """Submit one explicitly confirmed live micro-canary order after preview passes."""

        result = _submit_live_canary_payload(
            ranking_id=ranking_id,
            user_id=user_id,
            connection_id=connection_id,
            confirm=confirm,
        )
        click.echo(json.dumps(result, indent=2, default=str))

    @app.cli.command("live-funds-readiness")
    @click.option("--provider", type=click.Choice(["active", "hyperliquid", "kucoin"]), default="active", show_default=True)
    @click.option("--user-id", default=None, type=int)
    @click.option("--ranking-id", default=None, type=int)
    @click.option("--connection-id", default=None, type=int)
    @with_appcontext
    def live_funds_readiness(
        provider: str,
        user_id: int | None,
        ranking_id: int | None,
        connection_id: int | None,
    ) -> None:
        """Print readiness for a guarded first live-funds canary."""

        result = _live_funds_readiness_payload(
            provider=provider,
            user_id=user_id,
            ranking_id=ranking_id,
            connection_id=connection_id,
        )
        click.echo(json.dumps(result, indent=2, default=str))


def _live_canary_trade_payload(
    *,
    ranking_id: int,
    user_id: int,
    connection_id: int | None,
    submit: bool,
    record_preview_audit: bool = True,
    record_micro_submit_audit: bool = True,
) -> dict[str, object]:
    config_preview_only = _config_flag("CANARY_PREVIEW_ONLY", True)
    micro_guardrail = _live_micro_canary_guardrail()
    micro_preview_only = bool(micro_guardrail.get("enabled", False) and micro_guardrail.get("preview_only", True))
    preview_only = bool(config_preview_only or micro_preview_only or not submit)
    ranking = db.session.get(StrategyRanking, int(ranking_id))
    user = db.session.get(User, int(user_id))
    blockers: list[str] = []

    if ranking is None:
        return {
            "ready": False,
            "submitted": False,
            "real_order_submitted": False,
            "preview_only": preview_only,
            "blockers": ["ranking_not_found"],
            "ranking_id": ranking_id,
            "user_id": user_id,
            "mode": "live",
        }
    if user is None:
        return {
            "ready": False,
            "submitted": False,
            "real_order_submitted": False,
            "preview_only": preview_only,
            "blockers": ["user_not_found"],
            "ranking_id": ranking_id,
            "user_id": user_id,
            "mode": "live",
        }
    if ranking.rejected:
        blockers.append("ranking_rejected")
    if str(ranking.rejection_reason or "").strip():
        blockers.append("ranking_has_rejection_reason")
    explanation = ranking.ml_explanation if isinstance(ranking.ml_explanation, dict) else {}
    one_hour_pref = explanation.get("one_hour_live_preference")
    if (
        ranking.profile in {"aggressive_1h", "extreme_roi_experimental"}
        and isinstance(one_hour_pref, dict)
        and one_hour_pref.get("accepted_for_one_hour_live_preview") is False
    ):
        blockers.extend(str(item) for item in one_hour_pref.get("one_hour_live_blockers", []) or [])
        blockers.append("ranking_not_accepted_for_one_hour_live_preview")

    connection, connection_blocker = _live_canary_connection(user_id, connection_id)
    if connection_blocker:
        blockers.append(connection_blocker)
    if connection is not None and normalize_exchange_provider(getattr(ranking, "provider", "global")) not in {
        "global",
        normalize_exchange_provider(connection.provider),
    }:
        blockers.append("ranking_provider_mismatch")
    available_balance_usd: float | None = None
    if connection is not None:
        snapshot = get_service("trading_connections").account_snapshot(user_id, "live", connection.id)
        available_balance_usd = _live_micro_canary_available_balance_usd(snapshot.balances or [])
        blockers.extend(_provider_balance_readiness(connection.provider, snapshot.balances or []).get("blockers", []))

    signal_payload = _live_canary_signal_payload(ranking)
    signal = signal_payload.get("signal")
    diagnostics = dict(signal_payload.get("diagnostics") or {})
    blockers.extend(str(item) for item in diagnostics.get("blockers", []))

    intent: OrderIntent | None = None
    order_payload: dict[str, object] = {}
    risk_decision_payload: dict[str, object] = {
        "approved": False,
        "rule_name": "signal_unavailable",
        "reason": "A live canary signal could not be generated.",
        "details": {},
    }

    if signal is not None and not diagnostics.get("blockers"):
        intent, intent_blockers, order_payload = _live_canary_order_intent(
            ranking=ranking,
            user_id=user_id,
            connection_id=connection.id if connection is not None else connection_id,
            connection=connection,
            available_balance_usd=available_balance_usd,
            signal=signal,
            signal_diagnostics=diagnostics,
        )
        blockers.extend(intent_blockers)
        if intent is not None:
            has_trading_access = connection is not None
            risk_decision = get_service("risk_engine").evaluate(
                intent,
                _float(diagnostics.get("mid")),
                has_trading_access,
            )
            risk_decision_payload = risk_decision.as_dict()
            if not risk_decision.approved:
                blockers.append(f"risk:{risk_decision.rule_name}")

    ready = not blockers and bool(risk_decision_payload.get("approved", False)) and intent is not None
    submission_preflight = _live_canary_submission_preflight(
        submit_requested=submit,
        config_preview_only=config_preview_only,
        preview_ready=ready,
        ranking=ranking,
        user=user,
        connection=connection,
        intent=intent,
        base_blockers=blockers,
    )
    submission_blockers = [str(item) for item in submission_preflight.get("blockers", []) or []]
    submission_ready = ready and intent is not None and not submission_blockers
    output_blockers = list(dict.fromkeys(blockers))
    if submit:
        output_blockers = list(dict.fromkeys([*output_blockers, *submission_blockers]))
    live_micro_canary = dict(order_payload.get("live_micro_canary") or micro_guardrail)
    micro_active_blockers = [
        *list(live_micro_canary.get("blockers", []) or []),
        *list(live_micro_canary.get("submission_blockers", []) or []),
        *[
            item
            for item in [*output_blockers, *submission_blockers]
            if str(item)
            in {
                "live_mode_required",
                "live_micro_canary_preview_only_enabled",
                "live_micro_canary_account_usd_invalid",
                "live_micro_canary_max_allocation_invalid",
                "live_micro_canary_max_risk_pct_invalid",
                "live_micro_canary_max_leverage_invalid",
                "live_micro_canary_order_budget_invalid",
                "live_micro_canary_default_min_notional_invalid",
                "live_micro_canary_required",
                "live_micro_canary_live_submit_disabled",
                "live_micro_canary_daily_order_limit_reached",
                "live_micro_canary_leverage_exceeds_one",
                "live_micro_canary_balance_unavailable",
                "live_micro_canary_insufficient_balance",
                "live_micro_canary_order_budget_exceeded",
                "live_micro_canary_stop_loss_risk_exceeded",
                "exchange_min_notional_unknown",
                "exchange_min_notional_exceeds_micro_cap",
                "stop_loss_required",
                "take_profit_required",
                "averaging_down_disabled",
            }
        ],
    ]
    live_micro_canary["active_blockers"] = list(dict.fromkeys(str(item) for item in micro_active_blockers))
    signal_summary = (
        diagnostics.get("signal_payload", {}) if isinstance(diagnostics.get("signal_payload"), dict) else {}
    )
    projected_order = order_payload.get("projected_order", {})
    if not isinstance(projected_order, dict):
        projected_order = {}
    signal_quality = (
        diagnostics.get("signal_quality", {}) if isinstance(diagnostics.get("signal_quality"), dict) else {}
    )
    signal_metadata = signal_summary.get("metadata", {}) if isinstance(signal_summary.get("metadata"), dict) else {}
    confidence = (
        signal_summary.get("confidence")
        or signal_metadata.get("confidence")
        or (getattr(signal, "position_fraction", None) if signal is not None else None)
        or signal_quality.get("confidence")
    )
    reason = str(
        signal_summary.get("reason")
        or signal_summary.get("rationale")
        or risk_decision_payload.get("reason")
        or ""
    )
    payload: dict[str, object] = {
        "ready": ready,
        "submitted": False,
        "real_order_submitted": False,
        "preview_only": preview_only,
        "ranking_id": ranking.id,
        "user_id": user.id,
        "connection_id": connection.id if connection is not None else connection_id,
        "provider": normalize_exchange_provider(connection.provider if connection is not None else None),
        "collateral_asset": provider_collateral_asset(connection.provider if connection is not None else None),
        "mode": "live",
        "submit_requested": submit,
        "submission_ready": submission_ready,
        "blockers": output_blockers,
        "selected_symbol": ranking.symbol,
        "selected_strategy": ranking.strategy_name,
        "side": projected_order.get("side") or signal_summary.get("action"),
        "size": projected_order.get("quantity", 0.0),
        "confidence": confidence,
        "reason": reason,
        "ranking": _ranking_canary_summary(ranking),
        "connection": {
            "active_verified": connection is not None,
            "provider": connection.provider if connection is not None else None,
            "verification_status": connection.verification_status if connection is not None else None,
        },
        "signal": signal_summary,
        "signal_quality": signal_quality,
        "sizing": order_payload.get("sizing", {}),
        "projected_order": projected_order,
        "risk_decision": risk_decision_payload,
        "live_micro_canary": live_micro_canary,
        "notional_usd": projected_order.get("notional"),
        "estimated_loss_at_stop_usd": live_micro_canary.get("estimated_stop_loss_loss_usd"),
        "estimated_fees_slippage_usd": live_micro_canary.get("estimated_fees_slippage_usd"),
        "exchange_min_notional_usd": live_micro_canary.get("exchange_min_notional_usd"),
        "min_notional_order_used": live_micro_canary.get("min_notional_order_used"),
        "live_canary_readiness": {
            "ready": ready,
            "submission_ready": submission_ready,
            "preview_only": preview_only,
            "requires_exact_confirmation_for_submit": True,
            "uses_existing_live_caps": True,
            "optimizer_research_only": True,
            "canary_preview_only_config": config_preview_only,
            "strict_production_ready": submission_preflight.get("strict_production_ready"),
            "strict_production_blockers": submission_preflight.get("strict_production_blockers", []),
            "wallet_required": submission_preflight.get("wallet_required", False),
            "wallet_readiness": submission_preflight.get("wallet_readiness"),
            "confirmation_flags": submission_preflight.get("confirmation_flags", {}),
            "submission_blockers": submission_blockers,
            "real_order_submitted": False,
            "live_micro_canary": live_micro_canary,
        },
    }

    if preview_only and record_preview_audit:
        audit_id = _record_live_canary_preview_audit(
            ranking=ranking,
            user=user,
            connection=connection,
            payload=payload,
            intent=intent,
            submit_requested=submit,
            blocked_by_preview_only=bool(submit and (config_preview_only or micro_preview_only)),
        )
        if audit_id is not None:
            payload["canary_preview_audit_id"] = audit_id

    if submit:
        attempt_audit_id = _record_live_canary_submit_attempt(
            ranking=ranking,
            user=user,
            connection=connection,
            payload=payload,
            intent=intent,
            submission_ready=submission_ready,
            submission_blockers=submission_blockers,
        )
        payload["submission_attempt_audit_id"] = attempt_audit_id
        if record_micro_submit_audit and bool(live_micro_canary.get("enabled", False)):
            payload["live_micro_canary_submit_attempt_audit_id"] = _record_live_micro_canary_submit_audit(
                action="live_micro_canary_submit_attempt",
                message=f"Live micro-canary submit attempt evaluated for ranking {ranking.id}.",
                ranking=ranking,
                user=user,
                connection_id=connection.id if connection is not None else connection_id,
                payload=payload,
                blockers=output_blockers,
            )
        if config_preview_only:
            payload["submitted"] = False
            payload["real_order_submitted"] = False
            payload["submit_blocked"] = True
            payload["submit_block_reason"] = "CANARY_PREVIEW_ONLY is true; live canary submit is disabled."
            if record_micro_submit_audit and bool(live_micro_canary.get("enabled", False)):
                payload["live_micro_canary_submit_blocked_audit_id"] = _record_live_micro_canary_submit_audit(
                    action="live_micro_canary_submit_blocked",
                    message=f"Live micro-canary submit blocked for ranking {ranking.id}.",
                    ranking=ranking,
                    user=user,
                    connection_id=connection.id if connection is not None else connection_id,
                    payload=payload,
                    blockers=output_blockers,
                )
            return payload
        if micro_preview_only:
            payload["submitted"] = False
            payload["real_order_submitted"] = False
            payload["submit_blocked"] = True
            payload["submit_block_reason"] = "LIVE_MICRO_CANARY_PREVIEW_ONLY is true; live micro-canary submit is disabled."
            if record_micro_submit_audit and bool(live_micro_canary.get("enabled", False)):
                payload["live_micro_canary_submit_blocked_audit_id"] = _record_live_micro_canary_submit_audit(
                    action="live_micro_canary_submit_blocked",
                    message=f"Live micro-canary submit blocked for ranking {ranking.id}.",
                    ranking=ranking,
                    user=user,
                    connection_id=connection.id if connection is not None else connection_id,
                    payload=payload,
                    blockers=output_blockers,
                )
            return payload
        if not submission_ready or intent is None:
            payload["submitted"] = False
            payload["real_order_submitted"] = False
            payload["submit_blocked"] = True
            payload["submit_block_reason"] = _live_canary_submit_block_reason(output_blockers)
            if record_micro_submit_audit and bool(live_micro_canary.get("enabled", False)):
                payload["live_micro_canary_submit_blocked_audit_id"] = _record_live_micro_canary_submit_audit(
                    action="live_micro_canary_submit_blocked",
                    message=f"Live micro-canary submit blocked for ranking {ranking.id}.",
                    ranking=ranking,
                    user=user,
                    connection_id=connection.id if connection is not None else connection_id,
                    payload=payload,
                    blockers=output_blockers,
                )
            return payload
        order = get_service("order_manager").place_order(intent)
        order_status = str(getattr(order, "status", "") or "").lower()
        real_order_submitted = order_status not in {"rejected", "failed"}
        order_details = getattr(order, "details", {}) if isinstance(getattr(order, "details", {}), dict) else {}
        exchange_response = order_details.get("exchange_response", {}) if isinstance(order_details, dict) else {}
        fills = list(getattr(order, "fills", []) or [])
        audit = AuditLog(
            category="orders",
            action="live_canary_trade",
            message=f"Live canary trade attempted from ranking {ranking.id}.",
            user_id=user.id,
            trading_connection_id=intent.trading_connection_id,
        )
        audit.details = {
            "ranking_id": ranking.id,
            "optimizer_profile": ranking.profile,
            "strategy_name": ranking.strategy_name,
            "symbol": ranking.symbol,
            "timeframe": ranking.timeframe,
            "order_id": order.id,
            "order_status": getattr(order, "status", None),
            "real_order_submitted": real_order_submitted,
            "rejection_reason": getattr(order, "rejection_reason", None),
            "exchange_response": exchange_response,
            "fill_count": len(fills),
            "risk_decision": risk_decision_payload,
            "signal_quality": diagnostics.get("signal_quality", {}),
        }
        db.session.add(audit)
        db.session.commit()
        payload["submitted"] = True
        payload["real_order_submitted"] = real_order_submitted
        payload["preview_only"] = False
        payload["live_canary_readiness"]["preview_only"] = False
        payload["live_canary_readiness"]["real_order_submitted"] = real_order_submitted
        payload["order"] = {
            "id": order.id,
            "status": order.status,
            "risk_status": order.risk_status,
            "rejection_reason": order.rejection_reason,
            "client_order_id": order.client_order_id,
            "exchange_response": exchange_response,
            "fill_count": len(fills),
        }
        if record_micro_submit_audit and bool(live_micro_canary.get("enabled", False)):
            action = "live_micro_canary_submit_success" if real_order_submitted else "live_micro_canary_submit_rejected"
            payload["live_micro_canary_submit_final_audit_id"] = _record_live_micro_canary_submit_audit(
                action=action,
                message=f"Live micro-canary submit {'succeeded' if real_order_submitted else 'was rejected'} for ranking {ranking.id}.",
                ranking=ranking,
                user=user,
                connection_id=intent.trading_connection_id,
                payload=payload,
                order=order,
                blockers=output_blockers,
            )

    return payload


def _submit_live_canary_payload(
    *,
    ranking_id: int,
    user_id: int,
    connection_id: int | None,
    confirm: str,
) -> dict[str, object]:
    ranking = db.session.get(StrategyRanking, int(ranking_id))
    user = db.session.get(User, int(user_id))
    preview = _live_canary_trade_payload(
        ranking_id=ranking_id,
        user_id=user_id,
        connection_id=connection_id,
        submit=False,
        record_preview_audit=False,
    )
    resolved_connection_id = preview.get("connection_id") or connection_id
    try:
        resolved_connection_id = int(resolved_connection_id) if resolved_connection_id is not None else None
    except (TypeError, ValueError):
        resolved_connection_id = None
    attempt_audit_id = _record_live_micro_canary_submit_audit(
        action="live_micro_canary_submit_attempt",
        message=f"Live micro-canary submit command invoked for ranking {ranking_id}.",
        ranking=ranking,
        user=user,
        connection_id=resolved_connection_id,
        payload=preview,
        blockers=list(preview.get("blockers", []) or []),
    )

    micro_guardrail = _live_micro_canary_guardrail()
    live_micro = preview.get("live_micro_canary") if isinstance(preview.get("live_micro_canary"), dict) else micro_guardrail
    blockers: list[str] = []
    if ranking is None:
        blockers.append("ranking_not_found")
    if user is None:
        blockers.append("user_not_found")
    if str(current_app.config.get("APP_MODE", "paper")).lower() != "live":
        blockers.append("live_mode_required")
    if not _config_flag("ENABLE_LIVE_TRADING", False):
        blockers.append("live_trading_disabled")
    if not bool(micro_guardrail.get("enabled", False)):
        blockers.append("live_micro_canary_required")
    if not bool(micro_guardrail.get("live_submit_enabled", False)):
        blockers.append("live_micro_canary_live_submit_disabled")
    if _config_flag("CANARY_PREVIEW_ONLY", True):
        blockers.append("canary_preview_only_enabled")
    if bool(micro_guardrail.get("enabled", False)) and bool(micro_guardrail.get("preview_only", True)):
        blockers.append("live_micro_canary_preview_only_enabled")
    if bool(micro_guardrail.get("require_exact_confirmation", True)) and confirm != _live_micro_canary_confirmation_phrase():
        blockers.append("live_micro_canary_exact_confirmation_required")

    confirmation_flags = {
        "config_explicit_live_confirmed": _config_flag("EXPLICIT_LIVE_CONFIRMED", False),
        "config_secondary_confirmation": _config_flag("SECONDARY_CONFIRMATION", False),
        "setting_explicit_live_confirmed": bool(Setting.get_json("explicit_live_confirmed", False)),
        "setting_secondary_confirmation": bool(Setting.get_json("secondary_confirmation", False)),
    }
    if not confirmation_flags["config_explicit_live_confirmed"]:
        blockers.append("explicit_live_confirmed_required")
    if not confirmation_flags["config_secondary_confirmation"]:
        blockers.append("secondary_confirmation_required")
    if not confirmation_flags["setting_explicit_live_confirmed"]:
        blockers.append("setting_explicit_live_confirmed_required")
    if not confirmation_flags["setting_secondary_confirmation"]:
        blockers.append("setting_secondary_confirmation_required")
    if Setting.get_json("panic_lock", False):
        blockers.append("panic_lock_active")
    if resolved_connection_id is None:
        blockers.append("active_verified_live_connection_missing")
    max_daily = int(micro_guardrail.get("max_daily_live_orders") or 0)
    daily_success_count = _live_micro_canary_daily_success_count(user.id if user is not None else None)
    if max_daily > 0 and daily_success_count >= max_daily:
        blockers.append("live_micro_canary_daily_order_limit_reached")
    blockers.extend(str(item) for item in micro_guardrail.get("blockers", []) or [])
    blockers.extend(str(item) for item in live_micro.get("active_blockers", []) or [])

    if not bool(preview.get("ready", False)):
        blockers.append("live_canary_preview_not_ready")
        blockers.extend(str(item) for item in preview.get("blockers", []) or [])

    blockers = list(dict.fromkeys(str(item) for item in blockers if str(item)))
    base_payload = {
        "ready": False,
        "submitted": False,
        "order_id": None,
        "ranking_id": ranking_id,
        "user_id": user_id,
        "connection_id": resolved_connection_id,
        "symbol": preview.get("selected_symbol"),
        "strategy": preview.get("selected_strategy"),
        "notional_usd": preview.get("notional_usd"),
        "estimated_loss_at_stop_usd": preview.get("estimated_loss_at_stop_usd"),
        "estimated_fees_slippage_usd": preview.get("estimated_fees_slippage_usd"),
        "exchange_min_notional_usd": preview.get("exchange_min_notional_usd"),
        "min_notional_order_used": preview.get("min_notional_order_used"),
        "live_micro_canary": live_micro,
        "blockers": blockers,
        "attempt_audit_log_id": attempt_audit_id,
        "audit_log_id": attempt_audit_id,
        "preview": preview,
        "confirmation_flags": confirmation_flags,
        "daily_success_count": daily_success_count,
        "next_commands": {
            "preview": f"flask live-canary-trade --ranking-id {ranking_id} --user-id {user_id}",
            "strict_readiness": "flask production-readiness --strict",
            "find_ranking": "flask find-live-canary-ranking --research-depth quick --symbol BTC",
        },
    }
    if blockers:
        block_audit_id = _record_live_micro_canary_submit_audit(
            action="live_micro_canary_submit_blocked",
            message=f"Live micro-canary submit command blocked for ranking {ranking_id}.",
            ranking=ranking,
            user=user,
            connection_id=resolved_connection_id,
            payload=base_payload,
            blockers=blockers,
        )
        base_payload["blocked_audit_log_id"] = block_audit_id
        base_payload["audit_log_id"] = block_audit_id
        return base_payload

    submission = _live_canary_trade_payload(
        ranking_id=ranking_id,
        user_id=user_id,
        connection_id=resolved_connection_id,
        submit=True,
        record_preview_audit=False,
        record_micro_submit_audit=False,
    )
    order_payload = submission.get("order") if isinstance(submission.get("order"), dict) else {}
    submitted = bool(submission.get("submitted", False))
    real_order_submitted = bool(submission.get("real_order_submitted", False))
    final_blockers = [str(item) for item in submission.get("blockers", []) or []]
    final_action = (
        "live_micro_canary_submit_success"
        if submitted and real_order_submitted
        else "live_micro_canary_submit_rejected"
        if submitted
        else "live_micro_canary_submit_blocked"
    )
    final_audit_id = _record_live_micro_canary_submit_audit(
        action=final_action,
        message=(
            f"Live micro-canary submit command {'succeeded' if final_action.endswith('success') else 'completed without accepted live submission'} "
            f"for ranking {ranking_id}."
        ),
        ranking=ranking,
        user=user,
        connection_id=resolved_connection_id,
        payload={**submission, "live_micro_canary": submission.get("live_micro_canary", live_micro)},
        blockers=final_blockers,
    )
    return {
        "ready": submitted and real_order_submitted,
        "submitted": submitted,
        "real_order_submitted": real_order_submitted,
        "order_id": order_payload.get("id"),
        "ranking_id": ranking_id,
        "user_id": user_id,
        "connection_id": resolved_connection_id,
        "symbol": submission.get("selected_symbol"),
        "strategy": submission.get("selected_strategy"),
        "notional_usd": submission.get("notional_usd"),
        "estimated_loss_at_stop_usd": submission.get("estimated_loss_at_stop_usd"),
        "estimated_fees_slippage_usd": submission.get("estimated_fees_slippage_usd"),
        "exchange_min_notional_usd": submission.get("exchange_min_notional_usd"),
        "min_notional_order_used": submission.get("min_notional_order_used"),
        "live_micro_canary": submission.get("live_micro_canary", live_micro),
        "blockers": final_blockers,
        "attempt_audit_log_id": attempt_audit_id,
        "audit_log_id": final_audit_id,
        "order": order_payload,
        "preview": preview,
        "submission": submission,
        "next_commands": {
            "post_submit_readiness": f"flask live-funds-readiness --provider active --user-id {user_id} --ranking-id {ranking_id}",
            "restore_preview_only": "Set LIVE_MICRO_CANARY_PREVIEW_ONLY=true and CANARY_PREVIEW_ONLY=true after the canary attempt.",
        },
    }


def _live_canary_submission_preflight(
    *,
    submit_requested: bool,
    config_preview_only: bool,
    preview_ready: bool,
    ranking: StrategyRanking,
    user: User,
    connection: TradingConnection | None,
    intent: OrderIntent | None,
    base_blockers: list[str],
) -> dict[str, object]:
    blockers: list[str] = []
    strict_production_ready: bool | None = None
    strict_production_blockers: list[str] = []
    production: dict[str, object] = {}
    wallet_readiness: dict[str, object] | None = None
    wallet_required = _live_canary_requires_wallet_readiness(ranking, intent)
    confirmation_flags = {
        "config_explicit_live_confirmed": _config_flag("EXPLICIT_LIVE_CONFIRMED", False),
        "config_secondary_confirmation": _config_flag("SECONDARY_CONFIRMATION", False),
        "setting_explicit_live_confirmed": bool(Setting.get_json("explicit_live_confirmed", False)),
        "setting_secondary_confirmation": bool(Setting.get_json("secondary_confirmation", False)),
    }
    micro_guardrail = _live_micro_canary_guardrail()

    if str(current_app.config.get("APP_MODE", "paper")).lower() != "live":
        blockers.append("live_mode_required")
    if not _config_flag("ENABLE_LIVE_TRADING", False):
        blockers.append("live_trading_disabled")
    if config_preview_only:
        blockers.append("canary_preview_only_enabled")
    if bool(micro_guardrail.get("enabled", False)) and bool(micro_guardrail.get("preview_only", True)):
        blockers.append("live_micro_canary_preview_only_enabled")
    if submit_requested:
        if not bool(micro_guardrail.get("enabled", False)):
            blockers.append("live_micro_canary_required")
        if not bool(micro_guardrail.get("live_submit_enabled", False)):
            blockers.append("live_micro_canary_live_submit_disabled")
        max_daily = int(micro_guardrail.get("max_daily_live_orders") or 0)
        daily_success_count = _live_micro_canary_daily_success_count(user.id)
        if max_daily > 0 and daily_success_count >= max_daily:
            blockers.append("live_micro_canary_daily_order_limit_reached")
    blockers.extend(str(item) for item in micro_guardrail.get("blockers", []) or [])
    if not confirmation_flags["config_explicit_live_confirmed"]:
        blockers.append("explicit_live_confirmed_required")
    if not confirmation_flags["config_secondary_confirmation"]:
        blockers.append("secondary_confirmation_required")
    if not confirmation_flags["setting_explicit_live_confirmed"]:
        blockers.append("setting_explicit_live_confirmed_required")
    if not confirmation_flags["setting_secondary_confirmation"]:
        blockers.append("setting_secondary_confirmation_required")
    if Setting.get_json("panic_lock", False):
        blockers.append("panic_lock_active")
    if connection is None:
        blockers.append("active_verified_live_connection_missing")
    if ranking.rejected or str(ranking.rejection_reason or "").strip():
        blockers.append("accepted_ranking_required")
    if intent is None:
        blockers.append("order_intent_missing")
    elif not intent.stop_loss:
        blockers.append("stop_loss_required")
    elif submit_requested and bool(micro_guardrail.get("enabled", False)):
        intent_metadata = dict(intent.metadata or {})
        canary_sizing = (
            dict(intent_metadata.get("canary_sizing") or {})
            if isinstance(intent_metadata.get("canary_sizing"), dict)
            else {}
        )
        requested_leverage = _float(canary_sizing.get("requested_leverage"), intent.leverage)
        max_micro_leverage = _float(micro_guardrail.get("max_leverage"), 1.0)
        leverage_limit = min(1.0, max_micro_leverage) if max_micro_leverage > 0 else 1.0
        if requested_leverage > leverage_limit:
            blockers.append("live_micro_canary_leverage_exceeds_one")
    if base_blockers:
        blockers.extend(str(item) for item in base_blockers)
    if not preview_ready:
        blockers.append("preview_not_ready")

    if submit_requested:
        production = _production_readiness_payload()
        strict_production_ready = bool(production.get("ready", False))
        strict_production_blockers = [str(item) for item in production.get("blockers", []) or []]
        if not strict_production_ready:
            blockers.append("strict_readiness_required")
        if wallet_required:
            wallet = production.get("wallet") if isinstance(production.get("wallet"), dict) else _wallet_readiness_payload()
            wallet_readiness = dict(wallet)
            if not bool(wallet_readiness.get("ready", False)):
                blockers.append("wallet_readiness_required")

    return {
        "blockers": list(dict.fromkeys(blockers)),
        "strict_production_ready": strict_production_ready,
        "strict_production_blockers": strict_production_blockers,
        "wallet_required": wallet_required,
        "wallet_readiness": wallet_readiness,
        "confirmation_flags": confirmation_flags,
    }


def _live_canary_requires_wallet_readiness(ranking: StrategyRanking, intent: OrderIntent | None) -> bool:
    parameters = ranking.parameters if isinstance(ranking.parameters, dict) else {}
    metadata = dict(intent.metadata or {}) if intent is not None else {}
    wallet_markers = (
        "consumer_vault",
        "vault_cycle_id",
        "vault_leg_id",
        "wallet_funded",
        "wallet_funds_used",
    )
    return any(bool(parameters.get(key) or metadata.get(key)) for key in wallet_markers)


def _record_live_canary_submit_attempt(
    *,
    ranking: StrategyRanking,
    user: User,
    connection: TradingConnection | None,
    payload: dict[str, object],
    intent: OrderIntent | None,
    submission_ready: bool,
    submission_blockers: list[str],
) -> int:
    audit_connection_id = intent.trading_connection_id if intent is not None else None
    if audit_connection_id is None and connection is not None:
        audit_connection_id = connection.id
    audit = AuditLog(
        category="orders",
        action="live_canary_submit_attempt",
        message=f"Live canary submit attempt evaluated for ranking {ranking.id}.",
        user_id=user.id,
        trading_connection_id=audit_connection_id,
    )
    readiness = payload.get("live_canary_readiness", {}) if isinstance(payload.get("live_canary_readiness"), dict) else {}
    audit.details = {
        "ranking_id": ranking.id,
        "optimizer_profile": ranking.profile,
        "strategy_name": ranking.strategy_name,
        "symbol": ranking.symbol,
        "timeframe": ranking.timeframe,
        "connection_id": audit_connection_id,
        "provider": connection.provider if connection is not None else None,
        "submission_ready": submission_ready,
        "blockers": list(dict.fromkeys(submission_blockers)),
        "projected_order": payload.get("projected_order", {}),
        "sizing": payload.get("sizing", {}),
        "risk_decision": payload.get("risk_decision", {}),
        "signal_quality": payload.get("signal_quality", {}),
        "strict_production_ready": readiness.get("strict_production_ready"),
        "strict_production_blockers": readiness.get("strict_production_blockers", []),
        "wallet_required": readiness.get("wallet_required", False),
        "confirmation_flags": readiness.get("confirmation_flags", {}),
    }
    db.session.add(audit)
    db.session.commit()
    return int(audit.id)


def _live_canary_submit_block_reason(blockers: list[str]) -> str:
    unique = [str(item) for item in dict.fromkeys(blockers) if str(item).strip()]
    if not unique:
        return "Live canary submit blocked by readiness checks."
    return f"Live canary submit blocked: {unique[0]}."


def _config_flag(name: str, default: bool = False) -> bool:
    value = current_app.config.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _compact_rapid_ml_payload(payload: dict[str, object]) -> dict[str, object]:
    """Return operator-focused rapid ML output without deep model metadata."""

    provider_rows: list[dict[str, object]] = []
    cycles = [cycle for cycle in list(payload.get("cycles", []) or []) if isinstance(cycle, dict)]
    for cycle in cycles[-1:]:
        if not isinstance(cycle, dict):
            continue
        executions_by_connection = {
            int(item.get("connection_id") or 0): item
            for item in list(cycle.get("executions", []) or [])
            if isinstance(item, dict)
        }
        position_management_by_connection = {
            int(item.get("connection_id") or 0): item
            for item in list(cycle.get("position_management", []) or [])
            if isinstance(item, dict)
        }
        for analysis in list(cycle.get("analyses", []) or []):
            if not isinstance(analysis, dict):
                continue
            connection_id = int(analysis.get("connection_id") or 0)
            snapshot = analysis.get("snapshot") if isinstance(analysis.get("snapshot"), dict) else {}
            selected = analysis.get("selected") if isinstance(analysis.get("selected"), dict) else {}
            execution = executions_by_connection.get(connection_id, {})
            position_management = position_management_by_connection.get(connection_id, {})
            candidate_rows: list[dict[str, object]] = []
            candidates = [item for item in list(analysis.get("candidates", []) or []) if isinstance(item, dict)]
            selected_symbol = str(selected.get("symbol") or "").upper() if selected else ""
            ordered_candidates: list[dict[str, object]] = []
            seen_symbols: set[str] = set()
            if selected_symbol:
                for candidate in candidates:
                    if str(candidate.get("symbol") or "").upper() == selected_symbol:
                        ordered_candidates.append(candidate)
                        seen_symbols.add(selected_symbol)
                        break
            for candidate in candidates:
                symbol_key = str(candidate.get("symbol") or "").upper()
                if symbol_key in seen_symbols:
                    continue
                ordered_candidates.append(candidate)
                if len(ordered_candidates) >= 6:
                    break
            for candidate in ordered_candidates[:6]:
                if not isinstance(candidate, dict):
                    continue
                profitability = candidate.get("profitability") if isinstance(candidate.get("profitability"), dict) else {}
                ml_decisions = candidate.get("ml_decisions") if isinstance(candidate.get("ml_decisions"), dict) else {}
                signal = ml_decisions.get("pytorch_gru_signal") if isinstance(ml_decisions.get("pytorch_gru_signal"), dict) else {}
                risk_policy = (
                    ml_decisions.get("pytorch_risk_policy")
                    if isinstance(ml_decisions.get("pytorch_risk_policy"), dict)
                    else {}
                )
                candidate_rows.append(
                    {
                        "symbol": candidate.get("symbol"),
                        "action": candidate.get("action"),
                        "confidence": candidate.get("confidence"),
                        "signal_action": signal.get("action"),
                        "signal_expected_return": signal.get("expected_return"),
                        "risk_policy_action": risk_policy.get("action"),
                        "risk_policy_expected_return": risk_policy.get("expected_return"),
                        "edge_bps_after_costs": candidate.get("expected_edge_bps_after_costs"),
                        "total_cost_bps": profitability.get("total_cost_bps"),
                        "positive_edge_sources": profitability.get("positive_edge_sources", []),
                        "blockers": list(candidate.get("blockers", []) or [])[:8],
                    }
                )
            provider_rows.append(
                {
                    "provider": analysis.get("provider"),
                    "connection_id": connection_id,
                    "ready": bool(analysis.get("ready", False)),
                    "can_trade": bool(analysis.get("can_trade", False)),
                    "available_balance_usd": snapshot.get("available_balance_usd"),
                    "open_orders_count": snapshot.get("open_orders_count"),
                    "positions_count": snapshot.get("positions_count"),
                    "selected": {
                        "symbol": selected.get("symbol"),
                        "action": selected.get("action"),
                        "side": selected.get("side"),
                        "confidence": selected.get("confidence"),
                        "edge_bps_after_costs": selected.get("expected_edge_bps_after_costs"),
                        "opportunity_score": selected.get("opportunity_score"),
                        "ml_sizing": selected.get("ml_sizing", {}),
                        "all_futures_universe": bool(selected.get("rapid_ml_all_futures_universe")),
                        "futures_market_id": selected.get("futures_market_id"),
                    }
                    if selected
                    else None,
                    "candidate_summaries": candidate_rows,
                    "analysis_blockers": list(analysis.get("blockers", []) or [])[:10],
                    "execution_blockers": list(execution.get("blockers", []) or [])[:10] if isinstance(execution, dict) else [],
                    "execution_status": execution.get("status") if isinstance(execution, dict) else None,
                    "position_management": position_management if isinstance(position_management, dict) else {},
                    "order": execution.get("order") if isinstance(execution.get("order"), dict) else {},
                    "preview_ready": bool(execution.get("preview_ready", False)) if isinstance(execution, dict) else False,
                    "submitted": bool(execution.get("submitted", False)) if isinstance(execution, dict) else False,
                }
            )
    safety = payload.get("safety") if isinstance(payload.get("safety"), dict) else {}
    return {
        "ready": payload.get("ready"),
        "mode": payload.get("mode"),
        "session_id": payload.get("session_id"),
        "status": payload.get("status"),
        "provider_request": payload.get("provider_request"),
        "submit_requested": payload.get("submit_requested"),
        "real_submit_enabled": payload.get("real_submit_enabled"),
        "submitted_count": payload.get("submitted_count"),
        "real_order_submitted": payload.get("real_order_submitted"),
        "preview_ready_count": payload.get("preview_ready_count"),
        "rejected_decision_count": payload.get("rejected_decision_count"),
        "cycle_count": len(cycles),
        "latest_cycle": cycles[-1].get("cycle") if cycles else None,
        "capital": payload.get("capital", {}),
        "profitability_gate": safety.get("profitability_gate", {}),
        "blockers": payload.get("blockers", []),
        "providers": provider_rows,
        "performance": payload.get("performance", {}),
        "next_commands": payload.get("next_commands", {}),
        "notes": payload.get("notes", []),
    }


def _live_auto_canary_payload(
    *,
    user_id: int,
    provider: str,
    research_budget_minutes: float,
    max_submissions: int = 1,
    submit_interval_seconds: float = 0.0,
    submit: bool,
    connection_id: int | None = None,
) -> dict[str, object]:
    provider_request = str(provider or "both").lower()
    providers = _live_auto_provider_sequence(provider_request)
    submission_limit = max(1, min(int(max_submissions or 1), 10))
    submit_interval = max(0.0, float(submit_interval_seconds or 0.0))
    budget_seconds = max(60.0, float(research_budget_minutes or 10.0) * 60.0)
    discovery_connection_id = connection_id if provider_request != "both" else None
    discovery = _find_live_canary_ranking_payload(
        user_id=user_id,
        provider="active" if provider_request == "both" else provider_request,
        connection_id=discovery_connection_id,
        symbols=["BTC", "ETH", "SOL", "HYPE"],
        timeframes=["5m", "15m", "1h"],
        profiles=["short_term", "aggressive_1h"],
        max_parameter_sets=12,
        allocation_amount_usd=10.0,
        lock_duration_hours=1,
        auto_deploy_top_n=submission_limit,
        strategy_names=None,
        research_depth="ml",
        adaptive_research=True,
        deadline_monotonic=time.monotonic() + budget_seconds,
        market_data_mode="live",
        max_rankings=submission_limit,
    )

    ranking_ids = _live_auto_discovered_ranking_ids(
        discovery,
        provider="active" if provider_request == "both" else provider_request,
        limit=submission_limit,
    )
    ranking_id = ranking_ids[0] if ranking_ids else discovery.get("accepted_ranking_id")
    provider_results: list[dict[str, object]] = []
    blockers: list[str] = []
    stopped = False
    submissions_processed = 0
    if not ranking_ids:
        blockers.extend(str(item) for item in discovery.get("blockers", []) or ["accepted_ranking_missing"])
    else:
        for provider_name in providers:
            for candidate_ranking_id in ranking_ids:
                if stopped or (submit and submissions_processed >= submission_limit):
                    provider_results.append(
                        {
                            "provider": provider_name,
                            "ranking_id": int(candidate_ranking_id),
                            "ready": False,
                            "submitted": False,
                            "skipped": True,
                            "blockers": ["first_canary_submission_already_processed"],
                        }
                    )
                    continue
                result = _live_auto_canary_provider_step(
                    provider=provider_name,
                    user_id=user_id,
                    ranking_id=int(candidate_ranking_id),
                    connection_id=_live_auto_connection_id_for_provider(
                        connection_id,
                        provider=provider_name,
                        user_id=user_id,
                    ),
                    submit=submit,
                )
                provider_results.append(result)
                blockers.extend(str(item) for item in result.get("blockers", []) or [])
                if submit and "submission" in result:
                    submissions_processed += 1
                    if (
                        submit_interval > 0
                        and submissions_processed < submission_limit
                        and bool(result.get("real_order_submitted", False))
                    ):
                        time.sleep(submit_interval)
                if provider_name == "hyperliquid" and provider_request == "both" and submit:
                    submitted = bool(result.get("submitted", False))
                    verified = bool(result.get("post_submit_verified", False))
                    if not submitted or not verified:
                        stopped = True
                        break
            if stopped:
                continue

    ready = bool(ranking_id is not None) and all(bool(item.get("ready", False)) for item in provider_results)
    submitted_count = sum(1 for item in provider_results if bool(item.get("submitted", False)))
    real_submitted_count = sum(1 for item in provider_results if bool(item.get("real_order_submitted", False)))
    payload = {
        "ready": ready,
        "mode": "live",
        "user_id": user_id,
        "provider_request": provider_request,
        "providers_requested": providers,
        "connection_id": connection_id,
        "research_budget_minutes": float(research_budget_minutes or 10.0),
        "max_submissions": submission_limit,
        "submit_interval_seconds": submit_interval,
        "submit_requested": bool(submit),
        "accepted_ranking_id": ranking_id,
        "accepted_ranking_ids": ranking_ids,
        "blockers": list(dict.fromkeys(blockers)),
        "discovery": discovery,
        "provider_results": provider_results,
        "submitted_count": submitted_count,
        "real_submitted_count": real_submitted_count,
        "real_order_submitted": real_submitted_count > 0,
        "notes": [
            "Accepted-ranking gate remains mandatory.",
            "ML only prioritizes research sweeps and cannot override rejected rankings.",
            "Providers and rankings are processed sequentially.",
            "Raise --max-submissions only after reviewing preview output and live micro-canary limits.",
        ],
    }
    _record_live_auto_canary_audit(payload)
    return payload


def _live_auto_provider_sequence(provider: str) -> list[str]:
    if provider == "both":
        return ["hyperliquid", "kucoin"]
    if provider in {"hyperliquid", "kucoin"}:
        return [provider]
    return ["hyperliquid", "kucoin"]


def _live_auto_connection_id_for_provider(connection_id: int | None, *, provider: str, user_id: int) -> int | None:
    if connection_id is None:
        return None
    connection = db.session.get(TradingConnection, int(connection_id))
    if connection is None:
        return None
    if int(connection.user_id or 0) != int(user_id):
        return None
    if normalize_exchange_provider(connection.provider) != normalize_exchange_provider(provider):
        return None
    return int(connection.id)


def _live_auto_discovered_ranking_ids(discovery: dict[str, object], *, provider: str, limit: int) -> list[int]:
    ranking_ids: list[int] = []

    def add_id(value: object) -> None:
        try:
            ranking_id = int(value)
        except (TypeError, ValueError):
            return
        if ranking_id > 0 and ranking_id not in ranking_ids:
            ranking_ids.append(ranking_id)

    for value in list(discovery.get("accepted_ranking_ids", []) or []):
        add_id(value)
    add_id(discovery.get("accepted_ranking_id"))

    run_ids: list[int] = []
    for sweep in list(discovery.get("sweeps", []) or []):
        if not isinstance(sweep, dict):
            continue
        try:
            run_id = int(sweep.get("optimizer_run_id") or 0)
        except (TypeError, ValueError):
            run_id = 0
        if run_id > 0 and run_id not in run_ids:
            run_ids.append(run_id)

    provider_key = normalize_exchange_provider(provider)
    if provider_key == "active":
        provider_key = normalize_exchange_provider(discovery.get("provider") or discovery.get("provider_request") or "global")
    for run_id in run_ids:
        if len(ranking_ids) >= limit:
            break
        query = StrategyRanking.query.filter_by(optimizer_run_id=run_id, rejected=False).filter(
            (StrategyRanking.rejection_reason.is_(None)) | (StrategyRanking.rejection_reason == "")
        )
        if provider_key in {"hyperliquid", "kucoin"}:
            query = query.filter(StrategyRanking.provider.in_([provider_key, "global"]))
        for ranking in query.order_by(StrategyRanking.score.desc(), StrategyRanking.id.desc()).limit(limit).all():
            add_id(ranking.id)
            if len(ranking_ids) >= limit:
                break

    accepted_ids: list[int] = []
    for ranking_id in ranking_ids:
        if len(accepted_ids) >= limit:
            break
        ranking = db.session.get(StrategyRanking, ranking_id)
        if ranking is None or bool(ranking.rejected) or str(ranking.rejection_reason or "").strip():
            continue
        accepted_ids.append(ranking_id)
    return accepted_ids


def _live_auto_canary_provider_step(
    *,
    provider: str,
    user_id: int,
    ranking_id: int,
    connection_id: int | None = None,
    submit: bool,
) -> dict[str, object]:
    readiness = _live_funds_readiness_payload(
        provider=provider,
        user_id=user_id,
        ranking_id=ranking_id,
        connection_id=connection_id,
    )
    provider_item = _preferred_live_funds_connection(list(readiness.get("providers", []) or []))
    connection_id = int(provider_item["connection_id"]) if provider_item else None
    blockers = [str(item) for item in readiness.get("blockers", []) or []]
    if provider_item is None:
        blockers.append("active_verified_live_connection_missing")
    else:
        blockers.extend(_live_auto_provider_snapshot_blockers(provider_item))

    preview: dict[str, object] = {"ready": False, "blockers": ["connection_required"]}
    if connection_id is not None:
        preview = _live_canary_trade_payload(
            ranking_id=ranking_id,
            user_id=user_id,
            connection_id=connection_id,
            submit=False,
            record_preview_audit=True,
        )
        blockers.extend(str(item) for item in preview.get("blockers", []) or [])
    ready = not list(dict.fromkeys(blockers)) and bool(preview.get("ready", False))
    result: dict[str, object] = {
        "provider": provider,
        "ranking_id": ranking_id,
        "connection_id": connection_id,
        "ready": ready,
        "blockers": list(dict.fromkeys(blockers)),
        "readiness": readiness,
        "preview": preview,
        "submitted": False,
        "post_submit_verified": False,
    }
    if not submit:
        return result
    if not ready:
        result["submit_blocked"] = True
        return result

    submission = _live_canary_trade_payload(
        ranking_id=ranking_id,
        user_id=user_id,
        connection_id=connection_id,
        submit=True,
        record_preview_audit=False,
    )
    result["submission"] = submission
    result["submitted"] = bool(submission.get("submitted", False))
    result["real_order_submitted"] = bool(submission.get("real_order_submitted", False))
    post = _live_funds_readiness_payload(
        provider=provider,
        user_id=user_id,
        ranking_id=ranking_id,
        connection_id=connection_id,
    )
    result["post_submit_readiness"] = post
    result["post_submit_verified"] = _live_auto_submission_verified(submission, post)
    if not bool(result["post_submit_verified"]):
        result["blockers"] = list(dict.fromkeys([*result["blockers"], "post_submit_verification_failed"]))
    return result


def _live_auto_provider_snapshot_blockers(provider_item: dict[str, object]) -> list[str]:
    snapshot = provider_item.get("snapshot") if isinstance(provider_item.get("snapshot"), dict) else {}
    blockers: list[str] = []
    if int(snapshot.get("open_orders_count") or 0) > 0:
        blockers.append("unexpected_open_orders")
    if int(snapshot.get("positions_count") or 0) > 0:
        blockers.append("unexpected_open_positions")
    return blockers


def _live_auto_submission_verified(submission: dict[str, object], post_readiness: dict[str, object]) -> bool:
    if not bool(submission.get("real_order_submitted", False)):
        return False
    order = submission.get("order") if isinstance(submission.get("order"), dict) else {}
    status = str(order.get("status") or "").lower()
    if status not in {"submitted", "open", "filled"}:
        return False
    providers = list(post_readiness.get("providers", []) or [])
    if not providers:
        return False
    health = providers[0].get("health") if isinstance(providers[0], dict) else {}
    return bool(health.get("can_trade", False)) and not list(health.get("alerts", []) or [])


def _record_live_auto_canary_audit(payload: dict[str, object]) -> int:
    audit = AuditLog(
        category="orders",
        action="live_auto_canary",
        message="Live auto canary orchestration completed.",
        user_id=int(payload.get("user_id") or 0) or None,
    )
    audit.details = {
        "provider_request": payload.get("provider_request"),
        "accepted_ranking_id": payload.get("accepted_ranking_id"),
        "accepted_ranking_ids": payload.get("accepted_ranking_ids", []),
        "max_submissions": payload.get("max_submissions"),
        "submit_requested": payload.get("submit_requested"),
        "submitted_count": payload.get("submitted_count"),
        "real_submitted_count": payload.get("real_submitted_count"),
        "blockers": payload.get("blockers", []),
        "providers": [
            {
                "provider": item.get("provider"),
                "ranking_id": item.get("ranking_id"),
                "connection_id": item.get("connection_id"),
                "ready": item.get("ready"),
                "submitted": item.get("submitted"),
                "post_submit_verified": item.get("post_submit_verified"),
                "blockers": item.get("blockers", []),
            }
            for item in list(payload.get("provider_results", []) or [])
            if isinstance(item, dict)
        ],
    }
    db.session.add(audit)
    db.session.commit()
    return int(audit.id)


def _live_funds_readiness_payload(
    *,
    provider: str,
    user_id: int | None,
    ranking_id: int | None,
    connection_id: int | None,
) -> dict[str, object]:
    provider = str(provider or "active").lower()
    production_provider = provider if provider in {"hyperliquid", "kucoin"} else "global"
    production = _production_readiness_payload_for(provider=production_provider, horizon="1h")
    dependencies = _dependency_status()
    ranking = _select_live_canary_ranking(ranking_id)
    ranking_requested = ranking_id is not None
    user = db.session.get(User, int(user_id)) if user_id is not None else None
    connections = _live_funds_connections(provider, user_id, connection_id)
    blockers: list[str] = []

    if not bool(production.get("ready", False)):
        blockers.append("strict_readiness_required")
    if user_id is None:
        blockers.append("user_id_required")
    elif user is None:
        blockers.append("user_not_found")
    if ranking_requested and ranking is None:
        blockers.append("ranking_required")
    if not connections:
        blockers.append("active_verified_live_connection_missing")

    provider_results = [_live_funds_provider_readiness(connection, ranking) for connection in connections]
    for item in provider_results:
        blockers.extend(str(blocker) for blocker in item.get("blockers", []))

    canary_preview: dict[str, object] = {
        "ready": False,
        "blockers": ["user_id_required" if user_id is None else "ranking_required" if ranking is None else "connection_required"],
    }
    selected_connection = _preferred_live_funds_connection(provider_results)
    if user is not None and ranking is not None and selected_connection is not None:
        canary_preview = _live_canary_trade_payload(
            ranking_id=ranking.id,
            user_id=user.id,
            connection_id=int(selected_connection["connection_id"]),
            submit=False,
            record_preview_audit=False,
        )
        blockers.extend(str(blocker) for blocker in canary_preview.get("blockers", []))

    commands = _live_funds_next_commands(user, ranking, selected_connection)
    provider_ready = bool(provider_results) and all(bool(item.get("ready", False)) for item in provider_results)
    ready = not list(dict.fromkeys(blockers)) and (
        bool(canary_preview.get("ready", False)) if ranking_requested else provider_ready
    )
    return {
        "ready": ready,
        "mode": "live",
        "provider_request": provider,
        "blockers": list(dict.fromkeys(blockers)),
        "dependencies": dependencies,
        "strict_production_ready": bool(production.get("ready", False)),
        "strict_production_blockers": production.get("blockers", []),
        "first_canary_guardrail": _first_canary_guardrail(),
        "live_micro_canary": canary_preview.get("live_micro_canary", _live_micro_canary_guardrail()),
        "selected_ranking": _ranking_canary_summary(ranking) if ranking is not None else None,
        "providers": provider_results,
        "canary_preview": canary_preview,
        "next_commands": commands,
        "notes": [
            "This command never submits a live order.",
            "Keep CANARY_PREVIEW_ONLY=true until preview, readiness, and caps are reviewed.",
            "Submit only through submit-live-canary after all live and micro-canary gates are manually enabled.",
        ],
    }


def _live_funds_connections(provider: str, user_id: int | None, connection_id: int | None) -> list[TradingConnection]:
    query = TradingConnection.query.filter_by(verification_status="verified")
    if connection_id is None:
        query = query.filter_by(is_active=True)
    if user_id is not None:
        query = query.filter_by(user_id=int(user_id))
    if connection_id is not None:
        query = query.filter_by(id=int(connection_id))
    if provider in {"hyperliquid", "kucoin"}:
        query = query.filter_by(provider=provider)
    records = query.order_by(TradingConnection.updated_at.desc(), TradingConnection.id.desc()).all()
    return sorted(records, key=lambda item: (0 if item.provider == "hyperliquid" else 1 if item.provider == "kucoin" else 2, -int(item.id or 0)))


def _live_funds_provider_readiness(connection: TradingConnection, ranking: StrategyRanking | None) -> dict[str, object]:
    service = get_service("trading_connections")
    blockers: list[str] = []
    provider = normalize_exchange_provider(connection.provider)
    collateral_asset = provider_collateral_asset(provider)
    if ranking is not None and normalize_exchange_provider(getattr(ranking, "provider", "global")) not in {"global", provider}:
        blockers.append("ranking_provider_mismatch")
    health = _fresh_connection_health(connection, service)
    if not bool(health.get("can_trade", False)):
        blockers.append("active_connection_cannot_trade")
    snapshot = service.account_snapshot(connection.user_id, "live", connection.id)
    alerts = [str(item) for item in (snapshot.alerts or []) if str(item).strip()]
    if alerts:
        blockers.append("provider_snapshot_alerts")
    balance_readiness = _provider_balance_readiness(connection.provider, snapshot.balances or [])
    blockers.extend(str(item) for item in balance_readiness.get("blockers", []))
    details = _provider_readiness_details(connection, ranking)
    blockers.extend(str(item) for item in details.get("blockers", []))
    balances = list(snapshot.balances or [])
    positions = list(snapshot.positions or [])
    open_orders = list(snapshot.open_orders or [])
    return {
        "provider": provider,
        "collateral_asset": collateral_asset,
        "connection_id": connection.id,
        "user_id": connection.user_id,
        "ready": not blockers,
        "blockers": list(dict.fromkeys(blockers)),
        "health": health,
        "snapshot": {
            "balances": balances,
            "positions_count": len(positions),
            "open_orders_count": len(open_orders),
            "recent_fills_count": len(snapshot.recent_fills or []),
            "alerts": alerts,
        },
        "details": details,
        "balance_readiness": balance_readiness,
    }


def _provider_balance_readiness(provider: str, balances: list[dict[str, object]]) -> dict[str, object]:
    provider_key = normalize_exchange_provider(provider)
    collateral_asset = provider_collateral_asset(provider_key)

    margin_balance = 0.0
    spot_balance = 0.0
    any_balance = 0.0
    for balance in balances:
        if str(balance.get("asset", "")).upper() != collateral_asset:
            continue
        amount = max(_float(balance.get("withdrawable")), _float(balance.get("value")))
        any_balance = max(any_balance, amount)
        balance_type = str(balance.get("type", "")).lower()
        if balance_type in {"margin", "futures", "future", "contract"}:
            margin_balance = max(margin_balance, amount)
        elif balance_type == "spot":
            spot_balance = max(spot_balance, amount)

    if margin_balance > 0:
        funding_source = (
            "hyperliquid_margin_usdc"
            if provider_key == "hyperliquid" and collateral_asset == "USDC"
            else f"{provider_key}_margin_{collateral_asset.lower()}"
        )
        return {
            "ready": True,
            "blockers": [],
            "collateral_asset": collateral_asset,
            "funding_source": funding_source,
            "margin_balance": margin_balance,
            "spot_balance": spot_balance,
            f"margin_{collateral_asset.lower()}": margin_balance,
            f"spot_{collateral_asset.lower()}": spot_balance,
        }
    if spot_balance > 0:
        funding_source = (
            "hyperliquid_spot_usdc_unified_available_to_trade"
            if provider_key == "hyperliquid" and collateral_asset == "USDC"
            else f"{provider_key}_spot_{collateral_asset.lower()}_available_to_trade"
        )
        return {
            "ready": True,
            "blockers": [],
            "collateral_asset": collateral_asset,
            "funding_source": funding_source,
            "margin_balance": margin_balance,
            "spot_balance": spot_balance,
            f"margin_{collateral_asset.lower()}": margin_balance,
            f"spot_{collateral_asset.lower()}": spot_balance,
            "notes": [f"{provider_key.title()} reports margin=0 but spot {collateral_asset} is visible and may be available to trade."],
        }
    if any_balance > 0:
        return {
            "ready": True,
            "blockers": [],
            "collateral_asset": collateral_asset,
            "funding_source": f"{provider_key}_snapshot_{collateral_asset.lower()}",
            "margin_balance": margin_balance,
            "spot_balance": spot_balance,
            f"margin_{collateral_asset.lower()}": margin_balance,
            f"spot_{collateral_asset.lower()}": spot_balance,
            "balance": any_balance,
        }
    return {
        "ready": False,
        "blockers": [f"{provider_key}_{collateral_asset.lower()}_missing"],
        "collateral_asset": collateral_asset,
        "funding_source": "",
        "margin_balance": margin_balance,
        "spot_balance": spot_balance,
        f"margin_{collateral_asset.lower()}": margin_balance,
        f"spot_{collateral_asset.lower()}": spot_balance,
    }


def _provider_readiness_details(connection: TradingConnection, ranking: StrategyRanking | None) -> dict[str, object]:
    if connection.provider == "hyperliquid":
        return _hyperliquid_readiness_details(connection)
    if connection.provider == "kucoin":
        return _kucoin_readiness_details(connection, ranking)
    return {"blockers": ["provider_not_in_first_funds_scope"]}


def _hyperliquid_readiness_details(connection: TradingConnection) -> dict[str, object]:
    blockers: list[str] = []
    service = get_service("trading_connections")
    try:
        credentials = service.credentials_for_execution(connection.user_id, connection.id)
        has_secret = bool(credentials.api_secret)
        has_account = bool(credentials.wallet_address or credentials.api_key)
    except Exception as exc:  # noqa: BLE001
        return {
            "account_address_configured": bool(connection.wallet_address),
            "api_wallet_secret_configured": False,
            "blockers": ["credentials_unavailable"],
            "error": str(exc),
        }
    if not has_secret:
        blockers.append("hyperliquid_api_wallet_secret_missing")
    if not has_account:
        blockers.append("hyperliquid_account_address_missing")
    return {
        "provider": "hyperliquid",
        "collateral_asset": provider_collateral_asset("hyperliquid"),
        "account_address_configured": has_account,
        "api_wallet_secret_configured": has_secret,
        "blockers": blockers,
    }


def _kucoin_readiness_details(connection: TradingConnection, ranking: StrategyRanking | None) -> dict[str, object]:
    blockers: list[str] = []
    service = get_service("trading_connections")
    try:
        credentials = service.credentials_for_execution(connection.user_id, connection.id)
        has_credentials = bool(credentials.api_key and credentials.api_secret and credentials.passphrase)
    except Exception as exc:  # noqa: BLE001
        return {"credentials_configured": False, "blockers": ["credentials_unavailable"], "error": str(exc)}
    if not has_credentials:
        blockers.append("kucoin_credentials_missing")
    symbol = str(getattr(ranking, "symbol", "") or "").upper()
    symbol_map = {
        str(key).upper(): str(value).upper()
        for key, value in _json_object_config("KUCOIN_SYMBOL_MAP_JSON", KUCOIN_SYMBOLS).items()
    }
    contract_specs = {
        str(key).upper(): value
        for key, value in _json_object_config("KUCOIN_CONTRACT_SPECS_JSON", KUCOIN_CONTRACT_SPECS).items()
    }
    venue_symbol = str(symbol_map.get(symbol) or "").upper()
    symbol_mapped = bool(venue_symbol)
    contract_spec_available = bool(venue_symbol and isinstance(contract_specs.get(venue_symbol), dict))
    if symbol and not symbol_mapped:
        blockers.append("kucoin_symbol_mapping_missing")
    if symbol and symbol_mapped and not contract_spec_available:
        blockers.append("kucoin_contract_spec_missing")
    return {
        "credentials_configured": has_credentials,
        "provider": "kucoin",
        "collateral_asset": provider_collateral_asset("kucoin"),
        "symbol": symbol,
        "venue_symbol": venue_symbol or None,
        "symbol_mapped": symbol_mapped,
        "contract_spec_available": contract_spec_available,
        "margin_mode": current_app.config.get("KUCOIN_MARGIN_MODE", "ISOLATED"),
        "position_side": current_app.config.get("KUCOIN_POSITION_SIDE", "BOTH"),
        "blockers": blockers,
    }


def _preferred_live_funds_connection(provider_results: list[dict[str, object]]) -> dict[str, object] | None:
    for provider_name in ("hyperliquid", "kucoin"):
        for item in provider_results:
            if item.get("provider") == provider_name and bool(item.get("ready", False)):
                return item
    return provider_results[0] if provider_results else None


def _select_live_canary_ranking(ranking_id: int | None) -> StrategyRanking | None:
    if ranking_id is not None:
        return db.session.get(StrategyRanking, int(ranking_id))
    return (
        StrategyRanking.query.filter_by(rejected=False)
        .filter((StrategyRanking.rejection_reason.is_(None)) | (StrategyRanking.rejection_reason == ""))
        .order_by(StrategyRanking.created_at.desc(), StrategyRanking.score.desc())
        .first()
    )


def _live_funds_next_commands(
    user: User | None,
    ranking: StrategyRanking | None,
    provider_connection: dict[str, object] | None,
) -> dict[str, object]:
    commands: dict[str, object] = {
        "install_dependencies": "python3 -m pip install -r requirements.txt",
        "run_tests": "python3 -m pytest -q",
        "strict_readiness": "flask production-readiness --strict",
        "funds_readiness": "flask live-funds-readiness --provider active --user-id <id> --ranking-id <id>",
    }
    if user is not None and ranking is not None and provider_connection is not None:
        connection_id = int(provider_connection["connection_id"])
        commands["canary_preview"] = (
            f"flask live-canary-trade --ranking-id {ranking.id} --user-id {user.id} --connection-id {connection_id}"
        )
        commands["enable_real_submit"] = "Set CANARY_PREVIEW_ONLY=false in local environment only after reviewing preview output."
        commands["canary_submit"] = (
            f"flask live-canary-trade --ranking-id {ranking.id} --user-id {user.id} --connection-id {connection_id} "
            "--submit --confirm LIVE-CANARY-TRADE"
        )
        commands["submit_live_canary"] = (
            f"flask submit-live-canary --ranking-id {ranking.id} --user-id {user.id} "
            "--confirm LIVE-CANARY-TRADE"
        )
        commands["submit_live_canary_warning"] = (
            "This submit command still fails unless APP_MODE=live, live confirmations, micro-canary submit, "
            "and preview-only gates are manually enabled."
        )
        commands["restore_preview_only"] = "Set CANARY_PREVIEW_ONLY=true immediately after the canary attempt."
    return commands


def _live_micro_canary_preview_has_submit_blockers(readiness: dict[str, object] | None) -> bool:
    if not isinstance(readiness, dict):
        return True
    preview = readiness.get("canary_preview") if isinstance(readiness.get("canary_preview"), dict) else readiness
    if not bool(preview.get("ready", False)):
        return True
    micro = preview.get("live_micro_canary") if isinstance(preview.get("live_micro_canary"), dict) else {}
    blockers = [
        *list(micro.get("blockers", []) or []),
        *list(micro.get("active_blockers", []) or []),
    ]
    non_sizing_submit_gates = {
        "live_micro_canary_preview_only_enabled",
        "live_micro_canary_live_submit_disabled",
        "live_micro_canary_required",
    }
    return any(str(item).strip() and str(item) not in non_sizing_submit_gates for item in blockers)


def _first_canary_guardrail() -> dict[str, object]:
    preferred = max(0.0, _float(current_app.config.get("FIRST_CANARY_ALLOCATION_BUDGET_USDT"), 1.0))
    fallback = max(preferred, _float(current_app.config.get("FIRST_CANARY_FALLBACK_ALLOCATION_BUDGET_USDT"), 2.0))
    max_leverage = max(0.0, _float(current_app.config.get("FIRST_CANARY_MAX_LEVERAGE"), 1.0))
    return {
        "preferred_allocation_budget_usdt": preferred or 1.0,
        "fallback_allocation_budget_usdt": min(max(fallback, preferred or 1.0), 2.0),
        "max_leverage": max_leverage or 1.0,
        "fallback_policy": "Only use fallback when provider minimum order sizing blocks the preferred allocation budget.",
        "fallback_enabled": _config_flag("FIRST_CANARY_USE_MIN_SIZE_FALLBACK", False),
        "canary_preview_only": _config_flag("CANARY_PREVIEW_ONLY", True),
    }


def _live_micro_canary_guardrail() -> dict[str, object]:
    enabled = _config_flag("LIVE_MICRO_CANARY_ENABLED", False)
    account_usd = max(0.0, _float(current_app.config.get("LIVE_MICRO_CANARY_ACCOUNT_USD"), 10.0))
    max_allocation = _clamp(
        _float(current_app.config.get("LIVE_MICRO_CANARY_MAX_ALLOCATION_USD"), 1.0),
        0.0,
        2.0,
    )
    max_risk_pct = _clamp(
        _float(current_app.config.get("LIVE_MICRO_CANARY_MAX_RISK_PCT"), 0.01),
        0.0,
        0.01,
    )
    max_leverage = _clamp(
        _float(current_app.config.get("LIVE_MICRO_CANARY_MAX_LEVERAGE"), 1.0),
        0.0,
        1.0,
    )
    preview_only = _config_flag("LIVE_MICRO_CANARY_PREVIEW_ONLY", True)
    min_notional_buffer = max(
        0.0,
        _float(current_app.config.get("LIVE_MICRO_CANARY_MIN_NOTIONAL_BUFFER_USD"), 0.50),
    )
    max_daily_live_orders = max(0, int(_float(current_app.config.get("LIVE_MICRO_CANARY_MAX_DAILY_LIVE_ORDERS"), 1)))
    order_budget_usd = max(
        0.0,
        _float(current_app.config.get("LIVE_MICRO_CANARY_ORDER_BUDGET_USD"), 10.0),
    )
    default_min_notional = max(
        0.0,
        _float(current_app.config.get("LIVE_MICRO_CANARY_DEFAULT_MIN_NOTIONAL_USD"), 10.0),
    )
    blockers: list[str] = []
    if enabled:
        if account_usd <= 0:
            blockers.append("live_micro_canary_account_usd_invalid")
        if max_allocation <= 0:
            blockers.append("live_micro_canary_max_allocation_invalid")
        if max_risk_pct <= 0:
            blockers.append("live_micro_canary_max_risk_pct_invalid")
        if max_leverage <= 0:
            blockers.append("live_micro_canary_max_leverage_invalid")
        if order_budget_usd <= 0:
            blockers.append("live_micro_canary_order_budget_invalid")
        if default_min_notional <= 0:
            blockers.append("live_micro_canary_default_min_notional_invalid")

    return {
        "enabled": enabled,
        "account_usd": account_usd,
        "max_allocation_usd": max_allocation,
        "max_risk_pct": max_risk_pct,
        "account_risk_budget_usd": account_usd * max_risk_pct,
        "max_leverage": max_leverage,
        "require_stop_loss": _config_flag("LIVE_MICRO_CANARY_REQUIRE_STOP_LOSS", True),
        "require_take_profit": _config_flag("LIVE_MICRO_CANARY_REQUIRE_TAKE_PROFIT", True),
        "preview_only": preview_only,
        "live_submit_enabled": _config_flag("LIVE_MICRO_CANARY_LIVE_SUBMIT_ENABLED", False),
        "min_notional_buffer_usd": min_notional_buffer,
        "max_daily_live_orders": max_daily_live_orders,
        "require_exact_confirmation": _config_flag("LIVE_MICRO_CANARY_REQUIRE_EXACT_CONFIRMATION", True),
        "allow_min_notional_order": _config_flag("LIVE_MICRO_CANARY_ALLOW_MIN_NOTIONAL_ORDER", False),
        "order_budget_usd": order_budget_usd,
        "default_min_notional_usd": default_min_notional,
        "blockers": blockers,
        "submission_blockers": ["live_micro_canary_preview_only_enabled"] if enabled and preview_only else [],
    }


def _live_micro_canary_confirmation_phrase() -> str:
    phrase = str(current_app.config.get("LIVE_MICRO_CANARY_EXACT_CONFIRMATION", "LIVE-CANARY-TRADE") or "").strip()
    return phrase or "LIVE-CANARY-TRADE"


def _live_micro_canary_available_balance_usd(balances: list[dict[str, object]]) -> float:
    available = 0.0
    for balance in balances:
        asset = str(balance.get("asset", "")).upper()
        if asset not in {"USD", "USDC", "USDT"}:
            continue
        candidates = [
            _float(balance.get("withdrawable")),
            _float(balance.get("available")),
            _float(balance.get("free")),
            _float(balance.get("value")),
            _float(balance.get("amount")),
        ]
        available = max(available, *[value for value in candidates if value >= 0])
    return available


def _live_micro_canary_exchange_min_notional(
    *,
    connection: TradingConnection | None,
    symbol: str,
    mid: float,
) -> dict[str, object]:
    default = max(0.0, _float(current_app.config.get("LIVE_MICRO_CANARY_DEFAULT_MIN_NOTIONAL_USD"), 10.0))
    provider = str(getattr(connection, "provider", "") or "").lower()
    source = "default_min_notional"
    value = default
    if provider == "hyperliquid":
        source = "hyperliquid_default_min_notional"
    elif provider == "kucoin":
        symbol_map = {
            str(key).upper(): str(value).upper()
            for key, value in _json_object_config("KUCOIN_SYMBOL_MAP_JSON", KUCOIN_SYMBOLS).items()
        }
        contract_specs = {
            str(key).upper(): value
            for key, value in _json_object_config("KUCOIN_CONTRACT_SPECS_JSON", KUCOIN_CONTRACT_SPECS).items()
        }
        mapped = str(symbol_map.get(str(symbol or "").upper()) or "").upper()
        spec = contract_specs.get(mapped) if mapped else None
        if isinstance(spec, dict):
            explicit = max(_float(spec.get("min_notional")), _float(spec.get("minNotional")))
            if explicit > 0:
                value = explicit
                source = "kucoin_contract_min_notional"
            else:
                contract_size = max(
                    _float(spec.get("contract_size")),
                    _float(spec.get("contractSize")),
                    _float(spec.get("multiplier")),
                )
                min_size = max(_float(spec.get("min_size")), _float(spec.get("minSize")), _float(spec.get("lotSize")))
                if contract_size > 0 and min_size > 0 and mid > 0:
                    value = contract_size * min_size * mid
                    source = "kucoin_contract_size_min_notional"
        else:
            source = "kucoin_default_min_notional"
    elif provider:
        source = f"{provider}_default_min_notional"
    return {"value": value, "source": source, "provider": provider or None}


def _resolve_live_micro_canary_order_size(
    *,
    ranking: StrategyRanking,
    parameters: dict[str, object],
    connection: TradingConnection | None,
    mid: float,
    stop_loss: float,
    take_profit: float,
    leverage: float,
    requested_notional: float,
    normal_cap: float,
    stop_distance_pct: float,
    slippage_pct: float,
    available_balance_usd: float | None,
    micro_guardrail: dict[str, object],
) -> dict[str, object]:
    enabled = bool(micro_guardrail.get("enabled", False))
    requested_allocation = _float(ranking.allocation_amount_usd)
    parameter_cap = _float(parameters.get("allocation_cap_usd"))
    configured_cap = _float(micro_guardrail.get("max_allocation_usd"), 1.0)
    exchange_min = _live_micro_canary_exchange_min_notional(
        connection=connection,
        symbol=ranking.symbol,
        mid=mid,
    )
    exchange_min_notional = _float(exchange_min.get("value"))
    balance = _float(available_balance_usd)
    order_budget = _float(micro_guardrail.get("order_budget_usd"), 10.0)
    account_usd = _float(micro_guardrail.get("account_usd"), 10.0)
    risk_pct = _float(micro_guardrail.get("max_risk_pct"), 0.01)
    risk_budget = account_usd * risk_pct
    buffer_usd = _float(micro_guardrail.get("min_notional_buffer_usd"), 0.50)
    estimated_fees_slippage = max(0.0, requested_notional) * max(0.0, slippage_pct)
    reserve = max(buffer_usd, estimated_fees_slippage)
    final_notional = max(0.0, requested_notional)
    min_required = enabled and exchange_min_notional > final_notional + 1e-9
    min_allowed = bool(micro_guardrail.get("allow_min_notional_order", False))
    min_used = False
    blockers: list[str] = []

    if not enabled:
        return {
            "enabled": False,
            "requested_ranking_allocation_usd": requested_allocation,
            "configured_micro_allocation_cap_usd": configured_cap,
            "parameter_allocation_cap_usd": parameter_cap,
            "normal_micro_cap_usd": normal_cap,
            "exchange_min_notional_usd": exchange_min_notional,
            "exchange_min_notional_source": exchange_min.get("source"),
            "available_balance_usd": balance,
            "reserve_usd": reserve,
            "fee_slippage_reserve_usd": estimated_fees_slippage,
            "final_allowed_notional_usd": final_notional,
            "estimated_stop_loss_loss_usd": final_notional * max(0.0, stop_distance_pct),
            "estimated_fees_slippage_usd": estimated_fees_slippage,
            "min_notional_order_required": False,
            "min_notional_order_allowed": False,
            "min_notional_order_used": False,
            "blockers": [],
        }

    if exchange_min_notional <= 0:
        blockers.append("exchange_min_notional_unknown")
    elif min_required:
        if not min_allowed:
            blockers.append(
                "exchange_min_notional_exceeds_micro_cap"
                if exchange_min_notional > max(0.0, normal_cap) + 1e-9
                else "exchange_min_notional_requires_upsizing"
            )
        else:
            final_notional = max(final_notional, exchange_min_notional)
            min_used = True

    if stop_loss <= 0 and bool(micro_guardrail.get("require_stop_loss", True)):
        blockers.append("stop_loss_required")
    if take_profit <= 0 and bool(micro_guardrail.get("require_take_profit", True)):
        blockers.append("take_profit_required")
    if leverage <= 0 or leverage > _float(micro_guardrail.get("max_leverage"), 1.0) + 1e-9:
        blockers.append("live_micro_canary_leverage_exceeds_one")
    if order_budget > 0 and final_notional > order_budget + 1e-9:
        blockers.append("live_micro_canary_order_budget_exceeded")
    estimated_loss = final_notional * max(0.0, stop_distance_pct)
    estimated_fees_slippage = final_notional * max(0.0, slippage_pct)
    reserve = max(buffer_usd, estimated_fees_slippage)
    if balance <= 0:
        blockers.append("live_micro_canary_balance_unavailable")
    elif final_notional + reserve > balance + 1e-9:
        blockers.append("live_micro_canary_insufficient_balance")
    if risk_budget > 0 and estimated_loss > risk_budget + 1e-9:
        blockers.append("live_micro_canary_stop_loss_risk_exceeded")
    if final_notional <= 0:
        blockers.append("invalid_size")

    return {
        "enabled": True,
        "requested_ranking_allocation_usd": requested_allocation,
        "configured_micro_allocation_cap_usd": configured_cap,
        "parameter_allocation_cap_usd": parameter_cap,
        "normal_micro_cap_usd": normal_cap,
        "exchange_min_notional_usd": exchange_min_notional,
        "exchange_min_notional_source": exchange_min.get("source"),
        "available_balance_usd": balance,
        "reserve_usd": reserve,
        "fee_slippage_reserve_usd": estimated_fees_slippage,
        "final_allowed_notional_usd": final_notional,
        "estimated_stop_loss_loss_usd": estimated_loss,
        "estimated_fees_slippage_usd": estimated_fees_slippage,
        "min_notional_order_required": bool(min_required),
        "min_notional_order_allowed": bool(min_allowed),
        "min_notional_order_used": bool(min_used),
        "blockers": list(dict.fromkeys(blockers)),
    }


def _live_micro_canary_daily_success_count(user_id: int | None) -> int:
    today_utc = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    query = AuditLog.query.filter(
        AuditLog.action == "live_micro_canary_submit_success",
        AuditLog.created_at >= today_utc,
    )
    if user_id is not None:
        query = query.filter(AuditLog.user_id == int(user_id))
    return int(query.count())


def _record_live_micro_canary_submit_audit(
    *,
    action: str,
    message: str,
    ranking: StrategyRanking | None,
    user: User | None,
    connection_id: int | None,
    payload: dict[str, object],
    order: Order | None = None,
    blockers: list[str] | None = None,
) -> int:
    audit = AuditLog(
        category="orders",
        action=action,
        message=message,
        user_id=user.id if user is not None else None,
        trading_connection_id=connection_id,
    )
    order_payload = payload.get("order") if isinstance(payload.get("order"), dict) else {}
    audit.details = {
        "ranking_id": ranking.id if ranking is not None else payload.get("ranking_id"),
        "strategy_name": getattr(ranking, "strategy_name", None) or payload.get("selected_strategy"),
        "symbol": getattr(ranking, "symbol", None) or payload.get("selected_symbol"),
        "timeframe": getattr(ranking, "timeframe", None),
        "submitted": payload.get("submitted", False),
        "real_order_submitted": payload.get("real_order_submitted", False),
        "order_id": getattr(order, "id", None) or order_payload.get("id"),
        "order_status": getattr(order, "status", None) or order_payload.get("status"),
        "blockers": list(dict.fromkeys(str(item) for item in (blockers or payload.get("blockers", []) or []))),
        "notional_usd": payload.get("notional_usd"),
        "estimated_loss_at_stop_usd": payload.get("estimated_loss_at_stop_usd"),
        "estimated_fees_slippage_usd": payload.get("estimated_fees_slippage_usd"),
        "exchange_min_notional_usd": payload.get("exchange_min_notional_usd"),
        "min_notional_order_used": payload.get("min_notional_order_used"),
        "live_micro_canary": payload.get("live_micro_canary", {}),
        "risk_decision": payload.get("risk_decision", {}),
    }
    db.session.add(audit)
    db.session.commit()
    return int(audit.id)


def _json_object_config(key: str, default: dict[str, object] | None = None) -> dict[str, object]:
    value = current_app.config.get(key)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return dict(default or {})
        return parsed if isinstance(parsed, dict) else dict(default or {})
    return dict(default or {})


def _find_canary_ranking_timeout_seconds(value: float | None = None) -> float:
    raw = value if value is not None else current_app.config.get("FIND_CANARY_RANKING_TIMEOUT_SECONDS", 120.0)
    try:
        timeout = float(raw or 120.0)
    except (TypeError, ValueError):
        return 120.0
    return max(0.001, timeout)


def _find_canary_max_symbols() -> int:
    try:
        return max(1, int(current_app.config.get("FIND_CANARY_MAX_SYMBOLS", 4) or 4))
    except (TypeError, ValueError):
        return 4


def _find_canary_max_rankings() -> int:
    try:
        return max(1, int(current_app.config.get("FIND_CANARY_MAX_RANKINGS", 10) or 10))
    except (TypeError, ValueError):
        return 10


def _find_canary_per_sweep_timeout_seconds() -> float:
    try:
        return max(0.001, float(current_app.config.get("FIND_CANARY_PER_SWEEP_TIMEOUT_SECONDS", 30.0) or 30.0))
    except (TypeError, ValueError):
        return 30.0


def _find_canary_market_data_probe_limit() -> int:
    try:
        return max(30, min(int(current_app.config.get("FIND_CANARY_MARKET_DATA_PROBE_LIMIT", 250) or 250), 5_000))
    except (TypeError, ValueError):
        return 250


def _find_canary_mock_data_requested() -> bool:
    return _config_flag("FIND_CANARY_RANKING_ALLOW_MOCK_DATA", False)


def _find_canary_mock_data_allowed() -> bool:
    return bool(current_app.config.get("TESTING", False)) and _find_canary_mock_data_requested()


def _find_canary_candidate_symbols(
    *,
    provider: str,
    requested_symbols: list[str],
    explicit_symbols: bool,
    max_symbols: int,
) -> dict[str, object]:
    fallback_symbols_used = False
    if explicit_symbols:
        candidates = [_find_canary_normalize_symbol(symbol) for symbol in requested_symbols]
        candidates = [symbol for symbol in candidates if symbol]
    else:
        candidates = _find_canary_provider_fallback_symbols(provider)
        fallback_symbols_used = True
        if not candidates:
            candidates = ["BTC", "ETH", "SOL", "HYPE"]
            fallback_symbols_used = False
        candidates = _find_canary_ml_order_symbols(candidates, provider)

    limited, omitted = _find_canary_limited_symbols(candidates, max_symbols)
    return {
        "symbols": limited,
        "omitted_symbols": omitted,
        "candidate_order": limited,
        "fallback_symbols_used": fallback_symbols_used,
    }


def _find_canary_provider_fallback_symbols(provider: str) -> list[str]:
    configured = _json_object_config("FIND_CANARY_RANKING_FALLBACK_SYMBOLS")
    provider_key = str(provider or "active").lower()
    raw_symbols = configured.get(provider_key) or configured.get("active") or ["BTC", "ETH"]
    if isinstance(raw_symbols, str):
        raw_items = [item.strip() for item in raw_symbols.split(",")]
    elif isinstance(raw_symbols, list):
        raw_items = [str(item).strip() for item in raw_symbols]
    else:
        raw_items = []
    symbols: list[str] = []
    for item in raw_items:
        symbol = _find_canary_normalize_symbol(item)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def _find_canary_normalize_symbol(symbol: object) -> str:
    value = str(symbol or "").strip().upper()
    if not value:
        return ""
    for separator in ("-", "/", "_"):
        if separator in value:
            value = value.split(separator, 1)[0]
            break
    for suffix in ("USDC", "USDT", "USD", "PERP"):
        if len(value) > len(suffix) + 1 and value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _find_canary_ml_order_symbols(symbols: list[str], provider: str) -> list[str]:
    scored: list[tuple[float, int, str]] = []
    for index, symbol in enumerate(symbols):
        scored.append((_find_canary_symbol_ml_score(symbol, provider), -index, symbol))
    scored.sort(reverse=True)
    return [symbol for _score, _index, symbol in scored]


def _find_canary_symbol_ml_score(symbol: str, provider: str) -> float:
    context = {
        "symbol": symbol,
        "provider": str(provider or "active").lower(),
        "timeframe": "15m",
        "strategy_name": "canary_symbol_screen",
        "optimizer_profile": "find_live_canary_ranking",
        "lock_duration_hours": 1,
    }
    try:
        online_score = float(get_service("online_ranker").predict_score(context, "1h") or 0.0)
    except Exception:  # noqa: BLE001
        online_score = 0.0
    try:
        offline = get_service("offline_ranker").score_payload(context, "1h", base_score=online_score, rejected=False)
        return float(offline.get("blended_score", online_score) or online_score)
    except Exception:  # noqa: BLE001
        return online_score


def _find_canary_limited_symbols(symbols: list[str], max_symbols: int) -> tuple[list[str], list[str]]:
    normalized = [str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()]
    if not normalized:
        normalized = ["BTC", "ETH", "SOL", "HYPE"]
    limit = max(1, int(max_symbols or 1))
    return normalized[:limit], normalized[limit:]


def _find_canary_progress_state(
    *,
    started_at: float,
    timeout_seconds: float,
    symbols: list[str],
    omitted_symbols: list[str],
    profiles: list[str],
    max_symbols: int,
    max_rankings: int,
) -> dict[str, object]:
    return {
        "started_at": float(started_at),
        "timeout_seconds": float(timeout_seconds),
        "current_phase": "loading_services",
        "failed_phase": "loading_services",
        "symbols": list(symbols),
        "omitted_symbols": list(omitted_symbols),
        "profiles": list(profiles),
        "max_symbols": int(max_symbols),
        "max_rankings": int(max_rankings),
        "progress_events": [],
        "sweeps": [],
        "rejection_breakdown": {},
    }


def _find_canary_progress_reporter() -> Callable[..., None]:
    def _report(phase: str, **details: object) -> None:
        parts = [FIND_CANARY_PROGRESS_PREFIX, str(phase)]
        for key in sorted(details):
            value = details[key]
            if value is None or value == "" or value == []:
                continue
            parts.append(f"{key}={_find_canary_progress_value(value)}")
        click.echo(" ".join(parts), err=True)

    return _report


def _find_canary_progress_value(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _find_canary_phase(
    progress_state: dict[str, object] | None,
    progress: Callable[..., None] | None,
    phase: str,
    **details: object,
) -> None:
    if progress_state is not None:
        progress_state["current_phase"] = str(phase)
        progress_state["failed_phase"] = str(phase)
        started_at = float(progress_state.get("started_at") or time.monotonic())
        event = {
            "phase": str(phase),
            "elapsed_seconds": round(max(0.0, time.monotonic() - started_at), 3),
        }
        for key, value in details.items():
            if value is None or value == "":
                continue
            event[key] = value
        events = progress_state.setdefault("progress_events", [])
        if isinstance(events, list) and len(events) < 50:
            events.append(event)
    if progress is not None:
        progress(str(phase), **details)


def _find_canary_timeout_payload(
    *,
    user_id: int | None,
    provider: str,
    symbols: list[str],
    omitted_symbols: list[str],
    timeframes: list[str],
    profiles: list[str],
    research_depth: str,
    market_data_mode: str,
    timeout_seconds: float,
    started_at: float,
    progress_state: dict[str, object] | None,
) -> dict[str, object]:
    elapsed = max(0.0, time.monotonic() - float(started_at))
    failed_phase = "unknown"
    sweeps: list[dict[str, object]] = []
    rejection_breakdown: dict[str, int] = {}
    progress_events: list[dict[str, object]] = []
    max_symbols = _find_canary_max_symbols()
    max_rankings = _find_canary_max_rankings()
    candidate_order: list[object] = list(symbols)
    fallback_symbols_used = False
    market_data_readiness: dict[str, object] = {}
    if progress_state is not None:
        failed_phase = str(progress_state.get("failed_phase") or progress_state.get("current_phase") or "unknown")
        sweeps = [dict(item) for item in list(progress_state.get("sweeps") or []) if isinstance(item, dict)]
        rejection_breakdown = {
            str(key): int(value or 0)
            for key, value in dict(progress_state.get("rejection_breakdown") or {}).items()
        }
        progress_events = [
            dict(item) for item in list(progress_state.get("progress_events") or []) if isinstance(item, dict)
        ]
        max_symbols = int(progress_state.get("max_symbols") or max_symbols)
        max_rankings = int(progress_state.get("max_rankings") or max_rankings)
        candidate_order = list(progress_state.get("candidate_order") or symbols)
        fallback_symbols_used = bool(progress_state.get("fallback_symbols_used", False))
        market_data_readiness = dict(progress_state.get("market_data_readiness") or {})
    return {
        "ready": False,
        "mode": "live",
        "user_id": user_id,
        "provider_request": provider,
        "symbols": symbols,
        "omitted_symbols": omitted_symbols,
        "max_symbols": max_symbols,
        "timeframes": timeframes,
        "profiles": profiles,
        "max_rankings": max_rankings,
        "fallback_symbols_used": fallback_symbols_used,
        "candidate_order": candidate_order,
        "research_depth": str(research_depth or "standard").lower(),
        "market_data_mode": str(market_data_mode or "live"),
        "accepted_ranking_id": None,
        "selected_ranking": None,
        "timed_out": True,
        "failed_phase": failed_phase,
        "elapsed_seconds": round(elapsed, 3),
        "timeout_seconds": float(timeout_seconds),
        "blockers": ["find_canary_ranking_timeout"],
        "research_budget_exhausted": True,
        "research_budget_remaining_seconds": 0.0,
        "market_data_readiness": market_data_readiness,
        "no_accepted_ranking_reason": "timeout",
        "operator_next_steps": _find_canary_operator_next_steps("research_budget_exhausted"),
        **_find_canary_empty_fallback_result(),
        "sweeps": sweeps,
        "accepted_count": 0,
        "rejection_breakdown": rejection_breakdown,
        "research_diagnostics": _live_canary_research_diagnostics(sweeps, rejection_breakdown),
        "progress_events": progress_events,
        "next_commands": {
            "strict_readiness": "flask production-readiness --strict",
            "rerun_narrow": "flask find-live-canary-ranking --research-depth quick --symbol BTC",
        },
        "notes": [
            "This command never submits a live order.",
            "The ranking search timed out before an accepted canary candidate was selected.",
        ],
    }


def _find_canary_error_payload(
    *,
    user_id: int | None,
    provider: str,
    symbols: list[str],
    omitted_symbols: list[str],
    timeframes: list[str],
    profiles: list[str],
    max_symbols: int,
    max_rankings: int,
    research_depth: str,
    market_data_mode: str,
    failed_phase: str,
    error: str,
    blockers: list[str],
    progress_state: dict[str, object] | None,
) -> dict[str, object]:
    sweeps = [dict(item) for item in list((progress_state or {}).get("sweeps") or []) if isinstance(item, dict)]
    rejection_breakdown = {
        str(key): int(value or 0)
        for key, value in dict((progress_state or {}).get("rejection_breakdown") or {}).items()
    }
    progress_events = [
        dict(item) for item in list((progress_state or {}).get("progress_events") or []) if isinstance(item, dict)
    ]
    candidate_order = list((progress_state or {}).get("candidate_order") or symbols)
    fallback_symbols_used = bool((progress_state or {}).get("fallback_symbols_used", False))
    market_data_readiness = dict((progress_state or {}).get("market_data_readiness") or {})
    return {
        "ready": False,
        "mode": "live",
        "user_id": user_id,
        "provider_request": provider,
        "symbols": symbols,
        "omitted_symbols": omitted_symbols,
        "max_symbols": max_symbols,
        "timeframes": timeframes,
        "profiles": profiles,
        "max_rankings": max_rankings,
        "fallback_symbols_used": fallback_symbols_used,
        "candidate_order": candidate_order,
        "research_depth": str(research_depth or "standard").lower(),
        "market_data_mode": str(market_data_mode or "live"),
        "accepted_ranking_id": None,
        "selected_ranking": None,
        "timed_out": False,
        "failed_phase": failed_phase,
        "error": error,
        "blockers": list(dict.fromkeys(blockers)),
        "research_budget_exhausted": False,
        "research_budget_remaining_seconds": None,
        "market_data_readiness": market_data_readiness,
        "no_accepted_ranking_reason": failed_phase,
        "operator_next_steps": _find_canary_operator_next_steps(failed_phase),
        **_find_canary_empty_fallback_result(),
        "sweeps": sweeps,
        "accepted_count": 0,
        "rejection_breakdown": rejection_breakdown,
        "research_diagnostics": _live_canary_research_diagnostics(sweeps, rejection_breakdown),
        "progress_events": progress_events,
        "next_commands": {
            "strict_readiness": "flask production-readiness --strict",
            "funds_readiness": f"flask live-funds-readiness --provider {provider} --user-id {user_id}",
        },
        "notes": [
            "This command never submits a live order.",
            "The ranking search stopped before optimizer work because a readiness phase failed.",
        ],
    }


def _find_canary_market_data_preflight(
    *,
    symbols: list[str],
    timeframes: list[str],
    provider: str,
    deadline_monotonic: float | None,
    progress: Callable[..., None] | None,
    progress_state: dict[str, object] | None,
    market_data_mode: str,
) -> dict[str, object]:
    market_data = get_service("market_data")
    probe_limit = _find_canary_market_data_probe_limit()
    mock_requested = _find_canary_mock_data_requested()
    mock_allowed = _find_canary_mock_data_allowed()
    results: list[dict[str, object]] = []
    usable_candidates: list[dict[str, object]] = []
    blockers: list[str] = []

    for symbol in symbols:
        for timeframe in timeframes:
            if _live_canary_budget_insufficient(deadline_monotonic, min_seconds=1.0):
                blockers.append("market_data_probe_budget_exhausted")
                break
            _find_canary_phase(
                progress_state,
                progress,
                "fetching_candles",
                probe=True,
                symbol=symbol,
                timeframe=timeframe,
            )
            started = time.monotonic()
            data_source = "live"
            mock_data_used = False
            sanitized_error = ""
            candles: list[dict[str, object]] = []
            try:
                candles = market_data.get_candles(symbol, timeframe, mode=market_data_mode, limit=probe_limit)
            except Exception as exc:  # noqa: BLE001
                sanitized_error = _find_canary_sanitized_error(exc)
                if mock_allowed:
                    candles = _find_canary_mock_candles(timeframe, probe_limit)
                    data_source = "mock"
                    mock_data_used = True
            if len(candles) < 30 and mock_allowed:
                candles = _find_canary_mock_candles(timeframe, probe_limit)
                data_source = "mock"
                mock_data_used = True

            ready = len(candles) >= 30
            status = "ok" if ready else "insufficient_candles" if not sanitized_error else "error"
            row = {
                "symbol": symbol,
                "timeframe": timeframe,
                "ready": ready,
                "status": status,
                "candle_count": len(candles),
                "data_source": data_source,
                "mock_data_used": mock_data_used,
                "elapsed_seconds": round(max(0.0, time.monotonic() - started), 3),
            }
            if sanitized_error:
                row["error"] = sanitized_error
            results.append(row)
            if ready:
                usable_candidates.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "market_data_status": status,
                        "data_source": data_source,
                        "mock_data_used": mock_data_used,
                        "candle_count": len(candles),
                    }
                )
        if "market_data_probe_budget_exhausted" in blockers:
            break

    skipped_symbols = [
        symbol
        for symbol in symbols
        if not any(item.get("symbol") == symbol and item.get("ready") for item in results)
    ]
    if not usable_candidates:
        blockers.append("market_data_unavailable_for_all_symbols")
    if mock_requested and not mock_allowed:
        blockers.append("mock_data_disabled_in_live")

    return {
        "ready": bool(usable_candidates),
        "provider": str(provider or "active").lower(),
        "mode": str(market_data_mode or "live"),
        "probe_limit": probe_limit,
        "mock_data_requested": mock_requested,
        "mock_data_allowed": mock_allowed,
        "mock_data_disabled_in_live": mock_requested and not mock_allowed,
        "results": results,
        "usable_candidates": usable_candidates,
        "skipped_symbols": skipped_symbols,
        "blockers": list(dict.fromkeys(blockers)),
        "summary": {
            "candidate_count": len(usable_candidates),
            "checked_pairs": len(results),
            "skipped_symbol_count": len(skipped_symbols),
            "mock_data_used": any(bool(item.get("mock_data_used")) for item in results),
        },
    }


def _find_canary_market_data_unavailable_payload(
    *,
    user_id: int,
    provider: str,
    symbols: list[str],
    omitted_symbols: list[str],
    timeframes: list[str],
    profiles: list[str],
    max_symbols: int,
    max_rankings: int,
    research_depth: str,
    market_data_mode: str,
    fallback_symbols_used: bool,
    candidate_order: list[object],
    market_data_readiness: dict[str, object],
    progress_state: dict[str, object],
    timeout_value: float,
    payload_started_at: float,
) -> dict[str, object]:
    progress_events = [
        dict(item) for item in list(progress_state.get("progress_events") or []) if isinstance(item, dict)
    ]
    blockers = list(dict.fromkeys(str(item) for item in market_data_readiness.get("blockers", []) or []))
    return {
        "ready": False,
        "mode": "live",
        "user_id": user_id,
        "provider_request": provider,
        "symbols": symbols,
        "omitted_symbols": omitted_symbols,
        "max_symbols": max_symbols,
        "timeframes": timeframes,
        "profiles": profiles,
        "max_rankings": max_rankings,
        "research_depth": str(research_depth or "standard").lower(),
        "market_data_mode": str(market_data_mode or "live"),
        "accepted_ranking_id": None,
        "selected_ranking": None,
        "timed_out": False,
        "failed_phase": "fetching_candles",
        "elapsed_seconds": round(max(0.0, time.monotonic() - payload_started_at), 3),
        "timeout_seconds": timeout_value,
        "blockers": blockers or ["market_data_unavailable_for_all_symbols"],
        "research_budget_exhausted": False,
        "research_budget_remaining_seconds": None,
        "fallback_symbols_used": fallback_symbols_used,
        "candidate_order": candidate_order,
        "market_data_readiness": market_data_readiness,
        "no_accepted_ranking_reason": "market_data_unavailable_for_all_symbols",
        "operator_next_steps": _find_canary_operator_next_steps("market_data_unavailable_for_all_symbols"),
        **_find_canary_empty_fallback_result(),
        "sweeps": [],
        "accepted_count": 0,
        "rejection_breakdown": {},
        "research_diagnostics": _live_canary_research_diagnostics([], {}),
        "progress_events": progress_events,
        "next_commands": {
            "strict_readiness": "flask production-readiness --strict",
            "rerun_narrow": "flask find-live-canary-ranking --research-depth quick --symbol BTC",
        },
        "notes": [
            "This command never submits a live order.",
            "No optimizer sweep ran because usable live candle data was unavailable.",
        ],
    }


def _find_canary_operator_next_steps(reason: str) -> list[str]:
    if reason == "market_data_unavailable_for_all_symbols":
        return [
            "Check provider/network access for live candle data.",
            "Retry with one explicit liquid symbol, for example: flask find-live-canary-ranking --research-depth quick --symbol BTC.",
            "Review FIND_CANARY_RANKING_FALLBACK_SYMBOLS if the provider uses different supported symbols.",
        ]
    if reason == "all_candidates_rejected":
        return [
            "Review rejection_breakdown and top_rejected diagnostics before changing any gates.",
            "Retry later or use a narrower symbol/timeframe after market conditions change.",
        ]
    if reason == "optimizer_sweeps_incomplete":
        return [
            "Rerun with fewer symbols/timeframes or a lower --max-parameter-sets value.",
            "Only increase FIND_CANARY_PER_SWEEP_TIMEOUT_SECONDS after reviewing optimizer_runtime phase timings.",
            "Do not lower ranking gates to compensate for incomplete optimizer coverage.",
        ]
    if reason == "research_budget_exhausted":
        return [
            "Rerun with --research-depth quick or a single --symbol.",
            "Only increase timeout after confirming live market-data latency is acceptable.",
        ]
    return ["Review sweeps and market_data_readiness diagnostics before attempting a canary preview."]


def _find_canary_no_accepted_reason(
    sweeps: list[dict[str, object]],
    rejection_breakdown: dict[str, int],
    research_budget_exhausted: bool,
) -> str:
    if research_budget_exhausted:
        return "research_budget_exhausted"
    if any(sweep.get("error") for sweep in sweeps):
        return "optimizer_sweep_failed"
    if any(_find_canary_sweep_incomplete(sweep) for sweep in sweeps):
        return "optimizer_sweeps_incomplete"
    if sum(int(sweep.get("ranking_count") or 0) for sweep in sweeps) > 0 and rejection_breakdown:
        return "all_candidates_rejected"
    if sweeps:
        return "no_rankings_created"
    return "no_optimizer_sweeps_completed"


def _find_canary_sweep_incomplete(sweep: dict[str, object]) -> bool:
    return bool(
        sweep.get("timed_out")
        or sweep.get("partial_result")
        or str(sweep.get("partial_reason") or "").strip()
    )


def _find_canary_empty_fallback_result() -> dict[str, object]:
    return {
        "fallback_attempted": False,
        "fallback_used": False,
        "fallback_source": "none",
        "fallback_ranking_id": None,
        "fallback_blockers": [],
        "fallback_preview": None,
        "fallback_notes": [],
    }


def _find_canary_fallback_config() -> dict[str, object]:
    symbol = _find_canary_normalize_symbol(current_app.config.get("FIND_CANARY_FALLBACK_SYMBOL", "BTC"))
    if not symbol:
        symbol = "BTC"
    strategy = str(current_app.config.get("FIND_CANARY_FALLBACK_STRATEGY", "ema_crossover") or "ema_crossover").strip()
    timeframe = str(current_app.config.get("FIND_CANARY_FALLBACK_TIMEFRAME", "15m") or "15m").strip()
    recency_hours = max(0.0, _float(current_app.config.get("FIND_CANARY_FALLBACK_RECENCY_HOURS"), 24.0))
    allocation = _clamp(
        _float(current_app.config.get("FIND_CANARY_FALLBACK_ALLOCATION_USD"), 1.0),
        0.0,
        1.0,
    )
    stop_loss_pct = _clamp(
        _float(current_app.config.get("FIND_CANARY_FALLBACK_STOP_LOSS_PCT"), 0.005),
        0.001,
        0.02,
    )
    take_profit_pct = _clamp(
        _float(current_app.config.get("FIND_CANARY_FALLBACK_TAKE_PROFIT_PCT"), 0.012),
        stop_loss_pct,
        0.05,
    )
    return {
        "enabled": _config_flag("FIND_CANARY_FALLBACK_ENABLED", False),
        "symbol": symbol,
        "strategy_name": strategy or "ema_crossover",
        "timeframe": timeframe or "15m",
        "recency_hours": recency_hours,
        "allocation_amount_usd": allocation or 1.0,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
    }


def _find_canary_apply_fallback(
    *,
    user_id: int,
    provider: str,
    provider_connection: dict[str, object] | None,
    no_accepted_ranking_reason: str | None,
    progress: Callable[..., None] | None,
    progress_state: dict[str, object] | None,
) -> dict[str, object]:
    result = _find_canary_empty_fallback_result()
    config = _find_canary_fallback_config()
    config["provider"] = normalize_exchange_provider(provider)
    if no_accepted_ranking_reason != "optimizer_sweeps_incomplete":
        result["fallback_notes"] = ["Fallback only runs after optimizer_sweeps_incomplete."]
        return result
    if not bool(config.get("enabled", False)):
        result["fallback_notes"] = ["FIND_CANARY_FALLBACK_ENABLED=false; fallback ranking was not attempted."]
        return result

    result["fallback_attempted"] = True
    blockers: list[str] = []
    if not _config_flag("ENABLE_LIVE_TRADING", False):
        blockers.append("live_trading_disabled")
    if str(current_app.config.get("APP_MODE", "paper")).lower() != "live":
        blockers.append("live_mode_required")
    if not provider_connection or not bool(provider_connection.get("ready", False)):
        blockers.append("active_verified_live_connection_missing")
    connection_id = int(provider_connection["connection_id"]) if provider_connection and provider_connection.get("connection_id") else None
    if connection_id is None:
        blockers.append("fallback_connection_missing")
    if Setting.get_json("panic_lock", False):
        blockers.append("panic_lock_active")
    if blockers:
        result["fallback_blockers"] = list(dict.fromkeys(blockers))
        result["fallback_notes"] = ["Fallback blocked before ranking selection."]
        return result

    _find_canary_phase(
        progress_state,
        progress,
        "selecting_fallback_ranking",
        symbol=config["symbol"],
        strategy=config["strategy_name"],
        timeframe=config["timeframe"],
    )
    ranking = _find_canary_cached_fallback_ranking(config)
    source = "cached_ranking" if ranking is not None else "synthetic_ranking"
    synthetic_created = False
    if ranking is None:
        ranking = _find_canary_create_synthetic_fallback_ranking(config)
        synthetic_created = True

    preview = _live_canary_trade_payload(
        ranking_id=ranking.id,
        user_id=user_id,
        connection_id=connection_id,
        submit=False,
        record_preview_audit=False,
    )
    preview_blockers = [str(item) for item in preview.get("blockers", []) or []]
    blockers.extend(preview_blockers)
    if not bool(preview.get("ready", False)) and not preview_blockers:
        blockers.append("fallback_preview_not_ready")
    if blockers and synthetic_created:
        ranking.rejected = True
        ranking.rejection_reason = _find_canary_fallback_rejection_reason(blockers)
        db.session.commit()

    result.update(
        {
            "fallback_used": not blockers,
            "fallback_source": source if not blockers else "none",
            "fallback_ranking_id": ranking.id if not blockers else None,
            "fallback_blockers": list(dict.fromkeys(blockers)),
            "fallback_preview": preview,
            "fallback_notes": [
                "Fallback candidate passed live canary preview checks."
                if not blockers
                else "Fallback candidate failed live canary preview checks and remains blocked."
            ],
        }
    )
    result["fallback_audit_id"] = _record_find_canary_fallback_audit(
        user_id=user_id,
        connection_id=connection_id,
        ranking=ranking,
        fallback_result=result,
    )
    return result


def _find_canary_cached_fallback_ranking(config: dict[str, object]) -> StrategyRanking | None:
    provider = normalize_exchange_provider(config.get("provider"))
    query = StrategyRanking.query.filter_by(rejected=False, experimental=False).filter(
        (StrategyRanking.rejection_reason.is_(None)) | (StrategyRanking.rejection_reason == "")
    )
    if provider != "global":
        query = query.filter(StrategyRanking.provider.in_([provider, "global"]))
    query = query.filter(StrategyRanking.profile != "extreme_roi_experimental")
    recency_hours = _float(config.get("recency_hours"), 24.0)
    if recency_hours > 0:
        query = query.filter(StrategyRanking.created_at >= datetime.utcnow() - timedelta(hours=recency_hours))
    query = query.filter_by(
        symbol=str(config.get("symbol") or "BTC").upper(),
        strategy_name=str(config.get("strategy_name") or "ema_crossover"),
        timeframe=str(config.get("timeframe") or "15m"),
    )
    rankings = query.order_by(StrategyRanking.score.desc(), StrategyRanking.created_at.desc()).limit(10).all()
    for ranking in rankings:
        if _find_canary_ranking_has_stop_loss(ranking):
            return ranking
    return None


def _find_canary_ranking_has_stop_loss(ranking: StrategyRanking) -> bool:
    parameters = ranking.parameters if isinstance(ranking.parameters, dict) else {}
    return _float(parameters.get("stop_loss_pct"), _float(parameters.get("fallback_stop_loss_pct"))) > 0


def _find_canary_create_synthetic_fallback_ranking(config: dict[str, object]) -> StrategyRanking:
    provider = normalize_exchange_provider(config.get("provider"))
    symbol = str(config.get("symbol") or "BTC").upper()
    strategy_name = str(config.get("strategy_name") or "ema_crossover")
    timeframe = str(config.get("timeframe") or "15m")
    allocation = _float(config.get("allocation_amount_usd"), 1.0)
    stop_loss_pct = _float(config.get("stop_loss_pct"), 0.005)
    take_profit_pct = _float(config.get("take_profit_pct"), 0.012)
    parameters = _find_canary_strategy_defaults(strategy_name)
    parameters.update(
        {
            "risk_fraction": min(_float(current_app.config.get("RISK_PER_TRADE_PCT"), 0.01), 0.01),
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": max(take_profit_pct, stop_loss_pct),
            "allocation_cap_usd": allocation,
            "synthetic_live_canary_fallback": True,
            "find_canary_fallback": True,
        }
    )
    optimizer_run = OptimizerRun(profile="short_term", status="fallback")
    optimizer_run.symbols = [symbol]
    optimizer_run.timeframes = [timeframe]
    optimizer_run.config_payload = {
        "source": "find_live_canary_synthetic_fallback",
        "strategy_name": strategy_name,
        "symbol": symbol,
        "timeframe": timeframe,
        "allocation_amount_usd": allocation,
    }
    optimizer_run.result = {
        "synthetic_fallback": True,
        "accepted_count": 1,
        "ranking_count": 1,
        "reason": "optimizer_sweeps_incomplete",
    }
    optimizer_run.completed_at = datetime.utcnow()
    db.session.add(optimizer_run)
    db.session.flush()
    ranking = StrategyRanking(
        optimizer_run_id=optimizer_run.id,
        provider=provider,
        strategy_name=strategy_name,
        symbol=symbol,
        timeframe=timeframe,
        profile="short_term",
        score=0.001,
        total_return=0.0,
        net_return_after_costs=0.0,
        recent_performance_score=0.0,
        recent_1h_return=0.0,
        edge_score=0.0,
        expectancy=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        win_loss_ratio=0.0,
        cost_drag_bps=0.0,
        convex_edge_score=0.0,
        mfe_mae_ratio=1.0,
        max_drawdown=-stop_loss_pct,
        profit_factor=1.0,
        sharpe_like=0.0,
        sortino_like=0.0,
        trades_per_day=1.0,
        avg_trade_return=0.0,
        turnover_rate=0.0,
        turnover_after_fees=0.0,
        consistency=0.0,
        window_stability=0.0,
        accepted_window_ratio=0.0,
        win_rate=0.0,
        trade_count=1,
        allocation_amount_usd=allocation,
        lock_duration_hours=1,
        leverage=1.0,
        capacity_usd=allocation,
        universe_source="find_canary_fallback",
        rejected=False,
        rejection_reason="",
    )
    ranking.parameters = parameters
    ranking.ml_explanation = {
        "net_roi_v2": {
            "net_roi_v2_score": 0.0,
            "roi_quality_grade": "D",
            "roi_rejection_risk": "high",
            "regime_support": "fallback_requires_preview",
        },
        "find_canary_fallback": {
            "synthetic": True,
            "live_submittable_only_after_preview": True,
        },
    }
    ranking.warnings = [
        "Synthetic fallback created only because optimizer sweeps were incomplete.",
        "Use only after live canary preview reports ready=true.",
    ]
    db.session.add(ranking)
    db.session.commit()
    return ranking


def _find_canary_strategy_defaults(strategy_name: str) -> dict[str, object]:
    try:
        return dict(get_service("strategy_registry").build(strategy_name, {}).parameters)
    except Exception:  # noqa: BLE001
        return {}


def _find_canary_fallback_rejection_reason(blockers: list[str]) -> str:
    first = next((str(item) for item in blockers if str(item).strip()), "fallback_preview_not_ready")
    return f"fallback_preview_blocked:{first}"[:250]


def _record_find_canary_fallback_audit(
    *,
    user_id: int,
    connection_id: int | None,
    ranking: StrategyRanking,
    fallback_result: dict[str, object],
) -> int | None:
    try:
        audit = AuditLog(
            category="optimizer",
            action="find_live_canary_fallback_selected" if fallback_result.get("fallback_used") else "find_live_canary_fallback_rejected",
            message="find-live-canary-ranking evaluated a fallback ranking after incomplete optimizer sweeps.",
            user_id=user_id,
            trading_connection_id=connection_id,
        )
        audit.details = {
            "fallback_used": bool(fallback_result.get("fallback_used", False)),
            "fallback_source": fallback_result.get("fallback_source"),
            "fallback_ranking_id": ranking.id,
            "fallback_blockers": list(fallback_result.get("fallback_blockers", []) or []),
            "strategy_name": ranking.strategy_name,
            "symbol": ranking.symbol,
            "timeframe": ranking.timeframe,
            "synthetic": bool((ranking.parameters or {}).get("synthetic_live_canary_fallback", False)),
        }
        db.session.add(audit)
        db.session.commit()
        return int(audit.id)
    except Exception:  # noqa: BLE001
        db.session.rollback()
        return None


def _find_canary_sanitized_error(exc: Exception) -> str:
    value = str(exc)
    for token in ("api_key", "secret", "passphrase", "private_key"):
        value = value.replace(token, "[redacted]")
        value = value.replace(token.upper(), "[redacted]")
    return value[:300]


def _find_canary_mock_candles(timeframe: str, limit: int) -> list[dict[str, object]]:
    step_ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}.get(str(timeframe), 900_000)
    count = max(30, min(int(limit or 250), 500))
    start = int(time.time() * 1000) - (count * step_ms)
    rows: list[dict[str, object]] = []
    price = 100.0
    for index in range(count):
        price += 0.08 if index % 3 else -0.03
        rows.append(
            {
                "timestamp": start + index * step_ms,
                "open": price - 0.05,
                "high": price + 0.15,
                "low": price - 0.15,
                "close": price,
                "volume": 1000.0,
                "interval": timeframe,
            }
        )
    return rows


def _record_find_canary_timeout_audit(payload: dict[str, object]) -> int | None:
    if not has_app_context():
        return None
    user_id = payload.get("user_id")
    if user_id is None:
        return None
    try:
        audit = AuditLog(
            category="orders",
            action="find_live_canary_ranking_timeout",
            message="find-live-canary-ranking timed out before selecting a canary ranking.",
            user_id=int(user_id),
        )
        audit.details = {
            "phase": payload.get("failed_phase"),
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "timeout_seconds": payload.get("timeout_seconds"),
            "symbols": payload.get("symbols", []),
            "omitted_symbols": payload.get("omitted_symbols", []),
            "profiles": payload.get("profiles", []),
            "sweeps": payload.get("sweeps", []),
            "rejection_breakdown": payload.get("rejection_breakdown", {}),
        }
        db.session.add(audit)
        db.session.commit()
        return int(audit.id)
    except Exception:  # noqa: BLE001
        db.session.rollback()
        return None


def _find_live_canary_ranking_payload(
    *,
    user_id: int,
    provider: str,
    symbols: list[str],
    timeframes: list[str],
    profiles: list[str],
    max_parameter_sets: int,
    allocation_amount_usd: float,
    lock_duration_hours: int,
    auto_deploy_top_n: int,
    strategy_names: list[str] | None,
    omitted_symbols: list[str] | None = None,
    research_depth: str = "standard",
    adaptive_research: bool = True,
    deadline_monotonic: float | None = None,
    timeout_seconds: float | None = None,
    timeout_blocker: str = "research_budget_exhausted",
    explicit_symbols: bool = False,
    fallback_symbols_used: bool = False,
    candidate_order: list[object] | None = None,
    max_symbols: int | None = None,
    max_rankings: int | None = None,
    progress: Callable[..., None] | None = None,
    progress_state: dict[str, object] | None = None,
    market_data_mode: str = "live",
    connection_id: int | None = None,
) -> dict[str, object]:
    payload_started_at = time.monotonic()
    symbol_limit = max(1, int(max_symbols or _find_canary_max_symbols()))
    ranking_limit = max(1, int(max_rankings or _find_canary_max_rankings()))
    symbols, extra_omitted_symbols = _find_canary_limited_symbols(symbols, symbol_limit)
    omitted_symbols = list(omitted_symbols or []) + list(extra_omitted_symbols)
    candidate_order = list(candidate_order or symbols)
    timeout_value = float(timeout_seconds or _find_canary_ranking_timeout_seconds(None))
    if progress_state is None:
        progress_state = _find_canary_progress_state(
            started_at=payload_started_at,
            timeout_seconds=timeout_value,
            symbols=symbols,
            omitted_symbols=omitted_symbols,
            profiles=profiles,
            max_symbols=symbol_limit,
            max_rankings=ranking_limit,
        )
    else:
        progress_state["symbols"] = list(symbols)
        progress_state["omitted_symbols"] = list(omitted_symbols)
        progress_state["profiles"] = list(profiles)
        progress_state["max_symbols"] = symbol_limit
        progress_state["max_rankings"] = ranking_limit
    progress_state["candidate_order"] = list(candidate_order)
    progress_state["fallback_symbols_used"] = bool(fallback_symbols_used)
    progress_state["explicit_symbols"] = bool(explicit_symbols)
    _find_canary_phase(
        progress_state,
        progress,
        "loading_services",
        symbols=symbols,
        omitted_symbols=omitted_symbols,
        profiles=profiles,
        timeout_seconds=timeout_value,
        fallback_symbols_used=fallback_symbols_used,
    )
    search_started_at = datetime.utcnow()
    strict_provider = _live_canary_strict_provider(provider=provider, user_id=user_id, connection_id=connection_id)
    _find_canary_phase(progress_state, progress, "checking_strict_readiness", provider=strict_provider)
    try:
        _production_readiness_payload_for(provider=strict_provider, horizon="1h")
    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        return _find_canary_error_payload(
            user_id=user_id,
            provider=provider,
            symbols=symbols,
            omitted_symbols=omitted_symbols,
            timeframes=timeframes,
            profiles=profiles,
            max_symbols=symbol_limit,
            max_rankings=ranking_limit,
            research_depth=research_depth,
            market_data_mode=market_data_mode,
            failed_phase="checking_strict_readiness",
            error=str(exc),
            blockers=["strict_readiness_check_failed"],
            progress_state=progress_state,
        )
    _find_canary_phase(progress_state, progress, "checking_provider_readiness")
    try:
        readiness = _live_funds_readiness_payload(
            provider=provider,
            user_id=user_id,
            ranking_id=None,
            connection_id=connection_id,
        )
    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        return _find_canary_error_payload(
            user_id=user_id,
            provider=provider,
            symbols=symbols,
            omitted_symbols=omitted_symbols,
            timeframes=timeframes,
            profiles=profiles,
            max_symbols=symbol_limit,
            max_rankings=ranking_limit,
            research_depth=research_depth,
            market_data_mode=market_data_mode,
            failed_phase="checking_provider_readiness",
            error=str(exc),
            blockers=["provider_readiness_failed"],
            progress_state=progress_state,
        )
    provider_connection = _preferred_live_funds_connection(list(readiness.get("providers", []) or []))
    connection_id = int(provider_connection["connection_id"]) if provider_connection else None
    execution_provider = normalize_exchange_provider(
        provider_connection.get("provider") if provider_connection else provider
    )
    sweep_results: list[dict[str, object]] = []
    aggregate_rejections: dict[str, int] = {}
    progress_state["sweeps"] = sweep_results
    progress_state["rejection_breakdown"] = aggregate_rejections
    selected_ranking: StrategyRanking | None = None
    selected_readiness: dict[str, object] | None = None
    selected_mock_data_used = False
    research_budget_exhausted = False
    _find_canary_phase(
        progress_state,
        progress,
        "loading_market_universe",
        symbols=symbols,
        timeframes=timeframes,
        max_symbols=symbol_limit,
        max_rankings=ranking_limit,
    )
    market_data_readiness = _find_canary_market_data_preflight(
        symbols=symbols,
        timeframes=timeframes,
        provider=execution_provider,
        deadline_monotonic=deadline_monotonic,
        progress=progress,
        progress_state=progress_state,
        market_data_mode=market_data_mode,
    )
    progress_state["market_data_readiness"] = market_data_readiness
    usable_candidates = [
        dict(item)
        for item in list(market_data_readiness.get("usable_candidates") or [])
        if isinstance(item, dict)
    ]
    if not usable_candidates:
        _find_canary_phase(
            progress_state,
            progress,
            "no_eligible_ranking_found",
            reason="market_data_unavailable_for_all_symbols",
        )
        return _find_canary_market_data_unavailable_payload(
            user_id=user_id,
            provider=provider,
            symbols=symbols,
            omitted_symbols=omitted_symbols,
            timeframes=timeframes,
            profiles=profiles,
            max_symbols=symbol_limit,
            max_rankings=ranking_limit,
            research_depth=research_depth,
            market_data_mode=market_data_mode,
            fallback_symbols_used=fallback_symbols_used,
            candidate_order=candidate_order,
            market_data_readiness=market_data_readiness,
            progress_state=progress_state,
            timeout_value=timeout_value,
            payload_started_at=payload_started_at,
        )

    for profile in profiles:
        if _live_canary_budget_insufficient(deadline_monotonic):
            research_budget_exhausted = True
            break
        for candidate in usable_candidates:
            if _live_canary_budget_insufficient(deadline_monotonic):
                research_budget_exhausted = True
                break
            candidate_symbol = str(candidate.get("symbol") or "").upper()
            candidate_timeframe = str(candidate.get("timeframe") or "")
            profile_results = _run_live_canary_profile_sweeps(
                stage="requested",
                profile=profile,
                symbols=[candidate_symbol],
                timeframes=[candidate_timeframe],
                max_parameter_sets=max_parameter_sets,
                allocation_amount_usd=allocation_amount_usd,
                lock_duration_hours=lock_duration_hours,
                auto_deploy_top_n=auto_deploy_top_n,
                strategy_names=strategy_names,
                provider=execution_provider,
                deadline_monotonic=deadline_monotonic,
                progress=progress,
                progress_state=progress_state,
                market_data_mode=market_data_mode,
            )
            for item in profile_results:
                item["data_source"] = candidate.get("data_source", "live")
                item["mock_data_used"] = bool(candidate.get("mock_data_used", False))
                item["market_data_status"] = candidate.get("market_data_status", "")
                item["candle_count"] = candidate.get("candle_count")
                sweep_results.append(item)
                for reason, count in dict(item.get("rejection_breakdown") or {}).items():
                    aggregate_rejections[str(reason)] = aggregate_rejections.get(str(reason), 0) + int(count or 0)

                ranking = _accepted_ranking_for_sweep(item, profile, search_started_at, provider=execution_provider)
                if ranking is None:
                    continue

                _find_canary_phase(
                    progress_state,
                    progress,
                    "applying_live_canary_filters",
                    ranking_id=ranking.id,
                    profile=profile,
                )
                candidate_readiness = _live_funds_readiness_payload(
                    provider=provider,
                    user_id=user_id,
                    ranking_id=ranking.id,
                    connection_id=connection_id,
                )
                candidate_mock_data_used = bool(candidate.get("mock_data_used", False))
                canary_blockers = list(candidate_readiness.get("blockers", []) or [])
                if candidate_mock_data_used and "mock_data_used_not_live_submittable" not in canary_blockers:
                    canary_blockers.append("mock_data_used_not_live_submittable")
                item["accepted_ranking_id"] = ranking.id
                item["canary_ready"] = bool(candidate_readiness.get("ready", False)) and not candidate_mock_data_used
                item["canary_blockers"] = canary_blockers
                if bool(candidate_readiness.get("ready", False)) or candidate_mock_data_used:
                    selected_ranking = ranking
                    selected_readiness = dict(candidate_readiness)
                    if candidate_mock_data_used:
                        selected_readiness["ready"] = False
                        selected_readiness["blockers"] = canary_blockers
                    selected_mock_data_used = candidate_mock_data_used
                    _find_canary_phase(
                        progress_state,
                        progress,
                        "accepted_ranking_found",
                        ranking_id=ranking.id,
                        symbol=ranking.symbol,
                        profile=ranking.profile,
                        mock_data_used=candidate_mock_data_used,
                    )
                    break
            if selected_ranking is not None:
                break
        if selected_ranking is not None or research_budget_exhausted:
            break

    adaptive_stages: list[dict[str, object]] = []
    depth = str(research_depth or "standard").lower()
    if selected_ranking is None and adaptive_research and depth in {"standard", "deep", "ml"}:
        adaptive_stages = _live_canary_adaptive_research_stages(
            symbols=symbols,
            timeframes=timeframes,
            max_parameter_sets=max_parameter_sets,
            rejection_breakdown=aggregate_rejections,
            depth=depth,
        )
        if depth == "ml":
            adaptive_stages = _live_canary_ml_prioritize_stages(adaptive_stages, provider=execution_provider)
        for stage in adaptive_stages:
            if _live_canary_budget_insufficient(deadline_monotonic):
                research_budget_exhausted = True
                break
            profile = str(stage["profile"])
            stage_symbols = {str(symbol).upper() for symbol in list(stage["symbols"])}
            stage_timeframes = {str(timeframe) for timeframe in list(stage["timeframes"])}
            stage_candidates = [
                candidate
                for candidate in usable_candidates
                if str(candidate.get("symbol") or "").upper() in stage_symbols
                and str(candidate.get("timeframe") or "") in stage_timeframes
            ]
            for candidate in stage_candidates:
                if _live_canary_budget_insufficient(deadline_monotonic):
                    research_budget_exhausted = True
                    break
                candidate_symbol = str(candidate.get("symbol") or "").upper()
                candidate_timeframe = str(candidate.get("timeframe") or "")
                profile_results = _run_live_canary_profile_sweeps(
                    stage=str(stage["stage"]),
                    profile=profile,
                    symbols=[candidate_symbol],
                    timeframes=[candidate_timeframe],
                    max_parameter_sets=int(stage["max_parameter_sets"]),
                    allocation_amount_usd=allocation_amount_usd,
                    lock_duration_hours=int(stage.get("lock_duration_hours", lock_duration_hours)),
                    auto_deploy_top_n=auto_deploy_top_n,
                    strategy_names=list(stage["strategy_names"]) if stage.get("strategy_names") else None,
                    research_overlay=str(stage.get("research_overlay") or ""),
                    provider=execution_provider,
                    deadline_monotonic=deadline_monotonic,
                    progress=progress,
                    progress_state=progress_state,
                    market_data_mode=market_data_mode,
                )
                for item in profile_results:
                    item["data_source"] = candidate.get("data_source", "live")
                    item["mock_data_used"] = bool(candidate.get("mock_data_used", False))
                    item["market_data_status"] = candidate.get("market_data_status", "")
                    item["candle_count"] = candidate.get("candle_count")
                    sweep_results.append(item)
                    for reason, count in dict(item.get("rejection_breakdown") or {}).items():
                        aggregate_rejections[str(reason)] = aggregate_rejections.get(str(reason), 0) + int(count or 0)

                    ranking = _accepted_ranking_for_sweep(item, profile, search_started_at, provider=execution_provider)
                    if ranking is None:
                        continue

                    _find_canary_phase(
                        progress_state,
                        progress,
                        "applying_live_canary_filters",
                        ranking_id=ranking.id,
                        profile=profile,
                    )
                    candidate_readiness = _live_funds_readiness_payload(
                        provider=provider,
                        user_id=user_id,
                        ranking_id=ranking.id,
                        connection_id=connection_id,
                    )
                    candidate_mock_data_used = bool(candidate.get("mock_data_used", False))
                    canary_blockers = list(candidate_readiness.get("blockers", []) or [])
                    if candidate_mock_data_used and "mock_data_used_not_live_submittable" not in canary_blockers:
                        canary_blockers.append("mock_data_used_not_live_submittable")
                    item["accepted_ranking_id"] = ranking.id
                    item["canary_ready"] = bool(candidate_readiness.get("ready", False)) and not candidate_mock_data_used
                    item["canary_blockers"] = canary_blockers
                    if bool(candidate_readiness.get("ready", False)) or candidate_mock_data_used:
                        selected_ranking = ranking
                        selected_readiness = dict(candidate_readiness)
                        if candidate_mock_data_used:
                            selected_readiness["ready"] = False
                            selected_readiness["blockers"] = canary_blockers
                        selected_mock_data_used = candidate_mock_data_used
                        _find_canary_phase(
                            progress_state,
                            progress,
                            "accepted_ranking_found",
                            ranking_id=ranking.id,
                            symbol=ranking.symbol,
                            profile=ranking.profile,
                            mock_data_used=candidate_mock_data_used,
                        )
                        break
                if selected_ranking is not None:
                    break
            if selected_ranking is not None or research_budget_exhausted:
                break

    if selected_ranking is None and research_budget_exhausted and timeout_blocker == "find_canary_ranking_timeout":
        failed_phase = str(progress_state.get("failed_phase") or progress_state.get("current_phase") or "unknown")
        _find_canary_phase(progress_state, progress, "timeout", failed_phase=failed_phase)
        progress_state["failed_phase"] = failed_phase
        return _find_canary_timeout_payload(
            user_id=user_id,
            provider=provider,
            symbols=symbols,
            omitted_symbols=omitted_symbols,
            timeframes=timeframes,
            profiles=profiles,
            research_depth=research_depth,
            market_data_mode=market_data_mode,
            timeout_seconds=timeout_value,
            started_at=float(progress_state.get("started_at") or payload_started_at),
            progress_state=progress_state,
        )

    if selected_ranking is None:
        _find_canary_phase(
            progress_state,
            progress,
            "no_eligible_ranking_found",
            sweep_count=len(sweep_results),
            rejection_reasons=len(aggregate_rejections),
        )

    no_accepted_ranking_reason = (
        None if selected_ranking is not None else _find_canary_no_accepted_reason(sweep_results, aggregate_rejections, research_budget_exhausted)
    )
    fallback_result = _find_canary_empty_fallback_result()
    if selected_ranking is None:
        fallback_result = _find_canary_apply_fallback(
            user_id=user_id,
            provider=execution_provider,
            provider_connection=provider_connection,
            no_accepted_ranking_reason=no_accepted_ranking_reason,
            progress=progress,
            progress_state=progress_state,
        )
        if bool(fallback_result.get("fallback_used", False)) and fallback_result.get("fallback_ranking_id"):
            selected_ranking = db.session.get(StrategyRanking, int(fallback_result["fallback_ranking_id"]))
            selected_readiness = (
                dict(fallback_result.get("fallback_preview") or {})
                if isinstance(fallback_result.get("fallback_preview"), dict)
                else None
            )

    selected_payload = _ranking_canary_summary(selected_ranking) if selected_ranking is not None else None
    next_commands: dict[str, object] = {
        "strict_readiness": "flask production-readiness --strict",
        "funds_readiness": f"flask live-funds-readiness --provider {execution_provider} --user-id {user_id}",
    }
    if selected_ranking is not None and connection_id is not None and selected_readiness is not None:
        if bool(fallback_result.get("fallback_used", False)):
            next_commands = _live_funds_next_commands(db.session.get(User, int(user_id)), selected_ranking, provider_connection)
        else:
            next_commands = dict(selected_readiness.get("next_commands") or next_commands)
        if _live_micro_canary_preview_has_submit_blockers(selected_readiness):
            next_commands.pop("submit_live_canary", None)
            next_commands.pop("submit_live_canary_warning", None)

    blockers = [] if selected_ranking is not None else ["accepted_ranking_missing"]
    if selected_mock_data_used:
        blockers.append("mock_data_used_not_live_submittable")
    if research_budget_exhausted and selected_ranking is None:
        blockers.append("research_budget_exhausted")
    if selected_ranking is None and no_accepted_ranking_reason == "all_candidates_rejected":
        blockers.append("all_candidates_rejected")
    if selected_ranking is None and no_accepted_ranking_reason == "optimizer_sweeps_incomplete":
        blockers.append("optimizer_sweeps_incomplete")
    if (
        selected_ranking is None
        and bool(fallback_result.get("fallback_attempted", False))
        and not bool(fallback_result.get("fallback_used", False))
    ):
        blockers.append("fallback_unavailable")

    return {
        "ready": selected_ranking is not None
        and bool(selected_readiness and selected_readiness.get("ready", False))
        and not selected_mock_data_used,
        "mode": "live",
        "user_id": user_id,
        "provider_request": provider,
        "provider": execution_provider,
        "connection_id": connection_id,
        "collateral_asset": provider_collateral_asset(execution_provider),
        "symbols": symbols,
        "omitted_symbols": omitted_symbols,
        "max_symbols": symbol_limit,
        "timeframes": timeframes,
        "profiles": profiles,
        "max_rankings": ranking_limit,
        "fallback_symbols_used": bool(fallback_symbols_used),
        "candidate_order": candidate_order,
        "research_depth": depth,
        "market_data_mode": str(market_data_mode or "live"),
        "adaptive_research": bool(adaptive_research and depth in {"standard", "deep", "ml"}),
        "ml_research": _live_canary_ml_research_status(execution_provider) if depth == "ml" else None,
        "adaptive_stages": adaptive_stages,
        "accepted_ranking_id": selected_ranking.id if selected_ranking else None,
        "selected_ranking": selected_payload,
        "timed_out": False,
        "failed_phase": None,
        "elapsed_seconds": round(max(0.0, time.monotonic() - payload_started_at), 3),
        "timeout_seconds": timeout_value,
        "blockers": blockers,
        "research_budget_exhausted": research_budget_exhausted,
        "research_budget_remaining_seconds": _live_canary_budget_remaining(deadline_monotonic),
        "provider_readiness": readiness,
        "selected_readiness": selected_readiness,
        "market_data_readiness": market_data_readiness,
        "mock_data_used_for_selected_ranking": selected_mock_data_used,
        "no_accepted_ranking_reason": no_accepted_ranking_reason,
        "operator_next_steps": _find_canary_operator_next_steps(str(no_accepted_ranking_reason or "accepted_ranking_found")),
        **fallback_result,
        "sweeps": sweep_results,
        "accepted_count": 1 if selected_ranking is not None else 0,
        "rejection_breakdown": aggregate_rejections,
        "research_diagnostics": _live_canary_research_diagnostics(sweep_results, aggregate_rejections),
        "progress_events": list(progress_state.get("progress_events") or []),
        "next_commands": next_commands,
        "notes": [
            "This command never submits a live order.",
            "Rejected rankings are diagnostics only and remain blocked from live canary submission.",
            "Run live-canary-trade without --submit first when an accepted ranking is found.",
        ],
    }


def _live_canary_adaptive_research_stages(
    *,
    symbols: list[str],
    timeframes: list[str],
    max_parameter_sets: int,
    rejection_breakdown: dict[str, int],
    depth: str,
) -> list[dict[str, object]]:
    """Build follow-up optimizer sweeps that improve coverage without relaxing gates."""
    stable_timeframes = _ordered_subset(timeframes, ["15m", "1h", "5m"]) or list(timeframes)
    intraday_timeframes = _ordered_subset(timeframes, ["5m", "15m", "1h"]) or list(timeframes)
    parameter_sets = max(int(max_parameter_sets or 1), 12)
    stages: list[dict[str, object]] = []

    dominant_reason = max(rejection_breakdown, key=rejection_breakdown.get, default="")
    if dominant_reason in {"profit_factor_below_one", "negative_recent_1h_return", ""}:
        for symbol in symbols:
            stages.append(
                {
                    "stage": f"profit_factor_stability_focus_{symbol.lower()}",
                    "profile": "short_term",
                    "symbols": [symbol],
                    "timeframes": stable_timeframes,
                    "strategy_names": ["rsi_mean_reversion", "mean_reversion", "ema_crossover", "breakout"],
                    "max_parameter_sets": parameter_sets,
                    "research_overlay": "profit_factor_below_one",
                }
            )
    if dominant_reason in {"negative_recent_1h_return", "profit_factor_below_one", "low_trade_count", ""}:
        intraday_symbols = symbols if depth in {"deep", "ml"} else symbols[:2]
        for symbol in intraday_symbols:
            stages.append(
                {
                    "stage": f"intraday_breakout_focus_{symbol.lower()}",
                    "profile": "aggressive_1h",
                    "symbols": [symbol],
                    "timeframes": intraday_timeframes,
                    "strategy_names": ["scalping", "volatility_breakout", "breakout", "rsi_mean_reversion"],
                    "max_parameter_sets": parameter_sets,
                    "research_overlay": "negative_recent_1h_return",
                }
            )

    if depth in {"deep", "ml"}:
        per_symbol_timeframes = _ordered_subset(timeframes, ["15m", "5m", "1h"]) or list(timeframes)
        for symbol in symbols:
            stages.append(
                {
                    "stage": f"per_symbol_{symbol.lower()}_coverage",
                    "profile": "short_term",
                    "symbols": [symbol],
                    "timeframes": per_symbol_timeframes,
                    "strategy_names": None,
                    "max_parameter_sets": max(parameter_sets, 16),
                    "research_overlay": "low_trade_count",
                }
            )
    return stages


def _live_canary_ml_prioritize_stages(stages: list[dict[str, object]], *, provider: str = "global") -> list[dict[str, object]]:
    scored = []
    for index, stage in enumerate(stages):
        score_payload = _live_canary_stage_ml_score(stage, provider=provider)
        item = dict(stage)
        item["ml_priority_score"] = score_payload["score"]
        item["ml_priority_source"] = score_payload["source"]
        item["ml_offline_status"] = score_payload["offline_status"]
        scored.append((float(score_payload["score"]), -index, item))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [item for _score, _index, item in scored]


def _live_canary_stage_ml_score(stage: dict[str, object], *, provider: str = "global") -> dict[str, object]:
    symbol = str((stage.get("symbols") or [""])[0] if isinstance(stage.get("symbols"), list) else "")
    timeframes = stage.get("timeframes") if isinstance(stage.get("timeframes"), list) else []
    strategies = stage.get("strategy_names") if isinstance(stage.get("strategy_names"), list) else []
    provider_key = normalize_exchange_provider(provider)
    context = {
        "provider": provider_key,
        "execution_venue": provider_key,
        "collateral_asset": provider_collateral_asset(provider_key),
        "symbol": symbol,
        "timeframe": str(timeframes[0]) if timeframes else "",
        "strategy_name": str(strategies[0]) if strategies else "",
        "optimizer_profile": stage.get("profile"),
        "rejection_reason": stage.get("research_overlay"),
        "lock_duration_hours": 1,
        "allocation_amount_usd": 10.0,
    }
    online_score = 0.0
    try:
        online_score = float(get_service("online_ranker").predict_score(context, "1h") or 0.0)
    except Exception:  # noqa: BLE001
        online_score = 0.0
    offline_status = "unavailable"
    offline_prediction = 0.0
    source = "online_ranker"
    try:
        offline = get_service("offline_ranker").score_payload(context, "1h", base_score=online_score, rejected=False)
        offline_status = str(offline.get("status") or "unknown")
        offline_prediction = _float(offline.get("prediction"))
        if offline_status == "promoted":
            source = "offline_ranker"
    except Exception:  # noqa: BLE001
        offline_status = "offline_ranker_error"
    return {
        "score": online_score + offline_prediction,
        "source": source,
        "offline_status": offline_status,
    }


def _live_canary_ml_research_status(provider: str = "global") -> dict[str, object]:
    provider_key = normalize_exchange_provider(provider)
    try:
        readiness = get_service("offline_ranker").readiness("1h", require_blend=False, provider=provider_key)
    except Exception as exc:  # noqa: BLE001
        readiness = {"ready": False, "provider": provider_key, "blockers": [str(exc)], "promoted_model": None}
    return {
        "offline_ranker": readiness,
        "online_ranker_used_for_priority": True,
        "hard_rejection_override_allowed": False,
        "notes": [
            "ML only prioritizes research sweeps.",
            "ML never clears rejected=true or a non-empty rejection_reason.",
        ],
    }


def _live_canary_strict_provider(*, provider: str, user_id: int, connection_id: int | None) -> str:
    if connection_id is not None:
        connection = db.session.get(TradingConnection, int(connection_id))
        if connection is not None:
            return normalize_exchange_provider(connection.provider)
    provider_key = normalize_exchange_provider(provider)
    if provider_key != "active":
        return provider_key
    connections = _live_funds_connections("active", user_id, None)
    if connections:
        return normalize_exchange_provider(connections[0].provider)
    return "global"


def _live_canary_budget_remaining(deadline_monotonic: float | None) -> float | None:
    if deadline_monotonic is None:
        return None
    return max(0.0, float(deadline_monotonic) - time.monotonic())


def _live_canary_budget_exhausted(deadline_monotonic: float | None) -> bool:
    remaining = _live_canary_budget_remaining(deadline_monotonic)
    return remaining is not None and remaining <= 0.0


def _live_canary_budget_insufficient(deadline_monotonic: float | None, min_seconds: float = 5.0) -> bool:
    remaining = _live_canary_budget_remaining(deadline_monotonic)
    return remaining is not None and remaining <= max(0.0, float(min_seconds or 0.0))


def _ordered_subset(values: list[str], preferred: list[str]) -> list[str]:
    available = {str(value) for value in values}
    return [value for value in preferred if value in available]


def _compact_live_canary_rejected_rows(rows: list[object]) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("rejected"):
            continue
        compact.append(
            {
                "symbol": row.get("symbol"),
                "timeframe": row.get("timeframe"),
                "strategy_name": row.get("strategy_name"),
                "rejection_reason": row.get("rejection_reason") or row.get("no_trade_reason") or "rejected",
                "score": row.get("score"),
                "net_return_after_costs": row.get("net_return_after_costs"),
                "profit_factor": row.get("profit_factor"),
                "recent_1h_return": row.get("recent_1h_return"),
                "trade_count": row.get("trade_count"),
            }
        )
        if len(compact) >= 5:
            break
    return compact


def _live_canary_research_diagnostics(
    sweeps: list[dict[str, object]],
    rejection_breakdown: dict[str, int],
) -> dict[str, object]:
    timed_out = [sweep for sweep in sweeps if sweep.get("timed_out") or sweep.get("partial_result")]
    coverage = {
        "sweep_count": len(sweeps),
        "timed_out_sweeps": len(timed_out),
        "ranking_count": sum(int(sweep.get("ranking_count") or 0) for sweep in sweeps),
        "requested_stages": sorted({str(sweep.get("stage") or "requested") for sweep in sweeps}),
    }
    recommendations: list[str] = []
    if rejection_breakdown.get("profit_factor_below_one", 0) > 0:
        recommendations.append(
            "profit_factor_below_one dominated; keep the gate intact and prefer stability-focused reward/risk overlays or wait for cleaner tape."
        )
    if rejection_breakdown.get("negative_recent_1h_return", 0) > 0:
        recommendations.append(
            "negative_recent_1h_return appeared; avoid submitting stale edge and rerun discovery after the next market regime shift."
        )
    if rejection_breakdown.get("low_trade_count", 0) > 0:
        recommendations.append(
            "low_trade_count appeared; increase research depth or wait for more candles rather than lowering the trade-count gate."
        )
    if timed_out:
        recommendations.append(
            "At least one sweep was partial; use --research-depth deep or narrow --symbol/--timeframe to verify uncovered strategies."
        )
    return {
        "coverage": coverage,
        "top_rejection_reasons": dict(sorted(rejection_breakdown.items(), key=lambda item: item[1], reverse=True)[:5]),
        "recommendations": recommendations,
    }


def _run_live_canary_profile_sweeps(
    *,
    stage: str,
    profile: str,
    symbols: list[str],
    timeframes: list[str],
    max_parameter_sets: int,
    allocation_amount_usd: float,
    lock_duration_hours: int,
    auto_deploy_top_n: int,
    strategy_names: list[str] | None,
    provider: str,
    research_overlay: str = "",
    deadline_monotonic: float | None = None,
    progress: Callable[..., None] | None = None,
    progress_state: dict[str, object] | None = None,
    market_data_mode: str = "live",
) -> list[dict[str, object]]:
    if _live_canary_budget_exhausted(deadline_monotonic):
        return [
            _live_canary_budget_exhausted_sweep(
                stage=stage,
                profile=profile,
                symbols=symbols,
                timeframes=timeframes,
                strategy_names=strategy_names,
                max_parameter_sets=max_parameter_sets,
                research_overlay=research_overlay,
                fallback=False,
            )
        ]
    try:
        _find_canary_phase(
            progress_state,
            progress,
            "running_optimizer_sweep",
            stage=stage,
            profile=profile,
            symbols=symbols,
            fallback=False,
        )
        return [
            _run_live_canary_optimizer_sweep(
                stage=stage,
                profile=profile,
                symbols=symbols,
                timeframes=timeframes,
                max_parameter_sets=max_parameter_sets,
                allocation_amount_usd=allocation_amount_usd,
                lock_duration_hours=lock_duration_hours,
                auto_deploy_top_n=auto_deploy_top_n,
                strategy_names=strategy_names,
                provider=provider,
                research_overlay=research_overlay,
                fallback=False,
                deadline_monotonic=deadline_monotonic,
                progress=progress,
                progress_state=progress_state,
                market_data_mode=market_data_mode,
            )
        ]
    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        failed_phase = str((progress_state or {}).get("failed_phase") or "running_optimizer_sweep")
        results = [
            {
                "stage": stage,
                "profile": profile,
                "symbols": symbols,
                "timeframes": timeframes,
                "ranking_count": 0,
                "accepted_count": 0,
                "rejection_breakdown": {},
                "error": str(exc),
                "failed_phase": failed_phase,
                "fallback": False,
            }
        ]
        for symbol in symbols:
            if _live_canary_budget_exhausted(deadline_monotonic):
                results.append(
                    _live_canary_budget_exhausted_sweep(
                        stage=stage,
                        profile=profile,
                        symbols=[symbol],
                        timeframes=timeframes,
                        strategy_names=strategy_names,
                        max_parameter_sets=max_parameter_sets,
                        research_overlay=research_overlay,
                        fallback=True,
                    )
                )
                break
            try:
                _find_canary_phase(
                    progress_state,
                    progress,
                    "running_optimizer_sweep",
                    stage=stage,
                    profile=profile,
                    symbols=[symbol],
                    fallback=True,
                )
                results.append(
                    _run_live_canary_optimizer_sweep(
                        stage=stage,
                        profile=profile,
                        symbols=[symbol],
                        timeframes=timeframes,
                        max_parameter_sets=max_parameter_sets,
                        allocation_amount_usd=allocation_amount_usd,
                        lock_duration_hours=lock_duration_hours,
                        auto_deploy_top_n=auto_deploy_top_n,
                        strategy_names=strategy_names,
                        provider=provider,
                        research_overlay=research_overlay,
                        fallback=True,
                        deadline_monotonic=deadline_monotonic,
                        progress=progress,
                        progress_state=progress_state,
                        market_data_mode=market_data_mode,
                    )
                )
            except Exception as symbol_exc:  # noqa: BLE001
                db.session.rollback()
                failed_phase = str((progress_state or {}).get("failed_phase") or "running_optimizer_sweep")
                results.append(
                    {
                        "stage": stage,
                        "profile": profile,
                        "symbols": [symbol],
                        "timeframes": timeframes,
                        "ranking_count": 0,
                        "accepted_count": 0,
                        "rejection_breakdown": {},
                        "error": str(symbol_exc),
                        "failed_phase": failed_phase,
                        "fallback": True,
                    }
                )
        return results


def _live_canary_budget_exhausted_sweep(
    *,
    stage: str,
    profile: str,
    symbols: list[str],
    timeframes: list[str],
    strategy_names: list[str] | None,
    max_parameter_sets: int,
    research_overlay: str,
    fallback: bool,
) -> dict[str, object]:
    return {
        "stage": stage,
        "profile": profile,
        "symbols": symbols,
        "timeframes": timeframes,
        "strategy_names": list(strategy_names or []),
        "max_parameter_sets": int(max_parameter_sets or 0),
        "research_overlay": str(research_overlay or ""),
        "optimizer_run_id": None,
        "ranking_count": 0,
        "accepted_count": 0,
        "rejection_breakdown": {},
        "timed_out": True,
        "partial_result": True,
        "partial_reason": "research_budget_exhausted",
        "top_rejected": [],
        "fallback": fallback,
    }


def _run_live_canary_optimizer_sweep(
    *,
    stage: str,
    profile: str,
    symbols: list[str],
    timeframes: list[str],
    max_parameter_sets: int,
    allocation_amount_usd: float,
    lock_duration_hours: int,
    auto_deploy_top_n: int,
    strategy_names: list[str] | None,
    provider: str,
    fallback: bool,
    research_overlay: str = "",
    deadline_monotonic: float | None = None,
    progress: Callable[..., None] | None = None,
    progress_state: dict[str, object] | None = None,
    market_data_mode: str = "live",
) -> dict[str, object]:
    remaining = _live_canary_budget_remaining(deadline_monotonic)
    if remaining is not None and remaining <= 0.0:
        return _live_canary_budget_exhausted_sweep(
            stage=stage,
            profile=profile,
            symbols=symbols,
            timeframes=timeframes,
            strategy_names=strategy_names,
            max_parameter_sets=max_parameter_sets,
            research_overlay=research_overlay,
            fallback=fallback,
        )
    optimizer = get_service("strategy_optimizer")
    config = optimizer.default_config(
        symbols=symbols,
        timeframes=timeframes,
        strategy_names=strategy_names,
        profile=profile,
        allocation_amount_usd=allocation_amount_usd,
        lock_duration_hours=lock_duration_hours,
        universe_mode="configured",
        max_parallel_legs=int(current_app.config.get("VAULT_MAX_PARALLEL_LEGS", 1)),
        allow_leverage_experiment=False,
        mode=str(market_data_mode or "live"),
    )
    config.max_parameter_sets = max(1, int(max_parameter_sets or 1))
    config.auto_deploy_top_n = max(0, int(auto_deploy_top_n or 0))
    config.live_canary_research_overlay = str(research_overlay or "")
    config.provider = normalize_exchange_provider(provider)
    if profile == "aggressive_1h":
        config.high_upside_profile = False

    _find_canary_phase(
        progress_state,
        progress,
        "fetching_candles",
        stage=stage,
        profile=profile,
        symbols=symbols,
        timeframes=timeframes,
        fallback=fallback,
    )
    per_sweep_timeout = _find_canary_per_sweep_timeout_seconds()
    timeout_candidates = [_optimizer_timeout_seconds(), per_sweep_timeout]
    if remaining is not None:
        timeout_candidates.append(remaining)
    timeout_override = min(timeout_candidates)
    result = _run_optimizer_payload(config, timeout_seconds_override=timeout_override)
    top_rows = list(result.get("top", []) or [])
    timed_out = bool(result.get("timed_out", False))
    partial_result = bool(result.get("partial_result", False))
    optimizer_runtime = dict(result.get("optimizer_runtime") or {})
    partial_reason = str(result.get("partial_reason") or "")
    failed_phase = (
        _find_canary_optimizer_failed_phase(optimizer_runtime, progress_state)
        if timed_out or partial_result
        else None
    )
    _find_canary_phase(
        progress_state,
        progress,
        "applying_live_canary_filters",
        stage=stage,
        profile=profile,
        rankings=int(result.get("ranking_count") or 0),
        accepted=int(result.get("accepted_count") or 0),
        fallback=fallback,
    )
    return {
        "stage": stage,
        "profile": profile,
        "symbols": symbols,
        "timeframes": timeframes,
        "strategy_names": list(config.strategy_names or []),
        "max_parameter_sets": int(config.max_parameter_sets or 0),
        "research_overlay": config.live_canary_research_overlay,
        "market_data_mode": config.mode,
        "optimizer_run_id": result.get("optimizer_run_id"),
        "ranking_count": int(result.get("ranking_count") or 0),
        "accepted_count": int(result.get("accepted_count") or 0),
        "rejection_breakdown": dict(result.get("rejection_breakdown") or {}),
        "timed_out": timed_out,
        "partial_result": partial_result,
        "failed_phase": failed_phase,
        "partial_reason": partial_reason,
        "optimizer_runtime": optimizer_runtime,
        "optimizer_phase_seconds": dict(optimizer_runtime.get("phase_seconds") or {}),
        "slowest_market": dict(optimizer_runtime.get("slowest_market") or {}),
        "top_rejected": _compact_live_canary_rejected_rows(top_rows),
        "fallback": fallback,
    }


def _find_canary_optimizer_failed_phase(
    optimizer_runtime: dict[str, object],
    progress_state: dict[str, object] | None,
) -> str:
    phase_seconds = {
        str(key): float(value or 0.0)
        for key, value in dict(optimizer_runtime.get("phase_seconds") or {}).items()
    }
    if phase_seconds:
        dominant_phase = max(phase_seconds, key=phase_seconds.get)
        return {
            "candle_fetch": "fetching_candles",
            "slicing": "preparing_optimizer_windows",
            "backtest": "running_optimizer_backtests",
            "persistence": "persisting_optimizer_rankings",
            "finalize": "applying_live_canary_filters",
        }.get(dominant_phase, "running_optimizer_sweep")
    return str((progress_state or {}).get("failed_phase") or "fetching_candles")


def _run_optimizer_payload(config: object, *, timeout_seconds_override: float | None = None) -> dict[str, object]:
    timeout_seconds = _optimizer_timeout_seconds()
    if timeout_seconds_override is not None:
        timeout_seconds = max(0.0, float(timeout_seconds_override or 0.0))
    started_at = time.monotonic()
    config.runtime_deadline_monotonic = _optimizer_cooperative_deadline(started_at, timeout_seconds)
    try:
        with _optimizer_deadline(timeout_seconds):
            result = get_service("strategy_optimizer").run(config)
        return _optimization_result_with_diagnostics(
            config,
            result,
            started_at=started_at,
            timeout_seconds=timeout_seconds,
            timed_out=False,
        )
    except _OptimizerCliTimeout as exc:
        db.session.rollback()
        return _optimization_timeout_result(
            config,
            started_at=started_at,
            timeout_seconds=timeout_seconds,
            error=str(exc),
        )


def _accepted_ranking_for_sweep(
    sweep: dict[str, object],
    profile: str,
    search_started_at: datetime,
    *,
    provider: str,
) -> StrategyRanking | None:
    provider_key = normalize_exchange_provider(provider)
    optimizer_run_id = sweep.get("optimizer_run_id")
    query = StrategyRanking.query.filter_by(rejected=False, provider=provider_key).filter(
        (StrategyRanking.rejection_reason.is_(None)) | (StrategyRanking.rejection_reason == "")
    )
    if optimizer_run_id:
        query = query.filter_by(optimizer_run_id=int(optimizer_run_id))
    else:
        query = query.filter(StrategyRanking.created_at >= search_started_at, StrategyRanking.profile == profile)
    return query.order_by(StrategyRanking.score.desc(), StrategyRanking.id.desc()).first()


def _record_live_canary_preview_audit(
    *,
    ranking: StrategyRanking,
    user: User,
    connection: TradingConnection | None,
    payload: dict[str, object],
    intent: OrderIntent | None,
    submit_requested: bool,
    blocked_by_preview_only: bool,
) -> int | None:
    audit_connection_id = intent.trading_connection_id if intent is not None else None
    if audit_connection_id is None and connection is not None:
        audit_connection_id = connection.id
    audit = AuditLog(
        category="orders",
        action="live_canary_preview",
        message=f"Live canary preview generated from ranking {ranking.id}; no real order submitted.",
        user_id=user.id,
        trading_connection_id=audit_connection_id,
    )
    audit.details = {
        "preview_only": True,
        "real_order_submitted": False,
        "submit_requested": submit_requested,
        "blocked_by_preview_only": blocked_by_preview_only,
        "ranking_id": ranking.id,
        "optimizer_profile": ranking.profile,
        "strategy_name": ranking.strategy_name,
        "symbol": ranking.symbol,
        "timeframe": ranking.timeframe,
        "selected_symbol": payload.get("selected_symbol"),
        "selected_strategy": payload.get("selected_strategy"),
        "side": payload.get("side"),
        "size": payload.get("size"),
        "confidence": payload.get("confidence"),
        "reason": payload.get("reason"),
        "projected_order": payload.get("projected_order", {}),
        "risk_decision": payload.get("risk_decision", {}),
        "signal_quality": payload.get("signal_quality", {}),
        "blockers": payload.get("blockers", []),
    }
    db.session.add(audit)
    db.session.commit()
    return audit.id


def _live_canary_connection(user_id: int, connection_id: int | None) -> tuple[TradingConnection | None, str]:
    service = get_service("trading_connections")
    try:
        if connection_id is not None:
            connection = service.get_for_user(user_id, connection_id)
            if str(connection.verification_status) != "verified":
                return None, "active_verified_live_connection_missing"
            if not bool(service.provider_spec(connection.provider).get("tradable", False)):
                return None, "active_verified_live_connection_missing"
            health = _fresh_connection_health(connection, service)
            if not bool(health.get("can_trade", False)):
                return None, "active_connection_cannot_trade"
            return connection, ""
        connection = service.active_tradable_connection(user_id)
        if connection is None:
            return None, "active_verified_live_connection_missing"
        health = _fresh_connection_health(connection, service)
        if not bool(health.get("can_trade", False)):
            return None, "active_connection_cannot_trade"
        return connection, ""
    except Exception:  # noqa: BLE001
        return None, "active_verified_live_connection_missing"


def _live_canary_signal_payload(ranking: StrategyRanking) -> dict[str, object]:
    blockers: list[str] = []
    strategy_name = str(ranking.strategy_name or "")
    symbol = str(ranking.symbol or "").upper()
    timeframe = str(ranking.timeframe or current_app.config.get("DEFAULT_TIMEFRAME", "15m"))
    parameters = dict(ranking.parameters or {})
    parameters.update(
        {
            "optimizer_profile": ranking.profile,
            "optimizer_ranking_id": ranking.id,
            "lock_duration_hours": ranking.lock_duration_hours,
            "allocation_amount_usd": ranking.allocation_amount_usd,
            "max_drawdown": ranking.max_drawdown,
            "max_favorable_excursion": ranking.max_favorable_excursion,
            "max_adverse_excursion": ranking.max_adverse_excursion,
            "mfe_mae_ratio": ranking.mfe_mae_ratio,
            "expectancy": ranking.expectancy,
            "recent_1h_return": ranking.recent_1h_return,
            "cost_adjusted_recent_1h_return": ranking.cost_adjusted_recent_1h_return,
            "turnover_after_fees": ranking.turnover_after_fees,
            "trades_per_day": ranking.trades_per_day,
            "avg_trade_return": ranking.avg_trade_return,
            "avg_win": ranking.avg_win,
            "avg_loss": ranking.avg_loss,
        }
    )
    market_data = get_service("market_data")
    candles: list[dict[str, object]] = []
    mid = 0.0
    try:
        candles = market_data.get_candles(symbol, timeframe, mode="live", limit=200)
    except Exception as exc:  # noqa: BLE001
        blockers.append("market_data_unavailable")
        return {"signal": None, "diagnostics": {"blockers": blockers, "error": str(exc)}}
    try:
        mid = _float(market_data.get_mid_price(symbol, "live"))
    except Exception:  # noqa: BLE001
        mid = 0.0
    if mid <= 0 and candles:
        mid = _float(candles[-1].get("close") if isinstance(candles[-1], dict) else 0.0)
    if mid <= 0:
        blockers.append("price_unavailable")

    try:
        strategy = get_service("strategy_registry").build(strategy_name, parameters)
        signal = strategy.generate_signal(symbol=symbol, timeframe=timeframe, candles=candles, position={})
    except Exception as exc:  # noqa: BLE001
        return {
            "signal": None,
            "diagnostics": {
                "blockers": ["strategy_signal_error"],
                "error": str(exc),
                "mid": mid,
            },
        }

    signal_payload = signal.as_dict() if hasattr(signal, "as_dict") else {}
    if getattr(signal, "action", "hold") not in {"buy", "sell"}:
        blockers.append("signal_not_actionable")

    try:
        feature_payload = get_service("feature_engine").snapshot(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
        ).as_dict()
    except Exception:  # noqa: BLE001
        feature_payload = {}
    market_snapshot = _live_canary_market_snapshot(symbol, timeframe, mid, candles)
    signal_quality = {}
    if mid > 0:
        signal_quality = SignalQualityEvaluator(current_app.config, get_service("online_ranker")).evaluate(
            symbol=symbol,
            timeframe=timeframe,
            mode="live",
            run_parameters=parameters,
            signal=signal,
            feature_payload=feature_payload,
            mid=mid,
            market_snapshot=market_snapshot,
        )
        if signal_quality.get("no_trade_reason"):
            blockers.append(f"signal_quality:{signal_quality['no_trade_reason']}")

    return {
        "signal": signal,
        "diagnostics": {
            "blockers": blockers,
            "symbol": symbol,
            "timeframe": timeframe,
            "mid": mid,
            "candle_count": len(candles),
            "feature_payload": feature_payload,
            "market_snapshot": market_snapshot,
            "signal_payload": signal_payload,
            "signal_quality": signal_quality,
        },
    }


def _live_canary_market_snapshot(
    symbol: str,
    timeframe: str,
    mid: float,
    candles: list[dict[str, object]],
) -> dict[str, object]:
    snapshot: dict[str, object] = {}
    try:
        realtime = get_service("realtime_market").snapshot(symbol, "live", timeframe=timeframe)
        if isinstance(realtime, dict):
            snapshot.update(realtime)
    except Exception:  # noqa: BLE001
        pass
    try:
        structure = get_service("market_structure").snapshot(symbol, timeframe, mode="live")
        if isinstance(structure, dict):
            snapshot["market_structure"] = structure
            snapshot["market_structure_score"] = structure.get("score", 0.0)
            snapshot["market_structure_trend"] = structure.get("trend_score", structure.get("score", 0.0))
    except Exception:  # noqa: BLE001
        pass
    if "spread_bps" not in snapshot or "liquidity_usd" not in snapshot:
        book = {}
        try:
            book = get_service("market_data").get_order_book(symbol, "live")
        except Exception:  # noqa: BLE001
            book = {}
        book_payload = _order_book_quality(book, mid)
        snapshot.update({key: value for key, value in book_payload.items() if key not in snapshot or not snapshot.get(key)})
    if "volatility_pct" not in snapshot:
        snapshot["volatility_pct"] = _candle_volatility_pct(candles)
    snapshot.setdefault("source", "live_canary")
    snapshot.setdefault("signal_stability", 1.0)
    snapshot.setdefault("stale_data", False)
    snapshot.setdefault("stale_data_age_seconds", 0.0)
    return snapshot


def _live_canary_order_intent(
    *,
    ranking: StrategyRanking,
    user_id: int,
    connection_id: int | None,
    connection: TradingConnection | None = None,
    available_balance_usd: float | None = None,
    signal: object,
    signal_diagnostics: dict[str, object],
) -> tuple[OrderIntent | None, list[str], dict[str, object]]:
    blockers: list[str] = []
    mid = _float(signal_diagnostics.get("mid"))
    side = str(getattr(signal, "action", "")).lower()
    stop_loss = _float(getattr(signal, "stop_loss", None))
    take_profit = _float(getattr(signal, "take_profit", None))
    micro_guardrail = _live_micro_canary_guardrail()
    micro_enabled = bool(micro_guardrail.get("enabled", False))
    if micro_enabled:
        blockers.extend(str(item) for item in micro_guardrail.get("blockers", []) or [])
    if mid <= 0:
        blockers.append("price_unavailable")
    if stop_loss <= 0:
        blockers.append("stop_loss_required")
    elif side == "buy" and stop_loss >= mid:
        blockers.append("invalid_stop_loss")
    elif side == "sell" and stop_loss <= mid:
        blockers.append("invalid_stop_loss")
    if micro_enabled and bool(micro_guardrail.get("require_take_profit", True)) and take_profit <= 0:
        blockers.append("take_profit_required")
    if side not in {"buy", "sell"}:
        blockers.append("signal_not_actionable")

    parameters = dict(ranking.parameters or {})
    if micro_enabled and any(
        bool(parameters.get(key))
        for key in ("averaging_down", "allow_averaging_down", "dca_enabled", "scale_in_enabled")
    ):
        blockers.append("averaging_down_disabled")
    cap = _canary_effective_allocation_budget(ranking, parameters)
    if cap <= 0:
        blockers.append("allocation_budget_missing")
    stop_distance_pct = abs(mid - stop_loss) / max(mid, 1e-9) if mid > 0 and stop_loss > 0 else 0.0
    risk_fraction = _clamp(
        _float(parameters.get("risk_fraction"), _float(current_app.config.get("RISK_PER_TRADE_PCT"), 0.01)),
        0.0,
        _float(current_app.config.get("VAULT_MAX_RISK_FRACTION"), 0.03),
    )
    if micro_enabled:
        risk_fraction = min(risk_fraction, _float(micro_guardrail.get("max_risk_pct"), 0.01))
    account_risk_budget = (
        _float(micro_guardrail.get("account_usd"), 0.0) * risk_fraction
        if micro_enabled
        else None
    )
    risk_budget_candidates = [cap * risk_fraction]
    if account_risk_budget is not None and account_risk_budget >= 0:
        risk_budget_candidates.append(account_risk_budget)
    risk_budget = min(value for value in risk_budget_candidates if value >= 0)
    notional_by_risk = risk_budget / stop_distance_pct if stop_distance_pct > 0 else 0.0
    notional_by_signal = cap * _clamp(_float(getattr(signal, "position_fraction", 1.0), 1.0), 0.0, 1.0)
    candidate_notionals = [value for value in (cap, notional_by_risk, notional_by_signal) if value > 0]
    notional = min(candidate_notionals) if candidate_notionals else 0.0
    quantity = notional / mid if mid > 0 else 0.0
    if quantity <= 0:
        blockers.append("invalid_size")

    quality = signal_diagnostics.get("signal_quality") if isinstance(signal_diagnostics.get("signal_quality"), dict) else {}
    execution_style = str(quality.get("suggested_execution_style") or ranking.execution_style or "market")
    order_type = "limit" if execution_style == "maker_limit" else "market"
    limit_price = None
    if order_type == "limit" and mid > 0:
        spread_bps = _float(quality.get("spread_bps"), _float((signal_diagnostics.get("market_snapshot") or {}).get("spread_bps", 1.0)) if isinstance(signal_diagnostics.get("market_snapshot"), dict) else 1.0)
        offset = min(max(spread_bps / 20_000.0, 0.0001), 0.0015)
        limit_price = round(mid * (1 - offset if side == "buy" else 1 + offset), 8)

    metadata = {
        "live_canary": True,
        "optimizer_ranking_id": ranking.id,
        "optimizer_profile": ranking.profile,
        "strategy_name": ranking.strategy_name,
        "timeframe": ranking.timeframe,
        "duration_hours": ranking.lock_duration_hours,
        "high_upside_profile": bool(parameters.get("high_upside_profile", False)),
        "profit_objective_version": parameters.get("profit_objective_version", ""),
        "max_drawdown": ranking.max_drawdown,
        "expected_stop_loss": stop_loss,
        "expected_take_profit": take_profit,
        "signal_metadata": getattr(signal, "metadata", {}) or {},
        "signal_quality_breakdown": quality.get("signal_quality_breakdown", {}),
        "net_roi_score": quality.get("net_roi_score", parameters.get("net_roi_score", 0.0)),
        "net_roi_v2_score": quality.get("net_roi_v2_score", parameters.get("net_roi_v2_score", 0.0)),
        "roi_quality_grade": quality.get("roi_quality_grade", parameters.get("roi_quality_grade", "D")),
        "roi_rejection_risk": quality.get("roi_rejection_risk", parameters.get("roi_rejection_risk", "high")),
        "regime_support": quality.get("regime_support", parameters.get("regime_support", "regime-neutral")),
        "regime_bucket": quality.get("regime_bucket", parameters.get("regime_bucket", {})),
        "tail_loss_penalty": quality.get("tail_loss_penalty", parameters.get("tail_loss_penalty", 0.0)),
        "downside_asymmetry_penalty": quality.get(
            "downside_asymmetry_penalty",
            parameters.get("downside_asymmetry_penalty", 0.0),
        ),
        "cost_adjusted_breakout_potential": quality.get(
            "cost_adjusted_breakout_potential",
            parameters.get("cost_adjusted_breakout_potential", 0.0),
        ),
        "expected_fill_quality": quality.get("expected_fill_quality", parameters.get("expected_fill_quality", 0.0)),
        "churn_penalty": quality.get("churn_penalty", parameters.get("churn_penalty", 0.0)),
        "edge_after_cost_bps": quality.get("edge_after_cost_bps", parameters.get("edge_after_cost_bps", 0.0)),
        "cost_drag_bps": quality.get("cost_drag_bps", ranking.cost_drag_bps),
        "canary_sizing": {
            "effective_allocation_budget_usd": cap,
            "risk_fraction": risk_fraction,
            "risk_budget": risk_budget,
            "account_risk_budget": account_risk_budget,
            "stop_distance_pct": stop_distance_pct,
            "notional_by_risk": notional_by_risk,
            "notional_by_signal": notional_by_signal,
            "computed_notional": notional,
        },
        "live_micro_canary": micro_guardrail,
    }
    if blockers:
        return None, blockers, {
            "sizing": metadata["canary_sizing"],
            "projected_order": {},
            "live_micro_canary": micro_guardrail,
        }

    max_live_leverage = _float(current_app.config.get("MAX_LEVERAGE"), 1.0)
    max_canary_leverage = _float(current_app.config.get("FIRST_CANARY_MAX_LEVERAGE"), 1.0)
    max_micro_leverage = _float(micro_guardrail.get("max_leverage"), 0.0) if micro_enabled else 0.0
    leverage_caps = [value for value in (max_live_leverage, max_canary_leverage, max_micro_leverage) if value > 0]
    leverage_cap = min(leverage_caps) if leverage_caps else 1.0
    ranking_leverage = _float(ranking.leverage, 1.0)
    leverage = min(ranking_leverage if ranking_leverage > 0 else 1.0, leverage_cap)
    if leverage <= 0:
        blockers.append("invalid_leverage")
        return None, blockers, {
            "sizing": metadata["canary_sizing"],
            "projected_order": {},
            "live_micro_canary": micro_guardrail,
        }
    metadata["canary_sizing"]["max_leverage"] = leverage_cap
    metadata["canary_sizing"]["requested_leverage"] = ranking_leverage
    metadata["canary_sizing"]["computed_leverage"] = leverage
    slippage_pct = 0.0
    micro_sizing = _resolve_live_micro_canary_order_size(
        ranking=ranking,
        parameters=parameters,
        connection=connection,
        mid=mid,
        stop_loss=stop_loss,
        take_profit=take_profit,
        leverage=leverage,
        requested_notional=notional,
        normal_cap=cap,
        stop_distance_pct=stop_distance_pct,
        slippage_pct=slippage_pct,
        available_balance_usd=available_balance_usd,
        micro_guardrail=micro_guardrail,
    )
    live_micro_canary = {**micro_guardrail, **micro_sizing}
    live_micro_canary["active_blockers"] = list(
        dict.fromkeys(
            str(item)
            for item in [
                *list(micro_guardrail.get("blockers", []) or []),
                *list(micro_sizing.get("blockers", []) or []),
            ]
            if str(item)
        )
    )
    micro_blockers = [str(item) for item in micro_sizing.get("blockers", []) or []]
    if micro_blockers:
        blockers.extend(micro_blockers)
        metadata["live_micro_canary"] = live_micro_canary
        return None, list(dict.fromkeys(blockers)), {
            "sizing": metadata["canary_sizing"],
            "projected_order": {},
            "live_micro_canary": live_micro_canary,
        }
    notional = _float(micro_sizing.get("final_allowed_notional_usd"), notional)
    quantity = notional / mid if mid > 0 else 0.0
    metadata["canary_sizing"]["computed_notional"] = notional
    metadata["canary_sizing"]["final_allowed_notional"] = notional
    metadata["canary_sizing"]["micro_sizing"] = micro_sizing
    metadata["live_micro_canary"] = live_micro_canary

    intent = OrderIntent(
        symbol=ranking.symbol,
        side=side,
        quantity=quantity,
        mode="live",
        order_type=order_type,
        limit_price=limit_price,
        reduce_only=False,
        leverage=leverage,
        stop_loss=stop_loss,
        take_profit=take_profit if take_profit > 0 else None,
        strategy_name=ranking.strategy_name,
        timeframe=ranking.timeframe,
        slippage_pct=slippage_pct,
        user_id=user_id,
        trading_connection_id=connection_id,
        metadata=metadata,
    )
    return intent, [], {
        "sizing": metadata["canary_sizing"],
        "live_micro_canary": live_micro_canary,
        "projected_order": {
            "symbol": intent.symbol,
            "side": intent.side,
            "quantity": intent.quantity,
            "notional": notional,
            "order_type": intent.order_type,
            "limit_price": intent.limit_price,
            "leverage": intent.leverage,
            "stop_loss": intent.stop_loss,
            "take_profit": intent.take_profit,
            "slippage_pct": intent.slippage_pct,
        },
    }


def _ranking_canary_summary(ranking: StrategyRanking) -> dict[str, object]:
    explanation = ranking.ml_explanation if isinstance(ranking.ml_explanation, dict) else {}
    net_roi_v2 = explanation.get("net_roi_v2") if isinstance(explanation.get("net_roi_v2"), dict) else {}
    one_hour_pref = explanation.get("one_hour_live_preference") if isinstance(explanation.get("one_hour_live_preference"), dict) else {}
    return {
        "id": ranking.id,
        "provider": normalize_exchange_provider(getattr(ranking, "provider", "global")),
        "collateral_asset": provider_collateral_asset(getattr(ranking, "provider", "global")),
        "profile": ranking.profile,
        "strategy_name": ranking.strategy_name,
        "symbol": ranking.symbol,
        "timeframe": ranking.timeframe,
        "score": ranking.score,
        "rejected": bool(ranking.rejected),
        "rejection_reason": ranking.rejection_reason,
        "net_roi_v2_score": net_roi_v2.get("net_roi_v2_score", 0.0),
        "roi_quality_grade": net_roi_v2.get("roi_quality_grade", "D"),
        "roi_rejection_risk": net_roi_v2.get("roi_rejection_risk", "high"),
        "regime_support": net_roi_v2.get("regime_support", "regime-neutral"),
        "one_hour_high_upside_score": one_hour_pref.get("one_hour_high_upside_score", 0.0),
        "accepted_for_one_hour_live_preview": one_hour_pref.get("accepted_for_one_hour_live_preview"),
        "one_hour_live_blockers": one_hour_pref.get("one_hour_live_blockers", []),
    }


def _canary_effective_allocation_budget(ranking: StrategyRanking, parameters: dict[str, object]) -> float:
    first_canary = _first_canary_guardrail()
    micro_guardrail = _live_micro_canary_guardrail()
    first_canary_cap = (
        _float(first_canary.get("fallback_allocation_budget_usdt"), 2.0)
        if (
            not bool(micro_guardrail.get("enabled", False))
            and (bool(parameters.get("first_canary_use_min_size_fallback")) or _config_flag("FIRST_CANARY_USE_MIN_SIZE_FALLBACK", False))
        )
        else _float(first_canary.get("preferred_allocation_budget_usdt"), 1.0)
    )
    caps = [
        first_canary_cap,
        _float(ranking.allocation_amount_usd),
        _float(parameters.get("allocation_cap_usd")),
    ]
    if bool(micro_guardrail.get("enabled", False)):
        caps.append(_float(micro_guardrail.get("max_allocation_usd"), 1.0))
    positive = [value for value in caps if value > 0]
    return min(positive) if positive else 0.0


def _order_book_quality(book: dict[str, object], mid: float) -> dict[str, float]:
    levels = book.get("levels") if isinstance(book, dict) else None
    if not isinstance(levels, list) or len(levels) < 2 or mid <= 0:
        return {"spread_bps": 0.0, "liquidity_usd": 0.0}
    bids = levels[0] if isinstance(levels[0], list) else []
    asks = levels[1] if isinstance(levels[1], list) else []
    bid = _book_level_price(bids[0]) if bids else 0.0
    ask = _book_level_price(asks[0]) if asks else 0.0
    spread_bps = ((ask - bid) / mid) * 10_000 if bid > 0 and ask > 0 else 0.0
    liquidity = 0.0
    for row in [*bids[:5], *asks[:5]]:
        price = _book_level_price(row)
        size = _book_level_size(row)
        liquidity += max(price, 0.0) * max(size, 0.0)
    return {"spread_bps": spread_bps, "liquidity_usd": liquidity}


def _book_level_price(row: object) -> float:
    if isinstance(row, dict):
        return _float(row.get("px", row.get("price")))
    if isinstance(row, (list, tuple)) and row:
        return _float(row[0])
    return 0.0


def _book_level_size(row: object) -> float:
    if isinstance(row, dict):
        return _float(row.get("sz", row.get("size")))
    if isinstance(row, (list, tuple)) and len(row) > 1:
        return _float(row[1])
    return 0.0


def _candle_volatility_pct(candles: list[dict[str, object]]) -> float:
    closes = [_float(row.get("close")) for row in candles if isinstance(row, dict)]
    closes = [value for value in closes if value > 0]
    if len(closes) < 3:
        return 0.0
    returns = [abs((closes[index] - closes[index - 1]) / closes[index - 1]) for index in range(1, len(closes))]
    sample = returns[-20:] if len(returns) >= 20 else returns
    return sum(sample) / max(len(sample), 1) * 100


def _float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _json_float_map(value: str) -> dict[str, float]:
    raw = json.loads(value or "{}")
    if not isinstance(raw, dict):
        raise click.BadParameter("Expected a JSON object mapping duration buckets to numeric caps.")
    parsed: dict[str, float] = {}
    for key, item in raw.items():
        try:
            parsed[str(key).strip().lower()] = float(item)
        except (TypeError, ValueError) as exc:
            raise click.BadParameter(f"Cap for {key!r} must be numeric.") from exc
    return parsed


def _optimizer_timeout_seconds() -> float:
    try:
        return max(0.0, float(current_app.config.get("OPTIMIZER_MARKET_DATA_TIMEOUT_SECONDS", 120.0) or 0.0))
    except (TypeError, ValueError):
        return 120.0


def _optimizer_cooperative_deadline(started_at: float, timeout_seconds: float) -> float:
    timeout = max(0.0, float(timeout_seconds or 0.0))
    if timeout <= 0:
        return 0.0
    headroom = min(max(timeout * 0.05, 0.5), 5.0)
    return float(started_at) + max(timeout - headroom, timeout * 0.5)


@contextmanager
def _optimizer_deadline(timeout_seconds: float):
    timeout = max(0.0, float(timeout_seconds or 0.0))
    if timeout <= 0 or not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum, _frame) -> None:
        raise _OptimizerCliTimeout(timeout)

    signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


@contextmanager
def _find_canary_deadline(timeout_seconds: float, progress_state: dict[str, object] | None):
    timeout = max(0.0, float(timeout_seconds or 0.0))
    if timeout <= 0 or not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum, _frame) -> None:
        failed_phase = None
        if progress_state is not None:
            failed_phase = str(progress_state.get("failed_phase") or progress_state.get("current_phase") or "unknown")
        raise _FindCanaryRankingTimeout(timeout, failed_phase)

    signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


@contextmanager
def _high_upside_deadline(timeout_seconds: float, progress_state: dict[str, object] | None):
    timeout = max(0.0, float(timeout_seconds or 0.0))
    if timeout <= 0 or not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum, _frame) -> None:
        failed_phase = "unknown"
        if progress_state is not None:
            failed_phase = str(progress_state.get("failed_phase") or progress_state.get("current_phase") or "unknown")
        raise _HighUpsideDiscoveryTimeout(timeout, failed_phase)

    signal.signal(signal.SIGALRM, _raise_timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _optimization_result_with_diagnostics(
    config: object,
    result: dict[str, object],
    *,
    started_at: float,
    timeout_seconds: float,
    timed_out: bool,
) -> dict[str, object]:
    elapsed = max(time.monotonic() - started_at, 0.0)
    existing_runtime = dict((result or {}).get("optimizer_runtime") or {})
    timed_out_effective = bool(timed_out or (result or {}).get("timed_out", False) or existing_runtime.get("timed_out", False))
    payload: dict[str, object] = {
        **dict(result or {}),
        "offline_ml_readiness": _offline_ml_readiness(_duration_bucket(getattr(config, "lock_duration_hours", 1) or 1)),
        "live_canary_readiness": _optimizer_live_canary_readiness(config),
        "optimizer_runtime": {
            **existing_runtime,
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": elapsed,
            "timed_out": timed_out_effective,
            "source": "OPTIMIZER_MARKET_DATA_TIMEOUT_SECONDS",
            "market_data_cache": _market_data_cache_stats(),
        },
    }
    if getattr(config, "profile", "") == "dynamic_intraday":
        payload["dynamic_intraday_live_readiness"] = _dynamic_intraday_live_readiness(config)
    if bool(getattr(config, "high_upside_profile", False)):
        scanner_diagnostics = _bounded_optimizer_scanner_diagnostics(config, timeout_seconds=timeout_seconds)
        payload["scanner_diagnostics"] = scanner_diagnostics
        payload["high_upside_live_readiness"] = _high_upside_live_readiness(config, scanner_diagnostics)
    return payload


def _optimization_timeout_result(
    config: object,
    *,
    started_at: float,
    timeout_seconds: float,
    error: str,
) -> dict[str, object]:
    payload = _optimization_result_with_diagnostics(
        config,
        {
            "ok": False,
            "profile": getattr(config, "profile", ""),
            "timed_out": True,
            "ranking_count": 0,
            "accepted_count": 0,
            "top": [],
            "warnings": [
                "Optimizer command exceeded its configured runtime bound before completing research diagnostics.",
                "No live orders are created by run-optimization.",
            ],
            "error": error,
        },
        started_at=started_at,
        timeout_seconds=timeout_seconds,
        timed_out=True,
    )
    payload["optimizer_runtime"] = {
        **dict(payload.get("optimizer_runtime") or {}),
        "error": error,
        "action": "Increase OPTIMIZER_MARKET_DATA_TIMEOUT_SECONDS or narrow symbols/timeframes when market data is slow.",
    }
    return payload


def _discover_high_upside_vault_candidates_payload(
    *,
    user_id: int,
    provider_scope: str | None,
    providers: list[str],
    symbols: list[str],
    timeframes: list[str],
    profiles: list[str],
    strategy_names: list[str],
    max_symbols: int | None,
    max_sweeps: int | None,
    max_parameter_sets: int,
    allocation_amount_usd: float,
    lock_duration_hours: int,
    timeout_seconds: float,
    started_at: float,
    progress_state: dict[str, object],
    run_backtests: bool,
    objective: str = "risk_adjusted",
    target_roi_pct: float | None = None,
) -> dict[str, object]:
    _high_upside_phase(progress_state, "loading_services")
    config = current_app.config
    enabled = bool(config.get("HIGH_UPSIDE_DISCOVERY_ENABLED", False))
    selected_scope = str(provider_scope or config.get("HIGH_UPSIDE_PROVIDER_SCOPE", "multi_provider") or "multi_provider").lower()
    selected_timeframes = _high_upside_timeframes(timeframes)
    selected_profiles = _high_upside_profiles(profiles)
    selected_objective = _ml_objective(objective)
    target_roi = _ml_extreme_target_roi_pct(target_roi_pct, objective=selected_objective, horizon=_duration_bucket(lock_duration_hours or 1))
    symbol_cap = _high_upside_positive_int(max_symbols, "HIGH_UPSIDE_MAX_SYMBOLS", 12)
    sweep_cap = _high_upside_positive_int(max_sweeps, "HIGH_UPSIDE_MAX_SWEEPS", 8)
    requested_symbols = _dedupe_symbols_preserve_case(symbols)
    provider_sequence = _high_upside_provider_sequence(user_id, selected_scope, providers)
    deadline_monotonic = started_at + max(0.0, float(timeout_seconds or 0.0))
    blockers: list[str] = []

    if not enabled:
        blockers.append("HIGH_UPSIDE_DISCOVERY_ENABLED=false")
    if not provider_sequence:
        blockers.append("provider_universe_empty")

    base_config = SimpleNamespace(
        high_upside_profile=True,
        lock_duration_hours=max(1, int(lock_duration_hours or 1)),
        profile=selected_profiles[0] if selected_profiles else "aggressive_1h",
    )
    readiness = _high_upside_live_readiness(base_config, {})
    ml_readiness = _ml_suite_readiness(_duration_bucket(lock_duration_hours or 1))
    extreme_readiness = _ml_extreme_upside_readiness(_duration_bucket(lock_duration_hours or 1), selected_objective)
    ml_blockers = list(ml_readiness.get("blockers", []) or [])
    if blockers:
        return {
            "ready": False,
            "research_only": True,
            "live_orders_created": False,
            "submitted": False,
            "blockers": blockers,
            "provider_scope": selected_scope,
            "objective": selected_objective,
            "target_roi_pct": target_roi,
            "target_roi_policy": _ml_extreme_target_policy(selected_objective, target_roi),
            "providers_requested": provider_sequence,
            "providers": [],
            "accepted_ranking_ids": [],
            "accepted_rankings": [],
            "scanner_candidates": [],
            "rejection_breakdown": {},
            "high_upside_live_readiness": readiness,
            "ml_readiness": ml_readiness,
            "extreme_upside_readiness": extreme_readiness,
            "ml_models": ml_readiness.get("families", {}),
            "ml_blockers": ml_blockers,
            "next_commands": _high_upside_next_commands([]),
            "runtime": _high_upside_runtime(started_at, timeout_seconds, timed_out=False),
        }

    provider_results: list[dict[str, object]] = []
    accepted_rankings: list[dict[str, object]] = []
    scanner_candidates: list[dict[str, object]] = []
    rejection_breakdown: dict[str, int] = {}
    sweeps_used = 0

    for provider in provider_sequence:
        if _high_upside_deadline_reached(deadline_monotonic):
            progress_state["failed_phase"] = "provider_scan"
            raise _HighUpsideDiscoveryTimeout(timeout_seconds, "provider_scan")
        result = _high_upside_provider_result(
            provider=provider,
            user_id=user_id,
            requested_symbols=requested_symbols,
            timeframes=selected_timeframes,
            profiles=selected_profiles,
            strategy_names=strategy_names,
            symbol_cap=symbol_cap,
            sweep_cap=sweep_cap,
            sweeps_used=sweeps_used,
            max_parameter_sets=max_parameter_sets,
            allocation_amount_usd=allocation_amount_usd,
            lock_duration_hours=lock_duration_hours,
            deadline_monotonic=deadline_monotonic,
            timeout_seconds=timeout_seconds,
            progress_state=progress_state,
            run_backtests=run_backtests,
            objective=selected_objective,
            target_roi_pct=target_roi,
        )
        provider_results.append(result)
        sweeps_used += int(result.get("sweeps_used", 0) or 0)
        scanner_candidates.extend(list(result.get("scanner_candidates", []) or []))
        accepted_rankings.extend(list(result.get("accepted_rankings", []) or []))
        for reason, count in dict(result.get("rejection_breakdown", {}) or {}).items():
            rejection_breakdown[str(reason)] = rejection_breakdown.get(str(reason), 0) + int(count or 0)
        if sweeps_used >= sweep_cap:
            break

    accepted_rankings = _dedupe_rankings(accepted_rankings)
    accepted_ids = [int(row["id"]) for row in accepted_rankings if row.get("id") is not None]
    if not accepted_ids:
        blockers.append("accepted_ranking_missing")
    if all(not bool(row.get("market_data_supported", False)) for row in provider_results):
        blockers.append("provider_market_data_unavailable")

    readiness = _high_upside_live_readiness(
        base_config,
        {
            "accepted": scanner_candidates,
            "rejected": [],
            "rejection_breakdown": rejection_breakdown,
            "rejection_rate": _high_upside_rejection_rate(scanner_candidates, rejection_breakdown),
        },
    )

    return {
        "ready": bool(accepted_ids),
        "research_only": True,
        "live_orders_created": False,
        "submitted": False,
        "blockers": list(dict.fromkeys(blockers)),
        "provider_scope": selected_scope,
        "objective": selected_objective,
        "target_roi_pct": target_roi,
        "target_roi_policy": _ml_extreme_target_policy(selected_objective, target_roi),
        "providers_requested": provider_sequence,
        "timeframes": selected_timeframes,
        "profiles": selected_profiles,
        "evaluated_sides": ["long", "short"],
        "max_symbols": symbol_cap,
        "max_sweeps": sweep_cap,
        "sweeps_used": sweeps_used,
        "providers": provider_results,
        "scanner_candidates": scanner_candidates[:50],
        "accepted_ranking_ids": accepted_ids,
        "accepted_rankings": accepted_rankings[:20],
        "rejection_breakdown": rejection_breakdown,
        "high_upside_live_readiness": readiness,
        "ml_readiness": ml_readiness,
        "extreme_upside_readiness": extreme_readiness,
        "ml_models": ml_readiness.get("families", {}),
        "ml_blockers": ml_blockers,
        "next_commands": _high_upside_next_commands(accepted_ids),
        "operator_next_steps": _high_upside_operator_next_steps(bool(accepted_ids), readiness, blockers),
        "runtime": _high_upside_runtime(started_at, timeout_seconds, timed_out=False),
    }


def _high_upside_provider_result(
    *,
    provider: str,
    user_id: int,
    requested_symbols: list[str],
    timeframes: list[str],
    profiles: list[str],
    strategy_names: list[str],
    symbol_cap: int,
    sweep_cap: int,
    sweeps_used: int,
    max_parameter_sets: int,
    allocation_amount_usd: float,
    lock_duration_hours: int,
    deadline_monotonic: float,
    timeout_seconds: float,
    progress_state: dict[str, object],
    run_backtests: bool,
    objective: str,
    target_roi_pct: float,
) -> dict[str, object]:
    provider = _normalize_provider(provider)
    result: dict[str, object] = {
        "provider": provider,
        "objective": objective,
        "target_roi_pct": target_roi_pct,
        "market_data_supported": _high_upside_provider_market_data_supported(provider),
        "status": "pending",
        "blockers": [],
        "symbols": [],
        "stage1_diagnostics": {},
        "market_data_policy": _high_upside_market_data_policy(),
        "scanner_diagnostics": [],
        "scanner_candidates": [],
        "sweeps": [],
        "sweeps_used": 0,
        "accepted_rankings": [],
        "rejection_breakdown": {},
    }
    if not bool(result["market_data_supported"]):
        result["status"] = "skipped"
        result["blockers"] = ["provider_market_data_unavailable"]
        return result

    _high_upside_phase(progress_state, "loading_market_universe", provider=provider)
    candidate_plan = _high_upside_candidate_plan(
        provider=provider,
        requested_symbols=requested_symbols,
        timeframe=timeframes[0],
        symbol_cap=symbol_cap,
        deadline_monotonic=deadline_monotonic,
        timeout_seconds=timeout_seconds,
        progress_state=progress_state,
    )
    symbols = list(candidate_plan.get("symbols", []) or [])
    result["symbols"] = symbols
    result["stage1_diagnostics"] = candidate_plan
    if not symbols:
        result["status"] = "blocked"
        result["blockers"] = ["candidate_symbols_missing"]
        return result

    scanner_candidates: list[dict[str, object]] = []
    rejection_breakdown: dict[str, int] = {}
    scanner = get_service("market_scanner")
    market_mode = _high_upside_market_data_mode(provider)
    for timeframe in timeframes:
        if _high_upside_deadline_reached(deadline_monotonic):
            progress_state["failed_phase"] = "scoring_candidates"
            raise _HighUpsideDiscoveryTimeout(timeout_seconds, "scoring_candidates")
        _high_upside_phase(progress_state, "scoring_candidates", provider=provider, timeframe=timeframe)
        _high_upside_market_data_pace(progress_state, "scoring_candidates")
        scan_started = time.monotonic()
        try:
            scanner.score_candidates(
                symbols,
                mode=market_mode,
                timeframe=timeframe,
                duration_seconds=max(1, int(lock_duration_hours or 1)) * 3600,
                strategy_name=strategy_names[0] if strategy_names else "scalping",
                optimizer_profile=profiles[0] if profiles else "aggressive_1h",
            )
            diagnostics = dict(getattr(scanner, "last_scan_diagnostics", {}) or {})
        except Exception as exc:  # noqa: BLE001
            sanitized_error = _high_upside_sanitize_error(exc)
            diagnostics = {
                "accepted": [],
                "rejected": [],
                "rejection_breakdown": {"scanner_error": 1},
                "rejection_rate": 1.0,
                "error": sanitized_error,
                "rate_limited": _high_upside_error_is_rate_limited(sanitized_error),
            }
        diagnostics["scan_elapsed_seconds"] = max(time.monotonic() - scan_started, 0.0)
        diagnostics["stage1_symbol_count"] = len(symbols)
        diagnostics["failed_phase"] = diagnostics.get("failed_phase") or ""
        if _high_upside_diagnostics_rate_limited(diagnostics):
            _high_upside_record_rate_limit_backoff(provider, timeframe)
            result["blockers"] = list(dict.fromkeys([*list(result.get("blockers", []) or []), "provider_rate_limited"]))
        result["scanner_diagnostics"].append({"timeframe": timeframe, **diagnostics})
        for reason, count in dict(diagnostics.get("rejection_breakdown", {}) or {}).items():
            rejection_breakdown[str(reason)] = rejection_breakdown.get(str(reason), 0) + int(count or 0)
        for row in list(diagnostics.get("accepted", []) or []):
            ml_signal = _high_upside_ml_signal_score(
                row,
                provider=provider,
                timeframe=timeframe,
                lock_duration_hours=lock_duration_hours,
            )
            extreme_decision = _high_upside_extreme_upside_score(
                row,
                provider=provider,
                timeframe=timeframe,
                lock_duration_hours=lock_duration_hours,
                target_roi_pct=target_roi_pct,
                objective=objective,
            )
            if (
                bool(current_app.config.get("ML_SIGNAL_MODEL_ENABLED", False))
                and bool(current_app.config.get("HIGH_UPSIDE_REQUIRE_ML_SIGNAL", True))
                and not bool(ml_signal.get("ready_for_live", False))
            ):
                rejection_breakdown["ml_signal_not_ready"] = rejection_breakdown.get("ml_signal_not_ready", 0) + 1
                continue
            candidate = {
                **dict(row),
                "provider": provider,
                "timeframe": timeframe,
                "direction": _high_upside_direction(row),
                "market_data_mode": market_mode,
                "ml_decision": dict(row.get("ml_decision") or {}),
                "ml_universe_decision": dict(row.get("ml_universe_decision") or {}),
                "ml_signal_model": ml_signal,
                "ml_signal_priority_score": _high_upside_ml_signal_priority(ml_signal),
                "ml_extreme_upside_decision": extreme_decision,
                "extreme_upside_priority_score": _high_upside_extreme_priority(extreme_decision),
                "distance_to_target_pct": _float((extreme_decision.get("raw") or {}).get("distance_to_target_pct")),
            }
            scanner_candidates.append(candidate)

    scanner_candidates.sort(
        key=lambda item: (
            _float(item.get("extreme_upside_priority_score")) if objective == "extreme_upside" else 0.0,
            _float(item.get("ml_signal_priority_score")),
            _float(item.get("score")),
        ),
        reverse=True,
    )
    result["scanner_candidates"] = scanner_candidates
    if not run_backtests:
        result["status"] = "scored"
        result["rejection_breakdown"] = rejection_breakdown
        return result
    if not scanner_candidates:
        result["status"] = "blocked"
        result["blockers"] = ["scanner_accepted_candidates_missing"]
        result["rejection_breakdown"] = rejection_breakdown
        return result

    optimizer = get_service("strategy_optimizer")
    accepted_rankings: list[dict[str, object]] = []
    sweep_rows: list[dict[str, object]] = []
    for candidate in scanner_candidates:
        if sweeps_used + int(result["sweeps_used"]) >= sweep_cap:
            break
        for profile in profiles:
            if sweeps_used + int(result["sweeps_used"]) >= sweep_cap:
                break
            if _high_upside_deadline_reached(deadline_monotonic):
                progress_state["failed_phase"] = "running_backtests"
                raise _HighUpsideDiscoveryTimeout(timeout_seconds, "running_backtests")
            symbol = str(candidate.get("symbol") or "").upper()
            timeframe = str(candidate.get("timeframe") or timeframes[0])
            _high_upside_phase(progress_state, "running_backtests", provider=provider, symbol=symbol, timeframe=timeframe, profile=profile)
            _high_upside_market_data_pace(progress_state, "running_backtests")
            optimizer_config = optimizer.default_config(
                symbols=[symbol],
                timeframes=[timeframe],
                strategy_names=strategy_names or None,
                profile=profile,
                allocation_amount_usd=allocation_amount_usd,
                lock_duration_hours=lock_duration_hours,
                universe_mode="dynamic_liquid",
                max_parallel_legs=int(current_app.config.get("VAULT_MAX_PARALLEL_LEGS", 1) or 1),
                allow_leverage_experiment=True,
                mode=_high_upside_market_data_mode(provider),
            )
            optimizer_config.max_parameter_sets = max(1, int(max_parameter_sets or 1))
            optimizer_config.auto_deploy_top_n = 0
            optimizer_config.high_upside_profile = True
            optimizer_config.provider = provider
            optimizer_config.market_structure_features_enabled = True
            optimizer_config.market_structure_provider = str(current_app.config.get("MARKET_STRUCTURE_PROVIDER", "existing") or "existing")
            optimizer_config.pair_screening_enabled = bool(current_app.config.get("PAIR_SCREENING_ENABLED", False))
            optimizer_config.pair_trading_enabled = bool(current_app.config.get("PAIR_TRADING_ENABLED", False))
            optimizer_config.runtime_deadline_monotonic = max(0.0, deadline_monotonic - 1.0)
            try:
                run_result = optimizer.run(optimizer_config)
                run_id = int(run_result.get("optimizer_run_id") or 0)
                accepted = _high_upside_rankings_for_run(run_id, provider=provider)
                sweep = {
                    "provider": provider,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "profile": profile,
                    "objective": objective,
                    "target_roi_pct": target_roi_pct,
                    "optimizer_run_id": run_id,
                    "accepted_count": len(accepted),
                    "ranking_count": int(run_result.get("ranking_count", 0) or 0),
                    "rejection_breakdown": dict(run_result.get("rejection_breakdown", {}) or {}),
                    "optimizer_runtime": dict(run_result.get("optimizer_runtime", {}) or {}),
                    "live_orders_created": False,
                }
                accepted_rankings.extend(accepted)
                for reason, count in sweep["rejection_breakdown"].items():
                    rejection_breakdown[str(reason)] = rejection_breakdown.get(str(reason), 0) + int(count or 0)
            except Exception as exc:  # noqa: BLE001
                sweep = {
                    "provider": provider,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "profile": profile,
                    "objective": objective,
                    "target_roi_pct": target_roi_pct,
                    "accepted_count": 0,
                    "ranking_count": 0,
                    "error": str(exc),
                    "failed_phase": "running_backtests",
                    "live_orders_created": False,
                }
                rejection_breakdown["optimizer_error"] = rejection_breakdown.get("optimizer_error", 0) + 1
            sweep_rows.append(sweep)
            result["sweeps_used"] = int(result["sweeps_used"]) + 1

    result["status"] = "completed"
    result["sweeps"] = sweep_rows
    result["accepted_rankings"] = _dedupe_rankings(accepted_rankings)
    result["rejection_breakdown"] = rejection_breakdown
    if not result["accepted_rankings"]:
        result["blockers"] = ["accepted_ranking_missing"]
    return result


def _high_upside_timeout_payload(
    *,
    user_id: int,
    provider_scope: str | None,
    providers: list[str],
    symbols: list[str],
    timeframes: list[str],
    profiles: list[str],
    timeout_seconds: float,
    started_at: float,
    failed_phase: str,
    objective: str = "risk_adjusted",
    target_roi_pct: float | None = None,
) -> dict[str, object]:
    ml_readiness = _ml_suite_readiness("1h")
    selected_objective = _ml_objective(objective)
    target_roi = _ml_extreme_target_roi_pct(target_roi_pct, objective=selected_objective, horizon="1h")
    return {
        "ready": False,
        "research_only": True,
        "live_orders_created": False,
        "submitted": False,
        "timed_out": True,
        "failed_phase": failed_phase,
        "blockers": ["high_upside_discovery_timeout"],
        "provider_scope": provider_scope or current_app.config.get("HIGH_UPSIDE_PROVIDER_SCOPE", "multi_provider"),
        "objective": selected_objective,
        "target_roi_pct": target_roi,
        "target_roi_policy": _ml_extreme_target_policy(selected_objective, target_roi),
        "providers_requested": [_normalize_provider(item) for item in providers],
        "symbols": _normalize_symbols(symbols),
        "timeframes": _high_upside_timeframes(timeframes),
        "profiles": _high_upside_profiles(profiles),
        "accepted_ranking_ids": [],
        "accepted_rankings": [],
        "high_upside_live_readiness": _high_upside_live_readiness(
            SimpleNamespace(high_upside_profile=True, lock_duration_hours=1, profile="aggressive_1h"),
            {},
        ),
        "ml_readiness": ml_readiness,
        "extreme_upside_readiness": _ml_extreme_upside_readiness("1h", selected_objective),
        "ml_models": ml_readiness.get("families", {}),
        "ml_blockers": ml_readiness.get("blockers", []),
        "operator_next_steps": [
            "Narrow providers, symbols, timeframes, or max sweeps before increasing HIGH_UPSIDE_COMMAND_TIMEOUT_SECONDS.",
        ],
        "runtime": _high_upside_runtime(started_at, timeout_seconds, timed_out=True),
        "user_id": user_id,
    }


def _high_upside_timeout_seconds(value: float | None) -> float:
    if value is not None:
        return max(1.0, float(value))
    return max(1.0, _float(current_app.config.get("HIGH_UPSIDE_COMMAND_TIMEOUT_SECONDS"), 300.0))


def _high_upside_positive_int(value: int | None, config_key: str, default: int) -> int:
    raw = value if value is not None else current_app.config.get(config_key, default)
    try:
        return max(1, int(raw or default))
    except (TypeError, ValueError):
        return max(1, int(default))


def _high_upside_timeframes(values: list[str]) -> list[str]:
    configured = current_app.config.get("HIGH_UPSIDE_TIMEFRAMES", ["5m", "15m", "1h"])
    source = values or (list(configured) if isinstance(configured, (list, tuple)) else str(configured).split(","))
    return [item for item in dict.fromkeys(str(value).strip() for value in source if str(value).strip())]


def _high_upside_profiles(values: list[str]) -> list[str]:
    source = values or ["aggressive_1h"]
    return [item for item in dict.fromkeys(str(value).strip() for value in source if str(value).strip())]


def _normalize_symbols(values: list[str]) -> list[str]:
    return [item for item in dict.fromkeys(str(value).strip().upper() for value in values if str(value).strip())]


def _dedupe_symbols_preserve_case(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        symbol = str(value or "").strip()
        key = symbol.upper()
        if not symbol or key in seen:
            continue
        seen.add(key)
        result.append(symbol)
    return result


def _normalize_provider(provider: object) -> str:
    return str(provider or "").strip().lower().replace("-", "_")


def _high_upside_provider_sequence(user_id: int, provider_scope: str, providers: list[str]) -> list[str]:
    explicit = [_normalize_provider(item) for item in providers if _normalize_provider(item)]
    if explicit:
        return list(dict.fromkeys(explicit))
    if provider_scope in {"hyperliquid", "binance", "kucoin", "dydx", "uniswap"}:
        return [provider_scope]
    active = [
        _normalize_provider(connection.provider)
        for connection in TradingConnection.query.filter_by(user_id=user_id, is_active=True, verification_status="verified").all()
    ]
    if provider_scope == "active_provider":
        return list(dict.fromkeys(active))
    return list(dict.fromkeys([*active, "hyperliquid", "binance", "kucoin", "dydx", "uniswap"]))


def _high_upside_provider_market_data_supported(provider: str) -> bool:
    # KuCoin currently reuses the normalized public market-data feed while
    # provider-specific costs/collateral are carried through ranking metadata.
    return _normalize_provider(provider) in {"hyperliquid", "kucoin"}


def _high_upside_market_data_mode(provider: str) -> str:
    return "live" if _normalize_provider(provider) == "hyperliquid" else "live"


def _high_upside_candidate_symbols(provider: str, requested_symbols: list[str], timeframe: str, symbol_cap: int) -> list[str]:
    plan = _high_upside_candidate_plan(
        provider=provider,
        requested_symbols=requested_symbols,
        timeframe=timeframe,
        symbol_cap=symbol_cap,
        deadline_monotonic=0.0,
        timeout_seconds=0.0,
        progress_state={},
    )
    return list(plan.get("symbols", []) or [])


def _high_upside_candidate_plan(
    *,
    provider: str,
    requested_symbols: list[str],
    timeframe: str,
    symbol_cap: int,
    deadline_monotonic: float,
    timeout_seconds: float,
    progress_state: dict[str, object],
) -> dict[str, object]:
    """Build a bounded high-upside symbol set without broad candle/book fan-out."""

    provider = _normalize_provider(provider)
    market_mode = _high_upside_market_data_mode(provider)
    cap = max(1, int(symbol_cap or 1))
    stage1_max = max(cap, int(current_app.config.get("HIGH_UPSIDE_STAGE1_MAX_MIDS", 80) or 80))
    configured = _normalize_symbols(list(current_app.config.get("ALLOWED_SYMBOLS", []) or []))
    diagnostics: dict[str, object] = {
        "provider": provider,
        "timeframe": timeframe,
        "source": "explicit_symbols" if requested_symbols else "hyperliquid_mids_preselection",
        "symbol_cap": cap,
        "stage1_max_mids": stage1_max,
        "requested_symbols": _normalize_symbols(requested_symbols),
        "configured_seed_symbols": configured,
        "raw_mid_count": 0,
        "stage1_candidate_count": 0,
        "selected_count": 0,
        "skipped_symbols": [],
        "skip_breakdown": {},
        "omitted_symbols": [],
        "rate_limit_backoff_seconds": float(current_app.config.get("HIGH_UPSIDE_RATE_LIMIT_BACKOFF_SECONDS", 300.0) or 300.0),
    }

    if requested_symbols:
        rows = _high_upside_stage1_rows(_dedupe_symbols_preserve_case(requested_symbols), configured, diagnostics)
        selected = [row["symbol"] for row in rows[:cap]]
        diagnostics["stage1_candidate_count"] = len(rows)
        diagnostics["selected_count"] = len(selected)
        diagnostics["symbols"] = selected
        diagnostics["markets"] = rows[:cap]
        diagnostics["omitted_symbols"] = [row["symbol"] for row in rows[cap:]]
        return diagnostics

    raw_symbols: list[str] = []
    try:
        if deadline_monotonic and _high_upside_deadline_reached(deadline_monotonic):
            progress_state["failed_phase"] = "loading_market_universe"
            raise _HighUpsideDiscoveryTimeout(timeout_seconds, "loading_market_universe")
        if provider == "hyperliquid":
            universe = _hyperliquid_tradable_perp_universe(mode=market_mode)
            diagnostics["raw_mid_count"] = int(universe.get("total_mids", 0) or 0)
            diagnostics["tradable_pair_count"] = int(universe.get("tradable_count", 0) or 0)
            diagnostics["universe_skipped_count"] = len(list(universe.get("skipped", []) or []))
            diagnostics["universe_skipped_sample"] = list(universe.get("skipped", []) or [])[:20]
            diagnostics["skip_breakdown"] = _skip_breakdown(list(universe.get("skipped", []) or []))
            raw_symbols = [str(row.get("venue_symbol") or row.get("symbol") or "").strip() for row in list(universe.get("markets", []) or [])]
        else:
            market_data = get_service("market_data")
            mids = market_data.get_all_mids(market_mode)
            diagnostics["raw_mid_count"] = len(mids)
            raw_symbols = [str(symbol).strip() for symbol in mids if str(symbol).strip()]
    except _HighUpsideDiscoveryTimeout:
        raise
    except Exception as exc:  # noqa: BLE001
        diagnostics["source"] = "configured_fallback_after_mids_error"
        diagnostics["mids_error"] = _high_upside_sanitize_error(exc)
        raw_symbols = []

    rows = _high_upside_stage1_rows([*configured, *raw_symbols], configured, diagnostics)
    rows.sort(key=lambda row: (float(row["score"]), str(row["symbol"])), reverse=True)
    stage1_rows = rows[:stage1_max]
    selected = [row["symbol"] for row in stage1_rows[:cap]]
    diagnostics["stage1_candidate_count"] = len(stage1_rows)
    diagnostics["selected_count"] = len(selected)
    diagnostics["symbols"] = selected
    diagnostics["markets"] = stage1_rows[:cap]
    diagnostics["omitted_symbols"] = [row["symbol"] for row in stage1_rows[cap: min(len(stage1_rows), cap + 20)]]
    diagnostics["stage1_top"] = stage1_rows[: min(len(stage1_rows), 20)]
    if not selected and configured:
        diagnostics["source"] = "configured_fallback_after_empty_preselection"
        diagnostics["symbols"] = configured[:cap]
        diagnostics["selected_count"] = len(configured[:cap])
    return diagnostics


def _high_upside_stage1_rows(
    symbols: list[str],
    configured_symbols: list[str],
    diagnostics: dict[str, object],
) -> list[dict[str, object]]:
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    configured_rank = {symbol: len(configured_symbols) - index for index, symbol in enumerate(configured_symbols)}
    for raw_symbol in symbols:
        symbol = str(raw_symbol or "").strip()
        app_symbol = symbol.upper()
        dedupe_key = app_symbol
        if not symbol or dedupe_key in seen:
            continue
        reason = _high_upside_symbol_skip_reason(symbol)
        if reason:
            _high_upside_record_symbol_skip(diagnostics, symbol, reason)
            continue
        seen.add(dedupe_key)
        rows.append(
            {
                "symbol": symbol,
                "app_symbol": app_symbol,
                "venue_symbol": symbol,
                "score": float(configured_rank.get(app_symbol, 0)) + _high_upside_symbol_priority(app_symbol),
                "configured_seed": app_symbol in configured_rank,
            }
        )
    return rows


def _high_upside_symbol_skip_reason(symbol: str) -> str:
    raw = str(symbol or "").strip()
    normalized = raw.upper()
    if not normalized:
        return "empty_symbol"
    if normalized.startswith("@"):
        return "numeric_venue_symbol_unsupported"
    if any(char in normalized for char in {"/", "-", ":", " "}):
        return "mapped_pair_symbol_unsupported"
    if len(normalized) > 14:
        return "symbol_too_long"
    if normalized in {"USD", "USDC", "USDT", "USDE", "USDT0", "USDH"}:
        return "quote_or_stable_symbol"
    if not normalized.replace("_", "").isalnum():
        return "unsupported_symbol_characters"
    if normalized.isdigit():
        return "numeric_symbol_unsupported"
    return ""


def _high_upside_symbol_priority(symbol: str) -> float:
    majors = {"BTC": 50.0, "ETH": 45.0, "SOL": 35.0, "XRP": 25.0}
    if symbol in majors:
        return majors[symbol]
    if len(symbol) <= 5:
        return 10.0
    if len(symbol) <= 8:
        return 5.0
    return 1.0


def _high_upside_record_symbol_skip(diagnostics: dict[str, object], symbol: str, reason: str) -> None:
    skipped = diagnostics.setdefault("skipped_symbols", [])
    if isinstance(skipped, list) and len(skipped) < 50:
        skipped.append({"symbol": symbol, "reason": reason})
    breakdown = diagnostics.setdefault("skip_breakdown", {})
    if isinstance(breakdown, dict):
        breakdown[reason] = int(breakdown.get(reason, 0) or 0) + 1


def _hyperliquid_tradable_perp_universe(*, mode: str = "live") -> dict[str, object]:
    """Return exact Hyperliquid perp symbols that can be queried/submitted."""

    skipped: list[dict[str, object]] = []
    markets: list[dict[str, object]] = []
    try:
        market_data = get_service("market_data")
        client = getattr(market_data, "client", None)
        mids = dict(market_data.get_all_mids(mode) or {})
        meta: dict[str, object] = {}
        contexts: list[dict[str, object]] = []
        getter = getattr(client, "get_perp_meta_and_asset_contexts", None)
        if callable(getter):
            raw_meta, raw_contexts = getter(mode)
            meta = raw_meta if isinstance(raw_meta, dict) else {}
            contexts = [dict(item) for item in raw_contexts if isinstance(item, dict)] if isinstance(raw_contexts, list) else []
        universe = meta.get("universe") if isinstance(meta, dict) else None
        if not isinstance(universe, list) or not universe:
            universe = [{"name": symbol} for symbol in mids]
        else:
            known = {str(item.get("name") or item.get("coin") or item.get("symbol") or "").strip().upper() for item in universe if isinstance(item, dict)}
            universe = [
                *universe,
                *[{"name": symbol} for symbol in mids if str(symbol).strip().upper() not in known],
            ]
    except Exception as exc:  # noqa: BLE001
        return {
            "ready": False,
            "markets": [],
            "skipped": [{"symbol": "", "reason": "hyperliquid_universe_unavailable", "error": _high_upside_sanitize_error(exc)}],
            "total_mids": 0,
            "tradable_count": 0,
            "blockers": ["hyperliquid_universe_unavailable"],
        }

    seen: set[str] = set()
    for index, item in enumerate(universe):
        if not isinstance(item, dict):
            continue
        venue_symbol = str(item.get("name") or item.get("coin") or item.get("symbol") or "").strip()
        app_symbol = venue_symbol.upper()
        context = contexts[index] if index < len(contexts) and isinstance(contexts[index], dict) else {}
        reason = _hyperliquid_market_skip_reason(venue_symbol, item, context)
        mid = _hyperliquid_mid_for_symbol(venue_symbol, mids)
        if not reason and mid <= 0:
            reason = "mid_unavailable"
        if reason:
            skipped.append({"symbol": venue_symbol, "reason": reason})
            continue
        key = app_symbol
        if key in seen:
            skipped.append({"symbol": venue_symbol, "reason": "duplicate_symbol"})
            continue
        seen.add(key)
        markets.append(
            {
                "symbol": venue_symbol,
                "app_symbol": app_symbol,
                "venue_symbol": venue_symbol,
                "mid": mid,
                "max_leverage": _float(item.get("maxLeverage") or item.get("max_leverage") or context.get("maxLeverage"), 0.0),
                "open_interest": _float(context.get("openInterest") or context.get("open_interest"), 0.0),
                "source": "hyperliquid_meta",
            }
        )
    markets.sort(key=lambda row: (_float(row.get("open_interest"), 0.0), _float(row.get("mid"), 0.0), str(row.get("app_symbol"))), reverse=True)
    return {
        "ready": bool(markets),
        "markets": markets,
        "skipped": skipped,
        "total_mids": len(mids),
        "tradable_count": len(markets),
        "blockers": [] if markets else ["hyperliquid_tradable_pairs_missing"],
    }


def _hyperliquid_market_skip_reason(symbol: str, meta: dict[str, object], context: dict[str, object]) -> str:
    raw = str(symbol or "").strip()
    upper = raw.upper()
    if not raw:
        return "empty_symbol"
    if upper.startswith("@") or upper.isdigit():
        return "numeric_venue_symbol_unsupported"
    if any(char in raw for char in {"/", ":", " "}):
        return "mapped_pair_symbol_unsupported"
    if upper in {"USD", "USDC", "USDT", "USDE", "USDT0", "USDH"}:
        return "quote_or_stable_symbol"
    if bool(meta.get("isDelisted") or meta.get("delisted") or context.get("isDelisted") or context.get("delisted")):
        return "delisted"
    if not raw.replace("_", "").isalnum():
        return "unsupported_symbol_characters"
    return ""


def _hyperliquid_mid_for_symbol(symbol: str, mids: dict[str, object]) -> float:
    raw = str(symbol or "").strip()
    if raw in mids:
        return _float(mids.get(raw), 0.0)
    upper = raw.upper()
    if upper in mids:
        return _float(mids.get(upper), 0.0)
    for key, value in mids.items():
        if str(key).upper() == upper:
            return _float(value, 0.0)
    return 0.0


def _high_upside_market_data_policy() -> dict[str, object]:
    rate = _float(current_app.config.get("HIGH_UPSIDE_MARKET_DATA_RATE_LIMIT_PER_SECOND"), 4.0)
    return {
        "hyperliquid_first": True,
        "stage1_uses_mids_only": True,
        "unsupported_providers_skipped": True,
        "rate_limit_per_second": rate,
        "rate_limit_backoff_seconds": _float(current_app.config.get("HIGH_UPSIDE_RATE_LIMIT_BACKOFF_SECONDS"), 300.0),
        "stage1_max_mids": int(current_app.config.get("HIGH_UPSIDE_STAGE1_MAX_MIDS", 80) or 80),
    }


def _ml_history_candidate_plan(
    *,
    provider: str,
    requested_symbols: list[str],
    timeframe: str,
    symbol_cap: int,
    all_pairs: bool,
) -> dict[str, object]:
    provider_key = _normalize_provider(provider)
    if all_pairs and provider_key == "hyperliquid":
        universe = _hyperliquid_tradable_perp_universe(mode=_high_upside_market_data_mode(provider_key))
        markets = list(universe.get("markets", []) or [])
        selected = markets[: max(1, int(symbol_cap or len(markets) or 1))]
        return {
            "provider": provider_key,
            "timeframe": timeframe,
            "source": "hyperliquid_all_tradable_perps",
            "all_pairs": True,
            "symbol_cap": symbol_cap,
            "total_tradable_pairs": int(universe.get("tradable_count", len(markets)) or 0),
            "selected_count": len(selected),
            "symbols": [str(row.get("symbol") or "") for row in selected],
            "markets": selected,
            "skipped_symbols": list(universe.get("skipped", []) or [])[:100],
            "skip_breakdown": _skip_breakdown(list(universe.get("skipped", []) or [])),
            "blockers": list(universe.get("blockers", []) or []),
        }
    return _high_upside_candidate_plan(
        provider=provider_key,
        requested_symbols=requested_symbols,
        timeframe=timeframe,
        symbol_cap=symbol_cap,
        deadline_monotonic=0.0,
        timeout_seconds=0.0,
        progress_state={},
    )


def _skip_breakdown(rows: list[dict[str, object]]) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason") or "unknown")
        breakdown[reason] = breakdown.get(reason, 0) + 1
    return breakdown


def _ml_history_backfill_payload(
    *,
    provider_scope: str,
    providers: list[str],
    symbols: list[str],
    timeframes: list[str],
    max_symbols: int | None,
    lookback_days: int | None,
    user_id: int,
    all_pairs: bool = False,
) -> dict[str, object]:
    started_at = time.monotonic()
    blockers: list[str] = []
    if not bool(current_app.config.get("ML_HISTORY_BACKFILL_ENABLED", False)):
        blockers.append("ML_HISTORY_BACKFILL_ENABLED=false")
    all_pairs_requested = bool(all_pairs and not symbols)
    max_symbol_cap = 1_000 if all_pairs_requested else 250
    if all_pairs_requested and max_symbols is None:
        symbol_cap = max_symbol_cap
    else:
        symbol_cap = max(1, min(_high_upside_positive_int(max_symbols, "ML_HISTORY_BACKFILL_MAX_SYMBOLS", 50), max_symbol_cap))
    days = max(1, min(int(lookback_days or current_app.config.get("ML_HISTORY_BACKFILL_LOOKBACK_DAYS", 90) or 90), 365))
    selected_timeframes = _high_upside_timeframes(timeframes or ["5m", "15m", "1h"])
    provider_sequence = _ml_history_provider_sequence(user_id, provider_scope, providers)
    requested_symbols = _dedupe_symbols_preserve_case(symbols)
    if not provider_sequence:
        blockers.append("provider_universe_empty")
    if blockers:
        return {
            "ready": False,
            "research_only": True,
            "submitted": False,
            "live_orders_created": False,
            "blockers": blockers,
            "provider_scope": provider_scope,
            "providers": provider_sequence,
            "timeframes": selected_timeframes,
            "all_pairs": bool(all_pairs),
            "market_history": {"created_rows": 0, "status": "blocked"},
            "provider_skip_diagnostics": [],
            "runtime": {"elapsed_seconds": max(time.monotonic() - started_at, 0.0)},
        }

    provider_results: list[dict[str, object]] = []
    created_rows = 0
    market_data = get_service("market_data")
    for provider in provider_sequence:
        provider = _normalize_provider(provider)
        provider_result: dict[str, object] = {
            "provider": provider,
            "market_data_supported": _high_upside_provider_market_data_supported(provider),
            "symbols": [],
            "rows_created": 0,
            "skipped": [],
            "errors": [],
        }
        if not bool(provider_result["market_data_supported"]):
            provider_result["skipped"].append({"provider": provider, "reason": "provider_market_data_unavailable"})
            provider_results.append(provider_result)
            continue
        plan = _ml_history_candidate_plan(
            provider=provider,
            requested_symbols=requested_symbols,
            timeframe=selected_timeframes[0] if selected_timeframes else "1h",
            symbol_cap=symbol_cap,
            all_pairs=bool(all_pairs and not requested_symbols),
        )
        selected_markets = list(plan.get("markets", []) or [])[:symbol_cap]
        provider_result["symbols"] = [str(row.get("symbol") or "") for row in selected_markets]
        provider_result["markets"] = selected_markets
        provider_result["stage1_diagnostics"] = plan
        all_pairs_hyperliquid = bool(all_pairs and not requested_symbols and provider == "hyperliquid")
        order_book_enabled = (
            _config_flag("ML_HISTORY_BACKFILL_ALL_PAIRS_ORDER_BOOK_ENABLED", False)
            if all_pairs_hyperliquid
            else _config_flag("ML_HISTORY_BACKFILL_ORDER_BOOK_ENABLED", True)
        )
        request_sleep_seconds = max(0.0, _float(current_app.config.get("ML_HISTORY_BACKFILL_REQUEST_SLEEP_SECONDS"), 0.0))
        provider_result["order_book_enabled"] = bool(order_book_enabled)
        provider_result["request_sleep_seconds"] = request_sleep_seconds
        for market in selected_markets:
            symbol = str(market.get("symbol") or market.get("venue_symbol") or "").strip()
            venue_symbol = str(market.get("venue_symbol") or symbol).strip()
            app_symbol = str(market.get("app_symbol") or symbol.upper()).strip().upper()
            for timeframe in selected_timeframes:
                limit = _ml_history_candle_limit(timeframe, days)
                try:
                    candles = market_data.get_candles(venue_symbol, timeframe, mode=_high_upside_market_data_mode(provider), limit=limit)
                    if request_sleep_seconds > 0:
                        time.sleep(request_sleep_seconds)
                    book: dict[str, object] = {}
                    book_error = "order_book_backfill_disabled"
                    if order_book_enabled:
                        book_error = ""
                        try:
                            book = market_data.get_order_book(venue_symbol, _high_upside_market_data_mode(provider))
                            if request_sleep_seconds > 0:
                                time.sleep(request_sleep_seconds)
                        except Exception as exc:  # noqa: BLE001
                            book_error = _high_upside_sanitize_error(exc)
                    metrics = _ml_history_order_book_metrics(book, candles)
                    window_start, window_end = _ml_history_window_bounds(candles)
                    row = MLMarketHistory(
                        provider=provider,
                        symbol=app_symbol,
                        timeframe=timeframe,
                        mode=_high_upside_market_data_mode(provider),
                        source="ml_history_backfill",
                        status="ok" if candles else "empty",
                        error="" if candles else "candles_empty",
                        liquidity_usd=metrics["liquidity_usd"],
                        spread_bps=metrics["spread_bps"],
                        funding_rate=0.0,
                        window_start=window_start,
                        window_end=window_end,
                    )
                    row.candles = candles
                    row.order_book = book
                    row.funding = {}
                    row.diagnostics = {
                        "provider": provider,
                        "symbol": app_symbol,
                        "venue_symbol": venue_symbol,
                        "timeframe": timeframe,
                        "lookback_days": days,
                        "requested_limit": limit,
                        "order_book_enabled": bool(order_book_enabled),
                        "book_error": book_error,
                        "liquidation_buffer_pct": 0.0,
                    }
                    db.session.add(row)
                    db.session.flush()
                    created_rows += 1
                    provider_result["rows_created"] = int(provider_result["rows_created"]) + 1
                except Exception as exc:  # noqa: BLE001
                    error = _high_upside_sanitize_error(exc)
                    if _high_upside_error_is_rate_limited(error):
                        _high_upside_record_rate_limit_backoff(provider, timeframe)
                    row = MLMarketHistory(
                        provider=provider,
                        symbol=app_symbol,
                        timeframe=timeframe,
                        mode=_high_upside_market_data_mode(provider),
                        source="ml_history_backfill",
                        status="error",
                        error=error,
                    )
                    row.diagnostics = {"provider": provider, "symbol": app_symbol, "venue_symbol": venue_symbol, "timeframe": timeframe, "error": error}
                    db.session.add(row)
                    db.session.flush()
                    provider_result["errors"].append({"symbol": symbol, "timeframe": timeframe, "error": error})
        provider_results.append(provider_result)
    audit = AuditLog(
        category="ml",
        action="ml_history_backfill",
        message="ML market-history backfill completed.",
        user_id=user_id,
    )
    audit.details = _json_safe(
        {
            "provider_scope": provider_scope,
            "providers": provider_sequence,
            "timeframes": selected_timeframes,
            "created_rows": created_rows,
            "lookback_days": days,
            "max_symbols": symbol_cap,
            "all_pairs": bool(all_pairs),
        }
    )
    db.session.add(audit)
    db.session.commit()
    return {
        "ready": created_rows > 0,
        "research_only": True,
        "submitted": False,
        "live_orders_created": False,
        "blockers": [] if created_rows > 0 else ["market_history_rows_missing"],
        "provider_scope": provider_scope,
        "providers": provider_results,
        "timeframes": selected_timeframes,
        "all_pairs": bool(all_pairs),
        "market_history": {
            "created_rows": created_rows,
            "lookback_days": days,
            "max_symbols": symbol_cap,
            "audit_log_id": audit.id,
        },
        "provider_skip_diagnostics": [
            row for provider_result in provider_results for row in list(provider_result.get("skipped", []) or [])
        ],
        "runtime": {"elapsed_seconds": max(time.monotonic() - started_at, 0.0)},
    }


def _ml_history_provider_sequence(user_id: int, provider_scope: str, providers: list[str]) -> list[str]:
    explicit = [_normalize_provider(item) for item in providers if _normalize_provider(item)]
    if explicit:
        return list(dict.fromkeys(explicit))
    scope = _normalize_provider(provider_scope)
    all_providers = ["hyperliquid", "binance", "kucoin", "dydx", "uniswap"]
    if scope == "all":
        active = [
            _normalize_provider(connection.provider)
            for connection in TradingConnection.query.filter_by(user_id=user_id, is_active=True, verification_status="verified").all()
        ]
        return list(dict.fromkeys([*active, *all_providers]))
    return _high_upside_provider_sequence(user_id, scope, [])


def _ml_history_candle_limit(timeframe: str, lookback_days: int) -> int:
    minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}.get(str(timeframe), 60)
    requested = max(20, min(int((lookback_days * 24 * 60) / minutes), 5_000))
    max_per_request = max(
        20,
        min(_high_upside_positive_int(None, "ML_HISTORY_BACKFILL_MAX_CANDLES_PER_REQUEST", 5_000), 5_000),
    )
    return min(requested, max_per_request)


def _ml_history_window_bounds(candles: list[dict[str, object]]) -> tuple[datetime | None, datetime | None]:
    timestamps = [_float(row.get("timestamp"), 0.0) for row in candles if isinstance(row, dict) and row.get("timestamp") is not None]
    if not timestamps:
        return None, None
    return _ml_history_datetime(min(timestamps)), _ml_history_datetime(max(timestamps))


def _ml_history_datetime(timestamp: float) -> datetime | None:
    if timestamp <= 0:
        return None
    seconds = timestamp / 1000.0 if timestamp > 10_000_000_000 else timestamp
    try:
        return datetime.utcfromtimestamp(seconds)
    except (OverflowError, OSError, ValueError):
        return None


def _ml_history_order_book_metrics(book: dict[str, object], candles: list[dict[str, object]]) -> dict[str, float]:
    mid = _float((candles[-1] if candles else {}).get("close"), 0.0)
    levels = book.get("levels") if isinstance(book, dict) else None
    spread_bps = 0.0
    liquidity = 0.0
    if isinstance(levels, list) and len(levels) >= 2:
        bid_side = levels[0] if isinstance(levels[0], list) else []
        ask_side = levels[1] if isinstance(levels[1], list) else []
        bid = _ml_history_level_price(bid_side[0]) if bid_side else 0.0
        ask = _ml_history_level_price(ask_side[0]) if ask_side else 0.0
        if bid > 0 and ask > 0 and mid > 0:
            spread_bps = ((ask - bid) / mid) * 10_000
        for side in (bid_side, ask_side):
            for level in side[:10]:
                price = _ml_history_level_price(level)
                size = _ml_history_level_size(level)
                liquidity += max(price, 0.0) * max(size, 0.0)
    return {"spread_bps": max(0.0, spread_bps), "liquidity_usd": max(0.0, liquidity)}


def _ml_history_level_price(level: object) -> float:
    if isinstance(level, dict):
        return _float(level.get("px", level.get("price")), 0.0)
    if isinstance(level, (list, tuple)) and level:
        return _float(level[0], 0.0)
    return 0.0


def _ml_history_level_size(level: object) -> float:
    if isinstance(level, dict):
        return _float(level.get("sz", level.get("size")), 0.0)
    if isinstance(level, (list, tuple)) and len(level) > 1:
        return _float(level[1], 0.0)
    return 0.0


def _ml_feedback_sync_payload(*, horizon: str, provider: str = "all", sources: list[str], max_rows: int | None) -> dict[str, object]:
    started_at = time.monotonic()
    blockers: list[str] = []
    if not bool(current_app.config.get("ML_FEEDBACK_SYNC_ENABLED", False)):
        blockers.append("ML_FEEDBACK_SYNC_ENABLED=false")
    source_list = _ml_feedback_sources(sources)
    provider_filter = _normalize_provider(provider)
    if provider_filter in {"", "all", "global_all"}:
        provider_filter = "all"
    row_cap = max(1, min(int(max_rows or current_app.config.get("ML_FEEDBACK_SYNC_MAX_ROWS", 5000) or 5000), 50_000))
    horizon_filter = str(horizon or "all").strip().lower() or "all"
    if not source_list:
        blockers.append("ml_feedback_sources_empty")
    if blockers:
        return {
            "ready": False,
            "research_only": True,
            "submitted": False,
            "live_orders_created": False,
            "horizon": horizon_filter,
            "provider": provider_filter,
            "sources": source_list,
            "max_rows": row_cap,
            "feedback_sync": {"created_events": 0, "skipped_existing": 0, "status": "blocked"},
            "blockers": blockers,
            "runtime": {"elapsed_seconds": max(time.monotonic() - started_at, 0.0)},
        }

    ranker = OnlineRanker(current_app.config)
    feature_factory = MLFeatureFactory(current_app.config)
    created = 0
    skipped_existing = 0
    skipped_invalid = 0
    source_counts: dict[str, int] = {}
    diagnostics: list[dict[str, object]] = []

    def provider_allowed(value: object) -> bool:
        if provider_filter == "all":
            return True
        return normalize_exchange_provider(value) == provider_filter

    def can_add(source: str, source_id: object, event_horizon: str) -> bool:
        nonlocal skipped_existing
        if created >= row_cap:
            return False
        existing = MLTrainingEvent.query.filter_by(
            source=f"feedback_sync:{source}",
            source_id=str(source_id),
            horizon=event_horizon,
        ).first()
        if existing is not None:
            skipped_existing += 1
            return False
        return True

    def add_feedback(
        source: str,
        source_id: object,
        event_horizon: str,
        payload: dict[str, object],
        outcome: float,
        *,
        mode: str = "paper",
        metadata: dict[str, object] | None = None,
    ) -> None:
        nonlocal created, skipped_invalid
        event_horizon = str(event_horizon or "1h").lower()
        if horizon_filter != "all" and event_horizon != horizon_filter:
            return
        if not payload:
            skipped_invalid += 1
            return
        if not can_add(source, source_id, event_horizon):
            return
        result = ranker.update(
            extract_features(payload),
            outcome,
            horizon=event_horizon,
            source=f"feedback_sync:{source}",
            source_id=source_id,
            mode=str(mode or "paper").lower(),
            metadata={
                "feedback_sync": True,
                "source": source,
                "source_id": str(source_id),
                **dict(metadata or {}),
            },
        )
        if bool(result.get("updated", False)) or bool(result.get("quarantined", False)):
            created += 1
            source_counts[source] = source_counts.get(source, 0) + 1
        else:
            skipped_invalid += 1
            if len(diagnostics) < 25:
                diagnostics.append({"source": source, "source_id": source_id, "reason": result.get("reason", "not_updated")})

    if "rankings" in source_list:
        for ranking in StrategyRanking.query.order_by(StrategyRanking.created_at.asc(), StrategyRanking.id.asc()).limit(row_cap).all():
            if not provider_allowed(getattr(ranking, "provider", "global")):
                continue
            payload = _ml_feedback_ranking_payload(ranking)
            event_horizon = horizon_from_duration(ranking.lock_duration_hours or 1)
            add_feedback(
                "ranking",
                ranking.id,
                event_horizon,
                payload,
                outcome_from_result(payload),
                mode=payload.get("mode", "paper"),
                metadata={
                    "provider": normalize_exchange_provider(getattr(ranking, "provider", "global")),
                    "symbol": ranking.symbol,
                    "timeframe": ranking.timeframe,
                    "rejected": bool(ranking.rejected),
                    "rejection_reason": ranking.rejection_reason or "",
                },
            )

    if "backtests" in source_list:
        for run in BacktestRun.query.order_by(BacktestRun.created_at.asc(), BacktestRun.id.asc()).limit(row_cap).all():
            result = run.result if isinstance(run.result, dict) else {}
            if not result:
                continue
            run_provider = normalize_exchange_provider(
                result.get("provider")
                or (run.parameters or {}).get("provider")
                or (run.parameters or {}).get("execution_venue")
                or "global"
            )
            if not provider_allowed(run_provider):
                continue
            payload = feature_factory.build(
                symbol=run.symbol,
                timeframe=run.timeframe,
                optimizer_context={**provider_feature_context(run_provider), "strategy_name": run.strategy_name, **dict(run.parameters or {})},
                backtest_result=result,
                trade_outcomes=result.get("trades") if isinstance(result.get("trades"), list) else [],
            )
            event_horizon = horizon_from_duration(result.get("lock_duration_hours") or run.parameters.get("lock_duration_hours") or 1)
            add_feedback(
                "backtest",
                run.id,
                event_horizon,
                payload,
                outcome_from_result(result),
                mode="paper",
                metadata={"provider": run_provider, "symbol": run.symbol, "timeframe": run.timeframe, "strategy_name": run.strategy_name},
            )

    if "orders" in source_list:
        for order in Order.query.order_by(Order.created_at.asc(), Order.id.asc()).limit(row_cap).all():
            payload, outcome = _ml_feedback_order_payload(order)
            order_provider = normalize_exchange_provider(payload.get("provider") or payload.get("execution_venue") or "global")
            if not provider_allowed(order_provider):
                continue
            event_horizon = horizon_from_duration(order.details.get("lock_duration_hours") or 1)
            add_feedback(
                "order",
                order.id,
                event_horizon,
                payload,
                outcome,
                mode=order.mode,
                metadata={"provider": order_provider, "symbol": order.symbol, "status": order.status, "risk_status": order.risk_status},
            )

    if "vault_cycles" in source_list:
        for cycle in VaultCycle.query.order_by(VaultCycle.created_at.asc(), VaultCycle.id.asc()).limit(row_cap).all():
            payload = _ml_feedback_vault_cycle_payload(cycle)
            cycle_provider = normalize_exchange_provider(payload.get("provider") or payload.get("execution_venue") or "global")
            if not provider_allowed(cycle_provider):
                continue
            event_horizon = horizon_from_duration(cycle.lock_duration_hours or 1)
            add_feedback(
                "vault_cycle",
                cycle.id,
                event_horizon,
                payload,
                outcome_from_result(payload),
                mode=payload.get("mode", "paper"),
                metadata={"provider": cycle_provider, "status": cycle.status, "strategy_name": cycle.selected_strategy_name},
            )

    if "ops" in source_list:
        for audit in AuditLog.query.order_by(AuditLog.created_at.asc(), AuditLog.id.asc()).limit(row_cap).all():
            if "market_history" in source_list and created >= max(1, row_cap // 2):
                break
            payload = _ml_feedback_ops_audit_payload(audit)
            add_feedback(
                "ops_audit",
                audit.id,
                "ops",
                payload,
                _float(payload.get("ops_target"), 0.0),
                mode="paper",
                metadata={"action": audit.action, "category": audit.category},
            )
        for event in RiskEvent.query.order_by(RiskEvent.created_at.asc(), RiskEvent.id.asc()).limit(row_cap).all():
            if "market_history" in source_list and created >= max(1, row_cap // 2):
                break
            payload = _ml_feedback_risk_event_payload(event)
            add_feedback(
                "risk_event",
                event.id,
                "ops",
                payload,
                1.0,
                mode="paper",
                metadata={"rule_name": event.rule_name},
            )

    if "market_history" in source_list:
        for history in MLMarketHistory.query.order_by(MLMarketHistory.window_end.asc(), MLMarketHistory.id.asc()).limit(row_cap).all():
            if not provider_allowed(history.provider):
                continue
            payload, outcome = _ml_feedback_market_history_payload(feature_factory, history)
            add_feedback(
                "market_history",
                history.id,
                _ml_feedback_history_horizon(history.timeframe),
                payload,
                outcome,
                mode="paper",
                metadata={"provider": history.provider, "symbol": history.symbol, "timeframe": history.timeframe},
            )

    audit = AuditLog(
        category="ml",
        action="ml_feedback_sync",
        message="ML feedback sync completed.",
    )
    audit.details = _json_safe(
        {
            "horizon": horizon_filter,
            "provider": provider_filter,
            "sources": source_list,
            "created_events": created,
            "skipped_existing": skipped_existing,
            "skipped_invalid": skipped_invalid,
            "source_counts": source_counts,
        }
    )
    db.session.add(audit)
    db.session.commit()
    return {
        "ready": created > 0,
        "research_only": True,
        "submitted": False,
        "live_orders_created": False,
        "horizon": horizon_filter,
        "provider": provider_filter,
        "sources": source_list,
        "max_rows": row_cap,
        "feedback_sync": {
            "created_events": created,
            "skipped_existing": skipped_existing,
            "skipped_invalid": skipped_invalid,
            "source_counts": dict(sorted(source_counts.items())),
            "audit_log_id": audit.id,
            "status": "synced" if created > 0 else "no_new_feedback",
        },
        "diagnostics": diagnostics,
        "blockers": [] if created > 0 else ["ml_feedback_rows_missing"],
        "runtime": {"elapsed_seconds": max(time.monotonic() - started_at, 0.0)},
    }


def _ml_feedback_sources(sources: list[str]) -> list[str]:
    configured = current_app.config.get("ML_FEEDBACK_SYNC_SOURCES") or [
        "rankings",
        "backtests",
        "orders",
        "vault_cycles",
        "ops",
        "market_history",
    ]
    selected = [str(item or "").strip().lower() for item in (sources or []) if str(item or "").strip()]
    if not selected or "all" in selected:
        selected = [str(item or "").strip().lower() for item in configured if str(item or "").strip()]
    allowed = {"rankings", "backtests", "orders", "vault_cycles", "ops", "market_history"}
    return [item for item in dict.fromkeys(selected) if item in allowed]


def _ml_feedback_ranking_payload(ranking: StrategyRanking) -> dict[str, object]:
    explanation = ranking.ml_explanation if isinstance(ranking.ml_explanation, dict) else {}
    provider_key = normalize_exchange_provider(getattr(ranking, "provider", "global"))
    payload: dict[str, object] = {
        **provider_feature_context(provider_key),
        "strategy_name": ranking.strategy_name,
        "symbol": ranking.symbol,
        "venue_symbol": explanation.get("venue_symbol") or ranking.parameters.get("venue_symbol") or ranking.symbol,
        "timeframe": ranking.timeframe,
        "profile": ranking.profile,
        "optimizer_profile": ranking.profile,
        "mode": "paper",
        "net_return_after_costs": ranking.net_return_after_costs,
        "total_return": ranking.total_return,
        "recent_performance_score": ranking.recent_performance_score,
        "recent_1h_return": ranking.recent_1h_return,
        "max_drawdown": ranking.max_drawdown,
        "profit_factor": ranking.profit_factor,
        "sortino_like": ranking.sortino_like,
        "sharpe_like": ranking.sharpe_like,
        "consistency": ranking.consistency,
        "window_stability": ranking.window_stability,
        "accepted_window_ratio": ranking.accepted_window_ratio,
        "win_rate": ranking.win_rate,
        "trade_count": ranking.trade_count,
        "trades_per_day": ranking.trades_per_day,
        "avg_trade_return": ranking.avg_trade_return,
        "edge_score": ranking.edge_score,
        "expectancy": ranking.expectancy,
        "cost_drag_bps": ranking.cost_drag_bps,
        "turnover_after_fees": ranking.turnover_after_fees,
        "turnover_rate": ranking.turnover_rate,
        "allocation_amount_usd": ranking.allocation_amount_usd,
        "lock_duration_hours": ranking.lock_duration_hours,
        "leverage": ranking.leverage,
        "liquidation_buffer_pct": ranking.liquidation_buffer_pct,
        "capacity_usd": ranking.capacity_usd,
        "convex_edge_score": ranking.convex_edge_score,
        "mfe_mae_ratio": ranking.mfe_mae_ratio,
        "ml_score": ranking.ml_score,
        "ml_adjusted_score": ranking.ml_adjusted_score,
        "rejected": bool(ranking.rejected),
        "rejection_reason": ranking.rejection_reason or "",
    }
    for nested_key in ("net_roi", "net_roi_v2", "fibonacci", "ml_decision"):
        nested = explanation.get(nested_key) if isinstance(explanation.get(nested_key), dict) else {}
        payload.update({key: value for key, value in nested.items() if key not in payload})
    return payload


def _ml_feedback_order_payload(order: Order) -> tuple[dict[str, object], float]:
    realized = sum(
        float(fill.pnl or 0.0) - float(fill.fee or 0.0) - float(getattr(fill, "funding_fee", 0.0) or 0.0)
        for fill in order.fills
    )
    fill_price = _float(order.average_fill_price, _float(order.limit_price, 0.0))
    notional = abs(_float(order.filled_quantity, _float(order.quantity, 0.0)) * fill_price)
    return_after_costs = realized / max(notional, 1.0)
    details = order.details if isinstance(order.details, dict) else {}
    rejected = str(order.status or "").lower() in {"rejected", "failed", "cancelled"} or bool(order.rejection_reason)
    payload: dict[str, object] = {
        **details,
        **provider_feature_context(details.get("provider") or details.get("execution_venue")),
        "strategy_name": order.strategy_name,
        "symbol": order.symbol,
        "side": order.side,
        "mode": order.mode,
        "order_type": order.order_type,
        "status": order.status,
        "risk_status": order.risk_status,
        "rejection_reason": order.rejection_reason or "",
        "leverage": order.leverage,
        "stop_loss_present": order.stop_loss is not None,
        "take_profit_present": order.take_profit is not None,
        "allocation_amount_usd": notional,
        "net_return_after_costs": return_after_costs,
        "total_return": return_after_costs,
        "trade_count": max(len(order.fills), 1),
        "profit_factor": 1.2 if realized > 0 else 0.8,
        "consistency": 1.0 if realized > 0 else 0.0,
        "window_stability": 1.0 if order.status == "filled" else 0.0,
        "rejected": rejected,
    }
    outcome = outcome_from_result(payload)
    if rejected:
        outcome = min(outcome, -0.25)
    return payload, outcome


def _ml_feedback_vault_cycle_payload(cycle: VaultCycle) -> dict[str, object]:
    metadata = cycle.selection_metadata if isinstance(cycle.selection_metadata, dict) else {}
    starting = max(_float(cycle.starting_value_usd, 0.0), 1.0)
    ending = _float(cycle.final_settlement_amount, _float(cycle.current_estimated_value_usd, starting))
    cycle_return = (ending - starting) / starting
    return {
        **metadata,
        **provider_feature_context(metadata.get("provider") or metadata.get("execution_venue")),
        "strategy_name": cycle.selected_strategy_name,
        "profile": cycle.algorithm_profile,
        "mode": cycle.execution_mode,
        "status": cycle.status,
        "lock_duration_hours": cycle.lock_duration_hours,
        "allocation_amount_usd": cycle.starting_value_usd,
        "net_return_after_costs": cycle_return,
        "total_return": cycle_return,
        "recent_performance_score": cycle_return,
        "trade_count": max(len(cycle.allocation_legs or []), 1),
        "profit_factor": 1.2 if cycle_return > 0 else 0.8,
        "consistency": 1.0 if cycle_return > 0 else 0.0,
        "window_stability": 1.0 if str(cycle.status or "").lower() in {"completed", "settled"} else 0.0,
    }


def _ml_feedback_ops_audit_payload(audit: AuditLog) -> dict[str, object]:
    details = audit.details if isinstance(audit.details, dict) else {}
    text = f"{audit.category} {audit.action} {audit.message}".lower()
    bad_terms = ("failed", "rejected", "blocked", "timeout", "panic", "rate_limit", "429", "error", "unavailable")
    target = 1.0 if any(term in text for term in bad_terms) else 0.0
    return {
        **details,
        "category_hash": _stable_hash_bucket(audit.category),
        "action_hash": _stable_hash_bucket(audit.action),
        "message_length": len(audit.message or ""),
        "rate_limited": "429" in text or "rate_limit" in text,
        "failed": any(term in text for term in ("failed", "error", "unavailable")),
        "blocked": "blocked" in text or "rejected" in text,
        "ops_target": target,
    }


def _ml_feedback_risk_event_payload(event: RiskEvent) -> dict[str, object]:
    payload = event.payload if isinstance(event.payload, dict) else {}
    return {
        **payload,
        "rule_hash": _stable_hash_bucket(event.rule_name),
        "reason_length": len(event.reason or ""),
        "risk_block": True,
        "ops_target": 1.0,
    }


def _ml_feedback_market_history_payload(
    feature_factory: MLFeatureFactory,
    history: MLMarketHistory,
) -> tuple[dict[str, object], float]:
    candles = [row for row in history.candles if isinstance(row, dict)]
    if len(candles) < 4:
        return {}, 0.0
    split_index = max(2, min(len(candles) - 2, int(len(candles) * 0.7)))
    window = candles[: split_index + 1]
    current_close = _float(window[-1].get("close"), 0.0)
    final_close = _float(candles[-1].get("close"), 0.0)
    if current_close <= 0 or final_close <= 0:
        return {}, 0.0
    forward_return = (final_close - current_close) / current_close
    payload = feature_factory.build(
        symbol=history.symbol,
        timeframe=history.timeframe,
        candles=window,
        optimizer_context={
            "provider": history.provider,
            "strategy_name": "ml_feedback_market_history",
            "liquidity_usd": history.liquidity_usd,
            "spread_bps": history.spread_bps,
            "funding_rate": history.funding_rate,
        },
        trade_outcomes=[{"return": forward_return}],
        cutoff_timestamp=window[-1].get("timestamp"),
    )
    return payload, outcome_from_result(
        {
            "net_return_after_costs": forward_return,
            "total_return": forward_return,
            "recent_performance_score": forward_return,
            "profit_factor": 1.2 if forward_return > 0 else 0.8,
            "consistency": 1.0 if forward_return > 0 else 0.0,
            "window_stability": 1.0,
            "trade_count": 1,
        }
    )


def _ml_feedback_history_horizon(timeframe: str | None) -> str:
    value = str(timeframe or "1h").lower()
    if value in {"1m", "3m", "5m", "15m", "30m"}:
        return "1h"
    if value in {"1h", "4h"}:
        return "24h"
    return "24h"


def _stable_hash_bucket(value: object, buckets: int = 1000) -> float:
    text = str(value or "")
    total = 0
    for char in text:
        total = (total * 31 + ord(char)) % max(buckets, 1)
    return total / max(buckets, 1)


def _high_upside_market_data_pace(progress_state: dict[str, object], phase: str) -> None:
    rate = _float(current_app.config.get("HIGH_UPSIDE_MARKET_DATA_RATE_LIMIT_PER_SECOND"), 4.0)
    if rate <= 0:
        return
    now = time.monotonic()
    min_interval = 1.0 / max(rate, 0.001)
    last = _float(progress_state.get("last_high_upside_market_data_at"), 0.0)
    wait = max(0.0, min_interval - (now - last)) if last > 0 else 0.0
    if wait > 0:
        progress_state["rate_limit_wait_count"] = int(progress_state.get("rate_limit_wait_count", 0) or 0) + 1
        progress_state["rate_limit_wait_seconds"] = _float(progress_state.get("rate_limit_wait_seconds"), 0.0) + wait
        if not bool(current_app.config.get("TESTING", False)):
            _high_upside_phase(progress_state, "market_data_rate_limit_wait", phase=phase, wait_seconds=round(wait, 3))
            time.sleep(wait)
    progress_state["last_high_upside_market_data_at"] = time.monotonic()


def _high_upside_sanitize_error(exc: object) -> str:
    text = str(exc or "")
    if not text:
        return "unknown_error"
    return text.replace("\n", " ")[:500]


def _high_upside_error_is_rate_limited(error: object) -> bool:
    text = str(error or "").lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def _high_upside_diagnostics_rate_limited(diagnostics: dict[str, object]) -> bool:
    if bool(diagnostics.get("rate_limited", False)):
        return True
    breakdown = diagnostics.get("rejection_breakdown") if isinstance(diagnostics.get("rejection_breakdown"), dict) else {}
    if int((breakdown or {}).get("provider_rate_limited", 0) or 0) > 0:
        return True
    for row in list(diagnostics.get("rejected", []) or []):
        if isinstance(row, dict) and bool(row.get("rate_limited", False)):
            return True
    return False


def _high_upside_record_rate_limit_backoff(provider: str, timeframe: str) -> None:
    seconds = _float(current_app.config.get("HIGH_UPSIDE_RATE_LIMIT_BACKOFF_SECONDS"), 300.0)
    if seconds <= 0:
        return
    until = datetime.utcnow() + timedelta(seconds=seconds)
    Setting.set_json("high_upside_rate_limited_until", until.isoformat() + "Z")
    audit = AuditLog(
        category="risk",
        action="high_upside_provider_rate_limit_backoff",
        message="High-upside discovery observed provider rate limiting and set a temporary backoff.",
    )
    audit.details = {"provider": provider, "timeframe": timeframe, "backoff_until": until.isoformat() + "Z", "backoff_seconds": seconds}
    db.session.add(audit)
    db.session.commit()


def _high_upside_phase(progress_state: dict[str, object], phase: str, **details: object) -> None:
    progress_state["current_phase"] = phase
    payload = " ".join(f"{key}={value}" for key, value in details.items() if value not in {None, ""})
    click.echo(f"{HIGH_UPSIDE_DISCOVERY_PROGRESS_PREFIX} {phase}{(' ' + payload) if payload else ''}", err=True)


def _high_upside_deadline_reached(deadline_monotonic: float) -> bool:
    return bool(deadline_monotonic and time.monotonic() >= deadline_monotonic)


def _high_upside_direction(row: dict[str, object]) -> str:
    momentum = _float(row.get("momentum_acceleration"))
    expected_move = _float(row.get("cost_adjusted_expected_move"))
    if momentum < 0 and expected_move < 0:
        return "short"
    return "long"


def _ml_objective(value: object) -> str:
    objective = str(value or "risk_adjusted").strip().lower()
    if objective in {"extreme_upside", "extreme_roi_1h", "one_h10", "1h10", "one_hour_10x", "consistent_roi_1w"}:
        if objective in {"1h10", "one_hour_10x"}:
            return "one_h10"
        return objective
    return "risk_adjusted"


def _ml_extreme_target_roi_pct(value: float | None = None, *, objective: str = "extreme_upside", horizon: str = "1h") -> float:
    if value is not None:
        raw = value
    elif _ml_objective(objective) in {"one_h10", "1h10", "one_hour_10x"} or str(horizon or "").lower() == "1h10":
        raw = current_app.config.get("ML_TARGET_ROI_1H10_PCT", current_app.config.get("ONE_H10_TARGET_ROI_PCT", 1000.0))
    elif _ml_objective(objective) == "consistent_roi_1w" or str(horizon or "").lower() in {"1w", "7d"}:
        raw = current_app.config.get("ML_TARGET_ROI_1W_PCT", 100.0)
    elif _ml_objective(objective) == "extreme_roi_1h":
        raw = current_app.config.get("ML_TARGET_ROI_1H_PCT", 1000.0)
    else:
        raw = current_app.config.get("ML_EXTREME_UPSIDE_TARGET_ROI_PCT", 1000.0)
    return max(1.0, _float(raw, 1000.0))


def _ml_extreme_target_policy(objective: str, target_roi_pct: float) -> dict[str, object]:
    return {
        "objective": _ml_objective(objective),
        "target_roi_pct": target_roi_pct,
        "target_is_aspirational": True,
        "guarantees_profit": False,
        "hard_gates_remain_mandatory": [
            "profit_factor",
            "drawdown",
            "trade_count",
            "costs",
            "liquidity",
            "spread",
            "stop_loss",
            "take_profit",
            "RiskEngine",
        ],
    }


def _ml_extreme_upside_readiness(horizon: str, objective: str, *, provider: str = "global") -> dict[str, object]:
    provider_key = normalize_exchange_provider(provider)
    objective_key = _ml_objective(objective)
    if objective_key not in {"extreme_upside", "extreme_roi_1h", "consistent_roi_1w"}:
        return {
            "ready": True,
            "required": False,
            "blockers": [],
            "family": "pytorch_extreme_upside",
            "provider": provider_key,
        }
    family = "pytorch_roi_target" if objective_key in {"extreme_roi_1h", "consistent_roi_1w"} else "pytorch_extreme_upside"
    try:
        readiness = dict(get_service("ml_decision_engine").family_readiness(family, horizon, provider=provider_key))
    except Exception as exc:  # noqa: BLE001
        readiness = {"ready": False, "blockers": [_high_upside_sanitize_error(exc)], "family": family, "provider": provider_key}
    readiness["required"] = True
    return readiness


def _high_upside_ml_signal_score(
    row: dict[str, object],
    *,
    provider: str,
    timeframe: str,
    lock_duration_hours: int,
) -> dict[str, object]:
    horizon = _duration_bucket(lock_duration_hours or 1)
    context = {
        **dict(row or {}),
        "provider": provider,
        "timeframe": timeframe,
        "horizon": horizon,
        "high_upside_profile": True,
        "optimizer_profile": "aggressive_1h",
    }
    try:
        decision = dict(get_service("ml_decision_engine").decision("pytorch_gru_signal", context, horizon=horizon))
        payload = dict(decision.get("raw") or {})
        payload["ml_decision"] = decision
        return payload
    except Exception as exc:  # noqa: BLE001
        return {
            "enabled": bool(current_app.config.get("ML_SIGNAL_MODEL_ENABLED", False)),
            "status": "ml_signal_error",
            "ready_for_live": False,
            "action": "hold",
            "confidence": 0.0,
            "blockers": [_high_upside_sanitize_error(exc)],
        }


def _high_upside_extreme_upside_score(
    row: dict[str, object],
    *,
    provider: str,
    timeframe: str,
    lock_duration_hours: int,
    target_roi_pct: float,
    objective: str,
) -> dict[str, object]:
    objective_key = _ml_objective(objective)
    if objective_key not in {"extreme_upside", "extreme_roi_1h", "consistent_roi_1w"}:
        return {"ready": True, "required": False, "objective": "risk_adjusted"}
    horizon = _duration_bucket(lock_duration_hours or 1)
    family = "pytorch_roi_target" if objective_key in {"extreme_roi_1h", "consistent_roi_1w"} else "pytorch_extreme_upside"
    context = {
        **dict(row or {}),
        "provider": provider,
        "timeframe": timeframe,
        "horizon": horizon,
        "objective": objective_key,
        "target_roi_pct": target_roi_pct,
        "high_upside_profile": True,
        "optimizer_profile": "aggressive_1h",
        "allocation_amount_usd": _float(row.get("allocation_amount_usd"), 10.0),
        "hard_max_leverage": current_app.config.get("MAX_LEVERAGE", 1.0),
        "lock_duration_hours": max(1, int(lock_duration_hours or 1)),
    }
    try:
        decision = dict(get_service("ml_decision_engine").decision(family, context, horizon=horizon))
    except Exception as exc:  # noqa: BLE001
        decision = {
            "ready": False,
            "family": family,
            "action": "avoid",
            "blockers": [_high_upside_sanitize_error(exc)],
            "raw": {},
        }
    decision["objective"] = objective_key
    decision["target_roi_pct"] = target_roi_pct
    return decision


def _high_upside_extreme_priority(decision: dict[str, object]) -> float:
    raw = decision.get("raw") if isinstance(decision.get("raw"), dict) else {}
    if not bool(decision.get("ready", False)):
        return 0.0
    probability = _float(raw.get("extreme_upside_probability"), 0.0)
    distance = _float(raw.get("distance_to_target_pct"), 1000.0)
    confidence = _float(decision.get("confidence"), 0.0)
    return probability * 100.0 + confidence * 25.0 - distance / 100.0


def _high_upside_ml_signal_priority(payload: dict[str, object]) -> float:
    if not bool(payload.get("ready_for_live", False)):
        return 0.0
    confidence = _float(payload.get("confidence"))
    expected_return = abs(_float(payload.get("expected_return")))
    sizing_score = _float(payload.get("sizing_score"))
    return confidence * 10.0 + expected_return * 100.0 + sizing_score


def _high_upside_rankings_for_run(run_id: int, *, provider: str) -> list[dict[str, object]]:
    if run_id <= 0:
        return []
    provider_key = normalize_exchange_provider(provider)
    rankings = (
        StrategyRanking.query.filter_by(optimizer_run_id=run_id, provider=provider_key, rejected=False)
        .order_by(StrategyRanking.score.desc(), StrategyRanking.created_at.desc())
        .limit(20)
        .all()
    )
    return [_high_upside_ranking_payload(row, provider=provider_key) for row in rankings]


def _high_upside_ranking_payload(ranking: StrategyRanking, *, provider: str) -> dict[str, object]:
    return {
        "id": ranking.id,
        "provider": normalize_exchange_provider(getattr(ranking, "provider", None) or provider),
        "collateral_asset": provider_collateral_asset(getattr(ranking, "provider", None) or provider),
        "symbol": ranking.symbol,
        "timeframe": ranking.timeframe,
        "strategy_name": ranking.strategy_name,
        "profile": ranking.profile,
        "score": ranking.score,
        "rejected": bool(ranking.rejected),
        "rejection_reason": ranking.rejection_reason or "",
        "net_return_after_costs": ranking.net_return_after_costs,
        "profit_factor": ranking.profit_factor,
        "trade_count": ranking.trade_count,
        "max_drawdown": ranking.max_drawdown,
        "leverage": ranking.leverage,
        "parameters": dict(ranking.parameters or {}),
        "high_upside_profile": bool((ranking.parameters or {}).get("high_upside_profile", False)),
    }


def _dedupe_rankings(rankings: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[int] = set()
    result: list[dict[str, object]] = []
    for row in sorted(rankings, key=lambda item: _float(item.get("score")), reverse=True):
        ranking_id = int(row.get("id") or 0)
        if ranking_id <= 0 or ranking_id in seen:
            continue
        seen.add(ranking_id)
        result.append(row)
    return result


def _high_upside_rejection_rate(scanner_candidates: list[dict[str, object]], breakdown: dict[str, int]) -> float:
    rejected = sum(max(0, int(count or 0)) for count in breakdown.values())
    total = rejected + len(scanner_candidates)
    return rejected / total if total else 0.0


def _high_upside_runtime(started_at: float, timeout_seconds: float, *, timed_out: bool) -> dict[str, object]:
    return {
        "elapsed_seconds": max(time.monotonic() - started_at, 0.0),
        "timeout_seconds": timeout_seconds,
        "timed_out": bool(timed_out),
        "market_data_cache": _market_data_cache_stats(),
    }


def _high_upside_next_commands(accepted_ranking_ids: list[int]) -> dict[str, object]:
    if not accepted_ranking_ids:
        return {
            "research": "flask discover-high-upside-vault-candidates",
            "train_offline_ranker": "flask train-offline-ranker --provider <kucoin|hyperliquid> --horizon 1h --model both --confirm TRAIN-OFFLINE-RANKER",
            "train_ml_signal_model": "flask train-ml-signal-model --provider <kucoin|hyperliquid> --horizon 1h --model pytorch_gru --confirm TRAIN-ML-SIGNAL-MODEL",
            "train_ml_suite": "flask train-ml-suite --provider <kucoin|hyperliquid> --horizon 1h --model-family all --confirm TRAIN-ML-SUITE",
            "ml_readiness": "flask ml-readiness --horizon 1h",
        }
    first_id = accepted_ranking_ids[0]
    return {
        "preview_canary": f"flask live-canary-trade --ranking-id {first_id} --user-id <id>",
        "vault_preview": "Use the vault start flow; live execution still requires all vault and RiskEngine gates.",
        "warning": "No discovery command submits live orders. High-upside live auto execution remains blocked unless every explicit live gate is enabled.",
    }


def _high_upside_operator_next_steps(
    has_accepted: bool,
    readiness: dict[str, object],
    blockers: list[str],
) -> list[str]:
    if not has_accepted:
        return [
            "Review scanner and optimizer rejection breakdowns before expanding symbols or timeframes.",
            "Train and promote offline ML only after enough accepted historical rows exist.",
            "Train and promote the PyTorch ML signal model before enabling high-upside live auto execution.",
        ]
    if readiness.get("ready"):
        return ["Preview the accepted ranking through the existing canary or vault flow before any live execution."]
    return [
        "Accepted research rankings exist, but live high-upside execution is blocked.",
        "Review high_upside_live_readiness.blockers; do not lower risk gates to force a trade.",
    ] + [f"blocker: {item}" for item in blockers]


def _bounded_optimizer_scanner_diagnostics(config: object, *, timeout_seconds: float) -> dict[str, object]:
    try:
        with _optimizer_deadline(timeout_seconds):
            return _optimizer_scanner_diagnostics(config)
    except _OptimizerCliTimeout as exc:
        return {
            "accepted": [],
            "rejected": [],
            "rejection_breakdown": {"scanner_timeout": 1},
            "rejection_rate": 1.0,
            "cache_hit": False,
            "error": str(exc),
            "timed_out": True,
        }


def _market_data_cache_stats() -> dict[str, object]:
    try:
        market_data = get_service("market_data")
        if hasattr(market_data, "cache_stats"):
            return dict(market_data.cache_stats())
    except Exception:  # noqa: BLE001
        return {}
    return {}


def _scalar_count(sql: str) -> int:
    return int(db.session.execute(db.text(sql)).scalar() or 0)


def _database_backend() -> str:
    uri = str(current_app.config.get("SQLALCHEMY_DATABASE_URI", "") or "").lower()
    if uri.startswith("sqlite"):
        return "sqlite"
    if uri.startswith(("postgresql://", "postgresql+", "postgres://")):
        return "postgres"
    return "other"


def _table_exists(table: str) -> bool:
    if _database_backend() == "sqlite":
        rows = db.session.execute(db.text(f"PRAGMA table_info({table})")).mappings().all()
        return bool(rows)
    try:
        return bool(sa_inspect(db.engine).has_table(table))
    except Exception:  # noqa: BLE001
        return False


def _table_count(table: str) -> int:
    if not _table_exists(table):
        return 0
    return _scalar_count(f"SELECT COUNT(*) FROM {table}")


def _drop_legacy_tables() -> None:
    for table in ("paper_equity_snapshot", "paper_account"):
        db.session.execute(db.text(f"DROP TABLE IF EXISTS {table}"))


def _prune_efficiency_data_payload(
    *,
    protect_username: str,
    confirmed: bool,
    confirmation_value: str,
    no_trade_retention_hours: float | None,
    transient_retention_hours: float | None,
    vacuum: bool,
) -> dict[str, object]:
    username = str(protect_username or "").strip()
    blockers: list[str] = []
    warnings: list[str] = []
    if username.lower() != "sufyanh":
        blockers.append("protect_username_must_be_sufyanh")
    if confirmation_value and not confirmed:
        blockers.append("invalid_confirmation_phrase")
    protected_user = User.query.filter(db.func.lower(User.username) == username.lower()).one_or_none() if username else None
    if protected_user is None:
        blockers.append("protected_user_not_found")

    no_trade_hours = max(
        0.0,
        float(
            no_trade_retention_hours
            if no_trade_retention_hours is not None
            else current_app.config.get("NO_TRADE_AUDIT_RETENTION_HOURS", 24.0)
            or 24.0
        ),
    )
    transient_hours = max(
        0.0,
        float(
            transient_retention_hours
            if transient_retention_hours is not None
            else current_app.config.get("TRANSIENT_AUDIT_RETENTION_HOURS", 72.0)
            or 72.0
        ),
    )
    now = datetime.utcnow()
    no_trade_cutoff = now - timedelta(hours=no_trade_hours)
    transient_cutoff = now - timedelta(hours=transient_hours)
    transient_actions = ("provider_runtime_backoff", "one_h10_market_data_backoff")

    no_trade_query = AuditLog.query.filter(
        AuditLog.category == "strategy",
        AuditLog.action == "no_trade",
        AuditLog.created_at < no_trade_cutoff,
    )
    transient_query = AuditLog.query.filter(
        AuditLog.category == "strategy",
        AuditLog.action.in_(transient_actions),
        AuditLog.created_at < transient_cutoff,
    )
    candidates = {
        "strategy_no_trade": no_trade_query.count(),
        "strategy_transient_backoff": transient_query.count(),
    }
    protected_before = _protected_account_counts(protected_user.id if protected_user else None)
    backup_path = None

    if confirmed:
        backup_path = _sqlite_backup_path()
        if backup_path is None:
            blockers.append("automatic_sqlite_backup_unavailable")

    deleted = {"strategy_no_trade": 0, "strategy_transient_backoff": 0}
    vacuumed = False
    if confirmed and not blockers:
        deleted["strategy_no_trade"] = no_trade_query.delete(synchronize_session=False)
        deleted["strategy_transient_backoff"] = transient_query.delete(synchronize_session=False)
        db.session.commit()
        if vacuum:
            vacuumed = _vacuum_sqlite_database()
            if not vacuumed:
                warnings.append("sqlite_vacuum_skipped_or_unavailable")

    protected_after = _protected_account_counts(protected_user.id if protected_user else None)
    if confirmed and protected_before != protected_after:
        blockers.append("protected_account_counts_changed")

    return {
        "dry_run": not confirmed,
        "confirmed": confirmed and not blockers,
        "backup": str(backup_path) if backup_path else None,
        "protected_username": username,
        "protected_user_id": protected_user.id if protected_user else None,
        "retention": {
            "no_trade_hours": no_trade_hours,
            "transient_hours": transient_hours,
            "no_trade_cutoff": no_trade_cutoff.isoformat(),
            "transient_cutoff": transient_cutoff.isoformat(),
        },
        "candidates": candidates,
        "deleted": deleted,
        "vacuumed": vacuumed,
        "protected_counts_before": protected_before,
        "protected_counts_after": protected_after,
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
    }


def _protected_account_counts(user_id: int | None) -> dict[str, int]:
    if user_id is None:
        return {}
    return {
        "users": _scalar_count(f"SELECT COUNT(*) FROM \"user\" WHERE id = {int(user_id)}"),
        "wallet_balances": _scalar_count(f"SELECT COUNT(*) FROM wallet_balance WHERE user_id = {int(user_id)}"),
        "wallet_transactions": _scalar_count(f"SELECT COUNT(*) FROM wallet_transaction WHERE user_id = {int(user_id)}"),
        "trading_connections": _scalar_count(f"SELECT COUNT(*) FROM trading_connection WHERE user_id = {int(user_id)}"),
        "vault_cycles": _scalar_count(f"SELECT COUNT(*) FROM vault_cycle WHERE user_id = {int(user_id)}"),
        "orders": _scalar_count(f"SELECT COUNT(*) FROM \"order\" WHERE user_id = {int(user_id)}"),
        "fills": _scalar_count(
            f"SELECT COUNT(*) FROM fill WHERE order_id IN (SELECT id FROM \"order\" WHERE user_id = {int(user_id)})"
        ),
        "wallet_addresses": _table_user_count("wallet_address", int(user_id)),
        "deposit_addresses": _table_user_count("deposit_address", int(user_id)),
        "wallet_ledger_events": _table_user_count("wallet_ledger_event", int(user_id)),
        "wallet_withdrawals": _table_user_count("wallet_withdrawal", int(user_id)),
    }


def _table_user_count(table: str, user_id: int) -> int:
    if not _table_exists(table):
        return 0
    return _scalar_count(f"SELECT COUNT(*) FROM {table} WHERE user_id = {int(user_id)}")


def _vacuum_sqlite_database() -> bool:
    db_path = _sqlite_database_path()
    if not db_path:
        return False
    db.session.remove()
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("VACUUM")
        return True
    finally:
        connection.close()


def _load_json_file(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise click.ClickException(f"Expected {path} to contain a JSON object.")
    return value


def _production_account_readiness_payload(
    *,
    username: str = "sufyanh",
    user_id: int | None = None,
    expected_origin: str = "https://app.algvault.com",
    expected_wallet_snapshot: dict[str, object] | None = None,
    require_expected_snapshot: bool = True,
) -> dict[str, object]:
    blockers: list[str] = []
    warnings: list[str] = []
    expected = str(expected_origin or "").strip().rstrip("/")
    app_origin = str(current_app.config.get("PUBLIC_APP_ORIGIN") or "").strip().rstrip("/")
    api_origin = str(current_app.config.get("PUBLIC_API_ORIGIN") or "").strip().rstrip("/")
    deployment_target = str(current_app.config.get("DEPLOYMENT_TARGET") or "local")

    for key, origin in (("PUBLIC_APP_ORIGIN", app_origin), ("PUBLIC_API_ORIGIN", api_origin)):
        violations = public_origin_violations(origin, require_public_https=True)
        blockers.extend(f"{key.lower()}_{violation.replace(' ', '_').replace(',', '')}" for violation in violations)
        if expected and origin != expected:
            blockers.append(f"{key.lower()}_does_not_match_expected_origin")

    wallet_result = get_service("wallet_summary").account_funds_readiness(
        username=username,
        user_id=user_id,
        expected_snapshot=expected_wallet_snapshot,
        require_expected_snapshot=require_expected_snapshot,
    )
    blockers.extend(str(item) for item in wallet_result.get("blockers", []) or [])
    warnings.extend(str(item) for item in wallet_result.get("warnings", []) or [])

    return {
        "ready": not blockers,
        "production_url": {
            "expected_origin": expected,
            "public_app_origin": app_origin,
            "public_api_origin": api_origin,
            "deployment_target": deployment_target,
        },
        "account": wallet_result,
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "This command is read-only and does not call live exchanges or mutate balances.",
            "Use a profile-wallet-check JSON export from the local machine as --expected-wallet-snapshot for fund parity.",
        ],
    }


def _sqlite_backup_path() -> Path | None:
    uri = str(current_app.config.get("SQLALCHEMY_DATABASE_URI", ""))
    if not uri.startswith("sqlite:///"):
        return None
    db_path = Path(uri.removeprefix("sqlite:///"))
    if not db_path.is_absolute():
        db_path = Path(current_app.instance_path) / db_path
    if not db_path.exists():
        return None
    backup_path = db_path.with_name(f"{db_path.name}.backup-live-only-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    source = sqlite3.connect(str(db_path))
    target = sqlite3.connect(str(backup_path))
    try:
        with target:
            source.backup(target)
    finally:
        source.close()
        target.close()
    return backup_path


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _dependency_status() -> dict[str, bool]:
    return {
        "eth_account": _module_available("eth_account"),
        "bitcoinlib": _module_available("bitcoinlib"),
        "solders": _module_available("solders"),
        "xrpl": _module_available("xrpl"),
        "cryptography": _module_available("cryptography"),
        "joblib": _module_available("joblib"),
        "numpy": _module_available("numpy"),
        "scipy": _module_available("scipy"),
        "sklearn": _module_available("sklearn"),
        "xgboost": _module_available("xgboost"),
        "torch": _module_available("torch"),
    }


def _wallet_readiness_payload() -> dict[str, object]:
    custody = get_service("wallet_custody")
    readiness = custody.readiness()
    readiness.update(
        {
            "dependencies": _dependency_status(),
            "pending_withdrawals": WalletWithdrawal.query.filter(
                WalletWithdrawal.status.in_(["pending_approval", "pending_submission", "pending_gas_topup"])
            ).count(),
            "generated_address_count": WalletAddress.query.filter(
                WalletAddress.encrypted_metadata_json.like('%"custody": "in_app"%')
            ).count(),
            "sync_failures": WalletAuditLog.query.filter_by(action="wallet_deposit_sync_failed").count(),
        }
    )
    return readiness


def _reconcile_wallet_withdrawals_payload(
    *,
    username: str,
    withdrawal_id: int | None,
    confirmed: bool,
    confirmation_value: str = "",
) -> dict[str, object]:
    blockers: list[str] = []
    warnings: list[str] = []
    username_value = str(username or "").strip()
    user = None
    if username_value:
        user = User.query.filter(func.lower(User.username) == username_value.lower()).one_or_none()
        if user is None:
            blockers.append("username_not_found")
    if confirmation_value and not confirmed:
        blockers.append("invalid_confirmation")
    query = WalletWithdrawal.query
    if withdrawal_id is not None:
        query = query.filter(WalletWithdrawal.id == int(withdrawal_id))
    if user is not None:
        query = query.filter(WalletWithdrawal.user_id == user.id)
    query = query.filter(WalletWithdrawal.status.in_(["submitted", "pending_gas_topup"]))
    withdrawals = query.order_by(WalletWithdrawal.created_at.asc(), WalletWithdrawal.id.asc()).all()
    terminal_result: dict[str, object] | None = None
    if withdrawal_id is not None and not withdrawals and not blockers:
        existing = db.session.get(WalletWithdrawal, int(withdrawal_id))
        if existing is not None and (user is None or existing.user_id == user.id):
            terminal_result = {
                "withdrawal_id": existing.id,
                "asset": existing.asset,
                "amount": existing.amount,
                "status_before": existing.status,
                "provider_reference": existing.provider_reference,
                "terminal": existing.status in {"complete", "failed", "rejected"},
                "status_after": existing.status,
            }
        else:
            blockers.append("withdrawal_not_found")
    backup_path = None
    if confirmed and not blockers:
        backup_path = _sqlite_backup_path()
        if backup_path is None:
            blockers.append("automatic_sqlite_backup_unavailable")

    results: list[dict[str, object]] = [terminal_result] if terminal_result is not None else []
    if not blockers:
        custody = get_service("wallet_custody")
        treasury = get_service("platform_treasury")
        wallet_service = get_service("self_custody_wallet")
        for withdrawal in withdrawals:
            row: dict[str, object] = {
                "withdrawal_id": withdrawal.id,
                "asset": withdrawal.asset,
                "amount": withdrawal.amount,
                "status_before": withdrawal.status,
                "provider_reference": withdrawal.provider_reference,
            }
            if withdrawal.status == "pending_gas_topup":
                topup = treasury.refresh_withdrawal_topup(withdrawal)
                row["gas_topup"] = topup
                if confirmed and topup.get("status") == "complete":
                    submitted = wallet_service.submit_withdrawal(withdrawal, mode="live")
                    row["submit_after_topup"] = {
                        "status": submitted.status,
                        "provider_reference": submitted.provider_reference,
                        "failure_reason": submitted.failure_reason,
                    }
                    if submitted.status == "failed":
                        custody.release_failed_withdrawal(submitted)
                        custody._update_withdrawal_transaction(submitted, "failed", submitted.failure_reason or "Withdrawal failed after gas top-up.")  # noqa: SLF001
            if withdrawal.status == "submitted":
                row["reconciliation"] = custody.reconcile_withdrawal(withdrawal, commit=confirmed)
            row["status_after"] = withdrawal.status
            results.append(row)
        if confirmed:
            db.session.commit()
    elif confirmed:
        db.session.rollback()

    return {
        "dry_run": not confirmed,
        "confirmed": confirmed and not blockers,
        "backup": str(backup_path) if backup_path else None,
        "username": username_value or None,
        "withdrawal_id": withdrawal_id,
        "candidate_count": len(withdrawals),
        "results": results,
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
    }


def _repair_limited_cycle_payload(cycle_id: int, *, confirmed: bool, confirmation_value: str = "") -> dict[str, object]:
    cycle = db.session.get(VaultCycle, int(cycle_id))
    if cycle is None:
        return {
            "cycle_id": cycle_id,
            "preview_only": not confirmed,
            "ready": False,
            "repaired": False,
            "blockers": ["cycle_not_found"],
        }

    orders = _orders_for_cycle(cycle)
    fills = [fill for order in orders for fill in order.fills]
    run_ids = {leg.strategy_run_id for leg in cycle.allocation_legs if leg.strategy_run_id}
    if cycle.strategy_run_id:
        run_ids.add(cycle.strategy_run_id)
    runs = [db.session.get(StrategyRun, int(run_id)) for run_id in run_ids if run_id]
    runs = [run for run in runs if run is not None]
    reason = _cycle_no_execution_failure_reason(cycle, runs)
    already_repaired = cycle.execution_substatus == "failed_no_execution" and cycle.status in {"complete", "failed", "failed_no_execution"}

    blockers: list[str] = []
    if confirmation_value and not confirmed:
        blockers.append("confirmation_required")
    if orders:
        blockers.append("cycle_has_orders")
    if fills:
        blockers.append("cycle_has_fills")
    if cycle.status not in {"active", "settling", "error", "limited", "failed", "complete"} and not already_repaired:
        blockers.append("cycle_status_not_repairable")
    if cycle.execution_substatus not in {"limited", "error", "failed", "failed_no_execution", "validating_market", "initializing"} and not already_repaired:
        blockers.append("cycle_substatus_not_repairable")
    if not any((run.status == "error" or run.last_error) for run in runs) and not already_repaired:
        blockers.append("failed_strategy_run_missing")
    if not reason and not already_repaired:
        blockers.append("live_access_failure_missing")

    balance = WalletBalance.query.filter_by(user_id=cycle.user_id, asset=cycle.deposit_asset).one_or_none()
    release_amount = min(float(cycle.deposit_amount or 0.0), float(balance.locked_balance or 0.0)) if balance else 0.0
    result: dict[str, object] = {
        "cycle_id": cycle.id,
        "preview_only": not confirmed,
        "ready": not blockers,
        "repaired": False,
        "already_repaired": already_repaired,
        "blockers": list(dict.fromkeys(blockers)),
        "order_count": len(orders),
        "fill_count": len(fills),
        "strategy_run_ids": [run.id for run in runs],
        "failure_reason": reason,
        "release_amount": release_amount,
        "asset": cycle.deposit_asset,
    }
    if blockers or not confirmed or already_repaired:
        return result

    now = datetime.utcnow()
    if balance is not None and release_amount > 0:
        balance.locked_balance = max(0.0, float(balance.locked_balance or 0.0) - release_amount)
        balance.available_balance = float(balance.available_balance or 0.0) + release_amount
        if cycle.deposit_asset in {"USDC", "USDT", "USD"}:
            balance.estimated_usd_value = float(balance.available_balance or 0.0) + float(balance.locked_balance or 0.0)

    for run in runs:
        run.status = "stopped"
        run.manual_enabled = False
        if reason:
            run.last_error = run.last_error or reason
    for leg in cycle.allocation_legs:
        leg.status = "complete"
    metadata = cycle.selection_metadata
    metadata["no_order_failure_reason"] = reason
    metadata["repair_released_amount"] = release_amount
    metadata["repaired_at"] = now.isoformat() + "Z"
    cycle.selection_metadata = metadata
    summary = cycle.cycle_summary
    summary.update(
        {
            "no_order_failure_reason": reason,
            "order_count": 0,
            "fill_count": 0,
            "final_settlement_amount": float(cycle.deposit_amount or 0.0),
            "repair_released_amount": release_amount,
            "repaired_at": now.isoformat() + "Z",
        }
    )
    cycle.cycle_summary = summary
    cycle.status = "complete"
    cycle.execution_substatus = "failed_no_execution"
    cycle.live_validation_status = "failed"
    cycle.validation_failure_reason = reason
    cycle.final_settlement_amount = float(cycle.deposit_amount or 0.0)
    cycle.current_estimated_value_usd = float(cycle.starting_value_usd or cycle.current_estimated_value_usd or 0.0)
    cycle.settled_at = now
    db.session.add(
        WalletTransaction(
            vault_cycle_id=cycle.id,
            user_id=cycle.user_id,
            asset=cycle.deposit_asset,
            amount=release_amount,
            transaction_type="settlement",
            status="complete",
            note=f"No live order submitted; failed live connection cycle repaired. {reason}",
        )
    )
    audit = AuditLog(
        category="vault",
        action="repair_limited_cycle",
        message=f"Repaired no-order vault cycle {cycle.id}; released {release_amount:.8f} {cycle.deposit_asset}.",
    )
    audit.details = {
        "cycle_id": cycle.id,
        "user_id": cycle.user_id,
        "strategy_run_ids": [run.id for run in runs],
        "failure_reason": reason,
        "released_amount": release_amount,
        "order_count": 0,
        "fill_count": 0,
    }
    db.session.add(audit)
    db.session.commit()
    result.update(
        {
            "ready": True,
            "repaired": True,
            "status": cycle.status,
            "execution_substatus": cycle.execution_substatus,
            "final_settlement_amount": cycle.final_settlement_amount,
        }
    )
    return result


def _orders_for_cycle(cycle: VaultCycle) -> list[Order]:
    query = Order.query.filter_by(user_id=cycle.user_id)
    if cycle.execution_mode:
        query = query.filter_by(mode=cycle.execution_mode)
    query = query.filter(or_(Order.vault_cycle_id == cycle.id, Order.vault_cycle_id.is_(None)))
    return [
        order
        for order in query.order_by(Order.created_at.asc()).all()
        if _order_vault_cycle_id(order) == cycle.id
    ]


def _order_vault_cycle_id(order: Order) -> int | None:
    raw = order.vault_cycle_id
    if raw is None:
        raw = order.details.get("vault_cycle_id")
    try:
        return int(raw) if raw is not None and str(raw).strip() else None
    except (TypeError, ValueError):
        return None


def _cycle_no_execution_failure_reason(cycle: VaultCycle, runs: list[StrategyRun]) -> str:
    candidates = [cycle.validation_failure_reason or ""]
    candidates.extend(run.last_error or "" for run in runs)
    latest_audit = (
        AuditLog.query.filter(AuditLog.message.like(f"%Strategy run {cycle.strategy_run_id}%"))
        .order_by(AuditLog.created_at.desc())
        .first()
        if cycle.strategy_run_id
        else None
    )
    if latest_audit is not None:
        candidates.append(latest_audit.message or "")
    for raw in candidates:
        text = str(raw or "").strip()
        if not text:
            continue
        parsed = parse_exchange_failure(text)
        lowered = text.lower()
        if parsed.get("ip_whitelist_blocked") or any(
            token in lowered
            for token in (
                "invalid request ip",
                "exchange data unavailable",
                "credentials",
                "connection",
                "api",
                "providerrequesterror",
            )
        ):
            return text
    return ""


def _production_readiness_payload(*, provider: str = "global", horizon: str = "1h") -> dict[str, object]:
    blockers: list[str] = []
    warnings: list[str] = []
    provider_key = normalize_exchange_provider(provider)
    horizon_key = str(horizon or "1h").strip().lower() or "1h"
    wallet = _wallet_readiness_payload()
    dependencies = _dependency_status()
    risk_status = get_service("risk_engine").status("live")
    ml_suite = _ml_suite_readiness(horizon_key, provider=provider_key)
    connection_health = _refresh_active_connection_health(provider=provider_key)
    db_uri = str(current_app.config.get("SQLALCHEMY_DATABASE_URI", ""))
    deployment_target = str(current_app.config.get("DEPLOYMENT_TARGET", "local") or "local").strip().lower()
    db_backend = _database_backend()
    secret_key = str(current_app.config.get("SECRET_KEY", "") or "")
    totp_key = str(current_app.config.get("TOTP_ENCRYPTION_KEY", "") or "").strip()
    caps = current_app.config.get("WALLET_MAX_WITHDRAWAL_BY_ASSET") or {}
    custody_mode = str(current_app.config.get("WALLET_CUSTODY_MODE", "local_dev") or "local_dev").strip().lower()
    required_cap_assets = ("ETH", "USDC", "USDT", "BTC", "SOL", "XRP")

    if current_app.config.get("APP_MODE") != "live":
        blockers.append("APP_MODE must be live")
    if not bool(current_app.config.get("ENABLE_LIVE_TRADING", False)):
        blockers.append("ENABLE_LIVE_TRADING must be true")
    if "paper_trading" in current_app.extensions.get("services", {}):
        blockers.append("Simulated trading service must not be registered")
    if deployment_target in {"vps", "postgres", "production", "vercel"}:
        if db_backend != "postgres":
            blockers.append("DATABASE_URL must point at PostgreSQL for production deployment")
    elif not db_uri.startswith("sqlite:///") or db_uri in {"sqlite://", "sqlite:///:memory:"}:
        blockers.append("DATABASE_URL must point at a file-backed local SQLite database")
    if (
        deployment_target in {"vps", "postgres", "production", "vercel"}
        and bool(current_app.config.get("ENABLE_LIVE_TRADING", False))
        and str(current_app.config.get("WORKER_MODE", "web") or "web").strip().lower() == "web"
        and not bool(current_app.config.get("WORKER_PROCESS_CONFIGURED", False))
    ):
        blockers.append("production live trading requires a dedicated worker process")
    if secret_key in {"", "dev-secret-change-me"} or len(secret_key) < 32:
        blockers.append("FLASK_SECRET_KEY must be a non-default value with at least 32 characters")
    try:
        Fernet(totp_key.encode("utf-8"))
    except Exception:  # noqa: BLE001
        blockers.append("TOTP_ENCRYPTION_KEY must be a valid Fernet key")
    if not bool(current_app.config.get("WTF_CSRF_ENABLED", True)):
        blockers.append("CSRF protection must be enabled")
    if bool(risk_status.get("panic_lock", False)):
        blockers.append("Panic lock is active")
    if bool(risk_status.get("live_trading_blocked", False)):
        blockers.append("Live trading failure block is active")
    wallet_dependency_names = {"eth_account", "bitcoinlib", "solders", "xrpl", "cryptography"}
    missing_wallet = [name for name in sorted(wallet_dependency_names) if not dependencies.get(name, False)]
    if missing_wallet:
        warnings.append(
            "Optional wallet signing dependencies missing: "
            + ", ".join(missing_wallet)
            + "; install requirements-wallets.txt before broadcasting BTC, SOL, or XRP withdrawals."
        )
    missing_ml = [name for name in ("joblib", "numpy", "scipy", "sklearn", "xgboost") if not dependencies.get(name, False)]
    if bool(current_app.config.get("ML_OFFLINE_MODELS_ENABLED", False)) and missing_ml:
        blockers.append("Missing offline ML dependencies: " + ", ".join(missing_ml))
    if bool(current_app.config.get("ML_SIGNAL_MODEL_ENABLED", False)) and not dependencies.get("torch", False):
        blockers.append("Missing ML signal dependencies: torch")
    if bool(current_app.config.get("ML_ALL_AREAS_ENABLED", False)) and not bool(ml_suite.get("ready", False)):
        blockers.append("ML suite readiness failed: " + ", ".join(str(item) for item in ml_suite.get("blockers", [])[:5]))
    for health in connection_health:
        if not bool(health.get("can_trade", False)):
            provider = str(health.get("provider") or "exchange").title()
            reason = str(health.get("failure_reason") or "latest live access check failed")
            blockers.append(f"{provider} active connection cannot trade: {reason}")
            if health.get("client_ip"):
                warnings.append(f"{provider} API key whitelist must include current client IP {health.get('client_ip')}")
    if not bool(wallet.get("ready", False)):
        blockers.extend(str(item) for item in wallet.get("blockers", []))
    if deployment_target in {"vps", "postgres", "production", "vercel"} and bool(current_app.config.get("WALLET_WITHDRAWALS_ENABLED", False)):
        if custody_mode not in {"kms", "hsm", "mpc"}:
            blockers.append("production withdrawals require approved custody mode: kms, hsm, or mpc")
        if custody_mode == "mpc":
            if not str(current_app.config.get("WALLET_MPC_SIGNER_URL", "") or "").strip():
                blockers.append("WALLET_MPC_SIGNER_URL must be configured for mpc custody")
            if not str(current_app.config.get("WALLET_MPC_SIGNER_TOKEN", "") or "").strip():
                blockers.append("WALLET_MPC_SIGNER_TOKEN must be configured for mpc custody")
        if bool(current_app.config.get("WALLET_SIGNER_ISOLATION_REQUIRED", True)) and not bool(
            current_app.config.get("WALLET_SIGNER_ISOLATION_CONFIRMED", False)
        ):
            blockers.append("production withdrawals require confirmed signer isolation")
        if not bool(current_app.config.get("WALLET_SDK_CHECKS_PASSED", False)):
            blockers.append("production withdrawals require passing wallet SDK/integration checks")
        if float(current_app.config.get("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_WALLET", 0.0) or 0.0) <= 0:
            blockers.append("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_WALLET must be configured")
        asset_limits = current_app.config.get("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_ASSET") or {}
        if not isinstance(asset_limits, dict) or not asset_limits:
            blockers.append("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_ASSET_JSON must be configured")
        if float(current_app.config.get("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_DESTINATION", 0.0) or 0.0) <= 0:
            blockers.append("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_DESTINATION must be configured")
        if float(current_app.config.get("WALLET_DAILY_GLOBAL_WITHDRAWAL_LIMIT", 0.0) or 0.0) <= 0:
            blockers.append("WALLET_DAILY_GLOBAL_WITHDRAWAL_LIMIT must be configured")
    for asset in required_cap_assets:
        cap = caps.get(asset) if isinstance(caps, dict) else None
        if cap is None and isinstance(caps, dict):
            cap = caps.get(asset.lower())
        try:
            cap_value = float(cap or 0.0)
        except (TypeError, ValueError):
            cap_value = 0.0
        if cap_value <= 0:
            blockers.append(f"WALLET_MAX_WITHDRAWAL_BY_ASSET_JSON must set a positive {asset} cap")
    if not current_app.config.get("SIGNUP_INVITE_CODE"):
        warnings.append("SIGNUP_INVITE_CODE is empty; public registration is open")
    if not current_app.config.get("ADMIN_PASSWORD") and User.query.count() == 0:
        warnings.append("No seeded admin password and no users exist yet")
    web_concurrency = int(current_app.config.get("WEB_CONCURRENCY", 1) or 1)
    if deployment_target in {"vps", "postgres", "production"} and web_concurrency > 1:
        warnings.append("Use one Gunicorn worker for in-process strategy loops; scale with threads or a dedicated worker process.")
    if deployment_target in {"vps", "postgres", "production", "vercel"} and not bool(current_app.config.get("SESSION_COOKIE_SECURE", False)):
        warnings.append("SESSION_COOKIE_SECURE should be enabled behind HTTPS in production.")
    if deployment_target in {"vps", "postgres", "production", "vercel"} and not bool(current_app.config.get("PROXY_FIX_ENABLED", False)):
        warnings.append("PROXY_FIX_ENABLED should be true when running behind an HTTPS proxy or platform edge.")

    return {
        "ready": not blockers,
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "mode": {
            "app_mode": current_app.config.get("APP_MODE"),
            "enable_live_trading": bool(current_app.config.get("ENABLE_LIVE_TRADING", False)),
            "current_mode": Setting.get_json("current_mode", "live"),
            "deployment_target": deployment_target,
        },
        "database": {
            "uri": db_uri,
            "backend": db_backend,
            "sqlite_file": _sqlite_database_path(),
            "legacy_paper_tables": {
                "paper_account": _table_exists("paper_account"),
                "paper_equity_snapshot": _table_exists("paper_equity_snapshot"),
            },
        },
        "security": {
            "csrf_enabled": bool(current_app.config.get("WTF_CSRF_ENABLED", True)),
            "secret_key_configured": secret_key not in {"", "dev-secret-change-me"} and len(secret_key) >= 32,
            "totp_encryption_key_valid": "TOTP_ENCRYPTION_KEY must be a valid Fernet key" not in blockers,
            "signup_invite_configured": bool(current_app.config.get("SIGNUP_INVITE_CODE")),
            "session_cookie_secure": bool(current_app.config.get("SESSION_COOKIE_SECURE", False)),
            "session_cookie_httponly": bool(current_app.config.get("SESSION_COOKIE_HTTPONLY", True)),
            "secure_headers_enabled": bool(current_app.config.get("SECURE_HEADERS_ENABLED", True)),
            "rate_limit_enabled": bool(current_app.config.get("RATELIMIT_ENABLED", True)),
            "proxy_fix_enabled": bool(current_app.config.get("PROXY_FIX_ENABLED", False)),
        },
        "risk": risk_status,
        "connection_health": connection_health,
        "active_connections": _connection_readiness_summary(connection_health),
        "wallet": wallet,
        "provider": provider_key,
        "horizon": horizon_key,
        "offline_ml": _offline_ml_readiness(horizon_key, provider=provider_key),
        "ml_signal": _ml_signal_readiness(horizon_key, provider=provider_key),
        "ml_suite": ml_suite,
        "dependencies": dependencies,
    }


def _refresh_active_connection_health(provider: str = "global") -> list[dict[str, object]]:
    service = get_service("trading_connections")
    refreshed: list[dict[str, object]] = []
    provider_key = normalize_exchange_provider(provider)
    connections = TradingConnection.query.filter_by(is_active=True).order_by(
        TradingConnection.updated_at.desc(),
        TradingConnection.id.desc(),
    )
    if provider_key != "global":
        connections = connections.filter_by(provider=provider_key)
    for connection in connections:
        refreshed.append(_fresh_connection_health(connection, service))
    if refreshed:
        db.session.commit()
    if refreshed:
        return refreshed
    cached = active_connection_health()
    if provider_key != "global":
        return [row for row in cached if normalize_exchange_provider(row.get("provider")) == provider_key]
    return cached


def _fresh_connection_health(connection: TradingConnection, service: object | None = None) -> dict[str, object]:
    service = service or get_service("trading_connections")
    try:
        provider_spec = service.provider_spec(connection.provider)
        if not bool(provider_spec.get("tradable", False)):
            health = build_connection_health(
                connection,
                can_trade=False,
                alerts=[f"{connection.provider.title()} is not a tradable provider."],
            )
            return store_connection_health(connection, health)
        if str(connection.verification_status) != "verified":
            health = build_connection_health(
                connection,
                can_trade=False,
                alerts=["Connection is active but not verified."],
            )
            return store_connection_health(connection, health)
        snapshot = service.account_snapshot(connection.user_id, "live", connection.id)
        alerts = [str(item) for item in (snapshot.alerts or []) if str(item).strip()]
        can_trade = bool(service.can_trade(connection.user_id, "live", connection.id)) and not alerts
        if not can_trade and not alerts:
            alerts = ["Trading connector reports live trading unavailable."]
        health = build_connection_health(
            connection,
            can_trade=can_trade,
            alerts=alerts,
        )
    except Exception as exc:  # noqa: BLE001
        health = build_connection_health(
            connection,
            can_trade=False,
            alerts=[str(exc)],
            failure_reason=str(exc),
        )
    return store_connection_health(connection, health)


def _connection_readiness_summary(connection_health: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for health in connection_health:
        summaries.append(
            {
                "provider": health.get("provider"),
                "active_connection_id": health.get("connection_id"),
                "last_checked_at": health.get("last_checked_at"),
                "can_trade": bool(health.get("can_trade", False)),
                "failure_category": health.get("failure_category", ""),
                "actionable_blocker": "" if bool(health.get("can_trade", False)) else operator_connection_message(health),
            }
        )
    return summaries


def _sqlite_database_path() -> str | None:
    uri = str(current_app.config.get("SQLALCHEMY_DATABASE_URI", ""))
    if not uri.startswith("sqlite:///"):
        return None
    db_path = Path(uri.removeprefix("sqlite:///"))
    if not db_path.is_absolute():
        db_path = Path(current_app.instance_path) / db_path
    return str(db_path)


def _optimizer_scanner_diagnostics(config: object) -> dict[str, object]:
    scanner = get_service("market_scanner")
    symbols = [str(symbol).upper() for symbol in getattr(config, "symbols", []) if str(symbol).strip()]
    if not symbols:
        symbols = [str(symbol).upper() for symbol in current_app.config.get("ALLOWED_SYMBOLS", ["BTC"])]
    timeframes = list(getattr(config, "timeframes", []) or ["5m"])
    strategies = list(getattr(config, "strategy_names", []) or ["scalping"])
    try:
        scanner.score_candidates(
            symbols,
            mode=str(getattr(config, "mode", "testnet") or "testnet"),
            timeframe=str(timeframes[0]),
            duration_seconds=max(1, int(getattr(config, "lock_duration_hours", 1) or 1)) * 3600,
            strategy_name=str(strategies[0]),
            optimizer_profile=str(getattr(config, "profile", "")),
        )
    except Exception as exc:  # noqa: BLE001
        return {"accepted": [], "rejected": [], "rejection_breakdown": {}, "rejection_rate": 0.0, "error": str(exc)}
    return dict(getattr(scanner, "last_scan_diagnostics", {}) or {})


def _offline_ml_readiness(horizon: str = "1h", *, require_blend: bool = True, provider: str = "global") -> dict[str, object]:
    provider_key = normalize_exchange_provider(provider)
    try:
        readiness = get_service("offline_ranker").readiness(horizon, require_blend=require_blend, provider=provider_key)
    except Exception as exc:  # noqa: BLE001
        return {
            "ready": False,
            "horizon": horizon,
            "provider": provider_key,
            "blockers": [str(exc)],
            "promoted_model": None,
        }
    latest = (
        MLOfflineModel.query.filter_by(horizon=str(horizon or "global").lower(), provider=provider_key)
        .filter(MLOfflineModel.model_type.in_(["sklearn", "xgboost"]))
        .order_by(MLOfflineModel.created_at.desc())
        .first()
    )
    return {
        **readiness,
        "latest_model_id": latest.id if latest else None,
        "latest_model_status": latest.status if latest else None,
        "candidate_count": MLOfflineModel.query.filter_by(
            horizon=str(horizon or "global").lower(),
            provider=provider_key,
            status="candidate",
        )
        .filter(MLOfflineModel.model_type.in_(["sklearn", "xgboost"]))
        .count(),
    }


def _ml_signal_readiness(horizon: str = "1h", *, provider: str = "global") -> dict[str, object]:
    provider_key = normalize_exchange_provider(provider)
    try:
        readiness = get_service("ml_signal_model").readiness(
            horizon,
            require_promoted=bool(current_app.config.get("ML_SIGNAL_REQUIRE_PROMOTED", True)),
            provider=provider_key,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ready": False,
            "horizon": horizon,
            "provider": provider_key,
            "blockers": [str(exc)],
            "promoted_model": None,
        }
    latest = (
        MLOfflineModel.query.filter_by(
            horizon=str(horizon or "global").lower(),
            provider=provider_key,
            model_type="pytorch_gru",
        )
        .order_by(MLOfflineModel.created_at.desc())
        .first()
    )
    return {
        **readiness,
        "latest_model_id": latest.id if latest else None,
        "latest_model_status": latest.status if latest else None,
        "candidate_count": MLOfflineModel.query.filter_by(
            horizon=str(horizon or "global").lower(),
            provider=provider_key,
            model_type="pytorch_gru",
            status="candidate",
        ).count(),
    }


def _ml_suite_readiness(horizon: str = "1h", *, family: str = "all", provider: str = "global") -> dict[str, object]:
    provider_key = normalize_exchange_provider(provider)
    try:
        readiness = get_service("ml_decision_engine").readiness(horizon, family=family, provider=provider_key)
    except Exception as exc:  # noqa: BLE001
        return {
            "ready": False,
            "enabled": bool(current_app.config.get("ML_ALL_AREAS_ENABLED", False)),
            "horizon": horizon,
            "provider": provider_key,
            "family": family,
            "blockers": [str(exc)],
            "families": {},
        }
    latest_rows = (
        MLOfflineModel.query.filter_by(horizon=str(horizon or "global").lower(), provider=provider_key)
        .order_by(MLOfflineModel.created_at.desc())
        .limit(20)
        .all()
    )
    return {
        **readiness,
        "latest_models": [
            {
                "id": row.id,
                "provider": getattr(row, "provider", "global"),
                "model_type": row.model_type,
                "status": row.status,
                "feature_schema_version": row.feature_schema_version,
                "created_at": row.created_at,
                "promoted_at": row.promoted_at,
            }
            for row in latest_rows
        ],
        "torch_runtime": _torch_runtime_readiness(),
    }


def _ml_quality_report_payload(
    *,
    horizon: str = "1h",
    provider: str = "both",
    family: str = "all",
    candidate_limit: int = 3,
) -> dict[str, object]:
    horizon_key = str(horizon or "1h").lower()
    providers = ["hyperliquid", "kucoin"] if provider == "both" else [normalize_exchange_provider(provider)]
    families = list(ML_DECISION_MODEL_FAMILIES) if family == "all" else [str(family or "all")]
    reports: list[dict[str, object]] = []
    blocker_counts: dict[str, int] = {}
    for provider_key in providers:
        for family_key in families:
            report = _ml_quality_family_report(
                horizon=horizon_key,
                provider=provider_key,
                family=family_key,
                candidate_limit=candidate_limit,
            )
            reports.append(report)
            for blocker in list(report.get("blockers", []) or []):
                blocker_counts[str(blocker)] = blocker_counts.get(str(blocker), 0) + 1
    ready = all(bool(report.get("ready", False)) for report in reports)
    return {
        "ready": ready,
        "horizon": horizon_key,
        "provider_request": provider,
        "family_request": family,
        "providers": providers,
        "families": reports,
        "blocker_counts": dict(sorted(blocker_counts.items(), key=lambda item: (-item[1], item[0]))),
        "next_commands": {
            "readiness": "flask ml-readiness --horizon 1h --provider <provider>",
            "rapid_preview": "flask live-rapid-ml-trader --user-id 1 --capital-usd 5 --provider both --duration-minutes 0 --compact",
        },
        "notes": [
            "This command is read-only and never trains, promotes, submits orders, or changes balances.",
            "Promote only candidates whose diagnostics are ready=true.",
        ],
    }


def _sweep_ml_signal_payload(
    *,
    horizon: str,
    provider: str,
    model_type: str,
    objective: str,
    use_market_history: bool,
    thresholds: str,
    epochs: str,
    min_confidences: str,
    max_runs_per_provider: int,
) -> dict[str, object]:
    horizon_key = str(horizon or "1h").lower()
    providers = ["hyperliquid", "kucoin"] if provider == "both" else [normalize_exchange_provider(provider)]
    threshold_values = _csv_floats(thresholds, default=[0.0005, 0.001, 0.002, 0.003, 0.005])
    epoch_values = _csv_ints(epochs, default=[16, 32])
    confidence_values = _csv_floats(min_confidences, default=[0.55, 0.60])
    max_runs = max(1, int(max_runs_per_provider or 1))
    provider_reports: list[dict[str, object]] = []
    ready_commands: list[str] = []
    for provider_key in providers:
        attempts: list[dict[str, object]] = []
        ready_candidate_ids: list[int] = []
        run_count = 0
        for threshold in threshold_values:
            for epoch_count in epoch_values:
                for confidence in confidence_values:
                    if run_count >= max_runs:
                        break
                    run_count += 1
                    config = dict(current_app.config)
                    config.update(
                        {
                            "ML_SIGNAL_TARGET_RETURN_THRESHOLD": float(threshold),
                            "ML_SIGNAL_TRAINING_EPOCHS": int(epoch_count),
                            "ML_SIGNAL_MIN_CONFIDENCE": float(confidence),
                            "ML_SIGNAL_MAX_TRAINING_ROWS": min(
                                int(config.get("ML_SIGNAL_MAX_TRAINING_ROWS", 15_000) or 15_000),
                                int(config.get("ML_SIGNAL_SWEEP_MAX_TRAINING_ROWS", 6_000) or 6_000),
                            ),
                        }
                    )
                    result = MLSignalModel(config).train(
                        horizon_key,
                        model_type=model_type,
                        objective=objective,
                        use_market_history=use_market_history,
                        provider=provider_key,
                    )
                    model_payload = result.get("model") if isinstance(result.get("model"), dict) else {}
                    model_id = model_payload.get("id") if model_payload else None
                    row = db.session.get(MLOfflineModel, int(model_id)) if model_id is not None else None
                    diagnostics = MLSignalModel(config).promotion_diagnostics(row) if row is not None else {"ready": False, "blockers": result.get("blockers", [])}
                    ready = bool(diagnostics.get("ready", False))
                    if ready and model_id is not None:
                        ready_candidate_ids.append(int(model_id))
                        ready_commands.append(
                            f"ML_SIGNAL_MIN_CONFIDENCE={float(confidence):g} "
                            f"flask promote-ml-signal-model --horizon {horizon_key} --provider {provider_key} "
                            f"--model-id {int(model_id)} --confirm PROMOTE-ML-SIGNAL-MODEL"
                        )
                    metrics = model_payload.get("metrics") if isinstance(model_payload.get("metrics"), dict) else {}
                    attempts.append(
                        {
                            "model_id": model_id,
                            "ready": ready,
                            "blockers": list(diagnostics.get("blockers", []) or []),
                            "threshold": float(threshold),
                            "epochs": int(epoch_count),
                            "min_confidence": float(confidence),
                            "validation_loss": model_payload.get("validation_loss") if model_payload else None,
                            "accuracy": metrics.get("accuracy"),
                            "action_precision": metrics.get("action_precision"),
                            "action_rate": metrics.get("action_rate"),
                            "false_positive_rate": metrics.get("false_positive_rate"),
                        }
                    )
                if run_count >= max_runs:
                    break
            if run_count >= max_runs:
                break
        provider_reports.append(
            {
                "provider": provider_key,
                "ready": bool(ready_candidate_ids),
                "ready_candidate_ids": ready_candidate_ids,
                "attempt_count": len(attempts),
                "attempts": attempts,
            }
        )
    return {
        "ready": any(bool(report.get("ready")) for report in provider_reports),
        "research_only": True,
        "submitted": False,
        "live_orders_created": False,
        "horizon": horizon_key,
        "provider_request": provider,
        "providers": provider_reports,
        "next_commands": {
            "promote_ready_signal_candidates": ready_commands,
            "quality_report": "flask ml-quality-report --horizon 1h --provider both --model-family all --candidate-limit 3 --compact",
            "rapid_preview": "flask live-rapid-ml-trader --user-id 1 --capital-usd 5 --provider both --duration-minutes 0 --compact",
        },
        "notes": [
            "This command trains candidate signal models only; it never promotes models or submits orders.",
            "If a ready command includes ML_SIGNAL_MIN_CONFIDENCE, use the same value for quality report, preview, and submit.",
        ],
    }


def _csv_floats(value: str, *, default: list[float]) -> list[float]:
    values: list[float] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed = float(item)
        except ValueError:
            continue
        if parsed > 0:
            values.append(parsed)
    return values or list(default)


def _csv_ints(value: str, *, default: list[int]) -> list[int]:
    values: list[int] = []
    for item in str(value or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed = int(item)
        except ValueError:
            continue
        if parsed > 0:
            values.append(parsed)
    return values or list(default)


def _ml_quality_compact_payload(payload: dict[str, object]) -> dict[str, object]:
    """Reduce model diagnostics to the fields an operator needs before a rapid live preview."""

    provider_reports: dict[str, dict[str, object]] = {}
    retrain_commands: list[str] = []
    promote_commands: list[str] = []
    for report in list(payload.get("families", []) or []):
        if not isinstance(report, dict):
            continue
        provider = str(report.get("provider") or "unknown")
        provider_payload = provider_reports.setdefault(
            provider,
            {
                "provider": provider,
                "ready": True,
                "blockers": [],
                "blocked_family_count": 0,
                "blocked_families": [],
                "promoted_model_ids": {},
                "latest_candidate_ids": {},
                "ready_candidate_ids_by_family": {},
            },
        )
        blockers = [str(item) for item in list(report.get("blockers", []) or []) if str(item)]
        family_ready = bool(report.get("ready", False))
        if not family_ready:
            provider_payload["ready"] = False
        provider_payload["blockers"] = list(dict.fromkeys(list(provider_payload.get("blockers", []) or []) + blockers))

        promoted = report.get("promoted_model") if isinstance(report.get("promoted_model"), dict) else {}
        latest_candidates = [item for item in list(report.get("latest_candidates", []) or []) if isinstance(item, dict)]
        latest_ids = [item.get("id") for item in latest_candidates if item.get("id") is not None]
        latest_blockers: list[str] = []
        for candidate in latest_candidates:
            latest_blockers.extend(str(item) for item in list(candidate.get("blockers", []) or []) if str(item))
        ready_candidate_ids = [item for item in list(report.get("ready_candidate_ids", []) or []) if item is not None]
        family_name = str(report.get("family") or "")
        if promoted and promoted.get("id") is not None:
            promoted_model_ids = provider_payload["promoted_model_ids"]
            if isinstance(promoted_model_ids, dict):
                promoted_model_ids[family_name] = promoted.get("id")
        if latest_ids:
            latest_candidate_ids = provider_payload["latest_candidate_ids"]
            if isinstance(latest_candidate_ids, dict):
                latest_candidate_ids[family_name] = latest_ids
        if ready_candidate_ids:
            ready_candidates_by_family = provider_payload["ready_candidate_ids_by_family"]
            if isinstance(ready_candidates_by_family, dict):
                ready_candidates_by_family[family_name] = ready_candidate_ids
        if not family_ready:
            provider_payload["blocked_family_count"] = int(provider_payload.get("blocked_family_count", 0) or 0) + 1
            blocked_families = provider_payload["blocked_families"]
            if isinstance(blocked_families, list):
                blocked_families.append(
                    {
                        "family": family_name,
                        "blockers": blockers,
                        "candidate_blockers": list(dict.fromkeys(latest_blockers)),
                    }
                )

        commands = report.get("next_commands") if isinstance(report.get("next_commands"), dict) else {}
        train_command = commands.get("train_candidate")
        if not family_ready and isinstance(train_command, str):
            retrain_commands.append(train_command)
        for promote_command in list(commands.get("promote_ready_candidates", []) or []):
            if isinstance(promote_command, str):
                promote_commands.append(promote_command)

    return {
        "ready": bool(payload.get("ready", False)),
        "horizon": payload.get("horizon"),
        "provider_request": payload.get("provider_request"),
        "family_request": payload.get("family_request"),
        "blocker_counts": payload.get("blocker_counts", {}),
        "providers": list(provider_reports.values()),
        "next_commands": {
            **dict(payload.get("next_commands") if isinstance(payload.get("next_commands"), dict) else {}),
            "retrain_blocked_family_count": len(list(dict.fromkeys(retrain_commands))),
            "sample_retrain_blocked_families": list(dict.fromkeys(retrain_commands))[:4],
            "promote_ready_candidates": list(dict.fromkeys(promote_commands)),
        },
        "notes": payload.get("notes", []),
    }


def _ml_quality_family_report(
    *,
    horizon: str,
    provider: str,
    family: str,
    candidate_limit: int,
) -> dict[str, object]:
    family_key = str(family or "").lower()
    signal_model = get_service("ml_signal_model")
    decision_engine = get_service("ml_decision_engine")
    if family_key == ML_SIGNAL_FAMILY:
        readiness = signal_model.readiness(horizon, provider=provider)
        model_type = "pytorch_gru"
        diagnostics_for = lambda row: signal_model.promotion_diagnostics(row)
    else:
        readiness = decision_engine.family_readiness(family_key, horizon, provider=provider)
        model_type = family_key
        diagnostics_for = lambda row: decision_engine.promotion_diagnostics(row, family_key)
    candidates = (
        MLOfflineModel.query.filter_by(horizon=horizon, provider=provider, model_type=model_type)
        .order_by(MLOfflineModel.created_at.desc())
        .limit(max(1, int(candidate_limit or 1)))
        .all()
    )
    candidate_payloads = [
        _ml_quality_model_payload(row, diagnostics_for(row))
        for row in candidates
    ]
    promoted = readiness.get("promoted_model") if isinstance(readiness.get("promoted_model"), dict) else None
    promoted_id = int(promoted.get("id")) if promoted and promoted.get("id") is not None else None
    promoted_row = db.session.get(MLOfflineModel, promoted_id) if promoted_id else None
    promoted_diagnostics = diagnostics_for(promoted_row) if promoted_row is not None else None
    ready_candidates = [
        item
        for item in candidate_payloads
        if bool(item.get("ready", False)) and str(item.get("status") or "").lower() == "candidate"
    ]
    return {
        "provider": provider,
        "family": family_key,
        "model_type": model_type,
        "ready": bool(readiness.get("ready", False)),
        "blockers": list(readiness.get("blockers", []) or []),
        "promoted_model": _ml_quality_model_payload(promoted_row, promoted_diagnostics) if promoted_row is not None and promoted_diagnostics is not None else None,
        "latest_candidates": candidate_payloads,
        "ready_candidate_ids": [item.get("id") for item in ready_candidates],
        "next_commands": _ml_quality_next_commands(
            horizon=horizon,
            provider=provider,
            family=family_key,
            ready_candidate_ids=[int(item["id"]) for item in ready_candidates if item.get("id") is not None],
        ),
    }


def _ml_quality_model_payload(row: MLOfflineModel | None, diagnostics: dict[str, object] | None) -> dict[str, object] | None:
    if row is None:
        return None
    metrics = row.metrics if isinstance(row.metrics, dict) else {}
    walk_forward = metrics.get("walk_forward") if isinstance(metrics.get("walk_forward"), dict) else {}
    blockers = list((diagnostics or {}).get("blockers", []) or [])
    return {
        "id": row.id,
        "status": row.status,
        "provider": getattr(row, "provider", "global"),
        "model_type": row.model_type,
        "feature_schema_version": row.feature_schema_version,
        "ready": not blockers,
        "blockers": blockers,
        "training_rows": row.training_rows,
        "validation_rows": row.validation_rows,
        "validation_loss": row.validation_loss,
        "negative_error_rate": row.negative_error_rate,
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "action_rate": metrics.get("action_rate"),
            "false_positive_rate": metrics.get("false_positive_rate"),
            "walk_forward_false_positive_rate": walk_forward.get("false_positive_rate"),
            "walk_forward_mean_validation_loss": walk_forward.get("mean_validation_loss"),
            "target_distribution": metrics.get("target_distribution"),
        },
        "created_at": row.created_at,
        "promoted_at": row.promoted_at,
    }


def _ml_quality_next_commands(
    *,
    horizon: str,
    provider: str,
    family: str,
    ready_candidate_ids: list[int],
) -> dict[str, object]:
    if family == ML_SIGNAL_FAMILY:
        train = (
            "ML_SIGNAL_MODEL_ENABLED=true ML_OFFLINE_MODELS_ENABLED=true "
            f"flask train-ml-signal-model --horizon {horizon} --provider {provider} "
            "--model pytorch_gru --objective risk_adjusted --use-market-history "
            "--confirm TRAIN-ML-SIGNAL-MODEL"
        )
        promote_template = (
            f"flask promote-ml-signal-model --horizon {horizon} --provider {provider} "
            "--model-id <ready_candidate_id> --confirm PROMOTE-ML-SIGNAL-MODEL"
        )
    else:
        train = (
            "ML_ALL_AREAS_ENABLED=true ML_SIGNAL_MODEL_ENABLED=true ML_OFFLINE_MODELS_ENABLED=true "
            f"flask train-ml-suite --horizon {horizon} --provider {provider} "
            f"--model-family {family} --objective risk_adjusted --use-market-history "
            "--confirm TRAIN-ML-SUITE"
        )
        promote_template = (
            f"flask promote-ml-suite --horizon {horizon} --provider {provider} "
            "--model-id <ready_candidate_id> --confirm PROMOTE-ML-SUITE"
        )
    commands: dict[str, object] = {
        "train_candidate": train,
        "promote_template": promote_template,
    }
    if ready_candidate_ids:
        commands["promote_ready_candidates"] = [
            promote_template.replace("<ready_candidate_id>", str(model_id)) for model_id in ready_candidate_ids
        ]
    return commands


def _torch_runtime_readiness() -> dict[str, object]:
    """Return local PyTorch/MPS diagnostics without importing secrets or blocking disabled ML."""

    spec = importlib.util.find_spec("torch")
    if spec is None:
        return {
            "torch_installed": False,
            "torch_version": None,
            "mps_available": False,
            "status": "torch_missing",
            "blockers": ["torch_missing"],
        }
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {
            "torch_installed": False,
            "torch_version": None,
            "mps_available": False,
            "status": "torch_import_failed",
            "blockers": ["torch_import_failed"],
            "error": str(exc),
        }
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    try:
        mps_available = bool(mps_backend is not None and mps_backend.is_available())
    except Exception:  # noqa: BLE001
        mps_available = False
    return {
        "torch_installed": True,
        "torch_version": str(getattr(torch, "__version__", "")),
        "mps_available": mps_available,
        "status": "mps_ready" if mps_available else "mps_unavailable",
        "blockers": [] if mps_available else ["mps_unavailable"],
    }


def _ml_risk_preview_payload(
    *,
    ranking_id: int,
    horizon: str,
    user_id: int,
    connection_id: int | None,
    side: str,
) -> dict[str, object]:
    blockers: list[str] = []
    ranking = db.session.get(StrategyRanking, int(ranking_id))
    if ranking is None:
        return {
            "ready": False,
            "research_only": True,
            "submitted": False,
            "live_orders_created": False,
            "ranking_id": ranking_id,
            "blockers": ["ranking_not_found"],
        }
    if bool(ranking.rejected) or str(ranking.rejection_reason or "").strip():
        blockers.append("ranking_rejected")
    user = db.session.get(User, int(user_id))
    if user is None:
        blockers.append("user_missing")
    connection = _ml_live_vault_connection(user_id, "active", connection_id) if user is not None else None
    connection_blocker = _ml_live_vault_connection_blocker(connection) if connection is not None else "active_verified_connection_missing"
    if connection_blocker:
        blockers.append(connection_blocker)

    symbol = str(ranking.symbol or "").upper()
    side = str(side or "buy").lower()
    params = dict(ranking.parameters or {})
    horizon_key = str(horizon or "1h").lower()
    objective = "consistent_roi_1w" if horizon_key in {"1w", "7d"} else "extreme_roi_1h"
    target_roi = _ml_extreme_target_roi_pct(None, objective=objective, horizon=horizon_key)
    mid = 0.0
    try:
        mid = _float(get_service("market_data").get_mid_price(symbol, "live"))
    except Exception:  # noqa: BLE001
        mid = 0.0
    if mid <= 0:
        mid = _float(params.get("entry_price") or params.get("mid_price") or 100.0, 100.0)
    stop_pct = max(
        _float(params.get("stop_loss_pct")),
        _float(params.get("stop_loss_percentage")),
        0.005,
    )
    take_pct = max(
        _float(params.get("take_profit_pct")),
        _float(params.get("take_profit_percentage")),
        stop_pct * 1.5,
    )
    stop_loss = mid * (1.0 - stop_pct) if side == "buy" else mid * (1.0 + stop_pct)
    take_profit = mid * (1.0 + take_pct) if side == "buy" else mid * (1.0 - take_pct)
    cap = min(
        max(0.0, _float(current_app.config.get("ML_LIVE_HARD_CAP_USDC"), 10.0)),
        max(0.0, _float(ranking.allocation_amount_usd, 10.0)),
    )
    if cap <= 0:
        cap = max(0.0, _float(current_app.config.get("ML_LIVE_HARD_CAP_USDC"), 10.0))
    quantity = cap / mid if mid > 0 and cap > 0 else 0.0
    if quantity <= 0:
        blockers.append("quantity_unavailable")
    leverage = max(1.0, min(_float(ranking.leverage, 1.0), _float(current_app.config.get("MAX_LEVERAGE"), 1.0)))
    metadata = {
        **params,
        "ml_policy_required": True,
        "ml_governed_risk": True,
        "objective": objective,
        "target_roi_pct": target_roi,
        "optimizer_ranking_id": ranking.id,
        "optimizer_profile": ranking.profile,
        "strategy_name": ranking.strategy_name,
        "timeframe": ranking.timeframe,
        "duration_hours": 168 if horizon_key in {"1w", "7d"} else 1,
        "allocation_amount_usd": cap,
        "high_upside_profile": bool(params.get("high_upside_profile", False)),
        "profit_factor": ranking.profit_factor,
        "max_drawdown": ranking.max_drawdown,
        "recent_1h_return": ranking.recent_1h_return,
        "net_return_after_costs": ranking.net_return_after_costs,
        "liquidation_buffer_pct": ranking.liquidation_buffer_pct,
    }
    decisions: dict[str, object] = {}
    engine = get_service("ml_decision_engine")
    context = {
        **metadata,
        "symbol": symbol,
        "side": side,
        "horizon": horizon_key,
        "notional": cap,
        "reference_price": mid,
        "stop_loss_pct": stop_pct,
        "take_profit_pct": take_pct,
        "hard_max_leverage": current_app.config.get("MAX_LEVERAGE", 1.0),
        "ml_live_hard_cap_usdc": current_app.config.get("ML_LIVE_HARD_CAP_USDC", 10.0),
        "ml_live_hard_daily_loss_usdc": current_app.config.get("ML_LIVE_HARD_DAILY_LOSS_USDC", 0.50),
    }
    for family in (
        "pytorch_risk_policy",
        "pytorch_exit_policy",
        "pytorch_cap_policy",
        "pytorch_execution_policy",
        "pytorch_roi_target",
    ):
        decisions[family] = engine.decision(family, context, horizon=horizon_key)

    if blockers:
        risk_payload = {"approved": False, "rule_name": "preview_preflight_blocked", "reason": ", ".join(blockers)}
    else:
        intent = OrderIntent(
            symbol=symbol,
            side=side,
            quantity=quantity,
            mode="live",
            order_type="limit",
            limit_price=mid,
            reduce_only=False,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=ranking.strategy_name,
            timeframe=ranking.timeframe,
        slippage_pct=0.0,
            user_id=user_id,
            trading_connection_id=connection.id if connection else connection_id,
            metadata=metadata,
        )
        risk_payload = get_service("risk_engine").evaluate(
            intent,
            market_price=mid,
            has_trading_access=connection is not None and not connection_blocker,
        ).as_dict()
        if not bool(risk_payload.get("approved", False)):
            blockers.append(f"risk_preview_failed:{risk_payload.get('rule_name', 'unknown')}")

    blockers = list(dict.fromkeys(str(item) for item in blockers if str(item)))
    return {
        "ready": not blockers and bool(risk_payload.get("approved", False)),
        "research_only": True,
        "submitted": False,
        "live_orders_created": False,
        "ranking_id": ranking.id,
        "horizon": horizon_key,
        "objective": objective,
        "target_roi_pct": target_roi,
        "symbol": symbol,
        "side": side,
        "notional_usdc": cap,
        "leverage": leverage,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "ml_policy_decisions": decisions,
        "safety_envelope": dict((risk_payload.get("details") or {}).get("safety_envelope") or {}),
        "risk_decision": risk_payload,
        "blockers": blockers,
    }


def _ml_live_vault_preview_payload(
    *,
    user_id: int,
    provider: str,
    connection_id: int | None,
    cap_usdc: object,
    horizon: str,
    objective: str = "risk_adjusted",
    target_roi_pct: float | None = None,
    symbols: list[str] | None = None,
    timeout_seconds: float | None = None,
    record_audit: bool = True,
) -> dict[str, object]:
    """Build a one-shot ML vault preview without creating a cycle or submitting an order."""

    started_at = time.monotonic()
    blockers: list[str] = []
    provider = _normalize_provider(provider)
    horizon = str(horizon or "1h").strip().lower() or "1h"
    selected_objective = _ml_objective(objective)
    target_roi = _ml_extreme_target_roi_pct(target_roi_pct, objective=selected_objective, horizon=horizon)
    duration_hours = _ml_live_vault_horizon_hours(horizon)
    bucket = _duration_bucket(duration_hours)
    max_cap = max(0.0, _float(current_app.config.get("ML_LIVE_VAULT_MAX_CAP_USDC"), 10.0))

    user = db.session.get(User, user_id)
    if user is None:
        blockers.append("user_missing")
    connection = _ml_live_vault_connection(user_id, provider, connection_id) if user is not None else None
    if connection is None:
        blockers.append("active_verified_connection_missing")
    elif _ml_live_vault_connection_blocker(connection):
        blockers.append(_ml_live_vault_connection_blocker(connection))
    resolved_provider = normalize_exchange_provider(connection.provider if connection is not None else provider)
    collateral_asset = provider_collateral_asset(resolved_provider)
    cap_resolution = _ml_live_vault_resolve_cap(
        cap_usdc,
        user_id=user_id,
        connection=connection,
        provider=resolved_provider,
        collateral_asset=collateral_asset,
    )
    blockers.extend(cap_resolution.get("blockers", []) or [])
    cap = max(0.0, _float(cap_resolution.get("resolved_cap_usdc"), 0.0))
    if cap <= 0:
        blockers.append("cap_usdc_invalid")
    if max_cap > 0 and cap > max_cap + 1e-9:
        cap = max_cap

    strict_readiness = _production_readiness_payload_for(provider=resolved_provider, horizon=bucket)
    if not bool(strict_readiness.get("ready", False)):
        blockers.append("strict_production_readiness_failed")
    offline_ml = _offline_ml_readiness(bucket, require_blend=False, provider=resolved_provider)
    if bool(current_app.config.get("HIGH_UPSIDE_REQUIRE_PROMOTED_ML", True)) and not bool(offline_ml.get("ready", False)):
        blockers.append("promoted_offline_ml_missing")
    signal_readiness = _ml_signal_readiness(bucket, provider=resolved_provider)
    if bool(current_app.config.get("HIGH_UPSIDE_REQUIRE_ML_SIGNAL", True)) and not bool(signal_readiness.get("ready", False)):
        blockers.append("promoted_ml_signal_missing")
    ml_suite = _ml_suite_readiness(bucket, provider=resolved_provider)
    if bool(current_app.config.get("ML_REQUIRE_PROMOTED_FOR_LIVE", True)) and not bool(ml_suite.get("ready", False)):
        blockers.append("ml_suite_not_ready")
    extreme_readiness = _ml_extreme_upside_readiness(bucket, selected_objective, provider=resolved_provider)
    if selected_objective == "extreme_upside" and not bool(extreme_readiness.get("ready", False)):
        blockers.append("promoted_extreme_upside_model_missing")

    cap_readiness = _ml_live_vault_cap_readiness(bucket)
    blockers.extend(cap_readiness["blockers"])
    dynamic_caps = _ml_live_vault_dynamic_cap_payload(
        cap_usdc=cap,
        duration_bucket=bucket,
        objective=selected_objective,
        target_roi_pct=target_roi,
        provider=resolved_provider,
        context={},
    )
    blockers.extend(dynamic_caps.get("blockers", []) or [])
    cap = min(cap, _float(dynamic_caps.get("clipped_cap_usdc"), cap))

    requested_symbols = _normalize_symbols(symbols or [])
    discovery: dict[str, object] = {
        "ready": False,
        "accepted_ranking_ids": [],
        "accepted_rankings": [],
        "blockers": ["discovery_skipped_due_to_preflight_blockers"] if blockers else [],
    }
    if not blockers:
        try:
            discovery = _discover_high_upside_vault_candidates_payload(
                user_id=user_id,
                provider_scope=provider if provider != "active" else "active_provider",
                providers=[] if provider == "active" else [provider],
                symbols=requested_symbols,
                timeframes=[bucket if bucket in {"5m", "15m", "1h"} else "1h"],
                profiles=["aggressive_1h"],
                strategy_names=[],
                max_symbols=1,
                max_sweeps=1,
                max_parameter_sets=4,
                allocation_amount_usd=cap,
                lock_duration_hours=duration_hours,
                timeout_seconds=timeout_seconds or min(_high_upside_timeout_seconds(None), 90.0),
                started_at=started_at,
                progress_state={"phase": "ml_live_vault_preview_discovery"},
                run_backtests=True,
                objective=selected_objective,
                target_roi_pct=target_roi,
            )
        except Exception as exc:  # noqa: BLE001
            discovery = {
                "ready": False,
                "accepted_ranking_ids": [],
                "accepted_rankings": [],
                "blockers": ["high_upside_discovery_failed"],
                "error": _high_upside_sanitize_error(exc),
            }
    accepted_ids = [int(value) for value in list(discovery.get("accepted_ranking_ids", []) or []) if value is not None]
    if not accepted_ids:
        blockers.append("accepted_high_upside_ranking_missing")

    selected_ranking = db.session.get(StrategyRanking, accepted_ids[0]) if accepted_ids else None
    if selected_ranking is not None and (bool(selected_ranking.rejected) or str(selected_ranking.rejection_reason or "").strip()):
        blockers.append("selected_ranking_rejected")
    if selected_ranking is not None and normalize_exchange_provider(getattr(selected_ranking, "provider", "global")) != resolved_provider:
        blockers.append("selected_ranking_provider_mismatch")
    selection_payload: dict[str, object] = {}
    selected_leg: dict[str, object] = {}
    signal_preview: dict[str, object] = {}
    risk_payload: dict[str, object] = {"approved": False, "rule_name": "not_evaluated"}
    funds_readiness: dict[str, object] = {"ready": False, "blockers": ["accepted_ranking_missing"]}

    if selected_ranking is not None and connection is not None and not blockers:
        funds_readiness = _live_funds_readiness_payload(
            provider=provider,
            user_id=user_id,
            ranking_id=selected_ranking.id,
            connection_id=connection.id,
        )
        if not bool(funds_readiness.get("ready", False)):
            blockers.append("live_funds_readiness_failed")
        selection_payload, selected_leg = _ml_live_vault_selection_payload(
            selected_ranking=selected_ranking,
            user_id=user_id,
            connection_id=connection.id,
            provider=resolved_provider,
            collateral_asset=collateral_asset,
            cap_usdc=cap,
            duration_hours=duration_hours,
            requested_symbols=requested_symbols,
        )
        blockers.extend(selection_payload.get("blockers", []) or [])
        if selected_leg:
            dynamic_caps = _ml_live_vault_dynamic_cap_payload(
                cap_usdc=cap,
                duration_bucket=bucket,
                objective=selected_objective,
                target_roi_pct=target_roi,
                provider=resolved_provider,
                context={
                    **dict(selected_leg.get("parameters") or {}),
                    **provider_feature_context(resolved_provider),
                    "symbol": selected_leg.get("symbol"),
                    "timeframe": selected_leg.get("timeframe"),
                    "allocation_amount_usd": selected_leg.get("allocation_cap_usd"),
                    "allocation_budget_usdc": cap,
                    "hard_max_leverage": selected_leg.get("leverage", 1.0),
                },
            )
            blockers.extend(dynamic_caps.get("blockers", []) or [])
            clipped_cap = _float(dynamic_caps.get("clipped_cap_usdc"), cap)
            selected_leg["allocation_cap_usd"] = min(_float(selected_leg.get("allocation_cap_usd"), cap), clipped_cap)
            params = dict(selected_leg.get("parameters") or {})
            params["allocation_cap_usd"] = selected_leg["allocation_cap_usd"]
            params["ml_dynamic_cap_suggestion"] = dynamic_caps
            selected_leg["parameters"] = params
            signal_preview, risk_payload = _ml_live_vault_risk_preview(
                selected_leg=selected_leg,
                user_id=user_id,
                connection_id=connection.id,
                cap_usdc=cap,
                duration_hours=duration_hours,
            )
            blockers.extend(signal_preview.get("blockers", []) or [])
            if not bool(risk_payload.get("approved", False)):
                blockers.append(f"risk_dry_run_failed:{risk_payload.get('rule_name', 'unknown')}")

    blockers = list(dict.fromkeys([str(item) for item in blockers if str(item).strip()]))
    ready = not blockers and bool(selected_leg) and bool(risk_payload.get("approved", False))
    payload: dict[str, object] = {
        "ready": ready,
        "preview_only": True,
        "research_only": False,
        "submitted": False,
        "live_orders_created": False,
        "user_id": user_id,
        "provider": resolved_provider,
        "requested_provider": provider,
        "collateral_asset": collateral_asset,
        "connection_id": connection.id if connection else None,
        "cap_usdc": cap,
        "cap_usd": cap,
        "cap_resolution": cap_resolution,
        "cap_collateral_asset": collateral_asset,
        "horizon": horizon,
        "objective": selected_objective,
        "target_roi_pct": target_roi,
        "target_roi_policy": _ml_extreme_target_policy(selected_objective, target_roi),
        "duration_hours": duration_hours,
        "duration_bucket": bucket,
        "strict_readiness": strict_readiness,
        "funds_readiness": funds_readiness,
        "offline_ml_readiness": offline_ml,
        "ml_signal_readiness": signal_readiness,
        "ml_suite_readiness": ml_suite,
        "extreme_upside_readiness": extreme_readiness,
        "torch_runtime": ml_suite.get("torch_runtime", _torch_runtime_readiness()),
        "cap_readiness": cap_readiness,
        "dynamic_cap_suggestion": dynamic_caps,
        "clipped_cap_usdc": dynamic_caps.get("clipped_cap_usdc"),
        "clipped_cap_usd": dynamic_caps.get("clipped_cap_usdc"),
        "suggested_leverage": dynamic_caps.get("suggested_leverage"),
        "clipped_leverage": dynamic_caps.get("clipped_leverage"),
        "discovery": _ml_live_vault_compact_discovery(discovery),
        "selected_ranking": _ml_live_vault_ranking_payload(selected_ranking),
        "selected_cycle_intent": selection_payload,
        "selected_leg": selected_leg,
        "ml_model_ids": _ml_live_vault_model_ids(offline_ml, signal_readiness, ml_suite),
        "ml_signal_decision": signal_preview.get("ml_signal_decision", {}),
        "ml_extreme_upside_decision": dynamic_caps.get("extreme_upside_decision", {}),
        "ml_fibonacci_zones": dynamic_caps.get("fibonacci_zones", {}),
        "stop_loss": signal_preview.get("stop_loss"),
        "take_profit": signal_preview.get("take_profit"),
        "leverage": selected_leg.get("leverage"),
        "allocation_budget_usdc": cap,
        "allocation_budget_usd": cap,
        "risk_decision": risk_payload,
        "blockers": blockers,
        "next_commands": {},
        "runtime": {"elapsed_seconds": max(time.monotonic() - started_at, 0.0)},
    }
    if ready:
        payload["next_commands"] = {
            "one_shot": (
                "flask ml-live-vault-one-shot --preview-audit-id <preview_audit_id> "
                f"--user-id {user_id} --confirm {current_app.config.get('ML_LIVE_VAULT_EXACT_CONFIRMATION', 'ML-LIVE-VAULT-10USDC')}"
            ),
            "warning": "This starts a live vault cycle only if live env/config, DB confirmations, ML, caps, and RiskEngine gates still pass.",
        }
    if record_audit:
        audit = AuditLog(
            category="ml_live_vault",
            action="ml_live_vault_preview",
            message="ML live vault one-shot preview evaluated.",
            user_id=user_id,
            trading_connection_id=connection.id if connection else None,
        )
        audit.details = _json_safe(
            {
                "ready": ready,
                "payload": payload,
                "blockers": blockers,
                "expires_at": (datetime.utcnow() + timedelta(seconds=_ml_live_vault_preview_max_age_seconds())).isoformat(),
            }
        )
        db.session.add(audit)
        db.session.commit()
        payload["preview_audit_id"] = audit.id
        if ready:
            exact_confirmation = (
                current_app.config.get("ONE_H10_EXACT_CONFIRMATION", "ONE-H10-LIVE")
                if _is_one_h10_preview_payload(payload)
                else current_app.config.get("ML_LIVE_VAULT_EXACT_CONFIRMATION", "ML-LIVE-VAULT-10USDC")
            )
            payload["next_commands"]["one_shot"] = (
                f"flask ml-live-vault-one-shot --preview-audit-id {audit.id} "
                f"--user-id {user_id} --confirm {exact_confirmation}"
            )
    return payload


def _is_one_h10_preview_payload(payload: dict[str, object]) -> bool:
    objective = _ml_objective(payload.get("objective"))
    horizon = str(payload.get("horizon") or payload.get("duration_bucket") or "").strip().lower()
    return objective == "one_h10" or horizon == "1h10" or bool(payload.get("one_h10_vault"))


def _ml_live_vault_one_shot_payload(*, preview_audit_id: int, user_id: int, confirm: str) -> dict[str, object]:
    blockers: list[str] = []
    audit = db.session.get(AuditLog, preview_audit_id)
    if audit is None or audit.action != "ml_live_vault_preview":
        blockers.append("preview_audit_missing")
        preview_payload: dict[str, object] = {}
    else:
        preview_payload = dict((audit.details or {}).get("payload") or {})
        expected = str(
            current_app.config.get("ONE_H10_EXACT_CONFIRMATION", "ONE-H10-LIVE")
            if _is_one_h10_preview_payload(preview_payload)
            else current_app.config.get("ML_LIVE_VAULT_EXACT_CONFIRMATION", "ML-LIVE-VAULT-10USDC")
            or ""
        ).strip()
        if confirm != expected:
            blockers.append("exact_confirmation_missing")
        if audit.user_id != user_id:
            blockers.append("preview_user_mismatch")
        age = (datetime.utcnow() - audit.created_at).total_seconds()
        if age > _ml_live_vault_preview_max_age_seconds():
            blockers.append("preview_stale")
        if not bool(preview_payload.get("ready", False)):
            blockers.append("preview_not_ready")

    blockers.extend(_ml_live_vault_submit_gate_blockers(preview_payload, user_id))
    attempt_audit = AuditLog(
        category="ml_live_vault",
        action="ml_live_vault_one_shot_attempt",
        message="ML live vault one-shot submit attempt evaluated.",
        user_id=user_id,
        trading_connection_id=preview_payload.get("connection_id") if preview_payload else None,
    )
    attempt_audit.details = _json_safe(
        {
            "preview_audit_id": preview_audit_id,
            "blockers": blockers,
            "confirmed": "exact_confirmation_missing" not in blockers,
        }
    )
    db.session.add(attempt_audit)
    db.session.commit()

    refreshed: dict[str, object] = {}
    if not blockers and preview_payload:
        refreshed = _ml_live_vault_preview_payload(
            user_id=user_id,
            provider=str(preview_payload.get("provider") or "active"),
            connection_id=int(preview_payload.get("connection_id") or 0) or None,
            cap_usdc=_float(preview_payload.get("cap_usdc"), 10.0),
            horizon=str(preview_payload.get("horizon") or "1h"),
            objective=str(preview_payload.get("objective") or "risk_adjusted"),
            target_roi_pct=_float(preview_payload.get("target_roi_pct"), None),
            symbols=[str((preview_payload.get("selected_ranking") or {}).get("symbol") or "")],
            timeout_seconds=min(_high_upside_timeout_seconds(None), 90.0),
            record_audit=False,
        )
        if not bool(refreshed.get("ready", False)):
            blockers.append("fresh_preview_not_ready")
            blockers.extend(list(refreshed.get("blockers", []) or []))
        blockers.extend(_ml_live_vault_submit_gate_blockers(refreshed, user_id))

    blockers = list(dict.fromkeys([str(item) for item in blockers if str(item).strip()]))
    if blockers:
        blocked = AuditLog(
            category="ml_live_vault",
            action="ml_live_vault_one_shot_blocked",
            message="ML live vault one-shot blocked before cycle start.",
            user_id=user_id,
            trading_connection_id=preview_payload.get("connection_id") if preview_payload else None,
        )
        blocked.details = _json_safe(
            {
                "preview_audit_id": preview_audit_id,
                "attempt_audit_id": attempt_audit.id,
                "blockers": blockers,
                "fresh_preview": refreshed,
            }
        )
        db.session.add(blocked)
        db.session.commit()
        return {
            "ready": False,
            "submitted": False,
            "cycle_started": False,
            "live_orders_created": False,
            "preview_audit_id": preview_audit_id,
            "attempt_audit_id": attempt_audit.id,
            "blockers": blockers,
            "fresh_preview": refreshed,
        }

    cycle_payload = _create_ml_live_vault_cycle(refreshed or preview_payload, preview_audit_id)
    return {
        **cycle_payload,
        "attempt_audit_id": attempt_audit.id,
        "preview_audit_id": preview_audit_id,
    }


def _ml_auto_vault_cycle_payload(
    *,
    user_id: int,
    provider: str,
    connection_id: int | None,
    cap_usdc: object,
    horizon: str,
    objective: str,
    target_roi_pct: float | None,
    symbols: list[str],
    timeout_seconds: float | None,
    confirm: str,
) -> dict[str, object]:
    blockers: list[str] = []
    expected = str(current_app.config.get("ML_AUTO_VAULT_EXACT_CONFIRMATION", "ML-AUTO-VAULT-LIVE") or "").strip()
    if confirm != expected:
        blockers.append("exact_confirmation_missing")
    if not bool(current_app.config.get("ML_AUTO_VAULT_LIVE_ENABLED", False)):
        blockers.append("ML_AUTO_VAULT_LIVE_ENABLED=false")
    if not bool(current_app.config.get("ML_LIVE_VAULT_ONE_SHOT_ENABLED", False)):
        blockers.append("ML_LIVE_VAULT_ONE_SHOT_ENABLED=false")
    if not bool(current_app.config.get("ML_DYNAMIC_CAPS_ENABLED", False)):
        blockers.append("ML_DYNAMIC_CAPS_ENABLED=false")

    preview = _ml_live_vault_preview_payload(
        user_id=user_id,
        provider=provider,
        connection_id=connection_id,
        cap_usdc=cap_usdc,
        horizon=horizon,
        objective=objective,
        target_roi_pct=target_roi_pct,
        symbols=symbols,
        timeout_seconds=timeout_seconds,
        record_audit=True,
    )
    preview_audit_id = int(_float(preview.get("preview_audit_id"), 0.0))
    if not bool(preview.get("ready", False)):
        blockers.append("preview_not_ready")
        blockers.extend(list(preview.get("blockers", []) or []))
    blockers.extend(_ml_live_vault_submit_gate_blockers(preview, user_id))
    blockers = list(dict.fromkeys([str(item) for item in blockers if str(item).strip()]))
    attempt = AuditLog(
        category="ml_live_vault",
        action="ml_auto_vault_cycle_attempt",
        message="ML auto-vault cycle attempt evaluated.",
        user_id=user_id,
        trading_connection_id=preview.get("connection_id") if isinstance(preview, dict) else None,
    )
    attempt.details = _json_safe(
        {
            "preview_audit_id": preview_audit_id or None,
            "objective": _ml_objective(objective),
            "target_roi_pct": _ml_extreme_target_roi_pct(target_roi_pct, objective=objective),
            "blockers": blockers,
            "confirmed": confirm == expected,
        }
    )
    db.session.add(attempt)
    db.session.commit()
    if blockers:
        return {
            "ready": False,
            "submitted": False,
            "cycle_started": False,
            "live_orders_created": False,
            "preview_audit_id": preview_audit_id or None,
            "attempt_audit_id": attempt.id,
            "objective": _ml_objective(objective),
            "target_roi_pct": _ml_extreme_target_roi_pct(target_roi_pct, objective=objective),
            "blockers": blockers,
            "preview": preview,
            "live_order_path": "blocked_before_StrategyManager",
        }
    cycle_payload = _create_ml_live_vault_cycle(preview, preview_audit_id)
    return {
        **cycle_payload,
        "attempt_audit_id": attempt.id,
        "preview_audit_id": preview_audit_id,
        "objective": _ml_objective(objective),
        "target_roi_pct": _ml_extreme_target_roi_pct(target_roi_pct, objective=objective),
        "auto_vault_live_enabled": True,
    }


def _ml_vault_tick_payload(
    *,
    user_id: int,
    provider_scope: str | None,
    cap_usdc: float,
    objective: str,
    horizon: str,
    timeout_seconds: float | None,
    confirm: str,
) -> dict[str, object]:
    started_at = time.monotonic()
    scope = _normalize_provider(provider_scope or current_app.config.get("ML_VAULT_PROVIDER_SCOPE", "all"))
    selected_objective = _ml_objective(objective)
    target_roi = _ml_extreme_target_roi_pct(None, objective=selected_objective, horizon=horizon)
    cap = min(max(0.0, _float(cap_usdc, 10.0)), 10.0, _float(current_app.config.get("ML_VAULT_MAX_CAP_USDC"), 10.0))
    blockers: list[str] = []
    if confirm != "ML-VAULT-TICK":
        blockers.append("exact_confirmation_missing")
    if not bool(current_app.config.get("ML_CONTINUOUS_VAULT_ENABLED", False)):
        blockers.append("ML_CONTINUOUS_VAULT_ENABLED=false")
    if not bool(current_app.config.get("ML_VAULT_TICK_ENABLED", False)):
        blockers.append("ML_VAULT_TICK_ENABLED=false")
    if not bool(current_app.config.get("ML_AUTO_VAULT_LIVE_ENABLED", False)):
        blockers.append("ML_AUTO_VAULT_LIVE_ENABLED=false")
    if cap <= 0 or cap > 10.0:
        blockers.append("ML_VAULT_MAX_CAP_USDC_must_be_0_to_10")
    if _float(current_app.config.get("ML_VAULT_MAX_DAILY_LOSS_USDC"), 0.50) > 0.50:
        blockers.append("ML_VAULT_MAX_DAILY_LOSS_USDC_above_0.50")
    if int(current_app.config.get("ML_VAULT_MAX_ACTIVE_CYCLES", 1) or 1) != 1:
        blockers.append("ML_VAULT_MAX_ACTIVE_CYCLES_must_equal_1")
    if int(current_app.config.get("ML_VAULT_MAX_LIVE_CYCLES_PER_DAY", 1) or 1) != 1:
        blockers.append("ML_VAULT_MAX_LIVE_CYCLES_PER_DAY_must_equal_1")

    adaptive = _ml_vault_adaptive_cadence_payload()
    if adaptive.get("blocker"):
        blockers.append(str(adaptive["blocker"]))

    preview: dict[str, object] = {}
    exchange_gate: dict[str, object] = {"ready": False, "blockers": ["preview_not_evaluated"]}
    preview_audit_id = 0
    if not blockers:
        preview_provider = "active" if scope in {"all", "multi_provider", "active_provider"} else scope
        preview = _ml_live_vault_preview_payload(
            user_id=user_id,
            provider=preview_provider,
            connection_id=None,
            cap_usdc=cap,
            horizon=horizon,
            objective=selected_objective,
            target_roi_pct=target_roi,
            symbols=[],
            timeout_seconds=timeout_seconds,
            record_audit=True,
        )
        preview_audit_id = int(_float(preview.get("preview_audit_id"), 0.0))
        if not bool(preview.get("ready", False)):
            blockers.append("preview_not_ready")
            blockers.extend(list(preview.get("blockers", []) or []))
        exchange_gate = _ml_vault_exchange_leverage_gate(preview)
        if not bool(exchange_gate.get("ready", False)):
            blockers.extend(list(exchange_gate.get("blockers", []) or []))
        blockers.extend(_ml_live_vault_submit_gate_blockers(preview, user_id))

    blockers = list(dict.fromkeys([str(item) for item in blockers if str(item).strip()]))
    decision = {
        "ready": not blockers,
        "submitted": False,
        "cycle_started": False,
        "live_orders_created": False,
        "user_id": user_id,
        "provider_scope": scope,
        "cap_usdc": cap,
        "objective": selected_objective,
        "target_roi_pct": target_roi,
        "ml_vault_tick": {
            "enabled": bool(current_app.config.get("ML_VAULT_TICK_ENABLED", False)),
            "continuous_enabled": bool(current_app.config.get("ML_CONTINUOUS_VAULT_ENABLED", False)),
            "one_tick": True,
            "state_keys": [
                "ml_vault_last_tick_at",
                "ml_vault_last_live_cycle_at",
                "ml_vault_last_decision",
                "ml_vault_provider_backoff_until",
            ],
        },
        "adaptive_cadence": adaptive,
        "preview_audit_id": preview_audit_id or None,
        "preview": preview,
        "exchange_leverage_gate": exchange_gate,
        "blockers": blockers,
        "live_order_path": "blocked_before_StrategyManager" if blockers else "StrategyManager.start -> StrategyRunner -> OrderManager -> RiskEngine",
        "runtime": {"elapsed_seconds": max(time.monotonic() - started_at, 0.0)},
    }
    Setting.set_json("ml_vault_last_tick_at", datetime.utcnow().isoformat() + "Z")
    Setting.set_json("ml_vault_last_decision", _json_safe(decision))
    audit = AuditLog(
        category="ml_live_vault",
        action="ml_vault_tick",
        message="Continuous ML vault tick evaluated.",
        user_id=user_id,
        trading_connection_id=preview.get("connection_id") if isinstance(preview, dict) else None,
    )
    audit.details = _json_safe(decision)
    db.session.add(audit)
    db.session.commit()
    decision["tick_audit_id"] = audit.id
    if blockers:
        return decision

    cycle_payload = _create_ml_live_vault_cycle(preview, preview_audit_id)
    Setting.set_json("ml_vault_last_live_cycle_at", datetime.utcnow().isoformat() + "Z")
    Setting.set_json("ml_vault_last_decision", _json_safe({**decision, **cycle_payload}))
    db.session.commit()
    return {
        **decision,
        **cycle_payload,
        "tick_audit_id": audit.id,
        "preview_audit_id": preview_audit_id,
    }


def _ml_vault_adaptive_cadence_payload() -> dict[str, object]:
    last_tick_raw = Setting.get_json("ml_vault_last_tick_at", "")
    last_tick = _parse_iso_datetime(last_tick_raw)
    min_interval = _float(current_app.config.get("HIGH_UPSIDE_MIN_SCAN_INTERVAL_SECONDS"), 3600.0)
    fast_interval = _float(current_app.config.get("HIGH_UPSIDE_FAST_SCAN_INTERVAL_SECONDS"), 300.0)
    recommended = min_interval
    if bool(current_app.config.get("HIGH_UPSIDE_ADAPTIVE_CADENCE_ENABLED", False)) and bool(current_app.config.get("ML_ALL_AREAS_ENABLED", False)):
        recommended = min(min_interval, fast_interval)
    elapsed = (datetime.utcnow() - last_tick).total_seconds() if last_tick else None
    blocker = "ml_vault_tick_cadence_wait" if elapsed is not None and elapsed < recommended else ""
    return {
        "adaptive": bool(current_app.config.get("HIGH_UPSIDE_ADAPTIVE_CADENCE_ENABLED", False)),
        "last_tick_at": last_tick_raw,
        "elapsed_seconds": elapsed,
        "recommended_interval_seconds": recommended,
        "blocker": blocker,
        "policy": "ML cadence may speed scans within configured bounds; cooldowns and provider backoff can only slow execution.",
    }


def _parse_iso_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _ml_vault_exchange_leverage_gate(preview_payload: dict[str, object]) -> dict[str, object]:
    blockers: list[str] = []
    policy = str(current_app.config.get("ML_VAULT_LEVERAGE_POLICY", "exchange_max_gated") or "").strip().lower()
    if policy != "exchange_max_gated":
        blockers.append("ML_VAULT_LEVERAGE_POLICY_must_be_exchange_max_gated")
    connection_id = int(_float(preview_payload.get("connection_id"), 0.0))
    connection = db.session.get(TradingConnection, connection_id) if connection_id else None
    metadata = connection.provider_metadata if connection is not None and isinstance(connection.provider_metadata, dict) else {}
    exchange_max = _float(
        metadata.get("exchange_max_leverage", metadata.get("max_leverage", metadata.get("provider_max_leverage"))),
        0.0,
    )
    leg = preview_payload.get("selected_leg") if isinstance(preview_payload.get("selected_leg"), dict) else {}
    leverage = _float(leg.get("leverage") if isinstance(leg, dict) else 1.0, 1.0)
    selected = preview_payload.get("selected_ranking") if isinstance(preview_payload.get("selected_ranking"), dict) else {}
    ranking = db.session.get(StrategyRanking, int(_float(selected.get("id") if isinstance(selected, dict) else 0, 0.0))) if selected else None
    liquidation_buffer = _float(
        (ranking.liquidation_buffer_pct if ranking is not None else 0.0),
        _float((leg.get("parameters") or {}).get("liquidation_buffer_pct") if isinstance(leg, dict) else 0.0, 0.0),
    )
    min_buffer = _float(current_app.config.get("ML_VAULT_MIN_LIQUIDATION_BUFFER_PCT"), 0.20)
    if exchange_max <= 0:
        blockers.append("exchange_max_leverage_unavailable")
    elif leverage > exchange_max:
        blockers.append("leverage_above_exchange_max")
    if leverage <= 0:
        blockers.append("leverage_invalid")
    if liquidation_buffer < min_buffer:
        blockers.append("liquidation_buffer_below_ml_vault_minimum")
    if _float(preview_payload.get("cap_usdc"), 0.0) > 10.0:
        blockers.append("allocation_budget_above_10_usdc")
    return {
        "ready": not blockers,
        "policy": policy,
        "exchange_max_leverage": exchange_max,
        "requested_leverage": leverage,
        "liquidation_buffer_pct": liquidation_buffer,
        "min_liquidation_buffer_pct": min_buffer,
        "blockers": list(dict.fromkeys(blockers)),
    }


def _ml_live_vault_horizon_hours(horizon: str) -> int:
    value = str(horizon or "1h").strip().lower()
    if value.endswith("m"):
        return 1
    if value.endswith("h"):
        return max(1, int(_float(value[:-1], 1.0)))
    if value.endswith("d"):
        return max(1, int(_float(value[:-1], 1.0) * 24))
    return max(1, int(_float(value, 1.0)))


def _ml_live_vault_preview_max_age_seconds() -> float:
    return max(30.0, _float(current_app.config.get("ML_LIVE_VAULT_PREVIEW_MAX_AGE_SECONDS"), 600.0))


def _ml_live_vault_connection(user_id: int, provider: str, connection_id: int | None) -> TradingConnection | None:
    query = TradingConnection.query.filter_by(user_id=user_id, verification_status="verified")
    if connection_id is not None:
        query = query.filter_by(id=connection_id)
    else:
        query = query.filter_by(is_active=True)
    if provider and provider != "active":
        query = query.filter_by(provider=_normalize_provider(provider))
    return query.order_by(TradingConnection.updated_at.desc()).first()


def _ml_live_vault_connection_blocker(connection: TradingConnection) -> str:
    metadata = connection.provider_metadata or {}
    if metadata.get("can_trade") is False:
        return "connection_can_trade_false"
    try:
        service = get_service("trading_connections")
        if not bool(service.connector_for_user(connection.user_id, connection.id).can_trade("live")):
            return "connection_can_trade_false"
    except Exception as exc:  # noqa: BLE001
        parsed = parse_exchange_failure(str(exc))
        if parsed.get("provider_code") == "400006" or parsed.get("ip_whitelist_blocked"):
            return "connection_ip_whitelist_blocked"
        return "connection_can_trade_unavailable"
    return ""


def _ml_live_vault_resolve_cap(
    requested: object,
    *,
    user_id: int,
    connection: TradingConnection | None,
    provider: str,
    collateral_asset: str,
) -> dict[str, object]:
    request_text = str(requested if requested is not None else "").strip().lower()
    max_cap = max(0.0, _float(current_app.config.get("ML_LIVE_VAULT_MAX_CAP_USDC"), 10.0))
    blockers: list[str] = []
    local_available = _wallet_available_balance(user_id, collateral_asset)
    provider_available = 0.0
    if connection is not None:
        try:
            snapshot = get_service("trading_connections").account_snapshot(user_id, "live", connection.id)
            provider_available = _provider_available_collateral(collateral_asset, list(snapshot.balances or []))
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"provider_balance_unavailable:{_high_upside_sanitize_error(exc)}")
    elif request_text == "all":
        blockers.append("connection_required_for_all_cap")
    if request_text == "all":
        requested_cap = min(local_available, provider_available)
    else:
        requested_cap = _float(requested, 0.0)
    if requested_cap <= 0:
        blockers.append("cap_usdc_invalid")
    clipped = requested_cap
    if max_cap > 0:
        clipped = min(clipped, max_cap)
    if request_text == "all" and local_available <= 0:
        blockers.append("local_wallet_collateral_unavailable")
    if request_text == "all" and provider_available <= 0:
        blockers.append("provider_collateral_unavailable")
    return {
        "requested": str(requested),
        "requested_all": request_text == "all",
        "collateral_asset": str(collateral_asset).upper(),
        "local_available": local_available,
        "provider_available": provider_available,
        "configured_max_cap_usdc": max_cap,
        "requested_cap_usdc": requested_cap,
        "resolved_cap_usdc": max(0.0, clipped),
        "blockers": list(dict.fromkeys(blockers)),
        "policy": "all resolves to min(local unlocked collateral, provider available collateral), then configured caps clip notional",
    }


def _wallet_available_balance(user_id: int, asset: str) -> float:
    balance = WalletBalance.query.filter_by(user_id=user_id, asset=str(asset or "").upper()).one_or_none()
    return max(0.0, _float(balance.available_balance if balance is not None else 0.0, 0.0))


def _provider_available_collateral(asset: str, balances: list[dict[str, object]]) -> float:
    asset_key = str(asset or "").upper()
    total = 0.0
    for row in balances:
        if str(row.get("asset") or "").upper() != asset_key:
            continue
        value = _float(row.get("withdrawable"), _float(row.get("value"), 0.0))
        if value > 0:
            total += value
    return max(0.0, total)


def _ml_live_vault_cap_readiness(bucket: str) -> dict[str, object]:
    blockers: list[str] = []
    daily_loss = _float(current_app.config.get("HIGH_UPSIDE_MAX_DAILY_LOSS_USDC"), 0.0)
    ml_daily_loss = _float(current_app.config.get("ML_LIVE_VAULT_MAX_DAILY_LOSS_USDC"), 0.50)
    duration_cap = _duration_cap("HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION", bucket)
    max_daily_cycles = int(_float(current_app.config.get("HIGH_UPSIDE_MAX_LIVE_CYCLES_PER_DAY"), 1))
    max_active_cycles = int(_float(current_app.config.get("HIGH_UPSIDE_MAX_ACTIVE_CYCLES"), 1))
    if duration_cap <= 0 or duration_cap > 10.0:
        blockers.append("HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION_must_be_0_to_10")
    if daily_loss <= 0 or daily_loss > min(0.50, ml_daily_loss):
        blockers.append("HIGH_UPSIDE_MAX_DAILY_LOSS_USDC_must_be_0_to_0.50")
    if max_daily_cycles != 1:
        blockers.append("HIGH_UPSIDE_MAX_LIVE_CYCLES_PER_DAY_must_equal_1")
    if max_active_cycles != 1:
        blockers.append("HIGH_UPSIDE_MAX_ACTIVE_CYCLES_must_equal_1")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "duration_cap_usdc": duration_cap,
        "max_daily_loss_usdc": daily_loss,
        "ml_max_daily_loss_usdc": ml_daily_loss,
        "max_live_cycles_per_day": max_daily_cycles,
        "max_active_cycles": max_active_cycles,
    }


def _ml_live_vault_dynamic_cap_payload(
    *,
    cap_usdc: float,
    duration_bucket: str,
    objective: str,
    target_roi_pct: float,
    provider: str = "global",
    context: dict[str, object],
) -> dict[str, object]:
    """Return ML sizing suggestions clipped by deterministic live ceilings."""

    blockers: list[str] = []
    provider_key = normalize_exchange_provider(
        context.get("provider") or context.get("execution_venue") or provider
    )
    hard_caps = {
        "requested_cap_usdc": max(0.0, _float(cap_usdc, 0.0)),
        "ml_live_vault_max_cap_usdc": max(0.0, _float(current_app.config.get("ML_LIVE_VAULT_MAX_CAP_USDC"), 10.0)),
        "duration_cap_usdc": max(0.0, _duration_cap("HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION", duration_bucket)),
        "absolute_test_cap_usdc": 10.0,
    }
    positive_caps = [value for value in hard_caps.values() if value > 0]
    if len(positive_caps) < len(hard_caps):
        blockers.append("deterministic_hard_caps_missing")
    clipped_cap = min(positive_caps) if positive_caps else 0.0
    suggested_notional = min(max(0.0, _float(cap_usdc, 0.0)), clipped_cap) if clipped_cap > 0 else 0.0
    suggested_leverage = min(max(1.0, _float(context.get("leverage"), 1.0)), 1.0)
    suggested_risk_pct = min(
        max(0.0, _float(context.get("risk_pct"), _float(current_app.config.get("RISK_PER_TRADE_PCT"), 0.01))),
        0.01,
    )
    extreme_decision: dict[str, object] = {}
    fibonacci_decision: dict[str, object] = {}
    if _ml_objective(objective) == "extreme_upside":
        decision_context = {
            **provider_feature_context(provider_key),
            **context,
            "objective": "extreme_upside",
            "target_roi_pct": target_roi_pct,
            "allocation_amount_usd": cap_usdc,
            "allocation_budget_usdc": clipped_cap,
            "hard_max_leverage": 1.0,
            "lock_duration_hours": _ml_live_vault_horizon_hours(duration_bucket),
        }
        try:
            extreme_decision = dict(
                get_service("ml_decision_engine").decision(
                    "pytorch_extreme_upside",
                    decision_context,
                    horizon=duration_bucket,
                )
            )
        except Exception as exc:  # noqa: BLE001
            extreme_decision = {"ready": False, "blockers": [_high_upside_sanitize_error(exc)], "raw": {}}
        try:
            fibonacci_decision = dict(
                get_service("ml_decision_engine").decision(
                    "pytorch_fibonacci",
                    decision_context,
                    horizon=duration_bucket,
                )
            )
        except Exception as exc:  # noqa: BLE001
            fibonacci_decision = {"ready": False, "blockers": [_high_upside_sanitize_error(exc)], "raw": {}}
        if bool(current_app.config.get("ML_DYNAMIC_CAPS_ENABLED", False)) and bool(extreme_decision.get("ready", False)):
            raw = extreme_decision.get("raw") if isinstance(extreme_decision.get("raw"), dict) else {}
            suggested_notional = min(max(0.0, _float(raw.get("suggested_notional_usdc"), suggested_notional)), clipped_cap)
            suggested_leverage = min(max(1.0, _float(raw.get("suggested_leverage"), 1.0)), 1.0)
            suggested_risk_pct = min(max(0.0, _float(raw.get("suggested_risk_pct"), suggested_risk_pct)), 0.01)
    return {
        "enabled": bool(current_app.config.get("ML_DYNAMIC_CAPS_ENABLED", False)),
        "objective": _ml_objective(objective),
        "target_roi_pct": target_roi_pct,
        "hard_caps": hard_caps,
        "suggested_notional_usdc": suggested_notional,
        "clipped_cap_usdc": min(suggested_notional if suggested_notional > 0 else clipped_cap, clipped_cap) if clipped_cap > 0 else 0.0,
        "suggested_leverage": suggested_leverage,
        "clipped_leverage": min(suggested_leverage, 1.0),
        "suggested_risk_pct": suggested_risk_pct,
        "extreme_upside_decision": extreme_decision,
        "fibonacci_zones": dict((fibonacci_decision.get("raw") or {}) if isinstance(fibonacci_decision, dict) else {}),
        "blockers": list(dict.fromkeys(blockers)),
        "policy": "ML can tighten or suggest caps only; deterministic hard caps and RiskEngine remain authoritative.",
    }


def _ml_live_vault_selection_payload(
    *,
    selected_ranking: StrategyRanking,
    user_id: int,
    connection_id: int,
    provider: str,
    collateral_asset: str,
    cap_usdc: float,
    duration_hours: int,
    requested_symbols: list[str],
) -> tuple[dict[str, object], dict[str, object]]:
    blockers: list[str] = []
    provider_key = normalize_exchange_provider(provider)
    collateral = str(collateral_asset or provider_collateral_asset(provider_key)).upper()
    ranking_explanation = selected_ranking.ml_explanation if isinstance(selected_ranking.ml_explanation, dict) else {}
    venue_symbol = str(
        ranking_explanation.get("venue_symbol")
        or selected_ranking.parameters.get("venue_symbol")
        or selected_ranking.symbol
    ).strip()
    app_symbol = str(ranking_explanation.get("app_symbol") or selected_ranking.parameters.get("app_symbol") or selected_ranking.symbol).strip().upper()
    if normalize_exchange_provider(getattr(selected_ranking, "provider", "global")) != provider_key:
        blockers.append("selected_ranking_provider_mismatch")
    selector = get_service("vault_strategy_selector")
    allowed_symbols = requested_symbols or [selected_ranking.symbol]
    selection = selector.select(collateral, duration_hours, "live", cap_usdc, allowed_symbols=allowed_symbols, provider=provider_key)
    legs = selection.legs or [
        {
            "strategy_name": selection.strategy_name,
            "symbol": selection.symbol,
            "timeframe": selection.timeframe,
            "parameters": selection.parameters,
            "allocation_cap_usd": cap_usdc,
            "leverage": selection.parameters.get("leverage", 1.0),
            "optimizer_ranking_id": selection.metadata.get("optimizer_ranking_id"),
            "optimizer_profile": selection.metadata.get("optimizer_profile"),
        }
    ]
    if len(legs) != 1:
        blockers.append("ml_live_vault_requires_exactly_one_leg")
    leg = dict(legs[0]) if legs else {}
    ranking_id = int(leg.get("optimizer_ranking_id") or selection.metadata.get("optimizer_ranking_id") or 0)
    if ranking_id != selected_ranking.id:
        blockers.append("vault_selector_did_not_select_preview_ranking")
    allocation = min(max(0.0, _float(leg.get("allocation_cap_usd"), cap_usdc)), cap_usdc, 10.0)
    leverage = max(0.0, _float(leg.get("leverage"), selected_ranking.leverage or 1.0))
    if leverage <= 0 or leverage > 1.0:
        blockers.append("ml_live_vault_leverage_above_1x")
    params = {**selected_ranking.parameters, **dict(leg.get("parameters") or {})}
    if _float(params.get("stop_loss_pct"), 0.0) <= 0:
        blockers.append("stop_loss_required")
    if _float(params.get("take_profit_pct"), 0.0) <= 0:
        blockers.append("take_profit_required")
    params.update(
        {
            "high_upside_profile": True,
            "duration_hours": duration_hours,
            "lock_duration_hours": duration_hours,
            "allocation_cap_usd": allocation,
            "optimizer_ranking_id": selected_ranking.id,
            "optimizer_profile": selected_ranking.profile,
            "leverage": min(leverage if leverage > 0 else 1.0, 1.0),
            "user_id": user_id,
            "trading_connection_id": connection_id,
            "provider": provider_key,
            "execution_venue": provider_key,
            "collateral_asset": collateral,
            "venue_symbol": venue_symbol,
            "app_symbol": app_symbol,
            "ml_live_vault_one_shot": True,
        }
    )
    selected_leg = {
        "strategy_name": str(leg.get("strategy_name") or selected_ranking.strategy_name),
        "symbol": app_symbol,
        "venue_symbol": venue_symbol,
        "timeframe": str(leg.get("timeframe") or selected_ranking.timeframe),
        "parameters": params,
        "allocation_cap_usd": allocation,
        "leverage": params["leverage"],
        "optimizer_ranking_id": selected_ranking.id,
        "optimizer_profile": selected_ranking.profile,
        "provider": provider_key,
        "collateral_asset": collateral,
    }
    payload = {
        "profile": selection.profile,
        "strategy_name": selection.strategy_name,
        "symbol": selection.symbol,
        "timeframe": selection.timeframe,
        "mode": selection.mode,
        "execution_mode": selection.execution_mode,
        "live_validation_status": selection.live_validation_status,
        "metadata": selection.metadata,
        "blockers": blockers,
    }
    return payload, selected_leg


def _ml_live_vault_risk_preview(
    *,
    selected_leg: dict[str, object],
    user_id: int,
    connection_id: int,
    cap_usdc: float,
    duration_hours: int,
) -> tuple[dict[str, object], dict[str, object]]:
    blockers: list[str] = []
    params = dict(selected_leg.get("parameters") or {})
    provider = normalize_exchange_provider(params.get("provider") or selected_leg.get("provider"))
    collateral_asset = str(params.get("collateral_asset") or selected_leg.get("collateral_asset") or provider_collateral_asset(provider)).upper()
    symbol = str(selected_leg.get("symbol") or "").upper()
    venue_symbol = str(params.get("venue_symbol") or selected_leg.get("venue_symbol") or symbol).strip()
    timeframe = str(selected_leg.get("timeframe") or "1h")
    strategy_name = str(selected_leg.get("strategy_name") or "scalping")
    manager = get_service("strategy_manager")
    market_data = get_service("market_data")
    try:
        candles = market_data.get_candles(venue_symbol, timeframe, mode="live", limit=200)
    except Exception as exc:  # noqa: BLE001
        candles = []
        blockers.append(f"candles_unavailable:{_high_upside_sanitize_error(exc)}")
    if not candles:
        return {"blockers": blockers or ["candles_unavailable"], "ml_signal_decision": {}}, {
            "approved": False,
            "rule_name": "candles_unavailable",
            "reason": "Live candles are required for a risk dry-run.",
        }
    try:
        strategy = get_service("strategy_registry").build(strategy_name, params)
        base_signal = strategy.generate_signal(symbol=symbol, timeframe=timeframe, candles=candles, position={})
        run = SimpleNamespace(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            mode="live",
            parameters=params,
            user_id=user_id,
            trading_connection_id=connection_id,
        )
        feature_payload = manager._feature_payload(symbol, timeframe, candles, base_signal)
        signal = manager._high_upside_ml_signal(run, base_signal, candles, feature_payload, "live")
    except Exception as exc:  # noqa: BLE001
        return {"blockers": [f"ml_signal_preview_failed:{_high_upside_sanitize_error(exc)}"], "ml_signal_decision": {}}, {
            "approved": False,
            "rule_name": "ml_signal_preview_failed",
            "reason": str(exc),
        }
    if getattr(signal, "action", "hold") not in {"buy", "sell"}:
        blockers.append(str((getattr(signal, "metadata", {}) or {}).get("no_trade_reason") or "ml_signal_hold"))
    stop_loss = getattr(signal, "stop_loss", None)
    take_profit = getattr(signal, "take_profit", None)
    if stop_loss is None:
        blockers.append("stop_loss_required")
    if take_profit is None:
        blockers.append("take_profit_required")
    mid = _float((candles[-1] or {}).get("close"), 0.0)
    if mid <= 0:
        blockers.append("mid_price_unavailable")
    position_fraction = max(0.0, min(_float(getattr(signal, "position_fraction", 0.0), 0.0), 1.0))
    notional = min(cap_usdc, max(0.0, cap_usdc * position_fraction))
    quantity = notional / mid if mid > 0 and notional > 0 else 0.0
    if quantity <= 0:
        blockers.append("order_quantity_zero")
    metadata = {
        **params,
        **dict(getattr(signal, "metadata", {}) or {}),
        "high_upside_profile": True,
        "duration_hours": duration_hours,
        "account_equity_usd": cap_usdc,
        "provider": provider,
        "execution_venue": provider,
        "collateral_asset": collateral_asset,
        "venue_symbol": venue_symbol,
        "app_symbol": symbol,
        "expected_stop_loss": stop_loss,
        "expected_take_profit": take_profit,
        "ml_live_vault_one_shot": True,
    }
    if blockers:
        return {
            "action": getattr(signal, "action", "hold"),
            "blockers": list(dict.fromkeys(blockers)),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "notional_usdc": notional,
            "notional_usd": notional,
            "collateral_asset": collateral_asset,
            "ml_signal_decision": dict((getattr(signal, "metadata", {}) or {}).get("ml_signal_decision") or {}),
        }, {
            "approved": False,
            "rule_name": "signal_preview_blocked",
            "reason": ", ".join(list(dict.fromkeys(blockers))),
        }
    intent = OrderIntent(
        symbol=symbol,
        side="buy" if signal.action == "buy" else "sell",
        quantity=quantity,
        mode="live",
        order_type="limit",
        limit_price=mid,
        leverage=min(_float(selected_leg.get("leverage"), 1.0), 1.0),
        stop_loss=stop_loss,
        take_profit=take_profit,
        strategy_name=strategy_name,
        timeframe=timeframe,
            slippage_pct=0.0,
        user_id=user_id,
        trading_connection_id=connection_id,
        metadata=metadata,
    )
    decision = get_service("risk_engine").evaluate(intent, market_price=mid, has_trading_access=True)
    return {
        "action": signal.action,
        "blockers": [],
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "notional_usdc": notional,
        "notional_usd": notional,
        "collateral_asset": collateral_asset,
        "quantity": quantity,
        "mid_price": mid,
        "ml_signal_decision": dict((getattr(signal, "metadata", {}) or {}).get("ml_signal_decision") or {}),
    }, decision.as_dict()


def _ml_live_vault_submit_gate_blockers(preview_payload: dict[str, object], user_id: int) -> list[str]:
    blockers: list[str] = []
    if not bool(current_app.config.get("ML_LIVE_VAULT_ONE_SHOT_ENABLED", False)):
        blockers.append("ML_LIVE_VAULT_ONE_SHOT_ENABLED=false")
    if str(current_app.config.get("APP_MODE", "paper") or "paper").lower() != "live":
        blockers.append("APP_MODE must be live")
    if not bool(current_app.config.get("ENABLE_LIVE_TRADING", False)):
        blockers.append("ENABLE_LIVE_TRADING must be true")
    if not bool(current_app.config.get("EXPLICIT_LIVE_CONFIRMED", False)):
        blockers.append("EXPLICIT_LIVE_CONFIRMED must be true")
    if not bool(current_app.config.get("SECONDARY_CONFIRMATION", False)):
        blockers.append("SECONDARY_CONFIRMATION must be true")
    if not bool(Setting.get_json("explicit_live_confirmed", False)):
        blockers.append("db_explicit_live_confirmed_missing")
    if not bool(Setting.get_json("secondary_confirmation", False)):
        blockers.append("db_secondary_confirmation_missing")
    if Setting.get_json("panic_lock", False):
        blockers.append("panic_lock")
    if not bool(current_app.config.get("HIGH_UPSIDE_AUTO_LIVE_ENABLED", False)):
        blockers.append("HIGH_UPSIDE_AUTO_LIVE_ENABLED=false")
    if _is_one_h10_preview_payload(preview_payload) and not bool(current_app.config.get("ONE_H10_LIVE_ENABLED", False)):
        blockers.append("ONE_H10_LIVE_ENABLED=false")
    blockers.extend(_ml_live_vault_cap_readiness(str(preview_payload.get("duration_bucket") or "1h"))["blockers"])
    connection_id = int(_float(preview_payload.get("connection_id"), 0.0))
    connection = _ml_live_vault_connection(user_id, str(preview_payload.get("provider") or "active"), connection_id or None)
    if connection is None:
        blockers.append("active_verified_connection_missing")
    active_count = VaultCycle.query.filter_by(user_id=user_id, status="active", execution_mode="live").count()
    if active_count > 0:
        blockers.append("ml_live_vault_active_cycle_limit_reached")
    day_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    started_today = (
        AuditLog.query.filter(
            AuditLog.user_id == user_id,
            AuditLog.action == "ml_live_vault_one_shot_started",
            AuditLog.created_at >= day_start,
        ).count()
    )
    if started_today >= 1:
        blockers.append("ml_live_vault_daily_cycle_limit_reached")
    selected = preview_payload.get("selected_ranking") if isinstance(preview_payload.get("selected_ranking"), dict) else {}
    ranking_id = int(_float(selected.get("id") if isinstance(selected, dict) else 0, 0.0))
    ranking = db.session.get(StrategyRanking, ranking_id) if ranking_id else None
    if ranking is None:
        blockers.append("selected_ranking_missing")
    elif bool(ranking.rejected) or str(ranking.rejection_reason or "").strip():
        blockers.append("selected_ranking_rejected")
    elif normalize_exchange_provider(getattr(ranking, "provider", "global")) != normalize_exchange_provider(preview_payload.get("provider")):
        blockers.append("selected_ranking_provider_mismatch")
    leg = preview_payload.get("selected_leg") if isinstance(preview_payload.get("selected_leg"), dict) else {}
    if _float(leg.get("allocation_cap_usd") if isinstance(leg, dict) else 0.0, 0.0) > 10.0:
        blockers.append("leg_notional_above_10_usdc")
    if _float(leg.get("leverage") if isinstance(leg, dict) else 1.0, 1.0) > 1.0:
        blockers.append("leverage_above_1x")
    return blockers


def _create_ml_live_vault_cycle(preview_payload: dict[str, object], preview_audit_id: int) -> dict[str, object]:
    user_id = int(preview_payload.get("user_id") or 0)
    connection_id = int(preview_payload.get("connection_id") or 0)
    cap = min(_float(preview_payload.get("cap_usdc"), 10.0), 10.0)
    provider = normalize_exchange_provider(preview_payload.get("provider"))
    collateral_asset = str(preview_payload.get("collateral_asset") or provider_collateral_asset(provider)).upper()
    duration_hours = max(1, int(_float(preview_payload.get("duration_hours"), 1.0)))
    duration_seconds = duration_hours * 3600
    is_one_h10 = _is_one_h10_preview_payload(preview_payload)
    leg = dict(preview_payload.get("selected_leg") or {})
    params = dict(leg.get("parameters") or {})
    app_symbol = str(leg.get("symbol") or params.get("app_symbol") or "").upper()
    venue_symbol = str(leg.get("venue_symbol") or params.get("venue_symbol") or app_symbol).strip()
    balance = WalletBalance.query.filter_by(user_id=user_id, asset=collateral_asset).one_or_none()
    if balance is None or float(balance.available_balance or 0.0) + 1e-9 < cap:
        audit = AuditLog(
            category="ml_live_vault",
            action="ml_live_vault_one_shot_blocked",
            message="ML live vault one-shot blocked by wallet balance before cycle start.",
            user_id=user_id,
            trading_connection_id=connection_id or None,
        )
        audit.details = {"preview_audit_id": preview_audit_id, "blockers": ["wallet_balance_insufficient"]}
        db.session.add(audit)
        db.session.commit()
        return {
            "ready": False,
            "submitted": False,
            "cycle_started": False,
            "live_orders_created": False,
            "blockers": ["wallet_balance_insufficient"],
        }
    now = datetime.utcnow()
    balance.available_balance = float(balance.available_balance) - cap
    balance.locked_balance = float(balance.locked_balance) + cap
    balance.estimated_usd_value = balance.total_balance
    cycle = VaultCycle(
        user_id=user_id,
        trading_connection_id=connection_id or None,
        deposit_asset=collateral_asset,
        deposit_amount=cap,
        settlement_asset=collateral_asset,
        lock_duration_hours=duration_hours,
        lock_duration_seconds=duration_seconds,
        status="active",
        execution_substatus="ml_live_vault_one_shot",
        execution_mode="live",
        live_validation_status="passed",
        algorithm_profile="1H10" if is_one_h10 else "MLHighUpside",
        selected_strategy_name=str(leg.get("strategy_name") or "scalping"),
        selected_timeframe=str(leg.get("timeframe") or "1h"),
        started_at=now,
        unlocks_at=now + timedelta(seconds=duration_seconds),
        starting_value_usd=cap,
        current_estimated_value_usd=cap,
    )
    cycle.selection_metadata = _json_safe(
        {
            "ml_live_vault_one_shot": True,
            "one_h10_vault": is_one_h10,
            "vault_cycle_name": "1H10" if is_one_h10 else None,
            "ml_horizon": "1h10" if is_one_h10 else str(preview_payload.get("horizon") or "1h"),
            "target_amount_usd": cap * 10.0 if is_one_h10 else None,
            "target_roi_pct": current_app.config.get("ONE_H10_TARGET_ROI_PCT", 1000.0) if is_one_h10 else preview_payload.get("target_roi_pct"),
            "preview_audit_id": preview_audit_id,
            "selected_ranking": preview_payload.get("selected_ranking"),
            "ml_model_ids": preview_payload.get("ml_model_ids"),
            "risk_decision": preview_payload.get("risk_decision"),
            "provider": provider,
            "connection_id": connection_id,
            "collateral_asset": collateral_asset,
            "venue_symbol": venue_symbol,
            "app_symbol": app_symbol,
        }
    )
    db.session.add(cycle)
    db.session.flush()
    common = {
        "vault_cycle_id": cycle.id,
        "consumer_vault": True,
        "algorithm_profile": "1H10" if is_one_h10 else "MLHighUpside",
        "vault_cycle_name": "1H10" if is_one_h10 else None,
        "one_h10_vault": is_one_h10,
        "ml_horizon": "1h10" if is_one_h10 else str(preview_payload.get("horizon") or "1h"),
        "target_amount_usd": cap * 10.0 if is_one_h10 else None,
        "target_roi_pct": current_app.config.get("ONE_H10_TARGET_ROI_PCT", 1000.0) if is_one_h10 else preview_payload.get("target_roi_pct"),
        "execution_mode": "live",
        "live_validation_status": "passed",
        "lock_duration_hours": duration_hours,
        "lock_duration_seconds": duration_seconds,
        "allowed_symbols": [app_symbol],
        "user_id": user_id,
        "trading_connection_id": connection_id or None,
        "provider": provider,
        "execution_venue": provider,
        "collateral_asset": collateral_asset,
        "venue_symbol": venue_symbol,
        "app_symbol": app_symbol,
        "ml_live_vault_one_shot": True,
        "preview_audit_id": preview_audit_id,
    }
    params.update(common)
    params["allocation_cap_usd"] = min(_float(leg.get("allocation_cap_usd"), cap), cap, 10.0)
    params["leverage"] = min(_float(leg.get("leverage"), 1.0), 1.0)
    run = StrategyRun(
        strategy_name=str(leg.get("strategy_name") or "scalping"),
        symbol=app_symbol,
        timeframe=str(leg.get("timeframe") or "1h"),
        mode="live",
        user_id=user_id,
        trading_connection_id=connection_id or None,
        status="starting",
        lock_duration_seconds=duration_seconds,
        manual_enabled=True,
    )
    run.parameters = params
    db.session.add(run)
    db.session.flush()
    leg_model = VaultAllocationLeg(
        vault_cycle_id=cycle.id,
        strategy_run_id=run.id,
        optimizer_ranking_id=int(leg.get("optimizer_ranking_id") or 0) or None,
        symbol=run.symbol,
        timeframe=run.timeframe,
        provider=provider,
        trading_connection_id=connection_id or None,
        allocation_cap_usd=params["allocation_cap_usd"],
        leverage=params["leverage"],
        status="active",
    )
    leg_model.details = _json_safe(
        {
            "ml_live_vault_one_shot": True,
            "one_h10_vault": is_one_h10,
            "ml_horizon": "1h10" if is_one_h10 else str(preview_payload.get("horizon") or "1h"),
            "preview_audit_id": preview_audit_id,
            "provider": provider,
            "collateral_asset": collateral_asset,
            "venue_symbol": venue_symbol,
            "app_symbol": app_symbol,
        }
    )
    db.session.add(leg_model)
    db.session.flush()
    params["vault_leg_id"] = leg_model.id
    run.parameters = params
    cycle.strategy_run_id = run.id
    db.session.add(
        WalletTransaction(
            vault_cycle_id=cycle.id,
            user_id=user_id,
            asset=collateral_asset,
            amount=cap,
            transaction_type="allocation",
            status="complete",
            note="ML live vault one-shot cycle started.",
        )
    )
    started_audit = AuditLog(
        category="ml_live_vault",
        action="ml_live_vault_one_shot_started",
        message="ML live vault one-shot cycle started through StrategyManager.",
        user_id=user_id,
        trading_connection_id=connection_id or None,
    )
    started_audit.details = {
        "preview_audit_id": preview_audit_id,
        "cycle_id": cycle.id,
        "strategy_run_id": run.id,
        "provider": provider,
        "connection_id": connection_id,
        "collateral_asset": collateral_asset,
    }
    db.session.add(started_audit)
    db.session.commit()
    get_service("strategy_manager").start(run.id)
    return {
        "ready": True,
        "submitted": True,
        "cycle_started": True,
        "live_orders_created": "strategy_manager_deferred",
        "cycle_id": cycle.id,
        "strategy_run_id": run.id,
        "vault_leg_id": leg_model.id,
        "started_audit_id": started_audit.id,
        "allocation_budget_usdc": cap,
        "allocation_budget_usd": cap,
        "collateral_asset": collateral_asset,
        "provider": provider,
        "live_order_path": "StrategyManager.start -> StrategyRunner -> OrderManager -> RiskEngine",
        "blockers": [],
    }


def _ml_live_vault_compact_discovery(discovery: dict[str, object]) -> dict[str, object]:
    return {
        "ready": bool(discovery.get("ready", False)),
        "objective": discovery.get("objective"),
        "target_roi_pct": discovery.get("target_roi_pct"),
        "accepted_ranking_ids": list(discovery.get("accepted_ranking_ids", []) or []),
        "accepted_rankings": list(discovery.get("accepted_rankings", []) or [])[:3],
        "scanner_candidates": list(discovery.get("scanner_candidates", []) or [])[:3],
        "blockers": list(discovery.get("blockers", []) or []),
        "rejection_breakdown": dict(discovery.get("rejection_breakdown", {}) or {}),
        "runtime": discovery.get("runtime", {}),
    }


def _ml_live_vault_ranking_payload(ranking: StrategyRanking | None) -> dict[str, object] | None:
    if ranking is None:
        return None
    return {
        "id": ranking.id,
        "provider": normalize_exchange_provider(getattr(ranking, "provider", "global")),
        "collateral_asset": provider_collateral_asset(getattr(ranking, "provider", "global")),
        "strategy_name": ranking.strategy_name,
        "symbol": ranking.symbol,
        "timeframe": ranking.timeframe,
        "profile": ranking.profile,
        "score": ranking.score,
        "rejected": bool(ranking.rejected),
        "rejection_reason": ranking.rejection_reason or "",
        "profit_factor": ranking.profit_factor,
        "trade_count": ranking.trade_count,
        "net_return_after_costs": ranking.net_return_after_costs,
    }


def _ml_live_vault_model_ids(
    offline_ml: dict[str, object],
    signal_readiness: dict[str, object],
    ml_suite: dict[str, object],
) -> dict[str, object]:
    def model_id(payload: object) -> object:
        if not isinstance(payload, dict):
            return None
        nested = payload.get("promoted_model")
        if isinstance(nested, dict):
            return nested.get("model_id") or nested.get("id")
        return (
            payload.get("promoted_model_id")
            or payload.get("model_id")
            or payload.get("latest_model_id")
        )

    families = ml_suite.get("families") if isinstance(ml_suite.get("families"), dict) else {}
    return {
        "offline_model_id": model_id(offline_ml),
        "signal_model_id": model_id(signal_readiness),
        "suite_family_model_ids": {
            family: model_id(row)
            for family, row in families.items()
            if isinstance(row, dict)
        },
    }


def _json_safe(value: object) -> object:
    return json.loads(json.dumps(value, default=str))


def _optimizer_live_canary_readiness(config: object) -> dict[str, object]:
    profile = str(getattr(config, "profile", "") or "")
    ranking = (
        StrategyRanking.query.filter_by(profile=profile, rejected=False)
        .order_by(StrategyRanking.created_at.desc(), StrategyRanking.score.desc())
        .first()
    )
    active_verified_connections = TradingConnection.query.filter_by(
        is_active=True,
        verification_status="verified",
    ).count()
    blockers: list[str] = []
    if ranking is None:
        blockers.append("accepted_ranking_missing")
    if active_verified_connections <= 0:
        blockers.append("active_verified_live_connection_missing")
    risk_status = get_service("risk_engine").status("live")
    if bool(risk_status.get("panic_lock", False)):
        blockers.append("panic_lock")
    if bool(risk_status.get("live_trading_blocked", False)):
        blockers.append("live_trading_blocked")
    return {
        "ready_for_preview": ranking is not None,
        "ready_for_submit": not blockers,
        "blockers": blockers,
        "ranking_id": ranking.id if ranking else None,
        "profile": profile,
        "active_verified_connection_count": active_verified_connections,
        "command": (
            f"flask live-canary-trade --ranking-id {ranking.id} --user-id <id> --confirm LIVE-CANARY-TRADE"
            if ranking
            else "flask live-canary-trade --ranking-id <id> --user-id <id> --confirm LIVE-CANARY-TRADE"
        ),
        "submit_flag_required_for_live_order": "--submit",
        "optimizer_research_only": True,
        "uses_existing_live_caps": True,
        "canary_preview_only": bool(current_app.config.get("CANARY_PREVIEW_ONLY", True)),
    }


def _high_upside_live_readiness(config: object, scanner_diagnostics: dict[str, object] | None = None) -> dict[str, object]:
    risk_status = get_service("risk_engine").status("live")
    active_verified_connections = TradingConnection.query.filter_by(
        is_active=True,
        verification_status="verified",
    ).count()
    duration_bucket = _duration_bucket(getattr(config, "lock_duration_hours", 1))
    provider_key = normalize_exchange_provider(getattr(config, "provider", "global"))
    collateral_asset = provider_collateral_asset(provider_key)
    cap_usdc = _duration_cap("HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION", duration_bucket)
    cap_pct = _duration_cap("HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION", duration_bucket)
    max_daily_loss = float(current_app.config.get("HIGH_UPSIDE_MAX_DAILY_LOSS_USDC", 0.0) or 0.0)
    blockers: list[str] = []
    if not bool(getattr(config, "high_upside_profile", False)):
        blockers.append("high_upside_profile_not_requested")
    if str(current_app.config.get("APP_MODE", "paper") or "paper").lower() != "live":
        blockers.append("APP_MODE_not_live")
    if not bool(current_app.config.get("ENABLE_LIVE_TRADING", False)):
        blockers.append("ENABLE_LIVE_TRADING=false")
    if not bool(current_app.config.get("EXPLICIT_LIVE_CONFIRMED", False)) or not bool(Setting.get_json("explicit_live_confirmed", False)):
        blockers.append("explicit_live_confirmation_missing")
    if not bool(current_app.config.get("SECONDARY_CONFIRMATION", False)) or not bool(Setting.get_json("secondary_confirmation", False)):
        blockers.append("secondary_confirmation_missing")
    if not bool(current_app.config.get("HIGH_UPSIDE_PROFILE_ENABLED", False)):
        blockers.append("HIGH_UPSIDE_PROFILE_ENABLED=false")
    if not bool(current_app.config.get("HIGH_UPSIDE_LIVE_ELIGIBLE", False)):
        blockers.append("HIGH_UPSIDE_LIVE_ELIGIBLE=false")
    if not bool(current_app.config.get("HIGH_UPSIDE_AUTO_LIVE_ENABLED", False)):
        blockers.append("HIGH_UPSIDE_AUTO_LIVE_ENABLED=false")
    if bool(risk_status.get("panic_lock", False)):
        blockers.append("panic_lock")
    if bool(risk_status.get("live_trading_blocked", False)):
        blockers.append("live_trading_blocked")
    if bool((risk_status.get("high_upside") or {}).get("auto_disabled", False)):
        blockers.append("high_upside_auto_disabled")
    if active_verified_connections <= 0:
        blockers.append("active_verified_live_connection_missing")
    if max_daily_loss <= 0:
        blockers.append("HIGH_UPSIDE_MAX_DAILY_LOSS_USDC_missing")
    if cap_usdc <= 0:
        blockers.append("HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION_JSON_missing")
    if cap_pct <= 0:
        blockers.append("HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION_JSON_missing")
    rejection_rate = float((scanner_diagnostics or {}).get("rejection_rate", 0.0) or 0.0)
    max_rejection_rate = float(current_app.config.get("HIGH_UPSIDE_MAX_SCANNER_REJECTION_RATE", 0.65) or 0.65)
    if rejection_rate > max_rejection_rate:
        blockers.append("scanner_rejection_rate_breach")
    offline_ml = _offline_ml_readiness(duration_bucket, require_blend=False, provider=provider_key)
    if bool(current_app.config.get("HIGH_UPSIDE_REQUIRE_PROMOTED_ML", True)) and not bool(offline_ml.get("ready", False)):
        blockers.append("promoted_offline_ml_required")
    ml_signal = _ml_signal_readiness(duration_bucket, provider=provider_key)
    if bool(current_app.config.get("HIGH_UPSIDE_REQUIRE_ML_SIGNAL", True)) and not bool(ml_signal.get("ready", False)):
        blockers.append("promoted_ml_signal_required")
    ml_suite = _ml_suite_readiness(duration_bucket, provider=provider_key)
    if bool(current_app.config.get("ML_ALL_AREAS_ENABLED", False)) and not bool(ml_suite.get("ready", False)):
        blockers.append("ml_suite_required")
    continuous_controls = _high_upside_continuous_controls(duration_bucket, scanner_diagnostics or {}, offline_ml)
    blockers.extend(list(continuous_controls.get("blockers", []) or []))

    return {
        "requested": bool(getattr(config, "high_upside_profile", False)),
        "ready": not blockers,
        "provider": provider_key,
        "collateral_asset": collateral_asset,
        "blockers": blockers,
        "profile_enabled": bool(current_app.config.get("HIGH_UPSIDE_PROFILE_ENABLED", False)),
        "live_eligible": bool(current_app.config.get("HIGH_UPSIDE_LIVE_ELIGIBLE", False)),
        "auto_live_enabled": bool(current_app.config.get("HIGH_UPSIDE_AUTO_LIVE_ENABLED", False)),
        "app_mode": str(current_app.config.get("APP_MODE", "paper") or "paper"),
        "enable_live_trading": bool(current_app.config.get("ENABLE_LIVE_TRADING", False)),
        "confirmation_flags": {
            "config_explicit_live_confirmed": bool(current_app.config.get("EXPLICIT_LIVE_CONFIRMED", False)),
            "config_secondary_confirmation": bool(current_app.config.get("SECONDARY_CONFIRMATION", False)),
            "setting_explicit_live_confirmed": bool(Setting.get_json("explicit_live_confirmed", False)),
            "setting_secondary_confirmation": bool(Setting.get_json("secondary_confirmation", False)),
        },
        "auto_disabled": bool((risk_status.get("high_upside") or {}).get("auto_disabled", False)),
        "active_verified_connection_count": active_verified_connections,
        "duration_bucket": duration_bucket,
        "caps": {
            "max_daily_loss_usdc": max_daily_loss,
            "duration_cap_usdc": cap_usdc,
            "duration_cap_pct": cap_pct,
        },
        "scanner_rejection_rate": rejection_rate,
        "scanner_rejection_rate_limit": max_rejection_rate,
        "requires_candidate_tag": True,
        "requires_shadow_or_pre_live_validation": bool(getattr(config, "require_shadow_validation", True)),
        "requires_promoted_offline_ml": bool(current_app.config.get("HIGH_UPSIDE_REQUIRE_PROMOTED_ML", True)),
        "offline_ml_readiness": offline_ml,
        "requires_ml_signal": bool(current_app.config.get("HIGH_UPSIDE_REQUIRE_ML_SIGNAL", True)),
        "ml_signal_readiness": ml_signal,
        "ml_readiness": ml_suite,
        "ml_models": ml_suite.get("families", {}),
        "ml_blockers": ml_suite.get("blockers", []),
        "continuous_controls": continuous_controls,
        "no_default_live_cap_increase": True,
    }


def _high_upside_continuous_controls(
    duration_bucket: str,
    scanner_diagnostics: dict[str, object],
    offline_ml: dict[str, object],
) -> dict[str, object]:
    max_daily = max(0, int(current_app.config.get("HIGH_UPSIDE_MAX_LIVE_CYCLES_PER_DAY", 1) or 1))
    max_active = max(0, int(current_app.config.get("HIGH_UPSIDE_MAX_ACTIVE_CYCLES", 1) or 1))
    daily_count = _high_upside_daily_live_order_count()
    active_count = _high_upside_active_live_order_count()
    rejection_cooldown = _high_upside_recent_rejection_cooldown_seconds()
    loss_cooldown = _high_upside_loss_cooldown_seconds()
    backoff_seconds = _high_upside_rate_limit_backoff_remaining_seconds()
    rejection_rate = _float(scanner_diagnostics.get("rejection_rate"), 0.0)
    offline_ready = bool(offline_ml.get("ready", False))
    adaptive_enabled = bool(current_app.config.get("HIGH_UPSIDE_ADAPTIVE_CADENCE_ENABLED", False))
    min_interval = _float(current_app.config.get("HIGH_UPSIDE_MIN_SCAN_INTERVAL_SECONDS"), 3600.0)
    fast_interval = _float(current_app.config.get("HIGH_UPSIDE_FAST_SCAN_INTERVAL_SECONDS"), 300.0)
    recommended_interval = min_interval
    if adaptive_enabled and offline_ready and rejection_rate <= _float(current_app.config.get("HIGH_UPSIDE_MAX_SCANNER_REJECTION_RATE"), 0.65):
        recommended_interval = min(min_interval, fast_interval)
    ml_ops_anomaly_decision: dict[str, object] = {}

    blockers: list[str] = []
    if max_daily > 0 and daily_count >= max_daily:
        blockers.append("high_upside_daily_live_cycle_limit_reached")
    if max_active > 0 and active_count >= max_active:
        blockers.append("high_upside_active_cycle_limit_reached")
    if rejection_cooldown > 0:
        blockers.append("high_upside_rejection_cooldown_active")
    if loss_cooldown > 0:
        blockers.append("high_upside_loss_cooldown_active")
    if backoff_seconds > 0:
        blockers.append("high_upside_provider_rate_limit_backoff_active")
    if bool(current_app.config.get("ML_OPS_ANOMALY_ENABLED", False)):
        try:
            ml_ops_anomaly_decision = dict(
                get_service("ml_decision_engine").decision(
                    "pytorch_ops_anomaly",
                    {
                        "horizon": duration_bucket,
                        "scanner_rejection_rate": rejection_rate,
                        "rate_limited": backoff_seconds > 0,
                        "error_rate": rejection_rate,
                        "latency_ms": _float(scanner_diagnostics.get("latency_ms"), 0.0),
                    },
                    horizon=duration_bucket,
                )
            )
        except Exception as exc:  # noqa: BLE001
            ml_ops_anomaly_decision = {"ready": False, "blockers": [str(exc)]}
        anomaly_score = _float((ml_ops_anomaly_decision.get("raw") or {}).get("ops_anomaly_score"), 0.0)
        if bool(ml_ops_anomaly_decision.get("ready", False)) and anomaly_score >= 0.80:
            blockers.append("ml_ops_anomaly_block")
        if bool(ml_ops_anomaly_decision.get("ready", False)) and anomaly_score >= 0.50:
            recommended_interval = max(recommended_interval, min_interval)

    return {
        "adaptive_cadence_enabled": adaptive_enabled,
        "recommended_scan_interval_seconds": recommended_interval,
        "min_scan_interval_seconds": min_interval,
        "fast_scan_interval_seconds": fast_interval,
        "max_live_cycles_per_day": max_daily,
        "daily_live_order_count": daily_count,
        "max_active_cycles": max_active,
        "active_live_order_count": active_count,
        "loss_cooldown_seconds": _float(current_app.config.get("HIGH_UPSIDE_LOSS_COOLDOWN_SECONDS"), 3600.0),
        "loss_cooldown_remaining_seconds": loss_cooldown,
        "rejection_cooldown_seconds": _float(current_app.config.get("HIGH_UPSIDE_REJECTION_COOLDOWN_SECONDS"), 900.0),
        "rejection_cooldown_remaining_seconds": rejection_cooldown,
        "rate_limit_backoff_seconds": _float(current_app.config.get("HIGH_UPSIDE_RATE_LIMIT_BACKOFF_SECONDS"), 300.0),
        "rate_limit_backoff_remaining_seconds": backoff_seconds,
        "ml_ops_anomaly_decision": ml_ops_anomaly_decision,
        "blockers": blockers,
        "policy": "adaptive cadence can speed up only after promoted ML and clean scanner diagnostics; live execution remains RiskEngine-gated.",
        "duration_bucket": duration_bucket,
    }


def _high_upside_daily_live_order_count() -> int:
    since = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    count = 0
    for order in Order.query.filter(Order.mode == "live", Order.created_at >= since).order_by(Order.created_at.desc()).limit(500).all():
        if _high_upside_order_is_live_cycle(order) and str(order.status or "").lower() not in {"rejected", "failed", "cancelled"}:
            count += 1
    return count


def _high_upside_active_live_order_count() -> int:
    active_statuses = {"open", "submitted", "pending", "partially_filled"}
    count = 0
    for order in Order.query.filter_by(mode="live").order_by(Order.created_at.desc()).limit(500).all():
        if _high_upside_order_is_live_cycle(order) and str(order.status or "").lower() in active_statuses:
            count += 1
    return count


def _high_upside_order_is_live_cycle(order: Order) -> bool:
    details = order.details if isinstance(order.details, dict) else {}
    return bool(details.get("high_upside_profile") or details.get("optimizer_profile") == "aggressive_1h")


def _high_upside_recent_rejection_cooldown_seconds() -> float:
    cooldown = _float(current_app.config.get("HIGH_UPSIDE_REJECTION_COOLDOWN_SECONDS"), 900.0)
    if cooldown <= 0:
        return 0.0
    since = datetime.utcnow() - timedelta(seconds=cooldown)
    rejected = (
        Order.query.filter(Order.mode == "live", Order.created_at >= since, Order.status.in_(["rejected", "failed"]))
        .order_by(Order.created_at.desc())
        .limit(50)
        .all()
    )
    for order in rejected:
        if _high_upside_order_is_live_cycle(order):
            elapsed = (datetime.utcnow() - order.created_at).total_seconds()
            return max(0.0, cooldown - elapsed)
    return 0.0


def _high_upside_loss_cooldown_seconds() -> float:
    cooldown = _float(current_app.config.get("HIGH_UPSIDE_LOSS_COOLDOWN_SECONDS"), 3600.0)
    if cooldown <= 0:
        return 0.0
    since = datetime.utcnow() - timedelta(seconds=cooldown)
    fills = (
        db.session.query(Order, Fill)
        .join(Fill, Fill.order_id == Order.id)
        .filter(Order.mode == "live", Fill.fill_time >= since)
        .order_by(Fill.fill_time.desc())
        .limit(50)
        .all()
    )
    for order, fill in fills:
        fill_net = float(fill.pnl or 0.0) - float(fill.fee or 0.0) - float(getattr(fill, "funding_fee", 0.0) or 0.0)
        if _high_upside_order_is_live_cycle(order) and fill_net < 0:
            elapsed = (datetime.utcnow() - fill.fill_time).total_seconds()
            return max(0.0, cooldown - elapsed)
    return 0.0


def _high_upside_rate_limit_backoff_remaining_seconds() -> float:
    until = Setting.get_json("high_upside_rate_limited_until", None)
    if not until:
        return 0.0
    try:
        until_dt = datetime.fromisoformat(str(until).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if until_dt.tzinfo is not None:
        now = datetime.now(until_dt.tzinfo)
    else:
        now = datetime.utcnow()
    return max(0.0, (until_dt - now).total_seconds())


def _duration_bucket(duration_hours: object) -> str:
    try:
        duration = max(1, int(float(duration_hours or 1)))
    except (TypeError, ValueError):
        duration = 1
    if duration <= 1:
        return "1h"
    if duration <= 24:
        return "24h"
    if duration <= 48:
        return "48h"
    if duration <= 168:
        return "7d"
    return "custom"


def _duration_cap(config_key: str, bucket: str) -> float:
    mapping = current_app.config.get(config_key, {}) or {}
    if not isinstance(mapping, dict):
        return 0.0
    for lookup in (bucket, str(bucket).lower(), str(bucket).upper()):
        try:
            return float(mapping[lookup] or 0.0)
        except KeyError:
            continue
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _dynamic_intraday_live_readiness(config: object) -> dict[str, object]:
    risk_status = get_service("risk_engine").status("live")
    active_verified_connections = TradingConnection.query.filter_by(
        is_active=True,
        verification_status="verified",
    ).count()
    live_enabled = bool(current_app.config.get("ENABLE_LIVE_TRADING", False))
    dynamic_live_eligible = bool(getattr(config, "dynamic_intraday_live_eligible", False))
    require_shadow_validation = bool(getattr(config, "require_shadow_validation", True))
    daily_loss_limit = float(risk_status.get("daily_loss_limit", 0.0) or 0.0)
    daily_realized_pnl = float(risk_status.get("daily_realized_pnl", 0.0) or 0.0)
    daily_loss_used = abs(min(daily_realized_pnl, 0.0))
    warnings: list[str] = []
    if live_enabled:
        warnings.append(
            "ENABLE_LIVE_TRADING is true. Dynamic intraday live orders remain blocked unless live eligibility, connection, shadow validation, and hard risk gates all pass."
        )
    if dynamic_live_eligible:
        warnings.append("DYNAMIC_INTRADAY_LIVE_ELIGIBLE is enabled for this optimization config.")
    if not require_shadow_validation:
        warnings.append("Shadow validation was disabled for this optimizer config.")
    if bool(risk_status.get("panic_lock", False)):
        warnings.append("Panic lock is active; live orders are blocked.")
    if bool(risk_status.get("live_trading_blocked", False)):
        warnings.append("Live trading failure block is active.")
    if daily_loss_limit > 0 and daily_loss_used >= daily_loss_limit:
        warnings.append("Daily loss limit is already reached for live mode.")

    if not dynamic_live_eligible:
        candidate_status = "shadow_live_eligible"
    elif require_shadow_validation:
        candidate_status = "live_eligible_after_shadow_validation"
    else:
        candidate_status = "live_eligible_after_risk_checks"

    return {
        "profile": getattr(config, "profile", ""),
        "app_mode": current_app.config.get("APP_MODE"),
        "enable_live_trading": live_enabled,
        "dynamic_intraday_live_eligible": dynamic_live_eligible,
        "require_shadow_validation": require_shadow_validation,
        "candidate_default_status": candidate_status,
        "active_verified_connection_count": active_verified_connections,
        "has_active_verified_connection": active_verified_connections > 0,
        "panic_lock": bool(risk_status.get("panic_lock", False)),
        "live_trading_blocked": bool(risk_status.get("live_trading_blocked", False)),
        "daily_loss": {
            "realized_pnl": daily_realized_pnl,
            "loss_used": daily_loss_used,
            "limit": daily_loss_limit,
            "remaining": max(daily_loss_limit - daily_loss_used, 0.0) if daily_loss_limit > 0 else None,
        },
        "allocation_gate": {"source": "allocation_amount"},
        "slippage_gate": {"engine": "adaptive_ml", **dict(risk_status.get("adaptive_slippage", {}) or {})},
        "stop_loss_gate": {"required_for_opening_trades": True},
        "shadow_validation_gate": {
            "required": require_shadow_validation,
            "max_age_hours": float(current_app.config.get("ENSEMBLE_1H_SHADOW_VALIDATION_MAX_AGE_HOURS", 24.0) or 24.0),
        },
        "warnings": warnings,
    }
