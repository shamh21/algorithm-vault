"""Flask CLI commands for backtesting and optimization."""

from __future__ import annotations

import json
import importlib.util
import signal
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import click
from cryptography.fernet import Fernet
from flask import Flask, current_app
from flask.cli import with_appcontext

from .backtesting.engine import BacktestConfig
from .extensions import db
from .models import (
    AuditLog,
    BacktestRun,
    MLOfflineModel,
    Order,
    Setting,
    StrategyRanking,
    StrategyRun,
    TradingConnection,
    User,
    VaultCycle,
    WalletAddress,
    WalletAuditLog,
    WalletBalance,
    WalletTransaction,
    WalletWithdrawal,
)
from .runtime import get_service
from .services.connection_health import (
    active_connection_health,
    build_connection_health,
    parse_exchange_failure,
    store_connection_health,
)
from .services.order_manager import OrderIntent
from .services.signal_quality import SignalQualityEvaluator


class _OptimizerCliTimeout(TimeoutError):
    """Raised when the operator-facing optimizer command exceeds its deadline."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = float(timeout_seconds)
        super().__init__(
            f"run-optimization exceeded OPTIMIZER_MARKET_DATA_TIMEOUT_SECONDS={self.timeout_seconds:.3f}"
        )


def register_cli(app: Flask) -> None:
    """Register operational commands on the Flask app."""

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

    @app.cli.command("wallet-readiness")
    @with_appcontext
    def wallet_readiness() -> None:
        """Print live generated-wallet readiness diagnostics."""

        click.echo(json.dumps(_wallet_readiness_payload(), indent=2, default=str))

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
    @with_appcontext
    def production_readiness(strict: bool) -> None:
        """Print local live-production readiness diagnostics."""

        payload = _production_readiness_payload()
        click.echo(json.dumps(payload, indent=2, default=str))
        if strict and payload["blockers"]:
            raise click.exceptions.Exit(1)

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
    @click.option("--model", "model_types", default="both", show_default=True, type=click.Choice(["sklearn", "xgboost", "both"]))
    @click.option("--confirm", default="", help="Must be TRAIN-OFFLINE-RANKER.")
    @with_appcontext
    def train_offline_ranker(horizon: str, model_types: str, confirm: str) -> None:
        """Train candidate offline ranker artifacts from historical and quarantined outcomes."""

        if confirm != "TRAIN-OFFLINE-RANKER":
            raise click.ClickException("Refusing offline training. Pass --confirm TRAIN-OFFLINE-RANKER.")

        result = get_service("offline_ranker").train(horizon, model_types=model_types)
        click.echo(json.dumps(result, indent=2, default=str))

    @app.cli.command("promote-offline-ranker")
    @click.option("--horizon", default="1h", show_default=True)
    @click.option("--model-id", required=True, type=int)
    @click.option("--confirm", default="", help="Must be PROMOTE-OFFLINE-RANKER.")
    @with_appcontext
    def promote_offline_ranker(horizon: str, model_id: int, confirm: str) -> None:
        """Promote one candidate offline ranker artifact after guardrail checks."""

        if confirm != "PROMOTE-OFFLINE-RANKER":
            raise click.ClickException("Refusing offline promotion. Pass --confirm PROMOTE-OFFLINE-RANKER.")

        result = get_service("offline_ranker").promote(horizon, model_id=model_id)
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

        if submit and confirm != "LIVE-CANARY-TRADE":
            raise click.ClickException("Refusing live canary submission. Pass --confirm LIVE-CANARY-TRADE.")

        result = _live_canary_trade_payload(
            ranking_id=ranking_id,
            user_id=user_id,
            connection_id=connection_id,
            submit=submit,
        )
        click.echo(json.dumps(result, indent=2, default=str))


def _live_canary_trade_payload(
    *,
    ranking_id: int,
    user_id: int,
    connection_id: int | None,
    submit: bool,
) -> dict[str, object]:
    config_preview_only = _config_flag("CANARY_PREVIEW_ONLY", True)
    preview_only = bool(config_preview_only or not submit)
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
    output_blockers = list(dict.fromkeys(blockers))
    if submit and config_preview_only:
        output_blockers = list(dict.fromkeys([*output_blockers, "canary_preview_only_enabled"]))
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
        "mode": "live",
        "submit_requested": submit,
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
        "live_canary_readiness": {
            "ready": ready,
            "preview_only": preview_only,
            "requires_exact_confirmation_for_submit": True,
            "uses_existing_live_caps": True,
            "optimizer_research_only": True,
            "canary_preview_only_config": config_preview_only,
            "real_order_submitted": False,
        },
    }

    if preview_only:
        audit_id = _record_live_canary_preview_audit(
            ranking=ranking,
            user=user,
            connection=connection,
            payload=payload,
            intent=intent,
            submit_requested=submit,
            blocked_by_preview_only=bool(submit and config_preview_only),
        )
        if audit_id is not None:
            payload["canary_preview_audit_id"] = audit_id

    if submit:
        if config_preview_only:
            payload["submitted"] = False
            payload["real_order_submitted"] = False
            payload["submit_blocked"] = True
            payload["submit_block_reason"] = "CANARY_PREVIEW_ONLY is true; live canary submit is disabled."
            return payload
        if not ready or intent is None:
            payload["submitted"] = False
            payload["real_order_submitted"] = False
            payload["submit_blocked"] = True
            return payload
        order = get_service("order_manager").place_order(intent)
        audit = AuditLog(
            category="orders",
            action="live_canary_trade",
            message=f"Live canary trade submitted from ranking {ranking.id}.",
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
            "risk_decision": risk_decision_payload,
            "signal_quality": diagnostics.get("signal_quality", {}),
        }
        db.session.add(audit)
        db.session.commit()
        payload["submitted"] = True
        payload["real_order_submitted"] = True
        payload["preview_only"] = False
        payload["live_canary_readiness"]["preview_only"] = False
        payload["live_canary_readiness"]["real_order_submitted"] = True
        payload["order"] = {
            "id": order.id,
            "status": order.status,
            "risk_status": order.risk_status,
            "rejection_reason": order.rejection_reason,
            "client_order_id": order.client_order_id,
        }

    return payload


def _config_flag(name: str, default: bool = False) -> bool:
    value = current_app.config.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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
            if not connection.is_active:
                return None, "active_verified_live_connection_missing"
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
    signal: object,
    signal_diagnostics: dict[str, object],
) -> tuple[OrderIntent | None, list[str], dict[str, object]]:
    blockers: list[str] = []
    mid = _float(signal_diagnostics.get("mid"))
    side = str(getattr(signal, "action", "")).lower()
    stop_loss = _float(getattr(signal, "stop_loss", None))
    take_profit = _float(getattr(signal, "take_profit", None))
    if mid <= 0:
        blockers.append("price_unavailable")
    if stop_loss <= 0:
        blockers.append("stop_loss_required")
    elif side == "buy" and stop_loss >= mid:
        blockers.append("invalid_stop_loss")
    elif side == "sell" and stop_loss <= mid:
        blockers.append("invalid_stop_loss")
    if side not in {"buy", "sell"}:
        blockers.append("signal_not_actionable")

    parameters = dict(ranking.parameters or {})
    cap = _canary_effective_notional_cap(ranking, parameters)
    if cap <= 0:
        blockers.append("max_notional_missing")
    stop_distance_pct = abs(mid - stop_loss) / max(mid, 1e-9) if mid > 0 and stop_loss > 0 else 0.0
    risk_fraction = _clamp(
        _float(parameters.get("risk_fraction"), _float(current_app.config.get("RISK_PER_TRADE_PCT"), 0.01)),
        0.0,
        _float(current_app.config.get("VAULT_MAX_RISK_FRACTION"), 0.03),
    )
    risk_budget = cap * risk_fraction
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
            "effective_notional_cap": cap,
            "risk_fraction": risk_fraction,
            "risk_budget": risk_budget,
            "stop_distance_pct": stop_distance_pct,
            "notional_by_risk": notional_by_risk,
            "notional_by_signal": notional_by_signal,
            "computed_notional": notional,
        },
    }
    if blockers:
        return None, blockers, {
            "sizing": metadata["canary_sizing"],
            "projected_order": {},
        }

    intent = OrderIntent(
        symbol=ranking.symbol,
        side=side,
        quantity=quantity,
        mode="live",
        order_type=order_type,
        limit_price=limit_price,
        reduce_only=False,
        leverage=max(1.0, min(_float(ranking.leverage, 1.0), _float(current_app.config.get("MAX_LEVERAGE"), 1.0))),
        stop_loss=stop_loss,
        take_profit=take_profit if take_profit > 0 else None,
        strategy_name=ranking.strategy_name,
        timeframe=ranking.timeframe,
        slippage_pct=_float(current_app.config.get("MAX_SLIPPAGE_PCT"), 0.0) / 2,
        user_id=user_id,
        trading_connection_id=connection_id,
        metadata=metadata,
    )
    return intent, [], {
        "sizing": metadata["canary_sizing"],
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


def _canary_effective_notional_cap(ranking: StrategyRanking, parameters: dict[str, object]) -> float:
    caps = [
        _float(current_app.config.get("MAX_POSITION_NOTIONAL")),
        _float(ranking.allocation_amount_usd),
        _float(parameters.get("allocation_cap_usd")),
    ]
    if bool(parameters.get("high_upside_profile", False)):
        high_upside_cap = _float(current_app.config.get("HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD"))
        if high_upside_cap > 0:
            caps.append(high_upside_cap)
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


def _table_exists(table: str) -> bool:
    rows = db.session.execute(db.text(f"PRAGMA table_info({table})")).mappings().all()
    return bool(rows)


def _table_count(table: str) -> int:
    if not _table_exists(table):
        return 0
    return _scalar_count(f"SELECT COUNT(*) FROM {table}")


def _drop_legacy_tables() -> None:
    for table in ("paper_equity_snapshot", "paper_account"):
        db.session.execute(db.text(f"DROP TABLE IF EXISTS {table}"))


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
    }


def _wallet_readiness_payload() -> dict[str, object]:
    custody = get_service("wallet_custody")
    readiness = custody.readiness()
    readiness.update(
        {
            "dependencies": _dependency_status(),
            "pending_withdrawals": WalletWithdrawal.query.filter(
                WalletWithdrawal.status.in_(["pending_approval", "pending_submission"])
            ).count(),
            "generated_address_count": WalletAddress.query.filter(
                WalletAddress.encrypted_metadata_json.like('%"custody": "in_app"%')
            ).count(),
            "sync_failures": WalletAuditLog.query.filter_by(action="wallet_deposit_sync_failed").count(),
        }
    )
    return readiness


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
    return [
        order
        for order in query.order_by(Order.created_at.asc()).all()
        if order.details.get("vault_cycle_id") == cycle.id
    ]


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


def _production_readiness_payload() -> dict[str, object]:
    blockers: list[str] = []
    warnings: list[str] = []
    wallet = _wallet_readiness_payload()
    dependencies = _dependency_status()
    risk_status = get_service("risk_engine").status("live")
    connection_health = _refresh_active_connection_health()
    db_uri = str(current_app.config.get("SQLALCHEMY_DATABASE_URI", ""))
    secret_key = str(current_app.config.get("SECRET_KEY", "") or "")
    totp_key = str(current_app.config.get("TOTP_ENCRYPTION_KEY", "") or "").strip()
    caps = current_app.config.get("WALLET_MAX_WITHDRAWAL_BY_ASSET") or {}
    required_cap_assets = ("ETH", "USDC", "USDT", "BTC", "SOL", "XRP")

    if current_app.config.get("APP_MODE") != "live":
        blockers.append("APP_MODE must be live")
    if not bool(current_app.config.get("ENABLE_LIVE_TRADING", False)):
        blockers.append("ENABLE_LIVE_TRADING must be true")
    if "paper_trading" in current_app.extensions.get("services", {}):
        blockers.append("Simulated trading service must not be registered")
    if not db_uri.startswith("sqlite:///") or db_uri in {"sqlite://", "sqlite:///:memory:"}:
        blockers.append("DATABASE_URL must point at a file-backed local SQLite database")
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
        missing = missing_wallet
        blockers.append("Missing wallet dependencies: " + ", ".join(missing))
    missing_ml = [name for name in ("joblib", "numpy", "scipy", "sklearn", "xgboost") if not dependencies.get(name, False)]
    if bool(current_app.config.get("ML_OFFLINE_MODELS_ENABLED", False)) and missing_ml:
        blockers.append("Missing offline ML dependencies: " + ", ".join(missing_ml))
    for health in connection_health:
        if not bool(health.get("can_trade", False)):
            provider = str(health.get("provider") or "exchange").title()
            reason = str(health.get("failure_reason") or "latest live access check failed")
            blockers.append(f"{provider} active connection cannot trade: {reason}")
            if health.get("client_ip"):
                warnings.append(f"{provider} API key whitelist must include current client IP {health.get('client_ip')}")
    if not bool(wallet.get("ready", False)):
        blockers.extend(str(item) for item in wallet.get("blockers", []))
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
    if not current_app.config.get("ADMIN_PASSWORD") and _table_count("user") == 0:
        warnings.append("No seeded admin password and no users exist yet")

    return {
        "ready": not blockers,
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "mode": {
            "app_mode": current_app.config.get("APP_MODE"),
            "enable_live_trading": bool(current_app.config.get("ENABLE_LIVE_TRADING", False)),
            "current_mode": Setting.get_json("current_mode", "live"),
        },
        "database": {
            "uri": db_uri,
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
        },
        "risk": risk_status,
        "connection_health": connection_health,
        "wallet": wallet,
        "offline_ml": _offline_ml_readiness("1h"),
        "dependencies": dependencies,
    }


def _refresh_active_connection_health() -> list[dict[str, object]]:
    service = get_service("trading_connections")
    refreshed: list[dict[str, object]] = []
    connections = TradingConnection.query.filter_by(is_active=True).order_by(
        TradingConnection.updated_at.desc(),
        TradingConnection.id.desc(),
    )
    for connection in connections:
        refreshed.append(_fresh_connection_health(connection, service))
    if refreshed:
        db.session.commit()
    return refreshed or active_connection_health()


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


def _offline_ml_readiness(horizon: str = "1h", *, require_blend: bool = True) -> dict[str, object]:
    try:
        readiness = get_service("offline_ranker").readiness(horizon, require_blend=require_blend)
    except Exception as exc:  # noqa: BLE001
        return {
            "ready": False,
            "horizon": horizon,
            "blockers": [str(exc)],
            "promoted_model": None,
        }
    latest = (
        MLOfflineModel.query.filter_by(horizon=str(horizon or "global").lower())
        .order_by(MLOfflineModel.created_at.desc())
        .first()
    )
    return {
        **readiness,
        "latest_model_id": latest.id if latest else None,
        "latest_model_status": latest.status if latest else None,
        "candidate_count": MLOfflineModel.query.filter_by(horizon=str(horizon or "global").lower(), status="candidate").count(),
    }


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
    cap_usdc = _duration_cap("HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION", duration_bucket)
    cap_pct = _duration_cap("HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION", duration_bucket)
    max_notional = float(current_app.config.get("HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD", 0.0) or 0.0)
    max_daily_loss = float(current_app.config.get("HIGH_UPSIDE_MAX_DAILY_LOSS_USDC", 0.0) or 0.0)
    blockers: list[str] = []
    if not bool(getattr(config, "high_upside_profile", False)):
        blockers.append("high_upside_profile_not_requested")
    if not bool(current_app.config.get("HIGH_UPSIDE_PROFILE_ENABLED", False)):
        blockers.append("HIGH_UPSIDE_PROFILE_ENABLED=false")
    if not bool(current_app.config.get("HIGH_UPSIDE_LIVE_ELIGIBLE", False)):
        blockers.append("HIGH_UPSIDE_LIVE_ELIGIBLE=false")
    if bool(risk_status.get("panic_lock", False)):
        blockers.append("panic_lock")
    if bool(risk_status.get("live_trading_blocked", False)):
        blockers.append("live_trading_blocked")
    if bool((risk_status.get("high_upside") or {}).get("auto_disabled", False)):
        blockers.append("high_upside_auto_disabled")
    if active_verified_connections <= 0:
        blockers.append("active_verified_live_connection_missing")
    if max_notional <= 0:
        blockers.append("HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD_missing")
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
    offline_ml = _offline_ml_readiness(duration_bucket, require_blend=False)
    if bool(current_app.config.get("HIGH_UPSIDE_REQUIRE_PROMOTED_ML", True)) and not bool(offline_ml.get("ready", False)):
        blockers.append("promoted_offline_ml_required")

    return {
        "requested": bool(getattr(config, "high_upside_profile", False)),
        "ready": not blockers,
        "blockers": blockers,
        "profile_enabled": bool(current_app.config.get("HIGH_UPSIDE_PROFILE_ENABLED", False)),
        "live_eligible": bool(current_app.config.get("HIGH_UPSIDE_LIVE_ELIGIBLE", False)),
        "auto_disabled": bool((risk_status.get("high_upside") or {}).get("auto_disabled", False)),
        "active_verified_connection_count": active_verified_connections,
        "duration_bucket": duration_bucket,
        "caps": {
            "max_position_notional_usd": max_notional,
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
        "no_default_live_cap_increase": True,
    }


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
        "notional_gate": {"max_position_notional": float(risk_status.get("max_position_notional", 0.0) or 0.0)},
        "slippage_gate": {"max_slippage_pct": float(risk_status.get("max_slippage_pct", 0.0) or 0.0)},
        "stop_loss_gate": {"required_for_opening_trades": True},
        "shadow_validation_gate": {
            "required": require_shadow_validation,
            "max_age_hours": float(current_app.config.get("ENSEMBLE_1H_SHADOW_VALIDATION_MAX_AGE_HOURS", 24.0) or 24.0),
        },
        "warnings": warnings,
    }
