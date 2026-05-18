"""Flask application factory."""

# ruff: noqa: E402,I001
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

_DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"
_PRODUCTION_DOTENV_TARGETS = {"vps", "production", "prod", "postgres", "staging", "vercel"}


def _env_flag(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _should_load_repo_dotenv(environ: Mapping[str, str | None] = os.environ, *, pytest_loaded: bool | None = None) -> bool:
    if pytest_loaded is None:
        pytest_loaded = "pytest" in sys.modules
    if pytest_loaded:
        return False
    flask_cli_will_load_dotenv = _env_flag(environ.get("FLASK_RUN_FROM_CLI")) and not environ.get("FLASK_SKIP_DOTENV")
    if flask_cli_will_load_dotenv:
        return False
    if _env_flag(environ.get("VERCEL")):
        return False
    deployment_target = str(environ.get("DEPLOYMENT_TARGET") or "").strip().lower()
    if deployment_target in _PRODUCTION_DOTENV_TARGETS:
        return False
    app_env = str(environ.get("APP_ENV") or environ.get("FLASK_ENV") or environ.get("FLASK_CONFIG") or "").strip().lower()
    return app_env not in {"production", "prod"}


if _should_load_repo_dotenv():
    load_dotenv(_DOTENV_PATH)

from .backtesting.engine import BacktestEngine
from .backtesting.optimizer import StrategyOptimizer
from .backtesting.vault_simulator import VaultBacktestSimulator
from .auth import current_user, password_hash
from .cli import register_cli
from .config import public_origin_violations, selected_config_class
from .csrf import csrf_input, csrf_token, validate_csrf_request
from .extensions import db, migrate
from .features.engine import FeatureEngine
from .ml.offline_ranker import OfflineRanker
from .ml.online_ranker import OnlineRanker
from .ml.decision_engine import MLDecisionEngine
from .ml.features import MLFeatureFactory
from .ml.signal_model import MLSignalModel
from .models import (
    AuditLog,
    MLModelRegistry,
    MLOfflineModel,
    Order,
    Setting,
    StrategyRun,
    User,
    VaultCycle,
    WalletAuditLog,
    WalletBalance,
    WalletTransaction,
    WorkerLease,
)
from .admin_auth import admin_authenticated, admin_configured
from .routes.admin import admin_bp, profit_share_api_bp
from .routes.auth import auth_bp
from .services.execution import HyperliquidVenue
from .routes.backtests import backtests_bp
from .routes.consumer import consumer_bp
from .routes.dashboard import dashboard_bp
from .routes.internal_mpc_signer import internal_mpc_signer_bp
from .routes.settings import settings_bp
from .services.chart_stream import ChartStreamService
from .services.hyperliquid_client import HyperliquidClient
from .services.invite_profit_share import InviteProfitShareService
from .services.leveraged_markets import LeveragedMarketDiscoveryService
from .services.market_scanner import MarketScannerService
from .services.market_structure import MarketStructureService
from .services.market_universe import MarketUniverseService
from .services.market_data import MarketDataService
from .services.model_registry import ModelRegistryService
from .services.dashboard_service import DashboardPayloadService
from .services.dashboard_prediction import DashboardPredictionService
from .services.forecast_performance import ForecastPerformanceService
from .services.market_data_quality import MarketDataQualityService
from .services.ml_projection_engine import MLProjectionEngine
from .services.opportunity_scanner import DashboardOpportunityScanner
from .services.order_manager import OrderManager
from .services.one_h10_forecast import OneH10ForecastService
from .services.pair_screening import PairScreeningService
from .services.platform_treasury import PlatformTreasuryService
from .services.rapid_ml_trader import RapidMLTraderService
from .services.risk_engine import RiskEngine
from .services.realtime_market import RealtimeMarketService
from .services.self_custody_wallet import SelfCustodyWalletService
from .services.treasury_solvency import TreasurySolvencyEngine
from .services.strategy_runner import StrategyManager
from .services.trading_connections import TradingConnectionService
from .services.vault_activity import VaultCycleActivityService
from .services.vault_cycle_allocator import VaultCycleAllocator
from .services.vault_cycle_orchestrator import VaultCycleOrchestrator
from .services.vault_cycle_reporting import VaultCycleReportingService
from .services.vault_cycle_settlement import VaultCycleSettlementService
from .services.vault_cycle_trading_enforcer import VaultCycleTradingEnforcer
from .services.vault_cycle_transfers import VaultCycleTransferService
from .services.vault_coherence import VaultCoherenceService
from .services.vault_readiness import VaultReadinessService
from .services.vault_selector import VaultStrategySelector
from .services.wallet_addresses import WalletAddressService
from .services.wallet_activity import WalletActivityService
from .services.audit_events import register_audit_retention_listener
from .services.wallet_custody import RealWalletCustodyService
from .services.wallet_summary import WalletSummaryService
from .services.worker_lease import WorkerLeaseService
from .settings_validation import RuntimeConfigError, validate_runtime_config
from .services.withdrawal_config import wallet_withdrawals_enabled
from .strategies.registry import StrategyRegistry
from .utils import format_duration_seconds


class DatabaseStartupError(RuntimeError):
    """Raised when startup database checks cannot reach the configured database."""


def create_app(test_config: dict | None = None) -> Flask:
    """Create and configure the Flask application."""

    instance_path = os.getenv("FLASK_INSTANCE_PATH") or ("/tmp/algorithm-vault-instance" if os.getenv("VERCEL") else None)
    flask_kwargs: dict[str, Any] = {"template_folder": "../templates", "static_folder": "../static"}
    if instance_path:
        flask_kwargs["instance_path"] = instance_path
    app = Flask(__name__, **flask_kwargs)
    app.config.from_object(selected_config_class())
    if test_config:
        app.config.update(test_config)
        if app.config.get("TESTING") and "WTF_CSRF_ENABLED" not in test_config:
            app.config["WTF_CSRF_ENABLED"] = False
    _configure_runtime(app)
    _configure_engine_options(app)

    db.init_app(app)
    register_audit_retention_listener()
    if migrate is not None:
        migrate.init_app(app, db)

    strategy_registry = StrategyRegistry()
    hyperliquid_client = HyperliquidClient(app.config)
    market_data = MarketDataService(app.config, hyperliquid_client)
    realtime_market = RealtimeMarketService(app.config, market_data)
    market_structure = MarketStructureService(app.config, market_data)
    market_universe = MarketUniverseService(app.config, market_data)
    feature_engine = FeatureEngine()
    market_scanner = MarketScannerService(app.config, market_data, market_universe, feature_engine, online_ranker=None)
    execution_venue = HyperliquidVenue(hyperliquid_client, market_data)
    risk_engine = RiskEngine(app.config)
    trading_connections = TradingConnectionService(app.config)
    online_ranker = OnlineRanker(app.config)
    offline_ranker = OfflineRanker(app.config)
    ml_signal_model = MLSignalModel(app.config)
    ml_feature_factory = MLFeatureFactory(app.config, feature_engine)
    ml_decision_engine = MLDecisionEngine(app.config, signal_model=ml_signal_model)
    ml_decision_engine.feature_factory = ml_feature_factory
    model_registry = ModelRegistryService(app.config)
    vault_coherence = VaultCoherenceService(app.config)
    market_data_quality = MarketDataQualityService(app.config)
    dashboard_prediction = DashboardPredictionService(app.config)
    forecast_performance = ForecastPerformanceService(app.config)
    one_h10_forecast = OneH10ForecastService(app.config, ml_decision_engine, vault_coherence)
    market_scanner.online_ranker = online_ranker
    market_scanner.offline_ranker = offline_ranker
    market_scanner.ml_decision_engine = ml_decision_engine
    pair_screening = PairScreeningService(app.config, market_data, market_universe, market_structure, online_ranker)
    market_scanner.pair_screening = pair_screening
    leveraged_markets = LeveragedMarketDiscoveryService(app.config, market_data, trading_connections, ml_feature_factory)
    ml_projection_engine = MLProjectionEngine(
        app.config,
        market_data,
        feature_engine,
        one_h10_forecast,
        market_data_quality,
        dashboard_prediction,
    )
    dashboard_opportunities = DashboardOpportunityScanner(
        app.config,
        leveraged_markets,
        market_scanner,
        ml_projection_engine,
        trading_connections,
        forecast_performance,
    )
    order_manager = OrderManager(app.config, hyperliquid_client, market_data, risk_engine, trading_connections)
    dashboard_payload = DashboardPayloadService(app, app.config)
    chart_stream = ChartStreamService(app.config, dashboard_opportunities, dashboard_payload)
    rapid_ml_trader = RapidMLTraderService(
        app.config,
        trading_connections,
        market_data,
        ml_decision_engine,
        offline_ranker,
        order_manager,
        leveraged_markets,
    )
    backtest_engine = BacktestEngine(
        app.config, strategy_registry, market_data, ml_decision_engine=ml_decision_engine, ml_feature_factory=ml_feature_factory
    )
    backtest_vault_simulator = VaultBacktestSimulator(
        app.config,
        strategy_registry,
        market_data,
        backtest_engine,
        leveraged_markets=leveraged_markets,
        trading_connections=trading_connections,
        ml_projection_engine=ml_projection_engine,
        market_scanner=market_scanner,
        ml_decision_engine=ml_decision_engine,
    )
    strategy_optimizer = StrategyOptimizer(
        app.config,
        strategy_registry,
        market_data,
        backtest_engine,
        market_universe,
        online_ranker,
        market_structure,
        pair_screening,
        offline_ranker,
        ml_decision_engine,
        ml_feature_factory,
    )
    vault_strategy_selector = VaultStrategySelector(
        app.config,
        market_data,
        hyperliquid_client,
        market_universe,
        online_ranker,
        realtime_market,
        market_scanner,
        market_structure,
        pair_screening,
    )
    vault_strategy_selector.ml_decision_engine = ml_decision_engine
    vault_activity = VaultCycleActivityService()
    wallet_address_service = WalletAddressService(app.config)
    wallet_activity = WalletActivityService()
    wallet_custody = RealWalletCustodyService(app.config)
    platform_treasury = PlatformTreasuryService(app.config)
    invite_profit_share = InviteProfitShareService(app.config)
    treasury_solvency = TreasurySolvencyEngine(app.config)
    wallet_summary = WalletSummaryService()
    self_custody_wallet = SelfCustodyWalletService(app.config)
    strategy_manager = StrategyManager(
        app,
        app.config,
        strategy_registry,
        market_data,
        order_manager,
        feature_engine,
        online_ranker,
        realtime_market,
        ml_signal_model,
        ml_decision_engine,
    )
    vault_cycle_allocator = VaultCycleAllocator(app.config, trading_connections, leveraged_markets, market_scanner)
    vault_cycle_transfers = VaultCycleTransferService(app.config, trading_connections)
    vault_cycle_settlement = VaultCycleSettlementService(
        app.config,
        trading_connections,
        strategy_manager,
        vault_cycle_transfers,
    )
    vault_cycle_trading_enforcer = VaultCycleTradingEnforcer(
        app.config,
        trading_connections,
        vault_strategy_selector,
        strategy_manager,
        order_manager,
        leveraged_markets,
        market_data,
        vault_cycle_settlement,
    )
    vault_cycle_orchestrator = VaultCycleOrchestrator(
        app.config,
        trading_connections,
        vault_cycle_allocator,
        vault_cycle_transfers,
        vault_cycle_settlement,
        vault_strategy_selector,
        strategy_manager,
        vault_cycle_trading_enforcer,
    )
    vault_cycle_reporting = VaultCycleReportingService()
    vault_readiness = VaultReadinessService(app.config)
    worker_lease = WorkerLeaseService(app.config)

    app.extensions["services"] = {
        "strategy_registry": strategy_registry,
        "hyperliquid_client": hyperliquid_client,
        "execution_venue": execution_venue,
        "market_data": market_data,
        "realtime_market": realtime_market,
        "dashboard_payload": dashboard_payload,
        "market_data_quality": market_data_quality,
        "dashboard_prediction": dashboard_prediction,
        "forecast_performance": forecast_performance,
        "market_structure": market_structure,
        "market_universe": market_universe,
        "market_scanner": market_scanner,
        "pair_screening": pair_screening,
        "leveraged_markets": leveraged_markets,
        "ml_projection_engine": ml_projection_engine,
        "dashboard_opportunities": dashboard_opportunities,
        "chart_stream": chart_stream,
        "feature_engine": feature_engine,
        "risk_engine": risk_engine,
        "trading_connections": trading_connections,
        "online_ranker": online_ranker,
        "offline_ranker": offline_ranker,
        "ml_signal_model": ml_signal_model,
        "ml_feature_factory": ml_feature_factory,
        "ml_decision_engine": ml_decision_engine,
        "model_registry": model_registry,
        "vault_coherence": vault_coherence,
        "one_h10_forecast": one_h10_forecast,
        "order_manager": order_manager,
        "rapid_ml_trader": rapid_ml_trader,
        "backtest_engine": backtest_engine,
        "backtest_vault_simulator": backtest_vault_simulator,
        "strategy_optimizer": strategy_optimizer,
        "vault_strategy_selector": vault_strategy_selector,
        "vault_activity": vault_activity,
        "wallet_address_service": wallet_address_service,
        "wallet_activity": wallet_activity,
        "wallet_custody": wallet_custody,
        "platform_treasury": platform_treasury,
        "invite_profit_share": invite_profit_share,
        "treasury_solvency": treasury_solvency,
        "wallet_summary": wallet_summary,
        "self_custody_wallet": self_custody_wallet,
        "strategy_manager": strategy_manager,
        "vault_cycle_allocator": vault_cycle_allocator,
        "vault_cycle_transfers": vault_cycle_transfers,
        "vault_cycle_settlement": vault_cycle_settlement,
        "vault_cycle_trading_enforcer": vault_cycle_trading_enforcer,
        "vault_cycle_orchestrator": vault_cycle_orchestrator,
        "vault_cycle_reporting": vault_cycle_reporting,
        "vault_readiness": vault_readiness,
        "worker_lease": worker_lease,
    }

    app.register_blueprint(auth_bp)
    app.register_blueprint(consumer_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(profit_share_api_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(backtests_bp)
    app.register_blueprint(internal_mpc_signer_bp)
    register_cli(app)

    _register_operational_routes(app)
    _register_error_handlers(app)
    app.before_request(lambda: _block_when_database_startup_failed(app))
    app.before_request(lambda: _rate_limit_request(app))
    app.before_request(validate_csrf_request)
    app.after_request(lambda response: _set_response_headers(app, response))
    app.jinja_env.filters["duration"] = format_duration_seconds

    @app.context_processor
    def inject_globals():
        mode = "live"
        return {
            "current_app": app,
            "nav_mode": mode,
            "live_enabled": app.config["ENABLE_LIVE_TRADING"],
            "current_user": current_user(),
            "admin_authenticated": admin_authenticated(),
            "admin_configured": admin_configured(),
            "crypto_rail_assets": _crypto_rail_assets(),
            "csrf_token": csrf_token,
            "csrf_input": csrf_input,
            "format_duration_seconds": format_duration_seconds,
        }

    with app.app_context():
        try:
            _run_database_startup(app)
        except Exception as exc:
            if not _should_defer_database_startup_error(app, exc):
                raise
            _record_database_startup_error(app, exc)

    return app


class _JsonLogFormatter(logging.Formatter):
    """Small JSON formatter for production logs without request body data."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        try:
            if request:
                payload.update(
                    {
                        "method": request.method,
                        "path": request.path,
                        "remote_addr": request.remote_addr,
                    }
                )
        except RuntimeError:
            pass
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def _configure_runtime(app: Flask) -> None:
    _validate_public_origins(app)
    strict_config = not bool(app.config.get("TESTING", False)) or bool(app.config.get("STRICT_CONFIG_VALIDATION", False))
    validation = validate_runtime_config(app.config, strict=strict_config)
    app.config["RUNTIME_CONFIG_VALIDATION"] = validation
    if bool(app.config.get("PROXY_FIX_ENABLED", False)):
        app.wsgi_app = ProxyFix(
            app.wsgi_app,
            x_for=max(0, int(app.config.get("PROXY_FIX_X_FOR", 1) or 1)),
            x_proto=max(0, int(app.config.get("PROXY_FIX_X_PROTO", 1) or 1)),
            x_host=max(0, int(app.config.get("PROXY_FIX_X_HOST", 1) or 1)),
            x_port=max(0, int(app.config.get("PROXY_FIX_X_PORT", 1) or 1)),
            x_prefix=max(0, int(app.config.get("PROXY_FIX_X_PREFIX", 0) or 0)),
        )
    app.permanent_session_lifetime = timedelta(
        seconds=max(60, int(app.config.get("PERMANENT_SESSION_LIFETIME_SECONDS", 60 * 60 * 8) or 60 * 60 * 8))
    )
    _configure_logging(app)


def _validate_public_origins(app: Flask) -> None:
    deployment_target = str(app.config.get("DEPLOYMENT_TARGET", "local") or "local").strip().lower()
    require_public_https = deployment_target in {"vps", "production", "prod", "postgres", "vercel"}
    if not require_public_https:
        return

    errors: list[str] = []
    for key in ("PUBLIC_APP_ORIGIN", "PUBLIC_API_ORIGIN", "PUBLIC_LIVE_API_ORIGIN"):
        origin = str(app.config.get(key, "") or "").strip()
        if key == "PUBLIC_LIVE_API_ORIGIN" and not origin:
            continue
        violations = public_origin_violations(origin, require_public_https=True)
        if violations:
            errors.append(f"{key} {', '.join(violations)}")
    if errors:
        raise RuntimeError("Invalid production public origin configuration: " + "; ".join(errors))


def _configure_logging(app: Flask) -> None:
    level_name = str(app.config.get("LOG_LEVEL", "INFO") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter: logging.Formatter
    if str(app.config.get("LOG_FORMAT", "plain") or "plain").lower() == "json":
        formatter = _JsonLogFormatter()
    else:
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s")

    for logger_name in ("AlgorithmVault", "app"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        else:
            for handler in logger.handlers:
                handler.setFormatter(formatter)
    app.logger.setLevel(level)


def _register_operational_routes(app: Flask) -> None:
    @app.get("/manifest.json")
    def pwa_manifest_json():
        return send_from_directory(
            app.static_folder,
            "manifest.json",
            mimetype="application/manifest+json",
            max_age=0,
        )

    @app.get("/sw.js")
    def pwa_service_worker():
        return send_from_directory(
            Path(app.static_folder) / "js",
            "sw.js",
            mimetype="application/javascript",
            max_age=0,
        )

    @app.get("/icons/<path:filename>")
    def pwa_icon(filename: str):
        return send_from_directory(Path(app.static_folder) / "icons", filename)

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "service": app.config.get("APP_NAME", "Algorithm Vault")})

    @app.get("/readyz")
    def readyz():
        checks: dict[str, Any] = {
            "database": False,
            "services": False,
            "config": True,
        }
        status = 200
        startup_error = app.config.get("DATABASE_STARTUP_ERROR")
        if startup_error:
            checks["database_error"] = app.config.get("DATABASE_STARTUP_ERROR_CLASS", "DatabaseStartupError")
            checks["database_startup_blocked"] = True
            status = 503
        else:
            try:
                db.session.execute(text("SELECT 1"))
                checks["database"] = True
            except Exception as exc:  # noqa: BLE001
                current_error = str(exc.__class__.__name__)
                checks["database_error"] = current_error
                status = 503

        required_services = {"market_data", "risk_engine", "trading_connections", "strategy_manager"}
        registered = set(app.extensions.get("services", {}).keys())
        missing = sorted(required_services - registered)
        checks["services"] = not missing
        if missing:
            checks["missing_services"] = missing
            status = 503

        deployment_target = str(app.config.get("DEPLOYMENT_TARGET", "local") or "local")
        db_backend = _database_backend(app)
        validation = app.config.get("RUNTIME_CONFIG_VALIDATION")
        validation_blockers = list(getattr(validation, "blockers", ()))
        if validation_blockers:
            checks["config"] = False
            checks["config_blockers"] = validation_blockers
            status = 503
        if deployment_target in {"vps", "production", "postgres", "vercel"} and db_backend != "postgres":
            if bool(app.config.get("RECOVERY_SQLITE_ACTIVE", False)):
                recovery = _database_recovery_status(app)
                checks["database_recovery_mode"] = True
                checks["configured_postgres"] = recovery["configured_postgres"]
                checks["trading_disabled"] = recovery["trading_disabled"]
                checks["database_recovery_message"] = recovery["message"]
            else:
                checks["config"] = False
                checks["config_warning"] = "production target expects postgres"
                status = 503

        return jsonify({"ok": status == 200, "checks": checks, "deployment_target": deployment_target, "database": db_backend}), status

    @app.get("/ops/status")
    def ops_status():
        return jsonify(_operational_status(app))


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        if _prefers_json_response():
            return jsonify({"ok": False, "error": exc.name, "status": exc.code}), exc.code
        return (
            f"<!doctype html><title>{exc.code} {exc.name}</title><main><h1>{exc.name}</h1><p>{exc.description}</p></main>",
            exc.code,
            {"Content-Type": "text/html; charset=utf-8"},
        )

    @app.errorhandler(Exception)
    def unhandled_error(exc: Exception):
        if isinstance(exc, RuntimeConfigError):
            app.logger.error("Runtime configuration rejected: %s", exc)
        if app.config.get("TESTING"):
            raise exc
        app.logger.exception("Unhandled request error")
        if _prefers_json_response():
            return jsonify({"ok": False, "error": "internal_server_error", "status": 500}), 500
        return (
            "<!doctype html><title>Application Error</title><main><h1>Application Error</h1>"
            "<p>The request could not be completed.</p></main>",
            500,
            {"Content-Type": "text/html; charset=utf-8"},
        )


def _operational_status(app: Flask) -> dict[str, Any]:
    now = datetime.utcnow()
    validation = app.config.get("RUNTIME_CONFIG_VALIDATION")
    payload: dict[str, Any] = {
        "ok": True,
        "generated_at": now.isoformat(),
        "deployment_target": app.config.get("DEPLOYMENT_TARGET", "local"),
        "worker_mode": app.config.get("WORKER_MODE", "web"),
        "panic_lock": _safe_setting_json("panic_lock", False),
        "migration_version": _safe_scalar("SELECT version_num FROM alembic_version LIMIT 1"),
        "runtime_config": {
            "ok": bool(getattr(validation, "ok", True)),
            "blockers": list(getattr(validation, "blockers", ())),
            "app_mode": app.config.get("APP_MODE", "paper"),
            "database_backend": getattr(validation, "database_backend", _database_backend(app)),
            "database_recovery_mode": bool(app.config.get("RECOVERY_SQLITE_ACTIVE", False)),
            "live_trading_enabled": bool(app.config.get("ENABLE_LIVE_TRADING", False)),
            "custody_mode": app.config.get("WALLET_CUSTODY_MODE", "local_dev"),
            "withdrawals_enabled": wallet_withdrawals_enabled(app.config),
            "worker_process_configured": bool(app.config.get("WORKER_PROCESS_CONFIGURED", False)),
            "in_process_workers_enabled": bool(app.config.get("ENABLE_IN_PROCESS_WORKERS", False)),
        },
    }
    payload["database_recovery"] = _database_recovery_status(app)
    payload["workers"] = _worker_status(now)
    payload["trading"] = _trading_status(now, app)
    payload["wallets"] = _wallet_status()
    payload["models"] = _model_status()
    payload["treasury"] = _treasury_status(app)
    payload["live_operations"] = _live_operations_status(app, payload["runtime_config"], payload["treasury"])
    payload["observability"] = {
        "provider_failure_count_24h": _audit_count("provider", "failed"),
        "order_rejection_count_24h": _order_rejection_count(),
        "chart_refresh_lag_seconds": _chart_refresh_lag(app, now),
    }
    return payload


def _database_recovery_status(app: Flask) -> dict[str, Any]:
    active = bool(app.config.get("RECOVERY_SQLITE_ACTIVE", False))
    runtime_database = _database_backend(app)
    configured_postgres = _configured_postgres_status(app, probe=active)
    trading_disabled = active or not bool(app.config.get("ENABLE_LIVE_TRADING", False))
    if active and configured_postgres.get("available") is False:
        message = "Recovery SQLite is active because configured Postgres is unavailable; live trading remains disabled."
    elif active and configured_postgres.get("available") is True:
        message = "Recovery SQLite is active; complete the Postgres cutover checklist before disabling recovery mode."
    elif active:
        message = "Recovery SQLite is active; live trading remains disabled until a healthy Postgres cutover is verified."
    else:
        message = "Recovery SQLite is inactive."
    return {
        "active": active,
        "runtime_database": runtime_database,
        "configured_postgres": configured_postgres,
        "trading_disabled": trading_disabled,
        "message": message,
    }


def _configured_postgres_status(app: Flask, *, probe: bool) -> dict[str, Any]:
    configured_url = str(app.config.get("CONFIGURED_DATABASE_URL") or app.config.get("SQLALCHEMY_DATABASE_URI") or "").strip()
    backend = _database_backend_from_uri(configured_url)
    parsed = _safe_database_url(configured_url)
    status: dict[str, Any] = {
        "configured": bool(configured_url),
        "backend": backend,
        "host": parsed.host if parsed is not None else "",
        "available": None,
        "status": "not_configured" if not configured_url else "not_postgres",
        "error_category": "",
        "message": "",
    }
    if not configured_url:
        status["message"] = "DATABASE_URL is not configured."
        return status
    if backend != "postgres":
        status["message"] = "Configured DATABASE_URL is not PostgreSQL."
        return status
    if not probe:
        status["status"] = "skipped"
        status["message"] = "Configured Postgres probe skipped because recovery SQLite is inactive."
        return status

    now = time.monotonic()
    ttl_seconds = max(float(app.config.get("RECOVERY_POSTGRES_PROBE_TTL_SECONDS", 60.0) or 60.0), 0.0)
    cache = app.config.get("_RECOVERY_POSTGRES_PROBE_CACHE")
    if isinstance(cache, dict) and cache.get("url") == configured_url and float(cache.get("expires_at", 0.0) or 0.0) > now:
        cached_status = dict(cache.get("status") or {})
        if cached_status:
            return cached_status

    timeout_seconds = max(min(float(app.config.get("RECOVERY_POSTGRES_PROBE_TIMEOUT_SECONDS", 2.0) or 2.0), 10.0), 0.5)
    try:
        _probe_configured_postgres(configured_url, timeout_seconds=timeout_seconds)
    except Exception as exc:  # noqa: BLE001
        category = _classify_postgres_probe_error(exc)
        status.update(
            {
                "available": False,
                "status": "unavailable",
                "error_category": category,
                "message": f"Configured Postgres probe failed ({category}); recovery SQLite remains active.",
            }
        )
    else:
        status.update(
            {
                "available": True,
                "status": "healthy",
                "message": "Configured Postgres probe succeeded; complete migrations and cutover checks before disabling recovery SQLite.",
            }
        )
    app.config["_RECOVERY_POSTGRES_PROBE_CACHE"] = {"url": configured_url, "expires_at": now + ttl_seconds, "status": dict(status)}
    return status


def _safe_database_url(configured_url: str):
    if not configured_url:
        return None
    try:
        return make_url(configured_url)
    except Exception:  # noqa: BLE001
        return None


def _probe_configured_postgres(configured_url: str, *, timeout_seconds: float) -> None:
    engine = create_engine(
        configured_url,
        connect_args={"connect_timeout": max(int(timeout_seconds), 1)},
        future=True,
        poolclass=NullPool,
        pool_pre_ping=True,
    )
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1")).scalar()
    finally:
        engine.dispose()


def _classify_postgres_probe_error(exc: BaseException) -> str:
    message = " ".join(str(item).lower() for item in _exception_chain(exc))
    if "planlimitreached" in message or "plan limit" in message or "plan_limit" in message:
        return "plan_limit_reached"
    if "suspended" in message or "suspend" in message:
        return "suspended"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if "password authentication failed" in message or "authentication failed" in message:
        return "authentication_failed"
    if "could not translate host" in message or "name or service not known" in message:
        return "dns_failed"
    return "connection_failed"


def _live_operations_status(app: Flask, runtime_config: dict[str, Any], treasury: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    target = str(app.config.get("DEPLOYMENT_TARGET", "local") or "local").strip().lower()
    app_mode = str(app.config.get("APP_MODE", "paper") or "paper").strip().lower()
    db_backend = str(runtime_config.get("database_backend") or _database_backend(app))
    custody_mode = (
        str(runtime_config.get("custody_mode") or app.config.get("WALLET_CUSTODY_MODE", "local_dev") or "local_dev").strip().lower()
    )

    if target not in {"vercel", "production", "prod", "vps", "postgres", "staging"}:
        blockers.append("DEPLOYMENT_TARGET must be production-like for live operations")
    if db_backend != "postgres":
        blockers.append("live operations require PostgreSQL DATABASE_URL")
    if bool(runtime_config.get("database_recovery_mode", False)):
        blockers.append("recovery SQLite mode must be inactive for live operations")
    if app_mode != "live":
        blockers.append("APP_MODE must be live")
    if not bool(runtime_config.get("live_trading_enabled", False)):
        blockers.append("ENABLE_LIVE_TRADING must be enabled")
    if bool(runtime_config.get("in_process_workers_enabled", False)):
        blockers.append("in-process workers must stay disabled for Vercel web")
    if not bool(runtime_config.get("worker_process_configured", False)):
        blockers.append("dedicated worker process must be configured")
    if custody_mode not in {"kms", "hsm", "mpc"}:
        blockers.append("production custody mode must be kms, hsm, or mpc")
    if not bool(runtime_config.get("withdrawals_enabled", False)):
        blockers.append("wallet withdrawals are not enabled by server-side readiness gates")
    if treasury.get("ready") is not True:
        blockers.append("platform gas treasury is not ready")

    blockers.extend(str(blocker) for blocker in runtime_config.get("blockers", ()) if str(blocker))
    blockers.extend(str(blocker) for blocker in treasury.get("blockers", ()) if str(blocker))
    blockers = list(dict.fromkeys(blockers))

    reserve_health = treasury.get("reserve_health") if isinstance(treasury, dict) else {}
    reserve_state = str((reserve_health or {}).get("state") or "").strip().lower()
    if treasury.get("ready") is True and reserve_state in {"warning", "low", "critical", "emergency"}:
        warnings.append(f"platform treasury reserve health is {reserve_state}; gas-sponsored withdrawals may queue until funded")
    try:
        treasury_eth_balance = float(treasury.get("eth_balance") or 0.0)
    except (TypeError, ValueError):
        treasury_eth_balance = 0.0
    if treasury.get("ready") is True and treasury_eth_balance <= 0:
        warnings.append("platform treasury has 0 ETH available for gas top-ups")
    warnings = list(dict.fromkeys(warnings))

    trading_ready = (
        app_mode == "live"
        and bool(runtime_config.get("live_trading_enabled", False))
        and bool(runtime_config.get("worker_process_configured", False))
        and not bool(runtime_config.get("in_process_workers_enabled", False))
        and db_backend == "postgres"
        and not bool(runtime_config.get("database_recovery_mode", False))
    )
    withdrawals_ready = (
        bool(runtime_config.get("withdrawals_enabled", False))
        and custody_mode in {"kms", "hsm", "mpc"}
        and treasury.get("ready") is True
        and not blockers
    )
    return {
        "ready": not blockers,
        "trading_ready": trading_ready and not blockers,
        "withdrawals_ready": withdrawals_ready,
        "deployment_target": target,
        "database_backend": db_backend,
        "custody_mode": custody_mode,
        "blockers": blockers,
        "warnings": warnings,
    }


def _safe_scalar(statement: str) -> Any:
    try:
        return db.session.execute(text(statement)).scalar()
    except Exception:  # noqa: BLE001
        db.session.rollback()
        return None


def _safe_setting_json(key: str, default: Any) -> Any:
    try:
        return Setting.get_json(key, default)
    except Exception:  # noqa: BLE001
        db.session.rollback()
        return default


def _worker_status(now: datetime) -> dict[str, Any]:
    try:
        leases = WorkerLease.query.order_by(WorkerLease.lease_name.asc()).all()
        rows: list[dict[str, Any]] = []
        max_lag = 0.0
        for lease in leases:
            lag = _elapsed_seconds(now, lease.heartbeat_at) if lease.heartbeat_at else None
            if lag is not None:
                max_lag = max(max_lag, lag)
            rows.append(
                {
                    "lease_name": lease.lease_name,
                    "status": lease.status,
                    "heartbeat_lag_seconds": lag,
                    "expires_at": lease.expires_at.isoformat() if lease.expires_at else None,
                }
            )
        return {"available": True, "leases": rows, "max_lease_lag_seconds": max_lag}
    except Exception:  # noqa: BLE001
        db.session.rollback()
        return {"available": False, "leases": [], "max_lease_lag_seconds": None}


def _elapsed_seconds(now: datetime, then: datetime) -> float:
    if now.tzinfo is None and then.tzinfo is not None:
        now = now.replace(tzinfo=UTC)
    elif now.tzinfo is not None and then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return max(0.0, (now - then).total_seconds())


def _trading_status(now: datetime, app: Flask) -> dict[str, Any]:
    stale_after = max(60, int(app.config.get("STRATEGY_HEARTBEAT_PERSIST_SECONDS", 30) or 30) * 4)
    threshold = now - timedelta(seconds=stale_after)
    try:
        return {
            "available": True,
            "running_strategy_count": StrategyRun.query.filter_by(status="running").count(),
            "queued_strategy_count": StrategyRun.query.filter_by(status="queued").count(),
            "stale_strategy_count": StrategyRun.query.filter(
                StrategyRun.status == "running",
                StrategyRun.updated_at < threshold,
            ).count(),
            "stale_after_seconds": stale_after,
            "active_vault_cycle_count": VaultCycle.query.filter_by(status="active").count(),
        }
    except Exception:  # noqa: BLE001
        db.session.rollback()
        return {
            "available": False,
            "running_strategy_count": None,
            "queued_strategy_count": None,
            "stale_strategy_count": None,
            "stale_after_seconds": stale_after,
            "active_vault_cycle_count": None,
        }


def _wallet_status() -> dict[str, Any]:
    try:
        return {
            "available": True,
            "withdrawal_failures_24h": _wallet_audit_count("withdrawal_failed"),
            "withdrawal_safety_blocks_24h": _wallet_audit_count("withdrawal_blocked_by_safety_gate"),
            "wallet_sync_failures_24h": _wallet_audit_count("wallet_sync_failed"),
            "reconciliation_mismatches_24h": _wallet_audit_count("withdrawal_reconciled_failed"),
        }
    except Exception:  # noqa: BLE001
        db.session.rollback()
        return {
            "available": False,
            "withdrawal_failures_24h": None,
            "withdrawal_safety_blocks_24h": None,
            "wallet_sync_failures_24h": None,
            "reconciliation_mismatches_24h": None,
        }


def _model_status() -> dict[str, Any]:
    try:
        promoted = MLModelRegistry.query.filter_by(status="promoted").order_by(MLModelRegistry.promoted_at.desc()).limit(25).all()
        if not promoted:
            promoted_models = MLOfflineModel.query.filter_by(status="promoted").order_by(MLOfflineModel.promoted_at.desc()).limit(25).all()
            return {
                "available": True,
                "promoted_count": len(promoted_models),
                "drift_watch_count": sum(1 for record in promoted_models if float(record.drift or 0.0) > 0),
                "registry": [],
            }
        return {
            "available": True,
            "promoted_count": len(promoted),
            "drift_watch_count": sum(1 for record in promoted if record.drift_status in {"watch", "blocked"}),
            "registry": [
                {
                    "model_family": record.model_family,
                    "model_version": record.model_version,
                    "provider": record.provider,
                    "horizon": record.horizon,
                    "mode": record.mode,
                    "drift_status": record.drift_status,
                    "promoted_at": record.promoted_at.isoformat() if record.promoted_at else None,
                }
                for record in promoted
            ],
        }
    except Exception:  # noqa: BLE001
        db.session.rollback()
        return {"available": False, "promoted_count": None, "drift_watch_count": None, "registry": []}


def _treasury_status(app: Flask) -> dict[str, Any]:
    service = app.extensions.get("services", {}).get("platform_treasury")
    if service is None:
        return {"available": False}
    try:
        status = service.status(include_events=False)
        return status if isinstance(status, dict) else {"available": True}
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Treasury status failed")
        return {"available": False, "error": exc.__class__.__name__}


def _chart_refresh_lag(app: Flask, now: datetime) -> float | None:
    service = app.extensions.get("services", {}).get("chart_stream")
    last = getattr(service, "last_refresh_at", None)
    if last is None:
        return None
    try:
        return max(0.0, (now - last).total_seconds())
    except Exception:  # noqa: BLE001
        return None


def _order_rejection_count() -> int | None:
    try:
        return Order.query.filter(Order.status.in_(["rejected", "failed"])).count()
    except Exception:  # noqa: BLE001
        db.session.rollback()
        return None


def _audit_count(category: str, action_fragment: str) -> int | None:
    try:
        since = datetime.utcnow() - timedelta(days=1)
        return AuditLog.query.filter(
            AuditLog.created_at >= since,
            AuditLog.category == category,
            AuditLog.action.contains(action_fragment),
        ).count()
    except Exception:  # noqa: BLE001
        db.session.rollback()
        return None


def _wallet_audit_count(action: str) -> int | None:
    try:
        since = datetime.utcnow() - timedelta(days=1)
        return WalletAuditLog.query.filter(WalletAuditLog.created_at >= since, WalletAuditLog.action == action).count()
    except Exception:  # noqa: BLE001
        db.session.rollback()
        return None


def _prefers_json_response() -> bool:
    if request.path.startswith(("/api/", "/admin/api/")) or request.path in {"/healthz", "/readyz", "/ops/status"}:
        return True
    return request.accept_mimetypes["application/json"] >= request.accept_mimetypes["text/html"]


def _block_when_database_startup_failed(app: Flask) -> Response | None:
    if not app.config.get("DATABASE_STARTUP_ERROR"):
        return None
    path = request.path.lower()
    if (
        request.endpoint == "static"
        or path in {"/healthz", "/readyz", "/manifest.json", "/manifest.webmanifest", "/sw.js"}
        or path.startswith(("/static/", "/icons/"))
    ):
        return None

    status = 503
    if _prefers_json_response():
        response = jsonify({"ok": False, "error": "database_unavailable", "status": status})
        response.status_code = status
        return response
    return Response(
        "<!doctype html><title>Service Unavailable</title><main><h1>Service Unavailable</h1>"
        "<p>The database is temporarily unavailable. Trading actions are blocked until readiness recovers.</p></main>",
        status=status,
        content_type="text/html; charset=utf-8",
    )


def _rate_limit_request(app: Flask) -> Response | None:
    if app.config.get("TESTING") and not bool(app.config.get("RATELIMIT_FORCE_ENABLED", False)):
        return None
    if not bool(app.config.get("RATELIMIT_ENABLED", True)):
        return None
    if request.endpoint == "static" or request.path in {"/healthz", "/readyz"}:
        return None

    bucket_name, limit = _rate_limit_bucket(app)
    if not bucket_name or limit <= 0:
        return None

    window = max(1, int(app.config.get("RATELIMIT_WINDOW_SECONDS", 60) or 60))
    key = (_client_rate_key(), bucket_name)
    now = time.monotonic()
    store: defaultdict[tuple[str, str], deque[float]] = app.extensions.setdefault("rate_limit_store", defaultdict(deque))
    events = store[key]
    while events and now - events[0] >= window:
        events.popleft()
    if len(events) >= limit:
        retry_after = max(1, int(window - (now - events[0]))) if events else window
        payload = {"ok": False, "error": "rate_limited", "retry_after": retry_after}
        response = jsonify(payload) if _prefers_json_response() else Response("Too many requests.", status=429)
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response
    events.append(now)
    return None


def _rate_limit_bucket(app: Flask) -> tuple[str, int]:
    path = request.path.rstrip("/") or "/"
    unsafe = request.method in {"POST", "PUT", "PATCH", "DELETE"}
    if request.method == "POST" and path in {"/login", "/admin/api/sign-in"}:
        return "login", int(app.config.get("RATELIMIT_LOGIN_PER_WINDOW", 12) or 12)
    if path == "/setup-2fa":
        return "auth_setup", int(app.config.get("RATELIMIT_AUTH_SETUP_PER_WINDOW", 20) or 20)
    if path.startswith(("/api/", "/admin/api/")) or path.endswith("/stream"):
        return "api", int(app.config.get("RATELIMIT_API_PER_WINDOW", 180) or 180)
    if unsafe and (path.startswith("/admin") or path.startswith("/settings") or path.startswith("/wallet") or path.startswith("/vault")):
        return "unsafe", int(app.config.get("RATELIMIT_UNSAFE_PER_WINDOW", 60) or 60)
    return "", 0


def _client_rate_key() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "").split(",", 1)[0].strip()
    return forwarded or request.remote_addr or "unknown"


_LIVE_API_CORS_PATHS = {
    "/vault/readiness",
    "/api/vault/readiness",
    "/vault/preview-route",
    "/api/vault/routing-preview",
    "/vault/start-cycle",
    "/vault/cycles",
}
_LIVE_API_CORS_PATH_PREFIXES = (
    "/api/vault/cycles/",
    "/vault/start-status/",
    "/consumer/start-status/",
)


def _is_live_api_cors_request() -> bool:
    path = request.path.rstrip("/") or "/"
    return path in _LIVE_API_CORS_PATHS or any(request.path.startswith(prefix) for prefix in _LIVE_API_CORS_PATH_PREFIXES)


def _allowed_cors_origin(app: Flask) -> str:
    if not str(app.config.get("PUBLIC_LIVE_API_ORIGIN") or "").strip():
        return ""
    if not _is_live_api_cors_request():
        return ""
    origin = str(request.headers.get("Origin") or "").strip().rstrip("/")
    if not origin:
        return ""
    allowed = {
        str(item or "").strip().rstrip("/")
        for item in list(app.config.get("LIVE_API_CORS_ALLOWED_ORIGINS") or [])
        if str(item or "").strip() and str(item or "").strip() != "*"
    }
    public_app_origin = str(app.config.get("PUBLIC_APP_ORIGIN") or "").strip().rstrip("/")
    if public_app_origin and public_app_origin != "*":
        allowed.add(public_app_origin)
    return origin if origin in allowed else ""


def _set_response_headers(app: Flask, response: Response) -> Response:
    cors_origin = _allowed_cors_origin(app)
    if cors_origin:
        response.headers["Access-Control-Allow-Origin"] = cors_origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = request.headers.get(
            "Access-Control-Request-Headers",
            "Accept, Content-Type, X-CSRF-Token, X-Requested-With, Idempotency-Key, "
            "X-AlgVault-User-Id, X-AlgVault-Internal-Timestamp, X-AlgVault-Internal-Body-SHA256, "
            "X-AlgVault-Internal-Signature",
        )
        response.headers["Access-Control-Max-Age"] = "600"
        vary = response.headers.get("Vary", "")
        if "Origin" not in {item.strip() for item in vary.split(",") if item.strip()}:
            response.headers["Vary"] = f"{vary}, Origin".strip(", ")

    if bool(app.config.get("SECURE_HEADERS_ENABLED", True)):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if bool(app.config.get("SECURE_HEADERS_HSTS_ENABLED", False)):
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        if bool(app.config.get("SECURE_HEADERS_CSP_ENABLED", False)):
            live_api_origin = str(app.config.get("PUBLIC_LIVE_API_ORIGIN") or "").strip()
            connect_src = "'self'" + (f" {live_api_origin}" if live_api_origin else "")
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                f"img-src 'self' data:; connect-src {connect_src}; frame-ancestors 'none'",
            )

    path = request.path.lower()
    if path.startswith("/admin/api/") or (path.startswith("/api/vault-cycles/") and path.endswith("/process-profit-share")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    cacheable_static_response = request.endpoint == "static" or path in {"/manifest.json", "/sw.js"} or path.startswith("/icons/")
    if cacheable_static_response:
        if path.endswith("/sw.js") or path.endswith("static/js/sw.js"):
            max_age = max(0, int(app.config.get("SERVICE_WORKER_CACHE_SECONDS", 0) or 0))
            response.headers["Cache-Control"] = f"public, max-age={max_age}, must-revalidate"
            response.headers["Service-Worker-Allowed"] = "/"
        elif path.endswith("manifest.webmanifest") or path == "/manifest.json":
            max_age = max(0, int(app.config.get("SERVICE_WORKER_CACHE_SECONDS", 0) or 0))
            response.headers["Cache-Control"] = f"public, max-age={max_age}, must-revalidate"
        elif any(path.endswith(ext) for ext in (".css", ".js", ".png", ".jpg", ".jpeg", ".webp", ".svg", ".ico", ".woff2")):
            max_age = max(0, int(app.config.get("STATIC_CACHE_SECONDS", 31_536_000) or 31_536_000))
            response.headers["Cache-Control"] = f"public, max-age={max_age}, immutable"
    else:
        response.headers.setdefault("Cache-Control", "private, no-store, no-cache, must-revalidate, max-age=0")
        response.headers.setdefault("Pragma", "no-cache")
        response.headers.setdefault("Expires", "0")
    return response


def _database_backend(app: Flask) -> str:
    uri = str(app.config.get("SQLALCHEMY_DATABASE_URI", ""))
    return _database_backend_from_uri(uri)


def _database_backend_from_uri(uri: str) -> str:
    if uri.startswith(("postgresql://", "postgresql+", "postgres://")):
        return "postgres"
    if uri.startswith("sqlite"):
        return "sqlite"
    return "other"


def _configure_engine_options(app: Flask) -> None:
    uri = str(app.config.get("SQLALCHEMY_DATABASE_URI", ""))
    if uri.startswith(("postgresql://", "postgresql+", "postgres://")):
        options = dict(app.config.get("SQLALCHEMY_ENGINE_OPTIONS") or {})
        options.setdefault("pool_pre_ping", True)
        options.setdefault("pool_recycle", 1800)
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = options
        return
    if not uri.startswith("sqlite"):
        return
    options = dict(app.config.get("SQLALCHEMY_ENGINE_OPTIONS") or {})
    connect_args = dict(options.get("connect_args") or {})
    connect_args.setdefault("timeout", max(float(app.config.get("SQLITE_BUSY_TIMEOUT_MS", 10_000)) / 1000, 1.0))
    options["connect_args"] = connect_args
    if uri not in {"sqlite://", "sqlite:///:memory:"}:
        options.setdefault("poolclass", NullPool)
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = options


def _configure_sqlite_pragmas(app: Flask) -> None:
    uri = str(app.config.get("SQLALCHEMY_DATABASE_URI", ""))
    if not uri.startswith("sqlite"):
        return
    timeout_ms = max(int(app.config.get("SQLITE_BUSY_TIMEOUT_MS", 10_000)), 1000)
    db.session.execute(text(f"PRAGMA busy_timeout={timeout_ms}"))
    if bool(app.config.get("SQLITE_ENABLE_WAL", True)) and uri not in {"sqlite://", "sqlite:///:memory:"}:
        db.session.execute(text("PRAGMA journal_mode=WAL"))
        db.session.execute(text("PRAGMA synchronous=NORMAL"))


def _run_database_startup(app: Flask) -> None:
    _configure_sqlite_pragmas(app)
    if _skip_schema_bootstrap_requested():
        return
    if _schema_bootstrap_allowed(app):
        _create_all_tolerant()
        _ensure_schema()
    else:
        _verify_migrated_schema()
    db.session.commit()
    _seed_default_settings(app)
    admin_user = _seed_admin_user(app)
    db.session.commit()
    _migrate_legacy_wallet_rows(admin_user)
    _reset_stale_strategy_runs()


def _record_database_startup_error(app: Flask, exc: Exception) -> None:
    db.session.rollback()
    app.config["DATABASE_STARTUP_ERROR"] = str(exc)
    app.config["DATABASE_STARTUP_ERROR_CLASS"] = exc.__class__.__name__
    app.logger.exception("Database startup checks failed; serving readiness 503 until the runtime recovers.")


def _should_defer_database_startup_error(app: Flask, exc: Exception) -> bool:
    if not bool(app.config.get("DEFER_DATABASE_STARTUP_ERRORS", False)):
        return False
    if _is_missing_schema_error(exc):
        return False
    return isinstance(exc, DatabaseStartupError) or any(isinstance(item, OperationalError) for item in _exception_chain(exc))


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _is_missing_schema_error(exc: BaseException) -> bool:
    for item in _exception_chain(exc):
        message = str(item).lower()
        if "alembic_version" in message and (
            "no such table" in message or "does not exist" in message or "undefinedtable" in message or "undefined table" in message
        ):
            return True
    return False


def _skip_schema_bootstrap_requested() -> bool:
    return os.getenv("SKIP_SCHEMA_BOOTSTRAP", "").strip().lower() in {"1", "true", "yes", "on"}


def _schema_bootstrap_allowed(app: Flask) -> bool:
    if bool(app.config.get("TESTING", False)):
        return bool(app.config.get("SCHEMA_BOOTSTRAP_ENABLED", True))
    deployment_target = str(app.config.get("DEPLOYMENT_TARGET", "local") or "local").strip().lower()
    production_target = deployment_target in {"vps", "production", "prod", "postgres", "vercel"}
    if production_target:
        return bool(app.config.get("ALLOW_PRODUCTION_SCHEMA_BOOTSTRAP", False)) and bool(app.config.get("SCHEMA_BOOTSTRAP_ENABLED", False))
    return bool(app.config.get("SCHEMA_BOOTSTRAP_ENABLED", True))


def _verify_migrated_schema() -> None:
    try:
        version = db.session.execute(text("SELECT version_num FROM alembic_version LIMIT 1")).scalar()
    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        if not _is_missing_schema_error(exc):
            raise DatabaseStartupError("Database schema verification failed because the database is unavailable.") from exc
        raise RuntimeError("Database schema is not migrated; run `flask db upgrade` before starting production.") from exc
    if not version:
        raise RuntimeError("Database schema is not migrated; alembic_version is empty.")


def _create_all_tolerant() -> None:
    """Create schema while tolerating concurrent SQLite table creation."""

    for table in db.metadata.sorted_tables:
        try:
            table.create(bind=db.engine, checkfirst=True)
        except OperationalError as exc:
            if "already exists" not in str(exc).lower():
                raise
            db.session.rollback()


def _seed_default_settings(app: Flask) -> None:
    Setting.set_json("current_mode", "live")
    Setting.ensure_json("panic_lock", False)
    Setting.ensure_json("explicit_live_confirmed", bool(app.config["EXPLICIT_LIVE_CONFIRMED"]))
    Setting.ensure_json("secondary_confirmation", bool(app.config["SECONDARY_CONFIRMATION"]))
    Setting.ensure_json("live_trading_blocked", False)
    Setting.ensure_json("use_real_addresses", bool(app.config["USE_REAL_ADDRESSES"]))
    Setting.ensure_json("platform_treasury_paused", False)
    db.session.commit()


def _reset_stale_strategy_runs() -> None:
    for run in StrategyRun.query.filter(StrategyRun.status == "running").all():
        run.status = "stopped"
        run.manual_enabled = False
    db.session.commit()


def _seed_admin_user(app: Flask) -> User | None:
    admin = User.query.filter_by(role="admin").first()
    if admin is not None:
        return admin
    username = app.config.get("ADMIN_USERNAME", "admin")
    password = app.config.get("ADMIN_PASSWORD", "")
    if not username or not password:
        return None
    existing = User.query.filter_by(username=username).one_or_none()
    if existing is None:
        existing = User(username=username, password_hash=password_hash(password), role="admin")
        db.session.add(existing)
    else:
        existing.role = "admin"
        existing.password_hash = password_hash(password)
    db.session.commit()
    return existing


def _migrate_legacy_wallet_rows(admin: User | None) -> None:
    if admin is None:
        return

    for balance in WalletBalance.query.filter_by(user_id=None).all():
        existing = WalletBalance.query.filter_by(user_id=admin.id, asset=balance.asset).one_or_none()
        if existing and existing.id != balance.id:
            existing.available_balance += float(balance.available_balance or 0.0)
            existing.locked_balance += float(balance.locked_balance or 0.0)
            existing.estimated_usd_value += float(balance.estimated_usd_value or 0.0)
            db.session.delete(balance)
        else:
            balance.user_id = admin.id
    VaultCycle.query.filter_by(user_id=None).update({"user_id": admin.id})
    WalletTransaction.query.filter_by(user_id=None).update({"user_id": admin.id})
    db.session.commit()


def _ensure_schema() -> None:
    uri = str(db.engine.url)
    if not uri.startswith("sqlite"):
        return
    additions = {
        "user": {
            "role": "role VARCHAR(32) NOT NULL DEFAULT 'user'",
            "referral_invite_code_id": "referral_invite_code_id INTEGER",
            "totp_secret_encrypted": "totp_secret_encrypted TEXT",
            "two_factor_enabled_at": "two_factor_enabled_at DATETIME",
            "created_at": "created_at DATETIME",
            "updated_at": "updated_at DATETIME",
        },
        "wallet_balance": {
            "user_id": "user_id INTEGER",
            "active_deposit_address_id": "active_deposit_address_id INTEGER",
        },
        "vault_cycle": {
            "public_id": "public_id VARCHAR(40)",
            "user_id": "user_id INTEGER",
            "trading_connection_id": "trading_connection_id INTEGER",
            "execution_substatus": "execution_substatus VARCHAR(32) NOT NULL DEFAULT 'initializing'",
            "execution_mode": "execution_mode VARCHAR(32) NOT NULL DEFAULT 'live'",
            "live_validation_status": "live_validation_status VARCHAR(32) NOT NULL DEFAULT 'not_required'",
            "validation_started_at": "validation_started_at DATETIME",
            "validation_completed_at": "validation_completed_at DATETIME",
            "validation_failure_reason": "validation_failure_reason TEXT",
            "cycle_summary_json": "cycle_summary_json TEXT NOT NULL DEFAULT '{}'",
            "lock_duration_seconds": "lock_duration_seconds INTEGER NOT NULL DEFAULT 3600",
        },
        "wallet_transaction": {
            "user_id": "user_id INTEGER",
            "network": "network VARCHAR(64)",
            "withdraw_address": "withdraw_address TEXT",
        },
        "referral_invite_code": {
            "public_id": "public_id VARCHAR(40)",
            "profit_share_percent": "profit_share_percent FLOAT NOT NULL DEFAULT 0",
            "profit_share_wallet": "profit_share_wallet VARCHAR(120) NOT NULL DEFAULT 'sufyanh'",
            "profit_share_starts_at": "profit_share_starts_at DATETIME",
            "profit_share_ends_at": "profit_share_ends_at DATETIME",
            "profit_share_active": "profit_share_active BOOLEAN NOT NULL DEFAULT 1",
            "applies_to_vault_types_json": "applies_to_vault_types_json TEXT NOT NULL DEFAULT '[]'",
            "expires_at": "expires_at DATETIME",
            "assigned_role": "assigned_role VARCHAR(32) NOT NULL DEFAULT 'user'",
            "deleted_at": "deleted_at DATETIME",
        },
        "wallet_ledger_event": {
            "user_id": "user_id INTEGER NOT NULL",
            "deposit_address_id": "deposit_address_id INTEGER",
            "wallet_address_id": "wallet_address_id INTEGER",
            "asset": "asset VARCHAR(32) NOT NULL",
            "network": "network VARCHAR(64) NOT NULL",
            "address": "address TEXT NOT NULL",
            "event_type": "event_type VARCHAR(32) NOT NULL DEFAULT 'deposit'",
            "provider_reference": "provider_reference VARCHAR(180) NOT NULL",
            "idempotency_key": "idempotency_key VARCHAR(220) NOT NULL",
            "amount": "amount FLOAT NOT NULL DEFAULT 0",
            "confirmations": "confirmations INTEGER NOT NULL DEFAULT 0",
            "status": "status VARCHAR(32) NOT NULL DEFAULT 'pending'",
            "metadata_json": "metadata_json TEXT NOT NULL DEFAULT '{}'",
            "created_at": "created_at DATETIME",
            "updated_at": "updated_at DATETIME",
        },
        "trading_connection": {
            "verification_status": "verification_status VARCHAR(32) NOT NULL DEFAULT 'needs_verification'",
            "last_verified_at": "last_verified_at DATETIME",
            "last_verification_error": "last_verification_error TEXT",
            "provider_metadata_json": "provider_metadata_json TEXT NOT NULL DEFAULT '{}'",
        },
        "strategy_ranking": {
            "provider": "provider VARCHAR(64) NOT NULL DEFAULT 'global'",
            "profile": "profile VARCHAR(64) NOT NULL DEFAULT 'short_term'",
            "experimental": "experimental BOOLEAN NOT NULL DEFAULT 0",
            "risk_label": "risk_label VARCHAR(80) NOT NULL DEFAULT ''",
            "net_return_after_costs": "net_return_after_costs FLOAT NOT NULL DEFAULT 0",
            "recent_1h_return": "recent_1h_return FLOAT NOT NULL DEFAULT 0",
            "estimated_fees": "estimated_fees FLOAT NOT NULL DEFAULT 0",
            "edge_score": "edge_score FLOAT NOT NULL DEFAULT 0",
            "expectancy": "expectancy FLOAT NOT NULL DEFAULT 0",
            "avg_win": "avg_win FLOAT NOT NULL DEFAULT 0",
            "avg_loss": "avg_loss FLOAT NOT NULL DEFAULT 0",
            "win_loss_ratio": "win_loss_ratio FLOAT NOT NULL DEFAULT 0",
            "cost_drag_bps": "cost_drag_bps FLOAT NOT NULL DEFAULT 0",
            "convex_edge_score": "convex_edge_score FLOAT NOT NULL DEFAULT 0",
            "mfe_mae_ratio": "mfe_mae_ratio FLOAT NOT NULL DEFAULT 0",
            "capacity_multiple": "capacity_multiple FLOAT NOT NULL DEFAULT 0",
            "cost_adjusted_recent_1h_return": "cost_adjusted_recent_1h_return FLOAT NOT NULL DEFAULT 0",
            "decay_penalty": "decay_penalty FLOAT NOT NULL DEFAULT 0",
            "max_adverse_excursion": "max_adverse_excursion FLOAT NOT NULL DEFAULT 0",
            "max_favorable_excursion": "max_favorable_excursion FLOAT NOT NULL DEFAULT 0",
            "no_trade_reason": "no_trade_reason TEXT",
            "allocation_amount_usd": "allocation_amount_usd FLOAT NOT NULL DEFAULT 0",
            "lock_duration_hours": "lock_duration_hours INTEGER NOT NULL DEFAULT 0",
            "leverage": "leverage FLOAT NOT NULL DEFAULT 1",
            "liquidation_buffer_pct": "liquidation_buffer_pct FLOAT NOT NULL DEFAULT 1",
            "capacity_usd": "capacity_usd FLOAT NOT NULL DEFAULT 0",
            "universe_source": "universe_source VARCHAR(64) NOT NULL DEFAULT 'configured'",
            "vault_leg_count": "vault_leg_count INTEGER NOT NULL DEFAULT 1",
            "execution_style": "execution_style VARCHAR(32) NOT NULL DEFAULT 'market'",
            "funding_cost_estimate": "funding_cost_estimate FLOAT NOT NULL DEFAULT 0",
            "turnover_after_fees": "turnover_after_fees FLOAT NOT NULL DEFAULT 0",
            "window_stability": "window_stability FLOAT NOT NULL DEFAULT 0",
            "accepted_window_ratio": "accepted_window_ratio FLOAT NOT NULL DEFAULT 0",
            "win_rate": "win_rate FLOAT NOT NULL DEFAULT 0",
            "trade_count": "trade_count INTEGER NOT NULL DEFAULT 0",
            "ml_score": "ml_score FLOAT NOT NULL DEFAULT 0",
            "ml_adjusted_score": "ml_adjusted_score FLOAT NOT NULL DEFAULT 0",
            "ml_warmup": "ml_warmup BOOLEAN NOT NULL DEFAULT 1",
            "ml_explanation_json": "ml_explanation_json TEXT NOT NULL DEFAULT '{}'",
        },
        "strategy_run": {
            "lock_duration_seconds": "lock_duration_seconds INTEGER NOT NULL DEFAULT 3600",
            "user_id": "user_id INTEGER",
            "trading_connection_id": "trading_connection_id INTEGER",
        },
        "order": {
            "user_id": "user_id INTEGER",
            "trading_connection_id": "trading_connection_id INTEGER",
            "vault_cycle_id": "vault_cycle_id INTEGER",
            "vault_leg_id": "vault_leg_id INTEGER",
        },
        "fill": {
            "source_order_id": "source_order_id INTEGER",
            "exchange_order_id": "exchange_order_id VARCHAR(120)",
            "exchange_fill_id": "exchange_fill_id VARCHAR(180)",
            "funding_fee": "funding_fee FLOAT NOT NULL DEFAULT 0",
            "fee_known": "fee_known BOOLEAN NOT NULL DEFAULT 1",
            "realized_pnl_known": "realized_pnl_known BOOLEAN NOT NULL DEFAULT 1",
            "metadata_json": "metadata_json TEXT NOT NULL DEFAULT '{}'",
        },
        "position_snapshot": {
            "user_id": "user_id INTEGER",
            "trading_connection_id": "trading_connection_id INTEGER",
        },
        "audit_log": {
            "user_id": "user_id INTEGER",
            "trading_connection_id": "trading_connection_id INTEGER",
        },
        "risk_event": {
            "user_id": "user_id INTEGER",
            "trading_connection_id": "trading_connection_id INTEGER",
        },
        "wallet_withdrawal": {
            "trading_connection_id": "trading_connection_id INTEGER",
            "treasury_safety_status": "treasury_safety_status VARCHAR(32) NOT NULL DEFAULT 'unchecked'",
            "treasury_safety_reason": "treasury_safety_reason TEXT",
            "treasury_estimated_gas_eth": "treasury_estimated_gas_eth FLOAT NOT NULL DEFAULT 0",
            "treasury_safety_checked_at": "treasury_safety_checked_at DATETIME",
        },
        "platform_treasury_event": {
            "platform_treasury_job_id": "platform_treasury_job_id INTEGER",
            "wallet_ledger_event_id": "wallet_ledger_event_id INTEGER",
            "vault_cycle_id": "vault_cycle_id INTEGER",
            "referral_invite_code_id": "referral_invite_code_id INTEGER",
        },
        "treasury_reserve_state": {
            "target_reserve_eth": "target_reserve_eth FLOAT NOT NULL DEFAULT 0",
            "deficit_eth": "deficit_eth FLOAT NOT NULL DEFAULT 0",
        },
        "vault_allocation_leg": {
            "strategy_run_id": "strategy_run_id INTEGER",
            "optimizer_ranking_id": "optimizer_ranking_id INTEGER",
            "provider": "provider VARCHAR(64) NOT NULL DEFAULT 'global'",
            "trading_connection_id": "trading_connection_id INTEGER",
            "realized_pnl_usd": "realized_pnl_usd FLOAT NOT NULL DEFAULT 0",
            "unrealized_pnl_usd": "unrealized_pnl_usd FLOAT NOT NULL DEFAULT 0",
            "metadata_json": "metadata_json TEXT NOT NULL DEFAULT '{}'",
        },
        "ml_model_state": {
            "provider": "provider VARCHAR(64) NOT NULL DEFAULT 'global'",
        },
        "ml_offline_model": {
            "provider": "provider VARCHAR(64) NOT NULL DEFAULT 'global'",
        },
        "ml_training_event": {
            "provider": "provider VARCHAR(64) NOT NULL DEFAULT 'global'",
        },
        "leveraged_market": {
            "trading_connection_id": "trading_connection_id INTEGER",
        },
    }
    for table, columns in additions.items():
        existing = _table_columns(table)
        if not existing:
            continue
        quoted_table = _quote_sqlite_identifier(table)
        for name, ddl in columns.items():
            if name not in existing:
                db.session.execute(text(f"ALTER TABLE {quoted_table} ADD COLUMN {ddl}"))
    _backfill_public_ids()
    _backfill_order_vault_links()
    _mark_legacy_zero_live_fills_unknown()
    db.session.commit()
    _ensure_indexes()
    db.session.commit()


def _table_columns(table: str) -> set[str]:
    try:
        rows = db.session.execute(text(f"PRAGMA table_info({_quote_sqlite_identifier(table)})")).mappings().all()
    except Exception:  # noqa: BLE001
        return set()
    return {row["name"] for row in rows}


def _quote_sqlite_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _ensure_indexes() -> None:
    statements = (
        "CREATE INDEX IF NOT EXISTS ix_strategy_run_user_status_created ON strategy_run (user_id, status, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_strategy_run_status_updated ON strategy_run (status, updated_at)",
        "CREATE INDEX IF NOT EXISTS ix_strategy_run_connection_status_created ON strategy_run (trading_connection_id, status, created_at)",
        'CREATE INDEX IF NOT EXISTS ix_order_user_mode_created ON "order" (user_id, mode, created_at)',
        'CREATE INDEX IF NOT EXISTS ix_order_user_mode_status_created ON "order" (user_id, mode, status, created_at)',
        'CREATE INDEX IF NOT EXISTS ix_order_cycle_mode_created ON "order" (vault_cycle_id, mode, created_at)',
        'CREATE INDEX IF NOT EXISTS ix_order_leg_status_created ON "order" (vault_leg_id, status, created_at)',
        'CREATE INDEX IF NOT EXISTS ix_order_connection_status_created ON "order" (trading_connection_id, status, created_at)',
        "CREATE INDEX IF NOT EXISTS ix_vault_cycle_user_status_started ON vault_cycle (user_id, status, started_at)",
        "CREATE INDEX IF NOT EXISTS ix_vault_cycle_user_unlocks ON vault_cycle (user_id, unlocks_at)",
        "CREATE INDEX IF NOT EXISTS ix_vault_cycle_connection_status_started ON vault_cycle (trading_connection_id, status, started_at)",
        "CREATE INDEX IF NOT EXISTS ix_vault_leg_cycle_status ON vault_allocation_leg (vault_cycle_id, status)",
        "CREATE INDEX IF NOT EXISTS ix_vault_leg_run_status ON vault_allocation_leg (strategy_run_id, status)",
        "CREATE INDEX IF NOT EXISTS ix_vault_leg_connection_symbol_status ON vault_allocation_leg (trading_connection_id, symbol, status)",
        "CREATE INDEX IF NOT EXISTS ix_vault_cycle_allocation_cycle_status ON vault_cycle_allocation (vault_cycle_id, status)",
        "CREATE INDEX IF NOT EXISTS ix_vault_cycle_allocation_user_provider_status ON vault_cycle_allocation (user_id, provider, status)",
        "CREATE INDEX IF NOT EXISTS ix_vault_cycle_transfer_cycle_status ON vault_cycle_transfer (vault_cycle_id, status)",
        "CREATE INDEX IF NOT EXISTS ix_vault_cycle_transfer_allocation_direction ON vault_cycle_transfer (allocation_id, direction)",
        "CREATE INDEX IF NOT EXISTS ix_vault_cycle_transfer_user_direction_created ON vault_cycle_transfer (user_id, direction, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_vault_cycle_trade_cycle_created ON vault_cycle_trade (vault_cycle_id, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_vault_cycle_risk_cycle_created ON vault_cycle_risk_event (vault_cycle_id, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_vault_cycle_risk_user_severity_created ON vault_cycle_risk_event (user_id, severity, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_strategy_ranking_provider_profile_rejected_score_created ON strategy_ranking (provider, profile, rejected, score, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_strategy_ranking_symbol_profile_rejected_score ON strategy_ranking (symbol, profile, rejected, score)",
        "CREATE INDEX IF NOT EXISTS ix_strategy_ranking_run_rejected_score ON strategy_ranking (optimizer_run_id, rejected, score)",
        "CREATE INDEX IF NOT EXISTS ix_audit_log_created_id ON audit_log (created_at, id)",
        "CREATE INDEX IF NOT EXISTS ix_audit_log_category_action_created ON audit_log (category, action, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_audit_log_user_category_created ON audit_log (user_id, category, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_audit_log_connection_category_created ON audit_log (trading_connection_id, category, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_wallet_transaction_user_type_created ON wallet_transaction (user_id, transaction_type, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_wallet_transaction_cycle_type_status ON wallet_transaction (vault_cycle_id, transaction_type, status)",
        "CREATE INDEX IF NOT EXISTS ix_platform_treasury_wallet_network_active ON platform_treasury_wallet (network, is_active)",
        "CREATE INDEX IF NOT EXISTS ix_referral_invite_code_active_created ON referral_invite_code (is_active, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_referral_invite_code_public_id ON referral_invite_code (public_id)",
        "CREATE INDEX IF NOT EXISTS ix_referral_invite_code_profit_wallet ON referral_invite_code (profit_share_wallet)",
        "CREATE INDEX IF NOT EXISTS ix_vault_cycle_public_id ON vault_cycle (public_id)",
        "CREATE INDEX IF NOT EXISTS ix_user_referral_invite_code ON user (referral_invite_code_id)",
        "CREATE INDEX IF NOT EXISTS ix_invite_code_usage_code_status_used ON invite_code_usage (invite_code_id, status, used_at)",
        "CREATE INDEX IF NOT EXISTS ix_profit_share_payout_invite_status_created ON profit_share_payout (invite_code_id, status, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_admin_audit_entity_created ON admin_audit_log (entity_type, entity_public_id, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_platform_treasury_reserve_job_type_status_created ON platform_treasury_reserve_job (job_type, status, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_platform_treasury_reserve_job_user_type_created ON platform_treasury_reserve_job (user_id, job_type, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_platform_treasury_event_withdrawal_type_status ON platform_treasury_event (wallet_withdrawal_id, event_type, status)",
        "CREATE INDEX IF NOT EXISTS ix_platform_treasury_event_type_status_created ON platform_treasury_event (event_type, status, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_wallet_withdrawal_treasury_status_network ON wallet_withdrawal (treasury_safety_status, network, status)",
        "CREATE INDEX IF NOT EXISTS ix_treasury_alert_network_severity_created ON treasury_alert (network, severity, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_treasury_gas_usage_network_created ON treasury_gas_usage (network, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_treasury_forecast_network_window_created ON treasury_reserve_forecast (network, forecast_window, created_at)",
        "CREATE INDEX IF NOT EXISTS ix_ml_market_history_provider_symbol_tf_window ON ml_market_history (provider, symbol, timeframe, window_end)",
        "CREATE INDEX IF NOT EXISTS ix_ml_market_history_provider_status_fetched ON ml_market_history (provider, status, fetched_at)",
        "CREATE INDEX IF NOT EXISTS ix_optimizer_run_profile_status_created ON optimizer_run (profile, status, created_at)",
    )
    for statement in statements:
        try:
            db.session.execute(text(statement))
        except Exception:  # noqa: BLE001
            db.session.rollback()


def _backfill_public_ids() -> None:
    statements = (
        "UPDATE referral_invite_code SET public_id = 'inv_' || lower(hex(randomblob(11))) WHERE public_id IS NULL OR public_id = ''",
        "UPDATE vault_cycle SET public_id = 'vc_' || lower(hex(randomblob(11))) WHERE public_id IS NULL OR public_id = ''",
        "UPDATE referral_invite_code SET profit_share_percent = percent_profit WHERE COALESCE(profit_share_percent, 0) = 0 AND COALESCE(percent_profit, 0) > 0",
        "UPDATE referral_invite_code SET profit_share_wallet = 'sufyanh' WHERE profit_share_wallet IS NULL OR profit_share_wallet = ''",
    )
    for statement in statements:
        try:
            db.session.execute(text(statement))
        except Exception:  # noqa: BLE001
            db.session.rollback()


def _mark_legacy_zero_live_fills_unknown() -> None:
    fill_columns = _table_columns("fill")
    if not {"metadata_json", "fee_known", "realized_pnl_known"}.issubset(fill_columns):
        return
    try:
        db.session.execute(
            text(
                """
                UPDATE fill
                SET
                    fee_known = 0,
                    realized_pnl_known = 0,
                    metadata_json = '{"reconciliation_status":"legacy_unknown_zero_live_fill"}'
                WHERE id IN (
                    SELECT fill.id
                    FROM fill
                    JOIN "order" ON fill.order_id = "order".id
                    WHERE "order".mode = 'live'
                      AND ABS(COALESCE(fill.fee, 0)) < 0.000000000001
                      AND ABS(COALESCE(fill.pnl, 0)) < 0.000000000001
                      AND COALESCE(fill.metadata_json, '{}') = '{}'
                )
                """
            )
        )
    except Exception:  # noqa: BLE001
        db.session.rollback()


def _backfill_order_vault_links() -> None:
    order_columns = _table_columns("order")
    if not {"vault_cycle_id", "vault_leg_id", "metadata_json"}.issubset(order_columns):
        return
    for order in Order.query.filter(Order.vault_cycle_id.is_(None), Order.metadata_json.like("%vault_cycle_id%")).limit(5_000).all():
        details = order.details
        try:
            cycle_id = int(details.get("vault_cycle_id")) if details.get("vault_cycle_id") is not None else None
        except (TypeError, ValueError):
            cycle_id = None
        try:
            leg_id = int(details.get("vault_leg_id")) if details.get("vault_leg_id") is not None else None
        except (TypeError, ValueError):
            leg_id = None
        if cycle_id and order.vault_cycle_id is None:
            order.vault_cycle_id = cycle_id
        if leg_id and order.vault_leg_id is None:
            order.vault_leg_id = leg_id


def _crypto_rail_assets() -> list[dict]:
    from .models import WalletBalance

    user = current_user()
    symbols = ["BTC", "ETH", "USDT", "SOL", "XRP", "USDC"]
    movement = {
        "BTC": 1.24,
        "ETH": 0.84,
        "USDT": 0.01,
        "SOL": -0.72,
        "XRP": 0.33,
        "USDC": 0.0,
    }
    balances = {}
    try:
        query = WalletBalance.query.filter(WalletBalance.asset.in_(symbols))
        query = query.filter_by(user_id=user.id) if user is not None else query.filter_by(user_id=-1)
        balances = {balance.asset: balance for balance in query.all()}
    except Exception:  # noqa: BLE001
        balances = {}
    return [
        {
            "symbol": symbol,
            "balance": (balances[symbol].total_balance if symbol in balances else 0.0),
            "movement": movement.get(symbol, 0.0),
        }
        for symbol in symbols
    ]
