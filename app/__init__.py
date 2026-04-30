"""Flask application factory."""

from __future__ import annotations

from dotenv import load_dotenv
from flask import Flask
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from .backtesting.engine import BacktestEngine
from .backtesting.optimizer import StrategyOptimizer
from .auth import current_user, password_hash
from .cli import register_cli
from .config import BaseConfig
from .csrf import csrf_input, csrf_token, validate_csrf_request
from .extensions import db
from .features.engine import FeatureEngine
from .ml.offline_ranker import OfflineRanker
from .ml.online_ranker import OnlineRanker
from .models import Setting, StrategyRun, User, VaultCycle, WalletBalance, WalletTransaction
from .admin_auth import admin_authenticated, admin_configured
from .routes.admin import admin_bp
from .routes.auth import auth_bp
from .services.execution import HyperliquidVenue
from .routes.backtests import backtests_bp
from .routes.consumer import consumer_bp
from .routes.dashboard import dashboard_bp
from .routes.orders import orders_bp
from .routes.panic import panic_bp
from .routes.settings import settings_bp
from .services.hyperliquid_client import HyperliquidClient
from .services.market_scanner import MarketScannerService
from .services.market_structure import MarketStructureService
from .services.market_universe import MarketUniverseService
from .services.market_data import MarketDataService
from .services.order_manager import OrderManager
from .services.pair_screening import PairScreeningService
from .services.risk_engine import RiskEngine
from .services.realtime_market import RealtimeMarketService
from .services.self_custody_wallet import SelfCustodyWalletService
from .services.strategy_runner import StrategyManager
from .services.trading_connections import TradingConnectionService
from .services.vault_selector import VaultStrategySelector
from .services.wallet_addresses import WalletAddressService
from .services.wallet_custody import RealWalletCustodyService
from .strategies.registry import StrategyRegistry
from .utils import format_duration_seconds


def create_app(test_config: dict | None = None) -> Flask:
    """Create and configure the Flask application."""

    load_dotenv()
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.from_object(BaseConfig)
    if test_config:
        app.config.update(test_config)
        if app.config.get("TESTING") and "WTF_CSRF_ENABLED" not in test_config:
            app.config["WTF_CSRF_ENABLED"] = False
    _configure_engine_options(app)

    db.init_app(app)

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
    market_scanner.online_ranker = online_ranker
    market_scanner.offline_ranker = offline_ranker
    pair_screening = PairScreeningService(app.config, market_data, market_universe, market_structure, online_ranker)
    market_scanner.pair_screening = pair_screening
    order_manager = OrderManager(app.config, hyperliquid_client, market_data, risk_engine, trading_connections)
    backtest_engine = BacktestEngine(app.config, strategy_registry, market_data)
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
    wallet_address_service = WalletAddressService(app.config)
    wallet_custody = RealWalletCustodyService(app.config)
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
    )

    app.extensions["services"] = {
        "strategy_registry": strategy_registry,
        "hyperliquid_client": hyperliquid_client,
        "execution_venue": execution_venue,
        "market_data": market_data,
        "realtime_market": realtime_market,
        "market_structure": market_structure,
        "market_universe": market_universe,
        "market_scanner": market_scanner,
        "pair_screening": pair_screening,
        "feature_engine": feature_engine,
        "risk_engine": risk_engine,
        "trading_connections": trading_connections,
        "online_ranker": online_ranker,
        "offline_ranker": offline_ranker,
        "order_manager": order_manager,
        "backtest_engine": backtest_engine,
        "strategy_optimizer": strategy_optimizer,
        "vault_strategy_selector": vault_strategy_selector,
        "wallet_address_service": wallet_address_service,
        "wallet_custody": wallet_custody,
        "self_custody_wallet": self_custody_wallet,
        "strategy_manager": strategy_manager,
    }

    app.register_blueprint(auth_bp)
    app.register_blueprint(consumer_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(orders_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(panic_bp)
    app.register_blueprint(backtests_bp)
    register_cli(app)

    app.before_request(validate_csrf_request)
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
        _configure_sqlite_pragmas(app)
        _create_all_tolerant()
        _ensure_schema()
        _seed_default_settings(app)
        admin_user = _seed_admin_user(app)
        db.session.commit()
        _migrate_legacy_wallet_rows(admin_user)
        _reset_stale_strategy_runs()

    return app


def _configure_engine_options(app: Flask) -> None:
    uri = str(app.config.get("SQLALCHEMY_DATABASE_URI", ""))
    if not uri.startswith("sqlite"):
        return
    options = dict(app.config.get("SQLALCHEMY_ENGINE_OPTIONS") or {})
    connect_args = dict(options.get("connect_args") or {})
    connect_args.setdefault("timeout", max(float(app.config.get("SQLITE_BUSY_TIMEOUT_MS", 10_000)) / 1000, 1.0))
    options["connect_args"] = connect_args
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
    additions = {
        "user": {
            "role": "role VARCHAR(32) NOT NULL DEFAULT 'user'",
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
        },
        "vault_allocation_leg": {
            "strategy_run_id": "strategy_run_id INTEGER",
            "optimizer_ranking_id": "optimizer_ranking_id INTEGER",
            "realized_pnl_usd": "realized_pnl_usd FLOAT NOT NULL DEFAULT 0",
            "unrealized_pnl_usd": "unrealized_pnl_usd FLOAT NOT NULL DEFAULT 0",
            "metadata_json": "metadata_json TEXT NOT NULL DEFAULT '{}'",
        },
    }
    for table, columns in additions.items():
        existing = _table_columns(table)
        if not existing:
            continue
        for name, ddl in columns.items():
            if name not in existing:
                db.session.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl}"))
    db.session.commit()


def _table_columns(table: str) -> set[str]:
    try:
        rows = db.session.execute(text(f"PRAGMA table_info({table})")).mappings().all()
    except Exception:  # noqa: BLE001
        return set()
    return {row["name"] for row in rows}


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
        if user is not None:
            query = query.filter_by(user_id=user.id)
        else:
            query = query.filter_by(user_id=-1)
        balances = {
            balance.asset: balance
            for balance in query.all()
        }
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
