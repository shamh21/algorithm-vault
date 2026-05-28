"""Application configuration and environment parsing."""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from hyperliquid.utils import constants as hl_constants
except ImportError:  # pragma: no cover - import fallback for environments without the SDK

    class _FallbackConstants:
        TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"
        MAINNET_API_URL = "https://api.hyperliquid.xyz"

    hl_constants = _FallbackConstants()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _clean_public_origin(value: str | None, default: str = "") -> str:
    raw = str(value if value not in {None, ""} else default).strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        return raw.rstrip("/")
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}".rstrip("/")


def _parse_origin_list(value: str | None, defaults: list[str] | tuple[str, ...] = ()) -> list[str]:
    origins: list[str] = []
    for item in list(defaults) + [part for part in str(value or "").split(",")]:
        origin = _clean_public_origin(str(item or "").strip(), "")
        parsed = urlparse(origin)
        if origin == "*" or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if origin and origin not in origins:
            origins.append(origin)
    return origins


def _normalize_database_url(value: str | None, default: str) -> str:
    raw = str(value or default).strip() or default
    if raw.startswith("postgres://"):
        return "postgresql+psycopg://" + raw.removeprefix("postgres://")
    if raw.startswith("postgresql://"):
        return "postgresql+psycopg://" + raw.removeprefix("postgresql://")
    return raw


def _prepare_recovery_sqlite_database_url(default_url: str) -> tuple[str, bool]:
    if not _as_bool(os.getenv("ALGVAULT_RECOVERY_SQLITE_ENABLED"), default=False):
        return default_url, False

    bundle_raw = os.getenv("ALGVAULT_RECOVERY_SQLITE_BUNDLE", "app/recovery/algvault_sufyanh.seed").strip()
    runtime_raw = os.getenv("ALGVAULT_RECOVERY_SQLITE_PATH", "/tmp/algvault_sufyanh_recovery.sqlite").strip()
    bundle_path = Path(bundle_raw)
    if not bundle_path.is_absolute():
        bundle_path = Path(__file__).resolve().parents[1] / bundle_path
    runtime_path = Path(runtime_raw)
    if not runtime_path.is_absolute():
        runtime_path = Path("/tmp") / runtime_path

    if not bundle_path.exists():
        raise RuntimeError(f"Recovery SQLite bundle is missing: {bundle_path}")

    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    should_copy = not runtime_path.exists()
    if not should_copy:
        bundle_stat = bundle_path.stat()
        runtime_stat = runtime_path.stat()
        should_copy = runtime_stat.st_size != bundle_stat.st_size or runtime_stat.st_mtime < bundle_stat.st_mtime
    if should_copy:
        shutil.copy2(bundle_path, runtime_path)
    return f"sqlite:///{runtime_path}", True


def _is_local_or_private_hostname(hostname: str) -> bool:
    normalized = hostname.strip().lower().strip("[]")
    if normalized in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return bool(address.is_private or address.is_loopback or address.is_link_local)


def public_origin_violations(origin: str, *, require_public_https: bool) -> list[str]:
    if not require_public_https:
        return []
    if not origin:
        return ["is missing"]
    parsed = urlparse(origin)
    hostname = parsed.hostname or ""
    violations: list[str] = []
    if parsed.scheme != "https":
        violations.append("must use https")
    if not hostname:
        violations.append("must include a hostname")
    if parsed.username or parsed.password:
        violations.append("must not include credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        violations.append("must be an origin without path, query, or fragment")
    if hostname and _is_local_or_private_hostname(hostname):
        violations.append("must not use localhost, loopback, or private IP hosts")
    return violations


def _as_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _parse_allowed_symbols(value: str | None) -> list[str]:
    if not value:
        return ["BTC", "ETH", "SOL"]
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _parse_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_upper_csv(value: str | None, default: list[str]) -> list[str]:
    return [item.upper() for item in _parse_csv(value, default)]


def _parse_int_csv(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)
    parsed: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed.append(int(item))
        except ValueError:
            continue
    return parsed or list(default)


def _parse_float_map(value: str | None) -> dict[str, float]:
    if not value:
        return {}
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, float] = {}
    for key, item in raw.items():
        try:
            parsed[str(key).strip().lower()] = float(item)
        except (TypeError, ValueError):
            continue
    return parsed


def _parse_json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _normalize_mode(value: str | None, enable_live: bool) -> str:
    normalized = str(value or "paper").strip().lower()
    if normalized == "paper":
        return "paper"
    if normalized == "live" and enable_live:
        return "live"
    return "paper"


def _parse_deposit_address_book(value: str | None) -> dict[str, dict[str, list[str]]]:
    if not value:
        return {}
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(raw, dict):
        return {}
    book: dict[str, dict[str, list[str]]] = {}
    for asset, networks in raw.items():
        if not isinstance(networks, dict):
            continue
        asset_key = str(asset).upper()
        book[asset_key] = {}
        for network, addresses in networks.items():
            if isinstance(addresses, str):
                normalized = [addresses]
            elif isinstance(addresses, list):
                normalized = [str(item).strip() for item in addresses if str(item).strip()]
            else:
                normalized = []
            if normalized:
                book[asset_key][str(network)] = normalized
    return book


class BaseConfig:
    """Default configuration for local development and testing."""

    APP_NAME = os.getenv("APP_NAME", "Algorithm Vault")
    ASSET_VERSION = os.getenv("ASSET_VERSION", "1").strip() or "1"
    DEPLOYMENT_TARGET = os.getenv("DEPLOYMENT_TARGET", "local").strip().lower() or "local"
    PUBLIC_APP_ORIGIN = _clean_public_origin(os.getenv("PUBLIC_APP_ORIGIN"), "")
    PUBLIC_API_ORIGIN = _clean_public_origin(os.getenv("PUBLIC_API_ORIGIN"), PUBLIC_APP_ORIGIN)
    PUBLIC_LIVE_API_ORIGIN = _clean_public_origin(os.getenv("PUBLIC_LIVE_API_ORIGIN"), "")
    LIVE_API_CORS_ALLOWED_ORIGINS = _parse_origin_list(
        os.getenv("LIVE_API_CORS_ALLOWED_ORIGINS"),
        (PUBLIC_APP_ORIGIN, "https://algorithm-vault-chi.vercel.app"),
    )
    WEB_CONCURRENCY = _as_int(os.getenv("WEB_CONCURRENCY") or os.getenv("GUNICORN_WORKERS"), 1)
    GUNICORN_THREADS = _as_int(os.getenv("GUNICORN_THREADS"), 4)
    WORKER_MODE = os.getenv("WORKER_MODE", "web").strip().lower() or "web"
    ENABLE_IN_PROCESS_WORKERS = _as_bool(os.getenv("ENABLE_IN_PROCESS_WORKERS"), default=True)
    WORKER_PROCESS_CONFIGURED = _as_bool(os.getenv("WORKER_PROCESS_CONFIGURED"), default=False)
    WORKER_LEASE_TTL_SECONDS = max(5, _as_int(os.getenv("WORKER_LEASE_TTL_SECONDS"), 120))
    WORKER_POLL_SECONDS = max(1, _as_int(os.getenv("WORKER_POLL_SECONDS"), 15))
    STRATEGY_RUN_LEASE_TTL_SECONDS = max(30, _as_int(os.getenv("STRATEGY_RUN_LEASE_TTL_SECONDS"), 120))
    WORKER_STRATEGY_STARTER_ENABLED = _as_bool(os.getenv("WORKER_STRATEGY_STARTER_ENABLED"), default=True)
    WORKER_VAULT_ENFORCEMENT_ENABLED = _as_bool(os.getenv("WORKER_VAULT_ENFORCEMENT_ENABLED"), default=True)
    WORKER_TREASURY_SOLVENCY_ENABLED = _as_bool(os.getenv("WORKER_TREASURY_SOLVENCY_ENABLED"), default=True)
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin").strip() or "admin"
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()
    SIGNUP_INVITE_CODE = os.getenv("SIGNUP_INVITE_CODE", "").strip()
    TOTP_ENCRYPTION_KEY = os.getenv("TOTP_ENCRYPTION_KEY", "").strip()
    DEPOSIT_ADDRESS_BOOK = _parse_deposit_address_book(os.getenv("DEPOSIT_ADDRESSES_JSON"))
    CONFIGURED_DATABASE_URL = _normalize_database_url(os.getenv("DATABASE_URL"), "sqlite:///hyperliquid_dashboard.db")
    SQLALCHEMY_DATABASE_URI, RECOVERY_SQLITE_ACTIVE = _prepare_recovery_sqlite_database_url(CONFIGURED_DATABASE_URL)
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLITE_BUSY_TIMEOUT_MS = _as_int(os.getenv("SQLITE_BUSY_TIMEOUT_MS"), 10_000)
    SQLITE_ENABLE_WAL = _as_bool(os.getenv("SQLITE_ENABLE_WAL"), default=True)
    RECOVERY_POSTGRES_PROBE_TIMEOUT_SECONDS = _as_float(os.getenv("RECOVERY_POSTGRES_PROBE_TIMEOUT_SECONDS"), 2.0)
    RECOVERY_POSTGRES_PROBE_TTL_SECONDS = _as_float(os.getenv("RECOVERY_POSTGRES_PROBE_TTL_SECONDS"), 60.0)
    SCHEMA_BOOTSTRAP_ENABLED = False if RECOVERY_SQLITE_ACTIVE else _as_bool(os.getenv("SCHEMA_BOOTSTRAP_ENABLED"), default=True)
    ALLOW_PRODUCTION_SCHEMA_BOOTSTRAP = _as_bool(os.getenv("ALLOW_PRODUCTION_SCHEMA_BOOTSTRAP"), default=False)
    DEFER_DATABASE_STARTUP_ERRORS = _as_bool(
        os.getenv("DEFER_DATABASE_STARTUP_ERRORS"),
        default=_as_bool(os.getenv("VERCEL"), default=False),
    )

    ENABLE_LIVE_TRADING = False if RECOVERY_SQLITE_ACTIVE else _as_bool(os.getenv("ENABLE_LIVE_TRADING"), default=False)
    APP_MODE = _normalize_mode(os.getenv("APP_MODE", "paper"), enable_live=ENABLE_LIVE_TRADING)
    HL_TESTNET_BASE_URL = hl_constants.TESTNET_API_URL
    HL_MAINNET_BASE_URL = hl_constants.MAINNET_API_URL
    HL_WS_MAINNET_URL = os.getenv("HL_WS_MAINNET_URL", "wss://api.hyperliquid.xyz/ws").strip()
    HL_WS_TESTNET_URL = os.getenv("HL_WS_TESTNET_URL", "wss://api.hyperliquid-testnet.xyz/ws").strip()

    HL_ACCOUNT_ADDRESS = os.getenv("HL_ACCOUNT_ADDRESS", "").strip()
    HL_SECRET_KEY = os.getenv("HL_SECRET_KEY", "").strip()
    HL_VAULT_ADDRESS = os.getenv("HL_VAULT_ADDRESS", "").strip() or None
    HL_TIMEOUT_SECONDS = _as_float(os.getenv("HL_TIMEOUT_SECONDS"), 10.0)
    HYPERLIQUID_ACCOUNT = os.getenv("HYPERLIQUID_ACCOUNT", "").strip()
    HYPERLIQUID_ENV = os.getenv("HYPERLIQUID_ENV", "").strip().lower()
    HYPERLIQUID_BASE_URL = os.getenv("HYPERLIQUID_BASE_URL", "").strip()
    RUN_HYPERLIQUID_LIVE_TESTS = _as_bool(os.getenv("RUN_HYPERLIQUID_LIVE_TESTS"), default=False)
    RUN_HYPERLIQUID_FILL_TEST = _as_bool(os.getenv("RUN_HYPERLIQUID_FILL_TEST"), default=False)
    HYPERLIQUID_LIVE_TEST_MAX_NOTIONAL_USD = max(0.0, _as_float(os.getenv("HYPERLIQUID_LIVE_TEST_MAX_NOTIONAL_USD"), 0.0))
    HYPERLIQUID_MIN_ORDER_VALUE_USD = max(0.0, _as_float(os.getenv("HYPERLIQUID_MIN_ORDER_VALUE_USD"), 10.0))
    EXCHANGE_RETRY_ATTEMPTS = _as_int(os.getenv("EXCHANGE_RETRY_ATTEMPTS"), 3)
    EXCHANGE_RETRY_SLEEP_SECONDS = _as_float(os.getenv("EXCHANGE_RETRY_SLEEP_SECONDS"), 0.5)
    PROVIDER_TIMEOUT_SECONDS = _as_float(os.getenv("PROVIDER_TIMEOUT_SECONDS"), 10.0)
    PROVIDER_RETRY_ATTEMPTS = _as_int(os.getenv("PROVIDER_RETRY_ATTEMPTS"), EXCHANGE_RETRY_ATTEMPTS)
    PROVIDER_RETRY_SLEEP_SECONDS = _as_float(os.getenv("PROVIDER_RETRY_SLEEP_SECONDS"), EXCHANGE_RETRY_SLEEP_SECONDS)
    BINANCE_FUTURES_BASE_URL = os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com").strip()
    BINANCE_RECV_WINDOW_MS = _as_int(os.getenv("BINANCE_RECV_WINDOW_MS"), 5000)
    BINANCE_SYMBOL_MAP_JSON = os.getenv("BINANCE_SYMBOL_MAP_JSON", "").strip()
    KUCOIN_FUTURES_BASE_URL = os.getenv("KUCOIN_FUTURES_BASE_URL", "https://api-futures.kucoin.com").strip()
    KUCOIN_SPOT_BASE_URL = os.getenv("KUCOIN_SPOT_BASE_URL", "https://api.kucoin.com").strip()
    KUCOIN_API_KEY = os.getenv("KUCOIN_API_KEY", "").strip()
    KUCOIN_API_SECRET = os.getenv("KUCOIN_API_SECRET", "").strip()
    KUCOIN_API_PASSPHRASE = os.getenv("KUCOIN_API_PASSPHRASE", "").strip()
    KUCOIN_DEFAULT_MARKET_TYPE = os.getenv("KUCOIN_DEFAULT_MARKET_TYPE", "futures").strip().lower() or "futures"
    KUCOIN_DEFAULT_SPOT_QUOTE = os.getenv("KUCOIN_DEFAULT_SPOT_QUOTE", "USDT").strip().upper() or "USDT"
    KUCOIN_ACCOUNT_OVERVIEW_PATH = os.getenv("KUCOIN_ACCOUNT_OVERVIEW_PATH", "/api/v1/account-overview").strip()
    KUCOIN_ORDERS_PATH = os.getenv("KUCOIN_ORDERS_PATH", "/api/v1/orders").strip()
    KUCOIN_OPEN_ORDERS_PATH = os.getenv("KUCOIN_OPEN_ORDERS_PATH", "/api/v1/orders").strip()
    KUCOIN_POSITIONS_PATH = os.getenv("KUCOIN_POSITIONS_PATH", "/api/v1/positions").strip()
    KUCOIN_POSITION_MODE_PATH = os.getenv("KUCOIN_POSITION_MODE_PATH", "/api/v2/position/getPositionMode").strip()
    KUCOIN_FILLS_PATH = os.getenv("KUCOIN_FILLS_PATH", "/api/v1/recentDoneOrders").strip()
    KUCOIN_SPOT_ACCOUNTS_PATH = os.getenv("KUCOIN_SPOT_ACCOUNTS_PATH", "/api/v1/accounts").strip()
    KUCOIN_SUB_ACCOUNTS_PATH = os.getenv("KUCOIN_SUB_ACCOUNTS_PATH", "/api/v1/sub-accounts").strip()
    KUCOIN_SPOT_SYMBOLS_PATH = os.getenv("KUCOIN_SPOT_SYMBOLS_PATH", "/api/v2/symbols").strip()
    KUCOIN_SPOT_SYMBOL_PATH = os.getenv("KUCOIN_SPOT_SYMBOL_PATH", "/api/v2/symbols").strip()
    KUCOIN_SPOT_TICKER_PATH = os.getenv("KUCOIN_SPOT_TICKER_PATH", "/api/v1/market/orderbook/level1").strip()
    KUCOIN_SPOT_ORDERS_PATH = os.getenv("KUCOIN_SPOT_ORDERS_PATH", "/api/v1/hf/orders").strip()
    KUCOIN_SPOT_TEST_ORDER_PATH = os.getenv("KUCOIN_SPOT_TEST_ORDER_PATH", "/api/v1/hf/orders/test").strip()
    KUCOIN_SPOT_OPEN_ORDERS_PATH = os.getenv("KUCOIN_SPOT_OPEN_ORDERS_PATH", "/api/v1/hf/orders/active").strip()
    KUCOIN_SPOT_CLIENT_ORDER_PATH = os.getenv("KUCOIN_SPOT_CLIENT_ORDER_PATH", "/api/v1/hf/orders/client-order").strip()
    KUCOIN_SPOT_FILLS_PATH = os.getenv("KUCOIN_SPOT_FILLS_PATH", "/api/v1/hf/fills").strip()
    KUCOIN_ACTIVE_CONTRACTS_PATH = os.getenv("KUCOIN_ACTIVE_CONTRACTS_PATH", "/api/v1/contracts/active").strip()
    KUCOIN_DEPOSIT_ADDRESSES_PATH = os.getenv("KUCOIN_DEPOSIT_ADDRESSES_PATH", "/api/v3/deposit-addresses").strip()
    KUCOIN_WITHDRAWALS_PATH = os.getenv("KUCOIN_WITHDRAWALS_PATH", "/api/v3/withdrawals").strip()
    KUCOIN_UNIVERSAL_TRANSFER_PATH = os.getenv("KUCOIN_UNIVERSAL_TRANSFER_PATH", "/api/v3/accounts/universal-transfer").strip()
    KUCOIN_CONVERT_QUOTE_PATH = os.getenv("KUCOIN_CONVERT_QUOTE_PATH", "/api/v1/convert/quote").strip()
    KUCOIN_CONVERT_ORDER_PATH = os.getenv("KUCOIN_CONVERT_ORDER_PATH", "/api/v1/convert/order").strip()
    KUCOIN_SPOT_SYMBOL_MAP_JSON = os.getenv("KUCOIN_SPOT_SYMBOL_MAP_JSON", "").strip()
    KUCOIN_SYMBOL_MAP_JSON = os.getenv("KUCOIN_SYMBOL_MAP_JSON", "").strip()
    KUCOIN_CONTRACT_SPECS_JSON = os.getenv("KUCOIN_CONTRACT_SPECS_JSON", "").strip()
    KUCOIN_TEST_ACCOUNT = os.getenv("KUCOIN_TEST_ACCOUNT", "").strip()
    KUCOIN_TEST_SYMBOL = os.getenv("KUCOIN_TEST_SYMBOL", "").strip().upper()
    KUCOIN_MAX_TEST_NOTIONAL_USDT = _as_float(os.getenv("KUCOIN_MAX_TEST_NOTIONAL_USDT"), 0.0)
    KUCOIN_ENABLE_LIVE_TEST_TRADES = _as_bool(os.getenv("KUCOIN_ENABLE_LIVE_TEST_TRADES"), default=False)
    KUCOIN_ENABLE_FILL_TEST = _as_bool(os.getenv("KUCOIN_ENABLE_FILL_TEST"), default=False)
    KUCOIN_MARGIN_MODE = os.getenv("KUCOIN_MARGIN_MODE", "ISOLATED").strip().upper() or "ISOLATED"
    KUCOIN_POSITION_SIDE = os.getenv("KUCOIN_POSITION_SIDE", "BOTH").strip().upper() or "BOTH"
    KUCOIN_TIME_SYNC_ENABLED = os.getenv("KUCOIN_TIME_SYNC_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    KUCOIN_TIME_SYNC_TTL_SECONDS = float(os.getenv("KUCOIN_TIME_SYNC_TTL_SECONDS", "300"))
    DYDX_INDEXER_URL = os.getenv("DYDX_INDEXER_URL", "https://indexer.dydx.trade/v4").strip()
    DYDX_SYMBOL_MAP_JSON = os.getenv("DYDX_SYMBOL_MAP_JSON", "").strip()
    WALLETCONNECT_PROJECT_ID = os.getenv("WALLETCONNECT_PROJECT_ID", "").strip()
    UNISWAP_DELEGATED_TRADING_ENABLED = _as_bool(os.getenv("UNISWAP_DELEGATED_TRADING_ENABLED"), default=False)
    UNISWAP_API_KEY = os.getenv("UNISWAP_API_KEY", "").strip()
    UNISWAP_API_BASE_URL = os.getenv("UNISWAP_API_BASE_URL", "https://trade-api.gateway.uniswap.org/v1").strip()
    UNISWAP_PROTOCOLS = os.getenv("UNISWAP_PROTOCOLS", "V2,V3,V4").strip()
    UNISWAP_TOKEN_MAP_JSON = os.getenv("UNISWAP_TOKEN_MAP_JSON", "").strip()

    ALLOWED_SYMBOLS = _parse_allowed_symbols(os.getenv("ALLOWED_SYMBOLS"))
    DEFAULT_TIMEFRAME = os.getenv("DEFAULT_TIMEFRAME", "15m").strip()
    DEFAULT_PAPER_BALANCE = max(0.0, _as_float(os.getenv("DEFAULT_PAPER_BALANCE"), 1_000.0))
    BACKTEST_PAPER_BALANCE_USD = max(0.0, _as_float(os.getenv("BACKTEST_PAPER_BALANCE_USD"), 10_000.0))
    BACKTEST_ALLOCATION_DEFAULT_USD = max(0.0, _as_float(os.getenv("BACKTEST_ALLOCATION_DEFAULT_USD"), 10_000.0))
    BACKTEST_MAX_CHART_POINTS = _as_int(os.getenv("BACKTEST_MAX_CHART_POINTS"), 240)
    PAPER_BALANCE_MIN = _as_float(os.getenv("PAPER_BALANCE_MIN"), 0.0)
    PAPER_BALANCE_MAX = _as_float(os.getenv("PAPER_BALANCE_MAX"), 1_000_000.0)
    MAX_DAILY_LOSS_USDC = _as_float(os.getenv("MAX_DAILY_LOSS_USDC"), 5.0)
    MAX_BACKTEST_DRAWDOWN_PCT = _as_float(os.getenv("MAX_BACKTEST_DRAWDOWN_PCT"), 0.2)
    MAX_LEVERAGE = _as_float(os.getenv("MAX_LEVERAGE"), 3.0)
    LOSS_COOLDOWN_MINUTES = _as_int(os.getenv("LOSS_COOLDOWN_MINUTES"), 30)
    LOSS_STREAK_COOLDOWN_THRESHOLD = _as_int(os.getenv("LOSS_STREAK_COOLDOWN_THRESHOLD"), 3)
    MAX_TRADES_PER_WINDOW = _as_int(os.getenv("MAX_TRADES_PER_WINDOW"), 20)
    TRADE_WINDOW_MINUTES = _as_int(os.getenv("TRADE_WINDOW_MINUTES"), 60)
    RISK_PER_TRADE_PCT = _as_float(os.getenv("RISK_PER_TRADE_PCT"), 0.01)
    MIN_REWARD_RISK = _as_float(os.getenv("MIN_REWARD_RISK"), 1.0)
    FIXED_DOLLAR_SIZE = _as_float(os.getenv("FIXED_DOLLAR_SIZE"), 25.0)
    STRATEGY_POLL_SECONDS = _as_int(os.getenv("STRATEGY_POLL_SECONDS"), 20)
    STRATEGY_CHANGE_DRIVEN_LOOP_ENABLED = _as_bool(os.getenv("STRATEGY_CHANGE_DRIVEN_LOOP_ENABLED"), default=False)
    STRATEGY_IDLE_REEVAL_SECONDS = _as_float(os.getenv("STRATEGY_IDLE_REEVAL_SECONDS"), 15.0)
    STRATEGY_HEARTBEAT_PERSIST_SECONDS = _as_float(os.getenv("STRATEGY_HEARTBEAT_PERSIST_SECONDS"), 30.0)
    NO_TRADE_AUDIT_COMPACT_ENABLED = _as_bool(os.getenv("NO_TRADE_AUDIT_COMPACT_ENABLED"), default=True)
    NO_TRADE_AUDIT_THROTTLE_SECONDS = _as_float(os.getenv("NO_TRADE_AUDIT_THROTTLE_SECONDS"), 300.0)
    NO_TRADE_AUDIT_RETENTION_HOURS = _as_float(os.getenv("NO_TRADE_AUDIT_RETENTION_HOURS"), 24.0)
    TRANSIENT_AUDIT_RETENTION_HOURS = _as_float(os.getenv("TRANSIENT_AUDIT_RETENTION_HOURS"), 72.0)
    FEE_BPS = _as_float(os.getenv("FEE_BPS"), 5.0)
    SIM_SLIPPAGE_BPS = _as_float(os.getenv("SIM_SLIPPAGE_BPS"), 8.0)
    DASHBOARD_CANDLE_LIMIT = _as_int(os.getenv("DASHBOARD_CANDLE_LIMIT"), 200)
    DASHBOARD_ACCOUNT_SEGMENT_TTL_SECONDS = _as_float(os.getenv("DASHBOARD_ACCOUNT_SEGMENT_TTL_SECONDS"), 2.0)
    DASHBOARD_ACCOUNT_SEGMENT_STALE_SECONDS = _as_float(os.getenv("DASHBOARD_ACCOUNT_SEGMENT_STALE_SECONDS"), 10.0)
    DASHBOARD_TRADE_LIST_SEGMENT_TTL_SECONDS = _as_float(os.getenv("DASHBOARD_TRADE_LIST_SEGMENT_TTL_SECONDS"), 2.0)
    DASHBOARD_TRADE_LIST_STALE_SECONDS = _as_float(os.getenv("DASHBOARD_TRADE_LIST_STALE_SECONDS"), 10.0)
    DASHBOARD_STATIC_SEGMENT_TTL_SECONDS = _as_float(os.getenv("DASHBOARD_STATIC_SEGMENT_TTL_SECONDS"), 5.0)
    DASHBOARD_STATIC_SEGMENT_STALE_SECONDS = _as_float(os.getenv("DASHBOARD_STATIC_SEGMENT_STALE_SECONDS"), 30.0)
    DASHBOARD_PAGE_SIZE = _as_int(os.getenv("DASHBOARD_PAGE_SIZE"), 30)
    DASHBOARD_OPPORTUNITY_LIMIT = _as_int(os.getenv("DASHBOARD_OPPORTUNITY_LIMIT"), 30)
    DASHBOARD_OPPORTUNITY_TTL_SECONDS = _as_float(os.getenv("DASHBOARD_OPPORTUNITY_TTL_SECONDS"), 10.0)
    DASHBOARD_OPPORTUNITY_STALE_SECONDS = _as_float(os.getenv("DASHBOARD_OPPORTUNITY_STALE_SECONDS"), 45.0)
    DASHBOARD_OPPORTUNITY_DISCOVERY_REFRESH_ENABLED = _as_bool(
        os.getenv("DASHBOARD_OPPORTUNITY_DISCOVERY_REFRESH_ENABLED"),
        default=False,
    )
    DASHBOARD_OPPORTUNITY_DISCOVERY_REFRESH_SECONDS = _as_float(
        os.getenv("DASHBOARD_OPPORTUNITY_DISCOVERY_REFRESH_SECONDS"),
        300.0,
    )
    DASHBOARD_FORECAST_TTL_SECONDS = _as_float(os.getenv("DASHBOARD_FORECAST_TTL_SECONDS"), 90.0)
    DASHBOARD_FORECAST_MAX_ROWS = _as_int(os.getenv("DASHBOARD_FORECAST_MAX_ROWS"), 150)
    DASHBOARD_FORECAST_PREVIEW_ALLOCATION_USD = _as_float(os.getenv("DASHBOARD_FORECAST_PREVIEW_ALLOCATION_USD"), 10.0)
    DASHBOARD_CHART_CANDLE_LIMIT = _as_int(os.getenv("DASHBOARD_CHART_CANDLE_LIMIT"), 150)
    DASHBOARD_STREAM_INTERVAL_SECONDS = _as_float(os.getenv("DASHBOARD_STREAM_INTERVAL_SECONDS"), 10.0)
    EXPLICIT_LIVE_CONFIRMED = _as_bool(os.getenv("EXPLICIT_LIVE_CONFIRMED"), default=False)
    SECONDARY_CONFIRMATION = _as_bool(os.getenv("SECONDARY_CONFIRMATION"), default=False)
    SHADOW_LIVE_MIN_TRADES = _as_int(os.getenv("SHADOW_LIVE_MIN_TRADES"), 10)
    SHADOW_LIVE_MIN_HOURS = _as_float(os.getenv("SHADOW_LIVE_MIN_HOURS"), 24.0)
    VAULT_SHADOW_VALIDATION_MINUTES = _as_float(os.getenv("VAULT_SHADOW_VALIDATION_MINUTES"), 15.0)
    VAULT_SHADOW_MIN_SIGNALS = _as_int(os.getenv("VAULT_SHADOW_MIN_SIGNALS"), 2)
    VAULT_MAX_SPREAD_BPS = _as_float(os.getenv("VAULT_MAX_SPREAD_BPS"), 25.0)
    VAULT_MIN_LIQUIDITY_USD = _as_float(os.getenv("VAULT_MIN_LIQUIDITY_USD"), 1_000.0)
    VAULT_MAX_SLIPPAGE_BPS = _as_float(os.getenv("VAULT_MAX_SLIPPAGE_BPS"), 20.0)
    VAULT_SIGNAL_STABILITY_THRESHOLD = _as_float(os.getenv("VAULT_SIGNAL_STABILITY_THRESHOLD"), 0.65)
    VAULT_AGGRESSIVE_SIZE_MULTIPLIER = _as_float(os.getenv("VAULT_AGGRESSIVE_SIZE_MULTIPLIER"), 1.35)
    VAULT_LIVE_FALLBACK_POLICY = os.getenv("VAULT_LIVE_FALLBACK_POLICY", "live").strip().lower()
    VAULT_MIN_RISK_FRACTION = _as_float(os.getenv("VAULT_MIN_RISK_FRACTION"), RISK_PER_TRADE_PCT)
    VAULT_MAX_RISK_FRACTION = _as_float(os.getenv("VAULT_MAX_RISK_FRACTION"), 0.03)
    OPTIMIZER_TRAINING_WINDOW_DAYS = _as_int(os.getenv("OPTIMIZER_TRAINING_WINDOW_DAYS"), 14)
    OPTIMIZER_TESTING_WINDOW_DAYS = _as_int(os.getenv("OPTIMIZER_TESTING_WINDOW_DAYS"), 2)
    OPTIMIZER_STEP_DAYS = _as_int(os.getenv("OPTIMIZER_STEP_DAYS"), 1)
    OPTIMIZER_USE_FULL_HISTORY = _as_bool(os.getenv("OPTIMIZER_USE_FULL_HISTORY"), default=False)
    OPTIMIZER_RECENCY_WEIGHTING_ENABLED = _as_bool(
        os.getenv("OPTIMIZER_RECENCY_WEIGHTING_ENABLED"),
        default=True,
    )
    OPTIMIZER_DECAY_FACTOR = _as_float(os.getenv("OPTIMIZER_DECAY_FACTOR"), 0.9)
    OPTIMIZER_MARKET_DATA_TIMEOUT_SECONDS = _as_float(os.getenv("OPTIMIZER_MARKET_DATA_TIMEOUT_SECONDS"), 120.0)
    FIND_CANARY_RANKING_TIMEOUT_SECONDS = _as_float(os.getenv("FIND_CANARY_RANKING_TIMEOUT_SECONDS"), 120.0)
    FIND_CANARY_MAX_SYMBOLS = _as_int(os.getenv("FIND_CANARY_MAX_SYMBOLS"), 4)
    FIND_CANARY_MAX_RANKINGS = _as_int(os.getenv("FIND_CANARY_MAX_RANKINGS"), 10)
    FIND_CANARY_RANKING_ALLOW_MOCK_DATA = _as_bool(os.getenv("FIND_CANARY_RANKING_ALLOW_MOCK_DATA"), default=False)
    FIND_CANARY_RANKING_FALLBACK_SYMBOLS = os.getenv(
        "FIND_CANARY_RANKING_FALLBACK_SYMBOLS",
        '{"hyperliquid":["BTC","ETH"],"kucoin":["BTC","ETH"],"active":["BTC","ETH"]}',
    ).strip()
    FIND_CANARY_MARKET_DATA_PROBE_LIMIT = _as_int(os.getenv("FIND_CANARY_MARKET_DATA_PROBE_LIMIT"), 250)
    FIND_CANARY_PER_SWEEP_TIMEOUT_SECONDS = _as_float(os.getenv("FIND_CANARY_PER_SWEEP_TIMEOUT_SECONDS"), 30.0)
    FIND_CANARY_FALLBACK_ENABLED = _as_bool(os.getenv("FIND_CANARY_FALLBACK_ENABLED"), default=False)
    FIND_CANARY_FALLBACK_SYMBOL = os.getenv("FIND_CANARY_FALLBACK_SYMBOL", "BTC").strip().upper() or "BTC"
    FIND_CANARY_FALLBACK_STRATEGY = os.getenv("FIND_CANARY_FALLBACK_STRATEGY", "ema_crossover").strip() or "ema_crossover"
    FIND_CANARY_FALLBACK_TIMEFRAME = os.getenv("FIND_CANARY_FALLBACK_TIMEFRAME", "15m").strip() or "15m"
    FIND_CANARY_FALLBACK_RECENCY_HOURS = _as_float(os.getenv("FIND_CANARY_FALLBACK_RECENCY_HOURS"), 24.0)
    FIND_CANARY_FALLBACK_ALLOCATION_USD = _as_float(os.getenv("FIND_CANARY_FALLBACK_ALLOCATION_USD"), 1.0)
    FIND_CANARY_FALLBACK_STOP_LOSS_PCT = _as_float(os.getenv("FIND_CANARY_FALLBACK_STOP_LOSS_PCT"), 0.005)
    FIND_CANARY_FALLBACK_TAKE_PROFIT_PCT = _as_float(os.getenv("FIND_CANARY_FALLBACK_TAKE_PROFIT_PCT"), 0.012)
    OPTIMIZER_PREFILTER_ENABLED = _as_bool(os.getenv("OPTIMIZER_PREFILTER_ENABLED"), default=True)
    OPTIMIZER_SIGNAL_HISTORY_LIMIT = _as_int(os.getenv("OPTIMIZER_SIGNAL_HISTORY_LIMIT"), 120)
    MARKET_DATA_CACHE_TTL_SECONDS = _as_float(os.getenv("MARKET_DATA_CACHE_TTL_SECONDS"), 5.0)
    MARKET_DATA_LIVE_CANDLE_CACHE_SECONDS = _as_float(os.getenv("MARKET_DATA_LIVE_CANDLE_CACHE_SECONDS"), 55.0)
    MARKET_DATA_LIVE_ORDER_BOOK_CACHE_SECONDS = _as_float(os.getenv("MARKET_DATA_LIVE_ORDER_BOOK_CACHE_SECONDS"), 5.0)
    MARKET_DATA_LIVE_MIDS_CACHE_SECONDS = _as_float(os.getenv("MARKET_DATA_LIVE_MIDS_CACHE_SECONDS"), 5.0)
    MARKET_DATA_LIVE_STALE_SECONDS = _as_float(os.getenv("MARKET_DATA_LIVE_STALE_SECONDS"), 300.0)
    MARKET_DATA_LIVE_FAILURE_BACKOFF_SECONDS = _as_float(os.getenv("MARKET_DATA_LIVE_FAILURE_BACKOFF_SECONDS"), 30.0)
    TRADING_CONNECTION_LIVE_SNAPSHOT_CACHE_SECONDS = _as_float(os.getenv("TRADING_CONNECTION_LIVE_SNAPSHOT_CACHE_SECONDS"), 20.0)
    TRADING_CONNECTION_LIVE_SNAPSHOT_BACKOFF_SECONDS = _as_float(os.getenv("TRADING_CONNECTION_LIVE_SNAPSHOT_BACKOFF_SECONDS"), 30.0)
    TRADING_CONNECTION_LIVE_SNAPSHOT_STALE_SECONDS = _as_float(os.getenv("TRADING_CONNECTION_LIVE_SNAPSHOT_STALE_SECONDS"), 120.0)
    MARKET_DATA_RESEARCH_STALE_SECONDS = _as_float(os.getenv("MARKET_DATA_RESEARCH_STALE_SECONDS"), 600.0)
    MARKET_DATA_RESEARCH_CACHE_TTL_SECONDS = _as_float(os.getenv("MARKET_DATA_RESEARCH_CACHE_TTL_SECONDS"), 60.0)
    NET_ROI_MIN_EDGE_BPS = _as_float(os.getenv("NET_ROI_MIN_EDGE_BPS"), 4.0)
    NET_ROI_MAX_CHURN_PENALTY = _as_float(os.getenv("NET_ROI_MAX_CHURN_PENALTY"), 0.35)
    NET_ROI_MIN_FILL_QUALITY = _as_float(os.getenv("NET_ROI_MIN_FILL_QUALITY"), 0.55)
    NET_ROI_V2_ENABLED = _as_bool(os.getenv("NET_ROI_V2_ENABLED"), default=True)
    NET_ROI_V2_MIN_QUALITY_GRADE = os.getenv("NET_ROI_V2_MIN_QUALITY_GRADE", "B").strip().upper() or "B"
    ONE_HOUR_EDGE_V2_ENABLED = _as_bool(os.getenv("ONE_HOUR_EDGE_V2_ENABLED"), default=True)
    ONE_HOUR_MIN_EDGE_GRADE = os.getenv("ONE_HOUR_MIN_EDGE_GRADE", "B").strip().upper() or "B"
    ONE_HOUR_MAX_RAW_NET_GAP_PCT = _as_float(os.getenv("ONE_HOUR_MAX_RAW_NET_GAP_PCT"), 35.0)
    ONE_HOUR_MIN_EXECUTION_QUALITY = _as_float(os.getenv("ONE_HOUR_MIN_EXECUTION_QUALITY"), 0.60)
    SIGNAL_DEBOUNCE_SECONDS = _as_float(os.getenv("SIGNAL_DEBOUNCE_SECONDS"), 45.0)
    CANARY_PREVIEW_ONLY = _as_bool(os.getenv("CANARY_PREVIEW_ONLY"), default=True)
    FIRST_CANARY_ALLOCATION_BUDGET_USDT = _as_float(os.getenv("FIRST_CANARY_ALLOCATION_BUDGET_USDT"), 1.0)
    FIRST_CANARY_FALLBACK_ALLOCATION_BUDGET_USDT = _as_float(os.getenv("FIRST_CANARY_FALLBACK_ALLOCATION_BUDGET_USDT"), 2.0)
    FIRST_CANARY_USE_MIN_SIZE_FALLBACK = _as_bool(os.getenv("FIRST_CANARY_USE_MIN_SIZE_FALLBACK"), default=False)
    FIRST_CANARY_MAX_LEVERAGE = _as_float(os.getenv("FIRST_CANARY_MAX_LEVERAGE"), 1.0)
    LIVE_MICRO_CANARY_ENABLED = _as_bool(os.getenv("LIVE_MICRO_CANARY_ENABLED"), default=False)
    LIVE_MICRO_CANARY_ACCOUNT_USD = _as_float(os.getenv("LIVE_MICRO_CANARY_ACCOUNT_USD"), 10.0)
    LIVE_MICRO_CANARY_MAX_ALLOCATION_USD = _as_float(os.getenv("LIVE_MICRO_CANARY_MAX_ALLOCATION_USD"), 1.0)
    LIVE_MICRO_CANARY_MAX_RISK_PCT = _as_float(os.getenv("LIVE_MICRO_CANARY_MAX_RISK_PCT"), 0.01)
    LIVE_MICRO_CANARY_MAX_LEVERAGE = _as_float(os.getenv("LIVE_MICRO_CANARY_MAX_LEVERAGE"), 1.0)
    LIVE_MICRO_CANARY_REQUIRE_STOP_LOSS = _as_bool(os.getenv("LIVE_MICRO_CANARY_REQUIRE_STOP_LOSS"), default=True)
    LIVE_MICRO_CANARY_REQUIRE_TAKE_PROFIT = _as_bool(os.getenv("LIVE_MICRO_CANARY_REQUIRE_TAKE_PROFIT"), default=True)
    LIVE_MICRO_CANARY_PREVIEW_ONLY = _as_bool(os.getenv("LIVE_MICRO_CANARY_PREVIEW_ONLY"), default=True)
    LIVE_MICRO_CANARY_LIVE_SUBMIT_ENABLED = _as_bool(os.getenv("LIVE_MICRO_CANARY_LIVE_SUBMIT_ENABLED"), default=False)
    LIVE_MICRO_CANARY_MIN_NOTIONAL_BUFFER_USD = _as_float(os.getenv("LIVE_MICRO_CANARY_MIN_NOTIONAL_BUFFER_USD"), 0.50)
    LIVE_MICRO_CANARY_MAX_DAILY_LIVE_ORDERS = _as_int(os.getenv("LIVE_MICRO_CANARY_MAX_DAILY_LIVE_ORDERS"), 1)
    LIVE_MICRO_CANARY_REQUIRE_EXACT_CONFIRMATION = _as_bool(os.getenv("LIVE_MICRO_CANARY_REQUIRE_EXACT_CONFIRMATION"), default=True)
    LIVE_MICRO_CANARY_EXACT_CONFIRMATION = (
        os.getenv("LIVE_MICRO_CANARY_EXACT_CONFIRMATION", "LIVE-CANARY-TRADE").strip() or "LIVE-CANARY-TRADE"
    )
    LIVE_MICRO_CANARY_ALLOW_MIN_NOTIONAL_ORDER = _as_bool(os.getenv("LIVE_MICRO_CANARY_ALLOW_MIN_NOTIONAL_ORDER"), default=False)
    LIVE_MICRO_CANARY_ORDER_BUDGET_USD = _as_float(os.getenv("LIVE_MICRO_CANARY_ORDER_BUDGET_USD"), 10.0)
    LIVE_MICRO_CANARY_DEFAULT_MIN_NOTIONAL_USD = _as_float(os.getenv("LIVE_MICRO_CANARY_DEFAULT_MIN_NOTIONAL_USD"), 10.0)
    RAPID_ML_LIVE_ENABLED = _as_bool(os.getenv("RAPID_ML_LIVE_ENABLED"), default=False)
    RAPID_ML_PREVIEW_ONLY = _as_bool(os.getenv("RAPID_ML_PREVIEW_ONLY"), default=True)
    RAPID_ML_DECISION_INTERVAL_MS = max(250, _as_int(os.getenv("RAPID_ML_DECISION_INTERVAL_MS"), 1000))
    RAPID_ML_MAX_ORDER_RATE_PER_PROVIDER = max(0.0, _as_float(os.getenv("RAPID_ML_MAX_ORDER_RATE_PER_PROVIDER"), 1.0))
    RAPID_ML_SYMBOLS = os.getenv("RAPID_ML_SYMBOLS", "").strip()
    RAPID_ML_SYMBOLS_HYPERLIQUID = os.getenv("RAPID_ML_SYMBOLS_HYPERLIQUID", "").strip()
    RAPID_ML_SYMBOLS_KUCOIN = os.getenv("RAPID_ML_SYMBOLS_KUCOIN", "").strip()
    RAPID_ML_AUTO_FUTURES_UNIVERSE_ENABLED = _as_bool(os.getenv("RAPID_ML_AUTO_FUTURES_UNIVERSE_ENABLED"), default=True)
    RAPID_ML_UNIVERSE_REFRESH_SECONDS = max(0.0, _as_float(os.getenv("RAPID_ML_UNIVERSE_REFRESH_SECONDS"), 300.0))
    RAPID_ML_MAX_SYMBOLS_PER_PROVIDER = max(1, _as_int(os.getenv("RAPID_ML_MAX_SYMBOLS_PER_PROVIDER"), 48))
    RAPID_ML_ML_SIZING_ENABLED = _as_bool(os.getenv("RAPID_ML_ML_SIZING_ENABLED"), default=True)
    RAPID_ML_HARD_CAP_USDC = max(0.0, _as_float(os.getenv("RAPID_ML_HARD_CAP_USDC"), 0.0))
    RAPID_ML_AUTO_CLOSE_ENABLED = _as_bool(os.getenv("RAPID_ML_AUTO_CLOSE_ENABLED"), default=True)
    RAPID_ML_ROTATE_ENABLED = _as_bool(os.getenv("RAPID_ML_ROTATE_ENABLED"), default=True)
    RAPID_ML_ROTATE_MIN_SCORE_DELTA = max(0.0, _as_float(os.getenv("RAPID_ML_ROTATE_MIN_SCORE_DELTA"), 0.10))
    RAPID_ML_MANAGE_MANUAL_POSITIONS = _as_bool(os.getenv("RAPID_ML_MANAGE_MANUAL_POSITIONS"), default=False)
    RAPID_ML_FEATURE_TIMEFRAME = os.getenv("RAPID_ML_FEATURE_TIMEFRAME", "1m").strip().lower() or "1m"
    RAPID_ML_FEATURE_CANDLE_LIMIT = max(40, min(500, _as_int(os.getenv("RAPID_ML_FEATURE_CANDLE_LIMIT"), 200)))
    RAPID_ML_PREVIEW_SNAPSHOT_TTL_SECONDS = max(15.0, min(300.0, _as_float(os.getenv("RAPID_ML_PREVIEW_SNAPSHOT_TTL_SECONDS"), 120.0)))
    RAPID_ML_MAX_DAILY_LOSS_PCT = min(0.10, max(0.0001, _as_float(os.getenv("RAPID_ML_MAX_DAILY_LOSS_PCT"), 0.05)))
    RAPID_ML_MAX_POSITION_PCT = min(1.0, max(0.0001, _as_float(os.getenv("RAPID_ML_MAX_POSITION_PCT"), 0.20)))
    RAPID_ML_MIN_EDGE_BPS = _as_float(os.getenv("RAPID_ML_MIN_EDGE_BPS"), 5.0)
    RAPID_ML_MIN_CONFIDENCE = min(1.0, max(0.0, _as_float(os.getenv("RAPID_ML_MIN_CONFIDENCE"), 0.55)))
    RAPID_ML_MIN_EDGE_AGREEMENT = max(1, _as_int(os.getenv("RAPID_ML_MIN_EDGE_AGREEMENT"), 2))
    RAPID_ML_COST_RESERVE_BPS = max(0.0, _as_float(os.getenv("RAPID_ML_COST_RESERVE_BPS"), 2.0))
    RAPID_ML_SLIPPAGE_BPS = max(0.0, _as_float(os.getenv("RAPID_ML_SLIPPAGE_BPS"), 4.0))
    RAPID_ML_UNKNOWN_SPREAD_BPS = max(0.0, _as_float(os.getenv("RAPID_ML_UNKNOWN_SPREAD_BPS"), 5.0))
    RAPID_ML_MAX_DIRECTIONAL_EXPOSURE_PCT = min(1.0, max(0.0001, _as_float(os.getenv("RAPID_ML_MAX_DIRECTIONAL_EXPOSURE_PCT"), 1.0)))
    RAPID_ML_FEE_BPS_BY_PROVIDER_JSON = os.getenv(
        "RAPID_ML_FEE_BPS_BY_PROVIDER_JSON",
        '{"hyperliquid":8,"kucoin":10}',
    ).strip()
    RAPID_ML_MIN_NOTIONAL_BUFFER_USD = max(0.0, _as_float(os.getenv("RAPID_ML_MIN_NOTIONAL_BUFFER_USD"), 0.50))
    RAPID_ML_MAX_PROVIDER_FAILURES = max(1, _as_int(os.getenv("RAPID_ML_MAX_PROVIDER_FAILURES"), 3))
    RAPID_ML_CIRCUIT_BREAKER_WINDOW_MINUTES = max(1.0, _as_float(os.getenv("RAPID_ML_CIRCUIT_BREAKER_WINDOW_MINUTES"), 5.0))
    ONE_H10_LIVE_ENABLED = _as_bool(os.getenv("ONE_H10_LIVE_ENABLED"), default=False)
    ONE_H10_EXACT_CONFIRMATION = os.getenv("ONE_H10_EXACT_CONFIRMATION", "ONE-H10-LIVE").strip() or "ONE-H10-LIVE"
    ONE_H10_TARGET_ROI_PCT = _as_float(os.getenv("ONE_H10_TARGET_ROI_PCT"), 1000.0)
    ONE_H10_HORIZON_SECONDS = max(60, _as_int(os.getenv("ONE_H10_HORIZON_SECONDS"), 3600))
    ONE_H10_POLL_SECONDS = max(1.0, _as_float(os.getenv("ONE_H10_POLL_SECONDS"), 1.0))
    ONE_H10_REBALANCE_SECONDS = max(5.0, _as_float(os.getenv("ONE_H10_REBALANCE_SECONDS"), 15.0))
    ONE_H10_AUTO_RESUME_ACTIVE_RUNS = _as_bool(os.getenv("ONE_H10_AUTO_RESUME_ACTIVE_RUNS"), default=True)
    ONE_H10_FEATURE_TIMEFRAMES = _parse_csv(os.getenv("ONE_H10_FEATURE_TIMEFRAMES"), ["5m", "15m", "1h", "4h", "1d"])
    ONE_H10_REQUIRED_FEATURE_TIMEFRAMES = _parse_csv(os.getenv("ONE_H10_REQUIRED_FEATURE_TIMEFRAMES"), ["15m", "1h", "4h"])
    ONE_H10_FEATURE_REFRESH_SECONDS = _as_float(os.getenv("ONE_H10_FEATURE_REFRESH_SECONDS"), 3600.0)
    ONE_H10_FEATURE_MAX_MARKETS_PER_SYNC = _as_int(os.getenv("ONE_H10_FEATURE_MAX_MARKETS_PER_SYNC"), 10)
    ONE_H10_FEATURE_RATE_LIMIT_BACKOFF_SECONDS = _as_float(os.getenv("ONE_H10_FEATURE_RATE_LIMIT_BACKOFF_SECONDS"), 300.0)
    ONE_H10_START_SYNC_FEATURES = _as_bool(os.getenv("ONE_H10_START_SYNC_FEATURES"), default=False)
    ONE_H10_MARKET_DATA_BACKOFF_SECONDS = _as_float(os.getenv("ONE_H10_MARKET_DATA_BACKOFF_SECONDS"), 30.0)
    ONE_H10_MARKET_DATA_CACHE_SECONDS = _as_float(os.getenv("ONE_H10_MARKET_DATA_CACHE_SECONDS"), 5.0)
    ONE_H10_ACCOUNT_REFRESH_SECONDS = _as_float(os.getenv("ONE_H10_ACCOUNT_REFRESH_SECONDS"), 20.0)
    ONE_H10_PROVIDER_BACKOFF_ESCALATION_COUNT = _as_int(os.getenv("ONE_H10_PROVIDER_BACKOFF_ESCALATION_COUNT"), 3)
    ONE_H10_PROVIDER_BACKOFF_WINDOW_SECONDS = _as_float(os.getenv("ONE_H10_PROVIDER_BACKOFF_WINDOW_SECONDS"), 120.0)
    ONE_H10_ALL_PAIRS_ENABLED = _as_bool(os.getenv("ONE_H10_ALL_PAIRS_ENABLED"), default=True)
    ONE_H10_ALLOW_BOOTSTRAP_WITHOUT_PROMOTED_ML = _as_bool(
        os.getenv("ONE_H10_ALLOW_BOOTSTRAP_WITHOUT_PROMOTED_ML"),
        default=False,
    )
    ONE_H10_REQUIRE_PROMOTED_ML = (
        _as_bool(os.getenv("ONE_H10_REQUIRE_PROMOTED_ML"), default=True) or not ONE_H10_ALLOW_BOOTSTRAP_WITHOUT_PROMOTED_ML
    )
    ONE_H10_BOOTSTRAP_LIVE_ENABLED = _as_bool(os.getenv("ONE_H10_BOOTSTRAP_LIVE_ENABLED"), default=True)
    ONE_H10_MAX_PROVIDER_LEGS = _as_int(os.getenv("ONE_H10_MAX_PROVIDER_LEGS"), 3)
    ONE_H10_MAX_LEVERAGE = _as_float(os.getenv("ONE_H10_MAX_LEVERAGE"), MAX_LEVERAGE)
    ONE_H10_MIN_LIQUIDITY_USD = max(
        _as_float(os.getenv("ONE_H10_MIN_LIQUIDITY_USD"), VAULT_MIN_LIQUIDITY_USD),
        _as_float(os.getenv("ONE_H10_MIN_LIQUIDITY_FLOOR_USD"), 50_000.0),
    )
    ONE_H10_MAX_SLIPPAGE_BPS = min(
        _as_float(os.getenv("ONE_H10_MAX_SLIPPAGE_BPS"), VAULT_MAX_SLIPPAGE_BPS),
        _as_float(os.getenv("ONE_H10_MAX_SLIPPAGE_CAP_BPS"), 20.0),
    )
    ONE_H10_MIN_EDGE_AFTER_COST_BPS = _as_float(os.getenv("ONE_H10_MIN_EDGE_AFTER_COST_BPS"), NET_ROI_MIN_EDGE_BPS)
    ONE_H10_MAX_COST_DRAG_BPS = _as_float(
        os.getenv("ONE_H10_MAX_COST_DRAG_BPS"),
        _as_float(os.getenv("AGGRESSIVE_1H_MAX_COST_DRAG_BPS"), 18.0),
    )
    ONE_H10_MIN_RISK_REWARD = _as_float(os.getenv("ONE_H10_MIN_RISK_REWARD"), MIN_REWARD_RISK)
    ONE_H10_MAX_SIGNAL_AGE_SECONDS = _as_float(
        os.getenv("ONE_H10_MAX_SIGNAL_AGE_SECONDS"),
        float(ONE_H10_HORIZON_SECONDS),
    )
    ONE_H10_MIN_EXECUTION_QUALITY = _as_float(
        os.getenv("ONE_H10_MIN_EXECUTION_QUALITY"),
        ONE_HOUR_MIN_EXECUTION_QUALITY,
    )
    ONE_H10_MIN_FORECAST_CONFIDENCE = max(_as_float(os.getenv("ONE_H10_MIN_FORECAST_CONFIDENCE"), 0.25), 0.25)
    ONE_H10_MIN_BOOTSTRAP_CONFIDENCE = max(_as_float(os.getenv("ONE_H10_MIN_BOOTSTRAP_CONFIDENCE"), 0.15), 0.15)
    ONE_H10_DIRECTIONAL_THRESHOLD = _as_float(os.getenv("ONE_H10_DIRECTIONAL_THRESHOLD"), 0.04)
    ONE_H10_MIN_POSITION_FRACTION = _as_float(os.getenv("ONE_H10_MIN_POSITION_FRACTION"), 0.20)
    ONE_H10_MAX_POSITION_FRACTION = _as_float(os.getenv("ONE_H10_MAX_POSITION_FRACTION"), 0.75)
    ONE_H10_PROFIT_OPTIMIZER_ENABLED = _as_bool(os.getenv("ONE_H10_PROFIT_OPTIMIZER_ENABLED"), default=True)
    ONE_H10_MIN_PROFITABILITY_SCORE = _as_float(os.getenv("ONE_H10_MIN_PROFITABILITY_SCORE"), 0.35)
    ONE_H10_FIBONACCI_MIN_CONFIDENCE = max(_as_float(os.getenv("ONE_H10_FIBONACCI_MIN_CONFIDENCE"), 0.15), 0.15)
    ONE_H10_REJECT_ZERO_SPREAD = _as_bool(os.getenv("ONE_H10_REJECT_ZERO_SPREAD"), default=True)
    ONE_H10_ML_FORECAST_FAMILIES = _parse_csv(
        os.getenv("ONE_H10_ML_FORECAST_FAMILIES"),
        [
            "pytorch_fibonacci",
            "pytorch_roi_target",
            "pytorch_extreme_upside",
            "pytorch_cap_policy",
            "pytorch_exit_policy",
            "pytorch_execution_policy",
            "pytorch_risk_policy",
            "pytorch_optimizer_policy",
        ],
    )
    ONE_H10_ML_PROFIT_WEIGHT = _as_float(os.getenv("ONE_H10_ML_PROFIT_WEIGHT"), 0.65)
    ONE_H10_ML_EXPECTED_EDGE_CAP_BPS = _as_float(os.getenv("ONE_H10_ML_EXPECTED_EDGE_CAP_BPS"), 400.0)
    ONE_H10_MIN_MODEL_AGREEMENT = _as_float(os.getenv("ONE_H10_MIN_MODEL_AGREEMENT"), 0.55)
    ONE_H10_MODEL_AGREEMENT_WEIGHT = _as_float(os.getenv("ONE_H10_MODEL_AGREEMENT_WEIGHT"), 0.35)
    ONE_H10_MAX_TAKE_PROFIT_PCT = _as_float(os.getenv("ONE_H10_MAX_TAKE_PROFIT_PCT"), 0.35)
    ONE_H10_MAX_STOP_LOSS_PCT = _as_float(os.getenv("ONE_H10_MAX_STOP_LOSS_PCT"), 0.08)
    ONE_H10_BOOTSTRAP_FEATURE_BLOCKERS_ADVISORY = _as_bool(
        os.getenv("ONE_H10_BOOTSTRAP_FEATURE_BLOCKERS_ADVISORY"),
        default=True,
    )
    ONE_H10_ML_ADVISORY_BLOCKERS = _parse_csv(
        os.getenv("ONE_H10_ML_ADVISORY_BLOCKERS"),
        [
            "ml_fibonacci_confidence_below_minimum",
            "forecast_hold",
            "low_confidence",
            "ml_not_ready",
            "features_stale",
            "missing_fibonacci_features",
        ],
    )
    AGGRESSIVE_1H_ENABLED = _as_bool(os.getenv("AGGRESSIVE_1H_ENABLED"), default=False)
    ALLOW_AGGRESSIVE_LIVE_TRADING = _as_bool(os.getenv("ALLOW_AGGRESSIVE_LIVE_TRADING"), default=False)
    AGGRESSIVE_1H_MIN_TRADES = _as_int(os.getenv("AGGRESSIVE_1H_MIN_TRADES"), 8)
    AGGRESSIVE_1H_MAX_DRAWDOWN_PCT = _as_float(os.getenv("AGGRESSIVE_1H_MAX_DRAWDOWN_PCT"), 0.35)
    AGGRESSIVE_1H_MAX_PARAMETER_SETS = _as_int(os.getenv("AGGRESSIVE_1H_MAX_PARAMETER_SETS"), 25)
    AGGRESSIVE_1H_RISK_PER_TRADE_PCT = _as_float(os.getenv("AGGRESSIVE_1H_RISK_PER_TRADE_PCT"), 0.005)
    AGGRESSIVE_1H_POSITION_SIZE_FRACTION = _as_float(os.getenv("AGGRESSIVE_1H_POSITION_SIZE_FRACTION"), 0.12)
    AGGRESSIVE_1H_TIMEFRAMES = _parse_csv(os.getenv("AGGRESSIVE_1H_TIMEFRAMES"), ["1m", "5m"])
    AGGRESSIVE_1H_LIVE_CAP_USDC = _as_float(os.getenv("AGGRESSIVE_1H_LIVE_CAP_USDC"), 25.0)
    AGGRESSIVE_1H_LIVE_CAP_PCT = _as_float(os.getenv("AGGRESSIVE_1H_LIVE_CAP_PCT"), 0.02)
    AGGRESSIVE_MIN_EDGE_BPS = _as_float(os.getenv("AGGRESSIVE_MIN_EDGE_BPS"), 4.0)
    AGGRESSIVE_1H_MAX_COST_DRAG_BPS = _as_float(os.getenv("AGGRESSIVE_1H_MAX_COST_DRAG_BPS"), 18.0)
    AGGRESSIVE_1H_MIN_CAPACITY_MULTIPLE = _as_float(os.getenv("AGGRESSIVE_1H_MIN_CAPACITY_MULTIPLE"), 2.0)
    AGGRESSIVE_1H_MIN_MFE_MAE = _as_float(os.getenv("AGGRESSIVE_1H_MIN_MFE_MAE"), 1.5)
    AGGRESSIVE_1H_RECENCY_HALF_LIFE_HOURS = _as_float(os.getenv("AGGRESSIVE_1H_RECENCY_HALF_LIFE_HOURS"), 12.0)
    AGGRESSIVE_1H_MIN_WINDOW_STABILITY = _as_float(os.getenv("AGGRESSIVE_1H_MIN_WINDOW_STABILITY"), 0.55)
    EXTREME_ROI_ENABLED = _as_bool(os.getenv("EXTREME_ROI_ENABLED"), default=False)
    EXTREME_ROI_MIN_TRADES = _as_int(os.getenv("EXTREME_ROI_MIN_TRADES"), 10)
    EXTREME_ROI_MAX_DRAWDOWN_PCT = _as_float(os.getenv("EXTREME_ROI_MAX_DRAWDOWN_PCT"), 0.45)
    EXTREME_ROI_MAX_PARAMETER_SETS = _as_int(os.getenv("EXTREME_ROI_MAX_PARAMETER_SETS"), 40)
    EXTREME_ROI_RISK_PER_TRADE_PCT = _as_float(os.getenv("EXTREME_ROI_RISK_PER_TRADE_PCT"), 0.006)
    EXTREME_ROI_POSITION_SIZE_FRACTION = _as_float(os.getenv("EXTREME_ROI_POSITION_SIZE_FRACTION"), 0.16)
    EXTREME_ROI_TIMEFRAMES = _parse_csv(os.getenv("EXTREME_ROI_TIMEFRAMES"), ["1m", "5m"])
    EXTREME_ROI_MIN_EDGE_BPS = _as_float(os.getenv("EXTREME_ROI_MIN_EDGE_BPS"), 8.0)
    EXTREME_ROI_MIN_CONFIDENCE = _as_float(os.getenv("EXTREME_ROI_MIN_CONFIDENCE"), 0.62)
    EXTREME_ROI_MIN_LIQUIDITY_USD = _as_float(os.getenv("EXTREME_ROI_MIN_LIQUIDITY_USD"), 1_000.0)
    EXTREME_ROI_MIN_SIGNAL_STABILITY = _as_float(os.getenv("EXTREME_ROI_MIN_SIGNAL_STABILITY"), 0.55)
    DYNAMIC_UNIVERSE_ENABLED = _as_bool(os.getenv("DYNAMIC_UNIVERSE_ENABLED"), default=False)
    UNIVERSE_MAX_SYMBOLS = _as_int(os.getenv("UNIVERSE_MAX_SYMBOLS"), 20)
    UNIVERSE_MIN_LIQUIDITY_USD = _as_float(os.getenv("UNIVERSE_MIN_LIQUIDITY_USD"), 25_000.0)
    UNIVERSE_MAX_SPREAD_BPS = _as_float(os.getenv("UNIVERSE_MAX_SPREAD_BPS"), 15.0)
    UNIVERSE_REFRESH_SECONDS = _as_int(os.getenv("UNIVERSE_REFRESH_SECONDS"), 300)
    UNIVERSE_SYMBOL_BLACKLIST = _parse_upper_csv(os.getenv("UNIVERSE_SYMBOL_BLACKLIST"), [])
    HOT_TOKEN_SCAN_ENABLED = _as_bool(os.getenv("HOT_TOKEN_SCAN_ENABLED"), default=True)
    HOT_TOKEN_REFRESH_SECONDS = _as_int(os.getenv("HOT_TOKEN_REFRESH_SECONDS"), 180)
    HOT_TOKEN_MAX_CANDIDATES = _as_int(os.getenv("HOT_TOKEN_MAX_CANDIDATES"), 8)
    HOT_TOKEN_VOLUME_SPIKE_RATIO = _as_float(os.getenv("HOT_TOKEN_VOLUME_SPIKE_RATIO"), 1.8)
    HOT_TOKEN_MIN_VOLATILITY_PCT = _as_float(os.getenv("HOT_TOKEN_MIN_VOLATILITY_PCT"), 0.20)
    LEVERAGE_OPTIMIZER_ENABLED = _as_bool(os.getenv("LEVERAGE_OPTIMIZER_ENABLED"), default=False)
    AGGRESSIVE_MAX_TEST_LEVERAGE = _as_float(os.getenv("AGGRESSIVE_MAX_TEST_LEVERAGE"), 5.0)
    AGGRESSIVE_MAX_LIVE_LEVERAGE = _as_float(os.getenv("AGGRESSIVE_MAX_LIVE_LEVERAGE"), 3.0)
    MIN_LIQUIDATION_BUFFER_PCT = _as_float(os.getenv("MIN_LIQUIDATION_BUFFER_PCT"), 0.015)
    VAULT_MAX_PARALLEL_LEGS = _as_int(os.getenv("VAULT_MAX_PARALLEL_LEGS"), 3)
    VAULT_MIN_LEG_USD = _as_float(os.getenv("VAULT_MIN_LEG_USD"), 10.0)
    VAULT_MAX_SYMBOL_ALLOCATION_PCT = _as_float(os.getenv("VAULT_MAX_SYMBOL_ALLOCATION_PCT"), 0.50)
    VAULT_STRATEGY_BASKET_ENABLED = _as_bool(os.getenv("VAULT_STRATEGY_BASKET_ENABLED"), default=True)
    ENSEMBLE_ENHANCED_ENABLED = _as_bool(os.getenv("ENSEMBLE_ENHANCED_ENABLED"), default=False)
    ENSEMBLE_MAX_LEGS = _as_int(os.getenv("ENSEMBLE_MAX_LEGS", os.getenv("ENSEMBLE_1H_MAX_LEGS")), 5)
    ENSEMBLE_MIN_SHARPE = _as_float(os.getenv("ENSEMBLE_MIN_SHARPE"), 0.5)
    ENSEMBLE_MIN_EDGE_BPS = _as_float(os.getenv("ENSEMBLE_MIN_EDGE_BPS", os.getenv("ENSEMBLE_1H_MIN_EDGE_BPS")), AGGRESSIVE_MIN_EDGE_BPS)
    ENSEMBLE_LEARNING_DECAY = _as_float(os.getenv("ENSEMBLE_LEARNING_DECAY"), 0.8)
    ENSEMBLE_MAX_SYMBOL_PCT = _as_float(os.getenv("ENSEMBLE_MAX_SYMBOL_PCT", os.getenv("ENSEMBLE_1H_MAX_SYMBOL_PCT")), 0.50)
    ENSEMBLE_MAX_STRATEGY_PCT = _as_float(os.getenv("ENSEMBLE_MAX_STRATEGY_PCT", os.getenv("ENSEMBLE_1H_MAX_STRATEGY_PCT")), 0.60)
    FIB_MULTI_LOOKBACKS = _parse_int_csv(os.getenv("FIB_MULTI_LOOKBACKS", os.getenv("FIB_CONFLUENCE_LOOKBACKS")), [20, 50, 100])
    FIB_CONFLUENCE_THRESHOLD = _as_float(os.getenv("FIB_CONFLUENCE_THRESHOLD", os.getenv("ENSEMBLE_1H_MIN_CONFLUENCE")), 0.55)
    FIB_MULTI_TIMEFRAMES = _parse_csv(os.getenv("FIB_MULTI_TIMEFRAMES"), ["1h", "4h", "1d"])
    ENSEMBLE_1H_ENABLED = _as_bool(os.getenv("ENSEMBLE_1H_ENABLED"), default=False)
    ENSEMBLE_1H_MAX_LEGS = _as_int(os.getenv("ENSEMBLE_1H_MAX_LEGS"), 3)
    ENSEMBLE_1H_MIN_EDGE_BPS = _as_float(os.getenv("ENSEMBLE_1H_MIN_EDGE_BPS"), AGGRESSIVE_MIN_EDGE_BPS)
    ENSEMBLE_1H_MAX_SYMBOL_PCT = _as_float(os.getenv("ENSEMBLE_1H_MAX_SYMBOL_PCT"), 0.50)
    ENSEMBLE_1H_MAX_STRATEGY_PCT = _as_float(os.getenv("ENSEMBLE_1H_MAX_STRATEGY_PCT"), 0.60)
    ENSEMBLE_1H_MIN_CONFLUENCE = _as_float(os.getenv("ENSEMBLE_1H_MIN_CONFLUENCE"), 0.0)
    ENSEMBLE_1H_REQUIRE_SHADOW_VALIDATION = _as_bool(
        os.getenv("ENSEMBLE_1H_REQUIRE_SHADOW_VALIDATION"),
        default=True,
    )
    ENSEMBLE_1H_SHADOW_VALIDATION_MAX_AGE_HOURS = _as_float(
        os.getenv("ENSEMBLE_1H_SHADOW_VALIDATION_MAX_AGE_HOURS"),
        24.0,
    )
    ENSEMBLE_LEARNING_ENABLED = _as_bool(os.getenv("ENSEMBLE_LEARNING_ENABLED"), default=True)
    EXPERIMENTAL_DURATION_ENSEMBLE_ENABLED = _as_bool(os.getenv("EXPERIMENTAL_DURATION_ENSEMBLE_ENABLED"), default=False)
    EXPERIMENTAL_DURATION_ENSEMBLE_LIVE_ELIGIBLE = _as_bool(os.getenv("EXPERIMENTAL_DURATION_ENSEMBLE_LIVE_ELIGIBLE"), default=False)
    EXPERIMENTAL_DURATION_ENSEMBLE_MAX_LEGS = _as_int(os.getenv("EXPERIMENTAL_DURATION_ENSEMBLE_MAX_LEGS"), 5)
    EXPERIMENTAL_DURATION_ENSEMBLE_PRIMARY_METRIC = (
        os.getenv(
            "EXPERIMENTAL_DURATION_ENSEMBLE_PRIMARY_METRIC",
            "net_return_after_costs",
        ).strip()
        or "net_return_after_costs"
    )
    EXPERIMENTAL_LIVE_CAP_USDC_BY_DURATION = _parse_float_map(os.getenv("EXPERIMENTAL_LIVE_CAP_USDC_BY_DURATION"))
    EXPERIMENTAL_LIVE_CAP_PCT_BY_DURATION = _parse_float_map(os.getenv("EXPERIMENTAL_LIVE_CAP_PCT_BY_DURATION"))
    MAX_RETURN_OPTIMIZER_ENABLED = _as_bool(os.getenv("MAX_RETURN_OPTIMIZER_ENABLED"), default=False)
    MAX_RETURN_LIVE_ELIGIBLE = _as_bool(os.getenv("MAX_RETURN_LIVE_ELIGIBLE"), default=False)
    MAX_RETURN_MIN_NET_RETURN = _as_float(os.getenv("MAX_RETURN_MIN_NET_RETURN"), 0.0)
    MAX_RETURN_MAX_DRAWDOWN_PCT = _as_float(os.getenv("MAX_RETURN_MAX_DRAWDOWN_PCT"), 0.0)
    MARKET_STRUCTURE_FEATURES_ENABLED = _as_bool(os.getenv("MARKET_STRUCTURE_FEATURES_ENABLED"), default=False)
    MARKET_STRUCTURE_PROVIDER = os.getenv("MARKET_STRUCTURE_PROVIDER", "existing").strip() or "existing"
    MARKET_STRUCTURE_FAIL_CLOSED = _as_bool(os.getenv("MARKET_STRUCTURE_FAIL_CLOSED"), default=True)
    PAIR_SCREENING_ENABLED = _as_bool(os.getenv("PAIR_SCREENING_ENABLED"), default=False)
    PAIR_TRADING_ENABLED = _as_bool(os.getenv("PAIR_TRADING_ENABLED"), default=False)
    PAIR_LIVE_ELIGIBLE = _as_bool(os.getenv("PAIR_LIVE_ELIGIBLE"), default=False)
    PAIR_REQUIRE_SHADOW_VALIDATION = _as_bool(os.getenv("PAIR_REQUIRE_SHADOW_VALIDATION"), default=True)
    PAIR_MIN_CORRELATION = _as_float(os.getenv("PAIR_MIN_CORRELATION"), 0.75)
    PAIR_MAX_SPREAD_ZSCORE = _as_float(os.getenv("PAIR_MAX_SPREAD_ZSCORE"), 2.5)
    PAIR_MIN_LIQUIDITY_USD = _as_float(os.getenv("PAIR_MIN_LIQUIDITY_USD"), 25_000.0)
    PAIR_MAX_SPREAD_BPS = _as_float(os.getenv("PAIR_MAX_SPREAD_BPS"), 20.0)
    PAIR_MAX_LEGS_PER_CYCLE = _as_int(os.getenv("PAIR_MAX_LEGS_PER_CYCLE"), 2)
    FIB_CONFLUENCE_LOOKBACKS = _parse_int_csv(os.getenv("FIB_CONFLUENCE_LOOKBACKS"), [20, 50, 100])
    FIB_CONFLUENCE_CLUSTER_TOLERANCE_BPS = _as_float(os.getenv("FIB_CONFLUENCE_CLUSTER_TOLERANCE_BPS"), 18.0)
    AGGRESSIVE_1H_POLL_SECONDS = _as_float(os.getenv("AGGRESSIVE_1H_POLL_SECONDS"), 10.0)
    LIVE_CONNECTION_FAILURE_BACKOFF_SECONDS = _as_float(os.getenv("LIVE_CONNECTION_FAILURE_BACKOFF_SECONDS"), 60.0)
    VAULT_MAX_ACTIVE_CYCLES = _as_int(os.getenv("VAULT_MAX_ACTIVE_CYCLES"), 6)
    VAULT_START_ASYNC_ENABLED = _as_bool(os.getenv("VAULT_START_ASYNC_ENABLED"), default=False)
    VAULT_MAX_ACTIVE_CYCLES_PER_ASSET = _as_int(os.getenv("VAULT_MAX_ACTIVE_CYCLES_PER_ASSET"), 4)
    VAULT_MAX_ASSET_EXPOSURE_PCT = _as_float(os.getenv("VAULT_MAX_ASSET_EXPOSURE_PCT"), 0.75)
    VAULT_MAX_DURATION_EXPOSURE_PCT = _as_float(os.getenv("VAULT_MAX_DURATION_EXPOSURE_PCT"), 0.70)
    VAULT_MAX_STRATEGY_EXPOSURE_PCT = _as_float(os.getenv("VAULT_MAX_STRATEGY_EXPOSURE_PCT"), 0.70)
    VAULT_MAX_SYMBOL_EXPOSURE_PCT = _as_float(os.getenv("VAULT_MAX_SYMBOL_EXPOSURE_PCT"), 0.70)
    VAULT_MIN_AVAILABLE_RESERVE_USD = _as_float(os.getenv("VAULT_MIN_AVAILABLE_RESERVE_USD"), 5.0)
    VAULT_MIN_RISK_ADJUSTED_SCORE = _as_float(os.getenv("VAULT_MIN_RISK_ADJUSTED_SCORE"), 0.0)
    VAULT_CYCLE_ENGINE_ENABLED = _as_bool(os.getenv("VAULT_CYCLE_ENGINE_ENABLED"), default=False)
    VAULT_CYCLE_REAL_TRANSFERS_ENABLED = _as_bool(os.getenv("VAULT_CYCLE_REAL_TRANSFERS_ENABLED"), default=False)
    VAULT_CYCLE_AUTOMATIC_WITHDRAWALS_ENABLED = _as_bool(
        os.getenv("VAULT_CYCLE_AUTOMATIC_WITHDRAWALS_ENABLED"),
        default=False,
    )
    VAULT_CYCLE_PROVIDERS = _parse_csv(os.getenv("VAULT_CYCLE_PROVIDERS"), ["hyperliquid", "kucoin"])
    VAULT_CYCLE_ALLOWED_WITHDRAWAL_ADDRESSES = _parse_deposit_address_book(os.getenv("VAULT_CYCLE_ALLOWED_WITHDRAWAL_ADDRESSES_JSON"))
    VAULT_CYCLE_MAX_TRANSFER_USD = _as_float(os.getenv("VAULT_CYCLE_MAX_TRANSFER_USD"), 100.0)
    VAULT_CYCLE_DAILY_WITHDRAWAL_CAP_USD = _as_float(os.getenv("VAULT_CYCLE_DAILY_WITHDRAWAL_CAP_USD"), 100.0)
    VAULT_CYCLE_MAX_EXCHANGE_ALLOCATION_PCT = _as_float(os.getenv("VAULT_CYCLE_MAX_EXCHANGE_ALLOCATION_PCT"), 0.80)
    VAULT_CYCLE_MAX_SYMBOL_ALLOCATION_PCT = _as_float(os.getenv("VAULT_CYCLE_MAX_SYMBOL_ALLOCATION_PCT"), VAULT_MAX_SYMBOL_ALLOCATION_PCT)
    VAULT_CYCLE_MAX_CONCURRENT_POSITIONS = _as_int(os.getenv("VAULT_CYCLE_MAX_CONCURRENT_POSITIONS"), 3)
    VAULT_CYCLE_MIN_EXCHANGE_ALLOCATION_USD = _as_float(os.getenv("VAULT_CYCLE_MIN_EXCHANGE_ALLOCATION_USD"), 5.0)
    VAULT_CYCLE_TRANSFER_CONFIRMATION_TIMEOUT_SECONDS = _as_float(
        os.getenv("VAULT_CYCLE_TRANSFER_CONFIRMATION_TIMEOUT_SECONDS"),
        900.0,
    )
    VAULT_CYCLE_CONVERSION_ENABLED = _as_bool(os.getenv("VAULT_CYCLE_CONVERSION_ENABLED"), default=False)
    VAULT_CYCLE_CONVERSION_PROVIDER = os.getenv("VAULT_CYCLE_CONVERSION_PROVIDER", "kucoin").strip().lower() or "kucoin"
    VAULT_CYCLE_STABLECOIN_MAX_SLIPPAGE_BPS = _as_float(os.getenv("VAULT_CYCLE_STABLECOIN_MAX_SLIPPAGE_BPS"), 10.0)
    VAULT_CYCLE_REQUIRE_EXCHANGE_RESERVE = _as_bool(os.getenv("VAULT_CYCLE_REQUIRE_EXCHANGE_RESERVE"), default=True)
    VAULT_CYCLE_DEFAULT_NETWORK_BY_ASSET = _parse_json_object(os.getenv("VAULT_CYCLE_DEFAULT_NETWORK_BY_ASSET_JSON"))
    VAULT_CYCLE_ACTIVITY_ENFORCEMENT_ENABLED = _as_bool(os.getenv("VAULT_CYCLE_ACTIVITY_ENFORCEMENT_ENABLED"), default=True)
    VAULT_CYCLE_MAX_IDLE_SECONDS = _as_float(os.getenv("VAULT_CYCLE_MAX_IDLE_SECONDS"), 300.0)
    VAULT_CYCLE_RESCREEN_SECONDS = _as_float(os.getenv("VAULT_CYCLE_RESCREEN_SECONDS"), 180.0)
    VAULT_CYCLE_MIN_TRADES_PER_CYCLE = _as_int(os.getenv("VAULT_CYCLE_MIN_TRADES_PER_CYCLE"), 1)
    VAULT_CYCLE_TARGET_UTILIZATION_PCT = _as_float(os.getenv("VAULT_CYCLE_TARGET_UTILIZATION_PCT"), 0.60)
    VAULT_CYCLE_MIN_OPPORTUNITY_SCORE = _as_float(os.getenv("VAULT_CYCLE_MIN_OPPORTUNITY_SCORE"), 0.60)
    VAULT_CYCLE_ROTATION_SCORE_DELTA = _as_float(os.getenv("VAULT_CYCLE_ROTATION_SCORE_DELTA"), 0.10)
    VAULT_CYCLE_EXCHANGE_REBALANCE_ENABLED = _as_bool(os.getenv("VAULT_CYCLE_EXCHANGE_REBALANCE_ENABLED"), default=False)
    AGGRESSIVE_RISK_ADJUSTED_MIN_TRADES = _as_int(os.getenv("AGGRESSIVE_RISK_ADJUSTED_MIN_TRADES"), 6)
    AGGRESSIVE_RISK_ADJUSTED_MAX_DRAWDOWN_PCT = _as_float(
        os.getenv("AGGRESSIVE_RISK_ADJUSTED_MAX_DRAWDOWN_PCT"),
        0.25,
    )
    AGGRESSIVE_RISK_ADJUSTED_MAX_PARAMETER_SETS = _as_int(
        os.getenv("AGGRESSIVE_RISK_ADJUSTED_MAX_PARAMETER_SETS"),
        30,
    )
    DYNAMIC_INTRADAY_TIMEFRAMES = _parse_csv(os.getenv("DYNAMIC_INTRADAY_TIMEFRAMES"), ["1m", "5m", "15m"])
    DYNAMIC_INTRADAY_MIN_TRADES = _as_int(os.getenv("DYNAMIC_INTRADAY_MIN_TRADES"), 8)
    DYNAMIC_INTRADAY_MAX_DRAWDOWN_PCT = _as_float(os.getenv("DYNAMIC_INTRADAY_MAX_DRAWDOWN_PCT"), 0.25)
    DYNAMIC_INTRADAY_MAX_PARAMETER_SETS = _as_int(os.getenv("DYNAMIC_INTRADAY_MAX_PARAMETER_SETS"), 30)
    DYNAMIC_INTRADAY_MIN_EDGE_BPS = _as_float(os.getenv("DYNAMIC_INTRADAY_MIN_EDGE_BPS"), 6.0)
    DYNAMIC_INTRADAY_MIN_LIQUIDITY_USD = _as_float(os.getenv("DYNAMIC_INTRADAY_MIN_LIQUIDITY_USD"), 50_000.0)
    DYNAMIC_INTRADAY_MAX_SPREAD_BPS = _as_float(os.getenv("DYNAMIC_INTRADAY_MAX_SPREAD_BPS"), 12.0)
    DYNAMIC_INTRADAY_REQUIRE_SHADOW_VALIDATION = _as_bool(
        os.getenv("DYNAMIC_INTRADAY_REQUIRE_SHADOW_VALIDATION"),
        default=True,
    )
    DYNAMIC_INTRADAY_LIVE_ELIGIBLE = _as_bool(os.getenv("DYNAMIC_INTRADAY_LIVE_ELIGIBLE"), default=False)
    HIGH_UPSIDE_PROFILE_ENABLED = _as_bool(os.getenv("HIGH_UPSIDE_PROFILE_ENABLED"), default=False)
    HIGH_UPSIDE_LIVE_ELIGIBLE = _as_bool(os.getenv("HIGH_UPSIDE_LIVE_ELIGIBLE"), default=False)
    HIGH_UPSIDE_AUTO_LIVE_ENABLED = _as_bool(os.getenv("HIGH_UPSIDE_AUTO_LIVE_ENABLED"), default=False)
    HIGH_UPSIDE_DISCOVERY_ENABLED = _as_bool(os.getenv("HIGH_UPSIDE_DISCOVERY_ENABLED"), default=False)
    HIGH_UPSIDE_PROVIDER_SCOPE = os.getenv("HIGH_UPSIDE_PROVIDER_SCOPE", "multi_provider").strip().lower() or "multi_provider"
    HIGH_UPSIDE_MAX_SYMBOLS = _as_int(os.getenv("HIGH_UPSIDE_MAX_SYMBOLS"), 12)
    HIGH_UPSIDE_TIMEFRAMES = _parse_csv(os.getenv("HIGH_UPSIDE_TIMEFRAMES"), ["5m", "15m", "1h"])
    HIGH_UPSIDE_MAX_SWEEPS = _as_int(os.getenv("HIGH_UPSIDE_MAX_SWEEPS"), 8)
    HIGH_UPSIDE_COMMAND_TIMEOUT_SECONDS = _as_float(os.getenv("HIGH_UPSIDE_COMMAND_TIMEOUT_SECONDS"), 300.0)
    HIGH_UPSIDE_STAGE1_MAX_MIDS = _as_int(os.getenv("HIGH_UPSIDE_STAGE1_MAX_MIDS"), 80)
    HIGH_UPSIDE_ALL_PAIRS_ENABLED = _as_bool(os.getenv("HIGH_UPSIDE_ALL_PAIRS_ENABLED"), default=True)
    HIGH_UPSIDE_STAGE2_CANDIDATE_MULTIPLIER = _as_int(os.getenv("HIGH_UPSIDE_STAGE2_CANDIDATE_MULTIPLIER"), 3)
    HIGH_UPSIDE_BOUNDED_SCANNER_UNIVERSE = _as_bool(os.getenv("HIGH_UPSIDE_BOUNDED_SCANNER_UNIVERSE"), default=True)
    HIGH_UPSIDE_MARKET_DATA_RATE_LIMIT_PER_SECOND = _as_float(os.getenv("HIGH_UPSIDE_MARKET_DATA_RATE_LIMIT_PER_SECOND"), 4.0)
    HIGH_UPSIDE_RATE_LIMIT_BACKOFF_SECONDS = _as_float(os.getenv("HIGH_UPSIDE_RATE_LIMIT_BACKOFF_SECONDS"), 300.0)
    HIGH_UPSIDE_ADAPTIVE_CADENCE_ENABLED = _as_bool(os.getenv("HIGH_UPSIDE_ADAPTIVE_CADENCE_ENABLED"), default=False)
    HIGH_UPSIDE_MIN_SCAN_INTERVAL_SECONDS = _as_float(os.getenv("HIGH_UPSIDE_MIN_SCAN_INTERVAL_SECONDS"), 3600.0)
    HIGH_UPSIDE_FAST_SCAN_INTERVAL_SECONDS = _as_float(os.getenv("HIGH_UPSIDE_FAST_SCAN_INTERVAL_SECONDS"), 300.0)
    HIGH_UPSIDE_LOSS_COOLDOWN_SECONDS = _as_float(os.getenv("HIGH_UPSIDE_LOSS_COOLDOWN_SECONDS"), 3600.0)
    HIGH_UPSIDE_REJECTION_COOLDOWN_SECONDS = _as_float(os.getenv("HIGH_UPSIDE_REJECTION_COOLDOWN_SECONDS"), 900.0)
    HIGH_UPSIDE_MAX_LIVE_CYCLES_PER_DAY = _as_int(os.getenv("HIGH_UPSIDE_MAX_LIVE_CYCLES_PER_DAY"), 1)
    HIGH_UPSIDE_MAX_ACTIVE_CYCLES = _as_int(os.getenv("HIGH_UPSIDE_MAX_ACTIVE_CYCLES"), 1)
    HIGH_UPSIDE_MAX_DAILY_LOSS_USDC = _as_float(os.getenv("HIGH_UPSIDE_MAX_DAILY_LOSS_USDC"), 0.0)
    HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION = _parse_float_map(os.getenv("HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION_JSON"))
    HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION = _parse_float_map(os.getenv("HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION_JSON"))
    HIGH_UPSIDE_MAX_SCANNER_REJECTION_RATE = _as_float(os.getenv("HIGH_UPSIDE_MAX_SCANNER_REJECTION_RATE"), 0.65)
    ML_RANKER_ENABLED = _as_bool(os.getenv("ML_RANKER_ENABLED"), default=False)
    ML_SCORE_WEIGHT = _as_float(os.getenv("ML_SCORE_WEIGHT"), 0.15)
    ML_RANKER_LEARNING_RATE = _as_float(os.getenv("ML_RANKER_LEARNING_RATE"), 0.03)
    ML_RANKER_L2 = _as_float(os.getenv("ML_RANKER_L2"), 0.001)
    ML_MIN_TRAINING_EVENTS = _as_int(os.getenv("ML_MIN_TRAINING_EVENTS"), 25)
    ML_ALLOW_LIVE_UPDATES = _as_bool(os.getenv("ML_ALLOW_LIVE_UPDATES"), default=False)
    ML_LIVE_LEARNING_POLICY = os.getenv("ML_LIVE_LEARNING_POLICY", "quarantine").strip().lower()
    ML_LIVE_PROMOTION_MIN_EVENTS = _as_int(os.getenv("ML_LIVE_PROMOTION_MIN_EVENTS"), 25)
    ML_LIVE_PROMOTION_MAX_MEAN_LOSS = _as_float(os.getenv("ML_LIVE_PROMOTION_MAX_MEAN_LOSS"), 0.20)
    ML_LIVE_PROMOTION_MAX_NEGATIVE_ERROR_RATE = _as_float(os.getenv("ML_LIVE_PROMOTION_MAX_NEGATIVE_ERROR_RATE"), 0.55)
    ML_OFFLINE_MODELS_ENABLED = _as_bool(os.getenv("ML_OFFLINE_MODELS_ENABLED"), default=False)
    ML_OFFLINE_BLEND_ENABLED = _as_bool(os.getenv("ML_OFFLINE_BLEND_ENABLED"), default=False)
    ML_OFFLINE_MODEL_TYPES = _parse_csv(os.getenv("ML_OFFLINE_MODEL_TYPES"), ["sklearn", "xgboost"])
    ML_OFFLINE_SAFE_SCORING_MODEL_TYPES = _parse_csv(os.getenv("ML_OFFLINE_SAFE_SCORING_MODEL_TYPES"), ["sklearn"])
    ML_OFFLINE_SCORE_WEIGHT = _as_float(os.getenv("ML_OFFLINE_SCORE_WEIGHT"), 0.15)
    ML_OFFLINE_MIN_TRAINING_ROWS = _as_int(os.getenv("ML_OFFLINE_MIN_TRAINING_ROWS"), 250)
    ML_OFFLINE_MARKET_HISTORY_MAX_ROWS = _as_int(os.getenv("ML_OFFLINE_MARKET_HISTORY_MAX_ROWS"), 50_000)
    ML_OFFLINE_MARKET_HISTORY_SAMPLES_PER_WINDOW = _as_int(os.getenv("ML_OFFLINE_MARKET_HISTORY_SAMPLES_PER_WINDOW"), 250)
    ML_OFFLINE_MAX_VALIDATION_LOSS = _as_float(os.getenv("ML_OFFLINE_MAX_VALIDATION_LOSS"), 0.20)
    ML_OFFLINE_MAX_NEGATIVE_ERROR_RATE = _as_float(os.getenv("ML_OFFLINE_MAX_NEGATIVE_ERROR_RATE"), 0.55)
    ML_OFFLINE_MAX_MODEL_AGE_HOURS = _as_float(os.getenv("ML_OFFLINE_MAX_MODEL_AGE_HOURS"), 72.0)
    ML_OFFLINE_MAX_DRIFT = _as_float(os.getenv("ML_OFFLINE_MAX_DRIFT"), 0.35)
    ML_OFFLINE_MIN_TOP_DECILE_PRECISION = _as_float(os.getenv("ML_OFFLINE_MIN_TOP_DECILE_PRECISION"), 0.55)
    ML_OFFLINE_MAX_FALSE_POSITIVE_HIGH_UPSIDE_RATE = _as_float(
        os.getenv("ML_OFFLINE_MAX_FALSE_POSITIVE_HIGH_UPSIDE_RATE"),
        0.35,
    )
    ML_OFFLINE_MAX_CALIBRATION_ERROR = _as_float(os.getenv("ML_OFFLINE_MAX_CALIBRATION_ERROR"), 0.18)
    ML_ALL_AREAS_ENABLED = _as_bool(os.getenv("ML_ALL_AREAS_ENABLED"), default=False)
    ML_REQUIRE_PROMOTED_FOR_LIVE = _as_bool(os.getenv("ML_REQUIRE_PROMOTED_FOR_LIVE"), default=True)
    ML_ALLOW_STRATEGY_SIGNAL_OVERRIDE = _as_bool(os.getenv("ML_ALLOW_STRATEGY_SIGNAL_OVERRIDE"), default=True)
    ML_ALLOW_ALLOCATION_OVERRIDE = _as_bool(os.getenv("ML_ALLOW_ALLOCATION_OVERRIDE"), default=True)
    ML_ALLOW_EXECUTION_STYLE_SUGGESTIONS = _as_bool(os.getenv("ML_ALLOW_EXECUTION_STYLE_SUGGESTIONS"), default=True)
    ML_OPS_ANOMALY_ENABLED = _as_bool(os.getenv("ML_OPS_ANOMALY_ENABLED"), default=False)
    ML_DETERMINISTIC_SAFETY_GATES_REQUIRED = _as_bool(os.getenv("ML_DETERMINISTIC_SAFETY_GATES_REQUIRED"), default=True)
    ML_FIRST_STRATEGIES_ENABLED = _as_bool(os.getenv("ML_FIRST_STRATEGIES_ENABLED"), default=False)
    ML_FIBONACCI_MODEL_ENABLED = _as_bool(os.getenv("ML_FIBONACCI_MODEL_ENABLED"), default=False)
    ML_BACKTEST_SCORER_ENABLED = _as_bool(os.getenv("ML_BACKTEST_SCORER_ENABLED"), default=False)
    ML_OPTIMIZER_POLICY_ENABLED = _as_bool(os.getenv("ML_OPTIMIZER_POLICY_ENABLED"), default=False)
    ML_AUTO_LIVE_AFTER_PROMOTION_ENABLED = _as_bool(os.getenv("ML_AUTO_LIVE_AFTER_PROMOTION_ENABLED"), default=False)
    ML_DYNAMIC_CAPS_ENABLED = _as_bool(os.getenv("ML_DYNAMIC_CAPS_ENABLED"), default=False)
    ML_EXTREME_UPSIDE_MODEL_ENABLED = _as_bool(os.getenv("ML_EXTREME_UPSIDE_MODEL_ENABLED"), default=False)
    ML_EXTREME_UPSIDE_TARGET_ROI_PCT = _as_float(os.getenv("ML_EXTREME_UPSIDE_TARGET_ROI_PCT"), 1000.0)
    ML_RISK_POLICY_ENABLED = _as_bool(os.getenv("ML_RISK_POLICY_ENABLED"), default=False)
    ML_RISK_POLICY_APPROVE_THRESHOLD = _as_float(os.getenv("ML_RISK_POLICY_APPROVE_THRESHOLD"), 0.55)
    ML_RISK_POLICY_MIN_CONFIDENCE = _as_float(os.getenv("ML_RISK_POLICY_MIN_CONFIDENCE"), 0.55)
    ML_RISK_POLICY_TARGET_RETURN_THRESHOLD = _as_float(os.getenv("ML_RISK_POLICY_TARGET_RETURN_THRESHOLD"), 0.001)
    ML_RISK_POLICY_MAX_VALIDATION_LOSS = _as_float(os.getenv("ML_RISK_POLICY_MAX_VALIDATION_LOSS"), 0.80)
    ML_RISK_POLICY_MIN_APPROVAL_PRECISION = _as_float(os.getenv("ML_RISK_POLICY_MIN_APPROVAL_PRECISION"), 0.52)
    ML_RISK_POLICY_MIN_APPROVAL_COUNT = max(0, _as_int(os.getenv("ML_RISK_POLICY_MIN_APPROVAL_COUNT"), 5))
    ML_RISK_POLICY_MIN_APPROVAL_RATE = _as_float(os.getenv("ML_RISK_POLICY_MIN_APPROVAL_RATE"), 0.01)
    ML_RISK_POLICY_MAX_RECENT_VOLATILITY = _as_float(os.getenv("ML_RISK_POLICY_MAX_RECENT_VOLATILITY"), 0.01)
    ML_RISK_POLICY_MAX_RECENT_ABS_RETURN = _as_float(os.getenv("ML_RISK_POLICY_MAX_RECENT_ABS_RETURN"), 0.03)
    ML_RISK_POLICY_MAX_BACKTEST_DRAWDOWN_PCT = _as_float(os.getenv("ML_RISK_POLICY_MAX_BACKTEST_DRAWDOWN_PCT"), 0.05)
    ML_RISK_POLICY_MIN_BACKTEST_PROFIT_FACTOR = _as_float(os.getenv("ML_RISK_POLICY_MIN_BACKTEST_PROFIT_FACTOR"), 0.80)
    ML_ORDER_POLICY_ENABLED = _as_bool(os.getenv("ML_ORDER_POLICY_ENABLED"), default=False)
    ML_EXIT_POLICY_ENABLED = _as_bool(os.getenv("ML_EXIT_POLICY_ENABLED"), default=False)
    ML_CAP_POLICY_ENABLED = _as_bool(os.getenv("ML_CAP_POLICY_ENABLED"), default=False)
    ML_ROI_TARGET_POLICY_ENABLED = _as_bool(os.getenv("ML_ROI_TARGET_POLICY_ENABLED"), default=False)
    ML_POLICY_LIVE_AUTHORITY = os.getenv("ML_POLICY_LIVE_AUTHORITY", "guarded").strip().lower() or "guarded"
    ML_POLICY_SANDBOX_BYPASS_ENABLED = _as_bool(os.getenv("ML_POLICY_SANDBOX_BYPASS_ENABLED"), default=True)
    ML_LIVE_HARD_CAP_USDC = _as_float(os.getenv("ML_LIVE_HARD_CAP_USDC"), 10.0)
    ML_LIVE_HARD_DAILY_LOSS_USDC = _as_float(os.getenv("ML_LIVE_HARD_DAILY_LOSS_USDC"), 0.50)
    ML_TARGET_ROI_1H_PCT = _as_float(os.getenv("ML_TARGET_ROI_1H_PCT"), 1000.0)
    ML_TARGET_ROI_1H10_PCT = _as_float(os.getenv("ML_TARGET_ROI_1H10_PCT"), ONE_H10_TARGET_ROI_PCT)
    ML_TARGET_ROI_1W_PCT = _as_float(os.getenv("ML_TARGET_ROI_1W_PCT"), 100.0)
    ML_ONE_H10_MIN_TRAINING_ROWS = _as_int(os.getenv("ML_ONE_H10_MIN_TRAINING_ROWS"), 20)
    ML_AUTO_VAULT_LIVE_ENABLED = _as_bool(os.getenv("ML_AUTO_VAULT_LIVE_ENABLED"), default=False)
    ML_AUTO_VAULT_EXACT_CONFIRMATION = os.getenv("ML_AUTO_VAULT_EXACT_CONFIRMATION", "ML-AUTO-VAULT-LIVE").strip() or "ML-AUTO-VAULT-LIVE"
    ML_HISTORY_BACKFILL_ENABLED = _as_bool(os.getenv("ML_HISTORY_BACKFILL_ENABLED"), default=False)
    ML_HISTORY_BACKFILL_MAX_SYMBOLS = _as_int(os.getenv("ML_HISTORY_BACKFILL_MAX_SYMBOLS"), 50)
    ML_HISTORY_BACKFILL_LOOKBACK_DAYS = _as_int(os.getenv("ML_HISTORY_BACKFILL_LOOKBACK_DAYS"), 90)
    ML_HISTORY_BACKFILL_ORDER_BOOK_ENABLED = _as_bool(os.getenv("ML_HISTORY_BACKFILL_ORDER_BOOK_ENABLED"), default=True)
    ML_HISTORY_BACKFILL_ALL_PAIRS_ORDER_BOOK_ENABLED = _as_bool(
        os.getenv("ML_HISTORY_BACKFILL_ALL_PAIRS_ORDER_BOOK_ENABLED"),
        default=False,
    )
    ML_HISTORY_BACKFILL_REQUEST_SLEEP_SECONDS = _as_float(os.getenv("ML_HISTORY_BACKFILL_REQUEST_SLEEP_SECONDS"), 0.0)
    ML_HISTORY_BACKFILL_MAX_CANDLES_PER_REQUEST = _as_int(os.getenv("ML_HISTORY_BACKFILL_MAX_CANDLES_PER_REQUEST"), 5_000)
    ML_FEEDBACK_SYNC_ENABLED = _as_bool(os.getenv("ML_FEEDBACK_SYNC_ENABLED"), default=False)
    ML_FEEDBACK_SYNC_MAX_ROWS = _as_int(os.getenv("ML_FEEDBACK_SYNC_MAX_ROWS"), 5000)
    ML_FEEDBACK_SYNC_SOURCES = _parse_csv(
        os.getenv("ML_FEEDBACK_SYNC_SOURCES"),
        ["rankings", "backtests", "orders", "vault_cycles", "ops", "market_history"],
    )
    ML_CONTINUOUS_VAULT_ENABLED = _as_bool(os.getenv("ML_CONTINUOUS_VAULT_ENABLED"), default=False)
    ML_VAULT_TICK_ENABLED = _as_bool(os.getenv("ML_VAULT_TICK_ENABLED"), default=False)
    ML_VAULT_PROVIDER_SCOPE = os.getenv("ML_VAULT_PROVIDER_SCOPE", "all").strip().lower() or "all"
    ML_VAULT_MAX_CAP_USDC = _as_float(os.getenv("ML_VAULT_MAX_CAP_USDC"), 10.0)
    ML_VAULT_MAX_DAILY_LOSS_USDC = _as_float(os.getenv("ML_VAULT_MAX_DAILY_LOSS_USDC"), 0.50)
    ML_VAULT_MAX_ACTIVE_CYCLES = _as_int(os.getenv("ML_VAULT_MAX_ACTIVE_CYCLES"), 1)
    ML_VAULT_MAX_LIVE_CYCLES_PER_DAY = _as_int(os.getenv("ML_VAULT_MAX_LIVE_CYCLES_PER_DAY"), 1)
    ML_VAULT_LEVERAGE_POLICY = os.getenv("ML_VAULT_LEVERAGE_POLICY", "exchange_max_gated").strip().lower() or "exchange_max_gated"
    ML_VAULT_MIN_LIQUIDATION_BUFFER_PCT = _as_float(os.getenv("ML_VAULT_MIN_LIQUIDATION_BUFFER_PCT"), 0.20)
    ML_LIVE_VAULT_ONE_SHOT_ENABLED = _as_bool(os.getenv("ML_LIVE_VAULT_ONE_SHOT_ENABLED"), default=False)
    ML_LIVE_VAULT_MAX_CAP_USDC = _as_float(os.getenv("ML_LIVE_VAULT_MAX_CAP_USDC"), 10.0)
    ML_LIVE_VAULT_MAX_DAILY_LOSS_USDC = _as_float(os.getenv("ML_LIVE_VAULT_MAX_DAILY_LOSS_USDC"), 0.50)
    ML_LIVE_VAULT_PREVIEW_MAX_AGE_SECONDS = _as_float(os.getenv("ML_LIVE_VAULT_PREVIEW_MAX_AGE_SECONDS"), 600.0)
    ML_LIVE_VAULT_EXACT_CONFIRMATION = (
        os.getenv("ML_LIVE_VAULT_EXACT_CONFIRMATION", "ML-LIVE-VAULT-10USDC").strip() or "ML-LIVE-VAULT-10USDC"
    )
    ML_MIN_SIGNAL_CONFIDENCE = _as_float(os.getenv("ML_MIN_SIGNAL_CONFIDENCE"), 0.60)
    ML_MIN_FIB_CONFIDENCE = _as_float(os.getenv("ML_MIN_FIB_CONFIDENCE"), 0.55)
    ML_MAX_MODEL_AGE_HOURS = _as_float(os.getenv("ML_MAX_MODEL_AGE_HOURS"), 72.0)
    ML_BACKTEST_SCORER_WEIGHT = _as_float(os.getenv("ML_BACKTEST_SCORER_WEIGHT"), 0.10)
    ML_OPTIMIZER_POLICY_SKIP_THRESHOLD = _as_float(os.getenv("ML_OPTIMIZER_POLICY_SKIP_THRESHOLD"), -0.35)
    ML_SIGNAL_MODEL_ENABLED = _as_bool(os.getenv("ML_SIGNAL_MODEL_ENABLED"), default=False)
    ML_SIGNAL_MODEL_TYPES = _parse_csv(os.getenv("ML_SIGNAL_MODEL_TYPES"), ["pytorch_gru"])
    ML_SIGNAL_REQUIRE_PROMOTED = _as_bool(os.getenv("ML_SIGNAL_REQUIRE_PROMOTED"), default=True)
    ML_SIGNAL_MIN_CONFIDENCE = _as_float(os.getenv("ML_SIGNAL_MIN_CONFIDENCE"), 0.55)
    ML_SIGNAL_MIN_TRAINING_ROWS = _as_int(os.getenv("ML_SIGNAL_MIN_TRAINING_ROWS"), 500)
    ML_SIGNAL_MAX_TRAINING_ROWS = _as_int(os.getenv("ML_SIGNAL_MAX_TRAINING_ROWS"), 15_000)
    ML_SIGNAL_TRAINING_EPOCHS = _as_int(os.getenv("ML_SIGNAL_TRAINING_EPOCHS"), 16)
    ML_SIGNAL_TRAINING_BATCH_SIZE = max(128, _as_int(os.getenv("ML_SIGNAL_TRAINING_BATCH_SIZE"), 2048))
    ML_SIGNAL_HIDDEN_SIZE = max(0, _as_int(os.getenv("ML_SIGNAL_HIDDEN_SIZE"), 32))
    ML_SIGNAL_LEARNING_RATE = _as_float(os.getenv("ML_SIGNAL_LEARNING_RATE"), 0.01)
    ML_SIGNAL_SWEEP_MAX_TRAINING_ROWS = max(500, _as_int(os.getenv("ML_SIGNAL_SWEEP_MAX_TRAINING_ROWS"), 6_000))
    ML_SIGNAL_SEQUENCE_LENGTH = max(1, min(64, _as_int(os.getenv("ML_SIGNAL_SEQUENCE_LENGTH"), 8)))
    ML_SIGNAL_CLASS_BALANCE_ENABLED = _as_bool(os.getenv("ML_SIGNAL_CLASS_BALANCE_ENABLED"), default=True)
    ML_SIGNAL_MAX_CLASS_WEIGHT = _as_float(os.getenv("ML_SIGNAL_MAX_CLASS_WEIGHT"), 6.0)
    ML_SIGNAL_MAX_VALIDATION_LOSS = _as_float(os.getenv("ML_SIGNAL_MAX_VALIDATION_LOSS"), 0.20)
    ML_SIGNAL_MAX_CLASSIFICATION_LOSS = _as_float(os.getenv("ML_SIGNAL_MAX_CLASSIFICATION_LOSS"), 1.10)
    ML_SIGNAL_MIN_ACCURACY_EDGE = _as_float(os.getenv("ML_SIGNAL_MIN_ACCURACY_EDGE"), 0.0)
    ML_SIGNAL_MIN_ACTION_PRECISION = _as_float(os.getenv("ML_SIGNAL_MIN_ACTION_PRECISION"), 0.52)
    ML_SIGNAL_MIN_ACTION_COUNT = max(0, _as_int(os.getenv("ML_SIGNAL_MIN_ACTION_COUNT"), 5))
    ML_SIGNAL_MAX_FALSE_POSITIVE_RATE = _as_float(os.getenv("ML_SIGNAL_MAX_FALSE_POSITIVE_RATE"), 0.35)
    ML_SIGNAL_MIN_ACTION_RATE = _as_float(os.getenv("ML_SIGNAL_MIN_ACTION_RATE"), 0.005)
    ML_SIGNAL_MAX_ACTION_RATE = _as_float(os.getenv("ML_SIGNAL_MAX_ACTION_RATE"), 0.95)
    ML_SIGNAL_ALLOW_LIVE_OVERRIDE = _as_bool(os.getenv("ML_SIGNAL_ALLOW_LIVE_OVERRIDE"), default=False)
    ML_SIGNAL_TARGET_RETURN_THRESHOLD = _as_float(os.getenv("ML_SIGNAL_TARGET_RETURN_THRESHOLD"), 0.001)
    ML_SIGNAL_DIRECTIONAL_ACTION_ENABLED = _as_bool(os.getenv("ML_SIGNAL_DIRECTIONAL_ACTION_ENABLED"), default=True)
    ML_SIGNAL_MIN_ACTION_PROBABILITY = _as_float(os.getenv("ML_SIGNAL_MIN_ACTION_PROBABILITY"), 0.20)
    ML_SIGNAL_MIN_DIRECTIONAL_MARGIN = _as_float(os.getenv("ML_SIGNAL_MIN_DIRECTIONAL_MARGIN"), 0.05)
    ML_SIGNAL_MAX_HOLD_PROBABILITY_FOR_ACTION = _as_float(os.getenv("ML_SIGNAL_MAX_HOLD_PROBABILITY_FOR_ACTION"), 0.80)
    ML_SIGNAL_EXPECTED_RETURN_SCALE = _as_float(os.getenv("ML_SIGNAL_EXPECTED_RETURN_SCALE"), ML_SIGNAL_TARGET_RETURN_THRESHOLD)
    HIGH_UPSIDE_REQUIRE_PROMOTED_ML = _as_bool(os.getenv("HIGH_UPSIDE_REQUIRE_PROMOTED_ML"), default=True)
    HIGH_UPSIDE_REQUIRE_ML_SIGNAL = _as_bool(os.getenv("HIGH_UPSIDE_REQUIRE_ML_SIGNAL"), default=True)
    HIGH_UPSIDE_ML_SIGNAL_MODEL_TYPE = os.getenv("HIGH_UPSIDE_ML_SIGNAL_MODEL_TYPE", "pytorch_gru").strip().lower() or "pytorch_gru"
    ML_PREDICTION_CAP = _as_float(os.getenv("ML_PREDICTION_CAP"), 1.0)
    ML_TARGET_CAP = _as_float(os.getenv("ML_TARGET_CAP"), 1.0)
    ML_WEIGHT_CAP = _as_float(os.getenv("ML_WEIGHT_CAP"), 3.0)
    REALTIME_MARKET_ENABLED = _as_bool(os.getenv("REALTIME_MARKET_ENABLED"), default=False)
    REALTIME_MARKET_CONNECT_TIMEOUT_SECONDS = _as_float(os.getenv("REALTIME_MARKET_CONNECT_TIMEOUT_SECONDS"), 3.0)
    REALTIME_MARKET_MAX_STALE_SECONDS = _as_float(os.getenv("REALTIME_MARKET_MAX_STALE_SECONDS"), 15.0)
    WALLET_PROVIDER = os.getenv("WALLET_PROVIDER", "self_custody").strip() or "self_custody"
    WALLET_CUSTODY_MODE = (
        "local_dev" if RECOVERY_SQLITE_ACTIVE else os.getenv("WALLET_CUSTODY_MODE", "local_dev").strip().lower() or "local_dev"
    )
    WALLET_PAGE_LIVE_SYNC_ENABLED = _as_bool(
        os.getenv("WALLET_PAGE_LIVE_SYNC_ENABLED"),
        default=DEPLOYMENT_TARGET != "vercel" and not _as_bool(os.getenv("VERCEL"), default=False),
    )
    WALLET_SIGNER_ISOLATION_REQUIRED = _as_bool(os.getenv("WALLET_SIGNER_ISOLATION_REQUIRED"), default=True)
    WALLET_SIGNER_ISOLATION_CONFIRMED = _as_bool(os.getenv("WALLET_SIGNER_ISOLATION_CONFIRMED"), default=False)
    WALLET_SDK_CHECKS_PASSED = _as_bool(os.getenv("WALLET_SDK_CHECKS_PASSED"), default=False)
    WALLET_MPC_SIGNER_URL = os.getenv("WALLET_MPC_SIGNER_URL", "").strip()
    WALLET_MPC_SIGNER_TOKEN = os.getenv("WALLET_MPC_SIGNER_TOKEN", "").strip()
    WALLET_INTERNAL_MPC_SIGNER_ENABLED = _as_bool(os.getenv("WALLET_INTERNAL_MPC_SIGNER_ENABLED"), default=False)
    WALLET_MPC_SIGNER_ENCRYPTION_KEY = os.getenv("WALLET_MPC_SIGNER_ENCRYPTION_KEY", "").strip()
    WALLET_SIGNER_TIMEOUT_SECONDS = _as_float(os.getenv("WALLET_SIGNER_TIMEOUT_SECONDS"), 10.0)
    WALLET_EMERGENCY_STOP = _as_bool(os.getenv("WALLET_EMERGENCY_STOP"), default=False)
    USE_REAL_ADDRESSES = _as_bool(os.getenv("USE_REAL_ADDRESSES"), default=False)
    WALLET_REAL_CUSTODY_ENABLED = _as_bool(os.getenv("WALLET_REAL_CUSTODY_ENABLED"), default=False)
    WALLET_SELF_CUSTODY_ENABLED = _as_bool(os.getenv("WALLET_SELF_CUSTODY_ENABLED"), default=False)
    WALLET_ALLOW_IN_APP_KEYGEN = _as_bool(os.getenv("WALLET_ALLOW_IN_APP_KEYGEN"), default=False)
    WALLET_REQUIRE_WITHDRAWAL_APPROVAL = _as_bool(os.getenv("WALLET_REQUIRE_WITHDRAWAL_APPROVAL"), default=True)
    WALLET_WITHDRAWALS_ENABLED = False if RECOVERY_SQLITE_ACTIVE else _as_bool(os.getenv("WALLET_WITHDRAWALS_ENABLED"), default=False)
    WALLET_AUTO_ENABLE_WITHDRAWALS = _as_bool(os.getenv("WALLET_AUTO_ENABLE_WITHDRAWALS"), default=True)
    WALLET_AUTO_SWEEP_ENABLED = _as_bool(os.getenv("WALLET_AUTO_SWEEP_ENABLED"), default=False)
    WALLET_SWEEP_DESTINATION_ETH = os.getenv(
        "WALLET_SWEEP_DESTINATION_ETH",
        "0xcfc7d08f480E6F8c3631268ed49B44cdff389677",
    ).strip()
    WALLET_WITHDRAWAL_FEE_BPS = _as_float(os.getenv("WALLET_WITHDRAWAL_FEE_BPS"), 0.0)
    WALLET_WITHDRAWAL_FIXED_FEE_ETH = _as_float(os.getenv("WALLET_WITHDRAWAL_FIXED_FEE_ETH"), 0.0)
    WALLET_WITHDRAWAL_FEE_MAX_ETH = _as_float(os.getenv("WALLET_WITHDRAWAL_FEE_MAX_ETH"), 0.0)
    WALLET_MAX_SWEEP_ETH = _as_float(os.getenv("WALLET_MAX_SWEEP_ETH"), 0.0)
    WALLET_EVM_RPC_URL = os.getenv("WALLET_EVM_RPC_URL", "").strip()
    WALLET_EVM_NETWORKS = _parse_json_object(os.getenv("WALLET_EVM_NETWORKS_JSON"))
    WALLET_EVM_TOKEN_CONTRACTS = _parse_json_object(os.getenv("WALLET_EVM_TOKEN_CONTRACTS_JSON"))
    WALLET_EVM_MIN_GAS_PRICE_GWEI = _as_float(os.getenv("WALLET_EVM_MIN_GAS_PRICE_GWEI"), 2.0)
    WALLET_BROADCAST_VERIFY_ATTEMPTS = int(_as_float(os.getenv("WALLET_BROADCAST_VERIFY_ATTEMPTS"), 10))
    WALLET_BROADCAST_VERIFY_DELAY_SECONDS = _as_float(os.getenv("WALLET_BROADCAST_VERIFY_DELAY_SECONDS"), 2.0)
    WALLET_REQUIRED_CONFIRMATIONS = _parse_float_map(os.getenv("WALLET_REQUIRED_CONFIRMATIONS_JSON"))
    WALLET_BTC_INDEXER_URL = os.getenv("WALLET_BTC_INDEXER_URL", "").strip()
    WALLET_SOLANA_RPC_URL = os.getenv("WALLET_SOLANA_RPC_URL", "").strip()
    WALLET_XRP_RPC_URL = os.getenv("WALLET_XRP_RPC_URL", "").strip()
    WALLET_MAX_WITHDRAWAL_BY_ASSET = _parse_float_map(os.getenv("WALLET_MAX_WITHDRAWAL_BY_ASSET_JSON"))
    WALLET_DAILY_WITHDRAWAL_LIMIT_BY_WALLET = _as_float(os.getenv("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_WALLET"), 0.0)
    WALLET_DAILY_WITHDRAWAL_LIMIT_BY_ASSET = _parse_float_map(os.getenv("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_ASSET_JSON"))
    WALLET_DAILY_WITHDRAWAL_LIMIT_BY_DESTINATION = _as_float(os.getenv("WALLET_DAILY_WITHDRAWAL_LIMIT_BY_DESTINATION"), 0.0)
    WALLET_DAILY_GLOBAL_WITHDRAWAL_LIMIT = _as_float(os.getenv("WALLET_DAILY_GLOBAL_WITHDRAWAL_LIMIT"), 0.0)
    WALLET_PRICE_FEED_URL = os.getenv("WALLET_PRICE_FEED_URL", "").strip()
    ONRAMP_ENABLED = _as_bool(os.getenv("ONRAMP_ENABLED"), default=False)
    ONRAMP_PROVIDER_ENABLED = _as_bool(os.getenv("ONRAMP_PROVIDER_ENABLED"), default=ONRAMP_ENABLED)
    ONRAMP_PROVIDER = os.getenv("ONRAMP_PROVIDER", "custom_hosted").strip().lower() or "custom_hosted"
    ONRAMP_SUPPORTED_ASSETS = _parse_csv(os.getenv("ONRAMP_SUPPORTED_ASSETS"), [])
    ONRAMP_SUPPORTED_FIAT = _parse_csv(os.getenv("ONRAMP_SUPPORTED_FIAT"), ["USD"])
    ONRAMP_CUSTOM_SESSION_URL = os.getenv("ONRAMP_CUSTOM_SESSION_URL", "").strip()
    ONRAMP_CUSTOM_API_KEY = os.getenv("ONRAMP_CUSTOM_API_KEY", "").strip()
    ONRAMP_CUSTOM_WEBHOOK_SECRET = os.getenv("ONRAMP_CUSTOM_WEBHOOK_SECRET", "").strip()
    ONRAMP_CUSTOM_ALLOWED_ASSETS = _parse_json_object(os.getenv("ONRAMP_CUSTOM_ALLOWED_ASSETS_JSON"))
    ONRAMP_CUSTOM_MIN_FIAT_USD = max(0.0, _as_float(os.getenv("ONRAMP_CUSTOM_MIN_FIAT_USD"), 10.0))
    ONRAMP_CUSTOM_MAX_FIAT_USD = max(0.0, _as_float(os.getenv("ONRAMP_CUSTOM_MAX_FIAT_USD"), 5_000.0))
    ONRAMP_CUSTOM_TIMEOUT_SECONDS = max(1.0, _as_float(os.getenv("ONRAMP_CUSTOM_TIMEOUT_SECONDS"), 12.0))
    ONRAMP_CUSTOM_WEBHOOK_TOLERANCE_SECONDS = max(30, _as_int(os.getenv("ONRAMP_CUSTOM_WEBHOOK_TOLERANCE_SECONDS"), 300))
    APPLE_PAY_DIRECT_ENABLED = _as_bool(os.getenv("APPLE_PAY_DIRECT_ENABLED"), default=False)
    APPLE_PAY_CRYPTO_SALE_APPROVED = _as_bool(os.getenv("APPLE_PAY_CRYPTO_SALE_APPROVED"), default=False)
    APPLE_PAY_MERCHANT_ID = os.getenv("APPLE_PAY_MERCHANT_ID", "").strip()
    APPLE_PAY_DISPLAY_NAME = os.getenv("APPLE_PAY_DISPLAY_NAME", "AlgVault").strip() or "AlgVault"
    APPLE_PAY_DOMAIN = os.getenv("APPLE_PAY_DOMAIN", "").strip()
    APPLE_PAY_COUNTRY_CODE = os.getenv("APPLE_PAY_COUNTRY_CODE", "CA").strip().upper() or "CA"
    APPLE_PAY_SUPPORTED_NETWORKS = _parse_csv(os.getenv("APPLE_PAY_SUPPORTED_NETWORKS"), ["visa", "masterCard", "amex", "discover"])
    APPLE_PAY_MERCHANT_CERT_PATH = os.getenv("APPLE_PAY_MERCHANT_CERT_PATH", "").strip()
    APPLE_PAY_MERCHANT_KEY_PATH = os.getenv("APPLE_PAY_MERCHANT_KEY_PATH", "").strip()
    APPLE_PAY_MERCHANT_CERT_PEM = os.getenv("APPLE_PAY_MERCHANT_CERT_PEM", "").strip()
    APPLE_PAY_MERCHANT_KEY_PEM = os.getenv("APPLE_PAY_MERCHANT_KEY_PEM", "").strip()
    APPLE_PAY_GATEWAY_AUTHORIZE_URL = os.getenv("APPLE_PAY_GATEWAY_AUTHORIZE_URL", "").strip()
    APPLE_PAY_GATEWAY_REFUND_URL = os.getenv("APPLE_PAY_GATEWAY_REFUND_URL", "").strip()
    APPLE_PAY_GATEWAY_API_KEY = os.getenv("APPLE_PAY_GATEWAY_API_KEY", "").strip()
    APPLE_PAY_GATEWAY_WEBHOOK_SECRET = os.getenv("APPLE_PAY_GATEWAY_WEBHOOK_SECRET", "").strip()
    APPLE_PAY_GATEWAY_WEBHOOK_TOLERANCE_SECONDS = max(
        30,
        _as_int(os.getenv("APPLE_PAY_GATEWAY_WEBHOOK_TOLERANCE_SECONDS"), 300),
    )
    APPLE_PAY_BUY_ALLOWED_ASSETS = _parse_json_object(os.getenv("APPLE_PAY_BUY_ALLOWED_ASSETS_JSON"))
    APPLE_PAY_TREASURY_SOURCE_WALLETS = _parse_json_object(os.getenv("APPLE_PAY_TREASURY_SOURCE_WALLETS_JSON"))
    APPLE_PAY_ASSET_PRICE_USD = _parse_json_object(os.getenv("APPLE_PAY_ASSET_PRICE_USD_JSON"))
    APPLE_PAY_TREASURY_SOURCE_ADDRESS = os.getenv("APPLE_PAY_TREASURY_SOURCE_ADDRESS", "").strip()
    APPLE_PAY_TREASURY_FEE_ADDRESS = os.getenv("APPLE_PAY_TREASURY_FEE_ADDRESS", "").strip()
    APPLE_PAY_TREASURY_SIGNER_URL = os.getenv("APPLE_PAY_TREASURY_SIGNER_URL", "").strip()
    APPLE_PAY_TREASURY_SIGNER_TOKEN = os.getenv("APPLE_PAY_TREASURY_SIGNER_TOKEN", "").strip()
    APPLE_PAY_TREASURY_FEE_BPS = max(0.0, _as_float(os.getenv("APPLE_PAY_TREASURY_FEE_BPS"), 250.0))
    WALLET_BUY_PLATFORM_FEE_BPS = max(0.0, _as_float(os.getenv("WALLET_BUY_PLATFORM_FEE_BPS"), APPLE_PAY_TREASURY_FEE_BPS))
    WALLET_BUY_QUOTE_TTL_SECONDS = max(30, _as_int(os.getenv("WALLET_BUY_QUOTE_TTL_SECONDS"), 300))
    CARD_BUY_ENABLED = _as_bool(os.getenv("CARD_BUY_ENABLED"), default=False)
    CARD_GATEWAY_TOKENIZATION_URL = os.getenv("CARD_GATEWAY_TOKENIZATION_URL", "").strip()
    CARD_GATEWAY_AUTHORIZE_URL = os.getenv("CARD_GATEWAY_AUTHORIZE_URL", "").strip()
    CARD_GATEWAY_API_KEY = os.getenv("CARD_GATEWAY_API_KEY", "").strip()
    CARD_GATEWAY_WEBHOOK_SECRET = os.getenv("CARD_GATEWAY_WEBHOOK_SECRET", "").strip()
    CARD_GATEWAY_PUBLIC_CONFIG = _parse_json_object(os.getenv("CARD_GATEWAY_PUBLIC_CONFIG_JSON"))
    APPLE_PAY_EXECUTION_FEE_BUFFER_BPS = max(0.0, _as_float(os.getenv("APPLE_PAY_EXECUTION_FEE_BUFFER_BPS"), 500.0))
    APPLE_PAY_EXECUTION_FEE_USD_OVERRIDE = (
        _as_float(os.getenv("APPLE_PAY_EXECUTION_FEE_USD_OVERRIDE"), -1.0)
        if os.getenv("APPLE_PAY_EXECUTION_FEE_USD_OVERRIDE") is not None
        else None
    )
    APPLE_PAY_MIN_FIAT_USD = max(0.0, _as_float(os.getenv("APPLE_PAY_MIN_FIAT_USD"), 10.0))
    APPLE_PAY_MAX_FIAT_USD = max(0.0, _as_float(os.getenv("APPLE_PAY_MAX_FIAT_USD"), 5_000.0))
    APPLE_PAY_TIMEOUT_SECONDS = max(1.0, _as_float(os.getenv("APPLE_PAY_TIMEOUT_SECONDS"), 12.0))
    APPLE_PAY_ONEINCH_SLIPPAGE_PCT = max(0.0, _as_float(os.getenv("APPLE_PAY_ONEINCH_SLIPPAGE_PCT"), 0.5))
    APPLE_PAY_DOMAIN_ASSOCIATION = os.getenv("APPLE_PAY_DOMAIN_ASSOCIATION", "").strip()
    WORKER_APPLE_PAY_FULFILLMENT_ENABLED = _as_bool(os.getenv("WORKER_APPLE_PAY_FULFILLMENT_ENABLED"), default=True)
    ONEINCH_API_KEY = os.getenv("ONEINCH_API_KEY", "").strip()
    ONEINCH_API_BASE_URL = os.getenv("ONEINCH_API_BASE_URL", "https://api.1inch.com/swap/v6.1").strip()
    ONEINCH_REFERRER_ADDRESS = os.getenv("ONEINCH_REFERRER_ADDRESS", "").strip()
    ONEINCH_PARTNER_FEE_BPS = max(0.0, _as_float(os.getenv("ONEINCH_PARTNER_FEE_BPS"), 0.0))
    PLATFORM_GAS_TREASURY_ENABLED = _as_bool(os.getenv("PLATFORM_GAS_TREASURY_ENABLED"), default=False)
    TREASURY_ENCRYPTION_KEY = os.getenv("TREASURY_ENCRYPTION_KEY", "").strip()
    PLATFORM_GAS_TOPUP_MULTIPLIER = max(1.0, _as_float(os.getenv("PLATFORM_GAS_TOPUP_MULTIPLIER"), 2.0))
    PLATFORM_GAS_TOPUP_MAX_ETH = max(0.0, _as_float(os.getenv("PLATFORM_GAS_TOPUP_MAX_ETH"), 0.0))
    PLATFORM_GAS_TREASURY_DAILY_LIMIT_ETH = max(0.0, _as_float(os.getenv("PLATFORM_GAS_TREASURY_DAILY_LIMIT_ETH"), 0.0))
    PLATFORM_GAS_TREASURY_ROTATION_LOCK_ENABLED = _as_bool(
        os.getenv("PLATFORM_GAS_TREASURY_ROTATION_LOCK_ENABLED"),
        default=True,
    )
    WALLET_EVM_GAS_LIMIT_BUFFER_MULTIPLIER = max(1.0, _as_float(os.getenv("WALLET_EVM_GAS_LIMIT_BUFFER_MULTIPLIER"), 1.20))
    WALLET_EVM_FALLBACK_ETH_GAS_LIMIT = _as_int(os.getenv("WALLET_EVM_FALLBACK_ETH_GAS_LIMIT"), 21_000)
    WALLET_EVM_FALLBACK_ERC20_GAS_LIMIT = _as_int(os.getenv("WALLET_EVM_FALLBACK_ERC20_GAS_LIMIT"), 70_000)
    WALLET_EVM_ESTIMATE_FROM_ADDRESS = os.getenv("WALLET_EVM_ESTIMATE_FROM_ADDRESS", "").strip()
    PLATFORM_TREASURY_CONVERSION_PROVIDER = os.getenv("PLATFORM_TREASURY_CONVERSION_PROVIDER", "kucoin").strip().lower() or "kucoin"
    PLATFORM_TREASURY_CONVERSION_USER_ID = _as_int(os.getenv("PLATFORM_TREASURY_CONVERSION_USER_ID"), 0)
    PLATFORM_TREASURY_CONVERSION_CONNECTION_ID = _as_int(os.getenv("PLATFORM_TREASURY_CONVERSION_CONNECTION_ID"), 0)
    PLATFORM_TREASURY_ETH_USD_FALLBACK = max(0.0, _as_float(os.getenv("PLATFORM_TREASURY_ETH_USD_FALLBACK"), 3000.0))
    PLATFORM_TREASURY_RESERVE_JOB_MAX_ATTEMPTS = max(1, _as_int(os.getenv("PLATFORM_TREASURY_RESERVE_JOB_MAX_ATTEMPTS"), 3))
    PLATFORM_TREASURY_WITHDRAWAL_QUEUE_LIMIT = max(1, _as_int(os.getenv("PLATFORM_TREASURY_WITHDRAWAL_QUEUE_LIMIT"), 25))
    PLATFORM_TREASURY_DEFAULT_PROFIT_SHARE_PCT = min(
        100.0,
        max(0.0, _as_float(os.getenv("PLATFORM_TREASURY_DEFAULT_PROFIT_SHARE_PCT"), 50.0)),
    )
    TREASURY_SOLVENCY_ENABLED = _as_bool(os.getenv("TREASURY_SOLVENCY_ENABLED"), default=True)
    TREASURY_SOLVENCY_RECALC_INTERVAL_SECONDS = max(1.0, _as_float(os.getenv("TREASURY_SOLVENCY_RECALC_INTERVAL_SECONDS"), 30.0))
    TREASURY_HEALTHY_RATIO = max(0.0, _as_float(os.getenv("TREASURY_HEALTHY_RATIO"), 3.0))
    TREASURY_WARNING_RATIO = max(0.0, _as_float(os.getenv("TREASURY_WARNING_RATIO"), 1.5))
    TREASURY_LOW_RATIO = max(0.0, _as_float(os.getenv("TREASURY_LOW_RATIO"), 1.10))
    TREASURY_SOLVENCY_WITHDRAWAL_MIN_RATIO = max(0.0, _as_float(os.getenv("TREASURY_SOLVENCY_WITHDRAWAL_MIN_RATIO"), 1.10))
    TREASURY_SOLVENCY_BASE_SAFETY_MULTIPLIER = max(1.0, _as_float(os.getenv("TREASURY_SOLVENCY_BASE_SAFETY_MULTIPLIER"), 1.20))
    TREASURY_GAS_CONGESTION_MULTIPLIER_CAP = max(1.0, _as_float(os.getenv("TREASURY_GAS_CONGESTION_MULTIPLIER_CAP"), 3.0))
    TREASURY_GAS_VOLATILITY_WINDOW_HOURS = max(1.0, _as_float(os.getenv("TREASURY_GAS_VOLATILITY_WINDOW_HOURS"), 24.0))
    TREASURY_GAS_VOLATILITY_CAP = max(0.0, _as_float(os.getenv("TREASURY_GAS_VOLATILITY_CAP"), 0.75))
    TREASURY_FORECAST_HISTORY_HOURS = max(1.0, _as_float(os.getenv("TREASURY_FORECAST_HISTORY_HOURS"), 168.0))
    TREASURY_FORECAST_WINDOWS_HOURS = os.getenv("TREASURY_FORECAST_WINDOWS_HOURS", "1,6,24,168").strip() or "1,6,24,168"
    TREASURY_FORECAST_RETENTION_ROWS = max(24, _as_int(os.getenv("TREASURY_FORECAST_RETENTION_ROWS"), 96))
    TREASURY_REBALANCE_TARGET_RATIO = max(1.10, _as_float(os.getenv("TREASURY_REBALANCE_TARGET_RATIO"), 3.0))
    TREASURY_REBALANCE_SOURCE_ASSET = os.getenv("TREASURY_REBALANCE_SOURCE_ASSET", "USDC").strip().upper() or "USDC"
    TREASURY_REBALANCE_MIN_ETH = max(0.0, _as_float(os.getenv("TREASURY_REBALANCE_MIN_ETH"), 0.0))
    TREASURY_REBALANCE_MAX_ETH = max(0.0, _as_float(os.getenv("TREASURY_REBALANCE_MAX_ETH"), 0.0))
    TREASURY_REBALANCE_MAX_SLIPPAGE_BPS = max(0.0, _as_float(os.getenv("TREASURY_REBALANCE_MAX_SLIPPAGE_BPS"), 10.0))
    TREASURY_DEX_CONVERSIONS_ENABLED = _as_bool(os.getenv("TREASURY_DEX_CONVERSIONS_ENABLED"), default=False)

    WTF_CSRF_ENABLED = True
    TESTING = False
    PREFERRED_URL_SCHEME = os.getenv("PREFERRED_URL_SCHEME", "http").strip() or "http"
    SESSION_COOKIE_HTTPONLY = _as_bool(os.getenv("SESSION_COOKIE_HTTPONLY"), default=True)
    SESSION_COOKIE_SECURE = _as_bool(os.getenv("SESSION_COOKIE_SECURE"), default=False)
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax").strip() or "Lax"
    SESSION_COOKIE_DOMAIN = os.getenv("SESSION_COOKIE_DOMAIN", "").strip() or None
    PERMANENT_SESSION_LIFETIME_SECONDS = _as_int(os.getenv("PERMANENT_SESSION_LIFETIME_SECONDS"), 60 * 60 * 8)
    MAX_CONTENT_LENGTH = _as_int(os.getenv("MAX_CONTENT_LENGTH"), 2 * 1024 * 1024)
    PROXY_FIX_ENABLED = _as_bool(os.getenv("PROXY_FIX_ENABLED"), default=False)
    PROXY_FIX_X_FOR = _as_int(os.getenv("PROXY_FIX_X_FOR"), 1)
    PROXY_FIX_X_PROTO = _as_int(os.getenv("PROXY_FIX_X_PROTO"), 1)
    PROXY_FIX_X_HOST = _as_int(os.getenv("PROXY_FIX_X_HOST"), 1)
    PROXY_FIX_X_PORT = _as_int(os.getenv("PROXY_FIX_X_PORT"), 1)
    PROXY_FIX_X_PREFIX = _as_int(os.getenv("PROXY_FIX_X_PREFIX"), 0)
    SECURE_HEADERS_ENABLED = _as_bool(os.getenv("SECURE_HEADERS_ENABLED"), default=True)
    SECURE_HEADERS_HSTS_ENABLED = _as_bool(os.getenv("SECURE_HEADERS_HSTS_ENABLED"), default=False)
    SECURE_HEADERS_CSP_ENABLED = _as_bool(os.getenv("SECURE_HEADERS_CSP_ENABLED"), default=False)
    STATIC_CACHE_SECONDS = _as_int(os.getenv("STATIC_CACHE_SECONDS"), 31_536_000)
    SERVICE_WORKER_CACHE_SECONDS = _as_int(os.getenv("SERVICE_WORKER_CACHE_SECONDS"), 0)
    LOG_FORMAT = os.getenv("LOG_FORMAT", "plain").strip().lower() or "plain"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    RATELIMIT_ENABLED = _as_bool(os.getenv("RATELIMIT_ENABLED"), default=True)
    RATELIMIT_WINDOW_SECONDS = _as_int(os.getenv("RATELIMIT_WINDOW_SECONDS"), 60)
    RATELIMIT_LOGIN_PER_WINDOW = _as_int(os.getenv("RATELIMIT_LOGIN_PER_WINDOW"), 12)
    RATELIMIT_AUTH_SETUP_PER_WINDOW = _as_int(os.getenv("RATELIMIT_AUTH_SETUP_PER_WINDOW"), 20)
    RATELIMIT_API_PER_WINDOW = _as_int(os.getenv("RATELIMIT_API_PER_WINDOW"), 180)
    RATELIMIT_UNSAFE_PER_WINDOW = _as_int(os.getenv("RATELIMIT_UNSAFE_PER_WINDOW"), 60)

    @classmethod
    def export_defaults(cls) -> dict[str, Any]:
        """Return a serializable view of default values used in docs and tests."""

        return {
            "ASSET_VERSION": cls.ASSET_VERSION,
            "DEPLOYMENT_TARGET": cls.DEPLOYMENT_TARGET,
            "PUBLIC_APP_ORIGIN": cls.PUBLIC_APP_ORIGIN,
            "PUBLIC_API_ORIGIN": cls.PUBLIC_API_ORIGIN,
            "PUBLIC_LIVE_API_ORIGIN": cls.PUBLIC_LIVE_API_ORIGIN,
            "LIVE_API_CORS_ALLOWED_ORIGINS": list(cls.LIVE_API_CORS_ALLOWED_ORIGINS),
            "WEB_CONCURRENCY": cls.WEB_CONCURRENCY,
            "GUNICORN_THREADS": cls.GUNICORN_THREADS,
            "WORKER_MODE": cls.WORKER_MODE,
            "ENABLE_IN_PROCESS_WORKERS": cls.ENABLE_IN_PROCESS_WORKERS,
            "WORKER_PROCESS_CONFIGURED": cls.WORKER_PROCESS_CONFIGURED,
            "WORKER_LEASE_TTL_SECONDS": cls.WORKER_LEASE_TTL_SECONDS,
            "WORKER_POLL_SECONDS": cls.WORKER_POLL_SECONDS,
            "SCHEMA_BOOTSTRAP_ENABLED": cls.SCHEMA_BOOTSTRAP_ENABLED,
            "ALLOW_PRODUCTION_SCHEMA_BOOTSTRAP": cls.ALLOW_PRODUCTION_SCHEMA_BOOTSTRAP,
            "DEFER_DATABASE_STARTUP_ERRORS": cls.DEFER_DATABASE_STARTUP_ERRORS,
            "RECOVERY_SQLITE_ACTIVE": cls.RECOVERY_SQLITE_ACTIVE,
            "APP_MODE": cls.APP_MODE,
            "ENABLE_LIVE_TRADING": cls.ENABLE_LIVE_TRADING,
            "SQLITE_BUSY_TIMEOUT_MS": cls.SQLITE_BUSY_TIMEOUT_MS,
            "SQLITE_ENABLE_WAL": cls.SQLITE_ENABLE_WAL,
            "ALLOWED_SYMBOLS": list(cls.ALLOWED_SYMBOLS),
            "DEFAULT_TIMEFRAME": cls.DEFAULT_TIMEFRAME,
            "DEFAULT_PAPER_BALANCE": cls.DEFAULT_PAPER_BALANCE,
            "BACKTEST_PAPER_BALANCE_USD": cls.BACKTEST_PAPER_BALANCE_USD,
            "BACKTEST_ALLOCATION_DEFAULT_USD": cls.BACKTEST_ALLOCATION_DEFAULT_USD,
            "BACKTEST_MAX_CHART_POINTS": cls.BACKTEST_MAX_CHART_POINTS,
            "PAPER_BALANCE_MIN": cls.PAPER_BALANCE_MIN,
            "PAPER_BALANCE_MAX": cls.PAPER_BALANCE_MAX,
            "MAX_DAILY_LOSS_USDC": cls.MAX_DAILY_LOSS_USDC,
            "MAX_BACKTEST_DRAWDOWN_PCT": cls.MAX_BACKTEST_DRAWDOWN_PCT,
            "MAX_LEVERAGE": cls.MAX_LEVERAGE,
            "LOSS_COOLDOWN_MINUTES": cls.LOSS_COOLDOWN_MINUTES,
            "LOSS_STREAK_COOLDOWN_THRESHOLD": cls.LOSS_STREAK_COOLDOWN_THRESHOLD,
            "MAX_TRADES_PER_WINDOW": cls.MAX_TRADES_PER_WINDOW,
            "TRADE_WINDOW_MINUTES": cls.TRADE_WINDOW_MINUTES,
            "RISK_PER_TRADE_PCT": cls.RISK_PER_TRADE_PCT,
            "MIN_REWARD_RISK": cls.MIN_REWARD_RISK,
            "FIXED_DOLLAR_SIZE": cls.FIXED_DOLLAR_SIZE,
            "STRATEGY_POLL_SECONDS": cls.STRATEGY_POLL_SECONDS,
            "STRATEGY_RUN_LEASE_TTL_SECONDS": cls.STRATEGY_RUN_LEASE_TTL_SECONDS,
            "STRATEGY_CHANGE_DRIVEN_LOOP_ENABLED": cls.STRATEGY_CHANGE_DRIVEN_LOOP_ENABLED,
            "STRATEGY_IDLE_REEVAL_SECONDS": cls.STRATEGY_IDLE_REEVAL_SECONDS,
            "STRATEGY_HEARTBEAT_PERSIST_SECONDS": cls.STRATEGY_HEARTBEAT_PERSIST_SECONDS,
            "NO_TRADE_AUDIT_COMPACT_ENABLED": cls.NO_TRADE_AUDIT_COMPACT_ENABLED,
            "NO_TRADE_AUDIT_THROTTLE_SECONDS": cls.NO_TRADE_AUDIT_THROTTLE_SECONDS,
            "NO_TRADE_AUDIT_RETENTION_HOURS": cls.NO_TRADE_AUDIT_RETENTION_HOURS,
            "TRANSIENT_AUDIT_RETENTION_HOURS": cls.TRANSIENT_AUDIT_RETENTION_HOURS,
            "FEE_BPS": cls.FEE_BPS,
            "SIM_SLIPPAGE_BPS": cls.SIM_SLIPPAGE_BPS,
            "DASHBOARD_ACCOUNT_SEGMENT_TTL_SECONDS": cls.DASHBOARD_ACCOUNT_SEGMENT_TTL_SECONDS,
            "DASHBOARD_ACCOUNT_SEGMENT_STALE_SECONDS": cls.DASHBOARD_ACCOUNT_SEGMENT_STALE_SECONDS,
            "DASHBOARD_TRADE_LIST_SEGMENT_TTL_SECONDS": cls.DASHBOARD_TRADE_LIST_SEGMENT_TTL_SECONDS,
            "DASHBOARD_TRADE_LIST_STALE_SECONDS": cls.DASHBOARD_TRADE_LIST_STALE_SECONDS,
            "DASHBOARD_STATIC_SEGMENT_TTL_SECONDS": cls.DASHBOARD_STATIC_SEGMENT_TTL_SECONDS,
            "DASHBOARD_STATIC_SEGMENT_STALE_SECONDS": cls.DASHBOARD_STATIC_SEGMENT_STALE_SECONDS,
            "PROVIDER_TIMEOUT_SECONDS": cls.PROVIDER_TIMEOUT_SECONDS,
            "PROVIDER_RETRY_ATTEMPTS": cls.PROVIDER_RETRY_ATTEMPTS,
            "PROVIDER_RETRY_SLEEP_SECONDS": cls.PROVIDER_RETRY_SLEEP_SECONDS,
            "HYPERLIQUID_ACCOUNT": cls.HYPERLIQUID_ACCOUNT,
            "HYPERLIQUID_ENV": cls.HYPERLIQUID_ENV,
            "HYPERLIQUID_BASE_URL": cls.HYPERLIQUID_BASE_URL,
            "RUN_HYPERLIQUID_LIVE_TESTS": cls.RUN_HYPERLIQUID_LIVE_TESTS,
            "RUN_HYPERLIQUID_FILL_TEST": cls.RUN_HYPERLIQUID_FILL_TEST,
            "HYPERLIQUID_LIVE_TEST_MAX_NOTIONAL_USD": cls.HYPERLIQUID_LIVE_TEST_MAX_NOTIONAL_USD,
            "HYPERLIQUID_MIN_ORDER_VALUE_USD": cls.HYPERLIQUID_MIN_ORDER_VALUE_USD,
            "KUCOIN_CONTRACT_SPECS_JSON": cls.KUCOIN_CONTRACT_SPECS_JSON,
            "KUCOIN_DEFAULT_MARKET_TYPE": cls.KUCOIN_DEFAULT_MARKET_TYPE,
            "KUCOIN_DEFAULT_SPOT_QUOTE": cls.KUCOIN_DEFAULT_SPOT_QUOTE,
            "KUCOIN_SPOT_BASE_URL": cls.KUCOIN_SPOT_BASE_URL,
            "KUCOIN_SPOT_SYMBOL_MAP_JSON": cls.KUCOIN_SPOT_SYMBOL_MAP_JSON,
            "KUCOIN_TEST_ACCOUNT": cls.KUCOIN_TEST_ACCOUNT,
            "KUCOIN_TEST_SYMBOL": cls.KUCOIN_TEST_SYMBOL,
            "KUCOIN_MAX_TEST_NOTIONAL_USDT": cls.KUCOIN_MAX_TEST_NOTIONAL_USDT,
            "KUCOIN_ENABLE_LIVE_TEST_TRADES": cls.KUCOIN_ENABLE_LIVE_TEST_TRADES,
            "KUCOIN_ENABLE_FILL_TEST": cls.KUCOIN_ENABLE_FILL_TEST,
            "KUCOIN_MARGIN_MODE": cls.KUCOIN_MARGIN_MODE,
            "KUCOIN_POSITION_SIDE": cls.KUCOIN_POSITION_SIDE,
            "KUCOIN_SPOT_ACCOUNTS_PATH": cls.KUCOIN_SPOT_ACCOUNTS_PATH,
            "KUCOIN_SUB_ACCOUNTS_PATH": cls.KUCOIN_SUB_ACCOUNTS_PATH,
            "KUCOIN_SPOT_SYMBOLS_PATH": cls.KUCOIN_SPOT_SYMBOLS_PATH,
            "KUCOIN_SPOT_TICKER_PATH": cls.KUCOIN_SPOT_TICKER_PATH,
            "KUCOIN_SPOT_ORDERS_PATH": cls.KUCOIN_SPOT_ORDERS_PATH,
            "KUCOIN_SPOT_TEST_ORDER_PATH": cls.KUCOIN_SPOT_TEST_ORDER_PATH,
            "KUCOIN_SPOT_OPEN_ORDERS_PATH": cls.KUCOIN_SPOT_OPEN_ORDERS_PATH,
            "KUCOIN_SPOT_CLIENT_ORDER_PATH": cls.KUCOIN_SPOT_CLIENT_ORDER_PATH,
            "KUCOIN_SPOT_FILLS_PATH": cls.KUCOIN_SPOT_FILLS_PATH,
            "KUCOIN_ACTIVE_CONTRACTS_PATH": cls.KUCOIN_ACTIVE_CONTRACTS_PATH,
            "KUCOIN_DEPOSIT_ADDRESSES_PATH": cls.KUCOIN_DEPOSIT_ADDRESSES_PATH,
            "KUCOIN_WITHDRAWALS_PATH": cls.KUCOIN_WITHDRAWALS_PATH,
            "KUCOIN_UNIVERSAL_TRANSFER_PATH": cls.KUCOIN_UNIVERSAL_TRANSFER_PATH,
            "KUCOIN_CONVERT_QUOTE_PATH": cls.KUCOIN_CONVERT_QUOTE_PATH,
            "KUCOIN_CONVERT_ORDER_PATH": cls.KUCOIN_CONVERT_ORDER_PATH,
            "EXPLICIT_LIVE_CONFIRMED": cls.EXPLICIT_LIVE_CONFIRMED,
            "SECONDARY_CONFIRMATION": cls.SECONDARY_CONFIRMATION,
            "SHADOW_LIVE_MIN_TRADES": cls.SHADOW_LIVE_MIN_TRADES,
            "SHADOW_LIVE_MIN_HOURS": cls.SHADOW_LIVE_MIN_HOURS,
            "VAULT_SHADOW_VALIDATION_MINUTES": cls.VAULT_SHADOW_VALIDATION_MINUTES,
            "VAULT_SHADOW_MIN_SIGNALS": cls.VAULT_SHADOW_MIN_SIGNALS,
            "VAULT_MAX_SPREAD_BPS": cls.VAULT_MAX_SPREAD_BPS,
            "VAULT_MIN_LIQUIDITY_USD": cls.VAULT_MIN_LIQUIDITY_USD,
            "VAULT_MAX_SLIPPAGE_BPS": cls.VAULT_MAX_SLIPPAGE_BPS,
            "VAULT_SIGNAL_STABILITY_THRESHOLD": cls.VAULT_SIGNAL_STABILITY_THRESHOLD,
            "VAULT_AGGRESSIVE_SIZE_MULTIPLIER": cls.VAULT_AGGRESSIVE_SIZE_MULTIPLIER,
            "VAULT_LIVE_FALLBACK_POLICY": cls.VAULT_LIVE_FALLBACK_POLICY,
            "VAULT_MIN_RISK_FRACTION": cls.VAULT_MIN_RISK_FRACTION,
            "VAULT_MAX_RISK_FRACTION": cls.VAULT_MAX_RISK_FRACTION,
            "OPTIMIZER_TRAINING_WINDOW_DAYS": cls.OPTIMIZER_TRAINING_WINDOW_DAYS,
            "OPTIMIZER_TESTING_WINDOW_DAYS": cls.OPTIMIZER_TESTING_WINDOW_DAYS,
            "OPTIMIZER_STEP_DAYS": cls.OPTIMIZER_STEP_DAYS,
            "OPTIMIZER_USE_FULL_HISTORY": cls.OPTIMIZER_USE_FULL_HISTORY,
            "OPTIMIZER_RECENCY_WEIGHTING_ENABLED": cls.OPTIMIZER_RECENCY_WEIGHTING_ENABLED,
            "OPTIMIZER_DECAY_FACTOR": cls.OPTIMIZER_DECAY_FACTOR,
            "OPTIMIZER_MARKET_DATA_TIMEOUT_SECONDS": cls.OPTIMIZER_MARKET_DATA_TIMEOUT_SECONDS,
            "FIND_CANARY_RANKING_TIMEOUT_SECONDS": cls.FIND_CANARY_RANKING_TIMEOUT_SECONDS,
            "FIND_CANARY_MAX_SYMBOLS": cls.FIND_CANARY_MAX_SYMBOLS,
            "FIND_CANARY_MAX_RANKINGS": cls.FIND_CANARY_MAX_RANKINGS,
            "FIND_CANARY_RANKING_ALLOW_MOCK_DATA": cls.FIND_CANARY_RANKING_ALLOW_MOCK_DATA,
            "FIND_CANARY_RANKING_FALLBACK_SYMBOLS": cls.FIND_CANARY_RANKING_FALLBACK_SYMBOLS,
            "FIND_CANARY_MARKET_DATA_PROBE_LIMIT": cls.FIND_CANARY_MARKET_DATA_PROBE_LIMIT,
            "FIND_CANARY_PER_SWEEP_TIMEOUT_SECONDS": cls.FIND_CANARY_PER_SWEEP_TIMEOUT_SECONDS,
            "FIND_CANARY_FALLBACK_ENABLED": cls.FIND_CANARY_FALLBACK_ENABLED,
            "FIND_CANARY_FALLBACK_SYMBOL": cls.FIND_CANARY_FALLBACK_SYMBOL,
            "FIND_CANARY_FALLBACK_STRATEGY": cls.FIND_CANARY_FALLBACK_STRATEGY,
            "FIND_CANARY_FALLBACK_TIMEFRAME": cls.FIND_CANARY_FALLBACK_TIMEFRAME,
            "FIND_CANARY_FALLBACK_RECENCY_HOURS": cls.FIND_CANARY_FALLBACK_RECENCY_HOURS,
            "FIND_CANARY_FALLBACK_ALLOCATION_USD": cls.FIND_CANARY_FALLBACK_ALLOCATION_USD,
            "FIND_CANARY_FALLBACK_STOP_LOSS_PCT": cls.FIND_CANARY_FALLBACK_STOP_LOSS_PCT,
            "FIND_CANARY_FALLBACK_TAKE_PROFIT_PCT": cls.FIND_CANARY_FALLBACK_TAKE_PROFIT_PCT,
            "OPTIMIZER_PREFILTER_ENABLED": cls.OPTIMIZER_PREFILTER_ENABLED,
            "OPTIMIZER_SIGNAL_HISTORY_LIMIT": cls.OPTIMIZER_SIGNAL_HISTORY_LIMIT,
            "MARKET_DATA_CACHE_TTL_SECONDS": cls.MARKET_DATA_CACHE_TTL_SECONDS,
            "MARKET_DATA_LIVE_CANDLE_CACHE_SECONDS": cls.MARKET_DATA_LIVE_CANDLE_CACHE_SECONDS,
            "MARKET_DATA_LIVE_ORDER_BOOK_CACHE_SECONDS": cls.MARKET_DATA_LIVE_ORDER_BOOK_CACHE_SECONDS,
            "MARKET_DATA_LIVE_MIDS_CACHE_SECONDS": cls.MARKET_DATA_LIVE_MIDS_CACHE_SECONDS,
            "MARKET_DATA_LIVE_STALE_SECONDS": cls.MARKET_DATA_LIVE_STALE_SECONDS,
            "MARKET_DATA_LIVE_FAILURE_BACKOFF_SECONDS": cls.MARKET_DATA_LIVE_FAILURE_BACKOFF_SECONDS,
            "TRADING_CONNECTION_LIVE_SNAPSHOT_CACHE_SECONDS": cls.TRADING_CONNECTION_LIVE_SNAPSHOT_CACHE_SECONDS,
            "TRADING_CONNECTION_LIVE_SNAPSHOT_BACKOFF_SECONDS": cls.TRADING_CONNECTION_LIVE_SNAPSHOT_BACKOFF_SECONDS,
            "TRADING_CONNECTION_LIVE_SNAPSHOT_STALE_SECONDS": cls.TRADING_CONNECTION_LIVE_SNAPSHOT_STALE_SECONDS,
            "MARKET_DATA_RESEARCH_STALE_SECONDS": cls.MARKET_DATA_RESEARCH_STALE_SECONDS,
            "MARKET_DATA_RESEARCH_CACHE_TTL_SECONDS": cls.MARKET_DATA_RESEARCH_CACHE_TTL_SECONDS,
            "NET_ROI_MIN_EDGE_BPS": cls.NET_ROI_MIN_EDGE_BPS,
            "NET_ROI_MAX_CHURN_PENALTY": cls.NET_ROI_MAX_CHURN_PENALTY,
            "NET_ROI_MIN_FILL_QUALITY": cls.NET_ROI_MIN_FILL_QUALITY,
            "NET_ROI_V2_ENABLED": cls.NET_ROI_V2_ENABLED,
            "NET_ROI_V2_MIN_QUALITY_GRADE": cls.NET_ROI_V2_MIN_QUALITY_GRADE,
            "ONE_HOUR_EDGE_V2_ENABLED": cls.ONE_HOUR_EDGE_V2_ENABLED,
            "ONE_HOUR_MIN_EDGE_GRADE": cls.ONE_HOUR_MIN_EDGE_GRADE,
            "ONE_HOUR_MAX_RAW_NET_GAP_PCT": cls.ONE_HOUR_MAX_RAW_NET_GAP_PCT,
            "ONE_HOUR_MIN_EXECUTION_QUALITY": cls.ONE_HOUR_MIN_EXECUTION_QUALITY,
            "SIGNAL_DEBOUNCE_SECONDS": cls.SIGNAL_DEBOUNCE_SECONDS,
            "CANARY_PREVIEW_ONLY": cls.CANARY_PREVIEW_ONLY,
            "FIRST_CANARY_ALLOCATION_BUDGET_USDT": cls.FIRST_CANARY_ALLOCATION_BUDGET_USDT,
            "FIRST_CANARY_FALLBACK_ALLOCATION_BUDGET_USDT": cls.FIRST_CANARY_FALLBACK_ALLOCATION_BUDGET_USDT,
            "FIRST_CANARY_USE_MIN_SIZE_FALLBACK": cls.FIRST_CANARY_USE_MIN_SIZE_FALLBACK,
            "FIRST_CANARY_MAX_LEVERAGE": cls.FIRST_CANARY_MAX_LEVERAGE,
            "LIVE_MICRO_CANARY_ENABLED": cls.LIVE_MICRO_CANARY_ENABLED,
            "LIVE_MICRO_CANARY_ACCOUNT_USD": cls.LIVE_MICRO_CANARY_ACCOUNT_USD,
            "LIVE_MICRO_CANARY_MAX_ALLOCATION_USD": cls.LIVE_MICRO_CANARY_MAX_ALLOCATION_USD,
            "LIVE_MICRO_CANARY_MAX_RISK_PCT": cls.LIVE_MICRO_CANARY_MAX_RISK_PCT,
            "LIVE_MICRO_CANARY_MAX_LEVERAGE": cls.LIVE_MICRO_CANARY_MAX_LEVERAGE,
            "LIVE_MICRO_CANARY_REQUIRE_STOP_LOSS": cls.LIVE_MICRO_CANARY_REQUIRE_STOP_LOSS,
            "LIVE_MICRO_CANARY_REQUIRE_TAKE_PROFIT": cls.LIVE_MICRO_CANARY_REQUIRE_TAKE_PROFIT,
            "LIVE_MICRO_CANARY_PREVIEW_ONLY": cls.LIVE_MICRO_CANARY_PREVIEW_ONLY,
            "LIVE_MICRO_CANARY_LIVE_SUBMIT_ENABLED": cls.LIVE_MICRO_CANARY_LIVE_SUBMIT_ENABLED,
            "LIVE_MICRO_CANARY_MIN_NOTIONAL_BUFFER_USD": cls.LIVE_MICRO_CANARY_MIN_NOTIONAL_BUFFER_USD,
            "LIVE_MICRO_CANARY_MAX_DAILY_LIVE_ORDERS": cls.LIVE_MICRO_CANARY_MAX_DAILY_LIVE_ORDERS,
            "LIVE_MICRO_CANARY_REQUIRE_EXACT_CONFIRMATION": cls.LIVE_MICRO_CANARY_REQUIRE_EXACT_CONFIRMATION,
            "LIVE_MICRO_CANARY_EXACT_CONFIRMATION": cls.LIVE_MICRO_CANARY_EXACT_CONFIRMATION,
            "LIVE_MICRO_CANARY_ALLOW_MIN_NOTIONAL_ORDER": cls.LIVE_MICRO_CANARY_ALLOW_MIN_NOTIONAL_ORDER,
            "LIVE_MICRO_CANARY_ORDER_BUDGET_USD": cls.LIVE_MICRO_CANARY_ORDER_BUDGET_USD,
            "LIVE_MICRO_CANARY_DEFAULT_MIN_NOTIONAL_USD": cls.LIVE_MICRO_CANARY_DEFAULT_MIN_NOTIONAL_USD,
            "RAPID_ML_LIVE_ENABLED": cls.RAPID_ML_LIVE_ENABLED,
            "RAPID_ML_PREVIEW_ONLY": cls.RAPID_ML_PREVIEW_ONLY,
            "RAPID_ML_DECISION_INTERVAL_MS": cls.RAPID_ML_DECISION_INTERVAL_MS,
            "RAPID_ML_MAX_ORDER_RATE_PER_PROVIDER": cls.RAPID_ML_MAX_ORDER_RATE_PER_PROVIDER,
            "RAPID_ML_SYMBOLS": cls.RAPID_ML_SYMBOLS,
            "RAPID_ML_SYMBOLS_HYPERLIQUID": cls.RAPID_ML_SYMBOLS_HYPERLIQUID,
            "RAPID_ML_SYMBOLS_KUCOIN": cls.RAPID_ML_SYMBOLS_KUCOIN,
            "RAPID_ML_AUTO_FUTURES_UNIVERSE_ENABLED": cls.RAPID_ML_AUTO_FUTURES_UNIVERSE_ENABLED,
            "RAPID_ML_UNIVERSE_REFRESH_SECONDS": cls.RAPID_ML_UNIVERSE_REFRESH_SECONDS,
            "RAPID_ML_MAX_SYMBOLS_PER_PROVIDER": cls.RAPID_ML_MAX_SYMBOLS_PER_PROVIDER,
            "RAPID_ML_ML_SIZING_ENABLED": cls.RAPID_ML_ML_SIZING_ENABLED,
            "RAPID_ML_HARD_CAP_USDC": cls.RAPID_ML_HARD_CAP_USDC,
            "RAPID_ML_AUTO_CLOSE_ENABLED": cls.RAPID_ML_AUTO_CLOSE_ENABLED,
            "RAPID_ML_ROTATE_ENABLED": cls.RAPID_ML_ROTATE_ENABLED,
            "RAPID_ML_ROTATE_MIN_SCORE_DELTA": cls.RAPID_ML_ROTATE_MIN_SCORE_DELTA,
            "RAPID_ML_MANAGE_MANUAL_POSITIONS": cls.RAPID_ML_MANAGE_MANUAL_POSITIONS,
            "RAPID_ML_FEATURE_TIMEFRAME": cls.RAPID_ML_FEATURE_TIMEFRAME,
            "RAPID_ML_FEATURE_CANDLE_LIMIT": cls.RAPID_ML_FEATURE_CANDLE_LIMIT,
            "RAPID_ML_PREVIEW_SNAPSHOT_TTL_SECONDS": cls.RAPID_ML_PREVIEW_SNAPSHOT_TTL_SECONDS,
            "RAPID_ML_MAX_DAILY_LOSS_PCT": cls.RAPID_ML_MAX_DAILY_LOSS_PCT,
            "RAPID_ML_MAX_POSITION_PCT": cls.RAPID_ML_MAX_POSITION_PCT,
            "RAPID_ML_MIN_EDGE_BPS": cls.RAPID_ML_MIN_EDGE_BPS,
            "RAPID_ML_MIN_CONFIDENCE": cls.RAPID_ML_MIN_CONFIDENCE,
            "RAPID_ML_MIN_EDGE_AGREEMENT": cls.RAPID_ML_MIN_EDGE_AGREEMENT,
            "RAPID_ML_COST_RESERVE_BPS": cls.RAPID_ML_COST_RESERVE_BPS,
            "RAPID_ML_SLIPPAGE_BPS": cls.RAPID_ML_SLIPPAGE_BPS,
            "RAPID_ML_UNKNOWN_SPREAD_BPS": cls.RAPID_ML_UNKNOWN_SPREAD_BPS,
            "RAPID_ML_MAX_DIRECTIONAL_EXPOSURE_PCT": cls.RAPID_ML_MAX_DIRECTIONAL_EXPOSURE_PCT,
            "RAPID_ML_FEE_BPS_BY_PROVIDER_JSON": cls.RAPID_ML_FEE_BPS_BY_PROVIDER_JSON,
            "RAPID_ML_MIN_NOTIONAL_BUFFER_USD": cls.RAPID_ML_MIN_NOTIONAL_BUFFER_USD,
            "RAPID_ML_MAX_PROVIDER_FAILURES": cls.RAPID_ML_MAX_PROVIDER_FAILURES,
            "RAPID_ML_CIRCUIT_BREAKER_WINDOW_MINUTES": cls.RAPID_ML_CIRCUIT_BREAKER_WINDOW_MINUTES,
            "ONE_H10_LIVE_ENABLED": cls.ONE_H10_LIVE_ENABLED,
            "ONE_H10_EXACT_CONFIRMATION": cls.ONE_H10_EXACT_CONFIRMATION,
            "ONE_H10_TARGET_ROI_PCT": cls.ONE_H10_TARGET_ROI_PCT,
            "ONE_H10_HORIZON_SECONDS": cls.ONE_H10_HORIZON_SECONDS,
            "ONE_H10_POLL_SECONDS": cls.ONE_H10_POLL_SECONDS,
            "ONE_H10_REBALANCE_SECONDS": cls.ONE_H10_REBALANCE_SECONDS,
            "ONE_H10_AUTO_RESUME_ACTIVE_RUNS": cls.ONE_H10_AUTO_RESUME_ACTIVE_RUNS,
            "ONE_H10_FEATURE_TIMEFRAMES": list(cls.ONE_H10_FEATURE_TIMEFRAMES),
            "ONE_H10_REQUIRED_FEATURE_TIMEFRAMES": list(cls.ONE_H10_REQUIRED_FEATURE_TIMEFRAMES),
            "ONE_H10_FEATURE_REFRESH_SECONDS": cls.ONE_H10_FEATURE_REFRESH_SECONDS,
            "ONE_H10_FEATURE_MAX_MARKETS_PER_SYNC": cls.ONE_H10_FEATURE_MAX_MARKETS_PER_SYNC,
            "ONE_H10_FEATURE_RATE_LIMIT_BACKOFF_SECONDS": cls.ONE_H10_FEATURE_RATE_LIMIT_BACKOFF_SECONDS,
            "ONE_H10_START_SYNC_FEATURES": cls.ONE_H10_START_SYNC_FEATURES,
            "ONE_H10_MARKET_DATA_BACKOFF_SECONDS": cls.ONE_H10_MARKET_DATA_BACKOFF_SECONDS,
            "ONE_H10_MARKET_DATA_CACHE_SECONDS": cls.ONE_H10_MARKET_DATA_CACHE_SECONDS,
            "ONE_H10_ACCOUNT_REFRESH_SECONDS": cls.ONE_H10_ACCOUNT_REFRESH_SECONDS,
            "ONE_H10_PROVIDER_BACKOFF_ESCALATION_COUNT": cls.ONE_H10_PROVIDER_BACKOFF_ESCALATION_COUNT,
            "ONE_H10_PROVIDER_BACKOFF_WINDOW_SECONDS": cls.ONE_H10_PROVIDER_BACKOFF_WINDOW_SECONDS,
            "ONE_H10_ALL_PAIRS_ENABLED": cls.ONE_H10_ALL_PAIRS_ENABLED,
            "ONE_H10_ALLOW_BOOTSTRAP_WITHOUT_PROMOTED_ML": cls.ONE_H10_ALLOW_BOOTSTRAP_WITHOUT_PROMOTED_ML,
            "ONE_H10_REQUIRE_PROMOTED_ML": cls.ONE_H10_REQUIRE_PROMOTED_ML,
            "ONE_H10_BOOTSTRAP_LIVE_ENABLED": cls.ONE_H10_BOOTSTRAP_LIVE_ENABLED,
            "ONE_H10_MAX_PROVIDER_LEGS": cls.ONE_H10_MAX_PROVIDER_LEGS,
            "ONE_H10_MAX_LEVERAGE": cls.ONE_H10_MAX_LEVERAGE,
            "ONE_H10_MIN_LIQUIDITY_USD": cls.ONE_H10_MIN_LIQUIDITY_USD,
            "ONE_H10_MAX_SLIPPAGE_BPS": cls.ONE_H10_MAX_SLIPPAGE_BPS,
            "ONE_H10_MIN_FORECAST_CONFIDENCE": cls.ONE_H10_MIN_FORECAST_CONFIDENCE,
            "ONE_H10_MIN_BOOTSTRAP_CONFIDENCE": cls.ONE_H10_MIN_BOOTSTRAP_CONFIDENCE,
            "ONE_H10_DIRECTIONAL_THRESHOLD": cls.ONE_H10_DIRECTIONAL_THRESHOLD,
            "ONE_H10_MIN_POSITION_FRACTION": cls.ONE_H10_MIN_POSITION_FRACTION,
            "ONE_H10_MAX_POSITION_FRACTION": cls.ONE_H10_MAX_POSITION_FRACTION,
            "ONE_H10_PROFIT_OPTIMIZER_ENABLED": cls.ONE_H10_PROFIT_OPTIMIZER_ENABLED,
            "ONE_H10_MIN_PROFITABILITY_SCORE": cls.ONE_H10_MIN_PROFITABILITY_SCORE,
            "ONE_H10_FIBONACCI_MIN_CONFIDENCE": cls.ONE_H10_FIBONACCI_MIN_CONFIDENCE,
            "ONE_H10_REJECT_ZERO_SPREAD": cls.ONE_H10_REJECT_ZERO_SPREAD,
            "ONE_H10_ML_FORECAST_FAMILIES": list(cls.ONE_H10_ML_FORECAST_FAMILIES),
            "ONE_H10_ML_PROFIT_WEIGHT": cls.ONE_H10_ML_PROFIT_WEIGHT,
            "ONE_H10_ML_EXPECTED_EDGE_CAP_BPS": cls.ONE_H10_ML_EXPECTED_EDGE_CAP_BPS,
            "ONE_H10_MIN_MODEL_AGREEMENT": cls.ONE_H10_MIN_MODEL_AGREEMENT,
            "ONE_H10_MODEL_AGREEMENT_WEIGHT": cls.ONE_H10_MODEL_AGREEMENT_WEIGHT,
            "ONE_H10_MAX_TAKE_PROFIT_PCT": cls.ONE_H10_MAX_TAKE_PROFIT_PCT,
            "ONE_H10_MAX_STOP_LOSS_PCT": cls.ONE_H10_MAX_STOP_LOSS_PCT,
            "ONE_H10_BOOTSTRAP_FEATURE_BLOCKERS_ADVISORY": cls.ONE_H10_BOOTSTRAP_FEATURE_BLOCKERS_ADVISORY,
            "ONE_H10_ML_ADVISORY_BLOCKERS": list(cls.ONE_H10_ML_ADVISORY_BLOCKERS),
            "AGGRESSIVE_1H_ENABLED": cls.AGGRESSIVE_1H_ENABLED,
            "ALLOW_AGGRESSIVE_LIVE_TRADING": cls.ALLOW_AGGRESSIVE_LIVE_TRADING,
            "AGGRESSIVE_1H_MIN_TRADES": cls.AGGRESSIVE_1H_MIN_TRADES,
            "AGGRESSIVE_1H_MAX_DRAWDOWN_PCT": cls.AGGRESSIVE_1H_MAX_DRAWDOWN_PCT,
            "AGGRESSIVE_1H_MAX_PARAMETER_SETS": cls.AGGRESSIVE_1H_MAX_PARAMETER_SETS,
            "AGGRESSIVE_1H_RISK_PER_TRADE_PCT": cls.AGGRESSIVE_1H_RISK_PER_TRADE_PCT,
            "AGGRESSIVE_1H_POSITION_SIZE_FRACTION": cls.AGGRESSIVE_1H_POSITION_SIZE_FRACTION,
            "AGGRESSIVE_1H_TIMEFRAMES": list(cls.AGGRESSIVE_1H_TIMEFRAMES),
            "AGGRESSIVE_1H_LIVE_CAP_USDC": cls.AGGRESSIVE_1H_LIVE_CAP_USDC,
            "AGGRESSIVE_1H_LIVE_CAP_PCT": cls.AGGRESSIVE_1H_LIVE_CAP_PCT,
            "AGGRESSIVE_MIN_EDGE_BPS": cls.AGGRESSIVE_MIN_EDGE_BPS,
            "AGGRESSIVE_1H_MAX_COST_DRAG_BPS": cls.AGGRESSIVE_1H_MAX_COST_DRAG_BPS,
            "AGGRESSIVE_1H_MIN_CAPACITY_MULTIPLE": cls.AGGRESSIVE_1H_MIN_CAPACITY_MULTIPLE,
            "AGGRESSIVE_1H_MIN_MFE_MAE": cls.AGGRESSIVE_1H_MIN_MFE_MAE,
            "AGGRESSIVE_1H_RECENCY_HALF_LIFE_HOURS": cls.AGGRESSIVE_1H_RECENCY_HALF_LIFE_HOURS,
            "AGGRESSIVE_1H_MIN_WINDOW_STABILITY": cls.AGGRESSIVE_1H_MIN_WINDOW_STABILITY,
            "EXTREME_ROI_ENABLED": cls.EXTREME_ROI_ENABLED,
            "EXTREME_ROI_MIN_TRADES": cls.EXTREME_ROI_MIN_TRADES,
            "EXTREME_ROI_MAX_DRAWDOWN_PCT": cls.EXTREME_ROI_MAX_DRAWDOWN_PCT,
            "EXTREME_ROI_MAX_PARAMETER_SETS": cls.EXTREME_ROI_MAX_PARAMETER_SETS,
            "EXTREME_ROI_RISK_PER_TRADE_PCT": cls.EXTREME_ROI_RISK_PER_TRADE_PCT,
            "EXTREME_ROI_POSITION_SIZE_FRACTION": cls.EXTREME_ROI_POSITION_SIZE_FRACTION,
            "EXTREME_ROI_TIMEFRAMES": list(cls.EXTREME_ROI_TIMEFRAMES),
            "EXTREME_ROI_MIN_EDGE_BPS": cls.EXTREME_ROI_MIN_EDGE_BPS,
            "EXTREME_ROI_MIN_CONFIDENCE": cls.EXTREME_ROI_MIN_CONFIDENCE,
            "EXTREME_ROI_MIN_LIQUIDITY_USD": cls.EXTREME_ROI_MIN_LIQUIDITY_USD,
            "EXTREME_ROI_MIN_SIGNAL_STABILITY": cls.EXTREME_ROI_MIN_SIGNAL_STABILITY,
            "DYNAMIC_UNIVERSE_ENABLED": cls.DYNAMIC_UNIVERSE_ENABLED,
            "UNIVERSE_MAX_SYMBOLS": cls.UNIVERSE_MAX_SYMBOLS,
            "UNIVERSE_MIN_LIQUIDITY_USD": cls.UNIVERSE_MIN_LIQUIDITY_USD,
            "UNIVERSE_MAX_SPREAD_BPS": cls.UNIVERSE_MAX_SPREAD_BPS,
            "UNIVERSE_REFRESH_SECONDS": cls.UNIVERSE_REFRESH_SECONDS,
            "UNIVERSE_SYMBOL_BLACKLIST": list(cls.UNIVERSE_SYMBOL_BLACKLIST),
            "HOT_TOKEN_SCAN_ENABLED": cls.HOT_TOKEN_SCAN_ENABLED,
            "HOT_TOKEN_REFRESH_SECONDS": cls.HOT_TOKEN_REFRESH_SECONDS,
            "HOT_TOKEN_MAX_CANDIDATES": cls.HOT_TOKEN_MAX_CANDIDATES,
            "HOT_TOKEN_VOLUME_SPIKE_RATIO": cls.HOT_TOKEN_VOLUME_SPIKE_RATIO,
            "HOT_TOKEN_MIN_VOLATILITY_PCT": cls.HOT_TOKEN_MIN_VOLATILITY_PCT,
            "LEVERAGE_OPTIMIZER_ENABLED": cls.LEVERAGE_OPTIMIZER_ENABLED,
            "AGGRESSIVE_MAX_TEST_LEVERAGE": cls.AGGRESSIVE_MAX_TEST_LEVERAGE,
            "AGGRESSIVE_MAX_LIVE_LEVERAGE": cls.AGGRESSIVE_MAX_LIVE_LEVERAGE,
            "MIN_LIQUIDATION_BUFFER_PCT": cls.MIN_LIQUIDATION_BUFFER_PCT,
            "VAULT_MAX_PARALLEL_LEGS": cls.VAULT_MAX_PARALLEL_LEGS,
            "VAULT_MIN_LEG_USD": cls.VAULT_MIN_LEG_USD,
            "VAULT_MAX_SYMBOL_ALLOCATION_PCT": cls.VAULT_MAX_SYMBOL_ALLOCATION_PCT,
            "VAULT_STRATEGY_BASKET_ENABLED": cls.VAULT_STRATEGY_BASKET_ENABLED,
            "AGGRESSIVE_1H_POLL_SECONDS": cls.AGGRESSIVE_1H_POLL_SECONDS,
            "LIVE_CONNECTION_FAILURE_BACKOFF_SECONDS": cls.LIVE_CONNECTION_FAILURE_BACKOFF_SECONDS,
            "VAULT_MAX_ACTIVE_CYCLES": cls.VAULT_MAX_ACTIVE_CYCLES,
            "VAULT_START_ASYNC_ENABLED": cls.VAULT_START_ASYNC_ENABLED,
            "VAULT_MAX_ACTIVE_CYCLES_PER_ASSET": cls.VAULT_MAX_ACTIVE_CYCLES_PER_ASSET,
            "VAULT_MAX_ASSET_EXPOSURE_PCT": cls.VAULT_MAX_ASSET_EXPOSURE_PCT,
            "VAULT_MAX_DURATION_EXPOSURE_PCT": cls.VAULT_MAX_DURATION_EXPOSURE_PCT,
            "VAULT_MAX_STRATEGY_EXPOSURE_PCT": cls.VAULT_MAX_STRATEGY_EXPOSURE_PCT,
            "VAULT_MAX_SYMBOL_EXPOSURE_PCT": cls.VAULT_MAX_SYMBOL_EXPOSURE_PCT,
            "VAULT_MIN_AVAILABLE_RESERVE_USD": cls.VAULT_MIN_AVAILABLE_RESERVE_USD,
            "VAULT_MIN_RISK_ADJUSTED_SCORE": cls.VAULT_MIN_RISK_ADJUSTED_SCORE,
            "VAULT_CYCLE_ENGINE_ENABLED": cls.VAULT_CYCLE_ENGINE_ENABLED,
            "VAULT_CYCLE_REAL_TRANSFERS_ENABLED": cls.VAULT_CYCLE_REAL_TRANSFERS_ENABLED,
            "VAULT_CYCLE_AUTOMATIC_WITHDRAWALS_ENABLED": cls.VAULT_CYCLE_AUTOMATIC_WITHDRAWALS_ENABLED,
            "VAULT_CYCLE_PROVIDERS": list(cls.VAULT_CYCLE_PROVIDERS),
            "VAULT_CYCLE_ALLOWED_WITHDRAWAL_ADDRESSES": cls.VAULT_CYCLE_ALLOWED_WITHDRAWAL_ADDRESSES,
            "VAULT_CYCLE_MAX_TRANSFER_USD": cls.VAULT_CYCLE_MAX_TRANSFER_USD,
            "VAULT_CYCLE_DAILY_WITHDRAWAL_CAP_USD": cls.VAULT_CYCLE_DAILY_WITHDRAWAL_CAP_USD,
            "VAULT_CYCLE_MAX_EXCHANGE_ALLOCATION_PCT": cls.VAULT_CYCLE_MAX_EXCHANGE_ALLOCATION_PCT,
            "VAULT_CYCLE_MAX_SYMBOL_ALLOCATION_PCT": cls.VAULT_CYCLE_MAX_SYMBOL_ALLOCATION_PCT,
            "VAULT_CYCLE_MAX_CONCURRENT_POSITIONS": cls.VAULT_CYCLE_MAX_CONCURRENT_POSITIONS,
            "VAULT_CYCLE_MIN_EXCHANGE_ALLOCATION_USD": cls.VAULT_CYCLE_MIN_EXCHANGE_ALLOCATION_USD,
            "VAULT_CYCLE_TRANSFER_CONFIRMATION_TIMEOUT_SECONDS": cls.VAULT_CYCLE_TRANSFER_CONFIRMATION_TIMEOUT_SECONDS,
            "VAULT_CYCLE_CONVERSION_ENABLED": cls.VAULT_CYCLE_CONVERSION_ENABLED,
            "VAULT_CYCLE_CONVERSION_PROVIDER": cls.VAULT_CYCLE_CONVERSION_PROVIDER,
            "VAULT_CYCLE_STABLECOIN_MAX_SLIPPAGE_BPS": cls.VAULT_CYCLE_STABLECOIN_MAX_SLIPPAGE_BPS,
            "VAULT_CYCLE_REQUIRE_EXCHANGE_RESERVE": cls.VAULT_CYCLE_REQUIRE_EXCHANGE_RESERVE,
            "VAULT_CYCLE_DEFAULT_NETWORK_BY_ASSET": cls.VAULT_CYCLE_DEFAULT_NETWORK_BY_ASSET,
            "VAULT_CYCLE_ACTIVITY_ENFORCEMENT_ENABLED": cls.VAULT_CYCLE_ACTIVITY_ENFORCEMENT_ENABLED,
            "VAULT_CYCLE_MAX_IDLE_SECONDS": cls.VAULT_CYCLE_MAX_IDLE_SECONDS,
            "VAULT_CYCLE_RESCREEN_SECONDS": cls.VAULT_CYCLE_RESCREEN_SECONDS,
            "VAULT_CYCLE_MIN_TRADES_PER_CYCLE": cls.VAULT_CYCLE_MIN_TRADES_PER_CYCLE,
            "VAULT_CYCLE_TARGET_UTILIZATION_PCT": cls.VAULT_CYCLE_TARGET_UTILIZATION_PCT,
            "VAULT_CYCLE_MIN_OPPORTUNITY_SCORE": cls.VAULT_CYCLE_MIN_OPPORTUNITY_SCORE,
            "VAULT_CYCLE_ROTATION_SCORE_DELTA": cls.VAULT_CYCLE_ROTATION_SCORE_DELTA,
            "VAULT_CYCLE_EXCHANGE_REBALANCE_ENABLED": cls.VAULT_CYCLE_EXCHANGE_REBALANCE_ENABLED,
            "AGGRESSIVE_RISK_ADJUSTED_MIN_TRADES": cls.AGGRESSIVE_RISK_ADJUSTED_MIN_TRADES,
            "AGGRESSIVE_RISK_ADJUSTED_MAX_DRAWDOWN_PCT": cls.AGGRESSIVE_RISK_ADJUSTED_MAX_DRAWDOWN_PCT,
            "AGGRESSIVE_RISK_ADJUSTED_MAX_PARAMETER_SETS": cls.AGGRESSIVE_RISK_ADJUSTED_MAX_PARAMETER_SETS,
            "DYNAMIC_INTRADAY_TIMEFRAMES": list(cls.DYNAMIC_INTRADAY_TIMEFRAMES),
            "DYNAMIC_INTRADAY_MIN_TRADES": cls.DYNAMIC_INTRADAY_MIN_TRADES,
            "DYNAMIC_INTRADAY_MAX_DRAWDOWN_PCT": cls.DYNAMIC_INTRADAY_MAX_DRAWDOWN_PCT,
            "DYNAMIC_INTRADAY_MAX_PARAMETER_SETS": cls.DYNAMIC_INTRADAY_MAX_PARAMETER_SETS,
            "DYNAMIC_INTRADAY_MIN_EDGE_BPS": cls.DYNAMIC_INTRADAY_MIN_EDGE_BPS,
            "DYNAMIC_INTRADAY_MIN_LIQUIDITY_USD": cls.DYNAMIC_INTRADAY_MIN_LIQUIDITY_USD,
            "DYNAMIC_INTRADAY_MAX_SPREAD_BPS": cls.DYNAMIC_INTRADAY_MAX_SPREAD_BPS,
            "DYNAMIC_INTRADAY_REQUIRE_SHADOW_VALIDATION": cls.DYNAMIC_INTRADAY_REQUIRE_SHADOW_VALIDATION,
            "DYNAMIC_INTRADAY_LIVE_ELIGIBLE": cls.DYNAMIC_INTRADAY_LIVE_ELIGIBLE,
            "HIGH_UPSIDE_PROFILE_ENABLED": cls.HIGH_UPSIDE_PROFILE_ENABLED,
            "HIGH_UPSIDE_LIVE_ELIGIBLE": cls.HIGH_UPSIDE_LIVE_ELIGIBLE,
            "HIGH_UPSIDE_AUTO_LIVE_ENABLED": cls.HIGH_UPSIDE_AUTO_LIVE_ENABLED,
            "HIGH_UPSIDE_DISCOVERY_ENABLED": cls.HIGH_UPSIDE_DISCOVERY_ENABLED,
            "HIGH_UPSIDE_PROVIDER_SCOPE": cls.HIGH_UPSIDE_PROVIDER_SCOPE,
            "HIGH_UPSIDE_MAX_SYMBOLS": cls.HIGH_UPSIDE_MAX_SYMBOLS,
            "HIGH_UPSIDE_TIMEFRAMES": list(cls.HIGH_UPSIDE_TIMEFRAMES),
            "HIGH_UPSIDE_MAX_SWEEPS": cls.HIGH_UPSIDE_MAX_SWEEPS,
            "HIGH_UPSIDE_COMMAND_TIMEOUT_SECONDS": cls.HIGH_UPSIDE_COMMAND_TIMEOUT_SECONDS,
            "HIGH_UPSIDE_STAGE1_MAX_MIDS": cls.HIGH_UPSIDE_STAGE1_MAX_MIDS,
            "HIGH_UPSIDE_ALL_PAIRS_ENABLED": cls.HIGH_UPSIDE_ALL_PAIRS_ENABLED,
            "HIGH_UPSIDE_STAGE2_CANDIDATE_MULTIPLIER": cls.HIGH_UPSIDE_STAGE2_CANDIDATE_MULTIPLIER,
            "HIGH_UPSIDE_BOUNDED_SCANNER_UNIVERSE": cls.HIGH_UPSIDE_BOUNDED_SCANNER_UNIVERSE,
            "HIGH_UPSIDE_MARKET_DATA_RATE_LIMIT_PER_SECOND": cls.HIGH_UPSIDE_MARKET_DATA_RATE_LIMIT_PER_SECOND,
            "HIGH_UPSIDE_RATE_LIMIT_BACKOFF_SECONDS": cls.HIGH_UPSIDE_RATE_LIMIT_BACKOFF_SECONDS,
            "HIGH_UPSIDE_ADAPTIVE_CADENCE_ENABLED": cls.HIGH_UPSIDE_ADAPTIVE_CADENCE_ENABLED,
            "HIGH_UPSIDE_MIN_SCAN_INTERVAL_SECONDS": cls.HIGH_UPSIDE_MIN_SCAN_INTERVAL_SECONDS,
            "HIGH_UPSIDE_FAST_SCAN_INTERVAL_SECONDS": cls.HIGH_UPSIDE_FAST_SCAN_INTERVAL_SECONDS,
            "HIGH_UPSIDE_LOSS_COOLDOWN_SECONDS": cls.HIGH_UPSIDE_LOSS_COOLDOWN_SECONDS,
            "HIGH_UPSIDE_REJECTION_COOLDOWN_SECONDS": cls.HIGH_UPSIDE_REJECTION_COOLDOWN_SECONDS,
            "HIGH_UPSIDE_MAX_LIVE_CYCLES_PER_DAY": cls.HIGH_UPSIDE_MAX_LIVE_CYCLES_PER_DAY,
            "HIGH_UPSIDE_MAX_ACTIVE_CYCLES": cls.HIGH_UPSIDE_MAX_ACTIVE_CYCLES,
            "HIGH_UPSIDE_MAX_DAILY_LOSS_USDC": cls.HIGH_UPSIDE_MAX_DAILY_LOSS_USDC,
            "HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION": dict(cls.HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION),
            "HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION": dict(cls.HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION),
            "HIGH_UPSIDE_MAX_SCANNER_REJECTION_RATE": cls.HIGH_UPSIDE_MAX_SCANNER_REJECTION_RATE,
            "ML_RANKER_ENABLED": cls.ML_RANKER_ENABLED,
            "ML_SCORE_WEIGHT": cls.ML_SCORE_WEIGHT,
            "ML_RANKER_LEARNING_RATE": cls.ML_RANKER_LEARNING_RATE,
            "ML_RANKER_L2": cls.ML_RANKER_L2,
            "ML_MIN_TRAINING_EVENTS": cls.ML_MIN_TRAINING_EVENTS,
            "ML_ALLOW_LIVE_UPDATES": cls.ML_ALLOW_LIVE_UPDATES,
            "ML_LIVE_LEARNING_POLICY": cls.ML_LIVE_LEARNING_POLICY,
            "ML_LIVE_PROMOTION_MIN_EVENTS": cls.ML_LIVE_PROMOTION_MIN_EVENTS,
            "ML_LIVE_PROMOTION_MAX_MEAN_LOSS": cls.ML_LIVE_PROMOTION_MAX_MEAN_LOSS,
            "ML_LIVE_PROMOTION_MAX_NEGATIVE_ERROR_RATE": cls.ML_LIVE_PROMOTION_MAX_NEGATIVE_ERROR_RATE,
            "ML_OFFLINE_MODELS_ENABLED": cls.ML_OFFLINE_MODELS_ENABLED,
            "ML_OFFLINE_BLEND_ENABLED": cls.ML_OFFLINE_BLEND_ENABLED,
            "ML_OFFLINE_MODEL_TYPES": list(cls.ML_OFFLINE_MODEL_TYPES),
            "ML_OFFLINE_SAFE_SCORING_MODEL_TYPES": list(cls.ML_OFFLINE_SAFE_SCORING_MODEL_TYPES),
            "ML_OFFLINE_SCORE_WEIGHT": cls.ML_OFFLINE_SCORE_WEIGHT,
            "ML_OFFLINE_MIN_TRAINING_ROWS": cls.ML_OFFLINE_MIN_TRAINING_ROWS,
            "ML_OFFLINE_MARKET_HISTORY_MAX_ROWS": cls.ML_OFFLINE_MARKET_HISTORY_MAX_ROWS,
            "ML_OFFLINE_MARKET_HISTORY_SAMPLES_PER_WINDOW": cls.ML_OFFLINE_MARKET_HISTORY_SAMPLES_PER_WINDOW,
            "ML_OFFLINE_MAX_VALIDATION_LOSS": cls.ML_OFFLINE_MAX_VALIDATION_LOSS,
            "ML_OFFLINE_MAX_NEGATIVE_ERROR_RATE": cls.ML_OFFLINE_MAX_NEGATIVE_ERROR_RATE,
            "ML_OFFLINE_MAX_MODEL_AGE_HOURS": cls.ML_OFFLINE_MAX_MODEL_AGE_HOURS,
            "ML_OFFLINE_MAX_DRIFT": cls.ML_OFFLINE_MAX_DRIFT,
            "ML_OFFLINE_MIN_TOP_DECILE_PRECISION": cls.ML_OFFLINE_MIN_TOP_DECILE_PRECISION,
            "ML_OFFLINE_MAX_FALSE_POSITIVE_HIGH_UPSIDE_RATE": cls.ML_OFFLINE_MAX_FALSE_POSITIVE_HIGH_UPSIDE_RATE,
            "ML_OFFLINE_MAX_CALIBRATION_ERROR": cls.ML_OFFLINE_MAX_CALIBRATION_ERROR,
            "ML_ALL_AREAS_ENABLED": cls.ML_ALL_AREAS_ENABLED,
            "ML_REQUIRE_PROMOTED_FOR_LIVE": cls.ML_REQUIRE_PROMOTED_FOR_LIVE,
            "ML_ALLOW_STRATEGY_SIGNAL_OVERRIDE": cls.ML_ALLOW_STRATEGY_SIGNAL_OVERRIDE,
            "ML_ALLOW_ALLOCATION_OVERRIDE": cls.ML_ALLOW_ALLOCATION_OVERRIDE,
            "ML_ALLOW_EXECUTION_STYLE_SUGGESTIONS": cls.ML_ALLOW_EXECUTION_STYLE_SUGGESTIONS,
            "ML_OPS_ANOMALY_ENABLED": cls.ML_OPS_ANOMALY_ENABLED,
            "ML_DETERMINISTIC_SAFETY_GATES_REQUIRED": cls.ML_DETERMINISTIC_SAFETY_GATES_REQUIRED,
            "ML_FIRST_STRATEGIES_ENABLED": cls.ML_FIRST_STRATEGIES_ENABLED,
            "ML_FIBONACCI_MODEL_ENABLED": cls.ML_FIBONACCI_MODEL_ENABLED,
            "ML_BACKTEST_SCORER_ENABLED": cls.ML_BACKTEST_SCORER_ENABLED,
            "ML_OPTIMIZER_POLICY_ENABLED": cls.ML_OPTIMIZER_POLICY_ENABLED,
            "ML_AUTO_LIVE_AFTER_PROMOTION_ENABLED": cls.ML_AUTO_LIVE_AFTER_PROMOTION_ENABLED,
            "ML_DYNAMIC_CAPS_ENABLED": cls.ML_DYNAMIC_CAPS_ENABLED,
            "ML_EXTREME_UPSIDE_MODEL_ENABLED": cls.ML_EXTREME_UPSIDE_MODEL_ENABLED,
            "ML_EXTREME_UPSIDE_TARGET_ROI_PCT": cls.ML_EXTREME_UPSIDE_TARGET_ROI_PCT,
            "ML_RISK_POLICY_ENABLED": cls.ML_RISK_POLICY_ENABLED,
            "ML_RISK_POLICY_APPROVE_THRESHOLD": cls.ML_RISK_POLICY_APPROVE_THRESHOLD,
            "ML_RISK_POLICY_MIN_CONFIDENCE": cls.ML_RISK_POLICY_MIN_CONFIDENCE,
            "ML_RISK_POLICY_TARGET_RETURN_THRESHOLD": cls.ML_RISK_POLICY_TARGET_RETURN_THRESHOLD,
            "ML_RISK_POLICY_MAX_VALIDATION_LOSS": cls.ML_RISK_POLICY_MAX_VALIDATION_LOSS,
            "ML_RISK_POLICY_MIN_APPROVAL_PRECISION": cls.ML_RISK_POLICY_MIN_APPROVAL_PRECISION,
            "ML_RISK_POLICY_MIN_APPROVAL_COUNT": cls.ML_RISK_POLICY_MIN_APPROVAL_COUNT,
            "ML_RISK_POLICY_MIN_APPROVAL_RATE": cls.ML_RISK_POLICY_MIN_APPROVAL_RATE,
            "ML_RISK_POLICY_MAX_RECENT_VOLATILITY": cls.ML_RISK_POLICY_MAX_RECENT_VOLATILITY,
            "ML_RISK_POLICY_MAX_RECENT_ABS_RETURN": cls.ML_RISK_POLICY_MAX_RECENT_ABS_RETURN,
            "ML_RISK_POLICY_MAX_BACKTEST_DRAWDOWN_PCT": cls.ML_RISK_POLICY_MAX_BACKTEST_DRAWDOWN_PCT,
            "ML_RISK_POLICY_MIN_BACKTEST_PROFIT_FACTOR": cls.ML_RISK_POLICY_MIN_BACKTEST_PROFIT_FACTOR,
            "ML_ORDER_POLICY_ENABLED": cls.ML_ORDER_POLICY_ENABLED,
            "ML_EXIT_POLICY_ENABLED": cls.ML_EXIT_POLICY_ENABLED,
            "ML_CAP_POLICY_ENABLED": cls.ML_CAP_POLICY_ENABLED,
            "ML_ROI_TARGET_POLICY_ENABLED": cls.ML_ROI_TARGET_POLICY_ENABLED,
            "ML_POLICY_LIVE_AUTHORITY": cls.ML_POLICY_LIVE_AUTHORITY,
            "ML_POLICY_SANDBOX_BYPASS_ENABLED": cls.ML_POLICY_SANDBOX_BYPASS_ENABLED,
            "ML_LIVE_HARD_CAP_USDC": cls.ML_LIVE_HARD_CAP_USDC,
            "ML_LIVE_HARD_DAILY_LOSS_USDC": cls.ML_LIVE_HARD_DAILY_LOSS_USDC,
            "ML_TARGET_ROI_1H_PCT": cls.ML_TARGET_ROI_1H_PCT,
            "ML_TARGET_ROI_1H10_PCT": cls.ML_TARGET_ROI_1H10_PCT,
            "ML_TARGET_ROI_1W_PCT": cls.ML_TARGET_ROI_1W_PCT,
            "ML_ONE_H10_MIN_TRAINING_ROWS": cls.ML_ONE_H10_MIN_TRAINING_ROWS,
            "ML_AUTO_VAULT_LIVE_ENABLED": cls.ML_AUTO_VAULT_LIVE_ENABLED,
            "ML_AUTO_VAULT_EXACT_CONFIRMATION": cls.ML_AUTO_VAULT_EXACT_CONFIRMATION,
            "ML_HISTORY_BACKFILL_ENABLED": cls.ML_HISTORY_BACKFILL_ENABLED,
            "ML_HISTORY_BACKFILL_MAX_SYMBOLS": cls.ML_HISTORY_BACKFILL_MAX_SYMBOLS,
            "ML_HISTORY_BACKFILL_LOOKBACK_DAYS": cls.ML_HISTORY_BACKFILL_LOOKBACK_DAYS,
            "ML_HISTORY_BACKFILL_ORDER_BOOK_ENABLED": cls.ML_HISTORY_BACKFILL_ORDER_BOOK_ENABLED,
            "ML_HISTORY_BACKFILL_ALL_PAIRS_ORDER_BOOK_ENABLED": cls.ML_HISTORY_BACKFILL_ALL_PAIRS_ORDER_BOOK_ENABLED,
            "ML_HISTORY_BACKFILL_REQUEST_SLEEP_SECONDS": cls.ML_HISTORY_BACKFILL_REQUEST_SLEEP_SECONDS,
            "ML_HISTORY_BACKFILL_MAX_CANDLES_PER_REQUEST": cls.ML_HISTORY_BACKFILL_MAX_CANDLES_PER_REQUEST,
            "ML_FEEDBACK_SYNC_ENABLED": cls.ML_FEEDBACK_SYNC_ENABLED,
            "ML_FEEDBACK_SYNC_MAX_ROWS": cls.ML_FEEDBACK_SYNC_MAX_ROWS,
            "ML_FEEDBACK_SYNC_SOURCES": cls.ML_FEEDBACK_SYNC_SOURCES,
            "ML_CONTINUOUS_VAULT_ENABLED": cls.ML_CONTINUOUS_VAULT_ENABLED,
            "ML_VAULT_TICK_ENABLED": cls.ML_VAULT_TICK_ENABLED,
            "ML_VAULT_PROVIDER_SCOPE": cls.ML_VAULT_PROVIDER_SCOPE,
            "ML_VAULT_MAX_CAP_USDC": cls.ML_VAULT_MAX_CAP_USDC,
            "ML_VAULT_MAX_DAILY_LOSS_USDC": cls.ML_VAULT_MAX_DAILY_LOSS_USDC,
            "ML_VAULT_MAX_ACTIVE_CYCLES": cls.ML_VAULT_MAX_ACTIVE_CYCLES,
            "ML_VAULT_MAX_LIVE_CYCLES_PER_DAY": cls.ML_VAULT_MAX_LIVE_CYCLES_PER_DAY,
            "ML_VAULT_LEVERAGE_POLICY": cls.ML_VAULT_LEVERAGE_POLICY,
            "ML_VAULT_MIN_LIQUIDATION_BUFFER_PCT": cls.ML_VAULT_MIN_LIQUIDATION_BUFFER_PCT,
            "ML_LIVE_VAULT_ONE_SHOT_ENABLED": cls.ML_LIVE_VAULT_ONE_SHOT_ENABLED,
            "ML_LIVE_VAULT_MAX_CAP_USDC": cls.ML_LIVE_VAULT_MAX_CAP_USDC,
            "ML_LIVE_VAULT_MAX_DAILY_LOSS_USDC": cls.ML_LIVE_VAULT_MAX_DAILY_LOSS_USDC,
            "ML_LIVE_VAULT_PREVIEW_MAX_AGE_SECONDS": cls.ML_LIVE_VAULT_PREVIEW_MAX_AGE_SECONDS,
            "ML_LIVE_VAULT_EXACT_CONFIRMATION": cls.ML_LIVE_VAULT_EXACT_CONFIRMATION,
            "ML_MIN_SIGNAL_CONFIDENCE": cls.ML_MIN_SIGNAL_CONFIDENCE,
            "ML_MIN_FIB_CONFIDENCE": cls.ML_MIN_FIB_CONFIDENCE,
            "ML_MAX_MODEL_AGE_HOURS": cls.ML_MAX_MODEL_AGE_HOURS,
            "ML_BACKTEST_SCORER_WEIGHT": cls.ML_BACKTEST_SCORER_WEIGHT,
            "ML_OPTIMIZER_POLICY_SKIP_THRESHOLD": cls.ML_OPTIMIZER_POLICY_SKIP_THRESHOLD,
            "ML_SIGNAL_MODEL_ENABLED": cls.ML_SIGNAL_MODEL_ENABLED,
            "ML_SIGNAL_MODEL_TYPES": list(cls.ML_SIGNAL_MODEL_TYPES),
            "ML_SIGNAL_REQUIRE_PROMOTED": cls.ML_SIGNAL_REQUIRE_PROMOTED,
            "ML_SIGNAL_MIN_CONFIDENCE": cls.ML_SIGNAL_MIN_CONFIDENCE,
            "ML_SIGNAL_MIN_TRAINING_ROWS": cls.ML_SIGNAL_MIN_TRAINING_ROWS,
            "ML_SIGNAL_MAX_TRAINING_ROWS": cls.ML_SIGNAL_MAX_TRAINING_ROWS,
            "ML_SIGNAL_TRAINING_EPOCHS": cls.ML_SIGNAL_TRAINING_EPOCHS,
            "ML_SIGNAL_TRAINING_BATCH_SIZE": cls.ML_SIGNAL_TRAINING_BATCH_SIZE,
            "ML_SIGNAL_HIDDEN_SIZE": cls.ML_SIGNAL_HIDDEN_SIZE,
            "ML_SIGNAL_LEARNING_RATE": cls.ML_SIGNAL_LEARNING_RATE,
            "ML_SIGNAL_SWEEP_MAX_TRAINING_ROWS": cls.ML_SIGNAL_SWEEP_MAX_TRAINING_ROWS,
            "ML_SIGNAL_SEQUENCE_LENGTH": cls.ML_SIGNAL_SEQUENCE_LENGTH,
            "ML_SIGNAL_CLASS_BALANCE_ENABLED": cls.ML_SIGNAL_CLASS_BALANCE_ENABLED,
            "ML_SIGNAL_MAX_CLASS_WEIGHT": cls.ML_SIGNAL_MAX_CLASS_WEIGHT,
            "ML_SIGNAL_MAX_VALIDATION_LOSS": cls.ML_SIGNAL_MAX_VALIDATION_LOSS,
            "ML_SIGNAL_MAX_CLASSIFICATION_LOSS": cls.ML_SIGNAL_MAX_CLASSIFICATION_LOSS,
            "ML_SIGNAL_MIN_ACCURACY_EDGE": cls.ML_SIGNAL_MIN_ACCURACY_EDGE,
            "ML_SIGNAL_MIN_ACTION_PRECISION": cls.ML_SIGNAL_MIN_ACTION_PRECISION,
            "ML_SIGNAL_MIN_ACTION_COUNT": cls.ML_SIGNAL_MIN_ACTION_COUNT,
            "ML_SIGNAL_MAX_FALSE_POSITIVE_RATE": cls.ML_SIGNAL_MAX_FALSE_POSITIVE_RATE,
            "ML_SIGNAL_MIN_ACTION_RATE": cls.ML_SIGNAL_MIN_ACTION_RATE,
            "ML_SIGNAL_MAX_ACTION_RATE": cls.ML_SIGNAL_MAX_ACTION_RATE,
            "ML_SIGNAL_ALLOW_LIVE_OVERRIDE": cls.ML_SIGNAL_ALLOW_LIVE_OVERRIDE,
            "ML_SIGNAL_TARGET_RETURN_THRESHOLD": cls.ML_SIGNAL_TARGET_RETURN_THRESHOLD,
            "ML_SIGNAL_DIRECTIONAL_ACTION_ENABLED": cls.ML_SIGNAL_DIRECTIONAL_ACTION_ENABLED,
            "ML_SIGNAL_MIN_ACTION_PROBABILITY": cls.ML_SIGNAL_MIN_ACTION_PROBABILITY,
            "ML_SIGNAL_MIN_DIRECTIONAL_MARGIN": cls.ML_SIGNAL_MIN_DIRECTIONAL_MARGIN,
            "ML_SIGNAL_MAX_HOLD_PROBABILITY_FOR_ACTION": cls.ML_SIGNAL_MAX_HOLD_PROBABILITY_FOR_ACTION,
            "ML_SIGNAL_EXPECTED_RETURN_SCALE": cls.ML_SIGNAL_EXPECTED_RETURN_SCALE,
            "HIGH_UPSIDE_REQUIRE_PROMOTED_ML": cls.HIGH_UPSIDE_REQUIRE_PROMOTED_ML,
            "HIGH_UPSIDE_REQUIRE_ML_SIGNAL": cls.HIGH_UPSIDE_REQUIRE_ML_SIGNAL,
            "HIGH_UPSIDE_ML_SIGNAL_MODEL_TYPE": cls.HIGH_UPSIDE_ML_SIGNAL_MODEL_TYPE,
            "ML_PREDICTION_CAP": cls.ML_PREDICTION_CAP,
            "ML_TARGET_CAP": cls.ML_TARGET_CAP,
            "ML_WEIGHT_CAP": cls.ML_WEIGHT_CAP,
            "REALTIME_MARKET_ENABLED": cls.REALTIME_MARKET_ENABLED,
            "REALTIME_MARKET_MAX_STALE_SECONDS": cls.REALTIME_MARKET_MAX_STALE_SECONDS,
            "WALLET_PROVIDER": cls.WALLET_PROVIDER,
            "WALLET_CUSTODY_MODE": cls.WALLET_CUSTODY_MODE,
            "WALLET_PAGE_LIVE_SYNC_ENABLED": cls.WALLET_PAGE_LIVE_SYNC_ENABLED,
            "WALLET_SIGNER_ISOLATION_REQUIRED": cls.WALLET_SIGNER_ISOLATION_REQUIRED,
            "WALLET_SIGNER_ISOLATION_CONFIRMED": cls.WALLET_SIGNER_ISOLATION_CONFIRMED,
            "WALLET_SDK_CHECKS_PASSED": cls.WALLET_SDK_CHECKS_PASSED,
            "WALLET_MPC_SIGNER_URL": cls.WALLET_MPC_SIGNER_URL,
            "WALLET_MPC_SIGNER_TOKEN": "[configured]" if cls.WALLET_MPC_SIGNER_TOKEN else "",
            "WALLET_SIGNER_TIMEOUT_SECONDS": cls.WALLET_SIGNER_TIMEOUT_SECONDS,
            "WALLET_EMERGENCY_STOP": cls.WALLET_EMERGENCY_STOP,
            "USE_REAL_ADDRESSES": cls.USE_REAL_ADDRESSES,
            "WALLET_REAL_CUSTODY_ENABLED": cls.WALLET_REAL_CUSTODY_ENABLED,
            "WALLET_SELF_CUSTODY_ENABLED": cls.WALLET_SELF_CUSTODY_ENABLED,
            "WALLET_ALLOW_IN_APP_KEYGEN": cls.WALLET_ALLOW_IN_APP_KEYGEN,
            "WALLET_REQUIRE_WITHDRAWAL_APPROVAL": cls.WALLET_REQUIRE_WITHDRAWAL_APPROVAL,
            "WALLET_WITHDRAWALS_ENABLED": cls.WALLET_WITHDRAWALS_ENABLED,
            "WALLET_AUTO_ENABLE_WITHDRAWALS": cls.WALLET_AUTO_ENABLE_WITHDRAWALS,
            "WALLET_AUTO_SWEEP_ENABLED": cls.WALLET_AUTO_SWEEP_ENABLED,
            "WALLET_WITHDRAWAL_FEE_BPS": cls.WALLET_WITHDRAWAL_FEE_BPS,
            "WALLET_WITHDRAWAL_FIXED_FEE_ETH": cls.WALLET_WITHDRAWAL_FIXED_FEE_ETH,
            "WALLET_WITHDRAWAL_FEE_MAX_ETH": cls.WALLET_WITHDRAWAL_FEE_MAX_ETH,
            "WALLET_EVM_RPC_URL": cls.WALLET_EVM_RPC_URL,
            "WALLET_EVM_NETWORKS": cls.WALLET_EVM_NETWORKS,
            "WALLET_EVM_TOKEN_CONTRACTS": cls.WALLET_EVM_TOKEN_CONTRACTS,
            "WALLET_EVM_MIN_GAS_PRICE_GWEI": cls.WALLET_EVM_MIN_GAS_PRICE_GWEI,
            "WALLET_BROADCAST_VERIFY_ATTEMPTS": cls.WALLET_BROADCAST_VERIFY_ATTEMPTS,
            "WALLET_BROADCAST_VERIFY_DELAY_SECONDS": cls.WALLET_BROADCAST_VERIFY_DELAY_SECONDS,
            "WALLET_REQUIRED_CONFIRMATIONS": cls.WALLET_REQUIRED_CONFIRMATIONS,
            "WALLET_BTC_INDEXER_URL": cls.WALLET_BTC_INDEXER_URL,
            "WALLET_SOLANA_RPC_URL": cls.WALLET_SOLANA_RPC_URL,
            "WALLET_XRP_RPC_URL": cls.WALLET_XRP_RPC_URL,
            "WALLET_MAX_WITHDRAWAL_BY_ASSET": cls.WALLET_MAX_WITHDRAWAL_BY_ASSET,
            "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_WALLET": cls.WALLET_DAILY_WITHDRAWAL_LIMIT_BY_WALLET,
            "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_ASSET": cls.WALLET_DAILY_WITHDRAWAL_LIMIT_BY_ASSET,
            "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_DESTINATION": cls.WALLET_DAILY_WITHDRAWAL_LIMIT_BY_DESTINATION,
            "WALLET_DAILY_GLOBAL_WITHDRAWAL_LIMIT": cls.WALLET_DAILY_GLOBAL_WITHDRAWAL_LIMIT,
            "ONRAMP_PROVIDER_ENABLED": cls.ONRAMP_PROVIDER_ENABLED,
            "ONRAMP_PROVIDER": cls.ONRAMP_PROVIDER,
            "ONRAMP_SUPPORTED_ASSETS": cls.ONRAMP_SUPPORTED_ASSETS,
            "ONRAMP_SUPPORTED_FIAT": cls.ONRAMP_SUPPORTED_FIAT,
            "ONRAMP_CUSTOM_SESSION_URL": "[configured]" if cls.ONRAMP_CUSTOM_SESSION_URL else "",
            "ONRAMP_CUSTOM_API_KEY": "[configured]" if cls.ONRAMP_CUSTOM_API_KEY else "",
            "ONRAMP_CUSTOM_WEBHOOK_SECRET": "[configured]" if cls.ONRAMP_CUSTOM_WEBHOOK_SECRET else "",
            "ONRAMP_CUSTOM_ALLOWED_ASSETS": cls.ONRAMP_CUSTOM_ALLOWED_ASSETS,
            "ONRAMP_CUSTOM_MIN_FIAT_USD": cls.ONRAMP_CUSTOM_MIN_FIAT_USD,
            "ONRAMP_CUSTOM_MAX_FIAT_USD": cls.ONRAMP_CUSTOM_MAX_FIAT_USD,
            "ONRAMP_CUSTOM_TIMEOUT_SECONDS": cls.ONRAMP_CUSTOM_TIMEOUT_SECONDS,
            "APPLE_PAY_DIRECT_ENABLED": cls.APPLE_PAY_DIRECT_ENABLED,
            "APPLE_PAY_CRYPTO_SALE_APPROVED": cls.APPLE_PAY_CRYPTO_SALE_APPROVED,
            "APPLE_PAY_MERCHANT_ID": "[configured]" if cls.APPLE_PAY_MERCHANT_ID else "",
            "APPLE_PAY_DISPLAY_NAME": cls.APPLE_PAY_DISPLAY_NAME,
            "APPLE_PAY_DOMAIN": cls.APPLE_PAY_DOMAIN,
            "APPLE_PAY_COUNTRY_CODE": cls.APPLE_PAY_COUNTRY_CODE,
            "APPLE_PAY_SUPPORTED_NETWORKS": cls.APPLE_PAY_SUPPORTED_NETWORKS,
            "APPLE_PAY_MERCHANT_CERT_PATH": "[configured]" if cls.APPLE_PAY_MERCHANT_CERT_PATH else "",
            "APPLE_PAY_MERCHANT_KEY_PATH": "[configured]" if cls.APPLE_PAY_MERCHANT_KEY_PATH else "",
            "APPLE_PAY_MERCHANT_CERT_PEM": "[configured]" if cls.APPLE_PAY_MERCHANT_CERT_PEM else "",
            "APPLE_PAY_MERCHANT_KEY_PEM": "[configured]" if cls.APPLE_PAY_MERCHANT_KEY_PEM else "",
            "APPLE_PAY_GATEWAY_AUTHORIZE_URL": "[configured]" if cls.APPLE_PAY_GATEWAY_AUTHORIZE_URL else "",
            "APPLE_PAY_GATEWAY_REFUND_URL": "[configured]" if cls.APPLE_PAY_GATEWAY_REFUND_URL else "",
            "APPLE_PAY_GATEWAY_API_KEY": "[configured]" if cls.APPLE_PAY_GATEWAY_API_KEY else "",
            "APPLE_PAY_GATEWAY_WEBHOOK_SECRET": "[configured]" if cls.APPLE_PAY_GATEWAY_WEBHOOK_SECRET else "",
            "APPLE_PAY_GATEWAY_WEBHOOK_TOLERANCE_SECONDS": cls.APPLE_PAY_GATEWAY_WEBHOOK_TOLERANCE_SECONDS,
            "APPLE_PAY_BUY_ALLOWED_ASSETS": cls.APPLE_PAY_BUY_ALLOWED_ASSETS,
            "APPLE_PAY_TREASURY_SOURCE_WALLETS": "[configured]" if cls.APPLE_PAY_TREASURY_SOURCE_WALLETS else {},
            "APPLE_PAY_ASSET_PRICE_USD": "[configured]" if cls.APPLE_PAY_ASSET_PRICE_USD else {},
            "APPLE_PAY_TREASURY_SOURCE_ADDRESS": "[configured]" if cls.APPLE_PAY_TREASURY_SOURCE_ADDRESS else "",
            "APPLE_PAY_TREASURY_FEE_ADDRESS": "[configured]" if cls.APPLE_PAY_TREASURY_FEE_ADDRESS else "",
            "APPLE_PAY_TREASURY_SIGNER_URL": "[configured]" if cls.APPLE_PAY_TREASURY_SIGNER_URL else "",
            "APPLE_PAY_TREASURY_SIGNER_TOKEN": "[configured]" if cls.APPLE_PAY_TREASURY_SIGNER_TOKEN else "",
            "APPLE_PAY_TREASURY_FEE_BPS": cls.APPLE_PAY_TREASURY_FEE_BPS,
            "WALLET_BUY_PLATFORM_FEE_BPS": cls.WALLET_BUY_PLATFORM_FEE_BPS,
            "WALLET_BUY_QUOTE_TTL_SECONDS": cls.WALLET_BUY_QUOTE_TTL_SECONDS,
            "CARD_BUY_ENABLED": cls.CARD_BUY_ENABLED,
            "CARD_GATEWAY_TOKENIZATION_URL": "[configured]" if cls.CARD_GATEWAY_TOKENIZATION_URL else "",
            "CARD_GATEWAY_AUTHORIZE_URL": "[configured]" if cls.CARD_GATEWAY_AUTHORIZE_URL else "",
            "CARD_GATEWAY_API_KEY": "[configured]" if cls.CARD_GATEWAY_API_KEY else "",
            "CARD_GATEWAY_WEBHOOK_SECRET": "[configured]" if cls.CARD_GATEWAY_WEBHOOK_SECRET else "",
            "CARD_GATEWAY_PUBLIC_CONFIG": cls.CARD_GATEWAY_PUBLIC_CONFIG,
            "APPLE_PAY_EXECUTION_FEE_BUFFER_BPS": cls.APPLE_PAY_EXECUTION_FEE_BUFFER_BPS,
            "APPLE_PAY_MIN_FIAT_USD": cls.APPLE_PAY_MIN_FIAT_USD,
            "APPLE_PAY_MAX_FIAT_USD": cls.APPLE_PAY_MAX_FIAT_USD,
            "APPLE_PAY_TIMEOUT_SECONDS": cls.APPLE_PAY_TIMEOUT_SECONDS,
            "APPLE_PAY_ONEINCH_SLIPPAGE_PCT": cls.APPLE_PAY_ONEINCH_SLIPPAGE_PCT,
            "APPLE_PAY_DOMAIN_ASSOCIATION": "[configured]" if cls.APPLE_PAY_DOMAIN_ASSOCIATION else "",
            "WORKER_APPLE_PAY_FULFILLMENT_ENABLED": cls.WORKER_APPLE_PAY_FULFILLMENT_ENABLED,
            "ONEINCH_API_KEY": "[configured]" if cls.ONEINCH_API_KEY else "",
            "ONEINCH_API_BASE_URL": cls.ONEINCH_API_BASE_URL,
            "ONEINCH_REFERRER_ADDRESS": "[configured]" if cls.ONEINCH_REFERRER_ADDRESS else "",
            "ONEINCH_PARTNER_FEE_BPS": cls.ONEINCH_PARTNER_FEE_BPS,
            "PLATFORM_GAS_TREASURY_ENABLED": cls.PLATFORM_GAS_TREASURY_ENABLED,
            "TREASURY_ENCRYPTION_KEY": "[configured]" if cls.TREASURY_ENCRYPTION_KEY else "",
            "PLATFORM_GAS_TOPUP_MULTIPLIER": cls.PLATFORM_GAS_TOPUP_MULTIPLIER,
            "PLATFORM_GAS_TOPUP_MAX_ETH": cls.PLATFORM_GAS_TOPUP_MAX_ETH,
            "PLATFORM_GAS_TREASURY_DAILY_LIMIT_ETH": cls.PLATFORM_GAS_TREASURY_DAILY_LIMIT_ETH,
            "PLATFORM_GAS_TREASURY_ROTATION_LOCK_ENABLED": cls.PLATFORM_GAS_TREASURY_ROTATION_LOCK_ENABLED,
            "WALLET_EVM_GAS_LIMIT_BUFFER_MULTIPLIER": cls.WALLET_EVM_GAS_LIMIT_BUFFER_MULTIPLIER,
            "WALLET_EVM_FALLBACK_ETH_GAS_LIMIT": cls.WALLET_EVM_FALLBACK_ETH_GAS_LIMIT,
            "WALLET_EVM_FALLBACK_ERC20_GAS_LIMIT": cls.WALLET_EVM_FALLBACK_ERC20_GAS_LIMIT,
            "WALLET_EVM_ESTIMATE_FROM_ADDRESS": cls.WALLET_EVM_ESTIMATE_FROM_ADDRESS,
            "PLATFORM_TREASURY_CONVERSION_PROVIDER": cls.PLATFORM_TREASURY_CONVERSION_PROVIDER,
            "PLATFORM_TREASURY_CONVERSION_USER_ID": cls.PLATFORM_TREASURY_CONVERSION_USER_ID,
            "PLATFORM_TREASURY_CONVERSION_CONNECTION_ID": cls.PLATFORM_TREASURY_CONVERSION_CONNECTION_ID,
            "PLATFORM_TREASURY_ETH_USD_FALLBACK": cls.PLATFORM_TREASURY_ETH_USD_FALLBACK,
            "PLATFORM_TREASURY_RESERVE_JOB_MAX_ATTEMPTS": cls.PLATFORM_TREASURY_RESERVE_JOB_MAX_ATTEMPTS,
            "PLATFORM_TREASURY_WITHDRAWAL_QUEUE_LIMIT": cls.PLATFORM_TREASURY_WITHDRAWAL_QUEUE_LIMIT,
            "PLATFORM_TREASURY_DEFAULT_PROFIT_SHARE_PCT": cls.PLATFORM_TREASURY_DEFAULT_PROFIT_SHARE_PCT,
            "TREASURY_SOLVENCY_ENABLED": cls.TREASURY_SOLVENCY_ENABLED,
            "TREASURY_SOLVENCY_RECALC_INTERVAL_SECONDS": cls.TREASURY_SOLVENCY_RECALC_INTERVAL_SECONDS,
            "TREASURY_HEALTHY_RATIO": cls.TREASURY_HEALTHY_RATIO,
            "TREASURY_WARNING_RATIO": cls.TREASURY_WARNING_RATIO,
            "TREASURY_LOW_RATIO": cls.TREASURY_LOW_RATIO,
            "TREASURY_SOLVENCY_WITHDRAWAL_MIN_RATIO": cls.TREASURY_SOLVENCY_WITHDRAWAL_MIN_RATIO,
            "TREASURY_SOLVENCY_BASE_SAFETY_MULTIPLIER": cls.TREASURY_SOLVENCY_BASE_SAFETY_MULTIPLIER,
            "TREASURY_GAS_CONGESTION_MULTIPLIER_CAP": cls.TREASURY_GAS_CONGESTION_MULTIPLIER_CAP,
            "TREASURY_GAS_VOLATILITY_WINDOW_HOURS": cls.TREASURY_GAS_VOLATILITY_WINDOW_HOURS,
            "TREASURY_GAS_VOLATILITY_CAP": cls.TREASURY_GAS_VOLATILITY_CAP,
            "TREASURY_FORECAST_HISTORY_HOURS": cls.TREASURY_FORECAST_HISTORY_HOURS,
            "TREASURY_FORECAST_WINDOWS_HOURS": cls.TREASURY_FORECAST_WINDOWS_HOURS,
            "TREASURY_FORECAST_RETENTION_ROWS": cls.TREASURY_FORECAST_RETENTION_ROWS,
            "TREASURY_REBALANCE_TARGET_RATIO": cls.TREASURY_REBALANCE_TARGET_RATIO,
            "TREASURY_REBALANCE_SOURCE_ASSET": cls.TREASURY_REBALANCE_SOURCE_ASSET,
            "TREASURY_REBALANCE_MIN_ETH": cls.TREASURY_REBALANCE_MIN_ETH,
            "TREASURY_REBALANCE_MAX_ETH": cls.TREASURY_REBALANCE_MAX_ETH,
            "TREASURY_REBALANCE_MAX_SLIPPAGE_BPS": cls.TREASURY_REBALANCE_MAX_SLIPPAGE_BPS,
            "TREASURY_DEX_CONVERSIONS_ENABLED": cls.TREASURY_DEX_CONVERSIONS_ENABLED,
        }


class TestConfig(BaseConfig):
    """Testing configuration with an in-memory database."""

    TESTING = True
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    ENABLE_LIVE_TRADING = False
    APP_MODE = "paper"


class ProductionConfig(BaseConfig):
    """Production defaults for hosted deployments behind HTTPS."""

    DEPLOYMENT_TARGET = os.getenv("DEPLOYMENT_TARGET", "vps").strip().lower() or "vps"
    PUBLIC_APP_ORIGIN = _clean_public_origin(os.getenv("PUBLIC_APP_ORIGIN"), "https://app.algvault.com")
    PUBLIC_API_ORIGIN = _clean_public_origin(os.getenv("PUBLIC_API_ORIGIN"), PUBLIC_APP_ORIGIN)
    PUBLIC_LIVE_API_ORIGIN = _clean_public_origin(os.getenv("PUBLIC_LIVE_API_ORIGIN"), "")
    LIVE_API_CORS_ALLOWED_ORIGINS = _parse_origin_list(
        os.getenv("LIVE_API_CORS_ALLOWED_ORIGINS"),
        (PUBLIC_APP_ORIGIN, "https://algorithm-vault-chi.vercel.app"),
    )
    PREFERRED_URL_SCHEME = os.getenv("PREFERRED_URL_SCHEME", "https").strip() or "https"
    SESSION_COOKIE_SECURE = _as_bool(os.getenv("SESSION_COOKIE_SECURE"), default=True)
    SESSION_COOKIE_HTTPONLY = _as_bool(os.getenv("SESSION_COOKIE_HTTPONLY"), default=True)
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax").strip() or "Lax"
    SESSION_COOKIE_DOMAIN = os.getenv("SESSION_COOKIE_DOMAIN", "").strip() or None
    PROXY_FIX_ENABLED = _as_bool(os.getenv("PROXY_FIX_ENABLED"), default=True)
    SECURE_HEADERS_HSTS_ENABLED = _as_bool(os.getenv("SECURE_HEADERS_HSTS_ENABLED"), default=True)
    LOG_FORMAT = os.getenv("LOG_FORMAT", "json").strip().lower() or "json"
    WEB_CONCURRENCY = _as_int(os.getenv("WEB_CONCURRENCY") or os.getenv("GUNICORN_WORKERS"), 1)
    GUNICORN_THREADS = _as_int(os.getenv("GUNICORN_THREADS"), 4)
    SCHEMA_BOOTSTRAP_ENABLED = (
        False if BaseConfig.RECOVERY_SQLITE_ACTIVE else _as_bool(os.getenv("SCHEMA_BOOTSTRAP_ENABLED"), default=False)
    )
    ENABLE_IN_PROCESS_WORKERS = (
        False if BaseConfig.RECOVERY_SQLITE_ACTIVE else _as_bool(os.getenv("ENABLE_IN_PROCESS_WORKERS"), default=False)
    )


def selected_config_class() -> type[BaseConfig]:
    """Choose the default config object without exposing environment values."""

    explicit = (os.getenv("APP_ENV") or os.getenv("FLASK_ENV") or os.getenv("FLASK_CONFIG") or "").strip().lower()
    target = os.getenv("DEPLOYMENT_TARGET", "").strip().lower()
    if explicit in {"production", "prod"} or target in {"vps", "production", "postgres", "vercel"}:
        return ProductionConfig
    if explicit in {"testing", "test"}:
        return TestConfig
    return BaseConfig
