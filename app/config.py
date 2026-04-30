"""Application configuration and environment parsing."""

from __future__ import annotations

import json
import os
from typing import Any

try:
    from hyperliquid.utils import constants as hl_constants
except Exception:  # pragma: no cover - import fallback for environments without the SDK
    class _FallbackConstants:
        TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"
        MAINNET_API_URL = "https://api.hyperliquid.xyz"

    hl_constants = _FallbackConstants()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    return "live"


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
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin").strip() or "admin"
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()
    SIGNUP_INVITE_CODE = os.getenv("SIGNUP_INVITE_CODE", "").strip()
    TOTP_ENCRYPTION_KEY = os.getenv("TOTP_ENCRYPTION_KEY", "").strip()
    DEPOSIT_ADDRESS_BOOK = _parse_deposit_address_book(os.getenv("DEPOSIT_ADDRESSES_JSON"))
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///hyperliquid_dashboard.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLITE_BUSY_TIMEOUT_MS = _as_int(os.getenv("SQLITE_BUSY_TIMEOUT_MS"), 10_000)
    SQLITE_ENABLE_WAL = _as_bool(os.getenv("SQLITE_ENABLE_WAL"), default=True)

    ENABLE_LIVE_TRADING = _as_bool(os.getenv("ENABLE_LIVE_TRADING"), default=True)
    APP_MODE = _normalize_mode(os.getenv("APP_MODE", "live"), enable_live=ENABLE_LIVE_TRADING)
    HL_TESTNET_BASE_URL = hl_constants.TESTNET_API_URL
    HL_MAINNET_BASE_URL = hl_constants.MAINNET_API_URL
    HL_WS_MAINNET_URL = os.getenv("HL_WS_MAINNET_URL", "wss://api.hyperliquid.xyz/ws").strip()
    HL_WS_TESTNET_URL = os.getenv("HL_WS_TESTNET_URL", "wss://api.hyperliquid-testnet.xyz/ws").strip()

    HL_ACCOUNT_ADDRESS = os.getenv("HL_ACCOUNT_ADDRESS", "").strip()
    HL_SECRET_KEY = os.getenv("HL_SECRET_KEY", "").strip()
    HL_VAULT_ADDRESS = os.getenv("HL_VAULT_ADDRESS", "").strip() or None
    HL_TIMEOUT_SECONDS = _as_float(os.getenv("HL_TIMEOUT_SECONDS"), 10.0)
    EXCHANGE_RETRY_ATTEMPTS = _as_int(os.getenv("EXCHANGE_RETRY_ATTEMPTS"), 3)
    EXCHANGE_RETRY_SLEEP_SECONDS = _as_float(os.getenv("EXCHANGE_RETRY_SLEEP_SECONDS"), 0.5)
    PROVIDER_TIMEOUT_SECONDS = _as_float(os.getenv("PROVIDER_TIMEOUT_SECONDS"), 10.0)
    BINANCE_FUTURES_BASE_URL = os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com").strip()
    BINANCE_RECV_WINDOW_MS = _as_int(os.getenv("BINANCE_RECV_WINDOW_MS"), 5000)
    BINANCE_SYMBOL_MAP_JSON = os.getenv("BINANCE_SYMBOL_MAP_JSON", "").strip()
    KUCOIN_FUTURES_BASE_URL = os.getenv("KUCOIN_FUTURES_BASE_URL", "https://api-futures.kucoin.com").strip()
    KUCOIN_ACCOUNT_OVERVIEW_PATH = os.getenv("KUCOIN_ACCOUNT_OVERVIEW_PATH", "/api/v1/account-overview").strip()
    KUCOIN_ORDERS_PATH = os.getenv("KUCOIN_ORDERS_PATH", "/api/v1/orders").strip()
    KUCOIN_OPEN_ORDERS_PATH = os.getenv("KUCOIN_OPEN_ORDERS_PATH", "/api/v1/orders").strip()
    KUCOIN_POSITIONS_PATH = os.getenv("KUCOIN_POSITIONS_PATH", "/api/v1/positions").strip()
    KUCOIN_FILLS_PATH = os.getenv("KUCOIN_FILLS_PATH", "/api/v1/recentDoneOrders").strip()
    KUCOIN_SYMBOL_MAP_JSON = os.getenv("KUCOIN_SYMBOL_MAP_JSON", "").strip()
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
    PAPER_BALANCE_MIN = _as_float(os.getenv("PAPER_BALANCE_MIN"), 0.0)
    PAPER_BALANCE_MAX = _as_float(os.getenv("PAPER_BALANCE_MAX"), 1_000_000.0)
    MAX_DAILY_LOSS_USDC = _as_float(os.getenv("MAX_DAILY_LOSS_USDC"), 5.0)
    MAX_BACKTEST_DRAWDOWN_PCT = _as_float(os.getenv("MAX_BACKTEST_DRAWDOWN_PCT"), 0.2)
    MAX_LEVERAGE = _as_float(os.getenv("MAX_LEVERAGE"), 3.0)
    MAX_POSITION_NOTIONAL = _as_float(os.getenv("MAX_POSITION_NOTIONAL"), 20.0)
    MAX_SLIPPAGE_PCT = _as_float(os.getenv("MAX_SLIPPAGE_PCT"), 0.015)
    LOSS_COOLDOWN_MINUTES = _as_int(os.getenv("LOSS_COOLDOWN_MINUTES"), 30)
    LOSS_STREAK_COOLDOWN_THRESHOLD = _as_int(os.getenv("LOSS_STREAK_COOLDOWN_THRESHOLD"), 3)
    MAX_TRADES_PER_WINDOW = _as_int(os.getenv("MAX_TRADES_PER_WINDOW"), 20)
    TRADE_WINDOW_MINUTES = _as_int(os.getenv("TRADE_WINDOW_MINUTES"), 60)
    RISK_PER_TRADE_PCT = _as_float(os.getenv("RISK_PER_TRADE_PCT"), 0.01)
    MIN_REWARD_RISK = _as_float(os.getenv("MIN_REWARD_RISK"), 1.0)
    FIXED_DOLLAR_SIZE = _as_float(os.getenv("FIXED_DOLLAR_SIZE"), 25.0)
    STRATEGY_POLL_SECONDS = _as_int(os.getenv("STRATEGY_POLL_SECONDS"), 20)
    FEE_BPS = _as_float(os.getenv("FEE_BPS"), 5.0)
    SIM_SLIPPAGE_BPS = _as_float(os.getenv("SIM_SLIPPAGE_BPS"), 8.0)
    DASHBOARD_CANDLE_LIMIT = _as_int(os.getenv("DASHBOARD_CANDLE_LIMIT"), 200)
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
    OPTIMIZER_PREFILTER_ENABLED = _as_bool(os.getenv("OPTIMIZER_PREFILTER_ENABLED"), default=True)
    MARKET_DATA_CACHE_TTL_SECONDS = _as_float(os.getenv("MARKET_DATA_CACHE_TTL_SECONDS"), 5.0)
    MARKET_DATA_RESEARCH_CACHE_TTL_SECONDS = _as_float(os.getenv("MARKET_DATA_RESEARCH_CACHE_TTL_SECONDS"), 60.0)
    NET_ROI_MIN_EDGE_BPS = _as_float(os.getenv("NET_ROI_MIN_EDGE_BPS"), 4.0)
    NET_ROI_MAX_CHURN_PENALTY = _as_float(os.getenv("NET_ROI_MAX_CHURN_PENALTY"), 0.35)
    NET_ROI_MIN_FILL_QUALITY = _as_float(os.getenv("NET_ROI_MIN_FILL_QUALITY"), 0.55)
    NET_ROI_V2_ENABLED = _as_bool(os.getenv("NET_ROI_V2_ENABLED"), default=True)
    NET_ROI_V2_MIN_QUALITY_GRADE = os.getenv("NET_ROI_V2_MIN_QUALITY_GRADE", "B").strip().upper() or "B"
    SIGNAL_DEBOUNCE_SECONDS = _as_float(os.getenv("SIGNAL_DEBOUNCE_SECONDS"), 45.0)
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
    EXPERIMENTAL_DURATION_ENSEMBLE_PRIMARY_METRIC = os.getenv(
        "EXPERIMENTAL_DURATION_ENSEMBLE_PRIMARY_METRIC",
        "net_return_after_costs",
    ).strip() or "net_return_after_costs"
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
    VAULT_MAX_ACTIVE_CYCLES = _as_int(os.getenv("VAULT_MAX_ACTIVE_CYCLES"), 6)
    VAULT_MAX_ACTIVE_CYCLES_PER_ASSET = _as_int(os.getenv("VAULT_MAX_ACTIVE_CYCLES_PER_ASSET"), 4)
    VAULT_MAX_ASSET_EXPOSURE_PCT = _as_float(os.getenv("VAULT_MAX_ASSET_EXPOSURE_PCT"), 0.75)
    VAULT_MAX_DURATION_EXPOSURE_PCT = _as_float(os.getenv("VAULT_MAX_DURATION_EXPOSURE_PCT"), 0.70)
    VAULT_MAX_STRATEGY_EXPOSURE_PCT = _as_float(os.getenv("VAULT_MAX_STRATEGY_EXPOSURE_PCT"), 0.70)
    VAULT_MAX_SYMBOL_EXPOSURE_PCT = _as_float(os.getenv("VAULT_MAX_SYMBOL_EXPOSURE_PCT"), 0.70)
    VAULT_MIN_RISK_ADJUSTED_SCORE = _as_float(os.getenv("VAULT_MIN_RISK_ADJUSTED_SCORE"), 0.0)
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
    HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD = _as_float(os.getenv("HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD"), 0.0)
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
    ML_OFFLINE_SCORE_WEIGHT = _as_float(os.getenv("ML_OFFLINE_SCORE_WEIGHT"), 0.15)
    ML_OFFLINE_MIN_TRAINING_ROWS = _as_int(os.getenv("ML_OFFLINE_MIN_TRAINING_ROWS"), 250)
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
    HIGH_UPSIDE_REQUIRE_PROMOTED_ML = _as_bool(os.getenv("HIGH_UPSIDE_REQUIRE_PROMOTED_ML"), default=True)
    ML_PREDICTION_CAP = _as_float(os.getenv("ML_PREDICTION_CAP"), 1.0)
    ML_TARGET_CAP = _as_float(os.getenv("ML_TARGET_CAP"), 1.0)
    ML_WEIGHT_CAP = _as_float(os.getenv("ML_WEIGHT_CAP"), 3.0)
    REALTIME_MARKET_ENABLED = _as_bool(os.getenv("REALTIME_MARKET_ENABLED"), default=False)
    REALTIME_MARKET_CONNECT_TIMEOUT_SECONDS = _as_float(os.getenv("REALTIME_MARKET_CONNECT_TIMEOUT_SECONDS"), 3.0)
    REALTIME_MARKET_MAX_STALE_SECONDS = _as_float(os.getenv("REALTIME_MARKET_MAX_STALE_SECONDS"), 15.0)
    WALLET_PROVIDER = os.getenv("WALLET_PROVIDER", "self_custody").strip() or "self_custody"
    USE_REAL_ADDRESSES = _as_bool(os.getenv("USE_REAL_ADDRESSES"), default=False)
    WALLET_REAL_CUSTODY_ENABLED = _as_bool(os.getenv("WALLET_REAL_CUSTODY_ENABLED"), default=False)
    WALLET_SELF_CUSTODY_ENABLED = _as_bool(os.getenv("WALLET_SELF_CUSTODY_ENABLED"), default=False)
    WALLET_ALLOW_IN_APP_KEYGEN = _as_bool(os.getenv("WALLET_ALLOW_IN_APP_KEYGEN"), default=False)
    WALLET_REQUIRE_WITHDRAWAL_APPROVAL = _as_bool(os.getenv("WALLET_REQUIRE_WITHDRAWAL_APPROVAL"), default=True)
    WALLET_WITHDRAWALS_ENABLED = _as_bool(os.getenv("WALLET_WITHDRAWALS_ENABLED"), default=False)
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
    WALLET_REQUIRED_CONFIRMATIONS = _parse_float_map(os.getenv("WALLET_REQUIRED_CONFIRMATIONS_JSON"))
    WALLET_BTC_INDEXER_URL = os.getenv("WALLET_BTC_INDEXER_URL", "").strip()
    WALLET_SOLANA_RPC_URL = os.getenv("WALLET_SOLANA_RPC_URL", "").strip()
    WALLET_XRP_RPC_URL = os.getenv("WALLET_XRP_RPC_URL", "").strip()
    WALLET_MAX_WITHDRAWAL_BY_ASSET = _parse_float_map(os.getenv("WALLET_MAX_WITHDRAWAL_BY_ASSET_JSON"))
    WALLET_PRICE_FEED_URL = os.getenv("WALLET_PRICE_FEED_URL", "").strip()

    WTF_CSRF_ENABLED = True
    TESTING = False

    @classmethod
    def export_defaults(cls) -> dict[str, Any]:
        """Return a serializable view of default values used in docs and tests."""

        return {
            "APP_MODE": cls.APP_MODE,
            "ENABLE_LIVE_TRADING": cls.ENABLE_LIVE_TRADING,
            "SQLITE_BUSY_TIMEOUT_MS": cls.SQLITE_BUSY_TIMEOUT_MS,
            "SQLITE_ENABLE_WAL": cls.SQLITE_ENABLE_WAL,
            "ALLOWED_SYMBOLS": list(cls.ALLOWED_SYMBOLS),
            "DEFAULT_TIMEFRAME": cls.DEFAULT_TIMEFRAME,
            "DEFAULT_PAPER_BALANCE": cls.DEFAULT_PAPER_BALANCE,
            "PAPER_BALANCE_MIN": cls.PAPER_BALANCE_MIN,
            "PAPER_BALANCE_MAX": cls.PAPER_BALANCE_MAX,
            "MAX_DAILY_LOSS_USDC": cls.MAX_DAILY_LOSS_USDC,
            "MAX_BACKTEST_DRAWDOWN_PCT": cls.MAX_BACKTEST_DRAWDOWN_PCT,
            "MAX_LEVERAGE": cls.MAX_LEVERAGE,
            "MAX_POSITION_NOTIONAL": cls.MAX_POSITION_NOTIONAL,
            "MAX_SLIPPAGE_PCT": cls.MAX_SLIPPAGE_PCT,
            "LOSS_COOLDOWN_MINUTES": cls.LOSS_COOLDOWN_MINUTES,
            "LOSS_STREAK_COOLDOWN_THRESHOLD": cls.LOSS_STREAK_COOLDOWN_THRESHOLD,
            "MAX_TRADES_PER_WINDOW": cls.MAX_TRADES_PER_WINDOW,
            "TRADE_WINDOW_MINUTES": cls.TRADE_WINDOW_MINUTES,
            "RISK_PER_TRADE_PCT": cls.RISK_PER_TRADE_PCT,
            "MIN_REWARD_RISK": cls.MIN_REWARD_RISK,
            "FIXED_DOLLAR_SIZE": cls.FIXED_DOLLAR_SIZE,
            "STRATEGY_POLL_SECONDS": cls.STRATEGY_POLL_SECONDS,
            "FEE_BPS": cls.FEE_BPS,
            "SIM_SLIPPAGE_BPS": cls.SIM_SLIPPAGE_BPS,
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
            "OPTIMIZER_PREFILTER_ENABLED": cls.OPTIMIZER_PREFILTER_ENABLED,
            "MARKET_DATA_CACHE_TTL_SECONDS": cls.MARKET_DATA_CACHE_TTL_SECONDS,
            "MARKET_DATA_RESEARCH_CACHE_TTL_SECONDS": cls.MARKET_DATA_RESEARCH_CACHE_TTL_SECONDS,
            "NET_ROI_MIN_EDGE_BPS": cls.NET_ROI_MIN_EDGE_BPS,
            "NET_ROI_MAX_CHURN_PENALTY": cls.NET_ROI_MAX_CHURN_PENALTY,
            "NET_ROI_MIN_FILL_QUALITY": cls.NET_ROI_MIN_FILL_QUALITY,
            "NET_ROI_V2_ENABLED": cls.NET_ROI_V2_ENABLED,
            "NET_ROI_V2_MIN_QUALITY_GRADE": cls.NET_ROI_V2_MIN_QUALITY_GRADE,
            "SIGNAL_DEBOUNCE_SECONDS": cls.SIGNAL_DEBOUNCE_SECONDS,
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
            "VAULT_MAX_ACTIVE_CYCLES": cls.VAULT_MAX_ACTIVE_CYCLES,
            "VAULT_MAX_ACTIVE_CYCLES_PER_ASSET": cls.VAULT_MAX_ACTIVE_CYCLES_PER_ASSET,
            "VAULT_MAX_ASSET_EXPOSURE_PCT": cls.VAULT_MAX_ASSET_EXPOSURE_PCT,
            "VAULT_MAX_DURATION_EXPOSURE_PCT": cls.VAULT_MAX_DURATION_EXPOSURE_PCT,
            "VAULT_MAX_STRATEGY_EXPOSURE_PCT": cls.VAULT_MAX_STRATEGY_EXPOSURE_PCT,
            "VAULT_MAX_SYMBOL_EXPOSURE_PCT": cls.VAULT_MAX_SYMBOL_EXPOSURE_PCT,
            "VAULT_MIN_RISK_ADJUSTED_SCORE": cls.VAULT_MIN_RISK_ADJUSTED_SCORE,
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
            "HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD": cls.HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD,
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
            "ML_OFFLINE_SCORE_WEIGHT": cls.ML_OFFLINE_SCORE_WEIGHT,
            "ML_OFFLINE_MIN_TRAINING_ROWS": cls.ML_OFFLINE_MIN_TRAINING_ROWS,
            "ML_OFFLINE_MAX_VALIDATION_LOSS": cls.ML_OFFLINE_MAX_VALIDATION_LOSS,
            "ML_OFFLINE_MAX_NEGATIVE_ERROR_RATE": cls.ML_OFFLINE_MAX_NEGATIVE_ERROR_RATE,
            "ML_OFFLINE_MAX_MODEL_AGE_HOURS": cls.ML_OFFLINE_MAX_MODEL_AGE_HOURS,
            "ML_OFFLINE_MAX_DRIFT": cls.ML_OFFLINE_MAX_DRIFT,
            "ML_OFFLINE_MIN_TOP_DECILE_PRECISION": cls.ML_OFFLINE_MIN_TOP_DECILE_PRECISION,
            "ML_OFFLINE_MAX_FALSE_POSITIVE_HIGH_UPSIDE_RATE": cls.ML_OFFLINE_MAX_FALSE_POSITIVE_HIGH_UPSIDE_RATE,
            "ML_OFFLINE_MAX_CALIBRATION_ERROR": cls.ML_OFFLINE_MAX_CALIBRATION_ERROR,
            "HIGH_UPSIDE_REQUIRE_PROMOTED_ML": cls.HIGH_UPSIDE_REQUIRE_PROMOTED_ML,
            "ML_PREDICTION_CAP": cls.ML_PREDICTION_CAP,
            "ML_TARGET_CAP": cls.ML_TARGET_CAP,
            "ML_WEIGHT_CAP": cls.ML_WEIGHT_CAP,
            "REALTIME_MARKET_ENABLED": cls.REALTIME_MARKET_ENABLED,
            "REALTIME_MARKET_MAX_STALE_SECONDS": cls.REALTIME_MARKET_MAX_STALE_SECONDS,
            "WALLET_PROVIDER": cls.WALLET_PROVIDER,
            "USE_REAL_ADDRESSES": cls.USE_REAL_ADDRESSES,
            "WALLET_REAL_CUSTODY_ENABLED": cls.WALLET_REAL_CUSTODY_ENABLED,
            "WALLET_SELF_CUSTODY_ENABLED": cls.WALLET_SELF_CUSTODY_ENABLED,
            "WALLET_ALLOW_IN_APP_KEYGEN": cls.WALLET_ALLOW_IN_APP_KEYGEN,
            "WALLET_REQUIRE_WITHDRAWAL_APPROVAL": cls.WALLET_REQUIRE_WITHDRAWAL_APPROVAL,
            "WALLET_WITHDRAWALS_ENABLED": cls.WALLET_WITHDRAWALS_ENABLED,
            "WALLET_AUTO_SWEEP_ENABLED": cls.WALLET_AUTO_SWEEP_ENABLED,
            "WALLET_WITHDRAWAL_FEE_BPS": cls.WALLET_WITHDRAWAL_FEE_BPS,
            "WALLET_WITHDRAWAL_FIXED_FEE_ETH": cls.WALLET_WITHDRAWAL_FIXED_FEE_ETH,
            "WALLET_WITHDRAWAL_FEE_MAX_ETH": cls.WALLET_WITHDRAWAL_FEE_MAX_ETH,
            "WALLET_EVM_RPC_URL": cls.WALLET_EVM_RPC_URL,
            "WALLET_EVM_NETWORKS": cls.WALLET_EVM_NETWORKS,
            "WALLET_EVM_TOKEN_CONTRACTS": cls.WALLET_EVM_TOKEN_CONTRACTS,
            "WALLET_REQUIRED_CONFIRMATIONS": cls.WALLET_REQUIRED_CONFIRMATIONS,
            "WALLET_BTC_INDEXER_URL": cls.WALLET_BTC_INDEXER_URL,
            "WALLET_SOLANA_RPC_URL": cls.WALLET_SOLANA_RPC_URL,
            "WALLET_XRP_RPC_URL": cls.WALLET_XRP_RPC_URL,
            "WALLET_MAX_WITHDRAWAL_BY_ASSET": cls.WALLET_MAX_WITHDRAWAL_BY_ASSET,
        }


class TestConfig(BaseConfig):
    """Testing configuration with an in-memory database."""

    TESTING = True
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    ENABLE_LIVE_TRADING = False
    APP_MODE = "live"
