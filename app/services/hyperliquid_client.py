"""Official Hyperliquid SDK wrapper with retries and safe defaults."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from typing import Any

import requests

logger = logging.getLogger(__name__)

HYPERLIQUID_TESTNET_API_URL = "https://api.hyperliquid-testnet.xyz"
HYPERLIQUID_MAINNET_API_URL = "https://api.hyperliquid.xyz"
HYPERLIQUID_TEST_ACCOUNT = "sufyanh"
LIVE_TEST_CLIENT_ORDER_PREFIX = "codex-hl-test-"
VALID_MODES = {"testnet", "live"}
VALID_SIDES = {"buy", "sell"}
VALID_ORDER_TYPES = {"market", "limit"}
VALID_TIME_IN_FORCE = {"alo": "Alo", "ioc": "Ioc", "gtc": "Gtc"}
HYPERLIQUID_MIN_NOTIONAL_DEFAULT_USD = 10.0


@dataclass(slots=True)
class ClientSnapshot:
    """Normalized account snapshot for dashboard rendering."""

    mode: str
    balances: list[dict[str, Any]]
    positions: list[dict[str, Any]]
    open_orders: list[dict[str, Any]]
    recent_fills: list[dict[str, Any]]
    alerts: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class HyperliquidClient:
    """Encapsulates all Hyperliquid API access behind a safe service layer."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._public_info: dict[str, Any] = {}
        self._exchange_clients: dict[str, Any] = {}

    @property
    def retry_attempts(self) -> int:
        return max(1, self._safe_int(self.config.get("EXCHANGE_RETRY_ATTEMPTS", 3), 3))

    @property
    def retry_sleep_seconds(self) -> float:
        return max(0.0, self._safe_float(self.config.get("EXCHANGE_RETRY_SLEEP_SECONDS", 0.5), 0.5))

    def base_url_for_mode(self, mode: str) -> str:
        self._validate_mode(mode)

        if mode == "live":
            return self._normalize_base_url(self.config.get("HL_MAINNET_BASE_URL") or HYPERLIQUID_MAINNET_API_URL)

        return self._normalize_base_url(
            self.config.get("HYPERLIQUID_BASE_URL") or self.config.get("HL_TESTNET_BASE_URL") or HYPERLIQUID_TESTNET_API_URL
        )

    def has_account_address(self) -> bool:
        return bool(self.config.get("HL_ACCOUNT_ADDRESS"))

    def has_trading_credentials(self) -> bool:
        return bool(self.config.get("HL_SECRET_KEY"))

    def can_trade(self, mode: str) -> bool:
        self._validate_mode(mode)

        if mode == "live" and not self.config.get("ENABLE_LIVE_TRADING", False):
            return False
        if mode == "testnet" and self.testnet_live_test_guard_errors():
            return False

        return self.has_trading_credentials() and self.has_account_address()

    def testnet_live_test_guard_errors(self) -> list[str]:
        """Return fail-closed blockers for signed Hyperliquid testnet smoke tests."""

        errors: list[str] = []
        account = str(self.config.get("HYPERLIQUID_ACCOUNT") or "").strip()
        environment = str(self.config.get("HYPERLIQUID_ENV") or "").strip().lower()
        configured_base_url = self._normalize_base_url(self.config.get("HYPERLIQUID_BASE_URL") or "")

        if account != HYPERLIQUID_TEST_ACCOUNT:
            errors.append("HYPERLIQUID_ACCOUNT must be exactly sufyanh")
        if environment != "testnet":
            errors.append("HYPERLIQUID_ENV must be exactly testnet")
        if configured_base_url != HYPERLIQUID_TESTNET_API_URL:
            errors.append(f"HYPERLIQUID_BASE_URL must be exactly {HYPERLIQUID_TESTNET_API_URL}")
        if self.base_url_for_mode("testnet") != HYPERLIQUID_TESTNET_API_URL:
            errors.append("effective Hyperliquid testnet base URL is not the official testnet URL")
        if not bool(self.config.get("RUN_HYPERLIQUID_LIVE_TESTS", False)):
            errors.append("RUN_HYPERLIQUID_LIVE_TESTS=1 is required")
        if not self.has_account_address():
            errors.append("HL_ACCOUNT_ADDRESS is required for account queries")
        if not self.has_trading_credentials():
            errors.append("HL_SECRET_KEY is required for signed testnet trading")

        return errors

    def ensure_testnet_live_tests_enabled(self) -> None:
        blockers = self.testnet_live_test_guard_errors()
        if blockers:
            raise RuntimeError("hyperliquid_testnet_guard_failed: " + "; ".join(blockers))

    def ensure_testnet_account_funded(self, required_notional_usd: float) -> None:
        required = max(0.0, self._safe_float(required_notional_usd))
        balances = self.get_balances("testnet")
        available = 0.0
        account_value = 0.0
        for row in balances:
            if str(row.get("asset") or "").upper() != "USDC" or str(row.get("type") or "").lower() != "margin":
                continue
            available = max(available, self._safe_float(row.get("withdrawable")))
            account_value = max(account_value, self._safe_float(row.get("value")))

        if max(available, account_value) + 1e-9 < required:
            raise RuntimeError(
                "insufficient_testnet_funds: Hyperliquid testnet USDC margin is below "
                f"the required tiny smoke-test notional ({max(available, account_value):.6f} < {required:.6f})."
            )

    def account_snapshot(self, mode: str) -> ClientSnapshot:
        self._validate_mode(mode)

        alerts: list[str] = []

        if not self.has_account_address():
            alerts.append("No Hyperliquid account address configured; exchange views are read-only.")
            return ClientSnapshot(mode, [], [], [], [], alerts)

        try:
            retry = mode != "live"
            balances = self.get_balances(mode, retry=retry)
            positions = self.get_positions(mode, retry=retry)
            open_orders = self.get_open_orders(mode, retry=retry)
            recent_fills = self.get_recent_fills(mode, retry=retry)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to build Hyperliquid account snapshot for %s: %s", mode, exc)
            alerts.append(f"Exchange data unavailable: {exc}")
            balances, positions, open_orders, recent_fills = [], [], [], []

        if mode == "live" and not self.config.get("ENABLE_LIVE_TRADING", False):
            alerts.append("Live trading is disabled by environment configuration.")

        if mode in {"testnet", "live"} and not self.has_trading_credentials():
            alerts.append("Trading credentials are missing; order submission is disabled.")

        return ClientSnapshot(mode, balances, positions, open_orders, recent_fills, alerts)

    def get_balances(self, mode: str, *, retry: bool = True) -> list[dict[str, Any]]:
        info = self._get_public_info(mode)
        address = self._account_address()

        user_state = self._retry(
            lambda: info.user_state(address),
            f"{mode} user_state",
            attempts=self.retry_attempts if retry else 1,
            log_warnings=retry,
            wrap_error=retry,
        )
        spot_state = self._retry(
            lambda: info.spot_user_state(address),
            f"{mode} spot_user_state",
            attempts=self.retry_attempts if retry else 1,
            log_warnings=retry,
            wrap_error=retry,
        )

        margin_summary = user_state.get("marginSummary", {})

        balances = [
            {
                "asset": "USDC",
                "type": "margin",
                "value": self._safe_float(margin_summary.get("accountValue")),
                "withdrawable": self._safe_float(user_state.get("withdrawable")),
            }
        ]

        for balance in spot_state.get("balances", []):
            balances.append(
                {
                    "asset": balance.get("coin", balance.get("token", "spot")),
                    "type": "spot",
                    "value": self._safe_float(balance.get("total", balance.get("holdTotal"))),
                    "withdrawable": self._safe_float(balance.get("total")),
                }
            )

        return balances

    def get_positions(self, mode: str, *, retry: bool = True) -> list[dict[str, Any]]:
        info = self._get_public_info(mode)
        address = self._account_address()

        user_state = self._retry(
            lambda: info.user_state(address),
            f"{mode} positions",
            attempts=self.retry_attempts if retry else 1,
            log_warnings=retry,
            wrap_error=retry,
        )
        positions: list[dict[str, Any]] = []

        for item in user_state.get("assetPositions", []):
            position = item.get("position", {})
            quantity = self._safe_float(position.get("szi"))

            positions.append(
                {
                    "symbol": position.get("coin"),
                    "quantity": quantity,
                    "side": "long" if quantity > 0 else "short" if quantity < 0 else "flat",
                    "entry_price": self._safe_float(position.get("entryPx")),
                    "mark_value": self._safe_float(position.get("positionValue")),
                    "unrealized_pnl": self._safe_float(position.get("unrealizedPnl")),
                    "leverage": self._safe_float(position.get("leverage", {}).get("value"), 1.0),
                    "liquidation_price": self._safe_float(position.get("liquidationPx")),
                }
            )

        return positions

    def get_open_orders(self, mode: str, *, retry: bool = True) -> list[dict[str, Any]]:
        info = self._get_public_info(mode)
        address = self._account_address()

        orders = self._retry(
            lambda: info.open_orders(address),
            f"{mode} open_orders",
            attempts=self.retry_attempts if retry else 1,
            log_warnings=retry,
            wrap_error=retry,
        )

        return [
            {
                "symbol": order.get("coin"),
                "order_id": order.get("oid"),
                "side": self._normalize_side(order.get("side", order.get("dir", "B"))),
                "price": self._safe_float(order.get("limitPx", order.get("px"))),
                "size": self._safe_float(order.get("sz", order.get("origSz"))),
                "timestamp": order.get("timestamp"),
                "reduce_only": bool(order.get("reduceOnly", False)),
                "raw": order,
            }
            for order in orders
        ]

    def get_recent_fills(self, mode: str, *, retry: bool = True) -> list[dict[str, Any]]:
        info = self._get_public_info(mode)
        address = self._account_address()

        fills = self._retry(
            lambda: info.user_fills(address),
            f"{mode} user_fills",
            attempts=self.retry_attempts if retry else 1,
            log_warnings=retry,
            wrap_error=retry,
        )

        return [
            {
                "symbol": fill.get("coin"),
                "side": self._normalize_side(fill.get("side")),
                "price": self._safe_float(fill.get("px")),
                "size": self._safe_float(fill.get("sz")),
                "fee": self._safe_float(fill.get("fee")) if fill.get("fee") is not None else None,
                "fee_token": fill.get("feeToken"),
                "closed_pnl": self._safe_float(fill.get("closedPnl")) if fill.get("closedPnl") is not None else None,
                "exchange_order_id": str(fill.get("oid") or ""),
                "exchange_fill_id": str(fill.get("tid") or fill.get("hash") or ""),
                "timestamp": fill.get("time"),
                "direction": fill.get("dir"),
                "raw": fill,
            }
            for fill in fills[:25]
        ]

    def get_all_mids(self, mode: str, *, retry: bool = True) -> dict[str, float]:
        try:
            info = self._get_public_info(mode)
            mids = self._retry(
                lambda: info.all_mids(),
                f"{mode} all_mids",
                attempts=self.retry_attempts if retry else 1,
                log_warnings=retry,
                wrap_error=retry,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Hyperliquid SDK all_mids failed for %s; trying direct info fallback: %s", mode, self._sanitize_error_message(exc)
            )
            mids = self._public_info_post(
                mode,
                {"type": "allMids"},
                context=f"{mode} direct allMids",
                attempts=self.retry_attempts if retry else 1,
                log_warnings=retry,
            )

        return {str(symbol): price for symbol, raw_price in mids.items() if (price := self._safe_float(raw_price)) > 0}

    def get_perp_meta(self, mode: str) -> dict[str, Any]:
        """Return Hyperliquid perpetual metadata from the official info endpoint."""

        payload: Any = None
        try:
            info = self._get_public_info(mode)
            if hasattr(info, "meta"):
                payload = self._retry(lambda: info.meta(), f"{mode} meta")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hyperliquid SDK meta failed for %s; trying direct info fallback: %s", mode, self._sanitize_error_message(exc))
        if payload is None:
            payload = self._public_info_post(mode, {"type": "meta"}, context=f"{mode} direct meta")
        return payload if isinstance(payload, dict) else {}

    def get_perp_meta_and_asset_contexts(self, mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Return perpetual metadata and asset contexts when supported by the SDK."""

        payload: Any = None
        try:
            info = self._get_public_info(mode)
            if hasattr(info, "meta_and_asset_ctxs"):
                payload = self._retry(lambda: info.meta_and_asset_ctxs(), f"{mode} meta_and_asset_ctxs")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Hyperliquid SDK meta_and_asset_ctxs failed for %s; trying direct info fallback: %s",
                mode,
                self._sanitize_error_message(exc),
            )
        if payload is None:
            payload = self._public_info_post(mode, {"type": "metaAndAssetCtxs"}, context=f"{mode} direct metaAndAssetCtxs")
        if isinstance(payload, (list, tuple)) and len(payload) >= 2:
            meta = payload[0] if isinstance(payload[0], dict) else {}
            contexts = payload[1] if isinstance(payload[1], list) else []
            return meta, [dict(item) for item in contexts if isinstance(item, dict)]
        return {}, []

    def get_perp_asset_metadata(self, mode: str, symbol: str) -> dict[str, Any]:
        self._validate_symbol(symbol)
        meta, contexts = self.get_perp_meta_and_asset_contexts(mode)
        universe = meta.get("universe", []) if isinstance(meta, dict) else []
        for index, asset in enumerate(universe):
            if not isinstance(asset, dict) or asset.get("name") != symbol:
                continue
            payload = dict(asset)
            payload["asset_id"] = index
            if index < len(contexts) and isinstance(contexts[index], dict):
                payload["asset_context"] = dict(contexts[index])
            return payload
        raise ValueError(f"Hyperliquid symbol is not in perp metadata: {symbol}")

    def select_live_test_symbol(self, mode: str = "testnet", preferred: tuple[str, ...] = ("BTC", "ETH")) -> str:
        meta, _contexts = self.get_perp_meta_and_asset_contexts(mode)
        universe = meta.get("universe", []) if isinstance(meta, dict) else []
        available = [
            str(asset.get("name") or "")
            for asset in universe
            if isinstance(asset, dict) and str(asset.get("name") or "").strip() and not str(asset.get("name")).startswith(("#", "@"))
        ]
        mids = self.get_all_mids(mode)

        for symbol in preferred:
            if symbol in available and self._safe_float(mids.get(symbol)) > 0:
                return symbol

        for symbol in available:
            if self._safe_float(mids.get(symbol)) > 0:
                return symbol

        raise RuntimeError("No liquid Hyperliquid test symbol with a positive mid price is available.")

    def minimum_valid_order_size(self, mode: str, symbol: str, mid_price: float, *, min_notional_usd: float | None = None) -> float:
        self._validate_symbol(symbol)
        mid = self._require_positive_price(mid_price, symbol)
        metadata = self.get_perp_asset_metadata(mode, symbol)
        size_decimals = max(0, self._safe_int(metadata.get("szDecimals"), 0))
        notional = max(
            HYPERLIQUID_MIN_NOTIONAL_DEFAULT_USD,
            self._safe_float(self.config.get("HYPERLIQUID_MIN_ORDER_VALUE_USD"), HYPERLIQUID_MIN_NOTIONAL_DEFAULT_USD),
            self._safe_float(min_notional_usd, HYPERLIQUID_MIN_NOTIONAL_DEFAULT_USD),
        )
        step = Decimal("1").scaleb(-size_decimals)
        try:
            raw_size = Decimal(str(notional)) / Decimal(str(mid))
            normalized = raw_size.quantize(step, rounding=ROUND_CEILING)
        except (InvalidOperation, ValueError) as exc:
            raise RuntimeError(f"Unable to compute a valid Hyperliquid order size for {symbol}") from exc
        return self._safe_float(normalized)

    def normalize_limit_price(self, mode: str, symbol: str, price: float, *, round_up: bool = False) -> float:
        self._validate_symbol(symbol)
        value = self._require_positive_price(price, symbol)
        metadata = self.get_perp_asset_metadata(mode, symbol)
        size_decimals = max(0, self._safe_int(metadata.get("szDecimals"), 0))
        decimals = max(0, 6 - size_decimals)
        significant = Decimal(str(float(f"{value:.5g}")))
        step = Decimal("1").scaleb(-decimals)
        rounding = ROUND_CEILING if round_up else ROUND_FLOOR
        try:
            normalized = significant.quantize(step, rounding=rounding)
        except (InvalidOperation, ValueError) as exc:
            raise RuntimeError(f"Unable to normalize Hyperliquid limit price for {symbol}") from exc
        return self._require_positive_price(normalized, symbol)

    def live_test_order_plan(self, mode: str = "testnet", *, side: str = "buy", require_funds: bool = True) -> dict[str, Any]:
        if mode != "testnet":
            raise RuntimeError("Hyperliquid live-test order planning is testnet-only.")
        self.ensure_testnet_live_tests_enabled()
        side = str(side or "").lower()
        if side not in VALID_SIDES:
            raise ValueError(f"Unsupported order side: {side}")

        symbol = self.select_live_test_symbol(mode)
        mids = self.get_all_mids(mode)
        mid = self._require_positive_price(mids.get(symbol), symbol)
        size = self.minimum_valid_order_size(mode, symbol, mid)
        notional = size * mid
        max_notional = self._safe_float(self.config.get("HYPERLIQUID_LIVE_TEST_MAX_NOTIONAL_USD"), 0.0)
        min_notional = max(
            HYPERLIQUID_MIN_NOTIONAL_DEFAULT_USD,
            self._safe_float(self.config.get("HYPERLIQUID_MIN_ORDER_VALUE_USD"), HYPERLIQUID_MIN_NOTIONAL_DEFAULT_USD),
        )
        if max_notional > 0 and max_notional + 1e-9 < min_notional:
            raise RuntimeError(
                f"hyperliquid_test_max_notional_too_small: HYPERLIQUID_LIVE_TEST_MAX_NOTIONAL_USD must be at least {min_notional:.6f}."
            )
        if max_notional > 0 and notional > max_notional + max(mid * 1e-12, 1e-9):
            raise RuntimeError(
                "hyperliquid_test_max_notional_exceeded: computed minimum order notional "
                f"{notional:.6f} exceeds configured cap {max_notional:.6f}."
            )

        raw_price = mid * (0.90 if side == "buy" else 1.10)
        price = self.normalize_limit_price(mode, symbol, raw_price, round_up=side == "sell")
        required_notional = max(notional, min_notional)
        if require_funds:
            self.ensure_testnet_account_funded(required_notional)

        return {
            "mode": mode,
            "symbol": symbol,
            "side": side,
            "mid_price": mid,
            "limit_price": price,
            "quantity": size,
            "notional_usd": notional,
            "min_notional_usd": min_notional,
            "time_in_force": "Alo",
        }

    def get_order_book(self, mode: str, symbol: str, *, retry: bool = True) -> dict[str, Any]:
        self._validate_symbol(symbol)

        info = self._get_public_info(mode)
        return self._retry(
            lambda: info.l2_snapshot(symbol),
            f"{mode} l2_snapshot {symbol}",
            attempts=self.retry_attempts if retry else 1,
            log_warnings=retry,
            wrap_error=retry,
        )

    def get_candles(
        self,
        mode: str,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
        *,
        retry: bool = True,
    ) -> list[dict[str, Any]]:
        self._validate_symbol(symbol)

        if start_ms >= end_ms:
            raise ValueError("start_ms must be before end_ms")

        try:
            info = self._get_public_info(mode)
            candles = self._retry(
                lambda: info.candles_snapshot(symbol, timeframe, start_ms, end_ms),
                f"{mode} candles_snapshot {symbol} {timeframe}",
                attempts=self.retry_attempts if retry else 1,
                log_warnings=retry,
                wrap_error=retry,
            )
        except Exception as exc:  # noqa: BLE001
            if not retry:
                raise
            logger.warning(
                "Hyperliquid SDK candles_snapshot failed for %s %s %s; trying direct info fallback: %s",
                mode,
                symbol,
                timeframe,
                self._sanitize_error_message(exc),
            )
            candles = self._public_info_post(
                mode,
                {
                    "type": "candleSnapshot",
                    "req": {
                        "coin": symbol,
                        "interval": timeframe,
                        "startTime": start_ms,
                        "endTime": end_ms,
                    },
                },
                context=f"{mode} direct candleSnapshot {symbol} {timeframe}",
                attempts=self.retry_attempts if retry else 1,
                log_warnings=retry,
            )

        return candles if isinstance(candles, list) else []

    def place_order(
        self,
        mode: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        limit_price: float | None,
        reduce_only: bool,
        leverage: float,
        slippage_pct: float,
        *,
        client_order_id: str | None = None,
        time_in_force: str | None = None,
    ) -> dict[str, Any]:
        self._validate_mode(mode)
        self._validate_symbol(symbol)

        side = side.lower()
        order_type = order_type.lower()
        quantity = self._safe_float(quantity)
        leverage = self._safe_float(leverage, 1.0)
        slippage_pct = max(0.0, self._safe_float(slippage_pct))

        if side not in VALID_SIDES:
            raise ValueError(f"Unsupported order side: {side}")

        if order_type not in VALID_ORDER_TYPES:
            raise ValueError(f"Unsupported order type: {order_type}")

        if quantity <= 0:
            raise ValueError("quantity must be positive")

        exchange = self._get_exchange(mode)
        quantity = self._normalize_order_size(exchange, symbol, quantity)

        if quantity <= 0:
            raise ValueError(f"quantity is below Hyperliquid size precision for {symbol}")

        if order_type == "limit" and (limit_price is None or limit_price <= 0):
            raise ValueError("limit_price must be positive for limit orders")

        is_buy = side == "buy"

        if leverage > 0:
            self._retry(
                lambda: exchange.update_leverage(int(max(1, round(leverage))), symbol),
                f"{mode} update_leverage {symbol}",
            )

        price = limit_price
        tif = self._normalize_time_in_force(time_in_force) if time_in_force else "Gtc"

        if order_type == "market" or price is None:
            price = self._market_ioc_price(exchange, mode, symbol, is_buy, slippage_pct)
            tif = "Ioc"

        cloid = self._cloid_from_client_order_id(client_order_id) if client_order_id else None
        order_kwargs = {"reduce_only": reduce_only}
        if cloid is not None:
            order_kwargs["cloid"] = cloid

        response = self._retry(
            lambda: exchange.order(
                symbol,
                is_buy,
                quantity,
                price,
                {"limit": {"tif": tif}},
                **order_kwargs,
            ),
            f"{mode} order {symbol}",
        )

        return self._normalize_order_response(
            response,
            submitted_price=price,
            submitted_quantity=quantity,
            client_order_id=client_order_id,
            exchange_client_order_id=cloid.to_raw() if cloid is not None and hasattr(cloid, "to_raw") else None,
        )

    def _market_ioc_price(self, exchange: Any, mode: str, symbol: str, is_buy: bool, slippage_pct: float) -> float:
        if hasattr(exchange, "_slippage_price"):
            price = self._retry(
                lambda: exchange._slippage_price(symbol, is_buy, slippage_pct),  # noqa: SLF001
                f"{mode} slippage_price {symbol}",
            )
            return self._require_positive_price(price, symbol)

        mid = self.get_all_mids(mode).get(symbol, 0.0)

        if mid <= 0:
            raise RuntimeError(f"Mid price unavailable for {symbol}")

        direction = 1 + slippage_pct if is_buy else 1 - slippage_pct
        price = float(f"{mid * direction:.5g}")
        return self._require_positive_price(price, symbol)

    def _require_positive_price(self, price: Any, symbol: str) -> float:
        normalized = self._safe_float(price)
        if normalized <= 0:
            raise RuntimeError(f"Marketable price unavailable for {symbol}")
        return normalized

    def _normalize_order_size(self, exchange: Any, symbol: str, quantity: float) -> float:
        decimals = self._size_decimals(exchange, symbol)
        if decimals is None:
            return quantity
        decimals = max(0, int(decimals))
        try:
            normalized = Decimal(str(quantity)).quantize(Decimal("1").scaleb(-decimals), rounding=ROUND_FLOOR)
        except (InvalidOperation, ValueError):
            return 0.0
        return self._safe_float(normalized)

    @staticmethod
    def _size_decimals(exchange: Any, symbol: str) -> int | None:
        info = getattr(exchange, "info", None)
        if info is None:
            return None
        name_to_coin = getattr(info, "name_to_coin", {}) or {}
        coin = name_to_coin.get(symbol, symbol) if isinstance(name_to_coin, dict) else symbol
        coin_to_asset = getattr(info, "coin_to_asset", {}) or {}
        asset = coin_to_asset.get(coin) if isinstance(coin_to_asset, dict) else None
        if asset is None:
            return None
        asset_to_sz_decimals = getattr(info, "asset_to_sz_decimals", {}) or {}
        decimals = asset_to_sz_decimals.get(asset) if isinstance(asset_to_sz_decimals, dict) else None
        try:
            return int(decimals)
        except (TypeError, ValueError):
            return None

    def get_order_status(
        self,
        mode: str,
        *,
        exchange_order_id: str | int | None = None,
        client_order_id: str | None = None,
        retry: bool = True,
    ) -> dict[str, Any]:
        self._validate_mode(mode)
        if not exchange_order_id and not client_order_id:
            raise ValueError("exchange_order_id or client_order_id is required")

        info = self._get_public_info(mode)
        address = self._account_address()
        if client_order_id:
            cloid = self._cloid_from_client_order_id(client_order_id)
            if hasattr(info, "query_order_by_cloid"):
                payload = self._retry(
                    lambda: info.query_order_by_cloid(address, cloid),
                    f"{mode} order_status cloid",
                    attempts=self.retry_attempts if retry else 1,
                    log_warnings=retry,
                    wrap_error=retry,
                )
            else:
                payload = self._retry(
                    lambda: info.post("/info", {"type": "orderStatus", "user": address, "oid": cloid.to_raw()}),
                    f"{mode} order_status cloid",
                    attempts=self.retry_attempts if retry else 1,
                    log_warnings=retry,
                    wrap_error=retry,
                )
            return self._normalize_order_status_response(payload, client_order_id=client_order_id, exchange_client_order_id=cloid.to_raw())

        oid = int(str(exchange_order_id))
        if hasattr(info, "query_order_by_oid"):
            payload = self._retry(
                lambda: info.query_order_by_oid(address, oid),
                f"{mode} order_status {oid}",
                attempts=self.retry_attempts if retry else 1,
                log_warnings=retry,
                wrap_error=retry,
            )
        else:
            payload = self._retry(
                lambda: info.post("/info", {"type": "orderStatus", "user": address, "oid": oid}),
                f"{mode} order_status {oid}",
                attempts=self.retry_attempts if retry else 1,
                log_warnings=retry,
                wrap_error=retry,
            )
        return self._normalize_order_status_response(payload, exchange_order_id=str(oid))

    def cancel_order(
        self,
        mode: str,
        symbol: str,
        exchange_order_id: str,
        *,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        self._validate_mode(mode)
        self._validate_symbol(symbol)

        if not exchange_order_id and not client_order_id:
            raise ValueError("exchange_order_id or client_order_id is required")

        exchange = self._get_exchange(mode)

        if client_order_id and hasattr(exchange, "cancel_by_cloid"):
            cloid = self._cloid_from_client_order_id(client_order_id)
            response = self._retry(
                lambda: exchange.cancel_by_cloid(symbol, cloid),
                f"{mode} cancel {symbol}:cloid",
            )
            return {
                "status": "cancelled",
                "exchange_order_id": str(exchange_order_id or ""),
                "client_order_id": client_order_id,
                "exchange_client_order_id": cloid.to_raw(),
                "raw": response,
            }

        response = self._retry(lambda: exchange.cancel(symbol, int(exchange_order_id)), f"{mode} cancel {symbol}:{exchange_order_id}")

        return {"status": "cancelled", "exchange_order_id": str(exchange_order_id), "raw": response}

    def cancel_all_orders(self, mode: str) -> list[dict[str, Any]]:
        if not self.can_trade(mode):
            return []

        cancelled: list[dict[str, Any]] = []

        for order in self.get_open_orders(mode):
            symbol = str(order.get("symbol", ""))
            order_id = str(order.get("order_id", ""))

            try:
                cancelled.append(
                    {
                        "symbol": symbol,
                        "order_id": order_id,
                        "result": self.cancel_order(mode, symbol, order_id),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                error = self._sanitize_error_message(exc)
                logger.warning("Failed to cancel order %s:%s: %s", symbol, order_id, error)
                cancelled.append({"symbol": symbol, "order_id": order_id, "error": error})

        return cancelled

    def withdraw_from_bridge(self, mode: str, amount: float, destination: str) -> dict[str, Any]:
        """Submit a Hyperliquid bridge withdrawal through the official SDK."""

        self._validate_mode(mode)
        amount = self._safe_float(amount)
        destination = str(destination or "").strip()
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if not destination:
            raise ValueError("Withdrawal destination is required")

        exchange = self._get_exchange(mode)
        response = self._retry(
            lambda: exchange.withdraw_from_bridge(amount, destination),
            f"{mode} withdraw_from_bridge",
        )
        payload = response if isinstance(response, dict) else {"raw": response}
        return {
            "status": "submitted",
            "provider_reference": str(payload.get("hash") or payload.get("id") or payload.get("status") or ""),
            "raw": payload,
        }

    def flatten_all_positions(self, mode: str) -> list[dict[str, Any]]:
        if not self.can_trade(mode):
            return []

        exchange = self._get_exchange(mode)
        results: list[dict[str, Any]] = []

        for position in self.get_positions(mode):
            quantity = self._safe_float(position.get("quantity"))

            if abs(quantity) < 1e-9:
                continue

            symbol = str(position.get("symbol", ""))

            try:
                response = self._retry(
                    lambda s=symbol: exchange.market_close(s),
                    f"{mode} market_close {symbol}",
                )
                results.append({"symbol": symbol, "quantity": quantity, "result": response})
            except Exception as exc:  # noqa: BLE001
                error = self._sanitize_error_message(exc)
                logger.warning("Failed to flatten %s position: %s", symbol, error)
                results.append({"symbol": symbol, "quantity": quantity, "error": error})

        return results

    def _retry(
        self,
        fn: Callable[[], Any],
        context: str,
        *,
        attempts: int | None = None,
        log_warnings: bool = True,
        wrap_error: bool = True,
    ) -> Any:
        last_error: Exception | None = None
        retry_attempts = max(1, int(attempts or self.retry_attempts))

        for attempt in range(1, retry_attempts + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if log_warnings:
                    logger.warning(
                        "Hyperliquid call failed: %s attempt=%s/%s error=%s",
                        context,
                        attempt,
                        retry_attempts,
                        self._sanitize_error_message(exc),
                    )

                if attempt < retry_attempts:
                    time.sleep(self.retry_sleep_seconds * attempt)

        if last_error is not None and not wrap_error:
            raise last_error
        raise RuntimeError(f"Hyperliquid call failed after retries: {context}") from last_error

    def _public_info_post(
        self,
        mode: str,
        payload: dict[str, Any],
        *,
        context: str,
        attempts: int | None = None,
        log_warnings: bool = True,
    ) -> Any:
        return self._retry(
            lambda: self._public_info_once(mode, payload),
            context,
            attempts=attempts or self.retry_attempts,
            log_warnings=log_warnings,
            wrap_error=True,
        )

    def _public_info_once(self, mode: str, payload: dict[str, Any]) -> Any:
        self._validate_mode(mode)
        if mode == "testnet":
            self._ensure_testnet_public_base_url()
        response = requests.post(
            f"{self.base_url_for_mode(mode)}/info",
            json=payload,
            timeout=self._safe_float(self.config.get("HL_TIMEOUT_SECONDS"), 10.0),
        )
        response.raise_for_status()
        return response.json()

    def _get_public_info(self, mode: str) -> Any:
        self._validate_mode(mode)
        if mode == "testnet":
            self._ensure_testnet_public_base_url()

        if mode not in self._public_info:
            info_cls = _load_info_class()
            self._public_info[mode] = info_cls(
                self.base_url_for_mode(mode),
                skip_ws=True,
                timeout=self._safe_float(self.config.get("HL_TIMEOUT_SECONDS"), 10.0),
            )

        return self._public_info[mode]

    def _get_exchange(self, mode: str) -> Any:
        self._validate_mode(mode)

        if mode == "live" and not self.config.get("ENABLE_LIVE_TRADING", False):
            raise RuntimeError("Live trading is disabled by configuration")
        if mode == "testnet":
            self.ensure_testnet_live_tests_enabled()

        if not self.has_trading_credentials():
            raise RuntimeError("Trading credentials are not configured")

        if mode not in self._exchange_clients:
            eth_account_module = _load_eth_account()
            exchange_cls = _load_exchange_class()
            account = eth_account_module.Account.from_key(self.config["HL_SECRET_KEY"])

            self._exchange_clients[mode] = exchange_cls(
                account,
                self.base_url_for_mode(mode),
                account_address=self.config.get("HL_ACCOUNT_ADDRESS") or account.address,
                vault_address=self.config.get("HL_VAULT_ADDRESS"),
                timeout=self._safe_float(self.config.get("HL_TIMEOUT_SECONDS"), 10.0),
            )

        return self._exchange_clients[mode]

    def _account_address(self) -> str:
        address = self.config.get("HL_ACCOUNT_ADDRESS")

        if not address:
            raise RuntimeError("HL_ACCOUNT_ADDRESS is not configured")

        return str(address)

    def _ensure_testnet_public_base_url(self) -> None:
        base_url = self.base_url_for_mode("testnet")
        if base_url != HYPERLIQUID_TESTNET_API_URL:
            raise RuntimeError(
                f"hyperliquid_testnet_base_url_invalid: refusing to use non-testnet Hyperliquid URL for testnet mode ({base_url})."
            )

    @staticmethod
    def _normalize_base_url(value: Any) -> str:
        return str(value or "").strip().rstrip("/")

    @classmethod
    def _normalize_time_in_force(cls, value: str | None) -> str:
        if value is None:
            return "Gtc"
        key = str(value or "").strip().lower()
        if key not in VALID_TIME_IN_FORCE:
            supported = ", ".join(sorted(VALID_TIME_IN_FORCE.values()))
            raise ValueError(f"Unsupported Hyperliquid time_in_force '{value}'. Supported values: {supported}")
        return VALID_TIME_IN_FORCE[key]

    @classmethod
    def _cloid_from_client_order_id(cls, client_order_id: str | None) -> Any:
        raw = str(client_order_id or "").strip()
        if not raw:
            return None
        cloid_cls = _load_cloid_class()
        if re.fullmatch(r"0x[0-9a-fA-F]{32}", raw):
            return cloid_cls.from_str(raw.lower())
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
        return cloid_cls.from_str(f"0x{digest}")

    @classmethod
    def client_order_id_to_cloid(cls, client_order_id: str) -> str:
        cloid = cls._cloid_from_client_order_id(client_order_id)
        return cloid.to_raw() if cloid is not None and hasattr(cloid, "to_raw") else ""

    @staticmethod
    def _normalize_order_response(
        response: dict[str, Any],
        *,
        submitted_price: float,
        submitted_quantity: float | None = None,
        client_order_id: str | None = None,
        exchange_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        response_payload = response.get("response", {})
        if not isinstance(response_payload, dict):
            response_payload = {}
        data_payload = response_payload.get("data", {})
        if not isinstance(data_payload, dict):
            data_payload = {}
        statuses = data_payload.get("statuses", [])
        if not isinstance(statuses, list):
            statuses = []
        status = statuses[0] if statuses else {}

        exchange_order_id = None
        state = "submitted"
        fill_price = None
        filled_quantity = None
        error = None

        if "resting" in status:
            exchange_order_id = str(status["resting"].get("oid"))
            state = "open"
        elif "filled" in status:
            exchange_order_id = str(status["filled"].get("oid", ""))
            fill_price = HyperliquidClient._safe_float(status["filled"].get("avgPx"), submitted_price)
            filled_quantity = HyperliquidClient._safe_float(status["filled"].get("totalSz"), submitted_quantity or 0.0)
            state = "filled"
        elif "error" in status:
            state = "rejected"
            error = str(status["error"])
        elif response.get("error"):
            state = "rejected"
            error = str(response.get("error"))
        elif str(response.get("status", "")).lower() in {"err", "error", "rejected"}:
            state = "rejected"
            error = str(response.get("response") or response.get("message") or response.get("status"))

        result = {
            "raw": response,
            "status": state,
            "exchange_order_id": exchange_order_id,
            "fill_price": fill_price,
            "submitted_price": submitted_price,
        }

        if submitted_quantity is not None:
            result["submitted_quantity"] = submitted_quantity
        if filled_quantity is not None:
            result["filled_quantity"] = filled_quantity
        if client_order_id:
            result["client_order_id"] = client_order_id
        if exchange_client_order_id:
            result["exchange_client_order_id"] = exchange_client_order_id

        if error:
            result["error"] = HyperliquidClient._sanitize_error_message(error)
            result["error_category"] = HyperliquidClient.classify_error(error)

        return result

    @staticmethod
    def _normalize_order_status_response(
        response: Any,
        *,
        exchange_order_id: str | None = None,
        client_order_id: str | None = None,
        exchange_client_order_id: str | None = None,
    ) -> dict[str, Any]:
        payload = response if isinstance(response, dict) else {"raw": response}
        status_payload = payload.get("order") if isinstance(payload.get("order"), dict) else payload
        status_text = str(status_payload.get("status") or payload.get("status") or "").lower()
        normalized_status = {
            "open": "open",
            "filled": "filled",
            "canceled": "cancelled",
            "cancelled": "cancelled",
            "triggered": "triggered",
            "rejected": "rejected",
            "margincanceled": "cancelled",
        }.get(status_text, status_text or "unknown")
        result = {
            "status": normalized_status,
            "exchange_order_id": str(status_payload.get("oid") or exchange_order_id or ""),
            "client_order_id": client_order_id,
            "exchange_client_order_id": str(status_payload.get("cloid") or exchange_client_order_id or ""),
            "raw": payload,
        }
        if status_payload.get("error"):
            error = str(status_payload.get("error"))
            result["error"] = HyperliquidClient._sanitize_error_message(error)
            result["error_category"] = HyperliquidClient.classify_error(error)
        return result

    @staticmethod
    def classify_error(value: Any) -> str:
        text = str(value or "").lower()
        if any(marker in text for marker in ("missing", "not configured", "required")):
            return "missing_credentials"
        if any(marker in text for marker in ("insufficient margin", "insufficient_testnet_funds", "mintrade", "minimum value")):
            return "insufficient_funds"
        if any(marker in text for marker in ("mainnet", "testnet", "base url", "guard")):
            return "wrong_environment"
        if any(marker in text for marker in ("tick", "precision", "divisible", "decimal")):
            return "precision"
        if any(marker in text for marker in ("signature", "agent", "wallet")):
            return "signature"
        if any(marker in text for marker in ("429", "rate limit", "too many requests")):
            return "rate_limit"
        if any(marker in text for marker in ("symbol", "coin", "asset")):
            return "invalid_symbol"
        return "provider_error"

    @staticmethod
    def _sanitize_error_message(value: Any) -> str:
        text = str(value)
        text = re.sub(r"0x[0-9a-fA-F]{64}", "0x[redacted-private-key]", text)
        text = re.sub(r"0x[0-9a-fA-F]{40}", lambda match: f"{match.group(0)[:6]}...{match.group(0)[-4:]}", text)
        text = re.sub(r"signature['\"]?\s*[:=]\s*[^,}\s]+", "signature=[redacted]", text, flags=re.IGNORECASE)
        return text[:500]

    @staticmethod
    def _validate_mode(mode: str) -> None:
        if mode not in VALID_MODES:
            supported = ", ".join(sorted(VALID_MODES))
            raise ValueError(f"Unsupported mode '{mode}'. Supported modes: {supported}")

    @staticmethod
    def _validate_symbol(symbol: str) -> None:
        if not symbol or not isinstance(symbol, str):
            raise ValueError("symbol must be a non-empty string")
        if symbol.strip().startswith(("#", "@")):
            raise ValueError(f"Unsupported Hyperliquid indexed market symbol: {symbol}")

    @staticmethod
    def _normalize_side(value: Any) -> str:
        raw = str(value or "").lower()
        return "buy" if raw in {"b", "buy", "long"} else "sell"

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        if value is None:
            return default

        try:
            return int(value)
        except (TypeError, ValueError):
            return default


def _load_eth_account() -> Any:
    try:
        import eth_account
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("eth-account is not installed") from exc
    return eth_account


def _load_info_class() -> Any:
    try:
        from hyperliquid.info import Info
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("hyperliquid-python-sdk is not installed") from exc
    return Info


def _load_exchange_class() -> Any:
    try:
        from hyperliquid.exchange import Exchange
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("hyperliquid-python-sdk is not installed") from exc
    return Exchange


def _load_cloid_class() -> Any:
    try:
        from hyperliquid.utils.types import Cloid
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("hyperliquid-python-sdk is not installed") from exc
    return Cloid
