"""Official Hyperliquid SDK wrapper with retries and safe defaults."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Any, Callable


logger = logging.getLogger(__name__)

VALID_MODES = {"testnet", "live"}
VALID_SIDES = {"buy", "sell"}
VALID_ORDER_TYPES = {"market", "limit"}


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
            return str(self.config["HL_MAINNET_BASE_URL"])

        return str(self.config["HL_TESTNET_BASE_URL"])

    def has_account_address(self) -> bool:
        return bool(self.config.get("HL_ACCOUNT_ADDRESS"))

    def has_trading_credentials(self) -> bool:
        return bool(self.config.get("HL_SECRET_KEY"))

    def can_trade(self, mode: str) -> bool:
        self._validate_mode(mode)

        if mode == "live" and not self.config.get("ENABLE_LIVE_TRADING", False):
            return False

        return self.has_trading_credentials() and self.has_account_address()

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

        user_state = self._retry(lambda: info.user_state(address), f"{mode} user_state", attempts=self.retry_attempts if retry else 1, log_warnings=retry, wrap_error=retry)
        spot_state = self._retry(lambda: info.spot_user_state(address), f"{mode} spot_user_state", attempts=self.retry_attempts if retry else 1, log_warnings=retry, wrap_error=retry)

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

        user_state = self._retry(lambda: info.user_state(address), f"{mode} positions", attempts=self.retry_attempts if retry else 1, log_warnings=retry, wrap_error=retry)
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

        orders = self._retry(lambda: info.open_orders(address), f"{mode} open_orders", attempts=self.retry_attempts if retry else 1, log_warnings=retry, wrap_error=retry)

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

        fills = self._retry(lambda: info.user_fills(address), f"{mode} user_fills", attempts=self.retry_attempts if retry else 1, log_warnings=retry, wrap_error=retry)

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
        info = self._get_public_info(mode)
        mids = self._retry(
            lambda: info.all_mids(),
            f"{mode} all_mids",
            attempts=self.retry_attempts if retry else 1,
            log_warnings=retry,
            wrap_error=retry,
        )

        return {
            str(symbol): price
            for symbol, raw_price in mids.items()
            if (price := self._safe_float(raw_price)) > 0
        }

    def get_perp_meta(self, mode: str) -> dict[str, Any]:
        """Return Hyperliquid perpetual metadata from the official info endpoint."""

        info = self._get_public_info(mode)
        if not hasattr(info, "meta"):
            return {}
        payload = self._retry(lambda: info.meta(), f"{mode} meta")
        return payload if isinstance(payload, dict) else {}

    def get_perp_meta_and_asset_contexts(self, mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Return perpetual metadata and asset contexts when supported by the SDK."""

        info = self._get_public_info(mode)
        if not hasattr(info, "meta_and_asset_ctxs"):
            meta = self.get_perp_meta(mode)
            return meta, []
        payload = self._retry(lambda: info.meta_and_asset_ctxs(), f"{mode} meta_and_asset_ctxs")
        if isinstance(payload, (list, tuple)) and len(payload) >= 2:
            meta = payload[0] if isinstance(payload[0], dict) else {}
            contexts = payload[1] if isinstance(payload[1], list) else []
            return meta, [dict(item) for item in contexts if isinstance(item, dict)]
        return {}, []

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

        info = self._get_public_info(mode)

        candles = self._retry(
            lambda: info.candles_snapshot(symbol, timeframe, start_ms, end_ms),
            f"{mode} candles_snapshot {symbol} {timeframe}",
            attempts=self.retry_attempts if retry else 1,
            log_warnings=retry,
            wrap_error=retry,
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
        tif = "Gtc"

        if order_type == "market" or price is None:
            price = self._market_ioc_price(exchange, mode, symbol, is_buy, slippage_pct)
            tif = "Ioc"

        response = self._retry(
            lambda: exchange.order(
                symbol,
                is_buy,
                quantity,
                price,
                {"limit": {"tif": tif}},
                reduce_only=reduce_only,
            ),
            f"{mode} order {symbol}",
        )

        return self._normalize_order_response(response, submitted_price=price, submitted_quantity=quantity)

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

    def cancel_order(self, mode: str, symbol: str, exchange_order_id: str) -> dict[str, Any]:
        self._validate_mode(mode)
        self._validate_symbol(symbol)

        if not exchange_order_id:
            raise ValueError("exchange_order_id is required")

        exchange = self._get_exchange(mode)

        response = self._retry(
            lambda: exchange.cancel(symbol, int(exchange_order_id)),
            f"{mode} cancel {symbol}:{exchange_order_id}",
        )

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
                logger.warning("Failed to cancel order %s:%s: %s", symbol, order_id, exc)
                cancelled.append({"symbol": symbol, "order_id": order_id, "error": str(exc)})

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
                logger.warning("Failed to flatten %s position: %s", symbol, exc)
                results.append({"symbol": symbol, "quantity": quantity, "error": str(exc)})

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
                        exc,
                    )

                if attempt < retry_attempts:
                    time.sleep(self.retry_sleep_seconds * attempt)

        if last_error is not None and not wrap_error:
            raise last_error
        raise RuntimeError(f"Hyperliquid call failed after retries: {context}") from last_error

    def _get_public_info(self, mode: str) -> Any:
        self._validate_mode(mode)

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

    @staticmethod
    def _normalize_order_response(
        response: dict[str, Any],
        *,
        submitted_price: float,
        submitted_quantity: float | None = None,
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

        if error:
            result["error"] = error

        return result

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
