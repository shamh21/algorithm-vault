"""Live provider adapters for user-scoped trading connections."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import time
import uuid
from datetime import UTC, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlencode

import requests

from .failures import ProviderConnectionError
from .hyperliquid_client import ClientSnapshot
from .kucoin_compliance import kucoin_operator_region_status

logger = logging.getLogger(__name__)

BINANCE_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT"}
KUCOIN_SYMBOLS = {"BTC": "XBTUSDTM", "ETH": "ETHUSDTM", "SOL": "SOLUSDTM", "XRP": "XRPUSDTM"}
KUCOIN_SPOT_SYMBOLS = {"BTC": "BTC-USDT", "ETH": "ETH-USDT", "SOL": "SOL-USDT", "XRP": "XRP-USDT"}
KUCOIN_CONTRACT_SPECS = {
    "XBTUSDTM": {"contract_size": 0.001, "size_step": 1, "min_size": 1},
    "ETHUSDTM": {"contract_size": 0.01, "size_step": 1, "min_size": 1},
}
KUCOIN_TEST_ACCOUNT = "sufyanh"
KUCOIN_SPOT_CLIENT_ORDER_PREFIX = "codex-kucoin-"
DYDX_SYMBOLS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD"}
UNISWAP_TOKENS = {
    "ETH": {
        "chain_id": 1,
        "token_address": "0x0000000000000000000000000000000000000000",
        "token_decimals": 18,
        "quote_address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "quote_decimals": 6,
    },
    "BTC": {
        "chain_id": 1,
        "token_address": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
        "token_decimals": 8,
        "quote_address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "quote_decimals": 6,
    },
}


def _is_kucoin_region_restricted_error(value: object) -> bool:
    text = str(value or "").lower()
    return any(
        marker in text
        for marker in (
            "400302",
            "currently unavailable in the u.s",
            "restricted country",
            "restricted region",
            "restricted country/region",
            "current area: us",
        )
    )


def _kucoin_unavailable_alert(exc: object, *, spot: bool = False) -> str:
    if _is_kucoin_region_restricted_error(exc):
        return (
            "KuCoin region restricted: this runtime is not in a KuCoin-supported access region. "
            "Use only a compliant non-restricted fixed-egress server runtime or proxy for KuCoin routing."
        )
    label = "KuCoin spot unavailable" if spot else "KuCoin unavailable"
    return f"{label}: {exc}"


def _kucoin_egress_proxy_url(config: dict[str, Any]) -> str:
    return str(config.get("KUCOIN_EGRESS_PROXY_URL") or config.get("QUOTAGUARDSTATIC_URL") or "").strip()


def _configure_kucoin_egress_proxy(session: requests.Session, config: dict[str, Any]) -> None:
    proxy_url = _kucoin_egress_proxy_url(config)
    if not proxy_url:
        return
    session.trust_env = False
    session.proxies.update({"http": proxy_url, "https": proxy_url})


def _balance_available(balances: list[dict[str, Any]], asset: str) -> float:
    asset_key = str(asset or "").upper().strip()
    for row in balances or []:
        row_asset = str(row.get("asset") or row.get("currency") or "").upper().strip()
        if row_asset != asset_key:
            continue
        for key in ("withdrawable", "available", "available_balance", "free", "value", "total"):
            value = _safe_float(row.get(key))
            if value > 0:
                return value
    return 0.0


class ProviderRequestError(ProviderConnectionError):
    """Raised when a provider API returns an unusable response."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "provider",
        status_code: int | None = None,
        provider_code: str | None = None,
        transient: bool = False,
    ) -> None:
        self.provider = provider
        self.status_code = status_code
        self.provider_code = provider_code
        self.transient = transient
        super().__init__(_redact_provider_error(message))


class BinanceFuturesConnector:
    """Binance USD-M Futures REST adapter."""

    def __init__(self, config: dict[str, Any], credentials: Any, metadata: dict[str, Any] | None = None) -> None:
        if not credentials.api_key or not credentials.api_secret:
            raise RuntimeError("Binance API key and secret are required.")
        self.config = config
        self.credentials = credentials
        self.metadata = metadata or {}
        self.session = requests.Session()

    def can_trade(self, mode: str) -> bool:
        if mode != "live" or not bool(self.config.get("ENABLE_LIVE_TRADING", False)):
            return False
        account = self._signed("GET", "/fapi/v2/account")
        return bool(account.get("canTrade", True))

    def account_snapshot(self, mode: str) -> ClientSnapshot:
        if mode != "live":
            return ClientSnapshot(mode, [], [], [], [], ["Binance connector supports live USD-M futures only."])
        try:
            account = self._signed("GET", "/fapi/v2/account")
            balances = self._balances(account)
            positions = self._positions(account)
            open_orders = self.get_open_orders(mode)
            fills = self.get_recent_fills(mode)
            alerts = [] if bool(account.get("canTrade", True)) else ["Binance API key cannot trade USD-M futures."]
            return ClientSnapshot(mode, balances, positions, open_orders, fills, alerts)
        except Exception as exc:  # noqa: BLE001
            return ClientSnapshot(mode, [], [], [], [], [f"Binance unavailable: {exc}"])

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
        if mode != "live":
            raise RuntimeError("Binance connector supports live USD-M futures only.")
        venue_symbol = self._symbol(symbol)
        if leverage > 0:
            self._signed("POST", "/fapi/v1/leverage", {"symbol": venue_symbol, "leverage": int(max(1, round(leverage)))})
        params: dict[str, Any] = {
            "symbol": venue_symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": self._decimal(quantity),
            "reduceOnly": "true" if reduce_only else "false",
            "newClientOrderId": f"av-{uuid.uuid4().hex[:24]}",
            "newOrderRespType": "RESULT",
        }
        if order_type.lower() == "limit":
            if limit_price is None or limit_price <= 0:
                raise ValueError("Limit price is required for Binance limit orders.")
            params["timeInForce"] = "GTC"
            params["price"] = self._decimal(limit_price)
        response = self._signed("POST", "/fapi/v1/order", params)
        return self._normalize_order_response(response)

    def cancel_order(self, mode: str, symbol: str, exchange_order_id: str) -> dict[str, Any]:
        response = self._signed("DELETE", "/fapi/v1/order", {"symbol": self._symbol(symbol), "orderId": exchange_order_id})
        return {"status": "cancelled", "exchange_order_id": str(exchange_order_id), "raw": response}

    def cancel_all_orders(self, mode: str) -> list[dict[str, Any]]:
        results = []
        for order in self.get_open_orders(mode):
            try:
                results.append(self.cancel_order(mode, str(order["symbol"]), str(order["order_id"])))
            except Exception as exc:  # noqa: BLE001
                results.append({"symbol": order.get("symbol"), "order_id": order.get("order_id"), "error": str(exc)})
        return results

    def flatten_all_positions(self, mode: str) -> list[dict[str, Any]]:
        results = []
        for position in self.get_positions(mode):
            qty = _safe_float(position.get("quantity"))
            if abs(qty) < 1e-9:
                continue
            side = "sell" if qty > 0 else "buy"
            try:
                result = self.place_order(mode, str(position["symbol"]), side, abs(qty), "market", None, True, 1.0, 0.0)
                results.append({"symbol": position["symbol"], "quantity": qty, "result": result})
            except Exception as exc:  # noqa: BLE001
                results.append({"symbol": position.get("symbol"), "quantity": qty, "error": str(exc)})
        return results

    def get_positions(self, mode: str) -> list[dict[str, Any]]:
        return self._positions(self._signed("GET", "/fapi/v2/account"))

    def get_open_orders(self, mode: str) -> list[dict[str, Any]]:
        orders = self._signed("GET", "/fapi/v1/openOrders")
        if not isinstance(orders, list):
            return []
        return [
            {
                "symbol": self._internal_symbol(str(order.get("symbol", ""))),
                "order_id": str(order.get("orderId", "")),
                "side": str(order.get("side", "")).lower(),
                "price": _safe_float(order.get("price")),
                "size": _safe_float(order.get("origQty")),
                "timestamp": order.get("time"),
                "reduce_only": bool(order.get("reduceOnly", False)),
                "raw": order,
            }
            for order in orders
        ]

    def get_recent_fills(self, mode: str) -> list[dict[str, Any]]:
        fills: list[dict[str, Any]] = []
        for symbol in self._recent_symbols():
            try:
                rows = self._signed("GET", "/fapi/v1/userTrades", {"symbol": symbol, "limit": 10})
            except ProviderConnectionError:
                continue
            if isinstance(rows, list):
                for fill in rows:
                    fills.append(
                        {
                            "symbol": self._internal_symbol(str(fill.get("symbol", ""))),
                            "side": "buy" if bool(fill.get("buyer")) else "sell",
                            "price": _safe_float(fill.get("price")),
                            "size": _safe_float(fill.get("qty")),
                            "fee": _safe_float(fill.get("commission")) if fill.get("commission") is not None else None,
                            "fee_token": fill.get("commissionAsset"),
                            "closed_pnl": _safe_float(fill.get("realizedPnl")) if fill.get("realizedPnl") is not None else None,
                            "exchange_order_id": str(fill.get("orderId") or ""),
                            "exchange_fill_id": str(fill.get("id") or fill.get("tradeId") or ""),
                            "timestamp": fill.get("time"),
                            "raw": fill,
                        }
                    )
        return fills[:25]

    def withdraw_from_bridge(self, mode: str, amount: float, destination: str) -> dict[str, Any]:
        raise RuntimeError("Binance withdrawals are intentionally disabled in this app.")

    def _signed(self, method: str, path: str, params: dict[str, Any] | None = None) -> Any:
        params = dict(params or {})
        params.setdefault("recvWindow", int(self.config.get("BINANCE_RECV_WINDOW_MS", 5000)))
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params, doseq=True)
        signature = hmac.new(self.credentials.api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
        params["signature"] = signature
        return self._request(method, path, params=params, headers={"X-MBX-APIKEY": self.credentials.api_key})

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.session.request(method, self._base_url() + path, timeout=self._timeout(), **kwargs)
        if response.status_code >= 400:
            raise ProviderRequestError(response.text[:500])
        return response.json()

    def _base_url(self) -> str:
        return str(self.config.get("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com")).rstrip("/")

    def _timeout(self) -> float:
        return max(1.0, _safe_float(self.config.get("PROVIDER_TIMEOUT_SECONDS"), 10.0))

    def _symbol(self, symbol: str) -> str:
        mapping = _json_config(self.config, "BINANCE_SYMBOL_MAP_JSON", BINANCE_SYMBOLS)
        value = mapping.get(str(symbol).upper())
        if not value:
            raise ValueError(f"{symbol.upper()} is not mapped for Binance USD-M futures.")
        return str(value).upper()

    def _internal_symbol(self, venue_symbol: str) -> str:
        mapping = _json_config(self.config, "BINANCE_SYMBOL_MAP_JSON", BINANCE_SYMBOLS)
        reverse = {str(value).upper(): key for key, value in mapping.items()}
        return reverse.get(venue_symbol.upper(), venue_symbol.upper())

    def _recent_symbols(self) -> list[str]:
        configured = self.config.get("ALLOWED_SYMBOLS", ["BTC", "ETH", "SOL"])
        return [self._symbol(str(symbol)) for symbol in configured[:5] if str(symbol).strip()]

    def _balances(self, account: dict[str, Any]) -> list[dict[str, Any]]:
        balances = []
        for asset in account.get("assets", []):
            wallet = _safe_float(asset.get("walletBalance"))
            margin = _safe_float(asset.get("marginBalance"), wallet)
            if abs(wallet) < 1e-12 and abs(margin) < 1e-12:
                continue
            balances.append(
                {
                    "asset": asset.get("asset", "USDT"),
                    "type": "futures",
                    "value": margin,
                    "withdrawable": _safe_float(asset.get("availableBalance"), margin),
                }
            )
        if not balances and _safe_float(account.get("totalMarginBalance")) > 0:
            balances.append(
                {
                    "asset": "USDT",
                    "type": "futures",
                    "value": _safe_float(account.get("totalMarginBalance")),
                    "withdrawable": _safe_float(account.get("availableBalance")),
                }
            )
        return balances

    def _positions(self, account: dict[str, Any]) -> list[dict[str, Any]]:
        positions = []
        for item in account.get("positions", []):
            qty = _safe_float(item.get("positionAmt"))
            if abs(qty) < 1e-12:
                continue
            positions.append(
                {
                    "symbol": self._internal_symbol(str(item.get("symbol", ""))),
                    "quantity": qty,
                    "side": "long" if qty > 0 else "short",
                    "entry_price": _safe_float(item.get("entryPrice")),
                    "mark_price": _safe_float(item.get("markPrice")),
                    "mark_value": abs(qty) * _safe_float(item.get("markPrice")),
                    "unrealized_pnl": _safe_float(item.get("unrealizedProfit")),
                    "leverage": _safe_float(item.get("leverage"), 1.0),
                    "raw": item,
                }
            )
        return positions

    @staticmethod
    def _normalize_order_response(response: dict[str, Any]) -> dict[str, Any]:
        status = str(response.get("status", "NEW")).upper()
        normalized_status = "filled" if status == "FILLED" else "open" if status in {"NEW", "PARTIALLY_FILLED"} else status.lower()
        return {
            "status": normalized_status,
            "exchange_order_id": str(response.get("orderId") or response.get("clientOrderId") or ""),
            "fill_price": _safe_float(response.get("avgPrice")) or None,
            "filled_quantity": _safe_float(response.get("executedQty", response.get("origQty"))) or None,
            "submitted_price": _safe_float(response.get("price")) or None,
            "raw": response,
        }

    @staticmethod
    def _decimal(value: float | int | str) -> str:
        return format(float(value), "f").rstrip("0").rstrip(".")


class KucoinFuturesConnector:
    """KuCoin Futures/UTA signed REST adapter."""

    def __init__(self, config: dict[str, Any], credentials: Any, metadata: dict[str, Any] | None = None) -> None:
        if not credentials.api_key or not credentials.api_secret or not credentials.passphrase:
            raise RuntimeError("KuCoin API key, secret, and passphrase are required.")
        self.config = config
        self.credentials = credentials
        self.metadata = metadata or {}
        self.session = requests.Session()
        _configure_kucoin_egress_proxy(self.session, config)
        self._time_offset_ms = 0
        self._last_time_sync_monotonic = 0.0

    def can_trade(self, mode: str) -> bool:
        if mode != "live" or not bool(self.config.get("ENABLE_LIVE_TRADING", False)):
            return False
        if self._default_market_type() == "spot":
            self.get_spot_accounts(mode)
            return True
        self._account_overview()
        self._position_mode()
        return True

    def permission_probe(self, mode: str = "live") -> dict[str, Any]:
        """Run read-only KuCoin permission probes without submitting orders."""

        credentials_present = {
            "api_key": bool(self.credentials.api_key),
            "api_secret": bool(self.credentials.api_secret),
            "api_passphrase": bool(self.credentials.passphrase),
        }
        if mode != "live":
            return {
                "credentials_present": credentials_present,
                "general": {"status": "blocked", "message": "KuCoin permission probes require live mode."},
                "spot": {"status": "blocked", "message": "KuCoin permission probes require live mode."},
                "futures": {"status": "blocked", "message": "KuCoin permission probes require live mode."},
                "unified": {"status": "not_checked", "message": "Unified Account permission is operator-selected in KuCoin."},
            }

        return {
            "credentials_present": credentials_present,
            "general": self._permission_check("general", lambda: self.get_spot_accounts(mode, include_zero=False)),
            "spot": self._permission_check("spot", lambda: self.get_spot_accounts(mode, account_type="trade", include_zero=False)),
            "futures": self._permission_check("futures", lambda: (self._account_overview(), self._position_mode())),
            "unified": {
                "status": "operator_configured" if self._config_bool("KUCOIN_UNIFIED_ACCOUNT_ENABLED", False) else "not_checked",
                "message": "Unified Account is marked enabled in server config."
                if self._config_bool("KUCOIN_UNIFIED_ACCOUNT_ENABLED", False)
                else "Unified Account permission is selected in KuCoin and is not probed by this app.",
            },
        }

    def _permission_check(self, permission: str, callback: Any) -> dict[str, str]:
        try:
            callback()
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "message": _kucoin_unavailable_alert(exc, spot=permission == "spot")}
        return {"status": "ready", "message": f"KuCoin {permission} permission check passed."}

    def account_snapshot(self, mode: str) -> ClientSnapshot:
        if self._default_market_type() == "spot":
            return self.spot_account_snapshot(mode)
        if mode != "live":
            return ClientSnapshot(mode, [], [], [], [], ["KuCoin connector supports live futures only."])
        try:
            account = self._account_overview()
            return ClientSnapshot(
                mode, self._balances(account), self.get_positions(mode), self.get_open_orders(mode), self.get_recent_fills(mode), []
            )
        except Exception as exc:  # noqa: BLE001
            return ClientSnapshot(mode, [], [], [], [], [_kucoin_unavailable_alert(exc)])

    def spot_account_snapshot(self, mode: str) -> ClientSnapshot:
        if mode != "live":
            return ClientSnapshot(mode, [], [], [], [], ["KuCoin spot connector supports live mode only."])
        try:
            balances = self.get_spot_balances(mode)
            return ClientSnapshot(mode, balances, [], self.get_spot_open_orders(mode), self.get_spot_recent_fills(mode), [])
        except Exception as exc:  # noqa: BLE001
            return ClientSnapshot(mode, [], [], [], [], [_kucoin_unavailable_alert(exc, spot=True)])

    def get_balances(self, mode: str) -> list[dict[str, Any]]:
        if self._default_market_type() == "spot":
            return self.get_spot_balances(mode)
        return self._balances(self._account_overview())

    def get_spot_accounts(
        self,
        mode: str = "live",
        *,
        currency: str | None = None,
        account_type: str | None = None,
        include_zero: bool = True,
    ) -> list[dict[str, Any]]:
        if mode != "live":
            raise RuntimeError("KuCoin spot accounts are available in live mode only.")
        params: dict[str, Any] = {}
        if currency:
            params["currency"] = str(currency).upper().strip()
        if account_type:
            params["type"] = str(account_type).lower().strip()
        response = self._signed_spot("GET", self._path("KUCOIN_SPOT_ACCOUNTS_PATH", "/api/v1/accounts"), params=params or None)
        accounts = [self._normalize_spot_account(row) for row in _as_list(_response_data(response))]
        if include_zero:
            return accounts
        return [
            row for row in accounts if _safe_float(row.get("value")) or _safe_float(row.get("available")) or _safe_float(row.get("held"))
        ]

    def get_spot_balances(
        self,
        mode: str = "live",
        *,
        account_type: str = "trade",
        include_zero: bool = False,
    ) -> list[dict[str, Any]]:
        return self.get_spot_accounts(mode, account_type=account_type, include_zero=include_zero)

    def discover_spot_accounts(self, mode: str = "live", *, include_subaccounts: bool = True) -> dict[str, Any]:
        if mode != "live":
            raise RuntimeError("KuCoin account discovery is available in live mode only.")
        discovered: dict[str, Any] = {
            "provider": "kucoin",
            "account": str(self.config.get("KUCOIN_TEST_ACCOUNT") or self.metadata.get("account_label") or "").strip(),
            "spot_accounts": self.get_spot_accounts(mode, include_zero=False),
            "sub_accounts": [],
        }
        if include_subaccounts:
            try:
                response = self._signed_spot("GET", self._path("KUCOIN_SUB_ACCOUNTS_PATH", "/api/v1/sub-accounts"))
                discovered["sub_accounts"] = [
                    self._normalize_sub_account(row) for row in _as_list(_response_data(response)) if isinstance(row, dict)
                ]
            except Exception as exc:  # noqa: BLE001 - sub-account listing can be unavailable for sub-user keys.
                discovered["sub_account_error"] = _redact_provider_error(str(exc))
        return discovered

    def get_spot_markets(
        self,
        mode: str = "live",
        *,
        symbol: str | None = None,
        market: str | None = None,
    ) -> list[dict[str, Any]]:
        del mode  # Spot market metadata is public and host-stable.
        if symbol:
            venue_symbol = self._spot_symbol(symbol)
            path = f"{self._path('KUCOIN_SPOT_SYMBOL_PATH', '/api/v2/symbols')}/{venue_symbol}"
            payload = _request_with_retries(
                self.session,
                "GET",
                self._spot_base_url() + path,
                provider="KuCoin",
                attempts=self._retry_attempts(),
                sleep_seconds=self._retry_sleep_seconds(),
                timeout=self._timeout(),
            )
            data = _response_data(payload)
            return [self._normalize_spot_market(data)] if isinstance(data, dict) else []
        params = {"market": str(market).strip()} if market else None
        payload = _request_with_retries(
            self.session,
            "GET",
            self._spot_base_url() + self._path("KUCOIN_SPOT_SYMBOLS_PATH", "/api/v2/symbols"),
            provider="KuCoin",
            attempts=self._retry_attempts(),
            sleep_seconds=self._retry_sleep_seconds(),
            timeout=self._timeout(),
            params=params,
        )
        return [self._normalize_spot_market(row) for row in _as_list(_response_data(payload))]

    def get_spot_market(self, symbol: str, mode: str = "live") -> dict[str, Any]:
        markets = self.get_spot_markets(mode, symbol=symbol)
        if not markets:
            raise ValueError(f"{self._spot_symbol(symbol)} is not a known KuCoin spot market.")
        return markets[0]

    def get_spot_ticker(self, symbol: str, mode: str = "live") -> dict[str, Any]:
        del mode
        venue_symbol = self._spot_symbol(symbol)
        payload = _request_with_retries(
            self.session,
            "GET",
            self._spot_base_url() + self._path("KUCOIN_SPOT_TICKER_PATH", "/api/v1/market/orderbook/level1"),
            provider="KuCoin",
            attempts=max(1, min(self._retry_attempts(), 2)),
            sleep_seconds=self._retry_sleep_seconds(),
            timeout=self._timeout(),
            params={"symbol": venue_symbol},
        )
        data = _response_data(payload)
        row = data if isinstance(data, dict) else {}
        best_bid = _safe_float(row.get("bestBid", row.get("bestBidPrice")))
        best_ask = _safe_float(row.get("bestAsk", row.get("bestAskPrice")))
        last_price = _safe_float(row.get("price", row.get("last")))
        mid = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else last_price
        return {
            "symbol": venue_symbol,
            "price": last_price,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "timestamp": row.get("time"),
            "raw": row,
        }

    def get_spot_mid_price(self, symbol: str, mode: str = "live") -> float:
        ticker = self.get_spot_ticker(symbol, mode)
        price = _safe_float(ticker.get("mid_price"))
        if price <= 0:
            raise RuntimeError(f"provider_market_data_unavailable: kucoin spot mid_price {self._spot_symbol(symbol)}")
        return price

    def get_spot_open_orders(self, mode: str = "live", symbol: str | None = None) -> list[dict[str, Any]]:
        if mode != "live":
            return []
        params = {"symbol": self._spot_symbol(symbol)} if symbol else None
        response = self._signed_spot("GET", self._path("KUCOIN_SPOT_OPEN_ORDERS_PATH", "/api/v1/hf/orders/active"), params=params)
        return [self._normalize_spot_order(row) for row in _as_list(_response_data(response))]

    def get_spot_recent_fills(self, mode: str = "live", symbol: str | None = None, limit: int = 25) -> list[dict[str, Any]]:
        if mode != "live":
            return []
        params: dict[str, Any] = {"limit": max(1, min(int(_safe_float(limit, 25)), 100))}
        if symbol:
            params["symbol"] = self._spot_symbol(symbol)
        try:
            response = self._signed_spot("GET", self._path("KUCOIN_SPOT_FILLS_PATH", "/api/v1/hf/fills"), params=params)
        except ProviderConnectionError:
            return []
        data = _response_data(response)
        rows = data.get("items") if isinstance(data, dict) else data
        return [self._normalize_spot_fill(row) for row in _as_list(rows)][: int(params["limit"])]

    def build_spot_order_payload(
        self,
        symbol: str,
        side: str,
        quantity: float | str | None,
        order_type: str,
        limit_price: float | str | None = None,
        *,
        client_order_id: str | None = None,
        time_in_force: str | None = None,
        post_only: bool = False,
        funds: float | str | None = None,
        market: dict[str, Any] | None = None,
        reference_price: float | str | None = None,
    ) -> dict[str, Any]:
        venue_symbol = self._spot_symbol(symbol)
        side_key = str(side or "").lower().strip()
        type_key = str(order_type or "").lower().strip()
        if side_key not in {"buy", "sell"}:
            raise ValueError(f"Unsupported KuCoin spot order side: {side}")
        if type_key not in {"limit", "market"}:
            raise ValueError(f"Unsupported KuCoin spot order type: {order_type}")
        client_oid = str(client_order_id or self._new_spot_client_order_id()).strip()
        if not client_oid:
            raise ValueError("clientOid is required for KuCoin spot orders.")
        body: dict[str, Any] = {
            "clientOid": client_oid[:64],
            "symbol": venue_symbol,
            "side": side_key,
            "type": type_key,
        }
        if type_key == "limit":
            if limit_price is None or _safe_float(limit_price) <= 0:
                raise ValueError("limit_price must be positive for KuCoin spot limit orders.")
            if quantity is None or _safe_float(quantity) <= 0:
                raise ValueError("quantity must be positive for KuCoin spot limit orders.")
            tif = str(time_in_force or "GTC").upper().strip()
            if tif not in {"GTC", "GTT", "IOC", "FOK"}:
                raise ValueError("KuCoin spot timeInForce must be one of GTC, GTT, IOC, or FOK.")
            if post_only and tif in {"IOC", "FOK"}:
                raise ValueError("KuCoin spot postOnly orders cannot use IOC or FOK.")
            body["price"] = self._decimal(limit_price)
            body["size"] = self._decimal(quantity)
            body["timeInForce"] = tif
            if post_only:
                body["postOnly"] = True
        else:
            if quantity is None and funds is None:
                raise ValueError("KuCoin spot market orders require quantity/size or funds.")
            if quantity is not None:
                body["size"] = self._decimal(quantity)
            if funds is not None:
                body["funds"] = self._decimal(funds)
        if market is not None:
            self.validate_spot_order_payload(body, market=market, reference_price=reference_price)
        return body

    def validate_spot_order_payload(
        self,
        payload: dict[str, Any],
        *,
        market: dict[str, Any],
        reference_price: float | str | None = None,
    ) -> None:
        if not bool(market.get("enabled", market.get("enableTrading", True))):
            raise ValueError(f"KuCoin spot market {market.get('symbol') or payload.get('symbol')} is not enabled for trading.")
        base_increment = self._decimal_value(market.get("base_increment") or market.get("baseIncrement") or "0")
        quote_increment = self._decimal_value(market.get("quote_increment") or market.get("quoteIncrement") or "0")
        price_increment = self._decimal_value(market.get("price_increment") or market.get("priceIncrement") or "0")
        base_min = self._decimal_value(market.get("base_min_size") or market.get("baseMinSize") or "0")
        base_max = self._decimal_value(market.get("base_max_size") or market.get("baseMaxSize") or "0")
        min_funds = self._decimal_value(market.get("min_funds") or market.get("minFunds") or "0")
        size = self._decimal_value(payload.get("size")) if payload.get("size") is not None else None
        funds = self._decimal_value(payload.get("funds")) if payload.get("funds") is not None else None
        price = self._decimal_value(payload.get("price")) if payload.get("price") is not None else None

        if size is not None:
            if size <= 0:
                raise ValueError("KuCoin spot order size must be positive.")
            if base_min > 0 and size < base_min:
                raise ValueError(f"KuCoin spot order size {size} is below baseMinSize {base_min}.")
            if base_max > 0 and size > base_max:
                raise ValueError(f"KuCoin spot order size {size} exceeds baseMaxSize {base_max}.")
            if base_increment > 0 and not self._decimal_multiple(size, base_increment):
                raise ValueError(f"KuCoin spot order size {size} is not aligned to baseIncrement {base_increment}.")
        if price is not None:
            if price <= 0:
                raise ValueError("KuCoin spot limit price must be positive.")
            if price_increment > 0 and not self._decimal_multiple(price, price_increment):
                raise ValueError(f"KuCoin spot limit price {price} is not aligned to priceIncrement {price_increment}.")
        if funds is not None:
            if funds <= 0:
                raise ValueError("KuCoin spot order funds must be positive.")
            if quote_increment > 0 and not self._decimal_multiple(funds, quote_increment):
                raise ValueError(f"KuCoin spot order funds {funds} are not aligned to quoteIncrement {quote_increment}.")

        notional = Decimal("0")
        if price is not None and size is not None:
            notional = price * size
        elif funds is not None:
            notional = funds
        elif size is not None and reference_price is not None:
            notional = size * self._decimal_value(reference_price)
        if min_funds > 0 and notional > 0 and notional < min_funds:
            raise ValueError(f"KuCoin spot order notional {notional} is below minFunds {min_funds}.")

    def create_spot_test_order(
        self,
        mode: str,
        symbol: str,
        side: str,
        quantity: float | str | None,
        order_type: str,
        limit_price: float | str | None = None,
        *,
        client_order_id: str | None = None,
        time_in_force: str | None = None,
        post_only: bool = False,
        funds: float | str | None = None,
    ) -> dict[str, Any]:
        if mode != "live":
            raise RuntimeError("KuCoin spot test orders are available in live mode only.")
        market = self.get_spot_market(symbol, mode)
        payload = self.build_spot_order_payload(
            symbol,
            side,
            quantity,
            order_type,
            limit_price,
            client_order_id=client_order_id,
            time_in_force=time_in_force,
            post_only=post_only,
            funds=funds,
            market=market,
            reference_price=limit_price,
        )
        response = self._signed_spot("POST", self._path("KUCOIN_SPOT_TEST_ORDER_PATH", "/api/v1/hf/orders/test"), body=payload)
        data = _response_data(response)
        normalized = self._normalize_spot_order_response(
            data if isinstance(data, dict) else {}, fallback_client_oid=str(payload["clientOid"]), raw=response
        )
        normalized["test_order"] = True
        return normalized

    def place_spot_order(
        self,
        mode: str,
        symbol: str,
        side: str,
        quantity: float | str | None,
        order_type: str,
        limit_price: float | str | None = None,
        *,
        client_order_id: str | None = None,
        time_in_force: str | None = None,
        post_only: bool = False,
        funds: float | str | None = None,
    ) -> dict[str, Any]:
        if mode != "live":
            raise RuntimeError("KuCoin spot orders are available in live mode only.")
        market = self.get_spot_market(symbol, mode)
        reference_price = limit_price
        if reference_price is None and quantity is not None:
            reference_price = self.get_spot_mid_price(symbol, mode)
        payload = self.build_spot_order_payload(
            symbol,
            side,
            quantity,
            order_type,
            limit_price,
            client_order_id=client_order_id,
            time_in_force=time_in_force,
            post_only=post_only,
            funds=funds,
            market=market,
            reference_price=reference_price,
        )
        response = self._signed_spot("POST", self._path("KUCOIN_SPOT_ORDERS_PATH", "/api/v1/hf/orders"), body=payload)
        data = _response_data(response)
        return self._normalize_spot_order_response(
            data if isinstance(data, dict) else {}, fallback_client_oid=str(payload["clientOid"]), raw=response
        )

    def cancel_spot_order(
        self,
        mode: str,
        symbol: str,
        *,
        exchange_order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        if mode != "live":
            raise RuntimeError("KuCoin spot cancellations are available in live mode only.")
        venue_symbol = self._spot_symbol(symbol)
        if client_order_id:
            path = f"{self._path('KUCOIN_SPOT_CLIENT_ORDER_PATH', '/api/v1/hf/orders/client-order')}/{client_order_id}"
            response = self._signed_spot("DELETE", path, params={"symbol": venue_symbol})
            data = _response_data(response)
            return {
                "status": "cancelled",
                "exchange_order_id": str((data if isinstance(data, dict) else {}).get("orderId") or exchange_order_id or ""),
                "client_order_id": client_order_id,
                "raw": response,
            }
        if not exchange_order_id:
            raise ValueError("exchange_order_id or client_order_id is required for KuCoin spot cancellation.")
        path = f"{self._path('KUCOIN_SPOT_ORDERS_PATH', '/api/v1/hf/orders')}/{exchange_order_id}"
        response = self._signed_spot("DELETE", path, params={"symbol": venue_symbol})
        return {"status": "cancelled", "exchange_order_id": str(exchange_order_id), "raw": response}

    def get_spot_order_status(
        self,
        mode: str,
        symbol: str,
        *,
        exchange_order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        if mode != "live":
            raise RuntimeError("KuCoin spot order status is available in live mode only.")
        venue_symbol = self._spot_symbol(symbol)
        if client_order_id:
            path = f"{self._path('KUCOIN_SPOT_CLIENT_ORDER_PATH', '/api/v1/hf/orders/client-order')}/{client_order_id}"
            response = self._signed_spot("GET", path, params={"symbol": venue_symbol})
        elif exchange_order_id:
            path = f"{self._path('KUCOIN_SPOT_ORDERS_PATH', '/api/v1/hf/orders')}/{exchange_order_id}"
            response = self._signed_spot("GET", path, params={"symbol": venue_symbol})
        else:
            raise ValueError("exchange_order_id or client_order_id is required for KuCoin spot order status.")
        data = _response_data(response)
        return self._normalize_spot_order(data if isinstance(data, dict) else {})

    def kucoin_live_test_guard_errors(self, *, require_live_trading: bool = False, require_fill: bool = False) -> list[str]:
        errors: list[str] = []
        account = str(self.config.get("KUCOIN_TEST_ACCOUNT") or "").strip()
        symbol = str(self.config.get("KUCOIN_TEST_SYMBOL") or "").strip()
        raw_cap = self.config.get("KUCOIN_MAX_TEST_NOTIONAL_USDT")
        max_notional = _safe_float(raw_cap)
        if not self.credentials.api_key:
            errors.append("KUCOIN_API_KEY is required")
        if not self.credentials.api_secret:
            errors.append("KUCOIN_API_SECRET is required")
        if not self.credentials.passphrase:
            errors.append("KUCOIN_API_PASSPHRASE is required")
        if account != KUCOIN_TEST_ACCOUNT:
            errors.append(f"KUCOIN_TEST_ACCOUNT must be exactly {KUCOIN_TEST_ACCOUNT}")
        if not symbol:
            errors.append("KUCOIN_TEST_SYMBOL is required")
        if raw_cap in {None, ""} or max_notional <= 0:
            errors.append("KUCOIN_MAX_TEST_NOTIONAL_USDT must be set")
        elif max_notional > 5:
            errors.append("KUCOIN_MAX_TEST_NOTIONAL_USDT must be <= 5")
        if require_live_trading and not self._config_bool("KUCOIN_ENABLE_LIVE_TEST_TRADES", False):
            errors.append("KUCOIN_ENABLE_LIVE_TEST_TRADES=true is required")
        if require_fill and not self._config_bool("KUCOIN_ENABLE_FILL_TEST", False):
            errors.append("KUCOIN_ENABLE_FILL_TEST=true is required")
        region_status = kucoin_operator_region_status(self.config)
        if bool(region_status.get("restricted", False)):
            errors.append(f"KUCOIN_OPERATOR_REGION={region_status.get('label')} is restricted under KuCoin terms")
        if not self._config_bool("KUCOIN_COMPLIANCE_CONFIRMED", False):
            errors.append("KUCOIN_COMPLIANCE_CONFIRMED=true is required")
        if self._config_bool("KUCOIN_FIXED_EGRESS_REQUIRED", False) and not _kucoin_egress_proxy_url(self.config):
            errors.append("KUCOIN_EGRESS_PROXY_URL or QUOTAGUARDSTATIC_URL is required")
        return errors

    def kucoin_live_test_preflight_summary(self) -> dict[str, Any]:
        max_notional = _safe_float(self.config.get("KUCOIN_MAX_TEST_NOTIONAL_USDT"))
        proxy_url = _kucoin_egress_proxy_url(self.config)
        region_status = kucoin_operator_region_status(self.config)
        fixed_egress_status = (
            "restricted"
            if bool(region_status.get("restricted", False))
            else "ready"
            if self._config_bool("KUCOIN_COMPLIANCE_CONFIRMED", False) and proxy_url
            else "missing"
            if self._config_bool("KUCOIN_FIXED_EGRESS_REQUIRED", False)
            else "pending"
        )
        return {
            "account": str(self.config.get("KUCOIN_TEST_ACCOUNT") or "").strip(),
            "symbol": str(self.config.get("KUCOIN_TEST_SYMBOL") or "").strip(),
            "max_notional_usdt": max_notional if max_notional > 0 else None,
            "live_trading_enabled": self._config_bool("KUCOIN_ENABLE_LIVE_TEST_TRADES", False),
            "fill_test_enabled": self._config_bool("KUCOIN_ENABLE_FILL_TEST", False),
            "fixed_egress_required": self._config_bool("KUCOIN_FIXED_EGRESS_REQUIRED", False),
            "fixed_egress_configured": bool(proxy_url),
            "fixed_egress_status": fixed_egress_status,
            "compliance_confirmed": self._config_bool("KUCOIN_COMPLIANCE_CONFIRMED", False),
            "operator_region": str(region_status.get("region") or ""),
            "operator_region_label": str(region_status.get("label") or ""),
            "operator_region_restricted": bool(region_status.get("restricted", False)),
            "spot_base_url": self._spot_base_url(),
            "credentials_present": {
                "api_key": bool(self.credentials.api_key),
                "api_secret": bool(self.credentials.api_secret),
                "api_passphrase": bool(self.credentials.passphrase),
            },
            "missing_or_blocked": self.kucoin_live_test_guard_errors(),
        }

    def ensure_kucoin_live_test_guards(self, *, require_live_trading: bool = False, require_fill: bool = False) -> None:
        errors = self.kucoin_live_test_guard_errors(require_live_trading=require_live_trading, require_fill=require_fill)
        if errors:
            raise RuntimeError("kucoin_live_test_guard_failed: " + "; ".join(errors))

    def kucoin_spot_live_test_plan(
        self,
        *,
        side: str = "buy",
        require_funds: bool = True,
        post_only: bool = True,
    ) -> dict[str, Any]:
        self.ensure_kucoin_live_test_guards(require_live_trading=False)
        side_key = str(side or "buy").lower().strip()
        if side_key not in {"buy", "sell"}:
            raise ValueError("KuCoin live test side must be buy or sell.")
        symbol = str(self.config.get("KUCOIN_TEST_SYMBOL") or "").strip()
        market = self.get_spot_market(symbol, "live")
        ticker = self.get_spot_ticker(symbol, "live")
        best_bid = self._decimal_value(ticker.get("best_bid"))
        best_ask = self._decimal_value(ticker.get("best_ask"))
        mid_price = self._decimal_value(ticker.get("mid_price"))
        if mid_price <= 0:
            raise RuntimeError(f"KuCoin spot mid price unavailable for {market['symbol']}.")
        price_increment = self._decimal_value(market.get("price_increment"))
        base_increment = self._decimal_value(market.get("base_increment"))
        base_min = self._decimal_value(market.get("base_min_size"))
        min_funds = self._decimal_value(market.get("min_funds"))
        max_notional = self._decimal_value(self.config.get("KUCOIN_MAX_TEST_NOTIONAL_USDT"))
        if max_notional <= 0 or max_notional > Decimal("5"):
            raise RuntimeError("kucoin_test_cap_invalid: KUCOIN_MAX_TEST_NOTIONAL_USDT must be > 0 and <= 5.")
        if min_funds > max_notional:
            raise RuntimeError(f"kucoin_min_funds_exceeds_cap: {min_funds} > {max_notional}.")
        reference = best_bid if side_key == "buy" and best_bid > 0 else best_ask if side_key == "sell" and best_ask > 0 else mid_price
        raw_price = reference * (Decimal("0.90") if side_key == "buy" else Decimal("1.10"))
        price = self._round_decimal(raw_price, price_increment, ROUND_FLOOR if side_key == "buy" else ROUND_CEILING)
        if price <= 0:
            raise RuntimeError(f"KuCoin spot test price unavailable for {market['symbol']}.")
        min_size_by_funds = min_funds / price if min_funds > 0 else Decimal("0")
        size = max(base_min, self._round_decimal(min_size_by_funds, base_increment, ROUND_CEILING))
        if size <= 0:
            raise RuntimeError(f"KuCoin spot minimum size unavailable for {market['symbol']}.")
        notional = size * price
        if notional > max_notional:
            raise RuntimeError(f"kucoin_test_max_notional_exceeded: {notional} > {max_notional}.")
        quote_asset = str(market.get("quote") or "USDT").upper()
        base_asset = str(market.get("base") or "").upper()
        balances = self.get_spot_balances("live") if require_funds else []
        available_quote = _balance_available(balances, quote_asset) if require_funds else 0.0
        available_base = _balance_available(balances, base_asset) if require_funds else 0.0
        if require_funds:
            if side_key == "buy" and Decimal(str(available_quote)) + Decimal("0.000000001") < notional:
                raise RuntimeError(
                    f"insufficient_kucoin_test_funds: available {quote_asset} {available_quote:.8f} < required {self._format_decimal(notional)}."
                )
            if side_key == "sell" and Decimal(str(available_base)) + Decimal("0.000000001") < size:
                raise RuntimeError(
                    f"insufficient_kucoin_test_funds: available {base_asset} {available_base:.8f} < required {self._format_decimal(size)}."
                )
        return {
            "mode": "live",
            "symbol": market["symbol"],
            "internal_symbol": market.get("internal_symbol"),
            "side": side_key,
            "order_type": "limit",
            "time_in_force": "GTC",
            "post_only": bool(post_only),
            "limit_price": self._format_decimal(price),
            "quantity": self._format_decimal(size),
            "notional_usdt": float(notional),
            "max_notional_usdt": float(max_notional),
            "min_funds": float(min_funds),
            "quote_asset": quote_asset,
            "base_asset": base_asset,
            "available_quote": available_quote,
            "available_base": available_base,
            "ticker": {
                "best_bid": float(best_bid),
                "best_ask": float(best_ask),
                "mid_price": float(mid_price),
            },
        }

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
        if self._default_market_type() == "spot":
            if reduce_only:
                raise RuntimeError("KuCoin spot orders do not support reduce-only semantics.")
            if leverage != 1.0:
                raise RuntimeError("KuCoin spot orders do not support leverage.")
            return self.place_spot_order(
                mode,
                symbol,
                side,
                quantity,
                order_type,
                limit_price,
                client_order_id=client_order_id,
                time_in_force=time_in_force,
            )
        if mode != "live":
            raise RuntimeError("KuCoin connector supports live futures only.")
        venue_symbol = self._symbol(symbol)
        contracts = self._contract_size(venue_symbol, quantity)
        body: dict[str, Any] = {
            "clientOid": client_order_id or f"av-{uuid.uuid4().hex}",
            "symbol": venue_symbol,
            "marginMode": self._margin_mode(),
            "positionSide": self._position_side(),
            "side": side.lower(),
            "type": order_type.lower(),
            "size": contracts,
            "leverage": int(max(1, round(leverage))),
            "reduceOnly": bool(reduce_only),
        }
        if order_type.lower() == "limit":
            if limit_price is None or limit_price <= 0:
                raise ValueError("Limit price is required for KuCoin limit orders.")
            body["price"] = self._decimal(limit_price)
            body["timeInForce"] = str(time_in_force or "GTC").upper()
        response = self._signed("POST", self._path("KUCOIN_ORDERS_PATH", "/api/v1/orders"), body=body)
        data = _response_data(response)
        if not isinstance(data, dict):
            data = {}
        return self._normalize_order_response(data, fallback_client_oid=str(body["clientOid"]), submitted_price=limit_price, raw=response)

    def cancel_order(self, mode: str, symbol: str, exchange_order_id: str, *, client_order_id: str | None = None) -> dict[str, Any]:
        if self._default_market_type() == "spot":
            return self.cancel_spot_order(mode, symbol, exchange_order_id=exchange_order_id, client_order_id=client_order_id)
        response = self._signed("DELETE", f"{self._path('KUCOIN_ORDERS_PATH', '/api/v1/orders')}/{exchange_order_id}")
        return {"status": "cancelled", "exchange_order_id": str(exchange_order_id), "raw": response}

    def cancel_all_orders(self, mode: str) -> list[dict[str, Any]]:
        results = []
        for order in self.get_open_orders(mode):
            try:
                results.append(self.cancel_order(mode, str(order["symbol"]), str(order["order_id"])))
            except Exception as exc:  # noqa: BLE001
                results.append({"symbol": order.get("symbol"), "order_id": order.get("order_id"), "error": str(exc)})
        return results

    def flatten_all_positions(self, mode: str) -> list[dict[str, Any]]:
        results = []
        for position in self.get_positions(mode):
            qty = _safe_float(position.get("quantity"))
            if abs(qty) < 1e-9:
                continue
            side = "sell" if qty > 0 else "buy"
            try:
                result = self.place_order(mode, str(position["symbol"]), side, abs(qty), "market", None, True, 1.0, 0.0)
                results.append({"symbol": position["symbol"], "quantity": qty, "result": result})
            except Exception as exc:  # noqa: BLE001
                results.append({"symbol": position.get("symbol"), "quantity": qty, "error": str(exc)})
        return results

    def get_positions(self, mode: str) -> list[dict[str, Any]]:
        if self._default_market_type() == "spot":
            return []
        response = self._signed("GET", self._path("KUCOIN_POSITIONS_PATH", "/api/v1/positions"))
        rows = _as_list(_response_data(response))
        positions = []
        for item in rows:
            qty = _safe_float(item.get("currentQty", item.get("quantity")))
            if abs(qty) < 1e-12:
                continue
            positions.append(
                {
                    "symbol": self._internal_symbol(str(item.get("symbol", ""))),
                    "quantity": qty,
                    "side": "long" if qty > 0 else "short",
                    "entry_price": _safe_float(item.get("avgEntryPrice", item.get("entryPrice"))),
                    "mark_price": _safe_float(item.get("markPrice")),
                    "mark_value": _safe_float(item.get("posCost", item.get("value"))),
                    "unrealized_pnl": _safe_float(item.get("unrealisedPnl", item.get("unrealizedPnl"))),
                    "leverage": _safe_float(item.get("realLeverage", item.get("leverage")), 1.0),
                    "raw": item,
                }
            )
        return positions

    def get_open_orders(self, mode: str) -> list[dict[str, Any]]:
        if self._default_market_type() == "spot":
            return self.get_spot_open_orders(mode)
        response = self._signed("GET", self._path("KUCOIN_OPEN_ORDERS_PATH", "/api/v1/orders"), params={"status": "active"})
        data = _response_data(response)
        rows = _as_list(data.get("items") if isinstance(data, dict) else data)
        return [
            {
                "symbol": self._internal_symbol(str(order.get("symbol", ""))),
                "order_id": str(order.get("id", order.get("orderId", ""))),
                "side": str(order.get("side", "")).lower(),
                "price": _safe_float(order.get("price")),
                "size": _safe_float(order.get("size", order.get("quantity"))),
                "timestamp": order.get("createdAt", order.get("created_at")),
                "reduce_only": bool(order.get("reduceOnly", False)),
                "raw": order,
            }
            for order in rows
        ]

    def get_recent_fills(self, mode: str) -> list[dict[str, Any]]:
        if self._default_market_type() == "spot":
            return self.get_spot_recent_fills(mode)
        try:
            response = self._signed("GET", self._path("KUCOIN_FILLS_PATH", "/api/v1/recentDoneOrders"))
        except ProviderConnectionError:
            return []
        data = _response_data(response)
        rows = _as_list(data.get("items") if isinstance(data, dict) else data)
        return [
            {
                "symbol": self._internal_symbol(str(fill.get("symbol", ""))),
                "side": str(fill.get("side", "")).lower(),
                "price": _safe_float(fill.get("price")),
                "size": _safe_float(fill.get("size", fill.get("dealSize"))),
                "fee": _safe_float(fill.get("fee")) if fill.get("fee") is not None else None,
                "fee_token": fill.get("feeCurrency"),
                "closed_pnl": _safe_float(fill.get("realisedPnl", fill.get("pnl")))
                if fill.get("realisedPnl", fill.get("pnl")) is not None
                else None,
                "exchange_order_id": str(fill.get("orderId") or fill.get("order_id") or ""),
                "exchange_fill_id": str(fill.get("tradeId") or fill.get("id") or ""),
                "timestamp": fill.get("createdAt"),
                "raw": fill,
            }
            for fill in rows[:25]
        ]

    def withdraw_from_bridge(self, mode: str, amount: float, destination: str) -> dict[str, Any]:
        return self.withdraw_to_address(mode, "USDT", amount, destination)

    def deposit_address(self, mode: str, asset: str, network: str | None = None) -> dict[str, Any]:
        if mode != "live":
            raise RuntimeError("KuCoin deposit addresses are available in live mode only.")
        asset_key = str(asset or "").upper().strip()
        params: dict[str, Any] = {"currency": asset_key}
        chain = self._kucoin_chain(network)
        if chain:
            params["chain"] = chain
        response = self._signed_spot("GET", self._path("KUCOIN_DEPOSIT_ADDRESSES_PATH", "/api/v3/deposit-addresses"), params=params)
        data = _response_data(response)
        row: dict[str, Any]
        if isinstance(data, list):
            row = next((dict(item) for item in data if isinstance(item, dict)), {})
        elif isinstance(data, dict):
            row = data
        else:
            row = {}
        address = str(row.get("address") or row.get("toAddress") or "").strip()
        if not address:
            raise RuntimeError(f"KuCoin returned no deposit address for {asset_key}.")
        return {
            "asset": asset_key,
            "network": str(network or row.get("chain") or row.get("chainName") or ""),
            "address": address,
            "memo": str(row.get("memo") or row.get("tag") or ""),
            "status": "active",
            "raw": row or response,
        }

    def reserve_funds(self, mode: str, asset: str, amount: float) -> dict[str, Any]:
        snapshot = self.account_snapshot(mode)
        asset_key = str(asset or "").upper().strip()
        requested = max(0.0, float(amount or 0.0))
        available = _balance_available(snapshot.balances, asset_key)
        if available + 1e-9 < requested:
            raise RuntimeError(f"KuCoin {asset_key} reserve unavailable: {available:.6f} < {requested:.6f}.")
        return {
            "status": "confirmed",
            "provider_reference": f"kucoin-reserve-{uuid.uuid4().hex}",
            "asset": asset_key,
            "confirmed_amount": requested,
            "available_before": available,
            "raw": {"reserve_type": "exchange_balance"},
        }

    def internal_transfer(
        self,
        mode: str,
        asset: str,
        amount: float,
        *,
        from_account_type: str,
        to_account_type: str,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        if mode != "live":
            raise RuntimeError("KuCoin internal transfers are available in live mode only.")
        client_oid = str(client_reference or uuid.uuid4().hex)[:128]
        body = {
            "clientOid": client_oid,
            "type": "INTERNAL",
            "currency": str(asset or "").upper().strip(),
            "amount": self._decimal(max(0.0, float(amount or 0.0))),
            "fromAccountType": str(from_account_type or "MAIN").upper(),
            "toAccountType": str(to_account_type or "TRADE").upper(),
        }
        response = self._signed_spot("POST", self._path("KUCOIN_UNIVERSAL_TRANSFER_PATH", "/api/v3/accounts/universal-transfer"), body=body)
        data = _response_data(response)
        return {
            "status": "submitted",
            "provider_reference": str((data if isinstance(data, dict) else {}).get("orderId") or client_oid),
            "client_reference": client_oid,
            "raw": response,
        }

    def withdraw_to_address(
        self,
        mode: str,
        asset: str,
        amount: float,
        destination: str,
        network: str | None = None,
        memo: str | None = None,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        if mode != "live":
            raise RuntimeError("KuCoin withdrawals are available in live mode only.")
        asset_key = str(asset or "").upper().strip()
        requested = max(0.0, float(amount or 0.0))
        if requested <= 0:
            raise ValueError("KuCoin withdrawal amount must be positive.")
        body: dict[str, Any] = {
            "currency": asset_key,
            "toAddress": str(destination or "").strip(),
            "amount": self._decimal(requested),
            "withdrawType": "ADDRESS",
            "remark": str(client_reference or "algorithm-vault-cycle")[:64],
        }
        chain = self._kucoin_chain(network)
        if chain:
            body["chain"] = chain
        if memo:
            body["memo"] = str(memo)
        response = self._signed_spot("POST", self._path("KUCOIN_WITHDRAWALS_PATH", "/api/v3/withdrawals"), body=body)
        data = _response_data(response)
        provider_reference = str((data if isinstance(data, dict) else {}).get("withdrawalId") or "")
        return {
            "status": "submitted",
            "provider_reference": provider_reference,
            "asset": asset_key,
            "network": str(network or ""),
            "confirmed_amount": requested,
            "raw": response,
        }

    def transfer_status(self, mode: str, provider_reference: str, transfer_type: str | None = None) -> dict[str, Any]:
        return {
            "status": "submitted" if provider_reference else "unknown",
            "provider_reference": str(provider_reference or ""),
            "transfer_type": transfer_type or "withdrawal",
            "raw": {},
        }

    def convert_stablecoin(
        self,
        mode: str,
        from_asset: str,
        to_asset: str,
        amount: float,
        max_slippage_bps: float,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        if mode != "live":
            raise RuntimeError("KuCoin conversion is available in live mode only.")
        from_key = str(from_asset or "").upper().strip()
        to_key = str(to_asset or "").upper().strip()
        if {from_key, to_key} - {"USDC", "USDT", "ETH"}:
            raise RuntimeError("KuCoin conversion supports USDT, USDC, and ETH only.")
        requested = max(0.0, float(amount or 0.0))
        quote = self._signed_spot(
            "GET",
            self._path("KUCOIN_CONVERT_QUOTE_PATH", "/api/v1/convert/quote"),
            params={"fromCurrency": from_key, "toCurrency": to_key, "fromCurrencySize": self._decimal(requested)},
        )
        quote_data = _response_data(quote)
        if not isinstance(quote_data, dict) or not quote_data.get("quoteId"):
            raise RuntimeError("KuCoin conversion quote was unavailable.")
        from_size = _safe_float(quote_data.get("fromCurrencySize"), requested)
        to_size = _safe_float(quote_data.get("toCurrencySize"))
        expected = requested if from_key in {"USDC", "USDT"} and to_key in {"USDC", "USDT"} else 0.0
        if expected > 0 and to_size > 0:
            slippage_bps = abs((to_size - expected) / expected) * 10_000
            if slippage_bps > max(0.0, float(max_slippage_bps or 0.0)):
                raise RuntimeError(f"KuCoin conversion slippage {slippage_bps:.2f} bps exceeds configured limit.")
        body = {
            "clientOrderId": str(client_reference or uuid.uuid4().hex)[:128],
            "quoteId": str(quote_data["quoteId"]),
            "accountType": "BOTH",
        }
        order = self._signed_spot("POST", self._path("KUCOIN_CONVERT_ORDER_PATH", "/api/v1/convert/order"), body=body)
        order_data = _response_data(order)
        return {
            "status": "submitted",
            "provider_reference": str((order_data if isinstance(order_data, dict) else {}).get("orderId") or body["clientOrderId"]),
            "from_asset": from_key,
            "to_asset": to_key,
            "requested_amount": from_size or requested,
            "confirmed_amount": to_size,
            "raw": {"quote": quote, "order": order},
        }

    def discover_leveraged_markets(self, mode: str) -> list[dict[str, Any]]:
        if mode != "live":
            return []
        payload = _request_with_retries(
            self.session,
            "GET",
            self._base_url() + self._path("KUCOIN_ACTIVE_CONTRACTS_PATH", "/api/v1/contracts/active"),
            provider="KuCoin",
            attempts=self._retry_attempts(),
            sleep_seconds=self._retry_sleep_seconds(),
            timeout=self._timeout(),
        )
        rows = _response_data(payload)
        return [dict(item) for item in _as_list(rows) if isinstance(item, dict)]

    def get_candles(self, symbol: str, timeframe: str, mode: str, limit: int = 200) -> list[dict[str, Any]]:
        if mode != "live":
            raise RuntimeError("KuCoin connector supports live futures market data only.")
        venue_symbol = self._symbol(symbol)
        granularity = self._kline_granularity(timeframe)
        candle_limit = max(1, min(int(_safe_float(limit, 200)), 500))
        now = datetime.now(UTC)
        start = now - (timedelta(minutes=granularity) * (candle_limit + 5))
        payload = _request_with_retries(
            self.session,
            "GET",
            self._base_url() + self._path("KUCOIN_KLINE_PATH", "/api/v1/kline/query"),
            provider="KuCoin",
            attempts=max(1, min(self._retry_attempts(), 2)),
            sleep_seconds=self._retry_sleep_seconds(),
            timeout=self._timeout(),
            params={
                "symbol": venue_symbol,
                "granularity": granularity,
                "from": int(start.timestamp() * 1000),
                "to": int(now.timestamp() * 1000),
            },
        )
        rows = _response_data(payload)
        candles = [self._normalize_kline(row, timeframe) for row in (rows if isinstance(rows, list) else [])]
        result = [row for row in candles if row is not None]
        if not result:
            raise RuntimeError(f"provider_market_data_unavailable: kucoin candles {venue_symbol} {timeframe}")
        return sorted(result, key=lambda item: item["timestamp"])[-candle_limit:]

    def get_mid_price(self, symbol: str, mode: str) -> float:
        if mode != "live":
            raise RuntimeError("KuCoin connector supports live futures market data only.")
        venue_symbol = self._symbol(symbol)
        payload = _request_with_retries(
            self.session,
            "GET",
            self._base_url() + self._path("KUCOIN_TICKER_PATH", "/api/v1/ticker"),
            provider="KuCoin",
            attempts=max(1, min(self._retry_attempts(), 2)),
            sleep_seconds=self._retry_sleep_seconds(),
            timeout=self._timeout(),
            params={"symbol": venue_symbol},
        )
        data = _response_data(payload)
        if not isinstance(data, dict):
            raise RuntimeError(f"provider_market_data_unavailable: kucoin mid_price {venue_symbol}")
        best_bid = _safe_float(data.get("bestBidPrice", data.get("bidPrice")))
        best_ask = _safe_float(data.get("bestAskPrice", data.get("askPrice")))
        if best_bid > 0 and best_ask > 0:
            return (best_bid + best_ask) / 2.0
        price = _safe_float(data.get("price", data.get("markPrice", data.get("indexPrice"))))
        if price <= 0:
            raise RuntimeError(f"provider_market_data_unavailable: kucoin mid_price {venue_symbol}")
        return price

    def _account_overview(self) -> dict[str, Any]:
        path = self._path("KUCOIN_ACCOUNT_OVERVIEW_PATH", "/api/v1/account-overview")
        response = self._signed("GET", path, params={"currency": "USDT"})
        data = _response_data(response)
        return data if isinstance(data, dict) else {}

    def _position_mode(self) -> dict[str, Any]:
        path = self._path("KUCOIN_POSITION_MODE_PATH", "/api/v2/position/getPositionMode")
        response = self._signed("GET", path)
        data = _response_data(response)
        return data if isinstance(data, dict) else {}

    def _signed(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        return self._signed_with_base(method, path, params=params, body=body, base_url=self._base_url())

    def _signed_spot(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        return self._signed_with_base(method, path, params=params, body=body, base_url=self._spot_base_url())

    def _default_market_type(self) -> str:
        value = str(self.metadata.get("market_type") or self.config.get("KUCOIN_DEFAULT_MARKET_TYPE", "futures")).strip().lower()
        return "spot" if value == "spot" else "futures"

    def _signed_with_base(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        *,
        base_url: str,
    ) -> Any:
        params = dict(params or {})
        body_text = json.dumps(body or {}, separators=(",", ":"), ensure_ascii=False) if body else ""
        endpoint = path + (f"?{urlencode(params, doseq=True)}" if params else "")
        payload: Any = None
        for attempt in range(2):
            timestamp = str(self._timestamp_ms(force_sync=attempt > 0))
            pre_sign = f"{timestamp}{method.upper()}{endpoint}{body_text}"
            signature = base64.b64encode(
                hmac.new(self.credentials.api_secret.encode("utf-8"), pre_sign.encode("utf-8"), hashlib.sha256).digest()
            ).decode("utf-8")
            passphrase = base64.b64encode(
                hmac.new(self.credentials.api_secret.encode("utf-8"), self.credentials.passphrase.encode("utf-8"), hashlib.sha256).digest()
            ).decode("utf-8")
            headers = {
                "KC-API-KEY": self.credentials.api_key,
                "KC-API-SIGN": signature,
                "KC-API-TIMESTAMP": timestamp,
                "KC-API-PASSPHRASE": passphrase,
                "KC-API-KEY-VERSION": "2",
                "Content-Type": "application/json",
            }
            request_attempts = self._retry_attempts() if self._is_safe_retry_method(method) else 1
            payload = _request_with_retries(
                self.session,
                method,
                base_url + path,
                provider="KuCoin",
                attempts=request_attempts,
                sleep_seconds=self._retry_sleep_seconds(),
                timeout=self._timeout(),
                params=params or None,
                data=body_text if body else None,
                headers=headers,
            )
            if not self._is_invalid_timestamp_payload(payload) or attempt > 0:
                break
        if isinstance(payload, dict) and str(payload.get("code", "200000")) not in {"200000", "200", "0"}:
            provider_code = str(payload.get("code") or "")
            message = str(payload.get("msg") or payload.get("message") or payload)
            raise ProviderRequestError(
                json.dumps({"code": provider_code, "msg": message}),
                provider="KuCoin",
                provider_code=provider_code,
            )
        return payload

    @staticmethod
    def _is_safe_retry_method(method: str) -> bool:
        return str(method or "").upper() in {"GET", "HEAD", "OPTIONS"}

    def _timestamp_ms(self, *, force_sync: bool = False) -> int:
        if bool(self.config.get("KUCOIN_TIME_SYNC_ENABLED", True)):
            self._sync_server_time(force=force_sync)
        return int(time.time() * 1000) + int(self._time_offset_ms)

    def _sync_server_time(self, *, force: bool = False) -> None:
        if not bool(self.config.get("KUCOIN_TIME_SYNC_ENABLED", True)):
            return
        ttl = max(1.0, _safe_float(self.config.get("KUCOIN_TIME_SYNC_TTL_SECONDS"), 300.0))
        now_monotonic = time.monotonic()
        if not force and self._last_time_sync_monotonic and now_monotonic - self._last_time_sync_monotonic < ttl:
            return
        payload = _request_with_retries(
            self.session,
            "GET",
            (self._spot_base_url() if self._default_market_type() == "spot" else self._base_url())
            + self._path("KUCOIN_SERVER_TIME_PATH", "/api/v1/timestamp"),
            provider="KuCoin",
            attempts=max(1, min(self._retry_attempts(), 2)),
            sleep_seconds=self._retry_sleep_seconds(),
            timeout=self._timeout(),
        )
        server_ms = _safe_float(_response_data(payload))
        if server_ms <= 0:
            return
        local_ms = int(time.time() * 1000)
        self._time_offset_ms = int(server_ms) - local_ms
        self._last_time_sync_monotonic = now_monotonic

    @staticmethod
    def _is_invalid_timestamp_payload(payload: Any) -> bool:
        if not isinstance(payload, dict):
            return False
        code = str(payload.get("code") or "")
        message = str(payload.get("msg") or payload.get("message") or "").lower()
        return code == "400002" or "kc-api-timestamp" in message

    def _base_url(self) -> str:
        return str(self.config.get("KUCOIN_FUTURES_BASE_URL", "https://api-futures.kucoin.com")).rstrip("/")

    def _spot_base_url(self) -> str:
        return str(self.config.get("KUCOIN_SPOT_BASE_URL", "https://api.kucoin.com")).rstrip("/")

    def _timeout(self) -> float:
        return max(1.0, _safe_float(self.config.get("PROVIDER_TIMEOUT_SECONDS"), 10.0))

    def _retry_attempts(self) -> int:
        return max(1, int(_safe_float(self.config.get("PROVIDER_RETRY_ATTEMPTS", self.config.get("EXCHANGE_RETRY_ATTEMPTS", 3)), 3)))

    def _retry_sleep_seconds(self) -> float:
        return max(
            0.0, _safe_float(self.config.get("PROVIDER_RETRY_SLEEP_SECONDS", self.config.get("EXCHANGE_RETRY_SLEEP_SECONDS", 0.5)), 0.5)
        )

    def _path(self, key: str, default: str) -> str:
        return str(self.config.get(key, default)).strip() or default

    @staticmethod
    def _kucoin_chain(network: str | None) -> str:
        key = re.sub(r"[^A-Za-z0-9]", "", str(network or "")).upper()
        mapping = {
            "ETH": "eth",
            "ETHEREUM": "eth",
            "ERC20": "eth",
            "ARBITRUM": "arb",
            "ARBITRUMONE": "arb",
            "TRON": "trx",
            "TRC20": "trx",
            "BSC": "bsc",
            "BEP20": "bsc",
            "SOL": "sol",
            "SOLANA": "sol",
            "POLYGON": "matic",
            "MATIC": "matic",
            "BASE": "base",
        }
        return mapping.get(key, str(network or "").strip())

    def _margin_mode(self) -> str:
        value = str(self.metadata.get("margin_mode") or self.config.get("KUCOIN_MARGIN_MODE", "ISOLATED")).strip().upper()
        return value if value in {"ISOLATED", "CROSS"} else "ISOLATED"

    def _position_side(self) -> str:
        value = str(self.metadata.get("position_side") or self.config.get("KUCOIN_POSITION_SIDE", "BOTH")).strip().upper()
        return value if value in {"BOTH", "LONG", "SHORT"} else "BOTH"

    def _spot_symbol(self, symbol: str | None) -> str:
        raw_symbol = str(symbol or "").upper().strip().replace("/", "-").replace("_", "-")
        if not raw_symbol:
            raise ValueError("KuCoin spot symbol is required.")
        if "-" in raw_symbol:
            return raw_symbol
        mapping = _json_config(self.config, "KUCOIN_SPOT_SYMBOL_MAP_JSON", KUCOIN_SPOT_SYMBOLS)
        value = mapping.get(raw_symbol)
        if not value:
            quote = str(self.config.get("KUCOIN_DEFAULT_SPOT_QUOTE", "USDT") or "USDT").upper().strip()
            value = f"{raw_symbol}-{quote}"
        return str(value).upper().replace("/", "-")

    def _internal_spot_symbol(self, venue_symbol: str) -> str:
        raw_symbol = str(venue_symbol or "").upper().strip().replace("/", "-").replace("_", "-")
        mapping = _json_config(self.config, "KUCOIN_SPOT_SYMBOL_MAP_JSON", KUCOIN_SPOT_SYMBOLS)
        reverse = {str(value).upper().replace("/", "-"): key for key, value in mapping.items()}
        if raw_symbol in reverse:
            return reverse[raw_symbol]
        base, separator, quote = raw_symbol.partition("-")
        return base if separator and quote in {"USDT", "USDC", "USD"} else raw_symbol

    def _symbol(self, symbol: str) -> str:
        raw_symbol = str(symbol or "").upper()
        if raw_symbol.endswith(("USDTM", "USDM")) or raw_symbol.startswith("."):
            return raw_symbol
        mapping = _json_config(self.config, "KUCOIN_SYMBOL_MAP_JSON", KUCOIN_SYMBOLS)
        value = mapping.get(raw_symbol)
        if not value:
            raise ValueError(f"{raw_symbol} is not mapped for KuCoin futures.")
        return str(value).upper()

    def _internal_symbol(self, venue_symbol: str) -> str:
        mapping = _json_config(self.config, "KUCOIN_SYMBOL_MAP_JSON", KUCOIN_SYMBOLS)
        reverse = {str(value).upper(): key for key, value in mapping.items()}
        return reverse.get(venue_symbol.upper(), venue_symbol.upper())

    def _contract_size(self, venue_symbol: str, quantity: float) -> int:
        spec = _json_config(self.config, "KUCOIN_CONTRACT_SPECS_JSON", KUCOIN_CONTRACT_SPECS).get(venue_symbol.upper())
        if not isinstance(spec, dict):
            raise ValueError(f"{venue_symbol.upper()} is missing KUCOIN_CONTRACT_SPECS_JSON sizing metadata.")
        multiplier = _safe_float(spec.get("contract_size"))
        step = max(1, int(_safe_float(spec.get("size_step"), 1.0)))
        min_size = max(step, int(_safe_float(spec.get("min_size"), step)))
        if multiplier <= 0:
            raise ValueError(f"{venue_symbol.upper()} has invalid KuCoin contract_size metadata.")
        raw_contracts = float(quantity) / multiplier
        rounded = int(round(raw_contracts / step) * step)
        if rounded < min_size:
            raise ValueError(f"{venue_symbol.upper()} KuCoin contract size {rounded} is below minimum {min_size}.")
        if abs(rounded - raw_contracts) > 1e-9:
            raise ValueError(
                f"{venue_symbol.upper()} quantity {quantity} does not align to KuCoin contract_size={multiplier} and size_step={step}."
            )
        return rounded

    def _normalize_spot_account(self, row: dict[str, Any]) -> dict[str, Any]:
        asset = str(row.get("currency") or row.get("asset") or "").upper().strip()
        account_type = str(row.get("type") or row.get("accountType") or "").lower().strip()
        total = _safe_float(row.get("balance", row.get("total", row.get("value"))))
        available = _safe_float(row.get("available", row.get("availableBalance", row.get("free"))))
        held = _safe_float(row.get("holds", row.get("hold", row.get("holdBalance"))))
        return {
            "asset": asset,
            "currency": asset,
            "type": f"spot_{account_type}" if account_type else "spot",
            "account_type": account_type,
            "value": total,
            "total": total,
            "available": available,
            "withdrawable": available,
            "held": held,
            "raw": row,
        }

    @staticmethod
    def _normalize_sub_account(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": str(row.get("subName") or row.get("subAccountName") or "").strip(),
            "has_main_accounts": bool(row.get("mainAccounts")),
            "has_trade_accounts": bool(row.get("tradeAccounts")),
            "raw": row,
        }

    def _normalize_spot_market(self, row: dict[str, Any]) -> dict[str, Any]:
        symbol = str(row.get("symbol") or row.get("name") or "").upper().replace("/", "-")
        base = str(row.get("baseCurrency") or row.get("base") or "").upper()
        quote = str(row.get("quoteCurrency") or row.get("quote") or "").upper()
        return {
            "symbol": symbol,
            "internal_symbol": self._internal_spot_symbol(symbol),
            "base": base,
            "quote": quote,
            "fee_currency": row.get("feeCurrency"),
            "market": row.get("market"),
            "base_min_size": str(row.get("baseMinSize") or row.get("minBaseOrderSize") or "0"),
            "quote_min_size": str(row.get("quoteMinSize") or row.get("minQuoteOrderSize") or "0"),
            "base_max_size": str(row.get("baseMaxSize") or row.get("maxBaseOrderSize") or "0"),
            "quote_max_size": str(row.get("quoteMaxSize") or row.get("maxQuoteOrderSize") or "0"),
            "base_increment": str(row.get("baseIncrement") or row.get("baseOrderStep") or "0"),
            "quote_increment": str(row.get("quoteIncrement") or row.get("quoteOrderStep") or "0"),
            "price_increment": str(row.get("priceIncrement") or row.get("tickSize") or "0"),
            "price_limit_rate": str(row.get("priceLimitRate") or row.get("priceLimitRatio") or "0"),
            "min_funds": str(row.get("minFunds") or row.get("minQuoteOrderSize") or "0"),
            "enabled": bool(row.get("enableTrading", str(row.get("tradingStatus") or "1") == "1")),
            "margin_enabled": bool(row.get("isMarginEnabled", False)),
            "raw": row,
        }

    def _normalize_spot_order_response(self, payload: dict[str, Any], *, fallback_client_oid: str, raw: Any) -> dict[str, Any]:
        normalized = self._normalize_spot_order({**payload, "clientOid": payload.get("clientOid") or fallback_client_oid})
        if not normalized.get("exchange_order_id"):
            normalized["exchange_order_id"] = str(payload.get("orderId") or payload.get("id") or "")
        normalized["client_order_id"] = str(payload.get("clientOid") or fallback_client_oid)
        normalized["raw"] = raw
        if normalized["status"] == "unknown":
            normalized["status"] = "submitted"
        return normalized

    def _normalize_spot_order(self, order: dict[str, Any]) -> dict[str, Any]:
        raw_status = str(order.get("status") or order.get("state") or "").lower().strip()
        active = order.get("active", order.get("isActive"))
        cancel_exist = bool(order.get("cancelExist") or order.get("cancelled"))
        deal_size = _safe_float(order.get("dealSize", order.get("filledSize")))
        size = _safe_float(order.get("size", order.get("quantity")))
        if raw_status in {"done", "filled"} or (active is False and deal_size > 0 and (size <= 0 or deal_size >= size)):
            status = "filled"
        elif raw_status in {"cancelled", "canceled"} or cancel_exist or (active is False and deal_size < size):
            status = "cancelled"
        elif raw_status in {"active", "open"} or active is True or bool(order.get("inOrderBook")):
            status = "open"
        elif raw_status in {"rejected", "failed"} or order.get("rejectReason"):
            status = "rejected"
        else:
            status = raw_status or "unknown"
        return {
            "status": status,
            "symbol": self._internal_spot_symbol(str(order.get("symbol") or "")),
            "venue_symbol": str(order.get("symbol") or ""),
            "exchange_order_id": str(order.get("orderId") or order.get("id") or ""),
            "client_order_id": str(order.get("clientOid") or ""),
            "side": str(order.get("side") or "").lower(),
            "order_type": str(order.get("type") or "").lower(),
            "price": _safe_float(order.get("price")),
            "size": _safe_float(order.get("size")),
            "funds": _safe_float(order.get("funds")),
            "filled_quantity": deal_size or None,
            "filled_funds": _safe_float(order.get("dealFunds")) or None,
            "fee": _safe_float(order.get("fee")) if order.get("fee") is not None else None,
            "fee_token": order.get("feeCurrency"),
            "time_in_force": order.get("timeInForce"),
            "post_only": bool(order.get("postOnly", False)),
            "in_order_book": bool(order.get("inOrderBook", False)),
            "timestamp": order.get("createdAt", order.get("created_at")),
            "raw": order,
        }

    def _normalize_spot_fill(self, fill: dict[str, Any]) -> dict[str, Any]:
        return {
            "symbol": self._internal_spot_symbol(str(fill.get("symbol", ""))),
            "venue_symbol": str(fill.get("symbol") or ""),
            "side": str(fill.get("side") or "").lower(),
            "price": _safe_float(fill.get("price")),
            "size": _safe_float(fill.get("size", fill.get("dealSize"))),
            "funds": _safe_float(fill.get("funds", fill.get("dealFunds"))),
            "fee": _safe_float(fill.get("fee")) if fill.get("fee") is not None else None,
            "fee_token": fill.get("feeCurrency"),
            "liquidity": fill.get("liquidity"),
            "order_type": fill.get("type"),
            "exchange_order_id": str(fill.get("orderId") or fill.get("order_id") or ""),
            "exchange_fill_id": str(fill.get("tradeId") or fill.get("id") or ""),
            "timestamp": fill.get("createdAt"),
            "raw": fill,
        }

    def _balances(self, account: dict[str, Any]) -> list[dict[str, Any]]:
        value = _safe_float(account.get("accountEquity", account.get("marginBalance", account.get("balance"))))
        available = _safe_float(account.get("availableBalance", account.get("available", value)))
        if not value and not available:
            return []
        return [{"asset": account.get("currency", "USDT"), "type": "futures", "value": value, "withdrawable": available}]

    @staticmethod
    def _kline_granularity(timeframe: str) -> int:
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "8h": 480,
            "1d": 1440,
        }
        key = str(timeframe or "").strip().lower()
        if key not in mapping:
            raise ValueError(f"Unsupported KuCoin futures timeframe '{timeframe}'.")
        return mapping[key]

    @staticmethod
    def _normalize_kline(row: Any, timeframe: str) -> dict[str, Any] | None:
        if isinstance(row, dict):
            timestamp = _safe_float(row.get("time", row.get("timestamp", row.get("startAt", row.get("start")))))
            open_price = _safe_float(row.get("open"))
            high = _safe_float(row.get("high"))
            low = _safe_float(row.get("low"))
            close = _safe_float(row.get("close"))
            volume = _safe_float(row.get("volume", row.get("vol")))
        elif isinstance(row, (list, tuple)) and len(row) >= 6:
            values = list(row)
            timestamp = _safe_float(values[0])
            open_price = _safe_float(values[1])
            high = _safe_float(values[2])
            low = _safe_float(values[3])
            close = _safe_float(values[4])
            volume = _safe_float(values[5])
        else:
            return None
        if timestamp <= 0 or close <= 0:
            return None
        if timestamp > 10_000_000_000:
            timestamp = timestamp / 1000.0
        return {
            "timestamp": datetime.fromtimestamp(timestamp, tz=UTC).isoformat(),
            "timeframe": timeframe,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }

    def _new_spot_client_order_id(self) -> str:
        return f"{KUCOIN_SPOT_CLIENT_ORDER_PREFIX}{uuid.uuid4().hex[:24]}"

    def _config_bool(self, key: str, default: bool = False) -> bool:
        value = self.config.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _decimal_value(value: Any, default: str | Decimal = "0") -> Decimal:
        if value is None or value == "":
            return Decimal(str(default))
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return Decimal(str(default))

    @classmethod
    def _round_decimal(cls, value: Any, increment: Decimal, rounding: str) -> Decimal:
        decimal_value = cls._decimal_value(value)
        if increment <= 0:
            return decimal_value
        units = (decimal_value / increment).to_integral_value(rounding=rounding)
        return units * increment

    @staticmethod
    def _decimal_multiple(value: Decimal, increment: Decimal) -> bool:
        if increment <= 0:
            return True
        try:
            return (value % increment) == 0
        except InvalidOperation:
            return False

    @staticmethod
    def _format_decimal(value: Decimal) -> str:
        normalized = format(value.normalize(), "f")
        return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized

    @staticmethod
    def _normalize_order_response(
        payload: dict[str, Any],
        *,
        fallback_client_oid: str,
        submitted_price: float | None,
        raw: Any,
    ) -> dict[str, Any]:
        raw_status = str(payload.get("status") or payload.get("state") or "").lower()
        done = bool(payload.get("isActive") is False or payload.get("doneAt") or payload.get("endAt"))
        cancelled = bool(payload.get("cancelExist") or payload.get("cancelled"))
        deal_size = _safe_float(payload.get("dealSize", payload.get("filledSize")))
        size = _safe_float(payload.get("size", payload.get("quantity")))
        if raw_status in {"done", "filled"} or (done and deal_size > 0 and (size <= 0 or deal_size >= size)):
            status = "filled"
        elif raw_status in {"cancelled", "canceled"} or cancelled:
            status = "cancelled"
        elif raw_status in {"rejected", "failed"} or payload.get("rejectReason"):
            status = "rejected"
        elif raw_status in {"active", "open"} or payload.get("isActive") is True:
            status = "open"
        else:
            status = "submitted"
        result = {
            "status": status,
            "exchange_order_id": str(payload.get("orderId") or payload.get("id") or fallback_client_oid),
            "client_order_id": str(payload.get("clientOid") or fallback_client_oid),
            "fill_price": _safe_float(payload.get("avgDealPrice", payload.get("avgPrice"))) or None,
            "filled_quantity": deal_size or None,
            "submitted_price": submitted_price if submitted_price is not None else (_safe_float(payload.get("price")) or None),
            "raw": raw,
        }
        if payload.get("rejectReason"):
            result["error"] = str(payload["rejectReason"])
        return result

    @staticmethod
    def _decimal(value: float | int | str) -> str:
        return KucoinFuturesConnector._format_decimal(KucoinFuturesConnector._decimal_value(value))


class DydxV4Connector:
    """dYdX v4 adapter using permissioned trading keys."""

    def __init__(self, config: dict[str, Any], credentials: Any, metadata: dict[str, Any] | None = None) -> None:
        if not credentials.wallet_address:
            raise RuntimeError("dYdX owner wallet address is required.")
        if not credentials.api_secret:
            raise RuntimeError("dYdX permissioned trading private key is required.")
        self.config = config
        self.credentials = credentials
        self.metadata = metadata or {}
        self.session = requests.Session()

    def can_trade(self, mode: str) -> bool:
        if mode != "live" or not bool(self.config.get("ENABLE_LIVE_TRADING", False)):
            return False
        self._require_permissioned_setup()
        if not callable(self.config.get("DYDX_ORDER_EXECUTOR_FACTORY")):
            raise RuntimeError("Configure DYDX_ORDER_EXECUTOR_FACTORY to enable dYdX permissioned order signing.")
        return True

    def account_snapshot(self, mode: str) -> ClientSnapshot:
        if mode != "live":
            return ClientSnapshot(mode, [], [], [], [], ["dYdX connector supports live perpetuals only."])
        try:
            self._require_permissioned_setup()
            return ClientSnapshot(mode, self._balances(), self.get_positions(mode), [], [], [])
        except Exception as exc:  # noqa: BLE001
            return ClientSnapshot(mode, [], [], [], [], [f"dYdX unavailable: {exc}"])

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
        factory = self.config.get("DYDX_ORDER_EXECUTOR_FACTORY")
        if callable(factory):
            return factory(self.config, self.credentials, self.metadata).place_order(
                mode, symbol, side, quantity, order_type, limit_price, reduce_only, leverage, slippage_pct
            )
        raise RuntimeError("dYdX order signing requires a configured dydx-v4-client executor.")

    def cancel_order(self, mode: str, symbol: str, exchange_order_id: str) -> dict[str, Any]:
        factory = self.config.get("DYDX_ORDER_EXECUTOR_FACTORY")
        if callable(factory):
            return factory(self.config, self.credentials, self.metadata).cancel_order(mode, symbol, exchange_order_id)
        raise RuntimeError("dYdX cancel signing requires a configured dydx-v4-client executor.")

    def cancel_all_orders(self, mode: str) -> list[dict[str, Any]]:
        return []

    def flatten_all_positions(self, mode: str) -> list[dict[str, Any]]:
        results = []
        for position in self.get_positions(mode):
            qty = _safe_float(position.get("quantity"))
            if abs(qty) < 1e-9:
                continue
            side = "sell" if qty > 0 else "buy"
            try:
                result = self.place_order(mode, str(position["symbol"]), side, abs(qty), "market", None, True, 1.0, 0.0)
                results.append({"symbol": position["symbol"], "quantity": qty, "result": result})
            except Exception as exc:  # noqa: BLE001
                results.append({"symbol": position.get("symbol"), "quantity": qty, "error": str(exc)})
        return results

    def get_positions(self, mode: str) -> list[dict[str, Any]]:
        address = self.credentials.wallet_address
        subaccount = int(_safe_float(self.metadata.get("subaccount_number"), 0))
        path = f"/addresses/{address}/subaccountNumber/{subaccount}/perpetualPositions"
        try:
            payload = self._get(path, {"status": "OPEN"})
        except ProviderConnectionError:
            return []
        rows = _as_list(payload.get("positions") if isinstance(payload, dict) else payload)
        return [
            {
                "symbol": self._internal_symbol(str(item.get("market", item.get("ticker", "")))),
                "quantity": _safe_float(item.get("size")),
                "side": str(item.get("side", "")).lower(),
                "entry_price": _safe_float(item.get("entryPrice")),
                "mark_price": _safe_float(item.get("oraclePrice")),
                "unrealized_pnl": _safe_float(item.get("unrealizedPnl")),
                "leverage": _safe_float(item.get("leverage"), 1.0),
                "raw": item,
            }
            for item in rows
            if abs(_safe_float(item.get("size"))) > 1e-12
        ]

    def withdraw_from_bridge(self, mode: str, amount: float, destination: str) -> dict[str, Any]:
        raise RuntimeError("dYdX withdrawals are intentionally disabled in this app.")

    def _balances(self) -> list[dict[str, Any]]:
        address = self.credentials.wallet_address
        subaccount = int(_safe_float(self.metadata.get("subaccount_number"), 0))
        payload = self._get(f"/addresses/{address}/subaccountNumber/{subaccount}")
        account = payload.get("subaccount", payload) if isinstance(payload, dict) else {}
        equity = _safe_float(account.get("equity", account.get("freeCollateral")))
        if not equity:
            return []
        return [{"asset": "USDC", "type": "perpetual", "value": equity, "withdrawable": _safe_float(account.get("freeCollateral", equity))}]

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        response = self.session.get(self._indexer_url() + path, params=params or None, timeout=self._timeout())
        if response.status_code >= 400:
            raise ProviderRequestError(response.text[:500])
        return response.json()

    def _require_permissioned_setup(self) -> None:
        if not str(self.metadata.get("authenticator_id", "")).strip():
            raise RuntimeError("dYdX authenticator id is required.")
        if str(self.credentials.api_secret).strip().count(" ") >= 11:
            raise RuntimeError("dYdX seed phrases are not accepted; use a permissioned trading private key.")

    def _indexer_url(self) -> str:
        return str(self.config.get("DYDX_INDEXER_URL", "https://indexer.dydx.trade/v4")).rstrip("/")

    def _timeout(self) -> float:
        return max(1.0, _safe_float(self.config.get("PROVIDER_TIMEOUT_SECONDS"), 10.0))

    def _internal_symbol(self, venue_symbol: str) -> str:
        mapping = _json_config(self.config, "DYDX_SYMBOL_MAP_JSON", DYDX_SYMBOLS)
        reverse = {str(value).upper(): key for key, value in mapping.items()}
        return reverse.get(venue_symbol.upper(), venue_symbol.upper())

    @staticmethod
    def _sdk_available() -> bool:
        try:
            __import__("dydx_v4_client")
        except ImportError:
            return False
        return True


class UniswapDelegatedConnector:
    """Uniswap Trading API adapter gated by wallet delegation metadata."""

    def __init__(self, config: dict[str, Any], credentials: Any, metadata: dict[str, Any] | None = None) -> None:
        if not credentials.wallet_address:
            raise RuntimeError("Uniswap wallet address is required.")
        self.config = config
        self.credentials = credentials
        self.metadata = metadata or {}
        self.session = requests.Session()

    def can_trade(self, mode: str) -> bool:
        if mode != "live" or not bool(self.config.get("ENABLE_LIVE_TRADING", False)):
            return False
        self._require_delegation()
        return True

    def account_snapshot(self, mode: str) -> ClientSnapshot:
        try:
            self._require_delegation()
            return ClientSnapshot(
                mode,
                [{"asset": "Wallet", "type": "delegated", "value": 0.0, "withdrawable": 0.0}],
                [],
                [],
                [],
                [],
            )
        except Exception as exc:  # noqa: BLE001
            return ClientSnapshot(mode, [], [], [], [], [f"Uniswap delegation unavailable: {exc}"])

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
        if reduce_only or leverage != 1.0:
            raise RuntimeError("Uniswap swaps do not support reduce-only or leveraged order semantics.")
        self._require_delegation()
        token = self._token(symbol)
        exact_output = side.lower() == "buy"
        amount = self._swap_amount(quantity, token)
        quote = self._post(
            "/quote",
            {
                "tokenIn": token["quote_address"] if exact_output else token["token_address"],
                "tokenOut": token["token_address"] if exact_output else token["quote_address"],
                "tokenInChainId": int(token["chain_id"]),
                "tokenOutChainId": int(token["chain_id"]),
                "type": "EXACT_OUTPUT" if exact_output else "EXACT_INPUT",
                "amount": str(amount),
                "swapper": self.credentials.wallet_address,
                "slippageTolerance": max(0.01, slippage_pct * 100),
                "protocols": self._protocols(),
            },
        )
        permit_signature = str(self.metadata.get("permit2_signature", "") or "").strip()
        if quote.get("permitData") and not permit_signature:
            raise RuntimeError("Uniswap Permit2 signature is required for this quote.")
        endpoint = "/order" if str(quote.get("routing", "")).upper().startswith("DUTCH") else "/swap"
        payload: dict[str, Any] = {"quote": quote}
        if permit_signature:
            payload["signature"] = permit_signature
        result = self._post(endpoint, payload)
        tx = result.get("tx", result.get("transaction", {})) if isinstance(result, dict) else {}
        return {
            "status": "submitted",
            "exchange_order_id": str(result.get("requestId") or result.get("orderId") or tx.get("hash") or ""),
            "submitted_price": limit_price,
            "raw": result,
        }

    def cancel_order(self, mode: str, symbol: str, exchange_order_id: str) -> dict[str, Any]:
        raise RuntimeError("Uniswap swap cancellation is not available through this connector.")

    def cancel_all_orders(self, mode: str) -> list[dict[str, Any]]:
        return []

    def flatten_all_positions(self, mode: str) -> list[dict[str, Any]]:
        return []

    def get_positions(self, mode: str) -> list[dict[str, Any]]:
        return []

    def withdraw_from_bridge(self, mode: str, amount: float, destination: str) -> dict[str, Any]:
        raise RuntimeError("Uniswap withdrawals are not managed by this app.")

    def _require_delegation(self) -> None:
        if not bool(self.config.get("UNISWAP_DELEGATED_TRADING_ENABLED", False)):
            raise RuntimeError("Uniswap delegation trading is disabled by configuration.")
        if not str(self.config.get("UNISWAP_API_KEY", "")).strip():
            raise RuntimeError("UNISWAP_API_KEY is required.")
        if str(self.metadata.get("delegation_status", "")).strip().lower() != "approved":
            raise RuntimeError("Wallet delegation must be approved before Uniswap can activate.")
        if not str(self.metadata.get("session_topic", "")).strip():
            raise RuntimeError("WalletConnect/Reown session reference is required before Uniswap can activate.")
        expires_at = _parse_datetime(self.metadata.get("delegation_expires_at"))
        if expires_at is None or expires_at <= datetime.now(UTC):
            raise RuntimeError("Wallet delegation is expired or missing an expiry.")
        if _safe_float(self.metadata.get("daily_loss_usd")) <= 0:
            raise RuntimeError("Uniswap daily loss cap is required.")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "x-api-key": str(self.config.get("UNISWAP_API_KEY", "")),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        response = self.session.post(self._base_url() + path, json=payload, headers=headers, timeout=self._timeout())
        if response.status_code >= 400:
            raise ProviderRequestError(response.text[:500])
        data = response.json()
        if not isinstance(data, dict):
            raise ProviderRequestError("Uniswap returned a non-object response.")
        return data

    def _token(self, symbol: str) -> dict[str, Any]:
        mapping = _json_config(self.config, "UNISWAP_TOKEN_MAP_JSON", UNISWAP_TOKENS)
        token = mapping.get(str(symbol).upper())
        if not isinstance(token, dict):
            raise ValueError(f"{symbol.upper()} is not mapped for Uniswap swaps.")
        chain_id = int(_safe_float(self.metadata.get("chain_id"), token.get("chain_id", 1)))
        if int(token.get("chain_id", chain_id)) != chain_id:
            raise ValueError(f"{symbol.upper()} is not configured for chain {chain_id}.")
        return {**token, "chain_id": chain_id}

    def _swap_amount(self, quantity: float, token: dict[str, Any]) -> int:
        return max(1, int(float(quantity) * (10 ** int(token["token_decimals"]))))

    def _protocols(self) -> list[str]:
        raw = str(self.metadata.get("protocols") or self.config.get("UNISWAP_PROTOCOLS", "V2,V3,V4")).strip()
        return [item.strip().upper() for item in raw.split(",") if item.strip()]

    def _base_url(self) -> str:
        return str(self.config.get("UNISWAP_API_BASE_URL", "https://trade-api.gateway.uniswap.org/v1")).rstrip("/")

    def _timeout(self) -> float:
        return max(1.0, _safe_float(self.config.get("PROVIDER_TIMEOUT_SECONDS"), 10.0))


def _json_config(config: dict[str, Any], key: str, default: dict[str, Any]) -> dict[str, Any]:
    value = config.get(key)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return dict(default)
        return parsed if isinstance(parsed, dict) else dict(default)
    return dict(default)


def _response_data(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload


def _request_with_retries(
    session: requests.Session,
    method: str,
    url: str,
    *,
    provider: str,
    attempts: int,
    sleep_seconds: float,
    timeout: float,
    **kwargs: Any,
) -> Any:
    last_error: Exception | None = None
    attempts = max(1, int(attempts))
    for attempt in range(1, attempts + 1):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            if response.status_code >= 400:
                raise ProviderRequestError(
                    response.text[:500],
                    provider=provider,
                    status_code=response.status_code,
                    transient=_is_transient_status(response.status_code),
                )
            try:
                return response.json()
            except ValueError as exc:
                raise ProviderRequestError(
                    f"{provider} returned non-JSON response.",
                    provider=provider,
                    status_code=response.status_code,
                    transient=False,
                ) from exc
        except ProviderRequestError as exc:
            last_error = exc
            should_retry = bool(exc.transient)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            should_retry = True
        except requests.RequestException as exc:
            last_error = exc
            should_retry = False

        if attempt >= attempts or not should_retry:
            break
        logger.warning("%s request failed attempt=%s/%s error=%s", provider, attempt, attempts, _redact_provider_error(str(last_error)))
        time.sleep(max(0.0, sleep_seconds) * attempt)

    if isinstance(last_error, ProviderRequestError):
        raise last_error
    transient = isinstance(last_error, (requests.Timeout, requests.ConnectionError))
    raise ProviderRequestError(
        f"{provider} request failed after {attempts} attempt(s): {last_error}",
        provider=provider,
        transient=transient,
    ) from last_error


def _is_transient_status(status_code: int | None) -> bool:
    return int(status_code or 0) in {408, 425, 429, 500, 502, 503, 504}


def _redact_provider_error(message: object) -> str:
    text = str(message or "").strip()
    if not text:
        return ""
    replacements = [
        (r'("KC-API-KEY"\s*:\s*")[^"]+', r"\1[redacted]"),
        (r'("KC-API-SIGN"\s*:\s*")[^"]+', r"\1[redacted]"),
        (r'("KC-API-PASSPHRASE"\s*:\s*")[^"]+', r"\1[redacted]"),
        (r'("apiKey"\s*:\s*")[^"]+', r"\1[redacted]"),
        (r'("apiSecret"\s*:\s*")[^"]+', r"\1[redacted]"),
        (r"(0x)[0-9a-fA-F]{64}", r"\1[redacted]"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text[:500]


def _as_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_datetime(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
