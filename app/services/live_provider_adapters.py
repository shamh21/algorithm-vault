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
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import requests

from .hyperliquid_client import ClientSnapshot


logger = logging.getLogger(__name__)

BINANCE_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT"}
KUCOIN_SYMBOLS = {"BTC": "XBTUSDTM", "ETH": "ETHUSDTM", "SOL": "SOLUSDTM", "XRP": "XRPUSDTM"}
KUCOIN_CONTRACT_SPECS = {
    "XBTUSDTM": {"contract_size": 0.001, "size_step": 1, "min_size": 1},
    "ETHUSDTM": {"contract_size": 0.01, "size_step": 1, "min_size": 1},
}
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


class ProviderRequestError(RuntimeError):
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
            except Exception:
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
        self._time_offset_ms = 0
        self._last_time_sync_monotonic = 0.0

    def can_trade(self, mode: str) -> bool:
        if mode != "live" or not bool(self.config.get("ENABLE_LIVE_TRADING", False)):
            return False
        self._account_overview()
        self._position_mode()
        return True

    def account_snapshot(self, mode: str) -> ClientSnapshot:
        if mode != "live":
            return ClientSnapshot(mode, [], [], [], [], ["KuCoin connector supports live futures only."])
        try:
            account = self._account_overview()
            return ClientSnapshot(mode, self._balances(account), self.get_positions(mode), self.get_open_orders(mode), self.get_recent_fills(mode), [])
        except Exception as exc:  # noqa: BLE001
            return ClientSnapshot(mode, [], [], [], [], [f"KuCoin unavailable: {exc}"])

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
            raise RuntimeError("KuCoin connector supports live futures only.")
        venue_symbol = self._symbol(symbol)
        contracts = self._contract_size(venue_symbol, quantity)
        body: dict[str, Any] = {
            "clientOid": f"av-{uuid.uuid4().hex}",
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
            body["timeInForce"] = "GTC"
        response = self._signed("POST", self._path("KUCOIN_ORDERS_PATH", "/api/v1/orders"), body=body)
        data = _response_data(response)
        if not isinstance(data, dict):
            data = {}
        return self._normalize_order_response(data, fallback_client_oid=str(body["clientOid"]), submitted_price=limit_price, raw=response)

    def cancel_order(self, mode: str, symbol: str, exchange_order_id: str) -> dict[str, Any]:
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
        try:
            response = self._signed("GET", self._path("KUCOIN_FILLS_PATH", "/api/v1/recentDoneOrders"))
        except Exception:
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
        raise RuntimeError("KuCoin withdrawals are intentionally disabled in this app.")

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
        now = datetime.now(timezone.utc)
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
        params = dict(params or {})
        body_text = json.dumps(body or {}, separators=(",", ":")) if body else ""
        endpoint = path + (f"?{urlencode(params, doseq=True)}" if params else "")
        payload: Any = None
        for attempt in range(2):
            timestamp = str(self._timestamp_ms(force_sync=attempt > 0))
            pre_sign = f"{timestamp}{method.upper()}{endpoint}{body_text}"
            signature = base64.b64encode(hmac.new(self.credentials.api_secret.encode("utf-8"), pre_sign.encode("utf-8"), hashlib.sha256).digest()).decode("utf-8")
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
            payload = _request_with_retries(
                self.session,
                method,
                self._base_url() + path,
                provider="KuCoin",
                attempts=self._retry_attempts(),
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
            self._base_url() + self._path("KUCOIN_SERVER_TIME_PATH", "/api/v1/timestamp"),
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

    def _timeout(self) -> float:
        return max(1.0, _safe_float(self.config.get("PROVIDER_TIMEOUT_SECONDS"), 10.0))

    def _retry_attempts(self) -> int:
        return max(1, int(_safe_float(self.config.get("PROVIDER_RETRY_ATTEMPTS", self.config.get("EXCHANGE_RETRY_ATTEMPTS", 3)), 3)))

    def _retry_sleep_seconds(self) -> float:
        return max(0.0, _safe_float(self.config.get("PROVIDER_RETRY_SLEEP_SECONDS", self.config.get("EXCHANGE_RETRY_SLEEP_SECONDS", 0.5)), 0.5))

    def _path(self, key: str, default: str) -> str:
        return str(self.config.get(key, default)).strip() or default

    def _margin_mode(self) -> str:
        value = str(self.metadata.get("margin_mode") or self.config.get("KUCOIN_MARGIN_MODE", "ISOLATED")).strip().upper()
        return value if value in {"ISOLATED", "CROSS"} else "ISOLATED"

    def _position_side(self) -> str:
        value = str(self.metadata.get("position_side") or self.config.get("KUCOIN_POSITION_SIDE", "BOTH")).strip().upper()
        return value if value in {"BOTH", "LONG", "SHORT"} else "BOTH"

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
            "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
            "timeframe": timeframe,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }

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
        return format(float(value), "f").rstrip("0").rstrip(".")


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
        except Exception:
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
        except Exception:
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
        self._enforce_notional_cap(quantity, limit_price)
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
        if expires_at is None or expires_at <= datetime.now(timezone.utc):
            raise RuntimeError("Wallet delegation is expired or missing an expiry.")
        if _safe_float(self.metadata.get("max_notional_usd")) <= 0:
            raise RuntimeError("Uniswap max notional cap is required.")
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

    def _enforce_notional_cap(self, quantity: float, limit_price: float | None) -> None:
        if limit_price is None or limit_price <= 0:
            return
        notional = abs(float(quantity) * float(limit_price))
        if notional > _safe_float(self.metadata.get("max_notional_usd")):
            raise RuntimeError("Uniswap delegated order exceeds max notional cap.")

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
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
