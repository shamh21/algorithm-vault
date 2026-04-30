"""Execution venue abstractions for live trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .hyperliquid_client import HyperliquidClient
from .market_data import MarketDataService


VALID_ORDER_SIDES = {"buy", "sell"}
VALID_ORDER_TYPES = {"market", "limit"}


@dataclass(slots=True)
class VenueOrderRequest:
    """Exchange-neutral order request."""

    mode: str
    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    limit_price: float | None = None
    reduce_only: bool = False
    leverage: float = 1.0
    slippage_pct: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.side = self.side.lower()
        self.order_type = self.order_type.lower()

        if not self.mode:
            raise ValueError("mode is required")

        if not self.symbol:
            raise ValueError("symbol is required")

        if self.side not in VALID_ORDER_SIDES:
            raise ValueError(f"Unsupported order side: {self.side}")

        if self.order_type not in VALID_ORDER_TYPES:
            raise ValueError(f"Unsupported order type: {self.order_type}")

        if self.quantity <= 0:
            raise ValueError("quantity must be positive")

        if self.leverage <= 0:
            raise ValueError("leverage must be positive")

        if self.slippage_pct < 0:
            raise ValueError("slippage_pct cannot be negative")

        if self.order_type == "limit" and self.limit_price is None:
            raise ValueError("limit_price is required for limit orders")

        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be positive when provided")


@dataclass(slots=True)
class VenueOrderResult:
    """Exchange-neutral order result."""

    status: str
    exchange_order_id: str | None = None
    fill_price: float | None = None
    submitted_price: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exchange_order_id": self.exchange_order_id,
            "fill_price": self.fill_price,
            "submitted_price": self.submitted_price,
            "raw": dict(self.raw),
        }


@runtime_checkable
class ExecutionVenue(Protocol):
    """Protocol implemented by concrete execution venues."""

    def get_balances(self, mode: str) -> list[dict[str, Any]]:
        ...

    def get_positions(self, mode: str) -> list[dict[str, Any]]:
        ...

    def get_open_orders(self, mode: str) -> list[dict[str, Any]]:
        ...

    def get_recent_fills(self, mode: str) -> list[dict[str, Any]]:
        ...

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        mode: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def get_mid_price(self, symbol: str, mode: str) -> float:
        ...

    def get_order_book(self, symbol: str, mode: str) -> dict[str, Any]:
        ...

    def place_order(self, request: VenueOrderRequest) -> VenueOrderResult:
        ...

    def cancel_order(self, mode: str, symbol: str, exchange_order_id: str) -> dict[str, Any]:
        ...

    def cancel_all_orders(self, mode: str) -> list[dict[str, Any]]:
        ...

    def flatten_all_positions(self, mode: str) -> list[dict[str, Any]]:
        ...


class HyperliquidVenue:
    """ExecutionVenue adapter for the existing Hyperliquid client."""

    def __init__(self, client: HyperliquidClient, market_data: MarketDataService) -> None:
        self.client = client
        self.market_data = market_data

    def get_balances(self, mode: str) -> list[dict[str, Any]]:
        return self.client.get_balances(mode)

    def get_positions(self, mode: str) -> list[dict[str, Any]]:
        return self.client.get_positions(mode)

    def get_open_orders(self, mode: str) -> list[dict[str, Any]]:
        return self.client.get_open_orders(mode)

    def get_recent_fills(self, mode: str) -> list[dict[str, Any]]:
        return self.client.get_recent_fills(mode)

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        mode: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.market_data.get_candles(symbol, timeframe, mode=mode, limit=limit)

    def get_mid_price(self, symbol: str, mode: str) -> float:
        price = float(self.market_data.get_mid_price(symbol, mode))

        if price <= 0:
            raise ValueError(f"Invalid mid price for {symbol}: {price}")

        return price

    def get_order_book(self, symbol: str, mode: str) -> dict[str, Any]:
        return self.market_data.get_order_book(symbol, mode)

    def place_order(self, request: VenueOrderRequest) -> VenueOrderResult:
        response = self.client.place_order(
            request.mode,
            request.symbol,
            request.side,
            request.quantity,
            request.order_type,
            request.limit_price,
            request.reduce_only,
            request.leverage,
            request.slippage_pct,
        )

        raw = response.get("raw", response)

        return VenueOrderResult(
            status=str(response.get("status", "submitted")),
            exchange_order_id=response.get("exchange_order_id"),
            fill_price=_optional_float(response.get("fill_price")),
            submitted_price=_optional_float(response.get("submitted_price")),
            raw=raw if isinstance(raw, dict) else {"response": raw},
        )

    def cancel_order(self, mode: str, symbol: str, exchange_order_id: str) -> dict[str, Any]:
        if not exchange_order_id:
            raise ValueError("exchange_order_id is required")

        return self.client.cancel_order(mode, symbol, exchange_order_id)

    def cancel_all_orders(self, mode: str) -> list[dict[str, Any]]:
        return self.client.cancel_all_orders(mode)

    def flatten_all_positions(self, mode: str) -> list[dict[str, Any]]:
        return self.client.flatten_all_positions(mode)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None
