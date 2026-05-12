"""Deterministic fill simulation used by historical backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


VALID_SIDES = {"buy", "sell"}
VALID_ORDER_TYPES = {"market", "limit"}


@dataclass(slots=True)
class FillSimulation:
    """Result of a simulated fill decision."""

    status: str
    fill_price: float
    filled_quantity: float
    fee: float
    fill_fraction: float
    partial: bool
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "fill_price": self.fill_price,
            "filled_quantity": self.filled_quantity,
            "fee": self.fee,
            "fill_fraction": self.fill_fraction,
            "partial": self.partial,
            "details": dict(self.details),
        }


class FillModel(Protocol):
    """Protocol for pluggable deterministic or stochastic fill models."""

    def simulate(
        self,
        *,
        side: str,
        quantity: float,
        market_price: float,
        order_type: str,
        limit_price: float | None,
        slippage_pct: float,
        fee_rate: float,
        order_book: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FillSimulation:
        ...


class DeterministicFillModel:
    """Reproducible fill model with slippage, fees, and partial-fill hooks."""

    def simulate(
        self,
        *,
        side: str,
        quantity: float,
        market_price: float,
        order_type: str,
        limit_price: float | None,
        slippage_pct: float,
        fee_rate: float,
        order_book: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FillSimulation:
        metadata = metadata or {}

        side = str(side).lower()
        order_type = str(order_type).lower()

        if side not in VALID_SIDES:
            return self._rejected("invalid_side", side=side)

        if order_type not in VALID_ORDER_TYPES:
            return self._rejected("invalid_order_type", order_type=order_type)

        quantity = self._safe_float(quantity)
        market_price = self._safe_float(market_price)
        limit_price_value = self._safe_float(limit_price)
        slippage_pct = max(self._safe_float(slippage_pct), 0.0)
        fee_rate = max(self._safe_float(fee_rate), 0.0)

        if quantity <= 0:
            return self._rejected("invalid_quantity", quantity=quantity)

        if order_type == "limit" and limit_price_value <= 0:
            return self._rejected("invalid_limit_price", limit_price=limit_price)

        reference_price = limit_price_value if order_type == "limit" else market_price

        if reference_price <= 0:
            return self._rejected("invalid_reference_price", reference_price=reference_price)

        bid, ask = self._best_bid_ask(order_book)
        spread_bps = self._spread_bps(bid, ask, market_price)

        slipped_price = self._apply_slippage(
            side=side,
            reference_price=reference_price,
            slippage_pct=slippage_pct,
            bid=bid,
            ask=ask,
        )

        if order_type == "limit" and limit_price_value > 0:
            if side == "buy" and slipped_price > limit_price_value:
                return FillSimulation(
                    "open",
                    round(limit_price_value, 6),
                    0.0,
                    0.0,
                    0.0,
                    False,
                    {"reason": "limit_not_marketable", "spread_bps": spread_bps},
                )

            if side == "sell" and slipped_price < limit_price_value:
                return FillSimulation(
                    "open",
                    round(limit_price_value, 6),
                    0.0,
                    0.0,
                    0.0,
                    False,
                    {"reason": "limit_not_marketable", "spread_bps": spread_bps},
                )

            slipped_price = limit_price_value

        requested_fraction = self._clamp(
            self._safe_float(metadata.get("simulation_fill_fraction", metadata.get("paper_fill_fraction", 1.0))),
            0.0,
            1.0,
        )

        depth_fraction = self._depth_fraction(quantity, side, slipped_price, order_book)
        fill_fraction = min(requested_fraction, depth_fraction)

        filled_quantity = quantity * fill_fraction
        partial = 0.0 < filled_quantity < quantity

        if filled_quantity <= 0:
            status = "open"
        elif partial:
            status = "partially_filled"
        else:
            status = "filled"

        fill_price = round(slipped_price, 6)
        filled_quantity = round(filled_quantity, 10)
        fee = round(abs(fill_price * filled_quantity) * fee_rate, 10)

        return FillSimulation(
            status=status,
            fill_price=fill_price,
            filled_quantity=filled_quantity,
            fee=fee,
            fill_fraction=fill_fraction,
            partial=partial,
            details={
                "spread_bps": spread_bps,
                "requested_fill_fraction": requested_fraction,
                "depth_fraction": depth_fraction,
                "slippage_pct": slippage_pct,
                "fee_rate": fee_rate,
            },
        )

    @staticmethod
    def _apply_slippage(
        *,
        side: str,
        reference_price: float,
        slippage_pct: float,
        bid: float,
        ask: float,
    ) -> float:
        if side == "buy":
            slipped_price = reference_price * (1 + slippage_pct)
            return max(slipped_price, ask) if ask > 0 else slipped_price

        slipped_price = reference_price * (1 - slippage_pct)
        return min(slipped_price, bid) if bid > 0 else slipped_price

    @staticmethod
    def _best_bid_ask(order_book: dict[str, Any] | None) -> tuple[float, float]:
        if not order_book:
            return 0.0, 0.0

        levels = order_book.get("levels")

        if isinstance(levels, list) and len(levels) >= 2:
            bids = levels[0] or []
            asks = levels[1] or []

            bid = DeterministicFillModel._level_price(bids[0]) if bids else 0.0
            ask = DeterministicFillModel._level_price(asks[0]) if asks else 0.0

            return bid, ask

        bid = DeterministicFillModel._safe_float(
            order_book.get("bid") or order_book.get("best_bid")
        )
        ask = DeterministicFillModel._safe_float(
            order_book.get("ask") or order_book.get("best_ask")
        )

        return bid, ask

    @staticmethod
    def _depth_fraction(
        quantity: float,
        side: str,
        fill_price: float,
        order_book: dict[str, Any] | None,
    ) -> float:
        if quantity <= 0 or fill_price <= 0 or not order_book:
            return 1.0

        levels = order_book.get("levels")

        if not isinstance(levels, list) or len(levels) < 2:
            return 1.0

        book_side = levels[1] if side == "buy" else levels[0]
        available = 0.0

        for level in book_side:
            price = DeterministicFillModel._level_price(level)
            size = DeterministicFillModel._level_size(level)

            if price <= 0 or size <= 0:
                continue

            if side == "buy" and price <= fill_price:
                available += size
            elif side == "sell" and price >= fill_price:
                available += size

        if available <= 0:
            return 0.0

        return min(1.0, available / quantity)

    @staticmethod
    def _level_price(level: Any) -> float:
        if isinstance(level, dict):
            return DeterministicFillModel._safe_float(
                level.get("px", level.get("price", 0.0))
            )

        if isinstance(level, (list, tuple)) and level:
            return DeterministicFillModel._safe_float(level[0])

        return 0.0

    @staticmethod
    def _level_size(level: Any) -> float:
        if isinstance(level, dict):
            return DeterministicFillModel._safe_float(
                level.get("sz", level.get("size", 0.0))
            )

        if isinstance(level, (list, tuple)) and len(level) > 1:
            return DeterministicFillModel._safe_float(level[1])

        return 0.0

    @staticmethod
    def _spread_bps(bid: float, ask: float, market_price: float) -> float:
        midpoint = market_price if market_price > 0 else ((bid + ask) / 2 if bid > 0 and ask > 0 else 0.0)

        if midpoint <= 0 or bid <= 0 or ask <= 0 or ask < bid:
            return 0.0

        return ((ask - bid) / midpoint) * 10_000

    @staticmethod
    def _safe_float(value: Any) -> float:
        if value is None:
            return 0.0

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(value, upper))

    @staticmethod
    def _rejected(reason: str, **details: Any) -> FillSimulation:
        return FillSimulation(
            status="rejected",
            fill_price=0.0,
            filled_quantity=0.0,
            fee=0.0,
            fill_fraction=0.0,
            partial=False,
            details={"reason": reason, **details},
        )
