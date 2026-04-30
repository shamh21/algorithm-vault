"""Centralized live order entry, cancellation, and protective-exit logic."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..extensions import db
from ..models import AuditLog, Fill, Order, PositionSnapshot, Setting
from .hyperliquid_client import HyperliquidClient
from .market_data import MarketDataService
from .risk_engine import RiskEngine
from .db_retry import commit_with_retry
from .trading_connections import TradingConnectionService


TERMINAL_ORDER_STATUSES = {"cancelled", "filled", "rejected", "failed"}
OPEN_ORDER_STATUSES = {"open", "submitted", "pending"}


@dataclass(slots=True)
class OrderIntent:
    """Normalized order request shared by routes and strategies."""

    symbol: str
    side: str
    quantity: float
    mode: str
    order_type: str = "market"
    limit_price: float | None = None
    reduce_only: bool = False
    leverage: float = 1.0
    stop_loss: float | None = None
    take_profit: float | None = None
    strategy_name: str | None = None
    timeframe: str | None = None
    slippage_pct: float = 0.0
    user_id: int | None = None
    trading_connection_id: int | None = None
    idempotency_key: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper()
        self.side = self.side.lower()
        self.mode = self.mode.lower()
        self.order_type = self.order_type.lower()

        if not self.symbol:
            raise ValueError("symbol is required")

        if self.side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported order side: {self.side}")

        if self.order_type not in {"market", "limit"}:
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
            raise ValueError("limit_price must be positive")


class OrderManager:
    """Owns live order submission, cancellation, and exits."""

    def __init__(
        self,
        config: dict[str, Any],
        client: HyperliquidClient,
        market_data: MarketDataService,
        risk_engine: RiskEngine,
        trading_connections: TradingConnectionService | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.market_data = market_data
        self.risk_engine = risk_engine
        self.trading_connections = trading_connections

    def place_order(self, intent: OrderIntent) -> Order:
        if intent.mode != "live":
            raise ValueError("Live-only mode is enabled; non-live orders are not supported.")
        existing = Order.query.filter_by(client_order_id=intent.idempotency_key).one_or_none()
        if existing is not None:
            return existing

        if intent.mode == "live" and str(intent.metadata.get("optimizer_profile", "")).lower() in {"aggressive_1h", "extreme_roi_experimental", "dynamic_intraday"}:
            intent.metadata = {
                **dict(intent.metadata or {}),
                "account_equity_usd": self._live_account_equity_usd(intent.user_id, intent.trading_connection_id),
            }

        market_price = self._safe_market_price(intent.symbol, "live")
        can_trade = self._can_trade(intent)

        decision = self.risk_engine.evaluate(intent, market_price, can_trade)

        order = self._create_order(intent, decision.approved)

        if not decision.approved:
            order.status = "rejected"
            order.rejection_reason = decision.reason
            commit_with_retry()
            self.risk_engine.log_rejection(
                decision,
                {
                    "symbol": intent.symbol,
                    "mode": intent.mode,
                    "side": intent.side,
                    "user_id": intent.user_id,
                    "trading_connection_id": intent.trading_connection_id,
                },
                order.id,
            )
            return order

        try:
            self._exchange_submit(order, intent)
        except Exception as exc:  # noqa: BLE001
            self._handle_exchange_failure(order, intent, exc)

        if order.status not in {"failed", "rejected"}:
            self._record_snapshot_for_symbol(order.symbol, order.mode, order.user_id, order.trading_connection_id)

        return order

    def cancel_order(self, order_id: int) -> Order:
        order = Order.query.get_or_404(order_id)

        if order.status in TERMINAL_ORDER_STATUSES:
            return order

        try:
            if order.mode == "live" and order.exchange_order_id:
                self._connector_for_order(order).cancel_order(order.mode, order.symbol, order.exchange_order_id)

            order.status = "cancelled"
            self._audit(
                "cancel",
                f"Cancelled order {order.client_order_id}",
                {
                    "order_id": order.id,
                    "symbol": order.symbol,
                    "mode": order.mode,
                    "user_id": order.user_id,
                    "trading_connection_id": order.trading_connection_id,
                },
            )
            commit_with_retry()
        except Exception as exc:  # noqa: BLE001
            order.rejection_reason = str(exc)
            self._audit(
                "cancel_failed",
                f"Failed to cancel order {order.client_order_id}",
                {
                    "order_id": order.id,
                    "symbol": order.symbol,
                    "mode": order.mode,
                    "error": str(exc),
                    "user_id": order.user_id,
                    "trading_connection_id": order.trading_connection_id,
                },
            )
            commit_with_retry()
            raise

        return order

    def sync_exchange_orders(
        self,
        mode: str,
        user_id: int | None = None,
        trading_connection_id: int | None = None,
    ) -> list[dict[str, Any]]:
        if mode != "live":
            return []
        if self.trading_connections is None:
            return []
        if user_id is None:
            return []

        return self.trading_connections.connector_for_user(user_id, trading_connection_id).account_snapshot(mode).open_orders

    def cancel_all(
        self,
        mode: str,
        user_id: int | None = None,
        trading_connection_id: int | None = None,
    ) -> list[dict[str, Any]]:
        if mode != "live":
            return []

        results: list[dict[str, Any]] = []
        if user_id is not None and self.trading_connections is not None:
            results = self.trading_connections.connector_for_user(user_id, trading_connection_id).cancel_all_orders(mode)

        query = Order.query.filter_by(mode=mode).filter(Order.status.in_(list(OPEN_ORDER_STATUSES)))
        if user_id is not None:
            query = query.filter_by(user_id=int(user_id))
        if trading_connection_id is not None:
            query = query.filter_by(trading_connection_id=int(trading_connection_id))
        local_orders = query.all()

        for order in local_orders:
            order.status = "cancelled"

        commit_with_retry()
        return results

    def flatten_positions(
        self,
        mode: str,
        user_id: int | None = None,
        trading_connection_id: int | None = None,
    ) -> list[dict[str, Any]]:
        if mode != "live":
            return []

        if user_id is None or self.trading_connections is None:
            return []

        return self.trading_connections.connector_for_user(user_id, trading_connection_id).flatten_all_positions(mode)

    def _live_account_equity_usd(self, user_id: int | None, trading_connection_id: int | None) -> float:
        if user_id is None or self.trading_connections is None:
            return 0.0
        try:
            snapshot = self.trading_connections.account_snapshot(user_id, "live", trading_connection_id)
        except Exception:  # noqa: BLE001
            return 0.0
        return sum(
            self._safe_float(balance.get("value"))
            for balance in snapshot.balances
            if str(balance.get("asset", "")).upper() in {"USDC", "USD", "USDT"}
        )

    def current_position(
        self,
        symbol: str,
        mode: str,
        user_id: int | None = None,
        trading_connection_id: int | None = None,
    ) -> dict[str, Any]:
        symbol = symbol.upper()

        if mode != "live":
            raise ValueError("Live-only mode is enabled; non-live positions are not supported.")

        if user_id is None or self.trading_connections is None:
            return {
                "symbol": symbol,
                "quantity": 0.0,
                "entry_price": 0.0,
                "mark_price": self._safe_market_price(symbol, mode),
                "unrealized_pnl": 0.0,
                "leverage": 1.0,
            }

        for position in self.trading_connections.connector_for_user(user_id, trading_connection_id).get_positions(mode):
            if position.get("symbol") == symbol:
                return position

        return {
            "symbol": symbol,
            "quantity": 0.0,
            "entry_price": 0.0,
            "mark_price": self._safe_market_price(symbol, mode),
            "unrealized_pnl": 0.0,
            "leverage": 1.0,
        }

    def enforce_protective_exit(
        self,
        symbol: str,
        mode: str,
        user_id: int | None = None,
        trading_connection_id: int | None = None,
    ) -> Order | None:
        symbol = symbol.upper()
        position = self.current_position(symbol, mode, user_id, trading_connection_id)
        qty = self._safe_float(position.get("quantity"))

        if abs(qty) < 1e-9:
            return None

        source_order = (
            Order.query.filter_by(symbol=symbol, mode=mode, reduce_only=False, user_id=user_id)
            .filter(Order.status.in_(["filled", "open", "submitted"]))
            .order_by(Order.created_at.desc())
            .first()
        )

        if source_order is None:
            return None

        metadata = dict(source_order.details or {})

        if metadata.get("protective_triggered"):
            return None

        mark = self._safe_market_price(symbol, "live")
        if mark <= 0:
            return None

        trigger_reason = self._protective_trigger_reason(source_order, qty, mark)

        if trigger_reason is None:
            return None

        metadata["protective_triggered"] = trigger_reason
        metadata["protective_trigger_price"] = mark
        source_order.details = metadata

        exit_side = "sell" if qty > 0 else "buy"

        exit_order = self.place_order(
            OrderIntent(
                symbol=symbol,
                side=exit_side,
                quantity=abs(qty),
                mode=mode,
                order_type="market",
                reduce_only=True,
                leverage=1.0,
                stop_loss=source_order.stop_loss,
                take_profit=source_order.take_profit,
                user_id=user_id,
                trading_connection_id=trading_connection_id,
                idempotency_key=f"protective-{source_order.id}-{trigger_reason}",
                metadata={
                    "protective_exit": trigger_reason,
                    "source_order_id": source_order.id,
                    "vault_cycle_id": metadata.get("vault_cycle_id"),
                    "vault_leg_id": metadata.get("vault_leg_id"),
                    "execution_mode": metadata.get("execution_mode"),
                    "optimizer_profile": metadata.get("optimizer_profile"),
                    "experimental": metadata.get("experimental"),
                    "risk_label": metadata.get("risk_label"),
                    "algorithm_profile": metadata.get("algorithm_profile"),
                    "consumer_vault": metadata.get("consumer_vault"),
                    "execution_style": metadata.get("execution_style"),
                    "expected_stop_loss": metadata.get("expected_stop_loss"),
                    "expected_take_profit": metadata.get("expected_take_profit"),
                    "signal_metadata": metadata.get("signal_metadata", {}),
                    "edge_score": metadata.get("edge_score"),
                    "cost_drag_bps": metadata.get("cost_drag_bps"),
                },
            )
        )

        commit_with_retry()
        return exit_order

    def _create_order(self, intent: OrderIntent, approved: bool) -> Order:
        order = Order(
            user_id=intent.user_id,
            trading_connection_id=intent.trading_connection_id,
            client_order_id=intent.idempotency_key,
            mode=intent.mode,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            status="pending",
            strategy_name=intent.strategy_name,
            quantity=float(intent.quantity),
            limit_price=intent.limit_price,
            reduce_only=intent.reduce_only,
            leverage=float(intent.leverage),
            stop_loss=intent.stop_loss,
            take_profit=intent.take_profit,
            risk_status="approved" if approved else "rejected",
        )

        order.details = {
            **dict(intent.metadata or {}),
            "timeframe": intent.timeframe,
            "slippage_pct": intent.slippage_pct,
            "user_id": intent.user_id,
            "trading_connection_id": intent.trading_connection_id,
        }

        db.session.add(order)
        db.session.flush()

        return order

    def _exchange_submit(self, order: Order, intent: OrderIntent) -> None:
        max_slippage = self._safe_float(self.config.get("MAX_SLIPPAGE_PCT"), 0.0)
        slippage_pct = max(intent.slippage_pct, max_slippage / 2)

        if intent.user_id is not None and self.trading_connections is not None:
            connector = self.trading_connections.connector_for_user(intent.user_id, intent.trading_connection_id)
            response = connector.place_order(
                intent.mode,
                intent.symbol,
                intent.side,
                intent.quantity,
                intent.order_type,
                intent.limit_price,
                intent.reduce_only,
                intent.leverage,
                slippage_pct,
            )
        else:
            response = self.client.place_order(
                intent.mode,
                intent.symbol,
                intent.side,
                intent.quantity,
                intent.order_type,
                intent.limit_price,
                intent.reduce_only,
                intent.leverage,
                slippage_pct,
            )

        order.exchange_order_id = response.get("exchange_order_id")
        order.average_fill_price = response.get("fill_price") or response.get("submitted_price")
        order.status = response.get("status", "submitted")
        order.filled_quantity = intent.quantity if order.status == "filled" else 0.0

        details = dict(order.details or {})
        details["exchange_response"] = response
        order.details = details

        if order.status == "filled" and order.average_fill_price:
            db.session.add(
                Fill(
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.filled_quantity or order.quantity,
                    price=order.average_fill_price,
                    fee=0.0,
                    pnl=0.0,
                    simulated=False,
                )
            )

        self._audit(
            "submit",
            f"Submitted {order.mode} order for {order.symbol}.",
            {
                "order_id": order.id,
                "status": order.status,
                "mode": order.mode,
                "user_id": order.user_id,
                "trading_connection_id": order.trading_connection_id,
            },
        )

        commit_with_retry()

    def _handle_exchange_failure(self, order: Order, intent: OrderIntent, exc: Exception) -> None:
        order.status = "failed"
        order.rejection_reason = str(exc)

        if intent.mode == "live":
            Setting.set_json("live_trading_blocked", True)
            audit_message = "Live order failed; live trading was blocked until review."
        else:
            audit_message = f"{intent.mode} order failed."

        self._audit(
            "exchange_failure",
            audit_message,
            {
                "order_id": order.id,
                "mode": intent.mode,
                "symbol": intent.symbol,
                "error": str(exc),
                "user_id": intent.user_id,
                "trading_connection_id": intent.trading_connection_id,
            },
        )

        commit_with_retry()

    def _record_snapshot_for_symbol(
        self,
        symbol: str,
        mode: str,
        user_id: int | None = None,
        trading_connection_id: int | None = None,
    ) -> None:
        position = self.current_position(symbol, mode, user_id, trading_connection_id)

        quantity = self._safe_float(position.get("quantity"))
        mark_price = self._safe_float(position.get("mark_price")) or self._safe_float(position.get("entry_price"))

        snapshot = PositionSnapshot(
            mode=mode,
            user_id=user_id,
            trading_connection_id=trading_connection_id,
            symbol=symbol.upper(),
            quantity=quantity,
            average_entry_price=self._safe_float(position.get("entry_price")),
            mark_price=mark_price,
            unrealized_pnl=self._safe_float(position.get("unrealized_pnl")),
            leverage=self._safe_float(position.get("leverage"), 1.0),
            notional=abs(quantity) * mark_price,
            snapshot_time=datetime.now(timezone.utc),
        )

        db.session.add(snapshot)
        commit_with_retry()

    @staticmethod
    def _protective_trigger_reason(order: Order, qty: float, mark: float) -> str | None:
        if qty > 0:
            if order.stop_loss and mark <= order.stop_loss:
                return "stop_loss"
            if order.take_profit and mark >= order.take_profit:
                return "take_profit"

        if qty < 0:
            if order.stop_loss and mark >= order.stop_loss:
                return "stop_loss"
            if order.take_profit and mark <= order.take_profit:
                return "take_profit"

        return None

    def _audit(self, action: str, message: str, details: dict[str, Any]) -> None:
        audit = AuditLog(
            category="orders",
            action=action,
            message=message,
            user_id=details.get("user_id"),
            trading_connection_id=details.get("trading_connection_id"),
        )
        audit.details = details
        db.session.add(audit)

    def _can_trade(self, intent: OrderIntent) -> bool:
        if intent.mode != "live":
            return False
        if self.trading_connections is None:
            return self.client.can_trade(intent.mode)
        return self.trading_connections.can_trade(intent.user_id, intent.mode, intent.trading_connection_id)

    def _connector_for_order(self, order: Order):
        if self.trading_connections is None or order.user_id is None:
            return self.client
        return self.trading_connections.connector_for_user(order.user_id, order.trading_connection_id)

    def _safe_market_price(self, symbol: str, mode: str) -> float:
        try:
            return self.market_data.get_mid_price(symbol.upper(), mode)
        except Exception:  # noqa: BLE001
            return 0.0

    def _safe_order_book(self, symbol: str, mode: str) -> dict[str, Any]:
        try:
            return self.market_data.get_order_book(symbol.upper(), mode)
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default

        try:
            return float(value)
        except (TypeError, ValueError):
            return default
