"""Centralized live order entry, cancellation, and protective-exit logic."""

from __future__ import annotations

import uuid
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..extensions import db
from ..ml.decision_engine import MLDecisionEngine
from ..ml.signal_model import MLSignalModel
from ..models import AuditLog, Fill, Order, PositionSnapshot, Setting
from .hyperliquid_client import HyperliquidClient
from .market_data import MarketDataService
from .risk_engine import RiskEngine
from .db_retry import commit_with_retry
from .trading_connections import TradingConnectionService


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

        if intent.mode == "live" and (
            str(intent.metadata.get("optimizer_profile", "")).lower() in {"aggressive_1h", "extreme_roi_experimental", "dynamic_intraday"}
            or bool(intent.metadata.get("one_h10_vault"))
            or str(intent.metadata.get("ml_horizon", "")).lower() == "1h10"
        ):
            intent.metadata = {
                **dict(intent.metadata or {}),
                "account_equity_usd": self._live_account_equity_usd(intent.user_id, intent.trading_connection_id),
            }

        market_price = self._safe_market_price(
            intent.symbol,
            "live",
            user_id=intent.user_id,
            trading_connection_id=intent.trading_connection_id,
            provider=intent.metadata.get("provider"),
            venue_symbol=intent.metadata.get("venue_symbol"),
        )
        if market_price <= 0:
            market_price = self._safe_float(intent.metadata.get("reference_price"))
        if not intent.reduce_only:
            intent = self._apply_ml_execution_policy(intent, market_price)
        can_trade = self._can_trade(intent)

        risk_started_at = time.perf_counter()
        decision = self.risk_engine.evaluate(intent, market_price, can_trade)
        risk_latency_ms = (time.perf_counter() - risk_started_at) * 1000

        order = self._create_order(intent, decision.approved)
        details = dict(order.details or {})
        details["risk_decision"] = decision.as_dict()
        details["risk_latency_ms"] = risk_latency_ms
        if decision.details.get("ml_policy_decisions"):
            details["ml_policy_decisions"] = decision.details.get("ml_policy_decisions")
        if decision.reason:
            details["risk_rejection_reason"] = decision.reason
        if not decision.approved:
            details["blocker_category"] = self._blocker_category(decision.rule_name, decision.reason, decision.details)
        order.details = details

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

        adaptive_slippage = decision.details.get("adaptive_slippage") if isinstance(decision.details, dict) else None
        if isinstance(adaptive_slippage, dict):
            intent.metadata = {
                **dict(intent.metadata or {}),
                "adaptive_slippage": adaptive_slippage,
                "adaptive_slippage_pct": adaptive_slippage.get("max_acceptable_pct"),
            }
            details["adaptive_slippage"] = adaptive_slippage
            order.details = details

        try:
            self._exchange_submit(order, intent)
        except Exception as exc:  # noqa: BLE001
            self._handle_exchange_failure(order, intent, exc)

        if order.status not in {"failed", "rejected"}:
            self._record_snapshot_for_symbol(order.symbol, order.mode, order.user_id, order.trading_connection_id)

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

        if user_id is not None:
            self._invalidate_snapshot_cache(user_id, mode, trading_connection_id)
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

        flattened = self.trading_connections.connector_for_user(user_id, trading_connection_id).flatten_all_positions(mode)
        self._invalidate_snapshot_cache(user_id, mode, trading_connection_id)
        return flattened

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
                "mark_price": self._safe_market_price(symbol, mode, user_id=user_id, trading_connection_id=trading_connection_id),
                "unrealized_pnl": 0.0,
                "leverage": 1.0,
            }

        snapshot = self.trading_connections.account_snapshot(user_id, mode, trading_connection_id)
        if snapshot.alerts and not snapshot.positions:
            raise RuntimeError("; ".join(str(alert) for alert in snapshot.alerts))
        for position in snapshot.positions:
            position_symbol = str(position.get("symbol") or "")
            if position_symbol == symbol or position_symbol.upper() == symbol:
                return position

        return {
            "symbol": symbol,
            "quantity": 0.0,
            "entry_price": 0.0,
            "mark_price": 0.0 if mode == "live" else self._safe_market_price(symbol, mode, user_id=user_id, trading_connection_id=trading_connection_id),
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

        source_query = Order.query.filter_by(symbol=symbol, mode=mode, reduce_only=False, user_id=user_id)
        if trading_connection_id is not None:
            source_query = source_query.filter_by(trading_connection_id=trading_connection_id)
        source_order = (
            source_query.filter(Order.status.in_(["filled", "open", "submitted"]))
            .order_by(Order.created_at.desc())
            .first()
        )

        if source_order is None:
            return None

        metadata = dict(source_order.details or {})

        if metadata.get("protective_triggered"):
            return None

        if user_id is None and trading_connection_id is None:
            mark = self._safe_market_price(symbol, "live")
        else:
            mark = self._safe_market_price(symbol, "live", user_id=user_id, trading_connection_id=trading_connection_id)
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
                    "vault_cycle_name": metadata.get("vault_cycle_name"),
                    "consumer_vault": metadata.get("consumer_vault"),
                    "one_h10_vault": metadata.get("one_h10_vault"),
                    "ml_horizon": metadata.get("ml_horizon"),
                    "objective": metadata.get("objective"),
                    "ml_objective": metadata.get("ml_objective"),
                    "target_return_objective": metadata.get("target_return_objective"),
                    "ml_policy_required": metadata.get("ml_policy_required"),
                    "ml_governed_risk": metadata.get("ml_governed_risk"),
                    "provider": metadata.get("provider"),
                    "execution_venue": metadata.get("execution_venue"),
                    "collateral_asset": metadata.get("collateral_asset"),
                    "settlement_asset": metadata.get("settlement_asset"),
                    "allocation_cap_usd": metadata.get("allocation_cap_usd"),
                    "available_margin_usd": metadata.get("available_margin_usd"),
                    "account_equity_usd": metadata.get("account_equity_usd"),
                    "target_roi_pct": metadata.get("target_roi_pct"),
                    "target_amount_usd": metadata.get("target_amount_usd"),
                    "execution_style": metadata.get("execution_style"),
                    "expected_stop_loss": metadata.get("expected_stop_loss"),
                    "expected_take_profit": metadata.get("expected_take_profit"),
                    "signal_metadata": metadata.get("signal_metadata", {}),
                    "edge_score": metadata.get("edge_score"),
                    "cost_drag_bps": metadata.get("cost_drag_bps"),
                    "rapid_ml": metadata.get("rapid_ml"),
                    "rapid_ml_exit": bool(metadata.get("rapid_ml")),
                    "rapid_ml_exit_trigger": trigger_reason,
                    "rapid_ml_exit_source_order_id": source_order.id,
                    "rapid_ml_session_id": metadata.get("rapid_ml_session_id"),
                },
            )
        )

        commit_with_retry()
        return exit_order

    def _create_order(self, intent: OrderIntent, approved: bool) -> Order:
        vault_cycle_id = self._safe_int(intent.metadata.get("vault_cycle_id"))
        vault_leg_id = self._safe_int(intent.metadata.get("vault_leg_id"))
        order = Order(
            user_id=intent.user_id,
            trading_connection_id=intent.trading_connection_id,
            vault_cycle_id=vault_cycle_id if vault_cycle_id > 0 else None,
            vault_leg_id=vault_leg_id if vault_leg_id > 0 else None,
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
        adaptive_payload = (intent.metadata or {}).get("adaptive_slippage")
        adaptive_limit = self._safe_float(adaptive_payload.get("max_acceptable_pct"), intent.slippage_pct) if isinstance(adaptive_payload, dict) else intent.slippage_pct
        max_slippage = self._safe_float(
            (intent.metadata or {}).get("adaptive_slippage_pct"),
            adaptive_limit,
        )
        slippage_pct = self._submission_slippage_pct(intent.slippage_pct, max_slippage)

        if intent.user_id is not None and self.trading_connections is not None:
            connector = self.trading_connections.connector_for_user(intent.user_id, intent.trading_connection_id)
            submit_symbol = str(intent.metadata.get("venue_symbol") or intent.symbol).strip() or intent.symbol
            exchange_started_at = time.perf_counter()
            response = connector.place_order(
                intent.mode,
                submit_symbol,
                intent.side,
                intent.quantity,
                intent.order_type,
                intent.limit_price,
                intent.reduce_only,
                intent.leverage,
                slippage_pct,
            )
        else:
            submit_symbol = str(intent.metadata.get("venue_symbol") or intent.symbol).strip() or intent.symbol
            exchange_started_at = time.perf_counter()
            response = self.client.place_order(
                intent.mode,
                submit_symbol,
                intent.side,
                intent.quantity,
                intent.order_type,
                intent.limit_price,
                intent.reduce_only,
                intent.leverage,
                slippage_pct,
            )
        exchange_latency_ms = (time.perf_counter() - exchange_started_at) * 1000

        order.exchange_order_id = response.get("exchange_order_id")
        order.average_fill_price = response.get("fill_price") or response.get("submitted_price")
        order.status = response.get("status", "submitted")
        if response.get("error") or order.status in {"rejected", "failed"}:
            order.rejection_reason = str(response.get("error") or response.get("rejection_reason") or "Exchange rejected the order.")
        submitted_quantity = self._safe_float(response.get("submitted_quantity"))
        if submitted_quantity > 0:
            order.quantity = submitted_quantity
        order.filled_quantity = self._safe_float(response.get("filled_quantity"), order.quantity) if order.status == "filled" else 0.0

        details = dict(order.details or {})
        details["exchange_response"] = response
        details["exchange_latency_ms"] = exchange_latency_ms
        details["submitted_slippage_pct"] = slippage_pct
        if response.get("client_order_id"):
            details["provider_client_order_id"] = response.get("client_order_id")
        if response.get("fee") is not None:
            details["fee"] = response.get("fee")
        if response.get("slippage") is not None:
            details["reported_slippage"] = response.get("slippage")
        if response.get("error"):
            details["exchange_error"] = response.get("error")
        if order.status in {"rejected", "failed"} or response.get("error"):
            details["blocker_category"] = self._blocker_category("exchange_rejected", order.rejection_reason, response)
            if self._immediate_match_rejection(order.rejection_reason, response):
                details["execution_failure"] = "could_not_immediately_match"
                details["retryable_execution_failure"] = True
        order.details = details

        if order.status == "filled" and order.average_fill_price:
            fill = self._build_reconciled_fill(order, intent, response)
            details = dict(order.details or {})
            details["fill_reconciliation"] = fill.details
            order.details = details
            db.session.add(fill)

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

        self._invalidate_snapshot_cache(order.user_id, order.mode, order.trading_connection_id)
        commit_with_retry()

    def _invalidate_snapshot_cache(
        self,
        user_id: int | None,
        mode: str,
        trading_connection_id: int | None = None,
    ) -> None:
        if user_id is None or self.trading_connections is None:
            return
        self.trading_connections.invalidate_account_snapshot(
            user_id=user_id,
            mode=mode,
            connection_id=trading_connection_id,
        )

    def _build_reconciled_fill(self, order: Order, intent: OrderIntent, response: dict[str, Any]) -> Fill:
        matched_fill = self._matched_recent_fill(order, response)
        payloads = [response, response.get("raw") if isinstance(response, dict) else None]
        if matched_fill:
            payloads = [matched_fill, matched_fill.get("raw") if isinstance(matched_fill, dict) else None, *payloads]

        quantity = self._first_payload_float(payloads, ("filled_quantity", "size", "sz", "qty", "dealSize", "filledSize"), 0.0)
        if quantity <= 0:
            quantity = self._safe_float(order.filled_quantity or order.quantity)
        price = self._first_payload_float(payloads, ("fill_price", "avgPx", "avgPrice", "avgDealPrice", "price", "px"), 0.0)
        if price <= 0:
            price = self._safe_float(order.average_fill_price)

        fee_value = self._first_payload_optional_float(
            payloads,
            ("fee", "fees", "total_fee", "fee_usd", "feeUsd", "commission", "commissionAmount"),
        )
        fee_known = fee_value is not None
        fee_source = "exchange" if fee_known else "estimated"
        if fee_value is None:
            fee_value = self._estimated_trade_fee(intent, quantity, price)

        pnl_value = self._first_payload_optional_float(
            payloads,
            ("closed_pnl", "closedPnl", "realized_pnl", "realizedPnl", "realisedPnl", "pnl"),
        )
        realized_pnl_known = pnl_value is not None or not order.reduce_only
        pnl_source = "exchange" if pnl_value is not None else "opening_fill" if not order.reduce_only else "unknown"
        if pnl_value is None:
            pnl_value = 0.0

        funding_fee = self._first_payload_optional_float(
            payloads,
            ("funding_fee", "fundingFee", "funding_payment", "fundingPayment"),
        )
        funding_fee_known = funding_fee is not None
        if funding_fee is None:
            funding_fee = 0.0

        exchange_order_id = str(
            self._first_payload_value(payloads, ("exchange_order_id", "order_id", "orderId", "oid"))
            or order.exchange_order_id
            or ""
        ).strip() or None
        exchange_fill_id = str(
            self._first_payload_value(payloads, ("exchange_fill_id", "fill_id", "fillId", "trade_id", "tradeId", "tid", "hash", "id"))
            or ""
        ).strip() or None
        source_order_id = self._resolve_source_order_id(order)

        status = "reconciled"
        blockers: list[str] = []
        if not fee_known:
            status = "estimated_fee" if status == "reconciled" else status
            blockers.append("fee_estimated")
        if order.reduce_only and not realized_pnl_known:
            status = "unknown_realized_pnl"
            blockers.append("realized_pnl_unknown")
        if not funding_fee_known:
            blockers.append("funding_fee_not_reported")

        metadata = {
            "reconciliation_status": status,
            "fee_source": fee_source,
            "realized_pnl_source": pnl_source,
            "funding_fee_source": "exchange" if funding_fee_known else "not_reported",
            "reconciliation_blockers": blockers,
            "notional": abs(quantity * price),
            "exchange_order_id": exchange_order_id,
            "exchange_fill_id": exchange_fill_id,
        }
        if matched_fill:
            metadata["matched_recent_fill"] = matched_fill

        fill = Fill(
            order_id=order.id,
            source_order_id=source_order_id,
            exchange_order_id=exchange_order_id,
            exchange_fill_id=exchange_fill_id,
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            price=price,
            fee=max(self._safe_float(fee_value), 0.0),
            pnl=self._safe_float(pnl_value),
            funding_fee=max(self._safe_float(funding_fee), 0.0),
            fee_known=fee_known,
            realized_pnl_known=realized_pnl_known,
            simulated=False,
        )
        fill.details = metadata
        return fill

    def _matched_recent_fill(self, order: Order, response: dict[str, Any]) -> dict[str, Any] | None:
        if not bool(self.config.get("LIVE_FILL_RECONCILE_RECENT_FILLS_ENABLED", True)):
            return None
        connector = self._connector_for_order(order)
        getter = getattr(connector, "get_recent_fills", None)
        if not callable(getter):
            return None
        try:
            fills = getter(order.mode)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(fills, list):
            return None
        exchange_order_id = str(response.get("exchange_order_id") or order.exchange_order_id or "").strip()
        target_symbol = str(order.symbol or "").upper()
        target_side = str(order.side or "").lower()
        target_price = self._safe_float(order.average_fill_price)
        target_quantity = self._safe_float(order.filled_quantity or order.quantity)
        fallback: dict[str, Any] | None = None
        for fill in fills:
            if not isinstance(fill, dict):
                continue
            fill_symbol = str(fill.get("symbol") or "").upper()
            if fill_symbol and fill_symbol != target_symbol:
                continue
            fill_side = str(fill.get("side") or "").lower()
            if fill_side and fill_side != target_side:
                continue
            fill_order_id = str(
                fill.get("exchange_order_id")
                or fill.get("order_id")
                or (fill.get("raw") or {}).get("orderId")
                or (fill.get("raw") or {}).get("oid")
                or ""
            ).strip()
            if exchange_order_id and fill_order_id == exchange_order_id:
                return fill
            fill_price = self._safe_float(fill.get("price"))
            fill_quantity = self._safe_float(fill.get("size", fill.get("quantity")))
            price_matches = target_price <= 0 or fill_price <= 0 or abs(fill_price - target_price) / max(target_price, 1e-9) <= 0.0025
            quantity_matches = target_quantity <= 0 or fill_quantity <= 0 or abs(fill_quantity - target_quantity) <= max(target_quantity * 0.01, 1e-9)
            if price_matches and quantity_matches:
                fallback = fill
        return fallback

    def _resolve_source_order_id(self, order: Order) -> int | None:
        details = dict(order.details or {})
        for key in ("source_order_id", "rapid_ml_exit_source_order_id", "protective_exit_source_order_id"):
            raw_value = details.get(key)
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                continue
            if value > 0 and value != order.id:
                return value
        if not order.reduce_only:
            return None
        query = Order.query.filter(
            Order.id != order.id,
            Order.symbol == order.symbol,
            Order.mode == order.mode,
            Order.reduce_only.is_(False),
            Order.status.in_(["filled", "open", "submitted"]),
        )
        if order.user_id is not None:
            query = query.filter(Order.user_id == int(order.user_id))
        if order.trading_connection_id is not None:
            query = query.filter(Order.trading_connection_id == int(order.trading_connection_id))
        source = query.order_by(Order.created_at.desc()).first()
        return int(source.id) if source is not None else None

    def _estimated_trade_fee(self, intent: OrderIntent, quantity: float, price: float) -> float:
        notional = abs(self._safe_float(quantity) * self._safe_float(price))
        if notional <= 0:
            return 0.0
        provider = str((intent.metadata or {}).get("provider") or (intent.metadata or {}).get("execution_venue") or "").strip().lower()
        fee_bps = self._provider_fee_bps(provider)
        return notional * max(fee_bps, 0.0) / 10_000.0

    def _provider_fee_bps(self, provider: str) -> float:
        raw = self.config.get("RAPID_ML_FEE_BPS_BY_PROVIDER_JSON")
        payload: dict[str, Any] = {}
        if isinstance(raw, dict):
            payload = raw
        elif isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    payload = parsed
            except json.JSONDecodeError:
                payload = {}
        provider_key = str(provider or "").lower()
        if provider_key and provider_key in payload:
            return self._safe_float(payload.get(provider_key), self._safe_float(self.config.get("FEE_BPS"), 5.0))
        return self._safe_float(self.config.get("FEE_BPS"), 5.0)

    @classmethod
    def _first_payload_float(cls, payloads: list[Any], keys: tuple[str, ...], default: float = 0.0) -> float:
        value = cls._first_payload_optional_float(payloads, keys)
        return default if value is None else value

    @classmethod
    def _first_payload_optional_float(cls, payloads: list[Any], keys: tuple[str, ...]) -> float | None:
        value = cls._first_payload_value(payloads, keys)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _first_payload_value(cls, payloads: list[Any], keys: tuple[str, ...]) -> Any:
        for payload in payloads:
            value = cls._find_payload_value(payload, keys)
            if cls._payload_value_present(value):
                return value
        return None

    @classmethod
    def _find_payload_value(cls, payload: Any, keys: tuple[str, ...]) -> Any:
        if isinstance(payload, dict):
            for key in keys:
                if key in payload:
                    return payload[key]
            for value in payload.values():
                found = cls._find_payload_value(value, keys)
                if cls._payload_value_present(found):
                    return found
        elif isinstance(payload, list):
            for item in payload:
                found = cls._find_payload_value(item, keys)
                if cls._payload_value_present(found):
                    return found
        return None

    @staticmethod
    def _payload_value_present(value: Any) -> bool:
        return value is not None and not (isinstance(value, str) and not value.strip())

    @staticmethod
    def _submission_slippage_pct(intent_slippage: float, max_slippage: float) -> float:
        requested = max(0.0, OrderManager._safe_float(intent_slippage, 0.0))
        configured_max = max(0.0, OrderManager._safe_float(max_slippage, 0.0))

        if requested <= 0:
            return configured_max / 2 if configured_max > 0 else 0.0

        if configured_max <= 0:
            return requested

        return min(requested, configured_max)

    def _handle_exchange_failure(self, order: Order, intent: OrderIntent, exc: Exception) -> None:
        order.status = "failed"
        order.rejection_reason = str(exc)
        details = dict(order.details or {})
        details["exchange_error"] = str(exc)
        details["blocker_category"] = self._blocker_category("connector_error", str(exc), details)
        order.details = details

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

    def _apply_ml_execution_policy(self, intent: OrderIntent, market_price: float) -> OrderIntent:
        if not (
            bool(self.config.get("ML_ALL_AREAS_ENABLED", False))
            and bool(self.config.get("ML_ORDER_POLICY_ENABLED", False))
            and bool(self.config.get("ML_ALLOW_EXECUTION_STYLE_SUGGESTIONS", True))
        ):
            return intent
        metadata = dict(intent.metadata or {})
        horizon = self._ml_horizon(metadata)
        context = {
            **metadata,
            "symbol": intent.symbol,
            "side": intent.side,
            "horizon": horizon,
            "notional": max(0.0, self._safe_float(market_price) * self._safe_float(intent.quantity)),
            "reference_price": market_price,
            "order_type": intent.order_type,
            "slippage_pct": intent.slippage_pct,
            "hard_max_leverage": self.config.get("MAX_LEVERAGE", 1.0),
        }
        try:
            book = self._safe_order_book(
                intent.symbol,
                intent.mode,
                user_id=intent.user_id,
                trading_connection_id=intent.trading_connection_id,
                provider=metadata.get("provider"),
                venue_symbol=metadata.get("venue_symbol"),
            )
            if isinstance(book, dict):
                context["spread_bps"] = book.get("spread_bps", metadata.get("spread_bps", 0.0))
                context["expected_fill_quality"] = book.get("expected_fill_quality", metadata.get("expected_fill_quality", 0.0))
            decision = MLDecisionEngine(self.config, signal_model=MLSignalModel(self.config)).decision(
                "pytorch_execution_policy",
                context,
                horizon=horizon,
            )
        except Exception as exc:  # noqa: BLE001
            decision = {
                "ready": False,
                "family": "pytorch_execution_policy",
                "blockers": [str(exc)],
                "raw": {},
            }
        metadata["ml_execution_policy"] = decision
        raw = decision.get("raw") if isinstance(decision.get("raw"), dict) else {}
        if bool(decision.get("ready", False)) and self._safe_float(market_price) > 0:
            slippage = self._safe_float(raw.get("slippage_tolerance_pct"), intent.slippage_pct)
            adaptive = self.risk_engine.adaptive_slippage_metrics({**metadata, "spread_bps": context.get("spread_bps", 0.0)}, slippage_pct=slippage)
            max_slippage = self._safe_float(adaptive.get("max_acceptable_pct"), slippage)
            if max_slippage > 0:
                intent.slippage_pct = max(0.0, min(slippage, max_slippage))
                metadata["adaptive_slippage"] = adaptive
            if str(raw.get("order_type_suggestion") or "").lower() == "limit":
                offset_bps = max(0.0, self._safe_float(raw.get("limit_offset_bps"), 0.0))
                offset = market_price * offset_bps / 10_000.0
                limit_price = market_price - offset if intent.side == "buy" else market_price + offset
                if limit_price > 0:
                    intent.order_type = "limit"
                    intent.limit_price = limit_price
        intent.metadata = metadata
        return intent

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

    @staticmethod
    def _ml_horizon(metadata: dict[str, Any]) -> str:
        payload = metadata or {}
        markers = {
            str(payload.get("algorithm_profile") or "").strip().lower(),
            str(payload.get("vault_cycle_name") or "").strip().lower(),
            str(payload.get("ml_horizon") or "").strip().lower(),
            str(payload.get("objective") or "").strip().lower(),
        }
        if bool(payload.get("one_h10_vault")) or bool(markers & {"1h10", "one_h10", "one_hour_10x"}):
            return "1h10"
        return str(payload.get("duration_bucket") or payload.get("horizon") or payload.get("ml_horizon") or "1h").lower()

    @staticmethod
    def _blocker_category(rule_name: object, reason: object, details: object | None = None) -> str:
        text = " ".join(
            str(part or "")
            for part in (
                rule_name,
                reason,
                details if isinstance(details, str) else "",
            )
        ).lower()
        if "could not immediately match" in text or "immediately match against" in text:
            return "execution_not_immediately_matchable"
        if "429" in text or "rate limit" in text or "too many request" in text:
            return "rate_limited"
        if "leverage" in text:
            return "leverage_cap"
        if "dynamic_cap" in text or "notional" in text or "hard_cap" in text:
            return "dynamic_cap_breach"
        if "liquidity" in text:
            return "liquidity_too_low"
        if "slippage" in text or "spread" in text:
            return "slippage_too_high"
        if "min_notional" in text or "min size" in text or "minimum order" in text:
            return "min_notional"
        if "stop loss" in text or "take profit" in text or "missing_exit" in text:
            return "missing_stop_take_profit"
        if "ml" in text and ("not_ready" in text or "promoted" in text or "torch" in text):
            return "ml_not_ready"
        if "ml_signal_hold" in text or "hold" in text or "low_confidence" in text:
            return "ml_hold"
        if "exchange" in text and "reject" in text:
            return "exchange_rejected"
        if "connector" in text or "api" in text or "timeout" in text or "network" in text or "failed" in text:
            return "connector_error"
        return "risk_rejected"

    @staticmethod
    def _immediate_match_rejection(reason: object, response: object | None = None) -> bool:
        text = f"{reason or ''} {response or ''}".lower()
        return "could not immediately match" in text or "immediately match against" in text

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

    def _safe_market_price(
        self,
        symbol: str,
        mode: str,
        *,
        user_id: int | None = None,
        trading_connection_id: int | None = None,
        provider: str | None = None,
        venue_symbol: str | None = None,
    ) -> float:
        provider_key = str(provider or self._provider_for_connection(user_id, trading_connection_id) or "").strip().lower()
        market_symbol = str(venue_symbol or symbol).strip()
        if provider_key != "hyperliquid":
            market_symbol = market_symbol.upper()
        if provider_key and provider_key not in {"hyperliquid", "global"}:
            connector = self._connector_for_user(user_id, trading_connection_id)
            getter = getattr(connector, "get_mid_price", None) if connector is not None else None
            if callable(getter):
                try:
                    return float(getter(market_symbol, mode))
                except Exception:  # noqa: BLE001
                    return 0.0
            return 0.0
        try:
            return self.market_data.get_mid_price(market_symbol, mode)
        except Exception:  # noqa: BLE001
            return 0.0

    def _safe_order_book(
        self,
        symbol: str,
        mode: str,
        *,
        user_id: int | None = None,
        trading_connection_id: int | None = None,
        provider: str | None = None,
        venue_symbol: str | None = None,
    ) -> dict[str, Any]:
        provider_key = str(provider or self._provider_for_connection(user_id, trading_connection_id) or "").strip().lower()
        market_symbol = str(venue_symbol or symbol).strip()
        if provider_key != "hyperliquid":
            market_symbol = market_symbol.upper()
        if provider_key and provider_key not in {"hyperliquid", "global"}:
            connector = self._connector_for_user(user_id, trading_connection_id)
            getter = getattr(connector, "get_order_book", None) if connector is not None else None
            if callable(getter):
                try:
                    return dict(getter(market_symbol, mode) or {})
                except Exception:  # noqa: BLE001
                    return {}
            return {}
        try:
            return self.market_data.get_order_book(market_symbol, mode)
        except Exception:  # noqa: BLE001
            return {}

    def _connector_for_user(self, user_id: int | None, trading_connection_id: int | None):
        if user_id is None or self.trading_connections is None:
            return None
        try:
            return self.trading_connections.connector_for_user(user_id, trading_connection_id)
        except Exception:
            return None

    def _provider_for_connection(self, user_id: int | None, trading_connection_id: int | None) -> str:
        if user_id is None or trading_connection_id is None or self.trading_connections is None:
            return ""
        try:
            connection = self.trading_connections.get_for_user(user_id, trading_connection_id)
            return str(getattr(connection, "provider", "") or "")
        except Exception:
            return ""

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
