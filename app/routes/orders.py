"""Manual order entry and order history routes."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, flash, redirect, render_template, request, url_for

from ..auth import current_user
from ..admin_auth import require_admin
from ..models import Order
from ..runtime import get_current_mode, get_service
from ..services.order_manager import OrderIntent


orders_bp = Blueprint("orders", __name__, url_prefix="/admin/orders")


@orders_bp.before_request
def _protect_orders():
    return require_admin()


@orders_bp.get("/", strict_slashes=False)
def index():
    mode = get_current_mode()
    orders = Order.query.order_by(Order.created_at.desc()).limit(50).all()
    return render_template("orders.html", mode=mode, orders=orders)


@orders_bp.post("/place")
def place():
    order_manager = get_service("order_manager")
    mode = get_current_mode()
    user = current_user()
    connection = get_service("trading_connections").active_tradable_connection(user.id) if user is not None else None

    symbol = str(request.form.get("symbol", "BTC")).upper().strip()
    side = str(request.form.get("side", "buy")).strip().lower()
    order_type = str(request.form.get("order_type", "market")).strip().lower()

    quantity = _safe_float(request.form.get("quantity"), 0.0)
    limit_price = _optional_float(request.form.get("limit_price"))
    stop_loss = _optional_float(request.form.get("stop_loss"))
    take_profit = _optional_float(request.form.get("take_profit"))
    leverage = _safe_float(request.form.get("leverage"), 1.0)
    slippage_pct = _safe_float(request.form.get("slippage_pct"), 0.005)

    if not symbol:
        flash("Symbol is required.", "danger")
        return redirect(url_for("orders.index"))

    if side not in {"buy", "sell"}:
        flash("Side must be buy or sell.", "danger")
        return redirect(url_for("orders.index"))

    if order_type not in {"market", "limit"}:
        flash("Order type must be market or limit.", "danger")
        return redirect(url_for("orders.index"))

    if quantity <= 0:
        flash("Quantity must be greater than zero.", "danger")
        return redirect(url_for("orders.index"))

    if leverage <= 0:
        flash("Leverage must be greater than zero.", "danger")
        return redirect(url_for("orders.index"))

    if slippage_pct < 0:
        flash("Slippage cannot be negative.", "danger")
        return redirect(url_for("orders.index"))

    if order_type == "limit" and limit_price is None:
        flash("Limit price is required for limit orders.", "danger")
        return redirect(url_for("orders.index"))

    if mode == "live" and connection is None:
        flash("Connect, verify, and activate a live-ready trading account before submitting live orders.", "danger")
        return redirect(url_for("settings.connections"))

    try:
        order = order_manager.place_order(
            OrderIntent(
                symbol=symbol,
                side=side,
                quantity=quantity,
                mode=mode,
                order_type=order_type,
                limit_price=limit_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=leverage,
                slippage_pct=slippage_pct,
                user_id=user.id if user is not None else None,
                trading_connection_id=connection.id if connection is not None else None,
                metadata={"manual": True},
            )
        )
    except Exception as exc:  # noqa: BLE001
        flash(f"Order failed: {exc}", "danger")
        return redirect(url_for("orders.index"))

    if order.status == "rejected":
        flash(order.rejection_reason or "Order rejected.", "danger")
    else:
        flash(f"Order {order.client_order_id} submitted with status {order.status}.", "success")

    return redirect(url_for("orders.index"))


@orders_bp.post("/<int:order_id>/cancel")
def cancel(order_id: int):
    order_manager = get_service("order_manager")

    try:
        order = order_manager.cancel_order(order_id)
    except Exception as exc:  # noqa: BLE001
        flash(f"Cancel failed: {exc}", "danger")
        return redirect(url_for("orders.index"))

    flash(f"Order {order.client_order_id} status is now {order.status}.", "warning")
    return redirect(url_for("orders.index"))


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None

    return _safe_float(value, 0.0)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
