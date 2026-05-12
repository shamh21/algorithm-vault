"""Panic button routes."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, flash, redirect, render_template, url_for

from ..auth import current_user
from ..admin_auth import require_admin
from ..extensions import db
from ..models import AuditLog, StrategyRun, Setting
from ..runtime import get_current_mode, get_service


panic_bp = Blueprint("panic", __name__, url_prefix="/admin/panic")


@panic_bp.before_request
def _protect_panic():
    return require_admin()


@panic_bp.get("/", strict_slashes=False)
def index():
    audits = (
        AuditLog.query.filter_by(category="panic")
        .order_by(AuditLog.created_at.desc())
        .limit(20)
        .all()
    )

    return render_template(
        "panic.html",
        audits=audits,
        mode=get_current_mode(),
        panic_lock=Setting.get_json("panic_lock", False),
    )


@panic_bp.post("/activate")
def activate():
    mode = get_current_mode()
    user = current_user()
    connection = get_service("trading_connections").active_tradable_connection(user.id) if user is not None else None
    cancelled: list[dict[str, Any]] = []
    flattened: list[dict[str, Any]] = []
    errors: list[str] = []

    Setting.set_json("panic_lock", True)

    try:
        get_service("strategy_manager").stop_all()
        StrategyRun.query.update(
            {"status": "stopped", "manual_enabled": False},
            synchronize_session=False,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Strategy stop failed: {exc}")

    try:
        cancelled = get_service("order_manager").cancel_all(
            mode,
            user.id if user is not None else None,
            connection.id if connection is not None else None,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Order cancellation failed: {exc}")

    try:
        flattened = get_service("order_manager").flatten_positions(
            mode,
            user.id if user is not None else None,
            connection.id if connection is not None else None,
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Position flattening failed: {exc}")

    audit = AuditLog(
        category="panic",
        action="activate",
        message="Panic button activated: trading locked, strategies disabled, exits attempted.",
        user_id=user.id if user is not None else None,
        trading_connection_id=connection.id if connection is not None else None,
    )
    audit.details = {
        "mode": mode,
        "cancelled": cancelled,
        "flattened": flattened,
        "errors": errors,
    }
    db.session.add(audit)
    db.session.commit()

    if errors:
        flash("Panic lock activated, but some cleanup actions failed. Review the panic audit log.", "warning")
    else:
        flash("Panic button activated. Trading is locked until manually reset.", "danger")

    return redirect(url_for("dashboard.index"))
