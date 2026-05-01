"""Settings and mode management routes."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..auth import current_user, require_admin_user, require_authenticated_user
from ..extensions import db
from ..models import AuditLog, Setting, TradingConnection, WalletBalance
from ..runtime import available_modes, get_current_mode, get_service
from ..services.connection_health import latest_connection_health


settings_bp = Blueprint("settings", __name__, url_prefix="/settings")


@settings_bp.before_request
def _protect_settings():
    return require_authenticated_user()


@settings_bp.get("/")
def index():
    feature_engine = get_service("feature_engine")
    trading_connections = get_service("trading_connections")
    user = current_user()
    connections = TradingConnection.query.filter_by(user_id=user.id).order_by(TradingConnection.updated_at.desc()).all() if user else []

    return render_template(
        "settings.html",
        current_mode=get_current_mode(),
        modes=available_modes(),
        panic_lock=Setting.get_json("panic_lock", False),
        live_enabled=bool(current_app.config.get("ENABLE_LIVE_TRADING", False)),
        explicit_live_confirmed=Setting.get_json("explicit_live_confirmed", False),
        secondary_confirmation=Setting.get_json("secondary_confirmation", False),
        live_trading_blocked=Setting.get_json("live_trading_blocked", False),
        use_real_addresses=Setting.get_json("use_real_addresses", bool(current_app.config.get("USE_REAL_ADDRESSES", False))),
        external_adapter_status=feature_engine.external_status,
        pattern_model_status=feature_engine.pattern_status,
        trading_connections=connections,
        active_trading_connection=trading_connections.active_tradable_connection(user.id) if user else None,
    )


@settings_bp.get("/connections")
def connections():
    return _render_connections()


@settings_bp.get("/connections/<provider>")
def connection_provider(provider: str):
    return _render_connections(provider)


def _render_connections(provider: str | None = None):
    user = current_user()
    service = get_service("trading_connections")
    records = TradingConnection.query.filter_by(user_id=user.id).order_by(TradingConnection.updated_at.desc()).all() if user else []
    provider_specs = service.provider_specs()
    selected_provider = provider or request.args.get("provider") or "hyperliquid"
    if selected_provider not in provider_specs:
        selected_provider = "hyperliquid"
    selected_connection_id = _safe_int(request.args.get("connection_id"), 0)
    selected_connection = None
    if selected_connection_id:
        selected_connection = next((record for record in records if record.id == selected_connection_id), None)
    if selected_connection is None:
        selected_connection = next((record for record in records if record.provider == selected_provider), None)
    connection_health_by_id = {record.id: latest_connection_health(record.id) for record in records}
    return render_template(
        "connections.html",
        trading_connections=records,
        active_trading_connection=service.active_tradable_connection(user.id) if user else None,
        providers=provider_specs,
        selected_provider=selected_provider,
        selected_spec=provider_specs[selected_provider],
        selected_connection=selected_connection,
        selected_connection_health=connection_health_by_id.get(selected_connection.id, {}) if selected_connection else {},
        connection_health_by_id=connection_health_by_id,
    )


@settings_bp.post("/connections")
def save_connection():
    user = current_user()
    if user is None:
        return redirect(url_for("auth.login"))

    service = get_service("trading_connections")
    provider = str(request.form.get("provider", "hyperliquid")).strip().lower()
    spec = service.provider_spec(provider)
    connection_type = str(request.form.get("connection_type", spec["connection_type"])).strip().lower()
    api_key = str(request.form.get("api_key", "")).strip()
    api_secret = str(request.form.get("api_secret", "")).strip()
    passphrase = str(request.form.get("passphrase", "")).strip()
    wallet_address = str(request.form.get("wallet_address", "")).strip()
    metadata = {
        field["name"]: str(request.form.get(field["name"], "")).strip()
        for field in spec.get("fields", [])
        if field.get("storage") == "metadata"
    }

    try:
        connection = service.create_or_update(
            user_id=user.id,
            provider=provider,
            connection_type=connection_type,
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            wallet_address=wallet_address,
            metadata=metadata,
            is_active=False,
        )
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("settings.connection_provider", provider=provider))

    _audit(
        "trading_connection_saved",
        f"{provider.title()} trading connection saved.",
        {"user_id": user.id, "trading_connection_id": connection.id, "provider": provider},
    )
    db.session.commit()
    flash("Connection saved. Run verification before activating it for live trading.", "success")
    return redirect(url_for("settings.connection_provider", provider=provider, connection_id=connection.id))


@settings_bp.post("/connections/<int:connection_id>/verify")
def verify_connection(connection_id: int):
    user = current_user()
    if user is None:
        return redirect(url_for("auth.login"))
    service = get_service("trading_connections")
    try:
        result = service.verify_connection(user.id, connection_id)
    except PermissionError:
        flash("Trading connection was not found.", "danger")
        return redirect(url_for("settings.connections"))
    connection = result["connection"]
    _audit(
        "trading_connection_verified" if result["ok"] else "trading_connection_verification_failed",
        f"{connection.provider.title()} connection verification {'passed' if result['ok'] else 'failed'}.",
        {
            "user_id": user.id,
            "trading_connection_id": connection.id,
            "provider": connection.provider,
            "verification_status": connection.verification_status,
            "error": result.get("error"),
        },
    )
    db.session.commit()
    if result["ok"]:
        flash("Connection verified. Activate it to use live wallet, vault, and order workflows.", "success")
    else:
        flash(result.get("error") or "Connection verification failed.", "danger")
    return redirect(url_for("settings.connection_provider", provider=connection.provider, connection_id=connection.id))


@settings_bp.post("/connections/<int:connection_id>/activate")
def activate_connection(connection_id: int):
    user = current_user()
    if user is None:
        return redirect(url_for("auth.login"))
    service = get_service("trading_connections")
    try:
        connection = service.activate_verified(user.id, connection_id)
    except PermissionError:
        flash("Trading connection was not found.", "danger")
        return redirect(url_for("settings.connections"))
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("settings.connections"))
    _audit(
        "trading_connection_activated",
        f"{connection.provider.title()} trading connection activated.",
        {"user_id": user.id, "trading_connection_id": connection.id, "provider": connection.provider},
    )
    db.session.commit()
    flash("Trading connection activated.", "success")
    return redirect(url_for("consumer.home"))


@settings_bp.post("/connections/<int:connection_id>/delete")
def delete_connection(connection_id: int):
    user = current_user()
    if user is None:
        return redirect(url_for("auth.login"))
    try:
        get_service("trading_connections").delete(user_id=user.id, connection_id=connection_id)
    except PermissionError:
        flash("Trading connection was not found.", "danger")
        return redirect(url_for("settings.connections"))
    db.session.commit()
    flash("Trading connection deleted.", "warning")
    return redirect(url_for("settings.connections"))


@settings_bp.post("/mode")
def set_mode():
    Setting.set_json("current_mode", "live")
    _audit(
        "mode_change",
        "Confirmed live-only operating mode.",
        {"mode": "live"},
    )

    db.session.commit()
    flash("Live-only mode is active.", "success")
    return redirect(url_for("settings.index"))


@settings_bp.post("/address-mode")
def address_mode():
    guard = require_admin_user()
    if guard is not None:
        return guard

    enabled = request.form.get("use_real_addresses") == "on"
    Setting.set_json("use_real_addresses", enabled)
    _audit(
        "address_mode_change",
        f"Address mode changed to {'real-provider' if enabled else 'sandbox'} mode.",
        {"use_real_addresses": enabled},
    )
    db.session.commit()
    flash("Address mode updated.", "success")
    return redirect(url_for("settings.index"))


@settings_bp.post("/live-confirmations")
def live_confirmations():
    explicit = request.form.get("explicit_live_confirmed") == "on"
    secondary = request.form.get("secondary_confirmation") == "on"

    Setting.set_json("explicit_live_confirmed", explicit)
    Setting.set_json("secondary_confirmation", secondary)

    _audit(
        "live_confirmations",
        "Live trading confirmation flags updated.",
        {
            "explicit_live_confirmed": explicit,
            "secondary_confirmation": secondary,
        },
    )

    db.session.commit()
    flash("Live confirmation flags updated. Live orders are now controlled by per-user connections and hard risk limits.", "warning")
    return redirect(url_for("settings.index"))


@settings_bp.post("/live-block-reset")
def live_block_reset():
    Setting.set_json("live_trading_blocked", False)

    _audit(
        "live_block_reset",
        "Live trading failure block reset manually.",
        {"live_trading_blocked": False},
    )

    db.session.commit()
    flash("Live failure block reset. Live orders remain gated by per-user connections and hard risk limits.", "warning")
    return redirect(url_for("settings.index"))


@settings_bp.post("/panic-reset")
def panic_reset():
    Setting.set_json("panic_lock", False)

    _audit(
        "panic_reset",
        "Trading lock reset manually.",
        {"panic_lock": False},
    )

    db.session.commit()
    flash("Panic lock cleared. Trading remains manual and strategies must be restarted explicitly.", "warning")
    return redirect(url_for("settings.index"))


def _audit(action: str, message: str, details: dict[str, Any]) -> None:
    audit = AuditLog(category="settings", action=action, message=message)
    audit.details = details
    db.session.add(audit)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
