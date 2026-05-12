"""Settings and mode management routes."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, flash, redirect, render_template, request, url_for

from ..auth import current_user, require_authenticated_user
from ..extensions import db
from ..models import AuditLog, Setting, TradingConnection
from ..runtime import get_current_mode, get_service
from ..services.connection_health import build_connection_health, latest_connection_health, store_connection_health


settings_bp = Blueprint("settings", __name__, url_prefix="/settings")


@settings_bp.before_request
def _protect_settings():
    return require_authenticated_user()


@settings_bp.get("/")
def index():
    service = get_service("trading_connections")
    user = current_user()
    connections = TradingConnection.query.filter_by(user_id=user.id).order_by(TradingConnection.updated_at.desc()).all() if user else []
    provider_specs = service.provider_specs()
    connection_health_by_id = {record.id: latest_connection_health(record.id) for record in connections}
    enabled_connections = service.enabled_tradable_connections(user.id) if user else []

    return render_template(
        "settings.html",
        current_mode=get_current_mode(),
        panic_lock=Setting.get_json("panic_lock", False),
        explicit_live_confirmed=Setting.get_json("explicit_live_confirmed", False),
        secondary_confirmation=Setting.get_json("secondary_confirmation", False),
        live_trading_blocked=Setting.get_json("live_trading_blocked", False),
        trading_connections=connections,
        enabled_trading_connections=enabled_connections,
        enabled_provider_count=len(enabled_connections),
        providers=provider_specs,
        provider_cards=_provider_cards(connections, provider_specs, connection_health_by_id),
        connection_health_by_id=connection_health_by_id,
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
    enabled_connections = service.enabled_tradable_connections(user.id) if user else []
    return render_template(
        "connections.html",
        trading_connections=records,
        enabled_trading_connections=enabled_connections,
        enabled_provider_count=len(enabled_connections),
        providers=provider_specs,
        provider_cards=_provider_cards(records, provider_specs, connection_health_by_id),
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
    flash("Connection saved. Verify it before enabling live flows.", "success")
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
    if connection.is_active or connection.verification_status == "verified":
        snapshot = result.get("snapshot")
        alerts = list(getattr(snapshot, "alerts", []) or []) if snapshot is not None else []
        error = str(result.get("error") or "")
        health = build_connection_health(
            connection,
            can_trade=bool(result["ok"]) and not alerts,
            alerts=alerts or ([error] if error else []),
            failure_reason=error or "; ".join(alerts),
        )
        store_connection_health(connection, health)
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
        flash("Connection verified. Enable it to use live wallet, vault, and order workflows.", "success")
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
        f"{connection.provider.title()} trading connection enabled.",
        {"user_id": user.id, "trading_connection_id": connection.id, "provider": connection.provider},
    )
    db.session.commit()
    flash("Trading provider enabled.", "success")
    return redirect(url_for("settings.connection_provider", provider=connection.provider, connection_id=connection.id))


@settings_bp.post("/connections/<int:connection_id>/disable")
def disable_connection(connection_id: int):
    user = current_user()
    if user is None:
        return redirect(url_for("auth.login"))
    service = get_service("trading_connections")
    try:
        connection = service.disable(user.id, connection_id)
    except PermissionError:
        flash("Trading connection was not found.", "danger")
        return redirect(url_for("settings.connections"))
    _audit(
        "trading_connection_disabled",
        f"{connection.provider.title()} trading connection disabled.",
        {"user_id": user.id, "trading_connection_id": connection.id, "provider": connection.provider},
    )
    db.session.commit()
    flash("Trading provider disabled.", "warning")
    return redirect(url_for("settings.connection_provider", provider=connection.provider, connection_id=connection.id))


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


def _provider_cards(
    connections: list[TradingConnection],
    provider_specs: dict[str, dict[str, Any]],
    connection_health_by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    records_by_provider = {record.provider: record for record in reversed(connections)}
    cards: list[dict[str, Any]] = []
    for key, spec in provider_specs.items():
        connection = records_by_provider.get(key)
        status = str(getattr(connection, "verification_status", "") or "not_connected")
        enabled = bool(connection and connection.is_active and status == "verified" and spec.get("tradable"))
        health = connection_health_by_id.get(connection.id, {}) if connection and connection.id is not None else {}
        can_trade = bool(health.get("can_trade")) if health else False
        if health:
            health_label = "Online" if can_trade else "Blocked"
            health_class = "positive" if can_trade else "danger"
        else:
            health_label = "Not checked" if connection else "Not connected"
            health_class = "muted"
        cards.append(
            {
                "key": key,
                "label": spec.get("label", key.title()),
                "summary": spec.get("summary", ""),
                "tradable": bool(spec.get("tradable")),
                "verification_supported": bool(spec.get("verification_supported")),
                "connection": connection,
                "enabled": enabled,
                "status": status,
                "status_label": status.replace("_", " ").title(),
                "health": health,
                "health_label": health_label,
                "health_class": health_class,
                "wallet_preview": _connection_preview(connection),
                "can_enable": bool(connection and spec.get("tradable") and status == "verified" and not enabled),
                "can_disable": enabled,
            }
        )
    return cards


def _connection_preview(connection: TradingConnection | None) -> str:
    if connection is None:
        return "Not saved"
    if connection.wallet_address:
        value = str(connection.wallet_address)
        return f"{value[:10]}...{value[-6:]}" if len(value) > 20 else value
    if connection.encrypted_api_key:
        return "API key saved"
    if connection.encrypted_api_secret:
        return "Secret saved"
    return "Saved"


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
