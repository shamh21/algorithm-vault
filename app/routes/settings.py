"""Settings and mode management routes."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests
from flask import Blueprint, Response, current_app, flash, jsonify, redirect, render_template, request, url_for

from ..auth import current_user, require_authenticated_user
from ..extensions import db
from ..live_api_internal import is_live_api_internal_request, sign_live_api_internal_headers
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
        credential_saved_count=sum(1 for connection in connections if _has_saved_credential(connection)),
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
        existing_connection = service.get_for_user(user.id, connection_id)
    except PermissionError:
        if _internal_json_requested():
            return jsonify({"ok": False, "code": "connection_not_found", "message": "Trading connection was not found."}), 404
        flash("Trading connection was not found.", "danger")
        return redirect(url_for("settings.connections"))
    if existing_connection.provider == "kucoin" and _settings_verify_deferred_for_request():
        if _settings_live_api_proxy_enabled():
            return _proxy_kucoin_connection_verify(user.id, existing_connection)
        if _internal_json_requested():
            return (
                jsonify(
                    {
                        "ok": False,
                        "code": "live_api_proxy_required",
                        "message": _kucoin_fixed_egress_required_message(),
                        "connection": _connection_payload(existing_connection),
                    }
                ),
                409,
            )
        flash(_kucoin_fixed_egress_required_message(), "danger")
        return redirect(
            url_for("settings.connection_provider", provider=existing_connection.provider, connection_id=existing_connection.id)
        )
    try:
        result = service.verify_connection(user.id, connection_id)
    except PermissionError:
        if _internal_json_requested():
            return jsonify({"ok": False, "code": "connection_not_found", "message": "Trading connection was not found."}), 404
        flash("Trading connection was not found.", "danger")
        return redirect(url_for("settings.connections"))
    return _finish_verify_connection(result, internal_json=_internal_json_requested())


def _finish_verify_connection(result: dict[str, Any], *, internal_json: bool = False):
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
            "user_id": connection.user_id,
            "trading_connection_id": connection.id,
            "provider": connection.provider,
            "verification_status": connection.verification_status,
            "error": result.get("error"),
        },
    )
    db.session.commit()
    if internal_json:
        return jsonify(_verify_result_payload(result))
    if result["ok"]:
        flash("Connection verified. Enable it to use live wallet, vault, and order workflows.", "success")
    else:
        flash(result.get("error") or "Connection verification failed.", "danger")
    return redirect(url_for("settings.connection_provider", provider=connection.provider, connection_id=connection.id))


def _proxy_kucoin_connection_verify(user_id: int, connection: TradingConnection) -> Response:
    origin = _settings_live_api_internal_origin()
    if not origin:
        flash(_kucoin_fixed_egress_required_message(), "danger")
        return redirect(url_for("settings.connection_provider", provider=connection.provider, connection_id=connection.id))
    path = f"/settings/connections/{int(connection.id)}/verify"
    query_string = b"internal_json=1"
    body = b""
    headers = {
        "Accept": "application/json",
        "X-AlgVault-Forwarded-Origin": _request_origin(),
        **sign_live_api_internal_headers(
            current_app.config,
            method="POST",
            path=path,
            query_string=query_string,
            body=body,
            user_id=int(user_id),
        ),
    }
    try:
        upstream = requests.post(
            f"{origin}{path}?internal_json=1",
            data=body,
            headers=headers,
            timeout=float(current_app.config.get("LIVE_API_PROXY_TIMEOUT_SECONDS", 15.0) or 15.0),
        )
        payload = upstream.json()
    except (requests.RequestException, ValueError) as exc:
        current_app.logger.warning("KuCoin live API verification proxy failed connection_id=%s error=%s", connection.id, exc)
        flash("KuCoin verification requires the fixed-egress server runtime, but the live API proxy is unavailable.", "danger")
        return redirect(url_for("settings.connection_provider", provider=connection.provider, connection_id=connection.id))
    message = str(payload.get("message") or payload.get("error") or "")
    payload_connection = payload.get("connection") if isinstance(payload.get("connection"), dict) else {}
    provider = str(payload_connection.get("provider") or connection.provider)
    connection_id = int(payload_connection.get("id") or connection.id)
    if upstream.status_code >= 400 or not bool(payload.get("ok", False)):
        flash(message or "KuCoin verification failed on the fixed-egress server runtime.", "danger")
    else:
        flash(message or "Connection verified. Enable it to use live wallet, vault, and order workflows.", "success")
    return redirect(url_for("settings.connection_provider", provider=provider, connection_id=connection_id))


def _verify_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    connection = result["connection"]
    snapshot = result.get("snapshot")
    alerts = list(getattr(snapshot, "alerts", []) or []) if snapshot is not None else []
    message = (
        "Connection verified. Enable it to use live wallet, vault, and order workflows."
        if result.get("ok")
        else str(result.get("error") or "Connection verification failed.")
    )
    return {
        "ok": bool(result.get("ok", False)),
        "code": "trading_connection_verified" if result.get("ok") else "trading_connection_verification_failed",
        "message": message,
        "error": "" if result.get("ok") else message,
        "diagnostics": result.get("diagnostics", {}),
        "snapshot": {"balance_count": len(getattr(snapshot, "balances", []) or []), "alerts": alerts} if snapshot is not None else {},
        "connection": _connection_payload(connection),
    }


def _connection_payload(connection: TradingConnection) -> dict[str, Any]:
    return {
        "id": int(connection.id),
        "provider": connection.provider,
        "is_active": bool(connection.is_active),
        "verification_status": connection.verification_status,
    }


def _internal_json_requested() -> bool:
    return is_live_api_internal_request() and str(request.args.get("internal_json") or "").strip().lower() in {"1", "true", "yes"}


def _settings_verify_deferred_for_request() -> bool:
    if is_live_api_internal_request():
        return False
    if _settings_direct_kucoin_fixed_egress_configured():
        return False
    live_origin = _settings_public_live_api_origin()
    internal_origin = _settings_live_api_internal_origin() if _settings_live_api_proxy_enabled() else ""
    known_live_origins = {origin for origin in (live_origin, internal_origin) if origin}
    if known_live_origins:
        return _request_origin() not in known_live_origins
    return (
        bool(current_app.config.get("KUCOIN_FIXED_EGRESS_REQUIRED", False))
        and str(current_app.config.get("DEPLOYMENT_TARGET") or "").lower() == "vercel"
    )


def _settings_live_api_proxy_enabled() -> bool:
    return bool(
        current_app.config.get("LIVE_API_PROXY_ENABLED")
        and _settings_live_api_internal_origin()
        and str(current_app.config.get("LIVE_API_INTERNAL_TOKEN") or "").strip()
    )


def _settings_direct_kucoin_fixed_egress_configured() -> bool:
    if str(current_app.config.get("KUCOIN_EGRESS_PROXY_URL") or current_app.config.get("QUOTAGUARDSTATIC_URL") or "").strip():
        return True
    return bool(
        current_app.config.get("KUCOIN_NATIVE_STATIC_EGRESS_ENABLED")
        and str(current_app.config.get("KUCOIN_EGRESS_PUBLIC_IPS") or "").strip()
    )


def _settings_public_live_api_origin() -> str:
    return _origin_from_url(str(current_app.config.get("PUBLIC_LIVE_API_ORIGIN") or ""))


def _settings_live_api_internal_origin() -> str:
    return _origin_from_url(
        str(current_app.config.get("LIVE_API_INTERNAL_ORIGIN") or current_app.config.get("PUBLIC_LIVE_API_ORIGIN") or "")
    )


def _request_origin() -> str:
    return _origin_from_url(request.host_url)


def _origin_from_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except Exception:  # noqa: BLE001
        return raw.rstrip("/")
    if not parsed.scheme or not parsed.netloc:
        return raw.rstrip("/")
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}".rstrip("/")


def _kucoin_fixed_egress_required_message() -> str:
    return (
        "KuCoin verification must run through a configured fixed-egress server runtime or proxy. "
        "Do not whitelist the browser IP for production; whitelist the fixed server/proxy egress IP shown by KuCoin."
    )


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
        raw_status = str(getattr(connection, "verification_status", "") or "not_connected")
        status = "not_connected" if connection is None else raw_status
        enabled = bool(connection and connection.is_active and status == "verified" and spec.get("tradable"))
        health = connection_health_by_id.get(connection.id, {}) if connection and connection.id is not None else {}
        status_label, status_class, status_tone = _provider_status_display(connection, status)
        connection_label, connection_class = _provider_connection_display(connection, health)
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
                "status_label": status_label,
                "status_class": status_class,
                "status_tone": status_tone,
                "health": health,
                "connection_label": connection_label,
                "connection_class": connection_class,
                "trading_label": "Enabled" if enabled else "Disabled",
                "credential_label": _connection_preview(connection, str(spec.get("connection_type", ""))),
                "detail_note": _provider_detail_note(connection, health),
                "action_label": _provider_action_label(connection, status, enabled),
                "can_enable": bool(connection and spec.get("tradable") and status == "verified" and not enabled),
                "can_disable": enabled,
            }
        )
    return cards


def _provider_status_display(connection: TradingConnection | None, status: str) -> tuple[str, str, str]:
    if connection is None or status == "not_connected":
        return "Not connected", "muted", "muted"
    if status == "verified":
        return "Verified", "positive", "positive"
    if status in {"needs_verification", "failed", "error"}:
        return "Action needed", "warning", "warning"
    if status == "not_supported":
        return "Disabled", "muted", "muted"
    return "Action needed", "warning", "warning"


def _provider_connection_display(connection: TradingConnection | None, health: dict[str, Any]) -> tuple[str, str]:
    if connection is None:
        return "Not connected", "muted"
    if not health:
        return "Connection not checked", "muted"
    if bool(health.get("can_trade")):
        return "Online", "positive"
    if health.get("transient_failure"):
        return "Connection not checked", "warning"
    return "Action needed", "warning"


def _provider_action_label(connection: TradingConnection | None, status: str, enabled: bool) -> str:
    if connection is None:
        return "Connect"
    if enabled:
        return "Manage"
    if status == "verified":
        return "Manage"
    return "Review"


def _provider_detail_note(connection: TradingConnection | None, health: dict[str, Any]) -> str:
    if connection is None:
        return "Save limited credentials before verification or live trading can be enabled."
    if health.get("failure_reason"):
        return "Review the latest provider check before enabling trading."
    if not health:
        return "Connection has not been checked in this session."
    return ""


def _connection_preview(connection: TradingConnection | None, connection_type: str = "") -> str:
    if connection is None:
        return "No wallet connected" if connection_type == "wallet_delegation" else "No credentials saved"
    if connection.wallet_address:
        value = str(connection.wallet_address)
        return f"{value[:10]}...{value[-6:]}" if len(value) > 20 else value
    if connection.encrypted_api_key:
        return "API key saved"
    if connection.encrypted_api_secret or connection.encrypted_passphrase:
        return "Credentials saved"
    return "Connection saved"


def _has_saved_credential(connection: TradingConnection) -> bool:
    return bool(
        connection.wallet_address or connection.encrypted_api_key or connection.encrypted_api_secret or connection.encrypted_passphrase
    )


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
