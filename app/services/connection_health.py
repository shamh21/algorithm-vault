"""Live trading connection health snapshots stored without schema changes."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from ..models import Setting, TradingConnection


def connection_health_key(connection_id: int) -> str:
    return f"connection_health:{int(connection_id)}"


def latest_connection_health(connection_id: int | None) -> dict[str, Any]:
    if connection_id is None:
        return {}
    value = Setting.get_json(connection_health_key(connection_id), {})
    return value if isinstance(value, dict) else {}


def active_connection_health() -> list[dict[str, Any]]:
    health: list[dict[str, Any]] = []
    for connection in TradingConnection.query.filter_by(is_active=True).order_by(TradingConnection.updated_at.desc()).all():
        payload = latest_connection_health(connection.id)
        if payload:
            health.append(payload)
    return health


def store_connection_health(connection: TradingConnection, payload: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "connection_id": connection.id,
        "provider": connection.provider,
        "mode": "live",
        "last_checked_at": datetime.utcnow().isoformat() + "Z",
        **payload,
    }
    Setting.set_json(connection_health_key(connection.id), normalized)
    return normalized


def build_connection_health(
    connection: TradingConnection,
    *,
    can_trade: bool,
    alerts: list[str] | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    raw_failure = failure_reason or "; ".join(alerts or [])
    parsed = parse_exchange_failure(raw_failure)
    return {
        "connection_id": connection.id,
        "provider": connection.provider,
        "mode": "live",
        "can_trade": bool(can_trade),
        "alerts": list(alerts or []),
        "failure_reason": raw_failure,
        "provider_code": parsed.get("provider_code"),
        "client_ip": parsed.get("client_ip"),
        "ip_whitelist_blocked": bool(parsed.get("ip_whitelist_blocked", False)),
        "transient_failure": bool(parsed.get("transient_failure", False)),
        "failure_category": parsed.get("failure_category", ""),
    }


def parse_exchange_failure(message: object) -> dict[str, Any]:
    text = _stringify_failure(message)
    parsed: dict[str, Any] = {
        "provider_code": None,
        "client_ip": None,
        "ip_whitelist_blocked": False,
        "transient_failure": False,
        "failure_category": "",
    }
    if not text:
        return parsed
    payload = _json_failure_payload(text)
    if isinstance(payload, dict):
        code = payload.get("code") or payload.get("status") or payload.get("error_code")
        msg = payload.get("msg") or payload.get("message") or payload.get("error") or text
        if code is not None:
            parsed["provider_code"] = str(code)
        text = str(msg)
    code_match = re.search(r'["\']?code["\']?\s*[:=]\s*["\']?([A-Za-z0-9_-]+)', text)
    if code_match and not parsed["provider_code"]:
        parsed["provider_code"] = code_match.group(1)
    ip_match = re.search(
        r"(?:clientIp|client ip|current clientIp|current client ip)\s*(?:is)?\s*:?\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", text, re.I
    )
    if ip_match:
        parsed["client_ip"] = ip_match.group(1)
    parsed["ip_whitelist_blocked"] = "invalid request ip" in text.lower() or bool(parsed["client_ip"])
    lowered = text.lower()
    if parsed["ip_whitelist_blocked"]:
        parsed["failure_category"] = "ip_whitelist"
    credential_markers = (
        "invalid kc-api-key",
        "invalid api key",
        "api-key format invalid",
        "signature not valid",
        "invalid signature",
        "invalid passphrase",
        "api key not exists",
        "unauthorized",
        "cannot be decrypted",
    )
    permission_markers = (
        "permission",
        "forbidden",
        "futures permission",
        "api-permission",
        "not allowed",
    )
    symbol_markers = (
        "not mapped",
        "invalid symbol",
        "symbol not exists",
        "contract not exist",
        "contract_size",
        "sizing metadata",
    )
    if any(marker in lowered for marker in credential_markers):
        parsed["failure_category"] = "invalid_credentials"
    elif any(marker in lowered for marker in permission_markers):
        parsed["failure_category"] = "missing_permission"
    elif any(marker in lowered for marker in symbol_markers):
        parsed["failure_category"] = "bad_symbol_mapping"
    timeout_markers = (
        "timed out",
        "timeout",
        "max retries exceeded",
        "connectionerror",
        "failed to resolve",
        "nameresolutionerror",
        "nodename nor servname",
        "temporary failure",
    )
    if any(marker in lowered for marker in timeout_markers):
        parsed["transient_failure"] = True
        parsed["failure_category"] = "network_timeout" if "timed out" in lowered or "timeout" in lowered else "network_unavailable"
    return parsed


def operator_connection_message(health: dict[str, Any]) -> str:
    reason = str(health.get("failure_reason") or "Live exchange account snapshot failed.")
    provider = str(health.get("provider") or "exchange").title()
    client_ip = health.get("client_ip")
    if client_ip:
        return (
            f"{provider} blocked live trading: {reason}. Whitelist server egress IP {client_ip} "
            "on the exchange API key before starting a vault cycle."
        )
    if health.get("transient_failure"):
        return (
            f"{provider} live access is temporarily unavailable: {reason}. Wait for a fresh readiness check before starting a vault cycle."
        )
    return f"{provider} blocked live trading: {reason}"


def _stringify_failure(message: object) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message.strip()
    try:
        return json.dumps(message)
    except TypeError:
        return str(message)


def _json_failure_payload(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\})", stripped)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
