"""KuCoin IP and permission diagnostics for operator-facing readiness."""

from __future__ import annotations

import ipaddress
from typing import Any

import requests

from .kucoin_compliance import kucoin_operator_region_status


def build_kucoin_ip_diagnostics(
    config: dict[str, Any],
    *,
    operator_ip: str = "",
    resolved_server_ips: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build a no-secrets KuCoin IP restriction diagnostic payload."""

    operator_ips = _parse_ip_list(operator_ip)
    configured_egress_ips = _parse_ip_list(config.get("KUCOIN_EGRESS_PUBLIC_IPS") or config.get("KUCOIN_FIXED_EGRESS_PUBLIC_IPS") or "")
    server_egress_ips = list(dict.fromkeys([*configured_egress_ips, *list(resolved_server_ips or [])]))
    trusted_ips = _parse_ip_list(config.get("KUCOIN_TRUSTED_IPS") or config.get("KUCOIN_API_TRUSTED_IPS") or "")
    fixed_egress_configured = _fixed_egress_configured(config, configured_egress_ips)
    fixed_egress_required = bool(config.get("KUCOIN_FIXED_EGRESS_REQUIRED", False))
    mode = str(config.get("KUCOIN_IP_RESTRICTION_MODE") or "").strip().lower()
    if mode not in {"trusted", "unrestricted", "unknown", ""}:
        mode = "unknown"
    if not mode:
        mode = "trusted" if trusted_ips else "unknown"

    missing_from_trusted = [ip for ip in server_egress_ips if trusted_ips and ip not in trusted_ips]
    if mode == "unrestricted":
        status = "unrestricted"
        can_work = False
        message = "KuCoin API key is not configured for trusted-IP restriction."
    elif fixed_egress_required and not fixed_egress_configured:
        status = "fixed_egress_not_configured"
        can_work = False
        message = "Configure fixed egress before using KuCoin trusted-IP routing."
    elif mode == "trusted" and not server_egress_ips:
        status = "server_egress_ip_unknown"
        can_work = False
        message = "Server/proxy egress IP is not configured."
    elif mode == "trusted" and trusted_ips and missing_from_trusted:
        status = "trusted_ip_mismatch"
        can_work = False
        message = "KuCoin trusted IPs do not include every server/proxy egress IP."
    elif mode == "trusted" and not trusted_ips:
        status = "trusted_ips_missing"
        can_work = False
        message = "Set KUCOIN_TRUSTED_IPS to match the KuCoin key trusted-IP list."
    elif mode == "trusted":
        status = "ready"
        can_work = True
        message = "KuCoin trusted-IP routing can use the configured server/proxy egress IP."
    else:
        status = "unknown"
        can_work = False
        message = "KuCoin IP restriction mode is not configured."

    return {
        "operator_ip": operator_ips[0] if operator_ips else "",
        "operator_ips": operator_ips,
        "server_egress_ips": server_egress_ips,
        "trusted_ips": trusted_ips,
        "fixed_egress_configured": fixed_egress_configured,
        "fixed_egress_required": fixed_egress_required,
        "ip_restriction_mode": mode,
        "trusted_ip_status": status,
        "trusted_ip_can_work": can_work,
        "trusted_ip_message": message,
        "missing_trusted_ips": missing_from_trusted,
        "operator_ip_is_trading_egress": False,
    }


def build_kucoin_diagnostics_payload(
    config: dict[str, Any],
    *,
    operator_ip: str = "",
    connection: Any | None = None,
    permission_probe: dict[str, Any] | None = None,
    resolved_server_ips: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build the full no-secrets KuCoin diagnostic payload for UI/API use."""

    permission_payload = permission_probe or _permission_probe_unknown(connection)
    return {
        "provider": "kucoin",
        "ip_restriction": build_kucoin_ip_diagnostics(config, operator_ip=operator_ip, resolved_server_ips=resolved_server_ips),
        "permissions": permission_payload,
        "operator_region": kucoin_operator_region_status(config),
        "secrets_exposed": False,
    }


def resolve_server_egress_ip(config: dict[str, Any], *, timeout: float = 2.5) -> list[str]:
    """Resolve the current server/proxy public IP through an operator-triggered check."""

    url = str(config.get("KUCOIN_EGRESS_IP_CHECK_URL") or "https://api64.ipify.org").strip()
    if not url:
        return []
    session = requests.Session()
    proxy_url = str(config.get("KUCOIN_EGRESS_PROXY_URL") or config.get("QUOTAGUARDSTATIC_URL") or "").strip()
    if proxy_url:
        session.trust_env = False
        session.proxies.update({"http": proxy_url, "https": proxy_url})
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError:
        payload = response.text
    if isinstance(payload, dict):
        return _parse_ip_list(payload.get("ip") or payload.get("origin") or payload.get("address") or "")
    return _parse_ip_list(str(payload))


def _fixed_egress_configured(config: dict[str, Any], configured_egress_ips: list[str]) -> bool:
    if str(config.get("KUCOIN_EGRESS_PROXY_URL") or config.get("QUOTAGUARDSTATIC_URL") or "").strip():
        return True
    return bool(config.get("KUCOIN_NATIVE_STATIC_EGRESS_ENABLED") and configured_egress_ips)


def _permission_probe_unknown(connection: Any | None) -> dict[str, Any]:
    credentials_present = {
        "api_key": bool(getattr(connection, "encrypted_api_key", "")),
        "api_secret": bool(getattr(connection, "encrypted_api_secret", "")),
        "api_passphrase": bool(getattr(connection, "encrypted_passphrase", "")),
    }
    verified = bool(connection is not None and getattr(connection, "verification_status", "") == "verified")
    return {
        "credentials_present": credentials_present,
        "general": {"status": "not_checked", "message": "Use the live preflight to verify read-only account access."},
        "spot": {"status": "not_checked", "message": "Use the live preflight to verify spot account access."},
        "futures": {
            "status": "verified" if verified else "not_checked",
            "message": "Connection is verified." if verified else "Verify the KuCoin connection.",
        },
        "unified": {"status": "not_checked", "message": "Unified Account permission is operator-selected in KuCoin."},
    }


def _parse_ip_list(value: object) -> list[str]:
    items: list[str] = []
    raw_values = value if isinstance(value, (list, tuple, set)) else str(value or "").replace(";", ",").split(",")
    for raw in raw_values:
        text = str(raw or "").strip().strip("[]")
        if not text:
            continue
        try:
            items.append(str(ipaddress.ip_address(text)))
        except ValueError:
            continue
    return list(dict.fromkeys(items))
