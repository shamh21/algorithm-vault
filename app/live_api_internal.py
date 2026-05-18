"""Signed server-to-server helpers for Vercel to live API vault delegation."""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any

from flask import current_app, g, request

USER_HEADER = "X-AlgVault-User-Id"
TIMESTAMP_HEADER = "X-AlgVault-Internal-Timestamp"
BODY_SHA_HEADER = "X-AlgVault-Internal-Body-SHA256"
SIGNATURE_HEADER = "X-AlgVault-Internal-Signature"

_LIVE_API_INTERNAL_PATHS = {
    "/vault/readiness",
    "/api/vault/readiness",
    "/vault/preview-route",
    "/api/vault/routing-preview",
    "/api/vault/kucoin-diagnostics",
    "/vault/start-cycle",
    "/consumer/start",
    "/vault/start",
    "/vault/cycles",
}
_LIVE_API_INTERNAL_PATH_PREFIXES = (
    "/api/vault/cycles/",
    "/vault/cycles/",
    "/vault/start-status/",
    "/consumer/start-status/",
)


def live_api_internal_path_allowed(path: str) -> bool:
    clean = str(path or "").rstrip("/") or "/"
    return clean in _LIVE_API_INTERNAL_PATHS or any(str(path or "").startswith(prefix) for prefix in _LIVE_API_INTERNAL_PATH_PREFIXES)


def live_api_internal_user_id() -> int | None:
    if not is_live_api_internal_request():
        return None
    raw = str(request.headers.get(USER_HEADER) or "").strip()
    try:
        user_id = int(raw)
    except (TypeError, ValueError):
        return None
    return user_id if user_id > 0 else None


def is_live_api_internal_request() -> bool:
    cached = getattr(g, "_live_api_internal_valid", None)
    if cached is not None:
        return bool(cached)
    valid = _validate_live_api_internal_request()
    g._live_api_internal_valid = valid
    return valid


def sign_live_api_internal_headers(
    config: dict[str, Any],
    *,
    method: str,
    path: str,
    query_string: bytes | str = b"",
    body: bytes = b"",
    user_id: int,
    timestamp: int | None = None,
) -> dict[str, str]:
    token = str(config.get("LIVE_API_INTERNAL_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("LIVE_API_INTERNAL_TOKEN is required for live API proxy signing.")
    timestamp_value = str(int(timestamp or time.time()))
    body_sha = _body_sha(body)
    signature = _signature(
        token,
        method=method,
        path=path,
        query_string=query_string,
        user_id=str(int(user_id)),
        timestamp=timestamp_value,
        body_sha=body_sha,
    )
    return {
        USER_HEADER: str(int(user_id)),
        TIMESTAMP_HEADER: timestamp_value,
        BODY_SHA_HEADER: body_sha,
        SIGNATURE_HEADER: signature,
    }


def _validate_live_api_internal_request() -> bool:
    if not live_api_internal_path_allowed(request.path):
        return False
    token = str(current_app.config.get("LIVE_API_INTERNAL_TOKEN") or "").strip()
    if not token:
        return False
    user_id = str(request.headers.get(USER_HEADER) or "").strip()
    timestamp = str(request.headers.get(TIMESTAMP_HEADER) or "").strip()
    body_sha = str(request.headers.get(BODY_SHA_HEADER) or "").strip()
    supplied = str(request.headers.get(SIGNATURE_HEADER) or "").strip()
    if not user_id or not timestamp or not body_sha or not supplied:
        return False
    try:
        parsed_timestamp = int(timestamp)
        parsed_user_id = int(user_id)
    except (TypeError, ValueError):
        return False
    if parsed_user_id <= 0:
        return False
    skew = max(5, int(current_app.config.get("LIVE_API_INTERNAL_MAX_SKEW_SECONDS", 60) or 60))
    if abs(int(time.time()) - parsed_timestamp) > skew:
        return False
    body = request.get_data(cache=True) or b""
    if not hmac.compare_digest(body_sha, _body_sha(body)):
        return False
    expected = _signature(
        token,
        method=request.method,
        path=request.path,
        query_string=request.query_string,
        user_id=user_id,
        timestamp=timestamp,
        body_sha=body_sha,
    )
    return hmac.compare_digest(supplied, expected)


def _body_sha(body: bytes) -> str:
    return hashlib.sha256(body or b"").hexdigest()


def _signature(
    token: str,
    *,
    method: str,
    path: str,
    query_string: bytes | str,
    user_id: str,
    timestamp: str,
    body_sha: str,
) -> str:
    query = query_string.decode("utf-8", errors="surrogateescape") if isinstance(query_string, bytes) else str(query_string or "")
    canonical = "\n".join(
        [
            str(method or "").upper(),
            str(path or ""),
            query,
            str(user_id),
            str(timestamp),
            str(body_sha),
        ]
    )
    return hmac.new(token.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()
