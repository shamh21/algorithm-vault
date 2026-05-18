"""Lightweight CSRF helpers for plain Flask forms."""

from __future__ import annotations

import hmac
import secrets

from flask import abort, current_app, request, session

from .live_api_internal import is_live_api_internal_request

CSRF_SESSION_KEY = "_csrf_token"


def csrf_token() -> str:
    """Return the current session CSRF token, creating one if needed."""

    token = session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return str(token)


def csrf_input() -> str:
    """Return a hidden input field for templates that need manual rendering."""

    return f'<input type="hidden" name="csrf_token" value="{csrf_token()}">'


def validate_csrf_request() -> None:
    """Abort unsafe requests with an invalid or missing CSRF token."""

    if not bool(current_app.config.get("WTF_CSRF_ENABLED", True)):
        return
    if getattr(request, "routing_exception", None) is not None:
        return
    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return
    if is_live_api_internal_request():
        return
    if _csrf_exempt_auth_request():
        return

    expected = session.get(CSRF_SESSION_KEY)
    supplied = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
    if not expected or not supplied or not hmac.compare_digest(str(expected), str(supplied)):
        abort(400, description="Invalid or missing CSRF token.")


def _csrf_exempt_auth_request() -> bool:
    """Allow credential sign-in requests to recover from stale anonymous sessions.

    Login requests are rate-limited separately and do not mutate an already
    authenticated account. Exempting only these endpoints prevents a stale or
    rotated anonymous CSRF session cookie from blocking users before they can
    establish a fresh authenticated session.
    """

    path = request.path.rstrip("/") or "/"
    if path.startswith("/_internal/mpc-signer/"):
        return True
    return request.method == "POST" and path in {"/login", "/admin/api/sign-in"}
