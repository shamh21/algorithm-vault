"""Compatibility wrapper for protected admin routes."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from flask import Response, current_app, request

from .auth import current_user, require_admin_user, require_authenticated_user, two_factor_session_valid
from .models import User


_USER_BACKTEST_ENDPOINTS = {
    "backtests.index",
    "backtests.symbols_api",
    "backtests.quote_api",
    "backtests.run",
}
_USER_ADMIN_DECORATED_ENDPOINTS = {
    "admin.risk",
    "admin.risk_state",
    "admin.risk_audit_events",
    "admin.refresh_risk_exchange_limits",
}


def admin_configured() -> bool:
    return bool(User.query.filter_by(role="admin").first() or current_app.config.get("ADMIN_PASSWORD"))


def admin_authenticated() -> bool:
    user = current_user()
    return bool(user and user.is_admin and two_factor_session_valid(user))


def check_admin_credentials(username: str, password: str) -> bool:
    from .auth import password_matches

    user = User.query.filter_by(username=username, role="admin").one_or_none()
    return bool(user and password_matches(user, password))


def _authenticated_user_route() -> bool:
    """Return true only for explicitly user-accessible research routes."""
    return str(request.endpoint or "") in _USER_BACKTEST_ENDPOINTS


def require_admin() -> Response | None:
    if _authenticated_user_route():
        return require_authenticated_user()
    return require_admin_user()


def admin_required(fn: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        endpoint = str(request.endpoint or "")
        guard = require_authenticated_user() if endpoint in _USER_ADMIN_DECORATED_ENDPOINTS else require_admin_user()
        if guard is not None:
            return guard
        return fn(*args, **kwargs)

    return wrapper
