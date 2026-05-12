"""Compatibility wrapper for protected admin routes."""

from __future__ import annotations

from typing import Any, Callable

from flask import Response, current_app

from .auth import current_user, require_admin_user, two_factor_session_valid
from .models import User


def admin_configured() -> bool:
    return bool(User.query.filter_by(role="admin").first() or current_app.config.get("ADMIN_PASSWORD"))


def admin_authenticated() -> bool:
    user = current_user()
    return bool(user and user.is_admin and two_factor_session_valid(user))


def check_admin_credentials(username: str, password: str) -> bool:
    from .auth import password_matches

    user = User.query.filter_by(username=username, role="admin").one_or_none()
    return bool(user and password_matches(user, password))


def require_admin() -> Response | None:
    return require_admin_user()


def admin_required(fn: Callable[..., Any]) -> Callable[..., Any]:
    from .auth import admin_required as guard

    return guard(fn)
