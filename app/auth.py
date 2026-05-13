"""User authentication, TOTP encryption, and access guards."""

from __future__ import annotations

import base64
import hashlib
import io
from collections.abc import Callable
from functools import wraps
from typing import Any

import pyotp
import qrcode
from cryptography.fernet import Fernet
from flask import Response, current_app, flash, jsonify, redirect, request, session, url_for
from qrcode.image.svg import SvgPathImage
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db
from .models import User


def password_hash(password: str) -> str:
    return generate_password_hash(password)


def password_matches(user: User, password: str) -> bool:
    return check_password_hash(user.password_hash, password)


def current_user() -> User | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db.session.get(User, int(user_id))


def login_user(user: User, *, two_factor_verified: bool = False) -> None:
    session.clear()
    session["user_id"] = user.id
    session["two_factor_verified"] = bool(two_factor_verified)


def logout_user() -> None:
    session.clear()


def two_factor_session_valid(user: User) -> bool:
    return user.two_factor_enabled and bool(session.get("two_factor_verified"))


def encrypt_totp_secret(secret: str) -> str:
    return _fernet().encrypt(secret.encode("utf-8")).decode("utf-8")


def decrypt_totp_secret(user: User) -> str | None:
    if not user.totp_secret_encrypted:
        return None
    try:
        return _fernet().decrypt(user.totp_secret_encrypted.encode("utf-8")).decode("utf-8")
    except Exception:  # noqa: BLE001
        return None


def generate_totp_secret() -> str:
    return pyotp.random_base32()


def verify_totp(user: User, code: str) -> bool:
    secret = decrypt_totp_secret(user)
    if not secret:
        return False
    return pyotp.TOTP(secret).verify((code or "").strip(), valid_window=1)


def provisioning_uri(user: User, secret: str) -> str:
    issuer = current_app.config.get("APP_NAME", "Algorithm Vault")
    return pyotp.TOTP(secret).provisioning_uri(name=user.username, issuer_name=issuer)


def qr_code_data_uri(payload: str) -> str:
    image = qrcode.make(payload, image_factory=SvgPathImage)
    buffer = io.BytesIO()
    image.save(buffer)
    return "data:image/svg+xml;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def require_authenticated_user() -> Response | None:
    user = current_user()
    if user is None:
        if _expects_json_guard_response():
            return _json_guard_error("authentication_required", "Sign in to continue.", 401)
        flash("Sign in to continue.", "warning")
        return redirect(url_for("auth.login", next=request.full_path if request.query_string else request.path))
    if not user.two_factor_enabled:
        if _expects_json_guard_response():
            return _json_guard_error("two_factor_required", "Set up app-based 2FA before continuing.", 403)
        flash("Set up app-based 2FA before accessing your wallet.", "warning")
        return redirect(url_for("auth.setup_2fa"))
    if not two_factor_session_valid(user):
        if _expects_json_guard_response():
            return _json_guard_error("two_factor_required", "Enter your authenticator code to continue.", 401)
        flash("Enter your authenticator code to continue.", "warning")
        return redirect(url_for("auth.login", next=request.full_path if request.query_string else request.path))
    return None


def login_required(fn: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        guard = require_authenticated_user()
        if guard is not None:
            return guard
        return fn(*args, **kwargs)

    return wrapper


def admin_required(fn: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        guard = require_admin_user()
        if guard is not None:
            return guard
        return fn(*args, **kwargs)

    return wrapper


def require_admin_user() -> Response | None:
    guard = require_authenticated_user()
    if guard is not None:
        return guard
    user = current_user()
    if user is None or not user.is_admin:
        if _expects_json_guard_response():
            return _json_guard_error("admin_required", "Admin access is required.", 403)
        flash("Admin access is required for advanced controls.", "danger")
        return redirect(url_for("consumer.home"))
    return None


def _json_guard_error(code: str, message: str, status_code: int) -> Response:
    response = jsonify({"ok": False, "code": code, "error": message})
    response.status_code = status_code
    return response


def _expects_json_guard_response() -> bool:
    if request.path.startswith("/admin/api/"):
        return True
    if request.path.startswith("/api/vault-cycles/") and request.path.endswith("/process-profit-share"):
        return True
    if request.accept_mimetypes.best == "application/json":
        return True
    return bool(request.is_json)


def _fernet() -> Fernet:
    configured = str(current_app.config.get("TOTP_ENCRYPTION_KEY", "") or "").strip()
    if configured:
        try:
            return Fernet(configured.encode("utf-8"))
        except Exception:  # noqa: BLE001
            pass
    raw = str(current_app.config.get("SECRET_KEY", "dev-secret-change-me")).encode("utf-8")
    key = base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
    return Fernet(key)
