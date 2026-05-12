"""Username/password authentication and app-based 2FA routes."""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlparse

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..auth import (
    current_user,
    decrypt_totp_secret,
    encrypt_totp_secret,
    generate_totp_secret,
    login_user,
    logout_user,
    password_hash,
    password_matches,
    provisioning_uri,
    qr_code_data_uri,
    verify_totp,
)
from ..extensions import db
from ..models import ReferralInviteCode, TradingConnection, User


auth_bp = Blueprint("auth", __name__)


@auth_bp.get("/register")
def register():
    invite_required = bool(current_app.config.get("SIGNUP_INVITE_CODE")) or _managed_invites_available()
    return render_template(
        "auth/register.html",
        signup_enabled=True,
        invite_required=invite_required,
    )


@auth_bp.post("/register")
def register_post():
    invite = str(current_app.config.get("SIGNUP_INVITE_CODE", ""))
    username = _clean_username(request.form.get("username"))
    password = request.form.get("password", "")
    confirm = request.form.get("confirm_password", "")
    invite_code = str(request.form.get("invite_code", "")).strip()

    managed_invite = _managed_invite_for(invite_code)
    managed_required = _managed_invites_available()
    if managed_required and managed_invite is None:
        flash("Invite code is invalid.", "danger")
        return redirect(url_for("auth.register"))
    if invite and not managed_required and invite_code != invite:
        flash("Invite code is invalid.", "danger")
        return redirect(url_for("auth.register"))

    if len(username) < 3:
        flash("Choose a username with at least 3 characters.", "danger")
        return redirect(url_for("auth.register"))

    if len(password) < 8:
        flash("Choose a password with at least 8 characters.", "danger")
        return redirect(url_for("auth.register"))

    if password != confirm:
        flash("Password confirmation does not match.", "danger")
        return redirect(url_for("auth.register"))

    if User.query.filter_by(username=username).one_or_none() is not None:
        flash("That username is already registered.", "danger")
        return redirect(url_for("auth.register"))

    user = User(
        username=username,
        password_hash=password_hash(password),
        role="user",
        referral_invite_code_id=managed_invite.id if managed_invite is not None else None,
    )
    db.session.add(user)
    if managed_invite is not None:
        managed_invite.usage_count = int(managed_invite.usage_count or 0) + 1
    db.session.commit()

    login_user(user, two_factor_verified=False)
    flash("Account created. Set up app-based 2FA to continue.", "success")
    return redirect(url_for("auth.setup_2fa"))


@auth_bp.get("/login")
def login():
    return render_template(
        "auth/login.html",
        username=_clean_username(request.args.get("username")),
        next=request.args.get("next", ""),
    )


@auth_bp.post("/login")
def login_post():
    username = _clean_username(request.form.get("username"))
    password = request.form.get("password", "")
    code = str(request.form.get("totp_code", "")).strip()
    user = User.query.filter_by(username=username).one_or_none()

    if user is None or not password_matches(user, password):
        flash("Invalid username or password.", "danger")
        return redirect(url_for("auth.login"))

    if not user.two_factor_enabled:
        login_user(user, two_factor_verified=False)
        flash("Set up app-based 2FA before accessing your wallet.", "warning")
        return redirect(url_for("auth.setup_2fa"))

    if not code:
        flash("Enter your 6-digit authenticator code.", "warning")
        return render_template("auth/login.html", username=username, require_2fa=True)

    if not verify_totp(user, code):
        flash("Invalid authenticator code. Try again.", "danger")
        return render_template("auth/login.html", username=username, require_2fa=True)

    login_user(user, two_factor_verified=True)
    flash("Signed in.", "success")

    return redirect(_safe_next_url(request.args.get("next")))


@auth_bp.post("/logout")
def logout():
    logout_user()
    flash("Signed out.", "success")
    return redirect(url_for("auth.login"))


@auth_bp.route("/setup-2fa", methods=["GET", "POST"])
def setup_2fa():
    user = current_user()

    if user is None:
        flash("Sign in before setting up 2FA.", "warning")
        return redirect(url_for("auth.login", next=url_for("auth.setup_2fa")))

    if user.two_factor_enabled:
        flash("2FA is already enabled for this account.", "success")
        return redirect(_post_auth_redirect(user))

    secret = decrypt_totp_secret(user)

    if not secret:
        secret = generate_totp_secret()
        user.totp_secret_encrypted = encrypt_totp_secret(secret)
        db.session.commit()

    if request.method == "POST":
        code = str(request.form.get("totp_code", "")).strip()

        if not verify_totp(user, code):
            flash("Invalid authenticator code. Try again.", "danger")
        else:
            user.two_factor_enabled_at = datetime.now(timezone.utc)
            db.session.commit()

            login_user(user, two_factor_verified=True)
            flash("2FA enabled.", "success")
            return redirect(_post_auth_redirect(user))

    uri = provisioning_uri(user, secret)

    return render_template(
        "auth/setup_2fa.html",
        secret=secret,
        qr_code_uri=qr_code_data_uri(uri),
    )


def _clean_username(value: str | None) -> str:
    return str(value or "").strip().lower()


def _managed_invites_available() -> bool:
    return ReferralInviteCode.query.filter_by(is_active=True).first() is not None


def _managed_invite_for(code: str) -> ReferralInviteCode | None:
    value = str(code or "").strip()
    if not value:
        return None
    invite = ReferralInviteCode.query.filter_by(code=value, is_active=True).one_or_none()
    if invite is None or not invite.available:
        return None
    return invite


def _post_auth_redirect(user: User) -> str:
    if (
        not user.is_admin
        and current_app.config.get("ENABLE_LIVE_TRADING", False)
        and current_app.config.get("APP_MODE") == "live"
        and TradingConnection.query.filter_by(
            user_id=user.id,
            is_active=True,
            verification_status="verified",
            provider="hyperliquid",
        ).first() is None
    ):
        return url_for("settings.connections")
    return url_for("dashboard.index") if user.is_admin else url_for("consumer.home")


def _safe_next_url(next_url: str | None) -> str:
    if not next_url:
        return url_for("consumer.home")

    parsed = urlparse(next_url)

    if parsed.netloc or parsed.scheme or not next_url.startswith("/"):
        return url_for("consumer.home")

    return next_url
