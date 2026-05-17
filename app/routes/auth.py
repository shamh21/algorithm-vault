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
from ..models import InviteCodeUsage, ReferralInviteCode, TradingConnection, TreasuryReserveState, User
from ..services.withdrawal_config import wallet_withdrawals_enabled

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
        role=managed_invite.assigned_role if managed_invite is not None else "user",
        referral_invite_code_id=managed_invite.id if managed_invite is not None else None,
    )
    db.session.add(user)
    if managed_invite is not None:
        managed_invite.usage_count = int(managed_invite.usage_count or 0) + 1
        db.session.flush()
        usage = InviteCodeUsage(
            invite_code_id=managed_invite.id,
            invitee_user_id=user.id,
            status="accepted",
            accepted_disclosure_version="invite-profit-share-v1",
        )
        usage.details = {
            "invite_code": managed_invite.code,
            "profit_share_percent": managed_invite.effective_profit_share_percent,
            "profit_share_wallet": managed_invite.profit_share_wallet,
            "disclosure": "Using this invite code may allocate a percentage of positive Vault Cycle profit to sufyanh. It never applies to deposits, principal, or losses.",
        }
        db.session.add(usage)
    db.session.commit()

    login_user(user, two_factor_verified=False)
    flash("Account created. Set up app-based 2FA to continue.", "success")
    return redirect(url_for("auth.setup_2fa"))


@auth_bp.get("/login")
def login():
    next_url = request.args.get("next", "")
    return render_template(
        "auth/login.html",
        username=_clean_username(request.args.get("username")),
        next=next_url,
        next_destination=_next_destination_label(next_url),
        ops_snapshot=_login_ops_snapshot(),
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
        next_url = request.args.get("next", "")
        return render_template(
            "auth/login.html",
            username=username,
            require_2fa=True,
            next=next_url,
            next_destination=_next_destination_label(next_url),
            ops_snapshot=_login_ops_snapshot(),
        )

    if not verify_totp(user, code):
        flash("Invalid authenticator code. Try again.", "danger")
        next_url = request.args.get("next", "")
        return render_template(
            "auth/login.html",
            username=username,
            require_2fa=True,
            next=next_url,
            next_destination=_next_destination_label(next_url),
            ops_snapshot=_login_ops_snapshot(),
        )

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
            user.two_factor_enabled_at = datetime.now(timezone.utc)  # noqa: UP017
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
    return ReferralInviteCode.query.filter_by(is_active=True, deleted_at=None).first() is not None


def _managed_invite_for(code: str) -> ReferralInviteCode | None:
    value = str(code or "").strip().upper()
    if not value:
        return None
    invite = ReferralInviteCode.query.filter_by(code=value, is_active=True).with_for_update().one_or_none()
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
        ).first()
        is None
    ):
        return url_for("settings.connections")
    return url_for("consumer.home")


def _safe_next_url(next_url: str | None) -> str:
    if not next_url:
        return url_for("consumer.home")

    parsed = urlparse(next_url)

    if parsed.netloc or parsed.scheme or not next_url.startswith("/"):
        return url_for("consumer.home")

    return next_url


def _next_destination_label(next_url: str | None) -> str:
    parsed = urlparse(str(next_url or ""))
    path = parsed.path.rstrip("/") or "/"
    if path.startswith("/wallet"):
        return "Wallet"
    if path.startswith("/vault"):
        return "Vault"
    if path.startswith("/convert"):
        return "Convert"
    if path.startswith("/admin/backtests"):
        return "Backtests"
    if path.startswith("/admin/dashboard"):
        return "Trading Ops"
    if path.startswith("/admin"):
        return "Admin"
    return "Dashboard" if path and path != "/" else ""


def _login_ops_snapshot() -> dict[str, object]:
    validation = current_app.config.get("RUNTIME_CONFIG_VALIDATION")
    backend = str(getattr(validation, "database_backend", "") or current_app.config.get("SQLALCHEMY_DATABASE_URI", "")).lower()
    if "postgres" in backend:
        database_label = "Postgres"
    elif "sqlite" in backend:
        database_label = "SQLite"
    else:
        database_label = "Other"

    live_mode = str(current_app.config.get("APP_MODE", "paper") or "paper").lower() == "live" and bool(
        current_app.config.get("ENABLE_LIVE_TRADING", False)
    )
    withdrawals_enabled = wallet_withdrawals_enabled(current_app.config)
    treasury_enabled = bool(current_app.config.get("PLATFORM_GAS_TREASURY_ENABLED", False))
    reserve_state = ""
    reserve_balance: float | None = None
    try:
        state = TreasuryReserveState.query.filter_by(network="Ethereum").one_or_none()
        if state is not None:
            reserve_state = str(state.health_status or "").strip().lower()
            reserve_balance = float(state.total_eth_balance or 0.0)
    except Exception:  # noqa: BLE001
        reserve_state = "unavailable"

    tone = "ready"
    message = "Live operations are online with approval-gated withdrawals."
    if not live_mode:
        tone = "attention"
        message = "Live trading is not active for this runtime."
    elif not withdrawals_enabled:
        tone = "attention"
        message = "Withdrawals are disabled by server-side readiness gates."
    elif not treasury_enabled:
        tone = "attention"
        message = "Treasury gas funding is not enabled."
    elif reserve_state in {"warning", "low", "critical", "emergency"} or reserve_balance == 0:
        tone = "attention"
        message = "Withdrawals are enabled; token gas top-ups may queue until treasury reserve is funded."

    return {
        "tone": tone,
        "status_label": "Attention" if tone == "attention" else "Ready",
        "database": database_label,
        "mode": "Live" if live_mode else "Paper",
        "withdrawals": "Enabled" if withdrawals_enabled else "Blocked",
        "treasury": (reserve_state or "Ready").replace("_", " ").title() if treasury_enabled else "Disabled",
        "message": message,
    }
