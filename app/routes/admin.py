"""Admin authentication and advanced control routes."""

from __future__ import annotations

from urllib.parse import urlparse

from flask import Blueprint, flash, redirect, render_template, request, url_for

from ..auth import current_user
from ..admin_auth import admin_required
from ..extensions import db
from ..models import (
    AuditLog,
    DepositAddress,
    Order,
    RiskEvent,
    ShadowLiveObservation,
    StrategyRanking,
    StrategyRun,
    StrategyValidation,
    User,
    VaultCycle,
    WalletAddress,
    WalletAuditLog,
    WalletTransaction,
    WalletWithdrawal,
)
from ..runtime import get_current_mode, get_service
from ..services.db_retry import commit_with_retry


admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.get("/")
@admin_required
def index():
    return redirect(url_for("dashboard.index"))


@admin_bp.get("/login")
def login():
    return redirect(url_for("auth.login", next=_safe_next_url(request.args.get("next"))))


@admin_bp.post("/login")
def login_post():
    return redirect(url_for("auth.login", next=_safe_next_url(request.args.get("next"))), code=307)


@admin_bp.post("/logout")
def logout():
    return redirect(url_for("auth.logout"), code=307)


@admin_bp.get("/risk")
@admin_required
def risk():
    mode = get_current_mode()
    risk_engine = get_service("risk_engine")

    return render_template(
        "advanced/risk.html",
        mode=mode,
        risk_status=risk_engine.status(mode),
        recent_risk=RiskEvent.query.order_by(RiskEvent.created_at.desc()).limit(50).all(),
        audits=AuditLog.query.order_by(AuditLog.created_at.desc()).limit(50).all(),
    )


@admin_bp.get("/live-readiness")
@admin_required
def live_readiness():
    mode = get_current_mode()
    user = current_user()
    trading_connections = get_service("trading_connections")
    connection = trading_connections.active_tradable_connection(user.id) if user is not None else None
    risk_engine = get_service("risk_engine")
    wallet_readiness = get_service("wallet_custody").readiness()
    wallet_readiness.update(
        {
            "pending_withdrawals": WalletWithdrawal.query.filter(
                WalletWithdrawal.status.in_(["pending_approval", "pending_submission"])
            ).count(),
            "generated_address_count": WalletAddress.query.filter(
                WalletAddress.encrypted_metadata_json.like('%"custody": "in_app"%')
            ).count(),
            "sync_failures": WalletAuditLog.query.filter_by(action="wallet_deposit_sync_failed").count(),
        }
    )

    return render_template(
        "advanced/live_readiness.html",
        mode=mode,
        can_trade=trading_connections.can_trade(user.id if user else None, "live", connection.id if connection else None),
        has_account=connection is not None,
        risk_status=risk_engine.status(
            mode,
            user_id=user.id if user else None,
            trading_connection_id=connection.id if connection else None,
        ),
        active_cycles=(
            VaultCycle.query.filter(VaultCycle.status.in_(["active", "settling"]))
            .order_by(VaultCycle.started_at.desc())
            .limit(25)
            .all()
        ),
        recent_orders=Order.query.order_by(Order.created_at.desc()).limit(25).all(),
        recent_audits=(
            AuditLog.query.filter(AuditLog.category.in_(["vault", "orders", "panic"]))
            .order_by(AuditLog.created_at.desc())
            .limit(25)
            .all()
        ),
        wallet_readiness=wallet_readiness,
        recent_wallet_withdrawals=WalletWithdrawal.query.order_by(WalletWithdrawal.created_at.desc()).limit(10).all(),
        recent_wallet_audits=WalletAuditLog.query.order_by(WalletAuditLog.created_at.desc()).limit(10).all(),
        recent_risk=RiskEvent.query.order_by(RiskEvent.created_at.desc()).limit(25).all(),
    )


@admin_bp.get("/strategies")
@admin_required
def strategies():
    return render_template(
        "advanced/strategies.html",
        mode=get_current_mode(),
        strategy_runs=StrategyRun.query.order_by(StrategyRun.created_at.desc()).limit(50).all(),
        strategy_rankings=(
            StrategyRanking.query.order_by(
                StrategyRanking.rejected.asc(),
                StrategyRanking.score.desc(),
                StrategyRanking.created_at.desc(),
            )
            .limit(30)
            .all()
        ),
        validations=StrategyValidation.query.order_by(StrategyValidation.started_at.desc()).limit(30).all(),
        shadow_observations=ShadowLiveObservation.query.order_by(ShadowLiveObservation.created_at.desc()).limit(30).all(),
        vault_cycles=VaultCycle.query.order_by(VaultCycle.started_at.desc()).limit(30).all(),
    )


@admin_bp.get("/deposit-addresses")
@admin_required
def deposit_addresses():
    users = User.query.all()

    return render_template(
        "advanced/deposit_addresses.html",
        addresses=DepositAddress.query.order_by(DepositAddress.created_at.desc()).limit(200).all(),
        users={user.id: user for user in users},
        wallet_readiness=get_service("wallet_custody").readiness(),
    )


@admin_bp.get("/wallet-withdrawals")
@admin_required
def wallet_withdrawals():
    users = User.query.all()
    return render_template(
        "advanced/wallet_withdrawals.html",
        withdrawals=WalletWithdrawal.query.order_by(WalletWithdrawal.created_at.desc()).limit(200).all(),
        users={user.id: user for user in users},
        panic_lock=bool(get_service("risk_engine").status("live").get("panic_lock", False)),
    )


@admin_bp.post("/wallet-withdrawals/<int:withdrawal_id>/approve")
@admin_required
def approve_wallet_withdrawal(withdrawal_id: int):
    withdrawal = db.session.get(WalletWithdrawal, withdrawal_id)
    if withdrawal is None:
        flash("Withdrawal request was not found.", "danger")
        return redirect(url_for("admin.wallet_withdrawals"))
    if bool(get_service("risk_engine").status("live").get("panic_lock", False)):
        flash("Panic lock is active. Withdrawal approval is blocked.", "danger")
        return redirect(url_for("admin.wallet_withdrawals"))
    try:
        result = get_service("self_custody_wallet").approve_withdrawal(
            withdrawal,
            approved_by_user_id=current_user().id if current_user() else None,
            mode=get_current_mode(),
        )
    except RuntimeError as exc:
        db.session.rollback()
        flash(str(exc), "danger")
        return redirect(url_for("admin.wallet_withdrawals"))

    if result.status == "failed":
        get_service("wallet_custody").release_failed_withdrawal(result)
        _update_withdrawal_transaction(result, "failed", result.failure_reason or "Withdrawal approval failed.")
        flash(result.failure_reason or "Withdrawal approval failed.", "danger")
    else:
        _update_withdrawal_transaction(result, "pending_withdrawal", f"Withdrawal workflow {result.id}: {result.status}.")
        flash("Withdrawal approved and submitted to the live custody adapter.", "success")
    commit_with_retry()
    return redirect(url_for("admin.wallet_withdrawals"))


@admin_bp.post("/wallet-withdrawals/<int:withdrawal_id>/reject")
@admin_required
def reject_wallet_withdrawal(withdrawal_id: int):
    withdrawal = db.session.get(WalletWithdrawal, withdrawal_id)
    if withdrawal is None:
        flash("Withdrawal request was not found.", "danger")
        return redirect(url_for("admin.wallet_withdrawals"))
    reason = request.form.get("reason", "").strip() or "Withdrawal rejected by admin."
    try:
        result = get_service("self_custody_wallet").reject_withdrawal(
            withdrawal,
            rejected_by_user_id=current_user().id if current_user() else None,
            reason=reason,
        )
    except RuntimeError as exc:
        db.session.rollback()
        flash(str(exc), "danger")
        return redirect(url_for("admin.wallet_withdrawals"))

    get_service("wallet_custody").release_failed_withdrawal(result)
    _update_withdrawal_transaction(result, "rejected", result.failure_reason or reason)
    commit_with_retry()
    flash("Withdrawal rejected and locked funds released.", "success")
    return redirect(url_for("admin.wallet_withdrawals"))


def _update_withdrawal_transaction(withdrawal: WalletWithdrawal, status: str, note: str) -> None:
    transaction = (
        WalletTransaction.query.filter(
            WalletTransaction.user_id == withdrawal.user_id,
            WalletTransaction.asset == withdrawal.asset,
            WalletTransaction.transaction_type == "withdrawal",
            WalletTransaction.note.like(f"%Withdrawal workflow {withdrawal.id}:%"),
        )
        .order_by(WalletTransaction.created_at.desc())
        .first()
    )
    if transaction is None:
        return
    transaction.status = status
    transaction.note = note


def _safe_next_url(next_url: str | None) -> str:
    fallback = url_for("dashboard.index")

    if not next_url:
        return fallback

    parsed = urlparse(next_url)

    if parsed.scheme or parsed.netloc or not next_url.startswith("/"):
        return fallback

    return next_url
