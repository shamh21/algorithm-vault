"""Admin authentication and advanced control routes."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from flask import Blueprint, Response, current_app, flash, jsonify, redirect, render_template, request, stream_with_context, url_for
from sqlalchemy import func, or_

from ..admin_auth import admin_required
from ..auth import current_user
from ..csrf import csrf_token
from ..extensions import db
from ..models import (
    AdminAuditLog,
    AuditLog,
    DepositAddress,
    InviteCodeUsage,
    LeveragedMarket,
    Order,
    PlatformTreasuryReserveJob,
    ProfitSharePayout,
    ReferralInviteCode,
    RiskEvent,
    ShadowLiveObservation,
    StrategyRanking,
    StrategyRun,
    StrategyValidation,
    User,
    VaultCycle,
    VaultCycleSettlement,
    WalletAddress,
    WalletAuditLog,
    WalletTransaction,
    WalletWithdrawal,
)
from ..runtime import get_current_mode, get_service
from ..services.audit_events import get_audit_events_page
from ..services.connection_health import latest_connection_health
from ..services.db_retry import commit_with_retry
from ..services.invite_profit_share import DEFAULT_PROFIT_SHARE_WALLET, InviteProfitShareError
from ..services.provider_assets import normalize_provider

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")
profit_share_api_bp = Blueprint("profit_share_api", __name__)


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
    user = current_user()
    trading_connections = get_service("trading_connections")
    connection = trading_connections.active_tradable_connection(user.id) if user is not None else None
    audit_page = get_audit_events_page(request.args.get("audit_page", 1))

    return render_template(
        "advanced/risk.html",
        mode=mode,
        risk_status=risk_engine.status(
            mode,
            user_id=user.id if user else None,
            trading_connection_id=connection.id if connection else None,
        ),
        risk_state=_risk_state_payload(mode, user),
        audit_events=audit_page.events,
        audit_pagination=audit_page,
        audits=audit_page.records,
    )


@admin_bp.get("/risk/state")
@admin_required
def risk_state():
    return jsonify(_risk_state_payload(get_current_mode(), current_user()))


@admin_bp.post("/risk/config")
@admin_required
def save_risk_config():
    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    payload = payload or {}
    if bool(payload.get("daily_loss_unlimited", False)) and not bool(payload.get("confirm_unlimited_loss", False)):
        return jsonify({"ok": False, "error": "Unlimited loss limit requires explicit confirmation."}), 400

    risk_engine = get_service("risk_engine")
    controls = risk_engine.save_risk_controls(payload)
    exchange_limits = _exchange_limits_payload(current_user(), controls)
    exchange_max = _safe_float(exchange_limits.get("max_exchange_leverage"), 0.0)
    requested_leverage = _safe_float(controls.get("max_leverage"), 0.0)
    if exchange_max > 0 and requested_leverage > exchange_max:
        db.session.rollback()
        return jsonify({"ok": False, "error": f"Max leverage cannot exceed connected exchange max of {exchange_max:g}x."}), 400

    audit = AuditLog(
        category="risk",
        action="risk_controls_saved",
        message="Risk engine controls updated.",
        user_id=current_user().id if current_user() else None,
    )
    audit.details = {"controls": controls}
    db.session.add(audit)
    commit_with_retry()
    return jsonify({"ok": True, "controls": controls, "state": _risk_state_payload(get_current_mode(), current_user())})


@admin_bp.get("/risk/audit-events")
@admin_required
def risk_audit_events():
    audit_page = get_audit_events_page(request.args.get("page", 1))
    html = render_template("advanced/_audit_event_cards.html", audit_events=audit_page.events)
    return jsonify(
        {
            "ok": True,
            "html": html,
            "page": audit_page.page,
            "has_next": audit_page.has_next,
            "next_page": audit_page.next_page,
            "total": audit_page.total,
        }
    )


@admin_bp.post("/risk/exchange-limits/refresh")
@admin_required
def refresh_risk_exchange_limits():
    user = current_user()
    if user is None:
        return jsonify({"ok": False, "error": "Authentication required."}), 401
    results = get_service("leveraged_markets").sync_for_user(user.id, mode="live", feature_scope="all", persist_features=False)
    commit_with_retry()
    return jsonify({"ok": True, "results": results, "state": _risk_state_payload(get_current_mode(), user)})


def _risk_state_payload(mode: str, user: User | None) -> dict[str, object]:
    risk_engine = get_service("risk_engine")
    trading_connections = get_service("trading_connections")
    connection = trading_connections.active_tradable_connection(user.id) if user is not None else None
    status = risk_engine.status(
        mode,
        user_id=user.id if user else None,
        trading_connection_id=connection.id if connection else None,
    )
    controls = risk_engine.risk_controls()
    exchange_limits = _exchange_limits_payload(user, controls)
    latency = _exchange_latency_payload(user)
    slippage = _adaptive_slippage_payload(risk_engine, exchange_limits, latency)
    safety = _safety_status(status)
    health_score = _risk_health_score(status, exchange_limits, latency, slippage)
    volatility_state = str(slippage.get("volatility_state") or "Calm")

    return {
        "mode": mode,
        "controls": controls,
        "status": {
            "panic_lock": bool(status.get("panic_lock", False)),
            "live_trading_blocked": bool(status.get("live_trading_blocked", False)),
            "daily_realized_pnl": _safe_float(status.get("daily_realized_pnl"), 0.0),
            "daily_loss_limit": _safe_float(status.get("daily_loss_limit"), 0.0),
            "daily_loss_unlimited": bool(status.get("daily_loss_unlimited", False)),
            "cooldown_minutes_remaining": int(status.get("cooldown_minutes_remaining", 0) or 0),
        },
        "health_score": health_score,
        "safety_engine_status": safety,
        "volatility_state": volatility_state,
        "exchange_limits": exchange_limits,
        "latency": latency,
        "adaptive_slippage": slippage,
        "profiles": [
            {"key": "conservative", "label": "Conservative"},
            {"key": "balanced", "label": "Balanced"},
            {"key": "aggressive", "label": "Aggressive"},
            {"key": "maximum-performance", "label": "Maximum Performance"},
        ],
    }


def _exchange_limits_payload(user: User | None, controls: dict[str, object]) -> dict[str, object]:
    fallback = max(0.0, _safe_float(controls.get("max_leverage"), 1.0))
    providers: list[str] = []
    connection_ids: list[int] = []
    if user is not None:
        try:
            connections = get_service("trading_connections").enabled_tradable_connections(user.id)
        except Exception:  # noqa: BLE001
            connections = []
        providers = [normalize_provider(connection.provider) for connection in connections]
        connection_ids = [int(connection.id) for connection in connections if connection.id is not None]

    query = LeveragedMarket.query.filter_by(status="active")
    if connection_ids:
        query = query.filter(LeveragedMarket.trading_connection_id.in_(connection_ids))
    elif providers:
        query = query.filter(LeveragedMarket.provider.in_(providers))
    markets = query.order_by(LeveragedMarket.provider.asc(), LeveragedMarket.max_leverage.desc()).limit(250).all()
    rows: dict[str, dict[str, object]] = {}
    for market in markets:
        provider = normalize_provider(market.provider)
        row = rows.setdefault(provider, {"provider": provider, "label": provider.title(), "market_count": 0, "max_leverage": 0.0})
        row["market_count"] = int(row["market_count"]) + 1
        row["max_leverage"] = max(_safe_float(row["max_leverage"]), _safe_float(market.max_leverage))
    if not rows and providers:
        for provider in providers:
            rows[provider] = {"provider": provider, "label": provider.title(), "market_count": 0, "max_leverage": fallback}
    max_exchange = max([_safe_float(row["max_leverage"]) for row in rows.values()] or [fallback])
    effective = min(max_exchange, fallback) if max_exchange > 0 and fallback > 0 else max(max_exchange, fallback)
    return {
        "max_exchange_leverage": max_exchange,
        "effective_max_leverage": effective,
        "configured_max_leverage": fallback,
        "providers": list(rows.values()),
    }


def _exchange_latency_payload(user: User | None) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    if user is not None:
        try:
            connections = get_service("trading_connections").enabled_tradable_connections(user.id)
        except Exception:  # noqa: BLE001
            connections = []
        for connection in connections:
            health = latest_connection_health(connection.id)
            provider = normalize_provider(connection.provider)
            order_latencies = _recent_order_latencies(provider)
            latency_ms = _safe_float(health.get("latency_ms"), 0.0) or (sum(order_latencies) / len(order_latencies) if order_latencies else 0.0)
            rows.append(
                {
                    "provider": provider,
                    "label": provider.title(),
                    "latency_ms": latency_ms,
                    "quality": _latency_quality(latency_ms, bool(health.get("can_trade", False))),
                    "can_trade": bool(health.get("can_trade", False)),
                    "last_checked_at": health.get("last_checked_at", ""),
                }
            )
    average = sum(_safe_float(row["latency_ms"]) for row in rows) / len(rows) if rows else 0.0
    return {"average_ms": average, "providers": rows}


def _adaptive_slippage_payload(risk_engine, exchange_limits: dict[str, object], latency: dict[str, object]) -> dict[str, object]:
    markets = LeveragedMarket.query.filter_by(status="active").order_by(LeveragedMarket.liquidity_usd.desc(), LeveragedMarket.spread_bps.asc()).limit(25).all()
    spread = 8.0
    liquidity = 0.0
    if markets:
        spread = sum(max(0.0, _safe_float(market.spread_bps)) for market in markets) / len(markets)
        liquidity = sum(max(0.0, _safe_float(market.liquidity_usd)) for market in markets) / len(markets)
    metadata = {
        "spread_bps": spread,
        "liquidity_usd": liquidity,
        "exchange_latency_ms": _safe_float(latency.get("average_ms"), 0.0),
        "max_exchange_leverage": _safe_float(exchange_limits.get("max_exchange_leverage"), 0.0),
    }
    payload = dict(risk_engine.adaptive_slippage_metrics(metadata))
    estimate = _safe_float(payload.get("estimate_bps"), 0.0)
    payload["micro_chart"] = [round(max(0.0, estimate * factor), 2) for factor in (0.72, 0.86, 0.78, 1.0, 0.94, 1.08, 0.98)]
    return payload


def _safety_status(status: dict[str, object]) -> dict[str, str]:
    if bool(status.get("panic_lock", False)) or bool(status.get("live_trading_blocked", False)):
        return {"label": "Restricted", "tone": "error"}
    if int(status.get("cooldown_minutes_remaining", 0) or 0) > 0:
        return {"label": "Protective Action", "tone": "warning"}
    if bool(status.get("daily_loss_unlimited", False)):
        return {"label": "Monitoring", "tone": "warning"}
    return {"label": "Active", "tone": "success"}


def _risk_health_score(
    status: dict[str, object],
    exchange_limits: dict[str, object],
    latency: dict[str, object],
    slippage: dict[str, object],
) -> int:
    score = 100.0
    if bool(status.get("panic_lock", False)):
        score -= 40.0
    if bool(status.get("live_trading_blocked", False)):
        score -= 35.0
    if bool(status.get("daily_loss_unlimited", False)):
        score -= 12.0
    if _safe_float(exchange_limits.get("max_exchange_leverage"), 0.0) <= 0:
        score -= 18.0
    score -= min(_safe_float(latency.get("average_ms"), 0.0) / 40.0, 18.0)
    score -= max(0.0, 100.0 - _safe_float(slippage.get("execution_health"), 70.0)) * 0.22
    return int(max(0.0, min(round(score), 100.0)))


def _recent_order_latencies(provider: str) -> list[float]:
    values: list[float] = []
    for order in Order.query.filter_by(mode="live").order_by(Order.created_at.desc(), Order.id.desc()).limit(50).all():
        details = order.details or {}
        if provider and normalize_provider(details.get("provider") or details.get("execution_venue")) != provider:
            continue
        value = _safe_float(details.get("exchange_latency_ms"), 0.0)
        if value > 0:
            values.append(value)
        if len(values) >= 8:
            break
    return values


def _latency_quality(latency_ms: float, can_trade: bool) -> str:
    if not can_trade:
        return "Offline"
    if latency_ms <= 0:
        return "Unknown"
    if latency_ms < 180:
        return "Excellent"
    if latency_ms < 450:
        return "Stable"
    if latency_ms < 900:
        return "Elevated"
    return "Degraded"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
                WalletWithdrawal.status.in_(["pending_approval", "pending_submission", "pending_gas_topup"])
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


@admin_bp.route("/invite-codes", methods=["GET", "POST"])
@admin_required
def invite_codes():
    if request.method == "POST":
        invite_id = request.form.get("invite_id", "").strip()
        code = request.form.get("code", "").strip().upper() or secrets.token_urlsafe(9).replace("-", "").replace("_", "")[:12].upper()
        label = request.form.get("label", "").strip()
        try:
            percent_profit = max(0.0, min(float(request.form.get("percent_profit", "50") or 50.0), 100.0))
        except ValueError:
            percent_profit = 50.0
        try:
            max_uses = max(0, int(float(request.form.get("max_uses", "0") or 0)))
        except ValueError:
            max_uses = 0
        is_active = request.form.get("is_active") == "on"
        existing = ReferralInviteCode.query.filter_by(code=code).one_or_none()
        if invite_id:
            invite = db.session.get(ReferralInviteCode, int(invite_id))
            if invite is None:
                flash("Invite code was not found.", "danger")
                return redirect(url_for("admin.invite_codes"))
            if existing is not None and existing.id != invite.id:
                flash("Invite code already exists.", "danger")
                return redirect(url_for("admin.invite_codes"))
        else:
            if existing is not None:
                flash("Invite code already exists.", "danger")
                return redirect(url_for("admin.invite_codes"))
            invite = ReferralInviteCode(code=code, created_by_user_id=current_user().id if current_user() else None)
            db.session.add(invite)
        invite.code = code
        invite.label = label
        invite.percent_profit = percent_profit
        invite.profit_share_percent = percent_profit
        invite.profit_share_wallet = invite.profit_share_wallet or DEFAULT_PROFIT_SHARE_WALLET
        invite.max_uses = max_uses
        if invite.is_active and not is_active:
            invite.disabled_at = datetime.utcnow()
        invite.is_active = is_active
        invite.details = {
            **invite.details,
            "last_updated_by_user_id": current_user().id if current_user() else None,
            "last_updated_at": datetime.utcnow().isoformat(),
        }
        db.session.add(
            AuditLog(
                category="referral",
                action="invite_code_saved",
                message=f"Invite code {code} saved with {percent_profit:.2f}% treasury profit share.",
                user_id=current_user().id if current_user() else None,
            )
        )
        commit_with_retry()
        flash("Invite code saved.", "success")
        return redirect(url_for("admin.invite_codes"))

    return render_template(
        "advanced/invite_codes.html",
        invite_codes=ReferralInviteCode.query.order_by(ReferralInviteCode.created_at.desc()).limit(150).all(),
        users={user.id: user for user in User.query.all()},
    )


@admin_bp.get("/api/session")
@admin_required
def invite_admin_session():
    user = current_user()
    return jsonify(
        {
            "ok": True,
            "csrfToken": csrf_token(),
            "admin": {"username": user.username if user else "", "role": user.role if user else ""},
            "defaults": {"profitShareWallet": DEFAULT_PROFIT_SHARE_WALLET},
        }
    )


@admin_bp.route("/api/invite-codes", methods=["GET", "POST"])
@admin_required
def invite_codes_api():
    if request.method == "POST":
        payload = _request_payload()
        batch_count = _safe_int(payload.get("batchCount") or payload.get("batch_count") or 1, 1)
        batch_count = max(1, min(batch_count, 100))
        try:
            normalized = _normalized_invite_payload(payload, require_code=batch_count == 1)
        except ValueError as exc:
            return _json_error(str(exc), 400)

        created: list[ReferralInviteCode] = []
        for index in range(batch_count):
            code = normalized["code"] if batch_count == 1 and normalized.get("code") else _generate_invite_code(payload.get("codePrefix") or payload.get("code"))
            if ReferralInviteCode.query.filter_by(code=code).one_or_none() is not None:
                if batch_count == 1:
                    return _json_error("Invite code must be unique.", 409, "duplicate_invite_code")
                code = _unique_generated_invite_code(payload.get("codePrefix") or payload.get("code"))
            invite = ReferralInviteCode(code=code, created_by_user_id=current_user().id if current_user() else None)
            db.session.add(invite)
            _apply_invite_payload(invite, {**normalized, "code": code}, creating=True)
            db.session.flush()
            _record_admin_audit("invite_code_created", "invite_code", invite.public_id, {}, _invite_snapshot(invite), {"batch_index": index})
            created.append(invite)
        commit_with_retry()
        return jsonify({"ok": True, "inviteCodes": [_invite_payload(invite) for invite in created]}), 201

    status_filter = str(request.args.get("status", "all") or "all").strip().lower()
    search = str(request.args.get("search", "") or "").strip()
    sort = str(request.args.get("sort", "created_desc") or "created_desc").strip()
    query = ReferralInviteCode.query
    if status_filter != "deleted":
        query = query.filter(ReferralInviteCode.deleted_at.is_(None))
    if search:
        creator_ids = [row.id for row in User.query.filter(User.username.ilike(f"%{search}%")).limit(100).all()]
        query = query.filter(
            or_(
                ReferralInviteCode.code.ilike(f"%{search}%"),
                ReferralInviteCode.label.ilike(f"%{search}%"),
                ReferralInviteCode.profit_share_wallet.ilike(f"%{search}%"),
                ReferralInviteCode.created_by_user_id.in_(creator_ids) if creator_ids else False,
            )
        )
    invites = query.order_by(ReferralInviteCode.created_at.desc()).limit(500).all()
    if status_filter != "all":
        invites = [invite for invite in invites if invite.lifecycle_status == status_filter]
    rows = [_invite_payload(invite) for invite in invites]
    rows = _sort_invite_payloads(rows, sort)
    return jsonify({"ok": True, "inviteCodes": rows, "summary": _invite_summary(rows)})


@admin_bp.get("/api/invite-codes/<public_id>")
@admin_required
def invite_code_detail_api(public_id: str):
    invite = _invite_by_public_id(public_id)
    if invite is None:
        return _json_error("Invite code was not found.", 404, "invite_code_not_found")
    return jsonify({"ok": True, "inviteCode": _invite_payload(invite, include_recent=True)})


@admin_bp.patch("/api/invite-codes/<public_id>")
@admin_required
def update_invite_code_api(public_id: str):
    invite = _invite_by_public_id(public_id)
    if invite is None:
        return _json_error("Invite code was not found.", 404, "invite_code_not_found")
    payload = _request_payload()
    old = _invite_snapshot(invite)
    try:
        normalized = _normalized_invite_payload(payload, require_code=False, current=invite)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    sensitive_changed = (
        "profitSharePercent" in payload
        or "profit_share_percent" in payload
        or "profitShareWallet" in payload
        or "profit_share_wallet" in payload
    ) and (
        float(normalized.get("profit_share_percent", invite.effective_profit_share_percent)) != float(invite.effective_profit_share_percent)
        or str(normalized.get("profit_share_wallet", invite.profit_share_wallet)).lower() != str(invite.profit_share_wallet).lower()
    )
    if sensitive_changed and not bool(payload.get("confirmSensitiveChange")):
        return _json_error(
            "Changing this profit-share rule applies to future completed Vault Cycles for all users who joined with this code.",
            409,
            "confirmation_required",
        )
    if normalized.get("code") and normalized["code"] != invite.code:
        existing = ReferralInviteCode.query.filter_by(code=normalized["code"]).one_or_none()
        if existing is not None and existing.id != invite.id:
            return _json_error("Invite code must be unique.", 409, "duplicate_invite_code")
    _apply_invite_payload(invite, normalized, creating=False)
    db.session.flush()
    new = _invite_snapshot(invite)
    _record_admin_audit(
        "invite_code_updated",
        "invite_code",
        invite.public_id,
        old,
        new,
        {"sensitive_changed": sensitive_changed, "confirmation_reason": payload.get("confirmationReason", "")},
    )
    commit_with_retry()
    return jsonify({"ok": True, "inviteCode": _invite_payload(invite)})


@admin_bp.post("/api/invite-codes/<public_id>/disable")
@admin_required
def disable_invite_code_api(public_id: str):
    invite = _invite_by_public_id(public_id)
    if invite is None:
        return _json_error("Invite code was not found.", 404, "invite_code_not_found")
    old = _invite_snapshot(invite)
    invite.is_active = False
    invite.disabled_at = invite.disabled_at or datetime.utcnow()
    db.session.flush()
    _record_admin_audit("invite_code_disabled", "invite_code", invite.public_id, old, _invite_snapshot(invite), _request_payload())
    commit_with_retry()
    return jsonify({"ok": True, "inviteCode": _invite_payload(invite)})


@admin_bp.delete("/api/invite-codes/<public_id>")
@admin_required
def delete_invite_code_api(public_id: str):
    invite = _invite_by_public_id(public_id)
    if invite is None:
        return _json_error("Invite code was not found.", 404, "invite_code_not_found")
    old = _invite_snapshot(invite)
    invite.deleted_at = invite.deleted_at or datetime.utcnow()
    invite.is_active = False
    invite.disabled_at = invite.disabled_at or datetime.utcnow()
    db.session.flush()
    _record_admin_audit("invite_code_deleted", "invite_code", invite.public_id, old, _invite_snapshot(invite), _request_payload())
    commit_with_retry()
    return jsonify({"ok": True, "inviteCode": _invite_payload(invite)})


@admin_bp.get("/api/invite-codes/<public_id>/usages")
@admin_required
def invite_code_usages_api(public_id: str):
    invite = _invite_by_public_id(public_id)
    if invite is None:
        return _json_error("Invite code was not found.", 404, "invite_code_not_found")
    usages = InviteCodeUsage.query.filter_by(invite_code_id=invite.id).order_by(InviteCodeUsage.used_at.desc()).limit(250).all()
    return jsonify({"ok": True, "usages": [_usage_payload(usage) for usage in usages]})


@admin_bp.get("/api/invite-codes/<public_id>/profit-share-payouts")
@admin_required
def invite_code_profit_share_payouts_api(public_id: str):
    invite = _invite_by_public_id(public_id)
    if invite is None:
        return _json_error("Invite code was not found.", 404, "invite_code_not_found")
    payouts = ProfitSharePayout.query.filter_by(invite_code_id=invite.id).order_by(ProfitSharePayout.created_at.desc()).limit(250).all()
    return jsonify({"ok": True, "payouts": [_payout_payload(payout) for payout in payouts]})


@admin_bp.get("/api/profit-share-payouts")
@admin_required
def profit_share_payouts_api():
    status_filter = str(request.args.get("status", "all") or "all").strip().lower()
    query = ProfitSharePayout.query
    if status_filter != "all":
        query = query.filter_by(status=status_filter)
    payouts = query.order_by(ProfitSharePayout.created_at.desc()).limit(500).all()
    return jsonify({"ok": True, "payouts": [_payout_payload(payout) for payout in payouts]})


@admin_bp.get("/api/audit-logs")
@admin_required
def admin_audit_logs_api():
    entity_public_id = str(request.args.get("entityPublicId", "") or "").strip()
    query = AdminAuditLog.query
    if entity_public_id:
        query = query.filter_by(entity_public_id=entity_public_id)
    logs = query.order_by(AdminAuditLog.created_at.desc()).limit(500).all()
    return jsonify({"ok": True, "auditLogs": [_admin_audit_payload(log) for log in logs]})


@profit_share_api_bp.post("/api/vault-cycles/<public_id>/process-profit-share")
@admin_required
def process_vault_cycle_profit_share_api(public_id: str):
    cycle = VaultCycle.query.filter_by(public_id=public_id).one_or_none()
    if cycle is None:
        return _json_error("Vault Cycle was not found.", 404, "vault_cycle_not_found")
    settlement = VaultCycleSettlement.query.filter_by(vault_cycle_id=cycle.id).one_or_none()
    if settlement is None or settlement.status != "complete":
        return _json_error("Vault Cycle profit share can only be processed after completed settlement.", 409, "vault_cycle_not_complete")
    try:
        payload = get_service("invite_profit_share").process_cycle(
            cycle,
            settlement,
            available_credit_amount=cycle.final_settlement_amount or settlement.final_amount,
            debit_invitee_wallet=True,
        )
        if payload.get("applied"):
            settlement.details = {**settlement.details, "invite_profit_share": payload}
        commit_with_retry()
    except InviteProfitShareError as exc:
        db.session.rollback()
        return _json_error(str(exc), 409, "profit_share_failed")
    return jsonify({"ok": True, "profitShare": payload})


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
    elif result.status == "pending_gas_topup":
        _update_withdrawal_transaction(result, "pending_withdrawal", f"Withdrawal workflow {result.id}: pending_gas_topup.")
        flash("Withdrawal approved and waiting for platform treasury gas top-up.", "info")
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


@admin_bp.get("/platform-treasury")
@admin_required
def platform_treasury():
    status = get_service("platform_treasury").status()
    pending_gas_topups = (
        WalletWithdrawal.query.filter(WalletWithdrawal.status.in_(["pending_gas_topup", "queued_treasury_solvency"]))
        .order_by(WalletWithdrawal.created_at.desc())
        .limit(50)
        .all()
    )
    return render_template(
        "advanced/platform_treasury.html",
        treasury=status,
        pending_gas_topups=pending_gas_topups,
        reserve_jobs=PlatformTreasuryReserveJob.query.order_by(PlatformTreasuryReserveJob.created_at.desc()).limit(150).all(),
        panic_lock=bool(get_service("risk_engine").status("live").get("panic_lock", False)),
    )


@admin_bp.get("/api/platform-treasury/solvency")
@admin_required
def platform_treasury_solvency_api():
    network = request.args.get("network", "Ethereum").strip() or "Ethereum"
    recalculate = request.args.get("recalculate", "").lower() in {"1", "true", "yes"}
    return jsonify(get_service("treasury_solvency").solvency_payload(network=network, recalculate=recalculate))


@admin_bp.get("/api/platform-treasury/stream")
@admin_required
def platform_treasury_stream():
    network = request.args.get("network", "Ethereum").strip() or "Ethereum"
    once = request.args.get("once", "").lower() in {"1", "true", "yes"}
    interval = float(request.args.get("interval", current_app.config.get("TREASURY_SOLVENCY_RECALC_INTERVAL_SECONDS", 30.0)) or 30.0)
    solvency = get_service("treasury_solvency")
    events = solvency.event_stream(lambda: solvency.solvency_payload(network=network, recalculate=True), once=once, interval=interval)
    return Response(
        stream_with_context(events),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-store, no-cache, no-transform, must-revalidate, max-age=0", "X-Accel-Buffering": "no"},
    )


@admin_bp.post("/platform-treasury/rebalance")
@admin_required
def rebalance_platform_treasury():
    network = request.form.get("network", "Ethereum").strip() or "Ethereum"
    force = request.form.get("force", "").lower() in {"1", "true", "yes", "on"}
    try:
        result = get_service("treasury_solvency").rebalance_if_needed(network=network, force=force)
        commit_with_retry()
        flash("Treasury reserve rebalance queued." if result.get("created") else f"Treasury rebalance: {result.get('status', 'not needed')}.", "success")
    except RuntimeError as exc:
        db.session.rollback()
        flash(str(exc), "danger")
    return redirect(url_for("admin.platform_treasury"))


@admin_bp.post("/platform-treasury/create")
@admin_required
def create_platform_treasury():
    if request.form.get("confirm", "").strip() != "CREATE-PLATFORM-TREASURY":
        flash("Enter CREATE-PLATFORM-TREASURY to create the treasury wallet.", "danger")
        return redirect(url_for("admin.platform_treasury"))
    try:
        get_service("platform_treasury").create_wallet(created_by_user_id=current_user().id if current_user() else None)
        commit_with_retry()
        flash("Platform treasury wallet created.", "success")
    except RuntimeError as exc:
        db.session.rollback()
        flash(str(exc), "danger")
    return redirect(url_for("admin.platform_treasury"))


@admin_bp.post("/platform-treasury/rotate")
@admin_required
def rotate_platform_treasury():
    if request.form.get("confirm", "").strip() != "ROTATE-PLATFORM-TREASURY":
        flash("Enter ROTATE-PLATFORM-TREASURY to rotate the treasury wallet.", "danger")
        return redirect(url_for("admin.platform_treasury"))
    try:
        get_service("platform_treasury").rotate_wallet(created_by_user_id=current_user().id if current_user() else None)
        commit_with_retry()
        flash("Platform treasury wallet rotated.", "success")
    except RuntimeError as exc:
        db.session.rollback()
        flash(str(exc), "danger")
    return redirect(url_for("admin.platform_treasury"))


@admin_bp.post("/platform-treasury/pause")
@admin_required
def pause_platform_treasury():
    get_service("platform_treasury").set_paused(True, user_id=current_user().id if current_user() else None)
    commit_with_retry()
    flash("Platform treasury paused.", "warning")
    return redirect(url_for("admin.platform_treasury"))


@admin_bp.post("/platform-treasury/resume")
@admin_required
def resume_platform_treasury():
    get_service("platform_treasury").set_paused(False, user_id=current_user().id if current_user() else None)
    commit_with_retry()
    flash("Platform treasury resumed.", "success")
    return redirect(url_for("admin.platform_treasury"))


@admin_bp.post("/platform-treasury/process-queue")
@admin_required
def process_platform_treasury_queue():
    treasury = get_service("platform_treasury")
    result = treasury.process_solvency_cycle()
    commit_with_retry()
    flash(f"Processed {result['reserve_job_count']} reserve job(s) and {result['withdrawal_count']} withdrawal queue item(s).", "success")
    return redirect(url_for("admin.platform_treasury"))


@admin_bp.post("/platform-treasury/jobs/<int:job_id>/retry")
@admin_required
def retry_platform_treasury_job(job_id: int):
    try:
        get_service("platform_treasury").retry_reserve_job(job_id, user_id=current_user().id if current_user() else None)
        commit_with_retry()
        flash("Treasury reserve job retried.", "success")
    except RuntimeError as exc:
        db.session.rollback()
        flash(str(exc), "danger")
    return redirect(url_for("admin.platform_treasury"))


@admin_bp.post("/platform-treasury/top-up-withdrawal/<int:withdrawal_id>")
@admin_required
def top_up_withdrawal_gas(withdrawal_id: int):
    if request.form.get("confirm", "").strip() != "TOP-UP-WITHDRAWAL-GAS":
        flash("Enter TOP-UP-WITHDRAWAL-GAS to fund withdrawal gas.", "danger")
        return redirect(url_for("admin.platform_treasury"))
    withdrawal = db.session.get(WalletWithdrawal, withdrawal_id)
    if withdrawal is None:
        flash("Withdrawal request was not found.", "danger")
        return redirect(url_for("admin.platform_treasury"))
    try:
        result = get_service("platform_treasury").top_up_withdrawal_gas(withdrawal)
        commit_with_retry()
        if result.get("status") == "queued_treasury_solvency":
            flash("Withdrawal remains queued until treasury solvency recovers.", "warning")
        elif result.get("status") == "pending_approval":
            flash("Withdrawal is safe again and awaiting admin approval.", "info")
        else:
            flash("Treasury gas top-up submitted.", "success")
    except RuntimeError as exc:
        db.session.rollback()
        flash(str(exc), "danger")
    return redirect(url_for("admin.platform_treasury"))


def _request_payload() -> dict[str, Any]:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        return payload if isinstance(payload, dict) else {}
    return request.form.to_dict()


def _json_error(message: str, status: int = 400, code: str = "invalid_request"):
    return jsonify({"ok": False, "error": message, "code": code}), status


def _clean_invite_code(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip().upper() if ch.isalnum() or ch in {"-", "_"})


def _generate_invite_code(prefix: Any = "") -> str:
    clean_prefix = _clean_invite_code(prefix)[:8]
    token = secrets.token_urlsafe(8).replace("-", "").replace("_", "")[:10].upper()
    return f"{clean_prefix}{token}"[:24] if clean_prefix else token[:12]


def _unique_generated_invite_code(prefix: Any = "") -> str:
    for _ in range(20):
        code = _generate_invite_code(prefix)
        if ReferralInviteCode.query.filter_by(code=code).one_or_none() is None:
            return code
    raise RuntimeError("Unable to generate a unique invite code.")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_datetime(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid date value: {raw}") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)  # noqa: UP017
    return parsed


def _normalized_invite_payload(
    payload: dict[str, Any],
    *,
    require_code: bool,
    current: ReferralInviteCode | None = None,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if "code" in payload or require_code:
        code = _clean_invite_code(payload.get("code"))
        if require_code and not code:
            code = _generate_invite_code()
        if code and len(code) < 3:
            raise ValueError("Invite code must be at least 3 characters.")
        normalized["code"] = code
    if "label" in payload:
        normalized["label"] = str(payload.get("label") or "").strip()[:120]
    if "expirationDate" in payload or "expiresAt" in payload or "expires_at" in payload:
        expires_at = _parse_datetime(payload.get("expirationDate") or payload.get("expiresAt") or payload.get("expires_at"))
        if expires_at is not None and expires_at <= datetime.utcnow():
            raise ValueError("Expiration date cannot be in the past.")
        normalized["expires_at"] = expires_at
    if "maxUses" in payload or "max_uses" in payload:
        raw_max_uses = payload.get("maxUses", payload.get("max_uses"))
        if raw_max_uses in {None, ""}:
            normalized["max_uses"] = 0
        else:
            max_uses = _safe_int(raw_max_uses, -1)
            if max_uses <= 0:
                raise ValueError("Max uses must be a positive integer when provided.")
            normalized["max_uses"] = max_uses
    if "assignedRole" in payload or "assigned_role" in payload:
        role = str(payload.get("assignedRole", payload.get("assigned_role", "user")) or "user").strip().lower()
        if role == "admin":
            raise ValueError("Invite codes cannot assign admin access.")
        normalized["assigned_role"] = role or "user"
    if "profitSharePercent" in payload or "profit_share_percent" in payload or "percent_profit" in payload:
        percent = _safe_float(payload.get("profitSharePercent", payload.get("profit_share_percent", payload.get("percent_profit"))), -1.0)
        if percent < 0 or percent > 100:
            raise ValueError("Profit-share percentage must be between 0 and 100.")
        normalized["profit_share_percent"] = percent
    if "profitShareWallet" in payload or "profit_share_wallet" in payload:
        wallet = str(payload.get("profitShareWallet", payload.get("profit_share_wallet", DEFAULT_PROFIT_SHARE_WALLET)) or "").strip().lower()
        normalized["profit_share_wallet"] = wallet or DEFAULT_PROFIT_SHARE_WALLET
    if "profitShareStartsAt" in payload or "profit_share_starts_at" in payload:
        normalized["profit_share_starts_at"] = _parse_datetime(payload.get("profitShareStartsAt") or payload.get("profit_share_starts_at"))
    if "profitShareEndsAt" in payload or "profit_share_ends_at" in payload:
        normalized["profit_share_ends_at"] = _parse_datetime(payload.get("profitShareEndsAt") or payload.get("profit_share_ends_at"))
    if "profitShareActive" in payload or "profit_share_active" in payload:
        normalized["profit_share_active"] = bool(payload.get("profitShareActive", payload.get("profit_share_active")))
    if "appliesToVaultTypes" in payload or "applies_to_vault_types" in payload:
        vault_types = payload.get("appliesToVaultTypes", payload.get("applies_to_vault_types")) or []
        if isinstance(vault_types, str):
            vault_types = [item.strip() for item in vault_types.split(",")]
        normalized["applies_to_vault_types"] = [str(item).strip() for item in vault_types if str(item).strip()]
    if "isActive" in payload or "is_active" in payload:
        normalized["is_active"] = bool(payload.get("isActive", payload.get("is_active")))

    percent = normalized.get("profit_share_percent", current.effective_profit_share_percent if current is not None else 0.0)
    wallet = normalized.get("profit_share_wallet", current.profit_share_wallet if current is not None else DEFAULT_PROFIT_SHARE_WALLET)
    if float(percent or 0.0) > 0:
        if not wallet:
            raise ValueError("Profit-share wallet is required when percentage is greater than 0.")
        if User.query.filter_by(username=str(wallet).lower()).one_or_none() is None:
            raise ValueError(f"Destination wallet user '{wallet}' was not found.")
    starts_at = normalized.get("profit_share_starts_at", current.profit_share_starts_at if current is not None else None)
    ends_at = normalized.get("profit_share_ends_at", current.profit_share_ends_at if current is not None else None)
    if starts_at is not None and ends_at is not None and ends_at <= starts_at:
        raise ValueError("Profit-share end date must be after the start date.")
    return normalized


def _apply_invite_payload(invite: ReferralInviteCode, payload: dict[str, Any], *, creating: bool) -> None:
    if payload.get("code"):
        invite.code = payload["code"]
    if "label" in payload:
        invite.label = payload["label"]
    elif creating:
        invite.label = ""
    if "expires_at" in payload:
        invite.expires_at = payload["expires_at"]
    if "max_uses" in payload:
        invite.max_uses = int(payload["max_uses"] or 0)
    elif creating:
        invite.max_uses = 0
    if "assigned_role" in payload:
        invite.assigned_role = payload["assigned_role"]
    elif creating:
        invite.assigned_role = "user"
    if "profit_share_percent" in payload:
        invite.profit_share_percent = float(payload["profit_share_percent"])
        invite.percent_profit = float(payload["profit_share_percent"])
    elif creating:
        invite.profit_share_percent = 0.0
        invite.percent_profit = 0.0
    if "profit_share_wallet" in payload:
        invite.profit_share_wallet = payload["profit_share_wallet"]
    elif creating:
        invite.profit_share_wallet = DEFAULT_PROFIT_SHARE_WALLET
    if "profit_share_starts_at" in payload:
        invite.profit_share_starts_at = payload["profit_share_starts_at"]
    if "profit_share_ends_at" in payload:
        invite.profit_share_ends_at = payload["profit_share_ends_at"]
    if "profit_share_active" in payload:
        invite.profit_share_active = bool(payload["profit_share_active"])
    elif creating:
        invite.profit_share_active = True
    if "applies_to_vault_types" in payload:
        invite.applies_to_vault_types = payload["applies_to_vault_types"]
    elif creating:
        invite.applies_to_vault_types = []
    if "is_active" in payload:
        if invite.is_active and not bool(payload["is_active"]):
            invite.disabled_at = datetime.utcnow()
        invite.is_active = bool(payload["is_active"])
    elif creating:
        invite.is_active = True
    invite.details = {
        **invite.details,
        "last_updated_by_user_id": current_user().id if current_user() else None,
        "last_updated_at": datetime.utcnow().isoformat(),
    }


def _invite_by_public_id(public_id: str) -> ReferralInviteCode | None:
    return ReferralInviteCode.query.filter_by(public_id=str(public_id or "").strip()).one_or_none()


def _invite_snapshot(invite: ReferralInviteCode) -> dict[str, Any]:
    return {
        "publicId": invite.public_id,
        "code": invite.code,
        "label": invite.label,
        "expiresAt": _iso(invite.expires_at),
        "maxUses": int(invite.max_uses or 0),
        "currentUses": int(invite.usage_count or 0),
        "status": invite.lifecycle_status,
        "assignedRole": invite.assigned_role,
        "profitSharePercent": invite.effective_profit_share_percent,
        "profitShareWallet": invite.profit_share_wallet,
        "profitShareStartsAt": _iso(invite.profit_share_starts_at),
        "profitShareEndsAt": _iso(invite.profit_share_ends_at),
        "profitShareActive": bool(invite.profit_share_active),
        "appliesToVaultTypes": invite.applies_to_vault_types,
        "isActive": bool(invite.is_active),
    }


def _invite_payload(invite: ReferralInviteCode, *, include_recent: bool = False) -> dict[str, Any]:
    creator = db.session.get(User, invite.created_by_user_id) if invite.created_by_user_id else None
    usage_count = InviteCodeUsage.query.filter_by(invite_code_id=invite.id).count() or int(invite.usage_count or 0)
    total_profit = _sum_decimal(
        db.session.query(func.coalesce(func.sum(ProfitSharePayout.source_profit_amount), 0)).filter_by(invite_code_id=invite.id).scalar()
    )
    total_payout = _sum_decimal(
        db.session.query(func.coalesce(func.sum(ProfitSharePayout.payout_amount), 0))
        .filter_by(invite_code_id=invite.id, status="completed")
        .scalar()
    )
    payout_counts = {
        status: ProfitSharePayout.query.filter_by(invite_code_id=invite.id, status=status).count()
        for status in ("pending", "completed", "failed", "retryable")
    }
    payload = {
        **_invite_snapshot(invite),
        "createdBy": creator.username if creator else "",
        "createdAt": _iso(invite.created_at),
        "updatedAt": _iso(invite.updated_at),
        "disabledAt": _iso(invite.disabled_at),
        "deletedAt": _iso(invite.deleted_at),
        "currentUses": usage_count,
        "totalInviteeProfit": total_profit,
        "totalPaidToWallet": total_payout,
        "payoutCounts": payout_counts,
    }
    if include_recent:
        payload["recentUsages"] = [_usage_payload(row) for row in InviteCodeUsage.query.filter_by(invite_code_id=invite.id).order_by(InviteCodeUsage.used_at.desc()).limit(10).all()]
        payload["recentPayouts"] = [_payout_payload(row) for row in ProfitSharePayout.query.filter_by(invite_code_id=invite.id).order_by(ProfitSharePayout.created_at.desc()).limit(10).all()]
    return payload


def _usage_payload(usage: InviteCodeUsage) -> dict[str, Any]:
    return {
        "publicId": usage.public_id,
        "invitee": usage.invitee_user.username if usage.invitee_user else "",
        "usedAt": _iso(usage.used_at),
        "status": usage.status,
        "acceptedDisclosureVersion": usage.accepted_disclosure_version,
    }


def _payout_payload(payout: ProfitSharePayout) -> dict[str, Any]:
    return {
        "publicId": payout.public_id,
        "inviteCodePublicId": payout.invite_code.public_id if payout.invite_code else "",
        "inviteCode": payout.invite_code.code if payout.invite_code else "",
        "invitee": payout.invitee_user.username if payout.invitee_user else "",
        "vaultCyclePublicId": payout.vault_cycle.public_id if payout.vault_cycle else "",
        "sourceProfitAmount": float(payout.source_profit_amount or 0),
        "profitSharePercent": float(payout.profit_share_percent or 0),
        "payoutAmount": float(payout.payout_amount or 0),
        "asset": payout.asset,
        "destinationWallet": payout.destination_wallet,
        "status": payout.status,
        "idempotencyKey": payout.idempotency_key,
        "createdAt": _iso(payout.created_at),
        "completedAt": _iso(payout.completed_at),
        "failedReason": payout.failed_reason or "",
    }


def _admin_audit_payload(log: AdminAuditLog) -> dict[str, Any]:
    return {
        "publicId": log.public_id,
        "admin": log.admin_user.username if log.admin_user else "",
        "action": log.action,
        "entityType": log.entity_type,
        "entityPublicId": log.entity_public_id,
        "oldValue": log.old_value,
        "newValue": log.new_value,
        "ipAddress": log.ip_address,
        "createdAt": _iso(log.created_at),
        "metadata": log.details,
    }


def _record_admin_audit(
    action: str,
    entity_type: str,
    entity_public_id: str,
    old_value: dict[str, Any],
    new_value: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    user = current_user()
    log = AdminAuditLog(
        admin_user_id=user.id if user else None,
        action=action,
        entity_type=entity_type,
        entity_public_id=entity_public_id,
        ip_address=_request_ip(),
        user_agent=str(request.headers.get("User-Agent", ""))[:500],
    )
    log.old_value = old_value
    log.new_value = new_value
    log.details = metadata or {}
    db.session.add(log)


def _request_ip() -> str:
    forwarded = str(request.headers.get("X-Forwarded-For", "") or "").split(",", maxsplit=1)[0].strip()
    return forwarded or str(request.remote_addr or "")


def _sort_invite_payloads(rows: list[dict[str, Any]], sort: str) -> list[dict[str, Any]]:
    reverse = sort.endswith("_desc")
    key = sort.removesuffix("_desc").removesuffix("_asc")
    field_map = {
        "created": "createdAt",
        "creation": "createdAt",
        "expiration": "expiresAt",
        "expires": "expiresAt",
        "usage": "currentUses",
        "uses": "currentUses",
        "profit": "totalInviteeProfit",
        "payout": "totalPaidToWallet",
    }
    field = field_map.get(key, "createdAt")
    return sorted(rows, key=lambda row: row.get(field) or "", reverse=reverse)


def _invite_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "totalCodes": len(rows),
        "activeCodes": sum(1 for row in rows if row.get("status") == "active"),
        "totalUses": sum(int(row.get("currentUses") or 0) for row in rows),
        "totalInviteeProfit": sum(float(row.get("totalInviteeProfit") or 0.0) for row in rows),
        "totalPaidToWallet": sum(float(row.get("totalPaidToWallet") or 0.0) for row in rows),
        "failedPayouts": sum(int((row.get("payoutCounts") or {}).get("failed") or 0) for row in rows),
    }


def _sum_decimal(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.replace(microsecond=0).isoformat() + "Z"


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
