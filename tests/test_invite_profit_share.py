from __future__ import annotations

from datetime import datetime, timedelta

import pyotp
import pytest

from app.auth import encrypt_totp_secret, password_hash
from app.extensions import db
from app.models import (
    InviteCodeUsage,
    ProfitSharePayout,
    ReferralInviteCode,
    User,
    VaultCycle,
    VaultCycleSettlement,
    WalletBalance,
    WalletTransaction,
)
from app.services.vault_cycle_settlement import VaultCycleSettlementService

TOTP_SECRET = "JBSWY3DPEHPK3PXP"


class _AuditTransfer:
    def audit(self, *args, **kwargs) -> None:
        return None


def _user(username: str, *, role: str = "user") -> User:
    user = User(
        username=username,
        password_hash=password_hash("password123"),
        role=role,
        totp_secret_encrypted=encrypt_totp_secret(TOTP_SECRET),
        two_factor_enabled_at=datetime.utcnow(),
    )
    db.session.add(user)
    db.session.flush()
    return user


def _admin_client(app):
    admin = _user("invite-admin", role="admin")
    db.session.commit()
    client = app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = admin.id
        session["two_factor_verified"] = True
    return client


def _totp_code() -> str:
    return pyotp.TOTP(TOTP_SECRET).now()


def test_invite_admin_pwa_session_is_unauthenticated_safe_and_omits_data(app) -> None:
    client = app.test_client()

    response = client.get("/admin/api/session")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["authenticated"] is False
    assert payload["authorized"] is False
    assert payload["reason"] == "unauthenticated"
    assert payload["csrfToken"]
    assert "inviteCodes" not in payload
    assert "summary" not in payload


def test_invite_admin_pwa_rejects_unauthenticated_and_non_admin_api_access(app) -> None:
    _user("regular-user", role="user")
    db.session.commit()
    client = app.test_client()

    unauthenticated = client.get("/admin/api/invite-codes")
    assert unauthenticated.status_code == 401
    assert unauthenticated.get_json()["code"] == "authentication_required"

    signed_in = client.post(
        "/admin/api/sign-in",
        json={"username": "regular-user", "password": "password123", "totpCode": _totp_code()},
    )
    assert signed_in.status_code == 200
    payload = signed_in.get_json()
    assert payload["authenticated"] is True
    assert payload["authorized"] is False
    assert payload["reason"] == "access_denied"

    denied = client.get("/admin/api/invite-codes")
    assert denied.status_code == 403
    assert denied.get_json()["code"] == "access_denied"


def test_invite_admin_pwa_sign_in_sign_out_and_generic_failures(app) -> None:
    _user("sufyanh")
    _user("invite-admin-pwa", role="admin")
    db.session.commit()
    client = app.test_client()

    failed = client.post(
        "/admin/api/sign-in",
        json={"username": "invite-admin-pwa", "password": "wrong-password", "totpCode": _totp_code()},
    )
    assert failed.status_code == 401
    assert failed.get_json()["error"] == "Sign-in failed. Check your credentials and try again."

    invalid_totp = client.post(
        "/admin/api/sign-in",
        json={"username": "invite-admin-pwa", "password": "password123", "totpCode": "000000"},
    )
    assert invalid_totp.status_code == 401
    assert invalid_totp.get_json()["error"] == "Sign-in failed. Check your credentials and try again."

    signed_in = client.post(
        "/admin/api/sign-in",
        json={"username": "invite-admin-pwa", "password": "password123", "totpCode": _totp_code()},
    )
    assert signed_in.status_code == 200
    payload = signed_in.get_json()
    assert payload["authenticated"] is True
    assert payload["authorized"] is True
    assert payload["admin"] == {"username": "invite-admin-pwa", "role": "admin"}

    listed = client.get("/admin/api/invite-codes")
    assert listed.status_code == 200
    assert "inviteCodes" in listed.get_json()

    signed_out = client.post("/admin/api/sign-out")
    assert signed_out.status_code == 200
    assert signed_out.get_json()["authenticated"] is False

    blocked = client.get("/admin/api/invite-codes")
    assert blocked.status_code == 401


def test_admin_invite_code_api_create_search_update_and_disable(app) -> None:
    _user("sufyanh")
    db.session.commit()
    client = _admin_client(app)

    created = client.post(
        "/admin/api/invite-codes",
        json={
            "code": "sufyan20",
            "label": "Sufyan campaign",
            "profitSharePercent": 10,
            "profitShareWallet": "sufyanh",
            "maxUses": 5,
            "assignedRole": "premium",
            "profitShareActive": True,
            "isActive": True,
        },
    )

    assert created.status_code == 201
    payload = created.get_json()
    invite = payload["inviteCodes"][0]
    assert invite["code"] == "SUFYAN20"
    assert invite["profitShareWallet"] == "sufyanh"
    assert invite["profitSharePercent"] == 10

    duplicate = client.post("/admin/api/invite-codes", json={"code": "SUFYAN20", "profitSharePercent": 10, "profitShareWallet": "sufyanh"})
    assert duplicate.status_code == 409

    listed = client.get("/admin/api/invite-codes?search=sufyan&sort=created_desc")
    assert listed.status_code == 200
    assert listed.get_json()["summary"]["totalCodes"] == 1

    blocked = client.patch(
        f"/admin/api/invite-codes/{invite['publicId']}",
        json={"profitSharePercent": 12, "profitShareWallet": "sufyanh"},
    )
    assert blocked.status_code == 409
    assert blocked.get_json()["code"] == "confirmation_required"

    updated = client.patch(
        f"/admin/api/invite-codes/{invite['publicId']}",
        json={"profitSharePercent": 12, "profitShareWallet": "sufyanh", "confirmSensitiveChange": True},
    )
    assert updated.status_code == 200
    assert updated.get_json()["inviteCode"]["profitSharePercent"] == 12

    disabled = client.post(f"/admin/api/invite-codes/{invite['publicId']}/disable")
    assert disabled.status_code == 200
    assert disabled.get_json()["inviteCode"]["status"] == "disabled"


def test_signup_records_invite_usage_and_rejects_expired_codes(app) -> None:
    _user("sufyanh")
    expired = ReferralInviteCode(
        code="OLD20",
        label="Expired",
        profit_share_percent=10,
        percent_profit=10,
        profit_share_wallet="sufyanh",
        expires_at=datetime.utcnow() - timedelta(days=1),
    )
    valid = ReferralInviteCode(
        code="SUFYAN20",
        label="Valid",
        profit_share_percent=10,
        percent_profit=10,
        profit_share_wallet="sufyanh",
        assigned_role="premium",
    )
    db.session.add_all([expired, valid])
    db.session.commit()
    client = app.test_client()

    rejected = client.post(
        "/register",
        data={"username": "expired-user", "password": "password123", "confirm_password": "password123", "invite_code": "OLD20"},
    )
    assert rejected.status_code == 302
    assert User.query.filter_by(username="expired-user").one_or_none() is None

    accepted = client.post(
        "/register",
        data={"username": "new-user", "password": "password123", "confirm_password": "password123", "invite_code": "sufyan20"},
    )
    assert accepted.status_code == 302
    user = User.query.filter_by(username="new-user").one()
    assert user.role == "premium"
    assert user.referral_invite_code_id == valid.id
    assert InviteCodeUsage.query.filter_by(invitee_user_id=user.id, invite_code_id=valid.id).count() == 1
    assert db.session.get(ReferralInviteCode, valid.id).usage_count == 1


def test_settlement_profit_share_credits_sufyanh_without_touching_principal(app) -> None:
    invitee = _user("invitee")
    destination = _user("sufyanh")
    invite = ReferralInviteCode(code="SUFYAN20", profit_share_percent=10, percent_profit=10, profit_share_wallet="sufyanh", usage_count=1)
    db.session.add(invite)
    db.session.flush()
    invitee.referral_invite_code_id = invite.id
    cycle = VaultCycle(
        user_id=invitee.id,
        deposit_asset="USDT",
        deposit_amount=100.0,
        settlement_asset="USDT",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        starting_value_usd=100.0,
        current_estimated_value_usd=600.0,
        unlocks_at=datetime.utcnow(),
        algorithm_profile="VaultCycle",
    )
    settlement = VaultCycleSettlement(
        vault_cycle=cycle,
        user_id=invitee.id,
        settlement_asset="USDT",
        starting_value_usd=100.0,
        gross_pnl_usd=500.0,
        net_pnl_usd=500.0,
    )
    db.session.add_all([cycle, settlement, WalletBalance(user_id=invitee.id, asset="USDT", locked_balance=100.0)])
    db.session.flush()

    service = VaultCycleSettlementService(app.config, None, None, _AuditTransfer())
    service._credit_wallet(cycle, settlement, 600.0)
    db.session.commit()

    invitee_balance = WalletBalance.query.filter_by(user_id=invitee.id, asset="USDT").one()
    destination_balance = WalletBalance.query.filter_by(user_id=destination.id, asset="USDT").one()
    payout = ProfitSharePayout.query.filter_by(invite_code_id=invite.id, vault_cycle_id=cycle.id).one()
    assert payout.status == "completed"
    assert float(payout.source_profit_amount) == pytest.approx(500.0)
    assert float(payout.payout_amount) == pytest.approx(50.0)
    assert invitee_balance.available_balance == pytest.approx(550.0)
    assert invitee_balance.locked_balance == pytest.approx(0.0)
    assert destination_balance.available_balance == pytest.approx(50.0)
    assert WalletTransaction.query.filter_by(user_id=destination.id, transaction_type="invite_profit_share_credit").count() == 1

    app.extensions["services"]["invite_profit_share"].process_cycle(cycle, settlement, available_credit_amount=600.0)
    assert ProfitSharePayout.query.filter_by(invite_code_id=invite.id, vault_cycle_id=cycle.id).count() == 1


def test_profit_share_skips_zero_and_loss_cycles(app) -> None:
    invitee = _user("flat-invitee")
    _user("sufyanh")
    invite = ReferralInviteCode(code="FLAT", profit_share_percent=10, percent_profit=10, profit_share_wallet="sufyanh", usage_count=1)
    db.session.add(invite)
    db.session.flush()
    invitee.referral_invite_code_id = invite.id
    cycle = VaultCycle(
        user_id=invitee.id,
        deposit_asset="USDT",
        deposit_amount=100.0,
        settlement_asset="USDT",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        starting_value_usd=100.0,
        current_estimated_value_usd=95.0,
        unlocks_at=datetime.utcnow(),
        algorithm_profile="VaultCycle",
    )
    settlement = VaultCycleSettlement(
        vault_cycle=cycle,
        user_id=invitee.id,
        settlement_asset="USDT",
        starting_value_usd=100.0,
        gross_pnl_usd=-5.0,
        net_pnl_usd=-5.0,
        final_amount=95.0,
        status="complete",
    )
    db.session.add_all([cycle, settlement])
    db.session.flush()

    result = app.extensions["services"]["invite_profit_share"].process_cycle(cycle, settlement, available_credit_amount=95.0)

    assert result["applied"] is False
    assert result["reason"] == "non_positive_profit"
    assert ProfitSharePayout.query.count() == 0


def test_failed_wallet_credit_keeps_settlement_in_recovery(app) -> None:
    invitee = _user("missing-wallet-invitee")
    invite = ReferralInviteCode(code="MISSING", profit_share_percent=10, percent_profit=10, profit_share_wallet="missingwallet", usage_count=1)
    db.session.add(invite)
    db.session.flush()
    invitee.referral_invite_code_id = invite.id
    cycle = VaultCycle(
        user_id=invitee.id,
        deposit_asset="USDT",
        deposit_amount=100.0,
        settlement_asset="USDT",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        starting_value_usd=100.0,
        current_estimated_value_usd=600.0,
        unlocks_at=datetime.utcnow(),
        algorithm_profile="VaultCycle",
    )
    settlement = VaultCycleSettlement(
        vault_cycle=cycle,
        user_id=invitee.id,
        settlement_asset="USDT",
        starting_value_usd=100.0,
        gross_pnl_usd=500.0,
        net_pnl_usd=500.0,
    )
    db.session.add_all([cycle, settlement])
    db.session.flush()
    service = VaultCycleSettlementService(app.config, None, None, _AuditTransfer())

    service._credit_wallet(cycle, settlement, 600.0)
    service._credit_wallet(cycle, settlement, 600.0)

    payout = ProfitSharePayout.query.filter_by(invite_code_id=invite.id, vault_cycle_id=cycle.id).one()
    assert payout.status == "failed"
    assert "missingwallet" in payout.failed_reason
    assert settlement.status == "pending_recovery"
    assert cycle.execution_substatus == "settlement_pending_recovery"
    assert WalletTransaction.query.filter_by(user_id=invitee.id, transaction_type="settlement").count() == 0
