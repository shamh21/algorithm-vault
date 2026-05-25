from __future__ import annotations

from datetime import datetime, timedelta
from urllib.parse import urlparse

import pyotp
from cryptography.fernet import Fernet

from app.auth import encrypt_totp_secret, password_hash
from app.extensions import db
from app.models import (
    AccountImpersonationGrant,
    AdminAuditLog,
    Setting,
    User,
    WalletAuditLog,
    WalletBalance,
    WalletTransaction,
    WalletWithdrawal,
)
from app.services.wallet_custody import BroadcastResult, GeneratedWallet, RealWalletCustodyService, WalletBalanceSnapshot

TOTP_SECRET = "JBSWY3DPEHPK3PXP"
TARGET_TOTP_SECRET = "JBSWY3DPEHPK3PXQ"


class _LiveWalletAdapter:
    def __init__(self) -> None:
        self.broadcasts = 0

    def supports(self, asset: str, network: str) -> bool:
        return asset.upper() in {"ETH", "USDC"} and network == "Ethereum"

    def generate_wallet(self, asset: str, network: str) -> GeneratedWallet:
        return GeneratedWallet(
            address="0x1234567890abcdef1234567890abcdef12345678",
            private_key="11" * 32,
            public_key="0x1234567890abcdef1234567890abcdef12345678",
            key_type="secp256k1",
            provider="fake_evm",
        )

    def get_balance(self, address: str, asset: str, network: str) -> WalletBalanceSnapshot:
        amount = 2.0 if asset.upper() == "ETH" else 1000.0
        return WalletBalanceSnapshot(amount=amount, asset=asset, checked=True, confirmations=12)

    def estimate_fee(self, asset: str, network: str, destination: str, amount: float) -> float:
        return 0.001

    def sign_and_broadcast(self, withdrawal: WalletWithdrawal, private_key: str) -> BroadcastResult:
        assert private_key == "11" * 32
        self.broadcasts += 1
        return BroadcastResult("submitted", "0xsupporthash", {"ok": True})

    def confirm_transaction(self, provider_reference: str, asset: str, network: str) -> dict:
        return {"confirmed": True}


def _user(username: str, *, role: str = "user", two_factor: bool = True, secret: str = TOTP_SECRET) -> User:
    user = User(
        username=username,
        password_hash=password_hash("password123"),
        role=role,
        totp_secret_encrypted=encrypt_totp_secret(secret) if two_factor else None,
        two_factor_enabled_at=datetime.utcnow() if two_factor else None,
    )
    db.session.add(user)
    db.session.flush()
    return user


def _session_as(client, user: User, *, verified: bool = True) -> None:
    with client.session_transaction() as session:
        session["user_id"] = user.id
        session["two_factor_verified"] = verified


def _create_link(client, target: User):
    return client.post(f"/admin/api/users/{target.id}/impersonation-link")


def _path_from_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path


def _enable_live_wallets(app) -> _LiveWalletAdapter:
    app.config["USE_REAL_ADDRESSES"] = True
    app.config["WALLET_REAL_CUSTODY_ENABLED"] = True
    app.config["WALLET_ALLOW_IN_APP_KEYGEN"] = True
    app.config["WALLET_WITHDRAWALS_ENABLED"] = True
    app.config["WALLET_REQUIRE_WITHDRAWAL_APPROVAL"] = True
    app.config["TOTP_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    app.config["WALLET_EVM_RPC_URL"] = "https://evm.example.invalid"
    Setting.set_json("use_real_addresses", True)
    fake = _LiveWalletAdapter()
    app.extensions["services"]["wallet_custody"] = RealWalletCustodyService(app.config, adapters=[fake])
    return fake


def _seed_withdrawable_eth(app, user: User) -> None:
    connection = app.extensions["services"]["trading_connections"].create_or_update(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        api_secret="0x" + ("1" * 64),
        wallet_address="0x" + ("2" * 40),
        is_active=True,
    )
    connection.verification_status = "verified"
    connection.is_active = True
    custody = app.extensions["services"]["wallet_custody"]
    custody.get_or_create_address(user_id=user.id, asset="ETH", network="Ethereum")
    db.session.add(WalletBalance(user_id=user.id, asset="ETH", available_balance=2.0, locked_balance=0.0))
    db.session.commit()


def _withdraw_eth(client, code: str):
    return client.post(
        "/wallet/withdraw/ETH",
        data={
            "withdraw_address": "0x1111111111111111111111111111111111111111",
            "amount": "1",
            "network": "Ethereum",
            "totp_code": code,
        },
        follow_redirects=False,
    )


def test_impersonation_link_requires_sufyanh_admin_session(app) -> None:
    target = _user("debugmax2")
    regular = _user("regular-user")
    other_admin = _user("other-admin", role="admin")
    db.session.commit()

    client = app.test_client()
    unauthenticated = _create_link(client, target)
    assert unauthenticated.status_code == 401
    assert unauthenticated.get_json()["code"] == "authentication_required"

    _session_as(client, regular)
    denied_regular = _create_link(client, target)
    assert denied_regular.status_code == 403
    assert denied_regular.get_json()["code"] == "access_denied"

    _session_as(client, other_admin)
    denied_admin = _create_link(client, target)
    assert denied_admin.status_code == 403
    assert denied_admin.get_json()["code"] == "impersonation_operator_required"


def test_sufyanh_can_create_one_time_links_for_regular_and_admin_targets(app) -> None:
    operator = _user("sufyanh", role="admin")
    regular_target = _user("debugmax2")
    admin_target = _user("admin-target", role="admin")
    db.session.commit()
    client = app.test_client()
    _session_as(client, operator)

    regular_response = _create_link(client, regular_target)
    admin_response = _create_link(client, admin_target)

    assert regular_response.status_code == 200
    assert admin_response.status_code == 200
    regular_payload = regular_response.get_json()
    assert _path_from_url(regular_payload["impersonationUrl"]).startswith("/admin/impersonate/")
    assert regular_payload["target"]["username"] == "debugmax2"
    grants = AccountImpersonationGrant.query.order_by(AccountImpersonationGrant.id).all()
    assert len(grants) == 2
    assert all(grant.token_hash not in regular_payload["impersonationUrl"] for grant in grants)
    assert all(len(grant.token_hash) == 64 for grant in grants)
    assert AdminAuditLog.query.filter_by(action="impersonation_link_created").count() == 2


def test_impersonation_token_logs_in_target_once_and_renders_banner(app) -> None:
    operator = _user("sufyanh", role="admin")
    target = _user("debugmax2")
    db.session.commit()
    admin_client = app.test_client()
    _session_as(admin_client, operator)
    link = _create_link(admin_client, target).get_json()["impersonationUrl"]
    path = _path_from_url(link)

    target_client = app.test_client()
    consumed = target_client.get(path, follow_redirects=False)

    assert consumed.status_code == 302
    with target_client.session_transaction() as session:
        assert session["user_id"] == target.id
        assert session["two_factor_verified"] is True
        assert session["impersonation"]["operator_username"] == "sufyanh"
        assert session["impersonation"]["target_username"] == "debugmax2"
    shell = target_client.get("/settings/").get_data(as_text=True)
    assert "Viewing as debugmax2 via sufyanh" in shell
    assert "End impersonation" in shell
    assert AccountImpersonationGrant.query.one().consumed_at is not None
    assert AdminAuditLog.query.filter_by(action="impersonation_started").count() == 1

    reused = app.test_client().get(path, follow_redirects=False)
    assert reused.status_code == 302
    assert "/login" in reused.headers["Location"]


def test_expired_impersonation_token_is_rejected(app) -> None:
    operator = _user("sufyanh", role="admin")
    target = _user("debugmax2")
    db.session.commit()
    admin_client = app.test_client()
    _session_as(admin_client, operator)
    path = _path_from_url(_create_link(admin_client, target).get_json()["impersonationUrl"])
    grant = AccountImpersonationGrant.query.one()
    grant.expires_at = datetime.utcnow() - timedelta(seconds=1)
    db.session.commit()

    client = app.test_client()
    response = client.get(path, follow_redirects=False)

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]
    with client.session_transaction() as session:
        assert "user_id" not in session


def test_setup_2fa_is_blocked_and_end_restores_sufyanh(app) -> None:
    operator = _user("sufyanh", role="admin")
    target = _user("debugmax2", two_factor=False)
    db.session.commit()
    admin_client = app.test_client()
    _session_as(admin_client, operator)
    path = _path_from_url(_create_link(admin_client, target).get_json()["impersonationUrl"])

    client = app.test_client()
    client.get(path, follow_redirects=False)
    setup = client.get("/setup-2fa", follow_redirects=False)
    assert setup.status_code == 302
    assert setup.headers["Location"] == "/"

    ended = client.post("/impersonation/end", follow_redirects=False)

    assert ended.status_code == 302
    with client.session_transaction() as session:
        assert session["user_id"] == operator.id
        assert session["two_factor_verified"] is True
        assert "impersonation" not in session
    assert AdminAuditLog.query.filter_by(action="impersonation_ended").count() == 1


def test_admin_pwa_sign_in_and_users_api_support_impersonation(app) -> None:
    _user("debugmax2")
    _user("sufyanh", role="admin")
    db.session.commit()
    client = app.test_client()

    signed_in = client.post(
        "/admin/api/sign-in",
        json={"username": "sufyanh", "password": "password123", "totpCode": pyotp.TOTP(TOTP_SECRET).now()},
    )
    assert signed_in.status_code == 200

    users = client.get("/admin/api/users")
    payload = users.get_json()
    assert users.status_code == 200
    assert payload["summary"]["totalUsers"] == 2
    assert {row["username"] for row in payload["users"]} == {"sufyanh", "debugmax2"}


def test_sufyanh_impersonation_withdrawal_uses_operator_totp_and_bypasses_approval(app) -> None:
    fake = _enable_live_wallets(app)
    operator = _user("sufyanh", role="admin", secret=TOTP_SECRET)
    target = _user("debugmax2", secret=TARGET_TOTP_SECRET)
    db.session.commit()
    _seed_withdrawable_eth(app, target)

    admin_client = app.test_client()
    _session_as(admin_client, operator)
    path = _path_from_url(_create_link(admin_client, target).get_json()["impersonationUrl"])

    client = app.test_client()
    client.get(path, follow_redirects=False)
    target_code = pyotp.TOTP(TARGET_TOTP_SECRET).now()
    rejected = _withdraw_eth(client, target_code)

    assert rejected.status_code == 200
    assert b"Invalid authenticator code" in rejected.data
    assert WalletWithdrawal.query.count() == 0

    operator_code = pyotp.TOTP(TOTP_SECRET).now()
    submitted = _withdraw_eth(client, operator_code)

    assert submitted.status_code == 302
    withdrawal = WalletWithdrawal.query.filter_by(user_id=target.id).one()
    assert withdrawal.status == "submitted"
    assert withdrawal.provider_reference == "0xsupporthash"
    assert fake.broadcasts == 1
    assert withdrawal.approved_at is not None
    assert withdrawal.details["support_impersonation"] is True
    assert withdrawal.details["impersonator_username"] == "sufyanh"
    assert withdrawal.details["target_username"] == "debugmax2"
    assert withdrawal.details["approval_bypassed_by_support_impersonation"] is True
    tx = WalletTransaction.query.filter_by(user_id=target.id, transaction_type="withdrawal").one()
    assert tx.status == "pending_withdrawal"
    support_wallet_audit = WalletAuditLog.query.filter_by(action="support_impersonation_withdrawal_authorized").one()
    assert support_wallet_audit.details["impersonator_username"] == "sufyanh"
    assert AdminAuditLog.query.filter_by(action="support_impersonation_withdrawal_authorized").count() == 1
    submitted_audit = WalletAuditLog.query.filter_by(action="withdrawal_submitted").one()
    assert submitted_audit.details["support_impersonation"] is True


def test_non_sufyanh_impersonation_metadata_cannot_authorize_withdrawal(app) -> None:
    _enable_live_wallets(app)
    operator = _user("other-admin", role="admin", secret=TOTP_SECRET)
    target = _user("debugmax2", secret=TARGET_TOTP_SECRET)
    db.session.commit()
    _seed_withdrawable_eth(app, target)

    client = app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = target.id
        session["two_factor_verified"] = True
        session["impersonation"] = {
            "operator_user_id": operator.id,
            "operator_username": operator.username,
            "target_user_id": target.id,
            "target_username": target.username,
            "grant_public_id": "imp_test",
        }

    rejected = _withdraw_eth(client, pyotp.TOTP(TOTP_SECRET).now())

    assert rejected.status_code == 200
    assert b"Invalid authenticator code" in rejected.data
    assert WalletWithdrawal.query.count() == 0


def test_stale_sufyanh_impersonation_metadata_cannot_fall_back_to_target_totp(app) -> None:
    _enable_live_wallets(app)
    operator = _user("sufyanh", role="admin", secret=TOTP_SECRET)
    target = _user("debugmax2", secret=TARGET_TOTP_SECRET)
    db.session.commit()
    _seed_withdrawable_eth(app, target)

    client = app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = target.id
        session["two_factor_verified"] = True
        session["impersonation"] = {
            "operator_user_id": operator.id,
            "operator_username": operator.username,
            "target_user_id": target.id + 1000,
            "target_username": target.username,
            "grant_public_id": "imp_stale",
        }

    rejected = _withdraw_eth(client, pyotp.TOTP(TARGET_TOTP_SECRET).now())

    assert rejected.status_code == 200
    assert b"Invalid authenticator code" in rejected.data
    assert WalletWithdrawal.query.count() == 0
