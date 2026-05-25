from __future__ import annotations

from datetime import datetime, timedelta
from urllib.parse import urlparse

import pyotp

from app.auth import encrypt_totp_secret, password_hash
from app.extensions import db
from app.models import AccountImpersonationGrant, AdminAuditLog, User

TOTP_SECRET = "JBSWY3DPEHPK3PXP"


def _user(username: str, *, role: str = "user", two_factor: bool = True) -> User:
    user = User(
        username=username,
        password_hash=password_hash("password123"),
        role=role,
        totp_secret_encrypted=encrypt_totp_secret(TOTP_SECRET) if two_factor else None,
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
