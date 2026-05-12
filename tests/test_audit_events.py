from __future__ import annotations

from datetime import datetime, timedelta

from app.auth import password_hash
from app.extensions import db
from app.models import AuditLog, Setting, User
from app.services.audit_events import AUDIT_EVENT_UI_LIMIT, get_audit_events_page


def test_audit_feed_limits_newest_50_without_deleting_operational_history(app) -> None:
    _insert_audits(55)

    retained = AuditLog.query.order_by(AuditLog.created_at.asc(), AuditLog.id.asc()).all()
    page = get_audit_events_page(1, page_size=AUDIT_EVENT_UI_LIMIT)

    assert len(retained) == 55
    assert page.total == AUDIT_EVENT_UI_LIMIT
    assert {event["action_label"] for event in page.events}
    assert {record.action for record in page.records} == {f"event_{index:02d}" for index in range(5, 55)}


def test_critical_audit_events_survive_ui_feed_limit(app) -> None:
    old_critical = AuditLog(
        category="panic",
        action="panic_lock_activated",
        message="Panic lock activated for operator review.",
        created_at=datetime.utcnow() - timedelta(days=2),
    )
    old_critical.details = {"critical": True}
    db.session.add(old_critical)
    db.session.commit()

    _insert_audits(55)

    assert AuditLog.query.filter_by(category="panic", action="panic_lock_activated").one_or_none() is not None
    page = get_audit_events_page(1, page_size=AUDIT_EVENT_UI_LIMIT)
    assert all(event["action_label"] != "Panic Lock Activated" for event in page.events)


def test_admin_risk_audit_events_use_server_seeded_infinite_loading(app) -> None:
    client = _admin_client(app)
    _insert_audits(12, prefix="page")

    page_one = client.get("/admin/risk")
    page_two = client.get("/admin/risk/audit-events?page=2")

    assert page_one.status_code == 200
    assert b"audit-feed" in page_one.data
    assert b"Load more" in page_one.data
    for index in range(7, 12):
        assert f"page-message-{index:02d}".encode() in page_one.data
    assert b"page-message-06" not in page_one.data

    assert page_two.status_code == 200
    payload = page_two.get_json()
    assert payload["page"] == 2
    assert payload["has_next"] is True
    for index in range(2, 7):
        assert f"page-message-{index:02d}" in payload["html"]
    assert "page-message-07" not in payload["html"]


def test_admin_risk_audit_page_clamps_invalid_and_large_values(app) -> None:
    client = _admin_client(app)
    _insert_audits(12, prefix="clamp")

    invalid = client.get("/admin/risk?audit_page=0")
    too_large = client.get("/admin/risk?audit_page=99")

    assert invalid.status_code == 200
    assert b"Load more" in invalid.data
    assert b"clamp-message-11" in invalid.data
    assert b"clamp-message-06" not in invalid.data

    assert too_large.status_code == 200
    assert b"clamp-message-01" in too_large.data
    assert b"clamp-message-00" in too_large.data
    assert b"clamp-message-02" not in too_large.data


def test_admin_risk_state_and_config_endpoints(app) -> None:
    client = _admin_client(app)

    state = client.get("/admin/risk/state")
    assert state.status_code == 200
    payload = state.get_json()
    assert payload["controls"]["profile"] == "balanced"
    assert payload["adaptive_slippage"]["model"] == "adaptive_ml"

    missing_confirmation = client.post(
        "/admin/risk/config",
        json={"daily_loss_limit_pct": 20, "daily_loss_unlimited": True, "max_leverage": 2, "profile": "aggressive"},
    )
    assert missing_confirmation.status_code == 400

    saved = client.post(
        "/admin/risk/config",
        json={
            "daily_loss_limit_pct": 20,
            "daily_loss_unlimited": True,
            "confirm_unlimited_loss": True,
            "max_leverage": 2,
            "profile": "aggressive",
        },
    )
    assert saved.status_code == 200
    assert saved.get_json()["controls"]["daily_loss_unlimited"] is True
    assert Setting.get_json("risk_controls")["profile"] == "aggressive"


def _insert_audits(count: int, *, prefix: str = "event") -> None:
    base = datetime.utcnow() - timedelta(minutes=count)
    for index in range(count):
        audit = AuditLog(
            category="strategy",
            action=f"event_{index:02d}",
            message=f"{prefix}-message-{index:02d}",
            created_at=base + timedelta(minutes=index),
        )
        audit.details = {
            "provider": "hyperliquid",
            "symbol": "BTC",
            "run_id": index,
        }
        db.session.add(audit)
    db.session.commit()


def _admin_client(app):
    admin = User(
        username="audit-admin",
        password_hash=password_hash("password123"),
        role="admin",
        totp_secret_encrypted="configured",
        two_factor_enabled_at=datetime.utcnow(),
    )
    db.session.add(admin)
    db.session.commit()

    client = app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = admin.id
        session["two_factor_verified"] = True
    return client
