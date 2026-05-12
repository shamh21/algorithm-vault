from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app.auth import encrypt_totp_secret, password_hash
from app.extensions import db
from app.models import LeveragedMarket, Setting, TradingConnection, User, VaultCycle, WalletBalance
from app.routes.consumer import _persist_cycle_start_cycle_idempotency
from app.services.hyperliquid_client import ClientSnapshot
from app.services.vault_readiness import get_vault_cycle_readiness


def _user(username: str = "vaultready") -> User:
    user = User(username=username, password_hash=password_hash("password123"), role="user")
    user.totp_secret_encrypted = encrypt_totp_secret("JBSWY3DPEHPK3PXP")
    user.two_factor_enabled_at = datetime.utcnow()
    db.session.add(user)
    db.session.commit()
    return user


def _login(client, user: User) -> None:
    with client.session_transaction() as session:
        session["user_id"] = user.id
        session["two_factor_verified"] = True


def _confirm_live(app) -> None:
    app.config["EXPLICIT_LIVE_CONFIRMED"] = True
    app.config["SECONDARY_CONFIRMATION"] = True
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = False
    app.config["ONE_H10_BOOTSTRAP_LIVE_ENABLED"] = True
    Setting.set_json("explicit_live_confirmed", True)
    Setting.set_json("secondary_confirmation", True)
    db.session.commit()


def _kucoin_connection(app, user: User) -> TradingConnection:
    connection = app.extensions["services"]["trading_connections"].create_or_update(
        user_id=user.id,
        provider="kucoin",
        connection_type="cex_api_key",
        api_key="real-kucoin-key",
        api_secret="real-kucoin-secret",
        passphrase="real-kucoin-passphrase",
        is_active=False,
    )
    connection.verification_status = "verified"
    connection.is_active = True
    db.session.commit()
    return connection


def _missing_hyperliquid(user: User) -> TradingConnection:
    connection = TradingConnection(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        is_active=True,
        verification_status="verified",
    )
    db.session.add(connection)
    db.session.commit()
    return connection


def _ready_kucoin(app, monkeypatch, user: User) -> TradingConnection:
    connection = _kucoin_connection(app, user)
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=30.0, estimated_usd_value=30.0))
    db.session.add(
        LeveragedMarket(
            provider="kucoin",
            venue_symbol="XBTUSDTM",
            symbol="BTC",
            status="active",
            settlement_asset="USDT",
            max_leverage=20,
            liquidity_usd=1_000_000,
            last_seen_at=datetime.utcnow(),
        )
    )
    db.session.commit()
    service = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(service, "can_trade", lambda user_id, mode, connection_id=None: connection_id == connection.id)
    monkeypatch.setattr(
        service,
        "account_snapshot",
        lambda user_id, mode, connection_id=None: ClientSnapshot(
            mode,
            [{"asset": "USDT", "type": "futures", "value": 30.0, "withdrawable": 30.0}],
            [],
            [],
            [],
            [],
        ),
    )
    return connection


def _codes(payload: dict) -> set[str]:
    return {str(item.get("code")) for item in payload.get("active_blockers", []) + payload.get("exchange_blockers", [])}


def test_amount_zero_creates_amount_required_blocker(app) -> None:
    _confirm_live(app)
    user = _user("amountzero")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=10.0, estimated_usd_value=10.0))
    db.session.commit()

    payload = get_vault_cycle_readiness(user.id, amount=0, live_acknowledged=True)

    assert payload["ready"] is False
    assert "amount_required" in _codes(payload)
    assert payload["routing_preview"]["routes"] == []


def test_preview_route_max_uses_available_usdc_and_prices_at_one(app) -> None:
    _confirm_live(app)
    user = _user("maxpreview")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=27.802658, estimated_usd_value=27.802658))
    db.session.commit()
    client = app.test_client()
    _login(client, user)

    response = client.post(
        "/vault/preview-route",
        json={"deposit_asset": "USDC", "settlement_asset": "USDC", "max": True, "one_h10_live_ack": True},
        headers={"Accept": "application/json"},
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["amount"] == pytest.approx(27.802658)
    assert payload["notional_usd"] == pytest.approx(27.802658)


def test_hyperliquid_missing_credentials_has_specific_blocker(app) -> None:
    _confirm_live(app)
    user = _user("missinghl")
    _missing_hyperliquid(user)

    payload = get_vault_cycle_readiness(user.id, amount=10, live_acknowledged=True, enabled_exchanges=["hyperliquid"])

    assert "hyperliquid_credentials_missing" in _codes(payload)


def test_kucoin_ready_hyperliquid_blocked_allocates_kucoin_100(app, monkeypatch) -> None:
    _confirm_live(app)
    user = _user("kucoinonly")
    _ready_kucoin(app, monkeypatch, user)
    _missing_hyperliquid(user)

    payload = get_vault_cycle_readiness(user.id, amount=10, live_acknowledged=True, enabled_exchanges=["hyperliquid", "kucoin"])

    assert payload["ready"] is True
    assert payload["exchange_status"]["hyperliquid"]["allocation_pct"] == 0
    assert payload["exchange_status"]["kucoin"]["allocation_pct"] == pytest.approx(100)
    assert payload["routing_preview"]["routes"][0]["exchange"] == "kucoin"


def test_no_exchange_ready_blocks_json_start(app) -> None:
    _confirm_live(app)
    user = _user("noexchange")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=10.0, estimated_usd_value=10.0))
    db.session.commit()
    client = app.test_client()
    _login(client, user)

    response = client.post(
        "/vault/start-cycle",
        json={"deposit_amount": 5, "deposit_asset": "USDC", "settlement_asset": "USDC", "lock_duration": "1", "one_h10_live_ack": "on"},
        headers={"Accept": "application/json"},
    )

    payload = response.get_json()
    assert response.status_code == 400
    assert "no_exchange_ready" in {item["code"] for item in payload["blockers"]}
    assert VaultCycle.query.filter_by(user_id=user.id).count() == 0


def test_acknowledgement_missing_blocks_json_start(app, monkeypatch) -> None:
    _confirm_live(app)
    user = _user("ackmissing")
    _ready_kucoin(app, monkeypatch, user)
    client = app.test_client()
    _login(client, user)

    response = client.post(
        "/vault/start-cycle",
        json={"deposit_amount": 5, "deposit_asset": "USDC", "settlement_asset": "USDC", "lock_duration": "1", "providers": ["kucoin"]},
        headers={"Accept": "application/json"},
    )

    payload = response.get_json()
    assert response.status_code == 400
    assert "live_acknowledgement_required" in {item["code"] for item in payload["blockers"]}
    assert VaultCycle.query.filter_by(user_id=user.id).count() == 0


def test_panic_lock_blocks_json_start(app, monkeypatch) -> None:
    _confirm_live(app)
    user = _user("panicstart")
    _ready_kucoin(app, monkeypatch, user)
    Setting.set_json("panic_lock", True)
    db.session.commit()
    client = app.test_client()
    _login(client, user)

    response = client.post(
        "/vault/start-cycle",
        json={"deposit_amount": 5, "deposit_asset": "USDC", "settlement_asset": "USDC", "lock_duration": "1", "providers": ["kucoin"], "one_h10_live_ack": "on"},
        headers={"Accept": "application/json"},
    )

    payload = response.get_json()
    assert response.status_code == 400
    assert "panic_lock" in {item["code"] for item in payload["blockers"]}


def test_canary_preview_only_blocks_json_start(app, monkeypatch) -> None:
    _confirm_live(app)
    app.config["LIVE_MICRO_CANARY_ENABLED"] = True
    app.config["LIVE_MICRO_CANARY_PREVIEW_ONLY"] = True
    app.config["LIVE_MICRO_CANARY_LIVE_SUBMIT_ENABLED"] = False
    user = _user("canary")
    _ready_kucoin(app, monkeypatch, user)
    client = app.test_client()
    _login(client, user)

    response = client.post(
        "/vault/start-cycle",
        json={"deposit_amount": 5, "deposit_asset": "USDC", "settlement_asset": "USDC", "lock_duration": "1", "providers": ["kucoin"], "one_h10_live_ack": "on"},
        headers={"Accept": "application/json"},
    )

    payload = response.get_json()
    assert response.status_code == 400
    assert "canary_preview_only" in {item["code"] for item in payload["blockers"]}


def test_required_one_h10_ml_readiness_blocks_json_start(app, monkeypatch) -> None:
    _confirm_live(app)
    app.config["ML_ALL_AREAS_ENABLED"] = True
    app.config["ONE_H10_REQUIRE_PROMOTED_ML"] = True
    user = _user("mlblocked")
    _ready_kucoin(app, monkeypatch, user)
    engine = app.extensions["services"]["ml_decision_engine"]
    monkeypatch.setattr(engine, "family_readiness", lambda family, horizon, provider="global": {"ready": False, "blockers": ["missing_promoted_model"]})
    client = app.test_client()
    _login(client, user)

    response = client.post(
        "/vault/start-cycle",
        json={"deposit_amount": 5, "deposit_asset": "USDC", "settlement_asset": "USDC", "lock_duration": "1", "providers": ["kucoin"], "one_h10_live_ack": "on"},
        headers={"Accept": "application/json"},
    )

    payload = response.get_json()
    assert response.status_code == 400
    assert "ml_readiness_required" in {item["code"] for item in payload["blockers"]}


def test_duplicate_idempotency_key_returns_existing_cycle_without_creating_duplicate(app) -> None:
    _confirm_live(app)
    user = _user("duplicatecycle")
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDC",
        deposit_amount=5.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        status="active",
        algorithm_profile="1H10",
        started_at=datetime.utcnow(),
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
        starting_value_usd=5.0,
        current_estimated_value_usd=5.0,
    )
    db.session.add(cycle)
    db.session.commit()
    _persist_cycle_start_cycle_idempotency(user_id=user.id, idempotency_key="same-tap", cycle_id=cycle.id)
    client = app.test_client()
    _login(client, user)

    response = client.post(
        "/vault/start-cycle",
        json={"deposit_amount": 5, "deposit_asset": "USDC", "settlement_asset": "USDC", "lock_duration": "1", "one_h10_live_ack": "on"},
        headers={"Accept": "application/json", "Idempotency-Key": "same-tap"},
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["duplicate"] is True
    assert VaultCycle.query.filter_by(user_id=user.id).count() == 1


def test_vault_mobile_html_css_contains_safe_area_and_sticky_actions() -> None:
    template = Path("templates/vault.html").read_text()
    css = Path("static/css/app.css").read_text()

    assert "data-vault-top-blockers" in template
    assert "data-vault-result-sheet" in template
    assert "vault-sticky-actions" in template
    assert "env(safe-area-inset-bottom)" in css
    assert ".vault-sticky-actions" in css
    assert "-webkit-overflow-scrolling: touch" in css
