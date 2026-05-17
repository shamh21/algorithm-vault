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


def test_readiness_payload_classifies_objective_and_recovery_blockers(app) -> None:
    _confirm_live(app)
    app.config["RECOVERY_SQLITE_ACTIVE"] = True
    user = _user("recoveryblocked")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=30.0, estimated_usd_value=30.0))
    db.session.commit()

    payload = get_vault_cycle_readiness(user.id, amount=10, live_acknowledged=True, enabled_exchanges=["kucoin"])

    hard_codes = {str(item.get("code")) for item in payload["hard_blockers"]}
    clearable_codes = {str(item.get("code")) for item in payload["clearable_blockers"]}
    assert payload["ready"] is False
    assert payload["can_start"] is False
    assert payload["objective"]["horizon_seconds"] == app.config["ONE_H10_HORIZON_SECONDS"]
    assert payload["objective"]["target_multiplier"] == pytest.approx(10.0)
    assert "recovery_database_mode" in hard_codes
    assert "recovery_database_mode" not in clearable_codes


def test_readiness_payload_hard_blocks_missing_live_execution_runtime(app) -> None:
    _confirm_live(app)
    app.config["WORKER_MODE"] = "web"
    app.config["ENABLE_IN_PROCESS_WORKERS"] = False
    app.config["WORKER_PROCESS_CONFIGURED"] = False
    user = _user("runtimeblocked")
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=30.0, estimated_usd_value=30.0))
    db.session.commit()

    payload = get_vault_cycle_readiness(user.id, amount=10, live_acknowledged=True, enabled_exchanges=["kucoin"])

    hard_codes = {str(item.get("code")) for item in payload["hard_blockers"]}
    clearable_codes = {str(item.get("code")) for item in payload["clearable_blockers"]}
    assert payload["can_start"] is False
    assert "live_execution_runtime_missing" in hard_codes
    assert "live_execution_runtime_missing" not in clearable_codes


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


def test_vault_page_delegates_initial_preview_when_live_api_origin_configured(app, monkeypatch) -> None:
    _confirm_live(app)
    app.config["PUBLIC_LIVE_API_ORIGIN"] = "https://api.algvault.com"
    user = _user("delegatedpreview")
    _kucoin_connection(app, user)
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=30.0, estimated_usd_value=30.0))
    db.session.commit()
    service = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(
        service,
        "account_snapshot",
        lambda *args, **kwargs: pytest.fail("Vercel-side Vault render must not fetch exchange snapshots"),
    )
    client = app.test_client()
    _login(client, user)

    response = client.get("/vault")

    assert response.status_code == 200
    assert b"live_api_deferred" in response.data
    assert b"https://api.algvault.com" in response.data


def test_kucoin_region_restriction_is_structured_and_redacted(app, monkeypatch) -> None:
    _confirm_live(app)
    user = _user("kucoinrestricted")
    connection = _kucoin_connection(app, user)
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=30.0, estimated_usd_value=30.0))
    db.session.commit()
    raw_alert = (
        'KuCoin unavailable: {"code": "400302", "msg": "Our services are currently unavailable in the U.S. '
        'current ip: 3.86.110.165 and current area: US"}'
    )
    service = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(service, "can_trade", lambda user_id, mode, connection_id=None: False)
    monkeypatch.setattr(
        service,
        "account_snapshot",
        lambda user_id, mode, connection_id=None: (
            ClientSnapshot(mode, [], [], [], [], [raw_alert])
            if connection_id == connection.id
            else ClientSnapshot(mode, [], [], [], [], [])
        ),
    )

    payload = get_vault_cycle_readiness(
        user.id,
        amount=10,
        live_acknowledged=True,
        enabled_exchanges=["kucoin"],
    )

    assert "kucoin_geo_restricted" in _codes(payload)
    rendered = str(payload)
    assert "3.86.110.165" not in rendered
    assert '{"code": "400302"' not in rendered

    client = app.test_client()
    _login(client, user)
    response = client.get(
        "/api/vault/routing-preview?cycle_type=one_h10&amount=10&deposit_asset=USDC&settlement_asset=USDC&one_h10_live_ack=1&providers=kucoin",
        headers={"Accept": "application/json"},
    )
    response_text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "kucoin_geo_restricted" in response_text
    assert "3.86.110.165" not in response_text
    assert '{"code": "400302"' not in response_text
    assert "currently unavailable in the U.S." not in response_text


def test_live_api_cors_allows_configured_vault_origin(app) -> None:
    app.config["PUBLIC_LIVE_API_ORIGIN"] = "https://api.algvault.com"
    app.config["LIVE_API_CORS_ALLOWED_ORIGINS"] = ["https://app.algvault.com"]
    client = app.test_client()

    response = client.open(
        "/vault/readiness",
        method="OPTIONS",
        headers={
            "Origin": "https://app.algvault.com",
            "Access-Control-Request-Headers": "X-CSRF-Token, Idempotency-Key",
        },
    )

    assert response.headers["Access-Control-Allow-Origin"] == "https://app.algvault.com"
    assert response.headers["Access-Control-Allow-Credentials"] == "true"
    assert "X-CSRF-Token" in response.headers["Access-Control-Allow-Headers"]

    actual = client.get(
        "/vault/readiness",
        base_url="https://api.algvault.com",
        headers={"Origin": "https://app.algvault.com", "Accept": "application/json"},
    )

    assert actual.headers["Access-Control-Allow-Origin"] == "https://app.algvault.com"
    assert actual.headers["Access-Control-Allow-Credentials"] == "true"


def test_live_api_cors_rejects_disallowed_and_non_vault_origins(app) -> None:
    app.config["PUBLIC_LIVE_API_ORIGIN"] = "https://api.algvault.com"
    app.config["LIVE_API_CORS_ALLOWED_ORIGINS"] = ["*", "https://app.algvault.com"]
    client = app.test_client()

    disallowed = client.open(
        "/vault/readiness",
        method="OPTIONS",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Headers": "X-CSRF-Token"},
    )
    non_vault = client.open(
        "/healthz",
        method="OPTIONS",
        headers={"Origin": "https://app.algvault.com", "Access-Control-Request-Headers": "X-CSRF-Token"},
    )
    wildcard_origin = client.open(
        "/vault/readiness",
        method="OPTIONS",
        headers={"Origin": "*", "Access-Control-Request-Headers": "X-CSRF-Token"},
    )

    for response in (disallowed, non_vault, wildcard_origin):
        assert "Access-Control-Allow-Origin" not in response.headers
        assert "Access-Control-Allow-Credentials" not in response.headers


def test_session_cookie_domain_is_explicit_only(app) -> None:
    app.config["PUBLIC_APP_ORIGIN"] = "https://app.algvault.com"
    app.config["PUBLIC_LIVE_API_ORIGIN"] = "https://api.algvault.com"
    app.config["SESSION_COOKIE_DOMAIN"] = None

    assert app.config.get("SESSION_COOKIE_DOMAIN") is None
    assert app.session_interface.get_cookie_domain(app) is None

    app.config["SESSION_COOKIE_DOMAIN"] = ".algvault.com"

    assert app.session_interface.get_cookie_domain(app) == ".algvault.com"


def test_csp_connect_src_allows_live_api_without_wildcard(app) -> None:
    app.config["PUBLIC_LIVE_API_ORIGIN"] = "https://api.algvault.com"
    app.config["SECURE_HEADERS_CSP_ENABLED"] = True

    response = app.test_client().get("/login")
    csp = response.headers["Content-Security-Policy"]
    body = response.get_data(as_text=True)

    assert "connect-src 'self' https://api.algvault.com" in csp
    assert "*" not in csp
    assert "https://api.algvault.com" not in body


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
        json={
            "deposit_amount": 5,
            "deposit_asset": "USDC",
            "settlement_asset": "USDC",
            "lock_duration": "1",
            "providers": ["kucoin"],
            "one_h10_live_ack": "on",
        },
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
        json={
            "deposit_amount": 5,
            "deposit_asset": "USDC",
            "settlement_asset": "USDC",
            "lock_duration": "1",
            "providers": ["kucoin"],
            "one_h10_live_ack": "on",
        },
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
    monkeypatch.setattr(
        engine, "family_readiness", lambda family, horizon, provider="global": {"ready": False, "blockers": ["missing_promoted_model"]}
    )
    client = app.test_client()
    _login(client, user)

    response = client.post(
        "/vault/start-cycle",
        json={
            "deposit_amount": 5,
            "deposit_asset": "USDC",
            "settlement_asset": "USDC",
            "lock_duration": "1",
            "providers": ["kucoin"],
            "one_h10_live_ack": "on",
        },
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
    js = Path("static/js/vault.js").read_text()

    assert "data-vault-top-blockers" in template
    assert "data-vault-result-sheet" in template
    assert "vault-sticky-actions" in template
    assert "env(safe-area-inset-bottom)" in css
    assert ".vault-sticky-actions" in css
    assert "-webkit-overflow-scrolling: touch" in css
    assert "vaultApiUrl" in js
    assert 'credentials: "include"' in js


def _ready_hyperliquid_zero_collateral(app, monkeypatch, user: User) -> TradingConnection:
    service = app.extensions["services"]["trading_connections"]
    connection = service.create_or_update(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        api_secret="0x" + ("1" * 64),
        wallet_address="0x" + ("2" * 40),
        is_active=True,
    )
    connection.verification_status = "verified"
    connection.is_active = True
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=30.0, estimated_usd_value=30.0))
    db.session.add(
        LeveragedMarket(
            provider="hyperliquid",
            venue_symbol="BTC",
            symbol="BTC",
            status="active",
            settlement_asset="USDC",
            max_leverage=20,
            liquidity_usd=1_000_000,
            last_seen_at=datetime.utcnow(),
        )
    )
    db.session.commit()
    monkeypatch.setattr(service, "can_trade", lambda user_id, mode, connection_id=None: connection_id == connection.id)
    monkeypatch.setattr(
        service,
        "account_snapshot",
        lambda user_id, mode, connection_id=None: ClientSnapshot(mode, [], [], [], [], []),
    )
    return connection


def test_hyperliquid_zero_collateral_is_ready_auto_funded(app, monkeypatch) -> None:
    _confirm_live(app)
    user = _user("hlautofunded")
    _ready_hyperliquid_zero_collateral(app, monkeypatch, user)

    payload = get_vault_cycle_readiness(user.id, amount=10, live_acknowledged=True, enabled_exchanges=["hyperliquid"])

    status = payload["exchange_status"]["hyperliquid"]
    assert payload["ready"] is True
    assert status["ready"] is True
    assert status["enabled"] is True
    assert status["status"] == "ready_auto_funded"
    assert status["funding_status"] == "auto_funded"
    assert "hyperliquid_settlement_balance_unavailable" not in _codes(payload)
    assert "Auto-funded during vault cycle" in status["funding_label"]
    assert "Collateral is transferred at cycle start" in status["funding_detail"]


def test_hyperliquid_missing_wallet_returns_needs_wallet(app, monkeypatch) -> None:
    _confirm_live(app)
    user = _user("hlmissingwallet")
    service = app.extensions["services"]["trading_connections"]
    connection = TradingConnection(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        encrypted_api_secret=service._encrypt("0x" + ("1" * 64)),
        wallet_address="",
        is_active=True,
        verification_status="verified",
    )
    db.session.add(connection)
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=30.0, estimated_usd_value=30.0))
    db.session.commit()
    monkeypatch.setattr(service, "can_trade", lambda user_id, mode, connection_id=None: connection_id == connection.id)
    monkeypatch.setattr(
        service,
        "account_snapshot",
        lambda user_id, mode, connection_id=None: ClientSnapshot(
            mode,
            [{"asset": "USDC", "type": "margin", "value": 30.0, "withdrawable": 30.0}],
            [],
            [],
            [],
            [],
        ),
    )

    payload = get_vault_cycle_readiness(user.id, amount=10, live_acknowledged=True, enabled_exchanges=["hyperliquid"])

    assert payload["exchange_status"]["hyperliquid"]["status"] == "needs_wallet"
    assert "hyperliquid_wallet_not_verified" in _codes(payload)


def test_kucoin_400302_maps_to_geo_restricted_without_enabling(app, monkeypatch) -> None:
    _confirm_live(app)
    user = _user("kucoingeo")
    connection = _kucoin_connection(app, user)
    connection.verification_status = "geo_restricted"
    connection.is_active = False
    connection.last_verification_error = (
        '{"code":"400302","msg":"Sorry, the service is unavailable in your current area. IP: 98.84.12.34","detectedArea":"US"}'
    )
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=30.0, estimated_usd_value=30.0))
    db.session.commit()

    payload = get_vault_cycle_readiness(user.id, amount=10, live_acknowledged=True, enabled_exchanges=["kucoin"])

    status = payload["exchange_status"]["kucoin"]
    blocker = next(item for item in status["blockers"] if item["code"] == "kucoin_geo_restricted")
    assert status["enabled"] is True
    assert status["ready"] is False
    assert status["can_trade"] is False
    assert status["status"] == "geo_restricted"
    assert blocker["title"] == "Provider restricted"
    assert blocker["description"] == "KuCoin rejected verification from detected region: US."
    assert blocker["diagnostics"]["maskedIp"] == "98.84.xxx.xxx"
    assert "98.84.12.34" not in str(status)


def test_kucoin_verification_400302_stores_safe_geo_diagnostics(app, monkeypatch) -> None:
    user = _user("kucoinverifygeo")
    connection = _kucoin_connection(app, user)
    service = app.extensions["services"]["trading_connections"]

    class GeoRestrictedConnector:
        def can_trade(self, mode: str) -> bool:
            raise RuntimeError('{"code":"400302","msg":"unavailable in your current area: US from 98.84.12.34"}')

    monkeypatch.setattr(service, "_connector_for_connection", lambda saved: GeoRestrictedConnector())

    result = service.verify_connection(user.id, connection.id)

    assert result["ok"] is False
    assert connection.verification_status == "geo_restricted"
    assert connection.is_active is False
    assert result["diagnostics"]["providerCode"] == "400302"
    assert result["diagnostics"]["detectedArea"] == "US"
    assert result["diagnostics"]["maskedIp"] == "98.84.xxx.xxx"
    assert "98.84.12.34" not in result["error"]


def test_routing_preview_providers_include_new_readiness_fields(app, monkeypatch) -> None:
    _confirm_live(app)
    user = _user("previewfields")
    _ready_hyperliquid_zero_collateral(app, monkeypatch, user)
    client = app.test_client()
    _login(client, user)

    response = client.get(
        "/api/vault/routing-preview?amount=10&deposit_asset=USDC&settlement_asset=USDC&providers=hyperliquid&one_h10_live_ack=1",
        headers={"Accept": "application/json"},
    )

    payload = response.get_json()
    assert response.status_code == 200
    provider = next(item for item in payload["providers"] if item["provider"] == "hyperliquid")
    assert provider["ready"] is True
    assert provider["eligible"] is True
    assert provider["status"] == "ready_auto_funded"
    assert provider["readiness_state"] == "ready_auto_funded"
    assert provider["funding_status"] == "auto_funded"
    assert provider["funding_label"] == "Auto-funded during vault cycle"
    assert provider["funding_detail"] == "Collateral is transferred at cycle start and withdrawn after cycle completion."
    assert payload["can_start"] is True
    assert payload["objective"]["horizon_seconds"] == app.config["ONE_H10_HORIZON_SECONDS"]
    assert "hard_blockers" in payload
    assert "advisory_blockers" in payload
    assert "clearable_blockers" in payload
