from __future__ import annotations

import json
import sqlite3

from cryptography.fernet import Fernet

from app import create_app
from app.auth import password_hash
from app.extensions import db
from app.models import Setting, User


def _ready_config(db_path):
    return {
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_path}",
        "ENABLE_LIVE_TRADING": True,
        "APP_MODE": "live",
        "SECRET_KEY": "local-live-secret-key-for-readiness-12345",
        "ADMIN_PASSWORD": "admin-password-123",
        "SIGNUP_INVITE_CODE": "invite-code",
        "TOTP_ENCRYPTION_KEY": Fernet.generate_key().decode("utf-8"),
        "USE_REAL_ADDRESSES": True,
        "WALLET_REAL_CUSTODY_ENABLED": True,
        "WALLET_ALLOW_IN_APP_KEYGEN": True,
        "WALLET_WITHDRAWALS_ENABLED": True,
        "WALLET_REQUIRE_WITHDRAWAL_APPROVAL": True,
        "WALLET_EVM_RPC_URL": "https://evm.example.invalid",
        "WALLET_EVM_TOKEN_CONTRACTS": {
            "ETHEREUM": {
                "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                "USDC_DECIMALS": 6,
                "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                "USDT_DECIMALS": 6,
            }
        },
        "WALLET_BTC_INDEXER_URL": "https://btc.example.invalid",
        "WALLET_SOLANA_RPC_URL": "https://sol.example.invalid",
        "WALLET_XRP_RPC_URL": "https://xrp.example.invalid",
        "WALLET_MAX_WITHDRAWAL_BY_ASSET": {
            "ETH": 0.1,
            "USDC": 100.0,
            "USDT": 100.0,
            "BTC": 0.005,
            "SOL": 5.0,
            "XRP": 500.0,
        },
        "WTF_CSRF_ENABLED": True,
    }


def test_production_readiness_strict_fails_with_default_dev_secrets(app) -> None:
    result = app.test_cli_runner().invoke(args=["production-readiness", "--strict"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ready"] is False
    assert any("FLASK_SECRET_KEY" in blocker for blocker in payload["blockers"])
    assert any("TOTP_ENCRYPTION_KEY" in blocker for blocker in payload["blockers"])


def test_production_readiness_strict_passes_with_live_local_config(tmp_path) -> None:
    app = create_app(_ready_config(tmp_path / "ready.db"))
    with app.app_context():
        result = app.test_cli_runner().invoke(args=["production-readiness", "--strict"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["ready"] is True
        assert payload["blockers"] == []
        assert payload["database"]["legacy_paper_tables"] == {
            "paper_account": False,
            "paper_equity_snapshot": False,
        }


def test_production_readiness_blocks_failed_active_connection_health(tmp_path) -> None:
    app = create_app(_ready_config(tmp_path / "blocked.db"))
    with app.app_context():
        user = User(username="blocked", password_hash=password_hash("password123"), role="user")
        db.session.add(user)
        db.session.flush()
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
        Setting.set_json(
            f"connection_health:{connection.id}",
            {
                "connection_id": connection.id,
                "provider": "kucoin",
                "can_trade": False,
                "failure_reason": "Invalid request ip, the current clientIp is:209.52.132.232",
                "client_ip": "209.52.132.232",
                "last_checked_at": "2026-05-01T00:00:00Z",
            },
        )
        db.session.commit()

        result = app.test_cli_runner().invoke(args=["production-readiness", "--strict"])
        payload = json.loads(result.output)

        assert result.exit_code == 1
        assert payload["ready"] is False
        assert any("active connection cannot trade" in blocker for blocker in payload["blockers"])


def test_reset_local_state_backs_up_sqlite_before_clean_slate(tmp_path) -> None:
    db_path = tmp_path / "reset.db"
    app = create_app(_ready_config(db_path))
    with app.app_context():
        db.session.add(User(username="existing", password_hash=password_hash("password123"), role="user"))
        db.session.commit()

        result = app.test_cli_runner().invoke(args=["reset-local-state", "--confirm", "FULL-LIVE-RESET"])

        assert result.exit_code == 0
        payload = json.loads(result.output.split("\nWARNING", 1)[0])
        assert payload["backup"]
        backup_path = payload["backup"]
        with sqlite3.connect(backup_path) as connection:
            existing_count = connection.execute("SELECT COUNT(*) FROM user WHERE username = 'existing'").fetchone()[0]
        assert existing_count == 1
        assert User.query.count() == 0
