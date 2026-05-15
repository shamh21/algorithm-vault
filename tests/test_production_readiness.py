from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from cryptography.fernet import Fernet

from app import create_app
from app.auth import password_hash
from app.extensions import db
from app.models import (
    AuditLog,
    DepositAddress,
    Fill,
    Order,
    Setting,
    TradingConnection,
    User,
    VaultCycle,
    WalletAccount,
    WalletAddress,
    WalletBalance,
    WalletLedgerEvent,
    WalletTransaction,
    WalletWithdrawal,
)
from app.services.hyperliquid_client import ClientSnapshot


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
        "WALLET_REQUIRE_WITHDRAWAL_APPROVAL": False,
        "WALLET_CUSTODY_MODE": "mpc",
        "WALLET_MPC_SIGNER_URL": "https://signer.example.invalid",
        "WALLET_MPC_SIGNER_TOKEN": "test-signer-token",
        "WALLET_SIGNER_ISOLATION_CONFIRMED": True,
        "WALLET_SDK_CHECKS_PASSED": True,
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
        "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_WALLET": 100.0,
        "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_ASSET": {
            "ETH": 0.1,
            "USDC": 100.0,
            "USDT": 100.0,
            "BTC": 0.005,
            "SOL": 5.0,
            "XRP": 500.0,
        },
        "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_DESTINATION": 100.0,
        "WALLET_DAILY_GLOBAL_WITHDRAWAL_LIMIT": 250.0,
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


def test_production_readiness_accepts_postgres_for_vps_and_warns_on_extra_workers(tmp_path) -> None:
    app = create_app(_ready_config(tmp_path / "vps.db"))
    with app.app_context():
        app.config["DEPLOYMENT_TARGET"] = "vps"
        app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql+psycopg://bot:secret@db.internal/tradingbot"
        app.config["WEB_CONCURRENCY"] = 2
        app.config["WORKER_PROCESS_CONFIGURED"] = True

        result = app.test_cli_runner().invoke(args=["production-readiness", "--strict"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["ready"] is True
        assert payload["database"]["backend"] == "postgres"
        assert payload["mode"]["deployment_target"] == "vps"
        assert not any("PostgreSQL for VPS" in blocker for blocker in payload["blockers"])
        assert any("one Gunicorn worker" in warning for warning in payload["warnings"])


def test_production_readiness_accepts_vercel_with_dedicated_worker(tmp_path) -> None:
    app = create_app(_ready_config(tmp_path / "vercel.db"))
    with app.app_context():
        app.config["DEPLOYMENT_TARGET"] = "vercel"
        app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql+psycopg://bot:secret@db.internal/tradingbot"
        app.config["ENABLE_IN_PROCESS_WORKERS"] = False
        app.config["WORKER_MODE"] = "web"
        app.config["WORKER_PROCESS_CONFIGURED"] = True

        result = app.test_cli_runner().invoke(args=["production-readiness", "--strict"])
        payload = json.loads(result.output)

        assert result.exit_code == 0
        assert payload["ready"] is True
        assert payload["database"]["backend"] == "postgres"
        assert payload["mode"]["deployment_target"] == "vercel"


def test_production_readiness_blocks_vercel_live_without_worker_process(tmp_path) -> None:
    app = create_app(_ready_config(tmp_path / "vercel-worker.db"))
    with app.app_context():
        app.config["DEPLOYMENT_TARGET"] = "vercel"
        app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql+psycopg://bot:secret@db.internal/tradingbot"
        app.config["ENABLE_IN_PROCESS_WORKERS"] = False
        app.config["WORKER_MODE"] = "web"
        app.config["WORKER_PROCESS_CONFIGURED"] = False

        result = app.test_cli_runner().invoke(args=["production-readiness", "--strict"])
        payload = json.loads(result.output)

        assert result.exit_code == 1
        assert payload["ready"] is False
        assert any("dedicated worker process" in blocker for blocker in payload["blockers"])


def test_production_readiness_blocks_failed_active_connection_health(tmp_path, monkeypatch) -> None:
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
        service = app.extensions["services"]["trading_connections"]
        failure = "Invalid request ip, the current clientIp is:209.52.132.232"
        monkeypatch.setattr(
            service,
            "account_snapshot",
            lambda user_id, mode, connection_id=None: ClientSnapshot(mode, [], [], [], [], [failure]),
        )
        Setting.set_json(
            f"connection_health:{connection.id}",
            {
                "connection_id": connection.id,
                "provider": "kucoin",
                "can_trade": True,
                "failure_reason": "",
                "last_checked_at": "2026-05-01T00:00:00Z",
            },
        )
        db.session.commit()

        result = app.test_cli_runner().invoke(args=["production-readiness", "--strict"])
        payload = json.loads(result.output)

        assert result.exit_code == 1
        assert payload["ready"] is False
        assert any("active connection cannot trade" in blocker for blocker in payload["blockers"])
        assert payload["connection_health"][0]["can_trade"] is False
        assert payload["connection_health"][0]["client_ip"] == "209.52.132.232"
        assert payload["active_connections"][0]["provider"] == "hyperliquid"
        assert payload["active_connections"][0]["active_connection_id"] == connection.id
        assert "Whitelist current client IP 209.52.132.232" in payload["active_connections"][0]["actionable_blocker"]


def test_production_readiness_blocks_stale_encrypted_active_connection(tmp_path) -> None:
    config = _ready_config(tmp_path / "stale-secret.db")
    old_key = config["TOTP_ENCRYPTION_KEY"]
    app = create_app(config)
    with app.app_context():
        user = User(username="stale", password_hash=password_hash("password123"), role="user")
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
        db.session.commit()
        app.config["TOTP_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
        assert app.config["TOTP_ENCRYPTION_KEY"] != old_key

        result = app.test_cli_runner().invoke(args=["production-readiness", "--strict"])
        payload = json.loads(result.output)

        assert result.exit_code == 1
        assert payload["ready"] is False
        assert any("cannot be decrypted" in blocker for blocker in payload["blockers"])
        assert payload["connection_health"][0]["can_trade"] is False
        assert "cannot be decrypted" in payload["connection_health"][0]["failure_reason"]


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


def test_prune_efficiency_data_dry_run_and_confirm_preserve_sufyanh_records(tmp_path) -> None:
    db_path = tmp_path / "efficiency-prune.db"
    app = create_app(_ready_config(db_path))
    with app.app_context():
        user = User(username="sufyanh", password_hash=password_hash("password123"), role="user")
        db.session.add(user)
        db.session.flush()
        connection = TradingConnection(
            user_id=user.id,
            provider="hyperliquid",
            connection_type="cex_api_key",
            wallet_address="0x" + ("2" * 40),
            is_active=True,
            verification_status="verified",
        )
        db.session.add(connection)
        db.session.flush()
        balance = WalletBalance(user_id=user.id, asset="USDC", available_balance=12.5, locked_balance=1.0)
        transaction = WalletTransaction(user_id=user.id, asset="USDC", amount=12.5, transaction_type="deposit")
        deposit_address = DepositAddress(user_id=user.id, asset="USDC", network="ETHEREUM", address="0x" + ("3" * 40))
        wallet_account = WalletAccount(user_id=user.id, asset="USDC", network="ETHEREUM")
        db.session.add_all([balance, transaction, deposit_address, wallet_account])
        db.session.flush()
        wallet_address = WalletAddress(
            wallet_account_id=wallet_account.id,
            user_id=user.id,
            deposit_address_id=deposit_address.id,
            asset="USDC",
            network="ETHEREUM",
            address="0x" + ("4" * 40),
        )
        ledger = WalletLedgerEvent(
            user_id=user.id,
            deposit_address_id=deposit_address.id,
            wallet_address_id=wallet_address.id,
            asset="USDC",
            network="ETHEREUM",
            address=deposit_address.address,
            provider_reference="tx-1",
            idempotency_key="ledger-1",
            amount=12.5,
        )
        withdrawal = WalletWithdrawal(
            user_id=user.id,
            asset="USDC",
            network="ETHEREUM",
            destination_address="0x" + ("5" * 40),
            amount=1.0,
            idempotency_token="withdrawal-1",
        )
        cycle = VaultCycle(
            user_id=user.id,
            trading_connection_id=connection.id,
            deposit_asset="USDC",
            deposit_amount=10.0,
            settlement_asset="USDC",
            lock_duration_hours=1,
            execution_mode="live",
            unlocks_at=datetime.utcnow() + timedelta(hours=1),
        )
        order = Order(
            user_id=user.id,
            trading_connection_id=connection.id,
            vault_cycle_id=None,
            client_order_id="protected-order",
            mode="live",
            symbol="BTC",
            side="buy",
            quantity=0.1,
        )
        db.session.add_all([wallet_address, ledger, withdrawal, cycle, order])
        db.session.flush()
        order.vault_cycle_id = cycle.id
        fill = Fill(order_id=order.id, symbol="BTC", side="buy", quantity=0.1, price=100.0, simulated=False)
        old_at = datetime.utcnow() - timedelta(hours=48)
        recent_at = datetime.utcnow()
        stale_no_trade = AuditLog(
            category="strategy",
            action="no_trade",
            message="old no-trade noise",
            user_id=user.id,
            trading_connection_id=connection.id,
            created_at=old_at,
        )
        recent_no_trade = AuditLog(
            category="strategy",
            action="no_trade",
            message="recent no-trade retained",
            user_id=user.id,
            trading_connection_id=connection.id,
            created_at=recent_at,
        )
        stale_transient = AuditLog(
            category="strategy",
            action="provider_runtime_backoff",
            message="old transient provider noise",
            user_id=user.id,
            trading_connection_id=connection.id,
            created_at=old_at,
        )
        db.session.add_all([fill, stale_no_trade, recent_no_trade, stale_transient])
        db.session.commit()

        runner = app.test_cli_runner()
        dry_run = runner.invoke(
            args=[
                "prune-efficiency-data",
                "--protect-username",
                "sufyanh",
                "--no-trade-retention-hours",
                "1",
                "--transient-retention-hours",
                "1",
            ]
        )
        dry_payload = json.loads(dry_run.output)

        assert dry_run.exit_code == 0
        assert dry_payload["dry_run"] is True
        assert dry_payload["candidates"] == {"strategy_no_trade": 1, "strategy_transient_backoff": 1}
        assert AuditLog.query.filter_by(action="no_trade").count() == 2

        confirmed = runner.invoke(
            args=[
                "prune-efficiency-data",
                "--protect-username",
                "sufyanh",
                "--no-trade-retention-hours",
                "1",
                "--transient-retention-hours",
                "1",
                "--confirm",
                "PRUNE-EFFICIENCY-DATA",
            ]
        )
        payload = json.loads(confirmed.output)

        assert confirmed.exit_code == 0
        assert payload["confirmed"] is True
        assert Path(payload["backup"]).exists()
        assert payload["deleted"] == {"strategy_no_trade": 1, "strategy_transient_backoff": 1}
        assert payload["protected_counts_before"] == payload["protected_counts_after"]
        assert payload["protected_counts_after"]["users"] == 1
        assert payload["protected_counts_after"]["wallet_balances"] == 1
        assert payload["protected_counts_after"]["wallet_transactions"] == 1
        assert payload["protected_counts_after"]["trading_connections"] == 1
        assert payload["protected_counts_after"]["orders"] == 1
        assert payload["protected_counts_after"]["fills"] == 1
        assert WalletBalance.query.filter_by(user_id=user.id, asset="USDC").one().available_balance == 12.5
        assert WalletTransaction.query.filter_by(user_id=user.id).count() == 1
        assert TradingConnection.query.filter_by(user_id=user.id).count() == 1
        assert Order.query.filter_by(user_id=user.id).count() == 1
        assert Fill.query.count() == 1
        assert AuditLog.query.filter_by(action="provider_runtime_backoff").count() == 0
        remaining_no_trade = AuditLog.query.filter_by(action="no_trade").one()
        assert remaining_no_trade.message == "recent no-trade retained"
