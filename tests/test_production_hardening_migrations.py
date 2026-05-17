from __future__ import annotations

import importlib
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

import app as app_module
from app import create_app
from app.config import _prepare_recovery_sqlite_database_url

ROOT = Path(__file__).resolve().parents[1]


def test_recovery_sqlite_env_copies_bundle_and_overrides_database(tmp_path, monkeypatch) -> None:
    bundle = tmp_path / "account.seed"
    runtime = tmp_path / "runtime" / "account.sqlite"
    bundle.write_bytes(b"sqlite-seed")
    monkeypatch.setenv("ALGVAULT_RECOVERY_SQLITE_ENABLED", "true")
    monkeypatch.setenv("ALGVAULT_RECOVERY_SQLITE_BUNDLE", str(bundle))
    monkeypatch.setenv("ALGVAULT_RECOVERY_SQLITE_PATH", str(runtime))

    database_url, active = _prepare_recovery_sqlite_database_url("postgresql+psycopg://bot:secret@db/app")

    assert active is True
    assert database_url == f"sqlite:///{runtime}"
    assert runtime.read_bytes() == b"sqlite-seed"


def test_vercel_recovery_sqlite_serves_readiness_without_live_controls(tmp_path, monkeypatch) -> None:
    def fail_configured_postgres_probe(configured_url: str, *, timeout_seconds: float) -> None:
        assert "db.prisma.io" in configured_url
        assert timeout_seconds > 0
        raise RuntimeError("planLimitReached: Prisma Postgres resource has exceeded the plan limit")

    bundle = tmp_path / "account.seed"
    runtime = tmp_path / "runtime" / "account.sqlite"
    db_path = tmp_path / "source.sqlite"
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_path}",
            "SCHEMA_BOOTSTRAP_ENABLED": True,
        }
    )
    with app.app_context():
        from app.extensions import db

        db.create_all()
        db.session.execute(text("CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) NOT NULL)"))
        db.session.execute(text("DELETE FROM alembic_version"))
        db.session.execute(text("INSERT INTO alembic_version (version_num) VALUES ('recovery')"))
        db.session.commit()
    bundle.write_bytes(db_path.read_bytes())

    monkeypatch.setenv("ALGVAULT_RECOVERY_SQLITE_ENABLED", "true")
    monkeypatch.setenv("ALGVAULT_RECOVERY_SQLITE_BUNDLE", str(bundle))
    monkeypatch.setenv("ALGVAULT_RECOVERY_SQLITE_PATH", str(runtime))
    monkeypatch.setenv("DEPLOYMENT_TARGET", "vercel")
    monkeypatch.setenv("DATABASE_URL", "postgresql://bot:top-secret@db.prisma.io/postgres?sslmode=require")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("APP_MODE", "live")
    monkeypatch.setenv("ENABLE_IN_PROCESS_WORKERS", "true")
    monkeypatch.setenv("WALLET_WITHDRAWALS_ENABLED", "true")
    monkeypatch.setattr(app_module, "_probe_configured_postgres", fail_configured_postgres_probe)

    import app.config as config_module

    importlib.reload(config_module)
    try:
        recovery_app = create_app(
            {
                "TESTING": True,
                "PUBLIC_APP_ORIGIN": "https://algorithm-vault-chi.vercel.app",
                "PUBLIC_API_ORIGIN": "https://algorithm-vault-chi.vercel.app",
                "SECRET_KEY": "recovery-mode-secret-key-1234567890",
                "TOTP_ENCRYPTION_KEY": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
            }
        )

        ready = recovery_app.test_client().get("/readyz")
        payload = ready.get_json()
        body = ready.get_data(as_text=True)

        assert ready.status_code == 200
        assert payload["database"] == "sqlite"
        assert payload["checks"]["database_recovery_mode"] is True
        assert payload["checks"]["trading_disabled"] is True
        assert "Recovery SQLite is active" in payload["checks"]["database_recovery_message"]
        assert payload["checks"]["configured_postgres"]["backend"] == "postgres"
        assert payload["checks"]["configured_postgres"]["host"] == "db.prisma.io"
        assert payload["checks"]["configured_postgres"]["available"] is False
        assert payload["checks"]["configured_postgres"]["status"] == "unavailable"
        assert payload["checks"]["configured_postgres"]["error_category"] == "plan_limit_reached"
        assert "top-secret" not in body
        assert recovery_app.config["CONFIGURED_DATABASE_URL"].startswith("postgresql+psycopg://bot:top-secret@db.prisma.io")
        assert recovery_app.config["APP_MODE"] == "paper"
        assert recovery_app.config["ENABLE_LIVE_TRADING"] is False
        assert recovery_app.config["ENABLE_IN_PROCESS_WORKERS"] is False
        assert recovery_app.config["WALLET_WITHDRAWALS_ENABLED"] is False

        ops = recovery_app.test_client().get("/ops/status")
        ops_payload = ops.get_json()
        ops_body = ops.get_data(as_text=True)

        assert ops_payload["database_recovery"]["active"] is True
        assert ops_payload["database_recovery"]["runtime_database"] == "sqlite"
        assert ops_payload["database_recovery"]["trading_disabled"] is True
        assert ops_payload["database_recovery"]["configured_postgres"]["available"] is False
        assert ops_payload["runtime_config"]["database_recovery_mode"] is True
        assert ops_payload["runtime_config"]["live_trading_enabled"] is False
        assert "top-secret" not in ops_body
    finally:
        monkeypatch.undo()
        importlib.reload(config_module)


def test_alembic_upgrade_fresh_sqlite_creates_hardening_tables(tmp_path) -> None:
    db_path = tmp_path / "fresh-migrations.db"
    env = {
        **os.environ,
        "SKIP_SCHEMA_BOOTSTRAP": "1",
        "DATABASE_URL": f"sqlite:///{db_path}",
        "PUBLIC_APP_ORIGIN": "https://app.algvault.com",
        "PUBLIC_API_ORIGIN": "https://app.algvault.com",
    }
    result = subprocess.run(
        [sys.executable, "-m", "flask", "--app", "wsgi:app", "db", "upgrade"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        ml_columns = {row[1] for row in connection.execute("PRAGMA table_info(ml_offline_model)")}

    assert {"worker_lease", "worker_job_run", "ml_model_registry", "alembic_version"}.issubset(tables)
    assert {"feature_schema_hash", "dataset_hash", "rollback_target_model_id", "drift_status"}.issubset(ml_columns)


def test_startup_without_schema_bootstrap_does_not_create_hidden_schema(tmp_path) -> None:
    db_path = tmp_path / "unmigrated-prod.db"
    with pytest.raises(RuntimeError, match="Database schema is not migrated"):
        create_app(
            {
                "TESTING": True,
                "SCHEMA_BOOTSTRAP_ENABLED": False,
                "DEPLOYMENT_TARGET": "production",
                "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_path}",
                "PUBLIC_APP_ORIGIN": "https://app.algvault.com",
                "PUBLIC_API_ORIGIN": "https://app.algvault.com",
            }
        )

    with sqlite3.connect(db_path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "user" not in tables


def test_vercel_database_startup_error_returns_controlled_readiness_503(tmp_path, monkeypatch) -> None:
    def fail_schema_check() -> None:
        raise app_module.DatabaseStartupError("database unavailable")

    monkeypatch.setattr(app_module, "_verify_migrated_schema", fail_schema_check)
    app = create_app(
        {
            "TESTING": True,
            "DEFER_DATABASE_STARTUP_ERRORS": True,
            "SCHEMA_BOOTSTRAP_ENABLED": False,
            "DEPLOYMENT_TARGET": "vercel",
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'deferred.db'}",
            "PUBLIC_APP_ORIGIN": "https://app.algvault.com",
            "PUBLIC_API_ORIGIN": "https://app.algvault.com",
        }
    )

    client = app.test_client()
    health = client.get("/healthz")
    ready = client.get("/readyz")
    root = client.get("/", headers={"Accept": "text/html"})

    assert health.status_code == 200
    assert ready.status_code == 503
    assert ready.get_json()["checks"]["database_startup_blocked"] is True
    assert root.status_code == 503
    assert "Service Unavailable" in root.get_data(as_text=True)


def test_vercel_deferred_startup_still_fails_fast_for_unmigrated_schema(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Database schema is not migrated"):
        create_app(
            {
                "TESTING": True,
                "DEFER_DATABASE_STARTUP_ERRORS": True,
                "SCHEMA_BOOTSTRAP_ENABLED": False,
                "DEPLOYMENT_TARGET": "vercel",
                "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'unmigrated-vercel.db'}",
                "PUBLIC_APP_ORIGIN": "https://app.algvault.com",
                "PUBLIC_API_ORIGIN": "https://app.algvault.com",
            }
        )
