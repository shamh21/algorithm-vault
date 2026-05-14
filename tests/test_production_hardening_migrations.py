from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import app as app_module
from app import create_app

ROOT = Path(__file__).resolve().parents[1]


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
