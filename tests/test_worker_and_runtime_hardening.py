from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app import create_app
from app.extensions import db
from app.models import (
    AuditLog,
    MLModelRegistry,
    MLOfflineModel,
    StrategyRun,
    User,
    WalletAuditLog,
    WalletWithdrawal,
    WorkerJobRun,
    WorkerLease,
)
from app.services.model_registry import ModelRegistryService
from app.services.worker_lease import WorkerLeaseService
from app.settings_validation import RuntimeConfigError, validate_runtime_config
from app.workers.runner import _run_due_jobs


def test_worker_lease_blocks_duplicates_and_recovers_stale(app) -> None:
    with app.app_context():
        owner_a = WorkerLeaseService(app.config, owner_id="owner-a")
        owner_b = WorkerLeaseService(app.config, owner_id="owner-b")

        assert owner_a.acquire("strategy_starter:singleton", ttl_seconds=60) is not None
        assert owner_b.acquire("strategy_starter:singleton", ttl_seconds=60) is None

        lease = WorkerLease.query.filter_by(lease_name="strategy_starter:singleton").one()
        lease.expires_at = datetime.utcnow() - timedelta(seconds=1)
        db.session.commit()

        recovered = owner_b.acquire("strategy_starter:singleton", ttl_seconds=60)
        assert recovered is not None
        assert recovered.owner_id == "owner-b"
        assert AuditLog.query.filter_by(category="worker", action="worker_lease_acquire_failed").count() == 1


def test_strategy_run_lease_blocks_duplicate_owner_and_recovers_stale(app) -> None:
    with app.app_context():
        run = StrategyRun(
            strategy_name="ema_crossover",
            symbol="BTC",
            timeframe="1m",
            mode="live",
            status="queued",
            manual_enabled=True,
        )
        db.session.add(run)
        db.session.commit()
        owner_a = WorkerLeaseService(app.config, owner_id="strategy-owner-a")
        owner_b = WorkerLeaseService(app.config, owner_id="strategy-owner-b")

        assert owner_a.acquire_strategy_run(run.id, ttl_seconds=60) is not None
        assert owner_b.acquire_strategy_run(run.id, ttl_seconds=60) is None
        assert owner_a.heartbeat_strategy_run(run.id, ttl_seconds=60) is True

        lease = WorkerLease.query.filter_by(lease_name=WorkerLeaseService.strategy_run_lease_name(run.id)).one()
        lease.expires_at = datetime.utcnow() - timedelta(seconds=1)
        db.session.commit()

        recovered = owner_b.acquire_strategy_run(run.id, ttl_seconds=60)

        assert recovered is not None
        assert recovered.owner_id == "strategy-owner-b"
        assert owner_a.heartbeat_strategy_run(run.id, ttl_seconds=60) is False
        assert owner_b.heartbeat_strategy_run(run.id, ttl_seconds=60) is True


def test_worker_job_idempotency_prevents_duplicate_runs(app) -> None:
    with app.app_context():
        service = WorkerLeaseService(app.config, owner_id="worker")
        should_run, run = service.start_job("treasury_solvency", "treasury_solvency:bucket")
        assert should_run is True
        service.complete_job(run)

        duplicate, duplicate_run = service.start_job("treasury_solvency", "treasury_solvency:bucket")
        assert duplicate is False
        assert duplicate_run.id == run.id
        assert WorkerJobRun.query.count() == 1


def test_worker_runner_executes_one_shot_job_with_lease(app) -> None:
    with app.app_context():
        app.config["WORKER_MODE"] = "worker"
        service = WorkerLeaseService(app.config, owner_id="runner-test")

        results = _run_due_jobs(service, job_filter={"strategy_starter"})

        assert results[0]["ok"] is True
        assert results[0]["status"] == "complete"
        assert results[0]["job_name"] == "strategy_starter"
        assert WorkerLease.query.filter_by(lease_name="strategy_starter:singleton").one().status == "released"


def test_runtime_config_blocks_unsafe_production_withdrawals() -> None:
    config = {
        "DEPLOYMENT_TARGET": "production",
        "SQLALCHEMY_DATABASE_URI": "sqlite:///prod.db",
        "WORKER_MODE": "web",
        "ENABLE_IN_PROCESS_WORKERS": True,
        "ENABLE_LIVE_TRADING": True,
        "WALLET_WITHDRAWALS_ENABLED": True,
        "WALLET_CUSTODY_MODE": "encrypted_db",
    }

    validation = validate_runtime_config(config)
    assert validation.ok is False
    assert any("PostgreSQL" in blocker for blocker in validation.blockers)
    assert any("custody" in blocker for blocker in validation.blockers)
    with pytest.raises(RuntimeConfigError):
        validate_runtime_config(config, strict=True)


def test_runtime_config_treats_vercel_as_production() -> None:
    validation = validate_runtime_config(
        {
            "DEPLOYMENT_TARGET": "vercel",
            "SQLALCHEMY_DATABASE_URI": "sqlite:///prod.db",
            "WORKER_MODE": "web",
            "ENABLE_IN_PROCESS_WORKERS": False,
            "ENABLE_LIVE_TRADING": False,
            "WALLET_WITHDRAWALS_ENABLED": False,
        }
    )

    assert validation.ok is False
    assert any("PostgreSQL" in blocker for blocker in validation.blockers)


def test_runtime_config_blocks_vercel_in_process_web_workers() -> None:
    validation = validate_runtime_config(
        {
            "DEPLOYMENT_TARGET": "vercel",
            "SQLALCHEMY_DATABASE_URI": "postgresql+psycopg://bot:secret@db.internal/tradingbot",
            "WORKER_MODE": "web",
            "ENABLE_IN_PROCESS_WORKERS": True,
            "ENABLE_LIVE_TRADING": False,
            "SCHEMA_BOOTSTRAP_ENABLED": False,
            "WALLET_WITHDRAWALS_ENABLED": False,
        }
    )

    assert validation.ok is False
    assert any("in-process workers" in blocker for blocker in validation.blockers)


def test_runtime_config_blocks_vercel_live_without_worker_process() -> None:
    validation = validate_runtime_config(
        {
            "DEPLOYMENT_TARGET": "vercel",
            "SQLALCHEMY_DATABASE_URI": "postgresql+psycopg://bot:secret@db.internal/tradingbot",
            "WORKER_MODE": "web",
            "ENABLE_IN_PROCESS_WORKERS": False,
            "ENABLE_LIVE_TRADING": True,
            "SCHEMA_BOOTSTRAP_ENABLED": False,
            "WALLET_WITHDRAWALS_ENABLED": False,
        }
    )

    assert validation.ok is False
    assert any("dedicated worker process" in blocker for blocker in validation.blockers)


def test_runtime_config_accepts_vercel_live_with_dedicated_worker() -> None:
    validation = validate_runtime_config(
        {
            "DEPLOYMENT_TARGET": "vercel",
            "SQLALCHEMY_DATABASE_URI": "postgresql+psycopg://bot:secret@db.internal/tradingbot",
            "WORKER_MODE": "web",
            "ENABLE_IN_PROCESS_WORKERS": False,
            "WORKER_PROCESS_CONFIGURED": True,
            "ENABLE_LIVE_TRADING": True,
            "SCHEMA_BOOTSTRAP_ENABLED": False,
            "WALLET_WITHDRAWALS_ENABLED": False,
            "WALLET_CUSTODY_MODE": "local_dev",
        }
    )

    assert validation.ok is True
    assert validation.deployment_target == "vercel"


def test_wallet_safety_gate_blocks_weak_production_custody(tmp_path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'wallet-safety.db'}",
            "DEPLOYMENT_TARGET": "production",
            "PUBLIC_APP_ORIGIN": "https://app.algvault.com",
            "PUBLIC_API_ORIGIN": "https://app.algvault.com",
            "WALLET_WITHDRAWALS_ENABLED": True,
            "WALLET_CUSTODY_MODE": "encrypted_db",
            "WALLET_REQUIRE_WITHDRAWAL_APPROVAL": False,
        }
    )
    with app.app_context():
        user = User(username="wallet-prod", password_hash="hash", role="user")
        db.session.add(user)
        db.session.flush()
        withdrawal = WalletWithdrawal(
            user_id=user.id,
            asset="ETH",
            network="Ethereum",
            destination_address="0x" + ("1" * 40),
            amount=0.01,
            status="pending_submission",
            idempotency_token="prod-weak-custody",
        )
        db.session.add(withdrawal)
        db.session.flush()

        result = app.extensions["services"]["self_custody_wallet"].submit_withdrawal(withdrawal, mode="live")

        assert result.status == "failed_safety_gate"
        assert "production withdrawals require" in result.failure_reason
        assert WalletAuditLog.query.filter_by(action="withdrawal_blocked_by_safety_gate").count() == 1


def test_wallet_velocity_limit_blocks_excess_withdrawal(app) -> None:
    with app.app_context():
        app.config.update(
            {
                "WALLET_WITHDRAWALS_ENABLED": True,
                "WALLET_CUSTODY_MODE": "local_dev",
                "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_WALLET": 1.0,
            }
        )
        user = User(username="wallet-limit", password_hash="hash", role="user")
        db.session.add(user)
        db.session.flush()
        db.session.add(
            WalletWithdrawal(
                user_id=user.id,
                asset="ETH",
                network="Ethereum",
                destination_address="0x" + ("2" * 40),
                amount=0.8,
                status="submitted",
                idempotency_token="existing-limit",
            )
        )
        withdrawal = WalletWithdrawal(
            user_id=user.id,
            asset="ETH",
            network="Ethereum",
            destination_address="0x" + ("3" * 40),
            amount=0.3,
            status="pending_submission",
            idempotency_token="new-limit",
        )
        db.session.add(withdrawal)
        db.session.flush()

        result = app.extensions["services"]["self_custody_wallet"].submit_withdrawal(withdrawal, mode="live")

        assert result.status == "failed_safety_gate"
        assert "per-wallet daily withdrawal limit exceeded" in result.failure_reason


def test_model_registry_records_promotion_and_rollback_metadata(app) -> None:
    with app.app_context():
        service = ModelRegistryService(app.config)
        previous = MLOfflineModel(
            model_key="offline:test:prev",
            provider="global",
            horizon="1h",
            model_type="ridge",
            status="promoted",
            artifact_path="/tmp/prev.joblib",
            training_rows=10,
            validation_rows=5,
        )
        previous.feature_names = ["a", "b"]
        previous.metrics = {"validation_loss": 0.1}
        current = MLOfflineModel(
            model_key="offline:test:current",
            provider="global",
            horizon="1h",
            model_type="ridge",
            status="promoted",
            artifact_path="/tmp/current.joblib",
            training_rows=12,
            validation_rows=6,
        )
        current.feature_names = ["a", "b"]
        current.metrics = {"validation_loss": 0.05}
        db.session.add_all([previous, current])
        db.session.flush()

        service.record_promotion(previous, model_family="offline_ranker", promotion_source="test")
        db.session.commit()
        service.record_promotion(current, model_family="offline_ranker", promotion_source="test")
        db.session.commit()

        registry = MLModelRegistry.query.filter_by(model_key="offline:test:current").one()
        assert registry.rollback_target_model_id == previous.id
        assert registry.feature_schema_hash
        assert registry.dataset_hash
        assert AuditLog.query.filter_by(category="ml", action="model_promoted").count() >= 2


def test_ops_status_redacts_and_reports_core_state(app) -> None:
    with app.app_context():
        app.config["HL_SECRET_KEY"] = "super-secret"
        response = app.test_client().get("/ops/status")
        assert response.status_code == 200
        payload = response.get_json()
        body = response.get_data(as_text=True)

    assert payload["migration_version"] is None or isinstance(payload["migration_version"], str)
    assert "workers" in payload
    assert "wallets" in payload
    assert "models" in payload
    assert "super-secret" not in body


def test_thin_compatibility_entrypoints_register_cli_and_routes(app) -> None:
    import app.cli as cli_module
    import app.routes.consumer as consumer_module

    assert hasattr(cli_module, "register_cli")
    assert consumer_module.consumer_bp.name == "consumer"

    result = app.test_cli_runner().invoke(args=["worker", "start", "--help"])
    assert result.exit_code == 0
    assert "dedicated DB-lease worker" in result.output
