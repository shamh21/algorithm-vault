from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from cryptography.fernet import Fernet

import app as app_module
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
from app.services.withdrawal_config import automatic_withdrawal_blockers, wallet_withdrawals_enabled
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


def test_runtime_config_requires_mpc_signer_for_production_withdrawals() -> None:
    config = {
        "DEPLOYMENT_TARGET": "production",
        "SQLALCHEMY_DATABASE_URI": "postgresql+psycopg://bot:secret@db.internal/tradingbot",
        "WORKER_MODE": "web",
        "WORKER_PROCESS_CONFIGURED": True,
        "ENABLE_IN_PROCESS_WORKERS": False,
        "ENABLE_LIVE_TRADING": True,
        "SCHEMA_BOOTSTRAP_ENABLED": False,
        "WALLET_WITHDRAWALS_ENABLED": True,
        "WALLET_CUSTODY_MODE": "mpc",
        "WALLET_SIGNER_ISOLATION_CONFIRMED": True,
        "WALLET_SDK_CHECKS_PASSED": True,
        "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_WALLET": 100.0,
        "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_ASSET": {"USDC": 100.0},
        "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_DESTINATION": 100.0,
        "WALLET_DAILY_GLOBAL_WITHDRAWAL_LIMIT": 250.0,
    }

    validation = validate_runtime_config(config)

    assert validation.ok is False
    assert any("WALLET_MPC_SIGNER_URL" in blocker for blocker in validation.blockers)
    assert any("WALLET_MPC_SIGNER_TOKEN" in blocker for blocker in validation.blockers)


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
    assert any("FLASK_SECRET_KEY" in blocker for blocker in validation.blockers)
    assert any("TOTP_ENCRYPTION_KEY" in blocker for blocker in validation.blockers)


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
            "SECRET_KEY": "vercel-runtime-secret-key-123456789",
            "TOTP_ENCRYPTION_KEY": Fernet.generate_key().decode("utf-8"),
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
    assert "apple_pay_purchase" in payload
    assert payload["live_operations"]["ready"] is False
    assert any("live operations" in blocker for blocker in payload["live_operations"]["blockers"])
    assert "super-secret" not in body


def test_ops_status_reports_card_buy_readiness_without_secret_values(app) -> None:
    with app.app_context():
        app.config.update(
            {
                "CARD_BUY_ENABLED": True,
                "CARD_GATEWAY_TOKENIZATION_URL": "https://card-gateway.example/tokenize",
                "CARD_GATEWAY_AUTHORIZE_URL": "https://card-gateway.example/authorize",
                "CARD_GATEWAY_API_KEY": "card-secret",
                "CARD_GATEWAY_WEBHOOK_SECRET": "card-webhook-secret",
                "CARD_GATEWAY_PUBLIC_CONFIG": {"publishable_key": "pk_test_card"},
                "APPLE_PAY_CRYPTO_SALE_APPROVED": True,
                "APPLE_PAY_BUY_ALLOWED_ASSETS": {
                    "ETH": {"Ethereum": {"fulfillment_kind": "treasury_transfer", "chain_id": 1, "decimals": 18}},
                    "USDC": {
                        "Ethereum": {
                            "fulfillment_kind": "treasury_transfer",
                            "chain_id": 1,
                            "token_address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
                            "decimals": 6,
                        }
                    },
                    "USDT": {
                        "Ethereum": {
                            "fulfillment_kind": "treasury_transfer",
                            "chain_id": 1,
                            "token_address": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                            "decimals": 6,
                        }
                    },
                },
                "APPLE_PAY_TREASURY_SOURCE_WALLETS": {
                    "ETH": {"Ethereum": {"source_address": "0x2222222222222222222222222222222222222222", "signer_route": "evm"}},
                    "USDC": {"Ethereum": {"source_address": "0x2222222222222222222222222222222222222222", "signer_route": "evm"}},
                    "USDT": {"Ethereum": {"source_address": "0x2222222222222222222222222222222222222222", "signer_route": "evm"}},
                },
                "APPLE_PAY_TREASURY_SOURCE_ADDRESS": "0x2222222222222222222222222222222222222222",
                "APPLE_PAY_TREASURY_FEE_ADDRESS": "0x3333333333333333333333333333333333333333",
                "APPLE_PAY_TREASURY_SIGNER_URL": "https://signer.example/submit",
                "APPLE_PAY_TREASURY_SIGNER_TOKEN": "signer-secret",
                "APPLE_PAY_ASSET_PRICE_USD": {"ETH": 3000.0},
                "WALLET_EVM_RPC_URL": "https://evm.example.invalid",
                "WALLET_BUY_PLATFORM_FEE_BPS": 250.0,
                "ONEINCH_API_KEY": "oneinch-secret",
                "ONEINCH_API_BASE_URL": "https://api.1inch.com/swap/v6.1",
            }
        )

        response = app.test_client().get("/ops/status")
        payload = response.get_json()
        body = response.get_data(as_text=True)

    card_buy = payload["apple_pay_purchase"]["card_buy"]
    assert card_buy["enabled"] is True
    assert card_buy["provider"] == "custom_card_gateway"
    assert card_buy["gateway_tokenization_configured"] is True
    assert card_buy["gateway_authorize_configured"] is True
    assert card_buy["gateway_public_configured"] is True
    assert card_buy["treasury_fee_asset"] == "ETH"
    assert card_buy["treasury_fee_address_configured"] is True
    assert sorted(card_buy["allowed_assets"]) == ["ETH", "USDC", "USDT"]
    assert card_buy["oneinch_configured"] is True
    assert card_buy["oneinch_base_url"] == "https://api.1inch.com/swap/v6.1"
    assert "card-secret" not in body
    assert "card-webhook-secret" not in body
    assert "signer-secret" not in body
    assert "oneinch-secret" not in body


def test_ops_status_live_operations_call_out_recovery_mode(app, monkeypatch) -> None:
    def fail_configured_postgres_probe(configured_url: str, *, timeout_seconds: float) -> None:
        assert configured_url.startswith("postgresql+psycopg://")
        assert timeout_seconds > 0
        raise RuntimeError("planLimitReached")

    monkeypatch.setattr(app_module, "_probe_configured_postgres", fail_configured_postgres_probe)
    with app.app_context():
        app.config.update(
            {
                "DEPLOYMENT_TARGET": "vercel",
                "APP_MODE": "paper",
                "SQLALCHEMY_DATABASE_URI": "sqlite:///recovery.db",
                "CONFIGURED_DATABASE_URL": "postgresql+psycopg://bot:secret@db.prisma.io/postgres?sslmode=require",
                "RECOVERY_SQLITE_ACTIVE": True,
                "ENABLE_LIVE_TRADING": False,
                "ENABLE_IN_PROCESS_WORKERS": False,
                "WORKER_PROCESS_CONFIGURED": False,
                "WALLET_CUSTODY_MODE": "local_dev",
                "WALLET_WITHDRAWALS_ENABLED": False,
            }
        )
        app.config["RUNTIME_CONFIG_VALIDATION"] = validate_runtime_config(app.config)

        payload = app.test_client().get("/ops/status").get_json()

    live_operations = payload["live_operations"]
    database_recovery = payload["database_recovery"]
    assert live_operations["ready"] is False
    assert live_operations["database_backend"] == "sqlite"
    assert "live operations require PostgreSQL DATABASE_URL" in live_operations["blockers"]
    assert "recovery SQLite mode must be inactive for live operations" in live_operations["blockers"]
    assert database_recovery["active"] is True
    assert database_recovery["runtime_database"] == "sqlite"
    assert database_recovery["trading_disabled"] is True
    assert database_recovery["configured_postgres"]["host"] == "db.prisma.io"
    assert database_recovery["configured_postgres"]["available"] is False
    assert database_recovery["configured_postgres"]["error_category"] == "plan_limit_reached"


def test_live_operations_status_warns_when_treasury_ready_but_unfunded(app) -> None:
    app.config.update(
        {
            "DEPLOYMENT_TARGET": "vercel",
            "APP_MODE": "live",
        }
    )

    payload = app_module._live_operations_status(
        app,
        {
            "live_trading_enabled": True,
            "worker_process_configured": True,
            "in_process_workers_enabled": False,
            "database_backend": "postgres",
            "database_recovery_mode": False,
            "withdrawals_enabled": True,
            "custody_mode": "mpc",
            "blockers": [],
        },
        {
            "ready": True,
            "eth_balance": 0.0,
            "reserve_health": {"state": "emergency"},
            "blockers": [],
        },
    )

    assert payload["ready"] is True
    assert payload["withdrawals_ready"] is True
    assert any("reserve health is emergency" in warning for warning in payload["warnings"])
    assert any("0 ETH" in warning for warning in payload["warnings"])


def test_thin_compatibility_entrypoints_register_cli_and_routes(app) -> None:
    import app.cli as cli_module
    import app.routes.consumer as consumer_module

    assert hasattr(cli_module, "register_cli")
    assert consumer_module.consumer_bp.name == "consumer"

    result = app.test_cli_runner().invoke(args=["worker", "start", "--help"])
    assert result.exit_code == 0
    assert "dedicated DB-lease worker" in result.output


def test_worker_status_handles_timezone_aware_heartbeats(monkeypatch) -> None:
    import app as app_module

    class _Lease:
        lease_name = "test-aware-heartbeat"
        status = "held"
        heartbeat_at = datetime.now(UTC) - timedelta(seconds=7)
        expires_at = datetime.now(UTC) + timedelta(seconds=60)

    class _Query:
        def order_by(self, *_args, **_kwargs):
            return self

        def all(self):
            return [_Lease()]

    class _WorkerLease:
        lease_name = WorkerLease.lease_name
        query = _Query()

    monkeypatch.setattr(app_module, "WorkerLease", _WorkerLease)

    status = app_module._worker_status(datetime.utcnow())

    assert status["available"] is True
    assert status["leases"][0]["lease_name"] == "test-aware-heartbeat"
    assert status["max_lease_lag_seconds"] >= 0
    assert "expected_leases" in status
    assert "missing_expected_leases" in status
    assert "stale_expected_leases" in status
    assert "job_leases" in status


def test_wallet_withdrawals_auto_enable_for_ready_live_real_custody() -> None:
    config = {
        "APP_MODE": "live",
        "ENABLE_LIVE_TRADING": True,
        "USE_REAL_ADDRESSES": True,
        "WALLET_REAL_CUSTODY_ENABLED": True,
        "WALLET_ALLOW_IN_APP_KEYGEN": True,
        "WALLET_WITHDRAWALS_ENABLED": False,
        "WALLET_AUTO_ENABLE_WITHDRAWALS": True,
    }

    assert wallet_withdrawals_enabled(config) is True
    assert automatic_withdrawal_blockers(config) == []


def test_wallet_withdrawals_auto_enable_keeps_production_safeguards_required() -> None:
    unsafe = {
        "DEPLOYMENT_TARGET": "production",
        "APP_MODE": "live",
        "ENABLE_LIVE_TRADING": True,
        "USE_REAL_ADDRESSES": True,
        "WALLET_REAL_CUSTODY_ENABLED": True,
        "WALLET_ALLOW_IN_APP_KEYGEN": True,
        "WALLET_WITHDRAWALS_ENABLED": False,
        "WALLET_AUTO_ENABLE_WITHDRAWALS": True,
        "WALLET_CUSTODY_MODE": "encrypted_db",
    }
    safe = {
        **unsafe,
        "SQLALCHEMY_DATABASE_URI": "postgresql+psycopg://bot:secret@db.internal/tradingbot",
        "WORKER_MODE": "worker",
        "ENABLE_IN_PROCESS_WORKERS": False,
        "WORKER_PROCESS_CONFIGURED": True,
        "WALLET_CUSTODY_MODE": "kms",
        "WALLET_SIGNER_ISOLATION_CONFIRMED": True,
        "WALLET_SDK_CHECKS_PASSED": True,
        "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_WALLET": 1.0,
        "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_ASSET": {"ETH": 1.0},
        "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_DESTINATION": 1.0,
        "WALLET_DAILY_GLOBAL_WITHDRAWAL_LIMIT": 2.0,
    }

    assert wallet_withdrawals_enabled(unsafe) is False
    assert any("production withdrawals require" in blocker for blocker in automatic_withdrawal_blockers(unsafe))
    assert wallet_withdrawals_enabled(safe) is True
    assert validate_runtime_config(safe).withdrawals_enabled is True


def test_wallet_withdrawals_auto_enable_requires_mpc_signer() -> None:
    config = {
        "DEPLOYMENT_TARGET": "vercel",
        "APP_MODE": "live",
        "ENABLE_LIVE_TRADING": True,
        "USE_REAL_ADDRESSES": True,
        "WALLET_REAL_CUSTODY_ENABLED": True,
        "WALLET_ALLOW_IN_APP_KEYGEN": True,
        "WALLET_WITHDRAWALS_ENABLED": False,
        "WALLET_AUTO_ENABLE_WITHDRAWALS": True,
        "WALLET_CUSTODY_MODE": "mpc",
        "WALLET_SIGNER_ISOLATION_CONFIRMED": True,
        "WALLET_SDK_CHECKS_PASSED": True,
        "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_WALLET": 1.0,
        "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_ASSET": {"ETH": 1.0},
        "WALLET_DAILY_WITHDRAWAL_LIMIT_BY_DESTINATION": 1.0,
        "WALLET_DAILY_GLOBAL_WITHDRAWAL_LIMIT": 2.0,
    }

    blockers = automatic_withdrawal_blockers(config)

    assert wallet_withdrawals_enabled(config) is False
    assert "WALLET_MPC_SIGNER_URL is not configured" in blockers
    assert "WALLET_MPC_SIGNER_TOKEN is not configured" in blockers

    ready = {
        **config,
        "WALLET_MPC_SIGNER_URL": "https://signer.example.invalid",
        "WALLET_MPC_SIGNER_TOKEN": "test-token",
    }

    assert wallet_withdrawals_enabled(ready) is True
