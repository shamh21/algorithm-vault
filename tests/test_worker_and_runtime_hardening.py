from __future__ import annotations

import json
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
    Setting,
    StrategyRun,
    User,
    WalletAccount,
    WalletAddress,
    WalletAuditLog,
    WalletWithdrawal,
    WorkerJobRun,
    WorkerLease,
)
from app.services.model_registry import ModelRegistryService
from app.services.wallet_custody import BroadcastResult
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


def test_apple_pay_fulfillment_cron_requires_secret(app) -> None:
    with app.app_context():
        app.config["CRON_SECRET"] = "cron-secret"
        client = app.test_client()

        missing = client.get("/_internal/cron/apple-pay-fulfillment")
        invalid = client.get("/_internal/cron/apple-pay-fulfillment", headers={"Authorization": "Bearer wrong"})

    assert missing.status_code == 401
    assert invalid.status_code == 401


def test_apple_pay_fulfillment_cron_runs_single_job_with_sanitized_response(app) -> None:
    class _WalletApplePayPurchase:
        calls = 0

        def process_pending_orders(self):
            self.calls += 1
            return {"processed": 0, "orders": []}

    fake_purchase = _WalletApplePayPurchase()
    with app.app_context():
        app.config["CRON_SECRET"] = "cron-secret"
        app.extensions["services"]["wallet_apple_pay_purchase"] = fake_purchase
        client = app.test_client()

        response = client.get(
            "/_internal/cron/apple-pay-fulfillment",
            headers={"Authorization": "Bearer cron-secret", "x-vercel-id": "sfo1::cron-test"},
        )
        payload = response.get_json()
        lease = WorkerLease.query.filter_by(lease_name="apple_pay_fulfillment:singleton").one()
        run = WorkerJobRun.query.filter_by(job_name="apple_pay_fulfillment").one()

    assert response.status_code == 200
    assert payload == {
        "ok": True,
        "job": "apple_pay_fulfillment",
        "status": "complete",
        "processed": 0,
        "failed": 0,
        "result_count": 1,
    }
    assert fake_purchase.calls == 1
    assert lease.status == "released"
    assert lease.heartbeat_at is not None
    assert run.status == "complete"


def test_internal_mpc_signer_wallet_buy_treasury_transfer_is_idempotent(app, monkeypatch) -> None:
    from app.routes import internal_mpc_signer as signer_module

    class _FakeAdapter:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def sign_and_broadcast(self, withdrawal: WalletWithdrawal, private_key: str) -> BroadcastResult:
            self.calls.append(
                {
                    "asset": withdrawal.asset,
                    "destination": withdrawal.destination_address,
                    "amount": withdrawal.amount,
                    "private_key": private_key,
                }
            )
            tx_hash = "0xfee" if withdrawal.asset == "ETH" and withdrawal.destination_address.endswith("3333") else "0xuser"
            return BroadcastResult("submitted", tx_hash, {"raw_transaction": "0xsecret"})

    fake_adapter = _FakeAdapter()
    monkeypatch.setattr(signer_module, "_adapter_for", lambda _asset, _network: fake_adapter)

    key = Fernet.generate_key()
    source_address = "0x2222222222222222222222222222222222222222"
    with app.app_context():
        app.config.update(
            WALLET_INTERNAL_MPC_SIGNER_ENABLED=True,
            WALLET_MPC_SIGNER_TOKEN="signer-token",
            WALLET_MPC_SIGNER_ENCRYPTION_KEY=key.decode(),
        )
        user = User(username="treasury", password_hash="hash")
        db.session.add(user)
        db.session.flush()
        account = WalletAccount(user_id=user.id, provider="mpc_signer", asset="USDT", network="Ethereum")
        db.session.add(account)
        db.session.flush()
        wallet_address = WalletAddress(
            user_id=user.id,
            wallet_account_id=account.id,
            asset="USDT",
            network="Ethereum",
            address=source_address,
            status="active",
        )
        wallet_address.encrypted_metadata = {"custody": "mpc", "signer_key_id": "signer_usdt"}
        db.session.add(wallet_address)
        Setting.set_json(
            "internal_mpc_signer_keys_v1",
            {
                "keys": {
                    "signer_usdt": {
                        "user_id": user.id,
                        "asset": "USDT",
                        "network": "Ethereum",
                        "address": source_address,
                        "encrypted_private_key": Fernet(key).encrypt(b"private-key").decode(),
                    }
                }
            },
        )
        db.session.commit()
        client = app.test_client()
        payload = {
            "external_order_id": "wapo_test",
            "fulfillment_kind": "treasury_transfer",
            "asset": "USDT",
            "network": "Ethereum",
            "source_address": source_address,
            "destination_address": "0x1111111111111111111111111111111111111111",
            "amount": 10.5,
            "signer_key_id": "signer_usdt",
            "require_treasury_fee_transfer": True,
            "treasury_fee_address": "0x3333333333333333333333333333333333333333",
            "treasury_fee_eth_amount": 0.001,
        }

        first = client.post(
            "/_internal/mpc-signer/wallet-buy/treasury-transfer",
            json=payload,
            headers={"Authorization": "Bearer signer-token"},
        )
        second = client.post(
            "/_internal/mpc-signer/wallet-buy/treasury-transfer",
            json=payload,
            headers={"Authorization": "Bearer signer-token"},
        )
        withdrawals = WalletWithdrawal.query.filter_by(workflow_type="wallet_buy_fulfillment").all()

    assert first.status_code == 200
    assert first.get_json() == {
        "ok": True,
        "provider_reference": "0xuser",
        "status": "complete",
        "treasury_tx_hash": "0xfee",
        "tx_hash": "0xuser",
    }
    assert second.status_code == 200
    assert second.get_json() == first.get_json()
    assert len(fake_adapter.calls) == 2
    assert fake_adapter.calls[0]["private_key"] == "private-key"
    assert len(withdrawals) == 2
    assert {withdrawal.provider_reference for withdrawal in withdrawals} == {"0xuser", "0xfee"}
    assert "raw_transaction" not in first.get_data(as_text=True)
    assert "0xsecret" not in first.get_data(as_text=True)


def test_wallet_buy_treasury_source_discovery_redacts_key_material(app) -> None:
    from scripts.discover_wallet_buy_treasury_sources import build_wallet_buy_treasury_report

    with app.app_context():
        user = User(username="wallet-buy-treasury", password_hash="hash")
        db.session.add(user)
        db.session.flush()
        signer_records = {}
        for index, asset in enumerate(("ETH", "USDC", "USDT"), start=1):
            address = f"0x{index:040d}"
            signer_key_id = f"signer_{asset.lower()}"
            account = WalletAccount(user_id=user.id, provider="mpc_signer", asset=asset, network="Ethereum")
            db.session.add(account)
            db.session.flush()
            wallet_address = WalletAddress(
                user_id=user.id,
                wallet_account_id=account.id,
                asset=asset,
                network="Ethereum",
                address=address,
                status="active",
                onchain_balance=float(index),
                onchain_status="verified",
                onchain_checked_at=datetime.now(UTC).replace(tzinfo=None),
            )
            wallet_address.encrypted_metadata = {"custody": "mpc", "signer_key_id": signer_key_id}
            db.session.add(wallet_address)
            signer_records[signer_key_id] = {
                "asset": asset,
                "network": "Ethereum",
                "address": address,
                "encrypted_private_key": "do-not-print-private-key",
                "private_key": "also-do-not-print",
            }
        Setting.set_json("internal_mpc_signer_keys_v1", {"keys": signer_records})
        db.session.commit()

        report = build_wallet_buy_treasury_report()

    body = json.dumps(report)
    assert report["ready_for_env"] is True
    assert sorted(report["recommended_env_value"]) == ["ETH", "USDC", "USDT"]
    assert report["recommended_env_value"]["USDT"]["Ethereum"]["signer_key_id"] == "signer_usdt"
    assert report["fee_address_candidates"][0]["address"] == "0x0000000000000000000000000000000000000001"
    assert "encrypted_private_key" not in body
    assert "private_key" not in body
    assert "do-not-print-private-key" not in body
    assert "also-do-not-print" not in body


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
