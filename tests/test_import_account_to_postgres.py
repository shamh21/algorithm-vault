from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, select

import app.models  # noqa: F401
from app.extensions import db
from scripts.import_account_to_postgres import import_account, summary_to_dict


NOW = datetime(2026, 5, 12, 12, 0, 0)


def _sqlite_url(path: Path) -> str:
    return f"sqlite:///{path}"


def _create_schema(path: Path) -> None:
    engine = create_engine(_sqlite_url(path), future=True)
    try:
        db.metadata.create_all(engine)
    finally:
        engine.dispose()


def _seed_source(path: Path) -> None:
    engine = create_engine(_sqlite_url(path), future=True)
    tables = db.metadata.tables
    try:
        with engine.begin() as conn:
            conn.execute(
                tables["user"].insert(),
                [
                    {
                        "id": 1,
                        "username": "sufyanh",
                        "password_hash": "scrypt:real-hash",
                        "role": "admin",
                        "referral_invite_code_id": None,
                        "totp_secret_encrypted": "encrypted-totp",
                        "two_factor_enabled_at": NOW,
                        "created_at": NOW,
                        "updated_at": NOW,
                    },
                    {
                        "id": 2,
                        "username": "debugmax2",
                        "password_hash": "scrypt:debug-hash",
                        "role": "user",
                        "referral_invite_code_id": None,
                        "totp_secret_encrypted": None,
                        "two_factor_enabled_at": None,
                        "created_at": NOW,
                        "updated_at": NOW,
                    },
                ],
            )
            conn.execute(
                tables["trading_connection"].insert(),
                [
                    {
                        "id": 10,
                        "user_id": 1,
                        "provider": "kucoin",
                        "connection_type": "cex_api_key",
                        "encrypted_api_key": "encrypted-kucoin-key",
                        "encrypted_api_secret": "encrypted-kucoin-secret",
                        "encrypted_passphrase": "encrypted-kucoin-passphrase",
                        "wallet_address": None,
                        "is_active": True,
                        "verification_status": "verified",
                        "last_verified_at": NOW,
                        "last_verification_error": "",
                        "provider_metadata_json": "{}",
                        "created_at": NOW,
                        "updated_at": NOW,
                    },
                    {
                        "id": 11,
                        "user_id": 1,
                        "provider": "hyperliquid",
                        "connection_type": "cex_api_key",
                        "encrypted_api_key": "encrypted-hl-key",
                        "encrypted_api_secret": "encrypted-hl-secret",
                        "encrypted_passphrase": None,
                        "wallet_address": None,
                        "is_active": True,
                        "verification_status": "verified",
                        "last_verified_at": NOW,
                        "last_verification_error": "",
                        "provider_metadata_json": "{}",
                        "created_at": NOW,
                        "updated_at": NOW,
                    },
                    {
                        "id": 12,
                        "user_id": 2,
                        "provider": "debug",
                        "connection_type": "cex_api_key",
                        "encrypted_api_key": None,
                        "encrypted_api_secret": None,
                        "encrypted_passphrase": None,
                        "wallet_address": None,
                        "is_active": True,
                        "verification_status": "verified",
                        "last_verified_at": NOW,
                        "last_verification_error": "",
                        "provider_metadata_json": "{}",
                        "created_at": NOW,
                        "updated_at": NOW,
                    },
                ],
            )
            conn.execute(
                tables["deposit_address"].insert(),
                {
                    "id": 20,
                    "user_id": 1,
                    "asset": "USDC",
                    "network": "Arbitrum",
                    "address": "0xdeposit",
                    "version": 1,
                    "is_active": True,
                    "created_at": NOW,
                },
            )
            conn.execute(
                tables["wallet_account"].insert(),
                {
                    "id": 30,
                    "user_id": 1,
                    "provider": "self_custody",
                    "asset": "USDC",
                    "network": "Arbitrum",
                    "status": "active",
                    "encrypted_metadata_json": "{}",
                    "created_at": NOW,
                    "updated_at": NOW,
                },
            )
            conn.execute(
                tables["wallet_address"].insert(),
                {
                    "id": 40,
                    "wallet_account_id": 30,
                    "user_id": 1,
                    "deposit_address_id": 20,
                    "asset": "USDC",
                    "network": "Arbitrum",
                    "address": "0xwallet",
                    "status": "active",
                    "rotation_index": 1,
                    "encrypted_metadata_json": "{}",
                    "created_at": NOW,
                    "updated_at": NOW,
                },
            )
            conn.execute(
                tables["wallet_balance"].insert(),
                [
                    {
                        "id": 50,
                        "user_id": 1,
                        "active_deposit_address_id": 20,
                        "asset": "USDC",
                        "available_balance": 27.802657657549,
                        "locked_balance": 0.0,
                        "estimated_usd_value": 27.802657657549,
                        "created_at": NOW,
                        "updated_at": NOW,
                    },
                    {
                        "id": 51,
                        "user_id": 2,
                        "active_deposit_address_id": None,
                        "asset": "USDC",
                        "available_balance": 999.0,
                        "locked_balance": 0.0,
                        "estimated_usd_value": 999.0,
                        "created_at": NOW,
                        "updated_at": NOW,
                    },
                ],
            )
            conn.execute(
                tables["wallet_transaction"].insert(),
                {
                    "id": 60,
                    "user_id": 1,
                    "vault_cycle_id": 12345,
                    "asset": "USDC",
                    "amount": 27.802657657549,
                    "transaction_type": "deposit",
                    "status": "complete",
                    "network": "Arbitrum",
                    "withdraw_address": None,
                    "note": "restore test",
                    "created_at": NOW,
                },
            )
            conn.execute(
                tables["wallet_ledger_event"].insert(),
                {
                    "id": 70,
                    "user_id": 1,
                    "deposit_address_id": 20,
                    "wallet_address_id": 40,
                    "asset": "USDC",
                    "network": "Arbitrum",
                    "address": "0xwallet",
                    "event_type": "deposit",
                    "provider_reference": "chain-ref",
                    "idempotency_key": "ledger-key",
                    "amount": 27.802657657549,
                    "confirmations": 12,
                    "status": "complete",
                    "metadata_json": "{}",
                    "created_at": NOW,
                    "updated_at": NOW,
                },
            )
            conn.execute(
                tables["wallet_withdrawal"].insert(),
                {
                    "id": 80,
                    "user_id": 1,
                    "trading_connection_id": 10,
                    "wallet_account_id": 30,
                    "source_wallet_address_id": 40,
                    "source_deposit_address_id": 20,
                    "asset": "USDT",
                    "network": "Arbitrum",
                    "destination_address": "0xdestination",
                    "amount": 0.99,
                    "amount_eth": 0.0,
                    "fee_eth": 0.0,
                    "status": "failed",
                    "workflow_type": "manual_withdrawal",
                    "idempotency_token": "withdraw-token",
                    "provider_reference": "withdraw-ref",
                    "failure_reason": "receipt status 0x0",
                    "treasury_safety_status": "unchecked",
                    "treasury_estimated_gas_eth": 0.0,
                    "metadata_json": "{}",
                    "created_at": NOW,
                    "updated_at": NOW,
                },
            )
            conn.execute(
                tables["wallet_audit_log"].insert(),
                {
                    "id": 90,
                    "user_id": 1,
                    "wallet_account_id": 30,
                    "wallet_withdrawal_id": 80,
                    "category": "wallet",
                    "action": "withdrawal_failed",
                    "status": "recorded",
                    "message": "withdrawal failed",
                    "metadata_json": "{}",
                    "created_at": NOW,
                },
            )
            conn.execute(
                tables["audit_log"].insert(),
                {
                    "id": 100,
                    "user_id": 1,
                    "trading_connection_id": 10,
                    "category": "account_restore",
                    "action": "seed",
                    "message": "seeded",
                    "metadata_json": "{}",
                    "created_at": NOW,
                },
            )
    finally:
        engine.dispose()


def _fetch_all(path: Path, table_name: str):
    engine = create_engine(_sqlite_url(path), future=True)
    try:
        table = db.metadata.tables[table_name]
        with engine.connect() as conn:
            return [dict(row._mapping) for row in conn.execute(select(table)).fetchall()]
    finally:
        engine.dispose()


def test_import_account_dry_run_does_not_mutate_target(tmp_path):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _create_schema(source)
    _create_schema(target)
    _seed_source(source)

    summary = import_account(
        username="sufyanh",
        source=source,
        target_url=_sqlite_url(target),
        include_connections=True,
        dry_run=True,
    )

    assert summary.dry_run is True
    assert summary.selected_counts["user"] == 1
    assert summary.selected_counts["trading_connection"] == 2
    assert _fetch_all(target, "user") == []


def test_import_account_is_idempotent_and_excludes_debug_users(tmp_path):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _create_schema(source)
    _create_schema(target)
    _seed_source(source)

    first = import_account(
        username="sufyanh",
        source=source,
        target_url=_sqlite_url(target),
        include_connections=True,
        dry_run=False,
    )
    second = import_account(
        username="sufyanh",
        source=source,
        target_url=_sqlite_url(target),
        include_connections=True,
        dry_run=False,
    )

    users = _fetch_all(target, "user")
    assert [(row["id"], row["username"]) for row in users] == [(1, "sufyanh")]
    assert len(_fetch_all(target, "trading_connection")) == 2
    assert len(_fetch_all(target, "wallet_balance")) == 1
    assert len(_fetch_all(target, "wallet_transaction")) == 1
    assert _fetch_all(target, "wallet_transaction")[0]["vault_cycle_id"] is None
    assert second.deleted_counts["user"] == 1
    assert first.inserted_counts == second.inserted_counts

    payload = summary_to_dict(second)
    assert "debugmax2" not in str(payload)
    assert "encrypted_api_secret" not in str(payload)
    assert payload["wallet"]["available_assets"]["USDC"] == 27.802657657549


def test_import_account_without_connections_nulls_connection_references(tmp_path):
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _create_schema(source)
    _create_schema(target)
    _seed_source(source)

    summary = import_account(
        username="sufyanh",
        source=source,
        target_url=_sqlite_url(target),
        include_connections=False,
        dry_run=False,
    )

    assert summary.inserted_counts["trading_connection"] == 0
    assert _fetch_all(target, "trading_connection") == []
    assert _fetch_all(target, "wallet_withdrawal")[0]["trading_connection_id"] is None
    assert _fetch_all(target, "audit_log")[0]["trading_connection_id"] is None
