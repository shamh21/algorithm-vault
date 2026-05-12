#!/usr/bin/env python
"""Import one restored local account graph into the configured SQL database.

The script is intentionally narrow: it copies the selected user's auth,
wallet, ledger, audit, withdrawal, and optional encrypted exchange connection
rows from a local SQLite backup into the target DATABASE_URL.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import MetaData, create_engine, delete, insert, select, text
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.sql.sqltypes import Boolean, DateTime


INSERT_ORDER = (
    "user",
    "trading_connection",
    "deposit_address",
    "wallet_account",
    "wallet_address",
    "wallet_balance",
    "wallet_transaction",
    "wallet_ledger_event",
    "wallet_withdrawal",
    "wallet_audit_log",
    "audit_log",
)

DELETE_ORDER = tuple(reversed(INSERT_ORDER))
SECRET_COLUMNS = {
    "password_hash",
    "totp_secret_encrypted",
    "encrypted_api_key",
    "encrypted_api_secret",
    "encrypted_passphrase",
    "encrypted_metadata_json",
}


@dataclass(frozen=True)
class ImportSummary:
    username: str
    dry_run: bool
    source_user_id: int
    include_connections: bool
    target_backend: str
    selected_counts: dict[str, int]
    deleted_counts: dict[str, int]
    inserted_counts: dict[str, int]
    total_wallet_usd: float
    wallet_assets: dict[str, float]


class ImportAccountError(RuntimeError):
    """Raised when the account import cannot be completed safely."""


def normalize_database_url(raw_url: str) -> str:
    """Use psycopg v3 for Postgres URLs when no explicit SQLAlchemy driver is set."""

    value = (raw_url or "").strip()
    if value.startswith("postgres://"):
        return "postgresql+psycopg://" + value.removeprefix("postgres://")
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value.removeprefix("postgresql://")
    return value


def import_account(
    *,
    username: str,
    source: Path,
    target_url: str,
    include_connections: bool,
    dry_run: bool = False,
) -> ImportSummary:
    if not username.strip():
        raise ImportAccountError("username is required")
    if not source.exists():
        raise ImportAccountError(f"source database does not exist: {source}")
    if not target_url:
        raise ImportAccountError("target DATABASE_URL is required")

    normalized_url = normalize_database_url(target_url)
    source_conn = sqlite3.connect(source)
    source_conn.row_factory = sqlite3.Row
    try:
        selected = _collect_source_rows(source_conn, username=username, include_connections=include_connections)
        source_user_id = int(selected["user"][0]["id"])
        balances = selected.get("wallet_balance", [])
        wallet_assets = {str(row["asset"]): float(row["available_balance"] or 0.0) for row in balances}
        total_wallet_usd = sum(float(row["estimated_usd_value"] or 0.0) for row in balances)

        engine = create_engine(normalized_url, future=True)
        try:
            target_metadata = MetaData()
            target_metadata.reflect(bind=engine)
            _validate_target_tables(target_metadata, selected)

            if dry_run:
                return ImportSummary(
                    username=username,
                    dry_run=True,
                    source_user_id=source_user_id,
                    include_connections=include_connections,
                    target_backend=make_url(normalized_url).get_backend_name(),
                    selected_counts=_counts(selected),
                    deleted_counts={table: 0 for table in DELETE_ORDER},
                    inserted_counts={table: 0 for table in INSERT_ORDER},
                    total_wallet_usd=total_wallet_usd,
                    wallet_assets=wallet_assets,
                )

            with engine.begin() as target:
                deleted_counts = _delete_existing_account(target, target_metadata, username=username)
                _assert_no_id_conflicts(target, target_metadata, selected, username=username)
                inserted_counts = _insert_selected_rows(target, target_metadata, selected)
                _reset_postgres_sequences(target, target_metadata, selected)

            return ImportSummary(
                username=username,
                dry_run=False,
                source_user_id=source_user_id,
                include_connections=include_connections,
                target_backend=make_url(normalized_url).get_backend_name(),
                selected_counts=_counts(selected),
                deleted_counts=deleted_counts,
                inserted_counts=inserted_counts,
                total_wallet_usd=total_wallet_usd,
                wallet_assets=wallet_assets,
            )
        finally:
            engine.dispose()
    finally:
        source_conn.close()


def _collect_source_rows(
    source_conn: sqlite3.Connection,
    *,
    username: str,
    include_connections: bool,
) -> OrderedDict[str, list[dict[str, Any]]]:
    tables = _source_tables(source_conn)
    if "user" not in tables:
        raise ImportAccountError("source database is missing user table")

    user_rows = _fetch_source_rows(source_conn, "user", "username = ?", (username,))
    if not user_rows:
        raise ImportAccountError(f"source user not found: {username}")
    if len(user_rows) > 1:
        raise ImportAccountError(f"source has duplicate username rows: {username}")

    user_row = dict(user_rows[0])
    user_row["referral_invite_code_id"] = None
    source_user_id = int(user_row["id"])

    selected: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    selected["user"] = [user_row]
    selected["trading_connection"] = (
        _fetch_source_rows(source_conn, "trading_connection", "user_id = ?", (source_user_id,))
        if include_connections and "trading_connection" in tables
        else []
    )

    for table in (
        "deposit_address",
        "wallet_account",
        "wallet_address",
        "wallet_balance",
        "wallet_transaction",
        "wallet_ledger_event",
        "wallet_withdrawal",
        "wallet_audit_log",
        "audit_log",
    ):
        selected[table] = (
            _fetch_source_rows(source_conn, table, "user_id = ?", (source_user_id,)) if table in tables else []
        )

    return _sanitize_rows(selected, include_connections=include_connections)


def _source_tables(source_conn: sqlite3.Connection) -> set[str]:
    rows = source_conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {str(row[0]) for row in rows}


def _fetch_source_rows(
    source_conn: sqlite3.Connection,
    table: str,
    where_sql: str,
    params: tuple[Any, ...],
) -> list[dict[str, Any]]:
    quoted_table = '"' + table.replace('"', '""') + '"'
    rows = source_conn.execute(f"SELECT * FROM {quoted_table} WHERE {where_sql} ORDER BY id", params).fetchall()
    return [dict(row) for row in rows]


def _sanitize_rows(
    selected: OrderedDict[str, list[dict[str, Any]]],
    *,
    include_connections: bool,
) -> OrderedDict[str, list[dict[str, Any]]]:
    deposit_ids = _ids(selected.get("deposit_address", []))
    account_ids = _ids(selected.get("wallet_account", []))
    address_ids = _ids(selected.get("wallet_address", []))
    connection_ids = _ids(selected.get("trading_connection", [])) if include_connections else set()
    withdrawal_ids = _ids(selected.get("wallet_withdrawal", []))

    for row in selected.get("deposit_address", []):
        if row.get("rotated_from_id") not in deposit_ids:
            row["rotated_from_id"] = None

    sanitized_addresses = []
    for row in selected.get("wallet_address", []):
        if row.get("wallet_account_id") not in account_ids:
            continue
        if row.get("deposit_address_id") not in deposit_ids:
            row["deposit_address_id"] = None
        if row.get("rotated_from_id") not in address_ids:
            row["rotated_from_id"] = None
        sanitized_addresses.append(row)
    selected["wallet_address"] = sanitized_addresses
    address_ids = _ids(selected.get("wallet_address", []))

    for row in selected.get("wallet_balance", []):
        if row.get("active_deposit_address_id") not in deposit_ids:
            row["active_deposit_address_id"] = None

    for row in selected.get("wallet_transaction", []):
        row["vault_cycle_id"] = None

    for row in selected.get("wallet_ledger_event", []):
        if row.get("deposit_address_id") not in deposit_ids:
            row["deposit_address_id"] = None
        if row.get("wallet_address_id") not in address_ids:
            row["wallet_address_id"] = None

    for row in selected.get("wallet_withdrawal", []):
        if row.get("trading_connection_id") not in connection_ids:
            row["trading_connection_id"] = None
        if row.get("wallet_account_id") not in account_ids:
            row["wallet_account_id"] = None
        if row.get("source_wallet_address_id") not in address_ids:
            row["source_wallet_address_id"] = None
        if row.get("source_deposit_address_id") not in deposit_ids:
            row["source_deposit_address_id"] = None

    for row in selected.get("wallet_audit_log", []):
        if row.get("wallet_account_id") not in account_ids:
            row["wallet_account_id"] = None
        if row.get("wallet_withdrawal_id") not in withdrawal_ids:
            row["wallet_withdrawal_id"] = None

    for row in selected.get("audit_log", []):
        if row.get("trading_connection_id") not in connection_ids:
            row["trading_connection_id"] = None

    return selected


def _ids(rows: Iterable[dict[str, Any]]) -> set[int]:
    return {int(row["id"]) for row in rows if row.get("id") is not None}


def _validate_target_tables(metadata: MetaData, selected: OrderedDict[str, list[dict[str, Any]]]) -> None:
    missing = [table for table, rows in selected.items() if rows and table not in metadata.tables]
    if missing:
        raise ImportAccountError(f"target database is missing required tables: {', '.join(missing)}")


def _delete_existing_account(
    target: Connection,
    metadata: MetaData,
    *,
    username: str,
) -> dict[str, int]:
    user_table = metadata.tables["user"]
    existing_user_ids = [
        int(row[0])
        for row in target.execute(select(user_table.c.id).where(user_table.c.username == username)).fetchall()
    ]
    counts = {table: 0 for table in DELETE_ORDER}
    if not existing_user_ids:
        return counts

    for table_name in DELETE_ORDER:
        if table_name not in metadata.tables:
            continue
        table = metadata.tables[table_name]
        if table_name == "user":
            result = target.execute(delete(table).where(table.c.username == username))
        elif "user_id" in table.c:
            result = target.execute(delete(table).where(table.c.user_id.in_(existing_user_ids)))
        else:
            continue
        counts[table_name] = int(result.rowcount or 0)
    return counts


def _assert_no_id_conflicts(
    target: Connection,
    metadata: MetaData,
    selected: OrderedDict[str, list[dict[str, Any]]],
    *,
    username: str,
) -> None:
    for table_name, rows in selected.items():
        if not rows or table_name not in metadata.tables:
            continue
        table = metadata.tables[table_name]
        ids = sorted(_ids(rows))
        if not ids or "id" not in table.c:
            continue
        existing = [int(row[0]) for row in target.execute(select(table.c.id).where(table.c.id.in_(ids))).fetchall()]
        if existing:
            raise ImportAccountError(
                f"target {table_name} id conflict after deleting {username}: {existing[:10]}"
            )


def _insert_selected_rows(
    target: Connection,
    metadata: MetaData,
    selected: OrderedDict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    counts = {table: 0 for table in INSERT_ORDER}
    for table_name in INSERT_ORDER:
        rows = selected.get(table_name) or []
        if not rows or table_name not in metadata.tables:
            continue
        table = metadata.tables[table_name]
        prepared = [_prepare_row_for_target(table, row) for row in rows]
        if prepared:
            target.execute(insert(table), prepared)
        counts[table_name] = len(prepared)
    return counts


def _prepare_row_for_target(table: Any, row: dict[str, Any]) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for column in table.columns:
        if column.name not in row:
            continue
        prepared[column.name] = _coerce_value(column, row[column.name])
    return prepared


def _coerce_value(column: Any, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(column.type, Boolean):
        return bool(value)
    if isinstance(column.type, DateTime) and isinstance(value, str):
        normalized = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return value
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    return value


def _reset_postgres_sequences(
    target: Connection,
    metadata: MetaData,
    selected: OrderedDict[str, list[dict[str, Any]]],
) -> None:
    if target.dialect.name != "postgresql":
        return
    preparer = target.dialect.identifier_preparer
    for table_name, rows in selected.items():
        if not rows or table_name not in metadata.tables:
            continue
        table = metadata.tables[table_name]
        if "id" not in table.c:
            continue
        quoted = preparer.quote(table_name)
        target.execute(
            text(
                "SELECT setval("
                "pg_get_serial_sequence(:table_name, 'id'), "
                f"COALESCE((SELECT MAX(id) FROM {quoted}), 1), "
                f"(SELECT MAX(id) FROM {quoted}) IS NOT NULL"
                ")"
            ),
            {"table_name": table_name},
        )


def _counts(selected: OrderedDict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {table: len(rows) for table, rows in selected.items()}


def summary_to_dict(summary: ImportSummary) -> dict[str, Any]:
    return {
        "username": summary.username,
        "dry_run": summary.dry_run,
        "source_user_id": summary.source_user_id,
        "include_connections": summary.include_connections,
        "target_backend": summary.target_backend,
        "selected_counts": summary.selected_counts,
        "deleted_counts": summary.deleted_counts,
        "inserted_counts": summary.inserted_counts,
        "wallet": {
            "available_assets": summary.wallet_assets,
            "portfolio_total_usd": summary.total_wallet_usd,
        },
    }


def print_summary(summary: ImportSummary) -> None:
    payload = summary_to_dict(summary)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if any(column in json.dumps(payload) for column in SECRET_COLUMNS):
        raise ImportAccountError("summary unexpectedly included secret column names")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True, help="Username to import from the source SQLite database.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("instance/algorithm_vault_local_production.db"),
        help="Path to the restored local SQLite database.",
    )
    parser.add_argument(
        "--target-url",
        default=os.getenv("DATABASE_URL", ""),
        help="Target SQLAlchemy database URL. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--include-connections",
        action="store_true",
        help="Include encrypted exchange connection rows for the selected account.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Inspect rows without changing the target database.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv or sys.argv[1:]))
    try:
        summary = import_account(
            username=args.username,
            source=args.source,
            target_url=args.target_url,
            include_connections=bool(args.include_connections),
            dry_run=bool(args.dry_run),
        )
        print_summary(summary)
        return 0
    except ImportAccountError as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
