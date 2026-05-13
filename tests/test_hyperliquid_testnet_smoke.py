from __future__ import annotations

import base64
import hashlib
import os
import sqlite3
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet, InvalidToken

from app.services.hyperliquid_client import HYPERLIQUID_TESTNET_API_URL, HyperliquidClient

pytestmark = [pytest.mark.integration, pytest.mark.live]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _smoke_db_path() -> Path:
    configured = os.getenv("HYPERLIQUID_SMOKE_DB", "").strip()
    if configured:
        return Path(configured).expanduser()
    return _repo_root() / "instance" / "hyperliquid_dashboard.db"


def _read_only_env_enabled() -> bool:
    return (
        os.getenv("HYPERLIQUID_ACCOUNT", "").strip() == "sufyanh"
        and os.getenv("HYPERLIQUID_ENV", "").strip().lower() == "testnet"
        and os.getenv("HYPERLIQUID_BASE_URL", "").strip().rstrip("/") == HYPERLIQUID_TESTNET_API_URL
    )


def _signed_env_enabled() -> bool:
    return _read_only_env_enabled() and os.getenv("RUN_HYPERLIQUID_LIVE_TESTS", "").strip() == "1"


def _saved_hyperliquid_connection() -> dict[str, Any]:
    db_path = _smoke_db_path()
    if not db_path.exists():
        pytest.fail(f"Hyperliquid smoke DB not found: {db_path}")
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        """
        select tc.id, tc.wallet_address, tc.encrypted_api_secret, tc.encrypted_api_key
        from trading_connection tc
        join user u on u.id = tc.user_id
        where lower(u.username) = 'sufyanh'
          and tc.provider = 'hyperliquid'
        order by tc.is_active desc, tc.updated_at desc, tc.id desc
        limit 1
        """
    ).fetchone()
    if row is None:
        pytest.fail("Saved Hyperliquid connection for account/profile sufyanh was not found.")
    return dict(row)


def _decrypt_saved_secret(encrypted: str) -> str:
    configured_key = os.getenv("TOTP_ENCRYPTION_KEY", "").strip()
    if configured_key:
        key = configured_key.encode("utf-8")
    else:
        raw = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me").encode("utf-8")
        key = base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
    try:
        return Fernet(key).decrypt(encrypted.encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError) as exc:
        raise RuntimeError("saved_hyperliquid_secret_decrypt_failed: set TOTP_ENCRYPTION_KEY for the saved sufyanh connection.") from exc


def _client_from_saved_connection(*, require_secret: bool) -> HyperliquidClient:
    saved = _saved_hyperliquid_connection()
    account_address = str(saved.get("wallet_address") or "").strip()
    if not account_address:
        pytest.fail("Saved sufyanh Hyperliquid connection is missing the main account address.")

    secret = ""
    if require_secret:
        encrypted_secret = str(saved.get("encrypted_api_secret") or "").strip()
        if not encrypted_secret:
            pytest.fail("Saved sufyanh Hyperliquid connection is missing an encrypted API wallet secret.")
        secret = _decrypt_saved_secret(encrypted_secret)

    return HyperliquidClient(
        {
            "ENABLE_LIVE_TRADING": True,
            "HL_ACCOUNT_ADDRESS": account_address,
            "HL_SECRET_KEY": secret,
            "HL_TESTNET_BASE_URL": HYPERLIQUID_TESTNET_API_URL,
            "HL_MAINNET_BASE_URL": "https://api.hyperliquid.xyz",
            "HYPERLIQUID_ACCOUNT": os.getenv("HYPERLIQUID_ACCOUNT", "").strip(),
            "HYPERLIQUID_ENV": os.getenv("HYPERLIQUID_ENV", "").strip().lower(),
            "HYPERLIQUID_BASE_URL": os.getenv("HYPERLIQUID_BASE_URL", "").strip(),
            "RUN_HYPERLIQUID_LIVE_TESTS": os.getenv("RUN_HYPERLIQUID_LIVE_TESTS", "").strip() == "1",
            "HYPERLIQUID_LIVE_TEST_MAX_NOTIONAL_USD": float(os.getenv("HYPERLIQUID_LIVE_TEST_MAX_NOTIONAL_USD", "0") or 0),
            "HYPERLIQUID_MIN_ORDER_VALUE_USD": float(os.getenv("HYPERLIQUID_MIN_ORDER_VALUE_USD", "10") or 10),
            "EXCHANGE_RETRY_ATTEMPTS": 2,
            "EXCHANGE_RETRY_SLEEP_SECONDS": 0.25,
            "HL_TIMEOUT_SECONDS": 10.0,
        }
    )


def _assert_no_existing_testnet_positions(client: HyperliquidClient) -> None:
    positions = client.get_positions("testnet", retry=False)
    active = [row for row in positions if abs(float(row.get("quantity") or 0.0)) > 1e-9]
    if active:
        pytest.fail("unexpected_existing_testnet_positions: refusing smoke trade while the saved testnet account has open positions.")


def test_hyperliquid_testnet_read_only_smoke_for_saved_sufyanh_connection() -> None:
    if not _read_only_env_enabled():
        pytest.skip(
            "Set HYPERLIQUID_ACCOUNT=sufyanh, HYPERLIQUID_ENV=testnet, and "
            f"HYPERLIQUID_BASE_URL={HYPERLIQUID_TESTNET_API_URL} to run read-only testnet smoke."
        )

    client = _client_from_saved_connection(require_secret=False)
    meta, contexts = client.get_perp_meta_and_asset_contexts("testnet")
    universe = meta.get("universe", []) if isinstance(meta, dict) else []
    mids = client.get_all_mids("testnet")
    symbol = client.select_live_test_symbol("testnet")
    book = client.get_order_book("testnet", symbol)
    balances = client.get_balances("testnet")
    positions = client.get_positions("testnet")
    open_orders = client.get_open_orders("testnet")
    fills = client.get_recent_fills("testnet")

    assert universe
    assert contexts or isinstance(contexts, list)
    assert {"BTC", "ETH"} & {str(item.get("name")) for item in universe if isinstance(item, dict)}
    assert mids.get(symbol, 0.0) > 0
    assert isinstance(book, dict)
    assert any(row.get("asset") == "USDC" and row.get("type") == "margin" for row in balances)
    assert isinstance(positions, list)
    assert isinstance(open_orders, list)
    assert isinstance(fills, list)


def test_hyperliquid_testnet_alo_place_cancel_smoke_for_saved_sufyanh_connection() -> None:
    if not _signed_env_enabled():
        pytest.skip(
            "Set the read-only Hyperliquid testnet env vars plus RUN_HYPERLIQUID_LIVE_TESTS=1 "
            "to run the signed ALO place/cancel smoke."
        )

    client = _client_from_saved_connection(require_secret=True)
    _assert_no_existing_testnet_positions(client)
    plan = client.live_test_order_plan("testnet", side="buy", require_funds=True)
    client_order_id = f"codex-hl-test-{uuid.uuid4().hex}"
    symbol = str(plan["symbol"])
    exchange_order_id = ""

    try:
        response = client.place_order(
            "testnet",
            symbol,
            str(plan["side"]),
            float(plan["quantity"]),
            "limit",
            float(plan["limit_price"]),
            False,
            1.0,
            0.0,
            client_order_id=client_order_id,
            time_in_force="Alo",
        )
        exchange_order_id = str(response.get("exchange_order_id") or "")
        assert response["status"] == "open", response

        status = client.get_order_status("testnet", exchange_order_id=exchange_order_id, client_order_id=client_order_id)
        assert status["status"] in {"open", "unknown"}

        open_order_ids = {str(order.get("order_id")) for order in client.get_open_orders("testnet", retry=False)}
        assert exchange_order_id in open_order_ids
    finally:
        if exchange_order_id:
            client.cancel_order("testnet", symbol, exchange_order_id, client_order_id=client_order_id)
        client.flatten_all_positions("testnet")

    open_order_ids = {str(order.get("order_id")) for order in client.get_open_orders("testnet", retry=False)}
    assert exchange_order_id not in open_order_ids


def test_hyperliquid_testnet_optional_ioc_fill_and_flatten_smoke_for_saved_sufyanh_connection() -> None:
    if not _signed_env_enabled() or os.getenv("RUN_HYPERLIQUID_FILL_TEST", "").strip() != "1":
        pytest.skip("Set RUN_HYPERLIQUID_LIVE_TESTS=1 and RUN_HYPERLIQUID_FILL_TEST=1 to run the optional IOC fill smoke.")

    client = _client_from_saved_connection(require_secret=True)
    _assert_no_existing_testnet_positions(client)
    plan = client.live_test_order_plan("testnet", side="buy", require_funds=True)
    client_order_id = f"codex-hl-test-fill-{uuid.uuid4().hex}"
    symbol = str(plan["symbol"])
    exchange_order_id = ""

    try:
        response = client.place_order(
            "testnet",
            symbol,
            str(plan["side"]),
            float(plan["quantity"]),
            "market",
            None,
            False,
            1.0,
            0.001,
            client_order_id=client_order_id,
            time_in_force="Ioc",
        )
        exchange_order_id = str(response.get("exchange_order_id") or "")
        assert response["status"] == "filled", response
        assert float(response.get("filled_quantity") or 0) > 0

        fills = client.get_recent_fills("testnet", retry=False)
        assert any(str(fill.get("exchange_order_id")) == exchange_order_id for fill in fills)
        assert isinstance(client.get_balances("testnet", retry=False), list)
        assert isinstance(client.get_positions("testnet", retry=False), list)
    finally:
        client.flatten_all_positions("testnet")
        with suppress(Exception):
            client.cancel_order("testnet", symbol, exchange_order_id, client_order_id=client_order_id)
