from __future__ import annotations

import os
import time
import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from app.services.live_provider_adapters import KucoinFuturesConnector

pytestmark = [pytest.mark.integration, pytest.mark.live]


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_config() -> dict[str, Any]:
    return {
        "ENABLE_LIVE_TRADING": True,
        "KUCOIN_DEFAULT_MARKET_TYPE": "spot",
        "KUCOIN_SPOT_BASE_URL": os.getenv("KUCOIN_SPOT_BASE_URL", "https://api.kucoin.com").strip(),
        "KUCOIN_TEST_ACCOUNT": os.getenv("KUCOIN_TEST_ACCOUNT", "").strip(),
        "KUCOIN_TEST_SYMBOL": os.getenv("KUCOIN_TEST_SYMBOL", "").strip(),
        "KUCOIN_MAX_TEST_NOTIONAL_USDT": os.getenv("KUCOIN_MAX_TEST_NOTIONAL_USDT", "").strip(),
        "KUCOIN_ENABLE_LIVE_TEST_TRADES": _env_bool("KUCOIN_ENABLE_LIVE_TEST_TRADES"),
        "KUCOIN_ENABLE_FILL_TEST": _env_bool("KUCOIN_ENABLE_FILL_TEST"),
        "KUCOIN_TIME_SYNC_ENABLED": True,
        "KUCOIN_TIME_SYNC_TTL_SECONDS": 300,
        "PROVIDER_RETRY_ATTEMPTS": 2,
        "PROVIDER_RETRY_SLEEP_SECONDS": 0.25,
        "PROVIDER_TIMEOUT_SECONDS": 10,
    }


def _env_credentials() -> SimpleNamespace:
    return SimpleNamespace(
        api_key=os.getenv("KUCOIN_API_KEY", "").strip(),
        api_secret=os.getenv("KUCOIN_API_SECRET", "").strip(),
        passphrase=os.getenv("KUCOIN_API_PASSPHRASE", "").strip(),
        wallet_address="",
    )


def _connector() -> KucoinFuturesConnector:
    credentials = _env_credentials()
    missing = [
        name
        for name, value in (
            ("KUCOIN_API_KEY", credentials.api_key),
            ("KUCOIN_API_SECRET", credentials.api_secret),
            ("KUCOIN_API_PASSPHRASE", credentials.passphrase),
        )
        if not value
    ]
    if missing:
        config = _env_config()
        print(
            "KuCoin live preflight: "
            f"account={config['KUCOIN_TEST_ACCOUNT'] or '[missing]'} "
            f"symbol={config['KUCOIN_TEST_SYMBOL'] or '[missing]'} "
            f"max_notional_usdt={config['KUCOIN_MAX_TEST_NOTIONAL_USDT'] or None} "
            f"live_trading_enabled={config['KUCOIN_ENABLE_LIVE_TEST_TRADES']} "
            f"fill_test_enabled={config['KUCOIN_ENABLE_FILL_TEST']} "
            "credentials_present={'api_key': False, 'api_secret': False, 'api_passphrase': False}"
        )
        pytest.skip("KuCoin live smoke skipped: " + "; ".join(f"{name} is required" for name in missing))
    return KucoinFuturesConnector(_env_config(), credentials)


def _print_preflight(connector: KucoinFuturesConnector) -> None:
    summary = connector.kucoin_live_test_preflight_summary()
    print(
        "KuCoin live preflight: "
        f"account={summary['account'] or '[missing]'} "
        f"symbol={summary['symbol'] or '[missing]'} "
        f"max_notional_usdt={summary['max_notional_usdt']} "
        f"live_trading_enabled={summary['live_trading_enabled']} "
        f"fill_test_enabled={summary['fill_test_enabled']} "
        f"credentials_present={summary['credentials_present']}"
    )


def _skip_if_guarded(connector: KucoinFuturesConnector, *, require_live_trading: bool = False, require_fill: bool = False) -> None:
    _print_preflight(connector)
    blockers = connector.kucoin_live_test_guard_errors(require_live_trading=require_live_trading, require_fill=require_fill)
    if blockers:
        pytest.skip("KuCoin live smoke skipped: " + "; ".join(blockers))


def _poll_status(connector: KucoinFuturesConnector, symbol: str, client_oid: str, attempts: int = 6) -> dict[str, Any]:
    last: dict[str, Any] = {"status": "unknown", "client_order_id": client_oid}
    for _ in range(max(1, attempts)):
        try:
            last = connector.get_spot_order_status("live", symbol, client_order_id=client_oid)
        except Exception as exc:  # noqa: BLE001
            last = {"status": "unknown", "client_order_id": client_oid, "error": str(exc)}
        if last.get("status") in {"open", "filled", "cancelled", "rejected"}:
            return last
        time.sleep(0.5)
    return last


def test_kucoin_spot_live_read_only_preflight_for_sufyanh() -> None:
    connector = _connector()
    _skip_if_guarded(connector)

    connector._sync_server_time(force=True)
    discovery = connector.discover_spot_accounts("live")
    plan = connector.kucoin_spot_live_test_plan(require_funds=True)

    assert discovery["account"] == "sufyanh"
    assert discovery["spot_accounts"]
    assert plan["symbol"] == connector._spot_symbol(os.getenv("KUCOIN_TEST_SYMBOL"))
    assert plan["available_quote"] >= plan["notional_usdt"]


def test_kucoin_spot_live_test_order_endpoint_for_sufyanh() -> None:
    connector = _connector()
    _skip_if_guarded(connector)
    plan = connector.kucoin_spot_live_test_plan(require_funds=True)
    client_oid = f"codex-kucoin-test-{uuid.uuid4().hex[:20]}"

    response = connector.create_spot_test_order(
        "live",
        str(plan["symbol"]),
        str(plan["side"]),
        str(plan["quantity"]),
        "limit",
        str(plan["limit_price"]),
        client_order_id=client_oid,
        time_in_force="GTC",
        post_only=True,
    )

    assert response["test_order"] is True
    assert response["client_order_id"] == client_oid
    assert response["status"] in {"submitted", "open"}
    assert response.get("exchange_order_id")


def test_kucoin_spot_live_post_only_place_cancel_for_sufyanh() -> None:
    connector = _connector()
    _skip_if_guarded(connector, require_live_trading=True)
    plan = connector.kucoin_spot_live_test_plan(require_funds=True, post_only=True)
    client_oid = f"codex-kucoin-live-{uuid.uuid4().hex[:20]}"
    exchange_order_id = ""

    try:
        placed = connector.place_spot_order(
            "live",
            str(plan["symbol"]),
            str(plan["side"]),
            str(plan["quantity"]),
            "limit",
            str(plan["limit_price"]),
            client_order_id=client_oid,
            time_in_force="GTC",
            post_only=True,
        )
        exchange_order_id = str(placed.get("exchange_order_id") or "")
        assert placed["client_order_id"] == client_oid
        assert placed["status"] in {"submitted", "open"}

        status = _poll_status(connector, str(plan["symbol"]), client_oid)
        assert status["status"] in {"open", "submitted", "unknown"}
    finally:
        if exchange_order_id or client_oid:
            connector.cancel_spot_order("live", str(plan["symbol"]), exchange_order_id=exchange_order_id, client_order_id=client_oid)

    cancelled = _poll_status(connector, str(plan["symbol"]), client_oid)
    assert cancelled["status"] in {"cancelled", "unknown"}


def test_kucoin_spot_live_optional_fill_round_trip_for_sufyanh() -> None:
    connector = _connector()
    _skip_if_guarded(connector, require_live_trading=True, require_fill=True)
    plan = connector.kucoin_spot_live_test_plan(require_funds=True, post_only=False)
    symbol = str(plan["symbol"])
    buy_client_oid = f"codex-kucoin-fill-buy-{uuid.uuid4().hex[:16]}"
    sell_client_oid = f"codex-kucoin-fill-sell-{uuid.uuid4().hex[:16]}"
    bought_size = 0.0

    buy = connector.place_spot_order(
        "live",
        symbol,
        "buy",
        None,
        "market",
        client_order_id=buy_client_oid,
        funds=connector._decimal(plan["notional_usdt"]),
    )
    buy_status = _poll_status(connector, symbol, buy_client_oid)
    bought_size = float(buy_status.get("filled_quantity") or buy.get("filled_quantity") or 0.0)
    if bought_size <= 0:
        fills = connector.get_spot_recent_fills("live", symbol, limit=10)
        matching = [fill for fill in fills if str(fill.get("exchange_order_id")) == str(buy.get("exchange_order_id"))]
        bought_size = max((float(fill.get("size") or 0.0) for fill in matching), default=0.0)
    if bought_size <= 0:
        pytest.fail("KuCoin fill smoke could not confirm a filled buy size; stopping before sell.")

    try:
        sell = connector.place_spot_order(
            "live",
            symbol,
            "sell",
            connector._decimal(bought_size),
            "market",
            client_order_id=sell_client_oid,
        )
        sell_status = _poll_status(connector, symbol, sell_client_oid)
        assert sell["status"] in {"submitted", "filled"}
        assert sell_status["status"] in {"filled", "unknown"}
    finally:
        print(f"KuCoin optional fill smoke notional_usdt={plan['notional_usdt']:.8f} bought_size={bought_size:.12f}")
