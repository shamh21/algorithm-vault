from __future__ import annotations

import base64
import hashlib
import hmac
import json
from types import SimpleNamespace
from typing import Any

import pytest

import app.services.live_provider_adapters as adapters
from app.services.live_provider_adapters import KucoinFuturesConnector


def _credentials() -> SimpleNamespace:
    return SimpleNamespace(api_key="key", api_secret="secret", passphrase="passphrase", wallet_address="")


def _connector(config: dict[str, Any] | None = None) -> KucoinFuturesConnector:
    merged = {
        "ENABLE_LIVE_TRADING": True,
        "KUCOIN_SPOT_BASE_URL": "https://spot.example.test",
        "KUCOIN_TIME_SYNC_ENABLED": False,
        "PROVIDER_RETRY_ATTEMPTS": 3,
        "PROVIDER_RETRY_SLEEP_SECONDS": 0,
    }
    merged.update(config or {})
    return KucoinFuturesConnector(merged, _credentials())


def test_kucoin_spot_signed_request_uses_compact_body_and_no_unsafe_retry(monkeypatch) -> None:
    monkeypatch.setattr(adapters.time, "time", lambda: 1_700_000_000.0)
    calls: list[dict[str, Any]] = []

    def fake_request(session, method, url, *, provider, attempts, sleep_seconds, timeout, **kwargs):
        calls.append({"method": method, "url": url, "attempts": attempts, **kwargs})
        return {"code": "200000", "data": {"orderId": "order-1", "clientOid": "client-1"}}

    monkeypatch.setattr(adapters, "_request_with_retries", fake_request)
    connector = _connector()
    body = {"symbol": "BTC-USDT", "side": "buy", "type": "limit", "price": "100", "size": "0.01"}

    connector._signed_spot("POST", "/api/v1/hf/orders/test", body=body)

    call = calls[0]
    body_text = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
    pre_sign = f"1700000000000POST/api/v1/hf/orders/test{body_text}"
    expected_signature = base64.b64encode(hmac.new(b"secret", pre_sign.encode("utf-8"), hashlib.sha256).digest()).decode("utf-8")
    expected_passphrase = base64.b64encode(hmac.new(b"secret", b"passphrase", hashlib.sha256).digest()).decode("utf-8")
    assert call["attempts"] == 1
    assert call["data"] == body_text
    assert call["headers"]["KC-API-TIMESTAMP"] == "1700000000000"
    assert call["headers"]["KC-API-SIGN"] == expected_signature
    assert call["headers"]["KC-API-PASSPHRASE"] == expected_passphrase
    assert call["headers"]["KC-API-KEY-VERSION"] == "2"


def test_kucoin_spot_symbol_normalization() -> None:
    connector = _connector({"KUCOIN_SPOT_SYMBOL_MAP_JSON": '{"XBT":"BTC-USDT","DOGE":"DOGE-USDT"}'})

    assert connector._spot_symbol("btc-usdt") == "BTC-USDT"
    assert connector._spot_symbol("btc/usdt") == "BTC-USDT"
    assert connector._spot_symbol("DOGE") == "DOGE-USDT"
    assert connector._spot_symbol("SOL") == "SOL-USDT"
    assert connector._internal_spot_symbol("BTC-USDT") == "XBT"
    assert connector._internal_spot_symbol("ETH-USDT") == "ETH"


def test_kucoin_spot_balance_parsing() -> None:
    connector = _connector()

    parsed = connector._normalize_spot_account(
        {"currency": "USDT", "type": "trade", "balance": "5.25", "available": "4.75", "holds": "0.5"}
    )

    assert parsed["asset"] == "USDT"
    assert parsed["type"] == "spot_trade"
    assert parsed["value"] == pytest.approx(5.25)
    assert parsed["available"] == pytest.approx(4.75)
    assert parsed["withdrawable"] == pytest.approx(4.75)
    assert parsed["held"] == pytest.approx(0.5)


def test_kucoin_live_preflight_rejects_restricted_operator_region() -> None:
    connector = _connector(
        {
            "KUCOIN_FIXED_EGRESS_REQUIRED": True,
            "KUCOIN_EGRESS_PROXY_URL": "http://fixed-egress.test:8080",
            "KUCOIN_COMPLIANCE_CONFIRMED": True,
            "KUCOIN_OPERATOR_REGION": "British Columbia",
            "KUCOIN_TEST_ACCOUNT": "sufyanh",
            "KUCOIN_TEST_SYMBOL": "BTC-USDT",
            "KUCOIN_MAX_TEST_NOTIONAL_USDT": "1",
        }
    )

    summary = connector.kucoin_live_test_preflight_summary()

    assert summary["fixed_egress_status"] == "restricted"
    assert summary["operator_region_restricted"] is True
    assert "KUCOIN_OPERATOR_REGION=British Columbia is restricted under KuCoin terms" in summary["missing_or_blocked"]


def test_kucoin_live_preflight_ready_for_non_restricted_region_with_fixed_egress() -> None:
    connector = _connector(
        {
            "KUCOIN_FIXED_EGRESS_REQUIRED": True,
            "KUCOIN_EGRESS_PROXY_URL": "http://fixed-egress.test:8080",
            "KUCOIN_COMPLIANCE_CONFIRMED": True,
            "KUCOIN_OPERATOR_REGION": "Alberta",
            "KUCOIN_TEST_ACCOUNT": "sufyanh",
            "KUCOIN_TEST_SYMBOL": "BTC-USDT",
            "KUCOIN_MAX_TEST_NOTIONAL_USDT": "1",
        }
    )

    summary = connector.kucoin_live_test_preflight_summary()

    assert summary["fixed_egress_status"] == "ready"
    assert summary["operator_region_restricted"] is False
    assert summary["missing_or_blocked"] == []


def test_kucoin_live_preflight_ready_for_native_static_egress() -> None:
    connector = _connector(
        {
            "KUCOIN_FIXED_EGRESS_REQUIRED": True,
            "KUCOIN_NATIVE_STATIC_EGRESS_ENABLED": True,
            "KUCOIN_EGRESS_PUBLIC_IPS": "203.0.113.10",
            "KUCOIN_COMPLIANCE_CONFIRMED": True,
            "KUCOIN_OPERATOR_REGION": "Alberta",
            "KUCOIN_TEST_ACCOUNT": "sufyanh",
            "KUCOIN_TEST_SYMBOL": "BTC-USDT",
            "KUCOIN_MAX_TEST_NOTIONAL_USDT": "1",
        }
    )

    summary = connector.kucoin_live_test_preflight_summary()

    assert summary["fixed_egress_status"] == "ready"
    assert summary["fixed_egress_configured"] is True
    assert summary["missing_or_blocked"] == []


def test_kucoin_permission_probe_uses_read_only_endpoints(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_request(session, method, url, *, provider, attempts, sleep_seconds, timeout, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        if "/api/v1/accounts" in url:
            return {"code": "200000", "data": [{"currency": "USDT", "type": "trade", "balance": "2", "available": "2"}]}
        if "/api/v1/account-overview" in url:
            return {"code": "200000", "data": {"currency": "USDT", "availableBalance": "2"}}
        if "/api/v2/position/getPositionMode" in url:
            return {"code": "200000", "data": {"positionMode": "ONE_WAY"}}
        raise AssertionError(f"Unhandled KuCoin permission probe request: {method} {url}")

    monkeypatch.setattr(adapters, "_request_with_retries", fake_request)
    connector = _connector({"KUCOIN_UNIFIED_ACCOUNT_ENABLED": True})

    result = connector.permission_probe("live")

    assert result["general"]["status"] == "ready"
    assert result["spot"]["status"] == "ready"
    assert result["futures"]["status"] == "ready"
    assert result["unified"]["status"] == "operator_configured"
    assert calls
    assert all(call["method"] == "GET" for call in calls)
    assert not any("/orders" in call["url"] for call in calls)


def test_kucoin_spot_market_metadata_parsing_and_validation() -> None:
    connector = _connector()
    market = connector._normalize_spot_market(
        {
            "symbol": "BTC-USDT",
            "baseCurrency": "BTC",
            "quoteCurrency": "USDT",
            "baseMinSize": "0.00001",
            "baseMaxSize": "100",
            "baseIncrement": "0.00001",
            "quoteIncrement": "0.000001",
            "priceIncrement": "0.1",
            "minFunds": "0.1",
            "enableTrading": True,
        }
    )

    assert market["symbol"] == "BTC-USDT"
    assert market["internal_symbol"] == "BTC"
    assert market["base_increment"] == "0.00001"

    valid = connector.build_spot_order_payload("BTC", "buy", "0.00002", "limit", "50000.1", client_order_id="client-1", market=market)
    assert valid["symbol"] == "BTC-USDT"

    with pytest.raises(ValueError, match="below minFunds"):
        connector.build_spot_order_payload("BTC", "buy", "0.00001", "limit", "1000", client_order_id="client-2", market=market)

    with pytest.raises(ValueError, match="priceIncrement"):
        connector.build_spot_order_payload("BTC", "buy", "0.00002", "limit", "50000.12", client_order_id="client-3", market=market)


def test_kucoin_spot_order_payload_construction() -> None:
    connector = _connector()

    limit_payload = connector.build_spot_order_payload(
        "ETH-USDT",
        "sell",
        "0.01",
        "limit",
        "2500.1",
        client_order_id="codex-kucoin-unit",
        time_in_force="gtc",
        post_only=True,
    )
    market_payload = connector.build_spot_order_payload("ETH", "buy", None, "market", funds="1.25", client_order_id="codex-kucoin-market")

    assert limit_payload == {
        "clientOid": "codex-kucoin-unit",
        "symbol": "ETH-USDT",
        "side": "sell",
        "type": "limit",
        "price": "2500.1",
        "size": "0.01",
        "timeInForce": "GTC",
        "postOnly": True,
    }
    assert market_payload["funds"] == "1.25"
    assert market_payload["symbol"] == "ETH-USDT"

    with pytest.raises(ValueError, match="postOnly"):
        connector.build_spot_order_payload("ETH", "buy", "0.01", "limit", "2500", time_in_force="IOC", post_only=True)


def test_kucoin_spot_order_response_status_mapping() -> None:
    connector = _connector()

    open_order = connector._normalize_spot_order(
        {"id": "order-1", "clientOid": "client-1", "symbol": "BTC-USDT", "active": True, "size": "0.01"}
    )
    cancelled_order = connector._normalize_spot_order(
        {"id": "order-2", "clientOid": "client-2", "symbol": "BTC-USDT", "active": False, "cancelExist": True, "size": "0.01"}
    )
    filled_order = connector._normalize_spot_order(
        {"id": "order-3", "clientOid": "client-3", "symbol": "BTC-USDT", "active": False, "size": "0.01", "dealSize": "0.01", "fee": "0.1"}
    )

    assert open_order["status"] == "open"
    assert cancelled_order["status"] == "cancelled"
    assert filled_order["status"] == "filled"
    assert filled_order["fee"] == pytest.approx(0.1)


def test_kucoin_spot_mocked_integration_flows(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_request(session, method, url, *, provider, attempts, sleep_seconds, timeout, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        if url.endswith("/api/v2/symbols/BTC-USDT"):
            return {
                "code": "200000",
                "data": {
                    "symbol": "BTC-USDT",
                    "baseCurrency": "BTC",
                    "quoteCurrency": "USDT",
                    "baseMinSize": "0.00001",
                    "baseMaxSize": "100",
                    "baseIncrement": "0.00001",
                    "quoteIncrement": "0.000001",
                    "priceIncrement": "0.1",
                    "minFunds": "0.1",
                    "enableTrading": True,
                },
            }
        if url.endswith("/api/v1/accounts"):
            return {"code": "200000", "data": [{"currency": "USDT", "type": "trade", "balance": "2", "available": "2", "holds": "0"}]}
        if url.endswith("/api/v1/hf/orders/test"):
            return {"code": "200000", "data": {"orderId": "test-order-1", "clientOid": "test-client-1"}}
        if url.endswith("/api/v1/hf/orders") and method == "POST":
            return {"code": "200000", "data": {"orderId": "live-order-1", "clientOid": "live-client-1"}}
        if "/api/v1/hf/orders/client-order/" in url and method == "GET":
            return {"code": "200000", "data": {"id": "live-order-1", "clientOid": "live-client-1", "symbol": "BTC-USDT", "active": True}}
        if "/api/v1/hf/orders/client-order/" in url and method == "DELETE":
            return {"code": "200000", "data": {"clientOid": "live-client-1"}}
        raise AssertionError(f"Unhandled KuCoin mock request: {method} {url}")

    monkeypatch.setattr(adapters, "_request_with_retries", fake_request)
    connector = _connector()

    markets = connector.get_spot_markets(symbol="BTC-USDT")
    balances = connector.get_spot_balances("live")
    test_order = connector.create_spot_test_order("live", "BTC-USDT", "buy", "0.00001", "limit", "50000", client_order_id="test-client-1")
    payload = connector.build_spot_order_payload(
        "BTC", "buy", "0.00001", "limit", "50000", client_order_id="live-client-1", market=markets[0]
    )
    live_order = connector.place_spot_order("live", "BTC-USDT", "buy", "0.00001", "limit", "50000", client_order_id="live-client-1")
    status = connector.get_spot_order_status("live", "BTC-USDT", client_order_id="live-client-1")
    cancel = connector.cancel_spot_order("live", "BTC-USDT", client_order_id="live-client-1")

    assert markets[0]["symbol"] == "BTC-USDT"
    assert balances[0]["available"] == pytest.approx(2.0)
    assert test_order["test_order"] is True
    assert test_order["exchange_order_id"] == "test-order-1"
    assert payload["clientOid"] == "live-client-1"
    assert live_order["exchange_order_id"] == "live-order-1"
    assert status["status"] == "open"
    assert cancel["status"] == "cancelled"
