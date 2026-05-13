from __future__ import annotations

import pytest

import app.services.hyperliquid_client as hl_module
from app.services.hyperliquid_client import HYPERLIQUID_TESTNET_API_URL, HyperliquidClient


def _guard_config(**overrides):
    config = {
        "ENABLE_LIVE_TRADING": True,
        "HL_ACCOUNT_ADDRESS": "0x" + ("1" * 40),
        "HL_SECRET_KEY": "0x" + ("2" * 64),
        "HL_MAINNET_BASE_URL": "https://api.hyperliquid.xyz",
        "HL_TESTNET_BASE_URL": HYPERLIQUID_TESTNET_API_URL,
        "HYPERLIQUID_ACCOUNT": "sufyanh",
        "HYPERLIQUID_ENV": "testnet",
        "HYPERLIQUID_BASE_URL": HYPERLIQUID_TESTNET_API_URL,
        "RUN_HYPERLIQUID_LIVE_TESTS": True,
        "HYPERLIQUID_MIN_ORDER_VALUE_USD": 10.0,
        "EXCHANGE_RETRY_ATTEMPTS": 1,
        "EXCHANGE_RETRY_SLEEP_SECONDS": 0.0,
    }
    config.update(overrides)
    return config


def test_hyperliquid_testnet_guard_requires_exact_account_env_url_and_opt_in() -> None:
    client = HyperliquidClient(
        _guard_config(
            HYPERLIQUID_ACCOUNT="other",
            HYPERLIQUID_ENV="live",
            HYPERLIQUID_BASE_URL="https://api.hyperliquid.xyz",
            RUN_HYPERLIQUID_LIVE_TESTS=False,
        )
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.ensure_testnet_live_tests_enabled()

    message = str(exc_info.value)
    assert "hyperliquid_testnet_guard_failed" in message
    assert "HYPERLIQUID_ACCOUNT must be exactly sufyanh" in message
    assert "HYPERLIQUID_ENV must be exactly testnet" in message
    assert "HYPERLIQUID_BASE_URL must be exactly https://api.hyperliquid-testnet.xyz" in message
    assert "RUN_HYPERLIQUID_LIVE_TESTS=1 is required" in message
    assert client.can_trade("testnet") is False


def test_hyperliquid_testnet_exchange_uses_main_account_address_not_signer(monkeypatch) -> None:
    captured = {}

    class FakeLocalAccount:
        address = "0x" + ("a" * 40)

    class FakeEthAccountModule:
        class Account:
            @staticmethod
            def from_key(secret):
                captured["secret"] = secret
                return FakeLocalAccount()

    class FakeExchange:
        def __init__(self, account, base_url, account_address=None, vault_address=None, timeout=None):
            captured["account"] = account
            captured["base_url"] = base_url
            captured["account_address"] = account_address
            captured["vault_address"] = vault_address
            captured["timeout"] = timeout

    monkeypatch.setattr(hl_module, "_load_eth_account", lambda: FakeEthAccountModule)
    monkeypatch.setattr(hl_module, "_load_exchange_class", lambda: FakeExchange)

    client = HyperliquidClient(_guard_config(HL_ACCOUNT_ADDRESS="0x" + ("b" * 40), HL_VAULT_ADDRESS=None))
    client._get_exchange("testnet")

    assert captured["base_url"] == HYPERLIQUID_TESTNET_API_URL
    assert captured["account_address"] == "0x" + ("b" * 40)
    assert captured["account"].address == "0x" + ("a" * 40)
    assert captured["vault_address"] is None
    assert captured["secret"] == "0x" + ("2" * 64)


def test_hyperliquid_order_accepts_alo_tif_and_client_order_id(monkeypatch) -> None:
    captured = {}

    class FakeExchange:
        def update_leverage(self, leverage: int, symbol: str):
            return {"status": "ok"}

        def order(self, symbol, is_buy, quantity, price, order_type, reduce_only=False, cloid=None):
            captured.update(
                {
                    "symbol": symbol,
                    "is_buy": is_buy,
                    "quantity": quantity,
                    "price": price,
                    "order_type": order_type,
                    "reduce_only": reduce_only,
                    "cloid": cloid.to_raw() if cloid is not None else None,
                }
            )
            return {"status": "ok", "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": 123}}]}}}

    client = HyperliquidClient(_guard_config())
    monkeypatch.setattr(client, "_get_exchange", lambda mode: FakeExchange())
    local_client_id = "codex-hl-test-unit-1"

    result = client.place_order(
        "testnet",
        "BTC",
        "buy",
        0.001,
        "limit",
        50000.0,
        False,
        1.0,
        0.0,
        client_order_id=local_client_id,
        time_in_force="Alo",
    )

    assert result["status"] == "open"
    assert result["client_order_id"] == local_client_id
    assert result["exchange_client_order_id"] == HyperliquidClient.client_order_id_to_cloid(local_client_id)
    assert captured["order_type"] == {"limit": {"tif": "Alo"}}
    assert captured["cloid"] == result["exchange_client_order_id"]
    assert captured["reduce_only"] is False


def test_hyperliquid_order_status_can_query_by_local_client_order_id() -> None:
    captured = {}

    class FakeInfo:
        def query_order_by_cloid(self, user, cloid):
            captured["user"] = user
            captured["cloid"] = cloid.to_raw()
            return {"order": {"status": "open", "oid": 123, "cloid": cloid.to_raw()}}

    client = HyperliquidClient(_guard_config())
    client._public_info["testnet"] = FakeInfo()
    local_client_id = "codex-hl-test-status-1"

    status = client.get_order_status("testnet", client_order_id=local_client_id)

    assert captured["user"] == "0x" + ("1" * 40)
    assert captured["cloid"] == HyperliquidClient.client_order_id_to_cloid(local_client_id)
    assert status["status"] == "open"
    assert status["exchange_order_id"] == "123"
    assert status["client_order_id"] == local_client_id


def test_hyperliquid_live_test_plan_uses_metadata_and_fails_closed_without_funds() -> None:
    class FakeInfo:
        def meta_and_asset_ctxs(self):
            return [{"universe": [{"name": "BTC", "szDecimals": 5}, {"name": "ETH", "szDecimals": 4}]}, [{}, {}]]

        def all_mids(self):
            return {"BTC": "50000", "ETH": "2500"}

        def user_state(self, address):
            return {"marginSummary": {"accountValue": "0"}, "withdrawable": "0", "assetPositions": []}

        def spot_user_state(self, address):
            return {"balances": []}

    client = HyperliquidClient(_guard_config())
    client._public_info["testnet"] = FakeInfo()

    with pytest.raises(RuntimeError, match="insufficient_testnet_funds"):
        client.live_test_order_plan(require_funds=True)

    plan = client.live_test_order_plan(require_funds=False)
    assert plan["symbol"] == "BTC"
    assert plan["time_in_force"] == "Alo"
    assert plan["quantity"] == pytest.approx(0.0002)
    assert plan["limit_price"] < plan["mid_price"]


def test_hyperliquid_sanitizes_and_classifies_provider_errors() -> None:
    sanitized = HyperliquidClient._sanitize_error_message(
        "signature=0xabcdef private=0x" + ("1" * 64) + " account=0x" + ("2" * 40)
    )

    assert "0x" + ("1" * 64) not in sanitized
    assert "0x" + ("2" * 40) not in sanitized
    assert "0x2222...2222" in sanitized
    assert "signature=[redacted]" in sanitized
    assert HyperliquidClient.classify_error("Price must be divisible by tick size.") == "precision"
    assert HyperliquidClient.classify_error("Insufficient margin to place order.") == "insufficient_funds"
    assert HyperliquidClient.classify_error("429 too many requests") == "rate_limit"
