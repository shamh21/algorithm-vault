from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pyotp
import pytest
from cryptography.fernet import Fernet

from app.auth import encrypt_totp_secret, password_hash
from app.extensions import db
from app.models import (
    AuditLog,
    DepositAddress,
    LeveragedMarket,
    Order,
    PlatformTreasuryReserveJob,
    ReferralInviteCode,
    StrategyRun,
    TradingConnection,
    User,
    VaultCycle,
    VaultCycleAllocation,
    VaultCycleSettlement,
    VaultCycleTransfer,
    WalletBalance,
    WalletTransaction,
)
from app.services.failures import ProviderConnectionError
from app.services.hyperliquid_client import ClientSnapshot
from app.services.wallet_custody import BroadcastResult, GeneratedWallet, RealWalletCustodyService, WalletBalanceSnapshot


class _FakeVaultConnector:
    def __init__(self, asset: str = "USDT") -> None:
        self.asset = asset
        self.withdrawals = 0

    def reserve_funds(self, mode: str, asset: str, amount: float) -> dict[str, Any]:
        return {
            "status": "confirmed",
            "provider_reference": f"reserve-{asset}-{amount}",
            "confirmed_amount": amount,
        }

    def cancel_all_orders(self, mode: str) -> list[dict[str, Any]]:
        return []

    def flatten_all_positions(self, mode: str) -> list[dict[str, Any]]:
        return []

    def get_positions(self, mode: str) -> list[dict[str, Any]]:
        return []

    def withdraw_to_address(
        self,
        mode: str,
        asset: str,
        amount: float,
        destination: str,
        network: str | None = None,
        memo: str | None = None,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        self.withdrawals += 1
        return {
            "status": "confirmed",
            "provider_reference": f"withdraw-{client_reference}",
            "confirmed_amount": amount,
            "fee_amount": 0.0,
            "fee_asset": asset,
        }

    def transfer_status(self, mode: str, provider_reference: str, transfer_type: str | None = None) -> dict[str, Any]:
        return {
            "status": "confirmed",
            "provider_reference": provider_reference,
        }

    def convert_stablecoin(
        self,
        mode: str,
        from_asset: str,
        to_asset: str,
        amount: float,
        max_slippage_bps: float,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        return {
            "status": "confirmed",
            "provider_reference": f"convert-{client_reference}",
            "confirmed_amount": amount,
        }


class _FakeTreasuryConversionConnector:
    def __init__(self) -> None:
        self.conversions: list[dict[str, Any]] = []
        self.withdrawals: list[dict[str, Any]] = []

    def convert_stablecoin(
        self,
        mode: str,
        from_asset: str,
        to_asset: str,
        amount: float,
        max_slippage_bps: float,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        self.conversions.append({"from_asset": from_asset, "to_asset": to_asset, "amount": amount})
        return {
            "status": "confirmed",
            "provider_reference": f"treasury-convert-{client_reference}",
            "confirmed_amount": amount,
        }

    def withdraw_to_address(
        self,
        mode: str,
        asset: str,
        amount: float,
        destination: str,
        network: str | None = None,
        memo: str | None = None,
        client_reference: str | None = None,
    ) -> dict[str, Any]:
        self.withdrawals.append({"asset": asset, "amount": amount, "destination": destination})
        return {
            "status": "confirmed",
            "provider_reference": f"treasury-withdraw-{client_reference}",
            "confirmed_amount": amount,
        }


class _OnchainUsdtAdapter:
    def __init__(self, amount: float = 50.0) -> None:
        self.amount = amount

    def supports(self, asset: str, network: str) -> bool:
        return asset.upper() in {"ETH", "USDT"} and network == "Ethereum"

    def generate_wallet(self, asset: str, network: str) -> GeneratedWallet:
        return GeneratedWallet(
            address="0x1234567890abcdef1234567890abcdef12345678",
            private_key="11" * 32,
            public_key="0x1234567890abcdef1234567890abcdef12345678",
            key_type="secp256k1",
            provider="fake_evm",
        )

    def get_balance(self, address: str, asset: str, network: str) -> WalletBalanceSnapshot:
        amount = self.amount if asset.upper() == "USDT" else 0.01
        return WalletBalanceSnapshot(amount=amount, asset=asset, checked=True, confirmations=12)

    def estimate_fee(self, asset: str, network: str, destination: str, amount: float) -> float:
        return 0.001

    def sign_and_broadcast(self, withdrawal, private_key: str) -> BroadcastResult:
        return BroadcastResult("submitted", "0xunused", {})

    def confirm_transaction(self, provider_reference: str, asset: str, network: str) -> dict[str, Any]:
        return {"confirmed": True}


def _create_user(username: str = "vault-cycle") -> tuple[User, str]:
    user = User(username=username, password_hash=password_hash("password123"), role="user")
    secret = pyotp.random_base32()
    user.totp_secret_encrypted = encrypt_totp_secret(secret)
    user.two_factor_enabled_at = datetime.utcnow()
    db.session.add(user)
    db.session.commit()
    return user, secret


def _login(client, username: str, secret: str):
    return client.post(
        "/login",
        data={"username": username, "password": "password123", "totp_code": pyotp.TOTP(secret).now()},
        follow_redirects=False,
    )


def _connection(user: User, provider: str) -> TradingConnection:
    service = db.session
    connection = TradingConnection(
        user_id=user.id,
        provider=provider,
        connection_type="cex_api_key",
        is_active=True,
        verification_status="verified",
    )
    connection.provider_metadata = {"provider_label": provider.title(), "tradable": True, "last_verified_mode": "live"}
    if provider == "kucoin":
        connection.encrypted_api_key = "x"
        connection.encrypted_api_secret = "x"
        connection.encrypted_passphrase = "x"
    else:
        connection.wallet_address = "0x" + ("2" * 40)
        connection.encrypted_api_secret = "x"
    service.add(connection)
    service.commit()
    return connection


def _patch_market_data(app) -> None:
    market_data = app.extensions["services"]["market_data"]
    market_data.get_mid_price = lambda symbol, mode: 100.0
    market_data.get_order_book = lambda symbol, mode: {"levels": [[{"px": "99.95", "sz": "1000"}], [{"px": "100.05", "sz": "1000"}]]}
    market_data.get_candles = lambda symbol, timeframe, mode, limit: [
        {"timestamp": index, "open": 100, "high": 101, "low": 99, "close": 100 + index * 0.01, "volume": 1000} for index in range(80)
    ]


def _seed_market(provider: str, settlement_asset: str = "USDT") -> None:
    market = LeveragedMarket(
        provider=provider,
        venue_symbol="XBTUSDTM" if provider == "kucoin" else "BTC",
        symbol="BTC",
        status="active",
        settlement_asset=settlement_asset,
        max_leverage=3.0,
        liquidity_usd=250_000,
        spread_bps=4.0,
        funding_rate=0.0001,
    )
    market.raw = {"market_structure_score": 0.72, "ml_score": 0.66}
    db.session.add(market)
    db.session.commit()


def _seed_custom_market(
    provider: str,
    symbol: str,
    settlement_asset: str,
    *,
    liquidity_usd: float,
    spread_bps: float,
    market_structure_score: float,
    ml_score: float,
) -> None:
    venue_symbol = f"{symbol}USDTM" if provider == "kucoin" else symbol
    market = LeveragedMarket(
        provider=provider,
        venue_symbol=venue_symbol,
        symbol=symbol,
        status="active",
        settlement_asset=settlement_asset,
        max_leverage=3.0,
        liquidity_usd=liquidity_usd,
        spread_bps=spread_bps,
        funding_rate=0.0001,
    )
    market.raw = {"market_structure_score": market_structure_score, "ml_score": ml_score}
    db.session.add(market)
    db.session.commit()


def _snapshot_for(connection_ids: dict[int, str]):
    def account_snapshot(user_id: int, mode: str, connection_id: int | None = None):
        provider = connection_ids[int(connection_id or 0)]
        asset = "USDC" if provider == "hyperliquid" else "USDT"
        return ClientSnapshot(
            mode,
            [{"asset": asset, "type": "margin", "value": 100.0, "withdrawable": 100.0}],
            [],
            [],
            [],
            [],
        )

    return account_snapshot


def test_vault_cycle_allocator_excludes_unsupported_stablecoin_conversion(app) -> None:
    app.config["VAULT_CYCLE_CONVERSION_ENABLED"] = False
    user, _ = _create_user("allocator")
    hyperliquid = _connection(user, "hyperliquid")
    kucoin = _connection(user, "kucoin")
    _seed_market("hyperliquid", "USDC")
    _seed_market("kucoin", "USDT")
    trading = app.extensions["services"]["trading_connections"]
    trading.account_snapshot = _snapshot_for({hyperliquid.id: "hyperliquid", kucoin.id: "kucoin"})

    plans, blockers = app.extensions["services"]["vault_cycle_allocator"].allocate(
        user_id=user.id,
        amount_usd=20.0,
        settlement_asset="USDT",
        connections=[hyperliquid, kucoin],
        allowed_symbols=["BTC"],
        provider_filter=["hyperliquid", "kucoin"],
    )

    assert [plan.provider for plan in plans] == ["kucoin"]
    assert any(item["provider"] == "hyperliquid" and item["reason"] == "stablecoin_conversion_route_unavailable" for item in blockers)


def test_vault_cycle_route_creates_allocations_and_confirmed_reserve_transfer(app, monkeypatch) -> None:
    app.config.update(
        {
            "VAULT_CYCLE_ENGINE_ENABLED": True,
            "VAULT_CYCLE_REAL_TRANSFERS_ENABLED": True,
            "VAULT_CYCLE_PROVIDERS": ["kucoin"],
            "VAULT_CYCLE_REQUIRE_EXCHANGE_RESERVE": True,
        }
    )
    _patch_market_data(app)
    started: list[int] = []
    app.extensions["services"]["strategy_manager"].start = lambda run_id: started.append(run_id)
    user, secret = _create_user("route-cycle")
    kucoin = _connection(user, "kucoin")
    _seed_market("kucoin", "USDT")
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=50.0, estimated_usd_value=50.0))
    db.session.commit()
    trading = app.extensions["services"]["trading_connections"]
    trading.account_snapshot = _snapshot_for({kucoin.id: "kucoin"})
    trading.connector_for_user = lambda user_id, connection_id=None: _FakeVaultConnector("USDT")
    client = app.test_client()
    _login(client, user.username, secret)

    response = client.post(
        "/vault/cycles",
        data={
            "deposit_amount": "20",
            "deposit_asset": "USDT",
            "settlement_asset": "USDT",
            "lock_duration": "24",
            "providers": "kucoin",
            "idempotency_key": "route-cycle-1",
        },
        follow_redirects=False,
    )

    assert response.status_code in {302, 303}
    cycle = VaultCycle.query.filter_by(user_id=user.id, algorithm_profile="VaultCycle").one()
    enforcement = cycle.selection_metadata["active_trading_enforcement"]
    assert enforcement["status"] == "initialized"
    assert enforcement["minimum_trades_per_cycle"] == 1
    assert started == [cycle.strategy_run_id]
    assert all(leg.details["active_trading_enforced"] is True for leg in cycle.allocation_legs)
    assert VaultCycleAllocation.query.filter_by(vault_cycle_id=cycle.id, provider="kucoin", status="funded").count() == 1
    transfer = VaultCycleTransfer.query.filter_by(vault_cycle_id=cycle.id, direction="fund_exchange").one()
    assert transfer.status == "confirmed"
    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDT").one()
    assert balance.available_balance == 30
    assert balance.locked_balance == 20


def test_vault_cycle_materializes_onchain_surplus_before_allocation(app) -> None:
    app.config.update(
        {
            "VAULT_CYCLE_ENGINE_ENABLED": True,
            "VAULT_CYCLE_REAL_TRANSFERS_ENABLED": True,
            "VAULT_CYCLE_PROVIDERS": ["kucoin"],
            "VAULT_CYCLE_REQUIRE_EXCHANGE_RESERVE": True,
            "WALLET_REAL_CUSTODY_ENABLED": True,
            "WALLET_ALLOW_IN_APP_KEYGEN": True,
            "TOTP_ENCRYPTION_KEY": Fernet.generate_key().decode("utf-8"),
            "WALLET_EVM_RPC_URL": "https://evm.example.invalid",
            "WALLET_EVM_TOKEN_CONTRACTS": {
                "ETHEREUM": {
                    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
                    "USDT_DECIMALS": 6,
                }
            },
        }
    )
    adapter = _OnchainUsdtAdapter(amount=50.0)
    custody = RealWalletCustodyService(app.config, adapters=[adapter])
    app.extensions["services"]["wallet_custody"] = custody
    user, _ = _create_user("engine-onchain")
    kucoin = _connection(user, "kucoin")
    custody.get_or_create_address(user_id=user.id, asset="USDT", network="Ethereum")
    _seed_market("kucoin", "USDT")
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=40.0, estimated_usd_value=40.0))
    db.session.commit()
    trading = app.extensions["services"]["trading_connections"]
    trading.account_snapshot = _snapshot_for({kucoin.id: "kucoin"})
    trading.connector_for_user = lambda user_id, connection_id=None: _FakeVaultConnector("USDT")

    result = app.extensions["services"]["vault_cycle_orchestrator"].start_cycle(
        user=user,
        amount=50.0,
        deposit_asset="USDT",
        settlement_asset="USDT",
        duration_seconds=3600,
        providers=["kucoin"],
        allowed_symbols=["BTC"],
        idempotency_key="engine-onchain-surplus",
        start_strategy_runs=False,
    )

    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDT").one()
    reconciliation = WalletTransaction.query.filter_by(user_id=user.id, transaction_type="onchain_reconciliation").one()
    assert result["created"] is True
    assert balance.available_balance == pytest.approx(0.0)
    assert balance.locked_balance == pytest.approx(50.0)
    assert reconciliation.amount == pytest.approx(10.0)
    assert WalletTransaction.query.filter_by(user_id=user.id, transaction_type="allocation").one().amount == 50.0


def test_vault_start_routes_share_vault_cycle_orchestrator(app) -> None:
    app.config.update(
        {
            "VAULT_CYCLE_ENGINE_ENABLED": True,
            "VAULT_CYCLE_REAL_TRANSFERS_ENABLED": True,
            "VAULT_CYCLE_PROVIDERS": ["kucoin"],
            "VAULT_CYCLE_REQUIRE_EXCHANGE_RESERVE": True,
        }
    )
    _patch_market_data(app)
    started: list[int] = []
    app.extensions["services"]["strategy_manager"].start = lambda run_id: started.append(run_id)
    user, secret = _create_user("route-equivalence")
    kucoin = _connection(user, "kucoin")
    _seed_market("kucoin", "USDT")
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=100.0, estimated_usd_value=100.0))
    db.session.commit()
    trading = app.extensions["services"]["trading_connections"]
    trading.account_snapshot = _snapshot_for({kucoin.id: "kucoin"})
    trading.connector_for_user = lambda user_id, connection_id=None: _FakeVaultConnector("USDT")
    client = app.test_client()
    _login(client, user.username, secret)

    start_response = client.post(
        "/vault/start-cycle",
        data={
            "deposit_amount": "20",
            "deposit_asset": "USDT",
            "settlement_asset": "USDT",
            "lock_duration": "24",
            "providers": "kucoin",
            "idempotency_key": "start-route-engine",
        },
        headers={"Accept": "application/json"},
    )
    duplicate_response = client.post(
        "/vault/start-cycle",
        data={
            "deposit_amount": "20",
            "deposit_asset": "USDT",
            "settlement_asset": "USDT",
            "lock_duration": "24",
            "providers": "kucoin",
            "idempotency_key": "start-route-engine",
        },
        headers={"Accept": "application/json"},
    )
    cycles_response = client.post(
        "/vault/cycles",
        data={
            "deposit_amount": "20",
            "deposit_asset": "USDT",
            "settlement_asset": "USDT",
            "lock_duration": "24",
            "providers": "kucoin",
            "idempotency_key": "cycles-route-engine",
        },
        headers={"Accept": "application/json"},
    )
    cycles_duplicate_response = client.post(
        "/vault/cycles",
        data={
            "deposit_amount": "20",
            "deposit_asset": "USDT",
            "settlement_asset": "USDT",
            "lock_duration": "24",
            "providers": "kucoin",
            "idempotency_key": "cycles-route-engine",
        },
        headers={"Accept": "application/json"},
    )

    assert start_response.status_code == 201
    assert start_response.get_json()["code"] == "vault_cycle_started"
    assert duplicate_response.status_code == 200
    assert duplicate_response.get_json()["code"] == "vault_cycle_duplicate"
    assert cycles_response.status_code == 201
    assert cycles_response.get_json()["code"] == "vault_cycle_started"
    assert cycles_duplicate_response.status_code == 200
    assert cycles_duplicate_response.get_json()["code"] == "vault_cycle_duplicate"
    cycles = VaultCycle.query.filter_by(user_id=user.id, algorithm_profile="VaultCycle").order_by(VaultCycle.id.asc()).all()
    assert len(cycles) == 2
    assert all(cycle.selection_metadata["vault_cycle_engine"] is True for cycle in cycles)
    assert VaultCycleAllocation.query.filter(VaultCycleAllocation.vault_cycle_id.in_([cycle.id for cycle in cycles])).count() == 2
    assert WalletBalance.query.filter_by(user_id=user.id, asset="USDT").one().locked_balance == 40
    assert started


def test_vault_cycle_route_surfaces_typed_failure_codes(app) -> None:
    user, secret = _create_user("typed-failure")
    _connection(user, "kucoin")

    def fail_start_cycle(**_kwargs):
        raise ProviderConnectionError(
            "KuCoin account snapshot unavailable.",
            code="provider_snapshot_unavailable",
            context={"provider": "kucoin"},
        )

    app.extensions["services"]["vault_cycle_orchestrator"].start_cycle = fail_start_cycle
    client = app.test_client()
    _login(client, user.username, secret)

    response = client.post(
        "/vault/cycles",
        json={
            "amount": 20,
            "deposit_asset": "USDT",
            "settlement_asset": "USDT",
            "duration_hours": 24,
            "providers": ["kucoin"],
            "idempotency_key": "typed-failure-route",
        },
        headers={"Accept": "application/json"},
    )

    payload = response.get_json()
    assert response.status_code == 400
    assert payload["ok"] is False
    assert payload["code"] == "provider_snapshot_unavailable"
    assert payload["details"] == {"provider": "kucoin"}


def test_vault_cycle_enforcer_restarts_stale_strategy_run(app) -> None:
    app.config.update(
        {
            "VAULT_CYCLE_ENGINE_ENABLED": True,
            "VAULT_CYCLE_REAL_TRANSFERS_ENABLED": True,
            "VAULT_CYCLE_PROVIDERS": ["kucoin"],
            "VAULT_CYCLE_REQUIRE_EXCHANGE_RESERVE": True,
        }
    )
    _patch_market_data(app)
    started: list[int] = []
    app.extensions["services"]["strategy_manager"].start = lambda run_id: started.append(run_id)
    user, _ = _create_user("enforcer-restart")
    kucoin = _connection(user, "kucoin")
    _seed_market("kucoin", "USDT")
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=50.0, estimated_usd_value=50.0))
    db.session.commit()
    trading = app.extensions["services"]["trading_connections"]
    trading.account_snapshot = _snapshot_for({kucoin.id: "kucoin"})
    trading.connector_for_user = lambda user_id, connection_id=None: _FakeVaultConnector("USDT")
    result = app.extensions["services"]["vault_cycle_orchestrator"].start_cycle(
        user=user,
        amount=20.0,
        deposit_asset="USDT",
        settlement_asset="USDT",
        duration_seconds=3600,
        providers=["kucoin"],
        allowed_symbols=["BTC"],
        idempotency_key="restart-cycle",
        start_strategy_runs=False,
    )
    run = db.session.get(StrategyRun, result["run_ids"][0])
    run.status = "stopped"
    run.manual_enabled = False
    run.last_heartbeat_at = datetime.utcnow() - timedelta(minutes=20)
    db.session.commit()

    payloads = app.extensions["services"]["vault_cycle_trading_enforcer"].enforce_active_cycles(user.id)

    refreshed = db.session.get(StrategyRun, run.id)
    assert payloads[0]["start_run_ids"] == [run.id]
    assert started == [run.id]
    assert refreshed.manual_enabled is True
    assert refreshed.status == "starting"


def test_vault_cycle_enforcer_preserves_idle_capital_when_no_valid_setup(app) -> None:
    app.config.update(
        {
            "VAULT_CYCLE_ENGINE_ENABLED": True,
            "VAULT_CYCLE_REAL_TRANSFERS_ENABLED": True,
            "VAULT_CYCLE_PROVIDERS": ["kucoin"],
            "VAULT_CYCLE_REQUIRE_EXCHANGE_RESERVE": True,
            "VAULT_CYCLE_MAX_IDLE_SECONDS": 1,
            "VAULT_CYCLE_RESCREEN_SECONDS": 1,
            "VAULT_CYCLE_MIN_OPPORTUNITY_SCORE": 0.99,
        }
    )
    _patch_market_data(app)
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    app.extensions["services"]["order_manager"].current_position = lambda *args, **kwargs: {"quantity": 0.0}
    user, _ = _create_user("enforcer-idle")
    kucoin = _connection(user, "kucoin")
    _seed_market("kucoin", "USDT")
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=50.0, estimated_usd_value=50.0))
    db.session.commit()
    trading = app.extensions["services"]["trading_connections"]
    trading.account_snapshot = _snapshot_for({kucoin.id: "kucoin"})
    trading.connector_for_user = lambda user_id, connection_id=None: _FakeVaultConnector("USDT")
    result = app.extensions["services"]["vault_cycle_orchestrator"].start_cycle(
        user=user,
        amount=20.0,
        deposit_asset="USDT",
        settlement_asset="USDT",
        duration_seconds=3600,
        providers=["kucoin"],
        allowed_symbols=["BTC"],
        idempotency_key="idle-cycle",
        start_strategy_runs=False,
    )
    cycle = result["cycle"]
    cycle.started_at = datetime.utcnow() - timedelta(seconds=10)
    db.session.commit()

    app.extensions["services"]["vault_cycle_trading_enforcer"].enforce_active_cycles(user.id)

    refreshed = db.session.get(VaultCycle, cycle.id)
    enforcement = refreshed.selection_metadata["active_trading_enforcement"]
    assert enforcement["no_valid_setup"] is True
    assert enforcement["last_activity_snapshot"]["rescreen_reason"] == "max_idle_duration_exceeded"
    assert Order.query.filter_by(vault_cycle_id=cycle.id).count() == 0
    assert AuditLog.query.filter_by(user_id=user.id, action="active_trading_no_valid_setup").count() == 1


def test_vault_cycle_enforcer_rotates_idle_leg_to_stronger_opportunity(app) -> None:
    app.config.update(
        {
            "VAULT_CYCLE_ENGINE_ENABLED": True,
            "VAULT_CYCLE_REAL_TRANSFERS_ENABLED": True,
            "VAULT_CYCLE_PROVIDERS": ["kucoin"],
            "VAULT_CYCLE_REQUIRE_EXCHANGE_RESERVE": True,
            "VAULT_CYCLE_MAX_IDLE_SECONDS": 1,
            "VAULT_CYCLE_RESCREEN_SECONDS": 1,
            "VAULT_CYCLE_MIN_OPPORTUNITY_SCORE": 0.60,
            "VAULT_CYCLE_ROTATION_SCORE_DELTA": 0.05,
        }
    )
    _patch_market_data(app)
    started: list[int] = []
    stopped: list[int] = []
    app.extensions["services"]["strategy_manager"].start = lambda run_id: started.append(run_id)
    app.extensions["services"]["strategy_manager"].stop = lambda run_id: stopped.append(run_id)
    app.extensions["services"]["order_manager"].current_position = lambda *args, **kwargs: {"quantity": 0.0}
    user, _ = _create_user("enforcer-rotate")
    kucoin = _connection(user, "kucoin")
    _seed_custom_market("kucoin", "BTC", "USDT", liquidity_usd=250_000, spread_bps=8, market_structure_score=0.60, ml_score=0.45)
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=50.0, estimated_usd_value=50.0))
    db.session.commit()
    trading = app.extensions["services"]["trading_connections"]
    trading.account_snapshot = _snapshot_for({kucoin.id: "kucoin"})
    trading.connector_for_user = lambda user_id, connection_id=None: _FakeVaultConnector("USDT")
    result = app.extensions["services"]["vault_cycle_orchestrator"].start_cycle(
        user=user,
        amount=20.0,
        deposit_asset="USDT",
        settlement_asset="USDT",
        duration_seconds=3600,
        providers=["kucoin"],
        allowed_symbols=["BTC", "ETH"],
        idempotency_key="rotate-cycle",
        start_strategy_runs=False,
    )
    cycle = result["cycle"]
    old_run_id = result["run_ids"][0]
    _seed_custom_market("kucoin", "ETH", "USDT", liquidity_usd=2_000_000, spread_bps=1, market_structure_score=0.95, ml_score=0.95)
    cycle.started_at = datetime.utcnow() - timedelta(seconds=10)
    db.session.commit()

    payloads = app.extensions["services"]["vault_cycle_trading_enforcer"].enforce_active_cycles(user.id)

    refreshed = db.session.get(VaultCycle, cycle.id)
    active_symbols = {leg.symbol for leg in refreshed.allocation_legs if leg.status == "active"}
    rotated_symbols = {leg.symbol for leg in refreshed.allocation_legs if leg.status == "rotated"}
    assert payloads[0]["rotations"][0]["new_symbol"] == "ETH"
    assert "ETH" in active_symbols
    assert "BTC" in rotated_symbols
    assert old_run_id in stopped
    assert payloads[0]["rotations"][0]["new_run_id"] in started


def test_vault_cycle_enforcer_counts_rejected_attempts_without_utilization(app) -> None:
    app.config.update(
        {
            "VAULT_CYCLE_ENGINE_ENABLED": True,
            "VAULT_CYCLE_REAL_TRANSFERS_ENABLED": True,
            "VAULT_CYCLE_PROVIDERS": ["kucoin"],
            "VAULT_CYCLE_REQUIRE_EXCHANGE_RESERVE": True,
            "VAULT_CYCLE_MAX_IDLE_SECONDS": 1,
            "VAULT_CYCLE_RESCREEN_SECONDS": 1,
            "VAULT_CYCLE_MIN_OPPORTUNITY_SCORE": 0.99,
        }
    )
    _patch_market_data(app)
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    user, _ = _create_user("enforcer-rejected")
    kucoin = _connection(user, "kucoin")
    _seed_market("kucoin", "USDT")
    db.session.add(WalletBalance(user_id=user.id, asset="USDT", available_balance=50.0, estimated_usd_value=50.0))
    db.session.commit()
    trading = app.extensions["services"]["trading_connections"]
    trading.account_snapshot = _snapshot_for({kucoin.id: "kucoin"})
    trading.connector_for_user = lambda user_id, connection_id=None: _FakeVaultConnector("USDT")
    result = app.extensions["services"]["vault_cycle_orchestrator"].start_cycle(
        user=user,
        amount=20.0,
        deposit_asset="USDT",
        settlement_asset="USDT",
        duration_seconds=3600,
        providers=["kucoin"],
        allowed_symbols=["BTC"],
        idempotency_key="rejected-cycle",
        start_strategy_runs=False,
    )
    cycle = result["cycle"]
    order = Order(
        user_id=user.id,
        trading_connection_id=kucoin.id,
        vault_cycle_id=cycle.id,
        client_order_id="rejected-vault-order",
        mode="live",
        symbol="BTC",
        side="buy",
        order_type="market",
        status="rejected",
        strategy_name="mean_reversion",
        quantity=1.0,
        risk_status="rejected",
    )
    order.details = {"vault_cycle_id": cycle.id, "risk_rejection_reason": "stop_loss_required"}
    db.session.add(order)
    cycle.started_at = datetime.utcnow() - timedelta(seconds=10)
    db.session.commit()

    app.extensions["services"]["vault_cycle_trading_enforcer"].enforce_active_cycles(user.id)

    snapshot = db.session.get(VaultCycle, cycle.id).selection_metadata["active_trading_enforcement"]["last_activity_snapshot"]
    assert snapshot["rejected_order_attempts"] == 1
    assert snapshot["accepted_order_attempts"] == 0
    assert snapshot["trade_count"] == 0
    assert snapshot["capital_utilization_pct"] == 0


def test_vault_cycle_enforcer_soft_rebalances_weak_exchange_caps(app) -> None:
    app.config.update(
        {
            "VAULT_CYCLE_ENGINE_ENABLED": True,
            "VAULT_CYCLE_REAL_TRANSFERS_ENABLED": True,
            "VAULT_CYCLE_CONVERSION_ENABLED": True,
            "VAULT_CYCLE_PROVIDERS": ["hyperliquid", "kucoin"],
            "VAULT_CYCLE_REQUIRE_EXCHANGE_RESERVE": True,
            "VAULT_CYCLE_MAX_IDLE_SECONDS": 1,
            "VAULT_CYCLE_RESCREEN_SECONDS": 1,
            "VAULT_CYCLE_MIN_OPPORTUNITY_SCORE": 0.60,
        }
    )
    _patch_market_data(app)
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    app.extensions["services"]["order_manager"].current_position = lambda *args, **kwargs: {"quantity": 0.0}
    user, _ = _create_user("enforcer-rebalance")
    hyperliquid = _connection(user, "hyperliquid")
    kucoin = _connection(user, "kucoin")
    _seed_custom_market("hyperliquid", "BTC", "USDC", liquidity_usd=70_000, spread_bps=20, market_structure_score=0.20, ml_score=0.10)
    _seed_custom_market("kucoin", "ETH", "USDT", liquidity_usd=2_000_000, spread_bps=1, market_structure_score=0.95, ml_score=0.95)
    db.session.add(WalletBalance(user_id=user.id, asset="USDC", available_balance=50.0, estimated_usd_value=50.0))
    db.session.commit()
    trading = app.extensions["services"]["trading_connections"]
    trading.account_snapshot = _snapshot_for({hyperliquid.id: "hyperliquid", kucoin.id: "kucoin"})
    trading.connector_for_user = lambda user_id, connection_id=None: _FakeVaultConnector("USDC")
    result = app.extensions["services"]["vault_cycle_orchestrator"].start_cycle(
        user=user,
        amount=20.0,
        deposit_asset="USDC",
        settlement_asset="USDC",
        duration_seconds=3600,
        providers=["hyperliquid", "kucoin"],
        allowed_symbols=["BTC", "ETH"],
        idempotency_key="rebalance-cycle",
        start_strategy_runs=False,
    )
    cycle = result["cycle"]
    cycle.started_at = datetime.utcnow() - timedelta(seconds=10)
    db.session.commit()

    app.extensions["services"]["vault_cycle_trading_enforcer"].enforce_active_cycles(user.id)

    allocations = {allocation.provider: allocation for allocation in VaultCycleAllocation.query.filter_by(vault_cycle_id=cycle.id).all()}
    hyper_cap = allocations["hyperliquid"].constraints["effective_allocation_cap_usd"]
    kucoin_cap = allocations["kucoin"].constraints["effective_allocation_cap_usd"]
    assert kucoin_cap > hyper_cap
    assert (
        allocations["hyperliquid"].scores["active_trading_opportunity_score"]
        < allocations["kucoin"].scores["active_trading_opportunity_score"]
    )


def test_vault_cycle_settlement_confirms_withdrawal_and_is_idempotent(app) -> None:
    app.config.update(
        {
            "VAULT_CYCLE_ENGINE_ENABLED": True,
            "VAULT_CYCLE_REAL_TRANSFERS_ENABLED": True,
            "VAULT_CYCLE_AUTOMATIC_WITHDRAWALS_ENABLED": True,
            "VAULT_CYCLE_PROVIDERS": ["kucoin"],
            "VAULT_CYCLE_ALLOWED_WITHDRAWAL_ADDRESSES": {"USDT": {"Ethereum": ["0x1111111111111111111111111111111111111111"]}},
        }
    )
    _patch_market_data(app)
    app.extensions["services"]["strategy_manager"].start = lambda run_id: None
    app.extensions["services"]["strategy_manager"].stop = lambda run_id: None
    user, _ = _create_user("settlement-cycle")
    kucoin = _connection(user, "kucoin")
    _seed_market("kucoin", "USDT")
    db.session.add_all(
        [
            WalletBalance(user_id=user.id, asset="USDT", available_balance=50.0, estimated_usd_value=50.0),
            DepositAddress(
                user_id=user.id,
                asset="USDT",
                network="Ethereum",
                address="0x1111111111111111111111111111111111111111",
                is_active=True,
            ),
        ]
    )
    db.session.commit()
    fake = _FakeVaultConnector("USDT")
    trading = app.extensions["services"]["trading_connections"]
    trading.account_snapshot = _snapshot_for({kucoin.id: "kucoin"})
    trading.connector_for_user = lambda user_id, connection_id=None: fake

    result = app.extensions["services"]["vault_cycle_orchestrator"].start_cycle(
        user=user,
        amount=20.0,
        deposit_asset="USDT",
        settlement_asset="USDT",
        duration_seconds=60,
        providers=["kucoin"],
        allowed_symbols=["BTC"],
        idempotency_key="settle-cycle-1",
        start_strategy_runs=False,
    )
    cycle = result["cycle"]
    cycle.unlocks_at = datetime.utcnow() - timedelta(seconds=1)
    db.session.commit()

    first = app.extensions["services"]["vault_cycle_orchestrator"].resume_due_cycles(user.id)
    db.session.commit()
    second = app.extensions["services"]["vault_cycle_orchestrator"].resume_due_cycles(user.id)
    db.session.commit()

    refreshed = db.session.get(VaultCycle, cycle.id)
    balance = WalletBalance.query.filter_by(user_id=user.id, asset="USDT").one()
    assert first[0]["status"] == "complete"
    assert second == []
    assert refreshed.status == "complete"
    assert balance.available_balance == 50
    assert balance.locked_balance == 0
    assert WalletTransaction.query.filter_by(user_id=user.id, vault_cycle_id=cycle.id, transaction_type="settlement").count() == 1
    assert VaultCycleSettlement.query.filter_by(vault_cycle_id=cycle.id, status="complete").count() == 1
    assert fake.withdrawals == 1


def test_vault_settlement_deducts_gas_reserve_without_legacy_referral_profit_share(app, monkeypatch) -> None:
    app.config["PLATFORM_GAS_TREASURY_ENABLED"] = True
    app.config["TREASURY_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    user, _ = _create_user("profit-share")
    invite = ReferralInviteCode(code="HALFPROFIT", percent_profit=50.0, is_active=True, usage_count=1)
    db.session.add(invite)
    db.session.flush()
    user.referral_invite_code_id = invite.id
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDT",
        deposit_amount=100.0,
        settlement_asset="USDT",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        starting_value_usd=100.0,
        current_estimated_value_usd=200.0,
        unlocks_at=datetime.utcnow(),
        algorithm_profile="VaultCycle",
    )
    settlement = VaultCycleSettlement(vault_cycle=cycle, user_id=user.id, settlement_asset="USDT", starting_value_usd=100.0)
    db.session.add_all([cycle, settlement])
    db.session.flush()
    treasury = app.extensions["services"]["platform_treasury"]
    treasury.create_wallet(created_by_user_id=user.id)
    fake_conversion = _FakeTreasuryConversionConnector()
    monkeypatch.setattr(treasury, "_conversion_connector", lambda: fake_conversion)
    monkeypatch.setattr(treasury, "_asset_usd_price", lambda asset: 1.0)
    monkeypatch.setattr(
        treasury,
        "estimate_withdrawal_gas",
        lambda **kwargs: {
            "asset": kwargs["asset"],
            "network": kwargs["network"],
            "estimated_fee_eth": 0.001,
            "reserve_multiplier": 2.0,
            "reserve_eth_target": 0.002,
            "destination": "0x1111111111111111111111111111111111111111",
        },
    )

    payload = treasury.apply_vault_settlement_deductions(cycle, settlement, 200.0)

    assert payload["gas_reserve_asset"] == pytest.approx(0.002)
    assert payload["profit_share_asset"] == pytest.approx(0.0)
    assert payload["user_credit_amount"] == pytest.approx(199.998)
    assert payload["referral_invite_code"] == "HALFPROFIT"
    assert PlatformTreasuryReserveJob.query.filter_by(vault_cycle_id=cycle.id, job_type="vault_gas_reserve").one().status == "pending"
    assert PlatformTreasuryReserveJob.query.filter_by(vault_cycle_id=cycle.id, job_type="vault_profit_share").one_or_none() is None
    assert len(fake_conversion.conversions) == 0
    treasury.process_reserve_jobs()
    assert len(fake_conversion.conversions) == 1
    assert PlatformTreasuryReserveJob.query.filter_by(vault_cycle_id=cycle.id, job_type="vault_gas_reserve").one().status == "complete"


def test_vault_settlement_without_referral_has_no_default_profit_share(app, monkeypatch) -> None:
    app.config["PLATFORM_GAS_TREASURY_ENABLED"] = True
    app.config["TREASURY_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    user, _ = _create_user("default-share")
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDT",
        deposit_amount=100.0,
        settlement_asset="USDT",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        starting_value_usd=100.0,
        current_estimated_value_usd=200.0,
        unlocks_at=datetime.utcnow(),
        algorithm_profile="VaultCycle",
    )
    settlement = VaultCycleSettlement(vault_cycle=cycle, user_id=user.id, settlement_asset="USDT", starting_value_usd=100.0)
    db.session.add_all([cycle, settlement])
    db.session.flush()
    treasury = app.extensions["services"]["platform_treasury"]
    treasury.create_wallet(created_by_user_id=user.id)
    monkeypatch.setattr(treasury, "_conversion_connector", lambda: _FakeTreasuryConversionConnector())
    monkeypatch.setattr(treasury, "_asset_usd_price", lambda asset: 1.0)
    monkeypatch.setattr(
        treasury,
        "estimate_withdrawal_gas",
        lambda **kwargs: {
            "asset": kwargs["asset"],
            "network": kwargs["network"],
            "estimated_fee_eth": 0.0,
            "reserve_multiplier": 2.0,
            "reserve_eth_target": 0.0,
            "destination": "0x1111111111111111111111111111111111111111",
        },
    )

    payload = treasury.apply_vault_settlement_deductions(cycle, settlement, 200.0)

    assert payload["referral_percent"] == pytest.approx(0.0)
    assert payload["referral_invite_code"] == ""
    assert payload["profit_share_asset"] == pytest.approx(0.0)
    assert payload["user_credit_amount"] == pytest.approx(200.0)


def test_vault_settlement_skips_profit_share_when_gas_deduction_removes_profit(app, monkeypatch) -> None:
    app.config["PLATFORM_GAS_TREASURY_ENABLED"] = True
    app.config["TREASURY_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    user, _ = _create_user("gas-first")
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDT",
        deposit_amount=100.0,
        settlement_asset="USDT",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        starting_value_usd=100.0,
        current_estimated_value_usd=100.001,
        unlocks_at=datetime.utcnow(),
        algorithm_profile="VaultCycle",
    )
    settlement = VaultCycleSettlement(vault_cycle=cycle, user_id=user.id, settlement_asset="USDT", starting_value_usd=100.0)
    db.session.add_all([cycle, settlement])
    db.session.flush()
    treasury = app.extensions["services"]["platform_treasury"]
    treasury.create_wallet(created_by_user_id=user.id)
    monkeypatch.setattr(treasury, "_asset_usd_price", lambda asset: 1.0)
    monkeypatch.setattr(
        treasury,
        "estimate_withdrawal_gas",
        lambda **kwargs: {
            "asset": kwargs["asset"],
            "network": kwargs["network"],
            "estimated_fee_eth": 0.001,
            "reserve_multiplier": 2.0,
            "reserve_eth_target": 0.002,
            "destination": "0x1111111111111111111111111111111111111111",
        },
    )

    payload = treasury.apply_vault_settlement_deductions(cycle, settlement, 100.001)

    assert payload["gas_reserve_asset"] == pytest.approx(0.002)
    assert payload["profit_share_asset"] == pytest.approx(0.0)
    assert payload["user_credit_amount"] == pytest.approx(99.999)
    assert PlatformTreasuryReserveJob.query.filter_by(vault_cycle_id=cycle.id, job_type="vault_profit_share").one_or_none() is None
