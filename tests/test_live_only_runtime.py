from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy.exc import OperationalError

from app import _create_all_tolerant
from app.extensions import db
from app.models import (
    BacktestRun,
    Fill,
    MLModelState,
    MLTrainingEvent,
    Order,
    PositionSnapshot,
    Setting,
    StrategyRun,
    TradingConnection,
    User,
    VaultCycle,
    WalletBalance,
)
from app.runtime import available_modes, get_current_mode, market_mode_for
from app.routes import consumer as consumer_routes
from app.services.order_manager import OrderIntent


def test_runtime_is_always_live(app) -> None:
    Setting.set_json("current_mode", "paper")
    db.session.commit()

    assert get_current_mode() == "live"
    assert Setting.get_json("current_mode") == "live"
    assert available_modes() == ["live"]
    assert market_mode_for("paper") == "live"


def test_schema_create_tolerates_concurrent_existing_table(app, monkeypatch) -> None:
    table = next(iter(db.metadata.sorted_tables))
    original_create = table.create
    calls = {"count": 0}

    def fake_create(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OperationalError("CREATE TABLE", {}, Exception("table already exists"))
        return original_create(*args, **kwargs)

    monkeypatch.setattr(table, "create", fake_create)

    _create_all_tolerant()

    assert calls["count"] == 1


def test_live_only_order_manager_blocks_non_live_without_record(app) -> None:
    manager = app.extensions["services"]["order_manager"]

    try:
        manager.place_order(OrderIntent(symbol="BTC", side="buy", quantity=1.0, mode="paper", stop_loss=95.0))
    except ValueError as exc:
        assert "Live-only mode" in str(exc)
    else:
        raise AssertionError("non-live order was accepted")

    assert Order.query.count() == 0


def test_live_only_runtime_model_defaults_are_live(app) -> None:
    user = User(username="defaults", password_hash="hash", role="user")
    db.session.add(user)
    db.session.flush()
    model_state = MLModelState(model_key="defaults", horizon="1h")
    db.session.add(model_state)
    db.session.flush()

    run = StrategyRun(user_id=user.id, strategy_name="ema_crossover", symbol="BTC", timeframe="15m")
    order = Order(user_id=user.id, client_order_id="default-live-order", symbol="BTC", side="buy", quantity=0.1)
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDC",
        deposit_amount=100.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
    )
    event = MLTrainingEvent(model_state_id=model_state.id, source="test", source_id="1")
    db.session.add_all([run, order, cycle, event])
    db.session.flush()

    assert run.mode == "live"
    assert order.mode == "live"
    assert cycle.execution_mode == "live"
    assert event.mode == "live"


def test_wallet_balance_seed_is_live_only_and_does_not_read_simulated_portfolio(app, monkeypatch) -> None:
    user = User(username="walletseed", password_hash="hash", role="user")
    db.session.add(user)
    db.session.flush()

    monkeypatch.setattr(consumer_routes, "_sync_connection_balances", lambda user: None)
    monkeypatch.setattr(consumer_routes, "_sync_real_wallet_balances", lambda user: None)

    balances = consumer_routes._wallet_balances(user)

    assert balances
    assert WalletBalance.query.filter_by(user_id=user.id).count() == len(balances)
    assert all(balance.available_balance == 0.0 for balance in balances)
    assert all(balance.estimated_usd_value == 0.0 for balance in balances)


def test_cycle_orders_filters_to_live_cycle_records(app) -> None:
    user = User(username="cycleorders", password_hash="hash", role="user")
    db.session.add(user)
    db.session.flush()
    connection = TradingConnection(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        is_active=True,
        verification_status="verified",
    )
    db.session.add(connection)
    db.session.flush()
    cycle = VaultCycle(
        user_id=user.id,
        trading_connection_id=connection.id,
        deposit_asset="USDC",
        deposit_amount=100.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        execution_mode="live",
        unlocks_at=datetime.utcnow() + timedelta(hours=1),
    )
    db.session.add(cycle)
    db.session.flush()

    matching = Order(
        user_id=user.id,
        trading_connection_id=connection.id,
        client_order_id="cycle-match",
        mode="live",
        symbol="BTC",
        side="buy",
        quantity=0.1,
    )
    matching.details = {"vault_cycle_id": cycle.id}
    unrelated_same_user = Order(
        user_id=user.id,
        trading_connection_id=connection.id,
        client_order_id="cycle-other",
        mode="live",
        symbol="ETH",
        side="buy",
        quantity=0.1,
    )
    unrelated_same_user.details = {"vault_cycle_id": cycle.id + 1}
    non_live_same_cycle = Order(
        user_id=user.id,
        trading_connection_id=connection.id,
        client_order_id="cycle-paper",
        mode="paper",
        symbol="SOL",
        side="buy",
        quantity=0.1,
    )
    non_live_same_cycle.details = {"vault_cycle_id": cycle.id}
    db.session.add_all([matching, unrelated_same_user, non_live_same_cycle])
    db.session.commit()

    assert consumer_routes._cycle_orders(cycle) == [matching]


def test_live_only_clean_slate_purges_non_live_records_and_preserves_live_data(app) -> None:
    user = User(username="liveuser", password_hash="hash", role="user")
    db.session.add(user)
    db.session.flush()
    connection = TradingConnection(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        is_active=True,
        verification_status="verified",
    )
    db.session.add(connection)
    db.session.flush()

    live_order = Order(
        user_id=user.id,
        trading_connection_id=connection.id,
        client_order_id="live-order",
        mode="live",
        symbol="BTC",
        side="buy",
        quantity=0.1,
        status="submitted",
    )
    paper_order = Order(
        user_id=user.id,
        client_order_id="paper-order",
        mode="paper",
        symbol="BTC",
        side="buy",
        quantity=0.1,
        status="filled",
    )
    db.session.add_all([live_order, paper_order])
    db.session.flush()
    db.session.add(Fill(order_id=paper_order.id, symbol="BTC", side="buy", quantity=0.1, price=100.0, simulated=True))
    db.session.add_all(
        [
            StrategyRun(user_id=user.id, strategy_name="ema_crossover", symbol="BTC", timeframe="15m", mode="paper"),
            StrategyRun(user_id=user.id, strategy_name="ema_crossover", symbol="BTC", timeframe="15m", mode="live"),
            PositionSnapshot(user_id=user.id, mode="paper", symbol="BTC"),
            PositionSnapshot(user_id=user.id, trading_connection_id=connection.id, mode="live", symbol="BTC"),
            VaultCycle(
                user_id=user.id,
                deposit_asset="USDC",
                deposit_amount=100.0,
                settlement_asset="USDC",
                lock_duration_hours=1,
                execution_mode="paper",
                unlocks_at=datetime.utcnow() + timedelta(hours=1),
            ),
            VaultCycle(
                user_id=user.id,
                trading_connection_id=connection.id,
                deposit_asset="USDC",
                deposit_amount=100.0,
                settlement_asset="USDC",
                lock_duration_hours=1,
                execution_mode="live",
                unlocks_at=datetime.utcnow() + timedelta(hours=1),
            ),
        ]
    )
    db.session.execute(
        db.text(
            "CREATE TABLE paper_account (id INTEGER PRIMARY KEY, name VARCHAR(120), "
            "starting_balance FLOAT, cash FLOAT, realized_pnl FLOAT, base_asset VARCHAR(32))"
        )
    )
    db.session.execute(
        db.text(
            "CREATE TABLE paper_equity_snapshot (id INTEGER PRIMARY KEY, account_id INTEGER, "
            "cash FLOAT, equity FLOAT, realized_pnl FLOAT, unrealized_pnl FLOAT, position_value FLOAT)"
        )
    )
    db.session.execute(db.text("INSERT INTO paper_account (id, name, starting_balance, cash) VALUES (1, 'legacy', 1000, 1000)"))
    db.session.execute(db.text("INSERT INTO paper_equity_snapshot (id, account_id, cash, equity) VALUES (1, 1, 1000, 1000)"))
    Setting.set_json("current_mode", "paper")
    db.session.commit()

    result = app.test_cli_runner().invoke(args=["live-only-clean-slate", "--confirm", "LIVE-ONLY-RESET"])

    assert result.exit_code == 0
    assert User.query.count() == 1
    assert TradingConnection.query.count() == 1
    assert Order.query.filter_by(mode="paper").count() == 0
    assert Order.query.filter_by(mode="live").count() == 1
    assert Fill.query.count() == 0
    assert StrategyRun.query.filter_by(mode="paper").count() == 0
    assert StrategyRun.query.filter_by(mode="live").count() == 1
    assert PositionSnapshot.query.filter_by(mode="paper").count() == 0
    assert PositionSnapshot.query.filter_by(mode="live").count() == 1
    assert VaultCycle.query.filter_by(execution_mode="paper").count() == 0
    assert VaultCycle.query.filter_by(execution_mode="live").count() == 1
    assert not _table_exists("paper_account")
    assert not _table_exists("paper_equity_snapshot")
    assert Setting.get_json("current_mode") == "live"


def test_app_boots_without_paper_trading_service(app) -> None:
    assert "paper_trading" not in app.extensions["services"]
    assert not hasattr(app.extensions["services"]["order_manager"], "paper_portfolio_view")


def test_backtest_cli_is_historical_only_and_creates_no_runtime_records(app, monkeypatch) -> None:
    monkeypatch.setattr(
        app.extensions["services"]["backtest_engine"],
        "run",
        lambda config: {
            "total_return": 0.01,
            "max_drawdown": -0.01,
            "trade_count": 1,
            "trades_per_day": 0.5,
        },
    )

    result = app.test_cli_runner().invoke(args=["run-backtest", "--strategy", "ema_crossover", "--symbol", "BTC"])

    assert result.exit_code == 0
    assert BacktestRun.query.count() == 1
    assert Order.query.count() == 0
    assert StrategyRun.query.count() == 0
    assert VaultCycle.query.count() == 0


def _table_exists(table: str) -> bool:
    rows = db.session.execute(db.text(f"PRAGMA table_info({table})")).mappings().all()
    return bool(rows)
