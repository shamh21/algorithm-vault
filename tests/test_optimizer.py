from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from cryptography.fernet import Fernet

import app.cli as cli_module
from app.config import _normalize_mode
from app.extensions import db
from app.models import (
    AuditLog,
    MLMarketHistory,
    MLModelState,
    MLTrainingEvent,
    Order,
    OptimizerRun,
    Setting,
    StrategyRanking,
    StrategyRun,
    TradingConnection,
    User,
    VaultCycle,
    WalletBalance,
)
from app.ml.online_ranker import extract_features
from app.ml.decision_engine import MLDecisionEngine
from app.ml.offline_ranker import OfflineRanker
from app.strategies.base import Signal
from app.backtesting.optimizer import AGGRESSIVE_1H_WARNING, DYNAMIC_INTRADAY_WARNING, EXTREME_ROI_WARNING
from app.routes.backtests import _auto_deploy_rankings
from app.services.hyperliquid_client import ClientSnapshot
from app.services.order_manager import OrderIntent


def _optimizer_candles(days: int = 5) -> list[dict[str, Any]]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 100.0
    for index in range(days * 96):
        direction = 1 if (index // 8) % 2 == 0 else -1
        price += direction * 0.18
        timestamp = int((start + timedelta(minutes=15 * index)).timestamp() * 1000)
        rows.append(
            {
                "timestamp": timestamp,
                "open": price - 0.05,
                "high": price + 0.25,
                "low": price - 0.25,
                "close": price,
                "volume": 1000,
            }
        )
    return rows


def _minute_candles(hours: int = 75) -> list[dict[str, Any]]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 100.0
    for index in range(hours * 60):
        price += 0.02 if index % 2 == 0 else -0.01
        timestamp = int((start + timedelta(minutes=index)).timestamp() * 1000)
        rows.append(
            {
                "timestamp": timestamp,
                "open": price - 0.02,
                "high": price + 0.08,
                "low": price - 0.08,
                "close": price,
                "volume": 1000,
            }
        )
    return rows


class _CanaryStrategy:
    def generate_signal(self, *, symbol: str, timeframe: str, candles: list[dict[str, Any]], position: dict[str, Any]) -> Signal:
        return Signal(
            "buy",
            "canary test signal",
            timeframe,
            99.0,
            104.0,
            0.5,
            metadata={"signal_timestamp": candles[-1]["timestamp"] if candles else 0},
        )


class _NoTakeProfitCanaryStrategy:
    def generate_signal(self, *, symbol: str, timeframe: str, candles: list[dict[str, Any]], position: dict[str, Any]) -> Signal:
        return Signal(
            "buy",
            "canary test signal without take profit",
            timeframe,
            99.0,
            None,
            0.5,
            metadata={"signal_timestamp": candles[-1]["timestamp"] if candles else 0},
        )


class _NoStopCanaryStrategy:
    def generate_signal(self, *, symbol: str, timeframe: str, candles: list[dict[str, Any]], position: dict[str, Any]) -> Signal:
        return Signal(
            "buy",
            "canary test signal without stop",
            timeframe,
            None,
            104.0,
            0.5,
            metadata={"signal_timestamp": candles[-1]["timestamp"] if candles else 0},
        )


def _add_canary_ranking(*, profile: str = "short_term") -> tuple[User, TradingConnection, StrategyRanking]:
    user = User(username=f"canary-{profile}", password_hash="hash")
    db.session.add(user)
    db.session.flush()
    connection = TradingConnection(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        is_active=True,
        verification_status="verified",
        wallet_address="0x" + ("a" * 40),
    )
    optimizer_run = OptimizerRun(profile=profile, status="completed")
    db.session.add_all([connection, optimizer_run])
    db.session.flush()
    ranking = StrategyRanking(
        optimizer_run_id=optimizer_run.id,
        provider="hyperliquid",
        strategy_name="scalping",
        symbol="BTC",
        timeframe="1m",
        profile=profile,
        score=50.0,
        net_return_after_costs=0.03,
        recent_1h_return=0.02,
        max_drawdown=-0.02,
        profit_factor=1.5,
        edge_score=20.0,
        cost_drag_bps=2.0,
        expectancy=1.5,
        avg_win=0.01,
        avg_loss=-0.003,
        max_favorable_excursion=0.02,
        max_adverse_excursion=-0.005,
        mfe_mae_ratio=4.0,
        allocation_amount_usd=10.0,
        lock_duration_hours=1,
        leverage=1.0,
        window_stability=0.9,
        trade_count=12,
        rejected=False,
    )
    ranking.parameters = {"risk_fraction": 0.01, "take_profit_pct": 0.04, "stop_loss_pct": 0.01}
    ranking.ml_explanation = {
        "net_roi_v2": {
            "net_roi_v2_score": 45.0,
            "roi_quality_grade": "A",
            "roi_rejection_risk": "low",
            "regime_support": "regime-supported",
        }
    }
    db.session.add(ranking)
    db.session.commit()
    return user, connection, ranking


def _add_extra_canary_ranking(user: User, *, symbol: str = "ETH", provider: str = "hyperliquid") -> StrategyRanking:
    optimizer_run = OptimizerRun(profile="short_term", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    ranking = StrategyRanking(
        optimizer_run_id=optimizer_run.id,
        provider=provider,
        strategy_name="scalping",
        symbol=symbol,
        timeframe="1m",
        profile="short_term",
        score=49.0,
        net_return_after_costs=0.025,
        recent_1h_return=0.018,
        max_drawdown=-0.02,
        profit_factor=1.4,
        edge_score=18.0,
        cost_drag_bps=2.0,
        expectancy=1.2,
        avg_win=0.01,
        avg_loss=-0.003,
        max_favorable_excursion=0.02,
        max_adverse_excursion=-0.005,
        mfe_mae_ratio=4.0,
        allocation_amount_usd=10.0,
        lock_duration_hours=1,
        leverage=1.0,
        window_stability=0.9,
        trade_count=12,
        rejected=False,
        rejection_reason="",
    )
    ranking.parameters = {"risk_fraction": 0.01, "take_profit_pct": 0.04, "stop_loss_pct": 0.01}
    ranking.ml_explanation = {
        "net_roi_v2": {
            "net_roi_v2_score": 44.0,
            "roi_quality_grade": "A",
            "roi_rejection_risk": "low",
            "regime_support": "regime-supported",
        }
    }
    db.session.add(ranking)
    db.session.commit()
    return ranking


def _cli_stream(result, attr: str) -> str:
    try:
        return getattr(result, attr) or ""
    except ValueError:
        return ""


def _cli_json_payload(result) -> dict[str, Any]:
    raw = _cli_stream(result, "stdout") or result.output
    if not raw.lstrip().startswith("{"):
        start = raw.find("{")
        assert start >= 0
        raw = raw[start:]
    return json.loads(raw)


def _cli_combined_output(result) -> str:
    return "\n".join(part for part in [result.output, _cli_stream(result, "stderr")] if part)


def _patch_canary_market(app, monkeypatch, *, patch_connection_health: bool = True) -> None:
    market_data = app.extensions["services"]["market_data"]
    realtime_market = app.extensions["services"]["realtime_market"]
    market_structure = app.extensions["services"]["market_structure"]
    registry = app.extensions["services"]["strategy_registry"]
    trading_connections = app.extensions["services"]["trading_connections"]
    candles = [
        {"timestamp": index, "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.0 + index * 0.01, "volume": 1000.0}
        for index in range(200)
    ]
    monkeypatch.setattr(market_data, "get_candles", lambda symbol, timeframe, mode="live", limit=None: candles[-(limit or 200):])
    monkeypatch.setattr(market_data, "get_mid_price", lambda symbol, mode: 100.0)
    monkeypatch.setattr(
        market_data,
        "get_order_book",
        lambda symbol, mode: {"levels": [[{"px": "99.95", "sz": "10"}], [{"px": "100.05", "sz": "10"}]]},
    )
    monkeypatch.setattr(
        realtime_market,
        "snapshot",
        lambda symbol, mode, timeframe=None: {
            "source": "test",
            "spread_bps": 1.0,
            "liquidity_usd": 200_000.0,
            "volatility_pct": 0.4,
            "volatility_regime": "tradable",
            "signal_stability": 0.95,
        },
    )
    monkeypatch.setattr(market_structure, "snapshot", lambda symbol, timeframe, mode="live": {"score": 0.8, "trend_score": 0.8})
    monkeypatch.setattr(registry, "build", lambda name, parameters=None: _CanaryStrategy())
    if patch_connection_health:
        def account_snapshot(user_id, mode, connection_id=None):
            connection = db.session.get(TradingConnection, int(connection_id)) if connection_id else None
            if connection is not None and connection.provider == "kucoin":
                balances = [{"asset": "USDT", "type": "futures", "value": 100.0, "withdrawable": 0.0}]
            else:
                balances = [{"asset": "USDC", "type": "margin", "value": 100.0, "withdrawable": 100.0}]
            return ClientSnapshot(mode, balances, [], [], [], [])

        monkeypatch.setattr(
            trading_connections,
            "account_snapshot",
            account_snapshot,
        )
        monkeypatch.setattr(trading_connections, "can_trade", lambda user_id, mode, connection_id=None: mode == "live")


def _enable_canary_submit_gates(app, monkeypatch) -> None:
    app.config["CANARY_PREVIEW_ONLY"] = False
    app.config["EXPLICIT_LIVE_CONFIRMED"] = True
    app.config["SECONDARY_CONFIRMATION"] = True
    app.config["LIVE_MICRO_CANARY_ENABLED"] = True
    app.config["LIVE_MICRO_CANARY_ACCOUNT_USD"] = 10.0
    app.config["LIVE_MICRO_CANARY_MAX_ALLOCATION_USD"] = 1.0
    app.config["LIVE_MICRO_CANARY_MAX_RISK_PCT"] = 0.01
    app.config["LIVE_MICRO_CANARY_MAX_LEVERAGE"] = 1.0
    app.config["LIVE_MICRO_CANARY_REQUIRE_STOP_LOSS"] = True
    app.config["LIVE_MICRO_CANARY_REQUIRE_TAKE_PROFIT"] = True
    app.config["LIVE_MICRO_CANARY_PREVIEW_ONLY"] = False
    app.config["LIVE_MICRO_CANARY_LIVE_SUBMIT_ENABLED"] = True
    app.config["LIVE_MICRO_CANARY_ALLOW_MIN_NOTIONAL_ORDER"] = False
    app.config["LIVE_MICRO_CANARY_ORDER_BUDGET_USD"] = 10.0
    app.config["LIVE_MICRO_CANARY_DEFAULT_MIN_NOTIONAL_USD"] = 0.5
    app.config["LIVE_MICRO_CANARY_MIN_NOTIONAL_BUFFER_USD"] = 0.5
    Setting.set_json("explicit_live_confirmed", True)
    Setting.set_json("secondary_confirmation", True)
    db.session.commit()
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})


def _enable_micro_canary(app, *, max_allocation: float = 1.0, preview_only: bool = True) -> None:
    app.config["LIVE_MICRO_CANARY_ENABLED"] = True
    app.config["LIVE_MICRO_CANARY_ACCOUNT_USD"] = 10.0
    app.config["LIVE_MICRO_CANARY_MAX_ALLOCATION_USD"] = max_allocation
    app.config["LIVE_MICRO_CANARY_MAX_RISK_PCT"] = 0.01
    app.config["LIVE_MICRO_CANARY_MAX_LEVERAGE"] = 1.0
    app.config["LIVE_MICRO_CANARY_REQUIRE_STOP_LOSS"] = True
    app.config["LIVE_MICRO_CANARY_REQUIRE_TAKE_PROFIT"] = True
    app.config["LIVE_MICRO_CANARY_PREVIEW_ONLY"] = preview_only
    app.config["LIVE_MICRO_CANARY_LIVE_SUBMIT_ENABLED"] = False
    app.config["LIVE_MICRO_CANARY_ALLOW_MIN_NOTIONAL_ORDER"] = False
    app.config["LIVE_MICRO_CANARY_ORDER_BUDGET_USD"] = 10.0
    app.config["LIVE_MICRO_CANARY_DEFAULT_MIN_NOTIONAL_USD"] = max_allocation * 0.5
    app.config["LIVE_MICRO_CANARY_MIN_NOTIONAL_BUFFER_USD"] = 0.5


def _aggressive_window(
    *,
    total_return: float = 0.02,
    max_drawdown: float = -0.05,
    profit_factor: float = 1.4,
    trade_count: int = 8,
    trades_per_day: float = 48.0,
    hours_ago: int = 0,
    edge_score: float = 12.0,
    cost_drag_bps: float = 4.5,
    expectancy: float = 1.5,
    max_adverse_excursion: float = -0.004,
    max_favorable_excursion: float = 0.008,
    turnover_after_fees: float | None = None,
    no_trade_reason: str = "",
) -> dict[str, Any]:
    end = datetime(2026, 1, 4, tzinfo=timezone.utc) - timedelta(hours=hours_ago)
    row = {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "sharpe_like": 1.0,
        "sortino_like": 1.2,
        "trade_count": trade_count,
        "trades_per_day": trades_per_day,
        "average_return_per_trade": 0.002,
        "capital_turnover_rate": 1.1,
        "fees_paid": 1.25,
        "edge_score": edge_score,
        "expectancy": expectancy,
        "avg_win": 3.0,
        "avg_loss": 1.0,
        "win_loss_ratio": 3.0,
        "cost_drag_bps": cost_drag_bps,
        "max_adverse_excursion": max_adverse_excursion,
        "max_favorable_excursion": max_favorable_excursion,
        "no_trade_reason": no_trade_reason,
        "window_start": int((end - timedelta(hours=1)).timestamp() * 1000),
        "window_end": int(end.timestamp() * 1000),
    }
    if turnover_after_fees is not None:
        row["turnover_after_fees"] = turnover_after_fees
    return row


def _optimizer_window(
    *,
    total_return: float = 0.02,
    max_drawdown: float = -0.03,
    profit_factor: float = 1.4,
    trade_count: int = 8,
    turnover: float = 1.0,
    net_return_after_costs: float | None = None,
) -> dict[str, Any]:
    row = _aggressive_window(
        total_return=total_return,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        trade_count=trade_count,
    )
    row["capital_turnover_rate"] = turnover
    row["turnover_after_fees"] = max(turnover - 0.01, 0.0)
    row["win_rate"] = 0.55
    row["net_return_after_costs"] = total_return if net_return_after_costs is None else net_return_after_costs
    return row


def test_optimizer_persists_rankings_with_walk_forward_windows(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    optimizer.market_data.get_candles = lambda symbol, timeframe, mode, limit: _optimizer_candles()
    config = optimizer.default_config(symbols=["BTC"], timeframes=["15m"], strategy_names=["scalping"])
    config.training_window_days = 1
    config.testing_window_days = 1
    config.step_days = 1
    config.max_parameter_sets = 2
    config.min_trade_count = 1
    config.auto_deploy_top_n = 0

    result = optimizer.run(config)

    assert result["ranking_count"] == 2
    assert result["raw_upside_report"]["live_orders_created"] is False
    assert "raw_upside_leaderboard" in result
    assert result["optimizer_runtime"]["evaluated_candidates"] == 2
    assert result["optimizer_runtime"]["backtest_runs"] > 0
    assert result["optimizer_runtime"]["window_count"] > 0
    assert "phase_seconds" in result["optimizer_runtime"]
    assert result["optimizer_runtime"]["signal_history_limit"] == app.config["OPTIMIZER_SIGNAL_HISTORY_LIMIT"]
    assert OptimizerRun.query.first().status == "completed"
    assert StrategyRanking.query.count() == 2
    assert StrategyRun.query.count() == 0
    assert Order.query.count() == 0
    assert VaultCycle.query.count() == 0


def test_optimizer_window_slices_match_legacy_filtering(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    candles = _optimizer_candles()
    config = optimizer.default_config(symbols=["BTC"], timeframes=["15m"], strategy_names=["scalping"])
    config.training_window_days = 1
    config.testing_window_days = 1
    config.step_days = 1

    windows = optimizer._rolling_windows(candles, config)
    slices = optimizer._window_slices(candles, windows)

    assert len(slices) == len(windows)
    for window, sliced in zip(windows, slices):
        train_start, _test_start, test_end = window
        legacy = [row for row in candles if train_start <= int(row["timestamp"]) <= test_end]
        assert sliced.candles == legacy


def test_optimizer_cooperative_deadline_returns_partial_rankings(app, monkeypatch) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    optimizer.market_data.get_candles = lambda symbol, timeframe, mode, limit: _optimizer_candles()
    config = optimizer.default_config(symbols=["BTC"], timeframes=["15m"], strategy_names=["scalping"])
    config.training_window_days = 1
    config.testing_window_days = 1
    config.step_days = 1
    config.max_parameter_sets = 3
    config.min_trade_count = 1
    config.auto_deploy_top_n = 0
    checks = {"count": 0}

    def deadline_reached(_config):
        checks["count"] += 1
        return checks["count"] >= 3

    monkeypatch.setattr(optimizer, "_deadline_reached", deadline_reached)

    result = optimizer.run(config)

    assert result["partial_result"] is True
    assert result["timed_out"] is True
    assert result["partial_reason"] == "optimizer_deadline_reached"
    assert result["ranking_count"] == 1
    assert result["optimizer_runtime"]["evaluated_candidates"] == 1
    assert result["optimizer_runtime"]["partial_result"] is True
    assert OptimizerRun.query.first().status == "partial"
    assert StrategyRanking.query.count() == 1
    assert StrategyRun.query.count() == 0
    assert Order.query.count() == 0
    assert VaultCycle.query.count() == 0


def test_optimizer_prefilter_skips_untestable_candidates_with_reason(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    optimizer.market_data.get_candles = lambda symbol, timeframe, mode, limit: _minute_candles(hours=0)[:20]
    app.config["OPTIMIZER_PREFILTER_ENABLED"] = True
    config = optimizer.default_config(symbols=["BTC"], timeframes=["1m"], strategy_names=["scalping"])
    config.max_parameter_sets = 1
    config.auto_deploy_top_n = 0

    result = optimizer.run(config)

    assert result["ranking_count"] >= 1
    assert result["optimizer_runtime"]["skipped_candidates"] == result["ranking_count"]
    assert result["optimizer_runtime"]["skipped_reasons"] == {"insufficient_window_candles": result["ranking_count"]}
    assert all(row["rejected"] for row in result["top"])
    assert {row["rejection_reason"] for row in result["top"]} == {"insufficient_window_candles"}
    assert all("candidate_rejected:insufficient_window_candles" in row["live_blockers"] for row in result["top"])


def test_configured_optimizer_does_not_scan_dynamic_universe_for_capacity(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], timeframes=["1m"], strategy_names=["scalping"])
    config.universe_mode = "configured"

    def fail_liquid_universe(*_args, **_kwargs):
        raise AssertionError("configured optimizer should not scan the dynamic universe")

    optimizer.universe_service.liquid_universe = fail_liquid_universe

    assert optimizer._capacity_usd("BTC", config) == float(config.allocation_amount_usd or 0.0)
    assert optimizer._universe_source("BTC", config) == "configured"


def test_optimizer_reports_pair_screening_summary(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    optimizer.market_data.get_candles = lambda symbol, timeframe, mode, limit: _optimizer_candles()
    optimizer.market_data.get_order_book = lambda symbol, mode: {
        "levels": [[{"px": "99.99", "sz": "1000"}], [{"px": "100.01", "sz": "1000"}]]
    }
    config = optimizer.default_config(symbols=["BTC", "ETH"], timeframes=["15m"], strategy_names=["scalping"])
    config.training_window_days = 1
    config.testing_window_days = 1
    config.step_days = 1
    config.max_parameter_sets = 1
    config.min_trade_count = 1
    config.auto_deploy_top_n = 0
    config.pair_screening_enabled = True
    config.pair_trading_enabled = True

    result = optimizer.run(config)

    assert result["pair_screening_summary"]["enabled"] is True
    assert result["pair_screening_summary"]["trading_enabled"] is True
    assert "pair_candidates" in result
    assert "pair_rejection_breakdown" in result
    assert "pair_baseline_comparison" in result


def test_aggressive_1h_defaults_and_hourly_windows(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], strategy_names=["scalping"], profile="aggressive_1h")

    assert config.profile == "aggressive_1h"
    assert config.training_window_hours == 72
    assert config.testing_window_hours == 1
    assert config.step_hours == 1
    assert config.timeframes == ["1m", "5m"]
    assert config.max_parameter_sets == 25
    assert config.min_trade_count == 8

    windows = optimizer._rolling_windows(_minute_candles(), config)

    assert len(windows) == 2
    assert windows[0][1] - windows[0][0] == 72 * 60 * 60 * 1000
    assert windows[0][2] - windows[0][1] == 60 * 60 * 1000

    parameter_sets = optimizer._parameter_sets("scalping", 4, config)
    assert len(parameter_sets) <= 4
    assert any(item.get("breakeven_trigger_pct") == 0.002 for item in parameter_sets)
    assert any(item.get("take_profit_pct") == 0.009 for item in parameter_sets)


def test_aggressive_risk_adjusted_duration_defaults(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]

    one_hour = optimizer.default_config(profile="aggressive_risk_adjusted", lock_duration_hours=1)
    day = optimizer.default_config(profile="aggressive_risk_adjusted", lock_duration_hours=24)
    week = optimizer.default_config(profile="aggressive_risk_adjusted", lock_duration_hours=168)

    assert one_hour.timeframes == ["1m", "5m"]
    assert day.timeframes == ["5m", "15m", "1h"]
    assert week.timeframes == ["1h", "15m"]
    assert one_hour.testing_window_hours == 1
    assert day.testing_window_hours == 24
    assert week.testing_window_hours == 168
    assert "scalping" in one_hour.strategy_names
    assert "mean_reversion" in day.strategy_names
    assert "volatility_breakout" in week.strategy_names


def test_dynamic_intraday_defaults_use_dynamic_universe_and_shadow_gate(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(profile="dynamic_intraday")

    assert config.profile == "dynamic_intraday"
    assert config.timeframes == ["1m", "5m", "15m"]
    assert config.universe_mode == "dynamic_liquid"
    assert config.training_window_hours == 72
    assert config.testing_window_hours == 1
    assert config.require_shadow_validation is True
    assert config.dynamic_intraday_live_eligible is False
    assert config.market_structure_features_enabled is True
    assert "scalping" in config.strategy_names


def test_dynamic_intraday_scoring_reports_eligibility_and_rejects_bad_spread(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(profile="dynamic_intraday", strategy_names=["scalping"], allocation_amount_usd=100.0)
    config.min_trade_count = 1

    good = optimizer._aggregate_candidate(
        "BTC",
        "1m",
        "scalping",
        {"risk_fraction": 0.03},
        [_optimizer_window(total_return=0.03, net_return_after_costs=0.025, trade_count=10)],
        config,
        market_structure={
            "enabled": True,
            "score": 0.9,
            "book_depth_usd": 100_000.0,
            "spread_bps": 2.0,
            "spread_trend_bps": 2.0,
            "volume_impulse": 1.5,
            "volatility_regime": "tradable",
        },
    )
    bad_spread = optimizer._aggregate_candidate(
        "ETH",
        "1m",
        "scalping",
        {"risk_fraction": 0.03},
        [_optimizer_window(total_return=0.03, net_return_after_costs=0.025, trade_count=10)],
        config,
        market_structure={
            "enabled": True,
            "score": 0.4,
            "book_depth_usd": 100_000.0,
            "spread_bps": 50.0,
            "spread_trend_bps": 50.0,
            "volume_impulse": 1.0,
            "volatility_regime": "tradable",
        },
    )

    assert good["profile"] == "dynamic_intraday"
    assert good["experimental"] is False
    assert good["warning"] == DYNAMIC_INTRADAY_WARNING
    assert good["live_eligibility"]["status"] == "shadow_live_eligible"
    assert good["cost_drag_bps"] >= 20.0
    assert good["liquidity_capacity_usd"] == 100_000.0
    assert good["spread_bps"] == 2.0
    assert good["volatility_regime"] == "tradable"
    assert "recent_decay" in good
    assert not good["rejected"]
    assert bad_spread["rejected"]
    assert bad_spread["rejection_reason"] == "spread_above_threshold"
    diagnostics = optimizer._dynamic_intraday_diagnostics([good, bad_spread], config)
    assert diagnostics["enabled"] is True
    assert diagnostics["candidate_status_counts"] == {"shadow_live_eligible": 1, "paper_only": 1}
    assert diagnostics["top_candidates"][0]["spread_bps"] == 2.0


def test_dynamic_intraday_cli_accepts_dynamic_universe(app, monkeypatch) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    captured = {}

    def fake_run(config):
        captured["profile"] = config.profile
        captured["universe_mode"] = config.universe_mode
        captured["max_parallel_legs"] = config.max_parallel_legs
        return {"ok": True, "profile": config.profile}

    monkeypatch.setattr(optimizer, "run", fake_run)
    result = app.test_cli_runner().invoke(
        args=[
            "run-optimization",
            "--profile",
            "dynamic_intraday",
            "--universe-mode",
            "dynamic_liquid",
            "--max-parallel-legs",
            "3",
        ]
    )

    assert result.exit_code == 0
    assert captured == {"profile": "dynamic_intraday", "universe_mode": "dynamic_liquid", "max_parallel_legs": 3}
    assert '"profile": "dynamic_intraday"' in result.output


def test_dynamic_intraday_cli_reports_live_readiness_and_override_flags(app, monkeypatch) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    captured = {}

    def fake_run(config):
        captured["dynamic_intraday_live_eligible"] = config.dynamic_intraday_live_eligible
        captured["require_shadow_validation"] = config.require_shadow_validation
        return {"ok": True, "profile": config.profile, "top": []}

    monkeypatch.setattr(optimizer, "run", fake_run)
    result = app.test_cli_runner().invoke(
        args=[
            "run-optimization",
            "--profile",
            "dynamic_intraday",
            "--dynamic-intraday-live-eligible",
            "--no-require-shadow-validation",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    readiness = payload["dynamic_intraday_live_readiness"]
    assert captured == {"dynamic_intraday_live_eligible": True, "require_shadow_validation": False}
    assert readiness["dynamic_intraday_live_eligible"] is True
    assert readiness["require_shadow_validation"] is False
    assert readiness["candidate_default_status"] == "live_eligible_after_risk_checks"
    assert readiness["enable_live_trading"] is True
    assert readiness["warnings"]


def test_run_optimization_reports_high_upside_readiness_and_scanner_diagnostics(app, monkeypatch) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    scanner = app.extensions["services"]["market_scanner"]
    captured = {}

    def fake_run(config):
        captured["high_upside_profile"] = config.high_upside_profile
        return {"ok": True, "profile": config.profile, "top": []}

    def fake_score(*args, **kwargs):
        scanner.last_scan_diagnostics = {
            "accepted": [{"symbol": "BTC", "upside_screen_score": 2.5, "score_breakdown": {"volume_impulse": 1.0}}],
            "rejected": [{"symbol": "ETH", "rejection_reason": "cost_drag_above_threshold"}],
            "rejection_breakdown": {"cost_drag_above_threshold": 1},
            "rejection_rate": 0.5,
            "cache_hit": False,
        }
        return []

    monkeypatch.setattr(optimizer, "run", fake_run)
    monkeypatch.setattr(scanner, "score_candidates", fake_score)
    result = app.test_cli_runner().invoke(
        args=[
            "run-optimization",
            "--profile",
            "aggressive_1h",
            "--high-upside-profile",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert captured["high_upside_profile"] is True
    assert payload["scanner_diagnostics"]["accepted"][0]["symbol"] == "BTC"
    assert payload["high_upside_live_readiness"]["requested"] is True
    assert "HIGH_UPSIDE_PROFILE_ENABLED=false" in payload["high_upside_live_readiness"]["blockers"]


def test_high_upside_discovery_is_disabled_by_default_and_never_submits(app, monkeypatch) -> None:
    order_manager = app.extensions["services"]["order_manager"]
    monkeypatch.setattr(order_manager, "place_order", lambda intent: (_ for _ in ()).throw(AssertionError("no orders")))

    result = app.test_cli_runner().invoke(
        args=[
            "discover-high-upside-vault-candidates",
            "--provider",
            "hyperliquid",
            "--symbol",
            "BTC",
            "--timeframe",
            "5m",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is False
    assert payload["research_only"] is True
    assert payload["live_orders_created"] is False
    assert "HIGH_UPSIDE_DISCOVERY_ENABLED=false" in payload["blockers"]
    assert "ml_readiness" in payload
    assert "ml_models" in payload
    assert "ml_blockers" in payload


def test_ml_history_backfill_accepts_kucoin_provider_without_orders(app, monkeypatch) -> None:
    app.config["ML_HISTORY_BACKFILL_ENABLED"] = True
    market_data = app.extensions["services"]["market_data"]
    order_manager = app.extensions["services"]["order_manager"]
    monkeypatch.setattr(market_data, "get_candles", lambda symbol, timeframe, mode="live", limit=None: _optimizer_candles(days=2)[:120])
    monkeypatch.setattr(
        market_data,
        "get_order_book",
        lambda symbol, mode: {"levels": [[{"px": "100", "sz": "2"}], [{"px": "101", "sz": "2"}]]},
    )
    monkeypatch.setattr(order_manager, "place_order", lambda intent: (_ for _ in ()).throw(AssertionError("no orders")))

    result = app.test_cli_runner().invoke(
        args=[
            "ml-history-backfill",
            "--provider",
            "kucoin",
            "--symbol",
            "BTC",
            "--timeframe",
            "1h",
            "--max-symbols",
            "1",
            "--lookback-days",
            "2",
            "--confirm",
            "BACKFILL-ML-HISTORY",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["research_only"] is True
    assert payload["live_orders_created"] is False
    assert payload["ready"] is True
    assert payload["providers"][0]["provider"] == "kucoin"
    assert payload["providers"][0]["rows_created"] == 1
    assert payload["providers"][0]["market_data_supported"] is True
    assert "provider_market_data_unavailable" not in payload["providers"][0].get("blockers", [])
    row = MLMarketHistory.query.one()
    assert row.provider == "kucoin"
    assert row.symbol == "BTC"


def test_ml_history_backfill_persists_hyperliquid_rows(app, monkeypatch) -> None:
    app.config["ML_HISTORY_BACKFILL_ENABLED"] = True
    market_data = app.extensions["services"]["market_data"]
    monkeypatch.setattr(market_data, "get_candles", lambda symbol, timeframe, mode="live", limit=None: _optimizer_candles(days=2)[:120])
    monkeypatch.setattr(
        market_data,
        "get_order_book",
        lambda symbol, mode: {"levels": [[{"px": "100", "sz": "2"}], [{"px": "101", "sz": "2"}]]},
    )

    result = app.test_cli_runner().invoke(
        args=[
            "ml-history-backfill",
            "--provider",
            "hyperliquid",
            "--symbol",
            "BTC",
            "--timeframe",
            "1h",
            "--max-symbols",
            "1",
            "--lookback-days",
            "2",
            "--confirm",
            "BACKFILL-ML-HISTORY",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is True
    assert payload["market_history"]["created_rows"] == 1
    row = MLMarketHistory.query.one()
    assert row.provider == "hyperliquid"
    assert row.symbol == "BTC"
    assert row.candle_count == 120
    assert row.liquidity_usd > 0


def test_ml_history_backfill_all_pairs_preserves_hyperliquid_venue_symbols(app, monkeypatch) -> None:
    app.config["ML_HISTORY_BACKFILL_ENABLED"] = True
    market_data = app.extensions["services"]["market_data"]

    monkeypatch.setattr(
        market_data,
        "get_all_mids",
        lambda mode: {"BTC": 100.0, "kPEPE": 0.01, "@1": 1.0, "USDC": 1.0, "DELIST": 2.0},
    )
    monkeypatch.setattr(
        market_data.client,
        "get_perp_meta_and_asset_contexts",
        lambda mode: (
            {
                "universe": [
                    {"name": "BTC"},
                    {"name": "kPEPE"},
                    {"name": "@1"},
                    {"name": "USDC"},
                    {"name": "DELIST", "isDelisted": True},
                ]
            },
            [{"openInterest": "100"}, {"openInterest": "50"}, {}, {}, {}],
        ),
    )
    seen_symbols: list[str] = []

    def fake_candles(symbol, timeframe, mode="live", limit=None):
        seen_symbols.append(symbol)
        return _optimizer_candles(days=2)[:120]

    monkeypatch.setattr(market_data, "get_candles", fake_candles)
    monkeypatch.setattr(
        market_data,
        "get_order_book",
        lambda symbol, mode: {"levels": [[{"px": "100", "sz": "2"}], [{"px": "101", "sz": "2"}]]},
    )

    result = app.test_cli_runner().invoke(
        args=[
            "ml-history-backfill",
            "--provider",
            "hyperliquid",
            "--all-pairs",
            "--timeframe",
            "1h",
            "--confirm",
            "BACKFILL-ML-HISTORY",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    provider = payload["providers"][0]
    assert payload["market_history"]["created_rows"] == 2
    assert provider["stage1_diagnostics"]["total_tradable_pairs"] == 2
    assert provider["stage1_diagnostics"]["skip_breakdown"]["numeric_venue_symbol_unsupported"] == 1
    assert provider["stage1_diagnostics"]["skip_breakdown"]["quote_or_stable_symbol"] == 1
    assert provider["stage1_diagnostics"]["skip_breakdown"]["delisted"] == 1
    assert "kPEPE" in seen_symbols
    rows = MLMarketHistory.query.order_by(MLMarketHistory.symbol.asc()).all()
    assert [row.provider for row in rows] == ["hyperliquid", "hyperliquid"]
    assert sorted(row.symbol for row in rows) == ["BTC", "KPEPE"]
    assert {row.diagnostics["venue_symbol"] for row in rows} == {"BTC", "kPEPE"}


def test_offline_ranker_market_history_rows_reach_provider_minimum_without_future_leakage(app) -> None:
    app.config["ML_OFFLINE_MARKET_HISTORY_SAMPLES_PER_WINDOW"] = 250
    for index in range(9):
        row = MLMarketHistory(
            provider="hyperliquid",
            symbol=f"PAIR{index}",
            timeframe="15m",
            mode="live",
            status="ok",
        )
        row.candles = _optimizer_candles(days=14)
        row.diagnostics = {"venue_symbol": f"pair{index}"}
        db.session.add(row)
    db.session.commit()

    rows = OfflineRanker(app.config).training_rows("1h", provider="hyperliquid", use_market_history=True)

    assert len(rows) >= 2000
    assert {row.provider for row in rows} == {"hyperliquid"}
    assert rows[0].source == "ml_market_history:offline_ranker"


def test_train_ml_suite_can_include_market_history_rows(app, monkeypatch) -> None:
    app.config["ML_ALL_AREAS_ENABLED"] = True
    row = MLMarketHistory(
        provider="hyperliquid",
        symbol="BTC",
        timeframe="1h",
        mode="live",
        status="ok",
        liquidity_usd=10000.0,
        spread_bps=5.0,
    )
    row.candles = _optimizer_candles(days=2)[:120]
    row.order_book = {"levels": [[{"px": "100", "sz": "2"}], [{"px": "101", "sz": "2"}]]}
    db.session.add(row)
    db.session.commit()
    monkeypatch.setattr(MLDecisionEngine, "_module_available", staticmethod(lambda name: False))

    result = app.test_cli_runner().invoke(
        args=[
            "train-ml-suite",
            "--horizon",
            "1h",
            "--model-family",
            "pytorch_extreme_upside",
            "--objective",
            "extreme_upside",
            "--use-market-history",
            "--confirm",
            "TRAIN-ML-SUITE",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    family = payload["family_results"]["pytorch_extreme_upside"]
    assert payload["research_only"] is True
    assert family["training_rows"] >= 1
    assert family["training_dataset"]["market_history_rows"] >= 1
    assert "pytorch_extreme_upside:torch_missing" in payload["blockers"]


def test_train_ml_signal_model_can_use_market_history_rows(app, monkeypatch) -> None:
    app.config["ML_SIGNAL_MODEL_ENABLED"] = True
    app.config["ML_SIGNAL_MIN_TRAINING_ROWS"] = 5
    row = MLMarketHistory(
        provider="hyperliquid",
        symbol="BTC",
        timeframe="1h",
        mode="live",
        status="ok",
    )
    row.candles = _optimizer_candles(days=5)[:120]
    db.session.add(row)
    db.session.commit()
    monkeypatch.setattr(app.extensions["services"]["ml_signal_model"], "_module_available", lambda name: False if name == "torch" else True)

    result = app.test_cli_runner().invoke(
        args=[
            "train-ml-signal-model",
            "--horizon",
            "1h",
            "--model",
            "pytorch_gru",
            "--objective",
            "extreme_upside",
            "--use-market-history",
            "--confirm",
            "TRAIN-ML-SIGNAL-MODEL",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["trained"] is False
    assert payload["training_dataset"]["market_history_rows"] >= 1
    assert payload["target_distribution"]["rows"] >= payload["training_dataset"]["market_history_rows"]
    assert "torch_missing" in payload["blockers"]
    assert "insufficient_training_rows" not in payload["blockers"]


def test_market_history_training_rows_do_not_leak_forward_return_features(app) -> None:
    row = MLMarketHistory(
        provider="hyperliquid",
        symbol="BTC",
        timeframe="15m",
        mode="live",
        status="ok",
    )
    row.candles = _optimizer_candles(days=2)
    row.diagnostics = {"venue_symbol": "BTC"}
    db.session.add(row)
    db.session.commit()

    offline_rows = OfflineRanker(app.config).training_rows("1h", provider="hyperliquid", use_market_history=True)
    decision_rows = app.extensions["services"]["ml_decision_engine"].training_rows(
        "pytorch_extreme_upside",
        "1h",
        objective="extreme_upside",
        use_market_history=True,
        provider="hyperliquid",
    )
    signal_rows = app.extensions["services"]["ml_signal_model"].training_rows(
        "1h",
        objective="extreme_upside",
        use_market_history=True,
        provider="hyperliquid",
    )

    assert offline_rows
    assert decision_rows
    assert signal_rows
    assert any(abs(row.target) > 0 for row in offline_rows)
    assert any(abs(row.target) > 0 for row in decision_rows)
    assert any(abs(row.target_return) > 0 for row in signal_rows)
    target_only_features = {
        "net_return_after_costs",
        "total_return",
        "recent_return",
        "recent_1h_return",
        "recent_performance_score",
        "avg_trade_return",
    }
    for feature_row in [offline_rows[0].features, decision_rows[0].features, signal_rows[0].features]:
        assert {key: feature_row.get(key, 0.0) for key in target_only_features} == {
            key: 0.0 for key in target_only_features
        }


def test_ml_feedback_sync_blocks_by_default_without_orders(app) -> None:
    result = app.test_cli_runner().invoke(
        args=["ml-feedback-sync", "--horizon", "all", "--source", "all", "--confirm", "SYNC-ML-FEEDBACK"]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["research_only"] is True
    assert payload["live_orders_created"] is False
    assert payload["feedback_sync"]["created_events"] == 0
    assert "ML_FEEDBACK_SYNC_ENABLED=false" in payload["blockers"]
    assert Order.query.count() == 0


def test_ml_feedback_sync_creates_deduped_training_events(app) -> None:
    app.config["ML_FEEDBACK_SYNC_ENABLED"] = True
    _user, _connection, ranking = _add_canary_ranking(profile="aggressive_1h")

    result = app.test_cli_runner().invoke(
        args=[
            "ml-feedback-sync",
            "--horizon",
            "1h",
            "--source",
            "rankings",
            "--confirm",
            "SYNC-ML-FEEDBACK",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["feedback_sync"]["created_events"] == 1
    event = MLTrainingEvent.query.filter_by(
        source="feedback_sync:ranking",
        source_id=str(ranking.id),
        horizon="1h",
    ).one()
    assert event.features
    assert event.details["feedback_sync"] is True

    second = app.test_cli_runner().invoke(
        args=[
            "ml-feedback-sync",
            "--horizon",
            "1h",
            "--source",
            "rankings",
            "--confirm",
            "SYNC-ML-FEEDBACK",
        ]
    )
    second_payload = _cli_json_payload(second)
    assert second_payload["feedback_sync"]["created_events"] == 0
    assert second_payload["feedback_sync"]["skipped_existing"] >= 1


def test_ml_feedback_sync_filters_market_history_by_provider(app) -> None:
    app.config["ML_FEEDBACK_SYNC_ENABLED"] = True
    for provider in ("global", "hyperliquid"):
        row = MLMarketHistory(
            provider=provider,
            symbol="BTC",
            timeframe="15m",
            mode="live",
            status="ok",
        )
        row.candles = _optimizer_candles(days=2)[:120]
        db.session.add(row)
    db.session.commit()

    result = app.test_cli_runner().invoke(
        args=[
            "ml-feedback-sync",
            "--provider",
            "hyperliquid",
            "--horizon",
            "1h",
            "--source",
            "market_history",
            "--max-rows",
            "50",
            "--confirm",
            "SYNC-ML-FEEDBACK",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["feedback_sync"]["created_events"] == 1
    event = MLTrainingEvent.query.filter_by(source="feedback_sync:market_history").one()
    assert event.provider == "hyperliquid"
    assert event.details["provider"] == "hyperliquid"


def test_high_upside_discovery_scores_and_persists_accepted_ranking(app, monkeypatch) -> None:
    app.config["HIGH_UPSIDE_DISCOVERY_ENABLED"] = True
    scanner = app.extensions["services"]["market_scanner"]
    optimizer = app.extensions["services"]["strategy_optimizer"]
    order_manager = app.extensions["services"]["order_manager"]
    monkeypatch.setattr(order_manager, "place_order", lambda intent: (_ for _ in ()).throw(AssertionError("no orders")))

    def fake_score(symbols, **kwargs):
        scanner.last_scan_diagnostics = {
            "accepted": [
                {
                    "symbol": "BTC",
                    "score": 8.5,
                    "momentum_acceleration": 0.02,
                    "cost_adjusted_expected_move": 22.0,
                    "score_breakdown": {"technical": 2.0, "offline_ml": 0.5},
                    "offline_ml_status": "promoted",
                    "rejection_reason": "",
                }
            ],
            "rejected": [{"symbol": "ETH", "rejection_reason": "spread_above_threshold"}],
            "rejection_breakdown": {"spread_above_threshold": 1},
            "rejection_rate": 0.5,
        }
        return []

    def fake_run(config):
        run = OptimizerRun(profile=config.profile, status="completed")
        run.symbols = list(config.symbols)
        run.timeframes = list(config.timeframes)
        db.session.add(run)
        db.session.flush()
        ranking = StrategyRanking(
            optimizer_run_id=run.id,
            provider=config.provider,
            strategy_name="scalping",
            symbol=config.symbols[0],
            timeframe=config.timeframes[0],
            profile=config.profile,
            score=3.2,
            net_return_after_costs=0.04,
            recent_1h_return=0.02,
            max_drawdown=-0.04,
            profit_factor=1.6,
            trade_count=12,
            leverage=2.0,
            rejected=False,
        )
        ranking.parameters = {
            "high_upside_profile": True,
            "stop_loss_pct": 0.01,
            "take_profit_pct": 0.02,
            "leverage": 2.0,
        }
        db.session.add(ranking)
        db.session.commit()
        return {
            "optimizer_run_id": run.id,
            "ranking_count": 1,
            "accepted_count": 1,
            "top": [{"id": ranking.id, "rejected": False}],
            "rejection_breakdown": {},
            "optimizer_runtime": {"timed_out": False},
        }

    monkeypatch.setattr(scanner, "score_candidates", fake_score)
    monkeypatch.setattr(optimizer, "run", fake_run)

    result = app.test_cli_runner().invoke(
        args=[
            "discover-high-upside-vault-candidates",
            "--provider",
            "hyperliquid",
            "--symbol",
            "BTC",
            "--symbol",
            "ETH",
            "--timeframe",
            "5m",
            "--profile",
            "aggressive_1h",
            "--max-sweeps",
            "1",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is True
    assert payload["live_orders_created"] is False
    assert payload["accepted_ranking_ids"]
    assert payload["accepted_rankings"][0]["high_upside_profile"] is True
    assert payload["scanner_candidates"][0]["direction"] == "long"
    assert "ml_readiness" in payload
    assert "ml_models" in payload
    assert "ml_blockers" in payload
    assert payload["sweeps_used"] == 1
    assert "running_backtests" in _cli_combined_output(result)


def test_high_upside_discovery_extreme_objective_is_research_only(app, monkeypatch) -> None:
    app.config["HIGH_UPSIDE_DISCOVERY_ENABLED"] = True
    scanner = app.extensions["services"]["market_scanner"]
    optimizer = app.extensions["services"]["strategy_optimizer"]
    order_manager = app.extensions["services"]["order_manager"]
    monkeypatch.setattr(order_manager, "place_order", lambda intent: (_ for _ in ()).throw(AssertionError("no orders")))
    monkeypatch.setattr(
        app.extensions["services"]["ml_decision_engine"],
        "decision",
        lambda family, context, horizon="1h", candles=None: {
            "ready": family == "pytorch_extreme_upside",
            "family": family,
            "action": "pursue" if family == "pytorch_extreme_upside" else "hold",
            "confidence": 0.8,
            "blockers": [] if family == "pytorch_extreme_upside" else ["not_ready"],
            "raw": {
                "extreme_upside_probability": 0.9,
                "distance_to_target_pct": 100.0,
                "suggested_notional_usdc": 8.0,
                "suggested_leverage": 1.0,
            },
        },
    )

    def fake_score(symbols, **kwargs):
        scanner.last_scan_diagnostics = {
            "accepted": [
                {"symbol": "ETH", "score": 6.0, "momentum_acceleration": 0.01, "cost_adjusted_expected_move": 10.0},
                {"symbol": "BTC", "score": 4.0, "momentum_acceleration": 0.02, "cost_adjusted_expected_move": 12.0},
            ],
            "rejected": [],
            "rejection_breakdown": {},
            "rejection_rate": 0.0,
        }
        return []

    def fake_run(config):
        run = OptimizerRun(profile=config.profile, status="completed")
        run.symbols = list(config.symbols)
        run.timeframes = list(config.timeframes)
        db.session.add(run)
        db.session.flush()
        ranking = StrategyRanking(
            optimizer_run_id=run.id,
            provider=config.provider,
            strategy_name="scalping",
            symbol=config.symbols[0],
            timeframe=config.timeframes[0],
            profile=config.profile,
            score=3.2,
            net_return_after_costs=0.04,
            profit_factor=1.6,
            trade_count=12,
            max_drawdown=-0.02,
            leverage=1.0,
            rejected=False,
        )
        ranking.parameters = {"high_upside_profile": True, "stop_loss_pct": 0.01, "take_profit_pct": 0.02, "leverage": 1.0}
        db.session.add(ranking)
        db.session.commit()
        return {"optimizer_run_id": run.id, "ranking_count": 1, "accepted_count": 1, "rejection_breakdown": {}}

    monkeypatch.setattr(scanner, "score_candidates", fake_score)
    monkeypatch.setattr(optimizer, "run", fake_run)

    result = app.test_cli_runner().invoke(
        args=[
            "discover-high-upside-vault-candidates",
            "--provider",
            "hyperliquid",
            "--symbol",
            "BTC",
            "--symbol",
            "ETH",
            "--timeframe",
            "1h",
            "--profile",
            "aggressive_1h",
            "--max-sweeps",
            "1",
            "--objective",
            "extreme_upside",
            "--target-roi-pct",
            "1000",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["research_only"] is True
    assert payload["live_orders_created"] is False
    assert payload["objective"] == "extreme_upside"
    assert payload["target_roi_pct"] == 1000.0
    assert payload["target_roi_policy"]["target_is_aspirational"] is True
    assert payload["scanner_candidates"][0]["ml_extreme_upside_decision"]["action"] == "pursue"


def test_high_upside_discovery_accepts_kucoin_provider_for_research(app, monkeypatch) -> None:
    app.config["HIGH_UPSIDE_DISCOVERY_ENABLED"] = True
    scanner = app.extensions["services"]["market_scanner"]
    optimizer = app.extensions["services"]["strategy_optimizer"]
    order_manager = app.extensions["services"]["order_manager"]
    monkeypatch.setattr(order_manager, "place_order", lambda intent: (_ for _ in ()).throw(AssertionError("no orders")))
    monkeypatch.setattr(optimizer, "run", lambda config: (_ for _ in ()).throw(AssertionError("no optimizer sweep")))

    def fake_score(symbols, **kwargs):
        scanner.last_scan_diagnostics = {
            "accepted": [],
            "rejected": [{"symbol": "BTC", "reason": "scanner_test_rejected"}],
            "rejection_breakdown": {"scanner_test_rejected": 1},
            "rejection_rate": 1.0,
        }
        return []

    monkeypatch.setattr(scanner, "score_candidates", fake_score)

    result = app.test_cli_runner().invoke(
        args=[
            "discover-high-upside-vault-candidates",
            "--provider",
            "kucoin",
            "--symbol",
            "BTC",
            "--timeframe",
            "5m",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is False
    assert payload["providers"][0]["provider"] == "kucoin"
    assert payload["providers"][0]["status"] != "skipped"
    assert payload["providers"][0]["market_data_supported"] is True
    assert "provider_market_data_unavailable" not in payload["providers"][0].get("blockers", [])
    assert "accepted_ranking_missing" in payload["blockers"]


def test_high_upside_discovery_ml_cannot_accept_rejected_backtest(app, monkeypatch) -> None:
    app.config["HIGH_UPSIDE_DISCOVERY_ENABLED"] = True
    scanner = app.extensions["services"]["market_scanner"]
    optimizer = app.extensions["services"]["strategy_optimizer"]
    order_manager = app.extensions["services"]["order_manager"]
    monkeypatch.setattr(order_manager, "place_order", lambda intent: (_ for _ in ()).throw(AssertionError("no orders")))

    def fake_score(symbols, **kwargs):
        scanner.last_scan_diagnostics = {
            "accepted": [
                {
                    "symbol": "BTC",
                    "score": 20.0,
                    "momentum_acceleration": 0.03,
                    "cost_adjusted_expected_move": 30.0,
                    "offline_ml_status": "promoted",
                    "rejection_reason": "",
                }
            ],
            "rejected": [],
            "rejection_breakdown": {},
            "rejection_rate": 0.0,
        }
        return []

    def fake_run(config):
        run = OptimizerRun(profile=config.profile, status="completed")
        db.session.add(run)
        db.session.flush()
        ranking = StrategyRanking(
            optimizer_run_id=run.id,
            strategy_name="scalping",
            symbol=config.symbols[0],
            timeframe=config.timeframes[0],
            profile=config.profile,
            score=99.0,
            ml_adjusted_score=999.0,
            net_return_after_costs=-0.01,
            max_drawdown=-0.02,
            profit_factor=0.8,
            trade_count=20,
            rejected=True,
            rejection_reason="profit_factor_below_one",
        )
        ranking.parameters = {"high_upside_profile": True, "stop_loss_pct": 0.01, "take_profit_pct": 0.02}
        db.session.add(ranking)
        db.session.commit()
        return {
            "optimizer_run_id": run.id,
            "ranking_count": 1,
            "accepted_count": 0,
            "top": [{"id": ranking.id, "rejected": True, "rejection_reason": "profit_factor_below_one"}],
            "rejection_breakdown": {"profit_factor_below_one": 1},
            "optimizer_runtime": {"timed_out": False},
        }

    monkeypatch.setattr(scanner, "score_candidates", fake_score)
    monkeypatch.setattr(optimizer, "run", fake_run)

    result = app.test_cli_runner().invoke(
        args=[
            "discover-high-upside-vault-candidates",
            "--provider",
            "hyperliquid",
            "--symbol",
            "BTC",
            "--timeframe",
            "5m",
            "--profile",
            "aggressive_1h",
            "--max-sweeps",
            "1",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is False
    assert payload["accepted_ranking_ids"] == []
    assert "accepted_ranking_missing" in payload["blockers"]
    assert payload["rejection_breakdown"]["profit_factor_below_one"] == 1


def test_high_upside_discovery_uses_promoted_ml_signal_for_candidate_order(app, monkeypatch) -> None:
    app.config["HIGH_UPSIDE_DISCOVERY_ENABLED"] = True
    app.config["ML_SIGNAL_MODEL_ENABLED"] = True
    app.config["HIGH_UPSIDE_REQUIRE_ML_SIGNAL"] = True
    scanner = app.extensions["services"]["market_scanner"]
    signal_model = app.extensions["services"]["ml_signal_model"]
    order_manager = app.extensions["services"]["order_manager"]
    monkeypatch.setattr(order_manager, "place_order", lambda intent: (_ for _ in ()).throw(AssertionError("no orders")))

    def fake_score(symbols, **kwargs):
        scanner.last_scan_diagnostics = {
            "accepted": [
                {"symbol": "BTC", "score": 100.0, "momentum_acceleration": 0.01, "cost_adjusted_expected_move": 10.0},
                {"symbol": "ETH", "score": 10.0, "momentum_acceleration": 0.02, "cost_adjusted_expected_move": 20.0},
            ],
            "rejected": [],
            "rejection_breakdown": {},
            "rejection_rate": 0.0,
        }
        return []

    def fake_signal_score(context, horizon, **kwargs):
        symbol = context["symbol"]
        confidence = 0.91 if symbol == "ETH" else 0.62
        return {
            "enabled": True,
            "status": "promoted",
            "ready_for_live": True,
            "action": "buy",
            "confidence": confidence,
            "expected_return": 0.02 if symbol == "ETH" else 0.001,
            "sizing_score": confidence,
            "suggested_stop_loss_pct": 0.005,
            "suggested_take_profit_pct": 0.012,
            "blockers": [],
        }

    monkeypatch.setattr(scanner, "score_candidates", fake_score)
    monkeypatch.setattr(signal_model, "score_payload", fake_signal_score)

    result = app.test_cli_runner().invoke(
        args=[
            "discover-high-upside-vault-candidates",
            "--provider",
            "hyperliquid",
            "--symbol",
            "BTC",
            "--symbol",
            "ETH",
            "--timeframe",
            "5m",
            "--no-run-backtests",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["live_orders_created"] is False
    assert [row["symbol"] for row in payload["scanner_candidates"][:2]] == ["ETH", "BTC"]
    assert payload["scanner_candidates"][0]["ml_signal_model"]["status"] == "promoted"


def test_high_upside_discovery_preselects_and_filters_hyperliquid_symbols(app, monkeypatch) -> None:
    app.config["HIGH_UPSIDE_DISCOVERY_ENABLED"] = True
    app.config["ALLOWED_SYMBOLS"] = ["BTC", "ETH"]
    app.config["HIGH_UPSIDE_STAGE1_MAX_MIDS"] = 8
    market_data = app.extensions["services"]["market_data"]
    scanner = app.extensions["services"]["market_scanner"]
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        market_data,
        "get_all_mids",
        lambda mode: {
            "@1": 1.0,
            "kPEPE": 0.001,
            "BTC": 100.0,
            "ETH": 50.0,
            "AAVE": 10.0,
            "USDC": 1.0,
        },
    )

    def fake_score(symbols, **kwargs):
        captured["symbols"] = list(symbols)
        scanner.last_scan_diagnostics = {
            "accepted": [],
            "rejected": [{"symbol": symbol, "rejection_reason": "low_net_roi_edge"} for symbol in symbols],
            "rejection_breakdown": {"low_net_roi_edge": len(symbols)},
            "rejection_rate": 1.0,
        }
        return []

    monkeypatch.setattr(scanner, "score_candidates", fake_score)
    monkeypatch.setattr(
        app.extensions["services"]["order_manager"],
        "place_order",
        lambda intent: (_ for _ in ()).throw(AssertionError("no orders")),
    )

    result = app.test_cli_runner().invoke(
        args=[
            "discover-high-upside-vault-candidates",
            "--provider",
            "hyperliquid",
            "--timeframe",
            "5m",
            "--max-symbols",
            "2",
            "--no-run-backtests",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    provider = payload["providers"][0]
    assert captured["symbols"] == ["BTC", "ETH"]
    assert provider["stage1_diagnostics"]["selected_count"] == 2
    assert provider["stage1_diagnostics"]["skip_breakdown"]["numeric_venue_symbol_unsupported"] == 1
    assert "case_sensitive_venue_symbol_unsupported" not in provider["stage1_diagnostics"]["skip_breakdown"]
    assert "kPEPE" in provider["stage1_diagnostics"]["omitted_symbols"]
    assert provider["stage1_diagnostics"]["skip_breakdown"]["quote_or_stable_symbol"] == 1
    assert provider["market_data_policy"]["stage1_uses_mids_only"] is True
    assert payload["live_orders_created"] is False


def test_market_scanner_reports_provider_rate_limits(app, monkeypatch) -> None:
    app.config["HOT_TOKEN_SCAN_ENABLED"] = False
    scanner = app.extensions["services"]["market_scanner"]
    market_data = app.extensions["services"]["market_data"]
    monkeypatch.setattr(scanner.universe_service, "liquid_universe", lambda mode, timeframe: [])
    monkeypatch.setattr(market_data, "get_candles", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("429 too many requests")))

    scored = scanner.score_candidates(
        ["BTC"],
        mode="live",
        timeframe="5m",
        duration_seconds=3600,
        strategy_name="scalping",
        optimizer_profile="aggressive_1h",
    )

    assert scored == []
    assert scanner.last_scan_diagnostics["rejection_breakdown"]["provider_rate_limited"] == 1
    assert scanner.last_scan_diagnostics["rejected"][0]["rate_limited"] is True


def test_market_scanner_bounded_high_upside_skips_broad_universe_refresh(app, monkeypatch) -> None:
    app.config["HIGH_UPSIDE_PROFILE_ENABLED"] = True
    app.config["HIGH_UPSIDE_BOUNDED_SCANNER_UNIVERSE"] = True
    scanner = app.extensions["services"]["market_scanner"]
    market_data = app.extensions["services"]["market_data"]

    monkeypatch.setattr(
        scanner,
        "hot_tokens",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("hot token scan should be bounded out")),
    )
    monkeypatch.setattr(
        scanner.universe_service,
        "liquid_universe",
        lambda mode, timeframe: (_ for _ in ()).throw(AssertionError("broad universe refresh should be bounded out")),
    )
    monkeypatch.setattr(market_data, "get_candles", lambda *args, **kwargs: _optimizer_candles(days=2))
    monkeypatch.setattr(
        market_data,
        "get_order_book",
        lambda *args, **kwargs: {
            "levels": [
                [{"px": "99.9", "sz": "500"}],
                [{"px": "100.1", "sz": "500"}],
            ]
        },
    )

    scanner.score_candidates(
        ["BTC"],
        mode="live",
        timeframe="5m",
        duration_seconds=3600,
        strategy_name="scalping",
        optimizer_profile="aggressive_1h",
    )

    assert scanner.last_scan_diagnostics["bounded_universe"] is True
    assert scanner.last_scan_diagnostics["broad_universe_refresh_skipped"] is True
    assert scanner.last_scan_diagnostics["scan_key"]["symbols"] == "BTC"


def test_high_upside_live_readiness_and_risk_gate_require_auto_live_enabled(app) -> None:
    app.config["HIGH_UPSIDE_PROFILE_ENABLED"] = True
    app.config["HIGH_UPSIDE_LIVE_ELIGIBLE"] = True
    app.config["HIGH_UPSIDE_AUTO_LIVE_ENABLED"] = False
    app.config["HIGH_UPSIDE_REQUIRE_PROMOTED_ML"] = False
    app.config["HIGH_UPSIDE_MAX_DAILY_LOSS_USDC"] = 5.0
    app.config["HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION"] = {"1h": 10.0}
    app.config["HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION"] = {"1h": 1.0}
    readiness = cli_module._high_upside_live_readiness(
        SimpleNamespace(high_upside_profile=True, lock_duration_hours=1, profile="aggressive_1h"),
        {"accepted": [], "rejected": [], "rejection_breakdown": {}, "rejection_rate": 0.0},
    )
    assert "HIGH_UPSIDE_AUTO_LIVE_ENABLED=false" in readiness["blockers"]

    intent = OrderIntent(
        symbol="BTC",
        side="buy",
        quantity=0.01,
        mode="live",
        order_type="limit",
        limit_price=100.0,
        leverage=1.0,
        stop_loss=99.0,
        take_profit=103.0,
        strategy_name="scalping",
        timeframe="5m",
        metadata={"high_upside_profile": True, "duration_hours": 1, "max_drawdown": -0.01},
    )
    decision = app.extensions["services"]["risk_engine"].evaluate(intent, market_price=100.0, has_trading_access=True)
    assert decision.approved is False
    assert decision.rule_name == "high_upside_auto_live_disabled"


def test_high_upside_live_readiness_reports_continuous_cycle_limits(app) -> None:
    app.config["HIGH_UPSIDE_PROFILE_ENABLED"] = True
    app.config["HIGH_UPSIDE_LIVE_ELIGIBLE"] = True
    app.config["HIGH_UPSIDE_AUTO_LIVE_ENABLED"] = True
    app.config["HIGH_UPSIDE_REQUIRE_PROMOTED_ML"] = False
    app.config["HIGH_UPSIDE_MAX_DAILY_LOSS_USDC"] = 5.0
    app.config["HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION"] = {"1h": 10.0}
    app.config["HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION"] = {"1h": 1.0}
    app.config["HIGH_UPSIDE_MAX_LIVE_CYCLES_PER_DAY"] = 1
    app.config["HIGH_UPSIDE_MAX_ACTIVE_CYCLES"] = 1
    order = Order(
        client_order_id="high-upside-active",
        mode="live",
        symbol="BTC",
        side="buy",
        order_type="limit",
        status="pending",
        strategy_name="scalping",
        quantity=0.1,
        limit_price=100.0,
    )
    order.details = {"high_upside_profile": True, "optimizer_profile": "aggressive_1h"}
    db.session.add(order)
    db.session.commit()

    readiness = cli_module._high_upside_live_readiness(
        SimpleNamespace(high_upside_profile=True, lock_duration_hours=1, profile="aggressive_1h"),
        {"accepted": [], "rejected": [], "rejection_breakdown": {}, "rejection_rate": 0.0},
    )

    assert "high_upside_daily_live_cycle_limit_reached" in readiness["blockers"]
    assert "high_upside_active_cycle_limit_reached" in readiness["blockers"]
    assert readiness["continuous_controls"]["daily_live_order_count"] == 1
    assert readiness["continuous_controls"]["active_live_order_count"] == 1


def _add_ml_live_vault_fixture() -> tuple[User, TradingConnection, StrategyRanking]:
    user = User(username="ml-live-vault", password_hash="hash")
    db.session.add(user)
    db.session.flush()
    connection = TradingConnection(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        is_active=True,
        verification_status="verified",
        wallet_address="0x" + ("b" * 40),
    )
    connection.provider_metadata = {"can_trade": True}
    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add_all([connection, optimizer_run])
    db.session.flush()
    ranking = StrategyRanking(
        optimizer_run_id=optimizer_run.id,
        provider="hyperliquid",
        strategy_name="scalping",
        symbol="BTC",
        timeframe="5m",
        profile="aggressive_1h",
        score=25.0,
        net_return_after_costs=0.04,
        recent_1h_return=0.02,
        profit_factor=1.6,
        trade_count=16,
        max_drawdown=-0.02,
        edge_score=20.0,
        cost_drag_bps=2.0,
        window_stability=0.8,
        max_favorable_excursion=0.03,
        max_adverse_excursion=-0.005,
        leverage=1.0,
    )
    ranking.parameters = {
        "high_upside_profile": True,
        "stop_loss_pct": 0.005,
        "take_profit_pct": 0.012,
        "offline_ml_status": "promoted",
        "leverage": 1.0,
    }
    db.session.add(
        WalletBalance(
            user_id=user.id,
            asset="USDC",
            available_balance=10.0,
            locked_balance=0.0,
            estimated_usd_value=10.0,
        )
    )
    db.session.add(ranking)
    db.session.commit()
    return user, connection, ranking


def _enable_ml_live_vault_gates(app) -> None:
    app.config["APP_MODE"] = "live"
    app.config["ENABLE_LIVE_TRADING"] = True
    app.config["EXPLICIT_LIVE_CONFIRMED"] = True
    app.config["SECONDARY_CONFIRMATION"] = True
    app.config["ML_LIVE_VAULT_ONE_SHOT_ENABLED"] = True
    app.config["HIGH_UPSIDE_AUTO_LIVE_ENABLED"] = True
    app.config["HIGH_UPSIDE_PROFILE_ENABLED"] = True
    app.config["HIGH_UPSIDE_LIVE_ELIGIBLE"] = True
    app.config["HIGH_UPSIDE_MAX_DAILY_LOSS_USDC"] = 0.50
    app.config["HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION"] = {"1h": 10.0}
    app.config["HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION"] = {"1h": 1.0}
    app.config["HIGH_UPSIDE_MAX_LIVE_CYCLES_PER_DAY"] = 1
    app.config["HIGH_UPSIDE_MAX_ACTIVE_CYCLES"] = 1
    Setting.set_json("explicit_live_confirmed", True)
    Setting.set_json("secondary_confirmation", True)
    Setting.set_json("panic_lock", False)
    db.session.commit()


def test_ml_live_vault_all_cap_uses_local_and_provider_collateral(app, monkeypatch) -> None:
    user, connection, _ranking = _add_ml_live_vault_fixture()
    app.config["ML_LIVE_VAULT_MAX_CAP_USDC"] = 5.0
    service = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(
        service,
        "account_snapshot",
        lambda user_id, mode, connection_id: ClientSnapshot(
            mode,
            [{"asset": "USDC", "type": "spot", "value": 27.0, "withdrawable": 27.0}],
            [],
            [],
            [],
            [],
        ),
    )

    payload = cli_module._ml_live_vault_resolve_cap(
        "all",
        user_id=user.id,
        connection=connection,
        provider="hyperliquid",
        collateral_asset="USDC",
    )

    assert payload["requested_all"] is True
    assert payload["local_available"] == 10.0
    assert payload["provider_available"] == 27.0
    assert payload["requested_cap_usdc"] == 10.0
    assert payload["resolved_cap_usdc"] == 5.0
    assert payload["blockers"] == []


def _ml_live_vault_ready_preview(user: User, connection: TradingConnection, ranking: StrategyRanking) -> dict[str, Any]:
    return {
        "ready": True,
        "user_id": user.id,
        "provider": "hyperliquid",
        "collateral_asset": "USDC",
        "connection_id": connection.id,
        "cap_usdc": 10.0,
        "horizon": "1h",
        "duration_hours": 1,
        "duration_bucket": "1h",
        "selected_ranking": {
            "id": ranking.id,
            "provider": "hyperliquid",
            "collateral_asset": "USDC",
            "strategy_name": ranking.strategy_name,
            "symbol": ranking.symbol,
            "timeframe": ranking.timeframe,
            "profile": ranking.profile,
            "rejected": False,
            "rejection_reason": "",
        },
        "selected_leg": {
            "strategy_name": ranking.strategy_name,
            "provider": "hyperliquid",
            "collateral_asset": "USDC",
            "symbol": ranking.symbol,
            "timeframe": ranking.timeframe,
            "allocation_cap_usd": 10.0,
            "leverage": 1.0,
            "optimizer_ranking_id": ranking.id,
            "optimizer_profile": ranking.profile,
            "parameters": {
                "high_upside_profile": True,
                "stop_loss_pct": 0.005,
                "take_profit_pct": 0.012,
                "optimizer_ranking_id": ranking.id,
                "optimizer_profile": ranking.profile,
                "leverage": 1.0,
                "provider": "hyperliquid",
                "collateral_asset": "USDC",
            },
        },
        "risk_decision": {"approved": True, "rule_name": "ok"},
        "blockers": [],
    }


def test_ml_live_vault_preview_blocks_by_default_and_never_submits(app, monkeypatch) -> None:
    user, _, _ = _add_ml_live_vault_fixture()
    called = {"place_order": 0}

    def fail_place_order(intent):
        called["place_order"] += 1
        raise AssertionError("preview must not submit orders")

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=["ml-live-vault-preview", "--user-id", str(user.id), "--provider", "active", "--cap-usdc", "10", "--horizon", "1h"]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is False
    assert payload["submitted"] is False
    assert payload["live_orders_created"] is False
    assert called["place_order"] == 0
    assert "ml_suite_not_ready" in payload["blockers"] or "promoted_ml_signal_missing" in payload["blockers"]


def test_ml_live_vault_preview_extreme_objective_reports_clipped_dynamic_caps(app, monkeypatch) -> None:
    user, _, _ = _add_ml_live_vault_fixture()
    app.config["ML_DYNAMIC_CAPS_ENABLED"] = True
    app.config["ML_EXTREME_UPSIDE_MODEL_ENABLED"] = True
    app.config["ML_LIVE_VAULT_MAX_CAP_USDC"] = 10.0
    app.config["HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION"] = {"1h": 10.0}
    app.config["HIGH_UPSIDE_MAX_DAILY_LOSS_USDC"] = 0.50
    seen_contexts: list[dict[str, Any]] = []

    def fake_decision(family, context, horizon="1h", candles=None):
        seen_contexts.append(dict(context))
        return {
            "ready": True,
            "family": family,
            "action": "pursue" if family == "pytorch_extreme_upside" else "suggest",
            "confidence": 0.9,
            "blockers": [],
            "raw": {
                "suggested_notional_usdc": 100.0,
                "suggested_leverage": 5.0,
                "extreme_upside_probability": 0.95,
                "distance_to_target_pct": 50.0,
                "suggested_stop_loss_pct": 0.005,
                "suggested_take_profit_pct": 0.02,
            },
        }

    monkeypatch.setattr(
        app.extensions["services"]["ml_decision_engine"],
        "decision",
        fake_decision,
    )
    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", lambda intent: (_ for _ in ()).throw(AssertionError("no orders")))

    result = app.test_cli_runner().invoke(
        args=[
            "ml-live-vault-preview",
            "--user-id",
            str(user.id),
            "--provider",
            "active",
            "--cap-usdc",
            "10",
            "--horizon",
            "1h",
            "--objective",
            "extreme_upside",
            "--target-roi-pct",
            "1000",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["submitted"] is False
    assert payload["live_orders_created"] is False
    assert payload["objective"] == "extreme_upside"
    assert payload["dynamic_cap_suggestion"]["suggested_notional_usdc"] <= 10.0
    assert payload["clipped_leverage"] <= 1.0
    assert payload["target_roi_policy"]["guarantees_profit"] is False
    assert seen_contexts
    assert {row.get("provider") for row in seen_contexts} == {"hyperliquid"}
    assert {row.get("collateral_asset") for row in seen_contexts} == {"USDC"}


def test_ml_live_vault_model_ids_prefer_promoted_nested_models() -> None:
    payload = cli_module._ml_live_vault_model_ids(
        {"promoted_model": {"model_id": 27}, "latest_model_id": 28},
        {"promoted_model": {"id": 29}, "latest_model_id": 30},
        {
            "families": {
                "pytorch_extreme_upside": {"promoted_model": {"id": 31}},
                "pytorch_allocator": {"promoted_model_id": 41},
                "pytorch_candidate_only": {"latest_model_id": 99},
            }
        },
    )

    assert payload["offline_model_id"] == 27
    assert payload["signal_model_id"] == 29
    assert payload["suite_family_model_ids"]["pytorch_extreme_upside"] == 31
    assert payload["suite_family_model_ids"]["pytorch_allocator"] == 41
    assert payload["suite_family_model_ids"]["pytorch_candidate_only"] == 99


def test_torch_runtime_readiness_reports_missing_without_blocking_disabled_ml(app, monkeypatch) -> None:
    original_find_spec = cli_module.importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "torch":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(cli_module.importlib.util, "find_spec", fake_find_spec)

    payload = cli_module._torch_runtime_readiness()

    assert payload["status"] == "torch_missing"
    assert payload["blockers"] == ["torch_missing"]


def test_ml_live_vault_one_shot_rejects_wrong_confirmation(app) -> None:
    user, connection, ranking = _add_ml_live_vault_fixture()
    _enable_ml_live_vault_gates(app)
    audit = AuditLog(
        category="ml_live_vault",
        action="ml_live_vault_preview",
        message="ready preview",
        user_id=user.id,
        trading_connection_id=connection.id,
    )
    audit.details = {"payload": _ml_live_vault_ready_preview(user, connection, ranking)}
    db.session.add(audit)
    db.session.commit()

    result = app.test_cli_runner().invoke(
        args=["ml-live-vault-one-shot", "--preview-audit-id", str(audit.id), "--user-id", str(user.id), "--confirm", "WRONG"]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["cycle_started"] is False
    assert "exact_confirmation_missing" in payload["blockers"]
    assert VaultCycle.query.count() == 0


def test_ml_auto_vault_cycle_blocks_by_default_and_never_submits(app, monkeypatch) -> None:
    user, _, _ = _add_ml_live_vault_fixture()
    called = {"place_order": 0}

    def fail_place_order(intent):
        called["place_order"] += 1
        raise AssertionError("auto vault must block before orders by default")

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "ml-auto-vault-cycle",
            "--user-id",
            str(user.id),
            "--provider",
            "active",
            "--cap-usdc",
            "10",
            "--horizon",
            "1h",
            "--objective",
            "extreme_upside",
            "--confirm",
            "ML-AUTO-VAULT-LIVE",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["cycle_started"] is False
    assert payload["live_orders_created"] is False
    assert "ML_AUTO_VAULT_LIVE_ENABLED=false" in payload["blockers"]
    assert called["place_order"] == 0


def test_ml_vault_tick_blocks_by_default_and_never_submits(app, monkeypatch) -> None:
    user, _, _ = _add_ml_live_vault_fixture()
    called = {"place_order": 0}

    def fail_place_order(intent):
        called["place_order"] += 1
        raise AssertionError("tick must block before orders by default")

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "ml-vault-tick",
            "--user-id",
            str(user.id),
            "--provider-scope",
            "all",
            "--cap-usdc",
            "10",
            "--objective",
            "extreme_upside",
            "--confirm",
            "ML-VAULT-TICK",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["cycle_started"] is False
    assert payload["live_orders_created"] is False
    assert "ML_CONTINUOUS_VAULT_ENABLED=false" in payload["blockers"]
    assert "ML_VAULT_TICK_ENABLED=false" in payload["blockers"]
    assert called["place_order"] == 0
    assert Setting.get_json("ml_vault_last_decision", {})["ml_vault_tick"]["one_tick"] is True


def test_ml_vault_tick_blocks_without_exchange_leverage_gate(app, monkeypatch) -> None:
    user, connection, ranking = _add_ml_live_vault_fixture()
    _enable_ml_live_vault_gates(app)
    app.config["ML_CONTINUOUS_VAULT_ENABLED"] = True
    app.config["ML_VAULT_TICK_ENABLED"] = True
    app.config["ML_AUTO_VAULT_LIVE_ENABLED"] = True
    preview = _ml_live_vault_ready_preview(user, connection, ranking)
    monkeypatch.setattr(cli_module, "_ml_live_vault_preview_payload", lambda **kwargs: preview)
    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", lambda intent: (_ for _ in ()).throw(AssertionError("no orders")))

    result = app.test_cli_runner().invoke(
        args=[
            "ml-vault-tick",
            "--user-id",
            str(user.id),
            "--provider-scope",
            "all",
            "--cap-usdc",
            "10",
            "--objective",
            "extreme_upside",
            "--confirm",
            "ML-VAULT-TICK",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["cycle_started"] is False
    assert "exchange_max_leverage_unavailable" in payload["blockers"]


def test_ml_live_vault_one_shot_blocks_stale_preview(app) -> None:
    user, connection, ranking = _add_ml_live_vault_fixture()
    _enable_ml_live_vault_gates(app)
    app.config["ML_LIVE_VAULT_PREVIEW_MAX_AGE_SECONDS"] = 60
    audit = AuditLog(
        category="ml_live_vault",
        action="ml_live_vault_preview",
        message="old ready preview",
        user_id=user.id,
        trading_connection_id=connection.id,
        created_at=datetime.utcnow() - timedelta(minutes=5),
    )
    audit.details = {"payload": _ml_live_vault_ready_preview(user, connection, ranking)}
    db.session.add(audit)
    db.session.commit()

    result = app.test_cli_runner().invoke(
        args=[
            "ml-live-vault-one-shot",
            "--preview-audit-id",
            str(audit.id),
            "--user-id",
            str(user.id),
            "--confirm",
            "ML-LIVE-VAULT-10USDC",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["cycle_started"] is False
    assert "preview_stale" in payload["blockers"]
    assert VaultCycle.query.count() == 0


def test_ml_live_vault_one_shot_starts_cycle_through_strategy_manager(app, monkeypatch) -> None:
    user, connection, ranking = _add_ml_live_vault_fixture()
    _enable_ml_live_vault_gates(app)
    preview = _ml_live_vault_ready_preview(user, connection, ranking)
    audit = AuditLog(
        category="ml_live_vault",
        action="ml_live_vault_preview",
        message="ready preview",
        user_id=user.id,
        trading_connection_id=connection.id,
    )
    audit.details = {"payload": preview}
    db.session.add(audit)
    db.session.commit()
    monkeypatch.setattr(cli_module, "_ml_live_vault_preview_payload", lambda **kwargs: preview)
    inside_strategy_manager = {"active": False}
    calls = {"start": 0, "place_order": 0}

    def fake_place_order(intent: OrderIntent):
        assert inside_strategy_manager["active"] is True
        calls["place_order"] += 1
        order = Order(
            client_order_id=f"ml-live-vault-{calls['place_order']}",
            mode="live",
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            status="submitted",
            strategy_name=intent.strategy_name,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
        )
        order.details = {"ml_live_vault_one_shot": True}
        db.session.add(order)
        db.session.commit()
        return order

    def fake_start(run_id: int) -> None:
        calls["start"] += 1
        run = db.session.get(StrategyRun, run_id)
        assert run is not None
        inside_strategy_manager["active"] = True
        try:
            app.extensions["services"]["order_manager"].place_order(
                OrderIntent(
                    symbol=run.symbol,
                    side="buy",
                    quantity=0.01,
                    mode="live",
                    order_type="limit",
                    limit_price=100.0,
                    leverage=1.0,
                    stop_loss=99.0,
                    take_profit=102.0,
                    strategy_name=run.strategy_name,
                    timeframe=run.timeframe,
                    user_id=run.user_id,
                    trading_connection_id=run.trading_connection_id,
                    metadata={"high_upside_profile": True},
                )
            )
        finally:
            inside_strategy_manager["active"] = False

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fake_place_order)
    monkeypatch.setattr(app.extensions["services"]["strategy_manager"], "start", fake_start)

    result = app.test_cli_runner().invoke(
        args=[
            "ml-live-vault-one-shot",
            "--preview-audit-id",
            str(audit.id),
            "--user-id",
            str(user.id),
            "--confirm",
            "ML-LIVE-VAULT-10USDC",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["cycle_started"] is True
    assert calls == {"start": 1, "place_order": 1}
    assert VaultCycle.query.count() == 1
    assert StrategyRun.query.count() == 1
    assert AuditLog.query.filter_by(action="ml_live_vault_one_shot_started").count() == 1


def test_ml_auto_vault_cycle_happy_path_uses_strategy_manager(app, monkeypatch) -> None:
    user, connection, ranking = _add_ml_live_vault_fixture()
    _enable_ml_live_vault_gates(app)
    app.config["ML_AUTO_VAULT_LIVE_ENABLED"] = True
    app.config["ML_DYNAMIC_CAPS_ENABLED"] = True
    preview = _ml_live_vault_ready_preview(user, connection, ranking)
    preview.update(
        {
            "objective": "extreme_upside",
            "target_roi_pct": 1000.0,
            "dynamic_cap_suggestion": {"clipped_cap_usdc": 10.0, "clipped_leverage": 1.0, "blockers": []},
        }
    )
    monkeypatch.setattr(cli_module, "_ml_live_vault_preview_payload", lambda **kwargs: preview)
    inside_strategy_manager = {"active": False}
    calls = {"start": 0, "place_order": 0}

    def fake_place_order(intent: OrderIntent):
        assert inside_strategy_manager["active"] is True
        calls["place_order"] += 1
        order = Order(
            client_order_id=f"ml-auto-vault-{calls['place_order']}",
            mode="live",
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            status="submitted",
            strategy_name=intent.strategy_name,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
        )
        order.details = {"ml_live_vault_one_shot": True, "objective": "extreme_upside"}
        db.session.add(order)
        db.session.commit()
        return order

    def fake_start(run_id: int) -> None:
        calls["start"] += 1
        run = db.session.get(StrategyRun, run_id)
        assert run is not None
        inside_strategy_manager["active"] = True
        try:
            app.extensions["services"]["order_manager"].place_order(
                OrderIntent(
                    symbol=run.symbol,
                    side="buy",
                    quantity=0.01,
                    mode="live",
                    order_type="limit",
                    limit_price=100.0,
                    leverage=1.0,
                    stop_loss=99.0,
                    take_profit=102.0,
                    strategy_name=run.strategy_name,
                    timeframe=run.timeframe,
                    user_id=run.user_id,
                    trading_connection_id=run.trading_connection_id,
                    metadata={"high_upside_profile": True, "objective": "extreme_upside"},
                )
            )
        finally:
            inside_strategy_manager["active"] = False

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fake_place_order)
    monkeypatch.setattr(app.extensions["services"]["strategy_manager"], "start", fake_start)

    result = app.test_cli_runner().invoke(
        args=[
            "ml-auto-vault-cycle",
            "--user-id",
            str(user.id),
            "--provider",
            "active",
            "--cap-usdc",
            "10",
            "--horizon",
            "1h",
            "--objective",
            "extreme_upside",
            "--confirm",
            "ML-AUTO-VAULT-LIVE",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["cycle_started"] is True
    assert payload["objective"] == "extreme_upside"
    assert calls == {"start": 1, "place_order": 1}
    assert VaultCycle.query.count() == 1


def test_ml_vault_tick_happy_path_uses_strategy_manager(app, monkeypatch) -> None:
    user, connection, ranking = _add_ml_live_vault_fixture()
    _enable_ml_live_vault_gates(app)
    app.config["ML_CONTINUOUS_VAULT_ENABLED"] = True
    app.config["ML_VAULT_TICK_ENABLED"] = True
    app.config["ML_AUTO_VAULT_LIVE_ENABLED"] = True
    app.config["ML_VAULT_MAX_CAP_USDC"] = 10.0
    connection.provider_metadata = {"can_trade": True, "exchange_max_leverage": 20.0}
    db.session.commit()
    preview = _ml_live_vault_ready_preview(user, connection, ranking)
    monkeypatch.setattr(cli_module, "_ml_live_vault_preview_payload", lambda **kwargs: preview)
    inside_strategy_manager = {"active": False}
    calls = {"start": 0, "place_order": 0}

    def fake_place_order(intent: OrderIntent):
        assert inside_strategy_manager["active"] is True
        calls["place_order"] += 1
        order = Order(
            client_order_id=f"ml-vault-tick-{calls['place_order']}",
            mode="live",
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            status="submitted",
            strategy_name=intent.strategy_name,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
        )
        order.details = {"ml_vault_tick": True}
        db.session.add(order)
        db.session.commit()
        return order

    def fake_start(run_id: int) -> None:
        calls["start"] += 1
        run = db.session.get(StrategyRun, run_id)
        assert run is not None
        inside_strategy_manager["active"] = True
        try:
            app.extensions["services"]["order_manager"].place_order(
                OrderIntent(
                    symbol=run.symbol,
                    side="buy",
                    quantity=0.01,
                    mode="live",
                    order_type="limit",
                    limit_price=100.0,
                    leverage=1.0,
                    stop_loss=99.0,
                    take_profit=102.0,
                    strategy_name=run.strategy_name,
                    timeframe=run.timeframe,
                    user_id=run.user_id,
                    trading_connection_id=run.trading_connection_id,
                    metadata={"high_upside_profile": True, "ml_vault_tick": True},
                )
            )
        finally:
            inside_strategy_manager["active"] = False

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fake_place_order)
    monkeypatch.setattr(app.extensions["services"]["strategy_manager"], "start", fake_start)

    result = app.test_cli_runner().invoke(
        args=[
            "ml-vault-tick",
            "--user-id",
            str(user.id),
            "--provider-scope",
            "all",
            "--cap-usdc",
            "10",
            "--objective",
            "extreme_upside",
            "--confirm",
            "ML-VAULT-TICK",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["cycle_started"] is True
    assert payload["exchange_leverage_gate"]["ready"] is True
    assert calls == {"start": 1, "place_order": 1}


def test_vault_selector_requires_high_upside_exits_and_promoted_ml(app) -> None:
    selector = app.extensions["services"]["vault_strategy_selector"]
    app.config["HIGH_UPSIDE_REQUIRE_PROMOTED_ML"] = True
    ranking = StrategyRanking(
        strategy_name="scalping",
        symbol="BTC",
        timeframe="5m",
        profile="aggressive_1h",
        score=10.0,
        net_return_after_costs=0.03,
        recent_1h_return=0.02,
        profit_factor=1.5,
        trade_count=12,
        max_drawdown=-0.02,
        edge_score=12.0,
        cost_drag_bps=2.0,
        window_stability=0.8,
        max_favorable_excursion=0.02,
        max_adverse_excursion=-0.005,
        mfe_mae_ratio=4.0,
    )
    ranking.ml_explanation = {
        "net_roi": {
            "net_roi_score": 20.0,
            "expected_fill_quality": 0.9,
            "churn_penalty": 0.0,
            "edge_after_cost_bps": 12.0,
        },
        "net_roi_v2": {
            "net_roi_v2_score": 20.0,
            "roi_quality_grade": "A",
            "roi_rejection_risk": "low",
            "regime_support": "regime-supported",
        },
        "one_hour_edge_v2": {
            "one_hour_edge_v2": 20.0,
            "one_hour_edge_grade": "A",
            "expected_execution_quality": 0.9,
            "profitability_blockers": [],
        },
    }
    ranking.parameters = {"high_upside_profile": True, "stop_loss_pct": 0.01}

    assert selector._ranking_acceptable(ranking, "aggressive_1h", "Balanced") is False

    ranking.parameters = {
        "high_upside_profile": True,
        "stop_loss_pct": 0.01,
        "take_profit_pct": 0.02,
        "offline_ml_status": "promoted",
    }
    assert selector._ranking_acceptable(ranking, "aggressive_1h", "Balanced") is True


def test_run_optimization_outputs_raw_upside_research_report(app, monkeypatch) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    captured = {}

    def fake_run(config):
        captured["profile"] = config.profile
        return {
            "ok": True,
            "profile": config.profile,
            "top": [],
            "raw_upside_report": {
                "enabled": True,
                "research_only": True,
                "target_roi_pct": 1000.0,
                "target_roi_hit": True,
                "live_orders_created": False,
                "top_by_raw_upside": [
                    {
                        "symbol": "BTC",
                        "strategy_name": "scalping",
                        "raw_total_return_pct": 1200.0,
                        "target_roi_hit": True,
                        "rejected": True,
                        "rejection_reason": "high_drawdown",
                        "live_blockers": ["candidate_rejected:high_drawdown"],
                        "one_hour_edge_v2": 12.0,
                        "one_hour_edge_grade": "D",
                        "expected_execution_quality": 0.25,
                        "profitability_blockers": ["high_drawdown"],
                        "raw_vs_net_roi_gap": 950.0,
                    }
                ],
                "top_by_net_roi_v2": [],
                "rejected_high_raw_upside": [],
                "best_preview_candidate": {},
            },
            "one_hour_diagnostics": {
                "enabled": True,
                "primary_sort": "one_hour_edge_v2",
                "top_accepted": [],
                "top_rejected": [{"symbol": "BTC", "one_hour_edge_v2": 12.0, "profitability_blockers": ["high_drawdown"]}],
            },
        }

    monkeypatch.setattr(optimizer, "run", fake_run)

    for profile in ("extreme_roi_experimental", "aggressive_1h"):
        result = app.test_cli_runner().invoke(
            args=[
                "run-optimization",
                "--profile",
                profile,
                "--auto-deploy-top-n",
                "0",
            ]
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert captured["profile"] == profile
        assert payload["raw_upside_report"]["research_only"] is True
        assert payload["raw_upside_report"]["live_orders_created"] is False
        assert payload["raw_upside_report"]["top_by_raw_upside"][0]["raw_total_return_pct"] == 1200.0
        assert payload["raw_upside_report"]["top_by_raw_upside"][0]["one_hour_edge_grade"] == "D"
        assert payload["one_hour_diagnostics"]["primary_sort"] == "one_hour_edge_v2"
        assert payload["one_hour_diagnostics"]["top_rejected"][0]["profitability_blockers"] == ["high_drawdown"]
        assert payload["live_canary_readiness"]["canary_preview_only"] is True


def test_run_optimization_timeout_returns_actionable_diagnostics(app, monkeypatch) -> None:
    app.config["OPTIMIZER_MARKET_DATA_TIMEOUT_SECONDS"] = 0.01
    optimizer = app.extensions["services"]["strategy_optimizer"]

    def slow_run(config):
        time.sleep(1.0)
        return {"ok": True, "profile": config.profile}

    monkeypatch.setattr(optimizer, "run", slow_run)

    result = app.test_cli_runner().invoke(
        args=[
            "run-optimization",
            "--profile",
            "aggressive_1h",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["timed_out"] is True
    assert payload["optimizer_runtime"]["timed_out"] is True
    assert payload["optimizer_runtime"]["timeout_seconds"] == 0.01
    assert payload["optimizer_runtime"]["source"] == "OPTIMIZER_MARKET_DATA_TIMEOUT_SECONDS"
    assert "No live orders are created by run-optimization." in payload["warnings"]
    assert payload["offline_ml_readiness"]["horizon"] == "1h"


def test_run_optimization_preserves_cooperative_partial_runtime(app, monkeypatch) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]

    def partial_run(config):
        return {
            "profile": config.profile,
            "ranking_count": 1,
            "accepted_count": 1,
            "top": [{"symbol": "BTC", "strategy_name": "scalping"}],
            "partial_result": True,
            "timed_out": True,
            "partial_reason": "optimizer_deadline_reached",
            "warnings": ["No live orders are created by run-optimization."],
            "optimizer_runtime": {
                "evaluated_candidates": 1,
                "skipped_candidates": 0,
                "backtest_runs": 2,
                "window_count": 2,
                "phase_seconds": {"backtest": 0.1},
                "slowest_market": {"symbol": "BTC", "timeframe": "1m", "seconds": 0.2},
                "partial_result": True,
                "partial_reason": "optimizer_deadline_reached",
            },
        }

    monkeypatch.setattr(optimizer, "run", partial_run)

    result = app.test_cli_runner().invoke(
        args=[
            "run-optimization",
            "--profile",
            "aggressive_1h",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["partial_result"] is True
    assert payload["timed_out"] is True
    assert payload["optimizer_runtime"]["timed_out"] is True
    assert payload["optimizer_runtime"]["evaluated_candidates"] == 1
    assert payload["optimizer_runtime"]["backtest_runs"] == 2
    assert payload["optimizer_runtime"]["partial_reason"] == "optimizer_deadline_reached"
    assert payload["live_canary_readiness"]["optimizer_research_only"] is True
    assert payload["live_canary_readiness"]["canary_preview_only"] is True


def test_hourly_experimental_fetch_uses_bounded_buffer(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    captured: dict[str, int] = {}
    optimizer.market_data.get_candles = lambda symbol, timeframe, mode, limit: captured.setdefault("limit", limit) or _minute_candles()
    config = optimizer.default_config(symbols=["BTC"], timeframes=["1m"], strategy_names=["scalping"], profile="extreme_roi_experimental")
    config.lock_duration_hours = 1

    optimizer._fetch_candles("BTC", "1m", config)

    assert captured["limit"] <= 6_000


def test_optimizer_passes_configured_signal_history_limit(app, monkeypatch) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    optimizer.market_data.get_candles = lambda symbol, timeframe, mode, limit: _optimizer_candles()
    app.config["OPTIMIZER_SIGNAL_HISTORY_LIMIT"] = 64
    captured_limits: list[int] = []
    original_run = optimizer.runner.run

    def capture_run(backtest_config, candles):
        captured_limits.append(backtest_config.signal_history_limit)
        return original_run(backtest_config, candles)

    monkeypatch.setattr(optimizer.runner, "run", capture_run)
    config = optimizer.default_config(symbols=["BTC"], timeframes=["15m"], strategy_names=["scalping"])
    config.training_window_days = 1
    config.testing_window_days = 1
    config.step_days = 1
    config.max_parameter_sets = 1
    config.auto_deploy_top_n = 0

    result = optimizer.run(config)

    assert captured_limits
    assert set(captured_limits) == {64}
    assert result["optimizer_runtime"]["signal_history_limit"] == 64


def test_promote_live_ranker_cli_guardrail_failure_auto_disables_high_upside(app) -> None:
    result = app.test_cli_runner().invoke(
        args=["promote-live-ranker", "--horizon", "1h", "--confirm", "PROMOTE-LIVE-RANKER"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["promoted"] is False
    assert payload["blockers"] == ["insufficient_quarantined_events"]
    assert Setting.get_json("high_upside_live_disabled") is True
    assert Setting.get_json("high_upside_live_disabled_reason")["reason"] == "model_promotion_diagnostic_failure"


def test_promote_live_ranker_cli_promotes_quarantined_events(app) -> None:
    app.config["ML_LIVE_PROMOTION_MIN_EVENTS"] = 2
    app.config["ML_LIVE_PROMOTION_MAX_MEAN_LOSS"] = 1.0
    app.config["ML_LIVE_PROMOTION_MAX_NEGATIVE_ERROR_RATE"] = 1.0
    ranker = app.extensions["services"]["online_ranker"]
    features = extract_features({"strategy_name": "scalping", "net_return_after_costs": 0.04, "lock_duration_hours": 1})
    ranker.update(features, 0.2, horizon="1h", source="test", source_id="1", mode="live")
    ranker.update(features, 0.2, horizon="1h", source="test", source_id="2", mode="live")
    db.session.commit()

    result = app.test_cli_runner().invoke(
        args=["promote-live-ranker", "--horizon", "1h", "--confirm", "PROMOTE-LIVE-RANKER"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["promoted"] is True
    assert payload["promoted_count"] == 2
    assert MLModelState.query.filter_by(horizon="1h").one().update_count == 2


def test_offline_ranker_cli_training_and_promotion_commands_report_diagnostics(app, monkeypatch) -> None:
    offline_ranker = app.extensions["services"]["offline_ranker"]
    captured = {}

    def fake_train(horizon, *, model_types, provider):
        captured["train"] = {"horizon": horizon, "model_types": model_types, "provider": provider}
        return {"trained": True, "horizon": horizon, "provider": provider, "trained_models": [{"model_id": 7}], "blockers": []}

    def fake_promote(horizon, *, model_id, provider):
        captured["promote"] = {"horizon": horizon, "model_id": model_id, "provider": provider}
        return {"promoted": True, "horizon": horizon, "provider": provider, "model_id": model_id, "blockers": []}

    monkeypatch.setattr(offline_ranker, "train", fake_train)
    monkeypatch.setattr(offline_ranker, "promote", fake_promote)

    train_result = app.test_cli_runner().invoke(
        args=[
            "train-offline-ranker",
            "--horizon",
            "1h",
            "--model",
            "both",
            "--confirm",
            "TRAIN-OFFLINE-RANKER",
        ]
    )
    promote_result = app.test_cli_runner().invoke(
        args=[
            "promote-offline-ranker",
            "--horizon",
            "1h",
            "--model-id",
            "7",
            "--confirm",
            "PROMOTE-OFFLINE-RANKER",
        ]
    )

    assert train_result.exit_code == 0
    assert promote_result.exit_code == 0
    assert captured["train"] == {"horizon": "1h", "model_types": "both", "provider": "global"}
    assert captured["promote"] == {"horizon": "1h", "model_id": 7, "provider": "global"}
    assert json.loads(train_result.output)["trained"] is True
    assert json.loads(promote_result.output)["promoted"] is True


def test_activate_trading_connection_requires_exact_confirmation(app) -> None:
    result = app.test_cli_runner().invoke(
        args=["activate-trading-connection", "--user-id", "1", "--connection-id", "1", "--confirm", "WRONG"]
    )

    assert result.exit_code != 0
    assert "ACTIVATE-LIVE-CONNECTION" in result.output


def test_activate_trading_connection_enables_verified_provider_and_audits(app) -> None:
    user, hyperliquid, _ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    kucoin = service.create_or_update(
        user_id=user.id,
        provider="kucoin",
        connection_type="cex_api_key",
        api_key="key",
        api_secret="secret",
        passphrase="passphrase",
        is_active=False,
    )
    kucoin.verification_status = "verified"
    kucoin.is_active = False
    hyperliquid.is_active = True
    db.session.commit()

    result = app.test_cli_runner().invoke(
        args=[
            "activate-trading-connection",
            "--user-id",
            str(user.id),
            "--connection-id",
            str(kucoin.id),
            "--confirm",
            "ACTIVATE-LIVE-CONNECTION",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["activated"] is True
    assert payload["enabled"] is True
    assert payload["active_connection_id"] == kucoin.id
    assert payload["provider"] == "kucoin"
    assert set(payload["enabled_connection_ids"]) == {hyperliquid.id, kucoin.id}
    assert db.session.get(TradingConnection, kucoin.id).is_active is True
    assert db.session.get(TradingConnection, hyperliquid.id).is_active is True
    audit = AuditLog.query.filter_by(action="trading_connection_activated", trading_connection_id=kucoin.id).one()
    assert set(audit.details["enabled_connection_ids"]) == {hyperliquid.id, kucoin.id}


def test_provider_balance_readiness_uses_exchange_collateral_asset(app) -> None:
    kucoin = cli_module._provider_balance_readiness(
        "kucoin",
        [{"asset": "USDT", "type": "futures", "value": "10.0", "withdrawable": "0"}],
    )
    assert kucoin["ready"] is True
    assert kucoin["collateral_asset"] == "USDT"
    assert kucoin["margin_usdt"] == 10.0
    assert kucoin["funding_source"] == "kucoin_margin_usdt"

    missing = cli_module._provider_balance_readiness(
        "kucoin",
        [{"asset": "USDC", "type": "margin", "value": "25.0", "withdrawable": "25.0"}],
    )
    assert missing["ready"] is False
    assert missing["collateral_asset"] == "USDT"
    assert missing["blockers"] == ["kucoin_usdt_missing"]

    hyperliquid = cli_module._provider_balance_readiness(
        "hyperliquid",
        [{"asset": "USDC", "type": "margin", "value": "7.5", "withdrawable": "7.5"}],
    )
    assert hyperliquid["ready"] is True
    assert hyperliquid["collateral_asset"] == "USDC"
    assert hyperliquid["funding_source"] == "hyperliquid_margin_usdc"


def test_explicit_verified_connection_does_not_require_active_flag(app) -> None:
    user = User(username="explicit-inactive-provider", password_hash="hash")
    db.session.add(user)
    db.session.flush()
    active_kucoin = TradingConnection(
        user_id=user.id,
        provider="kucoin",
        connection_type="cex_api_key",
        is_active=True,
        verification_status="verified",
    )
    inactive_hyperliquid = TradingConnection(
        user_id=user.id,
        provider="hyperliquid",
        connection_type="cex_api_key",
        is_active=False,
        verification_status="verified",
        wallet_address="0x" + ("c" * 40),
    )
    db.session.add_all([active_kucoin, inactive_hyperliquid])
    db.session.commit()

    funds_connections = cli_module._live_funds_connections("hyperliquid", user.id, inactive_hyperliquid.id)
    vault_connection = cli_module._ml_live_vault_connection(user.id, "hyperliquid", inactive_hyperliquid.id)

    assert funds_connections == [inactive_hyperliquid]
    assert vault_connection == inactive_hyperliquid
    assert cli_module._ml_live_vault_connection(user.id, "hyperliquid", None) is None


def test_live_canary_preview_never_submits_order(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    order_manager = app.extensions["services"]["order_manager"]

    def fail_place_order(intent):
        raise AssertionError("preview mode must not submit an order")

    monkeypatch.setattr(order_manager, "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert payload["real_order_submitted"] is False
    assert payload["preview_only"] is True
    assert payload["ready"] is True
    assert payload["selected_symbol"] == "BTC"
    assert payload["selected_strategy"] == "scalping"
    assert payload["side"] == "buy"
    assert payload["size"] > 0
    assert payload["confidence"] == 0.5
    assert payload["projected_order"]["symbol"] == "BTC"
    assert payload["projected_order"]["notional"] <= 1.0
    assert payload["sizing"]["effective_allocation_budget_usd"] == 1.0
    assert payload["signal_quality"]["net_roi_v2_score"] > 0
    audit = AuditLog.query.filter_by(action="live_canary_preview").one()
    assert audit.details["preview_only"] is True
    assert audit.details["real_order_submitted"] is False
    assert audit.details["projected_order"]["symbol"] == "BTC"


def test_ml_risk_preview_never_submits_order(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()

    def fail_place_order(intent):
        raise AssertionError("ml-risk-preview must not submit an order")

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "ml-risk-preview",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
            "--horizon",
            "1h",
        ]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["research_only"] is True
    assert payload["submitted"] is False
    assert payload["live_orders_created"] is False
    assert payload["ranking_id"] == ranking.id
    assert "pytorch_risk_policy" in payload["ml_policy_decisions"]


def test_live_canary_can_use_two_usdt_min_size_fallback(app, monkeypatch) -> None:
    app.config["FIRST_CANARY_USE_MIN_SIZE_FALLBACK"] = True
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    db.session.commit()
    _patch_canary_market(app, monkeypatch)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is True
    assert payload["sizing"]["effective_allocation_budget_usd"] == 2.0
    assert payload["projected_order"]["notional"] <= 2.0


def test_live_micro_canary_default_mode_normalization_is_fail_closed() -> None:
    assert _normalize_mode(None, enable_live=False) == "paper"
    assert _normalize_mode("paper", enable_live=True) == "paper"
    assert _normalize_mode("live", enable_live=False) == "paper"
    assert _normalize_mode("invalid", enable_live=True) == "paper"
    assert _normalize_mode("live", enable_live=True) == "live"


def test_live_micro_canary_preview_caps_ten_usdt_account(app, monkeypatch) -> None:
    _enable_micro_canary(app)
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.allocation_amount_usd = 50.0
    ranking.leverage = 5.0
    ranking.parameters = {"risk_fraction": 0.03, "take_profit_pct": 0.04, "stop_loss_pct": 0.01, "allocation_cap_usd": 50.0}
    db.session.commit()
    _patch_canary_market(app, monkeypatch)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is True
    assert payload["live_micro_canary"]["enabled"] is True
    assert payload["live_micro_canary"]["account_usd"] == 10.0
    assert payload["live_micro_canary"]["max_allocation_usd"] == 1.0
    assert payload["sizing"]["effective_allocation_budget_usd"] == 1.0
    assert payload["sizing"]["risk_fraction"] <= 0.01
    assert payload["sizing"]["risk_budget"] <= 0.10
    assert payload["projected_order"]["notional"] <= 1.0
    assert payload["projected_order"]["leverage"] == 1.0


def test_live_micro_canary_two_usdt_cap_is_upper_bound(app, monkeypatch) -> None:
    _enable_micro_canary(app, max_allocation=2.0)
    app.config["FIRST_CANARY_ALLOCATION_BUDGET_USDT"] = 5.0
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.allocation_amount_usd = 50.0
    ranking.parameters = {"risk_fraction": 0.03, "take_profit_pct": 0.04, "stop_loss_pct": 0.01, "allocation_cap_usd": 50.0}
    db.session.commit()
    _patch_canary_market(app, monkeypatch)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is True
    assert payload["live_micro_canary"]["max_allocation_usd"] == 2.0
    assert payload["sizing"]["effective_allocation_budget_usd"] == 2.0
    assert payload["projected_order"]["notional"] <= 2.0


def test_live_micro_canary_requires_stop_loss(app, monkeypatch) -> None:
    _enable_micro_canary(app)
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(app.extensions["services"]["strategy_registry"], "build", lambda name, parameters=None: _NoStopCanaryStrategy())

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is False
    assert "stop_loss_required" in payload["blockers"]
    assert "stop_loss_required" in payload["live_micro_canary"]["active_blockers"]


def test_live_micro_canary_requires_take_profit(app, monkeypatch) -> None:
    _enable_micro_canary(app)
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(app.extensions["services"]["strategy_registry"], "build", lambda name, parameters=None: _NoTakeProfitCanaryStrategy())

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is False
    assert "take_profit_required" in payload["blockers"]
    assert "take_profit_required" in payload["live_micro_canary"]["active_blockers"]


def test_live_micro_canary_preview_only_blocks_real_submit(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    _enable_micro_canary(app, preview_only=True)
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)

    def fail_place_order(intent):
        raise AssertionError("micro-canary preview-only mode must not submit an order")

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
            "--submit",
            "--confirm",
            "LIVE-CANARY-TRADE",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert payload["real_order_submitted"] is False
    assert payload["submit_blocked"] is True
    assert payload["submit_block_reason"] == "LIVE_MICRO_CANARY_PREVIEW_ONLY is true; live micro-canary submit is disabled."
    assert "live_micro_canary_preview_only_enabled" in payload["blockers"]


def test_live_canary_submit_requires_live_mode(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    app.config["APP_MODE"] = "paper"
    app.config["ENABLE_LIVE_TRADING"] = False
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)

    def fail_place_order(intent):
        raise AssertionError("paper mode must block live canary submission")

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
            "--submit",
            "--confirm",
            "LIVE-CANARY-TRADE",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert payload["real_order_submitted"] is False
    assert "live_mode_required" in payload["blockers"]


def test_live_micro_canary_blocks_first_canary_min_size_bypass(app, monkeypatch) -> None:
    _enable_micro_canary(app)
    app.config["FIRST_CANARY_USE_MIN_SIZE_FALLBACK"] = True
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    db.session.commit()
    _patch_canary_market(app, monkeypatch)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is True
    assert payload["sizing"]["effective_allocation_budget_usd"] == 1.0
    assert payload["projected_order"]["notional"] <= 1.0


def test_live_funds_readiness_reports_missing_active_connection(app, monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})

    result = app.test_cli_runner().invoke(args=["live-funds-readiness", "--provider", "active"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is False
    assert "active_verified_live_connection_missing" in payload["blockers"]
    assert "user_id_required" in payload["blockers"]


def test_live_funds_readiness_reports_dependency_and_strict_readiness_blockers(app, monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": False, "blockers": ["Missing wallet dependencies: xrpl"]})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": False, "flask": False})

    result = app.test_cli_runner().invoke(args=["live-funds-readiness", "--provider", "active"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["strict_production_ready"] is False
    assert "strict_readiness_required" in payload["blockers"]
    assert payload["dependencies"]["pytest"] is False


def test_live_funds_readiness_reports_hyperliquid_ready_preview(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})

    result = app.test_cli_runner().invoke(
        args=[
            "live-funds-readiness",
            "--provider",
            "hyperliquid",
            "--user-id",
            str(user.id),
            "--ranking-id",
            str(ranking.id),
            "--connection-id",
            str(connection.id),
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is True
    assert payload["providers"][0]["provider"] == "hyperliquid"
    assert payload["providers"][0]["details"]["api_wallet_secret_configured"] is True
    assert payload["canary_preview"]["ready"] is True
    assert "canary_submit" in payload["next_commands"]
    assert AuditLog.query.filter_by(action="live_canary_preview").count() == 0


def test_live_funds_readiness_allows_hyperliquid_spot_unified_usdc(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(
        service,
        "account_snapshot",
        lambda user_id, mode, connection_id=None: ClientSnapshot(
            mode,
            [
                {"asset": "USDC", "type": "margin", "value": 0.0, "withdrawable": 0.0},
                {"asset": "USDC", "type": "spot", "value": 48.538977, "withdrawable": 48.538977},
            ],
            [],
            [],
            [],
            [],
        ),
    )
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})

    result = app.test_cli_runner().invoke(
        args=[
            "live-funds-readiness",
            "--provider",
            "hyperliquid",
            "--user-id",
            str(user.id),
            "--ranking-id",
            str(ranking.id),
            "--connection-id",
            str(connection.id),
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is True
    assert payload["providers"][0]["ready"] is True
    assert payload["providers"][0]["balance_readiness"]["ready"] is True
    assert payload["providers"][0]["balance_readiness"]["funding_source"] == "hyperliquid_spot_usdc_unified_available_to_trade"


def test_live_funds_readiness_blocks_hyperliquid_without_usdc(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(
        service,
        "account_snapshot",
        lambda user_id, mode, connection_id=None: ClientSnapshot(
            mode,
            [{"asset": "USDC", "type": "margin", "value": 0.0, "withdrawable": 0.0}],
            [],
            [],
            [],
            [],
        ),
    )
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})

    result = app.test_cli_runner().invoke(
        args=[
            "live-funds-readiness",
            "--provider",
            "hyperliquid",
            "--user-id",
            str(user.id),
            "--ranking-id",
            str(ranking.id),
            "--connection-id",
            str(connection.id),
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is False
    assert "hyperliquid_usdc_missing" in payload["blockers"]
    assert payload["providers"][0]["ready"] is False


def test_find_live_canary_ranking_stops_on_accepted_ranking(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    _enable_micro_canary(app)
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    monkeypatch.setattr(cli_module, "_find_canary_ml_order_symbols", lambda symbols, provider: symbols)
    optimizer = app.extensions["services"]["strategy_optimizer"]
    order_manager = app.extensions["services"]["order_manager"]
    captured: dict[str, object] = {}

    def fail_place_order(intent):
        raise AssertionError("find-live-canary-ranking must not submit orders")

    def fake_run(config):
        captured["symbols"] = list(config.symbols)
        captured["timeframes"] = list(config.timeframes)
        captured["mode"] = config.mode
        run = OptimizerRun(profile=config.profile, status="completed")
        run.symbols = list(config.symbols)
        run.timeframes = list(config.timeframes)
        db.session.add(run)
        db.session.flush()
        accepted = StrategyRanking(
            optimizer_run_id=run.id,
            provider=config.provider,
            strategy_name="scalping",
            symbol=config.symbols[0],
            timeframe=config.timeframes[0],
            profile=config.profile,
            score=9.0,
            total_return=0.02,
            net_return_after_costs=0.02,
            recent_performance_score=0.01,
            recent_1h_return=0.01,
            edge_score=15.0,
            expectancy=1.0,
            avg_win=2.0,
            avg_loss=1.0,
            win_loss_ratio=2.0,
            cost_drag_bps=2.0,
            convex_edge_score=10.0,
            mfe_mae_ratio=2.0,
            max_drawdown=-0.01,
            profit_factor=1.5,
            sharpe_like=1.2,
            sortino_like=1.1,
            trades_per_day=8.0,
            avg_trade_return=0.002,
            turnover_rate=0.2,
            turnover_after_fees=0.2,
            consistency=0.8,
            window_stability=0.9,
            accepted_window_ratio=0.8,
            win_rate=0.6,
            trade_count=12,
            allocation_amount_usd=10.0,
            lock_duration_hours=1,
            leverage=1.0,
            rejected=False,
            rejection_reason="",
        )
        accepted.parameters = {"risk_fraction": 0.01, "take_profit_pct": 0.04, "stop_loss_pct": 0.01}
        accepted.ml_explanation = {"net_roi_v2": {"net_roi_v2_score": 45.0, "roi_quality_grade": "A", "roi_rejection_risk": "low"}}
        db.session.add(accepted)
        db.session.commit()
        return {"optimizer_run_id": run.id, "ranking_count": 1, "accepted_count": 1, "top": [], "rejection_breakdown": {}}

    monkeypatch.setattr(optimizer, "run", fake_run)
    monkeypatch.setattr(order_manager, "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "find-live-canary-ranking",
            "--user-id",
            str(user.id),
            "--profile",
            "short_term",
            "--timeframe",
            "5m",
            "--research-depth",
            "quick",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is True
    assert payload["accepted_ranking_id"] is not None
    assert captured["symbols"] == ["BTC"]
    assert captured["timeframes"] == ["5m"]
    assert captured["mode"] == "live"
    assert payload["market_data_mode"] == "live"
    assert payload["fallback_symbols_used"] is True
    assert payload["candidate_order"] == ["BTC", "ETH"]
    assert payload["market_data_readiness"]["ready"] is True
    assert payload["sweeps"][0]["data_source"] == "live"
    assert payload["selected_readiness"]["live_micro_canary"]["enabled"] is True
    assert payload["selected_readiness"]["canary_preview"]["sizing"]["effective_allocation_budget_usd"] == 1.0
    assert payload["selected_readiness"]["canary_preview"]["projected_order"]["notional"] <= 1.0
    assert payload["fallback_attempted"] is False
    assert payload["fallback_used"] is False
    assert "canary_preview" in payload["next_commands"]
    assert payload["next_commands"]["submit_live_canary"] == (
        f"flask submit-live-canary --ranking-id {payload['accepted_ranking_id']} --user-id {user.id} "
        "--confirm LIVE-CANARY-TRADE"
    )
    combined = _cli_combined_output(result)
    assert "[find-live-canary-ranking] loading_services" in combined
    assert "[find-live-canary-ranking] accepted_ranking_found" in combined


def test_find_live_canary_ranking_reports_all_rejected(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    optimizer = app.extensions["services"]["strategy_optimizer"]
    calls: list[tuple[list[str], list[str]]] = []

    def fake_run(config):
        calls.append((list(config.symbols), list(config.timeframes)))
        run = OptimizerRun(profile=config.profile, status="completed")
        db.session.add(run)
        db.session.commit()
        return {
            "optimizer_run_id": run.id,
            "ranking_count": 2,
            "accepted_count": 0,
            "top": [],
            "rejection_breakdown": {"profit_factor_below_one": 2},
        }

    monkeypatch.setattr(optimizer, "run", fake_run)

    result = app.test_cli_runner().invoke(
        args=[
            "find-live-canary-ranking",
            "--user-id",
            str(user.id),
            "--profile",
            "short_term",
            "--research-depth",
            "quick",
            "--symbol",
            "BTC",
            "--timeframe",
            "5m",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is False
    assert payload["accepted_ranking_id"] is None
    assert "accepted_ranking_missing" in payload["blockers"]
    assert "all_candidates_rejected" in payload["blockers"]
    assert payload["no_accepted_ranking_reason"] == "all_candidates_rejected"
    assert payload["rejection_breakdown"] == {"profit_factor_below_one": 2}
    assert calls == [(["BTC"], ["5m"])]
    assert "canary_preview" not in payload["next_commands"]
    assert "[find-live-canary-ranking] no_eligible_ranking_found" in _cli_combined_output(result)


def test_find_live_canary_ranking_partial_sweeps_report_optimizer_incomplete(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    optimizer = app.extensions["services"]["strategy_optimizer"]

    def fake_run(config):
        run = OptimizerRun(profile=config.profile, status="partial")
        db.session.add(run)
        db.session.commit()
        return {
            "optimizer_run_id": run.id,
            "ranking_count": 1,
            "accepted_count": 0,
            "timed_out": True,
            "partial_result": True,
            "partial_reason": "optimizer_deadline_reached",
            "top": [
                {
                    "rejected": True,
                    "symbol": "BTC",
                    "timeframe": "5m",
                    "strategy_name": "breakout",
                    "rejection_reason": "profit_factor_below_one",
                }
            ],
            "rejection_breakdown": {"profit_factor_below_one": 1},
            "optimizer_runtime": {
                "phase_seconds": {
                    "candle_fetch": 0.1,
                    "backtest": 2.5,
                    "finalize": 0.2,
                },
                "slowest_market": {
                    "symbol": "BTC",
                    "timeframe": "5m",
                    "seconds": 0.1,
                    "candle_count": 200,
                    "window_count": 3,
                },
                "partial_result": True,
                "partial_reason": "optimizer_deadline_reached",
            },
        }

    monkeypatch.setattr(optimizer, "run", fake_run)

    result = app.test_cli_runner().invoke(
        args=[
            "find-live-canary-ranking",
            "--user-id",
            str(user.id),
            "--profile",
            "short_term",
            "--research-depth",
            "quick",
            "--symbol",
            "BTC",
            "--timeframe",
            "5m",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is False
    assert payload["accepted_ranking_id"] is None
    assert payload["no_accepted_ranking_reason"] == "optimizer_sweeps_incomplete"
    assert "optimizer_sweeps_incomplete" in payload["blockers"]
    assert "all_candidates_rejected" not in payload["blockers"]
    assert payload["rejection_breakdown"] == {"profit_factor_below_one": 1}
    sweep = payload["sweeps"][0]
    assert sweep["timed_out"] is True
    assert sweep["partial_result"] is True
    assert sweep["partial_reason"] == "optimizer_deadline_reached"
    assert sweep["failed_phase"] == "running_optimizer_backtests"
    assert sweep["optimizer_phase_seconds"]["backtest"] == 2.5
    assert sweep["slowest_market"]["symbol"] == "BTC"
    assert "max-parameter-sets" in payload["operator_next_steps"][0]
    assert payload["fallback_attempted"] is False
    assert payload["fallback_used"] is False


def test_find_live_canary_ranking_uses_cached_fallback_after_incomplete_sweeps(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    app.config["FIND_CANARY_FALLBACK_ENABLED"] = True
    app.config["FIND_CANARY_FALLBACK_STRATEGY"] = ranking.strategy_name
    app.config["FIND_CANARY_FALLBACK_TIMEFRAME"] = ranking.timeframe
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})

    def fake_run(config):
        run = OptimizerRun(profile=config.profile, status="partial")
        db.session.add(run)
        db.session.commit()
        return {
            "optimizer_run_id": run.id,
            "ranking_count": 1,
            "accepted_count": 0,
            "timed_out": True,
            "partial_result": True,
            "partial_reason": "optimizer_deadline_reached",
            "top": [{"rejected": True, "symbol": "BTC", "rejection_reason": "profit_factor_below_one"}],
            "rejection_breakdown": {"profit_factor_below_one": 1},
            "optimizer_runtime": {"phase_seconds": {"backtest": 2.0}, "partial_result": True},
        }

    def fail_place_order(intent):
        raise AssertionError("find-live-canary-ranking must not submit orders")

    preview_calls: list[dict[str, object]] = []

    def ready_preview(**kwargs):
        preview_calls.append(dict(kwargs))
        return {
            "ready": True,
            "blockers": [],
            "ranking_id": kwargs["ranking_id"],
            "user_id": kwargs["user_id"],
            "connection_id": kwargs["connection_id"],
            "submitted": False,
            "real_order_submitted": False,
            "preview_only": True,
            "next_commands": {"canary_preview": "flask live-canary-trade"},
        }

    monkeypatch.setattr(app.extensions["services"]["strategy_optimizer"], "run", fake_run)
    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)
    monkeypatch.setattr(cli_module, "_live_canary_trade_payload", ready_preview)

    result = app.test_cli_runner().invoke(
        args=[
            "find-live-canary-ranking",
            "--user-id",
            str(user.id),
            "--profile",
            "short_term",
            "--research-depth",
            "quick",
            "--symbol",
            "BTC",
            "--timeframe",
            "5m",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is True
    assert payload["accepted_ranking_id"] == ranking.id
    assert payload["no_accepted_ranking_reason"] == "optimizer_sweeps_incomplete"
    assert payload["fallback_attempted"] is True
    assert payload["fallback_used"] is True
    assert payload["fallback_source"] == "cached_ranking"
    assert payload["fallback_ranking_id"] == ranking.id
    assert payload["fallback_preview"]["ready"] is True
    expected_preview_call = {
        "ranking_id": ranking.id,
        "user_id": user.id,
        "connection_id": connection.id,
        "submit": False,
        "record_preview_audit": False,
    }
    assert expected_preview_call in preview_calls
    assert all(call["submit"] is False and call["record_preview_audit"] is False for call in preview_calls)
    assert payload["blockers"] == []
    assert "canary_preview" in payload["next_commands"]
    assert Order.query.count() == 0
    audit = AuditLog.query.filter_by(action="find_live_canary_fallback_selected").one()
    assert audit.details["fallback_source"] == "cached_ranking"


def test_find_live_canary_ranking_creates_synthetic_fallback_after_incomplete_sweeps(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    app.config["FIND_CANARY_FALLBACK_ENABLED"] = True
    app.config["FIND_CANARY_FALLBACK_STRATEGY"] = "scalping"
    app.config["FIND_CANARY_FALLBACK_TIMEFRAME"] = "5m"
    app.config["FIND_CANARY_FALLBACK_ALLOCATION_USD"] = 1.0
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})

    def fake_run(config):
        run = OptimizerRun(profile=config.profile, status="partial")
        db.session.add(run)
        db.session.commit()
        return {
            "optimizer_run_id": run.id,
            "ranking_count": 1,
            "accepted_count": 0,
            "timed_out": True,
            "partial_result": True,
            "partial_reason": "optimizer_deadline_reached",
            "top": [{"rejected": True, "symbol": "BTC", "rejection_reason": "profit_factor_below_one"}],
            "rejection_breakdown": {"profit_factor_below_one": 1},
            "optimizer_runtime": {"phase_seconds": {"backtest": 2.0}, "partial_result": True},
        }

    def fail_place_order(intent):
        raise AssertionError("find-live-canary-ranking must not submit orders")

    preview_calls: list[dict[str, object]] = []

    def ready_preview(**kwargs):
        preview_calls.append(dict(kwargs))
        return {
            "ready": True,
            "blockers": [],
            "ranking_id": kwargs["ranking_id"],
            "user_id": kwargs["user_id"],
            "connection_id": kwargs["connection_id"],
            "submitted": False,
            "real_order_submitted": False,
            "preview_only": True,
            "next_commands": {"canary_preview": "flask live-canary-trade"},
        }

    monkeypatch.setattr(app.extensions["services"]["strategy_optimizer"], "run", fake_run)
    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)
    monkeypatch.setattr(cli_module, "_live_canary_trade_payload", ready_preview)

    result = app.test_cli_runner().invoke(
        args=[
            "find-live-canary-ranking",
            "--user-id",
            str(user.id),
            "--profile",
            "short_term",
            "--research-depth",
            "quick",
            "--symbol",
            "BTC",
            "--timeframe",
            "5m",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is True
    assert payload["fallback_used"] is True
    assert payload["fallback_source"] == "synthetic_ranking"
    assert payload["fallback_ranking_id"] != ranking.id
    fallback = db.session.get(StrategyRanking, payload["fallback_ranking_id"])
    assert fallback is not None
    assert fallback.rejected is False
    assert fallback.symbol == "BTC"
    assert fallback.strategy_name == "scalping"
    assert fallback.allocation_amount_usd == 1.0
    assert fallback.leverage == 1.0
    assert fallback.parameters["synthetic_live_canary_fallback"] is True
    assert fallback.parameters["stop_loss_pct"] == app.config["FIND_CANARY_FALLBACK_STOP_LOSS_PCT"]
    assert payload["fallback_preview"]["ready"] is True
    expected_preview_call = {
        "ranking_id": fallback.id,
        "user_id": user.id,
        "connection_id": connection.id,
        "submit": False,
        "record_preview_audit": False,
    }
    assert expected_preview_call in preview_calls
    assert all(call["submit"] is False and call["record_preview_audit"] is False for call in preview_calls)
    assert Order.query.count() == 0


def test_find_live_canary_ranking_rejects_synthetic_fallback_when_preview_fails(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    app.config["FIND_CANARY_FALLBACK_ENABLED"] = True
    app.config["FIND_CANARY_FALLBACK_STRATEGY"] = "scalping"
    app.config["FIND_CANARY_FALLBACK_TIMEFRAME"] = "5m"
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})

    def fake_run(config):
        run = OptimizerRun(profile=config.profile, status="partial")
        db.session.add(run)
        db.session.commit()
        return {
            "optimizer_run_id": run.id,
            "ranking_count": 1,
            "accepted_count": 0,
            "timed_out": True,
            "partial_result": True,
            "partial_reason": "optimizer_deadline_reached",
            "top": [{"rejected": True, "symbol": "BTC", "rejection_reason": "profit_factor_below_one"}],
            "rejection_breakdown": {"profit_factor_below_one": 1},
            "optimizer_runtime": {"phase_seconds": {"backtest": 2.0}, "partial_result": True},
        }

    def blocked_preview(**kwargs):
        return {
            "ready": False,
            "blockers": ["stop_loss_required"],
            "ranking_id": kwargs["ranking_id"],
            "user_id": kwargs["user_id"],
            "connection_id": kwargs["connection_id"],
            "submitted": False,
            "real_order_submitted": False,
            "preview_only": True,
        }

    monkeypatch.setattr(app.extensions["services"]["strategy_optimizer"], "run", fake_run)
    monkeypatch.setattr(cli_module, "_live_canary_trade_payload", blocked_preview)

    result = app.test_cli_runner().invoke(
        args=[
            "find-live-canary-ranking",
            "--user-id",
            str(user.id),
            "--profile",
            "short_term",
            "--research-depth",
            "quick",
            "--symbol",
            "BTC",
            "--timeframe",
            "5m",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is False
    assert payload["accepted_ranking_id"] is None
    assert payload["fallback_attempted"] is True
    assert payload["fallback_used"] is False
    assert "stop_loss_required" in payload["fallback_blockers"]
    assert "fallback_unavailable" in payload["blockers"]
    rejected_fallback = StrategyRanking.query.filter_by(universe_source="find_canary_fallback").one()
    assert rejected_fallback.rejected is True
    assert rejected_fallback.rejection_reason.startswith("fallback_preview_blocked:stop_loss_required")


def test_find_live_canary_ranking_panic_lock_blocks_fallback(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    app.config["FIND_CANARY_FALLBACK_ENABLED"] = True
    app.config["FIND_CANARY_FALLBACK_STRATEGY"] = "scalping"
    app.config["FIND_CANARY_FALLBACK_TIMEFRAME"] = "5m"
    Setting.set_json("panic_lock", True)
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})

    def fake_run(config):
        run = OptimizerRun(profile=config.profile, status="partial")
        db.session.add(run)
        db.session.commit()
        return {
            "optimizer_run_id": run.id,
            "ranking_count": 1,
            "accepted_count": 0,
            "timed_out": True,
            "partial_result": True,
            "partial_reason": "optimizer_deadline_reached",
            "top": [{"rejected": True, "symbol": "BTC", "rejection_reason": "profit_factor_below_one"}],
            "rejection_breakdown": {"profit_factor_below_one": 1},
            "optimizer_runtime": {"phase_seconds": {"backtest": 2.0}, "partial_result": True},
        }

    monkeypatch.setattr(app.extensions["services"]["strategy_optimizer"], "run", fake_run)
    monkeypatch.setattr(
        cli_module,
        "_live_canary_trade_payload",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("panic lock must block fallback before preview")),
    )

    result = app.test_cli_runner().invoke(
        args=[
            "find-live-canary-ranking",
            "--user-id",
            str(user.id),
            "--profile",
            "short_term",
            "--research-depth",
            "quick",
            "--symbol",
            "BTC",
            "--timeframe",
            "5m",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["fallback_attempted"] is True
    assert payload["fallback_used"] is False
    assert "panic_lock_active" in payload["fallback_blockers"]
    assert "fallback_unavailable" in payload["blockers"]
    assert payload["fallback_preview"] is None
    assert StrategyRanking.query.filter_by(universe_source="find_canary_fallback").count() == 0


def test_find_live_canary_ranking_uses_provider_fallback_when_primary_market_data_fails(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    app.config["FIND_CANARY_RANKING_FALLBACK_SYMBOLS"] = {"active": ["BAD-USDC", "ETH-USDC"]}
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    monkeypatch.setattr(cli_module, "_find_canary_ml_order_symbols", lambda symbols, provider: symbols)
    market_data = app.extensions["services"]["market_data"]
    candles = [
        {"timestamp": index, "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.0 + index * 0.01, "volume": 1000.0}
        for index in range(80)
    ]

    def fake_candles(symbol, timeframe, mode="live", limit=None):
        if symbol == "BAD":
            raise RuntimeError("market data unavailable for BAD")
        return candles[-(limit or 80):]

    monkeypatch.setattr(market_data, "get_candles", fake_candles)
    optimizer = app.extensions["services"]["strategy_optimizer"]
    calls: list[list[str]] = []

    def fake_run(config):
        symbols = list(config.symbols)
        calls.append(symbols)
        run = OptimizerRun(profile=config.profile, status="completed")
        db.session.add(run)
        db.session.commit()
        return {
            "optimizer_run_id": run.id,
            "ranking_count": 1,
            "accepted_count": 0,
            "top": [],
            "rejection_breakdown": {"profit_factor_below_one": 1},
        }

    monkeypatch.setattr(optimizer, "run", fake_run)

    result = app.test_cli_runner().invoke(
        args=[
            "find-live-canary-ranking",
            "--user-id",
            str(user.id),
            "--profile",
            "short_term",
            "--research-depth",
            "quick",
            "--timeframe",
            "5m",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is False
    assert calls == [["ETH"]]
    assert payload["fallback_symbols_used"] is True
    assert payload["candidate_order"] == ["BAD", "ETH"]
    assert payload["market_data_readiness"]["skipped_symbols"] == ["BAD"]
    assert payload["market_data_readiness"]["results"][0]["status"] == "error"
    assert payload["market_data_readiness"]["results"][0]["error"] == "market data unavailable for BAD"
    assert payload["sweeps"][0]["symbols"] == ["ETH"]
    assert payload["sweeps"][0]["market_data_status"] == "ok"
    assert payload["rejection_breakdown"]["profit_factor_below_one"] == 1


def test_find_live_canary_ranking_reports_market_data_unavailable_for_all_symbols(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    monkeypatch.setattr(
        app.extensions["services"]["market_data"],
        "get_candles",
        lambda symbol, timeframe, mode="live", limit=None: (_ for _ in ()).throw(RuntimeError("provider candles unavailable")),
    )

    def fail_run(config):
        raise AssertionError("optimizer must not run without usable market data")

    monkeypatch.setattr(app.extensions["services"]["strategy_optimizer"], "run", fail_run)

    result = app.test_cli_runner().invoke(
        args=[
            "find-live-canary-ranking",
            "--user-id",
            str(user.id),
            "--profile",
            "short_term",
            "--research-depth",
            "quick",
            "--symbol",
            "BTC",
            "--timeframe",
            "5m",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is False
    assert payload["accepted_ranking_id"] is None
    assert payload["timed_out"] is False
    assert payload["failed_phase"] == "fetching_candles"
    assert payload["no_accepted_ranking_reason"] == "market_data_unavailable_for_all_symbols"
    assert "market_data_unavailable_for_all_symbols" in payload["blockers"]
    assert payload["market_data_readiness"]["results"][0]["status"] == "error"
    assert payload["market_data_readiness"]["results"][0]["error"] == "provider candles unavailable"
    assert payload["sweeps"] == []
    assert "provider/network" in payload["operator_next_steps"][0]


def test_find_live_canary_ranking_testing_mock_candles_complete_without_order_submission(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    app.config["FIND_CANARY_RANKING_ALLOW_MOCK_DATA"] = True
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    monkeypatch.setattr(
        app.extensions["services"]["market_data"],
        "get_candles",
        lambda symbol, timeframe, mode="live", limit=None: (_ for _ in ()).throw(RuntimeError("live candles unavailable")),
    )

    def fail_place_order(intent):
        raise AssertionError("find-live-canary-ranking must not submit orders")

    def fake_run(config):
        run = OptimizerRun(profile=config.profile, status="completed")
        run.symbols = list(config.symbols)
        run.timeframes = list(config.timeframes)
        db.session.add(run)
        db.session.flush()
        accepted = StrategyRanking(
            optimizer_run_id=run.id,
            provider=config.provider,
            strategy_name="scalping",
            symbol=config.symbols[0],
            timeframe=config.timeframes[0],
            profile=config.profile,
            score=9.0,
            total_return=0.02,
            net_return_after_costs=0.02,
            recent_performance_score=0.01,
            recent_1h_return=0.01,
            edge_score=15.0,
            expectancy=1.0,
            avg_win=2.0,
            avg_loss=1.0,
            win_loss_ratio=2.0,
            cost_drag_bps=2.0,
            convex_edge_score=10.0,
            mfe_mae_ratio=2.0,
            max_drawdown=-0.01,
            profit_factor=1.5,
            sharpe_like=1.2,
            sortino_like=1.1,
            trades_per_day=8.0,
            avg_trade_return=0.002,
            turnover_rate=0.2,
            turnover_after_fees=0.2,
            consistency=0.8,
            window_stability=0.9,
            accepted_window_ratio=0.8,
            win_rate=0.6,
            trade_count=12,
            allocation_amount_usd=10.0,
            lock_duration_hours=1,
            leverage=1.0,
            rejected=False,
            rejection_reason="",
        )
        accepted.parameters = {"risk_fraction": 0.01, "take_profit_pct": 0.04, "stop_loss_pct": 0.01}
        db.session.add(accepted)
        db.session.commit()
        return {"optimizer_run_id": run.id, "ranking_count": 1, "accepted_count": 1, "top": [], "rejection_breakdown": {}}

    monkeypatch.setattr(app.extensions["services"]["strategy_optimizer"], "run", fake_run)
    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "find-live-canary-ranking",
            "--user-id",
            str(user.id),
            "--profile",
            "short_term",
            "--research-depth",
            "quick",
            "--symbol",
            "BTC",
            "--timeframe",
            "5m",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is False
    assert payload["accepted_ranking_id"] is not None
    assert "mock_data_used_not_live_submittable" in payload["blockers"]
    assert payload["mock_data_used_for_selected_ranking"] is True
    assert payload["market_data_readiness"]["mock_data_allowed"] is True
    assert payload["market_data_readiness"]["summary"]["mock_data_used"] is True
    assert payload["sweeps"][0]["mock_data_used"] is True
    assert payload["sweeps"][0]["canary_ready"] is False
    assert Order.query.count() == 0


def test_find_live_canary_ranking_mock_data_is_blocked_outside_testing(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    app.config["FIND_CANARY_RANKING_ALLOW_MOCK_DATA"] = True
    app.config["TESTING"] = False
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    monkeypatch.setattr(
        app.extensions["services"]["market_data"],
        "get_candles",
        lambda symbol, timeframe, mode="live", limit=None: (_ for _ in ()).throw(RuntimeError("live candles unavailable")),
    )

    def fail_run(config):
        raise AssertionError("mock data must not enable optimizer outside TESTING")

    monkeypatch.setattr(app.extensions["services"]["strategy_optimizer"], "run", fail_run)

    result = app.test_cli_runner().invoke(
        args=[
            "find-live-canary-ranking",
            "--user-id",
            str(user.id),
            "--profile",
            "short_term",
            "--research-depth",
            "quick",
            "--symbol",
            "BTC",
            "--timeframe",
            "5m",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is False
    assert payload["market_data_readiness"]["mock_data_requested"] is True
    assert payload["market_data_readiness"]["mock_data_allowed"] is False
    assert "mock_data_disabled_in_live" in payload["market_data_readiness"]["blockers"]
    assert "market_data_unavailable_for_all_symbols" in payload["blockers"]
    assert payload["sweeps"] == []


def test_find_live_canary_ranking_preserves_explicit_order_and_ml_orders_only_fallback(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    app.config["FIND_CANARY_RANKING_FALLBACK_SYMBOLS"] = {"active": ["BTC", "ETH"]}
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    monkeypatch.setattr(cli_module, "_find_canary_symbol_ml_score", lambda symbol, provider: 10.0 if symbol == "ETH" else 1.0)
    optimizer = app.extensions["services"]["strategy_optimizer"]
    calls: list[list[str]] = []

    def fake_run(config):
        calls.append(list(config.symbols))
        run = OptimizerRun(profile=config.profile, status="completed")
        db.session.add(run)
        db.session.commit()
        return {
            "optimizer_run_id": run.id,
            "ranking_count": 0,
            "accepted_count": 0,
            "top": [],
            "rejection_breakdown": {},
        }

    monkeypatch.setattr(optimizer, "run", fake_run)

    explicit_result = app.test_cli_runner().invoke(
        args=[
            "find-live-canary-ranking",
            "--user-id",
            str(user.id),
            "--profile",
            "short_term",
            "--research-depth",
            "quick",
            "--symbol",
            "BTC",
            "--symbol",
            "ETH",
            "--timeframe",
            "5m",
        ]
    )
    explicit_payload = _cli_json_payload(explicit_result)

    assert explicit_result.exit_code == 0
    assert explicit_payload["candidate_order"] == ["BTC", "ETH"]
    assert calls == [["BTC"], ["ETH"]]

    calls.clear()
    fallback_result = app.test_cli_runner().invoke(
        args=[
            "find-live-canary-ranking",
            "--user-id",
            str(user.id),
            "--profile",
            "short_term",
            "--research-depth",
            "quick",
            "--timeframe",
            "5m",
        ]
    )
    fallback_payload = _cli_json_payload(fallback_result)

    assert fallback_result.exit_code == 0
    assert fallback_payload["fallback_symbols_used"] is True
    assert fallback_payload["candidate_order"] == ["ETH", "BTC"]
    assert calls == [["ETH"], ["BTC"]]


def test_find_live_canary_ranking_runs_adaptive_research_after_default_sweep(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    monkeypatch.setattr(cli_module, "_find_canary_ml_order_symbols", lambda symbols, provider: symbols)
    optimizer = app.extensions["services"]["strategy_optimizer"]
    calls: list[dict[str, object]] = []

    def fake_run(config):
        calls.append(
            {
                "symbols": list(config.symbols or []),
                "profile": config.profile,
                "strategy_names": list(config.strategy_names or []),
                "timeframes": list(config.timeframes or []),
                "max_parameter_sets": int(config.max_parameter_sets),
            }
        )
        run = OptimizerRun(profile=config.profile, status="completed")
        db.session.add(run)
        db.session.commit()
        return {
            "optimizer_run_id": run.id,
            "ranking_count": 2,
            "accepted_count": 0,
            "top": [],
            "rejection_breakdown": {"profit_factor_below_one": 2},
        }

    monkeypatch.setattr(optimizer, "run", fake_run)

    result = app.test_cli_runner().invoke(args=["find-live-canary-ranking", "--user-id", str(user.id)])

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is False
    assert payload["adaptive_research"] is True
    stage_names = [stage["stage"] for stage in payload["adaptive_stages"]]
    assert stage_names == [
        "profit_factor_stability_focus_btc",
        "profit_factor_stability_focus_eth",
        "intraday_breakout_focus_btc",
        "intraday_breakout_focus_eth",
    ]
    assert any(call["strategy_names"] == ["rsi_mean_reversion", "mean_reversion", "ema_crossover", "breakout"] for call in calls)
    assert any(call["strategy_names"] == ["scalping", "volatility_breakout", "breakout", "rsi_mean_reversion"] for call in calls)
    assert all(len(call["symbols"]) == 1 for call in calls)
    assert payload["market_data_readiness"]["summary"]["candidate_count"] == 6
    assert payload["research_diagnostics"]["coverage"]["sweep_count"] == len(calls)
    assert payload["rejection_breakdown"]["profit_factor_below_one"] == len(calls) * 2


def test_find_live_canary_ranking_ml_depth_prioritizes_without_accepting_rejected(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    optimizer = app.extensions["services"]["strategy_optimizer"]
    overlays: list[str] = []

    def fake_run(config):
        overlays.append(config.live_canary_research_overlay)
        run = OptimizerRun(profile=config.profile, status="completed")
        db.session.add(run)
        db.session.commit()
        return {
            "optimizer_run_id": run.id,
            "ranking_count": 1,
            "accepted_count": 0,
            "top": [{"rejected": True, "symbol": "BTC", "rejection_reason": "profit_factor_below_one"}],
            "rejection_breakdown": {"profit_factor_below_one": 1},
        }

    monkeypatch.setattr(optimizer, "run", fake_run)

    result = app.test_cli_runner().invoke(args=["find-live-canary-ranking", "--user-id", str(user.id), "--research-depth", "ml"])

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["accepted_ranking_id"] is None
    assert payload["ml_research"]["hard_rejection_override_allowed"] is False
    assert any(stage.get("ml_priority_source") for stage in payload["adaptive_stages"])
    assert "profit_factor_below_one" in overlays


def test_find_live_canary_ranking_timeout_exits_cleanly(app, monkeypatch) -> None:
    user = User(username="canary-timeout", password_hash="hash")
    db.session.add(user)
    db.session.commit()

    def slow_discovery(**kwargs):
        cli_module._find_canary_phase(
            kwargs.get("progress_state"),
            kwargs.get("progress"),
            "checking_provider_readiness",
        )
        time.sleep(1.0)
        return {"ready": True}

    monkeypatch.setattr(cli_module, "_find_live_canary_ranking_payload", slow_discovery)

    result = app.test_cli_runner().invoke(
        args=[
            "find-live-canary-ranking",
            "--user-id",
            str(user.id),
            "--timeout-seconds",
            "0.01",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is False
    assert payload["timed_out"] is True
    assert payload["failed_phase"] == "checking_provider_readiness"
    assert payload["accepted_ranking_id"] is None
    assert "find_canary_ranking_timeout" in payload["blockers"]
    assert "[find-live-canary-ranking] timeout" in _cli_combined_output(result)
    audit = AuditLog.query.filter_by(action="find_live_canary_ranking_timeout").one()
    assert audit.details["phase"] == "checking_provider_readiness"


def test_find_live_canary_ranking_symbol_cap_limits_work(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    app.config["FIND_CANARY_MAX_SYMBOLS"] = 2
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    optimizer = app.extensions["services"]["strategy_optimizer"]
    calls: list[list[str]] = []

    def fake_run(config):
        calls.append(list(config.symbols))
        run = OptimizerRun(profile=config.profile, status="completed")
        db.session.add(run)
        db.session.commit()
        return {
            "optimizer_run_id": run.id,
            "ranking_count": 0,
            "accepted_count": 0,
            "top": [],
            "rejection_breakdown": {},
        }

    monkeypatch.setattr(optimizer, "run", fake_run)

    result = app.test_cli_runner().invoke(
        args=[
            "find-live-canary-ranking",
            "--user-id",
            str(user.id),
            "--profile",
            "short_term",
            "--research-depth",
            "quick",
            "--timeframe",
            "5m",
            "--symbol",
            "ETH",
            "--symbol",
            "BTC",
            "--symbol",
            "SOL",
            "--symbol",
            "HYPE",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["symbols"] == ["ETH", "BTC"]
    assert payload["omitted_symbols"] == ["SOL", "HYPE"]
    assert payload["candidate_order"] == ["ETH", "BTC"]
    assert payload["max_symbols"] == 2
    assert calls == [["ETH"], ["BTC"]]


def test_find_live_canary_ranking_provider_failure_is_diagnostic(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    db.session.commit()
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    trading_connections = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(trading_connections, "can_trade", lambda user_id, mode, connection_id=None: True)

    def fail_snapshot(user_id, mode, connection_id=None):
        raise RuntimeError("provider snapshot unavailable")

    monkeypatch.setattr(trading_connections, "account_snapshot", fail_snapshot)

    result = app.test_cli_runner().invoke(
        args=["find-live-canary-ranking", "--user-id", str(user.id), "--profile", "short_term"]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["ready"] is False
    assert payload["failed_phase"] == "checking_provider_readiness"
    assert payload["error"] == "provider snapshot unavailable"
    assert "provider_readiness_failed" in payload["blockers"]


def test_live_auto_canary_no_accepted_ranking_does_not_preview_or_submit(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    db.session.commit()
    _patch_canary_market(app, monkeypatch)

    def fake_discovery(**kwargs):
        return {
            "ready": False,
            "accepted_ranking_id": None,
            "blockers": ["accepted_ranking_missing"],
            "rejection_breakdown": {"profit_factor_below_one": 3},
            "sweeps": [],
        }

    def fail_preview(*args, **kwargs):
        raise AssertionError("live-auto-canary must not preview without an accepted ranking")

    monkeypatch.setattr(cli_module, "_find_live_canary_ranking_payload", fake_discovery)
    monkeypatch.setattr(cli_module, "_live_canary_trade_payload", fail_preview)

    result = app.test_cli_runner().invoke(args=["live-auto-canary", "--user-id", str(user.id)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["accepted_ranking_id"] is None
    assert "accepted_ranking_missing" in payload["blockers"]
    assert payload["provider_results"] == []
    assert payload["real_order_submitted"] is False
    assert AuditLog.query.filter_by(action="live_auto_canary").count() == 1


def test_find_live_canary_ranking_respects_expired_research_budget(app, monkeypatch) -> None:
    user, connection, _ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    db.session.commit()
    _patch_canary_market(app, monkeypatch)

    def fail_sweep(*args, **kwargs):
        raise AssertionError("expired research budget must stop before optimizer sweeps")

    monkeypatch.setattr(cli_module, "_run_live_canary_profile_sweeps", fail_sweep)

    payload = cli_module._find_live_canary_ranking_payload(
        user_id=user.id,
        provider="active",
        symbols=["BTC", "ETH", "SOL", "HYPE"],
        timeframes=["5m", "15m", "1h"],
        profiles=["short_term", "aggressive_1h"],
        max_parameter_sets=12,
        allocation_amount_usd=10.0,
        lock_duration_hours=1,
        auto_deploy_top_n=1,
        strategy_names=None,
        research_depth="ml",
        adaptive_research=True,
        deadline_monotonic=time.monotonic() - 1.0,
    )

    assert payload["accepted_ranking_id"] is None
    assert payload["failed_phase"] == "fetching_candles"
    assert "market_data_probe_budget_exhausted" in payload["blockers"]
    assert "market_data_unavailable_for_all_symbols" in payload["blockers"]
    assert payload["sweeps"] == []


def test_live_auto_canary_previews_both_providers_without_submit(app, monkeypatch) -> None:
    user, hyperliquid, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    hyperliquid.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    kucoin = service.create_or_update(
        user_id=user.id,
        provider="kucoin",
        connection_type="cex_api_key",
        api_key="key",
        api_secret="secret",
        passphrase="passphrase",
        is_active=False,
    )
    kucoin.verification_status = "verified"
    kucoin.is_active = True
    app.config["KUCOIN_SYMBOL_MAP_JSON"] = '{"BTC":"XBTUSDTM"}'
    app.config["KUCOIN_CONTRACT_SPECS_JSON"] = '{"XBTUSDTM":{"contract_size":0.001,"size_step":1,"min_size":1}}'
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    monkeypatch.setattr(
        cli_module,
        "_find_live_canary_ranking_payload",
        lambda **kwargs: {"ready": True, "accepted_ranking_id": ranking.id, "blockers": [], "sweeps": []},
    )

    result = app.test_cli_runner().invoke(args=["live-auto-canary", "--user-id", str(user.id)])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is False
    assert payload["submitted_count"] == 0
    assert [item["provider"] for item in payload["provider_results"]] == ["hyperliquid", "kucoin"]
    assert payload["provider_results"][0]["preview"]["ready"] is True
    assert payload["provider_results"][1]["ready"] is False
    assert "ranking_provider_mismatch" in payload["provider_results"][1]["blockers"]
    assert Order.query.count() == 0


def test_kucoin_readiness_uses_connector_default_contract_metadata(app) -> None:
    user, _hyperliquid, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    kucoin = service.create_or_update(
        user_id=user.id,
        provider="kucoin",
        connection_type="cex_api_key",
        api_key="key",
        api_secret="secret",
        passphrase="passphrase",
        is_active=False,
    )
    kucoin.verification_status = "verified"
    ranking.provider = "kucoin"
    ranking.symbol = "BTC"
    app.config["KUCOIN_SYMBOL_MAP_JSON"] = ""
    app.config["KUCOIN_CONTRACT_SPECS_JSON"] = ""
    db.session.commit()

    details = cli_module._kucoin_readiness_details(kucoin, ranking)

    assert details["venue_symbol"] == "XBTUSDTM"
    assert details["symbol_mapped"] is True
    assert details["contract_spec_available"] is True
    assert details["blockers"] == []


def test_live_auto_canary_can_target_inactive_verified_kucoin_connection(app, monkeypatch) -> None:
    user, hyperliquid, _ranking = _add_canary_ranking()
    ranking = _add_extra_canary_ranking(user, symbol="BTC", provider="kucoin")
    service = app.extensions["services"]["trading_connections"]
    hyperliquid.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    kucoin = service.create_or_update(
        user_id=user.id,
        provider="kucoin",
        connection_type="cex_api_key",
        api_key="key",
        api_secret="secret",
        passphrase="passphrase",
        is_active=False,
    )
    kucoin.verification_status = "verified"
    kucoin.is_active = False
    app.config["KUCOIN_SYMBOL_MAP_JSON"] = ""
    app.config["KUCOIN_CONTRACT_SPECS_JSON"] = ""
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    captured: dict[str, object] = {}

    def fake_discovery(**kwargs):
        captured.update(kwargs)
        return {"ready": True, "accepted_ranking_id": ranking.id, "blockers": [], "sweeps": []}

    monkeypatch.setattr(cli_module, "_find_live_canary_ranking_payload", fake_discovery)

    result = app.test_cli_runner().invoke(
        args=[
            "live-auto-canary",
            "--user-id",
            str(user.id),
            "--provider",
            "kucoin",
            "--connection-id",
            str(kucoin.id),
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert captured["connection_id"] == kucoin.id
    assert payload["provider_results"][0]["provider"] == "kucoin"
    assert payload["provider_results"][0]["connection_id"] == kucoin.id
    assert payload["provider_results"][0]["readiness"]["providers"][0]["details"]["venue_symbol"] == "XBTUSDTM"
    assert payload["submitted_count"] == 0
    assert Order.query.count() == 0


def test_live_canary_ml_priority_uses_provider_scoped_offline_ranker(app, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeOfflineRanker:
        def score_payload(self, context, horizon, *, base_score=None, rejected=False):
            captured["score_context"] = dict(context)
            return {"status": "promoted", "prediction": 1.25}

        def readiness(self, horizon, *, require_blend=False, provider="global"):
            captured["readiness_provider"] = provider
            return {
                "ready": True,
                "horizon": horizon,
                "provider": provider,
                "blockers": [],
                "promoted_model": {"id": 71},
                "blend_enabled": True,
                "require_blend": require_blend,
            }

    class FakeOnlineRanker:
        def predict_score(self, context, horizon):
            return 0.5

    def fake_get_service(name):
        if name == "offline_ranker":
            return FakeOfflineRanker()
        if name == "online_ranker":
            return FakeOnlineRanker()
        raise AssertionError(name)

    monkeypatch.setattr(cli_module, "get_service", fake_get_service)

    stage = {
        "symbols": ["ETH"],
        "timeframes": ["1h"],
        "strategy_names": ["breakout"],
        "profile": "short_term",
        "research_overlay": "profit_factor_below_one",
    }

    score = cli_module._live_canary_stage_ml_score(stage, provider="kucoin")
    status = cli_module._live_canary_ml_research_status("kucoin")

    assert score["source"] == "offline_ranker"
    assert score["score"] == 1.75
    assert captured["score_context"]["provider"] == "kucoin"
    assert captured["score_context"]["execution_venue"] == "kucoin"
    assert status["offline_ranker"]["provider"] == "kucoin"
    assert captured["readiness_provider"] == "kucoin"


def test_live_auto_canary_rapid_previews_multiple_accepted_rankings(app, monkeypatch) -> None:
    user, hyperliquid, ranking = _add_canary_ranking()
    second = _add_extra_canary_ranking(user, symbol="ETH", provider="hyperliquid")
    service = app.extensions["services"]["trading_connections"]
    hyperliquid.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    monkeypatch.setattr(
        cli_module,
        "_find_live_canary_ranking_payload",
        lambda **kwargs: {
            "ready": True,
            "accepted_ranking_id": ranking.id,
            "accepted_ranking_ids": [ranking.id, second.id],
            "blockers": [],
            "sweeps": [],
        },
    )

    result = app.test_cli_runner().invoke(
        args=[
            "live-auto-canary",
            "--user-id",
            str(user.id),
            "--provider",
            "hyperliquid",
            "--max-submissions",
            "2",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["accepted_ranking_ids"] == [ranking.id, second.id]
    assert payload["submitted_count"] == 0
    assert [item["ranking_id"] for item in payload["provider_results"]] == [ranking.id, second.id]
    assert all(item["preview"]["ready"] for item in payload["provider_results"])
    assert Order.query.count() == 0


def test_live_auto_canary_submit_requires_exact_confirmation(app) -> None:
    result = app.test_cli_runner().invoke(args=["live-auto-canary", "--submit", "--confirm", "WRONG"])

    assert result.exit_code != 0
    assert "LIVE-CANARY-TRADE" in result.output


def test_live_auto_canary_preview_only_blocks_submit_and_skips_kucoin(app, monkeypatch) -> None:
    user, hyperliquid, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    hyperliquid.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    kucoin = service.create_or_update(
        user_id=user.id,
        provider="kucoin",
        connection_type="cex_api_key",
        api_key="key",
        api_secret="secret",
        passphrase="passphrase",
        is_active=False,
    )
    kucoin.verification_status = "verified"
    kucoin.is_active = True
    app.config["KUCOIN_SYMBOL_MAP_JSON"] = '{"BTC":"XBTUSDTM"}'
    app.config["KUCOIN_CONTRACT_SPECS_JSON"] = '{"XBTUSDTM":{"contract_size":0.001,"size_step":1,"min_size":1}}'
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    monkeypatch.setattr(
        cli_module,
        "_find_live_canary_ranking_payload",
        lambda **kwargs: {"ready": True, "accepted_ranking_id": ranking.id, "blockers": [], "sweeps": []},
    )

    result = app.test_cli_runner().invoke(
        args=["live-auto-canary", "--user-id", str(user.id), "--submit", "--confirm", "LIVE-CANARY-TRADE"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted_count"] == 0
    assert payload["provider_results"][0]["submission"]["submit_blocked"] is True
    assert payload["provider_results"][1]["skipped"] is True
    assert "first_canary_submission_already_processed" in payload["provider_results"][1]["blockers"]


def test_live_auto_canary_submits_hyperliquid_then_kucoin_after_verification(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    user, hyperliquid, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    hyperliquid.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    kucoin = service.create_or_update(
        user_id=user.id,
        provider="kucoin",
        connection_type="cex_api_key",
        api_key="key",
        api_secret="secret",
        passphrase="passphrase",
        is_active=False,
    )
    kucoin.verification_status = "verified"
    kucoin.is_active = True
    app.config["KUCOIN_SYMBOL_MAP_JSON"] = '{"BTC":"XBTUSDTM"}'
    app.config["KUCOIN_CONTRACT_SPECS_JSON"] = '{"XBTUSDTM":{"contract_size":0.001,"size_step":1,"min_size":1}}'
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    monkeypatch.setattr(
        cli_module,
        "_find_live_canary_ranking_payload",
        lambda **kwargs: {"ready": True, "accepted_ranking_id": ranking.id, "blockers": [], "sweeps": []},
    )
    submitted_connections: list[int] = []

    def fake_place_order(intent):
        submitted_connections.append(intent.trading_connection_id)
        return SimpleNamespace(
            id=100 + len(submitted_connections),
            status="submitted",
            risk_status="approved",
            rejection_reason=None,
            client_order_id=intent.idempotency_key,
        )

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fake_place_order)

    result = app.test_cli_runner().invoke(
        args=["live-auto-canary", "--user-id", str(user.id), "--submit", "--confirm", "LIVE-CANARY-TRADE"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted_count"] == 1
    assert submitted_connections == [hyperliquid.id]
    assert payload["provider_results"][0]["post_submit_verified"] is True
    assert payload["provider_results"][1]["skipped"] is True
    assert "first_canary_submission_already_processed" in payload["provider_results"][1]["blockers"]


def test_live_auto_canary_rapid_submits_multiple_accepted_rankings_when_requested(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    app.config["LIVE_MICRO_CANARY_MAX_DAILY_LIVE_ORDERS"] = 2
    user, hyperliquid, ranking = _add_canary_ranking()
    second = _add_extra_canary_ranking(user, symbol="ETH", provider="hyperliquid")
    service = app.extensions["services"]["trading_connections"]
    hyperliquid.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    monkeypatch.setattr(
        cli_module,
        "_find_live_canary_ranking_payload",
        lambda **kwargs: {
            "ready": True,
            "accepted_ranking_id": ranking.id,
            "accepted_ranking_ids": [ranking.id, second.id],
            "blockers": [],
            "sweeps": [],
        },
    )
    submitted_rankings: list[str] = []

    def fake_place_order(intent):
        submitted_rankings.append(intent.idempotency_key)
        return SimpleNamespace(
            id=100 + len(submitted_rankings),
            status="submitted",
            risk_status="approved",
            rejection_reason=None,
            client_order_id=intent.idempotency_key,
        )

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fake_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "live-auto-canary",
            "--user-id",
            str(user.id),
            "--provider",
            "hyperliquid",
            "--max-submissions",
            "2",
            "--submit",
            "--confirm",
            "LIVE-CANARY-TRADE",
        ]
    )

    assert result.exit_code == 0
    payload = _cli_json_payload(result)
    assert payload["submitted_count"] == 2
    assert payload["real_submitted_count"] == 2
    assert [item["ranking_id"] for item in payload["provider_results"]] == [ranking.id, second.id]
    assert len(submitted_rankings) == 2


def test_live_auto_canary_hyperliquid_failure_stops_kucoin(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    user, hyperliquid, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    hyperliquid.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    kucoin = service.create_or_update(
        user_id=user.id,
        provider="kucoin",
        connection_type="cex_api_key",
        api_key="key",
        api_secret="secret",
        passphrase="passphrase",
        is_active=False,
    )
    kucoin.verification_status = "verified"
    kucoin.is_active = True
    app.config["KUCOIN_SYMBOL_MAP_JSON"] = '{"BTC":"XBTUSDTM"}'
    app.config["KUCOIN_CONTRACT_SPECS_JSON"] = '{"XBTUSDTM":{"contract_size":0.001,"size_step":1,"min_size":1}}'
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})
    monkeypatch.setattr(
        cli_module,
        "_find_live_canary_ranking_payload",
        lambda **kwargs: {"ready": True, "accepted_ranking_id": ranking.id, "blockers": [], "sweeps": []},
    )

    def fake_place_order(intent):
        return SimpleNamespace(
            id=321,
            status="rejected",
            risk_status="approved",
            rejection_reason="provider rejected test order",
            client_order_id=intent.idempotency_key,
        )

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fake_place_order)

    result = app.test_cli_runner().invoke(
        args=["live-auto-canary", "--user-id", str(user.id), "--submit", "--confirm", "LIVE-CANARY-TRADE"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["provider_results"][0]["submitted"] is True
    assert payload["provider_results"][0]["real_order_submitted"] is False
    assert payload["provider_results"][0]["post_submit_verified"] is False
    assert payload["provider_results"][1]["skipped"] is True
    assert "first_canary_submission_already_processed" in payload["provider_results"][1]["blockers"]


def test_live_funds_readiness_reports_both_active_providers(app, monkeypatch) -> None:
    user, hyperliquid, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    hyperliquid.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    kucoin = service.create_or_update(
        user_id=user.id,
        provider="kucoin",
        connection_type="cex_api_key",
        api_key="key",
        api_secret="secret",
        passphrase="passphrase",
        is_active=False,
    )
    kucoin.verification_status = "verified"
    kucoin.is_active = True
    hyperliquid.is_active = True
    app.config["KUCOIN_SYMBOL_MAP_JSON"] = '{"BTC":"XBTUSDTM"}'
    app.config["KUCOIN_CONTRACT_SPECS_JSON"] = '{"XBTUSDTM":{"contract_size":0.001,"size_step":1,"min_size":1}}'
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})

    result = app.test_cli_runner().invoke(
        args=[
            "live-funds-readiness",
            "--provider",
            "active",
            "--user-id",
            str(user.id),
            "--ranking-id",
            str(ranking.id),
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    providers = [item["provider"] for item in payload["providers"]]
    assert providers == ["hyperliquid", "kucoin"]
    assert payload["providers"][1]["details"]["contract_spec_available"] is True
    assert payload["canary_preview"]["connection"]["provider"] == "hyperliquid"


def test_live_funds_readiness_reports_kucoin_missing_contract_specs(app, monkeypatch) -> None:
    user, _, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    kucoin = service.create_or_update(
        user_id=user.id,
        provider="kucoin",
        connection_type="cex_api_key",
        api_key="key",
        api_secret="secret",
        passphrase="passphrase",
        is_active=False,
    )
    kucoin.verification_status = "verified"
    kucoin.is_active = True
    app.config["KUCOIN_SYMBOL_MAP_JSON"] = '{"BTC":"XBTUSDTM"}'
    app.config["KUCOIN_CONTRACT_SPECS_JSON"] = "{}"
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(cli_module, "_dependency_status", lambda: {"pytest": True, "flask": True})

    result = app.test_cli_runner().invoke(
        args=[
            "live-funds-readiness",
            "--provider",
            "kucoin",
            "--user-id",
            str(user.id),
            "--ranking-id",
            str(ranking.id),
            "--connection-id",
            str(kucoin.id),
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is False
    assert payload["providers"][0]["provider"] == "kucoin"
    assert "kucoin_contract_spec_missing" in payload["providers"][0]["blockers"]


def test_submit_live_canary_blocks_paper_mode(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    app.config["APP_MODE"] = "paper"
    app.config["ENABLE_LIVE_TRADING"] = False
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(
        app.extensions["services"]["order_manager"],
        "place_order",
        lambda intent: (_ for _ in ()).throw(AssertionError("submit-live-canary must block in paper mode")),
    )

    result = app.test_cli_runner().invoke(
        args=["submit-live-canary", "--ranking-id", str(ranking.id), "--user-id", str(user.id), "--confirm", "LIVE-CANARY-TRADE"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert "live_mode_required" in payload["blockers"]
    assert "live_trading_disabled" in payload["blockers"]
    assert AuditLog.query.filter_by(action="live_micro_canary_submit_attempt").count() == 1
    assert AuditLog.query.filter_by(action="live_micro_canary_submit_blocked").count() == 1


def test_submit_live_canary_blocks_live_submit_disabled_and_wrong_confirmation(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    app.config["LIVE_MICRO_CANARY_LIVE_SUBMIT_ENABLED"] = False
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(
        app.extensions["services"]["order_manager"],
        "place_order",
        lambda intent: (_ for _ in ()).throw(AssertionError("blocked submit-live-canary must not place an order")),
    )

    result = app.test_cli_runner().invoke(
        args=["submit-live-canary", "--ranking-id", str(ranking.id), "--user-id", str(user.id), "--confirm", "WRONG"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert "live_micro_canary_live_submit_disabled" in payload["blockers"]
    assert "live_micro_canary_exact_confirmation_required" in payload["blockers"]


def test_submit_live_canary_blocks_micro_preview_only(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    app.config["LIVE_MICRO_CANARY_PREVIEW_ONLY"] = True
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(
        app.extensions["services"]["order_manager"],
        "place_order",
        lambda intent: (_ for _ in ()).throw(AssertionError("micro preview-only must block")),
    )

    result = app.test_cli_runner().invoke(
        args=["submit-live-canary", "--ranking-id", str(ranking.id), "--user-id", str(user.id), "--confirm", "LIVE-CANARY-TRADE"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert "live_micro_canary_preview_only_enabled" in payload["blockers"]


def test_submit_live_canary_blocks_missing_stop_and_take_profit(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(app.extensions["services"]["strategy_registry"], "build", lambda name, parameters=None: _NoStopCanaryStrategy())
    monkeypatch.setattr(
        app.extensions["services"]["order_manager"],
        "place_order",
        lambda intent: (_ for _ in ()).throw(AssertionError("missing stop/take profit must block")),
    )

    result = app.test_cli_runner().invoke(
        args=["submit-live-canary", "--ranking-id", str(ranking.id), "--user-id", str(user.id), "--confirm", "LIVE-CANARY-TRADE"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert "stop_loss_required" in payload["blockers"]

    monkeypatch.setattr(app.extensions["services"]["strategy_registry"], "build", lambda name, parameters=None: _NoTakeProfitCanaryStrategy())
    result = app.test_cli_runner().invoke(
        args=["submit-live-canary", "--ranking-id", str(ranking.id), "--user-id", str(user.id), "--confirm", "LIVE-CANARY-TRADE"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert "take_profit_required" in payload["blockers"]


def test_submit_live_canary_blocks_leverage_panic_unverified_and_balance(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    user, connection, ranking = _add_canary_ranking()
    ranking.leverage = 2.0
    connection.verification_status = "pending"
    Setting.set_json("panic_lock", True)
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(
        app.extensions["services"]["order_manager"],
        "place_order",
        lambda intent: (_ for _ in ()).throw(AssertionError("submit blockers must prevent order placement")),
    )

    result = app.test_cli_runner().invoke(
        args=["submit-live-canary", "--ranking-id", str(ranking.id), "--user-id", str(user.id), "--confirm", "LIVE-CANARY-TRADE"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert "panic_lock_active" in payload["blockers"]
    assert "active_verified_live_connection_missing" in payload["blockers"]

    Setting.set_json("panic_lock", False)
    connection.verification_status = "verified"
    db.session.commit()
    result = app.test_cli_runner().invoke(
        args=["submit-live-canary", "--ranking-id", str(ranking.id), "--user-id", str(user.id), "--confirm", "LIVE-CANARY-TRADE"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "live_micro_canary_leverage_exceeds_one" in payload["blockers"]

    ranking.leverage = 1.0
    db.session.commit()
    trading_connections = app.extensions["services"]["trading_connections"]
    monkeypatch.setattr(
        trading_connections,
        "account_snapshot",
        lambda user_id, mode, connection_id=None: ClientSnapshot(
            mode,
            [{"asset": "USDC", "type": "margin", "value": 0.9, "withdrawable": 0.9}],
            [],
            [],
            [],
            [],
        ),
    )
    result = app.test_cli_runner().invoke(
        args=["submit-live-canary", "--ranking-id", str(ranking.id), "--user-id", str(user.id), "--confirm", "LIVE-CANARY-TRADE"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "live_micro_canary_insufficient_balance" in payload["blockers"]


def test_submit_live_canary_blocks_min_notional_when_not_allowed(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    app.config["LIVE_MICRO_CANARY_DEFAULT_MIN_NOTIONAL_USD"] = 10.0
    app.config["LIVE_MICRO_CANARY_ALLOW_MIN_NOTIONAL_ORDER"] = False
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(
        app.extensions["services"]["order_manager"],
        "place_order",
        lambda intent: (_ for _ in ()).throw(AssertionError("min notional above cap must block")),
    )

    result = app.test_cli_runner().invoke(
        args=["submit-live-canary", "--ranking-id", str(ranking.id), "--user-id", str(user.id), "--confirm", "LIVE-CANARY-TRADE"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert "exchange_min_notional_exceeds_micro_cap" in payload["blockers"]
    assert payload["live_micro_canary"]["min_notional_order_required"] is True
    assert payload["live_micro_canary"]["min_notional_order_used"] is False


def test_submit_live_canary_blocks_daily_limit(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    user, connection, ranking = _add_canary_ranking()
    db.session.add(
        AuditLog(
            category="orders",
            action="live_micro_canary_submit_success",
            message="prior success",
            user_id=user.id,
            trading_connection_id=connection.id,
        )
    )
    db.session.commit()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(
        app.extensions["services"]["order_manager"],
        "place_order",
        lambda intent: (_ for _ in ()).throw(AssertionError("daily limit must block")),
    )

    result = app.test_cli_runner().invoke(
        args=["submit-live-canary", "--ranking-id", str(ranking.id), "--user-id", str(user.id), "--confirm", "LIVE-CANARY-TRADE"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert "live_micro_canary_daily_order_limit_reached" in payload["blockers"]


def test_submit_live_canary_allows_min_notional_once_when_confirmed(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    app.config["LIVE_MICRO_CANARY_DEFAULT_MIN_NOTIONAL_USD"] = 10.0
    app.config["LIVE_MICRO_CANARY_ALLOW_MIN_NOTIONAL_ORDER"] = True
    app.config["LIVE_MICRO_CANARY_ORDER_BUDGET_USD"] = 10.0
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)
    service = app.extensions["services"]["trading_connections"]
    order_manager = app.extensions["services"]["order_manager"]
    calls: list[object] = []
    original_place_order = order_manager.place_order

    class AcceptedConnector:
        def place_order(self, *args, **kwargs):
            return {
                "exchange_order_id": "micro-canary-1",
                "status": "filled",
                "fill_price": 100.0,
                "client_order_id": "provider-micro-1",
            }

        def get_positions(self, mode):
            return []

    def spy_place_order(intent):
        calls.append(intent)
        return original_place_order(intent)

    monkeypatch.setattr(service, "connector_for_user", lambda user_id, connection_id=None: AcceptedConnector())
    monkeypatch.setattr(order_manager, "place_order", spy_place_order)

    result = app.test_cli_runner().invoke(
        args=["submit-live-canary", "--ranking-id", str(ranking.id), "--user-id", str(user.id), "--confirm", "LIVE-CANARY-TRADE"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is True
    assert payload["ready"] is True
    assert payload["order_id"]
    assert payload["notional_usd"] == 10.0
    assert payload["min_notional_order_used"] is True
    assert payload["estimated_loss_at_stop_usd"] <= 0.10
    assert len(calls) == 1
    order = db.session.get(Order, payload["order_id"])
    assert order is not None
    assert order.leverage == 1.0
    assert order.stop_loss is not None
    assert order.take_profit is not None
    assert AuditLog.query.filter_by(action="live_micro_canary_submit_attempt").count() == 1
    assert AuditLog.query.filter_by(action="live_micro_canary_submit_success").count() == 1


def test_live_canary_submit_requires_exact_confirmation(app) -> None:
    user, _, ranking = _add_canary_ranking()

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--submit",
            "--confirm",
            "WRONG",
        ]
    )

    assert result.exit_code != 0
    assert "LIVE-CANARY-TRADE" in result.output


def test_live_canary_blocks_missing_verified_connection(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    connection.is_active = False
    db.session.commit()
    _patch_canary_market(app, monkeypatch)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is False
    assert "active_verified_live_connection_missing" in payload["blockers"]
    assert "risk:credentials_missing" in payload["blockers"]
    assert payload["submitted"] is False


def test_live_canary_blocks_stale_encrypted_active_connection(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    service = app.extensions["services"]["trading_connections"]
    connection.encrypted_api_secret = service._encrypt("0x" + ("1" * 64))
    db.session.commit()
    app.config["TOTP_ENCRYPTION_KEY"] = Fernet.generate_key().decode("utf-8")
    _patch_canary_market(app, monkeypatch, patch_connection_health=False)

    def fail_place_order(intent):
        raise AssertionError("stale encrypted credentials must block order submission")

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
            "--submit",
            "--confirm",
            "LIVE-CANARY-TRADE",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is False
    assert payload["submitted"] is False
    assert payload["real_order_submitted"] is False
    assert "active_connection_cannot_trade" in payload["blockers"]
    assert "risk:credentials_missing" in payload["blockers"]
    health = Setting.get_json(f"connection_health:{connection.id}", {})
    assert health["can_trade"] is False
    assert "cannot be decrypted" in health["failure_reason"]


def test_live_canary_blocks_one_hour_ranking_not_accepted_for_preview(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking(profile="aggressive_1h")
    ranking.ml_explanation = {
        **ranking.ml_explanation,
        "one_hour_live_preference": {
            "one_hour_high_upside_score": 120.0,
            "accepted_for_one_hour_live_preview": False,
            "one_hour_live_blockers": ["low_expected_fill_quality"],
        },
    }
    db.session.commit()
    _patch_canary_market(app, monkeypatch)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is False
    assert "low_expected_fill_quality" in payload["blockers"]
    assert "ranking_not_accepted_for_one_hour_live_preview" in payload["blockers"]
    assert payload["ranking"]["accepted_for_one_hour_live_preview"] is False
    assert payload["real_order_submitted"] is False


def test_live_canary_preview_only_blocks_submit_even_with_confirmation(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)

    def fail_place_order(intent):
        raise AssertionError("preview-only mode must not submit an order")

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
            "--submit",
            "--confirm",
            "LIVE-CANARY-TRADE",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is True
    assert payload["submitted"] is False
    assert payload["real_order_submitted"] is False
    assert payload["preview_only"] is True
    assert payload["submit_blocked"] is True
    assert payload["submit_block_reason"] == "CANARY_PREVIEW_ONLY is true; live canary submit is disabled."
    assert "canary_preview_only_enabled" in payload["blockers"]
    assert payload["projected_order"]["symbol"] == "BTC"
    audit = AuditLog.query.filter_by(action="live_canary_preview").one()
    assert audit.details["blocked_by_preview_only"] is True
    assert audit.details["submit_requested"] is True
    assert audit.details["real_order_submitted"] is False


def test_live_canary_submit_uses_order_manager_with_live_caps(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)
    service = app.extensions["services"]["trading_connections"]
    risk_engine = app.extensions["services"]["risk_engine"]
    captured: dict[str, object] = {"orders": [], "risk_calls": 0}

    original_evaluate = risk_engine.evaluate

    def spy_evaluate(*args, **kwargs):
        captured["risk_calls"] = int(captured["risk_calls"]) + 1
        return original_evaluate(*args, **kwargs)

    class AcceptedConnector:
        def place_order(self, *args, **kwargs):
            captured["orders"].append(args)
            return {
                "exchange_order_id": "canary-exchange-1",
                "status": "filled",
                "fill_price": 100.0,
                "client_order_id": "provider-client-1",
            }

        def get_positions(self, mode):
            return [
                {
                    "symbol": "BTC",
                    "quantity": 0.01,
                    "entry_price": 100.0,
                    "mark_price": 100.0,
                    "unrealized_pnl": 0.0,
                    "leverage": 1.0,
                }
            ]

    monkeypatch.setattr(risk_engine, "evaluate", spy_evaluate)
    monkeypatch.setattr(service, "connector_for_user", lambda user_id, connection_id=None: AcceptedConnector())

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
            "--submit",
            "--confirm",
            "LIVE-CANARY-TRADE",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is True
    assert payload["real_order_submitted"] is True
    assert payload["preview_only"] is False
    assert payload["submission_ready"] is True
    assert payload["submission_attempt_audit_id"]
    order = db.session.get(Order, payload["order"]["id"])
    assert order is not None
    assert order.status == "filled"
    assert order.exchange_order_id == "canary-exchange-1"
    assert order.details["live_canary"] is True
    assert order.details["optimizer_ranking_id"] == ranking.id
    assert order.trading_connection_id == connection.id
    assert order.quantity * 100.0 <= 1.0
    assert order.leverage == 1.0
    assert order.fills[0].price == 100.0
    assert int(captured["risk_calls"]) >= 2
    assert captured["orders"]


def test_live_canary_submit_blocks_panic_lock(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)
    Setting.set_json("panic_lock", True)
    db.session.commit()

    def fail_place_order(intent):
        raise AssertionError("panic lock must block canary submission")

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
            "--submit",
            "--confirm",
            "LIVE-CANARY-TRADE",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert payload["submit_blocked"] is True
    assert "panic_lock_active" in payload["blockers"]
    assert "risk:panic_lock" in payload["blockers"]
    assert AuditLog.query.filter_by(action="live_canary_submit_attempt").count() == 1


def test_live_canary_submit_blocks_missing_stop_loss(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)

    class NoStopCanaryStrategy:
        def generate_signal(self, *, symbol: str, timeframe: str, candles: list[dict[str, Any]], position: dict[str, Any]) -> Signal:
            return Signal(
                "buy",
                "missing stop",
                timeframe,
                None,
                104.0,
                0.5,
                metadata={"signal_timestamp": candles[-1]["timestamp"] if candles else 0},
            )

    monkeypatch.setattr(app.extensions["services"]["strategy_registry"], "build", lambda name, parameters=None: NoStopCanaryStrategy())

    def fail_place_order(intent):
        raise AssertionError("missing stop loss must block canary submission")

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
            "--submit",
            "--confirm",
            "LIVE-CANARY-TRADE",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert payload["submit_blocked"] is True
    assert "stop_loss_required" in payload["blockers"]
    assert payload["submission_attempt_audit_id"]


def test_live_canary_submit_requires_accepted_ranking(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    user, connection, ranking = _add_canary_ranking()
    ranking.rejected = True
    ranking.rejection_reason = "old_rejected"
    db.session.commit()
    _patch_canary_market(app, monkeypatch)

    def fail_place_order(intent):
        raise AssertionError("rejected ranking must block canary submission")

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
            "--submit",
            "--confirm",
            "LIVE-CANARY-TRADE",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert payload["submit_blocked"] is True
    assert "ranking_rejected" in payload["blockers"]
    assert "ranking_has_rejection_reason" in payload["blockers"]
    assert "accepted_ranking_required" in payload["blockers"]


def test_live_canary_submit_requires_strict_readiness(app, monkeypatch) -> None:
    app.config["CANARY_PREVIEW_ONLY"] = False
    app.config["EXPLICIT_LIVE_CONFIRMED"] = True
    app.config["SECONDARY_CONFIRMATION"] = True
    Setting.set_json("explicit_live_confirmed", True)
    Setting.set_json("secondary_confirmation", True)
    db.session.commit()
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": False, "blockers": ["strict blocker"]})

    def fail_place_order(intent):
        raise AssertionError("strict readiness failure must block canary submission")

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
            "--submit",
            "--confirm",
            "LIVE-CANARY-TRADE",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert payload["submission_ready"] is False
    assert "strict_readiness_required" in payload["blockers"]
    assert payload["live_canary_readiness"]["strict_production_ready"] is False
    assert payload["live_canary_readiness"]["strict_production_blockers"] == ["strict blocker"]


def test_live_canary_submit_requires_live_confirmation_flags(app, monkeypatch) -> None:
    app.config["CANARY_PREVIEW_ONLY"] = False
    monkeypatch.setattr(cli_module, "_production_readiness_payload", lambda: {"ready": True, "blockers": []})
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)

    def fail_place_order(intent):
        raise AssertionError("missing confirmation flags must block canary submission")

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fail_place_order)

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
            "--submit",
            "--confirm",
            "LIVE-CANARY-TRADE",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is False
    assert "explicit_live_confirmed_required" in payload["blockers"]
    assert "secondary_confirmation_required" in payload["blockers"]
    assert "setting_explicit_live_confirmed_required" in payload["blockers"]
    assert "setting_secondary_confirmation_required" in payload["blockers"]
    assert payload["live_canary_readiness"]["confirmation_flags"]["config_explicit_live_confirmed"] is False


def test_live_canary_provider_failure_records_order_and_audits(app, monkeypatch) -> None:
    _enable_canary_submit_gates(app, monkeypatch)
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)
    service = app.extensions["services"]["trading_connections"]

    class RejectingConnector:
        def place_order(self, *args, **kwargs):
            return {
                "exchange_order_id": "reject-1",
                "status": "rejected",
                "error": "provider rejected test order",
                "client_order_id": "provider-reject-1",
            }

        def get_positions(self, mode):
            return []

    monkeypatch.setattr(service, "connector_for_user", lambda user_id, connection_id=None: RejectingConnector())

    result = app.test_cli_runner().invoke(
        args=[
            "live-canary-trade",
            "--ranking-id",
            str(ranking.id),
            "--user-id",
            str(user.id),
            "--connection-id",
            str(connection.id),
            "--submit",
            "--confirm",
            "LIVE-CANARY-TRADE",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["submitted"] is True
    assert payload["real_order_submitted"] is False
    order = Order.query.one()
    assert order.status == "rejected"
    assert order.exchange_order_id == "reject-1"
    assert order.rejection_reason == "provider rejected test order"
    assert order.details["exchange_response"]["error"] == "provider rejected test order"
    assert AuditLog.query.filter_by(action="live_canary_submit_attempt").count() == 1
    final_audit = AuditLog.query.filter_by(action="live_canary_trade").one()
    assert final_audit.details["order_status"] == "rejected"
    assert final_audit.details["real_order_submitted"] is False


def test_reset_local_state_warns_when_live_config_enabled(app) -> None:
    result = app.test_cli_runner().invoke(args=["reset-local-state", "--confirm", "FULL-LIVE-RESET"])

    assert result.exit_code == 0
    assert "Current mode is live" in result.output
    assert "WARNING: ENABLE_LIVE_TRADING is true" in result.output


def test_extreme_roi_defaults_and_scoring_stay_experimental(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], strategy_names=["scalping"], profile="extreme_roi_experimental")

    assert config.profile == "extreme_roi_experimental"
    assert config.timeframes == ["1m", "5m"]
    assert config.testing_window_hours == 1
    assert config.allow_leverage_experiment is True
    assert config.max_parameter_sets == app.config["EXTREME_ROI_MAX_PARAMETER_SETS"]

    result = optimizer._aggregate_candidate(
        "BTC",
        "1m",
        "scalping",
        {"leverage": 2.0, "risk_fraction": 0.12},
        [_aggressive_window(total_return=0.04, edge_score=18.0, trade_count=12), _aggressive_window(total_return=0.05, edge_score=22.0, trade_count=12)],
        config,
    )

    assert not result["rejected"]
    assert result["profile"] == "extreme_roi_experimental"
    assert result["experimental"] is True
    assert result["risk_label"] == "Extreme Experimental Risk"
    assert result["warning"] == EXTREME_ROI_WARNING
    assert EXTREME_ROI_WARNING in result["warnings"]
    assert result["score"] > 0


def test_extreme_roi_keeps_hard_rejections(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], strategy_names=["scalping"], profile="extreme_roi_experimental")

    low_edge = optimizer._aggregate_candidate("BTC", "1m", "scalping", {}, [_aggressive_window(edge_score=1.0, trade_count=12)], config)
    high_drawdown = optimizer._aggregate_candidate("BTC", "1m", "scalping", {}, [_aggressive_window(max_drawdown=-0.9, edge_score=20.0, trade_count=12)], config)

    assert low_edge["rejected"]
    assert low_edge["rejection_reason"] == "low_edge_after_costs"
    assert high_drawdown["rejected"]
    assert high_drawdown["rejection_reason"] == "high_drawdown"


def test_extreme_roi_raw_upside_keeps_rejected_high_return_visible(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], strategy_names=["scalping"], profile="extreme_roi_experimental")
    config.min_trade_count = 1

    result = optimizer._aggregate_candidate(
        "BTC",
        "1m",
        "scalping",
        {"leverage": 2.0},
        [_aggressive_window(total_return=10.5, max_drawdown=-0.9, edge_score=80.0, trade_count=20)],
        config,
    )

    assert result["rejected"] is True
    assert result["rejection_reason"] == "high_drawdown"
    assert result["raw_total_return_pct"] == 1050.0
    assert result["target_roi_pct"] == 1000.0
    assert result["target_roi_hit"] is True
    assert result["distance_to_target_roi_pct"] == 0.0
    assert result["raw_upside_score"] >= result["raw_total_return_pct"]
    assert result["accepted_for_one_hour_live_preview"] is False
    assert "candidate_rejected:high_drawdown" in result["one_hour_live_blockers"]
    assert "candidate_rejected:high_drawdown" in result["live_blockers"]
    assert "excessive_drawdown" in result["live_blockers"]
    assert "net_roi_v2_score" in result
    assert "expected_fill_quality" in result
    assert "churn_penalty" in result


def test_raw_upside_report_surfaces_raw_and_net_roi_leaders(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], strategy_names=["scalping"], profile="extreme_roi_experimental")
    config.min_trade_count = 1
    accepted = optimizer._aggregate_candidate(
        "BTC",
        "1m",
        "scalping",
        {},
        [_aggressive_window(total_return=0.04, edge_score=45.0, trade_count=20)],
        config,
    )
    rejected_raw = optimizer._aggregate_candidate(
        "ETH",
        "1m",
        "scalping",
        {"leverage": 2.0},
        [_aggressive_window(total_return=10.2, max_drawdown=-0.9, edge_score=80.0, trade_count=20)],
        config,
    )

    report = optimizer._raw_upside_report([accepted, rejected_raw], config)

    assert report["enabled"] is True
    assert report["research_only"] is True
    assert report["live_orders_created"] is False
    assert report["target_roi_hit"] is True
    assert report["max_return_candidate"]["symbol"] == "ETH"
    assert report["max_return_candidate"]["rejected"] is True
    assert report["top_by_raw_upside"][0]["symbol"] == "ETH"
    assert report["rejected_high_raw_upside"][0]["rejection_reason"] == "high_drawdown"
    assert "distance_to_target_roi_pct" in report["top_by_raw_upside"][0]
    assert "accepted_for_one_hour_live_preview" in report["top_by_raw_upside"][0]
    assert report["top_by_net_roi_v2"][0]["net_roi_v2_score"] >= report["top_by_net_roi_v2"][-1]["net_roi_v2_score"]
    assert report["best_preview_candidate"]["symbol"] == "BTC"


def test_aggressive_warning_payload_and_acceptance(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], strategy_names=["scalping"], profile="aggressive_1h")
    result = optimizer._aggregate_candidate(
        "BTC",
        "1m",
        "scalping",
        {},
        [_aggressive_window(hours_ago=2, trades_per_day=12.0), _aggressive_window(total_return=0.03, trades_per_day=12.0)],
        config,
    )

    assert not result["rejected"]
    assert result["profile"] == "aggressive_1h"
    assert result["experimental"] is True
    assert result["risk_label"] == "Very High Risk"
    assert result["warning"] == AGGRESSIVE_1H_WARNING
    assert AGGRESSIVE_1H_WARNING in result["warnings"]
    assert result["recent_1h_return"] == 0.03
    assert result["estimated_fees"] == 2.5
    assert round(result["edge_score"], 6) == 12.0
    assert result["one_hour_high_upside_score"] > 0
    assert result["accepted_for_one_hour_live_preview"] is True
    assert result["one_hour_live_blockers"] == []
    assert round(result["expectancy"], 6) == 1.5
    assert round(result["cost_drag_bps"], 6) == 4.5
    assert result["net_roi_score"] > 0
    assert result["raw_total_return_pct"] > 2.0
    assert result["raw_upside_score"] >= result["raw_total_return_pct"]
    assert result["target_roi_hit"] is False
    assert "strict_readiness_required" in result["live_blockers"]
    assert result["expected_fill_quality"] >= app.config["NET_ROI_MIN_FILL_QUALITY"]
    assert result["one_hour_edge_v2"] > 0
    assert result["one_hour_edge_grade"] in {"A", "B"}
    assert result["expected_execution_quality"] >= app.config["ONE_HOUR_MIN_EXECUTION_QUALITY"]
    assert result["profitability_blockers"] == []
    assert "net_roi_v2_score" in result["candidate_quality_breakdown"]
    assert "net_return" in result["net_roi_components"]
    assert result["convex_edge_score"] > 0
    assert result["mfe_mae_ratio"] == 2.0
    assert "cost_adjusted_recent_1h_return" in result
    assert "decay_penalty" in result


def test_aggressive_1h_sort_prefers_convex_edge_over_raw_return(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], strategy_names=["scalping"], profile="aggressive_1h", allocation_amount_usd=100.0)
    config.min_trade_count = 1
    market_structure = {"enabled": True, "score": 0.8, "book_depth_usd": 100_000.0, "spread_bps": 1.0}

    raw_return_leader = optimizer._aggregate_candidate(
        "BTC",
        "1m",
        "scalping",
        {},
        [_aggressive_window(total_return=0.06, edge_score=20.0, max_favorable_excursion=0.008, max_adverse_excursion=-0.004)],
        config,
        market_structure=market_structure,
    )
    convex_leader = optimizer._aggregate_candidate(
        "ETH",
        "1m",
        "scalping",
        {},
        [_aggressive_window(total_return=0.04, edge_score=70.0, max_favorable_excursion=0.016, max_adverse_excursion=-0.004)],
        config,
        market_structure=market_structure,
    )

    assert raw_return_leader["total_return"] > convex_leader["total_return"]
    assert not raw_return_leader["rejected"]
    assert not convex_leader["rejected"]
    assert convex_leader["convex_edge_score"] > raw_return_leader["convex_edge_score"]
    assert convex_leader["net_roi_v2_score"] > raw_return_leader["net_roi_v2_score"]
    ranked = sorted([raw_return_leader, convex_leader], key=lambda item: optimizer._ranking_sort_key(item, config))
    assert ranked[0]["symbol"] == "ETH"


def test_aggressive_1h_net_roi_penalizes_costly_raw_return_leader(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], strategy_names=["scalping"], profile="aggressive_1h", allocation_amount_usd=100.0)
    config.min_trade_count = 1
    low_quality = optimizer._aggregate_candidate(
        "BTC",
        "1m",
        "scalping",
        {},
        [
            _aggressive_window(
                total_return=0.05,
                max_drawdown=-0.20,
                edge_score=20.0,
                cost_drag_bps=17.0,
                max_favorable_excursion=0.009,
                max_adverse_excursion=-0.004,
                turnover_after_fees=10.0,
            )
        ],
        config,
        market_structure={"enabled": True, "score": 0.1, "book_depth_usd": 10_000.0, "spread_bps": 17.0},
    )
    cleaner_edge = optimizer._aggregate_candidate(
        "ETH",
        "1m",
        "scalping",
        {},
        [_aggressive_window(total_return=0.04, edge_score=55.0, cost_drag_bps=3.0, max_favorable_excursion=0.016, max_adverse_excursion=-0.004)],
        config,
        market_structure={"enabled": True, "score": 0.9, "book_depth_usd": 250_000.0, "spread_bps": 1.0},
    )

    assert low_quality["total_return"] > cleaner_edge["total_return"]
    assert not low_quality["rejected"]
    assert not cleaner_edge["rejected"]
    assert cleaner_edge["net_roi_score"] > low_quality["net_roi_score"]
    assert cleaner_edge["net_roi_v2_score"] > low_quality["net_roi_v2_score"]
    assert cleaner_edge["one_hour_edge_v2"] > low_quality["one_hour_edge_v2"]
    assert "excessive_churn" in low_quality["profitability_blockers"]
    assert cleaner_edge["roi_rejection_risk"] != "high"
    ranked = sorted([low_quality, cleaner_edge], key=lambda item: optimizer._ranking_sort_key(item, config))
    assert ranked[0]["symbol"] == "ETH"


def test_aggressive_1h_rejects_cost_drag_capacity_and_weak_convexity(app, monkeypatch) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], strategy_names=["scalping"], profile="aggressive_1h", allocation_amount_usd=100.0)
    config.min_trade_count = 1
    monkeypatch.setattr(optimizer, "_capacity_usd", lambda *args, **kwargs: 0.0)

    high_cost = optimizer._aggregate_candidate(
        "BTC",
        "1m",
        "scalping",
        {},
        [_aggressive_window(total_return=0.08, edge_score=60.0, cost_drag_bps=30.0)],
        config,
        market_structure={"enabled": True, "score": 0.8, "book_depth_usd": 100_000.0, "spread_bps": 1.0},
    )
    low_capacity = optimizer._aggregate_candidate(
        "ETH",
        "1m",
        "scalping",
        {},
        [_aggressive_window(total_return=0.08, edge_score=60.0)],
        config,
        market_structure={"enabled": True, "score": 0.8, "book_depth_usd": 100.0, "spread_bps": 1.0},
    )
    weak_convexity = optimizer._aggregate_candidate(
        "SOL",
        "1m",
        "scalping",
        {},
        [_aggressive_window(total_return=0.08, edge_score=60.0, max_favorable_excursion=0.002, max_adverse_excursion=-0.004)],
        config,
        market_structure={"enabled": True, "score": 0.8, "book_depth_usd": 100_000.0, "spread_bps": 1.0},
    )

    assert high_cost["rejected"]
    assert high_cost["rejection_reason"] == "cost_drag_above_threshold"
    assert low_capacity["rejected"]
    assert low_capacity["rejection_reason"] == "insufficient_liquidity_capacity"
    assert weak_convexity["rejected"]
    assert weak_convexity["rejection_reason"] == "weak_convexity"


def test_aggressive_1h_diagnostics_report_top_accepted_and_rejected(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], strategy_names=["scalping"], profile="aggressive_1h", allocation_amount_usd=100.0)
    config.min_trade_count = 1
    accepted = optimizer._aggregate_candidate(
        "BTC",
        "1m",
        "scalping",
        {},
        [_aggressive_window(total_return=0.04, edge_score=70.0, max_favorable_excursion=0.016, max_adverse_excursion=-0.004)],
        config,
        market_structure={"enabled": True, "score": 0.8, "book_depth_usd": 100_000.0, "spread_bps": 1.0},
    )
    rejected = optimizer._aggregate_candidate(
        "ETH",
        "1m",
        "scalping",
        {},
        [_aggressive_window(total_return=0.08, edge_score=60.0, cost_drag_bps=30.0)],
        config,
        market_structure={"enabled": True, "score": 0.8, "book_depth_usd": 100_000.0, "spread_bps": 1.0},
    )

    diagnostics = optimizer._one_hour_diagnostics([accepted, rejected], config)

    assert diagnostics["enabled"] is True
    assert diagnostics["primary_sort"] == "one_hour_edge_v2"
    assert diagnostics["rejection_breakdown"] == {"cost_drag_above_threshold": 1}
    assert diagnostics["top_accepted"][0]["convex_edge_score"] == accepted["convex_edge_score"]
    assert diagnostics["top_accepted"][0]["net_roi_v2_score"] == accepted["net_roi_v2_score"]
    assert diagnostics["top_accepted"][0]["one_hour_edge_v2"] == accepted["one_hour_edge_v2"]
    assert diagnostics["top_accepted"][0]["expected_execution_quality"] == accepted["expected_execution_quality"]
    assert diagnostics["top_rejected"][0]["rejection_reason"] == "cost_drag_above_threshold"


def test_aggressive_rejection_rules(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], strategy_names=["scalping"], profile="aggressive_1h")

    cases = [
        (_aggressive_window(trade_count=2), "low_trade_count"),
        (_aggressive_window(max_drawdown=-0.5), "high_drawdown"),
        (_aggressive_window(total_return=-0.01), "negative_recent_1h_return"),
        (_aggressive_window(profit_factor=0.9), "profit_factor_below_one"),
    ]

    for window, reason in cases:
        result = optimizer._aggregate_candidate("BTC", "1m", "scalping", {}, [window], config)
        assert result["rejected"]
        assert result["rejection_reason"] == reason

    no_data = optimizer._aggregate_candidate("BTC", "1m", "scalping", {}, [], config)
    assert no_data["rejected"]
    assert no_data["rejection_reason"] == "no_test_window_data"

    low_edge = optimizer._aggregate_candidate("BTC", "1m", "scalping", {}, [_aggressive_window(edge_score=1.0)], config)
    assert low_edge["rejected"]
    assert low_edge["rejection_reason"] == "low_edge_after_costs"
    assert low_edge["no_trade_reason"] == "low_edge_after_costs"

    config.min_edge_bps = 20.0
    tuned_edge = optimizer._aggregate_candidate("BTC", "1m", "scalping", {}, [_aggressive_window(edge_score=12.0)], config)
    assert tuned_edge["rejected"]
    assert tuned_edge["rejection_reason"] == "low_edge_after_costs"


def test_optimizer_auto_deploy_disabled_in_live_only_runtime(app) -> None:
    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    ranking = StrategyRanking(
        optimizer_run_id=optimizer_run.id,
        strategy_name="scalping",
        symbol="BTC",
        timeframe="1m",
        profile="aggressive_1h",
        experimental=True,
        risk_label="Very High Risk",
        score=1.0,
        rejected=False,
    )
    ranking.parameters = {"risk_fraction": 0.02}
    db.session.add(ranking)
    db.session.commit()

    class Manager:
        def __init__(self) -> None:
            self.started: list[int] = []

        def start(self, run_id: int) -> None:
            self.started.append(run_id)

    manager = Manager()
    deployed = _auto_deploy_rankings({"optimizer_run_id": optimizer_run.id}, 1, manager)

    assert deployed == 0
    assert manager.started == []
    assert StrategyRun.query.count() == 0


def test_optimizer_persists_cost_adjusted_metrics(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    optimizer_run = OptimizerRun(profile="short_term", status="running")
    db.session.add(optimizer_run)
    db.session.flush()
    config = optimizer.default_config(symbols=["BTC"], timeframes=["15m"], strategy_names=["scalping"])
    config.min_trade_count = 1
    result = optimizer._aggregate_candidate(
        "BTC",
        "15m",
        "scalping",
        {},
        [_optimizer_window(net_return_after_costs=0.018), _optimizer_window(total_return=0.025, net_return_after_costs=0.02)],
        config,
    )

    optimizer._persist_ranking(optimizer_run, result)
    ranking = StrategyRanking.query.one()

    assert not result["rejected"]
    assert result["net_return_after_costs"] > 0
    assert result["turnover_after_fees"] > 0
    assert result["window_stability"] > 0
    assert ranking.net_return_after_costs == result["net_return_after_costs"]
    assert ranking.turnover_after_fees == result["turnover_after_fees"]
    assert ranking.win_rate == result["win_rate"]
    assert ranking.ml_explanation["raw_upside"]["raw_total_return_pct"] == result["raw_total_return_pct"]
    assert ranking.ml_explanation["raw_upside"]["target_roi_pct"] == 1000.0
    assert ranking.ml_explanation["raw_upside"]["live_blockers"] == result["live_blockers"]


def test_optimizer_rejects_high_turnover_low_net_return(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], timeframes=["15m"], strategy_names=["scalping"])
    config.min_trade_count = 1

    result = optimizer._aggregate_candidate(
        "BTC",
        "15m",
        "scalping",
        {},
        [_optimizer_window(total_return=0.002, max_drawdown=-0.001, net_return_after_costs=0.001, turnover=8.0)],
        config,
    )

    assert result["rejected"]
    assert result["rejection_reason"] == "high_turnover_low_net_return"


def test_optimizer_rejects_one_window_winners(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], timeframes=["15m"], strategy_names=["scalping"])
    config.min_trade_count = 1

    result = optimizer._aggregate_candidate(
        "BTC",
        "15m",
        "scalping",
        {},
        [
            _optimizer_window(total_return=0.02),
            _optimizer_window(total_return=-0.001, max_drawdown=-0.002),
            _optimizer_window(total_return=-0.001, max_drawdown=-0.002),
        ],
        config,
    )

    assert result["rejected"]
    assert result["rejection_reason"] == "one_window_winner"


def test_ml_optimizer_fields_and_training_do_not_override_rejections(app) -> None:
    app.config["ML_RANKER_ENABLED"] = True
    app.config["ML_MIN_TRAINING_EVENTS"] = 0
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(
        symbols=["BTC"],
        timeframes=["15m"],
        strategy_names=["scalping"],
        profile="aggressive_risk_adjusted",
        lock_duration_hours=24,
    )
    config.min_trade_count = 1
    config.auto_deploy_top_n = 0

    accepted = optimizer._aggregate_candidate("BTC", "15m", "scalping", {}, [_optimizer_window()], config)
    rejected = optimizer._aggregate_candidate(
        "BTC",
        "15m",
        "scalping",
        {},
        [_optimizer_window(max_drawdown=-0.9)],
        config,
    )

    assert "ml_score" in accepted
    assert "ml_adjusted_score" in accepted
    assert accepted["ml_warmup"] is False
    assert rejected["rejected"]
    assert rejected["rejection_reason"] == "high_drawdown"


def test_optimizer_trains_ranker_after_candidate_evaluation(app) -> None:
    app.config["ML_RANKER_ENABLED"] = True
    optimizer = app.extensions["services"]["strategy_optimizer"]
    optimizer.market_data.get_candles = lambda symbol, timeframe, mode, limit: _optimizer_candles()
    config = optimizer.default_config(
        symbols=["BTC"],
        timeframes=["15m"],
        strategy_names=["scalping"],
        profile="aggressive_risk_adjusted",
        lock_duration_hours=24,
    )
    config.training_window_hours = 24
    config.testing_window_hours = 24
    config.step_hours = 24
    config.max_parameter_sets = 1
    config.min_trade_count = 1
    config.auto_deploy_top_n = 0

    optimizer.run(config)

    assert MLModelState.query.filter_by(horizon="24h").one().update_count >= 1
    assert MLTrainingEvent.query.filter_by(source="optimizer", horizon="24h").count() >= 1


def test_enhanced_ensemble_backtest_reports_baseline_and_rejects_weak_candidates(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], strategy_names=["scalping"], profile="aggressive_1h")
    config.enhanced_ensemble_enabled = True
    config.ensemble_max_legs = 4
    config.ensemble_min_sharpe = 0.5
    config.min_edge_bps = 5.0
    rankings = [
        {
            "strategy_name": "scalping",
            "symbol": "BTC",
            "timeframe": "1m",
            "score": 10.0,
            "edge_score": 22.0,
            "net_return_after_costs": 0.05,
            "sharpe_like": 0.9,
            "sortino_like": 1.1,
            "max_drawdown": -0.03,
            "win_rate": 0.62,
            "expectancy": 1.5,
            "cost_drag_bps": 4.0,
            "rejected": False,
        },
        {
            "strategy_name": "rsi_mean_reversion",
            "symbol": "ETH",
            "timeframe": "1m",
            "score": 8.0,
            "edge_score": 18.0,
            "net_return_after_costs": 0.04,
            "sharpe_like": 0.8,
            "sortino_like": 1.0,
            "max_drawdown": -0.035,
            "win_rate": 0.58,
            "expectancy": 1.2,
            "cost_drag_bps": 5.0,
            "rejected": False,
        },
        {
            "strategy_name": "volatility_breakout",
            "symbol": "SOL",
            "timeframe": "5m",
            "score": 7.0,
            "edge_score": 16.0,
            "net_return_after_costs": 0.035,
            "sharpe_like": 0.7,
            "sortino_like": 0.9,
            "max_drawdown": -0.04,
            "win_rate": 0.56,
            "expectancy": 1.1,
            "cost_drag_bps": 5.5,
            "rejected": False,
        },
        {
            "strategy_name": "ema_crossover",
            "symbol": "XRP",
            "timeframe": "5m",
            "score": 9.0,
            "edge_score": 3.0,
            "net_return_after_costs": 0.06,
            "sharpe_like": 1.1,
            "sortino_like": 1.2,
            "rejected": False,
        },
        {
            "strategy_name": "breakout",
            "symbol": "BTC",
            "timeframe": "1m",
            "score": 6.0,
            "edge_score": 14.0,
            "net_return_after_costs": 0.03,
            "sharpe_like": 0.2,
            "sortino_like": 0.3,
            "rejected": False,
        },
        {
            "strategy_name": "rule_based_signal",
            "symbol": "BTC",
            "timeframe": "1m",
            "score": 12.0,
            "edge_score": 25.0,
            "net_return_after_costs": 0.08,
            "sharpe_like": 1.4,
            "sortino_like": 1.6,
            "rejected": True,
        },
    ]

    summary = optimizer._ensemble_backtest_summary(rankings, config)

    assert summary["enabled"] is True
    assert summary["accepted"] is True
    assert {leg["strategy_name"] for leg in summary["selected_legs"]} == {
        "scalping",
        "rsi_mean_reversion",
        "volatility_breakout",
    }
    assert round(sum(leg["weight"] for leg in summary["selected_legs"]), 6) == 1.0
    assert summary["baseline_best"]["strategy_name"] == "scalping"
    assert "improvement_vs_baseline_return" in summary
    assert summary["net_return_after_costs"] > 0


def test_duration_ensemble_backtest_uses_allocator_caps_and_absolute_return(app) -> None:
    app.config["EXPERIMENTAL_DURATION_ENSEMBLE_ENABLED"] = True
    app.config["ENSEMBLE_MAX_SYMBOL_PCT"] = 0.70
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(
        symbols=["BTC"],
        strategy_names=["rsi_mean_reversion", "ema_crossover", "volatility_breakout"],
        profile="aggressive_risk_adjusted",
        lock_duration_hours=24,
        allocation_amount_usd=100.0,
    )
    config.experimental_duration_ensemble_enabled = True
    config.ensemble_max_legs = 3
    config.ensemble_min_sharpe = 0.5
    config.min_edge_bps = 5.0
    rankings = [
        {
            "strategy_name": "rsi_mean_reversion",
            "symbol": "BTC",
            "timeframe": "15m",
            "score": 2.0,
            "edge_score": 18.0,
            "net_return_after_costs": 0.09,
            "sharpe_like": 0.8,
            "sortino_like": 1.0,
            "max_drawdown": -0.03,
            "win_rate": 0.62,
            "expectancy": 1.5,
            "cost_drag_bps": 4.0,
            "trade_count": 18,
            "profit_factor": 1.5,
            "rejected": False,
        },
        {
            "strategy_name": "ema_crossover",
            "symbol": "BTC",
            "timeframe": "15m",
            "score": 20.0,
            "edge_score": 18.0,
            "net_return_after_costs": 0.04,
            "sharpe_like": 0.8,
            "sortino_like": 1.0,
            "max_drawdown": -0.03,
            "win_rate": 0.58,
            "expectancy": 1.1,
            "cost_drag_bps": 5.0,
            "trade_count": 18,
            "profit_factor": 1.4,
            "rejected": False,
        },
        {
            "strategy_name": "volatility_breakout",
            "symbol": "ETH",
            "timeframe": "1h",
            "score": 7.0,
            "edge_score": 16.0,
            "net_return_after_costs": 0.035,
            "sharpe_like": 0.7,
            "sortino_like": 0.9,
            "max_drawdown": -0.04,
            "win_rate": 0.56,
            "expectancy": 1.1,
            "cost_drag_bps": 5.5,
            "trade_count": 18,
            "profit_factor": 1.3,
            "rejected": False,
        },
        {
            "strategy_name": "breakout",
            "symbol": "SOL",
            "timeframe": "15m",
            "score": 30.0,
            "edge_score": 25.0,
            "net_return_after_costs": 0.2,
            "sharpe_like": 1.5,
            "sortino_like": 1.8,
            "trade_count": 18,
            "rejected": True,
            "rejection_reason": "one_window_winner",
        },
    ]

    summary = optimizer._duration_ensemble_backtest_summary(rankings, config)

    assert summary["enabled"] is True
    assert summary["accepted"] is True
    assert summary["duration_bucket"] == "24h"
    assert summary["baseline_single_strategy"]["strategy_name"] == "rsi_mean_reversion"
    assert summary["selected_legs"][0]["strategy_name"] == "rsi_mean_reversion"
    assert summary["allocation_conservation"]["allocated_usd"] <= 100.0 + 1e-9
    assert round(sum(leg["effective_allocation_weight"] for leg in summary["selected_legs"]), 6) == 1.0
    assert summary["overfit_rejections"]["one_window_winner"] == 1
    assert "baseline_current_basket" in summary


def test_max_return_sort_prioritizes_net_return_after_hard_rejections(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(profile="aggressive_risk_adjusted", lock_duration_hours=24)
    config.max_return_optimizer_enabled = True
    rankings = [
        {
            "strategy_name": "high_score_low_return",
            "score": 99.0,
            "net_return_after_costs": 0.03,
            "recent_performance_score": 0.02,
            "expectancy": 1.0,
            "capacity_usd": 20_000.0,
            "cost_drag_bps": 4.0,
            "rejected": False,
        },
        {
            "strategy_name": "lower_score_high_return",
            "score": 4.0,
            "net_return_after_costs": 0.08,
            "recent_performance_score": 0.01,
            "expectancy": 0.5,
            "capacity_usd": 20_000.0,
            "cost_drag_bps": 5.0,
            "rejected": False,
        },
        {
            "strategy_name": "rejected_high_return",
            "score": 200.0,
            "net_return_after_costs": 0.20,
            "rejected": True,
            "rejection_reason": "one_window_winner",
        },
    ]

    rankings.sort(key=lambda item: optimizer._ranking_sort_key(item, config))

    assert rankings[0]["strategy_name"] == "lower_score_high_return"
    assert rankings[-1]["strategy_name"] == "rejected_high_return"


def test_max_return_rejects_non_positive_net_return_after_costs(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(symbols=["BTC"], strategy_names=["ema_crossover"], profile="short_term")
    config.max_return_optimizer_enabled = True

    result = optimizer._aggregate_candidate(
        "BTC",
        "15m",
        "ema_crossover",
        {},
        [_optimizer_window(total_return=0.02, net_return_after_costs=-0.001, profit_factor=1.8)],
        config,
    )

    assert result["rejected"] is True
    assert result["rejection_reason"] == "negative_net_return_after_costs"
    assert result["profit_objective_version"] == "max_return_v3"


def test_max_return_parameter_sets_expand_by_duration_bucket(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    one_hour = optimizer.default_config(strategy_names=["scalping"], profile="aggressive_risk_adjusted", lock_duration_hours=1)
    one_hour.max_return_optimizer_enabled = True
    weekly = optimizer.default_config(strategy_names=["volatility_breakout"], profile="aggressive_risk_adjusted", lock_duration_hours=168)
    weekly.max_return_optimizer_enabled = True

    short_params = optimizer._parameter_sets("scalping", 6, one_hour)
    weekly_params = optimizer._parameter_sets("volatility_breakout", 6, weekly)

    assert any(float(item.get("stop_loss_pct", 0.0) or 0.0) <= 0.007 for item in short_params)
    assert any(float(item.get("stop_loss_pct", 0.0) or 0.0) >= 0.012 for item in weekly_params)


def test_max_return_result_payloads_include_required_summaries(app) -> None:
    optimizer = app.extensions["services"]["strategy_optimizer"]
    config = optimizer.default_config(profile="aggressive_risk_adjusted", lock_duration_hours=24)
    config.max_return_optimizer_enabled = True
    rankings = [
        {
            "strategy_name": "rsi_mean_reversion",
            "symbol": "BTC",
            "timeframe": "15m",
            "score": 4.0,
            "net_return_after_costs": 0.08,
            "max_drawdown": -0.03,
            "market_structure": {"enabled": True, "coverage": 0.8, "provider": "existing", "score": 0.6},
            "market_structure_score": 0.6,
            "rejected": False,
            "lock_duration_hours": 24,
        },
        {
            "strategy_name": "breakout",
            "symbol": "ETH",
            "timeframe": "1h",
            "score": -1.0,
            "net_return_after_costs": 0.0,
            "rejected": True,
            "rejection_reason": "low_trade_count",
            "lock_duration_hours": 24,
        },
    ]

    summary = optimizer._max_return_summary(rankings, config, {"net_return_after_costs": 0.05})
    coverage = optimizer._market_structure_feature_coverage(rankings)
    leaders = optimizer._duration_return_leaders(rankings, config)

    assert summary["enabled"] is True
    assert summary["profit_objective_version"] == "max_return_v3"
    assert summary["top_candidate"]["strategy_name"] == "rsi_mean_reversion"
    assert coverage["enabled"] is True
    assert coverage["average_coverage"] == 0.8
    assert leaders["24h"]["strategy_name"] == "rsi_mean_reversion"
