from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from cryptography.fernet import Fernet

from app.extensions import db
from app.models import (
    AuditLog,
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
)
from app.ml.online_ranker import extract_features
from app.strategies.base import Signal
from app.backtesting.optimizer import AGGRESSIVE_1H_WARNING, DYNAMIC_INTRADAY_WARNING, EXTREME_ROI_WARNING
from app.routes.backtests import _auto_deploy_rankings
from app.services.hyperliquid_client import ClientSnapshot


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
        monkeypatch.setattr(
            trading_connections,
            "account_snapshot",
            lambda user_id, mode, connection_id=None: ClientSnapshot(
                mode,
                [{"asset": "USDC", "type": "margin", "value": 100.0, "withdrawable": 100.0}],
                [],
                [],
                [],
                [],
            ),
        )
        monkeypatch.setattr(trading_connections, "can_trade", lambda user_id, mode, connection_id=None: mode == "live")


def _aggressive_window(
    *,
    total_return: float = 0.02,
    max_drawdown: float = -0.05,
    profit_factor: float = 1.4,
    trade_count: int = 8,
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
        "trades_per_day": 48.0,
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
                    }
                ],
                "top_by_net_roi_v2": [],
                "rejected_high_raw_upside": [],
                "best_preview_candidate": {},
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

    def fake_train(horizon, *, model_types):
        captured["train"] = {"horizon": horizon, "model_types": model_types}
        return {"trained": True, "horizon": horizon, "trained_models": [{"model_id": 7}], "blockers": []}

    def fake_promote(horizon, *, model_id):
        captured["promote"] = {"horizon": horizon, "model_id": model_id}
        return {"promoted": True, "horizon": horizon, "model_id": model_id, "blockers": []}

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
    assert captured["train"] == {"horizon": "1h", "model_types": "both"}
    assert captured["promote"] == {"horizon": "1h", "model_id": 7}
    assert json.loads(train_result.output)["trained"] is True
    assert json.loads(promote_result.output)["promoted"] is True


def test_live_canary_preview_never_submits_order(app, monkeypatch) -> None:
    user, connection, ranking = _add_canary_ranking()
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
    assert payload["projected_order"]["notional"] <= app.config["MAX_POSITION_NOTIONAL"]
    assert payload["signal_quality"]["net_roi_v2_score"] > 0
    audit = AuditLog.query.filter_by(action="live_canary_preview").one()
    assert audit.details["preview_only"] is True
    assert audit.details["real_order_submitted"] is False
    assert audit.details["projected_order"]["symbol"] == "BTC"


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
    app.config["CANARY_PREVIEW_ONLY"] = False
    user, connection, ranking = _add_canary_ranking()
    _patch_canary_market(app, monkeypatch)
    captured = {}

    def fake_place_order(intent):
        captured["intent"] = intent
        return SimpleNamespace(
            id=123,
            status="submitted",
            risk_status="approved",
            rejection_reason=None,
            client_order_id=intent.idempotency_key,
        )

    monkeypatch.setattr(app.extensions["services"]["order_manager"], "place_order", fake_place_order)

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
    assert payload["order"]["id"] == 123
    assert captured["intent"].mode == "live"
    assert captured["intent"].trading_connection_id == connection.id
    assert captured["intent"].metadata["live_canary"] is True
    assert captured["intent"].metadata["optimizer_ranking_id"] == ranking.id
    assert captured["intent"].quantity * 100.0 <= app.config["MAX_POSITION_NOTIONAL"]


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
        [_aggressive_window(hours_ago=2), _aggressive_window(total_return=0.03)],
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
    assert diagnostics["primary_sort"] == "net_roi_v2_score"
    assert diagnostics["rejection_breakdown"] == {"cost_drag_above_threshold": 1}
    assert diagnostics["top_accepted"][0]["convex_edge_score"] == accepted["convex_edge_score"]
    assert diagnostics["top_accepted"][0]["net_roi_v2_score"] == accepted["net_roi_v2_score"]
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
