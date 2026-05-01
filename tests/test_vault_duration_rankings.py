from __future__ import annotations

from app.extensions import db
from app.models import MLModelState, OptimizerRun, StrategyRanking


def _patch_market(app) -> None:
    market_data = app.extensions["services"]["market_data"]
    market_data.get_candles = lambda symbol, timeframe, mode, limit: [
        {"timestamp": index, "close": 100.0 + index * 0.01, "volume": 1000.0}
        for index in range(max(limit, 40))
    ]
    market_data.get_order_book = lambda symbol, mode: {
        "levels": [[{"px": "99.95", "sz": "1000"}], [{"px": "100.05", "sz": "1000"}]]
    }


def _ranking(run_id: int, *, strategy: str, timeframe: str, duration: int, score: float) -> StrategyRanking:
    ranking = StrategyRanking(
        optimizer_run_id=run_id,
        strategy_name=strategy,
        symbol="BTC",
        timeframe=timeframe,
        profile="aggressive_risk_adjusted",
        score=score,
        net_return_after_costs=0.03,
        recent_performance_score=0.02,
        recent_1h_return=0.01,
        max_drawdown=-0.03,
        profit_factor=1.4,
        sortino_like=1.2,
        consistency=0.8,
        window_stability=0.9,
        accepted_window_ratio=0.8,
        trade_count=12,
        lock_duration_hours=duration,
        rejected=False,
    )
    ranking.parameters = {"risk_fraction": 0.02}
    return ranking


def test_vault_selector_prefers_duration_matched_aggressive_risk_adjusted_rankings(app) -> None:
    _patch_market(app)
    optimizer_run = OptimizerRun(profile="aggressive_risk_adjusted", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    db.session.add_all(
        [
            _ranking(optimizer_run.id, strategy="volatility_breakout", timeframe="1h", duration=168, score=9.0),
            _ranking(optimizer_run.id, strategy="mean_reversion", timeframe="15m", duration=24, score=3.0),
            _ranking(optimizer_run.id, strategy="breakout", timeframe="1h", duration=48, score=4.0),
        ]
    )
    db.session.commit()
    selector = app.extensions["services"]["vault_strategy_selector"]

    day = selector.select("USDC", 24, "paper", 100.0)
    two_day = selector.select("USDC", 48, "paper", 100.0)
    week = selector.select("USDC", 168, "paper", 100.0)

    assert day.strategy_name == "mean_reversion"
    assert two_day.strategy_name == "breakout"
    assert week.strategy_name == "volatility_breakout"
    assert day.metadata["optimizer_profile"] == "aggressive_risk_adjusted"


def test_vault_selector_uses_warmed_ml_score_to_nudge_ranking(app) -> None:
    app.config["ML_RANKER_ENABLED"] = True
    app.config["ML_MIN_TRAINING_EVENTS"] = 1
    app.config["ML_SCORE_WEIGHT"] = 1.0
    _patch_market(app)
    state = MLModelState(model_key="online_ranker:24h", horizon="24h", update_count=1)
    state.weights = {"strategy_name:breakout": 1.0}
    db.session.add(state)
    optimizer_run = OptimizerRun(profile="aggressive_risk_adjusted", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    db.session.add_all(
        [
            _ranking(optimizer_run.id, strategy="mean_reversion", timeframe="15m", duration=24, score=2.0),
            _ranking(optimizer_run.id, strategy="breakout", timeframe="15m", duration=24, score=1.5),
        ]
    )
    db.session.commit()

    selection = app.extensions["services"]["vault_strategy_selector"].select("USDC", 24, "paper", 100.0)

    assert selection.strategy_name == "breakout"


def _aggressive_1h_ranking(
    run_id: int,
    *,
    strategy: str,
    score: float,
    convex_edge_score: float,
    rejected: bool = False,
) -> StrategyRanking:
    ranking = StrategyRanking(
        optimizer_run_id=run_id,
        strategy_name=strategy,
        symbol="BTC",
        timeframe="1m",
        profile="aggressive_1h",
        experimental=True,
        risk_label="Very High Risk",
        score=score,
        convex_edge_score=convex_edge_score,
        cost_adjusted_recent_1h_return=0.02,
        mfe_mae_ratio=2.2,
        capacity_multiple=3.0,
        decay_penalty=0.0,
        net_return_after_costs=0.025,
        recent_1h_return=0.015,
        max_drawdown=-0.03,
        profit_factor=1.6,
        edge_score=12.0,
        cost_drag_bps=5.0,
        window_stability=0.85,
        trade_count=14,
        lock_duration_hours=1,
        rejected=rejected,
    )
    ranking.parameters = {"risk_fraction": 0.02, "stop_loss_pct": 0.003, "take_profit_pct": 0.007}
    return ranking


def test_one_hour_vault_selector_prefers_convex_edge_before_raw_score(app) -> None:
    _patch_market(app)
    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    db.session.add(
        _aggressive_1h_ranking(
            optimizer_run.id,
            strategy="scalping",
            score=0.5,
            convex_edge_score=50.0,
        )
    )
    for index in range(31):
        db.session.add(
            _aggressive_1h_ranking(
                optimizer_run.id,
                strategy=f"breakout_{index}",
                score=100.0 + index,
                convex_edge_score=1.0,
            )
        )
    db.session.commit()

    selection = app.extensions["services"]["vault_strategy_selector"].select("BTC", 1, "live", 10.0)

    assert selection.strategy_name == "scalping"
    assert selection.metadata["optimizer_profile"] == "aggressive_1h"
    assert selection.metadata["optimizer_convex_edge_score"] == 50.0


def test_one_hour_vault_selector_prefers_net_roi_before_convex_edge(app) -> None:
    _patch_market(app)
    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    convex_only = _aggressive_1h_ranking(
        optimizer_run.id,
        strategy="scalping",
        score=50.0,
        convex_edge_score=80.0,
    )
    convex_only.ml_explanation = {
        "net_roi": {
            "net_roi_score": 10.0,
            "expected_fill_quality": 0.9,
            "churn_penalty": 0.1,
            "edge_after_cost_bps": 12.0,
        }
    }
    net_roi_leader = _aggressive_1h_ranking(
        optimizer_run.id,
        strategy="breakout",
        score=25.0,
        convex_edge_score=40.0,
    )
    net_roi_leader.ml_explanation = {
        "net_roi": {
            "net_roi_score": 120.0,
            "expected_fill_quality": 0.95,
            "churn_penalty": 0.0,
            "edge_after_cost_bps": 24.0,
        }
    }
    db.session.add_all([convex_only, net_roi_leader])
    db.session.commit()

    selection = app.extensions["services"]["vault_strategy_selector"].select("BTC", 1, "live", 10.0)

    assert selection.strategy_name == "breakout"
    assert selection.metadata["optimizer_net_roi_score"] == 120.0
    assert selection.legs[0]["net_roi_score"] == 120.0


def test_one_hour_vault_selector_prefers_net_roi_v2_before_legacy_score(app) -> None:
    _patch_market(app)
    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    legacy_leader = _aggressive_1h_ranking(
        optimizer_run.id,
        strategy="scalping",
        score=80.0,
        convex_edge_score=80.0,
    )
    legacy_leader.ml_explanation = {
        "net_roi": {
            "net_roi_score": 150.0,
            "expected_fill_quality": 0.9,
            "churn_penalty": 0.1,
            "edge_after_cost_bps": 18.0,
        },
        "net_roi_v2": {
            "net_roi_v2_score": 15.0,
            "roi_quality_grade": "C",
            "roi_rejection_risk": "medium",
            "regime_support": "regime-neutral",
        },
    }
    v2_leader = _aggressive_1h_ranking(
        optimizer_run.id,
        strategy="breakout",
        score=25.0,
        convex_edge_score=20.0,
    )
    v2_leader.ml_explanation = {
        "net_roi": {
            "net_roi_score": 70.0,
            "expected_fill_quality": 0.95,
            "churn_penalty": 0.0,
            "edge_after_cost_bps": 26.0,
        },
        "net_roi_v2": {
            "net_roi_v2_score": 90.0,
            "roi_quality_grade": "A",
            "roi_rejection_risk": "low",
            "regime_support": "regime-supported",
            "tail_loss_penalty": 0.0,
            "downside_asymmetry_penalty": 0.0,
            "cost_adjusted_breakout_potential": 28.0,
        },
    }
    db.session.add_all([legacy_leader, v2_leader])
    db.session.commit()

    selection = app.extensions["services"]["vault_strategy_selector"].select("BTC", 1, "live", 10.0)

    assert selection.strategy_name == "breakout"
    assert selection.metadata["optimizer_net_roi_v2_score"] == 90.0
    assert selection.metadata["optimizer_roi_quality_grade"] == "A"
    assert selection.legs[0]["net_roi_v2_score"] == 90.0
    assert selection.legs[0]["regime_support"] == "regime-supported"


def test_one_hour_vault_selector_prefers_one_hour_edge_v2_before_raw_score(app) -> None:
    _patch_market(app)
    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    raw_score_leader = _aggressive_1h_ranking(
        optimizer_run.id,
        strategy="scalping",
        score=95.0,
        convex_edge_score=80.0,
    )
    raw_score_leader.ml_explanation = {
        "net_roi_v2": {
            "net_roi_v2_score": 70.0,
            "roi_quality_grade": "B",
            "roi_rejection_risk": "low",
            "regime_support": "regime-supported",
        },
        "one_hour_edge_v2": {
            "one_hour_edge_v2": 35.0,
            "one_hour_edge_grade": "B",
            "expected_execution_quality": 0.72,
            "profitability_blockers": [],
            "raw_vs_net_roi_gap": 5.0,
            "candidate_quality_breakdown": {"raw_upside_score": 120.0},
        },
    }
    edge_leader = _aggressive_1h_ranking(
        optimizer_run.id,
        strategy="volatility_breakout",
        score=45.0,
        convex_edge_score=35.0,
    )
    edge_leader.ml_explanation = {
        "net_roi_v2": {
            "net_roi_v2_score": 55.0,
            "roi_quality_grade": "B",
            "roi_rejection_risk": "low",
            "regime_support": "regime-supported",
        },
        "one_hour_edge_v2": {
            "one_hour_edge_v2": 115.0,
            "one_hour_edge_grade": "A",
            "expected_execution_quality": 0.94,
            "profitability_blockers": [],
            "raw_vs_net_roi_gap": 3.0,
            "candidate_quality_breakdown": {"net_roi_v2_score": 55.0},
        },
    }
    db.session.add_all([raw_score_leader, edge_leader])
    db.session.commit()

    selection = app.extensions["services"]["vault_strategy_selector"].select("BTC", 1, "live", 10.0)

    assert selection.strategy_name == "volatility_breakout"
    assert selection.metadata["optimizer_one_hour_edge_v2"] == 115.0
    assert selection.metadata["optimizer_one_hour_edge_grade"] == "A"
    assert selection.legs[0]["one_hour_edge_v2"] == 115.0


def test_one_hour_vault_selector_prefers_live_preview_high_upside_score(app) -> None:
    _patch_market(app)
    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    v2_leader = _aggressive_1h_ranking(
        optimizer_run.id,
        strategy="scalping",
        score=90.0,
        convex_edge_score=80.0,
    )
    v2_leader.ml_explanation = {
        "net_roi": {
            "net_roi_score": 120.0,
            "expected_fill_quality": 0.9,
            "churn_penalty": 0.1,
            "edge_after_cost_bps": 18.0,
        },
        "net_roi_v2": {
            "net_roi_v2_score": 90.0,
            "roi_quality_grade": "A",
            "roi_rejection_risk": "low",
            "regime_support": "regime-supported",
        },
        "one_hour_live_preference": {
            "one_hour_high_upside_score": 110.0,
            "one_hour_live_preference_rank": 110.0,
            "accepted_for_one_hour_live_preview": True,
            "one_hour_live_blockers": [],
        },
    }
    high_upside_leader = _aggressive_1h_ranking(
        optimizer_run.id,
        strategy="breakout",
        score=40.0,
        convex_edge_score=35.0,
    )
    high_upside_leader.ml_explanation = {
        "net_roi": {
            "net_roi_score": 80.0,
            "expected_fill_quality": 0.95,
            "churn_penalty": 0.05,
            "edge_after_cost_bps": 24.0,
        },
        "net_roi_v2": {
            "net_roi_v2_score": 50.0,
            "roi_quality_grade": "B",
            "roi_rejection_risk": "low",
            "regime_support": "regime-supported",
        },
        "one_hour_live_preference": {
            "one_hour_high_upside_score": 180.0,
            "one_hour_live_preference_rank": 180.0,
            "accepted_for_one_hour_live_preview": True,
            "one_hour_live_blockers": [],
        },
    }
    db.session.add_all([v2_leader, high_upside_leader])
    db.session.commit()

    selection = app.extensions["services"]["vault_strategy_selector"].select("BTC", 1, "live", 10.0)

    assert selection.strategy_name == "breakout"
    assert selection.metadata["optimizer_ranking_id"] == high_upside_leader.id


def test_one_hour_vault_selector_skips_ranking_blocked_for_live_preview(app) -> None:
    _patch_market(app)
    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    blocked = _aggressive_1h_ranking(
        optimizer_run.id,
        strategy="scalping",
        score=95.0,
        convex_edge_score=95.0,
    )
    blocked.ml_explanation = {
        "net_roi": {
            "net_roi_score": 140.0,
            "expected_fill_quality": 0.2,
            "churn_penalty": 0.1,
            "edge_after_cost_bps": 30.0,
        },
        "net_roi_v2": {
            "net_roi_v2_score": 120.0,
            "roi_quality_grade": "A",
            "roi_rejection_risk": "low",
            "regime_support": "regime-supported",
        },
        "one_hour_live_preference": {
            "one_hour_high_upside_score": 300.0,
            "accepted_for_one_hour_live_preview": False,
            "one_hour_live_blockers": ["low_expected_fill_quality"],
        },
    }
    db.session.add(blocked)
    db.session.commit()

    selection = app.extensions["services"]["vault_strategy_selector"].select("BTC", 1, "live", 10.0)

    assert selection.metadata["optimizer_ranking_id"] is None
    assert selection.strategy_name == "scalping"


def test_one_hour_vault_selector_fallback_stays_live_safe_without_accepted_candidate(app) -> None:
    _patch_market(app)
    optimizer_run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(optimizer_run)
    db.session.flush()
    db.session.add(
        _aggressive_1h_ranking(
            optimizer_run.id,
            strategy="scalping",
            score=25.0,
            convex_edge_score=40.0,
            rejected=True,
        )
    )
    db.session.commit()

    selection = app.extensions["services"]["vault_strategy_selector"].select("BTC", 1, "live", 10.0)

    assert selection.mode == "live"
    assert selection.execution_mode == "live"
    assert selection.metadata["optimizer_ranking_id"] is None
    assert selection.metadata["optimizer_profile"] == "aggressive_1h"
    assert selection.strategy_name == "scalping"
