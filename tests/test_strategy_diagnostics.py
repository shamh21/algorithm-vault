from __future__ import annotations

from datetime import datetime, timedelta

from app.models import StrategyRanking, StrategyRun
from app.routes.admin import _strategy_diagnostics, _strategy_ranking_diagnostic


def test_strategy_ranking_diagnostic_is_sanitized_and_risk_adjusted() -> None:
    ranking = StrategyRanking(
        optimizer_run_id=1,
        provider="global",
        strategy_name="scalping",
        symbol="ETH",
        timeframe="1m",
        score=0.72,
        ml_adjusted_score=0.68,
        net_return_after_costs=0.031,
        max_drawdown=0.044,
        win_rate=0.58,
        sharpe_like=1.4,
        rejected=False,
    )
    ranking.ml_explanation = {
        "net_roi": {"net_roi_score": 0.61},
        "net_roi_v2": {"net_roi_v2_score": 0.64},
        "one_hour_edge_v2": {
            "expected_execution_quality": 0.82,
            "candidate_quality_breakdown": {"expected_execution_quality": 0.82},
        },
    }
    ranking.warnings = ["stale market data reduced confidence"]

    diagnostic = _strategy_ranking_diagnostic(ranking)

    labels = {factor["label"] for factor in diagnostic["factors"]}
    assert {"Score", "Net ROI", "Drawdown", "Win Rate", "Execution"} <= labels
    assert diagnostic["selection_reason"].startswith("Candidate ranked by risk-adjusted score")
    assert diagnostic["blockers"] == ["stale market data reduced confidence"]
    assert "api" not in str(diagnostic).lower()
    assert "secret" not in str(diagnostic).lower()


def test_strategy_diagnostics_only_marks_active_stale_runs() -> None:
    active = StrategyRun(
        strategy_name="active",
        symbol="BTC",
        timeframe="1h",
        status="running",
        mode="paper",
        last_heartbeat_at=datetime.utcnow() - timedelta(minutes=10),
    )
    stopped = StrategyRun(
        strategy_name="stopped",
        symbol="ETH",
        timeframe="1h",
        status="stopped",
        mode="paper",
    )

    diagnostics = _strategy_diagnostics([active, stopped], [], [])

    assert diagnostics["stale_runs"] == 1
    rows = {row["name"]: row for row in diagnostics["run_rows"]}
    assert rows["active"]["stale"] is True
    assert rows["stopped"]["stale"] is False
