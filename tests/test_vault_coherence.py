from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.auth import password_hash
from app.extensions import db
from app.models import User, VaultCycle
from app.services.vault_coherence import (
    build_horizon_forecasts,
    calculate_coherence_summary,
    get_available_horizons,
    score_horizon_strategy,
)


NOW = datetime(2026, 5, 11, tzinfo=timezone.utc)


def _feature(direction: str = "bullish", *, updated_at: datetime | None = None, volatility: float = 0.008) -> dict[str, object]:
    sign = 1 if direction == "bullish" else -1
    return {
        "close": 100.0,
        "open": 98.8 if sign > 0 else 101.2,
        "high": 101.0 if sign > 0 else 102.0,
        "low": 98.0 if sign > 0 else 99.0,
        "trend_strength": 0.10 * sign,
        "ema_trend": 4.0 * sign,
        "macd_histogram": 1.4 * sign,
        "rsi": 66.0 if sign > 0 else 34.0,
        "atr_pct": volatility,
        "volatility": volatility,
        "spread_bps": 3.0,
        "volume_spike": {"ratio": 1.5},
        "updated_at": (updated_at or NOW).isoformat(),
    }


def _features(direction: str = "bullish", *, updated_at: datetime | None = None, volatility: float = 0.008) -> dict[str, object]:
    row = _feature(direction, updated_at=updated_at, volatility=volatility)
    return {
        "one_h10_horizon_features": {
            horizon: dict(row)
            for horizon in ("1m", "5m", "15m", "1h", "4h", "1d")
        }
    }


def _summary(features: dict[str, object], *, config: dict[str, object] | None = None):
    forecasts = build_horizon_forecasts(features, config=config or {}, now=NOW)
    strategies = [score_horizon_strategy(str(row["horizon"]), row, now=NOW) for row in forecasts]
    return forecasts, strategies, calculate_coherence_summary(forecasts, strategies, now=NOW)


def test_available_horizons_include_synthetic_7d() -> None:
    assert get_available_horizons() == ["1m", "5m", "15m", "45m", "1h", "4h", "1d", "7d"]


def test_aligned_bullish_horizons_are_ready() -> None:
    forecasts, _, summary = _summary(_features("bullish"))

    assert {row["direction"] for row in forecasts} == {"bullish"}
    assert summary["overallDirection"] == "bullish"
    assert summary["automationReadiness"] == "ready"
    assert summary["coherenceScore"] >= 90


def test_aligned_bearish_horizons_are_coherent() -> None:
    _, _, summary = _summary(_features("bearish"))

    assert summary["overallDirection"] == "bearish"
    assert summary["automationReadiness"] == "ready"
    assert summary["overallConfidence"] >= 58


def test_mixed_conflicting_horizons_reduce_readiness() -> None:
    features = {
        "one_h10_horizon_features": {
            "1m": _feature("bullish"),
            "5m": _feature("bullish"),
            "15m": _feature("bullish"),
            "1h": _feature("bearish"),
            "4h": _feature("bearish"),
            "1d": _feature("bearish"),
        }
    }
    _, _, summary = _summary(features)

    assert summary["overallDirection"] == "mixed"
    assert summary["automationReadiness"] == "notReady"
    assert "4h" in summary["conflictingHorizons"]
    assert any("conflict" in note.lower() for note in summary["riskNotes"])


def test_stale_feature_rows_reduce_readiness() -> None:
    stale_time = NOW - timedelta(hours=4)
    _, _, summary = _summary(
        _features("bullish", updated_at=stale_time),
        config={"ONE_H10_FEATURE_REFRESH_SECONDS": 1800},
    )

    assert summary["automationReadiness"] == "notReady"
    assert any("stale" in note.lower() for note in summary["riskNotes"])


def test_insufficient_horizon_data_is_not_ready() -> None:
    forecasts, _, summary = _summary({})

    assert all(row["dataQuality"] == "insufficient" for row in forecasts)
    assert summary["overallDirection"] == "neutral"
    assert summary["automationReadiness"] == "notReady"


def test_high_volatility_reduces_confidence() -> None:
    low_vol_forecasts, _, low_vol_summary = _summary(_features("bullish", volatility=0.008))
    high_vol_forecasts, _, high_vol_summary = _summary(_features("bullish", volatility=0.08))

    assert high_vol_summary["overallConfidence"] < low_vol_summary["overallConfidence"]
    assert high_vol_summary["automationReadiness"] != "ready"
    assert any(row["volatilityRisk"] == "high" for row in high_vol_forecasts)
    assert any(high["confidence"] < low["confidence"] for high, low in zip(high_vol_forecasts, low_vol_forecasts))


def test_strategy_forecast_disagreement_reduces_readiness() -> None:
    forecasts = build_horizon_forecasts(_features("bullish"), now=NOW)
    strategies = [score_horizon_strategy(str(row["horizon"]), row, now=NOW) for row in forecasts]
    for row in strategies[:3]:
        row["strategyBias"] = "short"
    summary = calculate_coherence_summary(forecasts, strategies, now=NOW)

    assert summary["automationReadiness"] != "ready"
    assert any("disagreement" in note.lower() for note in summary["riskNotes"])


def test_7d_horizon_is_derived_or_insufficient_when_daily_data_is_missing() -> None:
    forecasts = build_horizon_forecasts({"one_h10_horizon_features": {"4h": _feature("bullish")}}, now=NOW)
    seven_day = next(row for row in forecasts if row["horizon"] == "7d")

    assert seven_day["derived"] is True
    assert seven_day["sourceHorizon"] == "4h"
    assert seven_day["dataQuality"] == "insufficient"


def test_vault_cycle_status_payload_adds_coherence_fields(app) -> None:
    user = User(username="coherencepayload", password_hash=password_hash("password123"))
    db.session.add(user)
    db.session.flush()
    forecasts, strategies, summary = _summary(_features("bullish"))
    cycle = VaultCycle(
        user_id=user.id,
        deposit_asset="USDC",
        deposit_amount=25.0,
        settlement_asset="USDC",
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        status="active",
        algorithm_profile="1H10",
        started_at=NOW.replace(tzinfo=None),
        unlocks_at=(NOW + timedelta(hours=1)).replace(tzinfo=None),
        starting_value_usd=25.0,
        current_estimated_value_usd=25.0,
    )
    cycle.selection_metadata = {
        "horizon_forecasts": forecasts,
        "horizon_strategy_scores": strategies,
        "coherence_summary": summary,
        "cycle_status": {"phase": "ready", "phaseLabel": "Ready"},
    }
    db.session.add(cycle)
    db.session.commit()

    payload = app.extensions["services"]["vault_cycle_reporting"].status_payload(cycle)

    assert payload["status"] == "active"
    assert payload["horizon_forecasts"]
    assert payload["horizon_strategy_scores"]
    assert payload["coherence_summary"]["overallDirection"] == "bullish"
    assert payload["cycle_status"]["phase"] == "ready"
