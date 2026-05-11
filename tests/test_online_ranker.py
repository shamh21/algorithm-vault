from __future__ import annotations

import json
from datetime import datetime, timedelta

from app.extensions import db
from app.ml.decision_engine import FEATURE_SCHEMA_VERSION, MLDecisionEngine
from app.ml.features import ML_FEATURE_SCHEMA_VERSION, MLFeatureFactory
from app.ml.offline_ranker import OfflineRanker
from app.ml.online_ranker import OnlineRanker, extract_features, horizon_from_duration, outcome_from_result
from app.ml.signal_model import MLSignalModel, MLSignalTrainingRow
from app.models import (
    AuditLog,
    BacktestRun,
    MLOfflineModel,
    MLModelState,
    MLTrainingEvent,
    OptimizerRun,
    RiskEvent,
    Setting,
    StrategyRanking,
    VaultAllocationLeg,
    VaultCycle,
)
from app.strategies.base import Signal


def _config(**overrides):
    base = {
        "ML_RANKER_LEARNING_RATE": 0.05,
        "ML_RANKER_L2": 0.001,
        "ML_MIN_TRAINING_EVENTS": 2,
        "ML_ALLOW_LIVE_UPDATES": False,
        "ML_PREDICTION_CAP": 1.0,
        "ML_TARGET_CAP": 1.0,
        "ML_WEIGHT_CAP": 2.0,
        "ML_SCORE_WEIGHT": 0.15,
    }
    base.update(overrides)
    return base


def test_online_ranker_predictions_are_deterministic() -> None:
    ranker = OnlineRanker(
        _config(),
        state={
            "weights": {"net_return_after_costs": 0.4, "strategy_name:scalping": 0.2},
            "bias": 0.01,
            "update_count": 5,
        },
    )
    features = extract_features({"strategy_name": "scalping", "net_return_after_costs": 0.05, "lock_duration_hours": 1})

    assert ranker.predict_score(features, "1h") == ranker.predict_score(features, "1h")


def test_online_ranker_update_moves_prediction_toward_outcome() -> None:
    ranker = OnlineRanker(_config(), state={"weights": {}, "bias": 0.0, "update_count": 0})
    features = extract_features({"strategy_name": "scalping", "net_return_after_costs": 0.05, "lock_duration_hours": 1})

    before = ranker.predict_score(features, "1h")
    result = ranker.update(features, 0.5, horizon="1h", mode="paper")
    after = ranker.predict_score(features, "1h")

    assert result["updated"] is True
    assert after > before
    assert ranker.explain(features, "1h")["update_count"] == 1


def test_contextual_bandit_update_uses_decayed_state() -> None:
    ranker = OnlineRanker(
        _config(ENSEMBLE_LEARNING_DECAY=0.5),
        state={"weights": {"net_return_after_costs": 1.0}, "bias": 0.2, "update_count": 10},
    )
    features = extract_features({"strategy_name": "scalping", "net_return_after_costs": 0.04, "lock_duration_hours": 1})

    result = ranker.update_contextual_bandit(features, 0.5, horizon="1h", mode="paper", decay=0.5)
    explanation = ranker.explain(features, "bandit:1h")

    assert result["updated"] is True
    assert result["horizon"] == "bandit:1h"
    assert explanation["update_count"] == 6
    assert ranker.contextual_bandit_score(features, "1h") == ranker.predict_score(features, "bandit:1h")


def test_online_ranker_clips_features_and_predictions() -> None:
    ranker = OnlineRanker(
        _config(ML_PREDICTION_CAP=0.25),
        state={"weights": {"net_return_after_costs": 10.0}, "bias": 10.0, "update_count": 0},
    )
    features = extract_features({"net_return_after_costs": 100.0, "profit_factor": 99.0, "lock_duration_hours": 24})
    normalized = ranker.normalized_features(features)

    assert all(-1.0 <= value <= 1.0 for value in normalized.values())
    assert ranker.predict_score(features, "24h") == 0.25


def test_online_ranker_persists_and_reloads_state(app) -> None:
    app.config["ML_RANKER_ENABLED"] = True
    app.config["ML_MIN_TRAINING_EVENTS"] = 1
    features = extract_features({"strategy_name": "breakout", "net_return_after_costs": 0.04, "lock_duration_hours": 24})
    ranker = OnlineRanker(app.config)

    ranker.update(features, 0.4, horizon="24h", source="test", source_id="1", mode="paper")
    db.session.commit()
    reloaded = OnlineRanker(app.config)

    assert MLModelState.query.filter_by(horizon="24h").one().update_count == 1
    assert MLTrainingEvent.query.filter_by(horizon="24h").count() == 1
    assert reloaded.predict_score(features, "24h") == ranker.predict_score(features, "24h")


def test_online_ranker_blocks_live_updates_by_default(app) -> None:
    features = extract_features({"strategy_name": "scalping", "net_return_after_costs": 0.05, "lock_duration_hours": 1})
    result = OnlineRanker(app.config).update(features, 0.5, horizon="1h", mode="live")

    assert result["updated"] is False
    assert result["quarantined"] is True
    assert MLModelState.query.filter_by(horizon="1h").one().update_count == 0
    event = MLTrainingEvent.query.one()
    assert event.mode == "live"
    assert event.details["status"] == "quarantined"
    assert event.details["promotion_status"] == "pending"


def test_feature_helpers_bucket_horizons_and_outcomes_without_future_fields() -> None:
    features = extract_features(
        {
            "strategy_name": "ema_crossover",
            "symbol": "BTC",
            "timeframe": "15m",
            "lock_duration_hours": 48,
            "window_end": 123456,
            "future_return": 999,
            "net_return_after_costs": 0.03,
            "multi_timeframe_confluence": {
                "score": 0.72,
                "cluster_count": 4,
                "invalidation_distance_bps": 35.0,
                "volume_confirmation": True,
                "rsi_confirmation": True,
                "momentum_exhaustion": False,
                "trend_regime": "bullish",
            },
        }
    )

    assert horizon_from_duration(48) == "48h"
    assert features["horizon"] == "48h"
    assert features["market_regime"] == "bullish"
    assert features["multi_timeframe_confluence_score"] == 0.72
    assert features["multi_timeframe_cluster_count"] == 4
    assert features["mtf_volume_confirmation"] is True
    assert features["mtf_rsi_confirmation"] is True
    assert features["mtf_momentum_exhaustion"] is False


def test_extract_features_includes_market_structure_context() -> None:
    features = extract_features(
        {
            "strategy_name": "volatility_breakout",
            "lock_duration_hours": 24,
            "market_structure": {
                "enabled": True,
                "score": 0.64,
                "funding_rate": 0.0001,
                "open_interest_change_pct": 0.08,
                "book_depth_score": 0.9,
                "spread_trend_bps": 4.5,
                "liquidation_proxy": 0.2,
                "volume_impulse": 2.4,
                "volatility_regime": "tradable",
                "volatility_regime_score": 1.0,
            },
        }
    )

    assert features["market_structure_score"] == 0.64
    assert features["book_depth_score"] == 0.9
    assert features["volume_impulse"] == 2.4
    assert features["market_structure_regime"] == "tradable"
    assert features["market_structure_enabled"] is True
    assert "window_end" not in OnlineRanker(_config()).normalized_features(features)
    assert "future_return" not in OnlineRanker(_config()).normalized_features(features)
    assert outcome_from_result({"net_return_after_costs": 0.03, "profit_factor": 1.4, "trade_count": 8}) > 0


def test_ml_feature_factory_prevents_future_leakage_and_includes_baselines() -> None:
    candles = [
        {"timestamp": 1000, "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 10.0},
        {"timestamp": 2000, "open": 101.0, "high": 104.0, "low": 100.0, "close": 103.0, "volume": 11.0},
        {"timestamp": 3000, "open": 103.0, "high": 110.0, "low": 102.0, "close": 109.0, "volume": 99.0},
    ]
    signal = Signal("buy", "baseline", "1m", 99.0, 105.0, 0.25, {"confidence": 0.7})

    payload = MLFeatureFactory({}).build(
        symbol="BTC",
        timeframe="1m",
        candles=candles,
        deterministic_signal=signal,
        optimizer_context={"strategy_name": "scalping", "profit_factor": 1.3, "trade_count": 7},
        backtest_result={"net_return_after_costs": 0.04, "max_drawdown": -0.01},
        multi_timeframe={"windows": {"5m": {"trend_score": 0.8}, "1h": {"trend_score": 0.3}}, "score": 0.7},
        funding={"funding_rate": 0.0001, "interval_hours": 8},
        liquidation_context={"liquidation_buffer_pct": 0.25, "leverage": 2},
        trade_outcomes=[{"return": 0.01}, {"return": -0.005}],
        cutoff_timestamp=2000,
    )

    assert payload["ml_feature_schema_version"] == ML_FEATURE_SCHEMA_VERSION
    assert payload["latest_timestamp"] == 2000
    assert payload["candle_count"] == 2
    assert payload["deterministic_action"] == "buy"
    assert payload["deterministic_stop_loss"] == 99.0
    assert payload["strategy_name"] == "scalping"
    assert payload["backtest_net_return_after_costs"] == 0.04
    assert payload["multi_timeframe_trend_agreement"] == 1.0
    assert payload["funding_cost_bps"] == 1.0
    assert payload["liquidation_buffer_pct"] == 0.25
    assert payload["backtest_trade_outcome_count"] == 2


def test_extract_features_includes_upside_scanner_context() -> None:
    features = extract_features(
        {
            "strategy_name": "scalping",
            "lock_duration_hours": 1,
            "upside_screen_score": 12.0,
            "volume_impulse": 3.2,
            "volume_impulse_persistence": 2.1,
            "momentum_acceleration": 0.012,
            "volatility_compression": 0.4,
            "volatility_expansion": 1.8,
            "breakout_proximity_bps": 25.0,
            "spread_stability": 0.9,
            "depth_stability": 0.8,
            "stale_data_age_seconds": 45.0,
            "cost_adjusted_expected_move": 32.0,
            "liquidity_capacity_usd": 75_000.0,
            "volatility_regime": "tradable",
            "scanner_source": "dynamic_liquid",
            "cost_drag_bps": 14.0,
            "convex_edge_score": 18.0,
            "net_roi_v2_score": 42.0,
            "roi_quality_grade": "B",
            "roi_rejection_risk": "low",
            "regime_support": "regime-supported",
            "regime_bucket": {
                "volatility": "tradable",
                "spread_cost": "low_cost",
                "liquidity": "deep_liquidity",
                "trend_breakout": "breakout_supported",
            },
            "tail_loss_penalty": 0.02,
            "downside_asymmetry_penalty": 0.04,
            "cost_adjusted_breakout_potential": 38.0,
            "sustained_volume_impulse": 2.4,
            "pullback_quality": 0.8,
            "breakout_retest_success": 1.0,
            "volatility_expansion_after_compression": 0.6,
            "spread_stability_recent": 0.92,
            "cost_adjusted_expected_move_persistence": 24.0,
            "mfe_mae_ratio": 2.4,
            "rejection_reason": "spread_above_threshold",
            "high_upside_profile": True,
        }
    )
    normalized = OnlineRanker(_config()).normalized_features(features)

    assert features["upside_screen_score"] == 12.0
    assert features["liquidity_capacity"] == 75_000.0
    assert features["volatility_regime"] == "tradable"
    assert normalized["high_upside_profile"] == 1.0
    assert "scanner_source:dynamic_liquid" in normalized
    assert "roi_quality_grade:b" in normalized
    assert "regime_support:regime_supported" in normalized
    assert "spread_cost_regime:low_cost" in normalized
    assert "rejection_reason:spread_above_threshold" in normalized
    assert normalized["net_roi_v2_score"] > 0
    assert normalized["cost_adjusted_breakout_potential"] > 0
    assert normalized["volume_impulse_persistence"] > 0
    assert normalized["cost_adjusted_expected_move"] > 0


def test_live_ranker_promotion_guardrails_and_success(app) -> None:
    app.config["ML_LIVE_PROMOTION_MIN_EVENTS"] = 2
    app.config["ML_LIVE_PROMOTION_MAX_MEAN_LOSS"] = 1.0
    app.config["ML_LIVE_PROMOTION_MAX_NEGATIVE_ERROR_RATE"] = 1.0
    ranker = OnlineRanker(app.config)
    features = extract_features({"strategy_name": "scalping", "net_return_after_costs": 0.04, "lock_duration_hours": 1})

    blocked = ranker.promote_quarantined_events("1h")
    assert blocked["promoted"] is False
    assert blocked["blockers"] == ["insufficient_quarantined_events"]

    for source_id in ("a", "b"):
        ranker.update(features, 0.25, horizon="1h", source="test", source_id=source_id, mode="live")
    before = MLModelState.query.filter_by(horizon="1h").one().update_count
    promoted = ranker.promote_quarantined_events("1h")
    db.session.commit()

    assert promoted["promoted"] is True
    assert promoted["promoted_count"] == 2
    assert MLModelState.query.filter_by(horizon="1h").one().update_count == before + 2
    assert all(event.details["promotion_status"] == "promoted" for event in MLTrainingEvent.query.filter_by(mode="live").all())


class _FakeOfflineModel:
    def __init__(self, prediction: float = 0.2) -> None:
        self.prediction = prediction
        self.feature_importances_ = [1.0]

    def predict(self, rows):
        return [self.prediction for _row in rows]


class _FakeOfflineRanker(OfflineRanker):
    def _fit_model(self, model_type, train_x, train_y):
        return _FakeOfflineModel(0.2 if model_type == "sklearn" else 0.3)

    def _dump_artifact(self, payload, path):
        path.write_text("fake artifact", encoding="utf-8")

    def _load_artifact(self, path):
        return {"model": _FakeOfflineModel(0.4), "feature_names": ["net_return_after_costs"]}

    @staticmethod
    def _module_available(name):
        return True


def _ranking_row(run_id: int, index: int, net_return: float) -> StrategyRanking:
    ranking = StrategyRanking(
        optimizer_run_id=run_id,
        strategy_name="scalping",
        symbol=f"BTC{index}",
        timeframe="1m",
        profile="aggressive_1h",
        score=net_return,
        total_return=net_return,
        net_return_after_costs=net_return,
        recent_performance_score=net_return / 2,
        recent_1h_return=net_return / 3,
        max_drawdown=-0.02,
        profit_factor=1.5,
        trade_count=8,
        lock_duration_hours=1,
    )
    return ranking


def test_offline_ranker_filters_training_rows_by_provider(app) -> None:
    run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(run)
    db.session.flush()
    kucoin = _ranking_row(run.id, 1, 0.03)
    kucoin.provider = "kucoin"
    hyperliquid = _ranking_row(run.id, 2, 0.04)
    hyperliquid.provider = "hyperliquid"
    db.session.add_all([kucoin, hyperliquid])
    db.session.commit()

    ranker = OfflineRanker(app.config)
    kucoin_rows = ranker.training_rows("1h", provider="kucoin")
    hyperliquid_rows = ranker.training_rows("1h", provider="hyperliquid")
    all_rows = ranker.training_rows("1h")

    assert [row.provider for row in kucoin_rows] == ["kucoin"]
    assert [row.provider for row in hyperliquid_rows] == ["hyperliquid"]
    assert sorted(row.provider for row in all_rows) == ["hyperliquid", "kucoin"]


def test_offline_ranker_readiness_requires_promoted_model_for_provider(app, tmp_path) -> None:
    app.config["ML_OFFLINE_MODELS_ENABLED"] = True
    app.config["ML_OFFLINE_BLEND_ENABLED"] = True
    artifact = tmp_path / "kucoin-model.joblib"
    artifact.write_text("fake", encoding="utf-8")
    record = MLOfflineModel(
        model_key="offline_ranker:kucoin:1h:sklearn:test",
        provider="kucoin",
        horizon="1h",
        model_type="sklearn",
        status="promoted",
        artifact_path=str(artifact),
        training_rows=10,
        validation_rows=3,
        validation_loss=0.01,
        negative_error_rate=0.1,
        drift=0.01,
    )
    record.feature_names = ["net_return_after_costs"]
    record.metrics = {
        "calibration_error": 0.01,
        "top_decile_precision": 1.0,
        "false_positive_high_upside_rate": 0.0,
        "feature_importance": [],
    }
    db.session.add(record)
    db.session.commit()

    ranker = OfflineRanker(app.config, artifact_root=tmp_path)
    kucoin = ranker.readiness("1h", provider="kucoin")
    hyperliquid = ranker.readiness("1h", provider="hyperliquid")

    assert kucoin["ready"] is True
    assert kucoin["provider"] == "kucoin"
    assert kucoin["promoted_model"]["provider"] == "kucoin"
    assert hyperliquid["ready"] is False
    assert "promoted_model_missing" in hyperliquid["blockers"]


def test_offline_ranker_blocks_unsafe_promoted_scoring_model_type(app, tmp_path) -> None:
    app.config["ML_OFFLINE_MODELS_ENABLED"] = True
    app.config["ML_OFFLINE_BLEND_ENABLED"] = False
    app.config["ML_OFFLINE_SAFE_SCORING_MODEL_TYPES"] = ["sklearn"]
    artifact = tmp_path / "xgboost-model.joblib"
    artifact.write_text("do-not-load", encoding="utf-8")
    record = MLOfflineModel(
        model_key="offline_ranker:kucoin:1h:xgboost:test",
        provider="kucoin",
        horizon="1h",
        model_type="xgboost",
        status="promoted",
        artifact_path=str(artifact),
        training_rows=10,
        validation_rows=3,
        validation_loss=0.01,
        negative_error_rate=0.1,
        drift=0.01,
    )
    record.feature_names = ["net_return_after_costs"]
    record.metrics = {
        "calibration_error": 0.01,
        "top_decile_precision": 1.0,
        "false_positive_high_upside_rate": 0.0,
        "feature_importance": [],
    }
    db.session.add(record)
    db.session.commit()

    ranker = OfflineRanker(app.config, artifact_root=tmp_path)
    readiness = ranker.readiness("1h", provider="kucoin", require_blend=False)
    score = ranker.score_payload({"provider": "kucoin", "net_return_after_costs": 0.05}, "1h")

    assert readiness["ready"] is False
    assert "offline_model_type_not_safe_for_scoring:xgboost" in readiness["blockers"]
    assert score["status"] == "promoted_model_type_not_safe_for_scoring"
    assert score["blockers"] == ["offline_model_type_not_safe_for_scoring:xgboost"]


def test_offline_ranker_trains_candidate_models_and_promotes_with_guardrails(app, tmp_path) -> None:
    app.config["ML_OFFLINE_MODELS_ENABLED"] = True
    app.config["ML_OFFLINE_MIN_TRAINING_ROWS"] = 4
    app.config["ML_OFFLINE_MAX_NEGATIVE_ERROR_RATE"] = 1.0
    app.config["ML_OFFLINE_MAX_VALIDATION_LOSS"] = 1.0
    app.config["ML_OFFLINE_MAX_CALIBRATION_ERROR"] = 1.0
    app.config["ML_OFFLINE_MIN_TOP_DECILE_PRECISION"] = 0.0
    app.config["ML_OFFLINE_MAX_FALSE_POSITIVE_HIGH_UPSIDE_RATE"] = 1.0
    run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(run)
    db.session.flush()
    for index in range(6):
        db.session.add(_ranking_row(run.id, index, 0.02 + index * 0.002))
    db.session.commit()

    result = _FakeOfflineRanker(app.config, artifact_root=tmp_path).train("1h", model_types="both")

    assert result["trained"] is True
    assert len(result["trained_models"]) == 2
    assert MLOfflineModel.query.filter_by(horizon="1h", status="candidate").count() == 2
    assert result["trained_models"][0]["artifact_exists"] is True
    assert result["trained_models"][0]["feature_importance"]
    assert result["trained_models"][0]["feature_schema_version"] == "offline_ranker_v2"
    assert "calibration_error" in result["trained_models"][0]["metrics"]
    assert "top_decile_precision" in result["trained_models"][0]["metrics"]
    assert "false_positive_high_upside_rate" in result["trained_models"][0]["metrics"]
    suite_model = MLOfflineModel(
        model_key="ml_suite:global:1h:pytorch_roi_target:test",
        provider="global",
        horizon="1h",
        model_type="pytorch_roi_target",
        status="promoted",
        feature_schema_version="ml_decision_v1",
        training_rows=10,
        validation_rows=5,
    )
    db.session.add(suite_model)
    db.session.commit()
    model_id = result["trained_models"][0]["model_id"]
    promoted = _FakeOfflineRanker(app.config, artifact_root=tmp_path).promote("1h", model_id=model_id)
    assert promoted["promoted"] is True
    assert db.session.get(MLOfflineModel, model_id).status == "promoted"
    assert db.session.get(MLOfflineModel, suite_model.id).status == "promoted"
    assert promoted["training_rows"] == 4
    assert promoted["validation_rows"] == 2


def test_offline_ranker_only_blends_promoted_model_when_enabled(app, tmp_path) -> None:
    app.config["ML_OFFLINE_MODELS_ENABLED"] = True
    app.config["ML_OFFLINE_BLEND_ENABLED"] = False
    artifact = tmp_path / "model.joblib"
    artifact.write_text("fake", encoding="utf-8")
    record = MLOfflineModel(
        model_key="offline_ranker:1h:sklearn:test",
        horizon="1h",
        model_type="sklearn",
        status="promoted",
        artifact_path=str(artifact),
        training_rows=10,
        validation_rows=3,
        validation_loss=0.01,
        negative_error_rate=0.2,
        drift=0.01,
    )
    record.feature_names = ["net_return_after_costs"]
    record.metrics = {
        "calibration_error": 0.01,
        "top_decile_precision": 1.0,
        "false_positive_high_upside_rate": 0.0,
        "feature_importance": [],
    }
    db.session.add(record)
    db.session.commit()

    ranker = _FakeOfflineRanker(app.config, artifact_root=tmp_path)
    diagnostics = ranker.score_payload({"net_return_after_costs": 0.05, "lock_duration_hours": 1}, "1h", base_score=10.0)
    assert diagnostics["prediction"] == 0.4
    assert diagnostics["blend_applied"] is False
    assert diagnostics["blended_score"] == 10.0

    app.config["ML_OFFLINE_BLEND_ENABLED"] = True
    blended = ranker.score_payload({"net_return_after_costs": 0.05, "lock_duration_hours": 1}, "1h", base_score=10.0)
    assert blended["blend_applied"] is True
    assert blended["blended_score"] > 10.0


def test_offline_ranker_high_upside_readiness_does_not_require_blending(app, tmp_path) -> None:
    app.config["ML_OFFLINE_MODELS_ENABLED"] = True
    app.config["ML_OFFLINE_BLEND_ENABLED"] = False
    artifact = tmp_path / "model.joblib"
    artifact.write_text("fake", encoding="utf-8")
    record = MLOfflineModel(
        model_key="offline_ranker:1h:sklearn:high-upside",
        horizon="1h",
        model_type="sklearn",
        status="promoted",
        artifact_path=str(artifact),
        training_rows=10,
        validation_rows=3,
        validation_loss=0.01,
        negative_error_rate=0.2,
        drift=0.01,
    )
    record.feature_names = ["net_return_after_costs"]
    record.metrics = {
        "calibration_error": 0.01,
        "top_decile_precision": 1.0,
        "false_positive_high_upside_rate": 0.0,
        "feature_importance": [],
    }
    db.session.add(record)
    db.session.commit()

    ranker = _FakeOfflineRanker(app.config, artifact_root=tmp_path)

    ranking_readiness = ranker.readiness("1h")
    high_upside_readiness = ranker.readiness("1h", require_blend=False)

    assert ranking_readiness["ready"] is False
    assert "ML_OFFLINE_BLEND_ENABLED=false" in ranking_readiness["blockers"]
    assert high_upside_readiness["ready"] is True
    assert "ML_OFFLINE_BLEND_ENABLED=false" not in high_upside_readiness["blockers"]


def test_offline_ranker_v2_promotion_blocks_bad_calibration_metrics(app, tmp_path) -> None:
    app.config["ML_OFFLINE_MODELS_ENABLED"] = True
    artifact = tmp_path / "bad-model.joblib"
    artifact.write_text("fake", encoding="utf-8")
    record = MLOfflineModel(
        model_key="offline_ranker:1h:sklearn:bad-calibration",
        horizon="1h",
        model_type="sklearn",
        status="candidate",
        artifact_path=str(artifact),
        training_rows=25,
        validation_rows=5,
        validation_loss=0.01,
        negative_error_rate=0.2,
        drift=0.01,
    )
    record.feature_names = ["net_return_after_costs"]
    record.metrics = {
        "calibration_error": 0.4,
        "top_decile_precision": 0.2,
        "false_positive_high_upside_rate": 0.8,
    }
    db.session.add(record)
    db.session.commit()

    diagnostics = OfflineRanker(app.config, artifact_root=tmp_path).promotion_diagnostics(record)

    assert diagnostics["ready"] is False
    assert "calibration_error_above_threshold" in diagnostics["blockers"]
    assert "top_decile_precision_below_threshold" in diagnostics["blockers"]
    assert "false_positive_high_upside_rate_above_threshold" in diagnostics["blockers"]


def test_ml_signal_readiness_reports_torch_missing(app, monkeypatch) -> None:
    app.config["ML_SIGNAL_MODEL_ENABLED"] = True
    app.config["ML_SIGNAL_REQUIRE_PROMOTED"] = True
    monkeypatch.setattr(app.extensions["services"]["ml_signal_model"], "_module_available", lambda name: False if name == "torch" else True)
    readiness = app.extensions["services"]["ml_signal_model"].readiness("1h")

    assert readiness["ready"] is False
    assert "torch_missing" in readiness["blockers"]


def test_train_ml_signal_model_blocks_without_torch_and_rows(app, monkeypatch) -> None:
    app.config["ML_SIGNAL_MODEL_ENABLED"] = True
    app.config["ML_SIGNAL_MIN_TRAINING_ROWS"] = 5
    monkeypatch.setattr(app.extensions["services"]["ml_signal_model"], "_module_available", lambda name: False if name == "torch" else True)

    result = app.test_cli_runner().invoke(
        args=[
            "train-ml-signal-model",
            "--horizon",
            "1h",
            "--model",
            "pytorch_gru",
            "--confirm",
            "TRAIN-ML-SIGNAL-MODEL",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["trained"] is False
    assert "torch_missing" in payload["blockers"]
    assert "insufficient_training_rows" in payload["blockers"]


def test_ml_signal_promotion_blocks_bad_false_positive_rate(app, tmp_path) -> None:
    app.config["ML_SIGNAL_MODEL_ENABLED"] = True
    app.config["ML_SIGNAL_MAX_FALSE_POSITIVE_RATE"] = 0.10
    artifact = tmp_path / "signal.pt"
    artifact.write_text("fake artifact", encoding="utf-8")
    record = MLOfflineModel(
        model_key="ml_signal:1h:pytorch_gru:bad",
        horizon="1h",
        model_type="pytorch_gru",
        status="candidate",
        artifact_path=str(artifact),
        feature_schema_version="ml_signal_v1",
        training_rows=25,
        validation_rows=5,
        validation_loss=0.01,
        negative_error_rate=0.8,
        drift=0.0,
    )
    record.feature_names = ["net_return_after_costs"]
    record.metrics = {"false_positive_rate": 0.8}
    db.session.add(record)
    db.session.commit()

    result = MLSignalModel(app.config).promote("1h", model_id=record.id)

    assert result["promoted"] is False
    assert "signal_false_positive_rate_above_threshold" in result["blockers"]


def test_ml_signal_promotion_blocks_degenerate_hold_model(app, tmp_path) -> None:
    app.config["ML_SIGNAL_MODEL_ENABLED"] = True
    app.config["ML_SIGNAL_MIN_ACTION_RATE"] = 0.05
    artifact = tmp_path / "signal.pt"
    artifact.write_text("fake artifact", encoding="utf-8")
    record = MLOfflineModel(
        model_key="ml_signal:1h:pytorch_gru:hold_only",
        horizon="1h",
        model_type="pytorch_gru",
        status="candidate",
        artifact_path=str(artifact),
        feature_schema_version="ml_signal_v1",
        training_rows=100,
        validation_rows=20,
        validation_loss=0.01,
        negative_error_rate=0.0,
        drift=0.0,
    )
    record.feature_names = ["net_return_after_costs"]
    record.metrics = {"false_positive_rate": 0.0, "action_rate": 0.0}
    db.session.add(record)
    db.session.commit()

    result = MLSignalModel(app.config).promote("1h", model_id=record.id)

    assert result["promoted"] is False
    assert "signal_action_rate_below_threshold" in result["blockers"]


def test_ml_signal_promotion_blocks_always_action_model(app, tmp_path) -> None:
    app.config["ML_SIGNAL_MODEL_ENABLED"] = True
    app.config["ML_SIGNAL_MAX_ACTION_RATE"] = 0.95
    artifact = tmp_path / "signal.pt"
    artifact.write_text("fake artifact", encoding="utf-8")
    record = MLOfflineModel(
        model_key="ml_signal:1h:pytorch_gru:always_action",
        horizon="1h",
        model_type="pytorch_gru",
        status="candidate",
        artifact_path=str(artifact),
        feature_schema_version="ml_signal_v1",
        training_rows=100,
        validation_rows=20,
        validation_loss=0.01,
        negative_error_rate=0.0,
        drift=0.0,
    )
    record.feature_names = ["net_return_after_costs"]
    record.metrics = {"false_positive_rate": 0.0, "action_rate": 1.0}
    db.session.add(record)
    db.session.commit()

    result = MLSignalModel(app.config).promote("1h", model_id=record.id)

    assert result["promoted"] is False
    assert "signal_action_rate_above_threshold" in result["blockers"]


def test_ml_signal_promotion_requires_accuracy_above_majority_baseline(app, tmp_path) -> None:
    app.config["ML_SIGNAL_MODEL_ENABLED"] = True
    app.config["ML_SIGNAL_MAX_CLASSIFICATION_LOSS"] = 1.10
    artifact = tmp_path / "signal.pt"
    artifact.write_text("fake artifact", encoding="utf-8")
    record = MLOfflineModel(
        model_key="ml_signal:1h:pytorch_gru:below_baseline",
        horizon="1h",
        model_type="pytorch_gru",
        status="candidate",
        artifact_path=str(artifact),
        feature_schema_version="ml_signal_v1",
        training_rows=100,
        validation_rows=20,
        validation_loss=1.0,
        negative_error_rate=0.0,
        drift=0.0,
    )
    record.feature_names = ["net_return_after_costs"]
    record.metrics = {
        "accuracy": 0.35,
        "false_positive_rate": 0.0,
        "action_rate": 0.0,
        "target_distribution": {
            "rates": {"sell": 0.45, "hold": 0.10, "buy": 0.45},
        },
    }
    db.session.add(record)
    db.session.commit()

    result = MLSignalModel(app.config).promote("1h", model_id=record.id)

    assert result["promoted"] is False
    assert "signal_accuracy_below_baseline" in result["blockers"]


def test_ml_signal_promotion_blocks_metrics_below_live_confidence(app, tmp_path) -> None:
    app.config["ML_SIGNAL_MODEL_ENABLED"] = True
    app.config["ML_SIGNAL_MIN_CONFIDENCE"] = 0.60
    artifact = tmp_path / "signal.pt"
    artifact.write_text("fake artifact", encoding="utf-8")
    record = MLOfflineModel(
        model_key="ml_signal:1h:pytorch_gru:low_metric_confidence",
        horizon="1h",
        model_type="pytorch_gru",
        status="candidate",
        artifact_path=str(artifact),
        feature_schema_version="ml_signal_v1",
        training_rows=100,
        validation_rows=20,
        validation_loss=1.0,
        negative_error_rate=0.0,
        drift=0.0,
    )
    record.feature_names = ["net_return_after_costs"]
    record.metrics = {
        "accuracy": 0.50,
        "false_positive_rate": 0.0,
        "action_precision": 0.80,
        "action_rate": 0.10,
        "confidence_action_threshold": 0.50,
        "target_distribution": {"rates": {"sell": 0.40, "hold": 0.20, "buy": 0.40}},
    }
    db.session.add(record)
    db.session.commit()

    result = MLSignalModel(app.config).promote("1h", model_id=record.id)

    assert result["promoted"] is False
    assert "signal_metric_confidence_threshold_below_live" in result["blockers"]


def test_sweep_ml_signal_model_reports_ready_candidates_without_orders(app, monkeypatch, tmp_path) -> None:
    app.config["ML_SIGNAL_MODEL_ENABLED"] = True
    artifact = tmp_path / "signal.pt"
    artifact.write_text("fake artifact", encoding="utf-8")

    def fake_train(self, horizon, **kwargs):
        record = MLOfflineModel(
            model_key=f"ml_signal:{kwargs.get('provider', 'global')}:{horizon}:pytorch_gru:ready",
            provider=kwargs.get("provider", "global"),
            horizon=horizon,
            model_type="pytorch_gru",
            status="candidate",
            artifact_path=str(artifact),
            feature_schema_version="ml_signal_v1",
            training_rows=100,
            validation_rows=20,
            validation_loss=1.0,
            negative_error_rate=0.0,
            drift=0.0,
        )
        record.feature_names = ["score"]
        record.metrics = {
            "accuracy": 0.60,
            "false_positive_rate": 0.0,
            "action_precision": 0.75,
            "action_rate": 0.10,
            "action_count": 12,
            "confidence_action_threshold": self.config.get("ML_SIGNAL_MIN_CONFIDENCE", 0.60),
            "target_distribution": {"rates": {"sell": 0.40, "hold": 0.20, "buy": 0.40}},
        }
        db.session.add(record)
        db.session.commit()
        return {"trained": True, "model": {"id": record.id, "metrics": record.metrics, "validation_loss": record.validation_loss}, "blockers": []}

    monkeypatch.setattr(MLSignalModel, "train", fake_train)

    result = app.test_cli_runner().invoke(
        args=[
            "sweep-ml-signal-model",
            "--horizon",
            "1h",
            "--provider",
            "hyperliquid",
            "--thresholds",
            "0.001",
            "--epochs",
            "16",
            "--min-confidences",
            "0.6",
            "--max-runs-per-provider",
            "1",
            "--confirm",
            "SWEEP-ML-SIGNAL-MODEL",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is True
    assert payload["providers"][0]["ready_candidate_ids"]
    assert "promote-ml-signal-model" in payload["next_commands"]["promote_ready_signal_candidates"][0]


def test_ml_signal_sequence_examples_preserve_symbol_timelines(app) -> None:
    model = MLSignalModel({"ML_SIGNAL_SEQUENCE_LENGTH": 3})
    rows = [
        MLSignalTrainingRow({"price": 1.0}, 2, 0.01, datetime.utcnow(), "market", "hyperliquid", "hyperliquid:BTC:1m"),
        MLSignalTrainingRow({"price": 2.0}, 2, 0.01, datetime.utcnow(), "market", "hyperliquid", "hyperliquid:BTC:1m"),
        MLSignalTrainingRow({"price": 10.0}, 0, -0.01, datetime.utcnow(), "market", "hyperliquid", "hyperliquid:ETH:1m"),
        MLSignalTrainingRow({"price": 3.0}, 2, 0.01, datetime.utcnow(), "market", "hyperliquid", "hyperliquid:BTC:1m"),
    ]

    examples = model._sequence_examples(rows, ["price"], 3)

    assert examples[0] == [[1.0], [1.0], [1.0]]
    assert examples[1] == [[1.0], [1.0], [2.0]]
    assert examples[2] == [[10.0], [10.0], [10.0]]
    assert examples[3] == [[1.0], [2.0], [3.0]]


def test_ml_signal_directional_policy_can_sell_when_hold_is_largest(app) -> None:
    model = MLSignalModel(
        {
            "ML_SIGNAL_DIRECTIONAL_ACTION_ENABLED": True,
            "ML_SIGNAL_MIN_CONFIDENCE": 0.55,
            "ML_SIGNAL_MIN_ACTION_PROBABILITY": 0.15,
            "ML_SIGNAL_MIN_DIRECTIONAL_MARGIN": 0.05,
            "ML_SIGNAL_MAX_HOLD_PROBABILITY_FOR_ACTION": 0.80,
            "ML_SIGNAL_EXPECTED_RETURN_SCALE": 0.001,
        }
    )

    decision = model._probability_decision({"sell": 0.20, "hold": 0.70, "buy": 0.10})

    assert decision["action"] == "sell"
    assert decision["blockers"] == []
    assert decision["expected_return"] > 0
    assert decision["signed_expected_return"] < 0
    assert round(decision["directional_confidence"], 3) == 0.667


def test_ml_signal_directional_policy_blocks_weak_action_probability(app) -> None:
    model = MLSignalModel(
        {
            "ML_SIGNAL_DIRECTIONAL_ACTION_ENABLED": True,
            "ML_SIGNAL_MIN_CONFIDENCE": 0.55,
            "ML_SIGNAL_MIN_ACTION_PROBABILITY": 0.25,
            "ML_SIGNAL_MIN_DIRECTIONAL_MARGIN": 0.05,
            "ML_SIGNAL_MAX_HOLD_PROBABILITY_FOR_ACTION": 0.80,
            "ML_SIGNAL_EXPECTED_RETURN_SCALE": 0.001,
        }
    )

    decision = model._probability_decision({"sell": 0.18, "hold": 0.74, "buy": 0.08})

    assert decision["action"] == "hold"
    assert "ml_signal_action_probability_below_threshold" in decision["blockers"]


def test_ml_signal_promotion_blocks_tiny_action_sample(app, tmp_path) -> None:
    app.config["ML_SIGNAL_MODEL_ENABLED"] = True
    app.config["ML_SIGNAL_MIN_ACTION_COUNT"] = 10
    artifact = tmp_path / "signal.pt"
    artifact.write_text("fake artifact", encoding="utf-8")
    record = MLOfflineModel(
        model_key="ml_signal:1h:pytorch_gru:sparse_actions",
        horizon="1h",
        model_type="pytorch_gru",
        status="candidate",
        artifact_path=str(artifact),
        feature_schema_version="ml_signal_v1",
        training_rows=100,
        validation_rows=20,
        validation_loss=1.0,
        negative_error_rate=0.0,
        drift=0.0,
    )
    record.feature_names = ["score"]
    record.metrics = {
        "accuracy": 0.60,
        "false_positive_rate": 0.0,
        "action_precision": 0.80,
        "action_rate": 0.10,
        "action_count": 2,
        "confidence_action_threshold": 0.60,
        "target_distribution": {"rates": {"sell": 0.40, "hold": 0.20, "buy": 0.40}},
    }
    db.session.add(record)
    db.session.commit()

    result = MLSignalModel(app.config).promote("1h", model_id=record.id)

    assert result["promoted"] is False
    assert "signal_action_count_below_threshold" in result["blockers"]


def test_ml_decision_engine_readiness_reports_app_wide_family_blockers(app, monkeypatch) -> None:
    app.config["ML_ALL_AREAS_ENABLED"] = True
    monkeypatch.setattr(MLDecisionEngine, "_module_available", staticmethod(lambda name: False))

    readiness = app.extensions["services"]["ml_decision_engine"].readiness("1h", family="pytorch_allocator")

    assert readiness["ready"] is False
    assert readiness["families"]["pytorch_allocator"]["family"] == "pytorch_allocator"
    assert "pytorch_allocator:torch_missing" in readiness["blockers"]
    assert "pytorch_allocator:promoted_pytorch_allocator_missing" in readiness["blockers"]


def test_ml_decision_engine_signal_envelope_delegates_to_signal_model(app, monkeypatch) -> None:
    app.config["ML_ALL_AREAS_ENABLED"] = True
    signal_model = app.extensions["services"]["ml_signal_model"]

    def fake_score(context, horizon, **kwargs):
        return {
            "status": "promoted",
            "ready_for_live": True,
            "action": "buy",
            "confidence": 0.82,
            "expected_return": 0.012,
            "suggested_stop_loss_pct": 0.005,
            "suggested_take_profit_pct": 0.012,
            "sizing_score": 0.4,
            "blockers": [],
            "feature_schema_version": "ml_signal_v1",
            "model_id": 123,
        }

    monkeypatch.setattr(signal_model, "score_payload", fake_score)

    decision = app.extensions["services"]["ml_decision_engine"].decision(
        "pytorch_gru_signal",
        {"symbol": "BTC", "score": 4.2},
        horizon="1h",
    )

    assert decision["family"] == "pytorch_gru_signal"
    assert decision["ready"] is True
    assert decision["action"] == "buy"
    assert decision["model_id"] == 123
    assert decision["raw"]["ready_for_live"] is True


def test_ml_suite_promotion_blocks_bad_allocator_artifact(app, tmp_path) -> None:
    app.config["ML_ALL_AREAS_ENABLED"] = True
    app.config["ML_SIGNAL_MAX_VALIDATION_LOSS"] = 0.1
    artifact = tmp_path / "allocator.pt"
    artifact.write_text("fake allocator", encoding="utf-8")
    record = MLOfflineModel(
        model_key="ml_suite:1h:pytorch_allocator:bad",
        horizon="1h",
        model_type="pytorch_allocator",
        status="candidate",
        artifact_path=str(artifact),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        training_rows=100,
        validation_rows=20,
        validation_loss=0.8,
        negative_error_rate=0.1,
        drift=0.0,
    )
    record.feature_names = ["score"]
    record.metrics = {"false_positive_rate": 0.1}
    db.session.add(record)
    db.session.commit()

    result = app.test_cli_runner().invoke(
        args=[
            "promote-ml-suite",
            "--horizon",
            "1h",
            "--model-id",
            str(record.id),
            "--confirm",
            "PROMOTE-ML-SUITE",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["promoted"] is False
    assert "ml_decision_validation_loss_above_threshold" in payload["blockers"]


def test_ml_suite_promotion_uses_walk_forward_false_positive_rate(app, tmp_path) -> None:
    app.config["ML_ALL_AREAS_ENABLED"] = True
    app.config["ML_SIGNAL_MAX_VALIDATION_LOSS"] = 0.20
    app.config["ML_SIGNAL_MAX_FALSE_POSITIVE_RATE"] = 0.35
    artifact = tmp_path / "risk.pt"
    artifact.write_text("fake risk", encoding="utf-8")
    record = MLOfflineModel(
        model_key="ml_suite:1h:pytorch_risk_policy:bad_walk_forward",
        horizon="1h",
        model_type="pytorch_risk_policy",
        status="candidate",
        artifact_path=str(artifact),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        training_rows=100,
        validation_rows=20,
        validation_loss=0.01,
        negative_error_rate=0.1,
        drift=0.0,
    )
    record.feature_names = ["score"]
    record.metrics = {
        "false_positive_rate": 0.0,
        "walk_forward": {"false_positive_rate": 0.8, "prediction_source": "model"},
    }
    db.session.add(record)
    db.session.commit()

    result = app.test_cli_runner().invoke(
        args=[
            "promote-ml-suite",
            "--horizon",
            "1h",
            "--model-id",
            str(record.id),
            "--confirm",
            "PROMOTE-ML-SUITE",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["promoted"] is False
    assert "ml_decision_false_positive_rate_above_threshold" in payload["blockers"]


def test_ml_suite_promotion_ignores_legacy_heuristic_walk_forward_false_positive(app, tmp_path, monkeypatch) -> None:
    app.config["ML_ALL_AREAS_ENABLED"] = True
    app.config["ML_SIGNAL_MAX_VALIDATION_LOSS"] = 0.20
    app.config["ML_SIGNAL_MAX_FALSE_POSITIVE_RATE"] = 0.35
    monkeypatch.setattr(MLDecisionEngine, "_module_available", staticmethod(lambda name: True))
    artifact = tmp_path / "risk.pt"
    artifact.write_text("fake risk", encoding="utf-8")
    record = MLOfflineModel(
        model_key="ml_suite:1h:pytorch_risk_policy:legacy_walk_forward",
        horizon="1h",
        model_type="pytorch_risk_policy",
        status="candidate",
        artifact_path=str(artifact),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        training_rows=100,
        validation_rows=20,
        validation_loss=0.01,
        negative_error_rate=0.1,
        drift=0.0,
    )
    record.feature_names = ["score"]
    record.metrics = {
        "false_positive_rate": 0.0,
        "walk_forward": {"false_positive_rate": 0.8},
    }
    db.session.add(record)
    db.session.commit()

    payload = app.extensions["services"]["ml_decision_engine"].promotion_diagnostics(record, "pytorch_risk_policy")

    assert payload["ready"] is True
    assert "ml_decision_false_positive_rate_above_threshold" not in payload["blockers"]


def test_ml_suite_promotion_does_not_treat_signed_bias_as_false_positive(app, tmp_path, monkeypatch) -> None:
    app.config["ML_ALL_AREAS_ENABLED"] = True
    app.config["ML_SIGNAL_MAX_VALIDATION_LOSS"] = 0.20
    app.config["ML_SIGNAL_MAX_FALSE_POSITIVE_RATE"] = 0.35
    monkeypatch.setattr(MLDecisionEngine, "_module_available", staticmethod(lambda name: True))
    artifact = tmp_path / "allocator.pt"
    artifact.write_text("fake allocator", encoding="utf-8")
    record = MLOfflineModel(
        model_key="ml_suite:1h:pytorch_allocator:safe_fp_biased_errors",
        horizon="1h",
        model_type="pytorch_allocator",
        status="candidate",
        artifact_path=str(artifact),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        training_rows=100,
        validation_rows=20,
        validation_loss=0.01,
        negative_error_rate=0.9,
        drift=0.0,
    )
    record.feature_names = ["score"]
    record.metrics = {
        "false_positive_rate": 0.0,
        "walk_forward": {"false_positive_rate": 0.1},
    }
    db.session.add(record)
    db.session.commit()

    payload = app.extensions["services"]["ml_decision_engine"].promotion_diagnostics(record, "pytorch_allocator")

    assert payload["ready"] is True
    assert "ml_decision_false_positive_rate_above_threshold" not in payload["blockers"]


def test_ml_decision_engine_non_signal_training_rows_use_family_sources(app) -> None:
    run = OptimizerRun(profile="aggressive_1h", status="completed")
    db.session.add(run)
    db.session.flush()
    db.session.add(_ranking_row(run.id, 1, 0.04))
    cycle = VaultCycle(
        deposit_asset="USDC",
        deposit_amount=10.0,
        lock_duration_hours=1,
        lock_duration_seconds=3600,
        status="settled",
        started_at=datetime.utcnow() - timedelta(hours=2),
        unlocks_at=datetime.utcnow() - timedelta(hours=1),
        settled_at=datetime.utcnow(),
        starting_value_usd=10.0,
        current_estimated_value_usd=10.4,
        final_settlement_amount=10.4,
        algorithm_profile="aggressive_1h",
    )
    db.session.add(cycle)
    db.session.flush()
    leg = VaultAllocationLeg(
        vault_cycle_id=cycle.id,
        symbol="BTC",
        timeframe="5m",
        allocation_cap_usd=5.0,
        leverage=1.0,
        status="settled",
        realized_pnl_usd=0.25,
    )
    db.session.add(leg)
    audit = AuditLog(category="provider", action="provider_rate_limit", message="Hyperliquid HTTP 429 rate limit")
    db.session.add(audit)
    risk_event = RiskEvent(rule_name="provider_connection_failed", reason="provider connection failed")
    risk_event.payload = {"provider": "hyperliquid"}
    db.session.add(risk_event)
    backtest = BacktestRun(strategy_name="scalping", symbol="BTC", timeframe="1m")
    backtest.parameters = {"stop_loss_pct": 0.005, "take_profit_pct": 0.012}
    backtest.result = {
        "net_return_after_costs": 0.03,
        "total_return": 0.03,
        "profit_factor": 1.4,
        "trade_count": 6,
        "max_drawdown": -0.02,
        "trades": [
            {
                "return": 0.01,
                "fibonacci_levels": {"golden_zone": {"lower": 99.0, "upper": 100.0}},
                "entry_features": {"fibonacci_confluence": {"score": 0.7}},
            }
        ],
    }
    db.session.add(backtest)
    Setting.set_json("connection_health:hyperliquid", {"provider": "hyperliquid", "can_trade": True})
    db.session.commit()

    engine = app.extensions["services"]["ml_decision_engine"]
    universe_rows = engine.training_rows("pytorch_universe", "1h")
    allocator_rows = engine.training_rows("pytorch_allocator", "1h")
    ops_rows = engine.training_rows("pytorch_ops_anomaly", "1h")
    fib_rows = engine.training_rows("pytorch_fibonacci", "1h")
    backtest_rows = engine.training_rows("pytorch_backtest_scorer", "1h")
    optimizer_rows = engine.training_rows("pytorch_optimizer_policy", "1h")
    extreme_rows = engine.training_rows("pytorch_extreme_upside", "1h")
    risk_rows = engine.training_rows("pytorch_risk_policy", "1h", objective="extreme_roi_1h")
    exit_rows = engine.training_rows("pytorch_exit_policy", "1h", objective="extreme_roi_1h")
    cap_rows = engine.training_rows("pytorch_cap_policy", "1h", objective="extreme_roi_1h")
    execution_rows = engine.training_rows("pytorch_execution_policy", "1h", objective="extreme_roi_1h")
    roi_rows = engine.training_rows("pytorch_roi_target", "1h", objective="extreme_roi_1h")

    assert universe_rows
    assert universe_rows[0].source == "strategy_ranking"
    assert allocator_rows
    assert allocator_rows[0].source == "vault_allocation_leg"
    assert allocator_rows[0].target > 0
    assert len(ops_rows) >= 3
    assert any(row.target == 1.0 for row in ops_rows)
    assert any(row.target == 0.0 for row in ops_rows)
    assert any(row.source == "backtest_trade:fibonacci" for row in fib_rows)
    assert any(row.source == "backtest_run" for row in backtest_rows)
    assert any(row.source == "strategy_ranking:optimizer_policy" for row in optimizer_rows)
    assert any(row.source == "strategy_ranking:extreme_upside" for row in extreme_rows)
    assert any(row.source == "strategy_ranking:pytorch_risk_policy" for row in risk_rows)
    assert any(row.source == "backtest_run:pytorch_exit_policy" for row in exit_rows)
    assert any(row.source == "strategy_ranking:pytorch_cap_policy" for row in cap_rows)
    assert any(row.source == "backtest_run:pytorch_execution_policy" for row in execution_rows)
    assert any(row.source == "strategy_ranking:pytorch_roi_target" for row in roi_rows)


def test_promoted_non_signal_family_decisions_use_artifact_scores(app, tmp_path, monkeypatch) -> None:
    app.config["ML_ALL_AREAS_ENABLED"] = True
    app.config["ML_OPS_ANOMALY_ENABLED"] = True
    app.config["ML_FIBONACCI_MODEL_ENABLED"] = True
    app.config["ML_BACKTEST_SCORER_ENABLED"] = True
    app.config["ML_OPTIMIZER_POLICY_ENABLED"] = True
    app.config["ML_EXTREME_UPSIDE_MODEL_ENABLED"] = True
    app.config["ML_RISK_POLICY_ENABLED"] = True
    app.config["ML_EXIT_POLICY_ENABLED"] = True
    app.config["ML_CAP_POLICY_ENABLED"] = True
    app.config["ML_ORDER_POLICY_ENABLED"] = True
    app.config["ML_ROI_TARGET_POLICY_ENABLED"] = True
    artifact = tmp_path / "decision.pt"
    artifact.write_text("fake artifact", encoding="utf-8")
    for family in (
        "pytorch_allocator",
        "pytorch_universe",
        "pytorch_ops_anomaly",
        "pytorch_extreme_upside",
        "pytorch_fibonacci",
        "pytorch_backtest_scorer",
        "pytorch_optimizer_policy",
        "pytorch_risk_policy",
        "pytorch_exit_policy",
        "pytorch_cap_policy",
        "pytorch_execution_policy",
        "pytorch_roi_target",
    ):
        record = MLOfflineModel(
            model_key=f"ml_suite:1h:{family}:promoted",
            horizon="1h",
            model_type=family,
            status="promoted",
            artifact_path=str(artifact),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            training_rows=100,
            validation_rows=20,
            validation_loss=0.01,
            negative_error_rate=0.0,
            drift=0.0,
        )
        record.feature_names = ["score"]
        record.metrics = {"false_positive_rate": 0.0, "expected_return": -0.4}
        record.promoted_at = datetime.utcnow()
        db.session.add(record)
    db.session.commit()
    engine = app.extensions["services"]["ml_decision_engine"]
    monkeypatch.setattr(MLDecisionEngine, "_module_available", staticmethod(lambda name: True))
    monkeypatch.setattr(engine, "_load_decision_artifact", lambda path: {"feature_names": ["score"]})

    monkeypatch.setattr(engine, "_score_artifact", lambda artifact_payload, context: 0.72)
    allocator = engine.decision("pytorch_allocator", {"score": 1.0, "spread_bps": 2.0}, horizon="1h")
    universe = engine.decision("pytorch_universe", {"score": 1.0}, horizon="1h")
    fibonacci = engine.decision(
        "pytorch_fibonacci",
        {"score": 1.0, "fibonacci_confluence": {"score": 0.8, "nearest_support": 99.0}},
        horizon="1h",
    )
    backtest_score = engine.decision("pytorch_backtest_scorer", {"score": 1.0}, horizon="1h")
    monkeypatch.setattr(engine, "_score_artifact", lambda artifact_payload, context: -0.9)
    optimizer_policy = engine.decision("pytorch_optimizer_policy", {"score": -1.0}, horizon="1h")
    monkeypatch.setattr(engine, "_score_artifact", lambda artifact_payload, context: 0.91)
    ops = engine.decision("pytorch_ops_anomaly", {"latency_ms": 100.0}, horizon="1h")
    monkeypatch.setattr(engine, "_score_artifact", lambda artifact_payload, context: 0.73)
    extreme = engine.decision(
        "pytorch_extreme_upside",
        {"score": 1.0, "allocation_amount_usd": 10.0, "hard_max_notional_usdc": 10.0},
        horizon="1h",
    )
    risk_policy = engine.decision("pytorch_risk_policy", {"score": 1.0, "notional": 8.0}, horizon="1h")
    exit_policy = engine.decision(
        "pytorch_exit_policy",
        {"score": 1.0, "stop_loss_pct": 0.005, "take_profit_pct": 0.012},
        horizon="1h",
    )
    cap_policy = engine.decision(
        "pytorch_cap_policy",
        {"score": 1.0, "allocation_amount_usd": 50.0, "ml_live_hard_cap_usdc": 10.0},
        horizon="1h",
    )
    execution_policy = engine.decision(
        "pytorch_execution_policy",
        {"score": 1.0, "spread_bps": 4.0, "expected_fill_quality": 0.8},
        horizon="1h",
    )
    roi_target = engine.decision(
        "pytorch_roi_target",
        {"score": 1.0, "objective": "extreme_roi_1h", "target_roi_pct": 1000.0},
        horizon="1h",
    )

    assert allocator["ready"] is True
    assert allocator["action"] == "allocate"
    assert allocator["raw"]["sizing_score"] == 0.72
    assert universe["ready"] is True
    assert universe["action"] == "rank"
    assert universe["raw"]["prediction"] == 0.72
    assert fibonacci["ready"] is True
    assert fibonacci["action"] == "suggest"
    assert fibonacci["raw"]["suggested_stop_loss_pct"] > 0
    assert backtest_score["ready"] is True
    assert backtest_score["action"] == "score"
    assert backtest_score["raw"]["backtest_edge_prediction"] == 0.72
    assert optimizer_policy["ready"] is True
    assert optimizer_policy["action"] == "skip"
    assert optimizer_policy["raw"]["skip_reason"] == "ml_optimizer_policy_low_edge"
    assert ops["ready"] is True
    assert ops["action"] == "warn"
    assert ops["raw"]["ops_anomaly_score"] == 0.91
    assert extreme["ready"] is True
    assert extreme["action"] == "pursue"
    assert extreme["raw"]["objective"] == "extreme_upside"
    assert extreme["raw"]["target_roi_pct"] == 1000.0
    assert extreme["raw"]["suggested_notional_usdc"] <= 10.0
    assert risk_policy["action"] == "approve"
    assert risk_policy["expected_return"] == 0.73
    assert risk_policy["raw"]["risk_budget_usdc"] <= 10.0
    assert exit_policy["raw"]["suggested_stop_loss_pct"] > 0
    assert cap_policy["raw"]["suggested_notional_usdc"] <= 10.0
    assert execution_policy["raw"]["order_type_suggestion"] == "limit"
    assert roi_target["raw"]["objective"] == "extreme_roi_1h"
    assert roi_target["raw"]["target_roi_pct"] == 1000.0


def test_ml_decision_confidence_is_discounted_by_model_quality(app) -> None:
    engine = app.extensions["services"]["ml_decision_engine"]
    app.config["ML_TARGET_CAP"] = 1.0
    app.config["ML_SIGNAL_MAX_VALIDATION_LOSS"] = 0.20
    app.config["ML_OFFLINE_MAX_CALIBRATION_ERROR"] = 0.18
    clean_model = {
        "validation_loss": 0.0,
        "negative_error_rate": 0.0,
        "drift": 0.0,
        "metrics": {
            "validation_loss": 0.0,
            "false_positive_rate": 0.0,
            "drift": 0.0,
            "calibration_error": 0.0,
            "mean_absolute_error": 0.0,
        },
    }
    noisy_model = {
        "validation_loss": 0.20,
        "negative_error_rate": 0.50,
        "drift": 0.50,
        "metrics": {
            "validation_loss": 0.20,
            "false_positive_rate": 0.50,
            "drift": 0.50,
            "calibration_error": 0.18,
            "mean_absolute_error": 0.50,
            "approval_precision": 0.20,
        },
    }

    clean_confidence = engine._confidence({"score": 1.0}, clean_model, prediction=0.80)
    noisy_confidence = engine._confidence({"score": 1.0}, noisy_model, prediction=0.80)

    assert clean_confidence == 0.80
    assert noisy_confidence < clean_confidence
    assert noisy_confidence <= 0.25


def test_risk_policy_training_target_uses_approval_scale(app) -> None:
    engine = app.extensions["services"]["ml_decision_engine"]
    app.config["ML_RISK_POLICY_TARGET_RETURN_THRESHOLD"] = 0.001

    assert engine._risk_policy_training_target(0.002) == 1.0
    assert engine._risk_policy_training_target(0.001) == 1.0
    assert engine._risk_policy_training_target(0.0005) == 0.5
    assert engine._risk_policy_training_target(0.0) == 0.0
    assert engine._risk_policy_training_target(-0.001) == 0.0


def test_risk_policy_market_target_uses_risk_conditions(app) -> None:
    engine = app.extensions["services"]["ml_decision_engine"]
    app.config["ML_RISK_POLICY_MAX_RECENT_VOLATILITY"] = 0.01
    app.config["ML_RISK_POLICY_MAX_RECENT_ABS_RETURN"] = 0.03

    calm = [{"close": 100 + index * 0.02} for index in range(20)]
    jumpy = [{"close": 100.0}, {"close": 104.0}, *({"close": 104.1 + index * 0.01} for index in range(18))]

    assert engine._risk_policy_market_target(calm) == 1.0
    assert engine._risk_policy_market_target(jumpy) == 0.0


def test_risk_policy_action_uses_approval_thresholds(app) -> None:
    engine = app.extensions["services"]["ml_decision_engine"]
    app.config["ML_RISK_POLICY_APPROVE_THRESHOLD"] = 0.10
    app.config["ML_RISK_POLICY_MIN_CONFIDENCE"] = 0.10

    assert engine._action("pytorch_risk_policy", {}, 0.09, prediction=0.09) == "reject"
    assert engine._action("pytorch_risk_policy", {}, 0.10, prediction=0.10) == "approve"
    assert engine._action("pytorch_risk_policy", {}, 0.20, prediction=-0.20) == "reject"


def test_ml_readiness_cli_is_research_only_and_parseable(app) -> None:
    result = app.test_cli_runner().invoke(args=["ml-readiness", "--horizon", "1h"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["horizon"] == "1h"
    assert "pytorch_gru_signal" in payload["families"]
    assert "pytorch_allocator" in payload["families"]
    assert "pytorch_extreme_upside" in payload["families"]
    assert "pytorch_risk_policy" in payload["families"]


def test_ml_quality_report_summarizes_blocked_signal_models(app, tmp_path) -> None:
    app.config["ML_SIGNAL_MODEL_ENABLED"] = True
    app.config["ML_SIGNAL_MIN_ACTION_RATE"] = 0.05
    artifact = tmp_path / "signal.pt"
    artifact.write_text("fake artifact", encoding="utf-8")
    record = MLOfflineModel(
        model_key="ml_signal:hyperliquid:1h:pytorch_gru:hold_only",
        provider="hyperliquid",
        horizon="1h",
        model_type="pytorch_gru",
        status="promoted",
        artifact_path=str(artifact),
        feature_schema_version="ml_signal_v1",
        training_rows=100,
        validation_rows=20,
        validation_loss=0.01,
        negative_error_rate=0.0,
        drift=0.0,
        promoted_at=datetime.utcnow(),
    )
    record.feature_names = ["score"]
    record.metrics = {"false_positive_rate": 0.0, "action_rate": 0.0}
    db.session.add(record)
    db.session.commit()

    result = app.test_cli_runner().invoke(
        args=[
            "ml-quality-report",
            "--horizon",
            "1h",
            "--provider",
            "hyperliquid",
            "--model-family",
            "pytorch_gru_signal",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is False
    assert payload["families"][0]["provider"] == "hyperliquid"
    assert "signal_action_rate_below_threshold" in payload["families"][0]["blockers"]
    assert payload["families"][0]["promoted_model"]["id"] == record.id
    assert "train-ml-signal-model" in payload["families"][0]["next_commands"]["train_candidate"]

    compact = app.test_cli_runner().invoke(
        args=[
            "ml-quality-report",
            "--horizon",
            "1h",
            "--provider",
            "hyperliquid",
            "--model-family",
            "pytorch_gru_signal",
            "--compact",
        ]
    )

    assert compact.exit_code == 0
    compact_payload = json.loads(compact.output)
    assert compact_payload["ready"] is False
    assert compact_payload["providers"][0]["provider"] == "hyperliquid"
    assert compact_payload["providers"][0]["promoted_model_ids"]["pytorch_gru_signal"] == record.id
    assert "signal_action_rate_below_threshold" in compact_payload["providers"][0]["blockers"]
    assert compact_payload["next_commands"]["sample_retrain_blocked_families"]


def test_train_ml_suite_trains_non_signal_families_fail_closed_without_orders(app, monkeypatch) -> None:
    order_manager = app.extensions["services"]["order_manager"]
    app.config["ML_ALL_AREAS_ENABLED"] = True
    monkeypatch.setattr(MLDecisionEngine, "_module_available", staticmethod(lambda name: False))
    monkeypatch.setattr(order_manager, "place_order", lambda intent: (_ for _ in ()).throw(AssertionError("no orders")))

    result = app.test_cli_runner().invoke(
        args=[
            "train-ml-suite",
            "--horizon",
            "1h",
            "--model-family",
            "pytorch_allocator",
            "--confirm",
            "TRAIN-ML-SUITE",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["research_only"] is True
    assert payload["trained"] is False
    assert "pytorch_allocator:torch_missing" in payload["blockers"]
    assert "pytorch_allocator:insufficient_training_rows" in payload["blockers"]
    assert all("training_not_implemented" not in blocker for blocker in payload["blockers"])
