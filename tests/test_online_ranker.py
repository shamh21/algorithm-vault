from __future__ import annotations

from app.extensions import db
from app.ml.offline_ranker import OfflineRanker
from app.ml.online_ranker import OnlineRanker, extract_features, horizon_from_duration, outcome_from_result
from app.models import MLOfflineModel, MLModelState, MLTrainingEvent, OptimizerRun, StrategyRanking


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
    model_id = result["trained_models"][0]["model_id"]
    promoted = _FakeOfflineRanker(app.config, artifact_root=tmp_path).promote("1h", model_id=model_id)
    assert promoted["promoted"] is True
    assert db.session.get(MLOfflineModel, model_id).status == "promoted"
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
