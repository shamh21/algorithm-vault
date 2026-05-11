"""Offline ranker training, promotion, and promoted-score blending."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from flask import current_app, has_app_context

from ..extensions import db
from ..models import MLMarketHistory, MLOfflineModel, MLTrainingEvent, StrategyRanking
from ..services.provider_assets import normalize_provider, provider_feature_context
from .features import MLFeatureFactory
from .online_ranker import OnlineRanker, extract_features, horizon_from_duration, outcome_from_result


FEATURE_SCHEMA_VERSION = "offline_ranker_v2"


@dataclass(frozen=True, slots=True)
class OfflineTrainingRow:
    features: dict[str, float]
    target: float
    created_at: datetime
    source: str
    provider: str = "global"


class OfflineRanker:
    """Trains explicit offline models and uses only promoted artifacts for scoring."""

    def __init__(self, config: dict[str, Any], *, artifact_root: str | Path | None = None) -> None:
        self.config = config
        self.online_ranker = OnlineRanker(config)
        self.artifact_root = Path(artifact_root) if artifact_root else None

    def train(
        self,
        horizon: str,
        *,
        model_types: str | list[str] = "both",
        provider: str = "global",
        use_market_history: bool = False,
    ) -> dict[str, Any]:
        horizon_key = str(horizon or "global").lower()
        provider_key = normalize_provider(provider)
        requested = self._model_types(model_types)
        blockers: list[str] = []
        if not bool(self.config.get("ML_OFFLINE_MODELS_ENABLED", False)):
            blockers.append("ML_OFFLINE_MODELS_ENABLED=false")
        if not self._module_available("joblib"):
            blockers.append("joblib_missing")
        rows = self.training_rows(horizon_key, provider=provider_key, use_market_history=use_market_history)
        min_rows = int(self.config.get("ML_OFFLINE_MIN_TRAINING_ROWS", 250) or 250)
        if len(rows) < min_rows:
            blockers.append("insufficient_training_rows")
        if blockers:
            return {
                "trained": False,
                "horizon": horizon_key,
                "provider": provider_key,
                "requested_model_types": requested,
                "training_rows": len(rows),
                "min_training_rows": min_rows,
                "use_market_history": bool(use_market_history),
                "training_dataset": self._training_dataset_payload(rows),
                "blockers": blockers,
            }

        feature_names = sorted({key for row in rows for key in row.features})
        if not feature_names:
            return {
                "trained": False,
                "horizon": horizon_key,
                "provider": provider_key,
                "requested_model_types": requested,
                "training_rows": len(rows),
                "min_training_rows": min_rows,
                "use_market_history": bool(use_market_history),
                "training_dataset": self._training_dataset_payload(rows),
                "blockers": ["empty_feature_schema"],
            }

        x_all = [self._vector(row.features, feature_names) for row in rows]
        y_all = [row.target for row in rows]
        split_index = max(1, int(len(rows) * 0.8))
        if split_index >= len(rows):
            split_index = len(rows) - 1
        train_x, valid_x = x_all[:split_index], x_all[split_index:]
        train_y, valid_y = y_all[:split_index], y_all[split_index:]

        trained: list[dict[str, Any]] = []
        skipped: dict[str, str] = {}
        for model_type in requested:
            model = self._fit_model(model_type, train_x, train_y)
            if isinstance(model, str):
                skipped[model_type] = model
                continue
            predictions = [float(value) for value in model.predict(valid_x)] if valid_x else []
            metrics = self._metrics(valid_y, predictions)
            metrics["feature_importance"] = self._feature_importance(model, feature_names)
            artifact_path = self._artifact_path(provider_key, horizon_key, model_type)
            payload = {
                "model": model,
                "model_type": model_type,
                "horizon": horizon_key,
                "provider": provider_key,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_names": feature_names,
                "created_at": datetime.utcnow().isoformat(),
                "metrics": metrics,
                "feature_importance": metrics["feature_importance"],
            }
            self._dump_artifact(payload, artifact_path)
            record = MLOfflineModel(
                model_key=f"offline_ranker:{provider_key}:{horizon_key}:{model_type}:{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                provider=provider_key,
                horizon=horizon_key,
                model_type=model_type,
                status="candidate",
                artifact_path=str(artifact_path),
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                training_rows=len(train_x),
                validation_rows=len(valid_x),
                validation_loss=metrics["validation_loss"],
                negative_error_rate=metrics["negative_error_rate"],
                drift=metrics["drift"],
            )
            record.feature_names = feature_names
            record.metrics = metrics
            db.session.add(record)
            db.session.flush()
            trained.append(self._model_payload(record))

        db.session.commit()
        return {
            "trained": bool(trained),
            "horizon": horizon_key,
            "provider": provider_key,
            "requested_model_types": requested,
            "trained_models": trained,
            "skipped_models": skipped,
            "training_rows": len(rows),
            "feature_count": len(feature_names),
            "use_market_history": bool(use_market_history),
            "training_dataset": self._training_dataset_payload(rows),
            "blockers": [] if trained else ["no_model_trained"],
        }

    def promote(self, horizon: str, *, model_id: int, provider: str = "global") -> dict[str, Any]:
        horizon_key = str(horizon or "global").lower()
        provider_key = normalize_provider(provider)
        record = MLOfflineModel.query.filter_by(id=int(model_id), horizon=horizon_key, provider=provider_key).one_or_none()
        if record is None:
            return {"promoted": False, "horizon": horizon_key, "provider": provider_key, "model_id": model_id, "blockers": ["model_not_found"]}
        diagnostics = self.promotion_diagnostics(record)
        if not diagnostics["ready"]:
            return {"promoted": False, **diagnostics}

        for promoted in (
            MLOfflineModel.query.filter_by(horizon=horizon_key, provider=provider_key, status="promoted")
            .filter(MLOfflineModel.model_type.in_(self._model_types("both")))
            .all()
        ):
            if promoted.id != record.id:
                promoted.status = "archived"
        record.status = "promoted"
        record.promoted_at = datetime.utcnow()
        db.session.commit()
        return {"promoted": True, **self._model_payload(record), "blockers": []}

    def score_payload(
        self,
        context: dict[str, Any],
        horizon: str,
        *,
        base_score: float | None = None,
        rejected: bool = False,
    ) -> dict[str, Any]:
        horizon_key = str(horizon or "global").lower()
        provider_key = normalize_provider(context.get("provider") or context.get("execution_venue"))
        record = self.promoted_model(horizon_key, provider=provider_key, safe_scoring=True)
        unsafe_record = None
        if record is None:
            unsafe_record = self.promoted_model(horizon_key, provider=provider_key, safe_scoring=False)
        payload: dict[str, Any] = {
            "enabled": bool(self.config.get("ML_OFFLINE_MODELS_ENABLED", False)),
            "blend_enabled": bool(self.config.get("ML_OFFLINE_BLEND_ENABLED", False)),
            "blend_applied": False,
            "status": "no_promoted_model",
            "horizon": horizon_key,
            "provider": provider_key,
            "prediction": 0.0,
            "model_id": None,
            "model_type": None,
            "blended_score": base_score,
            "feature_drift": {},
            "blockers": [],
        }
        if record is None:
            if unsafe_record is not None:
                model_type = str(getattr(unsafe_record, "model_type", "") or "").lower()
                payload.update(
                    {
                        "status": "promoted_model_type_not_safe_for_scoring",
                        "model_id": getattr(unsafe_record, "id", None),
                        "model_type": model_type,
                        "blockers": [f"offline_model_type_not_safe_for_scoring:{model_type}"],
                    }
                )
                return payload
            payload["blockers"] = ["promoted_model_missing"]
            return payload
        diagnostics = self.promotion_diagnostics(record)
        payload.update(
            {
                "status": "promoted",
                "model_id": record.id,
                "model_type": record.model_type,
                "metrics": record.metrics,
                "blockers": diagnostics["blockers"],
            }
        )
        if diagnostics["blockers"]:
            payload["status"] = "promoted_blocked"
            return payload
        artifact = self._load_artifact(record.artifact_path)
        if isinstance(artifact, str):
            payload["status"] = "artifact_unavailable"
            payload["blockers"] = [artifact]
            return payload
        normalized = self.online_ranker.normalized_features(extract_features({**provider_feature_context(provider_key), **context}))
        feature_names = list(artifact.get("feature_names") or record.feature_names)
        vector = self._vector(normalized, feature_names)
        prediction = float(artifact["model"].predict([vector])[0])
        prediction_cap = float(self.config.get("ML_TARGET_CAP", 1.0) or 1.0)
        prediction = max(-prediction_cap, min(prediction, prediction_cap))
        payload["prediction"] = prediction
        payload["feature_drift"] = self._feature_drift(normalized, feature_names)
        if bool(payload["blend_enabled"]) and base_score is not None and not rejected:
            weight = float(self.config.get("ML_OFFLINE_SCORE_WEIGHT", 0.15) or 0.15)
            payload["blended_score"] = float(base_score) + weight * prediction
            payload["blend_applied"] = True
        return payload

    def readiness(self, horizon: str = "1h", *, require_blend: bool = True, provider: str = "global") -> dict[str, Any]:
        horizon_key = str(horizon or "global").lower()
        provider_key = normalize_provider(provider)
        record = self.promoted_model(horizon_key, provider=provider_key, safe_scoring=True)
        unsafe_record = None
        if record is None:
            unsafe_record = self.promoted_model(horizon_key, provider=provider_key, safe_scoring=False)
        blockers: list[str] = []
        if not bool(self.config.get("ML_OFFLINE_MODELS_ENABLED", False)):
            blockers.append("ML_OFFLINE_MODELS_ENABLED=false")
        if require_blend and not bool(self.config.get("ML_OFFLINE_BLEND_ENABLED", False)):
            blockers.append("ML_OFFLINE_BLEND_ENABLED=false")
        if record is None:
            if unsafe_record is not None:
                blockers.append(f"offline_model_type_not_safe_for_scoring:{str(unsafe_record.model_type or '').lower()}")
            else:
                blockers.append("promoted_model_missing")
        diagnostics = self.promotion_diagnostics(record) if record else {"blockers": []}
        blockers.extend(diagnostics.get("blockers", []))
        return {
            "ready": not blockers,
            "horizon": horizon_key,
            "provider": provider_key,
            "blockers": list(dict.fromkeys(blockers)),
            "promoted_model": self._model_payload(record) if record else None,
            "unsafe_promoted_model": self._model_payload(unsafe_record) if unsafe_record else None,
            "model_types": self._model_types("both"),
            "safe_scoring_model_types": self._safe_scoring_model_types(),
            "blend_enabled": bool(self.config.get("ML_OFFLINE_BLEND_ENABLED", False)),
            "require_blend": bool(require_blend),
        }

    def promotion_diagnostics(self, record: MLOfflineModel | None) -> dict[str, Any]:
        if record is None:
            return {"ready": False, "blockers": ["model_not_found"]}
        blockers: list[str] = []
        max_loss = float(self.config.get("ML_OFFLINE_MAX_VALIDATION_LOSS", 0.20) or 0.20)
        max_negative = float(self.config.get("ML_OFFLINE_MAX_NEGATIVE_ERROR_RATE", 0.55) or 0.55)
        max_age_hours = float(self.config.get("ML_OFFLINE_MAX_MODEL_AGE_HOURS", 72.0) or 72.0)
        max_drift = float(self.config.get("ML_OFFLINE_MAX_DRIFT", 0.35) or 0.35)
        min_top_decile_precision = self._config_float("ML_OFFLINE_MIN_TOP_DECILE_PRECISION", 0.55)
        max_false_positive = self._config_float("ML_OFFLINE_MAX_FALSE_POSITIVE_HIGH_UPSIDE_RATE", 0.35)
        max_calibration_error = self._config_float("ML_OFFLINE_MAX_CALIBRATION_ERROR", 0.18)
        metrics = record.metrics if isinstance(record.metrics, dict) else {}
        if record.feature_schema_version != FEATURE_SCHEMA_VERSION:
            blockers.append("feature_schema_version_mismatch")
        if int(record.training_rows or 0) <= 0 or int(record.validation_rows or 0) <= 0:
            blockers.append("insufficient_train_validation_split")
        if float(record.validation_loss or 0.0) > max_loss:
            blockers.append("validation_loss_above_threshold")
        if float(record.negative_error_rate or 0.0) > max_negative:
            blockers.append("negative_error_rate_above_threshold")
        if float(record.drift or 0.0) > max_drift:
            blockers.append("prediction_drift_above_threshold")
        if float(metrics.get("calibration_error", 0.0) or 0.0) > max_calibration_error:
            blockers.append("calibration_error_above_threshold")
        if float(metrics.get("top_decile_precision", 0.0) or 0.0) < min_top_decile_precision:
            blockers.append("top_decile_precision_below_threshold")
        if float(metrics.get("false_positive_high_upside_rate", 0.0) or 0.0) > max_false_positive:
            blockers.append("false_positive_high_upside_rate_above_threshold")
        if max_age_hours > 0:
            created_at = record.created_at or datetime.utcnow()
            if created_at < datetime.utcnow() - timedelta(hours=max_age_hours):
                blockers.append("model_age_above_threshold")
        if not record.artifact_path or not Path(record.artifact_path).exists():
            blockers.append("artifact_missing")
        return {
            "ready": not blockers,
            "blockers": blockers,
            **self._model_payload(record),
        }

    def promoted_model(self, horizon: str, *, provider: str = "global", safe_scoring: bool = False) -> MLOfflineModel | None:
        if not has_app_context():
            return None
        provider_key = normalize_provider(provider)
        query = (
            MLOfflineModel.query.filter_by(horizon=str(horizon or "global").lower(), provider=provider_key, status="promoted")
            .filter(MLOfflineModel.model_type.in_(self._model_types("both")))
        )
        if safe_scoring:
            safe_types = self._safe_scoring_model_types()
            if safe_types:
                query = query.filter(MLOfflineModel.model_type.in_(safe_types))
        return query.order_by(MLOfflineModel.promoted_at.desc(), MLOfflineModel.created_at.desc()).first()

    def training_rows(self, horizon: str, *, provider: str = "global", use_market_history: bool = False) -> list[OfflineTrainingRow]:
        if not has_app_context():
            return []
        horizon_key = str(horizon or "global").lower()
        provider_key = normalize_provider(provider)
        rows: list[OfflineTrainingRow] = []
        rankings = StrategyRanking.query.order_by(StrategyRanking.created_at.asc(), StrategyRanking.id.asc()).all()
        for ranking in rankings:
            if horizon_from_duration(ranking.lock_duration_hours or 1) != horizon_key:
                continue
            ranking_provider = normalize_provider(getattr(ranking, "provider", "global"))
            if provider_key != "global" and ranking_provider != provider_key:
                continue
            payload = self._ranking_payload(ranking)
            features = self.online_ranker.normalized_features(extract_features(payload))
            rows.append(
                OfflineTrainingRow(
                    features=features,
                    target=self._target_from_payload(payload),
                    created_at=ranking.created_at or datetime.utcnow(),
                    source="strategy_ranking",
                    provider=ranking_provider,
                )
            )
        events = (
            MLTrainingEvent.query.filter_by(horizon=horizon_key)
            .order_by(MLTrainingEvent.created_at.asc(), MLTrainingEvent.id.asc())
            .all()
        )
        for event in events:
            details = event.details or {}
            event_provider = normalize_provider(getattr(event, "provider", None) or details.get("provider") or (event.features or {}).get("provider"))
            if provider_key != "global" and event_provider != provider_key:
                continue
            if event.mode == "live" and details.get("status") not in {"quarantined", "promoted"}:
                continue
            target_payload = {**provider_feature_context(event_provider), **dict(details or {}), **dict(event.features or {})}
            rows.append(
                OfflineTrainingRow(
                    features=self.online_ranker.normalized_features(extract_features(target_payload)),
                    target=self._target_from_payload(target_payload, fallback=float(event.outcome or 0.0)),
                    created_at=event.created_at or datetime.utcnow(),
                    source=f"training_event:{event.source}",
                    provider=event_provider,
                )
            )
        if use_market_history:
            rows.extend(self._market_history_training_rows(horizon_key, provider=provider_key))
        rows.sort(key=lambda row: row.created_at)
        return rows

    def _market_history_training_rows(self, horizon: str, *, provider: str = "global") -> list[OfflineTrainingRow]:
        provider_key = normalize_provider(provider)
        query = MLMarketHistory.query.filter_by(status="ok")
        if provider_key != "global":
            query = query.filter(MLMarketHistory.provider == provider_key)
        histories = (
            query.order_by(MLMarketHistory.window_end.asc(), MLMarketHistory.fetched_at.asc(), MLMarketHistory.id.asc())
            .limit(5_000)
            .all()
        )
        rows: list[OfflineTrainingRow] = []
        factory = MLFeatureFactory(self.config)
        max_rows = max(1, int(self.config.get("ML_OFFLINE_MARKET_HISTORY_MAX_ROWS", 50_000) or 50_000))
        samples_per_window = max(1, int(self.config.get("ML_OFFLINE_MARKET_HISTORY_SAMPLES_PER_WINDOW", 250) or 250))
        horizon_minutes = self._horizon_minutes(horizon)
        for history in histories:
            if len(rows) >= max_rows:
                break
            candles = [row for row in history.candles if isinstance(row, dict)]
            if len(candles) < 40:
                continue
            timeframe_minutes = self._timeframe_minutes(history.timeframe)
            forward_steps = max(1, int(round(horizon_minutes / max(timeframe_minutes, 1))))
            min_window = max(30, forward_steps + 5)
            last_cutoff = len(candles) - forward_steps - 1
            if last_cutoff <= min_window:
                continue
            span = max(1, last_cutoff - min_window)
            stride = max(1, span // samples_per_window)
            for cutoff_index in range(min_window, last_cutoff + 1, stride):
                if len(rows) >= max_rows:
                    break
                feature_start = max(0, cutoff_index - 240)
                feature_window = candles[feature_start : cutoff_index + 1]
                current_close = self._first_float(feature_window[-1], "close")
                future_close = self._first_float(candles[cutoff_index + forward_steps], "close")
                if current_close <= 0 or future_close <= 0:
                    continue
                forward_return = (future_close - current_close) / current_close
                venue_symbol = ""
                diagnostics = history.diagnostics if isinstance(history.diagnostics, dict) else {}
                if isinstance(diagnostics, dict):
                    venue_symbol = str(diagnostics.get("venue_symbol") or "")
                provider_row = normalize_provider(history.provider)
                payload = factory.build(
                    symbol=history.symbol,
                    timeframe=history.timeframe,
                    candles=feature_window,
                    optimizer_context={
                        **provider_feature_context(provider_row),
                        "strategy_name": "ml_market_history_offline",
                        "provider": provider_row,
                        "execution_venue": provider_row,
                        "venue_symbol": venue_symbol or history.symbol,
                        "horizon": horizon,
                        "lock_duration_hours": max(1.0, horizon_minutes / 60.0),
                        "trade_count": 1,
                        "profit_factor": 1.0,
                        "consistency": 0.5,
                        "window_stability": 1.0,
                    },
                    cutoff_timestamp=feature_window[-1].get("timestamp"),
                )
                target_payload = {
                    **payload,
                    "net_return_after_costs": forward_return,
                    "total_return": forward_return,
                    "recent_performance_score": forward_return,
                    "profit_factor": 1.2 if forward_return > 0 else 0.8,
                    "consistency": 1.0 if forward_return > 0 else 0.0,
                    "window_stability": 1.0,
                    "trade_count": 1,
                }
                rows.append(
                    OfflineTrainingRow(
                        features=self.online_ranker.normalized_features(extract_features(payload)),
                        target=self._target_from_payload(target_payload, fallback=forward_return),
                        created_at=self._history_sample_time(feature_window[-1], history),
                        source="ml_market_history:offline_ranker",
                        provider=provider_row,
                    )
                )
        return rows

    @staticmethod
    def _training_dataset_payload(rows: list[OfflineTrainingRow]) -> dict[str, Any]:
        sources: dict[str, int] = {}
        providers: dict[str, int] = {}
        for row in rows:
            sources[row.source] = sources.get(row.source, 0) + 1
            providers[row.provider] = providers.get(row.provider, 0) + 1
        return {
            "row_count": len(rows),
            "sources": dict(sorted(sources.items())),
            "providers": dict(sorted(providers.items())),
            "market_history_rows": sum(count for source, count in sources.items() if source.startswith("ml_market_history:")),
            "leakage_policy": "features are built only from each row's pre-cutoff candle window; forward return is target-only",
        }

    @staticmethod
    def _timeframe_minutes(timeframe: str | None) -> int:
        value = str(timeframe or "1h").lower()
        return {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}.get(value, 60)

    @staticmethod
    def _horizon_minutes(horizon: str | None) -> int:
        value = str(horizon or "1h").lower()
        if value.endswith("m"):
            return max(1, int(OfflineRanker._safe_float(value[:-1], 1.0)))
        if value.endswith("h"):
            return max(1, int(OfflineRanker._safe_float(value[:-1], 1.0) * 60))
        if value.endswith("d"):
            return max(1, int(OfflineRanker._safe_float(value[:-1], 1.0) * 24 * 60))
        return max(1, int(OfflineRanker._safe_float(value, 1.0) * 60))

    @staticmethod
    def _history_sample_time(candle: dict[str, Any], history: MLMarketHistory) -> datetime:
        raw = OfflineRanker._safe_float(candle.get("timestamp"), 0.0)
        seconds = raw / 1000.0 if raw > 10_000_000_000 else raw
        if seconds > 0:
            try:
                return datetime.utcfromtimestamp(seconds)
            except (OverflowError, OSError, ValueError):
                pass
        return history.window_end or history.fetched_at or history.created_at or datetime.utcnow()

    def _fit_model(self, model_type: str, train_x: list[list[float]], train_y: list[float]) -> Any | str:
        model_type = str(model_type or "").lower()
        if model_type == "sklearn":
            if not self._module_available("sklearn.ensemble"):
                return "sklearn_missing"
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(n_estimators=64, min_samples_leaf=2, random_state=17)
            model.fit(train_x, train_y)
            return model
        if model_type == "xgboost":
            if not self._module_available("xgboost"):
                return "xgboost_missing"
            from xgboost import XGBRegressor

            model = XGBRegressor(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=17,
            )
            model.fit(train_x, train_y)
            return model
        return "unsupported_model_type"

    def _artifact_path(self, provider: str, horizon: str, model_type: str) -> Path:
        root = self.artifact_root or Path(current_app.instance_path) / "ml_models"
        root.mkdir(parents=True, exist_ok=True)
        provider_key = normalize_provider(provider)
        return root / f"offline-ranker-{provider_key}-{horizon}-{model_type}-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.joblib"

    def _dump_artifact(self, payload: dict[str, Any], path: Path) -> None:
        import joblib

        joblib.dump(payload, path)

    def _load_artifact(self, path: str) -> dict[str, Any] | str:
        if not self._module_available("joblib"):
            return "joblib_missing"
        artifact_path = Path(path or "")
        if not artifact_path.exists():
            return "artifact_missing"
        import joblib

        payload = joblib.load(artifact_path)
        return payload if isinstance(payload, dict) and "model" in payload else "artifact_invalid"

    @staticmethod
    def _metrics(targets: list[float], predictions: list[float]) -> dict[str, float]:
        if not targets or not predictions:
            return {
                "validation_loss": 0.0,
                "negative_error_rate": 0.0,
                "drift": 0.0,
                "mean_absolute_error": 0.0,
                "calibration_error": 0.0,
                "top_decile_precision": 0.0,
                "rank_correlation": 0.0,
                "false_positive_high_upside_rate": 0.0,
            }
        errors = [target - prediction for target, prediction in zip(targets, predictions)]
        loss = sum(error * error for error in errors) / len(errors)
        mae = sum(abs(error) for error in errors) / len(errors)
        negative_error_rate = sum(1 for error in errors if error < 0.0) / len(errors)
        drift = abs(sum(errors) / len(errors))
        calibration_error = OfflineRanker._calibration_error(targets, predictions)
        top_decile_precision, false_positive_high_upside_rate = OfflineRanker._top_decile_quality(targets, predictions)
        rank_correlation = OfflineRanker._rank_correlation(targets, predictions)
        return {
            "validation_loss": float(loss),
            "negative_error_rate": float(negative_error_rate),
            "drift": float(drift),
            "mean_absolute_error": float(mae),
            "calibration_error": float(calibration_error),
            "top_decile_precision": float(top_decile_precision),
            "rank_correlation": float(rank_correlation),
            "false_positive_high_upside_rate": float(false_positive_high_upside_rate),
        }

    def _target_from_payload(self, payload: dict[str, Any], *, fallback: float | None = None) -> float:
        base = outcome_from_result(payload)
        fallback_value = base if fallback is None else float(fallback or 0.0)
        net_return = self._first_float(payload, "net_return_after_costs", "total_return", default=fallback_value)
        drawdown = abs(min(self._first_float(payload, "max_drawdown", "drawdown"), 0.0))
        favorable = max(self._first_float(payload, "max_favorable_excursion"), 0.0)
        adverse = abs(min(self._first_float(payload, "max_adverse_excursion"), 0.0))
        mfe_mae = self._first_float(payload, "mfe_mae_ratio")
        if mfe_mae <= 0 and favorable > 0 and adverse > 0:
            mfe_mae = favorable / max(adverse, 1e-9)
        churn = max(self._first_float(payload, "churn_penalty", "turnover_after_fees", "turnover_rate"), 0.0)
        stability = max(
            self._first_float(payload, "window_stability", default=0.5),
            self._first_float(payload, "accepted_window_ratio", default=0.5),
        )
        net_roi_v2 = self._first_float(payload, "net_roi_v2_score") / 100.0
        tail_loss = max(self._first_float(payload, "tail_loss_penalty"), 0.0)
        downside = max(self._first_float(payload, "downside_asymmetry_penalty"), 0.0)
        breakout = max(self._first_float(payload, "cost_adjusted_breakout_potential"), 0.0) / 10_000.0
        target = (
            base * 0.35
            + net_return * 1.45
            + min(max(mfe_mae, 0.0), 8.0) * 0.035
            + max(stability - 0.5, -0.5) * 0.12
            + net_roi_v2 * 0.25
            + breakout * 0.35
            - drawdown * 0.9
            - min(churn, 10.0) * 0.015
            - tail_loss * 0.25
            - downside * 0.20
        )
        if bool(payload.get("rejected", False)):
            target -= 0.05
        cap = float(self.config.get("ML_TARGET_CAP", 1.0) or 1.0)
        return max(-cap, min(float(target if math.isfinite(target) else fallback_value), cap))

    @staticmethod
    def _calibration_error(targets: list[float], predictions: list[float]) -> float:
        paired = sorted(zip(predictions, targets), key=lambda item: item[0])
        if not paired:
            return 0.0
        bucket_count = min(5, len(paired))
        bucket_size = max(1, math.ceil(len(paired) / bucket_count))
        errors: list[float] = []
        for start in range(0, len(paired), bucket_size):
            bucket = paired[start : start + bucket_size]
            if not bucket:
                continue
            mean_prediction = sum(item[0] for item in bucket) / len(bucket)
            mean_target = sum(item[1] for item in bucket) / len(bucket)
            errors.append(abs(mean_target - mean_prediction))
        return sum(errors) / len(errors) if errors else 0.0

    @staticmethod
    def _top_decile_quality(targets: list[float], predictions: list[float]) -> tuple[float, float]:
        paired = sorted(zip(predictions, targets), key=lambda item: item[0], reverse=True)
        if not paired:
            return 0.0, 0.0
        take = max(1, math.ceil(len(paired) * 0.10))
        top = paired[:take]
        positive = sum(1 for _prediction, target in top if target > 0.0)
        false_positive = len(top) - positive
        return positive / len(top), false_positive / len(top)

    @staticmethod
    def _rank_correlation(targets: list[float], predictions: list[float]) -> float:
        if len(targets) < 2 or len(predictions) < 2:
            return 0.0
        target_ranks = OfflineRanker._ranks(targets)
        prediction_ranks = OfflineRanker._ranks(predictions)
        mean_target = sum(target_ranks) / len(target_ranks)
        mean_prediction = sum(prediction_ranks) / len(prediction_ranks)
        numerator = sum((a - mean_target) * (b - mean_prediction) for a, b in zip(target_ranks, prediction_ranks))
        denom_a = math.sqrt(sum((a - mean_target) ** 2 for a in target_ranks))
        denom_b = math.sqrt(sum((b - mean_prediction) ** 2 for b in prediction_ranks))
        if denom_a <= 0 or denom_b <= 0:
            return 0.0
        return numerator / (denom_a * denom_b)

    @staticmethod
    def _ranks(values: list[float]) -> list[float]:
        indexed = sorted(enumerate(values), key=lambda item: item[1])
        ranks = [0.0] * len(values)
        for rank, (index, _value) in enumerate(indexed, start=1):
            ranks[index] = float(rank)
        return ranks

    @staticmethod
    def _feature_importance(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
        raw_importance = getattr(model, "feature_importances_", None)
        if raw_importance is None:
            raw_importance = getattr(model, "coef_", None)
        if raw_importance is None:
            return []
        try:
            values = [abs(float(value)) for value in list(raw_importance)]
        except (TypeError, ValueError):
            return []
        total = sum(values)
        rows = [
            {
                "feature": feature,
                "importance": (value / total if total > 0 else value),
            }
            for feature, value in zip(feature_names, values)
        ]
        rows.sort(key=lambda item: float(item["importance"]), reverse=True)
        return rows[:20]

    @staticmethod
    def _vector(features: dict[str, float], feature_names: list[str]) -> list[float]:
        return [float(features.get(name, 0.0) or 0.0) for name in feature_names]

    def _model_types(self, value: str | list[str]) -> list[str]:
        configured = [str(item).strip().lower() for item in self.config.get("ML_OFFLINE_MODEL_TYPES", ["sklearn", "xgboost"])]
        if isinstance(value, str) and value.strip().lower() == "both":
            requested = configured
        elif isinstance(value, str):
            requested = [value.strip().lower()]
        else:
            requested = [str(item).strip().lower() for item in value]
        return [item for item in dict.fromkeys(requested) if item in {"sklearn", "xgboost"}]

    def _safe_scoring_model_types(self) -> list[str]:
        configured = self.config.get("ML_OFFLINE_SAFE_SCORING_MODEL_TYPES", ["sklearn"])
        if isinstance(configured, str):
            requested = [item.strip().lower() for item in configured.split(",") if item.strip()]
        else:
            requested = [str(item).strip().lower() for item in configured or [] if str(item).strip()]
        return [item for item in dict.fromkeys(requested) if item in {"sklearn", "xgboost"}]

    @staticmethod
    def _module_available(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            return False

    @staticmethod
    def _feature_drift(features: dict[str, float], feature_names: list[str]) -> dict[str, Any]:
        expected = set(feature_names)
        observed = set(features)
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        denominator = max(len(expected), 1)
        return {
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "missing_ratio": len(missing) / denominator,
            "unexpected_ratio": len(unexpected) / denominator,
            "missing_sample": missing[:8],
            "unexpected_sample": unexpected[:8],
        }

    @staticmethod
    def _model_payload(record: MLOfflineModel | None) -> dict[str, Any]:
        if record is None:
            return {}
        return {
            "model_id": record.id,
            "model_key": record.model_key,
            "provider": getattr(record, "provider", "global"),
            "horizon": record.horizon,
            "model_type": record.model_type,
            "status": record.status,
            "feature_schema_version": record.feature_schema_version,
            "feature_count": len(record.feature_names),
            "training_rows": int(record.training_rows or 0),
            "validation_rows": int(record.validation_rows or 0),
            "validation_loss": float(record.validation_loss or 0.0),
            "negative_error_rate": float(record.negative_error_rate or 0.0),
            "drift": float(record.drift or 0.0),
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "promoted_at": record.promoted_at.isoformat() if record.promoted_at else None,
            "metrics": record.metrics,
            "feature_importance": (record.metrics or {}).get("feature_importance", []) if isinstance(record.metrics, dict) else [],
            "artifact_exists": bool(record.artifact_path and Path(record.artifact_path).exists()),
        }

    @staticmethod
    def _ranking_payload(ranking: StrategyRanking) -> dict[str, Any]:
        explanation = ranking.ml_explanation if isinstance(ranking.ml_explanation, dict) else {}
        net_roi = explanation.get("net_roi") if isinstance(explanation.get("net_roi"), dict) else {}
        net_roi_v2 = explanation.get("net_roi_v2") if isinstance(explanation.get("net_roi_v2"), dict) else {}
        payload = {
            **provider_feature_context(getattr(ranking, "provider", "global")),
            "strategy_name": ranking.strategy_name,
            "symbol": ranking.symbol,
            "timeframe": ranking.timeframe,
            "profile": ranking.profile,
            "optimizer_profile": ranking.profile,
            "net_return_after_costs": ranking.net_return_after_costs,
            "total_return": ranking.total_return,
            "recent_performance_score": ranking.recent_performance_score,
            "recent_1h_return": ranking.recent_1h_return,
            "max_drawdown": ranking.max_drawdown,
            "profit_factor": ranking.profit_factor,
            "sortino_like": ranking.sortino_like,
            "sharpe_like": ranking.sharpe_like,
            "consistency": ranking.consistency,
            "window_stability": ranking.window_stability,
            "accepted_window_ratio": ranking.accepted_window_ratio,
            "win_rate": ranking.win_rate,
            "trade_count": ranking.trade_count,
            "trades_per_day": ranking.trades_per_day,
            "avg_trade_return": ranking.avg_trade_return,
            "edge_score": ranking.edge_score,
            "expectancy": ranking.expectancy,
            "cost_drag_bps": ranking.cost_drag_bps,
            "turnover_after_fees": ranking.turnover_after_fees,
            "allocation_amount_usd": ranking.allocation_amount_usd,
            "lock_duration_hours": ranking.lock_duration_hours,
            "leverage": ranking.leverage,
            "liquidation_buffer_pct": ranking.liquidation_buffer_pct,
            "capacity_usd": ranking.capacity_usd,
            "convex_edge_score": ranking.convex_edge_score,
            "mfe_mae_ratio": ranking.mfe_mae_ratio,
            "rejected": bool(ranking.rejected),
            "rejection_reason": ranking.rejection_reason or "",
        }
        payload.update(
            {
                "net_roi_score": net_roi.get("net_roi_score", 0.0),
                "expected_fill_quality": net_roi.get("expected_fill_quality", 0.0),
                "churn_penalty": net_roi.get("churn_penalty", 0.0),
                "edge_after_cost_bps": net_roi.get("edge_after_cost_bps", 0.0),
                "data_age_seconds": net_roi.get("data_age_seconds", 0.0),
                "net_roi_v2_score": net_roi_v2.get("net_roi_v2_score", 0.0),
                "roi_quality_grade": net_roi_v2.get("roi_quality_grade", "D"),
                "roi_rejection_risk": net_roi_v2.get("roi_rejection_risk", "high"),
                "regime_bucket": net_roi_v2.get("regime_bucket", {}),
                "regime_support": net_roi_v2.get("regime_support", "regime-neutral"),
                "regime_adjustment": net_roi_v2.get("regime_adjustment", 0.0),
                "regime_adjusted_expectancy": net_roi_v2.get("regime_adjusted_expectancy", 0.0),
                "tail_loss_penalty": net_roi_v2.get("tail_loss_penalty", 0.0),
                "downside_asymmetry_penalty": net_roi_v2.get("downside_asymmetry_penalty", 0.0),
                "cost_adjusted_breakout_potential": net_roi_v2.get("cost_adjusted_breakout_potential", 0.0),
            }
        )
        components = net_roi_v2.get("components") if isinstance(net_roi_v2.get("components"), dict) else {}
        if components:
            payload.setdefault("volatility_regime", (components.get("regime_bucket") or {}).get("volatility") if isinstance(components.get("regime_bucket"), dict) else "")
            for key in (
                "tail_loss_penalty",
                "downside_asymmetry_penalty",
                "cost_adjusted_breakout_potential",
                "regime_adjustment",
            ):
                payload[key] = components.get(key, payload.get(key, 0.0))
        return payload

    @staticmethod
    def _is_number(value: Any) -> bool:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(candidate)

    @staticmethod
    def _first_float(payload: dict[str, Any], *keys: str, default: float = 0.0) -> float:
        for key in keys:
            value = payload.get(key)
            if value in (None, ""):
                continue
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(candidate):
                return candidate
        return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value in (None, ""):
            return default
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return default
        return candidate if math.isfinite(candidate) else default

    def _config_float(self, key: str, default: float) -> float:
        value = self.config.get(key, default)
        if value in (None, ""):
            return default
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return default
        return candidate if math.isfinite(candidate) else default
