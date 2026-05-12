"""Shared ML decision envelopes for app-wide advisory decisions.

The decision engine gives discovery, selection, execution, and operational
diagnostics a common ML contract. It intentionally does not own irreversible
controls: live orders still flow through StrategyRunner, OrderManager, and
RiskEngine.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from flask import current_app, has_app_context

from ..extensions import db
from ..models import (
    AuditLog,
    BacktestRun,
    LeveragedMarketFeature,
    MLMarketHistory,
    MLOfflineModel,
    MLTrainingEvent,
    RiskEvent,
    Setting,
    StrategyRanking,
    VaultAllocationLeg,
    VaultCycle,
)
from ..services.model_registry import dataset_hash, feature_schema_hash
from ..services.provider_assets import normalize_provider, provider_feature_context
from .online_ranker import OnlineRanker, extract_features, horizon_from_duration, outcome_from_result
from .features import MLFeatureFactory, ML_FEATURE_SCHEMA_VERSION


FEATURE_SCHEMA_VERSION = "ml_decision_v1"
SIGNAL_FAMILY = "pytorch_gru_signal"
SIGNAL_MODEL_TYPE = "pytorch_gru"
EXTREME_UPSIDE_FAMILY = "pytorch_extreme_upside"
POLICY_FAMILIES = (
    "pytorch_risk_policy",
    "pytorch_exit_policy",
    "pytorch_cap_policy",
    "pytorch_execution_policy",
    "pytorch_roi_target",
)
MODEL_FAMILIES = (
    SIGNAL_FAMILY,
    EXTREME_UPSIDE_FAMILY,
    "pytorch_fibonacci",
    "pytorch_backtest_scorer",
    "pytorch_optimizer_policy",
    "pytorch_allocator",
    "pytorch_universe",
    "pytorch_ops_anomaly",
    *POLICY_FAMILIES,
)


@dataclass(slots=True)
class MLDecisionEnvelope:
    """Standard shape returned by every ML advisory decision."""

    family: str
    action: str = "hold"
    confidence: float = 0.0
    expected_return: float = 0.0
    uncertainty: float = 1.0
    model_id: int | None = None
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    blockers: list[str] = field(default_factory=list)
    fallback_used: bool = False
    audit_metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ready"] = not bool(self.blockers)
        return payload


@dataclass(frozen=True, slots=True)
class MLDecisionTrainingRow:
    """Feature/target row for non-signal ML decision families."""

    features: dict[str, float]
    target: float
    created_at: datetime
    source: str
    provider: str = "global"


class MLDecisionEngine:
    """App-wide ML decision facade with deterministic safety boundaries."""

    model_families = MODEL_FAMILIES

    def __init__(self, config: dict[str, Any], *, signal_model: Any | None = None) -> None:
        self.config = config
        self.signal_model = signal_model
        self.online_ranker = OnlineRanker(config)
        self.feature_factory = MLFeatureFactory(config)

    def readiness(self, horizon: str = "1h", *, family: str = "all", provider: str = "global") -> dict[str, Any]:
        horizon_key = self._horizon(horizon)
        provider_key = normalize_provider(provider)
        selected = self._selected_families(family)
        families = {name: self.family_readiness(name, horizon_key, provider=provider_key) for name in selected}
        blockers: list[str] = []
        for name, payload in families.items():
            blockers.extend(f"{name}:{item}" for item in payload.get("blockers", []))
        return {
            "ready": not blockers,
            "enabled": bool(self.config.get("ML_ALL_AREAS_ENABLED", False)),
            "horizon": horizon_key,
            "provider": provider_key,
            "family": family,
            "families": families,
            "blockers": list(dict.fromkeys(blockers)),
            "deterministic_safety_gates_required": bool(
                self.config.get("ML_DETERMINISTIC_SAFETY_GATES_REQUIRED", True)
            ),
            "live_requires_promoted": bool(self.config.get("ML_REQUIRE_PROMOTED_FOR_LIVE", True)),
        }

    def family_readiness(self, family: str, horizon: str = "1h", *, provider: str = "global") -> dict[str, Any]:
        family_key = self._family(family)
        horizon_key = self._horizon(horizon)
        provider_key = normalize_provider(provider)
        blockers: list[str] = []
        if not bool(self.config.get("ML_ALL_AREAS_ENABLED", False)):
            blockers.append("ML_ALL_AREAS_ENABLED=false")
        if family_key == SIGNAL_FAMILY:
            signal_payload = self._signal_readiness(horizon_key, provider=provider_key)
            blockers.extend(str(item) for item in signal_payload.get("blockers", []))
            return {
                "ready": not blockers,
                "family": family_key,
                "horizon": horizon_key,
                "provider": provider_key,
                "model_type": SIGNAL_MODEL_TYPE,
                "blockers": list(dict.fromkeys(blockers)),
                "promoted_model": signal_payload.get("promoted_model"),
                "enabled": bool(self.config.get("ML_ALL_AREAS_ENABLED", False))
                and bool(self.config.get("ML_SIGNAL_MODEL_ENABLED", False)),
                "source": "ml_signal_model",
            }
        family_flag = {
            EXTREME_UPSIDE_FAMILY: "ML_EXTREME_UPSIDE_MODEL_ENABLED",
            "pytorch_fibonacci": "ML_FIBONACCI_MODEL_ENABLED",
            "pytorch_backtest_scorer": "ML_BACKTEST_SCORER_ENABLED",
            "pytorch_optimizer_policy": "ML_OPTIMIZER_POLICY_ENABLED",
            "pytorch_ops_anomaly": "ML_OPS_ANOMALY_ENABLED",
            "pytorch_risk_policy": "ML_RISK_POLICY_ENABLED",
            "pytorch_exit_policy": "ML_EXIT_POLICY_ENABLED",
            "pytorch_cap_policy": "ML_CAP_POLICY_ENABLED",
            "pytorch_execution_policy": "ML_ORDER_POLICY_ENABLED",
            "pytorch_roi_target": "ML_ROI_TARGET_POLICY_ENABLED",
        }.get(family_key)
        if family_flag and not bool(self.config.get(family_flag, False)):
            blockers.append(f"{family_flag}=false")
        if not self._module_available("torch"):
            blockers.append("torch_missing")
        record = self.promoted_model(family_key, horizon_key, provider=provider_key)
        if bool(self.config.get("ML_REQUIRE_PROMOTED_FOR_LIVE", True)) and record is None:
            blockers.append(f"promoted_{family_key}_missing")
        if record is not None:
            blockers.extend(self.promotion_diagnostics(record, family_key).get("blockers", []))
        return {
            "ready": not blockers,
            "family": family_key,
            "horizon": horizon_key,
            "provider": provider_key,
            "model_type": family_key,
            "blockers": list(dict.fromkeys(blockers)),
            "promoted_model": self._model_payload(record) if record else None,
            "enabled": bool(self.config.get("ML_ALL_AREAS_ENABLED", False)),
            "source": "ml_decision_engine",
        }

    def decision(
        self,
        family: str,
        context: dict[str, Any] | None = None,
        *,
        horizon: str = "1h",
        candles: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        family_key = self._family(family)
        horizon_key = self._horizon(horizon)
        context_payload = dict(context or {})
        provider_key = normalize_provider(context_payload.get("provider") or context_payload.get("execution_venue"))
        context_payload = {**provider_feature_context(provider_key), **context_payload}
        if family_key == SIGNAL_FAMILY:
            return self._signal_decision(context_payload, horizon_key, candles or []).as_dict()

        readiness = self.family_readiness(family_key, horizon_key, provider=provider_key)
        record_payload = readiness.get("promoted_model") if isinstance(readiness.get("promoted_model"), dict) else {}
        if not bool(readiness.get("ready", False)):
            return MLDecisionEnvelope(
                family=family_key,
                action=self._default_action(family_key),
                blockers=list(readiness.get("blockers", []) or []),
                fallback_used=True,
                audit_metadata={
                    "horizon": horizon_key,
                    "provider": provider_key,
                    "status": "not_ready",
                    "deterministic_safety_gates_required": True,
                },
            ).as_dict()

        artifact_payload: dict[str, Any] = {}
        artifact_prediction = 0.0
        artifact = self._load_decision_artifact(record_payload.get("artifact_path"))
        if isinstance(artifact, str):
            return MLDecisionEnvelope(
                family=family_key,
                action=self._default_action(family_key),
                blockers=[artifact],
                fallback_used=True,
                audit_metadata={
                    "horizon": horizon_key,
                    "provider": provider_key,
                    "status": "artifact_unavailable",
                    "deterministic_safety_gates_required": True,
                },
            ).as_dict()
        artifact_payload = artifact
        artifact_prediction = self._score_artifact(artifact_payload, context_payload)
        confidence = self._confidence(context_payload, record_payload, prediction=artifact_prediction)
        expected_return = self._expected_return(context_payload, record_payload, prediction=artifact_prediction)
        raw_score = self._score(context_payload)
        sizing_score = max(0.0, min(artifact_prediction if family_key == "pytorch_allocator" else confidence, 1.0))
        ops_score = max(
            self._ops_anomaly_score(context_payload),
            max(0.0, min(artifact_prediction, 1.0)) if family_key == "pytorch_ops_anomaly" else 0.0,
        )
        family_raw = {
            "score": raw_score,
            "prediction": artifact_prediction,
            "sizing_score": sizing_score,
            "execution_style_suggestion": self._execution_style_suggestion(family_key, context_payload),
            "ops_anomaly_score": ops_score if family_key == "pytorch_ops_anomaly" else 0.0,
        }
        family_raw.update(self._family_specific_raw(family_key, context_payload, artifact_prediction))
        return MLDecisionEnvelope(
            family=family_key,
            action=self._action(family_key, context_payload, confidence, prediction=artifact_prediction),
            confidence=confidence,
            expected_return=expected_return,
            uncertainty=max(0.0, min(1.0, 1.0 - confidence)),
            model_id=self._safe_int(record_payload.get("id")),
            feature_schema_version=str(record_payload.get("feature_schema_version") or FEATURE_SCHEMA_VERSION),
            blockers=[],
            fallback_used=False,
            audit_metadata={
                "horizon": horizon_key,
                "provider": provider_key,
                "model_type": family_key,
                "deterministic_safety_gates_required": True,
                "can_override_safety": False,
                "feature_count": len(artifact_payload.get("feature_names") or []),
            },
            raw=family_raw,
        ).as_dict()

    def train_suite(
        self,
        horizon: str = "1h",
        *,
        family: str = "all",
        objective: str = "risk_adjusted",
        use_market_history: bool = False,
        provider: str = "global",
    ) -> dict[str, Any]:
        horizon_key = self._horizon(horizon)
        provider_key = normalize_provider(provider)
        selected = self._selected_families(family)
        results: dict[str, Any] = {}
        blockers: list[str] = []
        for family_key in selected:
            if family_key == SIGNAL_FAMILY:
                if self.signal_model is None:
                    result = {"trained": False, "blockers": ["ml_signal_model_service_unavailable"]}
                else:
                    result = self.signal_model.train(
                        horizon_key,
                        model_type=SIGNAL_MODEL_TYPE,
                        objective=objective,
                        use_market_history=use_market_history,
                        provider=provider_key,
                    )
                    result.setdefault("training_dataset", {})
                    result["training_dataset"].update(
                        {
                            "use_market_history": bool(use_market_history),
                            "objective": self._objective(objective),
                            "market_history_supported_for_signal_model": True,
                        }
                    )
            else:
                result = self.train_family(
                    horizon_key,
                    family=family_key,
                    objective=objective,
                    use_market_history=use_market_history,
                    provider=provider_key,
                )
            results[family_key] = result
            blockers.extend(f"{family_key}:{item}" for item in result.get("blockers", []) or [])
        return {
            "trained": any(bool(item.get("trained", False)) for item in results.values()),
            "research_only": True,
            "horizon": horizon_key,
            "provider": provider_key,
            "family": family,
            "objective": self._objective(objective),
            "use_market_history": bool(use_market_history),
            "family_results": results,
            "blockers": list(dict.fromkeys(blockers)),
        }

    def train_family(
        self,
        horizon: str = "1h",
        *,
        family: str,
        objective: str = "risk_adjusted",
        use_market_history: bool = False,
        provider: str = "global",
    ) -> dict[str, Any]:
        """Train one non-signal app-wide ML decision family.

        These models are advisory only: universe models reorder candidates,
        allocator models can tighten sizing, and ops models can warn/block scans.
        They never submit orders or clear deterministic safety gates.
        """

        family_key = self._family(family)
        horizon_key = self._horizon(horizon)
        objective_key = self._objective(objective)
        provider_key = normalize_provider(provider)
        training_started_at = datetime.utcnow()
        rows = self.training_rows(
            family_key,
            horizon_key,
            objective=objective_key,
            use_market_history=use_market_history,
            provider=provider_key,
        )
        target_distribution = self._target_distribution([row.target for row in rows])
        default_min_rows = int(self.config.get("ML_SIGNAL_MIN_TRAINING_ROWS", 500) or 500)
        if horizon_key == "1h10":
            default_min_rows = int(self.config.get("ML_ONE_H10_MIN_TRAINING_ROWS", 20) or 20)
        min_rows = max(2, default_min_rows)
        blockers: list[str] = []
        if family_key not in MODEL_FAMILIES or family_key == SIGNAL_FAMILY:
            blockers.append("unsupported_ml_decision_family")
        if not bool(self.config.get("ML_ALL_AREAS_ENABLED", False)):
            blockers.append("ML_ALL_AREAS_ENABLED=false")
        if not self._module_available("torch"):
            blockers.append("torch_missing")
        if len(rows) < min_rows:
            blockers.append("insufficient_training_rows")
        feature_names = sorted({key for row in rows for key in row.features})
        if not feature_names and not blockers:
            blockers.append("empty_feature_schema")
        if blockers:
            return {
                "trained": False,
                "research_only": True,
                "horizon": horizon_key,
                "provider": provider_key,
                "model_type": family_key,
                "objective": objective_key,
                "training_rows": len(rows),
                "validation_rows": 0,
                "min_training_rows": min_rows,
                "training_dataset": self._training_dataset_payload(rows, use_market_history=use_market_history),
                "walk_forward_metrics": {},
                "target_distribution": target_distribution,
                "metrics": {},
                "artifact_path": None,
                "blockers": list(dict.fromkeys(blockers)),
            }

        import torch
        from torch import nn

        x_all = [self._vector(row.features, feature_names) for row in rows]
        y_all = [row.target for row in rows]
        split_index = max(1, int(len(rows) * 0.8))
        if split_index >= len(rows):
            split_index = len(rows) - 1
        train_x, valid_x = x_all[:split_index], x_all[split_index:]
        train_y, valid_y = y_all[:split_index], y_all[split_index:]
        if not train_x or not valid_x:
            return {
                "trained": False,
                "research_only": True,
                "horizon": horizon_key,
                "provider": provider_key,
                "model_type": family_key,
                "objective": objective_key,
                "training_rows": len(train_x),
                "validation_rows": len(valid_x),
                "min_training_rows": min_rows,
                "training_dataset": self._training_dataset_payload(rows, use_market_history=use_market_history),
                "walk_forward_metrics": {},
                "target_distribution": target_distribution,
                "metrics": {},
                "artifact_path": None,
                "blockers": ["insufficient_train_validation_split"],
            }

        torch.manual_seed(17)
        risk_policy_classifier = family_key == "pytorch_risk_policy"
        hidden_size = max(8, min(64, len(feature_names) * 2))
        model_layers: list[Any] = [
            nn.Linear(len(feature_names), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ]
        if not risk_policy_classifier:
            model_layers.append(nn.Tanh())
        model = nn.Sequential(*model_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
        x_tensor = torch.tensor(train_x, dtype=torch.float32)
        if risk_policy_classifier:
            train_y_for_loss = [self._risk_policy_probability_target(value) for value in train_y]
            valid_y_for_loss = [self._risk_policy_probability_target(value) for value in valid_y]
            all_y_for_metrics = [self._risk_policy_probability_target(value) for value in y_all]
            positives = sum(1 for value in train_y_for_loss if value >= 0.5)
            negatives = max(len(train_y_for_loss) - positives, 0)
            pos_weight = torch.tensor([max(1.0, negatives / max(positives, 1))], dtype=torch.float32)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            y_tensor = torch.tensor(train_y_for_loss, dtype=torch.float32).reshape(-1, 1)
        else:
            valid_y_for_loss = valid_y
            all_y_for_metrics = y_all
            loss_fn = nn.MSELoss()
            y_tensor = torch.tensor(train_y, dtype=torch.float32).reshape(-1, 1)
        epochs = min(120, max(20, len(train_x) * 2))
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(x_tensor), y_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_output = model(torch.tensor(valid_x, dtype=torch.float32))
            all_output = model(torch.tensor(x_all, dtype=torch.float32))
            if risk_policy_classifier:
                validation_loss = float(
                    loss_fn(valid_output, torch.tensor(valid_y_for_loss, dtype=torch.float32).reshape(-1, 1)).item()
                )
                predictions = [
                    float(value)
                    for value in torch.sigmoid(valid_output).reshape(-1).detach().cpu().tolist()
                ]
                all_predictions = [
                    float(value)
                    for value in torch.sigmoid(all_output).reshape(-1).detach().cpu().tolist()
                ]
            else:
                predictions = [
                    float(value)
                    for value in valid_output.reshape(-1).detach().cpu().tolist()
                ]
                all_predictions = [
                    float(value)
                    for value in all_output.reshape(-1).detach().cpu().tolist()
                ]
                validation_loss = None
        if risk_policy_classifier:
            metrics = self._risk_policy_metrics(valid_y_for_loss, predictions, validation_loss=validation_loss)
        else:
            metrics = self._metrics(valid_y, predictions)
        metrics["target_distribution"] = target_distribution
        metrics["feature_importance"] = self._feature_importance(model, feature_names)
        walk_forward_metrics = self._walk_forward_metrics(
            all_y_for_metrics,
            all_predictions,
            prediction_source="model",
            false_positive_threshold=self._risk_policy_approve_threshold() if risk_policy_classifier else 0.1,
            positive_target_threshold=0.5 if risk_policy_classifier else 0.0,
        )
        metrics["walk_forward"] = walk_forward_metrics
        artifact_path = self._artifact_path(provider_key, horizon_key, family_key)
        artifact = {
            "model_type": family_key,
            "horizon": horizon_key,
            "provider": provider_key,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": feature_names,
            "input_size": len(feature_names),
            "hidden_size": hidden_size,
            "output_activation": "sigmoid" if risk_policy_classifier else "tanh",
            "state_dict": model.state_dict(),
            "created_at": datetime.utcnow().isoformat(),
            "metrics": metrics,
        }
        torch.save(artifact, artifact_path)
        record = MLOfflineModel(
            model_key=f"ml_suite:{provider_key}:{horizon_key}:{family_key}:{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            provider=provider_key,
            horizon=horizon_key,
            model_type=family_key,
            status="candidate",
            artifact_path=str(artifact_path),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            training_rows=len(train_x),
            validation_rows=len(valid_x),
            validation_loss=metrics["validation_loss"],
            negative_error_rate=metrics["negative_error_rate"],
            drift=metrics["drift"],
            feature_schema_hash=feature_schema_hash(FEATURE_SCHEMA_VERSION, feature_names),
            dataset_version=f"{provider_key}:{horizon_key}:{len(rows)}:{len(valid_x)}",
            dataset_hash=dataset_hash(
                {
                    "provider": provider_key,
                    "horizon": horizon_key,
                    "family": family_key,
                    "training_rows": len(rows),
                    "validation_rows": len(valid_x),
                    "feature_names": feature_names,
                    "target_distribution": target_distribution,
                }
            ),
            training_started_at=training_started_at,
            training_completed_at=datetime.utcnow(),
        )
        record.feature_names = feature_names
        record.metrics = metrics
        db.session.add(record)
        db.session.commit()
        return {
            "trained": True,
            "research_only": True,
            "horizon": horizon_key,
            "provider": provider_key,
            "model_type": family_key,
            "objective": objective_key,
            "training_rows": len(train_x),
            "validation_rows": len(valid_x),
            "training_dataset": self._training_dataset_payload(rows, use_market_history=use_market_history),
            "walk_forward_metrics": walk_forward_metrics,
            "target_distribution": target_distribution,
            "metrics": metrics,
            "artifact_path": str(artifact_path),
            "trained_models": [self._model_payload(record)],
            "blockers": [],
        }

    def promote_suite(self, horizon: str = "1h", *, model_id: int, provider: str = "global") -> dict[str, Any]:
        horizon_key = self._horizon(horizon)
        provider_key = normalize_provider(provider)
        record = db.session.get(MLOfflineModel, int(model_id)) if has_app_context() else None
        if (
            record is None
            or str(record.horizon or "").lower() != horizon_key
            or normalize_provider(getattr(record, "provider", "global")) != provider_key
        ):
            return {
                "promoted": False,
                "horizon": horizon_key,
                "provider": provider_key,
                "model_id": model_id,
                "blockers": ["ml_suite_model_not_found"],
            }
        model_type = str(record.model_type or "").lower()
        if model_type in {SIGNAL_MODEL_TYPE, SIGNAL_FAMILY}:
            if self.signal_model is None:
                return {
                    "promoted": False,
                    "horizon": horizon_key,
                    "provider": provider_key,
                    "model_id": model_id,
                    "blockers": ["ml_signal_model_service_unavailable"],
                }
            return self.signal_model.promote(horizon_key, model_id=int(model_id), provider=provider_key)
        family_key = self._family(model_type)
        diagnostics = self.promotion_diagnostics(record, family_key)
        if not diagnostics["ready"]:
            self._record_failed_promotion(family_key, provider_key, horizon_key, int(model_id), diagnostics["blockers"])
            return {"promoted": False, "horizon": horizon_key, "provider": provider_key, "model_id": model_id, **diagnostics}
        for promoted in MLOfflineModel.query.filter_by(
            horizon=horizon_key,
            provider=provider_key,
            model_type=family_key,
            status="promoted",
        ).all():
            if promoted.id != record.id:
                promoted.status = "archived"
        record.status = "promoted"
        record.promoted_at = datetime.utcnow()
        self._record_promotion(record, model_family=family_key, promotion_source="ml_decision.promote_suite")
        db.session.commit()
        return {
            "promoted": True,
            "family": family_key,
            "horizon": horizon_key,
            "provider": provider_key,
            "blockers": [],
            **(self._model_payload(record) or {}),
        }

    def _record_promotion(self, record: MLOfflineModel, *, model_family: str, promotion_source: str) -> None:
        if not has_app_context():
            return
        service = current_app.extensions.get("services", {}).get("model_registry")
        if service is not None:
            service.record_promotion(record, model_family=model_family, promotion_source=promotion_source)

    def _record_failed_promotion(
        self,
        model_family: str,
        provider: str,
        horizon: str,
        model_id: int | None,
        blockers: list[str],
    ) -> None:
        if not has_app_context():
            return
        service = current_app.extensions.get("services", {}).get("model_registry")
        if service is not None:
            service.record_failed_promotion(
                model_family=model_family,
                provider=provider,
                horizon=horizon,
                model_id=model_id,
                blockers=blockers,
            )
            db.session.commit()

    def promotion_diagnostics(self, record: MLOfflineModel | None, family: str | None = None) -> dict[str, Any]:
        if record is None:
            return {"ready": False, "blockers": ["ml_decision_model_not_found"]}
        family_key = self._family(family or record.model_type)
        blockers: list[str] = []
        if family_key not in MODEL_FAMILIES:
            blockers.append("unsupported_ml_decision_family")
        if str(record.model_type or "").lower() != family_key:
            blockers.append("ml_decision_model_type_mismatch")
        if family_key != SIGNAL_FAMILY and not self._module_available("torch"):
            blockers.append("torch_missing")
        if str(record.feature_schema_version or "") != FEATURE_SCHEMA_VERSION:
            blockers.append("ml_decision_feature_schema_version_mismatch")
        if int(record.training_rows or 0) <= 0 or int(record.validation_rows or 0) <= 0:
            blockers.append("insufficient_train_validation_split")
        max_validation_loss = (
            self.config.get("ML_RISK_POLICY_MAX_VALIDATION_LOSS", 0.75)
            if family_key == "pytorch_risk_policy"
            else self.config.get("ML_SIGNAL_MAX_VALIDATION_LOSS", 0.20)
        )
        if float(record.validation_loss or 0.0) > float(max_validation_loss or 0.20):
            blockers.append("ml_decision_validation_loss_above_threshold")
        metrics = record.metrics if isinstance(record.metrics, dict) else {}
        if family_key == "pytorch_risk_policy" and "approval_count" in metrics:
            approval_count = int(metrics.get("approval_count", 0) or 0)
            approval_rate = self._safe_float(metrics.get("approval_rate"))
            approval_precision = self._safe_float(metrics.get("approval_precision"))
            min_approval_count = max(0, int(self.config.get("ML_RISK_POLICY_MIN_APPROVAL_COUNT", 5) or 0))
            min_approval_rate = max(0.0, self._safe_float(self.config.get("ML_RISK_POLICY_MIN_APPROVAL_RATE"), 0.01))
            min_approval_precision = max(
                0.0,
                self._safe_float(self.config.get("ML_RISK_POLICY_MIN_APPROVAL_PRECISION"), 0.52),
            )
            if min_approval_count > 0 and approval_count < min_approval_count:
                blockers.append("risk_policy_approval_count_below_threshold")
            if min_approval_rate > 0 and approval_rate < min_approval_rate:
                blockers.append("risk_policy_approval_rate_below_threshold")
            if approval_count > 0 and approval_precision < min_approval_precision:
                blockers.append("risk_policy_approval_precision_below_threshold")
        false_positive = self._effective_false_positive_rate(record, metrics)
        if false_positive > float(self.config.get("ML_SIGNAL_MAX_FALSE_POSITIVE_RATE", 0.35) or 0.35):
            blockers.append("ml_decision_false_positive_rate_above_threshold")
        max_age_hours = float(
            self.config.get("ML_MAX_MODEL_AGE_HOURS", self.config.get("ML_OFFLINE_MAX_MODEL_AGE_HOURS", 72.0)) or 72.0
        )
        if max_age_hours > 0:
            created_at = record.created_at or datetime.utcnow()
            if created_at < datetime.utcnow() - timedelta(hours=max_age_hours):
                blockers.append("ml_decision_model_age_above_threshold")
        if not record.artifact_path or not Path(record.artifact_path).exists():
            blockers.append("ml_decision_artifact_missing")
        return {
            "ready": not blockers,
            "family": family_key,
            "blockers": blockers,
            "model": self._model_payload(record),
        }

    def _effective_false_positive_rate(self, record: MLOfflineModel, metrics: dict[str, Any]) -> float:
        rates: list[float] = []
        if "false_positive_rate" in metrics:
            rates.append(self._safe_float(metrics.get("false_positive_rate")))
        walk_forward = metrics.get("walk_forward")
        if (
            isinstance(walk_forward, dict)
            and "false_positive_rate" in walk_forward
            and walk_forward.get("prediction_source") == "model"
        ):
            rates.append(self._safe_float(walk_forward.get("false_positive_rate")))
        if not rates and record.negative_error_rate is not None:
            rates.append(self._safe_float(record.negative_error_rate))
        return max(rate for rate in rates if math.isfinite(rate))

    def promoted_model(self, family: str, horizon: str, *, provider: str = "global") -> MLOfflineModel | None:
        if not has_app_context():
            return None
        family_key = self._family(family)
        provider_key = normalize_provider(provider)
        horizon_key = self._horizon(horizon)
        record = (
            MLOfflineModel.query.filter_by(
                provider=provider_key,
                horizon=horizon_key,
                model_type=family_key,
                status="promoted",
            )
            .order_by(MLOfflineModel.promoted_at.desc(), MLOfflineModel.created_at.desc())
            .first()
        )
        if record is not None or provider_key == "global" or horizon_key != "1h10":
            return record
        return (
            MLOfflineModel.query.filter_by(
                provider="global",
                horizon=horizon_key,
                model_type=family_key,
                status="promoted",
            )
            .order_by(MLOfflineModel.promoted_at.desc(), MLOfflineModel.created_at.desc())
            .first()
        )

    def training_rows(
        self,
        family: str,
        horizon: str = "1h",
        *,
        objective: str = "risk_adjusted",
        use_market_history: bool = False,
        provider: str = "global",
    ) -> list[MLDecisionTrainingRow]:
        if not has_app_context():
            return []
        family_key = self._family(family)
        horizon_key = self._horizon(horizon)
        provider_key = normalize_provider(provider)
        if family_key == "pytorch_universe":
            rows = self._universe_training_rows(horizon_key)
        elif family_key == "pytorch_allocator":
            rows = self._allocator_training_rows(horizon_key)
        elif family_key == "pytorch_ops_anomaly":
            rows = self._ops_anomaly_training_rows(horizon_key)
        elif family_key == "pytorch_fibonacci":
            rows = self._fibonacci_training_rows(horizon_key)
        elif family_key == "pytorch_backtest_scorer":
            rows = self._backtest_scorer_training_rows(horizon_key)
        elif family_key == "pytorch_optimizer_policy":
            rows = self._optimizer_policy_training_rows(horizon_key)
        elif family_key == EXTREME_UPSIDE_FAMILY:
            rows = self._extreme_upside_training_rows(horizon_key)
        elif family_key in POLICY_FAMILIES:
            rows = self._policy_training_rows(family_key, horizon_key, objective=self._objective(objective))
        else:
            rows = []
        if horizon_key == "1h10" and family_key in {
            EXTREME_UPSIDE_FAMILY,
            "pytorch_universe",
            "pytorch_allocator",
            "pytorch_ops_anomaly",
            "pytorch_fibonacci",
            "pytorch_backtest_scorer",
            "pytorch_optimizer_policy",
            *POLICY_FAMILIES,
        }:
            rows.extend(self._one_h10_feature_training_rows(family_key, objective=self._objective(objective)))
        if use_market_history and family_key in {
            EXTREME_UPSIDE_FAMILY,
            "pytorch_universe",
            "pytorch_allocator",
            "pytorch_ops_anomaly",
            "pytorch_fibonacci",
            "pytorch_backtest_scorer",
            "pytorch_optimizer_policy",
            *POLICY_FAMILIES,
        }:
            rows.extend(
                self._market_history_training_rows(
                    family_key,
                    horizon_key,
                    objective=self._objective(objective),
                    provider=provider_key,
                )
            )
        if provider_key != "global":
            rows = [row for row in rows if normalize_provider(getattr(row, "provider", "global")) == provider_key]
        rows.sort(key=lambda row: row.created_at)
        return rows

    def _signal_readiness(self, horizon: str, *, provider: str = "global") -> dict[str, Any]:
        if self.signal_model is None:
            return {"ready": False, "blockers": ["ml_signal_model_service_unavailable"], "promoted_model": None}
        return dict(
            self.signal_model.readiness(
                horizon,
                require_promoted=bool(self.config.get("ML_SIGNAL_REQUIRE_PROMOTED", True)),
                provider=provider,
            )
        )

    def _signal_decision(
        self,
        context: dict[str, Any],
        horizon: str,
        candles: list[dict[str, Any]],
    ) -> MLDecisionEnvelope:
        if self.signal_model is None:
            return MLDecisionEnvelope(family=SIGNAL_FAMILY, blockers=["ml_signal_model_service_unavailable"], fallback_used=True)
        payload = dict(self.signal_model.score_payload(context, horizon, candles=candles))
        blockers = list(payload.get("blockers", []) or [])
        return MLDecisionEnvelope(
            family=SIGNAL_FAMILY,
            action=str(payload.get("action") or "hold"),
            confidence=self._safe_float(payload.get("confidence")),
            expected_return=self._safe_float(payload.get("expected_return")),
            uncertainty=max(0.0, min(1.0, 1.0 - self._safe_float(payload.get("confidence")))),
            model_id=self._safe_int(payload.get("model_id")),
            feature_schema_version=str(payload.get("feature_schema_version") or "ml_signal_v1"),
            blockers=blockers,
            fallback_used=not bool(payload.get("ready_for_live", False)),
            audit_metadata={
                "horizon": horizon,
                "status": payload.get("status"),
                "ready_for_live": bool(payload.get("ready_for_live", False)),
                "deterministic_safety_gates_required": True,
                "can_override_safety": False,
            },
            raw=payload,
        )

    def _selected_families(self, family: str) -> list[str]:
        family_key = str(family or "all").strip().lower()
        if family_key == "all":
            return list(MODEL_FAMILIES)
        return [self._family(family_key)]

    @staticmethod
    def _family(family: str | None) -> str:
        family_key = str(family or "").strip().lower()
        if family_key == SIGNAL_MODEL_TYPE:
            return SIGNAL_FAMILY
        if family_key in MODEL_FAMILIES:
            return family_key
        return family_key or SIGNAL_FAMILY

    @staticmethod
    def _horizon(horizon: str | None) -> str:
        return str(horizon or "1h").strip().lower() or "1h"

    @staticmethod
    def _module_available(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            return False

    @staticmethod
    def _default_action(family: str) -> str:
        return {
            SIGNAL_FAMILY: "hold",
            "pytorch_allocator": "hold",
            "pytorch_universe": "rank",
            "pytorch_ops_anomaly": "observe",
            EXTREME_UPSIDE_FAMILY: "avoid",
            "pytorch_fibonacci": "suggest",
            "pytorch_backtest_scorer": "score",
            "pytorch_optimizer_policy": "prioritize",
            "pytorch_risk_policy": "hold",
            "pytorch_exit_policy": "suggest",
            "pytorch_cap_policy": "cap",
            "pytorch_execution_policy": "route",
            "pytorch_roi_target": "score",
        }.get(family, "hold")

    def _action(self, family: str, context: dict[str, Any], confidence: float, *, prediction: float = 0.0) -> str:
        if family == "pytorch_universe":
            return "rank"
        if family == "pytorch_allocator":
            return "allocate" if prediction > 0 and confidence > 0 else "hold"
        if family == "pytorch_ops_anomaly":
            return "warn" if max(self._ops_anomaly_score(context), prediction) >= 0.7 else "observe"
        if family == EXTREME_UPSIDE_FAMILY:
            return "pursue" if prediction > 0 and confidence >= 0.60 else "avoid"
        if family == "pytorch_fibonacci":
            return "suggest" if prediction > 0 else "hold"
        if family == "pytorch_backtest_scorer":
            return "score"
        if family == "pytorch_optimizer_policy":
            threshold = self._safe_float(self.config.get("ML_OPTIMIZER_POLICY_SKIP_THRESHOLD"), -0.35)
            return "skip" if prediction < threshold and confidence >= 0.60 else "prioritize"
        if family == "pytorch_risk_policy":
            approve_threshold = self._safe_float(self.config.get("ML_RISK_POLICY_APPROVE_THRESHOLD"), 0.10)
            min_confidence = self._safe_float(self.config.get("ML_RISK_POLICY_MIN_CONFIDENCE"), 0.10)
            return "approve" if prediction >= approve_threshold and confidence >= min_confidence else "reject"
        if family == "pytorch_exit_policy":
            return "suggest" if prediction > -0.25 and confidence > 0 else "hold"
        if family == "pytorch_cap_policy":
            return "cap" if confidence > 0 else "hold"
        if family == "pytorch_execution_policy":
            return "route" if confidence > 0 else "hold"
        if family == "pytorch_roi_target":
            return "target_met_candidate" if prediction > 0 and confidence >= 0.50 else "target_unlikely"
        return "hold"

    def _confidence(
        self,
        context: dict[str, Any],
        record_payload: dict[str, Any],
        *,
        prediction: float | None = None,
    ) -> float:
        quality_multiplier = self._model_quality_multiplier(record_payload)
        if prediction is not None and math.isfinite(float(prediction)):
            return max(0.0, min(abs(float(prediction)) * quality_multiplier, 1.0))
        metrics = record_payload.get("metrics") if isinstance(record_payload.get("metrics"), dict) else {}
        metric_confidence = self._safe_float(metrics.get("confidence"), -1.0)
        if metric_confidence >= 0:
            return max(0.0, min(metric_confidence * quality_multiplier, 1.0))
        score = abs(self._score(context))
        quality = max(
            self._safe_float(context.get("expected_fill_quality")),
            self._safe_float(context.get("signal_stability")),
            self._safe_float(context.get("window_stability")),
        )
        return max(0.0, min((0.5 + min(score / 100.0, 0.3) + min(quality * 0.2, 0.2)) * quality_multiplier, 1.0))

    def _model_quality_multiplier(self, record_payload: dict[str, Any]) -> float:
        metrics = record_payload.get("metrics") if isinstance(record_payload.get("metrics"), dict) else {}
        if not metrics and not record_payload:
            return 1.0
        target_cap = max(1e-9, self._safe_float(self.config.get("ML_TARGET_CAP"), 1.0))
        max_loss = max(1e-9, self._safe_float(self.config.get("ML_SIGNAL_MAX_VALIDATION_LOSS"), 0.20))
        max_calibration = max(1e-9, self._safe_float(self.config.get("ML_OFFLINE_MAX_CALIBRATION_ERROR"), 0.18))
        validation_loss = self._safe_float(metrics.get("validation_loss"), self._safe_float(record_payload.get("validation_loss")))
        false_positive = self._safe_float(metrics.get("false_positive_rate"), self._safe_float(record_payload.get("negative_error_rate")))
        drift = self._safe_float(metrics.get("drift"), self._safe_float(record_payload.get("drift")))
        calibration_error = self._safe_float(metrics.get("calibration_error"))
        mean_absolute_error = self._safe_float(metrics.get("mean_absolute_error"))
        approval_precision = self._safe_float(metrics.get("approval_precision"), -1.0)
        penalty = (
            min(validation_loss / max_loss, 1.0) * 0.30
            + min(max(false_positive, 0.0), 1.0) * 0.25
            + min(abs(drift) / target_cap, 1.0) * 0.20
            + min(calibration_error / max_calibration, 1.0) * 0.15
            + min(mean_absolute_error / target_cap, 1.0) * 0.10
        )
        if approval_precision >= 0:
            penalty += min(max(1.0 - approval_precision, 0.0), 1.0) * 0.10
        return max(0.10, min(1.0 - penalty, 1.0))

    def _expected_return(
        self,
        context: dict[str, Any],
        record_payload: dict[str, Any],
        *,
        prediction: float | None = None,
    ) -> float:
        if prediction is not None and math.isfinite(float(prediction)):
            return float(prediction)
        metrics = record_payload.get("metrics") if isinstance(record_payload.get("metrics"), dict) else {}
        for key in ("expected_return", "mean_return", "target_return"):
            value = self._safe_float(metrics.get(key), math.nan)
            if math.isfinite(value):
                return value
        for key in ("expected_return", "net_return_after_costs", "recent_1h_return", "cost_adjusted_recent_1h_return"):
            value = self._safe_float(context.get(key), math.nan)
            if math.isfinite(value):
                return value
        return self._safe_float(context.get("cost_adjusted_expected_move")) / 10_000.0

    def _score(self, context: dict[str, Any]) -> float:
        for key in ("score", "upside_screen_score", "one_hour_edge_v2", "net_roi_v2_score", "net_roi_score"):
            value = self._safe_float(context.get(key), math.nan)
            if math.isfinite(value):
                return value
        return 0.0

    def _execution_style_suggestion(self, family: str, context: dict[str, Any]) -> str:
        if family != "pytorch_allocator":
            return ""
        spread = self._safe_float(context.get("spread_bps"))
        quality = self._safe_float(context.get("expected_fill_quality"), 1.0)
        if spread > 0 and spread <= float(self.config.get("VAULT_MAX_SPREAD_BPS", 25.0) or 25.0) and quality >= 0.75:
            return "maker_limit"
        return "risk_engine_default"

    @staticmethod
    def _ops_anomaly_score(context: dict[str, Any]) -> float:
        latency = MLDecisionEngine._safe_float(context.get("latency_ms"))
        error_rate = MLDecisionEngine._safe_float(context.get("error_rate"))
        rate_limited = 1.0 if bool(context.get("rate_limited", False)) else 0.0
        return max(0.0, min(latency / 5_000.0 + error_rate + rate_limited, 1.0))

    def _family_specific_raw(self, family: str, context: dict[str, Any], prediction: float) -> dict[str, Any]:
        if family == EXTREME_UPSIDE_FAMILY:
            return self._extreme_upside_payload(context, prediction)
        if family == "pytorch_fibonacci":
            return self._fibonacci_decision_payload(context, prediction)
        if family == "pytorch_backtest_scorer":
            return {
                "backtest_edge_prediction": prediction,
                "backtest_score_weight": self._safe_float(self.config.get("ML_BACKTEST_SCORER_WEIGHT"), 0.10),
            }
        if family == "pytorch_optimizer_policy":
            threshold = self._safe_float(self.config.get("ML_OPTIMIZER_POLICY_SKIP_THRESHOLD"), -0.35)
            return {
                "optimizer_policy_score": prediction,
                "skip_candidate": bool(prediction < threshold),
                "skip_reason": "ml_optimizer_policy_low_edge" if prediction < threshold else "",
            }
        if family == "pytorch_risk_policy":
            return self._risk_policy_payload(context, prediction)
        if family == "pytorch_exit_policy":
            return self._exit_policy_payload(context, prediction)
        if family == "pytorch_cap_policy":
            return self._cap_policy_payload(context, prediction)
        if family == "pytorch_execution_policy":
            return self._execution_policy_payload(context, prediction)
        if family == "pytorch_roi_target":
            return self._roi_target_payload(context, prediction)
        return {}

    def _extreme_upside_payload(self, context: dict[str, Any], prediction: float) -> dict[str, Any]:
        target_roi_pct = max(1.0, self._safe_float(self.config.get("ML_EXTREME_UPSIDE_TARGET_ROI_PCT"), 1000.0))
        target_return = target_roi_pct / 100.0
        expected_return = self._safe_float(
            context.get("net_return_after_costs"),
            self._safe_float(context.get("expected_return"), self._safe_float(context.get("recent_1h_return"))),
        )
        projected_roi_pct = max(prediction * target_roi_pct, expected_return * 100.0)
        distance_to_target_pct = max(0.0, target_roi_pct - projected_roi_pct)
        probability = max(0.0, min((prediction + 1.0) / 2.0, 1.0))
        allocation_budget = max(
            0.0,
            self._safe_float(context.get("allocation_budget_usdc"), self._safe_float(context.get("allocation_amount_usd"), 0.0)),
        )
        leverage_ceiling = max(1.0, self._safe_float(context.get("hard_max_leverage"), 1.0))
        cap_scale = max(0.05, min(probability, 1.0))
        suggested_notional = allocation_budget * cap_scale if allocation_budget > 0 else 0.0
        leverage_suggestion = min(leverage_ceiling, max(1.0, 1.0 + max(prediction, 0.0) * 0.5))
        return {
            "objective": "extreme_upside",
            "target_roi_pct": target_roi_pct,
            "target_return": target_return,
            "extreme_upside_probability": probability,
            "projected_roi_pct": projected_roi_pct,
            "distance_to_target_pct": distance_to_target_pct,
            "suggested_notional_usdc": suggested_notional,
            "suggested_leverage": leverage_suggestion,
            "suggested_risk_pct": max(0.0, min(probability * 0.01, 0.01)),
            "suggested_hold_duration_hours": max(1.0, self._safe_float(context.get("lock_duration_hours"), 1.0)),
            "dynamic_cap_policy": "suggestions_are_clipped_by_operator_caps_and_risk_engine",
        }

    def _fibonacci_decision_payload(self, context: dict[str, Any], prediction: float) -> dict[str, Any]:
        confidence = max(0.0, min(abs(prediction), 1.0))
        min_confidence = self._safe_float(
            self.config.get("ONE_H10_FIBONACCI_MIN_CONFIDENCE")
            if str(context.get("horizon") or context.get("ml_horizon") or "").lower() == "1h10"
            else self.config.get("ML_MIN_FIB_CONFIDENCE"),
            0.55,
        )
        confluence = context.get("fibonacci_confluence") if isinstance(context.get("fibonacci_confluence"), dict) else {}
        quality = max(confidence, self._safe_float(confluence.get("score")))
        stop_pct = max(0.0025, min(0.05, 0.004 + (1.0 - quality) * 0.012))
        take_pct = max(stop_pct * 1.5, min(0.12, stop_pct * (2.0 + quality)))
        if confidence < min_confidence:
            return {
                "target_zone_quality": quality,
                "invalidation_distance_pct": 0.0,
                "suggested_stop_loss_pct": 0.0,
                "suggested_take_profit_pct": 0.0,
                "blockers": ["ml_fibonacci_confidence_below_minimum"],
            }
        return {
            "target_zone_quality": quality,
            "invalidation_distance_pct": stop_pct,
            "suggested_stop_loss_pct": stop_pct,
            "suggested_take_profit_pct": take_pct,
            "nearest_support": confluence.get("nearest_support"),
            "nearest_resistance": confluence.get("nearest_resistance"),
        }

    def _risk_policy_payload(self, context: dict[str, Any], prediction: float) -> dict[str, Any]:
        confidence = max(0.0, min(abs(prediction), 1.0))
        approve_threshold = self._safe_float(self.config.get("ML_RISK_POLICY_APPROVE_THRESHOLD"), 0.10)
        min_confidence = self._safe_float(self.config.get("ML_RISK_POLICY_MIN_CONFIDENCE"), 0.10)
        hard_cap = max(0.0, self._safe_float(context.get("ml_live_hard_cap_usdc"), self.config.get("ML_LIVE_HARD_CAP_USDC", 10.0)))
        notional = max(0.0, self._safe_float(context.get("notional")))
        liquidation_buffer = self._safe_float(context.get("liquidation_buffer_pct"))
        liquidation_risk = max(0.0, min(1.0 - max(liquidation_buffer, 0.0), 1.0))
        return {
            "policy": "ml_risk_policy",
            "approve": bool(prediction >= approve_threshold and confidence >= min_confidence),
            "approve_probability": max(0.0, min(prediction, 1.0)),
            "approve_threshold": approve_threshold,
            "min_confidence": min_confidence,
            "risk_budget_usdc": min(value for value in [hard_cap, notional] if value > 0) if hard_cap > 0 or notional > 0 else 0.0,
            "liquidation_risk": liquidation_risk,
            "confidence": confidence,
            "hard_cap_usdc": hard_cap,
            "notional_usdc": notional,
        }

    def _exit_policy_payload(self, context: dict[str, Any], prediction: float) -> dict[str, Any]:
        confidence = max(0.0, min(abs(prediction), 1.0))
        base_stop = max(self._safe_float(context.get("stop_loss_pct")), self._safe_float(context.get("fallback_stop_loss_pct")), 0.003)
        base_take = max(self._safe_float(context.get("take_profit_pct")), self._safe_float(context.get("fallback_take_profit_pct")), base_stop * 1.5)
        stop_pct = max(0.001, min(0.25, base_stop * (1.0 + max(-prediction, 0.0))))
        take_pct = max(stop_pct * 1.2, min(2.0, base_take * (1.0 + max(prediction, 0.0))))
        return {
            "policy": "ml_exit_policy",
            "suggested_stop_loss_pct": stop_pct,
            "suggested_take_profit_pct": take_pct,
            "trailing_exit_pct": max(0.0, min(stop_pct * 0.75, 0.20)),
            "timeout_exit_hours": max(1.0, self._safe_float(context.get("lock_duration_hours"), 1.0)),
            "target_zone_quality": confidence,
            "invalidation_distance_pct": stop_pct,
            "blockers": [] if stop_pct > 0 and take_pct > 0 else ["ml_exit_policy_invalid_exit"],
        }

    def _cap_policy_payload(self, context: dict[str, Any], prediction: float) -> dict[str, Any]:
        hard_cap = max(0.0, self._safe_float(context.get("ml_live_hard_cap_usdc"), self.config.get("ML_LIVE_HARD_CAP_USDC", 10.0)))
        hard_daily_loss = max(
            0.0,
            self._safe_float(context.get("ml_live_hard_daily_loss_usdc"), self.config.get("ML_LIVE_HARD_DAILY_LOSS_USDC", 0.50)),
        )
        requested = max(0.0, self._safe_float(context.get("allocation_amount_usd"), self._safe_float(context.get("notional"))))
        scale = max(0.05, min((prediction + 1.0) / 2.0, 1.0))
        suggested_notional = requested * scale if requested > 0 else hard_cap * scale
        if hard_cap > 0:
            suggested_notional = min(suggested_notional, hard_cap)
        max_leverage = max(1.0, self._safe_float(context.get("exchange_max_leverage"), self._safe_float(context.get("hard_max_leverage"), 1.0)))
        leverage = min(max_leverage, max(1.0, 1.0 + max(prediction, 0.0) * max(max_leverage - 1.0, 0.0)))
        return {
            "policy": "ml_cap_policy",
            "suggested_notional_usdc": suggested_notional,
            "suggested_leverage": leverage,
            "suggested_daily_loss_usdc": min(hard_daily_loss, max(0.0, suggested_notional * 0.05)),
            "suggested_active_cycles": 1,
            "suggested_cooldown_seconds": 0 if prediction > 0 else 900,
            "hard_cap_usdc": hard_cap,
            "hard_daily_loss_usdc": hard_daily_loss,
            "clipped_by_hard_caps": hard_cap > 0,
        }

    def _execution_policy_payload(self, context: dict[str, Any], prediction: float) -> dict[str, Any]:
        spread_bps = self._safe_float(context.get("spread_bps"))
        fill_quality = self._safe_float(context.get("expected_fill_quality"), 0.0)
        maker = prediction > 0 and (spread_bps <= 25.0 or spread_bps <= 0) and fill_quality >= 0.5
        limit_offset_bps = max(1.0, min(max(spread_bps / 4.0, 1.0), 20.0))
        return {
            "policy": "ml_execution_policy",
            "order_type_suggestion": "limit" if maker else "market",
            "maker_taker_preference": "maker" if maker else "taker",
            "limit_offset_bps": limit_offset_bps,
            "slippage_tolerance_pct": max(0.0, min(limit_offset_bps / 10_000.0, 0.05)),
            "retry_policy": "no_retry",
        }

    def _roi_target_payload(self, context: dict[str, Any], prediction: float) -> dict[str, Any]:
        target_roi_pct = self._target_roi_pct(context.get("objective"), context.get("horizon"))
        expected_return = self._safe_float(
            context.get("net_return_after_costs"),
            self._safe_float(context.get("expected_return"), self._safe_float(context.get("recent_1h_return"))),
        )
        projected_roi_pct = max(prediction * target_roi_pct, expected_return * 100.0)
        probability = max(0.0, min((prediction + 1.0) / 2.0, 1.0))
        return {
            "policy": "ml_roi_target_policy",
            "objective": self._objective(str(context.get("objective") or "")),
            "target_roi_pct": target_roi_pct,
            "target_return": target_roi_pct / 100.0,
            "target_probability": probability,
            "projected_roi_pct": projected_roi_pct,
            "distance_to_target_pct": max(0.0, target_roi_pct - projected_roi_pct),
        }

    def _universe_training_rows(self, horizon: str) -> list[MLDecisionTrainingRow]:
        rows: list[MLDecisionTrainingRow] = []
        for ranking in StrategyRanking.query.order_by(StrategyRanking.created_at.asc(), StrategyRanking.id.asc()).all():
            if horizon_from_duration(ranking.lock_duration_hours or 1) != horizon:
                continue
            payload = self._ranking_payload(ranking)
            features = self._features(payload)
            rows.append(
                MLDecisionTrainingRow(
                    features=features,
                    target=self._bounded_target(outcome_from_result(payload)),
                    created_at=ranking.created_at or datetime.utcnow(),
                    source="strategy_ranking",
                    provider=normalize_provider(getattr(ranking, "provider", "global")),
                )
            )
        events = (
            MLTrainingEvent.query.filter_by(horizon=horizon)
            .order_by(MLTrainingEvent.created_at.asc(), MLTrainingEvent.id.asc())
            .all()
        )
        for event in events:
            details = event.details if isinstance(event.details, dict) else {}
            event_provider = normalize_provider(getattr(event, "provider", None) or details.get("provider") or (event.features or {}).get("provider"))
            payload = {**provider_feature_context(event_provider), **details, **dict(event.features or {})}
            rows.append(
                MLDecisionTrainingRow(
                    features=self._features(payload),
                    target=self._bounded_target(self._safe_float(event.outcome)),
                    created_at=event.created_at or datetime.utcnow(),
                    source=f"training_event:{event.source}",
                    provider=event_provider,
                )
            )
        return rows

    def _fibonacci_training_rows(self, horizon: str) -> list[MLDecisionTrainingRow]:
        rows: list[MLDecisionTrainingRow] = []
        for ranking in StrategyRanking.query.order_by(StrategyRanking.created_at.asc(), StrategyRanking.id.asc()).all():
            if horizon_from_duration(ranking.lock_duration_hours or 1) != horizon:
                continue
            payload = self._ranking_payload(ranking)
            payload["ml_feature_schema_version"] = ML_FEATURE_SCHEMA_VERSION
            rows.append(
                MLDecisionTrainingRow(
                    features=self._features(payload),
                    target=self._bounded_target(outcome_from_result(payload)),
                    created_at=ranking.created_at or datetime.utcnow(),
                    source="strategy_ranking:fibonacci",
                    provider=normalize_provider(getattr(ranking, "provider", "global")),
                )
            )
        for run in BacktestRun.query.order_by(BacktestRun.created_at.asc(), BacktestRun.id.asc()).limit(2_000).all():
            result = run.result if isinstance(run.result, dict) else {}
            trades = result.get("trades") if isinstance(result.get("trades"), list) else []
            for trade in trades:
                if not isinstance(trade, dict) or not trade.get("fibonacci_levels"):
                    continue
                payload = self.feature_factory.build(
                    symbol=run.symbol,
                    timeframe=run.timeframe,
                    optimizer_context={"strategy_name": run.strategy_name},
                    backtest_result={**result, **trade},
                )
                rows.append(
                    MLDecisionTrainingRow(
                        features=self._features(payload),
                        target=self._bounded_target(self._safe_float(trade.get("return"))),
                        created_at=run.created_at or datetime.utcnow(),
                        source="backtest_trade:fibonacci",
                    )
                )
        return rows

    def _one_h10_feature_training_rows(self, family: str, *, objective: str) -> list[MLDecisionTrainingRow]:
        """Bootstrap 1H10 ML families from persisted high-timeframe market features.

        These rows are separated under the 1h10 horizon and are replaced by
        outcome-backed feedback as live/paper fills accumulate. They let the
        1H10 policy learn relative opportunity quality from Fibonacci,
        volatility, liquidity, spread, funding, and trend feature distributions
        without borrowing authority from the generic 1h namespace.
        """

        rows: list[MLDecisionTrainingRow] = []
        objective_key = self._objective(objective)
        query = LeveragedMarketFeature.query.order_by(
            LeveragedMarketFeature.updated_at.asc(),
            LeveragedMarketFeature.id.asc(),
        ).limit(5_000)
        for row in query.all():
            features = row.features if isinstance(row.features, dict) else {}
            if not features:
                continue
            provider_key = normalize_provider(row.provider)
            payload = {
                **provider_feature_context(provider_key),
                **features,
                "provider": provider_key,
                "execution_venue": provider_key,
                "symbol": row.symbol,
                "timeframe": row.timeframe,
                "horizon": "1h10",
                "ml_horizon": "1h10",
                "objective": "one_h10",
                "one_h10_vault": True,
                "target_roi_pct": self._target_roi_pct(objective_key, "1h10"),
                "feature_schema_version": row.feature_schema_version,
            }
            target = self._one_h10_feature_target(payload, family)
            rows.append(
                MLDecisionTrainingRow(
                    features=self._features(payload),
                    target=target,
                    created_at=row.updated_at or row.created_at or datetime.utcnow(),
                    source=f"one_h10_feature_bootstrap:{family}",
                    provider=provider_key,
                )
            )
        return rows

    def _one_h10_feature_target(self, payload: dict[str, Any], family: str) -> float:
        confluence = payload.get("fibonacci_confluence") if isinstance(payload.get("fibonacci_confluence"), dict) else {}
        timing = payload.get("fibonacci_timing") if isinstance(payload.get("fibonacci_timing"), dict) else {}
        fib_score = self._safe_float(confluence.get("score"))
        range_position = self._safe_float(timing.get("range_position"), 0.5)
        trend = self._safe_float(payload.get("trend_strength"))
        ema_trend = self._safe_float(payload.get("ema_trend")) / max(abs(self._safe_float(payload.get("close"), 1.0)), 1.0)
        macd = self._safe_float(payload.get("macd_histogram")) / max(abs(self._safe_float(payload.get("close"), 1.0)), 1.0)
        rsi = self._safe_float(payload.get("rsi"), 50.0)
        atr = self._safe_float(payload.get("atr_pct"))
        volatility = max(self._safe_float(payload.get("volatility")), atr)
        liquidity = min(max(self._safe_float(payload.get("liquidity_usd")) / 1_000_000.0, 0.0), 1.0)
        depth = min(max(self._safe_float(payload.get("order_book_depth_usd")) / 250_000.0, 0.0), 1.0)
        spread_penalty = min(max(self._safe_float(payload.get("spread_bps")) / 20.0, 0.0), 1.0)
        funding_penalty = min(abs(self._safe_float(payload.get("funding_rate"))) / 0.001, 1.0)
        imbalance = self._safe_float(payload.get("order_book_imbalance"))
        directional = (trend * 4.0) + (ema_trend * 40.0) + (macd * 30.0) + (imbalance * 0.25)
        if rsi < 35:
            directional += 0.35
        elif rsi > 72:
            directional -= 0.35
        if range_position < 0.382:
            directional += fib_score * 0.2
        elif range_position > 0.786:
            directional -= fib_score * 0.2
        opportunity = abs(directional) + min(fib_score, 2.0) * 0.35 + min(volatility / 0.02, 1.0) * 0.25
        quality = liquidity * 0.25 + depth * 0.20 + max(0.0, 1.0 - spread_penalty) * 0.30 + max(0.0, 1.0 - funding_penalty) * 0.10
        raw = opportunity + quality - spread_penalty * 0.5 - funding_penalty * 0.15
        if family == "pytorch_risk_policy":
            raw = raw - max(0.0, volatility - 0.04) * 8.0 - spread_penalty * 0.5
        elif family == "pytorch_exit_policy":
            raw = abs(directional) + min(volatility / 0.03, 1.0) + min(fib_score, 2.0) * 0.2
        elif family == "pytorch_cap_policy":
            raw = min(liquidity + depth, 1.5) + max(0.0, 1.0 - spread_penalty) + min(abs(directional), 1.0)
        elif family == "pytorch_execution_policy":
            raw = max(0.0, 1.0 - spread_penalty) + depth + liquidity - funding_penalty * 0.2
        elif family == "pytorch_roi_target":
            raw = opportunity * 0.7 + min(volatility / 0.02, 1.0) * 0.5 + min(fib_score, 2.0) * 0.3
        elif family == "pytorch_universe":
            raw = opportunity + liquidity + depth - spread_penalty
        elif family == "pytorch_allocator":
            raw = min(liquidity + depth, 1.5) + min(abs(directional), 1.0) + max(0.0, 1.0 - spread_penalty)
        elif family == "pytorch_ops_anomaly":
            raw = 1.0 - min(spread_penalty + funding_penalty + max(0.0, volatility - 0.05) * 10.0, 1.0)
        elif family == EXTREME_UPSIDE_FAMILY:
            raw = opportunity * 0.8 + min(volatility / 0.02, 1.0) * 0.4 + min(fib_score, 2.0) * 0.3
        return self._bounded_target(raw / 3.0)

    def _backtest_scorer_training_rows(self, horizon: str) -> list[MLDecisionTrainingRow]:
        rows: list[MLDecisionTrainingRow] = []
        for run in BacktestRun.query.order_by(BacktestRun.created_at.asc(), BacktestRun.id.asc()).limit(2_000).all():
            result = run.result if isinstance(run.result, dict) else {}
            if not result:
                continue
            payload = self.feature_factory.build(
                symbol=run.symbol,
                timeframe=run.timeframe,
                optimizer_context={"strategy_name": run.strategy_name},
                backtest_result=result,
            )
            rows.append(
                MLDecisionTrainingRow(
                    features=self._features(payload),
                    target=self._bounded_target(outcome_from_result(result)),
                    created_at=run.created_at or datetime.utcnow(),
                    source="backtest_run",
                )
            )
        for ranking in StrategyRanking.query.order_by(StrategyRanking.created_at.asc(), StrategyRanking.id.asc()).all():
            if horizon_from_duration(ranking.lock_duration_hours or 1) != horizon:
                continue
            payload = self._ranking_payload(ranking)
            rows.append(
                MLDecisionTrainingRow(
                    features=self._features(payload),
                    target=self._bounded_target(outcome_from_result(payload)),
                    created_at=ranking.created_at or datetime.utcnow(),
                    source="strategy_ranking:backtest_scorer",
                    provider=normalize_provider(getattr(ranking, "provider", "global")),
                )
            )
        return rows

    def _optimizer_policy_training_rows(self, horizon: str) -> list[MLDecisionTrainingRow]:
        rows: list[MLDecisionTrainingRow] = []
        for ranking in StrategyRanking.query.order_by(StrategyRanking.created_at.asc(), StrategyRanking.id.asc()).all():
            if horizon_from_duration(ranking.lock_duration_hours or 1) != horizon:
                continue
            payload = self._ranking_payload(ranking)
            target = outcome_from_result(payload)
            if ranking.rejected:
                target = min(target, -0.25)
            rows.append(
                MLDecisionTrainingRow(
                    features=self._features(payload),
                    target=self._bounded_target(target),
                    created_at=ranking.created_at or datetime.utcnow(),
                    source="strategy_ranking:optimizer_policy",
                    provider=normalize_provider(getattr(ranking, "provider", "global")),
                )
            )
        return rows

    def _extreme_upside_training_rows(self, horizon: str) -> list[MLDecisionTrainingRow]:
        rows: list[MLDecisionTrainingRow] = []
        target_return = max(0.01, self._safe_float(self.config.get("ML_EXTREME_UPSIDE_TARGET_ROI_PCT"), 1000.0) / 100.0)
        for ranking in StrategyRanking.query.order_by(StrategyRanking.created_at.asc(), StrategyRanking.id.asc()).all():
            if horizon_from_duration(ranking.lock_duration_hours or 1) != horizon:
                continue
            payload = self._ranking_payload(ranking)
            payload.update(
                {
                    "objective": "extreme_upside",
                    "target_roi_pct": self.config.get("ML_EXTREME_UPSIDE_TARGET_ROI_PCT", 1000.0),
                }
            )
            outcome = outcome_from_result(payload)
            target = (outcome / target_return) if target_return > 0 else outcome
            if ranking.rejected:
                target = min(target, -0.35)
            rows.append(
                MLDecisionTrainingRow(
                    features=self._features(payload),
                    target=self._bounded_target(target),
                    created_at=ranking.created_at or datetime.utcnow(),
                    source="strategy_ranking:extreme_upside",
                    provider=normalize_provider(getattr(ranking, "provider", "global")),
                )
            )
        for run in BacktestRun.query.order_by(BacktestRun.created_at.asc(), BacktestRun.id.asc()).limit(2_000).all():
            result = run.result if isinstance(run.result, dict) else {}
            if not result:
                continue
            payload = self.feature_factory.build(
                symbol=run.symbol,
                timeframe=run.timeframe,
                optimizer_context={
                    "strategy_name": run.strategy_name,
                    "objective": "extreme_upside",
                    "target_roi_pct": self.config.get("ML_EXTREME_UPSIDE_TARGET_ROI_PCT", 1000.0),
                },
                backtest_result=result,
                trade_outcomes=result.get("trades") if isinstance(result.get("trades"), list) else [],
            )
            outcome = outcome_from_result(result)
            rows.append(
                MLDecisionTrainingRow(
                    features=self._features(payload),
                    target=self._bounded_target(outcome / target_return),
                    created_at=run.created_at or datetime.utcnow(),
                    source="backtest_run:extreme_upside",
                )
            )
        return rows

    def _allocator_training_rows(self, horizon: str) -> list[MLDecisionTrainingRow]:
        rows: list[MLDecisionTrainingRow] = []
        completed_statuses = {"completed", "settled", "limited", "failed", "error", "expired"}
        cycles = VaultCycle.query.order_by(VaultCycle.created_at.asc(), VaultCycle.id.asc()).all()
        for cycle in cycles:
            if horizon_from_duration(cycle.lock_duration_hours or 1) != horizon:
                continue
            if str(cycle.status or "").lower() not in completed_statuses and cycle.settled_at is None:
                continue
            cycle_return = self._cycle_return(cycle)
            for leg in list(cycle.allocation_legs or []):
                payload = self._allocation_leg_payload(cycle, leg)
                target = self._allocation_leg_target(cycle, leg, cycle_return)
                rows.append(
                    MLDecisionTrainingRow(
                    features=self._features(payload),
                    target=target,
                    created_at=leg.updated_at or leg.created_at or cycle.updated_at or datetime.utcnow(),
                    source="vault_allocation_leg",
                    provider=normalize_provider(payload.get("provider") or payload.get("execution_venue")),
                )
            )
        return rows

    def _ops_anomaly_training_rows(self, horizon: str) -> list[MLDecisionTrainingRow]:
        rows: list[MLDecisionTrainingRow] = []
        for audit in AuditLog.query.order_by(AuditLog.created_at.asc(), AuditLog.id.asc()).limit(2_000).all():
            payload = self._ops_payload_from_audit(audit)
            rows.append(
                MLDecisionTrainingRow(
                    features=self._features(payload),
                    target=self._ops_target(payload),
                    created_at=audit.created_at or datetime.utcnow(),
                    source=f"audit:{audit.action}",
                )
            )
        for event in RiskEvent.query.order_by(RiskEvent.created_at.asc(), RiskEvent.id.asc()).limit(2_000).all():
            payload = self._ops_payload_from_risk_event(event)
            rows.append(
                MLDecisionTrainingRow(
                    features=self._features(payload),
                    target=1.0,
                    created_at=event.created_at or datetime.utcnow(),
                    source=f"risk_event:{event.rule_name}",
                )
            )
        for setting in Setting.query.filter(Setting.key.like("connection_health:%")).all():
            payload = self._ops_payload_from_health_setting(setting)
            rows.append(
                MLDecisionTrainingRow(
                    features=self._features(payload),
                    target=0.0 if bool(payload.get("can_trade", False)) else 1.0,
                    created_at=setting.updated_at or datetime.utcnow(),
                    source="connection_health",
                )
            )
        return rows

    def _policy_training_rows(self, family: str, horizon: str, *, objective: str) -> list[MLDecisionTrainingRow]:
        rows: list[MLDecisionTrainingRow] = []
        objective_key = self._objective(objective)
        target_roi_pct = self._target_roi_pct(objective_key, horizon)
        target_return = max(0.01, target_roi_pct / 100.0)

        for ranking in StrategyRanking.query.order_by(StrategyRanking.created_at.asc(), StrategyRanking.id.asc()).all():
            if horizon_from_duration(ranking.lock_duration_hours or 1) != horizon:
                continue
            payload = self._ranking_payload(ranking)
            payload.update(
                {
                    "policy_family": family,
                    "objective": objective_key,
                    "target_roi_pct": target_roi_pct,
                    "ml_live_hard_cap_usdc": self.config.get("ML_LIVE_HARD_CAP_USDC", 10.0),
                    "ml_live_hard_daily_loss_usdc": self.config.get("ML_LIVE_HARD_DAILY_LOSS_USDC", 0.50),
                    "hard_max_leverage": self.config.get("MAX_LEVERAGE", 1.0),
                }
            )
            outcome = outcome_from_result(payload)
            if family == "pytorch_roi_target":
                outcome = outcome / target_return
            if family == "pytorch_risk_policy":
                outcome = 0.0 if bool(ranking.rejected) or str(getattr(ranking, "rejection_reason", "") or "").strip() else 1.0
            elif ranking.rejected:
                outcome = min(outcome, -0.35)
            rows.append(
                MLDecisionTrainingRow(
                    features=self._features(payload),
                    target=self._bounded_target(outcome),
                    created_at=ranking.created_at or datetime.utcnow(),
                    source=f"strategy_ranking:{family}",
                    provider=normalize_provider(getattr(ranking, "provider", "global")),
                )
            )

        for run in BacktestRun.query.order_by(BacktestRun.created_at.asc(), BacktestRun.id.asc()).limit(2_000).all():
            result = run.result if isinstance(run.result, dict) else {}
            if not result:
                continue
            payload = self.feature_factory.build(
                symbol=run.symbol,
                timeframe=run.timeframe,
                optimizer_context={
                    "strategy_name": run.strategy_name,
                    "policy_family": family,
                    "objective": objective_key,
                    "target_roi_pct": target_roi_pct,
                    "ml_live_hard_cap_usdc": self.config.get("ML_LIVE_HARD_CAP_USDC", 10.0),
                    "ml_live_hard_daily_loss_usdc": self.config.get("ML_LIVE_HARD_DAILY_LOSS_USDC", 0.50),
                    "hard_max_leverage": self.config.get("MAX_LEVERAGE", 1.0),
                },
                backtest_result=result,
                trade_outcomes=result.get("trades") if isinstance(result.get("trades"), list) else [],
            )
            outcome = outcome_from_result(result)
            if family == "pytorch_roi_target":
                outcome = outcome / target_return
            if family == "pytorch_risk_policy":
                outcome = self._risk_policy_backtest_target(result)
            rows.append(
                MLDecisionTrainingRow(
                    features=self._features(payload),
                    target=self._bounded_target(outcome),
                    created_at=run.created_at or datetime.utcnow(),
                    source=f"backtest_run:{family}",
                )
            )

        if family == "pytorch_risk_policy":
            for event in RiskEvent.query.order_by(RiskEvent.created_at.asc(), RiskEvent.id.asc()).limit(2_000).all():
                payload = self._ops_payload_from_risk_event(event)
                payload.update({"policy_family": family, "objective": objective_key, "target_roi_pct": target_roi_pct})
                rows.append(
                    MLDecisionTrainingRow(
                        features=self._features(payload),
                        target=-1.0,
                        created_at=event.created_at or datetime.utcnow(),
                        source=f"risk_event:{event.rule_name}",
                    )
                )

        return rows

    def _market_history_training_rows(
        self,
        family: str,
        horizon: str,
        *,
        objective: str,
        provider: str = "global",
    ) -> list[MLDecisionTrainingRow]:
        rows: list[MLDecisionTrainingRow] = []
        provider_key = normalize_provider(provider)
        objective_key = self._objective(objective)
        target_roi_pct = self._target_roi_pct(objective_key, horizon)
        target_return = max(0.01, target_roi_pct / 100.0)
        query = MLMarketHistory.query.filter_by(status="ok")
        if provider_key != "global":
            query = query.filter(MLMarketHistory.provider == provider_key)
        query = query.order_by(
            MLMarketHistory.window_end.asc(),
            MLMarketHistory.fetched_at.asc(),
            MLMarketHistory.id.asc(),
        ).limit(5_000)
        max_rows = max(1, int(self.config.get("ML_OFFLINE_MARKET_HISTORY_MAX_ROWS", 50_000) or 50_000))
        samples_per_window = max(1, int(self.config.get("ML_OFFLINE_MARKET_HISTORY_SAMPLES_PER_WINDOW", 250) or 250))
        horizon_minutes = self._horizon_minutes(horizon)
        for history in query.all():
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
            diagnostics = history.diagnostics if isinstance(history.diagnostics, dict) else {}
            venue_symbol = str(diagnostics.get("venue_symbol") or history.symbol) if isinstance(diagnostics, dict) else history.symbol
            for cutoff_index in range(min_window, last_cutoff + 1, stride):
                if len(rows) >= max_rows:
                    break
                feature_start = max(0, cutoff_index - 240)
                feature_window = candles[feature_start : cutoff_index + 1]
                cutoff = feature_window[-1]
                cutoff_ts = cutoff.get("timestamp")
                cutoff_close = self._safe_float(cutoff.get("close"))
                future_close = self._safe_float(candles[cutoff_index + forward_steps].get("close"))
                if cutoff_close <= 0 or future_close <= 0:
                    continue
                forward_return = (future_close - cutoff_close) / cutoff_close
                target = forward_return
                if family in {EXTREME_UPSIDE_FAMILY, "pytorch_roi_target"} or objective_key in {
                    "extreme_upside",
                    "extreme_roi_1h",
                    "consistent_roi_1w",
                }:
                    target = forward_return / target_return
                if family in {"pytorch_optimizer_policy", "pytorch_risk_policy"} and forward_return <= 0:
                    target = min(target, -0.25)
                if family == "pytorch_risk_policy":
                    target = self._risk_policy_market_target(feature_window)
                provider_row = normalize_provider(history.provider)
                payload = self.feature_factory.build(
                    symbol=history.symbol,
                    timeframe=history.timeframe,
                    candles=feature_window,
                    optimizer_context={
                        **provider_feature_context(provider_row),
                        "strategy_name": "ml_market_history",
                        "provider": provider_row,
                        "execution_venue": provider_row,
                        "venue_symbol": venue_symbol,
                        "objective": objective_key,
                        "target_roi_pct": target_roi_pct,
                        "policy_family": family if family in POLICY_FAMILIES else "",
                        "ml_live_hard_cap_usdc": self.config.get("ML_LIVE_HARD_CAP_USDC", 10.0),
                        "ml_live_hard_daily_loss_usdc": self.config.get("ML_LIVE_HARD_DAILY_LOSS_USDC", 0.50),
                    },
                    cutoff_timestamp=cutoff_ts,
                )
                rows.append(
                    MLDecisionTrainingRow(
                        features=self._features(payload),
                        target=self._bounded_target(target),
                        created_at=self._history_sample_time(cutoff, history),
                        source=f"ml_market_history:{family}",
                        provider=provider_row,
                    )
                )
        return rows

    def _features(self, payload: dict[str, Any]) -> dict[str, float]:
        return self.online_ranker.normalized_features(extract_features(payload))

    def _ranking_payload(self, ranking: StrategyRanking) -> dict[str, Any]:
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
            "turnover_rate": ranking.turnover_rate,
            "allocation_amount_usd": ranking.allocation_amount_usd,
            "lock_duration_hours": ranking.lock_duration_hours,
            "leverage": ranking.leverage,
            "liquidation_buffer_pct": ranking.liquidation_buffer_pct,
            "capacity_usd": ranking.capacity_usd,
            "convex_edge_score": ranking.convex_edge_score,
            "mfe_mae_ratio": ranking.mfe_mae_ratio,
            "ml_score": ranking.ml_score,
            "ml_adjusted_score": ranking.ml_adjusted_score,
            "rejected": bool(ranking.rejected),
            "rejection_reason": ranking.rejection_reason or "",
        }
        payload.update({key: value for key, value in net_roi.items() if key not in payload})
        payload.update({key: value for key, value in net_roi_v2.items() if key not in payload})
        return payload

    def _allocation_leg_payload(self, cycle: VaultCycle, leg: VaultAllocationLeg) -> dict[str, Any]:
        details = leg.details if isinstance(leg.details, dict) else {}
        ranking = leg.optimizer_ranking
        ranking_payload = self._ranking_payload(ranking) if ranking is not None else {}
        payload = {
            **ranking_payload,
            **details,
            "strategy_name": details.get("strategy_name") or ranking_payload.get("strategy_name") or cycle.selected_strategy_name or "",
            "symbol": leg.symbol,
            "timeframe": leg.timeframe,
            "profile": cycle.algorithm_profile,
            "optimizer_profile": cycle.algorithm_profile,
            "lock_duration_hours": cycle.lock_duration_hours,
            "duration_hours": cycle.lock_duration_hours,
            "starting_value_usd": cycle.starting_value_usd,
            "allocation_amount_usd": leg.allocation_cap_usd,
            "allocation_cap_usd": leg.allocation_cap_usd,
            "leverage": leg.leverage,
            "realized_pnl_usd": leg.realized_pnl_usd,
            "unrealized_pnl_usd": leg.unrealized_pnl_usd,
            "cycle_status": cycle.status,
            "leg_status": leg.status,
        }
        return payload

    def _allocation_leg_target(self, cycle: VaultCycle, leg: VaultAllocationLeg, cycle_return: float) -> float:
        cap = max(self._safe_float(leg.allocation_cap_usd), 1.0)
        pnl = self._safe_float(leg.realized_pnl_usd) + self._safe_float(leg.unrealized_pnl_usd)
        target = pnl / cap if cap > 0 else cycle_return
        if abs(target) <= 1e-12:
            target = cycle_return
        if str(cycle.status or "").lower() in {"failed", "error"} or str(leg.status or "").lower() in {"failed", "error"}:
            target -= 0.2
        return self._bounded_target(target)

    @staticmethod
    def _cycle_return(cycle: VaultCycle) -> float:
        starting = MLDecisionEngine._safe_float(cycle.starting_value_usd)
        final = MLDecisionEngine._safe_float(cycle.final_settlement_amount, math.nan)
        if not math.isfinite(final):
            final = MLDecisionEngine._safe_float(cycle.current_estimated_value_usd)
        if starting <= 0:
            return 0.0
        return MLDecisionEngine._bounded_target((final - starting) / starting)

    def _ops_payload_from_audit(self, audit: AuditLog) -> dict[str, Any]:
        details = audit.details if isinstance(audit.details, dict) else {}
        text = f"{audit.category} {audit.action} {audit.message}".lower()
        return {
            **details,
            "provider": details.get("provider") or details.get("provider_name") or "",
            "action": audit.action,
            "category": audit.category,
            "rate_limited": any(term in text for term in ("429", "rate_limit", "rate limit", "throttle")),
            "provider_failure": any(term in text for term in ("provider", "exchange", "market data", "order book", "connection")),
            "live_failure_block": any(term in text for term in ("blocked", "rejected", "failed", "panic", "readiness")),
            "stale_readiness": "stale" in text,
            "can_trade": bool(details.get("can_trade", False)),
            "latency_ms": self._safe_float(details.get("latency_ms")),
            "error_rate": 1.0 if any(term in text for term in ("failed", "error", "exception", "rejected")) else 0.0,
        }

    def _ops_payload_from_risk_event(self, event: RiskEvent) -> dict[str, Any]:
        payload = event.payload if isinstance(event.payload, dict) else {}
        text = f"{event.rule_name} {event.reason}".lower()
        return {
            **payload,
            "provider": payload.get("provider") or payload.get("provider_name") or "",
            "action": event.rule_name,
            "category": "risk",
            "rate_limited": any(term in text for term in ("429", "rate_limit", "rate limit", "throttle")),
            "provider_failure": "provider" in text or "connection" in text,
            "live_failure_block": True,
            "stale_readiness": "stale" in text,
            "can_trade": False,
            "error_rate": 1.0,
        }

    def _ops_payload_from_health_setting(self, setting: Setting) -> dict[str, Any]:
        value = Setting.get_json(setting.key, {})
        details = value if isinstance(value, dict) else {}
        last_checked = self._parse_datetime(details.get("checked_at") or details.get("last_checked_at"))
        age_seconds = (datetime.utcnow() - last_checked).total_seconds() if last_checked else 0.0
        return {
            **details,
            "provider": details.get("provider") or setting.key.split(":", 1)[-1],
            "category": "connection_health",
            "action": "connection_health_snapshot",
            "can_trade": bool(details.get("can_trade", False)),
            "rate_limited": bool(details.get("rate_limited", False)),
            "provider_failure": bool(details.get("provider_failure", False) or details.get("last_error")),
            "live_failure_block": not bool(details.get("can_trade", False)),
            "stale_readiness": bool(details.get("stale", False)) or age_seconds > 900,
            "latency_ms": self._safe_float(details.get("latency_ms")),
            "error_rate": 0.0 if bool(details.get("can_trade", False)) else 1.0,
            "stale_data_age_seconds": age_seconds,
        }

    @staticmethod
    def _ops_target(payload: dict[str, Any]) -> float:
        if any(
            bool(payload.get(key, False))
            for key in ("rate_limited", "provider_failure", "live_failure_block", "stale_readiness")
        ):
            return 1.0
        return 0.0

    def _artifact_path(self, provider: str, horizon: str, family: str) -> Path:
        root = Path(current_app.instance_path if has_app_context() else "instance") / "ml_models"
        root.mkdir(parents=True, exist_ok=True)
        provider_key = normalize_provider(provider)
        return root / f"ml-decision-{provider_key}-{family}-{horizon}-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.pt"

    def _load_decision_artifact(self, path: Any) -> dict[str, Any] | str:
        if not self._module_available("torch"):
            return "torch_missing"
        artifact_path = Path(str(path or ""))
        if not artifact_path.exists():
            return "ml_decision_artifact_missing"
        try:
            import torch

            payload = torch.load(artifact_path, map_location="cpu")
        except Exception:  # noqa: BLE001
            return "ml_decision_artifact_invalid"
        if not isinstance(payload, dict) or "state_dict" not in payload or "feature_names" not in payload:
            return "ml_decision_artifact_invalid"
        if str(payload.get("feature_schema_version") or "") != FEATURE_SCHEMA_VERSION:
            return "ml_decision_feature_schema_version_mismatch"
        return payload

    def _score_artifact(self, artifact: dict[str, Any], context: dict[str, Any]) -> float:
        import torch
        from torch import nn

        feature_names = [str(item) for item in list(artifact.get("feature_names") or [])]
        if not feature_names:
            return 0.0
        input_size = int(artifact.get("input_size") or len(feature_names))
        hidden_size = int(artifact.get("hidden_size") or max(8, min(64, input_size * 2)))
        output_activation = str(artifact.get("output_activation") or "tanh").strip().lower()
        model_layers: list[Any] = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ]
        if output_activation != "sigmoid":
            model_layers.append(nn.Tanh())
        model = nn.Sequential(*model_layers)
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        vector = self._vector(self._features(context), feature_names)
        with torch.no_grad():
            output = model(torch.tensor([vector], dtype=torch.float32)).reshape(-1)[0]
            if output_activation == "sigmoid":
                output = torch.sigmoid(output)
            prediction = float(output.item())
        return self._bounded_target(prediction)

    @staticmethod
    def _metrics(targets: list[float], predictions: list[float]) -> dict[str, float]:
        if not targets or not predictions:
            return {
                "validation_loss": 0.0,
                "negative_error_rate": 0.0,
                "drift": 0.0,
                "mean_absolute_error": 0.0,
                "false_positive_rate": 0.0,
                "confidence": 0.0,
                "expected_return": 0.0,
            }
        errors = [target - prediction for target, prediction in zip(targets, predictions)]
        loss = sum(error * error for error in errors) / len(errors)
        mae = sum(abs(error) for error in errors) / len(errors)
        negative_error_rate = sum(1 for error in errors if error < 0.0) / len(errors)
        false_positive_rate = sum(1 for target, prediction in zip(targets, predictions) if prediction > 0.1 and target <= 0.0) / len(errors)
        drift = abs(sum(errors) / len(errors))
        mean_prediction = sum(predictions) / len(predictions)
        return {
            "validation_loss": float(loss),
            "negative_error_rate": float(negative_error_rate),
            "drift": float(drift),
            "mean_absolute_error": float(mae),
            "false_positive_rate": float(false_positive_rate),
            "confidence": float(max(0.0, min(abs(mean_prediction), 1.0))),
            "expected_return": float(mean_prediction),
        }

    def _risk_policy_metrics(
        self,
        targets: list[float],
        predictions: list[float],
        *,
        validation_loss: float | None = None,
    ) -> dict[str, float]:
        if not targets or not predictions:
            return {
                "validation_loss": 0.0,
                "negative_error_rate": 0.0,
                "drift": 0.0,
                "mean_absolute_error": 0.0,
                "false_positive_rate": 0.0,
                "confidence": 0.0,
                "expected_return": 0.0,
                "accuracy": 0.0,
                "approval_precision": 0.0,
                "approval_rate": 0.0,
                "approval_count": 0,
            }
        target_values = [self._risk_policy_probability_target(value) for value in targets]
        prediction_values = [max(0.0, min(self._safe_float(value), 1.0)) for value in predictions]
        threshold = self._risk_policy_approve_threshold()
        actual_positive = [value >= 0.5 for value in target_values]
        predicted_positive = [value >= threshold for value in prediction_values]
        errors = [target - prediction for target, prediction in zip(target_values, prediction_values)]
        mse = sum(error * error for error in errors) / len(errors)
        mae = sum(abs(error) for error in errors) / len(errors)
        approvals = sum(1 for value in predicted_positive if value)
        true_approvals = sum(1 for actual, predicted in zip(actual_positive, predicted_positive) if actual and predicted)
        false_approvals = sum(1 for actual, predicted in zip(actual_positive, predicted_positive) if not actual and predicted)
        correct = sum(1 for actual, predicted in zip(actual_positive, predicted_positive) if bool(actual) == bool(predicted))
        mean_prediction = sum(prediction_values) / len(prediction_values)
        mean_target = sum(target_values) / len(target_values)
        return {
            "validation_loss": float(validation_loss if validation_loss is not None else mse),
            "negative_error_rate": float(sum(1 for error in errors if error < 0.0) / len(errors)),
            "drift": float(abs(mean_target - mean_prediction)),
            "mean_absolute_error": float(mae),
            "false_positive_rate": float(false_approvals / max(approvals, 1)),
            "confidence": float(max(0.0, min(mean_prediction, 1.0))),
            "expected_return": float(mean_prediction),
            "accuracy": float(correct / max(len(target_values), 1)),
            "approval_precision": float(true_approvals / max(approvals, 1)),
            "approval_rate": float(approvals / max(len(target_values), 1)),
            "approval_count": int(approvals),
            "approve_threshold": float(threshold),
        }

    @staticmethod
    def _target_distribution(targets: list[float]) -> dict[str, Any]:
        if not targets:
            return {"count": 0, "positive": 0, "negative": 0, "neutral": 0, "mean": 0.0}
        positive = sum(1 for value in targets if value > 0.0)
        negative = sum(1 for value in targets if value < 0.0)
        neutral = len(targets) - positive - negative
        return {
            "count": len(targets),
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "mean": float(sum(targets) / len(targets)),
        }

    @staticmethod
    def _training_dataset_payload(rows: list[MLDecisionTrainingRow], *, use_market_history: bool) -> dict[str, Any]:
        sources: dict[str, int] = {}
        for row in rows:
            source = str(row.source or "unknown")
            sources[source] = sources.get(source, 0) + 1
        return {
            "row_count": len(rows),
            "use_market_history": bool(use_market_history),
            "market_history_rows": sum(count for source, count in sources.items() if source.startswith("ml_market_history:")),
            "sources": sources,
            "leakage_policy": "features are built only from each row's pre-cutoff window; forward returns are target-only",
        }

    def _walk_forward_metrics(
        self,
        targets: list[float],
        predictions: list[float],
        *,
        prediction_source: str = "model",
        false_positive_threshold: float = 0.1,
        positive_target_threshold: float = 0.0,
    ) -> dict[str, float]:
        if len(targets) < 4 or len(predictions) < 4:
            return {
                "fold_count": 0,
                "mean_validation_loss": 0.0,
                "false_positive_rate": 0.0,
                "prediction_source": prediction_source,
            }
        fold_count = min(5, max(1, len(targets) // 4))
        fold_size = max(1, len(targets) // fold_count)
        losses: list[float] = []
        false_positive = 0
        observations = 0
        for fold in range(fold_count):
            start = fold * fold_size
            end = len(targets) if fold == fold_count - 1 else min(len(targets), start + fold_size)
            if start >= end:
                continue
            metrics = self._metrics(targets[start:end], predictions[start:end])
            losses.append(float(metrics.get("validation_loss", 0.0)))
            false_positive += sum(
                1
                for target, prediction in zip(targets[start:end], predictions[start:end])
                if prediction >= false_positive_threshold and target <= positive_target_threshold
            )
            observations += end - start
        return {
            "fold_count": len(losses),
            "mean_validation_loss": float(sum(losses) / len(losses)) if losses else 0.0,
            "false_positive_rate": float(false_positive / observations) if observations else 0.0,
            "prediction_source": prediction_source,
        }

    def _score_features_for_metrics(self, features: dict[str, float]) -> float:
        if not features:
            return 0.0
        values = [float(value or 0.0) for value in features.values()]
        return self._bounded_target(sum(values) / max(len(values), 1))

    def _target_roi_pct(self, objective: Any, horizon: Any = "1h") -> float:
        objective_key = self._objective(str(objective or ""))
        horizon_key = self._horizon(str(horizon or "1h"))
        if objective_key in {"one_h10", "1h10", "one_hour_10x"} or horizon_key == "1h10":
            return max(1.0, self._safe_float(self.config.get("ML_TARGET_ROI_1H10_PCT", self.config.get("ONE_H10_TARGET_ROI_PCT")), 1000.0))
        if objective_key == "extreme_roi_1h":
            return max(1.0, self._safe_float(self.config.get("ML_TARGET_ROI_1H_PCT"), 1000.0))
        if objective_key == "consistent_roi_1w":
            return max(1.0, self._safe_float(self.config.get("ML_TARGET_ROI_1W_PCT"), 100.0))
        if horizon_key in {"1w", "7d"}:
            return max(1.0, self._safe_float(self.config.get("ML_TARGET_ROI_1W_PCT"), 100.0))
        return max(1.0, self._safe_float(self.config.get("ML_EXTREME_UPSIDE_TARGET_ROI_PCT"), 1000.0))

    def _risk_policy_training_target(self, realized_return: Any) -> float:
        threshold = max(1e-6, self._safe_float(self.config.get("ML_RISK_POLICY_TARGET_RETURN_THRESHOLD"), 0.001))
        value = self._safe_float(realized_return)
        if value >= threshold:
            return 1.0
        if value <= 0:
            return 0.0
        return max(0.0, min(value / threshold, 1.0))

    def _risk_policy_backtest_target(self, result: dict[str, Any]) -> float:
        trade_count = max(0, int(self._safe_float(result.get("trade_count"))))
        trades = result.get("trades") if isinstance(result.get("trades"), list) else []
        if trade_count <= 0 and not trades:
            return 0.0
        max_drawdown = abs(min(self._safe_float(result.get("max_drawdown")), 0.0))
        drawdown_limit = max(0.0, self._safe_float(self.config.get("ML_RISK_POLICY_MAX_BACKTEST_DRAWDOWN_PCT"), 0.05))
        profit_factor = self._safe_float(result.get("profit_factor"), 1.0)
        min_profit_factor = max(0.0, self._safe_float(self.config.get("ML_RISK_POLICY_MIN_BACKTEST_PROFIT_FACTOR"), 0.80))
        if drawdown_limit > 0 and max_drawdown > drawdown_limit:
            return 0.0
        if profit_factor > 0 and profit_factor < min_profit_factor:
            return 0.0
        return 1.0

    def _risk_policy_market_target(self, candles: list[dict[str, Any]]) -> float:
        closes = [self._safe_float(row.get("close")) for row in candles if isinstance(row, dict)]
        closes = [value for value in closes if value > 0]
        if len(closes) < 10:
            return 0.0
        returns = [(end - start) / start for start, end in zip(closes, closes[1:]) if start > 0]
        if not returns:
            return 0.0
        mean_return = sum(returns) / len(returns)
        variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
        volatility = math.sqrt(max(variance, 0.0))
        max_abs_return = max(abs(value) for value in returns)
        volatility_limit = max(0.0, self._safe_float(self.config.get("ML_RISK_POLICY_MAX_RECENT_VOLATILITY"), 0.01))
        abs_return_limit = max(0.0, self._safe_float(self.config.get("ML_RISK_POLICY_MAX_RECENT_ABS_RETURN"), 0.03))
        if volatility_limit > 0 and volatility > volatility_limit:
            return 0.0
        if abs_return_limit > 0 and max_abs_return > abs_return_limit:
            return 0.0
        return 1.0

    def _risk_policy_probability_target(self, value: Any) -> float:
        return max(0.0, min(self._safe_float(value), 1.0))

    def _risk_policy_approve_threshold(self) -> float:
        return max(0.0, min(self._safe_float(self.config.get("ML_RISK_POLICY_APPROVE_THRESHOLD"), 0.55), 1.0))

    @staticmethod
    def _objective(value: str | None) -> str:
        objective = str(value or "risk_adjusted").strip().lower()
        if objective in {"extreme_upside", "extreme_roi_1h", "one_h10", "1h10", "one_hour_10x", "consistent_roi_1w"}:
            if objective in {"1h10", "one_hour_10x"}:
                return "one_h10"
            return objective
        return "risk_adjusted"

    @staticmethod
    def _feature_importance(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
        first_layer = None
        try:
            first_layer = model[0]
        except (TypeError, IndexError):
            return []
        weights = getattr(first_layer, "weight", None)
        if weights is None:
            return []
        try:
            values = [float(item) for item in weights.detach().abs().mean(dim=0).cpu().tolist()]
        except Exception:  # noqa: BLE001
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

    @staticmethod
    def _bounded_target(value: Any) -> float:
        cap = 1.0
        candidate = MLDecisionEngine._safe_float(value)
        return max(-cap, min(candidate, cap))

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if not value:
            return None
        text = str(value)
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    @staticmethod
    def _timeframe_minutes(timeframe: str | None) -> int:
        value = str(timeframe or "1h").lower()
        return {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}.get(value, 60)

    @staticmethod
    def _horizon_minutes(horizon: str | None) -> int:
        value = str(horizon or "1h").lower()
        if value.endswith("m"):
            return max(1, int(MLDecisionEngine._safe_float(value[:-1], 1.0)))
        if value.endswith("h"):
            return max(1, int(MLDecisionEngine._safe_float(value[:-1], 1.0) * 60))
        if value.endswith("d"):
            return max(1, int(MLDecisionEngine._safe_float(value[:-1], 1.0) * 24 * 60))
        return max(1, int(MLDecisionEngine._safe_float(value, 1.0) * 60))

    @staticmethod
    def _history_sample_time(candle: dict[str, Any], history: MLMarketHistory) -> datetime:
        raw = MLDecisionEngine._safe_float(candle.get("timestamp"), 0.0)
        seconds = raw / 1000.0 if raw > 10_000_000_000 else raw
        if seconds > 0:
            try:
                return datetime.utcfromtimestamp(seconds)
            except (OverflowError, OSError, ValueError):
                pass
        return history.window_end or history.fetched_at or history.created_at or datetime.utcnow()

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            result = int(value)
        except (TypeError, ValueError):
            return None
        return result if result > 0 else None

    @staticmethod
    def _model_payload(record: MLOfflineModel | None) -> dict[str, Any] | None:
        if record is None:
            return None
        return {
            "id": record.id,
            "model_key": record.model_key,
            "provider": getattr(record, "provider", "global"),
            "horizon": record.horizon,
            "model_type": record.model_type,
            "status": record.status,
            "feature_schema_version": record.feature_schema_version,
            "training_rows": record.training_rows,
            "validation_rows": record.validation_rows,
            "validation_loss": record.validation_loss,
            "negative_error_rate": record.negative_error_rate,
            "drift": record.drift,
            "metrics": record.metrics,
            "artifact_path": record.artifact_path,
            "artifact_exists": bool(record.artifact_path and Path(record.artifact_path).exists()),
            "created_at": record.created_at,
            "promoted_at": record.promoted_at,
        }
