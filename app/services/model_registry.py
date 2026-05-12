"""Auditable ML model registry and promotion metadata helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from ..extensions import db
from ..models import AuditLog, MLModelRegistry, MLOfflineModel
from .failures import ModelPromotionError


def feature_schema_hash(feature_schema_version: str, feature_names: list[str]) -> str:
    payload = {
        "feature_schema_version": str(feature_schema_version or ""),
        "feature_names": [str(item) for item in feature_names],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def dataset_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


class ModelRegistryService:
    """Persist governance metadata around existing model promotion flows."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def record_promotion(
        self,
        record: MLOfflineModel,
        *,
        model_family: str,
        promoted_by: str = "system",
        promotion_source: str = "promotion_flow",
        mode: str = "live",
    ) -> MLModelRegistry:
        if record is None:
            raise ModelPromotionError("Cannot promote a missing model.", code="model_missing")
        now = datetime.utcnow()
        family = str(model_family or record.model_type or "model").strip().lower()
        rollback_target = self.promoted_registry(family, record.provider, record.horizon)
        rollback_target_id = rollback_target.offline_model_id if rollback_target and rollback_target.offline_model_id != record.id else None
        schema_hash = record.feature_schema_hash or feature_schema_hash(record.feature_schema_version, record.feature_names)
        lineage = {
            "model_key": record.model_key,
            "provider": record.provider,
            "horizon": record.horizon,
            "model_type": record.model_type,
            "training_rows": record.training_rows,
            "validation_rows": record.validation_rows,
            "metrics": record.metrics,
        }
        data_hash = record.dataset_hash or dataset_hash(lineage)
        data_version = record.dataset_version or f"{record.provider}:{record.horizon}:{record.training_rows}:{record.validation_rows}"

        for existing in MLModelRegistry.query.filter_by(
            model_family=family,
            provider=record.provider,
            horizon=record.horizon,
            status="promoted",
        ).all():
            if existing.offline_model_id != record.id:
                existing.status = "archived"
                existing.mode = "shadow"

        record.feature_schema_hash = schema_hash
        record.dataset_hash = data_hash
        record.dataset_version = data_version
        record.promoted_by = promoted_by
        record.promotion_source = promotion_source
        record.rollback_target_model_id = rollback_target_id
        record.live_mode = mode
        record.drift_status = self._drift_status(record)
        governance = record.governance_metadata
        governance.update(
            {
                "promoted_by": promoted_by,
                "promotion_source": promotion_source,
                "rollback_target_model_id": rollback_target_id,
                "last_metrics": record.metrics,
            }
        )
        record.governance_metadata = governance

        registry = MLModelRegistry.query.filter_by(model_key=record.model_key).one_or_none()
        if registry is None:
            registry = MLModelRegistry(model_key=record.model_key, model_family=family)
            db.session.add(registry)
        registry.offline_model_id = record.id
        registry.model_family = family
        registry.model_version = str(record.id)
        registry.provider = record.provider
        registry.horizon = record.horizon
        registry.feature_schema_hash = schema_hash
        registry.dataset_version = data_version
        registry.dataset_hash = data_hash
        registry.trained_at = record.training_completed_at or record.created_at
        registry.promoted_at = record.promoted_at or now
        registry.promoted_by = promoted_by
        registry.promotion_source = promotion_source
        registry.rollback_target_model_id = rollback_target_id
        registry.mode = mode
        registry.drift_status = record.drift_status
        registry.status = "promoted"
        registry.metrics = record.metrics
        registry.details = {"offline_model_id": record.id, "artifact_path_present": bool(record.artifact_path)}
        self._audit("model_promoted", "promoted", registry)
        db.session.flush()
        return registry

    def record_failed_promotion(
        self,
        *,
        model_family: str,
        provider: str,
        horizon: str,
        model_id: int | None,
        blockers: list[str],
    ) -> None:
        audit = AuditLog(category="ml", action="model_promotion_failed", message="ML model promotion failed governance checks.")
        audit.details = {
            "model_family": model_family,
            "provider": provider,
            "horizon": horizon,
            "model_id": model_id,
            "blockers": blockers,
        }
        db.session.add(audit)
        db.session.flush()

    def rollback(self, *, model_family: str, provider: str, horizon: str, promoted_by: str = "system") -> MLModelRegistry:
        current = self.promoted_registry(model_family, provider, horizon)
        if current is None or not current.rollback_target_model_id:
            raise ModelPromotionError("No rollback target is available.", code="model_rollback_unavailable")
        target = db.session.get(MLOfflineModel, int(current.rollback_target_model_id))
        if target is None:
            raise ModelPromotionError("Rollback target model is missing.", code="model_rollback_target_missing")
        target.status = "promoted"
        target.promoted_at = datetime.utcnow()
        registry = self.record_promotion(
            target,
            model_family=model_family,
            promoted_by=promoted_by,
            promotion_source="rollback",
            mode="live",
        )
        self._audit("model_rolled_back", "promoted", registry)
        return registry

    def promoted_registry(self, model_family: str, provider: str, horizon: str) -> MLModelRegistry | None:
        return (
            MLModelRegistry.query.filter_by(
                model_family=str(model_family or "").strip().lower(),
                provider=str(provider or "global").strip().lower(),
                horizon=str(horizon or "global").strip().lower(),
                status="promoted",
            )
            .order_by(MLModelRegistry.promoted_at.desc(), MLModelRegistry.created_at.desc())
            .first()
        )

    def _drift_status(self, record: MLOfflineModel) -> str:
        max_drift = float(self.config.get("ML_OFFLINE_MAX_DRIFT", 0.35) or 0.35)
        drift = abs(float(record.drift or 0.0))
        if drift > max_drift:
            return "blocked"
        if drift > max_drift * 0.75:
            return "watch"
        return "ok"

    def _audit(self, action: str, status: str, registry: MLModelRegistry) -> None:
        audit = AuditLog(category="ml", action=action, message=f"ML registry {action.replace('_', ' ')}.")
        audit.details = {
            "model_key": registry.model_key,
            "model_family": registry.model_family,
            "model_version": registry.model_version,
            "provider": registry.provider,
            "horizon": registry.horizon,
            "status": status,
            "mode": registry.mode,
            "rollback_target_model_id": registry.rollback_target_model_id,
            "drift_status": registry.drift_status,
        }
        db.session.add(audit)
