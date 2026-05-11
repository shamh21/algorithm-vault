"""Promoted ML signal model for high-upside vault execution."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from flask import current_app, has_app_context

from ..extensions import db
from ..models import MLMarketHistory, MLOfflineModel
from ..services.provider_assets import normalize_provider, provider_feature_context
from .features import MLFeatureFactory
from .offline_ranker import OfflineRanker
from .online_ranker import OnlineRanker, extract_features


FEATURE_SCHEMA_VERSION = "ml_signal_v1"
ACTION_LABELS = ("sell", "hold", "buy")


@dataclass(frozen=True, slots=True)
class MLSignalTrainingRow:
    features: dict[str, float]
    target: int
    target_return: float
    created_at: datetime
    source: str
    provider: str = "global"
    sequence_key: str = "global"


class MLSignalModel:
    """Train, promote, and score ML-generated buy/sell/hold signals.

    The service is intentionally advisory unless a promoted artifact passes
    readiness. RiskEngine remains the live execution gate.
    """

    def __init__(self, config: dict[str, Any], *, artifact_root: str | Path | None = None) -> None:
        self.config = config
        self.online_ranker = OnlineRanker(config)
        self.offline_ranker = OfflineRanker(config)
        self.feature_factory = MLFeatureFactory(config)
        self.artifact_root = Path(artifact_root) if artifact_root else None

    def train(
        self,
        horizon: str = "1h",
        *,
        model_type: str = "pytorch_gru",
        objective: str = "risk_adjusted",
        use_market_history: bool = False,
        provider: str = "global",
    ) -> dict[str, Any]:
        horizon_key = str(horizon or "1h").lower()
        model_key = str(model_type or "pytorch_gru").lower()
        provider_key = normalize_provider(provider)
        raw_rows = self.training_rows(
            horizon_key,
            objective=objective,
            use_market_history=use_market_history,
            provider=provider_key,
        )
        raw_training_rows = len(raw_rows)
        rows = self._bounded_training_rows(raw_rows)
        min_rows = int(self.config.get("ML_SIGNAL_MIN_TRAINING_ROWS", 500) or 500)
        target_distribution = self._target_distribution(rows)
        blockers: list[str] = []
        if not bool(self.config.get("ML_SIGNAL_MODEL_ENABLED", False)):
            blockers.append("ML_SIGNAL_MODEL_ENABLED=false")
        if model_key != "pytorch_gru":
            blockers.append("unsupported_signal_model_type")
        if not self._module_available("torch"):
            blockers.append("torch_missing")
        if len(rows) < min_rows:
            blockers.append("insufficient_training_rows")
        if blockers:
            return {
                "trained": False,
                "horizon": horizon_key,
                "provider": provider_key,
                "model_type": model_key,
                "training_rows": len(rows),
                "raw_training_rows": raw_training_rows,
                "min_training_rows": min_rows,
                "training_dataset": self._training_dataset_payload(
                    rows,
                    use_market_history=use_market_history,
                    raw_rows=raw_training_rows,
                ),
                "target_distribution": target_distribution,
                "blockers": blockers,
            }

        torch = self._torch()
        feature_names = sorted({key for row in rows for key in row.features})
        if not feature_names:
            return {
                "trained": False,
                "horizon": horizon_key,
                "provider": provider_key,
                "model_type": model_key,
                "training_rows": len(rows),
                "raw_training_rows": raw_training_rows,
                "training_dataset": self._training_dataset_payload(
                    rows,
                    use_market_history=use_market_history,
                    raw_rows=raw_training_rows,
                ),
                "target_distribution": target_distribution,
                "blockers": ["empty_feature_schema"],
            }
        sequence_length = self._sequence_length()
        x_all = self._sequence_examples(rows, feature_names, sequence_length)
        y_all = [int(row.target) for row in rows]
        split_index = max(1, int(len(rows) * 0.8))
        if split_index >= len(rows):
            split_index = len(rows) - 1
        train_x, valid_x = x_all[:split_index], x_all[split_index:]
        train_y, valid_y = y_all[:split_index], y_all[split_index:]
        configured_hidden = int(self.config.get("ML_SIGNAL_HIDDEN_SIZE", 32) or 0)
        hidden_size = configured_hidden if configured_hidden > 0 else max(8, min(32, len(feature_names)))
        model = _TorchGRUSignalNet(torch, input_size=len(feature_names), hidden_size=hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.config.get("ML_SIGNAL_LEARNING_RATE", 0.01) or 0.01))
        class_weight_tensor, class_weights = self._class_weight_tensor(torch, train_y)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight_tensor)
        train_tensor = torch.tensor(train_x, dtype=torch.float32)
        label_tensor = torch.tensor(train_y, dtype=torch.long)
        model.train()
        epochs = max(1, int(self.config.get("ML_SIGNAL_TRAINING_EPOCHS", 16) or 16))
        batch_size = max(1, min(len(train_y), int(self.config.get("ML_SIGNAL_TRAINING_BATCH_SIZE", 2048) or 2048)))
        for _ in range(epochs):
            for start in range(0, len(train_y), batch_size):
                end = start + batch_size
                optimizer.zero_grad()
                loss = loss_fn(model(train_tensor[start:end]), label_tensor[start:end])
                loss.backward()
                optimizer.step()

        metrics = self._metrics(model, torch, valid_x, valid_y)
        metrics["target_distribution"] = target_distribution
        metrics["class_balance_enabled"] = bool(class_weights)
        metrics["class_weights"] = class_weights
        metrics["training_epochs"] = epochs
        metrics["training_batch_size"] = batch_size
        metrics["raw_training_rows"] = raw_training_rows
        metrics["used_training_rows"] = len(rows)
        metrics["sequence_length"] = sequence_length
        artifact_path = self._artifact_path(provider_key, horizon_key, model_key)
        torch.save(
            {
                "model_type": model_key,
                "horizon": horizon_key,
                "provider": provider_key,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "feature_names": feature_names,
                "input_size": len(feature_names),
                "hidden_size": hidden_size,
                "sequence_length": sequence_length,
                "state_dict": model.state_dict(),
                "created_at": datetime.utcnow().isoformat(),
                "metrics": metrics,
            },
            artifact_path,
        )
        record = MLOfflineModel(
            model_key=f"ml_signal:{provider_key}:{horizon_key}:{model_key}:{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            provider=provider_key,
            horizon=horizon_key,
            model_type=model_key,
            status="candidate",
            artifact_path=str(artifact_path),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            training_rows=len(train_x),
            validation_rows=len(valid_x),
            validation_loss=float(metrics["validation_loss"]),
            negative_error_rate=float(metrics["false_positive_rate"]),
            drift=0.0,
        )
        record.feature_names = feature_names
        record.metrics = metrics
        db.session.add(record)
        db.session.commit()
        return {
            "trained": True,
            "horizon": horizon_key,
            "provider": provider_key,
            "model_type": model_key,
            "training_rows": len(rows),
            "raw_training_rows": raw_training_rows,
            "training_dataset": self._training_dataset_payload(
                rows,
                use_market_history=use_market_history,
                raw_rows=raw_training_rows,
            ),
            "target_distribution": target_distribution,
            "model": self._model_payload(record),
            "blockers": [],
        }

    def promote(self, horizon: str = "1h", *, model_id: int, provider: str = "global") -> dict[str, Any]:
        horizon_key = str(horizon or "1h").lower()
        provider_key = normalize_provider(provider)
        record = MLOfflineModel.query.filter_by(
            id=int(model_id),
            provider=provider_key,
            horizon=horizon_key,
            model_type="pytorch_gru",
        ).one_or_none()
        if record is None:
            return {"promoted": False, "horizon": horizon_key, "provider": provider_key, "model_id": model_id, "blockers": ["signal_model_not_found"]}
        diagnostics = self.promotion_diagnostics(record)
        if not diagnostics["ready"]:
            return {"promoted": False, **diagnostics}
        for promoted in MLOfflineModel.query.filter_by(
            provider=provider_key,
            horizon=horizon_key,
            model_type="pytorch_gru",
            status="promoted",
        ).all():
            if promoted.id != record.id:
                promoted.status = "archived"
        record.status = "promoted"
        record.promoted_at = datetime.utcnow()
        db.session.commit()
        return {"promoted": True, **self._model_payload(record), "blockers": []}

    def readiness(self, horizon: str = "1h", *, require_promoted: bool = True, provider: str = "global") -> dict[str, Any]:
        horizon_key = str(horizon or "1h").lower()
        provider_key = normalize_provider(provider)
        record = self.promoted_model(horizon_key, provider=provider_key)
        blockers: list[str] = []
        if not bool(self.config.get("ML_SIGNAL_MODEL_ENABLED", False)):
            blockers.append("ML_SIGNAL_MODEL_ENABLED=false")
        if not self._module_available("torch"):
            blockers.append("torch_missing")
        if require_promoted and record is None:
            blockers.append("promoted_signal_model_missing")
        if record is not None:
            blockers.extend(self.promotion_diagnostics(record).get("blockers", []))
        return {
            "ready": not blockers,
            "horizon": horizon_key,
            "provider": provider_key,
            "model_type": str(self.config.get("HIGH_UPSIDE_ML_SIGNAL_MODEL_TYPE", "pytorch_gru") or "pytorch_gru"),
            "blockers": list(dict.fromkeys(blockers)),
            "promoted_model": self._model_payload(record) if record else None,
            "enabled": bool(self.config.get("ML_SIGNAL_MODEL_ENABLED", False)),
            "require_promoted": bool(require_promoted),
        }

    def score_payload(
        self,
        context: dict[str, Any],
        horizon: str = "1h",
        *,
        candles: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        horizon_key = str(horizon or "1h").lower()
        provider_key = normalize_provider((context or {}).get("provider") or (context or {}).get("execution_venue"))
        base = {
            "enabled": bool(self.config.get("ML_SIGNAL_MODEL_ENABLED", False)),
            "horizon": horizon_key,
            "provider": provider_key,
            "model_type": "pytorch_gru",
            "status": "disabled",
            "ready_for_live": False,
            "action": "hold",
            "probabilities": {"sell": 0.0, "hold": 1.0, "buy": 0.0},
            "confidence": 0.0,
            "expected_return": 0.0,
            "suggested_stop_loss_pct": 0.0,
            "suggested_take_profit_pct": 0.0,
            "sizing_score": 0.0,
            "position_fraction": 0.0,
            "blockers": [],
        }
        if not bool(base["enabled"]):
            base["blockers"] = ["ML_SIGNAL_MODEL_ENABLED=false"]
            return base
        if not self._module_available("torch"):
            base.update({"status": "torch_missing", "blockers": ["torch_missing"]})
            return base
        record = self.promoted_model(horizon_key, provider=provider_key)
        if record is None:
            base.update({"status": "no_promoted_signal_model", "blockers": ["promoted_signal_model_missing"]})
            return base
        diagnostics = self.promotion_diagnostics(record)
        if diagnostics.get("blockers"):
            base.update({"status": "promoted_blocked", "model_id": record.id, "blockers": diagnostics["blockers"]})
            return base
        torch = self._torch()
        artifact = self._load_artifact(record.artifact_path, torch)
        if isinstance(artifact, str):
            base.update({"status": artifact, "model_id": record.id, "blockers": [artifact]})
            return base
        feature_names = list(artifact.get("feature_names") or record.feature_names)
        candle_rows = list(candles or (context or {}).get("rapid_feature_candles") or [])
        sequence_length = max(1, int(artifact.get("sequence_length") or 1))
        sequence = self._prediction_feature_sequence(
            provider_key,
            context,
            candle_rows,
            feature_names,
            sequence_length,
        )
        vector = torch.tensor([sequence], dtype=torch.float32)
        model = _TorchGRUSignalNet(
            torch,
            input_size=int(artifact.get("input_size") or len(feature_names)),
            hidden_size=int(artifact.get("hidden_size") or 16),
        )
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(vector)[0]
            probs_tensor = torch.softmax(logits, dim=0)
        probabilities = {label: float(probs_tensor[index].item()) for index, label in enumerate(ACTION_LABELS)}
        probability_decision = self._probability_decision(probabilities)
        action = str(probability_decision["action"])
        confidence = self._safe_float(probability_decision.get("confidence"))
        expected_return = self._safe_float(probability_decision.get("expected_return"))
        stop_pct = self._config_or_context_float(context, "stop_loss_pct", 0.005)
        take_pct = self._config_or_context_float(context, "take_profit_pct", 0.012)
        sizing_score = max(0.0, min((confidence - 0.5) * 2.0, 1.0))
        blockers: list[str] = list(probability_decision.get("blockers", []) or [])
        if action in {"buy", "sell"} and stop_pct <= 0:
            blockers.append("ml_signal_stop_loss_missing")
        if action in {"buy", "sell"} and take_pct <= 0:
            blockers.append("ml_signal_take_profit_missing")
        return {
            **base,
            "status": "promoted",
            "model_id": record.id,
            "action": action,
            "probabilities": probabilities,
            "confidence": confidence,
            "expected_return": expected_return,
            "signed_expected_return": probability_decision.get("signed_expected_return"),
            "action_probability": probability_decision.get("action_probability"),
            "hold_probability": probability_decision.get("hold_probability"),
            "directional_confidence": probability_decision.get("directional_confidence"),
            "directional_margin": probability_decision.get("directional_margin"),
            "suggested_stop_loss_pct": stop_pct,
            "suggested_take_profit_pct": take_pct,
            "sizing_score": sizing_score,
            "position_fraction": sizing_score,
            "ready_for_live": action in {"buy", "sell"} and not blockers,
            "blockers": blockers,
            "metrics": record.metrics,
            "feature_schema_version": record.feature_schema_version,
            "sequence_length": sequence_length,
        }

    def promotion_diagnostics(self, record: MLOfflineModel | None) -> dict[str, Any]:
        if record is None:
            return {"ready": False, "blockers": ["signal_model_not_found"]}
        blockers: list[str] = []
        metrics = record.metrics if isinstance(record.metrics, dict) else {}
        if record.model_type != "pytorch_gru":
            blockers.append("signal_model_type_mismatch")
        if record.feature_schema_version != FEATURE_SCHEMA_VERSION:
            blockers.append("signal_feature_schema_version_mismatch")
        if int(record.training_rows or 0) <= 0 or int(record.validation_rows or 0) <= 0:
            blockers.append("insufficient_train_validation_split")
        max_loss = float(
            self.config.get(
                "ML_SIGNAL_MAX_CLASSIFICATION_LOSS",
                self.config.get("ML_SIGNAL_MAX_VALIDATION_LOSS", 1.10),
            )
            or 1.10
        )
        if float(record.validation_loss or 0.0) > max_loss:
            blockers.append("signal_validation_loss_above_threshold")
        metric_confidence = metrics.get("confidence_action_threshold")
        live_confidence = float(self.config.get("ML_SIGNAL_MIN_CONFIDENCE", 0.60) or 0.60)
        if metric_confidence is not None and float(metric_confidence or 0.0) + 1e-9 < live_confidence:
            blockers.append("signal_metric_confidence_threshold_below_live")
        action_rate = float(metrics.get("action_rate", 0.0) or 0.0)
        action_count = int(metrics.get("action_count", 0) or 0)
        action_precision = metrics.get("action_precision")
        if action_rate > 0 and action_precision is not None:
            min_precision = float(self.config.get("ML_SIGNAL_MIN_ACTION_PRECISION", 0.52) or 0.52)
            if float(action_precision or 0.0) < min_precision:
                blockers.append("signal_action_precision_below_threshold")
            min_action_count = max(0, int(self.config.get("ML_SIGNAL_MIN_ACTION_COUNT", 10) or 0))
            if min_action_count > 0 and action_count < min_action_count:
                blockers.append("signal_action_count_below_threshold")
        target_distribution = metrics.get("target_distribution") if isinstance(metrics.get("target_distribution"), dict) else {}
        target_rates = target_distribution.get("rates") if isinstance(target_distribution.get("rates"), dict) else {}
        if action_rate <= 0 and "accuracy" in metrics and target_rates:
            majority_baseline = max(float(value or 0.0) for value in target_rates.values())
            min_edge = float(self.config.get("ML_SIGNAL_MIN_ACCURACY_EDGE", 0.0) or 0.0)
            if float(metrics.get("accuracy") or 0.0) < majority_baseline + max(0.0, min_edge):
                blockers.append("signal_accuracy_below_baseline")
        false_positive = float(metrics.get("false_positive_rate", record.negative_error_rate or 0.0) or 0.0)
        if false_positive > float(self.config.get("ML_SIGNAL_MAX_FALSE_POSITIVE_RATE", 0.35) or 0.35):
            blockers.append("signal_false_positive_rate_above_threshold")
        min_action_rate = float(self.config.get("ML_SIGNAL_MIN_ACTION_RATE", 0.01) or 0.0)
        if min_action_rate > 0 and action_rate < min_action_rate:
            blockers.append("signal_action_rate_below_threshold")
        max_action_rate = float(self.config.get("ML_SIGNAL_MAX_ACTION_RATE", 1.0) or 1.0)
        if 0 < max_action_rate < 1.0 and action_rate > max_action_rate:
            blockers.append("signal_action_rate_above_threshold")
        max_age_hours = float(self.config.get("ML_OFFLINE_MAX_MODEL_AGE_HOURS", 72.0) or 72.0)
        if max_age_hours > 0:
            created_at = record.created_at or datetime.utcnow()
            if created_at < datetime.utcnow() - timedelta(hours=max_age_hours):
                blockers.append("signal_model_age_above_threshold")
        if not record.artifact_path or not Path(record.artifact_path).exists():
            blockers.append("signal_artifact_missing")
        return {"ready": not blockers, "blockers": blockers, **self._model_payload(record)}

    def promoted_model(self, horizon: str, *, provider: str = "global") -> MLOfflineModel | None:
        if not has_app_context():
            return None
        provider_key = normalize_provider(provider)
        return (
            MLOfflineModel.query.filter_by(
                provider=provider_key,
                horizon=str(horizon or "1h").lower(),
                model_type="pytorch_gru",
                status="promoted",
            )
            .order_by(MLOfflineModel.promoted_at.desc(), MLOfflineModel.created_at.desc())
            .first()
        )

    def training_rows(
        self,
        horizon: str,
        *,
        objective: str = "risk_adjusted",
        use_market_history: bool = False,
        provider: str = "global",
    ) -> list[MLSignalTrainingRow]:
        rows: list[MLSignalTrainingRow] = []
        provider_key = normalize_provider(provider)
        threshold = float(self.config.get("ML_SIGNAL_TARGET_RETURN_THRESHOLD", 0.001) or 0.001)
        for row in self.offline_ranker.training_rows(horizon, provider=provider_key):
            target_return = float(row.target or 0.0)
            target = 2 if target_return > threshold else 0 if target_return < -threshold else 1
            rows.append(
                MLSignalTrainingRow(
                    features=dict(row.features),
                    target=target,
                    target_return=target_return,
                    created_at=row.created_at,
                    source=row.source,
                    provider=row.provider,
                    sequence_key=f"{normalize_provider(row.provider)}:{row.source}",
                )
            )
        if use_market_history:
            rows.extend(self._market_history_training_rows(horizon, objective=objective, provider=provider_key))
        rows.sort(key=lambda row: row.created_at)
        return rows

    def _market_history_training_rows(self, horizon: str, *, objective: str, provider: str = "global") -> list[MLSignalTrainingRow]:
        if not has_app_context():
            return []
        rows: list[MLSignalTrainingRow] = []
        provider_key = normalize_provider(provider)
        max_market_rows = int(self.config.get("ML_SIGNAL_MAX_TRAINING_ROWS", 15_000) or 0)
        threshold = float(self.config.get("ML_SIGNAL_TARGET_RETURN_THRESHOLD", 0.001) or 0.001)
        target_roi_pct = max(1.0, self._safe_float(self.config.get("ML_EXTREME_UPSIDE_TARGET_ROI_PCT"), 1000.0))
        target_return = target_roi_pct / 100.0
        query = MLMarketHistory.query.filter_by(status="ok")
        if provider_key != "global":
            query = query.filter(MLMarketHistory.provider == provider_key)
        query = query.order_by(
            MLMarketHistory.window_end.desc(),
            MLMarketHistory.fetched_at.desc(),
            MLMarketHistory.id.desc(),
        ).limit(5_000)
        for history in query.all():
            if max_market_rows > 0 and len(rows) >= max_market_rows:
                break
            candles = [row for row in history.candles if isinstance(row, dict)]
            if len(candles) < 24:
                continue
            forward_steps = self._forward_steps(horizon, history.timeframe)
            min_window = max(12, forward_steps * 2)
            max_cutoff = len(candles) - forward_steps - 1
            if max_cutoff < min_window:
                continue
            step = max(1, min(forward_steps, max(1, len(candles) // 16)))
            for cutoff_index in range(min_window, max_cutoff + 1, step):
                window = candles[: cutoff_index + 1]
                current_close = self._safe_float(window[-1].get("close"))
                future_close = self._safe_float(candles[cutoff_index + forward_steps].get("close"))
                if current_close <= 0 or future_close <= 0:
                    continue
                forward_return = (future_close - current_close) / current_close
                effective_threshold = threshold
                if str(objective or "").lower() == "extreme_upside":
                    effective_threshold = max(threshold, min(target_return, 0.02))
                target = 2 if forward_return > effective_threshold else 0 if forward_return < -effective_threshold else 1
                payload = self.feature_factory.build(
                    symbol=history.symbol,
                    timeframe=history.timeframe,
                    candles=window,
                    optimizer_context={
                        "provider": history.provider,
                        "strategy_name": "ml_market_history_signal",
                        "objective": str(objective or "risk_adjusted").lower(),
                        "target_roi_pct": target_roi_pct,
                        "horizon": horizon,
                    },
                    cutoff_timestamp=window[-1].get("timestamp"),
                )
                rows.append(
                    MLSignalTrainingRow(
                        features=self.online_ranker.normalized_features(extract_features(payload)),
                        target=target,
                        target_return=forward_return,
                        created_at=self._history_row_time(history),
                        source="ml_market_history:signal",
                        provider=normalize_provider(history.provider),
                        sequence_key=f"{normalize_provider(history.provider)}:{history.symbol}:{history.timeframe}",
                    )
                )
                if max_market_rows > 0 and len(rows) >= max_market_rows:
                    break
        return rows

    @staticmethod
    def _forward_steps(horizon: str, timeframe: str | None) -> int:
        horizon_minutes = {
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "24h": 1440,
            "1d": 1440,
        }.get(str(horizon or "1h").lower(), 60)
        timeframe_minutes = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }.get(str(timeframe or "1h").lower(), 60)
        return max(1, int(round(horizon_minutes / max(timeframe_minutes, 1))))

    @staticmethod
    def _history_row_time(history: MLMarketHistory) -> datetime:
        return history.window_end or history.fetched_at or history.created_at or datetime.utcnow()

    @staticmethod
    def _target_distribution(rows: list[MLSignalTrainingRow]) -> dict[str, Any]:
        counts = {"sell": 0, "hold": 0, "buy": 0}
        for row in rows:
            label = ACTION_LABELS[int(row.target)] if 0 <= int(row.target) < len(ACTION_LABELS) else "hold"
            counts[label] += 1
        total = max(len(rows), 1)
        return {
            "rows": len(rows),
            "counts": counts,
            "rates": {label: count / total for label, count in counts.items()},
        }

    def _bounded_training_rows(self, rows: list[MLSignalTrainingRow]) -> list[MLSignalTrainingRow]:
        max_rows = int(self.config.get("ML_SIGNAL_MAX_TRAINING_ROWS", 50_000) or 0)
        if max_rows <= 0 or len(rows) <= max_rows:
            return list(rows)
        step = len(rows) / max_rows
        selected: list[MLSignalTrainingRow] = []
        last_index = -1
        for offset in range(max_rows):
            index = min(len(rows) - 1, int(offset * step))
            if index <= last_index:
                index = min(len(rows) - 1, last_index + 1)
            selected.append(rows[index])
            last_index = index
        return selected

    def _class_weight_tensor(self, torch: Any, labels: list[int]) -> tuple[Any | None, dict[str, float]]:
        if not bool(self.config.get("ML_SIGNAL_CLASS_BALANCE_ENABLED", True)):
            return None, {}
        if not labels:
            return None, {}
        counts = {index: 0 for index in range(len(ACTION_LABELS))}
        for label in labels:
            counts[int(label)] = counts.get(int(label), 0) + 1
        total = sum(counts.values())
        if total <= 0:
            return None, {}
        max_weight = max(float(self.config.get("ML_SIGNAL_MAX_CLASS_WEIGHT", 6.0) or 6.0), 1.0)
        min_weight = 1.0 / max_weight
        weights: list[float] = []
        payload: dict[str, float] = {}
        for index, label in enumerate(ACTION_LABELS):
            count = max(int(counts.get(index, 0)), 1)
            raw_weight = total / (len(ACTION_LABELS) * count)
            weight = min(max(raw_weight, min_weight), max_weight)
            weights.append(float(weight))
            payload[label] = float(weight)
        return torch.tensor(weights, dtype=torch.float32), payload

    @staticmethod
    def _training_dataset_payload(
        rows: list[MLSignalTrainingRow],
        *,
        use_market_history: bool,
        raw_rows: int | None = None,
    ) -> dict[str, Any]:
        sources: dict[str, int] = {}
        for row in rows:
            sources[row.source] = sources.get(row.source, 0) + 1
        raw_count = int(raw_rows if raw_rows is not None else len(rows))
        return {
            "rows": len(rows),
            "raw_rows": raw_count,
            "sampled": raw_count > len(rows),
            "use_market_history": bool(use_market_history),
            "market_history_rows": sources.get("ml_market_history:signal", 0),
            "sources": dict(sorted(sources.items())),
            "leakage_policy": "features_use_only_candles_at_or_before_cutoff",
        }

    def _metrics(self, model: Any, torch: Any, valid_x: list[list[list[float]]], valid_y: list[int]) -> dict[str, float]:
        if not valid_x or not valid_y:
            return {
                "validation_loss": 0.0,
                "accuracy": 0.0,
                "false_positive_rate": 0.0,
                "action_rate": 0.0,
                "action_count": 0,
            }
        tensor = torch.tensor(valid_x, dtype=torch.float32)
        labels = torch.tensor(valid_y, dtype=torch.long)
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            probability_rows = probabilities.detach().cpu().tolist()
        probability_decisions = [
            self._probability_decision(
                {label: float(row[index]) for index, label in enumerate(ACTION_LABELS)}
            )
            for row in probability_rows
        ]
        predictions = [ACTION_LABELS.index(str(decision.get("action") or "hold")) for decision in probability_decisions]
        confidences = [self._safe_float(decision.get("confidence")) for decision in probability_decisions]
        correct = sum(1 for actual, predicted in zip(valid_y, predictions) if int(actual) == int(predicted))
        min_confidence = float(self.config.get("ML_SIGNAL_MIN_CONFIDENCE", 0.60) or 0.60)
        actionable = [
            (actual, predicted)
            for actual, predicted, confidence in zip(valid_y, predictions, confidences)
            if int(predicted) != 1 and float(confidence) >= min_confidence
        ]
        action_correct = sum(1 for actual, predicted in actionable if int(actual) == int(predicted))
        false_positive = sum(
            1
            for actual, predicted, confidence in zip(valid_y, predictions, confidences)
            if int(predicted) != 1 and float(confidence) >= min_confidence and int(actual) == 1
        )
        return {
            "validation_loss": float(loss.item()),
            "accuracy": correct / max(len(valid_y), 1),
            "false_positive_rate": false_positive / max(len(actionable), 1),
            "action_precision": action_correct / max(len(actionable), 1),
            "action_rate": len(actionable) / max(len(valid_y), 1),
            "action_count": len(actionable),
            "confidence_action_threshold": min_confidence,
            "action_policy": "directional_probability"
            if bool(self.config.get("ML_SIGNAL_DIRECTIONAL_ACTION_ENABLED", True))
            else "argmax",
            "min_action_probability": self._min_action_probability(),
            "min_directional_margin": self._min_directional_margin(),
            "max_hold_probability_for_action": self._max_hold_probability_for_action(),
        }

    def _probability_decision(self, probabilities: dict[str, float]) -> dict[str, Any]:
        sell_probability = max(0.0, self._safe_float(probabilities.get("sell")))
        hold_probability = max(0.0, self._safe_float(probabilities.get("hold")))
        buy_probability = max(0.0, self._safe_float(probabilities.get("buy")))
        total_probability = sell_probability + hold_probability + buy_probability
        if total_probability > 0:
            sell_probability /= total_probability
            hold_probability /= total_probability
            buy_probability /= total_probability
        probabilities = {
            "sell": sell_probability,
            "hold": hold_probability,
            "buy": buy_probability,
        }
        if not bool(self.config.get("ML_SIGNAL_DIRECTIONAL_ACTION_ENABLED", True)):
            action = max(probabilities, key=probabilities.get)
            confidence = probabilities[action]
            signed_expected = (buy_probability - sell_probability) * float(self.config.get("ML_TARGET_CAP", 1.0) or 1.0)
            return {
                "action": action,
                "confidence": confidence,
                "expected_return": signed_expected,
                "signed_expected_return": signed_expected,
                "action_probability": probabilities[action],
                "hold_probability": hold_probability,
                "directional_confidence": 0.0,
                "directional_margin": abs(buy_probability - sell_probability),
                "blockers": [],
            }

        action_mass = buy_probability + sell_probability
        if buy_probability >= sell_probability:
            side = "buy"
            action_probability = buy_probability
            opposite_probability = sell_probability
            signed_direction = 1.0
        else:
            side = "sell"
            action_probability = sell_probability
            opposite_probability = buy_probability
            signed_direction = -1.0
        directional_margin = abs(action_probability - opposite_probability)
        directional_confidence = action_probability / max(action_mass, 1e-12)
        blockers: list[str] = []
        min_confidence = float(self.config.get("ML_SIGNAL_MIN_CONFIDENCE", 0.60) or 0.60)
        if directional_confidence < min_confidence:
            blockers.append("ml_signal_confidence_below_threshold")
        if action_probability < self._min_action_probability():
            blockers.append("ml_signal_action_probability_below_threshold")
        if directional_margin < self._min_directional_margin():
            blockers.append("ml_signal_directional_margin_below_threshold")
        if hold_probability > self._max_hold_probability_for_action():
            blockers.append("ml_signal_hold_probability_above_threshold")

        expected_scale = self._signal_expected_return_scale()
        signed_expected = (buy_probability - sell_probability) * expected_scale
        if blockers:
            return {
                "action": "hold",
                "confidence": max(hold_probability, directional_confidence),
                "expected_return": signed_expected,
                "signed_expected_return": signed_expected,
                "action_probability": action_probability,
                "hold_probability": hold_probability,
                "directional_confidence": directional_confidence,
                "directional_margin": directional_margin,
                "blockers": blockers,
            }
        return {
            "action": side,
            "confidence": directional_confidence,
            "expected_return": directional_margin * expected_scale,
            "signed_expected_return": signed_expected,
            "action_probability": action_probability,
            "hold_probability": hold_probability,
            "directional_confidence": directional_confidence,
            "directional_margin": directional_margin,
            "blockers": [],
        }

    def _min_action_probability(self) -> float:
        return max(0.0, min(1.0, self._safe_float(self.config.get("ML_SIGNAL_MIN_ACTION_PROBABILITY"), 0.20)))

    def _min_directional_margin(self) -> float:
        return max(0.0, min(1.0, self._safe_float(self.config.get("ML_SIGNAL_MIN_DIRECTIONAL_MARGIN"), 0.05)))

    def _max_hold_probability_for_action(self) -> float:
        return max(0.0, min(1.0, self._safe_float(self.config.get("ML_SIGNAL_MAX_HOLD_PROBABILITY_FOR_ACTION"), 0.80)))

    def _signal_expected_return_scale(self) -> float:
        return max(
            0.0,
            self._safe_float(
                self.config.get("ML_SIGNAL_EXPECTED_RETURN_SCALE"),
                self._safe_float(self.config.get("ML_SIGNAL_TARGET_RETURN_THRESHOLD"), 0.001),
            ),
        )

    def _artifact_path(self, provider: str, horizon: str, model_type: str) -> Path:
        root = self.artifact_root or Path(current_app.instance_path) / "ml_models"
        root.mkdir(parents=True, exist_ok=True)
        provider_key = normalize_provider(provider)
        return root / f"ml-signal-{provider_key}-{horizon}-{model_type}-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.pt"

    def _load_artifact(self, path: str, torch: Any) -> dict[str, Any] | str:
        artifact_path = Path(path or "")
        if not artifact_path.exists():
            return "signal_artifact_missing"
        try:
            payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(artifact_path, map_location="cpu")
        return payload if isinstance(payload, dict) and "state_dict" in payload else "signal_artifact_invalid"

    def _prediction_context(self, context: dict[str, Any], candles: list[dict[str, Any]]) -> dict[str, Any]:
        payload = dict(context or {})
        if candles:
            closes = [self._safe_float(row.get("close")) for row in candles if isinstance(row, dict)]
            closes = [value for value in closes if value > 0]
            volumes = [self._safe_float(row.get("volume")) for row in candles if isinstance(row, dict)]
            volumes = [value for value in volumes if value > 0]
            if len(closes) >= 2:
                payload["recent_return"] = (closes[-1] - closes[-2]) / closes[-2]
            if len(closes) >= 12:
                payload["recent_1h_return"] = (closes[-1] - closes[-12]) / closes[-12]
            if len(volumes) >= 8:
                avg_volume = sum(volumes[-8:-1]) / max(len(volumes[-8:-1]), 1)
                payload["volume_impulse"] = volumes[-1] / max(avg_volume, 1e-9)
        payload.setdefault("high_upside_profile", True)
        return payload

    def _prediction_feature_sequence(
        self,
        provider_key: str,
        context: dict[str, Any],
        candles: list[dict[str, Any]],
        feature_names: list[str],
        sequence_length: int,
    ) -> list[list[float]]:
        sequence_length = max(1, int(sequence_length or 1))
        usable_candles = [row for row in candles if isinstance(row, dict)]
        if sequence_length <= 1 or len(usable_candles) < max(12, sequence_length):
            features = self.online_ranker.normalized_features(
                extract_features({**provider_feature_context(provider_key), **self._prediction_context(context, usable_candles)})
            )
            return [self._vector(features, feature_names)] * sequence_length

        step = max(1, len(usable_candles) // sequence_length)
        cutoffs = list(range(max(12, len(usable_candles) - (sequence_length - 1) * step), len(usable_candles) + 1, step))
        cutoffs = cutoffs[-sequence_length:]
        while len(cutoffs) < sequence_length:
            cutoffs.insert(0, cutoffs[0] if cutoffs else len(usable_candles))
        rows: list[list[float]] = []
        safe_context = {key: value for key, value in dict(context or {}).items() if key != "rapid_feature_candles"}
        for cutoff in cutoffs:
            window = usable_candles[: max(1, min(len(usable_candles), int(cutoff)))]
            try:
                payload = self.feature_factory.build(
                    symbol=str(context.get("symbol") or context.get("provider_symbol") or "BTC"),
                    timeframe=str(context.get("rapid_feature_timeframe") or context.get("timeframe") or "1m"),
                    candles=window,
                    optimizer_context={
                        **provider_feature_context(provider_key),
                        **safe_context,
                        "provider": provider_key,
                        "execution_venue": provider_key,
                    },
                    provider_context={**provider_feature_context(provider_key), **safe_context},
                    order_book={},
                    cutoff_timestamp=window[-1].get("timestamp") if window else None,
                )
                features = self.online_ranker.normalized_features(extract_features(payload))
            except Exception:  # noqa: BLE001
                features = self.online_ranker.normalized_features(
                    extract_features({**provider_feature_context(provider_key), **self._prediction_context(context, window)})
                )
            rows.append(self._vector(features, feature_names))
        return rows[-sequence_length:]

    def _sequence_examples(
        self,
        rows: list[MLSignalTrainingRow],
        feature_names: list[str],
        sequence_length: int,
    ) -> list[list[list[float]]]:
        sequence_length = max(1, int(sequence_length or 1))
        if sequence_length <= 1:
            return [[self._vector(row.features, feature_names)] for row in rows]
        grouped_vectors: dict[str, list[list[float]]] = {}
        output: list[list[list[float]]] = []
        for row in rows:
            key = str(row.sequence_key or row.source or row.provider or "global")
            vector = self._vector(row.features, feature_names)
            history = grouped_vectors.setdefault(key, [])
            padded = ([history[0]] * max(0, sequence_length - len(history) - 1)) if history else []
            sequence = (padded + history + [vector])[-sequence_length:]
            while len(sequence) < sequence_length:
                sequence.insert(0, vector)
            output.append(sequence)
            history.append(vector)
        return output

    def _sequence_length(self) -> int:
        return max(1, min(64, int(self.config.get("ML_SIGNAL_SEQUENCE_LENGTH", 8) or 8)))

    @staticmethod
    def _vector(features: dict[str, float], feature_names: list[str]) -> list[float]:
        return [float(features.get(name, 0.0) or 0.0) for name in feature_names]

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    def _config_or_context_float(self, context: dict[str, Any], key: str, default: float) -> float:
        return max(self._safe_float(context.get(key), self._safe_float(self.config.get(key.upper()), default)), 0.0)

    @staticmethod
    def _module_available(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            return False

    @staticmethod
    def _torch() -> Any:
        import torch

        return torch

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
            "created_at": record.created_at,
            "promoted_at": record.promoted_at,
        }


def _TorchGRUSignalNet(torch: Any, *, input_size: int, hidden_size: int) -> Any:
    class GRUSignalNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.head = torch.nn.Linear(hidden_size, len(ACTION_LABELS))

        def forward(self, inputs: Any) -> Any:
            output, _ = self.gru(inputs)
            return self.head(output[:, -1, :])

    return GRUSignalNet()
