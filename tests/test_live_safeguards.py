from __future__ import annotations

from datetime import datetime, timedelta

from app.extensions import db
from app.models import Fill, MLOfflineModel, Order, Setting, StrategyValidation
from app.services.order_manager import OrderIntent
import app.services.risk_engine as risk_engine_module


def _intent() -> OrderIntent:
    return OrderIntent(
        symbol="BTC",
        side="buy",
        quantity=1.0,
        mode="live",
        stop_loss=95.0,
        strategy_name="ema_crossover",
        timeframe="15m",
    )


def _add_shadow_validation(strategy_name: str = "ema_crossover", timeframe: str = "15m", *, hours_ago: float = 0.0) -> None:
    db.session.add(
        StrategyValidation(
            strategy_name=strategy_name,
            symbol="BTC",
            timeframe=timeframe,
            stage="shadow_live",
            status="passed",
            completed_at=datetime.utcnow() - timedelta(hours=hours_ago),
        )
    )
    db.session.commit()


def _enable_high_upside_auto_live_gate(app) -> None:
    app.config["HIGH_UPSIDE_AUTO_LIVE_ENABLED"] = True
    app.config["HIGH_UPSIDE_REQUIRE_ML_SIGNAL"] = False
    app.config["EXPLICIT_LIVE_CONFIRMED"] = True
    app.config["SECONDARY_CONFIRMATION"] = True
    Setting.set_json("explicit_live_confirmed", True)
    Setting.set_json("secondary_confirmation", True)
    db.session.commit()


def test_live_does_not_require_legacy_confirmations(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]

    decision = risk_engine.evaluate(_intent(), market_price=100.0, has_trading_access=True)

    assert decision.approved


class _FakeMLPolicyEngine:
    def __init__(self, *args, action: str = "approve", **kwargs) -> None:
        self.action = action

    def family_readiness(self, family: str, horizon: str = "1h") -> dict:
        return {"ready": True, "family": family, "horizon": horizon, "blockers": []}

    def decision(self, family: str, context: dict, *, horizon: str = "1h", candles=None) -> dict:
        raw = {}
        action = "score"
        if family == "pytorch_risk_policy":
            action = self.action
            raw = {"approve": self.action == "approve", "risk_budget_usdc": min(context.get("notional", 0.0), 10.0)}
        elif family == "pytorch_exit_policy":
            action = "suggest"
            raw = {"suggested_stop_loss_pct": 0.005, "suggested_take_profit_pct": 0.012, "blockers": []}
        elif family == "pytorch_cap_policy":
            action = "cap"
            raw = {"suggested_notional_usdc": min(context.get("notional", 0.0), 10.0)}
        elif family == "pytorch_execution_policy":
            action = "route"
            raw = {"order_type_suggestion": "limit", "limit_offset_bps": 2.0}
        elif family == "pytorch_roi_target":
            action = "target_met_candidate"
            raw = {"target_probability": 0.75, "target_roi_pct": 1000.0}
        return {"ready": True, "family": family, "action": action, "blockers": [], "raw": raw}


def _enable_ml_policy(app) -> None:
    app.config["ML_RISK_POLICY_ENABLED"] = True
    app.config["ML_EXIT_POLICY_ENABLED"] = True
    app.config["ML_CAP_POLICY_ENABLED"] = True
    app.config["ML_ORDER_POLICY_ENABLED"] = True
    app.config["ML_ROI_TARGET_POLICY_ENABLED"] = True
    app.config["ML_POLICY_LIVE_AUTHORITY"] = "guarded"
    app.config["ML_LIVE_HARD_CAP_USDC"] = 10.0
    app.config["ML_LIVE_HARD_DAILY_LOSS_USDC"] = 0.50


def test_ml_policy_safety_envelope_blocks_missing_live_confirmations(app, monkeypatch) -> None:
    _enable_ml_policy(app)
    monkeypatch.setattr(risk_engine_module, "MLDecisionEngine", _FakeMLPolicyEngine)
    intent = _intent()
    intent.quantity = 0.05
    intent.take_profit = 105.0
    intent.metadata = {"ml_policy_required": True}

    decision = app.extensions["services"]["risk_engine"].evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "safety_envelope_blocked"
    assert "safety_explicit_live_confirmation_missing" in decision.details["blockers"]


def test_ml_policy_can_approve_inside_safety_envelope(app, monkeypatch) -> None:
    _enable_ml_policy(app)
    app.config["EXPLICIT_LIVE_CONFIRMED"] = True
    app.config["SECONDARY_CONFIRMATION"] = True
    Setting.set_json("explicit_live_confirmed", True)
    Setting.set_json("secondary_confirmation", True)
    db.session.commit()
    monkeypatch.setattr(risk_engine_module, "MLDecisionEngine", _FakeMLPolicyEngine)
    intent = _intent()
    intent.quantity = 0.05
    intent.take_profit = 105.0
    intent.metadata = {"ml_policy_required": True, "objective": "extreme_roi_1h"}

    decision = app.extensions["services"]["risk_engine"].evaluate(intent, market_price=100.0, has_trading_access=True)

    assert decision.approved
    assert decision.details["safety_envelope"]["ready"] is True
    assert "pytorch_risk_policy" in decision.details["ml_policy_decisions"]["ml_policy_decisions"]


def test_ml_policy_hard_cap_and_take_profit_remain_non_bypassable(app, monkeypatch) -> None:
    _enable_ml_policy(app)
    app.config["EXPLICIT_LIVE_CONFIRMED"] = True
    app.config["SECONDARY_CONFIRMATION"] = True
    Setting.set_json("explicit_live_confirmed", True)
    Setting.set_json("secondary_confirmation", True)
    db.session.commit()
    monkeypatch.setattr(risk_engine_module, "MLDecisionEngine", _FakeMLPolicyEngine)

    oversized = _intent()
    oversized.quantity = 0.2
    oversized.take_profit = 105.0
    oversized.metadata = {"ml_policy_required": True}
    cap_decision = app.extensions["services"]["risk_engine"].evaluate(oversized, market_price=100.0, has_trading_access=True)
    assert not cap_decision.approved
    assert cap_decision.rule_name == "safety_envelope_blocked"
    assert "safety_ml_live_hard_cap_breached" in cap_decision.details["blockers"]

    missing_take_profit = _intent()
    missing_take_profit.quantity = 0.05
    missing_take_profit.metadata = {"ml_policy_required": True}
    exit_decision = app.extensions["services"]["risk_engine"].evaluate(
        missing_take_profit,
        market_price=100.0,
        has_trading_access=True,
    )
    assert not exit_decision.approved
    assert exit_decision.rule_name == "take_profit_required"


def test_aggressive_1h_live_rejected_by_exposure_cap(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]
    _add_shadow_validation()
    intent = _intent()
    intent.metadata = {"optimizer_profile": "aggressive_1h"}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "aggressive_1h_live_cap"


def test_extreme_roi_live_rejected_by_exposure_cap(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]
    _add_shadow_validation()
    intent = _intent()
    intent.metadata = {"optimizer_profile": "extreme_roi_experimental"}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "aggressive_1h_live_cap"


def test_live_does_not_require_shadow_live_validation(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]

    decision = risk_engine.evaluate(_intent(), market_price=100.0, has_trading_access=True)

    assert decision.approved


def test_max_return_live_blocked_without_explicit_gate(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.metadata = {"profit_objective_version": "max_return_v3", "account_equity_usd": 2_000.0}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "max_return_live_disabled"


def test_dynamic_intraday_live_blocked_without_explicit_gate(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.metadata = {"optimizer_profile": "dynamic_intraday"}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "dynamic_intraday_live_disabled"


def test_dynamic_intraday_live_requires_shadow_validation_when_enabled(app) -> None:
    app.config["DYNAMIC_INTRADAY_LIVE_ELIGIBLE"] = True
    app.config["DYNAMIC_INTRADAY_REQUIRE_SHADOW_VALIDATION"] = True
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.quantity = 0.1
    intent.metadata = {"optimizer_profile": "dynamic_intraday", "account_equity_usd": 2_000.0}

    missing = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)
    assert not missing.approved
    assert missing.rule_name == "shadow_live_validation_required"

    _add_shadow_validation()
    allowed = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)
    assert allowed.approved


def test_high_upside_live_rejects_without_explicit_profile_flags(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.metadata = {"high_upside_profile": True, "duration_hours": 1, "account_equity_usd": 2_000.0}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "high_upside_profile_disabled"


def test_high_upside_live_requires_caps_and_shadow_validation(app) -> None:
    app.config["HIGH_UPSIDE_PROFILE_ENABLED"] = True
    app.config["HIGH_UPSIDE_LIVE_ELIGIBLE"] = True
    _enable_high_upside_auto_live_gate(app)
    app.config["HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD"] = 50.0
    app.config["HIGH_UPSIDE_MAX_DAILY_LOSS_USDC"] = 20.0
    app.config["HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION"] = {"1h": 100.0}
    app.config["HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION"] = {"1h": 0.5}
    app.config["HIGH_UPSIDE_REQUIRE_PROMOTED_ML"] = False
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.quantity = 0.2
    intent.metadata = {"high_upside_profile": True, "duration_hours": 1, "account_equity_usd": 2_000.0}

    missing = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)
    assert not missing.approved
    assert missing.rule_name == "shadow_live_validation_required"

    _add_shadow_validation()
    allowed = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)
    assert allowed.approved


def test_high_upside_live_fails_closed_when_required_caps_missing(app) -> None:
    app.config["HIGH_UPSIDE_PROFILE_ENABLED"] = True
    app.config["HIGH_UPSIDE_LIVE_ELIGIBLE"] = True
    _enable_high_upside_auto_live_gate(app)
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.metadata = {"high_upside_profile": True, "duration_hours": 1, "account_equity_usd": 2_000.0}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "high_upside_caps_missing"
    assert "HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD" in decision.details["missing"]


def test_high_upside_live_requires_promoted_offline_ml_by_default(app) -> None:
    app.config["HIGH_UPSIDE_PROFILE_ENABLED"] = True
    app.config["HIGH_UPSIDE_LIVE_ELIGIBLE"] = True
    _enable_high_upside_auto_live_gate(app)
    app.config["HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD"] = 50.0
    app.config["HIGH_UPSIDE_MAX_DAILY_LOSS_USDC"] = 20.0
    app.config["HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION"] = {"1h": 100.0}
    app.config["HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION"] = {"1h": 0.5}
    _add_shadow_validation()
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.quantity = 0.2
    intent.metadata = {"high_upside_profile": True, "duration_hours": 1, "account_equity_usd": 2_000.0}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "high_upside_promoted_ml_required"
    assert "promoted_model_missing" in decision.details["offline_ml_readiness"]["blockers"]


def test_high_upside_live_requires_promoted_ml_signal_when_enabled(app, monkeypatch) -> None:
    app.config["HIGH_UPSIDE_PROFILE_ENABLED"] = True
    app.config["HIGH_UPSIDE_LIVE_ELIGIBLE"] = True
    _enable_high_upside_auto_live_gate(app)
    app.config["HIGH_UPSIDE_REQUIRE_ML_SIGNAL"] = True
    app.config["HIGH_UPSIDE_REQUIRE_PROMOTED_ML"] = False
    app.config["ML_SIGNAL_MODEL_ENABLED"] = True
    app.config["ML_SIGNAL_REQUIRE_PROMOTED"] = True
    app.config["HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD"] = 50.0
    app.config["HIGH_UPSIDE_MAX_DAILY_LOSS_USDC"] = 20.0
    app.config["HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION"] = {"1h": 100.0}
    app.config["HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION"] = {"1h": 0.5}
    _add_shadow_validation()
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.quantity = 0.2
    intent.metadata = {"high_upside_profile": True, "duration_hours": 1, "account_equity_usd": 2_000.0}

    missing_model = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)
    assert not missing_model.approved
    assert missing_model.rule_name == "high_upside_promoted_ml_signal_required"

    class _ReadySignalModel:
        def __init__(self, config):
            self.config = config

        def readiness(self, horizon, *, require_promoted=True):
            return {"ready": True, "blockers": [], "horizon": horizon}

    monkeypatch.setattr(risk_engine_module, "MLSignalModel", _ReadySignalModel)
    missing_metadata = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)
    assert not missing_metadata.approved
    assert missing_metadata.rule_name == "high_upside_ml_signal_missing"

    intent.metadata["ml_signal_model"] = {
        "status": "promoted",
        "ready_for_live": True,
        "action": "buy",
        "confidence": 0.70,
        "blockers": [],
    }
    allowed = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)
    assert allowed.approved


def test_high_upside_live_accepts_promoted_ml_without_ranking_blend(app, tmp_path) -> None:
    app.config["HIGH_UPSIDE_PROFILE_ENABLED"] = True
    app.config["HIGH_UPSIDE_LIVE_ELIGIBLE"] = True
    _enable_high_upside_auto_live_gate(app)
    app.config["HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD"] = 50.0
    app.config["HIGH_UPSIDE_MAX_DAILY_LOSS_USDC"] = 20.0
    app.config["HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION"] = {"1h": 100.0}
    app.config["HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION"] = {"1h": 0.5}
    app.config["ML_OFFLINE_MODELS_ENABLED"] = True
    app.config["ML_OFFLINE_BLEND_ENABLED"] = False
    artifact = tmp_path / "promoted.joblib"
    artifact.write_text("fake artifact", encoding="utf-8")
    record = MLOfflineModel(
        model_key="offline_ranker:1h:sklearn:risk-gate",
        horizon="1h",
        model_type="sklearn",
        status="promoted",
        artifact_path=str(artifact),
        training_rows=25,
        validation_rows=5,
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
    _add_shadow_validation()
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.quantity = 0.2
    intent.metadata = {"high_upside_profile": True, "duration_hours": 1, "account_equity_usd": 2_000.0}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert decision.approved


def test_high_upside_live_auto_disables_on_offline_calibration_breach(app, tmp_path) -> None:
    app.config["HIGH_UPSIDE_PROFILE_ENABLED"] = True
    app.config["HIGH_UPSIDE_LIVE_ELIGIBLE"] = True
    _enable_high_upside_auto_live_gate(app)
    app.config["HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD"] = 50.0
    app.config["HIGH_UPSIDE_MAX_DAILY_LOSS_USDC"] = 20.0
    app.config["HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION"] = {"1h": 100.0}
    app.config["HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION"] = {"1h": 0.5}
    app.config["ML_OFFLINE_MODELS_ENABLED"] = True
    app.config["ML_OFFLINE_MAX_CALIBRATION_ERROR"] = 0.10
    artifact = tmp_path / "bad-calibration.joblib"
    artifact.write_text("fake artifact", encoding="utf-8")
    record = MLOfflineModel(
        model_key="offline_ranker:1h:sklearn:bad-calibration",
        horizon="1h",
        model_type="sklearn",
        status="promoted",
        artifact_path=str(artifact),
        training_rows=25,
        validation_rows=5,
        validation_loss=0.01,
        negative_error_rate=0.2,
        drift=0.01,
    )
    record.feature_names = ["net_return_after_costs"]
    record.metrics = {
        "calibration_error": 0.8,
        "top_decile_precision": 1.0,
        "false_positive_high_upside_rate": 0.0,
        "feature_importance": [],
    }
    db.session.add(record)
    db.session.commit()
    _add_shadow_validation()
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.quantity = 0.2
    intent.metadata = {"high_upside_profile": True, "duration_hours": 1, "account_equity_usd": 2_000.0}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "high_upside_promoted_ml_required"
    assert "calibration_error_above_threshold" in decision.details["offline_ml_readiness"]["blockers"]
    assert Setting.get_json("high_upside_live_disabled") is True
    assert Setting.get_json("high_upside_live_disabled_reason")["reason"] == "offline_model_diagnostic_breach"


def test_high_upside_daily_loss_breach_auto_disables_profile(app) -> None:
    app.config["HIGH_UPSIDE_PROFILE_ENABLED"] = True
    app.config["HIGH_UPSIDE_LIVE_ELIGIBLE"] = True
    _enable_high_upside_auto_live_gate(app)
    app.config["HIGH_UPSIDE_MAX_POSITION_NOTIONAL_USD"] = 50.0
    app.config["HIGH_UPSIDE_MAX_DAILY_LOSS_USDC"] = 1.0
    app.config["HIGH_UPSIDE_LIVE_CAP_USDC_BY_DURATION"] = {"1h": 100.0}
    app.config["HIGH_UPSIDE_LIVE_CAP_PCT_BY_DURATION"] = {"1h": 0.5}
    app.config["HIGH_UPSIDE_REQUIRE_PROMOTED_ML"] = False
    risk_engine = app.extensions["services"]["risk_engine"]
    _add_shadow_validation()
    loss_order = Order(
        client_order_id="high-upside-loss",
        mode="live",
        symbol="BTC",
        side="sell",
        order_type="market",
        status="filled",
        strategy_name="ema_crossover",
        quantity=1.0,
    )
    db.session.add(loss_order)
    db.session.flush()
    db.session.add(Fill(order_id=loss_order.id, symbol="BTC", side="sell", quantity=1.0, price=100.0, fee=0.1, pnl=-1.2, simulated=False))
    db.session.commit()
    intent = _intent()
    intent.quantity = 0.1
    intent.metadata = {"high_upside_profile": True, "duration_hours": 1, "account_equity_usd": 2_000.0}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "daily_loss_limit"
    assert Setting.get_json("high_upside_live_disabled") is True
    assert Setting.get_json("high_upside_live_disabled_reason")["reason"] == "daily_loss_limit_breach"


def test_max_return_live_eligible_reuses_existing_aggressive_caps(app) -> None:
    app.config["MAX_RETURN_LIVE_ELIGIBLE"] = True
    risk_engine = app.extensions["services"]["risk_engine"]
    _add_shadow_validation()
    intent = _intent()
    intent.quantity = 0.3
    intent.metadata = {
        "profit_objective_version": "max_return_v3",
        "duration_hours": 24,
        "account_equity_usd": 2_000.0,
    }

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "aggressive_1h_live_cap"
    assert decision.details["effective_cap"] == 25.0
    assert decision.details["duration_bucket"] == "24h"


def test_pair_live_blocked_without_explicit_gate(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.metadata = {"pair_group_id": "pair-stat-arb-btc-eth", "pair_mode": "stat_arb", "pair_symbol": "ETH"}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "pair_live_disabled"


def test_pair_live_requires_pair_shadow_validation(app) -> None:
    app.config["PAIR_LIVE_ELIGIBLE"] = True
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.quantity = 0.1
    intent.metadata = {
        "pair_group_id": "pair-stat-arb-btc-eth",
        "pair_mode": "stat_arb",
        "pair_symbol": "ETH",
        "account_equity_usd": 2_000.0,
    }

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "pair_shadow_live_validation_required"


def test_aggressive_1h_live_requires_recent_shadow_live_validation(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.quantity = 0.1
    intent.metadata = {"optimizer_profile": "aggressive_1h", "account_equity_usd": 2_000.0}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "shadow_live_validation_required"


def test_aggressive_1h_live_rejects_stale_shadow_live_validation(app) -> None:
    app.config["ENSEMBLE_1H_SHADOW_VALIDATION_MAX_AGE_HOURS"] = 1.0
    risk_engine = app.extensions["services"]["risk_engine"]
    _add_shadow_validation(hours_ago=2.0)
    intent = _intent()
    intent.quantity = 0.1
    intent.metadata = {"optimizer_profile": "aggressive_1h", "account_equity_usd": 2_000.0}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "shadow_live_validation_required"


def test_live_still_requires_stop_loss_for_opening_trade(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.stop_loss = None

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "stop_loss_required"


def test_aggressive_1h_live_allowed_when_hard_caps_pass(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]
    _add_shadow_validation()
    intent = _intent()
    intent.quantity = 0.1
    intent.metadata = {"optimizer_profile": "aggressive_1h", "account_equity_usd": 2_000.0}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert decision.approved


def test_aggressive_1h_live_rejected_when_exposure_cap_exceeded(app) -> None:
    app.config["AGGRESSIVE_1H_LIVE_CAP_USDC"] = 25.0
    app.config["AGGRESSIVE_1H_LIVE_CAP_PCT"] = 0.02
    risk_engine = app.extensions["services"]["risk_engine"]
    _add_shadow_validation()
    order = Order(
        client_order_id="aggressive-open-order",
        mode="live",
        symbol="BTC",
        side="buy",
        order_type="limit",
        status="open",
        strategy_name="ema_crossover",
        quantity=0.1,
        limit_price=100.0,
        stop_loss=95.0,
    )
    order.details = {"optimizer_profile": "aggressive_1h"}
    db.session.add(order)
    db.session.commit()
    intent = _intent()
    intent.quantity = 0.2
    intent.metadata = {"optimizer_profile": "aggressive_1h", "account_equity_usd": 2_000.0}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "aggressive_1h_live_cap"
    assert decision.reason == "Aggressive 1H live exposure exceeds configured cap."


def test_experimental_duration_live_uses_existing_caps_without_duration_map(app) -> None:
    app.config["EXPERIMENTAL_DURATION_ENSEMBLE_LIVE_ELIGIBLE"] = True
    risk_engine = app.extensions["services"]["risk_engine"]
    _add_shadow_validation()
    intent = _intent()
    intent.quantity = 0.3
    intent.metadata = {
        "ensemble_version": "duration_experimental_v1",
        "allocation_mode": "duration_experimental_ensemble",
        "duration_hours": 24,
        "account_equity_usd": 2_000.0,
    }

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "aggressive_1h_live_cap"
    assert decision.details["effective_cap"] == 25.0
    assert decision.details["duration_bucket"] == "24h"


def test_experimental_duration_live_cap_map_can_tighten_cap(app) -> None:
    app.config["EXPERIMENTAL_DURATION_ENSEMBLE_LIVE_ELIGIBLE"] = True
    app.config["EXPERIMENTAL_LIVE_CAP_USDC_BY_DURATION"] = {"24h": 10.0}
    app.config["EXPERIMENTAL_LIVE_CAP_PCT_BY_DURATION"] = {"24h": 0.01}
    risk_engine = app.extensions["services"]["risk_engine"]
    _add_shadow_validation()
    intent = _intent()
    intent.quantity = 0.2
    intent.metadata = {
        "ensemble_version": "duration_experimental_v1",
        "allocation_mode": "duration_experimental_ensemble",
        "duration_hours": 24,
        "account_equity_usd": 2_000.0,
    }

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "aggressive_1h_live_cap"
    assert decision.details["effective_cap"] == 10.0


def test_experimental_duration_live_cap_map_does_not_bypass_notional(app) -> None:
    app.config["EXPERIMENTAL_DURATION_ENSEMBLE_LIVE_ELIGIBLE"] = True
    app.config["EXPERIMENTAL_LIVE_CAP_USDC_BY_DURATION"] = {"24h": 1_000.0}
    app.config["EXPERIMENTAL_LIVE_CAP_PCT_BY_DURATION"] = {"24h": 1.0}
    app.config["MAX_POSITION_NOTIONAL"] = 15.0
    risk_engine = app.extensions["services"]["risk_engine"]
    _add_shadow_validation()
    intent = _intent()
    intent.quantity = 0.2
    intent.metadata = {
        "ensemble_version": "duration_experimental_v1",
        "allocation_mode": "duration_experimental_ensemble",
        "duration_hours": 24,
        "account_equity_usd": 2_000.0,
    }

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "max_notional"


def test_reduce_only_exit_is_not_blocked_by_opening_notional_cap(app) -> None:
    app.config["MAX_POSITION_NOTIONAL"] = 15.0
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.side = "sell"
    intent.quantity = 0.2
    intent.reduce_only = True
    intent.metadata = {"rapid_ml": True, "rapid_ml_exit": True}

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert decision.approved


def test_reduce_only_exit_is_not_blocked_by_allowed_symbols(app) -> None:
    app.config["ALLOWED_SYMBOLS"] = ["BTC", "ETH"]
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = OrderIntent(
        symbol="HYPE",
        side="buy",
        quantity=0.28,
        mode="live",
        reduce_only=True,
        stop_loss=42.8,
        take_profit=42.0,
        metadata={"rapid_ml": True, "rapid_ml_exit": True, "provider": "hyperliquid"},
    )

    decision = risk_engine.evaluate(intent, market_price=42.5, has_trading_access=True)

    assert decision.approved


def test_shadow_live_mode_is_not_supported_in_live_only_runtime(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]
    intent = _intent()
    intent.mode = "shadow_live"

    decision = risk_engine.evaluate(intent, market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "invalid_mode"


def test_daily_realized_pnl_filters_simulated_fills_once_by_mode(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]
    paper_order = Order(
        client_order_id="paper-fill",
        mode="paper",
        symbol="BTC",
        side="sell",
        order_type="market",
        status="filled",
        strategy_name="scalping",
        quantity=1.0,
    )
    live_order = Order(
        client_order_id="live-fill",
        mode="live",
        symbol="BTC",
        side="sell",
        order_type="market",
        status="filled",
        strategy_name="scalping",
        quantity=1.0,
    )
    db.session.add_all([paper_order, live_order])
    db.session.flush()
    db.session.add_all(
        [
            Fill(order_id=paper_order.id, symbol="BTC", side="sell", quantity=1.0, price=100.0, fee=0.5, pnl=5.0, simulated=True),
            Fill(order_id=paper_order.id, symbol="BTC", side="sell", quantity=1.0, price=100.0, fee=0.5, pnl=50.0, simulated=False),
            Fill(order_id=live_order.id, symbol="BTC", side="sell", quantity=1.0, price=100.0, fee=0.25, pnl=2.0, simulated=False),
            Fill(order_id=live_order.id, symbol="BTC", side="sell", quantity=1.0, price=100.0, fee=0.25, pnl=20.0, simulated=True),
        ]
    )
    db.session.commit()

    assert risk_engine.daily_realized_pnl("paper") == 4.5
    assert risk_engine.daily_realized_pnl("live") == 1.75


def test_live_filled_reduce_order_reconciles_fee_pnl_and_source_order(app, monkeypatch) -> None:
    manager = app.extensions["services"]["order_manager"]
    manager.trading_connections = None
    monkeypatch.setattr(manager.market_data, "get_mid_price", lambda symbol, mode: 100.0)

    source_order = Order(
        client_order_id="source-live-fill",
        mode="live",
        symbol="BTC",
        side="sell",
        order_type="market",
        status="filled",
        strategy_name="rapid_ml",
        quantity=1.0,
        filled_quantity=1.0,
        average_fill_price=100.0,
        reduce_only=False,
    )
    db.session.add(source_order)
    db.session.commit()

    class FakeClient:
        def can_trade(self, mode: str) -> bool:
            return True

        def place_order(self, *args, **kwargs):
            return {
                "status": "filled",
                "exchange_order_id": "exit-1",
                "fill_price": 97.0,
                "filled_quantity": 1.0,
                "raw": {"response": {"data": {"statuses": [{"filled": {"oid": "exit-1", "avgPx": "97", "totalSz": "1"}}]}}},
            }

        def get_recent_fills(self, mode: str):
            return [
                {
                    "symbol": "BTC",
                    "side": "buy",
                    "price": 97.0,
                    "size": 1.0,
                    "fee": 0.08,
                    "closed_pnl": 3.0,
                    "exchange_order_id": "exit-1",
                    "exchange_fill_id": "fill-1",
                }
            ]

    manager.client = FakeClient()

    order = manager.place_order(
        OrderIntent(
            symbol="BTC",
            side="buy",
            quantity=1.0,
            mode="live",
            reduce_only=True,
            stop_loss=105.0,
            strategy_name="rapid_ml",
            metadata={"source_order_id": source_order.id},
        )
    )

    fill = Fill.query.filter_by(order_id=order.id).one()
    assert fill.source_order_id == source_order.id
    assert fill.exchange_fill_id == "fill-1"
    assert fill.fee == 0.08
    assert fill.pnl == 3.0
    assert fill.fee_known is True
    assert fill.realized_pnl_known is True
    assert fill.details["reconciliation_status"] == "reconciled"
    assert app.extensions["services"]["risk_engine"].daily_realized_pnl("live") == 2.92


def test_unreconciled_reduce_fill_blocks_new_live_entries(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]
    order = Order(
        client_order_id="unknown-close",
        mode="live",
        symbol="BTC",
        side="buy",
        order_type="market",
        status="filled",
        strategy_name="rapid_ml",
        quantity=1.0,
        reduce_only=True,
    )
    db.session.add(order)
    db.session.flush()
    fill = Fill(
        order_id=order.id,
        symbol="BTC",
        side="buy",
        quantity=1.0,
        price=100.0,
        fee=0.05,
        pnl=0.0,
        simulated=False,
        realized_pnl_known=False,
        fee_known=False,
    )
    fill.details = {"reconciliation_status": "unknown_realized_pnl"}
    db.session.add(fill)
    db.session.commit()

    decision = risk_engine.evaluate(_intent(), market_price=100.0, has_trading_access=True)

    assert not decision.approved
    assert decision.rule_name == "unreconciled_live_fills"
