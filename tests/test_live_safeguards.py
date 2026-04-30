from __future__ import annotations

from datetime import datetime, timedelta

from app.extensions import db
from app.models import Fill, MLOfflineModel, Order, Setting, StrategyValidation
from app.services.order_manager import OrderIntent


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


def test_live_does_not_require_legacy_confirmations(app) -> None:
    risk_engine = app.extensions["services"]["risk_engine"]

    decision = risk_engine.evaluate(_intent(), market_price=100.0, has_trading_access=True)

    assert decision.approved


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


def test_high_upside_live_accepts_promoted_ml_without_ranking_blend(app, tmp_path) -> None:
    app.config["HIGH_UPSIDE_PROFILE_ENABLED"] = True
    app.config["HIGH_UPSIDE_LIVE_ELIGIBLE"] = True
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
