"""Dependency-free online ranker for strategy and vault candidate scoring."""

from __future__ import annotations

import math
import re
from typing import Any

from flask import has_app_context

from ..extensions import db
from ..models import MLModelState, MLTrainingEvent
from ..services.provider_assets import normalize_provider


NUMERIC_FEATURE_SCALES: dict[str, float] = {
    "net_return_after_costs": 0.25,
    "total_return": 0.30,
    "recent_return": 0.20,
    "recent_1h_return": 0.08,
    "recent_performance_score": 0.20,
    "max_drawdown": 0.50,
    "drawdown_penalty": 0.50,
    "profit_factor_edge": 3.0,
    "sortino_like": 5.0,
    "sharpe_like": 5.0,
    "consistency": 1.0,
    "window_stability": 1.0,
    "accepted_window_ratio": 1.0,
    "win_rate": 1.0,
    "trade_count": 100.0,
    "trades_per_day": 30.0,
    "avg_trade_return": 0.05,
    "edge_score": 75.0,
    "expectancy": 10.0,
    "cost_drag_bps": 100.0,
    "turnover_after_fees": 8.0,
    "turnover_rate": 8.0,
    "fees_to_equity": 0.10,
    "capacity_ratio": 4.0,
    "liquidity_usd": 250_000.0,
    "signal_stability": 1.0,
    "leverage": 5.0,
    "liquidation_buffer_pct": 0.25,
    "atr_pct": 0.06,
    "volatility": 0.08,
    "macd_histogram": 0.05,
    "bollinger_percent_b": 1.0,
    "bollinger_bandwidth": 0.20,
    "volume_spike_ratio": 5.0,
    "rule_score": 2.0,
    "fib_confluence_score": 1.0,
    "fib_cluster_count": 8.0,
    "fib_golden_zone_count": 3.0,
    "fib_nearest_level_bps": 100.0,
    "one_hour_confluence_score": 1.0,
    "multi_timeframe_confluence_score": 1.0,
    "multi_timeframe_cluster_count": 10.0,
    "multi_timeframe_invalidation_distance_bps": 200.0,
    "confluence_score": 1.0,
    "market_structure_score": 1.0,
    "pair_score": 1.0,
    "pair_correlation": 1.0,
    "pair_spread_zscore": 3.0,
    "pair_liquidity_balance": 1.0,
    "pair_spread_cost_bps": 100.0,
    "funding_rate": 0.01,
    "open_interest_change_pct": 0.25,
    "book_depth_score": 1.0,
    "spread_trend_bps": 100.0,
    "spread_depth_ratio": 100.0,
    "liquidation_proxy": 1.0,
    "volume_impulse": 5.0,
    "volume_impulse_persistence": 5.0,
    "momentum_acceleration": 0.05,
    "volatility_compression": 1.0,
    "volatility_expansion": 5.0,
    "breakout_proximity_bps": 250.0,
    "spread_stability": 1.0,
    "depth_stability": 1.0,
    "stale_data_age_seconds": 3_600.0,
    "cost_adjusted_expected_move": 100.0,
    "market_structure_trend": 1.0,
    "volatility_regime_score": 1.0,
    "ensemble_weight": 1.0,
    "mfe_mae_ratio": 8.0,
    "upside_screen_score": 25.0,
    "liquidity_capacity": 100_000.0,
    "convex_edge_score": 50.0,
    "net_roi_v2_score": 80.0,
    "regime_adjustment": 6.0,
    "regime_adjusted_expectancy": 10.0,
    "tail_loss_penalty": 1.0,
    "downside_asymmetry_penalty": 1.0,
    "cost_adjusted_breakout_potential": 120.0,
    "sustained_volume_impulse": 5.0,
    "pullback_quality": 1.0,
    "breakout_retest_success": 1.0,
    "volatility_expansion_after_compression": 5.0,
    "spread_stability_recent": 1.0,
    "cost_adjusted_expected_move_persistence": 100.0,
    "scanner_rejection_rate": 1.0,
    "offline_ml_prediction": 1.0,
    "duration_hours": 168.0,
    "order_book_depth_usd": 250_000.0,
    "fee_bps": 100.0,
    "maker_fee_bps": 100.0,
    "taker_fee_bps": 100.0,
}

CATEGORICAL_KEYS = (
    "strategy_name",
    "symbol",
    "timeframe",
    "horizon",
    "optimizer_profile",
    "market_regime",
    "market_structure_regime",
    "allocation_mode",
    "pair_mode",
    "pair_role",
    "no_trade_reason",
    "scanner_source",
    "volatility_regime",
    "rejection_reason",
    "roi_quality_grade",
    "roi_rejection_risk",
    "regime_support",
    "volatility_regime_bucket",
    "spread_cost_regime",
    "liquidity_regime",
    "trend_breakout_regime",
    "provider",
    "execution_venue",
    "collateral_asset",
)
BOOLEAN_FEATURES = {
    "aggressive_profile",
    "high_upside_profile",
    "volume_spike",
    "mtf_volume_confirmation",
    "mtf_rsi_confirmation",
    "mtf_momentum_exhaustion",
    "market_structure_enabled",
    "offline_ml_blend_enabled",
    "provider_is_hyperliquid",
    "provider_is_kucoin",
    "collateral_is_usdc",
    "collateral_is_usdt",
}

ONE_H10_HORIZON = "1h10"


def horizon_from_duration(duration_hours: int | float | str | None) -> str:
    """Map a lock duration to the model horizon key used for separate state."""

    duration = max(1, int(_safe_float(duration_hours, 1.0)))
    if duration <= 1:
        return "1h"
    if duration <= 24:
        return "24h"
    if duration <= 48:
        return "48h"
    if duration <= 168:
        return "7d"
    return "custom"


def is_one_h10_context(context: dict[str, Any] | Any) -> bool:
    payload = _as_dict(context)
    markers = {
        str(payload.get("algorithm_profile") or "").strip().lower(),
        str(payload.get("vault_cycle_name") or "").strip().lower(),
        str(payload.get("objective") or "").strip().lower(),
        str(payload.get("target_return_objective") or "").strip().lower(),
        str(payload.get("ml_horizon") or "").strip().lower(),
        str(payload.get("horizon") or "").strip().lower(),
    }
    return bool(payload.get("one_h10_vault")) or bool(payload.get("is_one_h10")) or bool(
        markers & {ONE_H10_HORIZON, "1h10", "one_h10", "one_hour_10x"}
    )


def horizon_from_context(context: dict[str, Any] | Any, duration_hours: int | float | str | None = None) -> str:
    payload = _as_dict(context)
    if is_one_h10_context(payload):
        return ONE_H10_HORIZON
    explicit = str(payload.get("ml_horizon") or "").strip().lower()
    if explicit:
        return explicit
    return horizon_from_duration(duration_hours if duration_hours is not None else _first_float(payload, "lock_duration_hours", "duration_hours") or 1)


def extract_features(context: dict[str, Any] | Any) -> dict[str, Any]:
    """Extract ranker-friendly features without reading future data.

    The caller is responsible for passing only prediction-time or out-of-sample
    metrics. This helper intentionally ignores timestamps and raw candle ranges.
    """

    payload = _as_dict(context)
    multi_timeframe_confluence = (
        payload.get("multi_timeframe_confluence") if isinstance(payload.get("multi_timeframe_confluence"), dict) else {}
    )
    market_structure = payload.get("market_structure") if isinstance(payload.get("market_structure"), dict) else {}
    pair_screening = payload.get("pair_screening") if isinstance(payload.get("pair_screening"), dict) else {}
    if "market_regime" not in payload and multi_timeframe_confluence.get("trend_regime"):
        payload["market_regime"] = multi_timeframe_confluence.get("trend_regime")
    if "market_structure_regime" not in payload and market_structure.get("volatility_regime"):
        payload["market_structure_regime"] = market_structure.get("volatility_regime")
    if "volatility_regime" not in payload and market_structure.get("volatility_regime"):
        payload["volatility_regime"] = market_structure.get("volatility_regime")
    if "scanner_source" not in payload and payload.get("source"):
        payload["scanner_source"] = payload.get("source")
    regime_bucket = payload.get("regime_bucket") if isinstance(payload.get("regime_bucket"), dict) else {}
    if regime_bucket:
        payload.setdefault("volatility_regime_bucket", regime_bucket.get("volatility"))
        payload.setdefault("spread_cost_regime", regime_bucket.get("spread_cost"))
        payload.setdefault("liquidity_regime", regime_bucket.get("liquidity"))
        payload.setdefault("trend_breakout_regime", regime_bucket.get("trend_breakout"))
    duration = _first_float(payload, "lock_duration_hours", "duration_hours")
    horizon = ONE_H10_HORIZON if is_one_h10_context(payload) else str(payload.get("horizon") or horizon_from_context(payload, duration or 1)).lower()
    payload["horizon"] = horizon
    features: dict[str, Any] = {
        "horizon": horizon,
        "duration_hours": duration or _duration_for_horizon(horizon),
    }

    for key in CATEGORICAL_KEYS:
        value = payload.get(key)
        if value not in {None, ""}:
            features[key] = str(value)

    net_return = _first_float(payload, "net_return_after_costs", "total_pnl_pct", "return_after_costs")
    total_return = _first_float(payload, "total_return", "return")
    recent_return = _first_float(payload, "recent_performance_score", "recent_return", "recent_1h_return")
    max_drawdown = _first_float(payload, "max_drawdown", "drawdown")
    profit_factor = _first_float(payload, "profit_factor")
    fees = _first_float(payload, "estimated_fees", "fees_paid")
    equity = max(_first_float(payload, "initial_balance", "starting_value_usd", default=0.0), 0.0)
    allocation = max(_first_float(payload, "allocation_amount_usd", default=0.0), 0.0)
    capacity = max(_first_float(payload, "capacity_usd", default=0.0), 0.0)
    volume_spike = payload.get("volume_spike") if isinstance(payload.get("volume_spike"), dict) else {}
    rule_decision = payload.get("rule_decision") if isinstance(payload.get("rule_decision"), dict) else {}
    fibonacci_confluence = payload.get("fibonacci_confluence") if isinstance(payload.get("fibonacci_confluence"), dict) else {}
    fibonacci_alignment = payload.get("fibonacci_alignment") if isinstance(payload.get("fibonacci_alignment"), dict) else {}
    one_hour_confluence = payload.get("one_hour_confluence") if isinstance(payload.get("one_hour_confluence"), dict) else {}
    max_adverse_excursion = abs(_first_float(payload, "max_adverse_excursion"))
    max_favorable_excursion = max(_first_float(payload, "max_favorable_excursion"), 0.0)
    mtf_score = _first_float(multi_timeframe_confluence, "score")
    scalar_confluence_score = _first_float(payload, "confluence_score")

    numeric = {
        "net_return_after_costs": net_return,
        "total_return": total_return,
        "recent_return": recent_return,
        "recent_1h_return": _first_float(payload, "recent_1h_return"),
        "recent_performance_score": _first_float(payload, "recent_performance_score"),
        "max_drawdown": max_drawdown,
        "drawdown_penalty": abs(min(max_drawdown, 0.0)),
        "profit_factor_edge": profit_factor - 1.0,
        "sortino_like": _first_float(payload, "sortino_like"),
        "sharpe_like": _first_float(payload, "sharpe_like"),
        "consistency": _first_float(payload, "consistency"),
        "window_stability": _first_float(payload, "window_stability"),
        "accepted_window_ratio": _first_float(payload, "accepted_window_ratio"),
        "win_rate": _first_float(payload, "win_rate"),
        "trade_count": _first_float(payload, "trade_count"),
        "trades_per_day": _first_float(payload, "trades_per_day"),
        "avg_trade_return": _first_float(payload, "avg_trade_return", "average_return_per_trade"),
        "edge_score": _first_float(payload, "edge_score"),
        "expectancy": _first_float(payload, "expectancy"),
        "cost_drag_bps": _first_float(payload, "cost_drag_bps"),
        "turnover_after_fees": _first_float(payload, "turnover_after_fees"),
        "turnover_rate": _first_float(payload, "turnover_rate", "capital_turnover_rate"),
        "fees_to_equity": fees / equity if equity > 0 else 0.0,
        "capacity_ratio": capacity / allocation if allocation > 0 and capacity > 0 else 1.0,
        "liquidity_usd": _first_float(payload, "liquidity_usd", "realtime_liquidity_usd", "capacity_usd"),
        "order_book_depth_usd": _first_float(payload, "order_book_depth_usd"),
        "signal_stability": _first_float(payload, "signal_stability", default=1.0),
        "fee_bps": _first_float(payload, "fee_bps", "estimated_fee_bps"),
        "maker_fee_bps": _first_float(payload, "maker_fee_bps"),
        "taker_fee_bps": _first_float(payload, "taker_fee_bps"),
        "leverage": _first_float(payload, "leverage", default=1.0),
        "liquidation_buffer_pct": _first_float(payload, "liquidation_buffer_pct", default=1.0),
        "atr_pct": _first_float(payload, "atr_pct"),
        "volatility": _first_float(payload, "volatility", "volatility_pct"),
        "macd_histogram": _first_float(payload, "macd_histogram"),
        "bollinger_percent_b": _safe_float((payload.get("bollinger_bands") or {}).get("percent_b")) if isinstance(payload.get("bollinger_bands"), dict) else 0.5,
        "bollinger_bandwidth": _safe_float((payload.get("bollinger_bands") or {}).get("bandwidth")) if isinstance(payload.get("bollinger_bands"), dict) else 0.0,
        "volume_spike_ratio": _safe_float(volume_spike.get("ratio")),
        "rule_score": _first_float(rule_decision, "score", "long_score", "short_score"),
        "fib_confluence_score": _first_float(fibonacci_confluence, "score"),
        "fib_cluster_count": _first_float(fibonacci_confluence, "cluster_count"),
        "fib_golden_zone_count": _first_float(fibonacci_confluence, "golden_zone_count"),
        "fib_nearest_level_bps": _nearest_fib_distance_bps(fibonacci_confluence, fibonacci_alignment),
        "one_hour_confluence_score": _first_float(one_hour_confluence, "score"),
        "multi_timeframe_confluence_score": mtf_score,
        "multi_timeframe_cluster_count": _first_float(multi_timeframe_confluence, "cluster_count"),
        "multi_timeframe_invalidation_distance_bps": _first_float(multi_timeframe_confluence, "invalidation_distance_bps"),
        "confluence_score": scalar_confluence_score or mtf_score,
        "market_structure_score": _first_float(payload, "market_structure_score") or _first_float(market_structure, "score"),
        "pair_score": _first_float(payload, "pair_score") or _first_float(pair_screening, "pair_score", "score"),
        "pair_correlation": _first_float(payload, "pair_correlation", "correlation") or _first_float(pair_screening, "correlation"),
        "pair_spread_zscore": abs(_first_float(payload, "pair_spread_zscore", "spread_zscore") or _first_float(pair_screening, "spread_zscore")),
        "pair_liquidity_balance": _first_float(payload, "pair_liquidity_balance", "liquidity_balance") or _first_float(pair_screening, "liquidity_balance"),
        "pair_spread_cost_bps": _first_float(payload, "pair_spread_cost_bps", "spread_cost_bps") or _first_float(pair_screening, "spread_cost_bps"),
        "funding_rate": _first_float(payload, "funding_rate") or _first_float(market_structure, "funding_rate"),
        "open_interest_change_pct": _first_float(payload, "open_interest_change_pct") or _first_float(market_structure, "open_interest_change_pct"),
        "book_depth_score": _first_float(payload, "book_depth_score") or _first_float(market_structure, "book_depth_score"),
        "spread_trend_bps": _first_float(payload, "spread_trend_bps") or _first_float(market_structure, "spread_trend_bps"),
        "spread_depth_ratio": _first_float(payload, "spread_depth_ratio") or _first_float(market_structure, "spread_depth_ratio"),
        "liquidation_proxy": _first_float(payload, "liquidation_proxy") or _first_float(market_structure, "liquidation_proxy"),
        "volume_impulse": _first_float(payload, "volume_impulse") or _first_float(market_structure, "volume_impulse"),
        "volume_impulse_persistence": _first_float(payload, "volume_impulse_persistence"),
        "momentum_acceleration": _first_float(payload, "momentum_acceleration"),
        "volatility_compression": _first_float(payload, "volatility_compression"),
        "volatility_expansion": _first_float(payload, "volatility_expansion"),
        "breakout_proximity_bps": _first_float(payload, "breakout_proximity_bps"),
        "spread_stability": _first_float(payload, "spread_stability"),
        "depth_stability": _first_float(payload, "depth_stability"),
        "stale_data_age_seconds": _first_float(payload, "stale_data_age_seconds"),
        "cost_adjusted_expected_move": _first_float(payload, "cost_adjusted_expected_move"),
        "market_structure_trend": _first_float(payload, "market_structure_trend"),
        "volatility_regime_score": _first_float(payload, "volatility_regime_score") or _first_float(market_structure, "volatility_regime_score"),
        "ensemble_weight": _first_float(payload, "ensemble_weight"),
        "mfe_mae_ratio": max_favorable_excursion / max(max_adverse_excursion, 1e-9),
        "upside_screen_score": _first_float(payload, "upside_screen_score"),
        "liquidity_capacity": _first_float(payload, "liquidity_capacity", "liquidity_capacity_usd"),
        "convex_edge_score": _first_float(payload, "convex_edge_score"),
        "net_roi_v2_score": _first_float(payload, "net_roi_v2_score"),
        "regime_adjustment": _first_float(payload, "regime_adjustment"),
        "regime_adjusted_expectancy": _first_float(payload, "regime_adjusted_expectancy"),
        "tail_loss_penalty": _first_float(payload, "tail_loss_penalty"),
        "downside_asymmetry_penalty": _first_float(payload, "downside_asymmetry_penalty"),
        "cost_adjusted_breakout_potential": _first_float(payload, "cost_adjusted_breakout_potential"),
        "sustained_volume_impulse": _first_float(payload, "sustained_volume_impulse"),
        "pullback_quality": _first_float(payload, "pullback_quality"),
        "breakout_retest_success": _first_float(payload, "breakout_retest_success"),
        "volatility_expansion_after_compression": _first_float(payload, "volatility_expansion_after_compression"),
        "spread_stability_recent": _first_float(payload, "spread_stability_recent"),
        "cost_adjusted_expected_move_persistence": _first_float(payload, "cost_adjusted_expected_move_persistence"),
        "scanner_rejection_rate": _first_float(payload, "scanner_rejection_rate"),
        "offline_ml_prediction": _first_float(payload, "offline_ml_prediction"),
    }

    for key, value in numeric.items():
        features[key] = value

    features["aggressive_profile"] = str(payload.get("optimizer_profile") or payload.get("profile") or "").lower() in {
        "aggressive_1h",
        "aggressive_risk_adjusted",
        "extreme_roi_experimental",
    }
    features["high_upside_profile"] = bool(payload.get("high_upside_profile", False))
    features["volume_spike"] = bool(volume_spike.get("is_spike", False))
    features["mtf_volume_confirmation"] = bool(multi_timeframe_confluence.get("volume_confirmation", False))
    features["mtf_rsi_confirmation"] = bool(multi_timeframe_confluence.get("rsi_confirmation", False))
    features["mtf_momentum_exhaustion"] = bool(multi_timeframe_confluence.get("momentum_exhaustion", False))
    features["market_structure_enabled"] = bool(market_structure.get("enabled", payload.get("market_structure_enabled", False)))
    features["offline_ml_blend_enabled"] = bool(payload.get("offline_ml_blend_enabled", False))
    features["provider_is_hyperliquid"] = str(payload.get("provider") or payload.get("execution_venue") or "").lower() == "hyperliquid"
    features["provider_is_kucoin"] = str(payload.get("provider") or payload.get("execution_venue") or "").lower() == "kucoin"
    features["collateral_is_usdc"] = str(payload.get("collateral_asset") or "").upper() == "USDC"
    features["collateral_is_usdt"] = str(payload.get("collateral_asset") or "").upper() == "USDT"
    return features


def outcome_from_result(result: dict[str, Any] | Any) -> float:
    """Convert an out-of-sample result payload into a bounded learning target."""

    payload = _as_dict(result)
    net_return = _first_float(payload, "net_return_after_costs", "total_return", "total_pnl_pct")
    recent = _first_float(payload, "recent_performance_score", "recent_return", "recent_1h_return")
    profit_factor = _first_float(payload, "profit_factor")
    consistency = _first_float(payload, "consistency", default=0.5)
    stability = _first_float(payload, "window_stability", "accepted_window_ratio", default=0.5)
    drawdown = abs(min(_first_float(payload, "max_drawdown", "drawdown"), 0.0))
    turnover = _first_float(payload, "turnover_after_fees", "turnover_rate")
    edge = _first_float(payload, "edge_score") / 10_000
    upside = _first_float(payload, "upside_screen_score") / 100.0
    convex = _first_float(payload, "convex_edge_score") / 100.0
    capacity_ratio = _first_float(payload, "capacity_ratio", default=1.0)
    trades = int(_first_float(payload, "trade_count"))
    low_trade_penalty = 0.04 if trades and trades < 5 else 0.0
    if trades <= 0:
        low_trade_penalty = 0.08

    outcome = (
        net_return * 1.4
        + recent * 0.7
        + max(profit_factor - 1.0, -1.0) * 0.05
        + (consistency - 0.5) * 0.08
        + (stability - 0.5) * 0.06
        + edge * 0.25
        + max(upside, 0.0) * 0.10
        + max(convex, 0.0) * 0.08
        + min(max(capacity_ratio - 1.0, 0.0), 4.0) * 0.015
        - drawdown * 0.8
        - max(turnover - 4.0, 0.0) * max(0.0, 0.01 - net_return) * 0.5
        - low_trade_penalty
    )
    if bool(payload.get("rejected", False)):
        outcome -= 0.05
    return _clip(outcome, -1.0, 1.0)


class OnlineRanker:
    """Small deterministic online linear model used to nudge candidate ranking."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        horizon: str = "global",
        state: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.horizon = str(horizon or "global")
        self._memory_state = state

    def predict_score(self, features: dict[str, Any], horizon: str | None = None) -> float:
        normalized = self.normalized_features(features)
        state = self._state_values(horizon or self._feature_horizon(features))
        score = state["bias"] + sum(state["weights"].get(key, 0.0) * value for key, value in normalized.items())
        return _clip(score, -state["prediction_cap"], state["prediction_cap"])

    def contextual_bandit_score(self, features: dict[str, Any], horizon: str | None = None) -> float:
        """Return the adaptive ensemble arm score for a strategy context."""

        return self.predict_score(features, self._bandit_horizon(horizon or self._feature_horizon(features)))

    def update_contextual_bandit(
        self,
        features: dict[str, Any],
        outcome: float,
        *,
        horizon: str | None = None,
        source: str = "vault_cycle",
        source_id: str | int | None = None,
        mode: str = "paper",
        metadata: dict[str, Any] | None = None,
        decay: float | None = None,
    ) -> dict[str, Any]:
        """Update the ensemble learner using exponentially decayed online state."""

        bandit_horizon = self._bandit_horizon(horizon or self._feature_horizon(features))
        decay_value = _clip(_safe_float(decay, self.config.get("ENSEMBLE_LEARNING_DECAY", 0.8)), 0.0, 1.0)
        if decay_value < 1.0:
            self._decay_state(bandit_horizon, decay_value)
        return self.update(
            features,
            outcome,
            horizon=bandit_horizon,
            source=source,
            source_id=source_id,
            mode=mode,
            metadata=metadata,
        )

    def update(
        self,
        features: dict[str, Any],
        outcome: float,
        *,
        horizon: str | None = None,
        source: str = "manual",
        source_id: str | int | None = None,
        mode: str = "paper",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        horizon_key = str(horizon or self._feature_horizon(features))
        mode = str(mode or "paper").lower()
        if mode == "live" and str(self.config.get("ML_LIVE_LEARNING_POLICY", "quarantine")).lower() == "quarantine":
            return self._record_quarantined_live_event(
                features,
                outcome,
                horizon=horizon_key,
                source=source,
                source_id=source_id,
                metadata=metadata,
            )
        if not self.should_update_from_mode(mode):
            return {"updated": False, "reason": "mode_not_allowed", "mode": mode, "horizon": horizon_key}

        normalized = self.normalized_features(features)
        return self._apply_normalized_update(
            normalized,
            outcome,
            horizon=horizon_key,
            source=source,
            source_id=source_id,
            mode=mode,
            metadata=metadata,
        )

    def _apply_normalized_update(
        self,
        normalized: dict[str, float],
        outcome: float,
        *,
        horizon: str,
        source: str,
        source_id: str | int | None,
        mode: str,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        horizon_key = str(horizon or self.horizon)
        state = self._state_values(horizon_key)
        prediction_before = _clip(
            state["bias"] + sum(state["weights"].get(key, 0.0) * value for key, value in normalized.items()),
            -state["prediction_cap"],
            state["prediction_cap"],
        )
        target = _clip(_safe_float(outcome), -float(self.config.get("ML_TARGET_CAP", 1.0)), float(self.config.get("ML_TARGET_CAP", 1.0)))
        error = target - prediction_before
        update_count = int(state["update_count"])
        learning_rate = state["learning_rate"] / math.sqrt(update_count + 1)
        l2 = state["l2"]
        weight_cap = state["weight_cap"]
        weights = dict(state["weights"])

        for key, value in normalized.items():
            current = weights.get(key, 0.0)
            regularized = current * (1.0 - learning_rate * l2)
            weights[key] = _clip(regularized + learning_rate * error * value, -weight_cap, weight_cap)

        bias = _clip(state["bias"] + learning_rate * error, -weight_cap, weight_cap)
        prediction_after = _clip(
            bias + sum(weights.get(key, 0.0) * value for key, value in normalized.items()),
            -state["prediction_cap"],
            state["prediction_cap"],
        )
        loss = error * error

        if self._memory_state is not None or not has_app_context():
            self._memory_state = {
                **state,
                "weights": weights,
                "bias": bias,
                "update_count": update_count + 1,
                "total_loss": state.get("total_loss", 0.0) + loss,
                "last_loss": loss,
            }
        else:
            record = self._model_state(horizon_key, create=True)
            record.weights = weights
            record.bias = bias
            record.update_count = update_count + 1
            record.total_loss = float(record.total_loss or 0.0) + loss
            record.last_loss = loss
            provider_key = normalize_provider((metadata or {}).get("provider") or (metadata or {}).get("execution_venue"))
            event = MLTrainingEvent(
                model_state_id=record.id,
                provider=provider_key,
                source=str(source or "unknown"),
                source_id=str(source_id) if source_id is not None else None,
                mode=mode,
                horizon=horizon_key,
                outcome=target,
                prediction_before=prediction_before,
                prediction_after=prediction_after,
                error=error,
            )
            event.features = normalized
            event.details = metadata or {}
            db.session.add(event)
            db.session.flush()

        return {
            "updated": True,
            "horizon": horizon_key,
            "prediction_before": prediction_before,
            "prediction_after": prediction_after,
            "outcome": target,
            "error": error,
            "learning_rate": learning_rate,
        }

    def _record_quarantined_live_event(
        self,
        features: dict[str, Any],
        outcome: float,
        *,
        horizon: str,
        source: str,
        source_id: str | int | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = self.normalized_features(features)
        state = self._state_values(horizon)
        prediction_before = _clip(
            state["bias"] + sum(state["weights"].get(key, 0.0) * value for key, value in normalized.items()),
            -state["prediction_cap"],
            state["prediction_cap"],
        )
        target = _clip(_safe_float(outcome), -float(self.config.get("ML_TARGET_CAP", 1.0)), float(self.config.get("ML_TARGET_CAP", 1.0)))
        error = target - prediction_before
        if has_app_context():
            record = self._model_state(horizon, create=True)
            provider_key = normalize_provider((metadata or {}).get("provider") or (metadata or {}).get("execution_venue"))
            event = MLTrainingEvent(
                model_state_id=record.id,
                provider=provider_key,
                source=str(source or "unknown"),
                source_id=str(source_id) if source_id is not None else None,
                mode="live",
                horizon=horizon,
                outcome=target,
                prediction_before=prediction_before,
                prediction_after=prediction_before,
                error=error,
            )
            event.features = normalized
            event.details = {
                **dict(metadata or {}),
                "status": "quarantined",
                "promotion_status": "pending",
                "learning_policy": "quarantine",
            }
            db.session.add(event)
            db.session.flush()
        return {
            "updated": False,
            "quarantined": True,
            "reason": "live_learning_quarantined",
            "horizon": horizon,
            "prediction_before": prediction_before,
            "prediction_after": prediction_before,
            "outcome": target,
            "error": error,
        }

    def promotion_diagnostics(self, horizon: str) -> dict[str, Any]:
        events = self._quarantined_events(horizon)
        losses = [float(event.error or 0.0) ** 2 for event in events]
        negative_error_rate = (
            sum(1 for event in events if float(event.error or 0.0) < 0.0) / len(events)
            if events
            else 0.0
        )
        mean_loss = sum(losses) / len(losses) if losses else 0.0
        drift = (
            abs(sum(float(event.outcome or 0.0) - float(event.prediction_before or 0.0) for event in events) / len(events))
            if events
            else 0.0
        )
        blockers: list[str] = []
        min_events = int(self.config.get("ML_LIVE_PROMOTION_MIN_EVENTS", 25))
        max_loss = float(self.config.get("ML_LIVE_PROMOTION_MAX_MEAN_LOSS", 0.20))
        max_negative = float(self.config.get("ML_LIVE_PROMOTION_MAX_NEGATIVE_ERROR_RATE", 0.55))
        if len(events) < min_events:
            blockers.append("insufficient_quarantined_events")
        if mean_loss > max_loss:
            blockers.append("mean_loss_above_threshold")
        if negative_error_rate > max_negative:
            blockers.append("negative_error_rate_above_threshold")
        if drift > float(self.config.get("ML_TARGET_CAP", 1.0)):
            blockers.append("prediction_drift_above_threshold")
        return {
            "horizon": str(horizon or "global"),
            "event_count": len(events),
            "mean_loss": mean_loss,
            "negative_error_rate": negative_error_rate,
            "drift": drift,
            "ready": not blockers,
            "blockers": blockers,
        }

    def promote_quarantined_events(self, horizon: str, *, source: str = "live_quarantine_promotion") -> dict[str, Any]:
        diagnostics = self.promotion_diagnostics(horizon)
        if not diagnostics["ready"]:
            return {"promoted": False, **diagnostics}
        events = self._quarantined_events(horizon)
        promoted = 0
        for event in events:
            result = self._apply_normalized_update(
                event.features,
                float(event.outcome or 0.0),
                horizon=str(horizon or event.horizon),
                source=source,
                source_id=event.id,
                mode="live_quarantine_promotion",
                metadata={"promoted_from_event_id": event.id},
            )
            details = dict(event.details or {})
            details["status"] = "promoted"
            details["promotion_status"] = "promoted"
            details["promotion_result"] = result
            event.details = details
            promoted += 1
        return {"promoted": True, "promoted_count": promoted, **self.promotion_diagnostics(horizon)}

    def _quarantined_events(self, horizon: str) -> list[MLTrainingEvent]:
        if not has_app_context():
            return []
        events = MLTrainingEvent.query.filter_by(mode="live", horizon=str(horizon or "global")).order_by(MLTrainingEvent.created_at.asc()).all()
        return [
            event
            for event in events
            if (event.details or {}).get("status") == "quarantined"
            and (event.details or {}).get("promotion_status", "pending") == "pending"
        ]

    def rank(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for candidate in candidates:
            features = candidate.get("features") if isinstance(candidate.get("features"), dict) else extract_features(candidate)
            horizon = self._feature_horizon(features)
            ml_score = self.predict_score(features, horizon)
            base_score = _safe_float(candidate.get("score"))
            adjusted = base_score + float(self.config.get("ML_SCORE_WEIGHT", 0.15)) * ml_score
            ranked.append({**candidate, "ml_score": ml_score, "ml_adjusted_score": adjusted})
        return sorted(ranked, key=lambda item: _safe_float(item.get("ml_adjusted_score")), reverse=True)

    def explain(self, features: dict[str, Any], horizon: str | None = None) -> dict[str, Any]:
        horizon_key = str(horizon or self._feature_horizon(features))
        normalized = self.normalized_features(features)
        state = self._state_values(horizon_key)
        contributions = [
            {"feature": key, "value": value, "weight": state["weights"].get(key, 0.0), "contribution": state["weights"].get(key, 0.0) * value}
            for key, value in normalized.items()
        ]
        contributions.sort(key=lambda item: abs(float(item["contribution"])), reverse=True)
        positives = [item for item in contributions if float(item["contribution"]) > 0][:5]
        negatives = [item for item in contributions if float(item["contribution"]) < 0][:5]
        return {
            "prediction": self.predict_score(features, horizon_key),
            "horizon": horizon_key,
            "bias": state["bias"],
            "update_count": state["update_count"],
            "warmup": state["update_count"] < int(self.config.get("ML_MIN_TRAINING_EVENTS", 25)),
            "top_positive": positives,
            "top_negative": negatives,
        }

    def normalized_features(self, features: dict[str, Any]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        payload = dict(features or {})

        for key in CATEGORICAL_KEYS:
            value = payload.get(key)
            if value in {None, ""}:
                continue
            normalized[f"{key}:{_token(value)}"] = 1.0

        for key, raw in payload.items():
            if key in CATEGORICAL_KEYS:
                continue
            if isinstance(raw, bool):
                if key in BOOLEAN_FEATURES:
                    normalized[key] = 1.0 if raw else 0.0
                continue
            if key not in NUMERIC_FEATURE_SCALES:
                continue
            value = _safe_float(raw, None)
            if value is None:
                continue
            scale = NUMERIC_FEATURE_SCALES.get(key, 1.0)
            normalized[key] = _clip(value / max(scale, 1e-9), -1.0, 1.0)

        return dict(sorted(normalized.items()))

    def is_warmed_up(self, horizon: str | None = None) -> bool:
        state = self._state_values(horizon or self.horizon)
        return int(state["update_count"]) >= int(self.config.get("ML_MIN_TRAINING_EVENTS", 25))

    def should_update_from_mode(self, mode: str) -> bool:
        mode = str(mode or "").lower()
        if mode in {"paper", "testnet"}:
            return True
        if mode == "live":
            return bool(self.config.get("ML_ALLOW_LIVE_UPDATES", False))
        return False

    def _model_state(self, horizon: str, *, create: bool) -> MLModelState:
        model_key = self._model_key(horizon)
        record = MLModelState.query.filter_by(model_key=model_key).one_or_none()
        if record is not None or not create:
            return record
        record = MLModelState(
            model_key=model_key,
            horizon=horizon,
            learning_rate=float(self.config.get("ML_RANKER_LEARNING_RATE", 0.03)),
            l2=float(self.config.get("ML_RANKER_L2", 0.001)),
            prediction_cap=float(self.config.get("ML_PREDICTION_CAP", 1.0)),
            weight_cap=float(self.config.get("ML_WEIGHT_CAP", 3.0)),
        )
        db.session.add(record)
        db.session.flush()
        return record

    def _state_values(self, horizon: str | None) -> dict[str, Any]:
        defaults = {
            "weights": {},
            "bias": 0.0,
            "learning_rate": float(self.config.get("ML_RANKER_LEARNING_RATE", 0.03)),
            "l2": float(self.config.get("ML_RANKER_L2", 0.001)),
            "prediction_cap": float(self.config.get("ML_PREDICTION_CAP", 1.0)),
            "weight_cap": float(self.config.get("ML_WEIGHT_CAP", 3.0)),
            "update_count": 0,
            "total_loss": 0.0,
            "last_loss": 0.0,
        }
        if self._memory_state is not None:
            return {**defaults, **self._memory_state, "weights": dict(self._memory_state.get("weights", {}))}
        if not has_app_context():
            return defaults
        record = self._model_state(str(horizon or self.horizon), create=False)
        if record is None:
            return defaults
        return {
            "weights": record.weights,
            "bias": float(record.bias or 0.0),
            "learning_rate": float(record.learning_rate or defaults["learning_rate"]),
            "l2": float(record.l2 or defaults["l2"]),
            "prediction_cap": float(record.prediction_cap or defaults["prediction_cap"]),
            "weight_cap": float(record.weight_cap or defaults["weight_cap"]),
            "update_count": int(record.update_count or 0),
            "total_loss": float(record.total_loss or 0.0),
            "last_loss": float(record.last_loss or 0.0),
        }

    def _feature_horizon(self, features: dict[str, Any]) -> str:
        return str((features or {}).get("horizon") or self.horizon or "global")

    @staticmethod
    def _model_key(horizon: str) -> str:
        return f"online_ranker:{str(horizon or 'global').lower()}"

    @staticmethod
    def _bandit_horizon(horizon: str | None) -> str:
        return f"bandit:{str(horizon or 'global').lower()}"

    def _decay_state(self, horizon: str, decay: float) -> None:
        decay = _clip(decay, 0.0, 1.0)
        if self._memory_state is not None:
            weights = {key: value * decay for key, value in dict(self._memory_state.get("weights", {})).items()}
            self._memory_state = {
                **self._memory_state,
                "weights": weights,
                "bias": _safe_float(self._memory_state.get("bias")) * decay,
                "update_count": max(0, int(int(self._memory_state.get("update_count", 0)) * decay)),
            }
            return
        if not has_app_context():
            return
        record = self._model_state(horizon, create=False)
        if record is None:
            return
        record.weights = {key: value * decay for key, value in record.weights.items()}
        record.bias = float(record.bias or 0.0) * decay
        record.update_count = max(0, int(int(record.update_count or 0) * decay))


def _as_dict(value: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    payload: dict[str, Any] = {}
    for key in dir(value):
        if key.startswith("_"):
            continue
        try:
            raw = getattr(value, key)
        except Exception:  # noqa: BLE001
            continue
        if callable(raw):
            continue
        payload[key] = raw
    return payload


def _first_float(payload: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in payload:
            return _safe_float(payload.get(key), default)
    return default


def _nearest_fib_distance_bps(confluence: dict[str, Any], alignment: dict[str, Any]) -> float:
    distances = [
        _safe_float(confluence.get("support_distance_bps")),
        _safe_float(confluence.get("resistance_distance_bps")),
        _safe_float(alignment.get("target_distance_bps")),
    ]
    positive = [float(value) for value in distances if value is not None and float(value) > 0]
    return min(positive) if positive else 0.0


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(float(value), upper))


def _token(value: Any) -> str:
    token = re.sub(r"[^a-z0-9_]+", "_", str(value).strip().lower())
    return token.strip("_") or "unknown"


def _duration_for_horizon(horizon: str) -> float:
    return {"1h": 1.0, "24h": 24.0, "48h": 48.0, "7d": 168.0}.get(str(horizon).lower(), 168.0)
