"""Signal quality scoring shared by strategy execution paths."""

from __future__ import annotations

from typing import Any

from ..features.fibonacci import FibonacciLevels, FibonacciService
from ..ml.online_ranker import extract_features, horizon_from_duration
from .net_roi import net_roi_diagnostics, net_roi_v2_diagnostics, one_hour_edge_v2_diagnostics


EXPERIMENTAL_PROFILES = {"aggressive_1h", "dynamic_intraday", "extreme_roi_experimental"}


class SignalQualityEvaluator:
    """Combines strategy, feature, ML, Fibonacci, and market-quality signals."""

    def __init__(self, config: dict[str, Any], online_ranker: Any | None = None) -> None:
        self.config = config
        self.online_ranker = online_ranker

    def evaluate(
        self,
        *,
        symbol: str,
        timeframe: str,
        mode: str,
        run_parameters: dict[str, Any],
        signal: Any,
        feature_payload: dict[str, Any],
        mid: float,
        market_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if getattr(signal, "action", "hold") not in {"buy", "sell"} or mid <= 0:
            return {}

        side = "buy" if signal.action == "buy" else "sell"
        market_snapshot = market_snapshot or {}
        expected_move_bps = self._expected_move_bps(signal, run_parameters, mid)
        spread_bps = self._float(market_snapshot.get("spread_bps"))
        fee_bps = self._float(self.config.get("FEE_BPS")) * 2
        slippage_bps = self._float(self.config.get("SIM_SLIPPAGE_BPS"))
        atr_pct = self._float(feature_payload.get("atr_pct"))
        volatility_pct = self._float(market_snapshot.get("volatility_pct"))
        liquidity_usd = self._float(market_snapshot.get("liquidity_usd"))
        stability = self._float(market_snapshot.get("signal_stability"), 1.0)

        volatility_penalty_bps = min(35.0, max(0.0, atr_pct * 10_000 * 0.05 + volatility_pct * 0.8))
        liquidity_penalty_bps = self._liquidity_penalty(liquidity_usd)
        stability_penalty_bps = max(0.0, 1.0 - stability) * 12.0
        cost_drag_bps = spread_bps + fee_bps + slippage_bps + liquidity_penalty_bps

        fibonacci = self._fibonacci_score(feature_payload, side, mid, self._float(getattr(signal, "take_profit", 0.0)))
        feature = self._feature_score(feature_payload, side)
        one_hour = self._one_hour_confluence_score(feature_payload, side, run_parameters, market_snapshot)
        impulse_bps = self._trade_impulse_bps(market_snapshot.get("recent_trades"), side)
        ml_payload = self._ml_payload(symbol, timeframe, run_parameters, feature_payload)
        ml_bonus_bps = 0.0 if ml_payload.get("warmup", True) else max(-8.0, min(12.0, self._float(ml_payload.get("score")) * 12.0))

        confluence_bonus_bps = fibonacci["bonus_bps"] + feature["bonus_bps"] + one_hour["bonus_bps"] + ml_bonus_bps
        edge_score = (
            expected_move_bps
            - cost_drag_bps
            - volatility_penalty_bps
            - stability_penalty_bps
            + impulse_bps
            + confluence_bonus_bps
        )
        confidence = self._clamp(
            0.45
            + edge_score / 80.0
            + confluence_bonus_bps / 70.0
            + one_hour["score"] * 0.12
            + stability * 0.18
            - max(spread_bps - 8.0, 0.0) / 120.0
            - max(volatility_pct - 2.0, 0.0) / 12.0,
            0.0,
            1.0,
        )

        reasons = [*feature["reasons"], *fibonacci["reasons"], *one_hour["reasons"]]
        if impulse_bps > 0:
            reasons.append("recent trades support signal direction")
        elif impulse_bps < 0:
            reasons.append("recent trades oppose signal direction")
        if ml_bonus_bps > 0:
            reasons.append("online ranker supports candidate")
        elif ml_bonus_bps < 0:
            reasons.append("online ranker penalizes candidate")
        if market_snapshot.get("source"):
            reasons.append(f"market source: {market_snapshot.get('source')}")

        no_trade_reason = ""
        profile = str(run_parameters.get("optimizer_profile", "")).lower()
        if profile in EXPERIMENTAL_PROFILES:
            min_edge = self._min_edge(profile)
            min_confidence = self._float(self.config.get("EXTREME_ROI_MIN_CONFIDENCE"), 0.62) if profile == "extreme_roi_experimental" else 0.0
            min_liquidity = self._float(self.config.get("EXTREME_ROI_MIN_LIQUIDITY_USD"), 0.0) if profile == "extreme_roi_experimental" else 0.0
            min_stability = self._float(self.config.get("EXTREME_ROI_MIN_SIGNAL_STABILITY"), 0.55) if profile == "extreme_roi_experimental" else 0.0
            min_one_hour_confluence = self._float(self.config.get("ENSEMBLE_1H_MIN_CONFLUENCE"), 0.0) if profile == "aggressive_1h" else 0.0
            max_cost_drag = self._float(self.config.get("AGGRESSIVE_1H_MAX_COST_DRAG_BPS"), 18.0) if profile == "aggressive_1h" else 0.0
            min_one_hour_stability = self._float(self.config.get("AGGRESSIVE_1H_MIN_WINDOW_STABILITY"), 0.55) if profile == "aggressive_1h" else 0.0
            volatility_regime = str(market_snapshot.get("volatility_regime", "") or "").lower()
            if edge_score < min_edge:
                no_trade_reason = "low_edge_after_costs"
            elif max_cost_drag and cost_drag_bps > max_cost_drag:
                no_trade_reason = "cost_drag_above_threshold"
            elif min_confidence and confidence < min_confidence:
                no_trade_reason = "low_signal_confluence"
            elif min_one_hour_confluence and one_hour["score"] < min_one_hour_confluence:
                no_trade_reason = "low_1h_confluence"
            elif min_liquidity and liquidity_usd > 0 and liquidity_usd < min_liquidity:
                no_trade_reason = "insufficient_realtime_liquidity"
            elif min_one_hour_stability and stability < min_one_hour_stability:
                no_trade_reason = "unstable_realtime_signal"
            elif volatility_regime == "dislocated":
                no_trade_reason = "dislocated_volatility_regime"
            elif min_stability and stability < min_stability:
                no_trade_reason = "unstable_realtime_signal"

        maker_threshold = max(cost_drag_bps * 2.5, 18.0 if profile == "aggressive_1h" else 12.0)
        execution_style = "maker_limit" if edge_score > maker_threshold else "market"
        roi_context = {
            "edge_score": edge_score + cost_drag_bps,
            "expected_move_bps": expected_move_bps,
            "cost_drag_bps": cost_drag_bps,
            "spread_bps": spread_bps,
            "projected_slippage_bps": slippage_bps,
            "liquidity_usd": liquidity_usd,
            "volatility_regime": market_snapshot.get("volatility_regime", ""),
            "signal_stability": stability,
            "market_structure_score": market_snapshot.get("market_structure_score", 0.0),
            "market_structure_trend": market_snapshot.get("market_structure_trend", market_snapshot.get("market_structure_score", 0.0)),
            "stale_data": market_snapshot.get("stale_data", False),
            "stale_data_age_seconds": market_snapshot.get("stale_data_age_seconds", 0.0),
            "ml_score": ml_payload.get("score", 0.0),
            "window_stability": market_snapshot.get("window_stability", stability),
            "accepted_window_ratio": market_snapshot.get("accepted_window_ratio", stability),
            "max_drawdown": run_parameters.get("max_drawdown", market_snapshot.get("max_drawdown", 0.0)),
            "max_favorable_excursion": run_parameters.get("max_favorable_excursion", market_snapshot.get("max_favorable_excursion", 0.0)),
            "max_adverse_excursion": run_parameters.get("max_adverse_excursion", market_snapshot.get("max_adverse_excursion", 0.0)),
            "mfe_mae_ratio": run_parameters.get("mfe_mae_ratio", market_snapshot.get("mfe_mae_ratio", 0.0)),
            "avg_win": run_parameters.get("avg_win", 0.0),
            "avg_loss": run_parameters.get("avg_loss", 0.0),
            "expectancy": run_parameters.get("expectancy", edge_score),
            "recent_1h_return": run_parameters.get("recent_1h_return", 0.0),
            "cost_adjusted_recent_1h_return": run_parameters.get("cost_adjusted_recent_1h_return", 0.0),
            "cost_adjusted_expected_move": market_snapshot.get("cost_adjusted_expected_move", edge_score),
            "breakout_proximity_bps": market_snapshot.get("breakout_proximity_bps", 250.0),
            "turnover_after_fees": run_parameters.get("turnover_after_fees", 0.0),
            "trades_per_day": run_parameters.get("trades_per_day", 0.0),
            "avg_trade_return": run_parameters.get("avg_trade_return", 0.0),
        }
        roi_payload = net_roi_diagnostics(roi_context, self.config)
        roi_v2_payload = net_roi_v2_diagnostics(roi_context, self.config)
        one_hour_edge_payload = one_hour_edge_v2_diagnostics(
            {
                **roi_context,
                **roi_payload,
                **roi_v2_payload,
                "raw_upside_score": run_parameters.get("raw_upside_score", 0.0),
                "raw_total_return_pct": run_parameters.get("raw_total_return_pct", 0.0),
                "raw_net_return_pct": run_parameters.get("raw_net_return_pct", 0.0),
            },
            self.config,
        )
        if mode == "live" and not no_trade_reason:
            debounce_age = self._float(run_parameters.get("last_signal_age_seconds"), 1_000_000.0)
            debounce_seconds = self._float(self.config.get("SIGNAL_DEBOUNCE_SECONDS"), 45.0)
            if (
                debounce_age < debounce_seconds
                and str(run_parameters.get("last_signal_action") or "").lower() == getattr(signal, "action", "")
            ):
                no_trade_reason = "signal_debounce_active"
            elif self._float(roi_payload.get("expected_fill_quality")) < self._float(self.config.get("NET_ROI_MIN_FILL_QUALITY"), 0.55):
                no_trade_reason = "low_expected_fill_quality"
            elif execution_style == "market" and edge_score < self._float(self.config.get("NET_ROI_MIN_EDGE_BPS"), 4.0):
                no_trade_reason = "low_net_roi_edge"
            elif self._float(roi_payload.get("churn_penalty")) > self._float(self.config.get("NET_ROI_MAX_CHURN_PENALTY"), 0.35):
                no_trade_reason = "excessive_churn"
            elif bool(roi_payload.get("stale_data", False)):
                no_trade_reason = "stale_signal_market_data"
            elif str(roi_v2_payload.get("regime_support", "")).lower() == "regime-fragile":
                no_trade_reason = "fragile_roi_v2_regime"
            elif str(roi_v2_payload.get("roi_rejection_risk", "")).lower() == "high":
                no_trade_reason = "high_roi_v2_rejection_risk"
            elif self._float(one_hour_edge_payload.get("expected_execution_quality")) < self._float(self.config.get("ONE_HOUR_MIN_EXECUTION_QUALITY"), 0.60):
                no_trade_reason = "low_one_hour_execution_quality"
            elif str(one_hour_edge_payload.get("one_hour_edge_grade", "D")).upper() not in {"A", "B"}:
                no_trade_reason = "low_one_hour_edge_grade"

        signal_quality_breakdown = {
            "raw_strategy": {
                "action": getattr(signal, "action", "hold"),
                "rationale": getattr(signal, "rationale", ""),
                "expected_move_bps": expected_move_bps,
                "position_fraction": self._float(getattr(signal, "position_fraction", 0.0)),
            },
            "confluence": {
                "bonus_bps": confluence_bonus_bps,
                "fibonacci": fibonacci,
                "features": feature,
                "one_hour": one_hour,
            },
            "ml": ml_payload,
            "cost_drag": {
                "cost_drag_bps": cost_drag_bps,
                "spread_bps": spread_bps,
                "projected_slippage_bps": slippage_bps,
                "liquidity_penalty_bps": liquidity_penalty_bps,
            },
            "fill_quality": {
                "expected_fill_quality": roi_payload["expected_fill_quality"],
                "churn_penalty": roi_payload["churn_penalty"],
                "execution_style": execution_style,
            },
            "regime": {
                "support": roi_v2_payload["regime_support"],
                "bucket": roi_v2_payload["regime_bucket"],
                "volatility_regime": market_snapshot.get("volatility_regime", ""),
            },
            "risk": {
                "roi_quality_grade": roi_v2_payload["roi_quality_grade"],
                "roi_rejection_risk": roi_v2_payload["roi_rejection_risk"],
                "one_hour_edge_grade": one_hour_edge_payload["one_hour_edge_grade"],
                "profitability_blockers": one_hour_edge_payload["profitability_blockers"],
                "no_trade_reason": no_trade_reason,
            },
        }

        return {
            "edge_score": edge_score,
            "net_roi_score": roi_payload["net_roi_score"],
            "net_roi_v2_score": roi_v2_payload["net_roi_v2_score"],
            "one_hour_edge_v2": one_hour_edge_payload["one_hour_edge_v2"],
            "one_hour_edge_grade": one_hour_edge_payload["one_hour_edge_grade"],
            "expected_execution_quality": one_hour_edge_payload["expected_execution_quality"],
            "profitability_blockers": one_hour_edge_payload["profitability_blockers"],
            "raw_vs_net_roi_gap": one_hour_edge_payload["raw_vs_net_roi_gap"],
            "candidate_quality_breakdown": one_hour_edge_payload["candidate_quality_breakdown"],
            "roi_quality_grade": roi_v2_payload["roi_quality_grade"],
            "roi_rejection_risk": roi_v2_payload["roi_rejection_risk"],
            "regime_support": roi_v2_payload["regime_support"],
            "regime_bucket": roi_v2_payload["regime_bucket"],
            "tail_loss_penalty": roi_v2_payload["tail_loss_penalty"],
            "downside_asymmetry_penalty": roi_v2_payload["downside_asymmetry_penalty"],
            "cost_adjusted_breakout_potential": roi_v2_payload["cost_adjusted_breakout_potential"],
            "expected_fill_quality": roi_payload["expected_fill_quality"],
            "churn_penalty": roi_payload["churn_penalty"],
            "edge_after_cost_bps": edge_score,
            "data_age_seconds": roi_payload["data_age_seconds"],
            "net_roi_components": roi_payload["net_roi_components"],
            "net_roi_v2_components": roi_v2_payload["net_roi_v2_components"],
            "signal_quality_breakdown": signal_quality_breakdown,
            "confidence": confidence,
            "expected_move_bps": expected_move_bps,
            "cost_drag_bps": cost_drag_bps,
            "spread_bps": spread_bps,
            "projected_slippage_bps": slippage_bps,
            "volatility_penalty_bps": volatility_penalty_bps,
            "liquidity_penalty_bps": liquidity_penalty_bps,
            "stability_penalty_bps": stability_penalty_bps,
            "realtime_liquidity_usd": liquidity_usd,
            "realtime_volatility_pct": volatility_pct,
            "signal_stability": stability,
            "market_source": market_snapshot.get("source", "unknown"),
            "trade_impulse_bps": impulse_bps,
            "confluence_bonus_bps": confluence_bonus_bps,
            "fibonacci_alignment": fibonacci,
            "feature_confluence": feature,
            "one_hour_confluence": one_hour,
            "ml_signal_quality": ml_payload,
            "quality_reasons": reasons,
            "suggested_execution_style": execution_style,
            "no_trade_reason": no_trade_reason,
            "minimum_edge_bps": self._min_edge(profile),
        }

    def _expected_move_bps(self, signal: Any, parameters: dict[str, Any], mid: float) -> float:
        take_profit = self._float(getattr(signal, "take_profit", 0.0))
        if take_profit > 0:
            return abs(take_profit - mid) / mid * 10_000
        return self._float(parameters.get("take_profit_pct")) * 10_000

    def _feature_score(self, feature_payload: dict[str, Any], side: str) -> dict[str, Any]:
        bonus = 0.0
        reasons: list[str] = []
        trend = self._float(feature_payload.get("ema_trend"))
        trend_strength = min(abs(self._float(feature_payload.get("trend_strength"))) * 12.0, 1.0)
        if (side == "buy" and trend > 0) or (side == "sell" and trend < 0):
            bonus += 4.0 + trend_strength * 5.0
            reasons.append("EMA trend supports direction")
        elif trend:
            bonus -= 5.0
            reasons.append("EMA trend conflicts with direction")

        pattern = feature_payload.get("pattern_prediction") if isinstance(feature_payload.get("pattern_prediction"), dict) else {}
        label = str(pattern.get("label", "")).lower()
        confidence = self._float(pattern.get("confidence"))
        probability = self._float(pattern.get("probability"), 0.5)
        if label in {side, "bullish" if side == "buy" else "bearish"} or (side == "buy" and probability > 0.55) or (side == "sell" and probability < 0.45):
            bonus += min(confidence * 8.0, 6.0)
            reasons.append("pattern model supports direction")

        volume = feature_payload.get("volume_spike") if isinstance(feature_payload.get("volume_spike"), dict) else {}
        if bool(volume.get("is_spike", False)):
            bonus += min(self._float(volume.get("ratio")) * 1.5, 6.0)
            reasons.append("volume spike confirms move")

        external = feature_payload.get("external_scores") if isinstance(feature_payload.get("external_scores"), dict) else {}
        scores = [self._float(item.get("score")) for item in external.values() if isinstance(item, dict)]
        if scores:
            avg = sum(scores) / len(scores)
            directional = avg if side == "buy" else -avg
            bonus += max(-5.0, min(5.0, directional * 5.0))
            if directional > 0.1:
                reasons.append("external scores support direction")
        return {"bonus_bps": bonus, "reasons": reasons}

    def _fibonacci_score(self, feature_payload: dict[str, Any], side: str, mid: float, take_profit: float) -> dict[str, Any]:
        raw = feature_payload.get("fibonacci_levels") if isinstance(feature_payload.get("fibonacci_levels"), dict) else {}
        levels = self._levels_from_dict(raw)
        confluence = feature_payload.get("fibonacci_confluence") if isinstance(feature_payload.get("fibonacci_confluence"), dict) else {}
        if levels is None:
            confluence_bonus = self._fib_confluence_bonus(confluence, side)
            return {
                "bonus_bps": confluence_bonus["bonus_bps"],
                "reasons": confluence_bonus["reasons"],
                "confluence": confluence,
            }
        bonus = 0.0
        reasons: list[str] = []
        if (side == "buy" and levels.trend == "up") or (side == "sell" and levels.trend == "down"):
            bonus += 5.0
            reasons.append("Fibonacci swing trend supports direction")
        if FibonacciService.price_in_golden_zone(mid, levels):
            bonus += 4.0
            reasons.append("price is inside Fibonacci golden zone")
        target_direction = "above" if side == "buy" else "below"
        nearest = FibonacciService.nearest_level(take_profit or mid, levels, target_direction)
        distance_bps = abs((nearest - (take_profit or mid)) / mid * 10_000) if nearest and mid > 0 else 0.0
        if nearest and distance_bps <= 35:
            bonus += 4.0
            reasons.append("take profit aligns with Fibonacci extension/retracement")
        confluence_bonus = self._fib_confluence_bonus(confluence, side)
        bonus += confluence_bonus["bonus_bps"]
        reasons.extend(confluence_bonus["reasons"])
        return {
            "bonus_bps": bonus,
            "reasons": reasons,
            "trend": levels.trend,
            "nearest_target": nearest,
            "target_distance_bps": distance_bps,
            "confluence": confluence,
        }

    def _fib_confluence_bonus(self, confluence: dict[str, Any], side: str) -> dict[str, Any]:
        if not confluence:
            return {"bonus_bps": 0.0, "reasons": []}

        score = self._float(confluence.get("score"))
        cluster_count = int(self._float(confluence.get("cluster_count")))
        golden_zone_count = int(self._float(confluence.get("golden_zone_count")))
        trend_bias = str(confluence.get("trend_bias", "flat")).lower()
        support_distance = self._float(confluence.get("support_distance_bps"))
        resistance_distance = self._float(confluence.get("resistance_distance_bps"))
        tolerance = self._float(self.config.get("FIB_CONFLUENCE_CLUSTER_TOLERANCE_BPS"), 18.0)
        bonus = max(0.0, min(score, 1.0)) * 7.0
        reasons: list[str] = []

        if score > 0:
            reasons.append("multi-lookback Fibonacci confluence is present")
        if cluster_count >= 2:
            bonus += min(cluster_count, 5) * 0.8
            reasons.append("Fibonacci level cluster near price")
        if golden_zone_count:
            bonus += min(golden_zone_count, 3) * 1.2
            reasons.append("multiple Fibonacci golden zones overlap")
        if (side == "buy" and trend_bias == "up") or (side == "sell" and trend_bias == "down"):
            bonus += 2.5
            reasons.append("Fibonacci confluence trend bias supports direction")
        elif trend_bias in {"up", "down"}:
            bonus -= 2.0
            reasons.append("Fibonacci confluence trend bias conflicts with direction")
        if side == "buy" and 0 < support_distance <= tolerance:
            bonus += 2.0
            reasons.append("nearby Fibonacci support improves buy setup")
        if side == "sell" and 0 < resistance_distance <= tolerance:
            bonus += 2.0
            reasons.append("nearby Fibonacci resistance improves sell setup")

        return {"bonus_bps": max(-4.0, min(12.0, bonus)), "reasons": reasons}

    def _one_hour_confluence_score(
        self,
        feature_payload: dict[str, Any],
        side: str,
        parameters: dict[str, Any],
        market_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        profile = str(parameters.get("optimizer_profile", "")).lower()
        has_duration = "lock_duration_hours" in parameters or "duration_hours" in parameters
        duration = int(self._float(parameters.get("lock_duration_hours", parameters.get("duration_hours", 0)), 0.0))
        if profile != "aggressive_1h" and (not has_duration or duration > 1):
            return {"score": 0.0, "bonus_bps": 0.0, "reasons": []}

        trend = self._float(feature_payload.get("ema_trend"))
        trend_strength = abs(self._float(feature_payload.get("trend_strength")))
        macd_histogram = self._float(feature_payload.get("macd_histogram"))
        rsi = self._float(feature_payload.get("rsi"), 50.0)
        atr_pct = self._float(feature_payload.get("atr_pct"))
        volatility_pct = self._float(market_snapshot.get("volatility_pct"))
        stability = self._float(market_snapshot.get("signal_stability"), 1.0)
        bollinger = feature_payload.get("bollinger_bands") if isinstance(feature_payload.get("bollinger_bands"), dict) else {}
        percent_b = self._float(bollinger.get("percent_b"), 0.5)

        directional = 0.0
        reasons: list[str] = []
        if (side == "buy" and trend > 0 and macd_histogram >= 0) or (side == "sell" and trend < 0 and macd_histogram <= 0):
            directional += 0.35
            reasons.append("1h entry has trend and MACD confirmation")
        elif trend or macd_histogram:
            directional -= 0.25
            reasons.append("1h entry conflicts with trend or MACD")

        if trend_strength >= 0.002:
            directional += min(trend_strength * 25.0, 0.25)
        if stability >= 0.70:
            directional += 0.12
        elif stability < 0.55:
            directional -= 0.15
            reasons.append("real-time signal stability is weak")

        chop_penalty = 0.0
        if trend_strength < 0.0008 and (atr_pct > 0.006 or volatility_pct > 2.5):
            chop_penalty = 0.30
            reasons.append("anti-chop filter penalized noisy low-trend market")

        exhaustion_penalty = 0.0
        if side == "buy" and (rsi >= 78.0 or percent_b >= 0.97):
            exhaustion_penalty = 0.22
            reasons.append("momentum exhaustion penalized late buy")
        if side == "sell" and (0 < rsi <= 22.0 or percent_b <= 0.03):
            exhaustion_penalty = 0.22
            reasons.append("momentum exhaustion penalized late sell")

        score = self._clamp(0.45 + directional - chop_penalty - exhaustion_penalty, 0.0, 1.0)
        bonus = (score - 0.45) * 18.0
        return {
            "score": score,
            "bonus_bps": max(-8.0, min(10.0, bonus)),
            "trend_strength": trend_strength,
            "anti_chop_penalty": chop_penalty,
            "exhaustion_penalty": exhaustion_penalty,
            "reasons": reasons,
        }

    def _ml_payload(self, symbol: str, timeframe: str, parameters: dict[str, Any], feature_payload: dict[str, Any]) -> dict[str, Any]:
        if self.online_ranker is None or not bool(self.config.get("ML_RANKER_ENABLED", False)):
            return {"score": 0.0, "warmup": True}
        duration = parameters.get("lock_duration_hours") or parameters.get("duration_hours") or 1
        horizon = horizon_from_duration(duration)
        features = extract_features(
            {
                **feature_payload,
                "strategy_name": parameters.get("strategy_name"),
                "symbol": symbol,
                "timeframe": timeframe,
                "optimizer_profile": parameters.get("optimizer_profile"),
                "lock_duration_hours": duration,
                "horizon": horizon,
                "leverage": parameters.get("leverage", 1.0),
            }
        )
        explanation = self.online_ranker.explain(features, horizon)
        return {"score": self._float(explanation.get("prediction")), "warmup": bool(explanation.get("warmup", True)), "update_count": explanation.get("update_count", 0)}

    def _trade_impulse_bps(self, trades: Any, side: str) -> float:
        if not isinstance(trades, list) or len(trades) < 2:
            return 0.0
        first = self._float(trades[0].get("px") if isinstance(trades[0], dict) else None)
        last = self._float(trades[-1].get("px") if isinstance(trades[-1], dict) else None)
        if first <= 0 or last <= 0:
            return 0.0
        impulse = (last - first) / first * 10_000
        if side == "sell":
            impulse *= -1
        return max(-15.0, min(15.0, impulse * 0.25))

    def _liquidity_penalty(self, liquidity_usd: float) -> float:
        if liquidity_usd <= 0:
            return 0.0
        floor = self._float(self.config.get("EXTREME_ROI_MIN_LIQUIDITY_USD"), 1_000.0)
        if liquidity_usd >= floor:
            return 0.0
        return min(20.0, (floor - liquidity_usd) / max(floor, 1.0) * 20.0)

    def _min_edge(self, profile: str) -> float:
        if profile == "extreme_roi_experimental":
            return self._float(self.config.get("EXTREME_ROI_MIN_EDGE_BPS"), 8.0)
        return self._float(self.config.get("AGGRESSIVE_MIN_EDGE_BPS"), 4.0)

    def _levels_from_dict(self, raw: dict[str, Any]) -> FibonacciLevels | None:
        if not raw:
            return None
        try:
            return FibonacciLevels(
                swing_high=self._float(raw.get("swing_high")),
                swing_low=self._float(raw.get("swing_low")),
                trend=str(raw.get("trend", "flat")),
                lookback=int(self._float(raw.get("lookback"), 0)),
                retracements={str(k): self._float(v) for k, v in dict(raw.get("retracements") or {}).items()},
                extensions={str(k): self._float(v) for k, v in dict(raw.get("extensions") or {}).items()},
                golden_zone={str(k): self._float(v) for k, v in dict(raw.get("golden_zone") or {}).items()},
            )
        except Exception:
            return None

    @staticmethod
    def _clamp(value: float, floor: float, cap: float) -> float:
        return max(floor, min(cap, value))

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
