"""Consumer vault strategy selection and market-readiness scoring."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from ..features.fibonacci import FibonacciService
from ..features.multi_timeframe import MultiTimeframeConfluenceService
from ..ml.online_ranker import ONE_H10_HORIZON, OnlineRanker, extract_features, horizon_from_duration
from ..models import StrategyRanking
from ..runtime import market_mode_for
from .ensemble_allocator import DURATION_ENSEMBLE_VERSION, ENSEMBLE_VERSION, EnhancedEnsembleAllocator
from .net_roi import net_roi_diagnostics, net_roi_v2_diagnostics, one_hour_edge_v2_diagnostics
from .provider_assets import normalize_provider, provider_collateral_asset, provider_feature_context


SUPPORTED_VAULT_TIMEFRAMES = {"1m", "5m", "15m", "1h"}


@dataclass(frozen=True, slots=True)
class VaultSelection:
    profile: str
    strategy_name: str
    symbol: str
    timeframe: str
    mode: str
    execution_mode: str
    live_validation_status: str
    execution_substatus: str
    parameters: dict[str, Any]
    metadata: dict[str, Any]
    legs: list[dict[str, Any]]


class VaultStrategySelector:
    """Chooses an admin-visible strategy while exposing only profiles to users."""

    def __init__(
        self,
        config: dict[str, Any],
        market_data: Any,
        client: Any,
        universe_service: Any | None = None,
        online_ranker: OnlineRanker | None = None,
        realtime_market: Any | None = None,
        market_scanner: Any | None = None,
        market_structure: Any | None = None,
        pair_screening: Any | None = None,
    ) -> None:
        self.config = config
        self.market_data = market_data
        self.client = client
        self.universe_service = universe_service
        self.online_ranker = online_ranker or OnlineRanker(config)
        self.realtime_market = realtime_market
        self.market_scanner = market_scanner
        self.market_structure = market_structure
        self.pair_screening = pair_screening
        self.ml_decision_engine = None
        self.fibonacci_service = FibonacciService()
        self.multi_timeframe_confluence = MultiTimeframeConfluenceService(market_data, config, self.fibonacci_service)
        self.enhanced_allocator = EnhancedEnsembleAllocator(config, self.online_ranker)

    def select(
        self,
        asset: str,
        duration_hours: int,
        current_mode: str,
        allocation_amount_usd: float = 0.0,
        allowed_symbols: list[str] | None = None,
        provider: str | None = None,
    ) -> VaultSelection:
        asset = str(asset or "").upper()
        current_mode = str(current_mode or "live").lower()
        provider_key = normalize_provider(provider)
        collateral_asset = provider_collateral_asset(provider_key)

        symbol = self._resolve_symbol(asset, allowed_symbols)
        base = self._base_profile(duration_hours)
        base["allowed_symbols"] = self._normalize_symbols(allowed_symbols)
        base["duration_hours"] = duration_hours
        base["provider"] = provider_key
        base["collateral_asset"] = collateral_asset
        market_mode = market_mode_for(current_mode)
        timeframe = self._normalize_timeframe(str(base["timeframe"]))
        regime = self._market_regime(symbol, timeframe, market_mode)

        ranking = self._best_ranking(symbol, base, regime.get("preferred_strategy"), provider=provider_key)
        if ranking is not None:
            base["strategy_name"] = ranking.strategy_name
            base["timeframe"] = self._normalize_timeframe(ranking.timeframe)
            base["parameters"] = {**base["parameters"], **dict(ranking.parameters or {})}

        timeframe = self._normalize_timeframe(str(base["timeframe"]))

        realtime = self._market_snapshot(symbol, timeframe, market_mode)
        volatility_pct = self._safe_float(realtime.get("volatility_pct")) or self._recent_volatility_pct(
            symbol,
            timeframe,
            market_mode,
        )
        spread_bps = self._safe_float(realtime.get("spread_bps")) or self._spread_bps(symbol, market_mode)
        liquidity_usd = self._safe_float(realtime.get("liquidity_usd")) or self._book_liquidity_usd(symbol, market_mode)
        estimated_slippage_bps = self._estimated_slippage_bps(spread_bps, liquidity_usd)
        signal_stability = self._safe_float(realtime.get("signal_stability"), 1.0)
        reference_price = self._reference_price(symbol, market_mode, realtime)
        fib_confluence = self._fib_confluence(symbol, timeframe, market_mode, reference_price)
        multi_timeframe_confluence = self._multi_timeframe_confluence(
            symbol,
            timeframe,
            market_mode,
            reference_price,
        )
        market_structure = self._market_structure_snapshot(symbol, timeframe, market_mode)
        market_structure_score = self._safe_float(market_structure.get("score")) if isinstance(market_structure, dict) else 0.0
        pair_candidates = self._pair_candidates(symbol, timeframe, market_mode, duration_hours, base["allowed_symbols"])

        parameters = dict(base["parameters"])
        profile = str(base["profile"])
        ranking_roi_payload = self._ranking_net_roi_payload(ranking)
        ranking_roi_v2_payload = self._ranking_net_roi_v2_payload(ranking)
        ranking_one_hour_edge_payload = self._ranking_one_hour_edge_payload(ranking)

        metadata: dict[str, Any] = {
            **provider_feature_context(provider_key),
            "asset": asset,
            "collateral_asset": collateral_asset,
            "symbol": symbol,
            "duration_hours": duration_hours,
            "duration_bucket": EnhancedEnsembleAllocator.duration_bucket(duration_hours),
            "selection_reasons": list(base["reasons"]),
            "volatility_pct": volatility_pct,
            "spread_bps": spread_bps,
            "liquidity_usd": liquidity_usd,
            "estimated_slippage_bps": estimated_slippage_bps,
            "signal_stability": signal_stability,
            "fib_confluence": fib_confluence,
            "fibonacci_confluence": fib_confluence,
            "multi_timeframe_confluence": multi_timeframe_confluence,
            "market_structure": market_structure,
            "market_structure_score": market_structure_score,
            "pair_candidates": [candidate.as_dict() if hasattr(candidate, "as_dict") else dict(candidate) for candidate in pair_candidates[:5]],
            "pair_screening_summary": self._pair_screening_summary(pair_candidates),
            "market_data_source": realtime.get("source", "http"),
            "is_realtime_market": bool(realtime.get("is_realtime", False)),
            "recent_trade_count": len(realtime.get("recent_trades") or []),
            "optimizer_ranking_id": ranking.id if ranking else None,
            "optimizer_provider": normalize_provider(getattr(ranking, "provider", provider_key)) if ranking else None,
            "optimizer_score": float(ranking.score) if ranking else None,
            "optimizer_profile": ranking.profile if ranking else base.get("optimizer_profile"),
            "optimizer_recent_1h_return": float(ranking.recent_1h_return) if ranking else None,
            "optimizer_convex_edge_score": float(getattr(ranking, "convex_edge_score", 0.0)) if ranking else None,
            "optimizer_mfe_mae_ratio": float(getattr(ranking, "mfe_mae_ratio", 0.0)) if ranking else None,
            "optimizer_capacity_multiple": float(getattr(ranking, "capacity_multiple", 0.0)) if ranking else None,
            "optimizer_cost_adjusted_recent_1h_return": float(getattr(ranking, "cost_adjusted_recent_1h_return", 0.0)) if ranking else None,
            "optimizer_decay_penalty": float(getattr(ranking, "decay_penalty", 0.0)) if ranking else None,
            "optimizer_net_roi_score": ranking_roi_payload["net_roi_score"] if ranking else None,
            "optimizer_net_roi_v2_score": ranking_roi_v2_payload["net_roi_v2_score"] if ranking else None,
            "optimizer_one_hour_edge_v2": ranking_one_hour_edge_payload.get("one_hour_edge_v2") if ranking else None,
            "optimizer_one_hour_edge_grade": ranking_one_hour_edge_payload.get("one_hour_edge_grade") if ranking else None,
            "optimizer_expected_execution_quality": ranking_one_hour_edge_payload.get("expected_execution_quality") if ranking else None,
            "optimizer_profitability_blockers": ranking_one_hour_edge_payload.get("profitability_blockers", []) if ranking else [],
            "optimizer_raw_vs_net_roi_gap": ranking_one_hour_edge_payload.get("raw_vs_net_roi_gap") if ranking else None,
            "optimizer_candidate_quality_breakdown": ranking_one_hour_edge_payload.get("candidate_quality_breakdown", {}) if ranking else {},
            "optimizer_roi_quality_grade": ranking_roi_v2_payload["roi_quality_grade"] if ranking else None,
            "optimizer_roi_rejection_risk": ranking_roi_v2_payload["roi_rejection_risk"] if ranking else None,
            "optimizer_regime_support": ranking_roi_v2_payload["regime_support"] if ranking else None,
            "optimizer_regime_bucket": ranking_roi_v2_payload["regime_bucket"] if ranking else None,
            "optimizer_tail_loss_penalty": ranking_roi_v2_payload["tail_loss_penalty"] if ranking else None,
            "optimizer_downside_asymmetry_penalty": ranking_roi_v2_payload["downside_asymmetry_penalty"] if ranking else None,
            "optimizer_cost_adjusted_breakout_potential": ranking_roi_v2_payload["cost_adjusted_breakout_potential"] if ranking else None,
            "optimizer_expected_fill_quality": ranking_roi_payload["expected_fill_quality"] if ranking else None,
            "optimizer_churn_penalty": ranking_roi_payload["churn_penalty"] if ranking else None,
            "market_regime": regime.get("name"),
            "preferred_strategy": regime.get("preferred_strategy"),
            "allowed_symbols": base["allowed_symbols"],
        }
        if duration_hours <= 1:
            target_roi_pct = self._safe_float(
                parameters.get("target_roi_pct", self.config.get("ML_TARGET_ROI_1H10_PCT", self.config.get("ONE_H10_TARGET_ROI_PCT", 1000.0))),
                1000.0,
            )
            target_multiplier = 10.0
            metadata.update(
                {
                    "vault_cycle_name": "1H10",
                    "algorithm_profile": "1H10",
                    "ml_horizon": ONE_H10_HORIZON,
                    "one_h10_vault": True,
                    "target_roi_pct": target_roi_pct,
                    "target_multiplier": target_multiplier,
                    "target_return_objective": "one_h10",
                    "target_amount_usd": float(allocation_amount_usd or 0.0) * target_multiplier,
                    "target_copy": "1H10 aims to 10x the user's input amount in 1 hour.",
                    "non_guarantee_notice": "The 10x figure is a strategy objective, not a guaranteed return.",
                }
            )
            parameters.update(
                {
                    "vault_cycle_name": "1H10",
                    "algorithm_profile": "1H10",
                    "ml_horizon": ONE_H10_HORIZON,
                    "one_h10_vault": True,
                    "target_roi_pct": target_roi_pct,
                    "target_multiplier": target_multiplier,
                    "target_amount_usd": float(allocation_amount_usd or 0.0) * target_multiplier,
                }
            )
        if bool(parameters.get("high_upside_profile", False)):
            metadata["high_upside_profile"] = True
            metadata["selection_reasons"].append("high-upside profile tag requires explicit live caps and pre-live validation")
        if bool(self.config.get("MAX_RETURN_OPTIMIZER_ENABLED", False)):
            metadata["profit_objective_version"] = "max_return_v3"
            metadata["selection_reasons"].append("max-return v3 objective ranks accepted candidates by net return after costs")

        profile = self._apply_profile_adjustments(profile, parameters, metadata, volatility_pct)

        limited_reasons = self._limited_reasons(spread_bps, liquidity_usd, estimated_slippage_bps, signal_stability)
        if limited_reasons:
            profile = "Conservative"
            self._scale_risk(parameters, 0.5, floor=0.01)
            metadata["selection_reasons"].extend(limited_reasons)

        mode, execution_mode, live_status, substatus = self._execution_state(
            current_mode,
            limited_reasons,
            metadata,
        )

        parameters["risk_fraction"] = self._clamp(
            self._safe_float(parameters.get("risk_fraction"), 0.03),
            self._safe_float(self.config.get("VAULT_MIN_RISK_FRACTION"), 0.005),
            self._safe_float(self.config.get("VAULT_MAX_RISK_FRACTION"), 0.09),
        )
        parameters["leverage"] = self._adaptive_leverage(
            parameters,
            profile,
            volatility_pct,
            mode,
        )
        allocation_decision = self._allocation_ml_decision(
            ranking=ranking,
            parameters=parameters,
            metadata=metadata,
            duration_hours=duration_hours,
        )
        if allocation_decision:
            metadata["ml_decision"] = allocation_decision
            metadata["ml_allocator_decision"] = allocation_decision
            if (
                bool(allocation_decision.get("ready", False))
                and bool(self.config.get("ML_ALLOW_ALLOCATION_OVERRIDE", True))
                and bool(self.config.get("ML_DETERMINISTIC_SAFETY_GATES_REQUIRED", True))
            ):
                sizing_score = self._safe_float((allocation_decision.get("raw") or {}).get("sizing_score"), 1.0)
                if 0 < sizing_score < 1:
                    parameters["risk_fraction"] = min(
                        parameters["risk_fraction"],
                        max(self._safe_float(self.config.get("VAULT_MIN_RISK_FRACTION"), 0.005), parameters["risk_fraction"] * sizing_score),
                    )
        legs = self._allocation_legs(
            base=base,
            selected_ranking=ranking,
            selected_parameters=parameters,
            profile=profile,
            metadata=metadata,
            allocation_amount_usd=float(allocation_amount_usd or 0.0),
            mode=mode,
            market_mode=market_mode,
        )
        legs = [self._provider_tagged_leg(leg, provider_key, collateral_asset) for leg in legs]
        metadata["vault_leg_count"] = len(legs)

        return VaultSelection(
            profile=profile,
            strategy_name=str(base["strategy_name"]),
            symbol=symbol,
            timeframe=timeframe,
            mode=mode,
            execution_mode=execution_mode,
            live_validation_status=live_status,
            execution_substatus=substatus,
            parameters=parameters,
            metadata=metadata,
            legs=legs,
        )

    def _base_profile(self, duration_hours: int) -> dict[str, Any]:
        duration_hours = max(1, int(duration_hours or 1))

        if duration_hours <= 1:
            optimizer_profiles = ["aggressive_risk_adjusted", "aggressive_1h"]
            if bool(self.config.get("EXTREME_ROI_ENABLED", False)):
                optimizer_profiles.insert(0, "extreme_roi_experimental")
            return {
                "profile": "1H10",
                "optimizer_profile": "aggressive_1h",
                "optimizer_profiles": optimizer_profiles,
                "strategy_name": "scalping",
                "timeframe": "1m",
                "parameters": {
                    "one_h10_vault": True,
                    "ml_horizon": ONE_H10_HORIZON,
                    "objective": "one_h10",
                    "high_upside_profile": True,
                    "target_roi_pct": self._safe_float(
                        self.config.get("ML_TARGET_ROI_1H10_PCT", self.config.get("ONE_H10_TARGET_ROI_PCT", 1000.0)),
                        1000.0,
                    ),
                    "momentum_lookback": 4,
                    "minimum_move_pct": 0.0015,
                    "risk_fraction": self._default_risk_fraction(),
                    "stop_loss_pct": 0.003,
                    "take_profit_pct": 0.0045,
                    "leverage": 1.0,
                },
                "reasons": ["1H10 aims to 10x the user's input amount in 1 hour while remaining risk-gated"],
            }

        if duration_hours <= 24:
            return {
                "profile": "Conservative",
                "optimizer_profile": "aggressive_risk_adjusted",
                "optimizer_profiles": ["aggressive_risk_adjusted", None],
                "strategy_name": "mean_reversion",
                "timeframe": "5m",
                "parameters": {
                    "risk_fraction": self._default_risk_fraction(),
                    "entry_threshold_pct": 0.014,
                    "stop_loss_pct": 0.006,
                    "take_profit_pct": 0.011,
                    "leverage": 1.0,
                },
                "reasons": ["short duration uses tighter mean-reversion settings"],
            }

        if duration_hours <= 72:
            return {
                "profile": "Balanced",
                "optimizer_profile": "aggressive_risk_adjusted",
                "optimizer_profiles": ["aggressive_risk_adjusted", None],
                "strategy_name": "rule_based_signal",
                "timeframe": self._normalize_timeframe(self.config.get("DEFAULT_TIMEFRAME", "15m")),
                "parameters": {
                    "risk_fraction": self._default_risk_fraction(),
                    "minimum_signal_score": 0.9,
                    "fallback_stop_loss_pct": 0.007,
                    "fallback_take_profit_pct": 0.012,
                    "leverage": 1.0,
                },
                "reasons": ["medium duration uses balanced multi-factor allocation"],
            }

        return {
            "profile": "Aggressive",
            "optimizer_profile": "aggressive_risk_adjusted",
            "optimizer_profiles": ["aggressive_risk_adjusted", None],
            "strategy_name": "volatility_breakout",
            "timeframe": "1h",
            "parameters": {
                "risk_fraction": self._default_risk_fraction(),
                "lookback": 24,
                "range_multiplier": 1.5,
                "stop_loss_pct": 0.012,
                "take_profit_pct": 0.026,
                "leverage": 1.0,
            },
            "reasons": ["long duration uses volatility breakout allocation"],
        }

    def _best_ranking(
        self,
        symbol: str,
        base: dict[str, Any],
        preferred_strategy: str | None = None,
        provider: str | None = None,
    ) -> StrategyRanking | None:
        optimizer_profiles = list(base.get("optimizer_profiles") or [base.get("optimizer_profile")])
        profile = str(base.get("profile", ""))
        duration_hours = int(base.get("duration_hours", 0) or 0)
        provider_key = normalize_provider(provider or base.get("provider"))

        for optimizer_profile in optimizer_profiles:
            query = StrategyRanking.query.filter_by(symbol=symbol, rejected=False)
            query = self._provider_filtered_query(query, provider_key)

            if optimizer_profile:
                query = query.filter_by(profile=str(optimizer_profile))
            else:
                query = query.filter_by(experimental=False)

            candidates = self._ordered_rankings_query(query, optimizer_profile).limit(30).all()
            accepted = [
                ranking
                for ranking in candidates
                if self._ranking_acceptable(ranking, optimizer_profile, profile)
            ]
            accepted.sort(
                key=lambda row: (
                    self._provider_match_score(row, provider_key),
                    self._duration_match_score(row.lock_duration_hours, duration_hours),
                    self._one_hour_candidate_score(row, duration_hours)
                    if optimizer_profile in {"aggressive_1h", "extreme_roi_experimental"}
                    else self._ranking_score(row, duration_hours),
                    row.strategy_name == preferred_strategy,
                    row.created_at,
                ),
                reverse=True,
            )
            if accepted:
                return accepted[0]

        return None

    @staticmethod
    def _provider_filtered_query(query: Any, provider: str | None) -> Any:
        provider_key = normalize_provider(provider)
        if provider_key and provider_key != "global":
            return query.filter(StrategyRanking.provider.in_([provider_key, "global"]))
        return query

    @staticmethod
    def _provider_match_score(ranking: StrategyRanking, provider: str | None) -> int:
        provider_key = normalize_provider(provider)
        if not provider_key or provider_key == "global":
            return 1
        ranking_provider = normalize_provider(getattr(ranking, "provider", "global"))
        if ranking_provider == provider_key:
            return 2
        if ranking_provider == "global":
            return 1
        return 0

    def _ordered_rankings_query(self, query: Any, optimizer_profile: Any) -> Any:
        if str(optimizer_profile or "") == "aggressive_1h":
            return query.order_by(
                StrategyRanking.convex_edge_score.desc(),
                StrategyRanking.cost_adjusted_recent_1h_return.desc(),
                StrategyRanking.score.desc(),
                StrategyRanking.created_at.desc(),
            )
        return query.order_by(StrategyRanking.score.desc(), StrategyRanking.created_at.desc())

    def _ranking_acceptable(self, ranking: StrategyRanking, optimizer_profile: Any, profile: str) -> bool:
        if ranking is None or self._safe_float(ranking.score) <= 0:
            return False

        max_drawdown = self._safe_float(ranking.max_drawdown)
        drawdown_cap = self._safe_float(self.config.get("MAX_BACKTEST_DRAWDOWN_PCT"), 0.2)
        if optimizer_profile in {"aggressive_1h", "extreme_roi_experimental"}:
            if str(ranking.rejection_reason or "").strip():
                return False
            if self._is_high_upside_ranking(ranking):
                if not self._ranking_has_required_exits(ranking):
                    return False
                if bool(self.config.get("HIGH_UPSIDE_REQUIRE_PROMOTED_ML", True)) and not self._ranking_has_promoted_ml(ranking):
                    return False
            if self._safe_float(ranking.net_return_after_costs) < 0:
                return False
            if self._safe_float(ranking.recent_1h_return) <= 0:
                return False
            roi_payload = self._ranking_net_roi_payload(ranking)
            roi_v2_payload = self._ranking_net_roi_v2_payload(ranking)
            one_hour_edge = self._ranking_one_hour_edge_payload(ranking)
            explanation = ranking.ml_explanation if isinstance(ranking.ml_explanation, dict) else {}
            has_net_roi_payload = isinstance(explanation.get("net_roi"), dict)
            has_net_roi_v2_payload = isinstance(explanation.get("net_roi_v2"), dict)
            has_one_hour_edge_payload = isinstance(explanation.get("one_hour_edge_v2"), dict)
            edge_after_cost = self._safe_float(roi_payload.get("edge_after_cost_bps"))
            if has_net_roi_payload and edge_after_cost < self._safe_float(self.config.get("NET_ROI_MIN_EDGE_BPS"), 4.0):
                return False
            fill_quality = self._safe_float(roi_payload.get("expected_fill_quality"))
            if has_net_roi_payload and fill_quality < self._safe_float(self.config.get("NET_ROI_MIN_FILL_QUALITY"), 0.55):
                return False
            if has_one_hour_edge_payload:
                min_grade = str(self.config.get("ONE_HOUR_MIN_EDGE_GRADE", "B") or "B").upper()
                grade = str(one_hour_edge.get("one_hour_edge_grade", "D") or "D").upper()
                grade_order = {"A": 4, "B": 3, "C": 2, "D": 1}
                if grade_order.get(grade, 0) < grade_order.get(min_grade, 3):
                    return False
                if self._safe_float(one_hour_edge.get("expected_execution_quality")) < self._safe_float(self.config.get("ONE_HOUR_MIN_EXECUTION_QUALITY"), 0.60):
                    return False
                hard_blockers = {
                    "low_edge_after_costs",
                    "low_expected_execution_quality",
                    "fragile_regime",
                    "high_roi_rejection_risk",
                    "stale_data",
                }
                if hard_blockers.intersection(set(one_hour_edge.get("profitability_blockers", []) or [])):
                    return False
            if has_net_roi_v2_payload and str(roi_v2_payload.get("roi_rejection_risk", "high") or "high").lower() == "high":
                return False
            if has_net_roi_v2_payload and str(roi_v2_payload.get("regime_support", "regime-neutral") or "").lower() == "regime-fragile":
                return False
            live_pref = self._ranking_one_hour_live_preference_payload(ranking)
            if live_pref and live_pref.get("accepted_for_one_hour_live_preview") is False:
                return False
            min_pf = 1.0 if optimizer_profile == "extreme_roi_experimental" else 1.2
            if self._safe_float(ranking.profit_factor) < min_pf:
                return False
            min_trades_key = "EXTREME_ROI_MIN_TRADES" if optimizer_profile == "extreme_roi_experimental" else "AGGRESSIVE_1H_MIN_TRADES"
            if int(ranking.trade_count or 0) < int(self.config.get(min_trades_key, 8)):
                return False
            drawdown_key = "EXTREME_ROI_MAX_DRAWDOWN_PCT" if optimizer_profile == "extreme_roi_experimental" else "AGGRESSIVE_1H_MAX_DRAWDOWN_PCT"
            if max_drawdown <= -abs(self._safe_float(self.config.get(drawdown_key), 0.35)):
                return False
            edge_score = self._safe_float(getattr(ranking, "edge_score", 0.0))
            if str(getattr(ranking, "no_trade_reason", "") or "").strip():
                return False
            edge_key = "EXTREME_ROI_MIN_EDGE_BPS" if optimizer_profile == "extreme_roi_experimental" else "AGGRESSIVE_MIN_EDGE_BPS"
            if edge_score and edge_score < self._safe_float(self.config.get(edge_key), 4.0):
                return False
            if optimizer_profile == "aggressive_1h":
                cost_drag = self._safe_float(getattr(ranking, "cost_drag_bps", 0.0))
                max_cost_drag = self._safe_float(self.config.get("AGGRESSIVE_1H_MAX_COST_DRAG_BPS"), 18.0)
                if cost_drag > 0 and cost_drag > max_cost_drag:
                    return False
                stability = self._safe_float(getattr(ranking, "window_stability", 0.0))
                min_stability = self._safe_float(self.config.get("AGGRESSIVE_1H_MIN_WINDOW_STABILITY"), 0.55)
                if stability > 0 and stability < min_stability:
                    return False
                favorable = max(self._safe_float(getattr(ranking, "max_favorable_excursion", 0.0)), 0.0)
                adverse = abs(min(self._safe_float(getattr(ranking, "max_adverse_excursion", 0.0)), 0.0))
                mfe_mae = self._safe_float(getattr(ranking, "mfe_mae_ratio", 0.0))
                if mfe_mae <= 0 and favorable > 0 and adverse > 0:
                    mfe_mae = favorable / max(adverse, 1e-9)
                min_mfe_mae = self._safe_float(self.config.get("AGGRESSIVE_1H_MIN_MFE_MAE"), 1.5)
                if favorable > 0 and adverse > 0 and mfe_mae < min_mfe_mae:
                    return False
                allocation = self._safe_float(getattr(ranking, "allocation_amount_usd", 0.0))
                capacity_multiple = self._safe_float(getattr(ranking, "capacity_multiple", 0.0))
                if capacity_multiple <= 0 and allocation > 0 and self._safe_float(getattr(ranking, "capacity_usd", 0.0)) > 0:
                    capacity_multiple = (self._safe_float(getattr(ranking, "capacity_usd", 0.0)) / 0.05) / allocation
                min_capacity_multiple = self._safe_float(self.config.get("AGGRESSIVE_1H_MIN_CAPACITY_MULTIPLE"), 2.0)
                if allocation > 0 and capacity_multiple > 0 and capacity_multiple < min_capacity_multiple:
                    return False
        if optimizer_profile == "aggressive_risk_adjusted":
            if self._safe_float(ranking.net_return_after_costs) <= 0:
                return False
            if self._safe_float(ranking.profit_factor) < 1.05:
                return False
            if int(ranking.trade_count or 0) < int(self.config.get("AGGRESSIVE_RISK_ADJUSTED_MIN_TRADES", 6)):
                return False
            if max_drawdown <= -abs(self._safe_float(self.config.get("AGGRESSIVE_RISK_ADJUSTED_MAX_DRAWDOWN_PCT"), 0.25)):
                return False

        if profile == "Conservative" and max_drawdown <= -abs(drawdown_cap):
            return False

        return True

    @staticmethod
    def _is_high_upside_ranking(ranking: StrategyRanking) -> bool:
        params = ranking.parameters if isinstance(ranking.parameters, dict) else {}
        value = params.get("high_upside_profile", False)
        return value is True or str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _ranking_has_required_exits(self, ranking: StrategyRanking) -> bool:
        params = ranking.parameters if isinstance(ranking.parameters, dict) else {}
        stop_loss = max(
            self._safe_float(params.get("stop_loss_pct")),
            self._safe_float(params.get("stop_loss")),
            self._safe_float(params.get("stop_loss_price")),
        )
        take_profit = max(
            self._safe_float(params.get("take_profit_pct")),
            self._safe_float(params.get("take_profit")),
            self._safe_float(params.get("take_profit_price")),
        )
        return stop_loss > 0 and take_profit > 0

    def _ranking_has_promoted_ml(self, ranking: StrategyRanking) -> bool:
        params = ranking.parameters if isinstance(ranking.parameters, dict) else {}
        explanation = ranking.ml_explanation if isinstance(ranking.ml_explanation, dict) else {}
        scanner = explanation.get("scanner") if isinstance(explanation.get("scanner"), dict) else {}
        return str(
            params.get("offline_ml_status")
            or scanner.get("offline_ml_status")
            or explanation.get("offline_ml_status")
            or ""
        ).lower() == "promoted"

    def _one_hour_candidate_score(self, ranking: StrategyRanking, duration_hours: int) -> float:
        one_hour_edge = self._ranking_one_hour_edge_payload(ranking)
        explanation = ranking.ml_explanation if isinstance(ranking.ml_explanation, dict) else {}
        has_stored_one_hour_edge = isinstance(explanation.get("one_hour_edge_v2"), dict)
        edge_v2 = self._safe_float(one_hour_edge.get("one_hour_edge_v2"))
        execution_quality = self._safe_float(one_hour_edge.get("expected_execution_quality"))
        live_pref = self._ranking_one_hour_live_preference_payload(ranking)
        high_upside_score = self._safe_float(live_pref.get("one_hour_high_upside_score")) if live_pref else 0.0
        live_pref_rank = self._safe_float(live_pref.get("one_hour_live_preference_rank")) if live_pref else 0.0
        net_roi_v2 = self._ranking_net_roi_v2_score(ranking)
        net_roi = self._ranking_net_roi_score(ranking)
        convex = self._safe_float(getattr(ranking, "convex_edge_score", 0.0))
        if has_stored_one_hour_edge and edge_v2:
            return edge_v2 + max(execution_quality, 0.0) * 4.0 + max(net_roi_v2, 0.0) * 0.04
        if high_upside_score:
            return high_upside_score + max(live_pref_rank, 0.0) * 0.1 + max(net_roi_v2, 0.0) * 0.05
        if net_roi_v2:
            return net_roi_v2 + max(net_roi, 0.0) * 0.05 + max(convex, 0.0) * 0.01
        if net_roi:
            return net_roi + max(convex, 0.0) * 0.01
        if convex:
            return convex
        return self._ranking_score(ranking, duration_hours)

    def _ranking_net_roi_v2_score(self, ranking: StrategyRanking | None) -> float:
        if ranking is None:
            return 0.0
        return self._safe_float(self._ranking_net_roi_v2_payload(ranking).get("net_roi_v2_score"))

    def _ranking_net_roi_v2_payload(self, ranking: StrategyRanking | None) -> dict[str, Any]:
        if ranking is None:
            return {
                "net_roi_v2_score": 0.0,
                "roi_quality_grade": "D",
                "roi_rejection_risk": "high",
                "regime_bucket": {},
                "regime_support": "regime-neutral",
                "tail_loss_penalty": 0.0,
                "downside_asymmetry_penalty": 0.0,
                "cost_adjusted_breakout_potential": 0.0,
                "net_roi_v2_components": {},
            }
        explanation = ranking.ml_explanation if isinstance(ranking.ml_explanation, dict) else {}
        stored = explanation.get("net_roi_v2") if isinstance(explanation.get("net_roi_v2"), dict) else {}
        if stored and "net_roi_v2_score" in stored:
            return {
                "net_roi_v2_score": self._safe_float(stored.get("net_roi_v2_score")),
                "roi_quality_grade": str(stored.get("roi_quality_grade", "D") or "D"),
                "roi_rejection_risk": str(stored.get("roi_rejection_risk", "high") or "high"),
                "regime_bucket": stored.get("regime_bucket", {}) if isinstance(stored.get("regime_bucket"), dict) else {},
                "regime_support": str(stored.get("regime_support", "regime-neutral") or "regime-neutral"),
                "tail_loss_penalty": self._safe_float(stored.get("tail_loss_penalty")),
                "downside_asymmetry_penalty": self._safe_float(stored.get("downside_asymmetry_penalty")),
                "cost_adjusted_breakout_potential": self._safe_float(stored.get("cost_adjusted_breakout_potential")),
                "net_roi_v2_components": stored.get("components", {}) if isinstance(stored.get("components"), dict) else {},
            }
        return net_roi_v2_diagnostics(
            {
                "net_return_after_costs": ranking.net_return_after_costs,
                "total_return": ranking.total_return,
                "recent_performance_score": ranking.recent_performance_score,
                "recent_1h_return": ranking.recent_1h_return,
                "cost_adjusted_recent_1h_return": ranking.cost_adjusted_recent_1h_return,
                "max_drawdown": ranking.max_drawdown,
                "profit_factor": ranking.profit_factor,
                "window_stability": ranking.window_stability,
                "accepted_window_ratio": ranking.accepted_window_ratio,
                "edge_score": ranking.edge_score + ranking.cost_drag_bps,
                "expectancy": ranking.expectancy,
                "avg_win": ranking.avg_win,
                "avg_loss": ranking.avg_loss,
                "cost_drag_bps": ranking.cost_drag_bps,
                "turnover_after_fees": ranking.turnover_after_fees,
                "turnover_rate": ranking.turnover_rate,
                "trades_per_day": ranking.trades_per_day,
                "avg_trade_return": ranking.avg_trade_return,
                "allocation_amount_usd": ranking.allocation_amount_usd,
                "capacity_usd": ranking.capacity_usd,
                "capacity_multiple": ranking.capacity_multiple,
                "max_favorable_excursion": ranking.max_favorable_excursion,
                "max_adverse_excursion": ranking.max_adverse_excursion,
                "mfe_mae_ratio": ranking.mfe_mae_ratio,
                "decay_penalty": ranking.decay_penalty,
                "ml_score": ranking.ml_score,
            },
            self.config,
        )

    @staticmethod
    def _ranking_one_hour_live_preference_payload(ranking: StrategyRanking | None) -> dict[str, Any]:
        if ranking is None:
            return {}
        explanation = ranking.ml_explanation if isinstance(ranking.ml_explanation, dict) else {}
        payload = explanation.get("one_hour_live_preference")
        return payload if isinstance(payload, dict) else {}

    def _ranking_one_hour_edge_payload(self, ranking: StrategyRanking | None) -> dict[str, Any]:
        if ranking is None:
            return {}
        explanation = ranking.ml_explanation if isinstance(ranking.ml_explanation, dict) else {}
        stored = explanation.get("one_hour_edge_v2") if isinstance(explanation.get("one_hour_edge_v2"), dict) else {}
        if stored and "one_hour_edge_v2" in stored:
            return {
                "one_hour_edge_v2": self._safe_float(stored.get("one_hour_edge_v2")),
                "one_hour_edge_grade": str(stored.get("one_hour_edge_grade", "D") or "D"),
                "expected_execution_quality": self._safe_float(stored.get("expected_execution_quality")),
                "profitability_blockers": list(stored.get("profitability_blockers", []) or []),
                "raw_vs_net_roi_gap": self._safe_float(stored.get("raw_vs_net_roi_gap")),
                "candidate_quality_breakdown": stored.get("candidate_quality_breakdown", {}) if isinstance(stored.get("candidate_quality_breakdown"), dict) else {},
            }
        return one_hour_edge_v2_diagnostics(
            {
                "net_return_after_costs": ranking.net_return_after_costs,
                "total_return": ranking.total_return,
                "recent_performance_score": ranking.recent_performance_score,
                "recent_1h_return": ranking.recent_1h_return,
                "cost_adjusted_recent_1h_return": ranking.cost_adjusted_recent_1h_return,
                "max_drawdown": ranking.max_drawdown,
                "profit_factor": ranking.profit_factor,
                "window_stability": ranking.window_stability,
                "accepted_window_ratio": ranking.accepted_window_ratio,
                "edge_score": ranking.edge_score + ranking.cost_drag_bps,
                "expectancy": ranking.expectancy,
                "avg_win": ranking.avg_win,
                "avg_loss": ranking.avg_loss,
                "cost_drag_bps": ranking.cost_drag_bps,
                "turnover_after_fees": ranking.turnover_after_fees,
                "turnover_rate": ranking.turnover_rate,
                "trades_per_day": ranking.trades_per_day,
                "avg_trade_return": ranking.avg_trade_return,
                "allocation_amount_usd": ranking.allocation_amount_usd,
                "capacity_usd": ranking.capacity_usd,
                "capacity_multiple": ranking.capacity_multiple,
                "max_favorable_excursion": ranking.max_favorable_excursion,
                "max_adverse_excursion": ranking.max_adverse_excursion,
                "mfe_mae_ratio": ranking.mfe_mae_ratio,
                "decay_penalty": ranking.decay_penalty,
                "ml_score": ranking.ml_score,
            },
            self.config,
        )

    def _ranking_net_roi_score(self, ranking: StrategyRanking | None) -> float:
        if ranking is None:
            return 0.0
        return self._safe_float(self._ranking_net_roi_payload(ranking).get("net_roi_score"))

    def _ranking_net_roi_payload(self, ranking: StrategyRanking | None) -> dict[str, Any]:
        if ranking is None:
            return {
                "net_roi_score": 0.0,
                "expected_fill_quality": 0.0,
                "churn_penalty": 0.0,
                "edge_after_cost_bps": 0.0,
                "data_age_seconds": 0.0,
                "net_roi_components": {},
            }
        explanation = ranking.ml_explanation if isinstance(ranking.ml_explanation, dict) else {}
        stored = explanation.get("net_roi") if isinstance(explanation.get("net_roi"), dict) else {}
        if stored and "net_roi_score" in stored:
            return {
                "net_roi_score": self._safe_float(stored.get("net_roi_score")),
                "expected_fill_quality": self._safe_float(stored.get("expected_fill_quality")),
                "churn_penalty": self._safe_float(stored.get("churn_penalty")),
                "edge_after_cost_bps": self._safe_float(stored.get("edge_after_cost_bps")),
                "data_age_seconds": self._safe_float(stored.get("data_age_seconds")),
                "net_roi_components": stored.get("components", {}) if isinstance(stored.get("components"), dict) else {},
            }
        return net_roi_diagnostics(
            {
                "net_return_after_costs": ranking.net_return_after_costs,
                "total_return": ranking.total_return,
                "recent_performance_score": ranking.recent_performance_score,
                "recent_1h_return": ranking.recent_1h_return,
                "cost_adjusted_recent_1h_return": ranking.cost_adjusted_recent_1h_return,
                "max_drawdown": ranking.max_drawdown,
                "profit_factor": ranking.profit_factor,
                "window_stability": ranking.window_stability,
                "accepted_window_ratio": ranking.accepted_window_ratio,
                "edge_score": ranking.edge_score + ranking.cost_drag_bps,
                "expectancy": ranking.expectancy,
                "cost_drag_bps": ranking.cost_drag_bps,
                "turnover_after_fees": ranking.turnover_after_fees,
                "turnover_rate": ranking.turnover_rate,
                "trades_per_day": ranking.trades_per_day,
                "avg_trade_return": ranking.avg_trade_return,
                "allocation_amount_usd": ranking.allocation_amount_usd,
                "capacity_usd": ranking.capacity_usd,
                "capacity_multiple": ranking.capacity_multiple,
                "max_favorable_excursion": ranking.max_favorable_excursion,
                "max_adverse_excursion": ranking.max_adverse_excursion,
                "mfe_mae_ratio": ranking.mfe_mae_ratio,
                "decay_penalty": ranking.decay_penalty,
                "ml_score": ranking.ml_score,
            },
            self.config,
        )

    def _duration_match_score(self, ranking_duration: int | None, requested_duration: int) -> int:
        if not requested_duration:
            return 0
        if not ranking_duration:
            return 1
        ranking_bucket = horizon_from_duration(ranking_duration)
        requested_bucket = horizon_from_duration(requested_duration)
        if ranking_bucket == requested_bucket:
            return 2
        return 0

    def _ranking_score(self, ranking: StrategyRanking, duration_hours: int) -> float:
        base_score = self._safe_float(ranking.score)
        if not bool(self.config.get("ML_RANKER_ENABLED", False)):
            return base_score
        horizon = horizon_from_duration(duration_hours or ranking.lock_duration_hours or 0)
        if not self.online_ranker.is_warmed_up(horizon):
            return base_score
        roi_v2_payload = self._ranking_net_roi_v2_payload(ranking)
        features = extract_features(
            {
                **provider_feature_context(getattr(ranking, "provider", "global")),
                "strategy_name": ranking.strategy_name,
                "symbol": ranking.symbol,
                "timeframe": ranking.timeframe,
                "optimizer_profile": ranking.profile,
                "lock_duration_hours": duration_hours or ranking.lock_duration_hours,
                "horizon": horizon,
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
                "net_roi_v2_score": roi_v2_payload.get("net_roi_v2_score"),
                "roi_quality_grade": roi_v2_payload.get("roi_quality_grade"),
                "roi_rejection_risk": roi_v2_payload.get("roi_rejection_risk"),
                "regime_support": roi_v2_payload.get("regime_support"),
                "tail_loss_penalty": roi_v2_payload.get("tail_loss_penalty"),
                "downside_asymmetry_penalty": roi_v2_payload.get("downside_asymmetry_penalty"),
                "cost_adjusted_breakout_potential": roi_v2_payload.get("cost_adjusted_breakout_potential"),
                "expectancy": ranking.expectancy,
                "cost_drag_bps": ranking.cost_drag_bps,
                "turnover_after_fees": ranking.turnover_after_fees,
                "turnover_rate": ranking.turnover_rate,
                "estimated_fees": ranking.estimated_fees,
                "allocation_amount_usd": ranking.allocation_amount_usd,
                "leverage": ranking.leverage,
                "liquidation_buffer_pct": ranking.liquidation_buffer_pct,
                "capacity_usd": ranking.capacity_usd,
            }
        )
        return base_score + self._safe_float(self.config.get("ML_SCORE_WEIGHT"), 0.15) * self.online_ranker.predict_score(features, horizon)

    @staticmethod
    def _provider_tagged_leg(leg: dict[str, Any], provider: str, collateral_asset: str) -> dict[str, Any]:
        provider_key = normalize_provider(provider)
        collateral = str(collateral_asset or provider_collateral_asset(provider_key)).upper()
        tagged = dict(leg or {})
        tagged["provider"] = provider_key
        tagged["execution_venue"] = provider_key
        tagged["collateral_asset"] = collateral
        params = dict(tagged.get("parameters") or {})
        params.update(provider_feature_context(provider_key))
        params["collateral_asset"] = collateral
        tagged["parameters"] = params
        return tagged

    def _allocation_ml_decision(
        self,
        *,
        ranking: StrategyRanking | None,
        parameters: dict[str, Any],
        metadata: dict[str, Any],
        duration_hours: int,
    ) -> dict[str, Any]:
        if self.ml_decision_engine is None:
            return {}
        if not bool(self.config.get("ML_ALL_AREAS_ENABLED", False)):
            return {}
        horizon = str(parameters.get("ml_horizon") or metadata.get("ml_horizon") or horizon_from_duration(duration_hours or 1)).lower()
        ranking_payload = {
            "ranking_id": ranking.id,
            "score": ranking.score,
            "net_return_after_costs": ranking.net_return_after_costs,
            "profit_factor": ranking.profit_factor,
            "trade_count": ranking.trade_count,
            "max_drawdown": ranking.max_drawdown,
        } if ranking is not None else {}
        try:
            return dict(
                self.ml_decision_engine.decision(
                    "pytorch_allocator",
                    {
                        **ranking_payload,
                        **dict(parameters or {}),
                        **dict(metadata or {}),
                        "horizon": horizon,
                        "duration_hours": duration_hours,
                    },
                    horizon=horizon,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "family": "pytorch_allocator",
                "ready": False,
                "action": "hold",
                "blockers": [str(exc)],
                "audit_metadata": {"status": "ml_allocator_decision_error"},
            }

    def _allocation_legs(
        self,
        *,
        base: dict[str, Any],
        selected_ranking: StrategyRanking | None,
        selected_parameters: dict[str, Any],
        profile: str,
        metadata: dict[str, Any],
        allocation_amount_usd: float,
        mode: str,
        market_mode: str,
    ) -> list[dict[str, Any]]:
        max_legs = max(1, self._config_int("VAULT_MAX_PARALLEL_LEGS", 3))
        if self._experimental_duration_ensemble_enabled(base):
            max_legs = max(2, min(self._config_int("EXPERIMENTAL_DURATION_ENSEMBLE_MAX_LEGS", 5), 5))
        if self._enhanced_ensemble_enabled(base):
            max_legs = max(2, min(self._config_int("ENSEMBLE_MAX_LEGS", 5), 5))
        if self._one_hour_ensemble_enabled(base):
            max_legs = max(1, self._config_int("ENSEMBLE_1H_MAX_LEGS", max_legs))
        min_leg_usd = self._safe_float(self.config.get("VAULT_MIN_LEG_USD"), 10.0)
        fallback_leg = self._leg_from_selection(
            base,
            selected_ranking,
            selected_parameters,
            allocation_amount_usd,
            profile,
            metadata,
            mode,
        )

        if allocation_amount_usd < min_leg_usd * 2 or max_legs <= 1:
            if self._experimental_duration_ensemble_enabled(base) or self._enhanced_ensemble_enabled(base):
                metadata["allocation_mode"] = "single_best"
                metadata["skip_reason"] = "allocation_too_small_for_ensemble"
                metadata["rejected_candidate_count"] = 0
                metadata["cap_blocked_count"] = 0
            return [fallback_leg]

        pair_legs = self._pair_stat_arb_legs(
            base=base,
            selected_parameters=selected_parameters,
            profile=profile,
            metadata=metadata,
            allocation_amount_usd=allocation_amount_usd,
            mode=mode,
            min_leg_usd=min_leg_usd,
        )
        if len(pair_legs) >= 2:
            metadata["allocation_mode"] = "pair_stat_arb"
            metadata["selected_strategies"] = [str(leg.get("strategy_name")) for leg in pair_legs]
            metadata["individual_weights"] = {
                str(leg.get("symbol")): self._safe_float(leg.get("effective_allocation_weight"))
                for leg in pair_legs
            }
            return pair_legs

        if self._experimental_duration_ensemble_enabled(base):
            confluence = metadata.get("multi_timeframe_confluence") if isinstance(metadata.get("multi_timeframe_confluence"), dict) else {}
            if not bool(confluence.get("passed", False)):
                metadata["allocation_mode"] = "single_best"
                metadata["skip_reason"] = str(confluence.get("skip_reason") or "multi_timeframe_confluence_failed")
                metadata["ensemble_version"] = DURATION_ENSEMBLE_VERSION
                metadata["rejected_candidate_count"] = 0
                metadata["cap_blocked_count"] = 0
                return [fallback_leg]
            duration_bucket = str(metadata.get("duration_bucket") or EnhancedEnsembleAllocator.duration_bucket(base.get("duration_hours")))
            ensemble_id = f"ensemble-duration-{duration_bucket}-{str(metadata.get('symbol') or '').lower()}"
            metadata["ensemble_id"] = ensemble_id
            metadata["ensemble_version"] = DURATION_ENSEMBLE_VERSION
            metadata["duration_ensemble_enabled"] = True
            metadata["ensemble_primary_metric"] = str(
                self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_PRIMARY_METRIC", "net_return_after_costs")
            )
            candidates = self._duration_ensemble_rankings(base, market_mode, max_legs)
            ranked, skipped = self.enhanced_allocator.rank(
                candidates,
                duration_hours=int(base.get("duration_hours", 1) or 1),
                metadata=metadata,
            )
            metadata["enhanced_ensemble_skipped"] = skipped
            if len(ranked) >= 2:
                legs = self.enhanced_allocator.allocate(
                    ranked,
                    base_parameters=base["parameters"],
                    allocation_amount_usd=allocation_amount_usd,
                    profile=profile,
                    metadata=metadata,
                    adaptive_leverage=self._adaptive_leverage,
                    min_leg_usd=min_leg_usd,
                    mode=mode,
                )
                if len(legs) >= 2:
                    metadata["allocation_mode"] = "duration_experimental_ensemble"
                    metadata["ensemble_learning_enabled"] = bool(self.config.get("ENSEMBLE_LEARNING_ENABLED", True))
                    metadata["selected_strategies"] = [str(leg.get("strategy_name")) for leg in legs]
                    metadata["individual_weights"] = {
                        str(leg.get("strategy_name")): self._safe_float(leg.get("effective_allocation_weight"))
                        for leg in legs
                    }
                    metadata["target_weights"] = {
                        str(leg.get("strategy_name")): self._safe_float(leg.get("target_ensemble_weight"))
                        for leg in legs
                    }
                    metadata["ml_rank_scores"] = {
                        str(leg.get("strategy_name")): self._safe_float(leg.get("ml_rank_score"))
                        for leg in legs
                    }
                    metadata["confluence_score"] = self._safe_float(confluence.get("score"))
                    metadata["baseline_comparison"] = self._baseline_comparison(candidates, selected_ranking)
                    return legs
            metadata["allocation_mode"] = "single_best"
            metadata["skip_reason"] = "not_enough_duration_ensemble_candidates"
            metadata["rejected_candidate_count"] = len(skipped) + max(0, len(candidates) - len(ranked))
            metadata["cap_blocked_count"] = int(metadata.get("cap_blocked_count", 0) or 0)
            return [fallback_leg]

        if self._enhanced_ensemble_enabled(base):
            confluence = metadata.get("multi_timeframe_confluence") if isinstance(metadata.get("multi_timeframe_confluence"), dict) else {}
            if not bool(confluence.get("passed", False)):
                metadata["allocation_mode"] = "single_best"
                metadata["skip_reason"] = str(confluence.get("skip_reason") or "multi_timeframe_confluence_failed")
                metadata["ensemble_version"] = ENSEMBLE_VERSION
                metadata["rejected_candidate_count"] = 0
                metadata["cap_blocked_count"] = 0
                return [fallback_leg]
            ensemble_id = f"ensemble-v2-1h-{str(metadata.get('symbol') or '').lower()}-{int(base.get('duration_hours', 1) or 1)}h"
            metadata["ensemble_id"] = ensemble_id
            candidates = self._enhanced_ensemble_rankings(base, market_mode, max_legs)
            ranked, skipped = self.enhanced_allocator.rank(
                candidates,
                duration_hours=int(base.get("duration_hours", 1) or 1),
                metadata=metadata,
            )
            metadata["enhanced_ensemble_skipped"] = skipped
            if len(ranked) >= 2:
                legs = self.enhanced_allocator.allocate(
                    ranked,
                    base_parameters=base["parameters"],
                    allocation_amount_usd=allocation_amount_usd,
                    profile=profile,
                    metadata=metadata,
                    adaptive_leverage=self._adaptive_leverage,
                    min_leg_usd=min_leg_usd,
                    mode=mode,
                )
                if len(legs) >= 2:
                    metadata["allocation_mode"] = "enhanced_1h_ensemble"
                    metadata["ensemble_version"] = ENSEMBLE_VERSION
                    metadata["ensemble_learning_enabled"] = bool(self.config.get("ENSEMBLE_LEARNING_ENABLED", True))
                    metadata["selected_strategies"] = [str(leg.get("strategy_name")) for leg in legs]
                    metadata["individual_weights"] = {
                        str(leg.get("strategy_name")): self._safe_float(leg.get("ensemble_weight"))
                        for leg in legs
                    }
                    metadata["ml_rank_scores"] = {
                        str(leg.get("strategy_name")): self._safe_float(leg.get("ml_rank_score"))
                        for leg in legs
                    }
                    metadata["confluence_score"] = self._safe_float(confluence.get("score"))
                    return legs
            metadata["allocation_mode"] = "single_best"
            metadata["skip_reason"] = "not_enough_enhanced_ensemble_candidates"
            metadata["ensemble_version"] = ENSEMBLE_VERSION
            metadata["rejected_candidate_count"] = len(skipped) + max(0, len(candidates) - len(ranked))
            metadata["cap_blocked_count"] = int(metadata.get("cap_blocked_count", 0) or 0)
            return [fallback_leg]

        if self._one_hour_ensemble_enabled(base):
            ensemble_id = f"ensemble-1h-{str(metadata.get('symbol') or '').lower()}-{int(base.get('duration_hours', 1) or 1)}h"
            candidates = self._ranked_ensemble_candidates(base, market_mode, max_legs)
            if len(candidates) >= 2:
                legs = self._legs_from_rankings(
                    candidates,
                    base,
                    allocation_amount_usd,
                    profile,
                    metadata,
                    mode,
                    min_leg_usd,
                    allocation_mode="1h_live_aggressive_ensemble",
                    ensemble_id=ensemble_id,
                )
                if len(legs) >= 2:
                    metadata["allocation_mode"] = "1h_live_aggressive_ensemble"
                    metadata["ensemble_id"] = ensemble_id
                    metadata["ensemble_learning_enabled"] = bool(self.config.get("ENSEMBLE_LEARNING_ENABLED", True))
                    return legs
            metadata["skip_reason"] = "not_enough_accepted_1h_ensemble_candidates"

        candidates: list[StrategyRanking] = []
        if bool(self.config.get("DYNAMIC_UNIVERSE_ENABLED", False)):
            candidates = self._ranked_universe_candidates(base, market_mode, max_legs)

        if len(candidates) < 2 and bool(self.config.get("VAULT_STRATEGY_BASKET_ENABLED", True)):
            candidates = self._ranked_strategy_basket_candidates(base, market_mode, max_legs)

        if len(candidates) >= 2:
            legs = self._legs_from_rankings(candidates, base, allocation_amount_usd, profile, metadata, mode, min_leg_usd)
            if len(legs) >= 2:
                metadata["allocation_mode"] = "ranked_strategy_basket"
                return legs

        if bool(self.config.get("DYNAMIC_UNIVERSE_ENABLED", False)):
            scanned_legs = self._scanned_candidate_legs(
                base=base,
                selected_parameters=selected_parameters,
                profile=profile,
                allocation_amount_usd=allocation_amount_usd,
                mode=mode,
                market_mode=market_mode,
                max_legs=max_legs,
                min_leg_usd=min_leg_usd,
            )
            if len(scanned_legs) >= 2:
                metadata["allocation_mode"] = "market_scanner"
                return scanned_legs

        metadata["allocation_mode"] = "single_best"
        return [fallback_leg]

    def _legs_from_rankings(
        self,
        candidates: list[StrategyRanking],
        base: dict[str, Any],
        allocation_amount_usd: float,
        profile: str,
        metadata: dict[str, Any],
        mode: str,
        min_leg_usd: float,
        allocation_mode: str = "ranked_strategy_basket",
        ensemble_id: str | None = None,
    ) -> list[dict[str, Any]]:
        duration_hours = int(base.get("duration_hours", 0) or 0)
        weights = [
            self._ensemble_weight(row, duration_hours)
            if allocation_mode == "1h_live_aggressive_ensemble"
            else max(
                self._ranking_net_roi_v2_score(row),
                self._ranking_net_roi_score(row),
                self._safe_float(row.edge_score),
                self._safe_float(row.score),
                0.01,
            )
            for row in candidates
        ]
        total_weight = sum(weights) or 1.0
        max_symbol_pct = self._clamp(
            self._safe_float(
                self.config.get(
                    "ENSEMBLE_1H_MAX_SYMBOL_PCT" if allocation_mode == "1h_live_aggressive_ensemble" else "VAULT_MAX_SYMBOL_ALLOCATION_PCT"
                ),
                0.50,
            ),
            0.05,
            1.0,
        )
        max_strategy_pct = self._clamp(
            self._safe_float(self.config.get("ENSEMBLE_1H_MAX_STRATEGY_PCT"), 0.60),
            0.05,
            1.0,
        )
        legs: list[dict[str, Any]] = []
        allocated = 0.0
        allocated_by_symbol: dict[str, float] = {}
        allocated_by_strategy: dict[str, float] = {}

        for ranking, weight in zip(candidates, weights):
            raw_allocation = allocation_amount_usd * (weight / total_weight)
            if allocation_mode == "1h_live_aggressive_ensemble":
                symbol_room = allocation_amount_usd * max_symbol_pct - allocated_by_symbol.get(str(ranking.symbol), 0.0)
                strategy_room = allocation_amount_usd * max_strategy_pct - allocated_by_strategy.get(str(ranking.strategy_name), 0.0)
                allocation_cap = min(raw_allocation, allocation_amount_usd * max_symbol_pct, symbol_room, strategy_room)
            else:
                allocation_cap = min(raw_allocation, allocation_amount_usd * max_symbol_pct)
            if allocation_mode == "1h_live_aggressive_ensemble":
                allocation_cap = min(allocation_cap, strategy_room)
            if allocation_cap + 1e-9 < min_leg_usd:
                continue
            params = {**dict(base["parameters"]), **dict(ranking.parameters or {})}
            roi_payload = self._ranking_net_roi_payload(ranking)
            roi_v2_payload = self._ranking_net_roi_v2_payload(ranking)
            params["net_roi_score"] = roi_payload["net_roi_score"]
            params["net_roi_v2_score"] = roi_v2_payload["net_roi_v2_score"]
            one_hour_edge_payload = self._ranking_one_hour_edge_payload(ranking)
            params["one_hour_edge_v2"] = one_hour_edge_payload.get("one_hour_edge_v2")
            params["one_hour_edge_grade"] = one_hour_edge_payload.get("one_hour_edge_grade")
            params["expected_execution_quality"] = one_hour_edge_payload.get("expected_execution_quality")
            params["profitability_blockers"] = one_hour_edge_payload.get("profitability_blockers", [])
            params["raw_vs_net_roi_gap"] = one_hour_edge_payload.get("raw_vs_net_roi_gap")
            params["candidate_quality_breakdown"] = one_hour_edge_payload.get("candidate_quality_breakdown", {})
            params["roi_quality_grade"] = roi_v2_payload["roi_quality_grade"]
            params["roi_rejection_risk"] = roi_v2_payload["roi_rejection_risk"]
            params["regime_support"] = roi_v2_payload["regime_support"]
            params["regime_bucket"] = roi_v2_payload["regime_bucket"]
            params["tail_loss_penalty"] = roi_v2_payload["tail_loss_penalty"]
            params["downside_asymmetry_penalty"] = roi_v2_payload["downside_asymmetry_penalty"]
            params["cost_adjusted_breakout_potential"] = roi_v2_payload["cost_adjusted_breakout_potential"]
            params["expected_fill_quality"] = roi_payload["expected_fill_quality"]
            params["churn_penalty"] = roi_payload["churn_penalty"]
            params["edge_after_cost_bps"] = roi_payload["edge_after_cost_bps"]
            params["leverage"] = self._adaptive_leverage(params, profile, metadata.get("volatility_pct", 0.0), mode)
            params["strategy_name"] = ranking.strategy_name
            leg = {
                "strategy_name": ranking.strategy_name,
                "symbol": ranking.symbol,
                "timeframe": self._normalize_timeframe(ranking.timeframe),
                "parameters": params,
                "allocation_cap_usd": allocation_cap,
                "leverage": params["leverage"],
                "optimizer_ranking_id": ranking.id,
                "optimizer_profile": ranking.profile,
                "edge_score": self._safe_float(ranking.edge_score),
                "net_roi_score": roi_payload["net_roi_score"],
                "net_roi_v2_score": roi_v2_payload["net_roi_v2_score"],
                "one_hour_edge_v2": one_hour_edge_payload.get("one_hour_edge_v2"),
                "one_hour_edge_grade": one_hour_edge_payload.get("one_hour_edge_grade"),
                "expected_execution_quality": one_hour_edge_payload.get("expected_execution_quality"),
                "profitability_blockers": one_hour_edge_payload.get("profitability_blockers", []),
                "raw_vs_net_roi_gap": one_hour_edge_payload.get("raw_vs_net_roi_gap"),
                "candidate_quality_breakdown": one_hour_edge_payload.get("candidate_quality_breakdown", {}),
                "roi_quality_grade": roi_v2_payload["roi_quality_grade"],
                "roi_rejection_risk": roi_v2_payload["roi_rejection_risk"],
                "regime_support": roi_v2_payload["regime_support"],
                "regime_bucket": roi_v2_payload["regime_bucket"],
                "tail_loss_penalty": roi_v2_payload["tail_loss_penalty"],
                "downside_asymmetry_penalty": roi_v2_payload["downside_asymmetry_penalty"],
                "cost_adjusted_breakout_potential": roi_v2_payload["cost_adjusted_breakout_potential"],
                "expected_fill_quality": roi_payload["expected_fill_quality"],
                "churn_penalty": roi_payload["churn_penalty"],
                "edge_after_cost_bps": roi_payload["edge_after_cost_bps"],
                "execution_style": ranking.execution_style or "market",
                "universe_source": ranking.universe_source or "strategy_basket",
                "allocation_mode": allocation_mode,
                "ensemble_id": ensemble_id,
                "ensemble_weight": weight / total_weight if total_weight > 0 else 0.0,
                "target_ensemble_weight": weight / total_weight if total_weight > 0 else 0.0,
                "effective_allocation_weight": 0.0,
                "cap_limited": allocation_cap + 1e-9 < raw_allocation,
                "cap_limit_reason": "allocation_cap" if allocation_cap + 1e-9 < raw_allocation else "",
                "fib_confluence": metadata.get("fib_confluence", {}),
                "market_regime": metadata.get("market_regime"),
                "skip_reason": "",
            }
            legs.append(leg)
            allocated += allocation_cap
            allocated_by_symbol[str(ranking.symbol)] = allocated_by_symbol.get(str(ranking.symbol), 0.0) + allocation_cap
            allocated_by_strategy[str(ranking.strategy_name)] = allocated_by_strategy.get(str(ranking.strategy_name), 0.0) + allocation_cap

        if not legs:
            return []

        if allocation_amount_usd > allocated and legs:
            first = legs[0]
            if allocation_mode == "1h_live_aggressive_ensemble":
                room = allocation_amount_usd * max_symbol_pct - allocated_by_symbol.get(str(first.get("symbol")), 0.0)
                room = min(room, allocation_amount_usd * max_strategy_pct - allocated_by_strategy.get(str(first.get("strategy_name")), 0.0))
            else:
                room = allocation_amount_usd * max_symbol_pct - first["allocation_cap_usd"]
            legs[0]["allocation_cap_usd"] += max(0.0, min(room, allocation_amount_usd - allocated))

        actual_allocated = sum(self._safe_float(leg.get("allocation_cap_usd")) for leg in legs)
        metadata["allocation_conservation"] = {
            "requested_allocation_usd": allocation_amount_usd,
            "allocated_usd": actual_allocated,
            "unallocated_usd": max(0.0, allocation_amount_usd - actual_allocated),
            "within_total_cap": actual_allocated <= allocation_amount_usd + 1e-9,
        }
        for leg in legs:
            effective_weight = self._safe_float(leg.get("allocation_cap_usd")) / max(actual_allocated, 1e-9)
            leg["effective_allocation_weight"] = effective_weight
            leg["ensemble_weight"] = effective_weight

        return legs

    def _scanned_candidate_legs(
        self,
        *,
        base: dict[str, Any],
        selected_parameters: dict[str, Any],
        profile: str,
        allocation_amount_usd: float,
        mode: str,
        market_mode: str,
        max_legs: int,
        min_leg_usd: float,
    ) -> list[dict[str, Any]]:
        if self.market_scanner is None:
            return []
        timeframe = self._normalize_timeframe(str(base.get("timeframe", "5m")))
        symbols = self._candidate_symbols(base, market_mode, timeframe)
        scored = self.market_scanner.score_candidates(
            symbols,
            mode=market_mode,
            timeframe=timeframe,
            duration_seconds=max(1, int(base.get("duration_hours", 1) or 1)) * 3600,
            strategy_name=str(base.get("strategy_name")),
            optimizer_profile=str(base.get("optimizer_profile")),
        )
        candidates = [candidate for candidate in scored if candidate.score > 0]
        candidates.sort(
            key=lambda candidate: (
                self._safe_float(candidate.features.get("net_roi_v2_score")),
                self._safe_float(candidate.features.get("net_roi_score")),
                candidate.score,
            ),
            reverse=True,
        )
        candidates = candidates[:max_legs]
        if len(candidates) < 2:
            return []

        max_symbol_pct = self._clamp(self._safe_float(self.config.get("VAULT_MAX_SYMBOL_ALLOCATION_PCT"), 0.50), 0.05, 1.0)
        weights = [
            max(
                self._safe_float(candidate.features.get("net_roi_v2_score")),
                self._safe_float(candidate.features.get("net_roi_score")),
                candidate.score,
                0.01,
            )
            for candidate in candidates
        ]
        total_weight = sum(weights) or 1.0
        legs: list[dict[str, Any]] = []
        for candidate, weight in zip(candidates, weights):
            allocation_cap = min(allocation_amount_usd * (weight / total_weight), allocation_amount_usd * max_symbol_pct)
            if allocation_cap + 1e-9 < min_leg_usd:
                continue
            params = dict(selected_parameters)
            pair_payload = candidate.features.get("pair_screening") if isinstance(candidate.features.get("pair_screening"), dict) else {}
            if pair_payload:
                params.update(
                    {
                        "pair_group_id": pair_payload.get("pair_group_id"),
                        "pair_mode": pair_payload.get("pair_mode"),
                        "pair_symbol": pair_payload.get("pair_symbol"),
                        "hedge_ratio": pair_payload.get("hedge_ratio"),
                        "spread_zscore": pair_payload.get("spread_zscore"),
                        "spread_half_life": pair_payload.get("spread_half_life"),
                        "pair_score": pair_payload.get("pair_score", pair_payload.get("score")),
                        "correlation": pair_payload.get("correlation"),
                        "pair_signal": pair_payload.get("pair_signal", {}),
                    }
                )
            volatility_pct = float(candidate.features.get("volatility", 0.0) or 0.0) * 100
            params["leverage"] = self._adaptive_leverage(params, profile, volatility_pct, mode)
            params["scanner_source"] = candidate.source
            params["upside_screen_score"] = candidate.features.get("upside_screen_score")
            params["scanner_score_breakdown"] = candidate.features.get("scanner_score_breakdown", {})
            params["cost_drag_bps"] = candidate.features.get("cost_drag_bps")
            params["liquidity_capacity"] = candidate.features.get("liquidity_capacity")
            params["liquidity_capacity_usd"] = candidate.features.get("liquidity_capacity_usd")
            params["volatility_regime"] = candidate.features.get("volatility_regime")
            params["volume_impulse"] = candidate.features.get("volume_impulse")
            params["rejection_reason"] = candidate.rejection_reason
            params["net_roi_score"] = candidate.features.get("net_roi_score")
            params["net_roi_v2_score"] = candidate.features.get("net_roi_v2_score")
            params["one_hour_edge_v2"] = candidate.features.get("one_hour_edge_v2")
            params["one_hour_edge_grade"] = candidate.features.get("one_hour_edge_grade")
            params["expected_execution_quality"] = candidate.features.get("expected_execution_quality")
            params["profitability_blockers"] = candidate.features.get("profitability_blockers", [])
            params["raw_vs_net_roi_gap"] = candidate.features.get("raw_vs_net_roi_gap")
            params["candidate_quality_breakdown"] = candidate.features.get("candidate_quality_breakdown", {})
            params["roi_quality_grade"] = candidate.features.get("roi_quality_grade")
            params["roi_rejection_risk"] = candidate.features.get("roi_rejection_risk")
            params["regime_support"] = candidate.features.get("regime_support")
            params["regime_bucket"] = candidate.features.get("regime_bucket")
            params["tail_loss_penalty"] = candidate.features.get("tail_loss_penalty")
            params["downside_asymmetry_penalty"] = candidate.features.get("downside_asymmetry_penalty")
            params["cost_adjusted_breakout_potential"] = candidate.features.get("cost_adjusted_breakout_potential")
            params["expected_fill_quality"] = candidate.features.get("expected_fill_quality")
            params["churn_penalty"] = candidate.features.get("churn_penalty")
            params["edge_after_cost_bps"] = candidate.features.get("edge_after_cost_bps")
            legs.append(
                {
                    "strategy_name": str(base.get("strategy_name")),
                    "symbol": candidate.symbol,
                    "timeframe": timeframe,
                    "parameters": params,
                    "allocation_cap_usd": allocation_cap,
                    "leverage": params["leverage"],
                    "optimizer_ranking_id": None,
                    "optimizer_profile": base.get("optimizer_profile"),
                    "edge_score": candidate.score,
                    "net_roi_score": candidate.features.get("net_roi_score"),
                    "net_roi_v2_score": candidate.features.get("net_roi_v2_score"),
                    "one_hour_edge_v2": candidate.features.get("one_hour_edge_v2"),
                    "one_hour_edge_grade": candidate.features.get("one_hour_edge_grade"),
                    "expected_execution_quality": candidate.features.get("expected_execution_quality"),
                    "profitability_blockers": candidate.features.get("profitability_blockers", []),
                    "raw_vs_net_roi_gap": candidate.features.get("raw_vs_net_roi_gap"),
                    "candidate_quality_breakdown": candidate.features.get("candidate_quality_breakdown", {}),
                    "roi_quality_grade": candidate.features.get("roi_quality_grade"),
                    "roi_rejection_risk": candidate.features.get("roi_rejection_risk"),
                    "regime_support": candidate.features.get("regime_support"),
                    "regime_bucket": candidate.features.get("regime_bucket"),
                    "tail_loss_penalty": candidate.features.get("tail_loss_penalty"),
                    "downside_asymmetry_penalty": candidate.features.get("downside_asymmetry_penalty"),
                    "cost_adjusted_breakout_potential": candidate.features.get("cost_adjusted_breakout_potential"),
                    "expected_fill_quality": candidate.features.get("expected_fill_quality"),
                    "churn_penalty": candidate.features.get("churn_penalty"),
                    "edge_after_cost_bps": candidate.features.get("edge_after_cost_bps"),
                    "upside_screen_score": candidate.features.get("upside_screen_score"),
                    "scanner_score_breakdown": candidate.features.get("scanner_score_breakdown", {}),
                    "cost_drag_bps": candidate.features.get("cost_drag_bps"),
                    "liquidity_capacity_usd": candidate.features.get("liquidity_capacity_usd"),
                    "volatility_regime": candidate.features.get("volatility_regime"),
                    "rejection_reason": candidate.rejection_reason,
                    "execution_style": "market",
                    "universe_source": candidate.source,
                    "allocation_mode": "market_scanner",
                    "ensemble_id": None,
                    "ensemble_weight": weight / total_weight if total_weight > 0 else 0.0,
                    "fib_confluence": {},
                    "market_regime": None,
                    "pair_group_id": pair_payload.get("pair_group_id"),
                    "pair_mode": pair_payload.get("pair_mode"),
                    "pair_symbol": pair_payload.get("pair_symbol"),
                    "hedge_ratio": pair_payload.get("hedge_ratio"),
                    "spread_zscore": pair_payload.get("spread_zscore"),
                    "spread_half_life": pair_payload.get("spread_half_life"),
                    "pair_score": pair_payload.get("pair_score", pair_payload.get("score")),
                    "correlation": pair_payload.get("correlation"),
                    "pair_signal": pair_payload.get("pair_signal", {}),
                    "skip_reason": "",
                }
            )
        return legs

    def _leg_from_selection(
        self,
        base: dict[str, Any],
        ranking: StrategyRanking | None,
        parameters: dict[str, Any],
        allocation_amount_usd: float,
        profile: str,
        metadata: dict[str, Any],
        mode: str,
    ) -> dict[str, Any]:
        params = dict(parameters)
        roi_payload = self._ranking_net_roi_payload(ranking)
        roi_v2_payload = self._ranking_net_roi_v2_payload(ranking)
        one_hour_edge_payload = self._ranking_one_hour_edge_payload(ranking)
        params["net_roi_score"] = roi_payload["net_roi_score"]
        params["net_roi_v2_score"] = roi_v2_payload["net_roi_v2_score"]
        params["one_hour_edge_v2"] = one_hour_edge_payload.get("one_hour_edge_v2")
        params["one_hour_edge_grade"] = one_hour_edge_payload.get("one_hour_edge_grade")
        params["expected_execution_quality"] = one_hour_edge_payload.get("expected_execution_quality")
        params["profitability_blockers"] = one_hour_edge_payload.get("profitability_blockers", [])
        params["raw_vs_net_roi_gap"] = one_hour_edge_payload.get("raw_vs_net_roi_gap")
        params["candidate_quality_breakdown"] = one_hour_edge_payload.get("candidate_quality_breakdown", {})
        params["roi_quality_grade"] = roi_v2_payload["roi_quality_grade"]
        params["roi_rejection_risk"] = roi_v2_payload["roi_rejection_risk"]
        params["regime_support"] = roi_v2_payload["regime_support"]
        params["regime_bucket"] = roi_v2_payload["regime_bucket"]
        params["tail_loss_penalty"] = roi_v2_payload["tail_loss_penalty"]
        params["downside_asymmetry_penalty"] = roi_v2_payload["downside_asymmetry_penalty"]
        params["cost_adjusted_breakout_potential"] = roi_v2_payload["cost_adjusted_breakout_potential"]
        params["expected_fill_quality"] = roi_payload["expected_fill_quality"]
        params["churn_penalty"] = roi_payload["churn_penalty"]
        params["edge_after_cost_bps"] = roi_payload["edge_after_cost_bps"]
        params["leverage"] = self._adaptive_leverage(params, profile, metadata.get("volatility_pct", 0.0), mode)
        return {
            "strategy_name": str(base["strategy_name"]),
            "symbol": str(metadata.get("symbol") or base.get("symbol") or ""),
            "timeframe": self._normalize_timeframe(str(base["timeframe"])),
            "parameters": params,
            "allocation_cap_usd": allocation_amount_usd,
            "leverage": params["leverage"],
            "optimizer_ranking_id": ranking.id if ranking else None,
            "optimizer_profile": ranking.profile if ranking else base.get("optimizer_profile"),
            "edge_score": self._safe_float(ranking.edge_score) if ranking else 0.0,
            "net_roi_score": roi_payload["net_roi_score"],
            "net_roi_v2_score": roi_v2_payload["net_roi_v2_score"],
            "one_hour_edge_v2": one_hour_edge_payload.get("one_hour_edge_v2"),
            "one_hour_edge_grade": one_hour_edge_payload.get("one_hour_edge_grade"),
            "expected_execution_quality": one_hour_edge_payload.get("expected_execution_quality"),
            "profitability_blockers": one_hour_edge_payload.get("profitability_blockers", []),
            "raw_vs_net_roi_gap": one_hour_edge_payload.get("raw_vs_net_roi_gap"),
            "candidate_quality_breakdown": one_hour_edge_payload.get("candidate_quality_breakdown", {}),
            "roi_quality_grade": roi_v2_payload["roi_quality_grade"],
            "roi_rejection_risk": roi_v2_payload["roi_rejection_risk"],
            "regime_support": roi_v2_payload["regime_support"],
            "regime_bucket": roi_v2_payload["regime_bucket"],
            "tail_loss_penalty": roi_v2_payload["tail_loss_penalty"],
            "downside_asymmetry_penalty": roi_v2_payload["downside_asymmetry_penalty"],
            "cost_adjusted_breakout_potential": roi_v2_payload["cost_adjusted_breakout_potential"],
            "expected_fill_quality": roi_payload["expected_fill_quality"],
            "churn_penalty": roi_payload["churn_penalty"],
            "edge_after_cost_bps": roi_payload["edge_after_cost_bps"],
            "execution_style": ranking.execution_style if ranking and ranking.execution_style else "market",
            "universe_source": ranking.universe_source if ranking and ranking.universe_source else "configured",
            "allocation_mode": metadata.get("allocation_mode", "single_best"),
            "ensemble_id": metadata.get("ensemble_id"),
            "ensemble_version": metadata.get("ensemble_version"),
            "ensemble_weight": 1.0,
            "target_ensemble_weight": 1.0,
            "effective_allocation_weight": 1.0,
            "cap_limited": False,
            "cap_limit_reason": "",
            "ml_rank_score": 0.0,
            "multi_timeframe_confluence": metadata.get("multi_timeframe_confluence", {}),
            "confluence_score": self._safe_float((metadata.get("multi_timeframe_confluence") or {}).get("score")) if isinstance(metadata.get("multi_timeframe_confluence"), dict) else 0.0,
            "profit_objective_version": metadata.get("profit_objective_version", ""),
            "high_upside_profile": bool(params.get("high_upside_profile") or metadata.get("high_upside_profile", False)),
            "market_structure": metadata.get("market_structure", {}),
            "market_structure_score": self._safe_float(metadata.get("market_structure_score")),
            "fib_confluence": metadata.get("fib_confluence", {}),
            "market_regime": metadata.get("market_regime"),
            "pair_group_id": metadata.get("pair_group_id"),
            "pair_mode": metadata.get("pair_mode"),
            "pair_symbol": metadata.get("pair_symbol"),
            "pair_role": metadata.get("pair_role"),
            "hedge_ratio": metadata.get("hedge_ratio"),
            "spread_zscore": metadata.get("spread_zscore"),
            "spread_half_life": metadata.get("spread_half_life"),
            "pair_score": metadata.get("pair_score"),
            "correlation": metadata.get("correlation"),
            "pair_signal": metadata.get("pair_signal", {}),
            "pair_skip_reason": metadata.get("pair_skip_reason", ""),
            "skip_reason": metadata.get("skip_reason", ""),
        }

    def _pair_stat_arb_legs(
        self,
        *,
        base: dict[str, Any],
        selected_parameters: dict[str, Any],
        profile: str,
        metadata: dict[str, Any],
        allocation_amount_usd: float,
        mode: str,
        min_leg_usd: float,
    ) -> list[dict[str, Any]]:
        if not bool(self.config.get("PAIR_TRADING_ENABLED", False)):
            return []
        max_pair_legs = max(0, self._config_int("PAIR_MAX_LEGS_PER_CYCLE", 2))
        if max_pair_legs < 2:
            metadata["pair_skip_reason"] = "pair_max_legs_below_two"
            return []
        candidates = metadata.get("pair_candidates") if isinstance(metadata.get("pair_candidates"), list) else []
        candidate = next(
            (
                item
                for item in candidates
                if isinstance(item, dict) and str(item.get("pair_mode") or "").lower() == "stat_arb"
            ),
            None,
        )
        if not candidate:
            metadata["pair_skip_reason"] = "no_stat_arb_pair_candidate"
            return []
        signal = candidate.get("pair_signal") if isinstance(candidate.get("pair_signal"), dict) else {}
        base_symbol = str(candidate.get("base_symbol") or "").upper()
        pair_symbol = str(candidate.get("pair_symbol") or "").upper()
        if not base_symbol or not pair_symbol:
            metadata["pair_skip_reason"] = "pair_symbols_missing"
            return []
        roles = [
            (base_symbol, str(signal.get("base_side") or "buy").lower(), pair_symbol),
            (pair_symbol, str(signal.get("pair_side") or "sell").lower(), base_symbol),
        ]
        if any(side not in {"buy", "sell"} for _, side, _ in roles):
            metadata["pair_skip_reason"] = "pair_signal_missing_side"
            return []

        max_symbol_pct = self._clamp(self._safe_float(self.config.get("VAULT_MAX_SYMBOL_ALLOCATION_PCT"), 0.50), 0.05, 1.0)
        per_leg_target = allocation_amount_usd / 2
        per_leg_cap = min(per_leg_target, allocation_amount_usd * max_symbol_pct)
        if per_leg_cap + 1e-9 < min_leg_usd:
            metadata["pair_skip_reason"] = "pair_allocation_below_min_leg"
            return []

        group_id = str(candidate.get("pair_group_id") or f"pair-stat-arb-{base_symbol.lower()}-{pair_symbol.lower()}")
        total_allocated = per_leg_cap * 2
        legs: list[dict[str, Any]] = []
        for symbol, side, peer in roles:
            params = dict(selected_parameters)
            params["strategy_name"] = str(base.get("strategy_name"))
            params["ensemble_adapter"] = "pairs_trading"
            params["pair_group_id"] = group_id
            params["pair_mode"] = "stat_arb"
            params["pair_symbol"] = peer
            params["pair_role"] = "long" if side == "buy" else "short"
            params["pair_forced_side"] = side
            params["pair_signal"] = signal
            params["pair_score"] = self._safe_float(candidate.get("pair_score", candidate.get("score")))
            params["correlation"] = self._safe_float(candidate.get("correlation"))
            params["spread_zscore"] = self._safe_float(candidate.get("spread_zscore"))
            params["spread_half_life"] = self._safe_float(candidate.get("spread_half_life"))
            params["hedge_ratio"] = self._safe_float(candidate.get("hedge_ratio"), 1.0)
            params["leverage"] = self._adaptive_leverage(params, profile, self._safe_float(metadata.get("volatility_pct")), mode)
            leg = {
                "strategy_name": str(base.get("strategy_name")),
                "symbol": symbol,
                "timeframe": self._normalize_timeframe(str(base.get("timeframe", "5m"))),
                "parameters": params,
                "allocation_cap_usd": per_leg_cap,
                "leverage": params["leverage"],
                "optimizer_ranking_id": None,
                "optimizer_profile": base.get("optimizer_profile"),
                "edge_score": self._safe_float(candidate.get("pair_score", candidate.get("score"))),
                "execution_style": "market",
                "universe_source": "pair_screening",
                "allocation_mode": "pair_stat_arb",
                "ensemble_id": group_id,
                "ensemble_version": "pair_screening_v1",
                "ensemble_adapter": "pairs_trading",
                "ensemble_weight": 0.5,
                "target_ensemble_weight": 0.5,
                "effective_allocation_weight": per_leg_cap / max(total_allocated, 1e-9),
                "cap_limited": per_leg_cap + 1e-9 < per_leg_target,
                "cap_limit_reason": "symbol_cap" if per_leg_cap + 1e-9 < per_leg_target else "",
                "ml_rank_score": self._safe_float(candidate.get("ml_score")),
                "multi_timeframe_confluence": metadata.get("multi_timeframe_confluence", {}),
                "confluence_score": self._safe_float((metadata.get("multi_timeframe_confluence") or {}).get("score")) if isinstance(metadata.get("multi_timeframe_confluence"), dict) else 0.0,
                "fib_confluence": metadata.get("fib_confluence", {}),
                "market_regime": metadata.get("market_regime"),
                "pair_group_id": group_id,
                "pair_mode": "stat_arb",
                "pair_symbol": peer,
                "pair_role": "long" if side == "buy" else "short",
                "hedge_ratio": self._safe_float(candidate.get("hedge_ratio"), 1.0),
                "spread_zscore": self._safe_float(candidate.get("spread_zscore")),
                "spread_half_life": self._safe_float(candidate.get("spread_half_life")),
                "pair_score": self._safe_float(candidate.get("pair_score", candidate.get("score"))),
                "correlation": self._safe_float(candidate.get("correlation")),
                "pair_signal": signal,
                "pair_skip_reason": "",
                "skip_reason": "",
            }
            legs.append(leg)

        if len(legs) != 2 or total_allocated > allocation_amount_usd + 1e-9:
            metadata["pair_skip_reason"] = "pair_allocation_failed_conservation"
            return []
        metadata["pair_group_id"] = group_id
        metadata["pair_mode"] = "stat_arb"
        metadata["pair_score"] = self._safe_float(candidate.get("pair_score", candidate.get("score")))
        metadata["correlation"] = self._safe_float(candidate.get("correlation"))
        metadata["spread_zscore"] = self._safe_float(candidate.get("spread_zscore"))
        metadata["spread_half_life"] = self._safe_float(candidate.get("spread_half_life"))
        metadata["allocation_conservation"] = {
            "requested_allocation_usd": allocation_amount_usd,
            "allocated_usd": total_allocated,
            "unallocated_usd": max(0.0, allocation_amount_usd - total_allocated),
            "within_total_cap": total_allocated <= allocation_amount_usd + 1e-9,
        }
        return legs

    def _ranked_universe_candidates(
        self,
        base: dict[str, Any],
        market_mode: str,
        max_legs: int,
    ) -> list[StrategyRanking]:
        if self.universe_service is None:
            return []
        timeframe = self._normalize_timeframe(str(base.get("timeframe", "5m")))
        symbols = self._candidate_symbols(base, market_mode, timeframe)
        if not symbols:
            return []

        optimizer_profiles = list(base.get("optimizer_profiles") or [base.get("optimizer_profile")])
        profile = str(base.get("profile", ""))
        duration_hours = int(base.get("duration_hours", 0) or 0)
        for optimizer_profile in optimizer_profiles:
            query = StrategyRanking.query.filter(StrategyRanking.symbol.in_(symbols), StrategyRanking.rejected.is_(False))
            query = self._provider_filtered_query(query, base.get("provider"))
            if optimizer_profile:
                query = query.filter_by(profile=str(optimizer_profile))
            else:
                query = query.filter_by(experimental=False)

            candidates = self._ordered_rankings_query(query, optimizer_profile).limit(max_legs * 6).all()
            accepted = [
                ranking
                for ranking in candidates
                if self._ranking_acceptable(ranking, optimizer_profile, profile)
            ]
            accepted.sort(
                key=lambda row: (
                    self._provider_match_score(row, base.get("provider")),
                    self._duration_match_score(row.lock_duration_hours, duration_hours),
                    self._one_hour_candidate_score(row, duration_hours) if optimizer_profile == "aggressive_1h" else 0.0,
                    self._ranking_net_roi_score(row),
                    self._safe_float(row.edge_score),
                    self._ranking_score(row, duration_hours),
                    row.created_at,
                ),
                reverse=True,
            )
            if accepted:
                return self._diversified_rankings(accepted, max_legs)
        return []

    def _ranked_strategy_basket_candidates(
        self,
        base: dict[str, Any],
        market_mode: str,
        max_legs: int,
    ) -> list[StrategyRanking]:
        timeframe = self._normalize_timeframe(str(base.get("timeframe", "5m")))
        symbols = self._candidate_symbols(base, market_mode, timeframe)
        if not symbols:
            return []

        optimizer_profiles = list(base.get("optimizer_profiles") or [base.get("optimizer_profile")])
        profile = str(base.get("profile", ""))
        preferred_strategy = str(base.get("strategy_name") or "")
        duration_hours = int(base.get("duration_hours", 0) or 0)
        for optimizer_profile in optimizer_profiles:
            query = StrategyRanking.query.filter(StrategyRanking.symbol.in_(symbols), StrategyRanking.rejected.is_(False))
            query = self._provider_filtered_query(query, base.get("provider"))
            if optimizer_profile:
                query = query.filter_by(profile=str(optimizer_profile))
            else:
                query = query.filter_by(experimental=False)

            candidates = self._ordered_rankings_query(query, optimizer_profile).limit(max_legs * 12).all()
            accepted = [
                ranking
                for ranking in candidates
                if self._ranking_acceptable(ranking, optimizer_profile, profile)
            ]
            accepted.sort(
                key=lambda row: (
                    self._provider_match_score(row, base.get("provider")),
                    self._duration_match_score(row.lock_duration_hours, duration_hours),
                    self._one_hour_candidate_score(row, duration_hours) if optimizer_profile == "aggressive_1h" else 0.0,
                    self._ranking_net_roi_score(row),
                    str(row.strategy_name or "") == preferred_strategy,
                    self._safe_float(row.edge_score),
                    self._safe_float(row.recent_1h_return),
                    self._ranking_score(row, duration_hours),
                    row.created_at,
                ),
                reverse=True,
            )
            diversified = self._diversified_rankings(accepted, max_legs)
            if len(diversified) >= 2:
                return diversified
        return []

    def _duration_ensemble_rankings(
        self,
        base: dict[str, Any],
        market_mode: str,
        max_legs: int,
    ) -> list[StrategyRanking]:
        timeframe = self._normalize_timeframe(str(base.get("timeframe", "1m")))
        symbols = self._candidate_symbols(base, market_mode, timeframe)
        if not symbols:
            return []

        optimizer_profiles = list(base.get("optimizer_profiles") or [base.get("optimizer_profile")])
        profile = str(base.get("profile", ""))
        duration_hours = int(base.get("duration_hours", 1) or 1)
        library = EnhancedEnsembleAllocator.strategy_library(duration_hours)
        for optimizer_profile in optimizer_profiles:
            query = StrategyRanking.query.filter(
                StrategyRanking.symbol.in_(symbols),
                StrategyRanking.strategy_name.in_(library),
                StrategyRanking.rejected.is_(False),
            )
            query = self._provider_filtered_query(query, base.get("provider"))
            if optimizer_profile:
                query = query.filter_by(profile=str(optimizer_profile))
            else:
                query = query.filter_by(experimental=False)

            candidates = query.order_by(
                StrategyRanking.net_return_after_costs.desc(),
                StrategyRanking.score.desc(),
                StrategyRanking.created_at.desc(),
            ).limit(max_legs * 24).all()
            accepted = [
                ranking
                for ranking in candidates
                if self._ranking_acceptable(ranking, optimizer_profile, profile)
            ]
            accepted.sort(
                key=lambda row: (
                    self._provider_match_score(row, base.get("provider")),
                    *self._max_return_ranking_key(row, duration_hours),
                )
                if bool(self.config.get("MAX_RETURN_OPTIMIZER_ENABLED", False))
                else (
                    self._provider_match_score(row, base.get("provider")),
                    self._duration_match_score(row.lock_duration_hours, duration_hours),
                    self._ranking_net_roi_score(row),
                    self._safe_float(row.net_return_after_costs),
                    self._ranking_score(row, duration_hours),
                    self._safe_float(row.edge_score),
                    row.created_at,
                ),
                reverse=True,
            )
            if accepted:
                return accepted
        return []

    def _ranked_ensemble_candidates(
        self,
        base: dict[str, Any],
        market_mode: str,
        max_legs: int,
    ) -> list[StrategyRanking]:
        timeframe = self._normalize_timeframe(str(base.get("timeframe", "1m")))
        symbols = self._candidate_symbols(base, market_mode, timeframe)
        if not symbols:
            return []

        optimizer_profiles = list(base.get("optimizer_profiles") or [base.get("optimizer_profile")])
        profile = str(base.get("profile", ""))
        duration_hours = int(base.get("duration_hours", 1) or 1)
        min_edge = self._safe_float(self.config.get("ENSEMBLE_1H_MIN_EDGE_BPS"), self._safe_float(self.config.get("AGGRESSIVE_MIN_EDGE_BPS"), 4.0))

        for optimizer_profile in optimizer_profiles:
            query = StrategyRanking.query.filter(StrategyRanking.symbol.in_(symbols), StrategyRanking.rejected.is_(False))
            query = self._provider_filtered_query(query, base.get("provider"))
            if optimizer_profile:
                query = query.filter_by(profile=str(optimizer_profile))
            else:
                query = query.filter_by(experimental=False)

            candidates = self._ordered_rankings_query(query, optimizer_profile).limit(max_legs * 16).all()
            accepted = [
                ranking
                for ranking in candidates
                if self._ranking_acceptable(ranking, optimizer_profile, profile)
                and self._safe_float(ranking.edge_score) >= min_edge
            ]
            accepted.sort(
                key=lambda row: (
                    self._provider_match_score(row, base.get("provider")),
                    self._duration_match_score(row.lock_duration_hours, duration_hours),
                    self._one_hour_candidate_score(row, duration_hours),
                    self._ranking_net_roi_score(row),
                    self._ensemble_weight(row, duration_hours),
                    self._safe_float(row.recent_1h_return),
                    row.created_at,
                ),
                reverse=True,
            )
            diversified = self._diversified_rankings(accepted, max_legs)
            if len(diversified) >= 2:
                return diversified
        return []

    def _enhanced_ensemble_rankings(
        self,
        base: dict[str, Any],
        market_mode: str,
        max_legs: int,
    ) -> list[StrategyRanking]:
        timeframe = self._normalize_timeframe(str(base.get("timeframe", "1m")))
        symbols = self._candidate_symbols(base, market_mode, timeframe)
        if not symbols:
            return []

        optimizer_profiles = list(base.get("optimizer_profiles") or [base.get("optimizer_profile")])
        profile = str(base.get("profile", ""))
        duration_hours = int(base.get("duration_hours", 1) or 1)
        library = {"scalping", "rsi_mean_reversion", "volatility_breakout", "breakout", "ema_crossover", "rule_based_signal"}
        for optimizer_profile in optimizer_profiles:
            query = StrategyRanking.query.filter(
                StrategyRanking.symbol.in_(symbols),
                StrategyRanking.strategy_name.in_(library),
                StrategyRanking.rejected.is_(False),
            )
            query = self._provider_filtered_query(query, base.get("provider"))
            if optimizer_profile:
                query = query.filter_by(profile=str(optimizer_profile))
            else:
                query = query.filter_by(experimental=False)

            candidates = self._ordered_rankings_query(query, optimizer_profile).limit(max_legs * 20).all()
            accepted = [
                ranking
                for ranking in candidates
                if self._ranking_acceptable(ranking, optimizer_profile, profile)
            ]
            accepted.sort(
                key=lambda row: (
                    self._provider_match_score(row, base.get("provider")),
                    self._duration_match_score(row.lock_duration_hours, duration_hours),
                    self._ranking_net_roi_score(row),
                    self._safe_float(row.sharpe_like),
                    self._safe_float(row.sortino_like),
                    self._safe_float(row.edge_score),
                    self._safe_float(row.recent_1h_return),
                    self._ranking_score(row, duration_hours),
                    row.created_at,
                ),
                reverse=True,
            )
            if accepted:
                return accepted
        return []

    def _ensemble_weight(self, ranking: StrategyRanking, duration_hours: int) -> float:
        net_roi = max(self._ranking_net_roi_score(ranking), 0.0)
        edge = max(self._safe_float(ranking.edge_score), 0.0)
        net_return = max(self._safe_float(ranking.net_return_after_costs), self._safe_float(ranking.total_return), 0.0)
        recent = max(self._safe_float(ranking.recent_1h_return), self._safe_float(ranking.recent_performance_score), 0.0)
        profit_factor = max(self._safe_float(ranking.profit_factor) - 1.0, 0.0)
        win_rate_edge = max(self._safe_float(ranking.win_rate) - 0.5, 0.0)
        expectancy = max(self._safe_float(ranking.expectancy), 0.0)
        favorable = max(self._safe_float(ranking.max_favorable_excursion), 0.0)
        adverse = abs(min(self._safe_float(ranking.max_adverse_excursion), 0.0))
        mfe_mae_ratio = favorable / max(adverse, 1e-9) if favorable > 0 else 0.0
        convex = max(self._safe_float(getattr(ranking, "convex_edge_score", 0.0)), 0.0)
        stability = max(self._safe_float(ranking.window_stability), self._safe_float(ranking.accepted_window_ratio), 0.0)
        capacity_ratio = 1.0
        allocation = self._safe_float(ranking.allocation_amount_usd)
        if allocation > 0 and self._safe_float(ranking.capacity_usd) > 0:
            capacity_ratio = min(self._safe_float(ranking.capacity_usd) / allocation, 5.0)
        ml_bonus = 0.0
        if bool(self.config.get("ML_RANKER_ENABLED", False)) and bool(self.config.get("ENSEMBLE_LEARNING_ENABLED", True)):
            horizon = horizon_from_duration(duration_hours or ranking.lock_duration_hours or 1)
            if self.online_ranker.is_warmed_up(horizon):
                ml_bonus = max(self.online_ranker.predict_score(extract_features(ranking), horizon), 0.0) * 10.0
        cost_penalty = max(self._safe_float(ranking.cost_drag_bps) - 18.0, 0.0) * 0.15

        return max(
            0.01,
            edge
            + net_roi
            + convex
            + net_return * 120.0
            + recent * 180.0
            + profit_factor * 8.0
            + win_rate_edge * 12.0
            + expectancy
            + min(mfe_mae_ratio, 6.0)
            + stability * 6.0
            + capacity_ratio
            + ml_bonus
            - cost_penalty,
        )

    def _max_return_ranking_key(self, ranking: StrategyRanking, duration_hours: int) -> tuple[Any, ...]:
        favorable = max(self._safe_float(ranking.max_favorable_excursion), 0.0)
        adverse = abs(min(self._safe_float(ranking.max_adverse_excursion), 0.0))
        mfe_mae = favorable / max(adverse, 1e-9) if favorable > 0 else 0.0
        liquidity = self._safe_float(ranking.capacity_usd)
        cost_drag = self._safe_float(ranking.cost_drag_bps)
        return (
            self._duration_match_score(ranking.lock_duration_hours, duration_hours),
            self._ranking_net_roi_score(ranking),
            self._safe_float(ranking.net_return_after_costs),
            max(self._safe_float(ranking.recent_performance_score), self._safe_float(ranking.recent_1h_return)),
            self._safe_float(ranking.expectancy),
            min(mfe_mae, 8.0),
            liquidity,
            -cost_drag,
            self._ranking_score(ranking, duration_hours),
            ranking.created_at,
        )

    @staticmethod
    def _diversified_rankings(candidates: list[StrategyRanking], max_legs: int) -> list[StrategyRanking]:
        selected: list[StrategyRanking] = []
        seen_keys: set[tuple[str, str, str]] = set()
        seen_strategies: set[str] = set()
        for require_new_strategy in (True, False):
            for ranking in candidates:
                key = (str(ranking.symbol or ""), str(ranking.strategy_name or ""), str(ranking.timeframe or ""))
                if key in seen_keys:
                    continue
                if require_new_strategy and str(ranking.strategy_name or "") in seen_strategies:
                    continue
                selected.append(ranking)
                seen_keys.add(key)
                seen_strategies.add(str(ranking.strategy_name or ""))
                if len(selected) >= max_legs:
                    return selected
        return selected

    def _adaptive_leverage(
        self,
        parameters: dict[str, Any],
        profile: str,
        volatility_pct: float,
        mode: str,
    ) -> float:
        requested = max(1.0, self._safe_float(parameters.get("leverage"), 1.0))
        if not bool(self.config.get("LEVERAGE_OPTIMIZER_ENABLED", False)):
            requested = 1.0
        hard_cap = min(self._safe_float(self.config.get("MAX_LEVERAGE"), 3.0), 3.0)
        cap = min(self._safe_float(self.config.get("AGGRESSIVE_MAX_TEST_LEVERAGE"), 3.0), hard_cap)
        if mode == "live":
            cap = min(
                cap,
                hard_cap,
                self._safe_float(self.config.get("AGGRESSIVE_MAX_LIVE_LEVERAGE"), 3.0),
            )
        if profile not in {"Aggressive", "1H10"}:
            cap = min(cap, 2.0)
        if volatility_pct >= self._safe_float(self.config.get("VAULT_HIGH_VOLATILITY_PCT"), 2.5):
            cap = min(cap, 1.5)
        return self._clamp(requested, 1.0, max(1.0, cap))

    def _market_regime(self, symbol: str, timeframe: str, mode: str) -> dict[str, str]:
        try:
            candles = self.market_data.get_candles(symbol, timeframe, mode=mode, limit=40)
        except Exception:  # noqa: BLE001
            return {"name": "unknown", "preferred_strategy": "scalping"}

        closes = [self._safe_float(row.get("close")) for row in candles if isinstance(row, dict)]
        closes = [value for value in closes if value > 0]
        if len(closes) < 14:
            return {"name": "unknown", "preferred_strategy": "scalping"}

        recent = closes[-12:]
        previous = closes[-24:-12] if len(closes) >= 24 else closes[:-12]
        recent_range = (max(recent) - min(recent)) / max(recent[-1], 1e-9)
        previous_range = (max(previous) - min(previous)) / max(previous[-1], 1e-9) if previous else recent_range
        momentum = (closes[-1] - closes[-6]) / max(closes[-6], 1e-9)
        recent_returns = [
            (recent[index] - recent[index - 1]) / max(recent[index - 1], 1e-9)
            for index in range(1, len(recent))
        ]
        sign_changes = sum(
            1
            for index in range(1, len(recent_returns))
            if recent_returns[index] * recent_returns[index - 1] < 0
        )
        distance_from_mean = abs(closes[-1] - mean(recent)) / max(mean(recent), 1e-9)

        if recent_range > max(previous_range * 1.4, 0.006):
            return {"name": "range_expansion", "preferred_strategy": "volatility_breakout"}
        if abs(momentum) > 0.006:
            return {"name": "fast_momentum", "preferred_strategy": "scalping"}
        if sign_changes >= 5 and distance_from_mean > 0.003:
            return {"name": "overextended_chop", "preferred_strategy": "rsi_mean_reversion"}

        return {"name": "balanced", "preferred_strategy": str(self._base_profile(1)["strategy_name"])}

    def _reference_price(self, symbol: str, mode: str, realtime: dict[str, Any]) -> float:
        for key in ("mid", "mid_price", "price", "last_price"):
            value = self._safe_float(realtime.get(key))
            if value > 0:
                return value
        try:
            return self._safe_float(self.market_data.get_mid_price(symbol, mode))
        except Exception:  # noqa: BLE001
            return 0.0

    def _fib_confluence(self, symbol: str, timeframe: str, mode: str, price: float) -> dict[str, Any]:
        if price <= 0:
            return {"score": 0.0, "cluster_count": 0, "golden_zone_count": 0, "trend_bias": "flat", "levels": []}
        lookbacks = self.config.get("FIB_CONFLUENCE_LOOKBACKS", [20, 50, 100])
        if isinstance(lookbacks, str):
            lookbacks = lookbacks.split(",")
        parsed_lookbacks: list[int] = []
        for value in lookbacks:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                parsed_lookbacks.append(parsed)
        lookbacks = parsed_lookbacks or [20, 50, 100]
        try:
            candles = self.market_data.get_candles(symbol, timeframe, mode=mode, limit=max(max(lookbacks), 100))
            return self.fibonacci_service.confluence(
                candles,
                price,
                lookbacks=lookbacks,
                tolerance_bps=self._safe_float(self.config.get("FIB_CONFLUENCE_CLUSTER_TOLERANCE_BPS"), 18.0),
            ).as_dict()
        except Exception as exc:  # noqa: BLE001
            return {"score": 0.0, "error": str(exc), "cluster_count": 0, "golden_zone_count": 0, "trend_bias": "flat", "levels": []}

    def _multi_timeframe_confluence(self, symbol: str, timeframe: str, mode: str, price: float) -> dict[str, Any]:
        if not (
            bool(self.config.get("ENSEMBLE_ENHANCED_ENABLED", False))
            or bool(self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_ENABLED", False))
            or bool(self.config.get("MAX_RETURN_OPTIMIZER_ENABLED", False))
        ):
            return {}
        try:
            return self.multi_timeframe_confluence.score(
                symbol=symbol,
                entry_timeframe=timeframe,
                mode=mode,
                price=price,
                side="buy",
            ).as_dict()
        except Exception as exc:  # noqa: BLE001
            return {
                "score": 0.0,
                "passed": False,
                "skip_reason": str(exc),
                "timeframes": [],
                "timeframe_scores": [],
            }

    def _market_structure_snapshot(self, symbol: str, timeframe: str, mode: str) -> dict[str, Any]:
        if self.market_structure is None:
            return {}
        try:
            snapshot = self.market_structure.snapshot(symbol, timeframe, mode=mode)
        except Exception as exc:  # noqa: BLE001
            return {"enabled": bool(self.config.get("MARKET_STRUCTURE_FEATURES_ENABLED", False)), "score": 0.0, "error": str(exc)}
        return snapshot if isinstance(snapshot, dict) else {}

    def _pair_candidates(
        self,
        symbol: str,
        timeframe: str,
        mode: str,
        duration_hours: int,
        allowed_symbols: list[str],
    ) -> list[Any]:
        if self.pair_screening is None or not bool(self.config.get("PAIR_SCREENING_ENABLED", False)):
            return []
        symbols = list(dict.fromkeys([symbol, *self._normalize_symbols(allowed_symbols)]))
        try:
            return self.pair_screening.screen(
                symbols,
                mode=mode,
                timeframe=timeframe,
                duration_hours=max(1, int(duration_hours or 1)),
                pair_mode="both",
            )
        except Exception:  # noqa: BLE001
            return []

    def _pair_screening_summary(self, candidates: list[Any]) -> dict[str, Any]:
        enabled = bool(self.config.get("PAIR_SCREENING_ENABLED", False))
        payloads = [candidate.as_dict() if hasattr(candidate, "as_dict") else dict(candidate) for candidate in candidates]
        top = payloads[0] if payloads else {}
        return {
            "enabled": enabled,
            "trading_enabled": bool(self.config.get("PAIR_TRADING_ENABLED", False)),
            "candidate_count": len(payloads),
            "top_pair": {
                "pair_group_id": top.get("pair_group_id"),
                "pair_mode": top.get("pair_mode"),
                "base_symbol": top.get("base_symbol"),
                "pair_symbol": top.get("pair_symbol"),
                "pair_score": top.get("pair_score", top.get("score")),
                "correlation": top.get("correlation"),
                "spread_zscore": top.get("spread_zscore"),
            }
            if top
            else {},
            "rejection_breakdown": dict(getattr(self.pair_screening, "last_rejections", {}) or {}),
        }

    def _experimental_duration_ensemble_enabled(self, base: dict[str, Any]) -> bool:
        return int(base.get("duration_hours", 0) or 0) >= 1 and bool(self.config.get("EXPERIMENTAL_DURATION_ENSEMBLE_ENABLED", False))

    def _enhanced_ensemble_enabled(self, base: dict[str, Any]) -> bool:
        return int(base.get("duration_hours", 0) or 0) <= 1 and bool(self.config.get("ENSEMBLE_ENHANCED_ENABLED", False))

    def _one_hour_ensemble_enabled(self, base: dict[str, Any]) -> bool:
        return int(base.get("duration_hours", 0) or 0) <= 1 and bool(self.config.get("ENSEMBLE_1H_ENABLED", False))

    def _apply_profile_adjustments(
        self,
        profile: str,
        parameters: dict[str, Any],
        metadata: dict[str, Any],
        volatility_pct: float,
    ) -> str:
        if profile in {"Aggressive", "1H10"}:
            multiplier = self._safe_float(self.config.get("VAULT_AGGRESSIVE_SIZE_MULTIPLIER"), 1.35)
            self._scale_risk(parameters, multiplier, cap=0.09)
            metadata["selection_reasons"].append(f"{profile} profile applied adaptive sizing under risk caps")

        if volatility_pct >= self._safe_float(self.config.get("VAULT_HIGH_VOLATILITY_PCT"), 2.5):
            self._scale_risk(parameters, 0.5, floor=0.01)
            metadata["selection_reasons"].append("high volatility reduced allocation sizing")
            return "Conservative"

        return profile

    def _execution_state(
        self,
        current_mode: str,
        limited_reasons: list[str],
        metadata: dict[str, Any],
    ) -> tuple[str, str, str, str]:
        mode = current_mode
        execution_mode = current_mode
        live_status = "not_required"
        substatus = "initializing"

        if mode == "live":
            if limited_reasons:
                return self._fallback_execution(metadata, "market conditions failed pre-allocation checks")

            metadata["selection_reasons"].append("live execution uses the authenticated user's active trading connection")
            return "live", "live", "not_required", "executing"

        return "live", "live", "failed", "limited"

    def _fallback_execution(self, metadata: dict[str, Any], reason: str) -> tuple[str, str, str, str]:
        fallback = "live"
        metadata["fallback_mode"] = fallback
        metadata["fallback_reason"] = reason
        return fallback, fallback, "failed", "limited"

    def _recent_volatility_pct(self, symbol: str, timeframe: str, mode: str) -> float:
        try:
            candles = self.market_data.get_candles(symbol, timeframe, mode=mode, limit=32)
        except Exception:  # noqa: BLE001
            return 0.0

        closes = [self._safe_float(candle.get("close")) for candle in candles]
        closes = [close for close in closes if close > 0]

        if len(closes) < 3:
            return 0.0

        returns = [
            abs((closes[index] - closes[index - 1]) / closes[index - 1])
            for index in range(1, len(closes))
            if closes[index - 1] > 0
        ]

        return mean(returns) * 100 if returns else 0.0

    def _spread_bps(self, symbol: str, mode: str) -> float:
        bid, ask = self._best_bid_ask(symbol, mode)
        mid = (bid + ask) / 2
        if bid <= 0 or ask <= 0 or ask < bid or mid <= 0:
            return 0.0
        return ((ask - bid) / mid) * 10_000

    def _book_liquidity_usd(self, symbol: str, mode: str) -> float:
        try:
            book = self.market_data.get_order_book(symbol, mode)
        except Exception:  # noqa: BLE001
            return 0.0

        levels = book.get("levels", []) if isinstance(book, dict) else []
        if not isinstance(levels, list):
            return 0.0

        total = 0.0
        depth = max(1, self._config_int("VAULT_BOOK_DEPTH_LEVELS", 5))

        for side in levels[:2]:
            if not isinstance(side, list):
                continue

            for row in side[:depth]:
                price, size = self._level_price_size(row)
                total += price * size

        return total

    def _best_bid_ask(self, symbol: str, mode: str) -> tuple[float, float]:
        try:
            book = self.market_data.get_order_book(symbol, mode)
            levels = book.get("levels", []) if isinstance(book, dict) else []
            bid = self._level_price_size(levels[0][0])[0]
            ask = self._level_price_size(levels[1][0])[0]
            return bid, ask
        except Exception:  # noqa: BLE001
            return 0.0, 0.0

    def _estimated_slippage_bps(self, spread_bps: float, liquidity_usd: float) -> float:
        max_slippage = self._safe_float(self.config.get("VAULT_MAX_SLIPPAGE_BPS"), 20.0)

        if liquidity_usd <= 0:
            return max(spread_bps, max_slippage + 1)

        target_order_usd = self._safe_float(self.config.get("VAULT_ESTIMATED_ORDER_USD"), 10_000.0)
        impact_bps = min(25.0, (target_order_usd / liquidity_usd) * 10)

        return spread_bps + impact_bps

    def _limited_reasons(
        self,
        spread_bps: float,
        liquidity_usd: float,
        estimated_slippage_bps: float,
        signal_stability: float = 1.0,
    ) -> list[str]:
        reasons: list[str] = []

        if spread_bps > self._safe_float(self.config.get("VAULT_MAX_SPREAD_BPS"), 25.0):
            reasons.append("wide spread limited execution")

        if liquidity_usd < self._safe_float(self.config.get("VAULT_MIN_LIQUIDITY_USD"), 1_000.0):
            reasons.append("insufficient visible liquidity limited execution")

        if estimated_slippage_bps > self._safe_float(self.config.get("VAULT_MAX_SLIPPAGE_BPS"), 20.0):
            reasons.append("projected slippage limited execution")

        if signal_stability < self._safe_float(self.config.get("VAULT_SIGNAL_STABILITY_THRESHOLD"), 0.65):
            reasons.append("unstable real-time signal limited execution")

        return reasons

    def _market_snapshot(self, symbol: str, timeframe: str, mode: str) -> dict[str, Any]:
        if self.realtime_market is None:
            return {}
        try:
            snapshot = self.realtime_market.snapshot(symbol, mode, timeframe=timeframe)
        except Exception:  # noqa: BLE001
            return {}
        return snapshot if isinstance(snapshot, dict) else {}

    def _resolve_symbol(self, asset: str, allowed_symbols: list[str] | None = None) -> str:
        allowed = self._normalize_symbols(allowed_symbols)
        return asset if asset in allowed else (allowed[0] if allowed else "BTC")

    def _candidate_symbols(self, base: dict[str, Any], market_mode: str, timeframe: str) -> list[str]:
        allowed = self._normalize_symbols(base.get("allowed_symbols"))
        if self.market_scanner is not None:
            return self.market_scanner.candidate_symbols(allowed, mode=market_mode, timeframe=timeframe)
        discovered = self.universe_service.symbols(market_mode, timeframe) if self.universe_service is not None else []
        return list(dict.fromkeys([*allowed, *discovered]))

    def _normalize_symbols(self, symbols: Any = None) -> list[str]:
        raw = symbols or self.config.get("ALLOWED_SYMBOLS", ["BTC"])
        if isinstance(raw, str):
            raw = raw.split(",")
        return [str(symbol).upper() for symbol in raw if str(symbol).strip()]

    def _default_risk_fraction(self) -> float:
        return self._clamp(
            self._safe_float(self.config.get("RISK_PER_TRADE_PCT"), 0.01),
            0.0,
            self._safe_float(self.config.get("VAULT_MAX_RISK_FRACTION"), 0.03),
        )

    def _normalize_timeframe(self, timeframe: Any) -> str:
        value = str(timeframe or self.config.get("DEFAULT_TIMEFRAME", "15m"))
        return value if value in SUPPORTED_VAULT_TIMEFRAMES else "15m"

    def _fallback_mode(self) -> str:
        return "live"

    @staticmethod
    def _level_price_size(row: Any) -> tuple[float, float]:
        if isinstance(row, dict):
            return (
                VaultStrategySelector._safe_float(row.get("px", row.get("price"))),
                VaultStrategySelector._safe_float(row.get("sz", row.get("size"))),
            )

        if isinstance(row, (list, tuple)) and len(row) >= 2:
            return VaultStrategySelector._safe_float(row[0]), VaultStrategySelector._safe_float(row[1])

        return 0.0, 0.0

    @staticmethod
    def _scale_risk(
        parameters: dict[str, Any],
        multiplier: float,
        *,
        floor: float = 0.0,
        cap: float = 1.0,
    ) -> None:
        current = VaultStrategySelector._safe_float(parameters.get("risk_fraction"), 0.03)
        parameters["risk_fraction"] = VaultStrategySelector._clamp(current * multiplier, floor, cap)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        if upper < lower:
            upper = lower
        return max(lower, min(value, upper))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default

        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _baseline_comparison(self, candidates: list[StrategyRanking], selected_ranking: StrategyRanking | None) -> dict[str, Any]:
        accepted = [candidate for candidate in candidates if candidate is not None]
        best = max(
            accepted,
            key=lambda row: (
                self._safe_float(row.net_return_after_costs),
                self._safe_float(row.score),
            ),
            default=None,
        )
        return {
            "baseline_single_strategy": {
                "strategy_name": best.strategy_name if best else None,
                "symbol": best.symbol if best else None,
                "timeframe": best.timeframe if best else None,
                "net_return_after_costs": self._safe_float(best.net_return_after_costs) if best else 0.0,
                "score": self._safe_float(best.score) if best else 0.0,
            },
            "selected_single_strategy": {
                "strategy_name": selected_ranking.strategy_name if selected_ranking else None,
                "symbol": selected_ranking.symbol if selected_ranking else None,
                "timeframe": selected_ranking.timeframe if selected_ranking else None,
                "net_return_after_costs": self._safe_float(selected_ranking.net_return_after_costs) if selected_ranking else 0.0,
                "score": self._safe_float(selected_ranking.score) if selected_ranking else 0.0,
            },
        }

    def _config_int(self, key: str, default: int = 0) -> int:
        try:
            return int(self.config.get(key, default))
        except (TypeError, ValueError):
            return default
