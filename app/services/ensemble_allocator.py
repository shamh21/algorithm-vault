"""Enhanced 1h ensemble allocation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ml.online_ranker import OnlineRanker, extract_features, horizon_from_duration
from ..models import StrategyRanking


ENSEMBLE_VERSION = "enhanced_v2"
DURATION_ENSEMBLE_VERSION = "duration_experimental_v1"
MAX_RETURN_OBJECTIVE_VERSION = "max_return_v3"
ADAPTER_STRATEGIES = {
    "scalping": "scalping",
    "rsi_mean_reversion": "rsi_mean_reversion",
    "volatility_breakout": "volatility_breakout",
    "momentum_breakout": "breakout",
    "ema_crossover": "ema_crossover",
    "rule_based_filters": "rule_based_signal",
}
UNAVAILABLE_ADAPTERS = {"pairs_trading", "sentiment_driven_signals"}
DURATION_STRATEGY_LIBRARY = {
    "1h": {"scalping", "rsi_mean_reversion", "volatility_breakout", "breakout", "ema_crossover", "rule_based_signal"},
    "24h": {"rsi_mean_reversion", "ema_crossover", "rule_based_signal", "volatility_breakout", "breakout"},
    "48h": {"ema_crossover", "rule_based_signal", "volatility_breakout", "breakout"},
    "72h": {"ema_crossover", "rule_based_signal", "volatility_breakout", "breakout"},
    "7d": {"volatility_breakout", "rule_based_signal", "ema_crossover", "breakout"},
}


@dataclass(frozen=True, slots=True)
class EnhancedCandidate:
    ranking: StrategyRanking
    adapter_name: str
    score: float
    ml_score: float
    skip_reason: str = ""


class EnhancedEnsembleAllocator:
    """Scores and allocates enhanced 1h ensemble legs without bypassing hard risk gates."""

    def __init__(self, config: dict[str, Any], online_ranker: OnlineRanker | None = None) -> None:
        self.config = config
        self.online_ranker = online_ranker or OnlineRanker(config)

    def rank(
        self,
        rankings: list[StrategyRanking],
        *,
        duration_hours: int,
        metadata: dict[str, Any],
    ) -> tuple[list[EnhancedCandidate], list[dict[str, Any]]]:
        min_edge = self._float(self.config.get("ENSEMBLE_MIN_EDGE_BPS"), self._float(self.config.get("ENSEMBLE_1H_MIN_EDGE_BPS"), 4.0))
        min_sharpe = self._float(self.config.get("ENSEMBLE_MIN_SHARPE"), 0.5)
        unavailable_adapters = set(UNAVAILABLE_ADAPTERS)
        if metadata.get("pair_candidates") or metadata.get("pair_group_id"):
            unavailable_adapters.discard("pairs_trading")
        skipped = [
            {"adapter": adapter, "skip_reason": "adapter_unavailable_or_data_missing"}
            for adapter in sorted(unavailable_adapters)
        ]
        library = self.strategy_library(duration_hours)
        duration_ensemble = bool(metadata.get("duration_ensemble_enabled", False))
        primary_metric = str(metadata.get("ensemble_primary_metric") or "score")
        candidates: list[EnhancedCandidate] = []

        for ranking in rankings:
            if str(ranking.strategy_name or "") not in library:
                skipped.append({"strategy_name": ranking.strategy_name, "skip_reason": "strategy_not_in_duration_library"})
                continue
            adapter_name = self._adapter_name(str(ranking.strategy_name or ""))
            if not adapter_name:
                skipped.append({"strategy_name": ranking.strategy_name, "skip_reason": "strategy_not_in_enhanced_library"})
                continue
            if self._float(ranking.edge_score) < min_edge:
                skipped.append({"strategy_name": ranking.strategy_name, "skip_reason": "edge_below_threshold"})
                continue
            if self._float(ranking.sharpe_like) < min_sharpe and self._float(ranking.sortino_like) < min_sharpe:
                skipped.append({"strategy_name": ranking.strategy_name, "skip_reason": "risk_adjusted_return_below_threshold"})
                continue
            if str(getattr(ranking, "no_trade_reason", "") or "").strip():
                skipped.append({"strategy_name": ranking.strategy_name, "skip_reason": "ranking_has_no_trade_reason"})
                continue

            ml_score = self._bandit_score(ranking, duration_hours, metadata)
            score = self._score(ranking, ml_score, metadata)
            candidates.append(EnhancedCandidate(ranking=ranking, adapter_name=adapter_name, score=score, ml_score=ml_score))

        if duration_ensemble and primary_metric == "net_return_after_costs":
            candidates.sort(
                key=lambda item: (
                    self._float(item.ranking.net_return_after_costs),
                    self._float(item.ranking.recent_performance_score),
                    self._float(item.ranking.expectancy),
                    self._float(item.ranking.max_favorable_excursion) / max(abs(min(self._float(item.ranking.max_adverse_excursion), 0.0)), 1e-9),
                    item.score,
                    self._float(item.ranking.edge_score),
                    item.ranking.created_at,
                ),
                reverse=True,
            )
        else:
            candidates.sort(key=lambda item: (item.score, item.ranking.created_at), reverse=True)
        return candidates, skipped

    def allocate(
        self,
        candidates: list[EnhancedCandidate],
        *,
        base_parameters: dict[str, Any],
        allocation_amount_usd: float,
        profile: str,
        metadata: dict[str, Any],
        adaptive_leverage: Any,
        min_leg_usd: float,
        mode: str,
    ) -> list[dict[str, Any]]:
        max_legs_key = "EXPERIMENTAL_DURATION_ENSEMBLE_MAX_LEGS" if metadata.get("duration_ensemble_enabled") else "ENSEMBLE_MAX_LEGS"
        max_legs = max(2, min(self._int(self.config.get(max_legs_key), 5), 5))
        max_symbol_pct = self._clamp(self._float(self.config.get("ENSEMBLE_MAX_SYMBOL_PCT"), 0.50), 0.05, 1.0)
        max_strategy_pct = self._clamp(self._float(self.config.get("ENSEMBLE_MAX_STRATEGY_PCT"), 0.60), 0.05, 1.0)
        selected = self._diversify(candidates, max_legs)
        if len(selected) < 2:
            metadata["cap_blocked_count"] = 0
            metadata["rejected_candidate_count"] = max(0, len(candidates) - len(selected))
            return []

        total_score = sum(max(item.score, 0.01) for item in selected) or 1.0
        allocated_by_symbol: dict[str, float] = {}
        allocated_by_strategy: dict[str, float] = {}
        legs: list[dict[str, Any]] = []
        total_allocated = 0.0
        cap_blocked_count = 0
        version = DURATION_ENSEMBLE_VERSION if metadata.get("duration_ensemble_enabled") else ENSEMBLE_VERSION
        allocation_mode = "duration_experimental_ensemble" if metadata.get("duration_ensemble_enabled") else "enhanced_1h_ensemble"
        profit_objective_version = str(metadata.get("profit_objective_version") or "")
        market_structure = metadata.get("market_structure") if isinstance(metadata.get("market_structure"), dict) else {}
        market_structure_score = self._float(metadata.get("market_structure_score"), self._float(market_structure.get("score")))

        for item in selected:
            ranking = item.ranking
            weight = max(item.score, 0.01) / total_score
            symbol = str(ranking.symbol or "")
            strategy = str(ranking.strategy_name or "")
            symbol_room = allocation_amount_usd * max_symbol_pct - allocated_by_symbol.get(symbol, 0.0)
            strategy_room = allocation_amount_usd * max_strategy_pct - allocated_by_strategy.get(strategy, 0.0)
            raw_allocation = allocation_amount_usd * weight
            remaining_room = allocation_amount_usd - total_allocated
            allocation_cap = min(raw_allocation, symbol_room, strategy_room, remaining_room)
            cap_limit_reason = ""
            if allocation_cap + 1e-9 < raw_allocation:
                if remaining_room <= min(symbol_room, strategy_room):
                    cap_limit_reason = "total_allocation_cap"
                elif symbol_room <= strategy_room:
                    cap_limit_reason = "symbol_cap"
                else:
                    cap_limit_reason = "strategy_cap"
            if allocation_cap + 1e-9 < min_leg_usd:
                cap_blocked_count += 1
                continue

            params = {**dict(base_parameters), **dict(ranking.parameters or {})}
            params["strategy_name"] = strategy
            params["ensemble_adapter"] = item.adapter_name
            params["ensemble_version"] = version
            params["leverage"] = adaptive_leverage(params, profile, self._float(metadata.get("volatility_pct")), mode)
            leg = {
                "strategy_name": strategy,
                "symbol": symbol,
                "timeframe": str(ranking.timeframe or "1m"),
                "parameters": params,
                "allocation_cap_usd": allocation_cap,
                "leverage": params["leverage"],
                "optimizer_ranking_id": ranking.id,
                "optimizer_profile": ranking.profile,
                "edge_score": self._float(ranking.edge_score),
                "execution_style": ranking.execution_style or "market",
                "universe_source": ranking.universe_source or "enhanced_ensemble",
                "allocation_mode": allocation_mode,
                "ensemble_id": metadata.get("ensemble_id"),
                "ensemble_version": version,
                "ensemble_adapter": item.adapter_name,
                "profit_objective_version": profit_objective_version,
                "market_structure": market_structure,
                "market_structure_score": market_structure_score,
                "ensemble_weight": weight,
                "target_ensemble_weight": weight,
                "effective_allocation_weight": 0.0,
                "cap_limited": bool(cap_limit_reason),
                "cap_limit_reason": cap_limit_reason,
                "ml_rank_score": item.ml_score,
                "multi_timeframe_confluence": metadata.get("multi_timeframe_confluence", {}),
                "confluence_score": self._float((metadata.get("multi_timeframe_confluence") or {}).get("score")),
                "fib_confluence": metadata.get("fib_confluence", {}),
                "market_regime": metadata.get("market_regime"),
                "skip_reason": "",
            }
            legs.append(leg)
            allocated_by_symbol[symbol] = allocated_by_symbol.get(symbol, 0.0) + allocation_cap
            allocated_by_strategy[strategy] = allocated_by_strategy.get(strategy, 0.0) + allocation_cap
            total_allocated += allocation_cap

        metadata["cap_blocked_count"] = cap_blocked_count
        metadata["rejected_candidate_count"] = max(0, len(candidates) - len(selected)) + cap_blocked_count
        metadata["allocation_conservation"] = {
            "requested_allocation_usd": allocation_amount_usd,
            "allocated_usd": total_allocated,
            "unallocated_usd": max(0.0, allocation_amount_usd - total_allocated),
            "within_total_cap": total_allocated <= allocation_amount_usd + 1e-9,
        }
        if len(legs) < 2 or total_allocated > allocation_amount_usd + 1e-9:
            return []
        for leg in legs:
            effective_weight = self._float(leg.get("allocation_cap_usd")) / max(total_allocated, 1e-9)
            leg["effective_allocation_weight"] = effective_weight
            leg["ensemble_weight"] = effective_weight
        return legs

    def _score(self, ranking: StrategyRanking, ml_score: float, metadata: dict[str, Any]) -> float:
        edge = max(self._float(ranking.edge_score), 0.0)
        net_return = max(self._float(ranking.net_return_after_costs), self._float(ranking.total_return), 0.0)
        recent = max(self._float(ranking.recent_1h_return), self._float(ranking.recent_performance_score), 0.0)
        sharpe = max(self._float(ranking.sharpe_like), 0.0)
        sortino = max(self._float(ranking.sortino_like), 0.0)
        expectancy = max(self._float(ranking.expectancy), 0.0)
        win_rate_edge = max(self._float(ranking.win_rate) - 0.5, 0.0)
        favorable = max(self._float(ranking.max_favorable_excursion), 0.0)
        adverse = abs(min(self._float(ranking.max_adverse_excursion), 0.0))
        mfe_mae = favorable / max(adverse, 1e-9) if favorable > 0 else 0.0
        drawdown_penalty = abs(min(self._float(ranking.max_drawdown), 0.0)) * 45.0
        cost_penalty = max(self._float(ranking.cost_drag_bps) - 18.0, 0.0) * 0.20
        productive_frequency = min(max(self._float(ranking.trades_per_day), self._float(ranking.trade_count)) / 24.0, 1.0) * 7.0
        confluence_bonus = self._float((metadata.get("multi_timeframe_confluence") or {}).get("score")) * 12.0
        market_structure = metadata.get("market_structure") if isinstance(metadata.get("market_structure"), dict) else {}
        market_structure_score = self._float(metadata.get("market_structure_score"), self._float(market_structure.get("score")))
        market_structure_bonus = market_structure_score * 10.0
        spread_penalty = max(self._float(market_structure.get("spread_trend_bps")) - 12.0, 0.0) * 0.05
        volume_bonus = min(max(self._float(market_structure.get("volume_impulse")), 0.0), 5.0) * 0.4
        capacity_bonus = min(max(self._float(ranking.capacity_usd), 0.0) / max(self._float(ranking.allocation_amount_usd), 1.0), 5.0)

        return max(
            0.01,
            edge
            + net_return * 140.0
            + recent * 180.0
            + sharpe * 4.0
            + sortino * 3.0
            + expectancy
            + win_rate_edge * 12.0
            + min(mfe_mae, 8.0)
            + productive_frequency
            + confluence_bonus
            + market_structure_bonus
            + volume_bonus
            + capacity_bonus
            + max(ml_score, 0.0) * 12.0
            - drawdown_penalty
            - cost_penalty
            - spread_penalty,
        )

    def _bandit_score(self, ranking: StrategyRanking, duration_hours: int, metadata: dict[str, Any]) -> float:
        if not bool(self.config.get("ENSEMBLE_LEARNING_ENABLED", True)):
            return 0.0
        horizon = horizon_from_duration(duration_hours or ranking.lock_duration_hours or 1)
        features = extract_features(
            {
                **metadata,
                "strategy_name": ranking.strategy_name,
                "symbol": ranking.symbol,
                "timeframe": ranking.timeframe,
                "optimizer_profile": ranking.profile,
                "lock_duration_hours": duration_hours or ranking.lock_duration_hours,
                "net_return_after_costs": ranking.net_return_after_costs,
                "recent_1h_return": ranking.recent_1h_return,
                "sharpe_like": ranking.sharpe_like,
                "sortino_like": ranking.sortino_like,
                "edge_score": ranking.edge_score,
                "cost_drag_bps": ranking.cost_drag_bps,
                "max_drawdown": ranking.max_drawdown,
                "max_favorable_excursion": ranking.max_favorable_excursion,
                "max_adverse_excursion": ranking.max_adverse_excursion,
            }
        )
        return self.online_ranker.contextual_bandit_score(features, horizon)

    @staticmethod
    def _adapter_name(strategy_name: str) -> str:
        for adapter, actual in ADAPTER_STRATEGIES.items():
            if strategy_name == actual:
                return adapter
        return ""

    @staticmethod
    def duration_bucket(duration_hours: int | float | str | None) -> str:
        duration = max(1, int(EnhancedEnsembleAllocator._float(duration_hours, 1.0)))
        if duration <= 1:
            return "1h"
        if duration <= 24:
            return "24h"
        if duration <= 48:
            return "48h"
        if duration <= 72:
            return "72h"
        return "7d"

    @staticmethod
    def strategy_library(duration_hours: int | float | str | None) -> set[str]:
        return set(DURATION_STRATEGY_LIBRARY.get(EnhancedEnsembleAllocator.duration_bucket(duration_hours), DURATION_STRATEGY_LIBRARY["7d"]))

    @staticmethod
    def _diversify(candidates: list[EnhancedCandidate], max_legs: int) -> list[EnhancedCandidate]:
        selected: list[EnhancedCandidate] = []
        seen_strategies: set[str] = set()
        seen_symbols: set[str] = set()
        for require_new_strategy, require_new_symbol in ((True, True), (True, False), (False, False)):
            for candidate in candidates:
                strategy = str(candidate.ranking.strategy_name or "")
                symbol = str(candidate.ranking.symbol or "")
                if candidate in selected:
                    continue
                if require_new_strategy and strategy in seen_strategies:
                    continue
                if require_new_symbol and symbol in seen_symbols:
                    continue
                selected.append(candidate)
                seen_strategies.add(strategy)
                seen_symbols.add(symbol)
                if len(selected) >= max_legs:
                    return selected
        return selected

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(value, upper))

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
