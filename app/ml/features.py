"""Leakage-safe ML feature payload builder.

The factory centralizes the feature contract used by ML-first strategies,
optimizer policy, Fibonacci interpretation, and backtest scoring. Callers pass
only the candle window that is legal at that decision point; the factory never
fetches extra data or looks beyond that window.
"""

from __future__ import annotations

from typing import Any

from ..features.engine import FeatureEngine
from ..services.provider_assets import provider_feature_context


ML_FEATURE_SCHEMA_VERSION = "ml_feature_v1"


class MLFeatureFactory:
    """Build normalized app-wide ML context payloads without future leakage."""

    def __init__(self, config: dict[str, Any], feature_engine: FeatureEngine | None = None) -> None:
        self.config = config
        self.feature_engine = feature_engine or FeatureEngine()

    def build(
        self,
        *,
        symbol: str,
        timeframe: str,
        candles: list[dict[str, Any]] | None = None,
        deterministic_signal: Any | None = None,
        optimizer_context: dict[str, Any] | None = None,
        backtest_result: dict[str, Any] | None = None,
        market_structure: dict[str, Any] | None = None,
        order_book: dict[str, Any] | None = None,
        multi_timeframe: dict[str, Any] | None = None,
        funding: dict[str, Any] | None = None,
        liquidation_context: dict[str, Any] | None = None,
        provider_context: dict[str, Any] | None = None,
        trade_outcomes: list[dict[str, Any]] | None = None,
        cutoff_timestamp: int | float | None = None,
    ) -> dict[str, Any]:
        safe_candles = self._candles_before(candles or [], cutoff_timestamp)
        feature_snapshot = self._feature_snapshot(symbol, timeframe, safe_candles)
        payload: dict[str, Any] = {
            "ml_feature_schema_version": ML_FEATURE_SCHEMA_VERSION,
            "symbol": str(symbol or "").upper(),
            "timeframe": str(timeframe or ""),
            "candle_count": len(safe_candles),
            "latest_timestamp": self._safe_float((safe_candles[-1] if safe_candles else {}).get("timestamp")),
            **feature_snapshot,
        }
        payload.update(self._signal_payload(deterministic_signal))
        payload.update(self._optimizer_payload(optimizer_context or {}))
        payload.update(self._backtest_payload(backtest_result or {}, trade_outcomes or []))
        payload.update(self._market_payload(market_structure or {}, order_book or {}, payload))
        payload.update(self._multi_timeframe_payload(multi_timeframe or {}))
        payload.update(self._funding_payload(funding or {}, market_structure or {}))
        payload.update(self._liquidation_payload(liquidation_context or {}, optimizer_context or {}))
        payload.update(self._provider_payload(provider_context or optimizer_context or {}))
        return payload

    def _candles_before(
        self,
        candles: list[dict[str, Any]],
        cutoff_timestamp: int | float | None,
    ) -> list[dict[str, Any]]:
        if cutoff_timestamp is None:
            return [dict(row) for row in candles if isinstance(row, dict)]
        cutoff = self._safe_float(cutoff_timestamp)
        return [
            dict(row)
            for row in candles
            if isinstance(row, dict) and self._safe_float(row.get("timestamp")) <= cutoff
        ]

    def _feature_snapshot(self, symbol: str, timeframe: str, candles: list[dict[str, Any]]) -> dict[str, Any]:
        if not candles:
            return {
                "feature_snapshot": {},
                "fibonacci_levels": {},
                "fibonacci_confluence": {},
            }
        try:
            snapshot = self.feature_engine.snapshot(symbol=symbol, timeframe=timeframe, candles=candles).as_dict()
        except Exception as exc:  # noqa: BLE001
            return {
                "feature_snapshot": {},
                "fibonacci_levels": {},
                "fibonacci_confluence": {},
                "feature_factory_error": str(exc),
            }
        return {
            **snapshot,
            "feature_snapshot": snapshot,
            "fibonacci_levels": dict(snapshot.get("fibonacci_levels") or {}),
            "fibonacci_confluence": dict(snapshot.get("fibonacci_confluence") or {}),
        }

    def _signal_payload(self, signal: Any | None) -> dict[str, Any]:
        if signal is None:
            return {}
        metadata = getattr(signal, "metadata", {}) or {}
        return {
            "deterministic_action": str(getattr(signal, "action", "hold") or "hold"),
            "deterministic_rationale": str(getattr(signal, "rationale", "") or ""),
            "deterministic_confidence": self._safe_float(metadata.get("confidence")),
            "deterministic_position_fraction": self._safe_float(getattr(signal, "position_fraction", 0.0)),
            "deterministic_stop_loss": self._safe_float(getattr(signal, "stop_loss", 0.0)),
            "deterministic_take_profit": self._safe_float(getattr(signal, "take_profit", 0.0)),
            "rule_decision": metadata.get("rule_decision", {}),
            "baseline_strategy_metadata": metadata,
        }

    def _optimizer_payload(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            **self._provider_payload(context),
            "strategy_name": context.get("strategy_name", context.get("strategy", "")),
            "profile": context.get("profile", context.get("optimizer_profile", "")),
            "optimizer_profile": context.get("optimizer_profile", context.get("profile", "")),
            "score": self._safe_float(context.get("score")),
            "net_return_after_costs": self._safe_float(context.get("net_return_after_costs")),
            "total_return": self._safe_float(context.get("total_return")),
            "recent_1h_return": self._safe_float(context.get("recent_1h_return")),
            "max_drawdown": self._safe_float(context.get("max_drawdown")),
            "profit_factor": self._safe_float(context.get("profit_factor")),
            "trade_count": self._safe_float(context.get("trade_count")),
            "edge_score": self._safe_float(context.get("edge_score")),
            "cost_drag_bps": self._safe_float(context.get("cost_drag_bps")),
            "liquidity_usd": self._safe_float(context.get("liquidity_usd", context.get("liquidity_capacity_usd"))),
            "spread_bps": self._safe_float(context.get("spread_bps")),
            "rejected": bool(context.get("rejected", False)),
            "rejection_reason": str(context.get("rejection_reason", "") or ""),
        }

    def _provider_payload(self, context: dict[str, Any]) -> dict[str, Any]:
        provider = context.get("provider") or context.get("execution_venue") or context.get("venue")
        payload = provider_feature_context(provider)
        collateral = str(context.get("collateral_asset") or payload["collateral_asset"]).upper()
        payload.update(
            {
                "collateral_asset": collateral,
                "collateral_is_usdc": collateral == "USDC",
                "collateral_is_usdt": collateral == "USDT",
                "fee_bps": self._safe_float(context.get("fee_bps", context.get("estimated_fee_bps"))),
                "maker_fee_bps": self._safe_float(context.get("maker_fee_bps")),
                "taker_fee_bps": self._safe_float(context.get("taker_fee_bps")),
            }
        )
        return payload

    def _backtest_payload(self, result: dict[str, Any], trade_outcomes: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        trades = trade_outcomes or (result.get("trades") if isinstance(result.get("trades"), list) else [])
        returns = [
            self._safe_float(trade.get("return", trade.get("net_return_after_costs")))
            for trade in trades
            if isinstance(trade, dict)
        ]
        if returns:
            payload.update(
                {
                    "backtest_trade_outcome_count": len(returns),
                    "backtest_trade_outcome_mean_return": sum(returns) / len(returns),
                    "backtest_trade_outcome_best_return": max(returns),
                    "backtest_trade_outcome_worst_return": min(returns),
                    "backtest_trade_outcome_positive_rate": sum(1 for value in returns if value > 0) / len(returns),
                }
            )
        if not result:
            return payload
        payload.update(
            {
                "backtest_total_return": self._safe_float(result.get("total_return")),
                "backtest_net_return_after_costs": self._safe_float(result.get("net_return_after_costs")),
                "backtest_profit_factor": self._safe_float(result.get("profit_factor")),
                "backtest_trade_count": self._safe_float(result.get("trade_count")),
                "backtest_max_drawdown": self._safe_float(result.get("max_drawdown")),
                "backtest_edge_score": self._safe_float(result.get("edge_score")),
                "backtest_cost_drag_bps": self._safe_float(result.get("cost_drag_bps")),
                "walk_forward_split": result.get("walk_forward_split", {}),
                "ml_decision_snapshots": result.get("ml_decision_snapshots", []),
            }
        )
        return {
            **payload,
        }

    def _market_payload(
        self,
        market_structure: dict[str, Any],
        order_book: dict[str, Any],
        base_payload: dict[str, Any],
    ) -> dict[str, Any]:
        spread_bps = self._safe_float(market_structure.get("spread_bps", market_structure.get("spread_trend_bps")))
        if spread_bps <= 0:
            spread_bps = self._order_book_spread_bps(order_book, self._safe_float(base_payload.get("close")))
        return {
            "market_structure": market_structure,
            "market_structure_score": self._safe_float(market_structure.get("score")),
            "funding_rate": self._safe_float(market_structure.get("funding_rate")),
            "book_depth_score": self._safe_float(market_structure.get("book_depth_score")),
            "order_book_depth_usd": self._order_book_depth_usd(order_book),
            "spread_bps": spread_bps,
        }

    def _multi_timeframe_payload(self, context: dict[str, Any]) -> dict[str, Any]:
        if not context:
            return {}
        windows = context.get("windows") if isinstance(context.get("windows"), dict) else {}
        trend_scores = [
            self._safe_float(row.get("trend_score", row.get("score")))
            for row in windows.values()
            if isinstance(row, dict)
        ]
        agreement = 0.0
        if trend_scores:
            positive = sum(1 for value in trend_scores if value > 0)
            negative = sum(1 for value in trend_scores if value < 0)
            agreement = max(positive, negative) / len(trend_scores)
        return {
            "multi_timeframe_context": context,
            "multi_timeframe_window_count": len(windows),
            "multi_timeframe_trend_agreement": agreement,
            "multi_timeframe_confluence_score": self._safe_float(context.get("confluence_score", context.get("score"))),
            "multi_timeframe_primary_timeframe": context.get("primary_timeframe", ""),
        }

    def _funding_payload(self, funding: dict[str, Any], market_structure: dict[str, Any]) -> dict[str, Any]:
        rate = self._safe_float(funding.get("funding_rate"), self._safe_float(market_structure.get("funding_rate")))
        interval_hours = max(self._safe_float(funding.get("interval_hours"), 8.0), 1.0)
        return {
            "funding_rate": rate,
            "funding_interval_hours": interval_hours,
            "funding_cost_bps": rate * 10_000,
        }

    def _liquidation_payload(self, liquidation_context: dict[str, Any], optimizer_context: dict[str, Any]) -> dict[str, Any]:
        leverage = self._safe_float(
            liquidation_context.get("leverage"),
            self._safe_float(optimizer_context.get("leverage"), 1.0),
        )
        buffer_pct = self._safe_float(
            liquidation_context.get("liquidation_buffer_pct"),
            self._safe_float(optimizer_context.get("liquidation_buffer_pct")),
        )
        return {
            "liquidation_buffer_pct": buffer_pct,
            "liquidation_price": self._safe_float(liquidation_context.get("liquidation_price")),
            "max_leverage": leverage,
        }

    @staticmethod
    def _order_book_spread_bps(order_book: dict[str, Any], mid: float) -> float:
        levels = order_book.get("levels") if isinstance(order_book, dict) else None
        if not isinstance(levels, list) or len(levels) < 2 or not levels[0] or not levels[1] or mid <= 0:
            return 0.0
        first_bid = levels[0][0]
        first_ask = levels[1][0]
        bid = MLFeatureFactory._safe_float(first_bid.get("px") if isinstance(first_bid, dict) else first_bid[0])
        ask = MLFeatureFactory._safe_float(first_ask.get("px") if isinstance(first_ask, dict) else first_ask[0])
        return ((ask - bid) / mid) * 10_000 if bid > 0 and ask > 0 else 0.0

    @staticmethod
    def _order_book_depth_usd(order_book: dict[str, Any]) -> float:
        levels = order_book.get("levels") if isinstance(order_book, dict) else None
        if not isinstance(levels, list):
            return 0.0
        total = 0.0
        for side in levels[:2]:
            if not isinstance(side, list):
                continue
            for level in side[:10]:
                if isinstance(level, dict):
                    price = MLFeatureFactory._safe_float(level.get("px", level.get("price")))
                    size = MLFeatureFactory._safe_float(level.get("sz", level.get("size")))
                elif isinstance(level, (list, tuple)) and len(level) >= 2:
                    price = MLFeatureFactory._safe_float(level[0])
                    size = MLFeatureFactory._safe_float(level[1])
                else:
                    continue
                total += max(0.0, price) * max(0.0, size)
        return total

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return default
        return candidate if candidate == candidate and abs(candidate) != float("inf") else default
