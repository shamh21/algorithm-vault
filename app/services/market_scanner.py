"""Cached market scanner for hot-token and candidate scoring."""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import mean
from typing import Any
from flask import has_app_context

from ..extensions import db
from ..features.engine import FeatureEngine
from ..ml.online_ranker import extract_features, horizon_from_duration
from ..models import AuditLog, Setting
from .market_data import MarketDataService
from .net_roi import net_roi_diagnostics, net_roi_v2_diagnostics
from .tradability import (
    best_bid_ask,
    book_liquidity_usd,
    cost_drag_bps,
    level_price_size,
    market_structure_score,
    safe_float,
    spread_bps,
    volatility_pct,
    volatility_regime,
)


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    symbol: str
    score: float
    technical_score: float
    ml_score: float
    hot_score: float
    source: str
    features: dict[str, Any]
    score_breakdown: dict[str, float] | None = None
    rejection_reason: str = ""
    stale_data: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": self.score,
            "technical_score": self.technical_score,
            "ml_score": self.ml_score,
            "hot_score": self.hot_score,
            "source": self.source,
            "features": dict(self.features),
            "score_breakdown": dict(self.score_breakdown or {}),
            "rejection_reason": self.rejection_reason,
            "stale_data": self.stale_data,
        }


class MarketScannerService:
    """Scores candidate symbols using hot-token, TA, and online ranker signals."""

    def __init__(
        self,
        config: dict[str, Any],
        market_data: MarketDataService,
        universe_service: Any,
        feature_engine: FeatureEngine,
        online_ranker: Any | None = None,
        offline_ranker: Any | None = None,
        pair_screening: Any | None = None,
    ) -> None:
        self.config = config
        self.market_data = market_data
        self.universe_service = universe_service
        self.feature_engine = feature_engine
        self.online_ranker = online_ranker
        self.offline_ranker = offline_ranker
        self.pair_screening = pair_screening
        self._hot_cache: dict[tuple[str, str], tuple[float, list[dict[str, Any]]]] = {}
        self._score_cache: dict[tuple[str, str, str, int, str, str], tuple[float, list[ScoredCandidate]]] = {}
        self._diagnostic_cache: dict[tuple[str, str, str, int, str, str], dict[str, Any]] = {}
        self.last_scan_diagnostics: dict[str, Any] = {
            "accepted": [],
            "rejected": [],
            "rejection_breakdown": {},
            "cache_hit": False,
            "rejection_rate": 0.0,
        }

    def candidate_symbols(
        self,
        allowed_symbols: list[str] | tuple[str, ...] | None,
        *,
        mode: str,
        timeframe: str,
    ) -> list[str]:
        configured = [str(symbol).upper() for symbol in (allowed_symbols or self.config.get("ALLOWED_SYMBOLS", [])) if str(symbol).strip()]
        hot = [item["symbol"] for item in self.hot_tokens(mode=mode, timeframe=timeframe)]
        pair_symbols: list[str] = []
        if self.pair_screening is not None and bool(self.config.get("PAIR_SCREENING_ENABLED", False)):
            try:
                for candidate in self.pair_screening.screen(configured, mode=mode, timeframe=timeframe, duration_hours=1, pair_mode="both"):
                    payload = candidate.as_dict()
                    pair_symbols.extend(
                        [
                            str(payload.get("base_symbol") or "").upper(),
                            str(payload.get("pair_symbol") or "").upper(),
                            str(payload.get("leader_symbol") or "").upper(),
                        ]
                    )
            except Exception:  # noqa: BLE001
                pair_symbols = []
        return list(dict.fromkeys(symbol for symbol in [*configured, *hot, *pair_symbols] if symbol))

    def hot_tokens(self, *, mode: str, timeframe: str) -> list[dict[str, Any]]:
        if not bool(self.config.get("HOT_TOKEN_SCAN_ENABLED", True)):
            return []
        cache_key = (str(mode or "testnet"), str(timeframe or "5m"))
        cached_at, cached = self._hot_cache.get(cache_key, (0.0, []))
        ttl = max(0, int(self.config.get("HOT_TOKEN_REFRESH_SECONDS", 180) or 180))
        if cached and ttl and time.time() - cached_at < ttl:
            return list(cached)

        symbols = [item.symbol for item in self.universe_service.liquid_universe(cache_key[0], cache_key[1])]
        scored: list[dict[str, Any]] = []
        for symbol in symbols:
            try:
                candles = self.market_data.get_candles(symbol, cache_key[1], mode=cache_key[0], limit=60)
            except Exception:  # noqa: BLE001
                continue
            score, payload = self._hot_score(symbol, candles)
            if score <= 0:
                continue
            scored.append({"symbol": symbol, "score": score, **payload})

        scored.sort(key=lambda item: float(item["score"]), reverse=True)
        limit = max(1, int(self.config.get("HOT_TOKEN_MAX_CANDIDATES", 8) or 8))
        result = scored[:limit]
        self._hot_cache[cache_key] = (time.time(), result)
        return list(result)

    def score_candidates(
        self,
        symbols: list[str] | tuple[str, ...],
        *,
        mode: str,
        timeframe: str,
        duration_seconds: int,
        strategy_name: str,
        optimizer_profile: str,
    ) -> list[ScoredCandidate]:
        normalized = tuple(dict.fromkeys(str(symbol).upper() for symbol in symbols if str(symbol).strip()))
        if not normalized:
            return []
        started_at = time.perf_counter()
        cache_key = (
            ",".join(normalized),
            str(mode or "testnet"),
            str(timeframe or "5m"),
            int(duration_seconds or 0),
            str(strategy_name or ""),
            str(optimizer_profile or ""),
        )
        cached_at, cached = self._score_cache.get(cache_key, (0.0, []))
        ttl = max(0, int(self.config.get("HOT_TOKEN_REFRESH_SECONDS", 180) or 180))
        if cache_key in self._score_cache and ttl and time.time() - cached_at < ttl:
            self.last_scan_diagnostics = {
                **dict(self._diagnostic_cache.get(cache_key, {})),
                "cache_hit": True,
                "scan_runtime_seconds": max(time.perf_counter() - started_at, 0.0),
                "market_data_cache": self._market_data_cache_stats(),
            }
            return list(cached)

        hot_lookup = {item["symbol"]: item for item in self.hot_tokens(mode=mode, timeframe=timeframe)}
        universe_lookup = {
            item.symbol: item.as_dict()
            for item in self.universe_service.liquid_universe(mode, timeframe)
            if hasattr(item, "as_dict")
        }
        pair_lookup = self._pair_lookup(
            list(normalized),
            mode=mode,
            timeframe=timeframe,
            duration_hours=max(duration_seconds / 3600, 1),
        )
        horizon = horizon_from_duration(max(int(duration_seconds or 3600), 1) / 3600)
        scored: list[ScoredCandidate] = []
        rejected: list[dict[str, Any]] = []
        for symbol in normalized:
            try:
                candles = self.market_data.get_candles(symbol, timeframe, mode=mode, limit=120)
                snapshot = self.feature_engine.snapshot(symbol=symbol, timeframe=timeframe, candles=candles)
                try:
                    book = self.market_data.get_order_book(symbol, mode)
                except Exception:  # noqa: BLE001
                    book = {}
            except Exception:  # noqa: BLE001
                rejected.append(self._diagnostic_row(symbol, "market_data_unavailable", source="market_data"))
                continue
            features = snapshot.as_dict()
            tradability = self._tradability_payload(symbol, candles, book, universe_lookup.get(symbol, {}))
            high_upside_metrics = self._high_upside_metrics(candles, tradability)
            features["tradability"] = tradability
            features["market_structure_score"] = tradability["market_structure_score"]
            features["cost_drag_bps"] = tradability["cost_drag_bps"]
            features["spread_bps"] = tradability["spread_bps"]
            features["liquidity_usd"] = tradability["liquidity_usd"]
            features["volatility_regime"] = tradability["volatility_regime"]
            features["scanner_source"] = tradability.get("source", "scanner")
            features["momentum_acceleration"] = high_upside_metrics["momentum_acceleration"]
            features["volume_impulse"] = high_upside_metrics["volume_impulse"]
            features["volume_impulse_persistence"] = high_upside_metrics["volume_impulse_persistence"]
            features["volatility_compression"] = high_upside_metrics["volatility_compression"]
            features["volatility_expansion"] = high_upside_metrics["volatility_expansion"]
            features["breakout_proximity_bps"] = high_upside_metrics["breakout_proximity_bps"]
            features["spread_stability"] = high_upside_metrics["spread_stability"]
            features["depth_stability"] = high_upside_metrics["depth_stability"]
            features["sustained_volume_impulse"] = high_upside_metrics["sustained_volume_impulse"]
            features["pullback_quality"] = high_upside_metrics["pullback_quality"]
            features["breakout_retest_success"] = high_upside_metrics["breakout_retest_success"]
            features["volatility_expansion_after_compression"] = high_upside_metrics["volatility_expansion_after_compression"]
            features["spread_stability_recent"] = high_upside_metrics["spread_stability_recent"]
            features["cost_adjusted_expected_move_persistence"] = high_upside_metrics["cost_adjusted_expected_move_persistence"]
            features["stale_data_age_seconds"] = high_upside_metrics["stale_data_age_seconds"]
            features["cost_adjusted_expected_move"] = high_upside_metrics["cost_adjusted_expected_move"]
            features["market_structure_trend"] = high_upside_metrics["market_structure_trend"]
            features["liquidity_capacity"] = high_upside_metrics["liquidity_capacity"]
            features["liquidity_capacity_usd"] = high_upside_metrics["liquidity_capacity"]
            features["volatility_regime_score"] = high_upside_metrics["volatility_regime_score"]
            features["stale_data"] = high_upside_metrics["stale_data"]
            pair_payload = pair_lookup.get(symbol, {})
            if pair_payload:
                features["pair_screening"] = pair_payload
            hot_score = float(hot_lookup.get(symbol, {}).get("score", 0.0) or 0.0)
            technical_score = self._technical_score(features)
            ranker_features = extract_features(
                {
                    **features,
                    **pair_payload,
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "optimizer_profile": optimizer_profile,
                    "lock_duration_hours": max(duration_seconds / 3600, 1),
                    "horizon": horizon,
                    "edge_score": technical_score,
                    "upside_screen_score": high_upside_metrics["raw_upside_score"],
                    "scanner_source": features["scanner_source"],
                    "volatility_regime": features["volatility_regime"],
                    "volume_spike": features.get("volume_spike", {}),
                }
            )
            ml_score = 0.0
            if self.online_ranker is not None and bool(self.config.get("ML_RANKER_ENABLED", False)):
                ml_score = float(self.online_ranker.predict_score(ranker_features, horizon) or 0.0)
            offline_payload = self._offline_score_payload(
                {
                    **features,
                    **pair_payload,
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "optimizer_profile": optimizer_profile,
                    "lock_duration_hours": max(duration_seconds / 3600, 1),
                    "horizon": horizon,
                    "scanner_source": features["scanner_source"],
                },
                horizon,
            )
            features["offline_ml_prediction"] = offline_payload.get("prediction", 0.0)
            features["offline_ml_status"] = offline_payload.get("status", "no_promoted_model")
            features["offline_ml_blend_enabled"] = offline_payload.get("blend_enabled", False)
            features["offline_ml_explanation"] = offline_payload
            pair_bonus = float(pair_payload.get("pair_score", pair_payload.get("score", 0.0)) or 0.0) * float(
                self.config.get("PAIR_SCANNER_WEIGHT", 0.20) or 0.20
            )
            tradability_bonus = tradability["market_structure_score"] * 1.5 - max(tradability["cost_drag_bps"] - 18.0, 0.0) / 20
            score_breakdown = {
                "technical": technical_score,
                "hot": hot_score,
                "pair": pair_bonus,
                "tradability": tradability_bonus,
                "momentum_acceleration": high_upside_metrics["momentum_acceleration"] * 35.0,
                "volume_impulse": min(high_upside_metrics["volume_impulse"], 5.0) * 0.8,
                "volatility_regime": high_upside_metrics["volatility_regime_score"] * 1.2,
                "liquidity_capacity": min(high_upside_metrics["liquidity_capacity"] / 100_000.0, 5.0) * 0.35,
                "market_structure": tradability["market_structure_score"] * 1.5,
                "ml": float(self.config.get("ML_SCORE_WEIGHT", 0.15) or 0.15) * ml_score,
                "offline_ml": (
                    float(self.config.get("ML_OFFLINE_SCORE_WEIGHT", 0.15) or 0.15)
                    * float(offline_payload.get("prediction", 0.0) or 0.0)
                    if bool(offline_payload.get("blend_enabled", False)) and offline_payload.get("status") == "promoted"
                    else 0.0
                ),
                "volume_persistence": min(high_upside_metrics["volume_impulse_persistence"], 5.0) * 0.45,
                "sustained_volume": min(high_upside_metrics["sustained_volume_impulse"], 5.0) * 0.35,
                "volatility_transition": (
                    high_upside_metrics["volatility_compression"] * 0.35
                    + min(high_upside_metrics["volatility_expansion"], 5.0) * 0.18
                ),
                "volatility_expansion_after_compression": min(high_upside_metrics["volatility_expansion_after_compression"], 5.0) * 0.18,
                "breakout_proximity": max(0.0, 1.0 - high_upside_metrics["breakout_proximity_bps"] / 150.0) * 1.1,
                "breakout_retest_success": high_upside_metrics["breakout_retest_success"] * 0.75,
                "pullback_quality": high_upside_metrics["pullback_quality"] * 0.55,
                "expected_move_after_cost": max(high_upside_metrics["cost_adjusted_expected_move"], 0.0) / 100.0,
                "expected_move_persistence": max(high_upside_metrics["cost_adjusted_expected_move_persistence"], 0.0) / 130.0,
                "cost_penalty": -max(tradability["cost_drag_bps"] - 18.0, 0.0) / 12.0,
                "stale_penalty": -2.0 if high_upside_metrics["stale_data"] else 0.0,
            }
            upside_screen_score = sum(score_breakdown.values())
            roi_context = {
                    **features,
                    **pair_payload,
                    "score": upside_screen_score,
                    "edge_score": high_upside_metrics["cost_adjusted_expected_move"],
                    "expected_move_bps": high_upside_metrics["cost_adjusted_expected_move"] + tradability["cost_drag_bps"],
                    "cost_drag_bps": tradability["cost_drag_bps"],
                    "spread_bps": tradability["spread_bps"],
                    "liquidity_usd": tradability["liquidity_usd"],
                    "liquidity_capacity_usd": high_upside_metrics["liquidity_capacity"],
                    "market_structure_score": tradability["market_structure_score"],
                    "recent_1h_return": high_upside_metrics["momentum_acceleration"],
                    "offline_ml_prediction": features["offline_ml_prediction"],
                    "window_stability": high_upside_metrics["depth_stability"],
                    "volatility_regime": features["volatility_regime"],
            }
            roi_payload = net_roi_diagnostics(
                roi_context,
                self.config,
            )
            roi_v2_payload = net_roi_v2_diagnostics(roi_context, self.config)
            offline_payload = self._offline_score_payload(
                {
                    **roi_context,
                    **roi_payload,
                    **roi_v2_payload,
                    "strategy_name": strategy_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "optimizer_profile": optimizer_profile,
                    "lock_duration_hours": max(duration_seconds / 3600, 1),
                    "horizon": horizon,
                    "scanner_source": features["scanner_source"],
                },
                horizon,
            )
            features["offline_ml_prediction"] = offline_payload.get("prediction", 0.0)
            features["offline_ml_status"] = offline_payload.get("status", "no_promoted_model")
            features["offline_ml_blend_enabled"] = offline_payload.get("blend_enabled", False)
            features["offline_ml_explanation"] = offline_payload
            score_breakdown["offline_ml"] = (
                float(self.config.get("ML_OFFLINE_SCORE_WEIGHT", 0.15) or 0.15)
                * float(offline_payload.get("prediction", 0.0) or 0.0)
                if bool(offline_payload.get("blend_enabled", False)) and offline_payload.get("status") == "promoted"
                else 0.0
            )
            upside_screen_score = sum(score_breakdown.values())
            roi_context["score"] = upside_screen_score
            features.update(roi_payload)
            features.update(roi_v2_payload)
            features["upside_screen_score"] = upside_screen_score
            score_breakdown["net_roi"] = float(roi_payload["net_roi_score"])
            score_breakdown["net_roi_v2"] = float(roi_v2_payload["net_roi_v2_score"])
            features["scanner_score_breakdown"] = dict(score_breakdown)
            score = upside_screen_score + float(roi_payload["net_roi_score"])
            if bool(self.config.get("NET_ROI_V2_ENABLED", True)):
                score += float(roi_v2_payload["net_roi_v2_score"]) * 0.25
            source = "pair_screening" if pair_bonus > hot_score and pair_bonus > 0 else "hot_token" if hot_score > 0 else "configured"
            rejection_reason = self._scanner_rejection_reason(tradability, {**high_upside_metrics, **roi_payload, **roi_v2_payload}, score)
            candidate = ScoredCandidate(
                symbol,
                score,
                technical_score,
                ml_score,
                hot_score,
                source,
                features,
                score_breakdown=score_breakdown,
                rejection_reason=rejection_reason,
                stale_data=bool(high_upside_metrics["stale_data"]),
            )
            if rejection_reason:
                rejected.append(self._diagnostic_row(symbol, rejection_reason, candidate=candidate))
                continue
            scored.append(candidate)

        scored.sort(key=lambda item: item.score, reverse=True)
        self._score_cache[cache_key] = (time.time(), scored)
        diagnostics = self._scan_diagnostics(cache_key, scored, rejected, runtime_seconds=max(time.perf_counter() - started_at, 0.0))
        self._diagnostic_cache[cache_key] = diagnostics
        self.last_scan_diagnostics = dict(diagnostics)
        return list(scored)

    def _hot_score(self, symbol: str, candles: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
        closes = [self._float(row.get("close")) for row in candles if isinstance(row, dict)]
        volumes = [self._float(row.get("volume")) for row in candles if isinstance(row, dict)]
        closes = [value for value in closes if value > 0]
        volumes = [value for value in volumes if value > 0]
        if len(closes) < 6 or len(volumes) < 6:
            return 0.0, {"reason": "insufficient_history"}

        recent_return = (closes[-1] - closes[-6]) / closes[-6]
        returns = [abs((closes[index] - closes[index - 1]) / closes[index - 1]) for index in range(1, len(closes))]
        volatility_pct = mean(returns[-20:]) * 100 if returns else 0.0
        volume_base = mean(volumes[-21:-1]) if len(volumes) > 21 else mean(volumes[:-1])
        volume_ratio = volumes[-1] / volume_base if volume_base > 0 else 0.0
        min_volume_ratio = float(self.config.get("HOT_TOKEN_VOLUME_SPIKE_RATIO", 1.8) or 1.8)
        min_volatility = float(self.config.get("HOT_TOKEN_MIN_VOLATILITY_PCT", 0.20) or 0.20)
        if volume_ratio < min_volume_ratio and volatility_pct < min_volatility and recent_return <= 0:
            return 0.0, {"reason": "not_hot"}

        score = max(recent_return * 100, 0.0) + max(volume_ratio - 1.0, 0.0) + volatility_pct
        return score, {
            "recent_return": recent_return,
            "volatility_pct": volatility_pct,
            "volume_ratio": volume_ratio,
            "source": "hot_token_scan",
        }

    def _technical_score(self, features: dict[str, Any]) -> float:
        trend = self._float(features.get("trend_strength"))
        rsi = self._float(features.get("rsi"), 50.0)
        macd_histogram = self._float(features.get("macd_histogram"))
        volume = features.get("volume_spike") if isinstance(features.get("volume_spike"), dict) else {}
        bands = features.get("bollinger_bands") if isinstance(features.get("bollinger_bands"), dict) else {}
        percent_b = self._float(bands.get("percent_b"), 0.5)
        volume_bonus = min(self._float(volume.get("ratio")) - 1.0, 3.0) if volume.get("is_spike") else 0.0
        rsi_bias = 0.0
        if 45 <= rsi <= 68:
            rsi_bias = 0.4
        elif rsi > 78:
            rsi_bias = -0.8
        elif rsi < 25:
            rsi_bias = -0.3
        band_bias = 0.3 if 0.45 <= percent_b <= 0.9 else -0.2 if percent_b > 1.05 else 0.0
        return trend * 100 + macd_histogram * 10 + volume_bonus + rsi_bias + band_bias

    def _tradability_payload(
        self,
        symbol: str,
        candles: list[dict[str, Any]],
        book: dict[str, Any],
        universe_payload: dict[str, Any],
    ) -> dict[str, Any]:
        if universe_payload:
            return {
                "symbol": symbol,
                "spread_bps": self._float(universe_payload.get("spread_bps")),
                "liquidity_usd": self._float(universe_payload.get("liquidity_usd")),
                "volatility_pct": self._float(universe_payload.get("volatility_pct")),
                "cost_drag_bps": self._float(universe_payload.get("cost_drag_bps")),
                "market_structure_score": self._float(universe_payload.get("market_structure_score")),
                "volatility_regime": str(universe_payload.get("volatility_regime", "unknown")),
                "source": universe_payload.get("source", "dynamic_liquid"),
            }
        spread = spread_bps(book)
        liquidity_usd = book_liquidity_usd(book, depth=max(1, int(self.config.get("VAULT_BOOK_DEPTH_LEVELS", 5))))
        volatility = volatility_pct(candles, lookback=30)
        regime, volatility_score = volatility_regime(volatility)
        structure_score = market_structure_score(
            liquidity_usd=liquidity_usd,
            spread=spread,
            volatility_score=volatility_score,
            min_liquidity_usd=float(self.config.get("UNIVERSE_MIN_LIQUIDITY_USD", 25_000.0) or 25_000.0),
            max_spread_bps=float(self.config.get("UNIVERSE_MAX_SPREAD_BPS", 15.0) or 15.0),
        )
        return {
            "symbol": symbol,
            "spread_bps": spread,
            "liquidity_usd": liquidity_usd,
            "volatility_pct": volatility,
            "cost_drag_bps": cost_drag_bps(
                spread=spread,
                fee_bps=float(self.config.get("FEE_BPS", 5.0) or 5.0),
                slippage_bps=float(self.config.get("SIM_SLIPPAGE_BPS", 8.0) or 8.0),
            ),
            "market_structure_score": structure_score,
            "volatility_regime": regime,
            "source": "scanner",
        }

    def _high_upside_metrics(self, candles: list[dict[str, Any]], tradability: dict[str, Any]) -> dict[str, Any]:
        closes = [self._float(row.get("close")) for row in candles if isinstance(row, dict)]
        volumes = [self._float(row.get("volume")) for row in candles if isinstance(row, dict)]
        closes = [value for value in closes if value > 0]
        volumes = [value for value in volumes if value > 0]
        short_return = self._window_return(closes, 6)
        medium_return = self._window_return(closes, 24)
        momentum_acceleration = short_return - (medium_return / 4.0 if medium_return else 0.0)
        volume_base = mean(volumes[-31:-1]) if len(volumes) > 31 else mean(volumes[:-1]) if len(volumes) > 1 else 0.0
        volume_impulse = volumes[-1] / volume_base if volume_base > 0 and volumes else 0.0
        recent_volume = mean(volumes[-4:-1]) if len(volumes) > 4 else volumes[-1] if volumes else 0.0
        volume_impulse_persistence = recent_volume / volume_base if volume_base > 0 else 0.0
        sustained_volume = mean(volumes[-8:]) if len(volumes) >= 8 else recent_volume
        sustained_volume_impulse = sustained_volume / volume_base if volume_base > 0 else 0.0
        recent_volatility = self._return_volatility(closes[-8:])
        base_volatility = self._return_volatility(closes[-40:])
        volatility_compression = max(0.0, 1.0 - (recent_volatility / max(base_volatility, 1e-9))) if base_volatility > 0 else 0.0
        volatility_expansion = recent_volatility / max(base_volatility, 1e-9) if base_volatility > 0 else 0.0
        previous_volatility = self._return_volatility(closes[-24:-8]) if len(closes) >= 24 else base_volatility
        previous_compression = max(0.0, 1.0 - (previous_volatility / max(base_volatility, 1e-9))) if base_volatility > 0 else 0.0
        volatility_expansion_after_compression = previous_compression * max(volatility_expansion - 1.0, 0.0)
        recent_high = max(closes[-30:]) if closes else 0.0
        breakout_proximity_bps = max((recent_high - closes[-1]) / closes[-1] * 10_000, 0.0) if closes and closes[-1] > 0 else 0.0
        prior_high = max(closes[-40:-8]) if len(closes) >= 40 else recent_high
        recent_low = min(closes[-12:]) if len(closes) >= 12 else min(closes) if closes else 0.0
        pullback_depth = (recent_high - recent_low) / recent_high if recent_high > 0 else 0.0
        pullback_quality = max(0.0, min(1.0, 1.0 - abs(pullback_depth - 0.012) / 0.05)) if closes else 0.0
        breakout_retest_success = (
            1.0
            if closes
            and prior_high > 0
            and max(closes[-8:]) >= prior_high
            and min(closes[-4:]) >= prior_high * 0.995
            else 0.0
        )
        max_spread = self._float(self.config.get("UNIVERSE_MAX_SPREAD_BPS"), 15.0)
        min_liquidity = self._float(self.config.get("UNIVERSE_MIN_LIQUIDITY_USD"), 25_000.0)
        spread_stability = max(0.0, 1.0 - self._float(tradability.get("spread_bps")) / max(max_spread, 1e-9))
        spread_stability_recent = spread_stability
        depth_stability = min(self._float(tradability.get("liquidity_usd")) / max(min_liquidity, 1e-9), 2.0) / 2.0
        expected_move_bps = max(short_return, momentum_acceleration, 0.0) * 10_000
        cost_adjusted_expected_move = expected_move_bps - self._float(tradability.get("cost_drag_bps"))
        persistent_move = max(short_return, 0.0) * 0.6 + max(medium_return, 0.0) * 0.4
        cost_adjusted_expected_move_persistence = persistent_move * 10_000 - self._float(tradability.get("cost_drag_bps"))
        regime = str(tradability.get("volatility_regime", "unknown"))
        regime_score = {
            "compressed": 0.55,
            "tradable": 1.0,
            "elevated": 0.70,
            "dislocated": 0.15,
        }.get(regime, 0.0)
        return {
            "momentum_acceleration": momentum_acceleration,
            "volume_impulse": volume_impulse,
            "liquidity_capacity": self._float(tradability.get("liquidity_usd")) * 0.05,
            "volatility_regime_score": regime_score,
            "volume_impulse_persistence": volume_impulse_persistence,
            "sustained_volume_impulse": sustained_volume_impulse,
            "volatility_compression": volatility_compression,
            "volatility_expansion": volatility_expansion,
            "volatility_expansion_after_compression": volatility_expansion_after_compression,
            "breakout_proximity_bps": breakout_proximity_bps,
            "pullback_quality": pullback_quality,
            "breakout_retest_success": breakout_retest_success,
            "spread_stability": spread_stability,
            "spread_stability_recent": spread_stability_recent,
            "depth_stability": depth_stability,
            "stale_data_age_seconds": self._stale_data_age_seconds(candles),
            "cost_adjusted_expected_move": cost_adjusted_expected_move,
            "cost_adjusted_expected_move_persistence": cost_adjusted_expected_move_persistence,
            "market_structure_trend": self._float(tradability.get("market_structure_score")),
            "raw_upside_score": (
                momentum_acceleration * 35.0
                + min(volume_impulse, 5.0) * 0.8
                + min(volume_impulse_persistence, 5.0) * 0.45
                + min(sustained_volume_impulse, 5.0) * 0.35
                + max(0.0, 1.0 - breakout_proximity_bps / 150.0) * 1.1
                + pullback_quality * 0.55
                + breakout_retest_success * 0.75
                + max(cost_adjusted_expected_move, 0.0) / 100.0
                + max(cost_adjusted_expected_move_persistence, 0.0) / 130.0
                + regime_score
            ),
            "stale_data": self._stale_data(candles),
        }

    def _offline_score_payload(self, context: dict[str, Any], horizon: str) -> dict[str, Any]:
        if self.offline_ranker is None:
            return {"status": "offline_ranker_unavailable", "prediction": 0.0, "blend_enabled": False}
        try:
            return dict(self.offline_ranker.score_payload(context, horizon))
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "offline_ranker_error",
                "prediction": 0.0,
                "blend_enabled": False,
                "blockers": [str(exc)],
            }

    def _scanner_rejection_reason(
        self,
        tradability: dict[str, Any],
        metrics: dict[str, Any],
        score: float,
    ) -> str:
        min_liquidity = self._float(self.config.get("UNIVERSE_MIN_LIQUIDITY_USD"), 25_000.0)
        max_spread = self._float(self.config.get("UNIVERSE_MAX_SPREAD_BPS"), 15.0)
        max_cost = max(18.0, self._float(self.config.get("AGGRESSIVE_1H_MAX_COST_DRAG_BPS"), 18.0))
        high_upside_enabled = bool(self.config.get("HIGH_UPSIDE_PROFILE_ENABLED", False))
        if self._float(tradability.get("liquidity_usd")) < min_liquidity:
            return "liquidity_below_threshold"
        if self._float(tradability.get("spread_bps")) > max_spread:
            return "spread_above_threshold"
        if high_upside_enabled and self._float(tradability.get("cost_drag_bps")) > max_cost:
            return "cost_drag_above_threshold"
        if str(tradability.get("volatility_regime")) == "dislocated":
            return "dislocated_volatility_regime"
        if bool(self.config.get("REALTIME_MARKET_ENABLED", False)) and bool(metrics.get("stale_data", False)):
            return "stale_market_data"
        if (
            high_upside_enabled
            and self._float(metrics.get("edge_after_cost_bps")) < self._float(self.config.get("NET_ROI_MIN_EDGE_BPS"), 4.0)
            and self._float(metrics.get("net_roi_score")) <= 0.0
        ):
            return "low_net_roi_edge"
        if high_upside_enabled and self._float(metrics.get("expected_fill_quality"), 1.0) < self._float(self.config.get("NET_ROI_MIN_FILL_QUALITY"), 0.55):
            return "low_expected_fill_quality"
        if high_upside_enabled and self._float(metrics.get("churn_penalty")) > self._float(self.config.get("NET_ROI_MAX_CHURN_PENALTY"), 0.35):
            return "excessive_churn"
        if high_upside_enabled and score <= 0:
            return "non_positive_upside_score"
        return ""

    def _scan_diagnostics(
        self,
        cache_key: tuple[str, str, str, int, str, str],
        accepted: list[ScoredCandidate],
        rejected: list[dict[str, Any]],
        *,
        runtime_seconds: float = 0.0,
    ) -> dict[str, Any]:
        rejected_breakdown: dict[str, int] = {}
        for row in rejected:
            reason = str(row.get("rejection_reason") or "unknown")
            rejected_breakdown[reason] = rejected_breakdown.get(reason, 0) + 1
        total = len(accepted) + len(rejected)
        rejection_rate = len(rejected) / total if total else 0.0
        diagnostics = {
            "scan_key": {
                "symbols": cache_key[0],
                "mode": cache_key[1],
                "timeframe": cache_key[2],
                "duration_seconds": cache_key[3],
                "strategy_name": cache_key[4],
                "optimizer_profile": cache_key[5],
            },
            "accepted": [self._diagnostic_row(candidate.symbol, "", candidate=candidate) for candidate in accepted[:20]],
            "rejected": rejected[:20],
            "rejection_breakdown": rejected_breakdown,
            "rejection_rate": rejection_rate,
            "cache_hit": False,
            "scan_runtime_seconds": runtime_seconds,
            "market_data_cache": self._market_data_cache_stats(),
        }
        if (
            bool(self.config.get("HIGH_UPSIDE_PROFILE_ENABLED", False))
            and total > 0
            and rejection_rate > self._float(self.config.get("HIGH_UPSIDE_MAX_SCANNER_REJECTION_RATE"), 0.65)
        ):
            self._disable_high_upside("scanner_rejection_rate_breach", {"rejection_rate": rejection_rate, "breakdown": rejected_breakdown})
            diagnostics["high_upside_auto_disabled"] = True
        else:
            diagnostics["high_upside_auto_disabled"] = False
        return diagnostics

    def _market_data_cache_stats(self) -> dict[str, Any]:
        if hasattr(self.market_data, "cache_stats"):
            try:
                return dict(self.market_data.cache_stats())
            except Exception:  # noqa: BLE001
                return {}
        return {}

    def _diagnostic_row(
        self,
        symbol: str,
        rejection_reason: str,
        *,
        source: str = "",
        candidate: ScoredCandidate | None = None,
    ) -> dict[str, Any]:
        if candidate is None:
            return {
                "symbol": symbol,
                "source": source,
                "score": 0.0,
                "net_roi_score": 0.0,
                "net_roi_v2_score": 0.0,
                "roi_quality_grade": "D",
                "roi_rejection_risk": "high",
                "regime_bucket": {},
                "regime_support": "regime-neutral",
                "expected_fill_quality": 0.0,
                "churn_penalty": 0.0,
                "edge_after_cost_bps": 0.0,
                "data_age_seconds": 0.0,
                "upside_screen_score": 0.0,
                "score_breakdown": {},
                "spread_bps": 0.0,
                "liquidity_usd": 0.0,
                "volatility_regime": "unknown",
                "cost_drag_bps": 0.0,
                "stale_data": False,
                "momentum_acceleration": 0.0,
                "volume_impulse": 0.0,
                "volume_impulse_persistence": 0.0,
                "volatility_compression": 0.0,
                "volatility_expansion": 0.0,
                "breakout_proximity_bps": 0.0,
                "spread_stability": 0.0,
                "depth_stability": 0.0,
                "sustained_volume_impulse": 0.0,
                "pullback_quality": 0.0,
                "breakout_retest_success": 0.0,
                "volatility_expansion_after_compression": 0.0,
                "spread_stability_recent": 0.0,
                "cost_adjusted_expected_move_persistence": 0.0,
                "stale_data_age_seconds": 0.0,
                "cost_adjusted_expected_move": 0.0,
                "liquidity_capacity_usd": 0.0,
                "market_structure_trend": 0.0,
                "offline_ml_prediction": 0.0,
                "offline_ml_status": "no_promoted_model",
                "offline_ml_blend_enabled": False,
                "rejection_reason": rejection_reason,
            }
        features = candidate.features or {}
        return {
            "symbol": candidate.symbol,
            "source": candidate.source,
            "score": candidate.score,
            "net_roi_score": self._float(features.get("net_roi_score")),
            "net_roi_v2_score": self._float(features.get("net_roi_v2_score")),
            "roi_quality_grade": str(features.get("roi_quality_grade", "D")),
            "roi_rejection_risk": str(features.get("roi_rejection_risk", "high")),
            "regime_bucket": dict(features.get("regime_bucket") or {}),
            "regime_support": str(features.get("regime_support", "regime-neutral")),
            "expected_fill_quality": self._float(features.get("expected_fill_quality")),
            "churn_penalty": self._float(features.get("churn_penalty")),
            "edge_after_cost_bps": self._float(features.get("edge_after_cost_bps")),
            "data_age_seconds": self._float(features.get("data_age_seconds", features.get("stale_data_age_seconds"))),
            "upside_screen_score": self._float(features.get("upside_screen_score"), candidate.score),
            "score_breakdown": dict(candidate.score_breakdown or features.get("scanner_score_breakdown") or {}),
            "spread_bps": self._float(features.get("spread_bps")),
            "liquidity_usd": self._float(features.get("liquidity_usd")),
            "volatility_regime": str(features.get("volatility_regime", "unknown")),
            "cost_drag_bps": self._float(features.get("cost_drag_bps")),
            "stale_data": bool(candidate.stale_data or features.get("stale_data", False)),
            "momentum_acceleration": self._float(features.get("momentum_acceleration")),
            "volume_impulse": self._float(features.get("volume_impulse")),
            "volume_impulse_persistence": self._float(features.get("volume_impulse_persistence")),
            "volatility_compression": self._float(features.get("volatility_compression")),
            "volatility_expansion": self._float(features.get("volatility_expansion")),
            "breakout_proximity_bps": self._float(features.get("breakout_proximity_bps")),
            "spread_stability": self._float(features.get("spread_stability")),
            "depth_stability": self._float(features.get("depth_stability")),
            "sustained_volume_impulse": self._float(features.get("sustained_volume_impulse")),
            "pullback_quality": self._float(features.get("pullback_quality")),
            "breakout_retest_success": self._float(features.get("breakout_retest_success")),
            "volatility_expansion_after_compression": self._float(features.get("volatility_expansion_after_compression")),
            "spread_stability_recent": self._float(features.get("spread_stability_recent")),
            "cost_adjusted_expected_move_persistence": self._float(features.get("cost_adjusted_expected_move_persistence")),
            "stale_data_age_seconds": self._float(features.get("stale_data_age_seconds")),
            "cost_adjusted_expected_move": self._float(features.get("cost_adjusted_expected_move")),
            "liquidity_capacity_usd": self._float(features.get("liquidity_capacity_usd", features.get("liquidity_capacity"))),
            "market_structure_trend": self._float(features.get("market_structure_trend")),
            "offline_ml_prediction": self._float(features.get("offline_ml_prediction")),
            "offline_ml_status": str(features.get("offline_ml_status", "no_promoted_model")),
            "offline_ml_blend_enabled": bool(features.get("offline_ml_blend_enabled", False)),
            "rejection_reason": rejection_reason or candidate.rejection_reason,
        }

    def _disable_high_upside(self, reason: str, details: dict[str, Any]) -> None:
        if not has_app_context():
            return
        Setting.set_json("high_upside_live_disabled", True)
        Setting.set_json("high_upside_live_disabled_reason", {"reason": reason, **dict(details or {})})
        db.session.add(
            AuditLog(
                category="risk",
                action="high_upside_auto_disabled",
                message=f"High-upside live profile auto-disabled: {reason}",
            )
        )
        db.session.flush()

    def _window_return(self, closes: list[float], lookback: int) -> float:
        if len(closes) <= lookback or closes[-lookback] <= 0:
            return 0.0
        return (closes[-1] - closes[-lookback]) / closes[-lookback]

    def _stale_data(self, candles: list[dict[str, Any]]) -> bool:
        if not candles:
            return True
        timestamp = self._float(candles[-1].get("timestamp")) if isinstance(candles[-1], dict) else 0.0
        if timestamp <= 0:
            return False
        if timestamp > 10_000_000_000:
            timestamp /= 1000.0
        max_age = self._float(self.config.get("REALTIME_MARKET_MAX_STALE_SECONDS"), 120.0)
        return time.time() - timestamp > max(max_age * 10, 3600.0)

    def _stale_data_age_seconds(self, candles: list[dict[str, Any]]) -> float:
        if not candles:
            return 1_000_000_000.0
        timestamp = self._float(candles[-1].get("timestamp")) if isinstance(candles[-1], dict) else 0.0
        if timestamp <= 0:
            return 0.0
        if timestamp > 10_000_000_000:
            timestamp /= 1000.0
        return max(time.time() - timestamp, 0.0)

    def _return_volatility(self, closes: list[float]) -> float:
        returns = [
            abs((closes[index] - closes[index - 1]) / closes[index - 1])
            for index in range(1, len(closes))
            if closes[index - 1] > 0
        ]
        return mean(returns) if returns else 0.0

    def _pair_lookup(
        self,
        symbols: list[str],
        *,
        mode: str,
        timeframe: str,
        duration_hours: float,
    ) -> dict[str, dict[str, Any]]:
        if self.pair_screening is None or not bool(self.config.get("PAIR_SCREENING_ENABLED", False)):
            return {}
        try:
            candidates = self.pair_screening.screen(
                symbols,
                mode=mode,
                timeframe=timeframe,
                duration_hours=duration_hours,
                pair_mode="both",
            )
        except Exception:  # noqa: BLE001
            return {}
        lookup: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            payload = candidate.as_dict()
            for symbol_key in ("base_symbol", "pair_symbol", "leader_symbol"):
                symbol = str(payload.get(symbol_key) or "").upper()
                if not symbol:
                    continue
                existing = lookup.get(symbol)
                if existing is None or float(payload.get("pair_score", 0.0) or 0.0) > float(existing.get("pair_score", 0.0) or 0.0):
                    lookup[symbol] = payload
        return lookup

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        return safe_float(value, default)

    def _book_spread_bps(self, book: dict[str, Any]) -> float:
        return spread_bps(book)

    def _book_liquidity_usd(self, book: dict[str, Any]) -> float:
        return book_liquidity_usd(book, depth=max(1, int(self.config.get("VAULT_BOOK_DEPTH_LEVELS", 5))))

    _best_bid_ask = staticmethod(best_bid_ask)
    _level_price_size = staticmethod(level_price_size)
    _volatility_pct = staticmethod(volatility_pct)
    _volatility_regime = staticmethod(volatility_regime)
