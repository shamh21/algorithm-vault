"""Fail-closed pair discovery and pair-trading candidate scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from statistics import mean, pstdev
from typing import Any

from ..ml.online_ranker import extract_features, horizon_from_duration


@dataclass(frozen=True, slots=True)
class PairCandidate:
    """Ranked opportunity between two symbols."""

    base_symbol: str
    pair_symbol: str
    pair_mode: str
    score: float
    correlation: float
    spread_zscore: float
    hedge_ratio: float
    spread_half_life: float
    liquidity_balance: float
    liquidity_usd: float
    spread_cost_bps: float
    market_structure_score: float
    optimizer_edge_score: float
    ml_score: float
    leader_symbol: str
    laggard_symbol: str
    pair_signal: dict[str, Any]
    skip_reason: str = ""

    @property
    def pair_group_id(self) -> str:
        symbols = "-".join(sorted((self.base_symbol, self.pair_symbol))).lower()
        return f"pair-{self.pair_mode}-{symbols}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "pair_group_id": self.pair_group_id,
            "base_symbol": self.base_symbol,
            "pair_symbol": self.pair_symbol,
            "pair_mode": self.pair_mode,
            "score": self.score,
            "pair_score": self.score,
            "correlation": self.correlation,
            "spread_zscore": self.spread_zscore,
            "hedge_ratio": self.hedge_ratio,
            "spread_half_life": self.spread_half_life,
            "liquidity_balance": self.liquidity_balance,
            "liquidity_usd": self.liquidity_usd,
            "spread_cost_bps": self.spread_cost_bps,
            "market_structure_score": self.market_structure_score,
            "optimizer_edge_score": self.optimizer_edge_score,
            "ml_score": self.ml_score,
            "leader_symbol": self.leader_symbol,
            "laggard_symbol": self.laggard_symbol,
            "pair_signal": dict(self.pair_signal),
            "skip_reason": self.skip_reason,
        }


class PairScreeningService:
    """Finds stat-arb and relative-strength pair candidates from existing data."""

    def __init__(
        self,
        config: dict[str, Any],
        market_data: Any,
        universe_service: Any | None = None,
        market_structure: Any | None = None,
        online_ranker: Any | None = None,
    ) -> None:
        self.config = config
        self.market_data = market_data
        self.universe_service = universe_service
        self.market_structure = market_structure
        self.online_ranker = online_ranker
        self.last_rejections: dict[str, int] = {}

    def screen(
        self,
        symbols: list[str] | tuple[str, ...] | None = None,
        *,
        mode: str = "testnet",
        timeframe: str = "5m",
        duration_hours: int | float = 24,
        pair_mode: str = "both",
        limit: int | None = None,
    ) -> list[PairCandidate]:
        """Return ranked pair candidates, failing closed to an empty list."""

        self.last_rejections = {}
        if not bool(self.config.get("PAIR_SCREENING_ENABLED", False)):
            return []

        universe = self._symbol_universe(symbols, mode, timeframe)
        if len(universe) < 2:
            self._reject("insufficient_universe")
            return []

        candle_limit = max(40, int(self._float(self.config.get("PAIR_LOOKBACK_CANDLES"), 120)))
        candles_by_symbol: dict[str, list[dict[str, Any]]] = {}
        books_by_symbol: dict[str, dict[str, Any]] = {}
        for symbol in universe:
            try:
                candles = self.market_data.get_candles(symbol, timeframe, mode=mode, limit=candle_limit)
                book = self.market_data.get_order_book(symbol, mode)
            except Exception:  # noqa: BLE001
                self._reject("market_data_unavailable")
                continue
            if len(self._closes(candles)) < 30:
                self._reject("insufficient_history")
                continue
            candles_by_symbol[symbol] = candles
            books_by_symbol[symbol] = book if isinstance(book, dict) else {}

        candidates: list[PairCandidate] = []
        modes = {"stat_arb", "relative_strength"} if pair_mode == "both" else {str(pair_mode)}
        for base_symbol, pair_symbol in combinations(sorted(candles_by_symbol), 2):
            metrics = self._pair_metrics(
                base_symbol,
                pair_symbol,
                candles_by_symbol[base_symbol],
                candles_by_symbol[pair_symbol],
                books_by_symbol.get(base_symbol, {}),
                books_by_symbol.get(pair_symbol, {}),
                mode=mode,
                timeframe=timeframe,
                duration_hours=duration_hours,
            )
            if metrics is None:
                continue
            if "stat_arb" in modes:
                candidate = self._candidate_from_metrics(metrics, "stat_arb", duration_hours)
                if candidate is not None:
                    candidates.append(candidate)
            if "relative_strength" in modes:
                candidate = self._candidate_from_metrics(metrics, "relative_strength", duration_hours)
                if candidate is not None:
                    candidates.append(candidate)

        candidates.sort(key=lambda item: item.score, reverse=True)
        max_candidates = limit if limit is not None else int(self._float(self.config.get("PAIR_MAX_CANDIDATES"), 8))
        return candidates[: max(1, int(max_candidates or 1))]

    def _pair_metrics(
        self,
        base_symbol: str,
        pair_symbol: str,
        base_candles: list[dict[str, Any]],
        pair_candles: list[dict[str, Any]],
        base_book: dict[str, Any],
        pair_book: dict[str, Any],
        *,
        mode: str,
        timeframe: str,
        duration_hours: int | float,
    ) -> dict[str, Any] | None:
        base_closes = self._closes(base_candles)
        pair_closes = self._closes(pair_candles)
        length = min(len(base_closes), len(pair_closes))
        if length < 30:
            self._reject("insufficient_history")
            return None
        base_closes = base_closes[-length:]
        pair_closes = pair_closes[-length:]
        base_returns = self._returns(base_closes)
        pair_returns = self._returns(pair_closes)
        correlation = self._correlation(base_returns, pair_returns)
        if math.isnan(correlation):
            self._reject("correlation_unavailable")
            return None

        hedge_ratio = self._hedge_ratio(base_closes, pair_closes)
        spreads = [base - hedge_ratio * pair for base, pair in zip(base_closes, pair_closes)]
        spread_std = pstdev(spreads) if len(spreads) >= 2 else 0.0
        spread_zscore = (spreads[-1] - mean(spreads)) / spread_std if spread_std > 0 else 0.0
        half_life = self._spread_half_life(spreads)
        base_liquidity = self._book_liquidity_usd(base_book)
        pair_liquidity = self._book_liquidity_usd(pair_book)
        base_spread = self._book_spread_bps(base_book, base_closes[-1])
        pair_spread = self._book_spread_bps(pair_book, pair_closes[-1])
        liquidity_floor = self._float(self.config.get("PAIR_MIN_LIQUIDITY_USD"), 25_000.0)
        spread_cap = self._float(self.config.get("PAIR_MAX_SPREAD_BPS"), 20.0)
        if min(base_liquidity, pair_liquidity) < liquidity_floor:
            self._reject("liquidity_below_threshold")
            return None
        spread_cost_bps = base_spread + pair_spread
        if spread_cost_bps > spread_cap:
            self._reject("spread_cost_above_threshold")
            return None
        liquidity_balance = min(base_liquidity, pair_liquidity) / max(max(base_liquidity, pair_liquidity), 1e-9)
        if liquidity_balance < self._float(self.config.get("PAIR_MIN_LIQUIDITY_BALANCE"), 0.20):
            self._reject("liquidity_imbalance")
            return None

        base_return = (base_closes[-1] - base_closes[max(0, length - 12)]) / max(base_closes[max(0, length - 12)], 1e-9)
        pair_return = (pair_closes[-1] - pair_closes[max(0, length - 12)]) / max(pair_closes[max(0, length - 12)], 1e-9)
        leader_symbol = base_symbol if base_return >= pair_return else pair_symbol
        laggard_symbol = pair_symbol if leader_symbol == base_symbol else base_symbol
        market_structure_score = self._market_structure_score(
            base_symbol,
            pair_symbol,
            timeframe,
            mode,
            base_candles,
            pair_candles,
            base_book,
            pair_book,
        )

        return {
            "base_symbol": base_symbol,
            "pair_symbol": pair_symbol,
            "correlation": correlation,
            "spread_zscore": spread_zscore,
            "hedge_ratio": hedge_ratio,
            "spread_half_life": half_life,
            "liquidity_balance": liquidity_balance,
            "liquidity_usd": min(base_liquidity, pair_liquidity),
            "spread_cost_bps": spread_cost_bps,
            "market_structure_score": market_structure_score,
            "base_return": base_return,
            "pair_return": pair_return,
            "leader_symbol": leader_symbol,
            "laggard_symbol": laggard_symbol,
            "duration_hours": duration_hours,
        }

    def _candidate_from_metrics(
        self,
        metrics: dict[str, Any],
        pair_mode: str,
        duration_hours: int | float,
    ) -> PairCandidate | None:
        correlation = self._float(metrics.get("correlation"))
        spread_zscore = self._float(metrics.get("spread_zscore"))
        abs_zscore = abs(spread_zscore)
        min_corr = self._float(self.config.get("PAIR_MIN_CORRELATION"), 0.75)
        max_zscore = self._float(self.config.get("PAIR_MAX_SPREAD_ZSCORE"), 2.5)
        if pair_mode == "stat_arb":
            if correlation < min_corr:
                self._reject("correlation_below_threshold")
                return None
            if abs_zscore > max_zscore:
                self._reject("spread_zscore_above_threshold")
                return None
            pair_signal = self._stat_arb_signal(metrics)
            divergence_score = min(abs_zscore / max(max_zscore, 1e-9), 1.0)
            mode_bonus = divergence_score * 0.25
        else:
            relative_corr_floor = max(0.35, min_corr * 0.50)
            if correlation < relative_corr_floor:
                self._reject("relative_strength_correlation_below_floor")
                return None
            pair_signal = self._relative_strength_signal(metrics)
            relative_gap = abs(self._float(metrics.get("base_return")) - self._float(metrics.get("pair_return")))
            mode_bonus = min(relative_gap * 20.0, 0.35)

        liquidity_score = min(self._float(metrics.get("liquidity_usd")) / max(self._float(self.config.get("PAIR_MIN_LIQUIDITY_USD"), 25_000.0), 1.0), 3.0)
        cost_penalty = self._float(metrics.get("spread_cost_bps")) / max(self._float(self.config.get("PAIR_MAX_SPREAD_BPS"), 20.0), 1.0)
        half_life_penalty = min(self._float(metrics.get("spread_half_life")) / 200.0, 0.5)
        optimizer_edge = self._optimizer_edge_score(metrics)
        ml_score = self._ml_score(metrics, pair_mode, duration_hours)
        score = max(
            0.0,
            correlation * 0.35
            + self._float(metrics.get("liquidity_balance")) * 0.15
            + min(liquidity_score, 1.0) * 0.10
            + self._float(metrics.get("market_structure_score")) * 0.12
            + optimizer_edge * 0.12
            + max(ml_score, 0.0) * 0.08
            + mode_bonus
            - cost_penalty * 0.10
            - half_life_penalty * 0.05,
        )
        if score <= 0:
            self._reject("pair_score_non_positive")
            return None

        return PairCandidate(
            base_symbol=str(metrics.get("base_symbol") or ""),
            pair_symbol=str(metrics.get("pair_symbol") or ""),
            pair_mode=pair_mode,
            score=score,
            correlation=correlation,
            spread_zscore=spread_zscore,
            hedge_ratio=self._float(metrics.get("hedge_ratio"), 1.0),
            spread_half_life=self._float(metrics.get("spread_half_life")),
            liquidity_balance=self._float(metrics.get("liquidity_balance")),
            liquidity_usd=self._float(metrics.get("liquidity_usd")),
            spread_cost_bps=self._float(metrics.get("spread_cost_bps")),
            market_structure_score=self._float(metrics.get("market_structure_score")),
            optimizer_edge_score=optimizer_edge,
            ml_score=ml_score,
            leader_symbol=str(metrics.get("leader_symbol") or ""),
            laggard_symbol=str(metrics.get("laggard_symbol") or ""),
            pair_signal=pair_signal,
        )

    def _symbol_universe(self, symbols: list[str] | tuple[str, ...] | None, mode: str, timeframe: str) -> list[str]:
        configured = [str(symbol).upper() for symbol in (symbols or self.config.get("ALLOWED_SYMBOLS", [])) if str(symbol).strip()]
        discovered: list[str] = []
        if self.universe_service is not None:
            try:
                discovered = [str(symbol).upper() for symbol in self.universe_service.symbols(mode, timeframe)]
            except Exception:  # noqa: BLE001
                discovered = []
        max_symbols = max(2, int(self._float(self.config.get("PAIR_UNIVERSE_MAX_SYMBOLS"), 12)))
        return list(dict.fromkeys([*configured, *discovered]))[:max_symbols]

    def _market_structure_score(
        self,
        base_symbol: str,
        pair_symbol: str,
        timeframe: str,
        mode: str,
        base_candles: list[dict[str, Any]],
        pair_candles: list[dict[str, Any]],
        base_book: dict[str, Any],
        pair_book: dict[str, Any],
    ) -> float:
        if self.market_structure is None:
            return 0.0
        scores: list[float] = []
        for symbol, candles, book in (
            (base_symbol, base_candles, base_book),
            (pair_symbol, pair_candles, pair_book),
        ):
            try:
                snapshot = self.market_structure.snapshot(symbol, timeframe, mode=mode, candles=candles, order_book=book)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(snapshot, dict):
                scores.append(self._float(snapshot.get("score")))
        return mean(scores) if scores else 0.0

    def _optimizer_edge_score(self, metrics: dict[str, Any]) -> float:
        edges = self.config.get("PAIR_OPTIMIZER_EDGE_BY_SYMBOL") or {}
        if not isinstance(edges, dict):
            return 0.0
        values: list[float] = []
        for symbol in (metrics.get("base_symbol"), metrics.get("pair_symbol")):
            value = edges.get(str(symbol).upper()) if symbol is not None else None
            if value is not None:
                values.append(self._float(value))
        return min(max(mean(values), 0.0), 1.0) if values else 0.0

    def _ml_score(self, metrics: dict[str, Any], pair_mode: str, duration_hours: int | float) -> float:
        if self.online_ranker is None or not bool(self.config.get("ML_RANKER_ENABLED", False)):
            return 0.0
        horizon = horizon_from_duration(duration_hours)
        features = extract_features(
            {
                "symbol": metrics.get("base_symbol"),
                "timeframe": "pair",
                "horizon": horizon,
                "duration_hours": duration_hours,
                "pair_mode": pair_mode,
                "pair_score": 0.0,
                "pair_correlation": metrics.get("correlation"),
                "pair_spread_zscore": abs(self._float(metrics.get("spread_zscore"))),
                "pair_liquidity_balance": metrics.get("liquidity_balance"),
                "pair_spread_cost_bps": metrics.get("spread_cost_bps"),
                "market_structure_score": metrics.get("market_structure_score"),
                "liquidity_usd": metrics.get("liquidity_usd"),
            }
        )
        try:
            return float(self.online_ranker.predict_score(features, horizon) or 0.0)
        except Exception:  # noqa: BLE001
            return 0.0

    @staticmethod
    def _stat_arb_signal(metrics: dict[str, Any]) -> dict[str, Any]:
        base_symbol = str(metrics.get("base_symbol") or "")
        pair_symbol = str(metrics.get("pair_symbol") or "")
        zscore = PairScreeningService._float(metrics.get("spread_zscore"))
        if zscore >= 0:
            return {
                "base_side": "sell",
                "pair_side": "buy",
                "long_symbol": pair_symbol,
                "short_symbol": base_symbol,
                "reason": "spread_above_mean",
            }
        return {
            "base_side": "buy",
            "pair_side": "sell",
            "long_symbol": base_symbol,
            "short_symbol": pair_symbol,
            "reason": "spread_below_mean",
        }

    @staticmethod
    def _relative_strength_signal(metrics: dict[str, Any]) -> dict[str, Any]:
        leader = str(metrics.get("leader_symbol") or "")
        laggard = str(metrics.get("laggard_symbol") or "")
        return {
            "leader_symbol": leader,
            "laggard_symbol": laggard,
            "directional_symbol": leader,
            "peer_symbol": laggard,
            "reason": "leader_outperforming_peer",
        }

    @staticmethod
    def _closes(candles: list[dict[str, Any]]) -> list[float]:
        closes = [PairScreeningService._float(row.get("close")) for row in candles if isinstance(row, dict)]
        return [value for value in closes if value > 0]

    @staticmethod
    def _returns(values: list[float]) -> list[float]:
        return [
            (values[index] - values[index - 1]) / max(values[index - 1], 1e-9)
            for index in range(1, len(values))
            if values[index - 1] > 0
        ]

    @staticmethod
    def _correlation(a_values: list[float], b_values: list[float]) -> float:
        length = min(len(a_values), len(b_values))
        if length < 2:
            return math.nan
        a = a_values[-length:]
        b = b_values[-length:]
        a_mean = mean(a)
        b_mean = mean(b)
        covariance = sum((a_item - a_mean) * (b_item - b_mean) for a_item, b_item in zip(a, b)) / length
        a_std = pstdev(a)
        b_std = pstdev(b)
        if a_std <= 0 or b_std <= 0:
            return math.nan
        return max(-1.0, min(covariance / (a_std * b_std), 1.0))

    @staticmethod
    def _hedge_ratio(base_closes: list[float], pair_closes: list[float]) -> float:
        length = min(len(base_closes), len(pair_closes))
        if length < 2:
            return 1.0
        base = base_closes[-length:]
        pair = pair_closes[-length:]
        pair_mean = mean(pair)
        base_mean = mean(base)
        variance = sum((value - pair_mean) ** 2 for value in pair) / length
        if variance <= 0:
            return base[-1] / max(pair[-1], 1e-9)
        covariance = sum((base_item - base_mean) * (pair_item - pair_mean) for base_item, pair_item in zip(base, pair)) / length
        return covariance / variance

    @staticmethod
    def _spread_half_life(spreads: list[float]) -> float:
        if len(spreads) < 3:
            return 0.0
        lagged = spreads[:-1]
        current = spreads[1:]
        corr = PairScreeningService._correlation(lagged, current)
        if math.isnan(corr) or abs(corr) <= 0 or abs(corr) >= 1:
            return float(len(spreads))
        return max(1.0, min(-math.log(2) / math.log(abs(corr)), float(len(spreads))))

    @staticmethod
    def _book_liquidity_usd(book: dict[str, Any]) -> float:
        levels = book.get("levels", []) if isinstance(book, dict) else []
        if not isinstance(levels, list):
            return 0.0
        total = 0.0
        for side in levels[:2]:
            if not isinstance(side, list):
                continue
            for row in side[:5]:
                price, size = PairScreeningService._level_price_size(row)
                total += max(price, 0.0) * max(size, 0.0)
        return total

    @staticmethod
    def _book_spread_bps(book: dict[str, Any], fallback_mid: float) -> float:
        levels = book.get("levels", []) if isinstance(book, dict) else []
        try:
            bid = PairScreeningService._level_price_size(levels[0][0])[0]
            ask = PairScreeningService._level_price_size(levels[1][0])[0]
        except Exception:  # noqa: BLE001
            return 0.0
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else fallback_mid
        if bid <= 0 or ask <= 0 or ask < bid or mid <= 0:
            return 0.0
        return ((ask - bid) / mid) * 10_000

    @staticmethod
    def _level_price_size(row: Any) -> tuple[float, float]:
        if isinstance(row, dict):
            return PairScreeningService._float(row.get("px", row.get("price"))), PairScreeningService._float(row.get("sz", row.get("size")))
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            return PairScreeningService._float(row[0]), PairScreeningService._float(row[1])
        return 0.0, 0.0

    def _reject(self, reason: str) -> None:
        self.last_rejections[reason] = self.last_rejections.get(reason, 0) + 1

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
