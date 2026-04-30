"""Dependency-light market-structure feature extraction."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any


class MarketStructureService:
    """Build neutral-to-positive market-structure features from existing data."""

    def __init__(self, config: dict[str, Any], market_data: Any) -> None:
        self.config = config
        self.market_data = market_data

    def snapshot(
        self,
        symbol: str,
        timeframe: str,
        mode: str = "testnet",
        *,
        candles: list[dict[str, Any]] | None = None,
        order_book: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return fail-closed market-structure features for ranking."""

        provider = str(self.config.get("MARKET_STRUCTURE_PROVIDER", "existing") or "existing")
        if not bool(self.config.get("MARKET_STRUCTURE_FEATURES_ENABLED", False)):
            return self._neutral(provider=provider, enabled=False)

        try:
            candle_rows = candles if candles is not None else self.market_data.get_candles(symbol, timeframe, mode=mode, limit=120)
            book = order_book if order_book is not None else self.market_data.get_order_book(symbol, mode)
            features = self._from_existing_data(symbol, candle_rows or [], book if isinstance(book, dict) else {})
            optional = self._optional_provider_features(symbol, mode, provider)
            features.update(optional)
            features["score"] = self._score(features)
            features["provider"] = provider
            features["enabled"] = True
            features["fail_closed"] = False
            features["coverage"] = self._coverage(features)
            return features
        except Exception as exc:  # noqa: BLE001
            if bool(self.config.get("MARKET_STRUCTURE_FAIL_CLOSED", True)):
                return self._neutral(provider=provider, enabled=True, error=str(exc))
            raise

    def _from_existing_data(
        self,
        symbol: str,
        candles: list[dict[str, Any]],
        book: dict[str, Any],
    ) -> dict[str, Any]:
        closes = [self._float(row.get("close")) for row in candles if isinstance(row, dict)]
        closes = [value for value in closes if value > 0]
        volumes = [max(self._float(row.get("volume")), 0.0) for row in candles if isinstance(row, dict)]
        returns = [
            (closes[index] - closes[index - 1]) / max(closes[index - 1], 1e-9)
            for index in range(1, len(closes))
            if closes[index - 1] > 0
        ]

        levels = book.get("levels", []) if isinstance(book, dict) else []
        bids = levels[0] if len(levels) > 0 and isinstance(levels[0], list) else []
        asks = levels[1] if len(levels) > 1 and isinstance(levels[1], list) else []
        bid, bid_size = self._level_price_size(bids[0]) if bids else (0.0, 0.0)
        ask, ask_size = self._level_price_size(asks[0]) if asks else (0.0, 0.0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else closes[-1] if closes else 0.0
        spread_bps = ((ask - bid) / mid) * 10_000 if bid > 0 and ask >= bid and mid > 0 else 0.0
        depth = self._book_depth_usd(bids, asks)
        depth_scale = max(self._float(self.config.get("MARKET_STRUCTURE_DEPTH_USD_SCALE"), 100_000.0), 1.0)
        book_depth_score = min(depth / depth_scale, 1.0)

        recent_volume = volumes[-1] if volumes else 0.0
        base_volume = mean(volumes[-21:-1]) if len(volumes) >= 21 else mean(volumes[:-1]) if len(volumes) > 1 else recent_volume
        volume_impulse = max(0.0, min((recent_volume / max(base_volume, 1e-9)) - 1.0, 5.0)) if recent_volume > 0 else 0.0
        volatility = pstdev(returns[-30:]) if len(returns) >= 2 else 0.0
        volatility_regime, volatility_score = self._volatility_regime(volatility)
        spread_depth_ratio = spread_bps / max(book_depth_score * 100.0, 1.0)

        return {
            "symbol": symbol,
            "funding_rate": 0.0,
            "open_interest_change_pct": 0.0,
            "book_depth_usd": depth,
            "book_depth_score": book_depth_score,
            "best_bid_size": bid_size,
            "best_ask_size": ask_size,
            "spread_bps": spread_bps,
            "spread_trend_bps": spread_bps,
            "spread_depth_ratio": spread_depth_ratio,
            "liquidation_proxy": 0.0,
            "volume_impulse": volume_impulse,
            "volatility_pct": volatility * 100,
            "volatility_regime": volatility_regime,
            "volatility_regime_score": volatility_score,
            "source": "existing",
        }

    def _optional_provider_features(self, symbol: str, mode: str, provider: str) -> dict[str, Any]:
        if provider == "existing":
            return {}

        features: dict[str, Any] = {}
        for name, output_key in (
            ("get_funding_rate", "funding_rate"),
            ("get_open_interest_change", "open_interest_change_pct"),
            ("get_liquidation_proxy", "liquidation_proxy"),
        ):
            method = getattr(self.market_data, name, None)
            if method is None:
                continue
            features[output_key] = self._float(method(symbol, mode))
        return features

    def _score(self, features: dict[str, Any]) -> float:
        depth = self._float(features.get("book_depth_score"))
        volume = min(self._float(features.get("volume_impulse")) / 3.0, 1.0)
        volatility = self._float(features.get("volatility_regime_score"))
        oi = min(abs(self._float(features.get("open_interest_change_pct"))) / 0.20, 1.0)
        liquidation = min(self._float(features.get("liquidation_proxy")), 1.0)
        spread_penalty = min(max(self._float(features.get("spread_trend_bps")) - 5.0, 0.0) / 45.0, 1.0)
        funding_penalty = min(abs(self._float(features.get("funding_rate"))) / 0.001, 1.0) * 0.08
        score = (
            depth * 0.30
            + volume * 0.22
            + volatility * 0.20
            + oi * 0.12
            + max(0.0, 1.0 - liquidation) * 0.08
            + max(0.0, 1.0 - spread_penalty) * 0.08
            - funding_penalty
        )
        return max(0.0, min(score, 1.0))

    def _book_depth_usd(self, bids: list[Any], asks: list[Any]) -> float:
        depth = max(1, int(self._float(self.config.get("VAULT_BOOK_DEPTH_LEVELS"), 5)))
        total = 0.0
        for side in (bids[:depth], asks[:depth]):
            for row in side:
                price, size = self._level_price_size(row)
                total += max(price, 0.0) * max(size, 0.0)
        return total

    @staticmethod
    def _volatility_regime(volatility: float) -> tuple[str, float]:
        if volatility <= 0:
            return "unknown", 0.0
        if volatility < 0.002:
            return "compressed", 0.45
        if volatility <= 0.015:
            return "tradable", 1.0
        if volatility <= 0.035:
            return "elevated", 0.65
        return "dislocated", 0.20

    @staticmethod
    def _coverage(features: dict[str, Any]) -> float:
        keys = (
            "book_depth_usd",
            "spread_bps",
            "volume_impulse",
            "volatility_pct",
            "funding_rate",
            "open_interest_change_pct",
            "liquidation_proxy",
        )
        present = sum(1 for key in keys if features.get(key) not in {None, ""})
        return present / len(keys)

    @staticmethod
    def _neutral(*, provider: str, enabled: bool, error: str = "") -> dict[str, Any]:
        return {
            "enabled": enabled,
            "provider": provider,
            "score": 0.0,
            "coverage": 0.0,
            "funding_rate": 0.0,
            "open_interest_change_pct": 0.0,
            "book_depth_usd": 0.0,
            "book_depth_score": 0.0,
            "spread_bps": 0.0,
            "spread_trend_bps": 0.0,
            "liquidation_proxy": 0.0,
            "volume_impulse": 0.0,
            "volatility_pct": 0.0,
            "volatility_regime": "unknown",
            "volatility_regime_score": 0.0,
            "fail_closed": bool(error),
            "error": error,
        }

    @staticmethod
    def _level_price_size(row: Any) -> tuple[float, float]:
        if isinstance(row, dict):
            return (
                MarketStructureService._float(row.get("px", row.get("price"))),
                MarketStructureService._float(row.get("sz", row.get("size"))),
            )
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            return MarketStructureService._float(row[0]), MarketStructureService._float(row[1])
        return 0.0, 0.0

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
