"""Paper Vault ensemble simulation for the Backtests PWA."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from ..backtesting.engine import BacktestConfig
from ..models import BacktestRun, LeveragedMarket
from ..services.provider_assets import normalize_provider, provider_collateral_asset
from ..services.tradability import book_liquidity_usd, cost_drag_bps, spread_bps, volatility_pct, volatility_regime


AUTO_STRATEGIES = (
    "breakout",
    "ema_crossover",
    "mean_reversion",
    "rsi_mean_reversion",
    "rule_based_signal",
    "scalping",
    "volatility_breakout",
)

PUBLIC_TIMEFRAMES = (
    {"value": "live", "label": "LIVE", "source": "1m"},
    {"value": "5m", "label": "5M", "source": "5m"},
    {"value": "15m", "label": "15M", "source": "15m"},
    {"value": "45m", "label": "45M", "source": "15m"},
    {"value": "4h", "label": "4HR", "source": "4h"},
    {"value": "1d", "label": "1D", "source": "4h"},
)

_PUBLIC_TIMEFRAME_VALUES = {item["value"] for item in PUBLIC_TIMEFRAMES}


@dataclass(frozen=True, slots=True)
class SimulationInput:
    provider: str
    symbol: str
    venue_symbol: str
    timeframe: str
    allocation_usd: float
    cycle: str = "1h10"


class VaultBacktestSimulator:
    """Runs an auto-optimized paper vault simulation without live order side effects."""

    def __init__(
        self,
        config: dict[str, Any],
        registry: Any,
        market_data: Any,
        backtest_engine: Any,
        *,
        leveraged_markets: Any | None = None,
        trading_connections: Any | None = None,
        ml_projection_engine: Any | None = None,
        market_scanner: Any | None = None,
        ml_decision_engine: Any | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.market_data = market_data
        self.backtest_engine = backtest_engine
        self.leveraged_markets = leveraged_markets
        self.trading_connections = trading_connections
        self.ml_projection_engine = ml_projection_engine
        self.market_scanner = market_scanner
        self.ml_decision_engine = ml_decision_engine

    def timeframes(self) -> list[dict[str, str]]:
        return [dict(item) for item in PUBLIC_TIMEFRAMES]

    def allocation_cap_usd(self) -> float:
        paper_balance = float(self.config.get("BACKTEST_PAPER_BALANCE_USD", 10_000.0) or 10_000.0)
        return min(max(paper_balance, 0.0), 10_000.0)

    def allocation_default_usd(self) -> float:
        default = float(self.config.get("BACKTEST_ALLOCATION_DEFAULT_USD", 10_000.0) or 10_000.0)
        return min(max(default, 0.0), self.allocation_cap_usd())

    def symbol_payload(
        self,
        *,
        user: Any | None,
        query: str = "",
        cursor: int = 0,
        limit: int = 40,
        refresh: bool = False,
        mode: str = "live",
    ) -> dict[str, Any]:
        if refresh and user is not None and self.leveraged_markets is not None:
            try:
                self.leveraged_markets.sync_for_user(user.id, mode=mode, feature_scope="allowed", persist_features=False)
            except Exception:
                pass

        rows = self._symbol_rows(user=user)
        query_key = str(query or "").strip().upper()
        if query_key:
            rows = [
                row
                for row in rows
                if query_key in str(row.get("symbol", "")).upper()
                or query_key in str(row.get("venue_symbol", "")).upper()
                or query_key in str(row.get("provider_label", "")).upper()
            ]
        offset = max(int(cursor or 0), 0)
        page_size = max(1, min(int(limit or 40), 80))
        page = rows[offset : offset + page_size]
        next_cursor = offset + len(page)
        return {
            "ok": True,
            "symbols": page,
            "total": len(rows),
            "count": len(page),
            "cursor": str(offset),
            "next_cursor": str(next_cursor) if next_cursor < len(rows) else None,
            "has_more": next_cursor < len(rows),
            "updated_at": self._utc_now(),
        }

    def quote_payload(
        self,
        *,
        provider: str = "",
        symbol: str = "",
        venue_symbol: str = "",
        allocation_usd: float = 0.0,
        mode: str = "live",
    ) -> dict[str, Any]:
        symbol_key = str(symbol or "BTC").upper().strip()
        provider_key = normalize_provider(provider, default="global")
        venue_key = str(venue_symbol or symbol_key).upper().strip()
        allocation = max(self._safe_float(allocation_usd), 0.0)
        mid = self._mid_price(venue_key or symbol_key, mode=mode)
        market = self._market(provider_key, symbol_key, venue_key)
        if mid <= 0 and market is not None:
            raw = market.raw if hasattr(market, "raw") else {}
            mid = self._safe_float(raw.get("mark") or raw.get("markPx") or raw.get("mid") or raw.get("lastTradePrice"))
        amount = allocation / mid if mid > 0 else 0.0
        precision = 8 if mid >= 1000 else 6
        return {
            "ok": True,
            "provider": provider_key,
            "symbol": symbol_key,
            "venue_symbol": venue_key,
            "allocation_usd": allocation,
            "mid": mid,
            "asset_amount": amount,
            "asset_amount_formatted": f"{amount:,.{precision}f}",
            "quote_asset": "USDT",
            "updated_at": self._utc_now(),
        }

    def parse_input(self, form: Any) -> SimulationInput:
        symbol = str(form.get("symbol", "BTC")).upper().strip()
        provider = normalize_provider(form.get("provider"), default="global")
        venue_symbol = str(form.get("venue_symbol") or symbol).upper().strip()
        timeframe = self.normalize_timeframe(str(form.get("timeframe", "live")))
        allocation = self._safe_float(form.get("allocation_amount_usd"), self.allocation_default_usd())
        cap = self.allocation_cap_usd()
        if not symbol:
            raise ValueError("Symbol is required.")
        if allocation <= 0:
            raise ValueError("Test allocation amount must be greater than zero.")
        if allocation > cap:
            raise ValueError(f"Test allocation amount cannot exceed ${cap:,.2f} paper funds.")
        return SimulationInput(
            provider=provider,
            symbol=symbol,
            venue_symbol=venue_symbol or symbol,
            timeframe=timeframe,
            allocation_usd=allocation,
            cycle="1h10",
        )

    def run(self, request_input: SimulationInput) -> dict[str, Any]:
        quote = self.quote_payload(
            provider=request_input.provider,
            symbol=request_input.symbol,
            venue_symbol=request_input.venue_symbol,
            allocation_usd=request_input.allocation_usd,
            mode="live",
        )
        market = self._market(request_input.provider, request_input.symbol, request_input.venue_symbol)
        candles = self._simulation_candles(request_input.venue_symbol or request_input.symbol, request_input.timeframe)
        if len(candles) < 30:
            raise RuntimeError("Backtest failed: insufficient market history for vault simulation.")
        book = self._order_book(request_input.venue_symbol or request_input.symbol, mode="live")
        profile = self._market_profile(market, candles, book, quote)
        auto = self._auto_controls(market, profile)
        strategy_results = self._strategy_results(request_input, candles, auto)
        weights = self._strategy_weights(strategy_results)
        combined = self._combine_results(request_input, strategy_results, weights, candles)
        chart = self._projection_chart(request_input, candles, profile)
        overlays = dict(chart.get("overlays") or {})

        result = {
            "vault_simulation": True,
            "summary": {
                "strategy": "Vault Ensemble Auto",
                "symbol": request_input.symbol,
                "venue_symbol": request_input.venue_symbol,
                "provider": request_input.provider,
                "provider_label": self._provider_label(request_input.provider),
                "timeframe": self._timeframe_label(request_input.timeframe),
                "duration": "1H10",
                "allocation": request_input.allocation_usd,
                "paper_balance": self.allocation_cap_usd(),
                "converted_amount": quote["asset_amount"],
                "converted_amount_formatted": quote["asset_amount_formatted"],
                "quote_asset": "USDT",
            },
            "metrics": combined["metrics"],
            "charts": {
                **combined["charts"],
                "candles": chart.get("candles") or self._chart_candles(candles),
                "liquidity_depth": self._liquidity_depth(book),
                "slippage_simulation": self._slippage_simulation(auto, profile),
            },
            "overlays": overlays,
            "autopilot": {
                "enabled": True,
                "status": "optimized",
                "confidence": auto["confidence"],
                "market_regime": profile["volatility_regime"],
                "strategy_count": len(strategy_results),
                "active_strategy_count": len([item for item in weights if item["enabled"]]),
                "objective": "risk-adjusted 1H10 return",
                "model_stack": ["ensemble_ranker", "execution_cost_model", "fibonacci_timing", "risk_allocator"],
            },
            "strategy_weights": weights,
            "execution_quality": {
                "auto_leverage": auto["leverage"],
                "max_exchange_leverage": auto["max_exchange_leverage"],
                "fee_bps": auto["fee_bps"],
                "slippage_bps": auto["slippage_bps"],
                "spread_bps": profile["spread_bps"],
                "liquidity_usd": profile["liquidity_usd"],
                "cost_drag_bps": auto["cost_drag_bps"],
                "fill_quality": profile["fill_quality"],
                "liquidity_quality": profile["liquidity_quality"],
                "liquidation_buffer_pct": auto["liquidation_buffer_pct"],
                "execution_style": "adaptive_market_limit",
            },
            "system_metrics": {
                "fees": "auto exchange maker/taker model",
                "slippage": "live depth and volatility adjusted",
                "exits": "dynamic volatility, trend, Fibonacci, and confidence exits",
                "sizing": auto["sizing_policy"],
                "cycle": "AI optimized high-frequency vault cycle",
            },
            "quote": quote,
            "market_profile": profile,
            "strategy_results": strategy_results,
            "generated_at": self._utc_now(),
        }
        return {
            "record": {
                "strategy_name": "vault_ensemble_auto",
                "symbol": request_input.symbol,
                "timeframe": request_input.timeframe,
            },
            "parameters": self._parameters(request_input, auto, weights, profile),
            "result": self._json_safe(result),
        }

    def response_payload(self, run: BacktestRun) -> dict[str, Any]:
        result = run.result if isinstance(run.result, dict) else {}
        if not result.get("vault_simulation"):
            return {"ok": False, "error": "Unsupported backtest payload."}
        payload = {
            "ok": True,
            "run_id": run.id,
            "summary": result.get("summary", {}),
            "metrics": result.get("metrics", {}),
            "charts": result.get("charts", {}),
            "overlays": result.get("overlays", {}),
            "autopilot": result.get("autopilot", {}),
            "strategy_weights": result.get("strategy_weights", []),
            "execution_quality": result.get("execution_quality", {}),
            "system_metrics": result.get("system_metrics", {}),
            "quote": result.get("quote", {}),
            "result": result,
        }
        return self._json_safe(payload)

    def normalize_timeframe(self, timeframe: str) -> str:
        value = str(timeframe or "live").strip().lower()
        aliases = {
            "": "live",
            "realtime": "live",
            "rt": "live",
            "4hr": "4h",
            "4hour": "4h",
            "4hours": "4h",
            "240m": "4h",
            "1day": "1d",
            "24h": "1d",
        }
        value = aliases.get(value, value)
        return value if value in _PUBLIC_TIMEFRAME_VALUES else "live"

    def engine_timeframe(self, public_timeframe: str) -> str:
        return {
            "live": "1m",
            "45m": "15m",
            "4h": "1h",
            "1d": "1h",
        }.get(self.normalize_timeframe(public_timeframe), self.normalize_timeframe(public_timeframe))

    def _parameters(
        self,
        request_input: SimulationInput,
        auto: dict[str, Any],
        weights: list[dict[str, Any]],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "initial_balance": request_input.allocation_usd,
            "allocation_amount_usd": request_input.allocation_usd,
            "fee_bps": auto["fee_bps"],
            "slippage_bps": auto["slippage_bps"],
            "stop_loss_pct": auto["stop_loss_pct"],
            "take_profit_pct": auto["take_profit_pct"],
            "sizing_mode": auto["sizing_policy"],
            "leverage": auto["leverage"],
            "parameters": {
                "sandbox_backtest": True,
                "simulated_capital_only": True,
                "paper_balance_usd": self.allocation_cap_usd(),
                "provider": request_input.provider,
                "venue_symbol": request_input.venue_symbol,
                "vault_cycle_duration": "1h10",
                "lock_duration_seconds": 70 * 60,
                "lock_duration_hours": 1,
                "one_h10_vault": True,
                "ml_horizon": "1h10",
                "allocation_cap_usd": request_input.allocation_usd,
                "auto_leverage": auto["leverage"],
                "auto_cost_model": {
                    "fee_bps": auto["fee_bps"],
                    "slippage_bps": auto["slippage_bps"],
                    "cost_drag_bps": auto["cost_drag_bps"],
                },
                "auto_exits": {
                    "stop_loss_pct": auto["stop_loss_pct"],
                    "take_profit_pct": auto["take_profit_pct"],
                    "policy": "volatility_trend_fibonacci_confidence",
                },
                "auto_sizing_policy": auto["sizing_policy"],
                "ensemble": {
                    "strategy_weights": weights,
                    "market_regime": profile["volatility_regime"],
                    "confidence": auto["confidence"],
                },
            },
        }

    def _symbol_rows(self, *, user: Any | None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        market_rows = self._active_markets(user=user)
        seen: set[tuple[str, str, str]] = set()
        for market in market_rows:
            provider = normalize_provider(getattr(market, "provider", ""), default="global")
            symbol = str(getattr(market, "symbol", "") or "").upper()
            venue_symbol = str(getattr(market, "venue_symbol", symbol) or symbol).upper()
            if not symbol:
                continue
            key = (provider, symbol, venue_symbol)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "provider": provider,
                    "provider_label": self._provider_label(provider),
                    "symbol": symbol,
                    "venue_symbol": venue_symbol,
                    "settlement_asset": str(getattr(market, "settlement_asset", provider_collateral_asset(provider)) or "").upper(),
                    "max_leverage": self._safe_float(getattr(market, "max_leverage", 1.0), 1.0),
                    "liquidity_usd": self._safe_float(getattr(market, "liquidity_usd", 0.0)),
                    "spread_bps": self._safe_float(getattr(market, "spread_bps", 0.0)),
                    "fee_bps": self._safe_float(getattr(market, "fee_bps", self.config.get("FEE_BPS", 5.0))),
                    "compatibility_badges": [self._provider_label(provider), str(getattr(market, "settlement_asset", provider_collateral_asset(provider)) or "").upper()],
                    "category": "Connected markets",
                    "token_icon": symbol[:1],
                    "favorite": False,
                    "recent": self._recent_symbol(symbol),
                }
            )
        if rows:
            rows.sort(
                key=lambda item: (
                    not bool(item.get("recent")),
                    -self._safe_float(item.get("liquidity_usd")),
                    str(item.get("symbol")),
                    str(item.get("provider")),
                )
            )
            return rows
        for symbol in self.config.get("ALLOWED_SYMBOLS", ["BTC", "ETH", "SOL"]):
            symbol_key = str(symbol or "").upper().strip()
            if not symbol_key:
                continue
            rows.append(
                {
                    "provider": "global",
                    "provider_label": "Configured",
                    "symbol": symbol_key,
                    "venue_symbol": symbol_key,
                    "settlement_asset": "USDT",
                    "max_leverage": self._safe_float(self.config.get("MAX_LEVERAGE", 1.0), 1.0),
                    "liquidity_usd": 0.0,
                    "spread_bps": 0.0,
                    "fee_bps": self._safe_float(self.config.get("FEE_BPS", 5.0), 5.0),
                    "compatibility_badges": ["Configured"],
                    "category": "Configured assets",
                    "token_icon": symbol_key[:1],
                    "favorite": symbol_key in {"BTC", "ETH"},
                    "recent": self._recent_symbol(symbol_key),
                }
            )
        return rows

    def _active_markets(self, *, user: Any | None) -> list[LeveragedMarket]:
        if self.leveraged_markets is None:
            return []
        try:
            markets = list(self.leveraged_markets.active_markets())
        except Exception:
            return []
        if user is None or self.trading_connections is None:
            return markets
        try:
            connections = list(self.trading_connections.verified_tradable_connections(user.id))
        except Exception:
            return markets
        connection_ids = {int(connection.id) for connection in connections if getattr(connection, "id", None)}
        providers = {normalize_provider(getattr(connection, "provider", ""), default="global") for connection in connections}
        scoped = [
            market
            for market in markets
            if int(getattr(market, "trading_connection_id", 0) or 0) in connection_ids
            or normalize_provider(getattr(market, "provider", ""), default="global") in providers
        ]
        return scoped or markets

    def _market(self, provider: str, symbol: str, venue_symbol: str = "") -> LeveragedMarket | None:
        if self.leveraged_markets is None:
            return None
        provider_key = normalize_provider(provider, default="global")
        symbol_key = str(symbol or "").upper()
        venue_key = str(venue_symbol or "").upper()
        try:
            candidates = list(self.leveraged_markets.active_markets(provider=None if provider_key == "global" else provider_key))
        except Exception:
            return None
        for market in candidates:
            market_symbol = str(getattr(market, "symbol", "") or "").upper()
            market_venue = str(getattr(market, "venue_symbol", "") or "").upper()
            if venue_key and market_venue == venue_key:
                return market
            if symbol_key and market_symbol == symbol_key:
                return market
        return None

    def _simulation_candles(self, symbol: str, public_timeframe: str) -> list[dict[str, Any]]:
        public = self.normalize_timeframe(public_timeframe)
        if public == "45m":
            source = self._safe_candles(symbol, "15m", limit=210)
            return self._aggregate_candles(source, group_size=3)[-120:]
        if public == "1d":
            source = self._safe_candles(symbol, "4h", limit=240)
            return self._aggregate_candles(source, group_size=6)[-120:]
        source = {"live": "1m"}.get(public, public)
        return self._safe_candles(symbol, source, limit=180)[-160:]

    def _safe_candles(self, symbol: str, timeframe: str, *, limit: int) -> list[dict[str, Any]]:
        try:
            rows = list(self.market_data.get_candles(symbol, timeframe, mode="live", limit=limit) or [])
        except Exception:
            try:
                rows = list(self.market_data.get_candles(symbol, timeframe, mode="testnet", limit=limit) or [])
            except Exception:
                rows = []
        return [self._normalize_candle(row, index) for index, row in enumerate(rows) if isinstance(row, dict)]

    def _aggregate_candles(self, candles: list[dict[str, Any]], *, group_size: int) -> list[dict[str, Any]]:
        aggregated: list[dict[str, Any]] = []
        for start in range(0, len(candles), max(1, int(group_size))):
            group = candles[start : start + max(1, int(group_size))]
            if not group:
                continue
            aggregated.append(
                {
                    "timestamp": int(group[-1]["timestamp"]),
                    "open": group[0]["open"],
                    "high": max(row["high"] for row in group),
                    "low": min(row["low"] for row in group),
                    "close": group[-1]["close"],
                    "volume": sum(row.get("volume", 0.0) for row in group),
                }
            )
        return aggregated

    def _normalize_candle(self, row: dict[str, Any], index: int) -> dict[str, Any]:
        close = self._safe_float(row.get("close", row.get("c", row.get("price", 0.0))))
        timestamp = row.get("timestamp", row.get("time", row.get("t", index)))
        timestamp_value = self._safe_float(timestamp, index)
        if timestamp_value > 10_000_000_000:
            timestamp_value /= 1000.0
        return {
            "timestamp": int(timestamp_value),
            "open": self._safe_float(row.get("open", row.get("o", close)), close),
            "high": self._safe_float(row.get("high", row.get("h", close)), close),
            "low": self._safe_float(row.get("low", row.get("l", close)), close),
            "close": close,
            "volume": self._safe_float(row.get("volume", row.get("v", 0.0))),
        }

    def _market_profile(
        self,
        market: LeveragedMarket | None,
        candles: list[dict[str, Any]],
        book: dict[str, Any],
        quote: dict[str, Any],
    ) -> dict[str, Any]:
        mid = self._safe_float(quote.get("mid")) or self._safe_float(candles[-1].get("close") if candles else 0.0)
        spread = self._safe_float(getattr(market, "spread_bps", 0.0) if market is not None else 0.0)
        if spread <= 0:
            spread = spread_bps(book, mid)
        liquidity = self._safe_float(getattr(market, "liquidity_usd", 0.0) if market is not None else 0.0)
        if liquidity <= 0:
            liquidity = book_liquidity_usd(book, depth=max(1, int(self.config.get("VAULT_BOOK_DEPTH_LEVELS", 5) or 5)))
        vol = volatility_pct(candles)
        regime, regime_score = volatility_regime(vol)
        min_liquidity = float(self.config.get("UNIVERSE_MIN_LIQUIDITY_USD", 25_000.0) or 25_000.0)
        max_spread = float(self.config.get("UNIVERSE_MAX_SPREAD_BPS", 15.0) or 15.0)
        liquidity_quality = min(liquidity / max(min_liquidity * 4, 1.0), 1.0)
        spread_quality = max(0.0, 1.0 - spread / max(max_spread, 1.0))
        fill_quality = max(0.0, min((liquidity_quality * 0.45) + (spread_quality * 0.35) + (regime_score * 0.20), 1.0))
        return {
            "mid": mid,
            "spread_bps": spread,
            "liquidity_usd": liquidity,
            "volatility_pct": vol,
            "volatility_regime": regime,
            "volatility_score": regime_score,
            "liquidity_quality": liquidity_quality,
            "spread_quality": spread_quality,
            "fill_quality": fill_quality,
        }

    def _auto_controls(self, market: LeveragedMarket | None, profile: dict[str, Any]) -> dict[str, Any]:
        fee_bps = self._safe_float(getattr(market, "fee_bps", self.config.get("FEE_BPS", 5.0)) if market is not None else self.config.get("FEE_BPS", 5.0), 5.0)
        max_exchange = self._safe_float(getattr(market, "max_leverage", self.config.get("MAX_LEVERAGE", 1.0)) if market is not None else self.config.get("MAX_LEVERAGE", 1.0), 1.0)
        configured_max = min(
            max_exchange,
            self._safe_float(self.config.get("MAX_LEVERAGE", max_exchange), max_exchange),
            self._safe_float(self.config.get("ONE_H10_MAX_LEVERAGE", max_exchange), max_exchange),
        )
        confidence = max(0.05, min(profile["fill_quality"] * 0.72 + profile["volatility_score"] * 0.28, 1.0))
        volatility_drag = 1.0 + max(profile["volatility_pct"], 0.0) / 2.5
        leverage = max(1.0, min(configured_max, configured_max * (0.35 + confidence * 0.65) / volatility_drag))
        slippage = (
            self._safe_float(self.config.get("SIM_SLIPPAGE_BPS", 8.0), 8.0) * 0.45
            + profile["spread_bps"] * 0.35
            + profile["volatility_pct"] * 3.5
            + (1.0 - profile["liquidity_quality"]) * 5.0
        )
        stop = min(max((profile["volatility_pct"] / 100.0) * 1.6, 0.004), 0.08)
        take = min(max(stop * (1.45 + confidence), 0.008), 0.16)
        cost_drag = cost_drag_bps(spread=profile["spread_bps"], fee_bps=fee_bps, slippage_bps=slippage)
        return {
            "confidence": confidence,
            "fee_bps": max(fee_bps, 0.0),
            "slippage_bps": max(slippage, 0.0),
            "cost_drag_bps": cost_drag,
            "leverage": round(leverage, 3),
            "max_exchange_leverage": max_exchange,
            "stop_loss_pct": stop,
            "take_profit_pct": take,
            "risk_per_trade_pct": min(max(0.006 + confidence * 0.018, 0.006), 0.03),
            "liquidation_buffer_pct": max(0.05, self._safe_float(self.config.get("MIN_LIQUIDATION_BUFFER_PCT", 0.015), 0.015) + (1.0 - confidence) * 0.12),
            "sizing_policy": "ml_volatility_risk_weighted",
        }

    def _strategy_results(
        self,
        request_input: SimulationInput,
        candles: list[dict[str, Any]],
        auto: dict[str, Any],
    ) -> list[dict[str, Any]]:
        engine_timeframe = self.engine_timeframe(request_input.timeframe)
        available = set(self.registry.names())
        strategy_names = [name for name in AUTO_STRATEGIES if name in available] or sorted(available)
        results: list[dict[str, Any]] = []
        for strategy_name in strategy_names:
            parameters = dict(self.registry.definition(strategy_name).get("parameters") or {})
            parameters.update(
                {
                    "stop_loss_pct": auto["stop_loss_pct"],
                    "take_profit_pct": auto["take_profit_pct"],
                    "risk_fraction": min(auto["risk_per_trade_pct"] * 4.0, 0.12),
                    "leverage": auto["leverage"],
                    "one_h10_vault": True,
                    "ml_horizon": "1h10",
                }
            )
            config = BacktestConfig(
                strategy_name=strategy_name,
                symbol=request_input.venue_symbol or request_input.symbol,
                timeframe=engine_timeframe,
                mode="live",
                initial_balance=request_input.allocation_usd,
                fee_bps=auto["fee_bps"],
                slippage_bps=auto["slippage_bps"],
                stop_loss_pct=auto["stop_loss_pct"],
                take_profit_pct=auto["take_profit_pct"],
                position_size_fraction=1.0,
                parameters=parameters,
                sizing_mode="risk_based",
                fixed_dollar_size=request_input.allocation_usd,
                risk_per_trade_pct=auto["risk_per_trade_pct"],
                max_daily_loss=self._safe_float(self.config.get("MAX_DAILY_LOSS_USDC", 100.0), 100.0),
                max_drawdown_pct=self._safe_float(self.config.get("MAX_BACKTEST_DRAWDOWN_PCT", 0.2), 0.2),
                loss_streak_cooldown=int(self.config.get("LOSS_STREAK_COOLDOWN_THRESHOLD", 3) or 3),
                cooldown_minutes=int(self.config.get("LOSS_COOLDOWN_MINUTES", 30) or 30),
                max_trades_per_window=int(self.config.get("MAX_TRADES_PER_WINDOW", 5) or 5),
                trade_window_minutes=int(self.config.get("TRADE_WINDOW_MINUTES", 60) or 60),
                intrabar_model="conservative",
                allocation_amount_usd=request_input.allocation_usd,
                leverage=auto["leverage"],
                min_liquidation_buffer_pct=auto["liquidation_buffer_pct"],
                funding_cost_bps=self._safe_float(self.config.get("FUNDING_COST_BPS", 0.0), 0.0),
            )
            try:
                result = dict(self.backtest_engine.run(config, list(candles)))
                error = ""
            except Exception as exc:  # noqa: BLE001
                result = {}
                error = str(exc)
            score = self._strategy_score(result, auto)
            results.append(
                {
                    "strategy_name": strategy_name,
                    "label": self._strategy_label(strategy_name),
                    "score": score,
                    "result": result,
                    "error": error,
                    "total_return": self._safe_float(result.get("total_return")) if result else 0.0,
                    "max_drawdown": self._safe_float(result.get("max_drawdown")) if result else 0.0,
                    "win_rate": self._safe_float(result.get("win_rate")) if result else 0.0,
                    "trade_count": int(result.get("trade_count") or 0) if result else 0,
                }
            )
        return results

    def _strategy_score(self, result: dict[str, Any], auto: dict[str, Any]) -> float:
        if not result:
            return -1.0
        total_return = self._safe_float(result.get("total_return"))
        drawdown = abs(self._safe_float(result.get("max_drawdown")))
        win_rate = self._safe_float(result.get("win_rate"))
        trades = min(int(result.get("trade_count") or 0), 30) / 30.0
        cost_penalty = self._safe_float(auto.get("cost_drag_bps")) / 10_000.0
        return total_return - drawdown * 0.45 + win_rate * 0.08 + trades * 0.05 - cost_penalty

    def _strategy_weights(self, strategy_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ordered = sorted(strategy_results, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        active_names = {item["strategy_name"] for item in ordered[:4] if not item.get("error")}
        if not active_names and ordered:
            active_names = {ordered[0]["strategy_name"]}
        raw_weights = {
            item["strategy_name"]: max(float(item.get("score", 0.0)) + 0.05, 0.01)
            for item in ordered
            if item["strategy_name"] in active_names
        }
        total = sum(raw_weights.values()) or 1.0
        payload: list[dict[str, Any]] = []
        for item in strategy_results:
            enabled = item["strategy_name"] in active_names
            reason = ""
            if not enabled:
                reason = item.get("error") or "weaker_risk_adjusted_edge"
            payload.append(
                {
                    "strategy_name": item["strategy_name"],
                    "label": item["label"],
                    "enabled": enabled,
                    "weight": (raw_weights.get(item["strategy_name"], 0.0) / total) if enabled else 0.0,
                    "score": item["score"],
                    "total_return": item["total_return"],
                    "win_rate": item["win_rate"],
                    "trade_count": item["trade_count"],
                    "disabled_reason": reason,
                }
            )
        return payload

    def _combine_results(
        self,
        request_input: SimulationInput,
        strategy_results: list[dict[str, Any]],
        weights: list[dict[str, Any]],
        candles: list[dict[str, Any]],
    ) -> dict[str, Any]:
        weight_by_strategy = {item["strategy_name"]: float(item.get("weight", 0.0) or 0.0) for item in weights if item.get("enabled")}
        active = [item for item in strategy_results if weight_by_strategy.get(item["strategy_name"], 0.0) > 0]
        if not active:
            equity_curve = self._flat_equity_curve(request_input.allocation_usd, candles)
            drawdown_curve = [{"timestamp": row["timestamp"], "drawdown": 0.0} for row in equity_curve]
            trades: list[dict[str, Any]] = []
            fees = 0.0
        else:
            equity_curve = self._weighted_equity_curve(request_input.allocation_usd, active, weight_by_strategy, candles)
            drawdown_curve = self._drawdown_curve(equity_curve)
            trades = []
            fees = 0.0
            for item in active:
                weight = weight_by_strategy[item["strategy_name"]]
                result = item.get("result") or {}
                fees += self._safe_float(result.get("fees_paid")) * weight
                trades.extend(result.get("trades") if isinstance(result.get("trades"), list) else [])
        final_equity = self._safe_float(equity_curve[-1].get("equity"), request_input.allocation_usd) if equity_curve else request_input.allocation_usd
        total_return = (final_equity - request_input.allocation_usd) / max(request_input.allocation_usd, 1e-9)
        max_drawdown = min([self._safe_float(row.get("drawdown")) for row in drawdown_curve] or [0.0])
        wins = len([trade for trade in trades if self._safe_float(trade.get("pnl")) > 0])
        trade_count = len(trades)
        win_rate = wins / trade_count if trade_count else 0.0
        charts = self._charts_from_curves(equity_curve, drawdown_curve, request_input.allocation_usd, trades)
        return {
            "metrics": {
                "roi": total_return,
                "pnl": final_equity - request_input.allocation_usd,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "trades": trade_count,
                "fees": fees,
                "ending_balance": final_equity,
                "profit_factor": self._profit_factor(trades),
            },
            "charts": charts,
        }

    def _weighted_equity_curve(
        self,
        allocation: float,
        active: list[dict[str, Any]],
        weight_by_strategy: dict[str, float],
        candles: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        curves: list[tuple[float, list[dict[str, Any]]]] = []
        for item in active:
            result = item.get("result") or {}
            curve = result.get("equity_curve") if isinstance(result.get("equity_curve"), list) else []
            if curve:
                curves.append((weight_by_strategy[item["strategy_name"]], curve))
        if not curves:
            return self._flat_equity_curve(allocation, candles)
        length = max(len(curve) for _, curve in curves)
        combined: list[dict[str, Any]] = []
        for index in range(length):
            timestamp = 0
            equity_delta = 0.0
            for weight, curve in curves:
                point = curve[min(index, len(curve) - 1)]
                timestamp = int(self._safe_float(point.get("timestamp"), timestamp))
                equity_delta += (self._safe_float(point.get("equity"), allocation) - allocation) * weight
            combined.append({"timestamp": timestamp or index, "equity": allocation + equity_delta})
        return combined

    def _flat_equity_curve(self, allocation: float, candles: list[dict[str, Any]]) -> list[dict[str, Any]]:
        source = candles[-60:] if candles else [{"timestamp": int(time.time())}]
        return [{"timestamp": int(row.get("timestamp", index)), "equity": allocation} for index, row in enumerate(source)]

    def _drawdown_curve(self, equity_curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
        peak = 0.0
        rows: list[dict[str, Any]] = []
        for point in equity_curve:
            equity = self._safe_float(point.get("equity"))
            peak = max(peak, equity)
            drawdown = (equity - peak) / max(peak, 1e-9)
            rows.append({"timestamp": int(self._safe_float(point.get("timestamp"))), "drawdown": drawdown})
        return rows

    def _charts_from_curves(
        self,
        equity_curve: list[dict[str, Any]],
        drawdown_curve: list[dict[str, Any]],
        allocation: float,
        trades: list[dict[str, Any]],
    ) -> dict[str, Any]:
        equity = [{"x": self._safe_float(row.get("timestamp")), "y": self._safe_float(row.get("equity"), allocation)} for row in equity_curve]
        pnl = [{"x": row["x"], "y": row["y"] - allocation} for row in equity]
        growth = [{"x": row["x"], "y": (row["y"] - allocation) / max(allocation, 1e-9)} for row in equity]
        drawdown = [{"x": self._safe_float(row.get("timestamp")), "y": self._safe_float(row.get("drawdown"))} for row in drawdown_curve]
        wins = len([trade for trade in trades if self._safe_float(trade.get("pnl")) > 0])
        losses = len([trade for trade in trades if self._safe_float(trade.get("pnl")) < 0])
        flat = max(0, len(trades) - wins - losses)
        return {
            "equity": self._downsample(equity),
            "pnl": self._downsample(pnl),
            "drawdown": self._downsample(drawdown),
            "growth": self._downsample(growth),
            "profit_curve": self._downsample(pnl),
            "win_loss": {"wins": wins, "losses": losses, "flat": flat},
            "trade_distribution": self._trade_distribution(trades),
        }

    def _projection_chart(
        self,
        request_input: SimulationInput,
        candles: list[dict[str, Any]],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        fallback = {"candles": self._chart_candles(candles), "overlays": self._fallback_overlays(candles, profile)}
        if self.ml_projection_engine is None:
            return fallback
        features = {
            "symbol": request_input.symbol,
            "provider": request_input.provider,
            "close": profile["mid"],
            "atr_pct": max(profile["volatility_pct"] / 100.0, 0.002),
            "volatility": max(profile["volatility_pct"] / 100.0, 0.002),
            "spread_bps": profile["spread_bps"],
            "liquidity_usd": profile["liquidity_usd"],
            "ml_horizon": "1h10",
            "one_h10_vault": True,
        }
        forecast = {}
        try:
            forecast = dict(
                self.ml_projection_engine.forecast_from_features(
                    features,
                    provider=request_input.provider,
                    symbol=request_input.symbol,
                    allocation_cap_usd=request_input.allocation_usd,
                    available_margin_usd=request_input.allocation_usd,
                    market=self._market(request_input.provider, request_input.symbol, request_input.venue_symbol),
                )
            )
        except Exception:
            forecast = {}
        try:
            return dict(
                self.ml_projection_engine.chart_payload(
                    provider=request_input.provider,
                    symbol=request_input.symbol,
                    venue_symbol=request_input.venue_symbol,
                    mode="live",
                    timeframe=request_input.timeframe,
                    forecast=forecast,
                    features=features,
                )
            )
        except Exception:
            return fallback

    def _fallback_overlays(self, candles: list[dict[str, Any]], profile: dict[str, Any]) -> dict[str, Any]:
        if not candles:
            return {}
        last = candles[-1]
        price = self._safe_float(last.get("close"), profile.get("mid", 1.0))
        timestamp = int(self._safe_float(last.get("timestamp"), time.time()))
        step = 70 * 60 / 8
        volatility = max(profile["volatility_pct"] / 100.0, 0.002)
        path = []
        upper = []
        lower = []
        for index in range(1, 9):
            point_time = timestamp + step * index
            value = price * (1 + volatility * math.sin(index / 8 * math.pi) * 0.2)
            band = price * volatility * (0.8 + index / 8)
            path.append({"time": point_time, "value": value})
            upper.append({"time": point_time, "value": value + band})
            lower.append({"time": point_time, "value": max(value - band, 0.0)})
        return {
            "path": path,
            "confidence_band": {"upper": upper, "lower": lower},
            "zones": {
                "entry": {"price": price},
                "exit": {"price": price * (1 + volatility * 2)},
                "stop_loss": {"price": price * (1 - volatility * 1.5)},
            },
            "fibonacci_time_zones": [
                {"time": float(timestamp + 70 * 60 * ratio), "ratio": ratio}
                for ratio in (0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
            ],
        }

    def _order_book(self, symbol: str, *, mode: str) -> dict[str, Any]:
        try:
            return dict(self.market_data.get_order_book(symbol, mode) or {})
        except Exception:
            return {}

    def _mid_price(self, symbol: str, *, mode: str) -> float:
        try:
            return self._safe_float(self.market_data.get_mid_price(symbol, mode))
        except Exception:
            try:
                return self._safe_float(self.market_data.get_mid_price(symbol, "testnet"))
            except Exception:
                return 0.0

    def _chart_candles(self, candles: list[dict[str, Any]]) -> list[dict[str, float]]:
        return [
            {
                "time": self._safe_float(row.get("timestamp")),
                "open": self._safe_float(row.get("open")),
                "high": self._safe_float(row.get("high")),
                "low": self._safe_float(row.get("low")),
                "close": self._safe_float(row.get("close")),
            }
            for row in candles[-150:]
        ]

    def _liquidity_depth(self, book: dict[str, Any]) -> list[dict[str, float]]:
        levels = book.get("levels", []) if isinstance(book, dict) else []
        rows: list[dict[str, float]] = []
        for side_index, side in enumerate(levels[:2] if isinstance(levels, list) else []):
            side_name = "bid" if side_index == 0 else "ask"
            cumulative = 0.0
            for level in (side or [])[:12]:
                price = self._safe_float(level.get("px", level.get("price")) if isinstance(level, dict) else level[0] if isinstance(level, (list, tuple)) and level else 0.0)
                size = self._safe_float(level.get("sz", level.get("size")) if isinstance(level, dict) else level[1] if isinstance(level, (list, tuple)) and len(level) > 1 else 0.0)
                cumulative += price * size
                rows.append({"side": side_name, "price": price, "notional": cumulative})
        return rows

    def _slippage_simulation(self, auto: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, float]]:
        base = self._safe_float(auto.get("slippage_bps"))
        liquidity_drag = (1.0 - self._safe_float(profile.get("liquidity_quality"))) * 4.0
        return [
            {"size_pct": pct, "bps": base + liquidity_drag * (pct / 100.0) ** 1.25}
            for pct in (10, 25, 50, 75, 100)
        ]

    def _trade_distribution(self, trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
        buckets = [
            {"label": "< -2%", "min": -math.inf, "max": -0.02, "count": 0},
            {"label": "-2% to 0", "min": -0.02, "max": 0.0, "count": 0},
            {"label": "0 to 2%", "min": 0.0, "max": 0.02, "count": 0},
            {"label": "> 2%", "min": 0.02, "max": math.inf, "count": 0},
        ]
        for trade in trades:
            value = self._safe_float(trade.get("return"))
            for bucket in buckets:
                if bucket["min"] <= value < bucket["max"]:
                    bucket["count"] += 1
                    break
        return [{"label": bucket["label"], "count": bucket["count"]} for bucket in buckets]

    def _profit_factor(self, trades: list[dict[str, Any]]) -> float:
        wins = sum(max(self._safe_float(trade.get("pnl")), 0.0) for trade in trades)
        losses = abs(sum(min(self._safe_float(trade.get("pnl")), 0.0) for trade in trades))
        return wins / losses if losses > 0 else (wins if wins > 0 else 0.0)

    def _downsample(self, series: list[dict[str, float]]) -> list[dict[str, float]]:
        max_points = max(12, int(self.config.get("BACKTEST_MAX_CHART_POINTS", 240) or 240))
        if len(series) <= max_points:
            return series
        step = math.ceil(len(series) / max_points)
        sampled = series[::step]
        if sampled[-1] != series[-1]:
            sampled.append(series[-1])
        return sampled

    def _recent_symbol(self, symbol: str) -> bool:
        try:
            return BacktestRun.query.filter_by(symbol=str(symbol).upper()).count() > 0
        except Exception:
            return False

    def _timeframe_label(self, timeframe: str) -> str:
        normalized = self.normalize_timeframe(timeframe)
        for item in PUBLIC_TIMEFRAMES:
            if item["value"] == normalized:
                return item["label"]
        return normalized.upper()

    @staticmethod
    def _provider_label(provider: str) -> str:
        provider_key = normalize_provider(provider, default="global")
        return {
            "hyperliquid": "Hyperliquid",
            "kucoin": "KuCoin",
            "global": "Configured",
        }.get(provider_key, provider_key.replace("_", " ").title())

    @staticmethod
    def _strategy_label(strategy_name: str) -> str:
        return str(strategy_name or "").replace("_", " ").title()

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            try:
                parsed = float(default)
            except (TypeError, ValueError):
                return 0.0
        return parsed if math.isfinite(parsed) else float(default or 0.0)

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        return value

    @staticmethod
    def _utc_now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
