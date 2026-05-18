"""Exchange-level allocation scoring for broader Vault Cycle automation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..models import LeveragedMarket, TradingConnection
from .provider_assets import normalize_provider, provider_collateral_asset


@dataclass(frozen=True, slots=True)
class VaultCycleAllocationPlan:
    connection: TradingConnection
    provider: str
    settlement_asset: str
    collateral_asset: str
    target_amount: float
    allocation_weight: float
    scores: dict[str, Any]
    constraints: dict[str, Any]


class VaultCycleAllocator:
    """Ranks enabled exchanges and produces capped provider allocations."""

    def __init__(self, config: dict[str, Any], trading_connections: Any, leveraged_markets: Any, market_scanner: Any | None = None) -> None:
        self.config = config
        self.trading_connections = trading_connections
        self.leveraged_markets = leveraged_markets
        self.market_scanner = market_scanner

    def allocate(
        self,
        *,
        user_id: int,
        amount_usd: float,
        settlement_asset: str,
        connections: list[TradingConnection],
        allowed_symbols: list[str] | None = None,
        provider_filter: list[str] | None = None,
    ) -> tuple[list[VaultCycleAllocationPlan], list[dict[str, Any]]]:
        amount = max(0.0, float(amount_usd or 0.0))
        settlement = str(settlement_asset or "").upper().strip()
        supported = {normalize_provider(provider) for provider in (provider_filter or self.config.get("VAULT_CYCLE_PROVIDERS", []))}
        min_allocation = max(0.0, float(self.config.get("VAULT_CYCLE_MIN_EXCHANGE_ALLOCATION_USD", 5.0) or 0.0))
        max_exchange_pct = self._bounded_pct(self.config.get("VAULT_CYCLE_MAX_EXCHANGE_ALLOCATION_PCT", 0.80), 0.80)
        max_symbol_pct = self._bounded_pct(self.config.get("VAULT_CYCLE_MAX_SYMBOL_ALLOCATION_PCT", 0.50), 0.50)
        max_positions = max(1, int(self.config.get("VAULT_CYCLE_MAX_CONCURRENT_POSITIONS", 3) or 3))

        candidates: list[dict[str, Any]] = []
        blockers: list[dict[str, Any]] = []
        for connection in connections:
            provider = normalize_provider(connection.provider)
            if supported and provider not in supported:
                blockers.append({"provider": provider, "reason": "provider_not_enabled_for_vault_cycle"})
                continue
            collateral = provider_collateral_asset(provider)
            conversion = self._conversion_policy(provider, collateral, settlement)
            if not conversion["supported"]:
                blockers.append({"provider": provider, "reason": conversion["reason"], "collateral_asset": collateral})
                continue
            try:
                snapshot = self.trading_connections.account_snapshot(user_id, "live", connection.id)
            except Exception as exc:  # noqa: BLE001
                blockers.append({"provider": provider, "reason": f"account_snapshot_failed: {exc}"})
                continue
            alerts = [str(alert) for alert in getattr(snapshot, "alerts", []) or [] if str(alert).strip()]
            if alerts:
                blockers.append({"provider": provider, "reason": "; ".join(alerts[:2])})
                continue
            available = self._balance_available(getattr(snapshot, "balances", []) or [], collateral)
            if bool(self.config.get("VAULT_CYCLE_REQUIRE_EXCHANGE_RESERVE", True)) and available + 1e-9 < min_allocation:
                blockers.append({"provider": provider, "reason": f"insufficient_{collateral.lower()}_exchange_reserve", "available": available})
                continue
            market_score = self._market_score(provider, allowed_symbols)
            reliability = self._provider_reliability(connection)
            transfer_cost = self._transfer_cost_penalty(provider, collateral, settlement)
            reserve_score = min(max(available / max(amount, min_allocation, 1.0), 0.0), 1.25)
            risk_adjusted = max(
                0.0,
                (market_score["opportunity_score"] * 0.38)
                + (market_score["liquidity_score"] * 0.18)
                + (market_score["spread_quality_score"] * 0.14)
                + (market_score["funding_score"] * 0.08)
                + (market_score["structure_score"] * 0.08)
                + (reliability * 0.08)
                + (reserve_score * 0.06)
                - transfer_cost,
            )
            if risk_adjusted <= 0:
                blockers.append({"provider": provider, "reason": "risk_adjusted_score_non_positive"})
                continue
            candidates.append(
                {
                    "connection": connection,
                    "provider": provider,
                    "settlement_asset": settlement,
                    "collateral_asset": collateral,
                    "available": available,
                    "risk_adjusted_score": risk_adjusted,
                    "scores": {
                        **market_score,
                        "exchange_reserve_score": reserve_score,
                        "exchange_reliability_score": reliability,
                        "transfer_cost_penalty": transfer_cost,
                        "risk_adjusted_score": risk_adjusted,
                        "conversion_required": collateral != settlement,
                        "conversion_supported": conversion["supported"],
                        "conversion_from": settlement if collateral != settlement else "",
                        "conversion_to": collateral if collateral != settlement else "",
                        "conversion_status": "planned" if collateral != settlement else "not_required",
                    },
                    "constraints": {
                        "max_exchange_allocation_pct": max_exchange_pct,
                        "max_symbol_allocation_pct": max_symbol_pct,
                        "max_concurrent_positions": max_positions,
                        "min_exchange_allocation_usd": min_allocation,
                        "available_exchange_reserve": available,
                    },
                }
            )

        if not candidates:
            return [], blockers

        total_score = sum(float(item["risk_adjusted_score"]) for item in candidates)
        cap_amount = amount * max_exchange_pct if max_exchange_pct > 0 else amount
        raw_allocations: list[dict[str, Any]] = []
        for item in candidates:
            share = float(item["risk_adjusted_score"]) / max(total_score, 1e-9)
            target = min(amount * share, cap_amount)
            if bool(self.config.get("VAULT_CYCLE_REQUIRE_EXCHANGE_RESERVE", True)):
                target = min(target, float(item["available"]))
            raw_allocations.append({**item, "target_amount": target})

        raw_allocations = [item for item in raw_allocations if float(item["target_amount"]) + 1e-9 >= min_allocation]
        if not raw_allocations:
            blockers.append({"provider": "all", "reason": "all_allocations_below_minimum"})
            return [], blockers

        allocated_total = sum(float(item["target_amount"]) for item in raw_allocations)
        plans = []
        for item in raw_allocations:
            target_amount = float(item["target_amount"])
            scores = dict(item["scores"])
            if bool(scores.get("conversion_required")):
                scores["conversion_amount"] = target_amount
            plans.append(
                VaultCycleAllocationPlan(
                    connection=item["connection"],
                    provider=item["provider"],
                    settlement_asset=item["settlement_asset"],
                    collateral_asset=item["collateral_asset"],
                    target_amount=target_amount,
                    allocation_weight=target_amount / max(allocated_total, 1e-9),
                    scores=scores,
                    constraints=item["constraints"],
                )
            )
        return plans, blockers

    def _market_score(self, provider: str, allowed_symbols: list[str] | None) -> dict[str, float]:
        markets = self.leveraged_markets.active_markets(provider=provider, symbols=allowed_symbols) if self.leveraged_markets else []
        if not markets:
            return {
                "opportunity_score": 0.55,
                "liquidity_score": 0.45,
                "spread_quality_score": 0.45,
                "funding_score": 0.50,
                "structure_score": 0.50,
                "ml_rank_score": 0.0,
                "strategy_suitability_score": 0.50,
                "best_symbol": "",
            }
        best = max(markets, key=self._market_opportunity)
        liquidity = min(math.log10(max(float(best.liquidity_usd or 0.0), 1.0)) / 8.0, 1.0)
        spread = 1.0 - min(max(float(best.spread_bps or 0.0), 0.0) / max(float(self.config.get("VAULT_MAX_SPREAD_BPS", 25.0) or 25.0), 1.0), 1.0)
        funding = 1.0 - min(abs(float(best.funding_rate or 0.0)) * 100.0, 1.0)
        raw = best.raw if hasattr(best, "raw") else {}
        ml_score = self._safe_float(raw.get("ml_score") or raw.get("rank_score"))
        structure = self._safe_float(raw.get("market_structure_score"), 0.55)
        opportunity = self._market_opportunity(best)
        return {
            "opportunity_score": opportunity,
            "liquidity_score": liquidity,
            "spread_quality_score": max(0.0, spread),
            "funding_score": max(0.0, funding),
            "structure_score": max(0.0, min(structure, 1.0)),
            "ml_rank_score": max(0.0, min(ml_score, 1.0)),
            "strategy_suitability_score": max(0.0, min((opportunity + structure) / 2.0, 1.0)),
            "best_symbol": str(best.symbol or ""),
            "best_market_id": int(best.id or 0),
            "best_venue_symbol": str(best.venue_symbol or ""),
            "spread_bps": float(best.spread_bps or 0.0),
            "funding_rate": float(best.funding_rate or 0.0),
            "liquidity_usd": float(best.liquidity_usd or 0.0),
            "max_leverage": float(best.max_leverage or 1.0),
        }

    def _market_opportunity(self, market: LeveragedMarket) -> float:
        liquidity = min(math.log10(max(float(market.liquidity_usd or 0.0), 1.0)) / 8.0, 1.0)
        spread_penalty = min(max(float(market.spread_bps or 0.0), 0.0) / 50.0, 1.0)
        leverage = min(float(market.max_leverage or 1.0) / max(float(self.config.get("MAX_LEVERAGE", 3.0) or 3.0), 1.0), 1.0)
        funding_penalty = min(abs(float(market.funding_rate or 0.0)) * 100.0, 0.35)
        return max(0.0, min((liquidity * 0.55) + (leverage * 0.25) + ((1.0 - spread_penalty) * 0.20) - funding_penalty, 1.0))

    @staticmethod
    def _balance_available(balances: list[dict[str, Any]], asset: str) -> float:
        asset_key = str(asset or "").upper().strip()
        for row in balances:
            row_asset = str(row.get("asset") or row.get("currency") or "").upper().strip()
            if row_asset != asset_key:
                continue
            for key in ("withdrawable", "available", "available_balance", "free", "value", "total"):
                try:
                    value = float(row.get(key, 0.0) or 0.0)
                except (TypeError, ValueError):
                    value = 0.0
                if value > 0:
                    return value
        return 0.0

    @staticmethod
    def _provider_reliability(connection: TradingConnection) -> float:
        metadata = connection.provider_metadata or {}
        if metadata.get("last_verified_mode") == "live" and connection.verification_status == "verified":
            return 0.9
        return 0.55

    def _conversion_policy(self, provider: str, collateral: str, settlement: str) -> dict[str, Any]:
        if collateral == settlement:
            return {"supported": True, "reason": ""}
        if (
            {collateral, settlement}.issubset({"USDC", "USDT"})
            and provider in {"hyperliquid", "kucoin"}
            and bool(self.config.get("VAULT_CYCLE_CONVERSION_ENABLED", False))
        ):
            return {"supported": True, "reason": ""}
        return {"supported": False, "reason": "stablecoin_conversion_route_unavailable"}

    def _transfer_cost_penalty(self, provider: str, collateral: str, settlement: str) -> float:
        penalty = 0.02 if collateral != settlement else 0.0
        if provider == "kucoin":
            penalty += 0.01
        return penalty

    @staticmethod
    def _bounded_pct(value: Any, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = default
        return max(0.0, min(parsed, 1.0))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
