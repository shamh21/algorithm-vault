"""Unified Vault Cycle readiness diagnostics.

This module is intentionally fail-closed for live execution while keeping
exchange-specific blockers separate from blockers that stop the whole cycle.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from flask import current_app, has_app_context

from ..extensions import db
from ..ml.online_ranker import ONE_H10_HORIZON
from ..models import LeveragedMarket, LeveragedMarketFeature, RiskEvent, Setting, TradingConnection, User, WalletBalance
from .one_h10_quality import ONE_H10_HORIZON_SECONDS
from .oneinch_funding import OneInchFundingConnector
from .provider_assets import normalize_provider, provider_collateral_asset
from .vault_allocation_assets import BASE_VAULT_ALLOCATION_ASSETS, asset_usd_price
from .worker_lease import in_process_workers_enabled

logger = logging.getLogger(__name__)

SUPPORTED_SETTLEMENT_ASSETS = set(BASE_VAULT_ALLOCATION_ASSETS)
VAULT_READINESS_EXCHANGES = ("hyperliquid", "kucoin")
EXCHANGE_LABELS = {
    "hyperliquid": "Hyperliquid",
    "kucoin": "KuCoin",
}
REQUIRED_ML_FAMILIES = (
    "pytorch_fibonacci",
    "pytorch_risk_policy",
    "pytorch_exit_policy",
    "pytorch_cap_policy",
    "pytorch_execution_policy",
    "pytorch_roi_target",
)
EXCHANGE_READINESS_STATES = {
    "disabled",
    "ready",
    "ready_auto_funded",
    "needs_wallet",
    "needs_api_credentials",
    "needs_verification",
    "geo_restricted",
    "provider_unavailable",
    "credential_error",
    "transfer_failed",
    "blocked",
}


PLACEHOLDER_MARKERS = {
    "",
    "...",
    "changeme",
    "change_me",
    "placeholder",
    "replace_me",
    "your_api_key",
    "your_api_secret",
    "your_secret",
    "example",
    "demo",
    "test",
}


def get_vault_cycle_readiness(
    user_id: int | None,
    cycle: str = "1H10",
    settlement_asset: str = "USDC",
    amount: float | None = None,
    enabled_exchanges: list[str] | tuple[str, ...] | None = None,
    live_acknowledged: bool = False,
    idempotency_key: str | None = None,
    *,
    deposit_asset: str | None = None,
    enforce_ml_gate: bool | None = None,
    require_market_metadata: bool = False,
) -> dict[str, Any]:
    """Return a structured readiness payload for Vault Cycle UI and start flow."""

    service = None
    try:
        service = current_app.extensions.get("services", {}).get("vault_readiness")
    except RuntimeError:
        service = None
    if service is None:
        service = VaultReadinessService(current_app.config)
    return service.get_vault_cycle_readiness(
        user_id=user_id,
        cycle=cycle,
        settlement_asset=settlement_asset,
        amount=amount,
        enabled_exchanges=enabled_exchanges,
        live_acknowledged=live_acknowledged,
        idempotency_key=idempotency_key,
        deposit_asset=deposit_asset,
        enforce_ml_gate=enforce_ml_gate,
        require_market_metadata=require_market_metadata,
    )


class VaultReadinessService:
    """Builds one shared readiness decision for Vault Cycle surfaces."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def get_vault_cycle_readiness(
        self,
        user_id: int | None,
        cycle: str = "1H10",
        settlement_asset: str = "USDC",
        amount: float | None = None,
        enabled_exchanges: list[str] | tuple[str, ...] | None = None,
        live_acknowledged: bool = False,
        idempotency_key: str | None = None,
        *,
        deposit_asset: str | None = None,
        enforce_ml_gate: bool | None = None,
        require_market_metadata: bool = False,
    ) -> dict[str, Any]:
        cycle_key = str(cycle or "1H10").strip().upper()
        settlement = str(settlement_asset or "USDC").strip().upper()
        funding_asset = str(deposit_asset or settlement or "USDC").strip().upper()
        notional_amount = self._safe_float(amount, 0.0)
        requested_exchanges = self._enabled_exchanges(enabled_exchanges)
        active_blockers: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        user = db.session.get(User, int(user_id)) if user_id is not None else None

        if user is None:
            active_blockers.append(
                self._blocker(
                    "user_missing",
                    "User required",
                    "Sign in before checking or starting a Vault Cycle.",
                    "critical",
                    "Sign in and retry the readiness check.",
                )
            )

        if cycle_key not in {"1H10", "ONE_H10"}:
            active_blockers.append(
                self._blocker(
                    "cycle_invalid",
                    "Unsupported cycle",
                    f"{cycle_key or 'This cycle'} is not a supported Vault Cycle.",
                    "blocker",
                    "Select the 1H10 vault cycle.",
                )
            )
        elif not bool(self.config.get("ONE_H10_LIVE_ENABLED", False)):
            active_blockers.append(
                self._blocker(
                    "one_h10_live_disabled",
                    "1H10 live disabled",
                    "1H10 live execution is disabled by configuration.",
                    "blocker",
                    "Set ONE_H10_LIVE_ENABLED=true only after paper and backtest validation.",
                )
            )

        if settlement not in SUPPORTED_SETTLEMENT_ASSETS:
            active_blockers.append(
                self._blocker(
                    "settlement_asset_unsupported",
                    "Unsupported settlement asset",
                    f"{settlement or 'The selected asset'} is not supported for Vault Cycle settlement.",
                    "blocker",
                    "Choose USDC or another configured wallet asset.",
                )
            )

        local_balance = self._wallet_balance(user.id if user else None, funding_asset)
        available_funding = float(local_balance.available_balance or 0.0) if local_balance is not None else 0.0
        if user is not None:
            verified_funding = self._verified_spendable_amount(user.id, funding_asset)
            if verified_funding is not None:
                available_funding = verified_funding
        price, price_warning = self._asset_usd_price(funding_asset)
        if price_warning is not None:
            warnings.append(price_warning)

        if notional_amount <= 0:
            active_blockers.append(
                self._blocker(
                    "amount_required",
                    "Amount required",
                    "Enter an amount greater than 0 before starting a 1H10 vault cycle.",
                    "blocker",
                    "Use MAX or enter an amount within your available balance.",
                )
            )
        elif available_funding + 1e-9 < notional_amount:
            active_blockers.append(
                self._blocker(
                    "amount_exceeds_balance",
                    "Amount exceeds balance",
                    f"{notional_amount:g} {funding_asset} exceeds the available funding balance of {available_funding:g} {funding_asset}.",
                    "blocker",
                    "Tap MAX or enter a smaller amount.",
                )
            )

        if not live_acknowledged:
            active_blockers.append(
                self._blocker(
                    "live_acknowledgement_required",
                    "Live acknowledgement required",
                    "Confirm the 1H10 live execution acknowledgement before starting.",
                    "blocker",
                    "Review the live execution acknowledgement and check the confirmation box.",
                )
            )

        self._append_live_gate_blockers(active_blockers)

        if price <= 0 and notional_amount > 0:
            active_blockers.append(
                self._blocker(
                    "price_unavailable",
                    "Price unavailable",
                    f"A USD estimate is unavailable for {funding_asset}.",
                    "blocker",
                    "Use USDC/USDT or retry after market data recovers.",
                )
            )
        notional_usd = max(0.0, notional_amount * max(price, 0.0))

        ml_readiness = self._one_h10_ml_readiness("global")
        if self._should_enforce_ml_gate(enforce_ml_gate) and not bool(ml_readiness.get("ready", False)):
            active_blockers.append(
                self._blocker(
                    "ml_readiness_required",
                    "ML readiness required",
                    "1H10 live execution requires the promoted ML policy families to be ready.",
                    "blocker",
                    "Promote or repair the required 1H10 ML families before live execution.",
                )
            )

        active_blockers.extend(self._critical_safety_event_blockers(user.id if user else None))
        market_refresh_results = self._refresh_one_h10_market_data(
            user.id if user else None,
            requested_exchanges,
        )

        exchange_status: dict[str, dict[str, Any]] = {}
        for exchange in VAULT_READINESS_EXCHANGES:
            exchange_status[exchange] = self._exchange_status(
                user=user,
                exchange=exchange,
                enabled=exchange in requested_exchanges,
                settlement_asset=settlement,
                deposit_asset=funding_asset,
                amount=notional_amount,
                require_market_metadata=require_market_metadata,
                market_refresh_results=market_refresh_results,
            )

        ready_exchanges = [
            exchange for exchange, status in exchange_status.items() if bool(status.get("enabled")) and bool(status.get("ready"))
        ]
        if not ready_exchanges:
            active_blockers.append(
                self._blocker(
                    "no_exchange_ready",
                    "No exchange ready",
                    "No enabled exchange has a verified live connection, provider verification, market metadata, and trading access.",
                    "blocker",
                    "Verify at least one enabled exchange connection and retry.",
                )
            )

        routes = self._allocate_routes(
            ready_exchanges=ready_exchanges,
            exchange_status=exchange_status,
            notional_usd=notional_usd,
            amount=notional_amount,
        )
        ready_exchange_count = len(ready_exchanges)
        total_exchange_count = len(requested_exchanges)
        exchange_blockers = [
            blocker for status in exchange_status.values() for blocker in list(status.get("blockers") or []) if bool(status.get("enabled"))
        ]
        all_blockers = list(active_blockers) + exchange_blockers
        ready = not active_blockers and ready_exchange_count > 0 and notional_amount > 0
        hard_blockers = self._hard_blockers(all_blockers)
        advisory_blockers = self._advisory_blockers(warnings, ml_readiness)
        clearable_blockers = self._clearable_blockers(all_blockers, advisory_blockers)
        mode = "live_ready" if ready else self._mode_for(active_blockers, ready_exchange_count)
        routing_summary = self._routing_summary(routes, exchange_status, notional_amount, ready_exchange_count)
        payload = {
            "ready": ready,
            "ok": ready,
            "can_start": ready,
            "mode": mode,
            "state_label": self._state_label(active_blockers, exchange_blockers, ready),
            "cycle": "1H10",
            "objective": self._objective_payload(),
            "settlement_asset": settlement,
            "deposit_asset": funding_asset,
            "amount": notional_amount,
            "notional_usd": notional_usd,
            "available_balance": available_funding,
            "ready_exchange_count": ready_exchange_count,
            "total_exchange_count": total_exchange_count,
            "active_blockers": active_blockers,
            "exchange_blockers": exchange_blockers,
            "all_blockers": all_blockers,
            "hard_blockers": hard_blockers,
            "advisory_blockers": advisory_blockers,
            "clearable_blockers": clearable_blockers,
            "warnings": warnings,
            "exchange_status": exchange_status,
            "routing_preview": {
                "notional_usd": notional_usd if notional_amount > 0 else 0.0,
                "routes": routes if notional_amount > 0 else [],
                "summary": routing_summary,
            },
            "ml_readiness": ml_readiness,
            "market_refresh_results": market_refresh_results,
            "idempotency_key": str(idempotency_key or ""),
        }
        self._log_exchange_decisions(exchange_status)
        return payload

    def _objective_payload(self) -> dict[str, Any]:
        target_roi_pct = max(0.0, self._safe_float(self.config.get("ONE_H10_TARGET_ROI_PCT"), 1000.0))
        horizon_seconds = max(60, int(self._safe_float(self.config.get("ONE_H10_HORIZON_SECONDS"), ONE_H10_HORIZON_SECONDS)))
        return {
            "profile": "1H10",
            "name": "1 hour / 10x target",
            "target_multiplier": max(1.0, target_roi_pct / 100.0),
            "target_roi_pct": target_roi_pct,
            "horizon_seconds": horizon_seconds,
            "horizon_label": "1 hour",
            "copy": "Optimizes routing toward a 1-hour / 10x target objective without guaranteeing returns.",
        }

    def _hard_blockers(self, blockers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        hard_severities = {"critical", "blocker"}
        return [dict(item) for item in blockers if str(item.get("severity") or "").lower() in hard_severities]

    def _advisory_blockers(self, warnings: list[dict[str, Any]], ml_readiness: dict[str, Any]) -> list[dict[str, Any]]:
        advisory = [dict(item) for item in warnings if isinstance(item, dict)]
        for blocker in ml_readiness.get("advisory_blockers", []) or []:
            advisory.append(
                self._blocker(
                    "ml_readiness_advisory",
                    "ML promotion advisory",
                    str(blocker),
                    "warning",
                    "Bootstrap 1H10 mode can continue, but promoted ML should be repaired before increasing allocation.",
                )
            )
        return advisory

    def _clearable_blockers(
        self,
        blockers: list[dict[str, Any]],
        advisory_blockers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        clearable = [
            dict(item)
            for item in blockers
            if str(item.get("severity") or "").lower() != "critical" and str(item.get("fix_hint") or "").strip()
        ]
        clearable.extend(dict(item) for item in advisory_blockers if str(item.get("fix_hint") or "").strip())
        return clearable

    def _exchange_status(
        self,
        *,
        user: User | None,
        exchange: str,
        enabled: bool,
        settlement_asset: str,
        deposit_asset: str,
        amount: float,
        require_market_metadata: bool,
        market_refresh_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        label = EXCHANGE_LABELS.get(exchange, exchange.title())
        blockers: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        connection = self._connection_for(user.id if user else None, exchange)
        collateral_asset = provider_collateral_asset(exchange)
        available_margin = 0.0
        markets = self._active_markets(exchange, connection.id if connection is not None else None)
        market_freshness = self._one_h10_market_freshness(exchange, markets, market_refresh_results or [])

        if not enabled:
            blockers.append(
                self._blocker(
                    f"{exchange}_disabled",
                    f"{label} disabled",
                    f"{label} is disabled for this route preview.",
                    "info",
                    "Enable the exchange if you want to route Vault Cycle capital there.",
                    exchange=exchange,
                )
            )
        if connection is None:
            blockers.append(
                self._blocker(
                    f"{exchange}_credentials_missing",
                    f"{label} setup incomplete",
                    f"{label} API credentials are missing or not verified.",
                    "blocker",
                    f"Add and verify {label} credentials before routing funds there.",
                    exchange=exchange,
                )
            )
        else:
            connection_warnings, connection_blockers = self._connection_diagnostics(connection, exchange)
            warnings.extend(connection_warnings)
            blockers.extend(connection_blockers)
            credential_warnings, credential_blockers = self._credential_diagnostics(connection, exchange)
            warnings.extend(credential_warnings)
            blockers.extend(credential_blockers)
            if not self._has_trading_permission(connection, exchange):
                blockers.append(
                    self._blocker(
                        f"{exchange}_trading_permission_missing",
                        f"{label} trading permission missing",
                        f"{label} did not confirm live trading permission or its trading endpoint was unreachable.",
                        "blocker",
                        f"Verify the {label} connection with live trading permission enabled.",
                        exchange=exchange,
                    )
                )
            snapshot = self._account_snapshot(user.id if user else None, connection)
            if snapshot is None:
                blockers.append(
                    self._blocker(
                        f"{exchange}_balance_fetch_failed",
                        f"{label} balance unavailable",
                        f"{label} balances could not be fetched.",
                        "blocker",
                        f"Re-verify {label} credentials and retry the balance check.",
                        exchange=exchange,
                    )
                )
            else:
                alerts = [str(alert) for alert in (getattr(snapshot, "alerts", []) or []) if str(alert).strip()]
                geo_restriction = self._kucoin_geo_restriction(alerts if exchange == "kucoin" else [])
                if geo_restriction is not None:
                    blockers.append(geo_restriction)
                elif alerts:
                    blockers.append(
                        self._blocker(
                            f"{exchange}_connection_failed",
                            f"{label} connection failed",
                            self._sanitize_provider_message("; ".join(alerts[:2])),
                            "blocker",
                            f"Resolve the {label} connection alert, then verify the connection again.",
                            exchange=exchange,
                        )
                    )
                available_margin = self._snapshot_available_margin(snapshot, collateral_asset)
                if available_margin <= 0:
                    if exchange == "hyperliquid":
                        funding_state = self._hyperliquid_auto_funding_state(
                            user=user,
                            connection=connection,
                            deposit_asset=deposit_asset,
                            collateral_asset=collateral_asset,
                            settlement_asset=settlement_asset,
                            amount=amount,
                        )
                        if bool(funding_state.get("ready")):
                            warnings.append(
                                self._blocker(
                                    "hyperliquid_auto_funding_pending",
                                    "Auto-funded during cycle",
                                    "USDT is converted to USDC server-side and transferred to Hyperliquid before trading starts.",
                                    "warning",
                                    "Keep Hyperliquid enabled; funding is handled by the Vault Cycle transfer step.",
                                    exchange=exchange,
                                )
                            )
                        else:
                            blockers.append(
                                self._blocker(
                                    str(funding_state.get("code") or "hyperliquid_auto_funding_unavailable"),
                                    str(funding_state.get("title") or "Hyperliquid auto-funding unavailable"),
                                    str(funding_state.get("description") or "Hyperliquid has no usable USDC collateral and auto-funding is not ready."),
                                    "blocker",
                                    str(funding_state.get("fix_hint") or "Configure and verify the server-side USDT to USDC funding route."),
                                    exchange=exchange,
                                )
                            )
                    else:
                        blockers.append(
                            self._blocker(
                                f"{exchange}_settlement_balance_unavailable",
                                f"{label} collateral unavailable",
                                f"{label} has no usable {collateral_asset} collateral for live Vault execution.",
                                "blocker",
                                f"Fund the {label} futures account with {collateral_asset} or reduce the route.",
                                exchange=exchange,
                            )
                        )

        if exchange == "hyperliquid":
            market_warnings, market_blockers = self._hyperliquid_specific_blockers(
                connection, settlement_asset, markets, require_market_metadata, market_freshness
            )
            warnings.extend(market_warnings)
            blockers.extend(market_blockers)
        elif exchange == "kucoin":
            market_warnings, market_blockers = self._kucoin_specific_blockers(connection, markets, require_market_metadata)
            warnings.extend(market_warnings)
            blockers.extend(market_blockers)

        market_warning = self._market_warning(exchange, markets)
        if market_warning is not None:
            warnings.append(market_warning)

        blocking = [item for item in blockers if item.get("severity") in {"blocker", "critical"}]
        auto_funded = exchange == "hyperliquid" and available_margin <= 0 and not blocking
        ready = enabled and not blocking and (available_margin > 0 or auto_funded)
        status = self._exchange_readiness_state(
            exchange=exchange,
            enabled=enabled,
            ready=ready,
            auto_funded=auto_funded,
            blockers=blockers,
            connection=connection,
        )
        score = self._exchange_score(ready=ready, available_margin=available_margin, markets=markets, exchange=exchange)
        return {
            "enabled": enabled,
            "eligible": ready,
            "ready": ready,
            "score": score,
            "allocation_pct": 0,
            "allocation_weight": 0.0,
            "notional_usd": 0.0,
            "target_amount": 0.0,
            "available_margin_usd": available_margin,
            "collateral_asset": collateral_asset,
            "connected": connection is not None,
            "verified": bool(connection and connection.verification_status == "verified"),
            "can_trade": ready,
            "status": status,
            "readiness_state": status,
            "funding_status": "auto_funded" if auto_funded else "available" if available_margin > 0 else "unavailable",
            "funding_label": "Auto-funded during vault cycle"
            if auto_funded
            else f"{collateral_asset} collateral available"
            if available_margin > 0
            else f"{collateral_asset} collateral unavailable",
            "funding_detail": "USDT is converted to USDC server-side and transferred to Hyperliquid before trading starts."
            if auto_funded
            else "",
            "label": label,
            "blockers": blockers,
            "warnings": warnings,
            "market_data_freshness": market_freshness,
            "trading_connection_id": connection.id if connection is not None else None,
        }

    def _refresh_one_h10_market_data(self, user_id: int | None, exchanges: list[str]) -> list[dict[str, Any]]:
        if user_id is None or not bool(self.config.get("ONE_H10_READINESS_REFRESH_MARKET_DATA", True)):
            return []
        if not any(exchange in {"hyperliquid", "kucoin"} for exchange in exchanges):
            return []
        if not has_app_context():
            return []
        try:
            service = current_app.extensions.get("services", {}).get("leveraged_markets")
        except RuntimeError:
            return []
        if service is None:
            return []
        try:
            logger.info("1H10 market data refresh start user_id=%s exchanges=%s", user_id, ",".join(exchanges))
            rows = list(service.sync_for_user(user_id, mode="live", feature_scope="all", persist_features=True))
            db.session.flush()
            logger.info("1H10 market data refresh end user_id=%s rows=%s", user_id, len(rows))
            return rows
        except Exception as exc:  # noqa: BLE001
            db.session.rollback()
            logger.warning("1H10 market data refresh failed user_id=%s: %s", user_id, exc)
            return [{"skipped": True, "reason": str(exc), "provider": "global"}]

    def _one_h10_market_freshness(
        self,
        provider: str,
        markets: list[LeveragedMarket],
        refresh_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if provider not in {"hyperliquid", "kucoin"}:
            return {}
        required = self._required_feature_timeframes()
        max_age = self._max_feature_age_seconds()
        min_candles = max(0, int(self._safe_float(self.config.get("ONE_H10_MIN_CANDLES_PER_HORIZON"), 40.0)))
        market_ids = [int(market.id) for market in markets if market.id is not None]
        feature_rows = (
            LeveragedMarketFeature.query.filter(LeveragedMarketFeature.leveraged_market_id.in_(market_ids)).all() if market_ids else []
        )
        by_market: dict[int, dict[str, LeveragedMarketFeature]] = {}
        for row in feature_rows:
            by_market.setdefault(int(row.leveraged_market_id), {})[str(row.timeframe)] = row
        now = datetime.utcnow()
        eligible_markets = 0
        stale_horizons: set[str] = set()
        missing_horizons: set[str] = set()
        insufficient_horizons: set[str] = set()
        candle_counts: dict[str, int] = {}
        last_refresh: datetime | None = None
        max_age_seconds = 0.0
        market_rows: list[dict[str, Any]] = []
        for market in markets[:25]:
            rows = by_market.get(int(market.id or 0), {})
            row_ready = True
            horizon_rows: dict[str, dict[str, Any]] = {}
            for timeframe in required:
                feature = rows.get(timeframe)
                if feature is None:
                    missing_horizons.add(timeframe)
                    row_ready = False
                    candle_counts[timeframe] = max(candle_counts.get(timeframe, 0), 0)
                    horizon_rows[timeframe] = {"missing": True, "stale": True, "candle_count": 0, "age_seconds": None}
                    continue
                features = dict(feature.features or {})
                updated_at = self._parse_datetime(features.get("last_successful_refresh_at") or feature.updated_at)
                age = (now - updated_at).total_seconds() if updated_at is not None else None
                count = int(self._safe_float(features.get("candle_count"), 0.0))
                candle_counts[timeframe] = max(candle_counts.get(timeframe, 0), count)
                horizon_max_age = self._horizon_feature_max_age_seconds(timeframe, max_age)
                stale = updated_at is None or (horizon_max_age > 0 and age is not None and age > horizon_max_age)
                if stale:
                    stale_horizons.add(timeframe)
                    row_ready = False
                if count < min_candles:
                    insufficient_horizons.add(timeframe)
                    row_ready = False
                if updated_at is not None and (last_refresh is None or updated_at > last_refresh):
                    last_refresh = updated_at
                if age is not None:
                    max_age_seconds = max(max_age_seconds, age)
                horizon_rows[timeframe] = {
                    "missing": False,
                    "stale": stale,
                    "candle_count": count,
                    "age_seconds": age,
                    "max_age_seconds": horizon_max_age,
                    "last_successful_refresh_at": updated_at.isoformat() if updated_at is not None else None,
                    "source": features.get("market_data_source") or ("cache" if stale else "live"),
                }
            if row_ready:
                eligible_markets += 1
            market_rows.append(
                {
                    "symbol": market.symbol,
                    "venue_symbol": market.venue_symbol,
                    "ready": row_ready,
                    "horizons": horizon_rows,
                }
            )
        provider_refresh = [row for row in refresh_results if str(row.get("provider") or "").lower() in {provider, "global"}]
        return {
            "provider": provider,
            "ready": bool(eligible_markets > 0),
            "markets_checked": len(markets),
            "eligible_markets": eligible_markets,
            "required_horizons": required,
            "stale_horizons": sorted(stale_horizons),
            "missing_horizons": sorted(missing_horizons),
            "insufficient_horizons": sorted(insufficient_horizons),
            "candle_count_by_horizon": candle_counts,
            "last_successful_refresh_at": last_refresh.isoformat() if last_refresh is not None else None,
            "age_seconds": max_age_seconds if last_refresh is not None else None,
            "source": "live" if any(int(row.get("features_attempted", 0) or 0) > 0 for row in provider_refresh) else "cache",
            "refresh_results": provider_refresh[:3],
            "markets": market_rows[:5],
        }

    def _required_feature_timeframes(self) -> list[str]:
        raw = self.config.get("ONE_H10_REQUIRED_FEATURE_TIMEFRAMES", ["15m", "1h", "4h"])
        if isinstance(raw, str):
            raw = [item.strip() for item in raw.split(",") if item.strip()]
        return [str(item).strip() for item in (raw or []) if str(item).strip()]

    def _max_feature_age_seconds(self) -> float:
        default = self._safe_float(self.config.get("ONE_H10_FEATURE_REFRESH_SECONDS"), 3600.0)
        return max(0.0, self._safe_float(self.config.get("ONE_H10_MARKET_DATA_FRESHNESS_SECONDS"), default))

    def _horizon_feature_max_age_seconds(self, timeframe: str, base_max_age: float) -> float:
        base = max(0.0, self._safe_float(base_max_age, 0.0))
        seconds = self._timeframe_seconds(timeframe)
        if seconds <= 0:
            return base
        return max(base, seconds + base)

    @staticmethod
    def _timeframe_seconds(timeframe: str) -> float:
        value = str(timeframe or "").strip().lower()
        if not value:
            return 0.0
        try:
            amount = float(value[:-1])
        except ValueError:
            return 0.0
        unit = value[-1]
        if unit == "m":
            return amount * 60.0
        if unit == "h":
            return amount * 3600.0
        if unit == "d":
            return amount * 86400.0
        return 0.0

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value.replace(tzinfo=None)
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(UTC).replace(tzinfo=None)
        return parsed

    def _exchange_readiness_state(
        self,
        *,
        exchange: str,
        enabled: bool,
        ready: bool,
        auto_funded: bool,
        blockers: list[dict[str, Any]],
        connection: TradingConnection | None,
    ) -> str:
        if not enabled:
            return "disabled"
        if ready and auto_funded:
            return "ready_auto_funded"
        if ready:
            return "ready"
        codes = {str(item.get("code") or "") for item in blockers}
        if f"{exchange}_geo_restricted" in codes:
            return "geo_restricted"
        if any("wallet" in code for code in codes):
            return "needs_wallet"
        if any("credentials_missing" in code for code in codes):
            return "needs_api_credentials"
        if any("credentials" in code for code in codes):
            return "credential_error"
        if any("not_verified" in code for code in codes) or (connection is not None and connection.verification_status != "verified"):
            return "needs_verification"
        if any("transfer" in code for code in codes):
            return "transfer_failed"
        if any("unavailable" in code or "connection_failed" in code for code in codes):
            return "provider_unavailable"
        return "blocked"

    def _kucoin_geo_restriction(self, messages: list[Any], metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
        metadata = metadata or {}
        diagnostics = metadata.get("provider_diagnostics") if isinstance(metadata, dict) else None
        if isinstance(diagnostics, dict) and str(diagnostics.get("providerCode") or diagnostics.get("provider_code") or "") == "400302":
            detected = str(diagnostics.get("detectedArea") or "restricted region").upper()
            return self._geo_restricted_blocker(detected, diagnostics)
        combined = " ".join(str(message or "") for message in messages if str(message or "").strip())
        if not combined:
            return None
        lowered = combined.lower()
        if "400302" not in combined and not any(
            phrase in lowered
            for phrase in ("unavailable in your current area", "unavailable in the detected region", "restricted region", "current area")
        ):
            return None
        detected = (
            self._detected_area_from_text(combined) or "US"
            if " us" in f" {lowered}" or "united states" in lowered
            else self._detected_area_from_text(combined) or "restricted region"
        )
        diagnostics = {
            "providerCode": "400302" if "400302" in combined else "",
            "detectedArea": str(detected).upper() if len(str(detected)) <= 3 else str(detected),
            "egressRegion": os.environ.get("VERCEL_REGION") or os.environ.get("AWS_REGION") or "unknown",
            "maskedIp": self._mask_ip(combined),
        }
        return self._geo_restricted_blocker(str(diagnostics["detectedArea"]), diagnostics)

    def _geo_restricted_blocker(self, detected_area: str, diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
        area = str(detected_area or "restricted region").upper()
        blocker = self._blocker(
            "kucoin_geo_restricted",
            "Provider restricted",
            f"KuCoin rejected verification from detected region: {area}.",
            "blocker",
            "Recheck provider after server-region update or use another supported exchange.",
            exchange="kucoin",
            docs_or_action_label="Recheck provider",
        )
        if diagnostics:
            blocker["diagnostics"] = {
                key: value
                for key, value in {
                    "providerCode": diagnostics.get("providerCode"),
                    "detectedArea": diagnostics.get("detectedArea"),
                    "egressRegion": diagnostics.get("egressRegion"),
                    "maskedIp": diagnostics.get("maskedIp"),
                }.items()
                if value
            }
        return blocker

    @staticmethod
    def _sanitize_provider_message(message: str) -> str:
        return re.sub(r"\b(\d{1,3})\.(\d{1,3})\.\d{1,3}\.\d{1,3}\b", r"\1.\2.xxx.xxx", str(message or ""))[:500]

    @staticmethod
    def _mask_ip(message: str) -> str:
        match = re.search(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b", str(message or ""))
        return f"{match.group(1)}.{match.group(2)}.xxx.xxx" if match else ""

    @staticmethod
    def _detected_area_from_text(message: str) -> str:
        text = str(message or "")
        for pattern in (
            r'"(?:detectedArea|detected_area|area|country|region)"\s*:\s*"?([A-Za-z]{2,32})',
            r"detected region[:= ]+([A-Za-z]{2,32})",
            r"current area[:= ]+([A-Za-z]{2,32})",
        ):
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return ""
        if isinstance(payload, dict):
            for key in ("detectedArea", "detected_area", "area", "country", "region"):
                if payload.get(key):
                    return str(payload[key])
        return ""

    def _connection_for(self, user_id: int | None, exchange: str) -> TradingConnection | None:
        if user_id is None:
            return None
        provider = normalize_provider(exchange)
        active = (
            TradingConnection.query.filter_by(user_id=int(user_id), provider=provider, is_active=True)
            .order_by(TradingConnection.updated_at.desc(), TradingConnection.id.desc())
            .first()
        )
        if active is not None:
            return active
        return (
            TradingConnection.query.filter_by(user_id=int(user_id), provider=provider)
            .order_by(TradingConnection.updated_at.desc(), TradingConnection.id.desc())
            .first()
        )

    def _connection_diagnostics(self, connection: TradingConnection, exchange: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        label = EXCHANGE_LABELS.get(exchange, exchange.title())
        warnings: list[dict[str, Any]] = []
        blockers: list[dict[str, Any]] = []
        if not bool(connection.is_active):
            warnings.append(
                self._blocker(
                    f"{exchange}_connection_inactive",
                    f"{label} inactive",
                    f"{label} is verified but not marked active. The explicit route selection will still require live trading and balance checks.",
                    "warning",
                    f"Activate the verified {label} connection for default routing.",
                    exchange=exchange,
                )
            )
        if exchange == "kucoin":
            geo_restriction = self._kucoin_geo_restriction([connection.last_verification_error or ""], connection.provider_metadata)
            if geo_restriction is not None:
                blockers.append(geo_restriction)
                return warnings, blockers
        if connection.verification_status != "verified":
            blockers.append(
                self._blocker(
                    f"{exchange}_connection_not_verified",
                    f"{label} not verified",
                    f"{label} must be verified before live Vault routing.",
                    "blocker",
                    f"Run connection verification for {label}.",
                    exchange=exchange,
                )
            )
        return warnings, blockers

    def _credential_diagnostics(self, connection: TradingConnection, exchange: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        label = EXCHANGE_LABELS.get(exchange, exchange.title())
        warnings: list[dict[str, Any]] = []
        blockers: list[dict[str, Any]] = []
        stored_required = {
            "hyperliquid": [("api_secret", connection.encrypted_api_secret), ("wallet_address", connection.wallet_address)],
            "kucoin": [
                ("api_key", connection.encrypted_api_key),
                ("api_secret", connection.encrypted_api_secret),
                ("passphrase", connection.encrypted_passphrase),
            ],
        }.get(exchange, [])
        missing_fields = [field for field, value in stored_required if not str(value or "").strip()]
        if missing_fields:
            blockers.append(
                self._blocker(
                    f"{exchange}_credentials_missing",
                    f"{label} credentials missing",
                    f"{label} is missing required credential fields: {', '.join(missing_fields)}.",
                    "blocker",
                    f"Add the missing {label} credential fields and verify the connection.",
                    exchange=exchange,
                )
            )
            return warnings, blockers

        try:
            credentials = self._trading_connections().credentials_for_execution(connection.user_id, connection.id)
        except Exception:  # noqa: BLE001
            warnings.append(
                self._blocker(
                    f"{exchange}_credentials_decrypt_unverified",
                    f"{label} credential decrypt check inconclusive",
                    f"{label} credentials are stored, but the readiness diagnostic could not decrypt them for placeholder detection.",
                    "warning",
                    f"Re-verify {label} if trading or balance checks fail.",
                    exchange=exchange,
                )
            )
            return warnings, blockers

        plaintext_values = [
            getattr(credentials, "api_key", ""),
            getattr(credentials, "api_secret", ""),
            getattr(credentials, "passphrase", ""),
            getattr(credentials, "wallet_address", ""),
        ]
        if any(self._looks_placeholder(value) for value in plaintext_values if str(value or "").strip()):
            blockers.append(
                self._blocker(
                    f"{exchange}_credentials_placeholder",
                    f"{label} credentials look like placeholders",
                    f"{label} credential fields contain placeholder-looking values.",
                    "blocker",
                    f"Replace placeholder values with real {label} API credentials, then verify.",
                    exchange=exchange,
                )
            )
        return warnings, blockers

    def _has_trading_permission(self, connection: TradingConnection, exchange: str) -> bool:
        try:
            return bool(self._trading_connections().can_trade(connection.user_id, "live", connection.id))
        except Exception:  # noqa: BLE001
            logger.warning("%s readiness can_trade failed for connection_id=%s", exchange, connection.id)
            return False

    def _account_snapshot(self, user_id: int | None, connection: TradingConnection | None) -> Any | None:
        if user_id is None or connection is None:
            return None
        try:
            return self._trading_connections().account_snapshot(user_id, "live", connection.id)
        except Exception:  # noqa: BLE001
            logger.warning("Vault readiness balance fetch failed for provider=%s connection_id=%s", connection.provider, connection.id)
            return None

    def _hyperliquid_specific_blockers(
        self,
        connection: TradingConnection | None,
        settlement_asset: str,
        markets: list[LeveragedMarket],
        require_market_metadata: bool,
        market_freshness: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        warnings: list[dict[str, Any]] = []
        blockers: list[dict[str, Any]] = []
        if not bool(self.config.get("ENABLE_LIVE_TRADING", False)):
            blockers.append(
                self._blocker(
                    "hyperliquid_live_mode_disabled",
                    "Hyperliquid live mode disabled",
                    "Live trading is disabled by configuration.",
                    "blocker",
                    "Set ENABLE_LIVE_TRADING=true only after live readiness is validated.",
                    exchange="hyperliquid",
                )
            )
        if settlement_asset != "USDC":
            blockers.append(
                self._blocker(
                    "hyperliquid_usdc_settlement_unavailable",
                    "Hyperliquid requires USDC",
                    "Hyperliquid perpetuals use USDC collateral for this Vault routing flow.",
                    "blocker",
                    "Select USDC settlement for Hyperliquid or disable Hyperliquid for this route.",
                    exchange="hyperliquid",
                )
            )
        if connection is not None and not str(connection.wallet_address or "").strip():
            blockers.append(
                self._blocker(
                    "hyperliquid_wallet_not_verified",
                    "Hyperliquid wallet not verified",
                    "Hyperliquid requires a verified account wallet address.",
                    "blocker",
                    "Add the Hyperliquid account address and re-run verification.",
                    exchange="hyperliquid",
                )
            )
        if not markets:
            target = blockers if require_market_metadata else warnings
            target.append(
                self._blocker(
                    "hyperliquid_market_metadata_unavailable",
                    "Hyperliquid markets unavailable",
                    "Hyperliquid supported markets are not loaded for 1H10 routing.",
                    "blocker" if require_market_metadata else "warning",
                    "Refresh market discovery before routing to Hyperliquid.",
                    exchange="hyperliquid",
                )
            )
        elif require_market_metadata and not bool((market_freshness or {}).get("ready", False)):
            freshness = market_freshness or {}
            stale = [str(item) for item in freshness.get("stale_horizons", []) if str(item)]
            missing = [str(item) for item in freshness.get("missing_horizons", []) if str(item)]
            insufficient = [str(item) for item in freshness.get("insufficient_horizons", []) if str(item)]
            if missing:
                code = "missing_candles"
                title = "Hyperliquid candles missing"
                detail = f"Missing required Hyperliquid candle horizon: {', '.join(missing[:4])}."
            elif insufficient:
                code = "insufficient_candles"
                title = "Hyperliquid candles insufficient"
                detail = f"Insufficient Hyperliquid candle history for: {', '.join(insufficient[:4])}."
            else:
                code = "stale_candles"
                title = "Hyperliquid candles stale"
                detail = f"Stale Hyperliquid candle horizon: {', '.join(stale[:4]) or 'required horizon'}."
            blockers.append(
                self._blocker(
                    code,
                    title,
                    detail,
                    "blocker",
                    "Refresh Hyperliquid market data and retry; execution remains blocked until fresh candles pass.",
                    exchange="hyperliquid",
                )
            )
        return warnings, blockers

    def _kucoin_specific_blockers(
        self,
        connection: TradingConnection | None,
        markets: list[LeveragedMarket],
        require_market_metadata: bool,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if markets:
            return [], []
        payload = self._blocker(
            "kucoin_market_metadata_unavailable",
            "KuCoin contracts unavailable",
            "KuCoin futures contract metadata is not loaded for 1H10 routing.",
            "blocker" if require_market_metadata else "warning",
            "Refresh KuCoin market discovery before routing to KuCoin.",
            exchange="kucoin",
        )
        if require_market_metadata:
            return [], [payload]
        return [payload], []

    def _append_live_gate_blockers(self, blockers: list[dict[str, Any]]) -> None:
        if bool(self.config.get("RECOVERY_SQLITE_ACTIVE", False)):
            blockers.append(
                self._blocker(
                    "recovery_database_mode",
                    "Recovery database mode",
                    "Live 1H10 execution is disabled while the app is running on recovery SQLite.",
                    "critical",
                    "Attach healthy production Postgres and set ALGVAULT_RECOVERY_SQLITE_ENABLED=false before live allocation.",
                )
            )
        if not bool(self.config.get("ENABLE_LIVE_TRADING", False)):
            blockers.append(
                self._blocker(
                    "live_trading_disabled",
                    "Live trading disabled",
                    "Live trading is disabled by configuration.",
                    "critical",
                    "Set ENABLE_LIVE_TRADING=true only after the full live checklist passes.",
                )
            )
        elif not self._live_execution_runtime_available():
            blockers.append(
                self._blocker(
                    "live_execution_runtime_missing",
                    "Live execution runtime missing",
                    "Live 1H10 execution requires a worker runtime or a configured live execution process.",
                    "critical",
                    "Configure a dedicated worker/cron execution path before live allocation.",
                )
            )
        if Setting.get_json("panic_lock", False):
            blockers.append(
                self._blocker(
                    "panic_lock",
                    "Panic lock active",
                    "Trading is disabled until the panic lock is manually reset.",
                    "critical",
                    "Review the safety event and reset the panic lock only when it is safe.",
                )
            )
        if Setting.get_json("live_trading_blocked", False):
            blockers.append(
                self._blocker(
                    "live_trading_blocked",
                    "Live trading blocked",
                    "A persistent live-trading block is active.",
                    "critical",
                    "Resolve the live-trading block in settings before starting a live cycle.",
                )
            )
        if not bool(self.config.get("EXPLICIT_LIVE_CONFIRMED", False)) or not bool(Setting.get_json("explicit_live_confirmed", False)):
            blockers.append(
                self._blocker(
                    "explicit_live_confirmation_missing",
                    "Explicit live confirmation missing",
                    "1H10 live execution requires explicit live trading confirmation.",
                    "blocker",
                    "Complete the explicit live trading confirmation.",
                )
            )
        if not bool(self.config.get("SECONDARY_CONFIRMATION", False)) or not bool(Setting.get_json("secondary_confirmation", False)):
            blockers.append(
                self._blocker(
                    "secondary_live_confirmation_missing",
                    "Secondary confirmation missing",
                    "1H10 live execution requires secondary live trading confirmation.",
                    "blocker",
                    "Complete the secondary live trading confirmation.",
                )
            )
        if bool(self.config.get("LIVE_MICRO_CANARY_ENABLED", False)) and (
            bool(self.config.get("LIVE_MICRO_CANARY_PREVIEW_ONLY", True))
            or not bool(self.config.get("LIVE_MICRO_CANARY_LIVE_SUBMIT_ENABLED", False))
        ):
            blockers.append(
                self._blocker(
                    "canary_preview_only",
                    "Canary preview-only",
                    "Live micro-canary is configured for preview-only mode, so live submit is blocked.",
                    "blocker",
                    "Set LIVE_MICRO_CANARY_PREVIEW_ONLY=false and LIVE_MICRO_CANARY_LIVE_SUBMIT_ENABLED=true after canary approval.",
                )
            )

    def _critical_safety_event_blockers(self, user_id: int | None) -> list[dict[str, Any]]:
        blockers: list[dict[str, Any]] = []
        stored = Setting.get_json("critical_safety_events", [])
        if isinstance(stored, list):
            unresolved = [
                item
                for item in stored
                if isinstance(item, dict) and str(item.get("status") or "open").lower() not in {"resolved", "closed"}
            ]
            if unresolved:
                blockers.append(
                    self._blocker(
                        "critical_safety_event_unresolved",
                        "Critical safety event unresolved",
                        "A critical safety event is still open.",
                        "critical",
                        "Resolve the critical safety event before live execution.",
                    )
                )
        if user_id is None:
            return blockers
        since = datetime.utcnow() - timedelta(hours=24)
        recent = (
            RiskEvent.query.filter(RiskEvent.user_id == int(user_id), RiskEvent.created_at >= since)
            .order_by(RiskEvent.created_at.desc())
            .limit(10)
            .all()
        )
        for event in recent:
            payload = event.payload or {}
            severity = str(payload.get("severity") or "").lower()
            status = str(payload.get("status") or "open").lower()
            if severity == "critical" and status not in {"resolved", "closed"}:
                blockers.append(
                    self._blocker(
                        "critical_safety_event_unresolved",
                        "Critical safety event unresolved",
                        event.reason or "A critical safety event is still open.",
                        "critical",
                        "Resolve the critical safety event before live execution.",
                    )
                )
                break
        return blockers

    def _allocate_routes(
        self,
        *,
        ready_exchanges: list[str],
        exchange_status: dict[str, dict[str, Any]],
        notional_usd: float,
        amount: float,
    ) -> list[dict[str, Any]]:
        if amount <= 0 or notional_usd <= 0 or not ready_exchanges:
            for status in exchange_status.values():
                status["allocation_pct"] = 0
                status["allocation_weight"] = 0.0
                status["notional_usd"] = 0.0
                status["target_amount"] = 0.0
            return []
        weights = {exchange: max(float(exchange_status[exchange].get("available_margin_usd") or 0.0), 0.0) for exchange in ready_exchanges}
        if not any(value > 0 for value in weights.values()):
            weights = {exchange: max(float(exchange_status[exchange].get("score") or 0.0), 1.0) for exchange in ready_exchanges}
        total_weight = sum(weights.values()) or float(len(ready_exchanges))
        routes: list[dict[str, Any]] = []
        for index, exchange in enumerate(ready_exchanges):
            if len(ready_exchanges) == 1:
                weight = 1.0
            elif index == len(ready_exchanges) - 1:
                weight = max(0.0, 1.0 - sum(float(route["allocation_weight"]) for route in routes))
            else:
                weight = weights[exchange] / max(total_weight, 1e-9)
            allocation_pct = round(weight * 100.0, 2)
            route_notional = notional_usd * weight
            target_amount = amount * weight
            status = exchange_status[exchange]
            status["allocation_pct"] = allocation_pct
            status["allocation_weight"] = weight
            status["notional_usd"] = route_notional
            status["target_amount"] = target_amount
            routes.append(
                {
                    "exchange": exchange,
                    "label": status.get("label") or EXCHANGE_LABELS.get(exchange, exchange.title()),
                    "allocation_pct": allocation_pct,
                    "allocation_weight": weight,
                    "notional_usd": route_notional,
                    "target_amount": target_amount,
                    "score": status.get("score", 0),
                }
            )
        for exchange, status in exchange_status.items():
            if exchange not in ready_exchanges:
                status["allocation_pct"] = 0
                status["allocation_weight"] = 0.0
                status["notional_usd"] = 0.0
                status["target_amount"] = 0.0
        return routes

    def _routing_summary(
        self,
        routes: list[dict[str, Any]],
        exchange_status: dict[str, dict[str, Any]],
        amount: float,
        ready_exchange_count: int,
    ) -> str:
        if amount <= 0:
            return "Enter amount to generate route."
        if not routes:
            return "No enabled exchange is ready for this amount."
        ready_parts = [f"{route['allocation_pct']:.0f}% {route.get('label') or route['exchange'].title()}" for route in routes]
        blocked = [
            str(status.get("label") or exchange.title())
            for exchange, status in exchange_status.items()
            if bool(status.get("enabled")) and not bool(status.get("ready"))
        ]
        if blocked:
            return f"{' / '.join(ready_parts)} / {', '.join(blocked)} blocked"
        if ready_exchange_count == 1:
            return ready_parts[0]
        return " / ".join(ready_parts)

    def _mode_for(self, active_blockers: list[dict[str, Any]], ready_exchange_count: int) -> str:
        if any(item.get("severity") == "critical" for item in active_blockers):
            return "blocked"
        if ready_exchange_count > 0:
            return "bootstrap" if any(item.get("code") == "amount_required" for item in active_blockers) else "blocked"
        return "blocked"

    def _state_label(self, active_blockers: list[dict[str, Any]], exchange_blockers: list[dict[str, Any]], ready: bool) -> str:
        if ready:
            return "Live Ready"
        codes = {str(item.get("code") or "") for item in active_blockers}
        if "amount_required" in codes:
            return "Amount Required"
        if "ml_readiness_required" in codes:
            return "ML Readiness Required"
        if codes.intersection({"panic_lock", "live_trading_disabled", "live_trading_blocked", "canary_preview_only"}):
            return "Risk Review Required"
        if "no_exchange_ready" in codes or exchange_blockers:
            return "Exchange Setup Required"
        return "Blocked"

    def _one_h10_ml_readiness(self, provider: str = "global") -> dict[str, Any]:
        try:
            engine = current_app.extensions.get("services", {}).get("ml_decision_engine")
            if engine is None:
                raise RuntimeError("ML decision engine is unavailable")
            families = {
                family: dict(engine.family_readiness(family, ONE_H10_HORIZON, provider=provider)) for family in REQUIRED_ML_FAMILIES
            }
            blockers: list[str] = []
            for family, payload in families.items():
                blockers.extend(f"{family}:{item}" for item in payload.get("blockers", []) or [])
            promoted_blockers = list(dict.fromkeys(blockers))
            promoted_ready = not promoted_blockers
            bootstrap_enabled = bool(self.config.get("ONE_H10_BOOTSTRAP_LIVE_ENABLED", True)) and not bool(
                self.config.get("ONE_H10_REQUIRE_PROMOTED_ML", True)
            )
            execution_ready = promoted_ready or bootstrap_enabled
            readiness_mode = "promoted" if promoted_ready else "bootstrap" if bootstrap_enabled else "blocked"
            return {
                "ready": execution_ready,
                "execution_ready": execution_ready,
                "promoted_ready": promoted_ready,
                "bootstrap_enabled": bootstrap_enabled,
                "mode": readiness_mode,
                "display_status": "Ready" if promoted_ready else "Bootstrap Ready" if bootstrap_enabled else "ML Readiness Required",
                "enabled": bool(self.config.get("ML_ALL_AREAS_ENABLED", False)) or bootstrap_enabled,
                "horizon": ONE_H10_HORIZON,
                "provider": provider,
                "family": "one_h10_live_execution",
                "objective": "one_h10",
                "families": families,
                "required_families": list(REQUIRED_ML_FAMILIES),
                "blockers": [] if execution_ready else promoted_blockers,
                "advisory_blockers": promoted_blockers if execution_ready and not promoted_ready else [],
                "promoted_blockers": promoted_blockers,
                "source": "vault_readiness",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ready": False,
                "enabled": False,
                "horizon": ONE_H10_HORIZON,
                "provider": provider,
                "blockers": [str(exc)],
                "source": "vault_readiness_ml_error",
            }

    def _should_enforce_ml_gate(self, enforce_ml_gate: bool | None) -> bool:
        if enforce_ml_gate is not None:
            return bool(enforce_ml_gate)
        return bool(self.config.get("ONE_H10_REQUIRE_PROMOTED_ML", True)) and bool(self.config.get("ML_ALL_AREAS_ENABLED", False))

    def _live_execution_runtime_available(self) -> bool:
        return bool(self.config.get("WORKER_PROCESS_CONFIGURED", False)) or in_process_workers_enabled(self.config)

    def _active_markets(self, exchange: str, connection_id: int | None) -> list[LeveragedMarket]:
        query = LeveragedMarket.query.filter_by(provider=normalize_provider(exchange), status="active")
        if connection_id is not None:
            provider_rows = query.filter(
                (LeveragedMarket.trading_connection_id == int(connection_id)) | (LeveragedMarket.trading_connection_id.is_(None))
            ).order_by(LeveragedMarket.liquidity_usd.desc(), LeveragedMarket.id.asc())
        else:
            provider_rows = query.order_by(LeveragedMarket.liquidity_usd.desc(), LeveragedMarket.id.asc())
        return provider_rows.limit(50).all()

    def _market_warning(self, exchange: str, markets: list[LeveragedMarket]) -> dict[str, Any] | None:
        if not markets:
            return None
        stale_seconds = float(self.config.get("MARKET_DATA_LIVE_STALE_SECONDS", 90.0) or 90.0)
        newest = max((market.last_seen_at for market in markets if market.last_seen_at), default=None)
        if newest is None:
            return None
        age = (datetime.utcnow() - newest).total_seconds()
        if age > stale_seconds:
            return self._blocker(
                f"{exchange}_market_metadata_stale",
                f"{EXCHANGE_LABELS.get(exchange, exchange.title())} metadata stale",
                f"{EXCHANGE_LABELS.get(exchange, exchange.title())} market metadata is {int(age)} seconds old.",
                "warning",
                "Refresh market data before live execution.",
                exchange=exchange,
            )
        return None

    def _exchange_score(
        self,
        *,
        ready: bool,
        available_margin: float,
        markets: list[LeveragedMarket],
        exchange: str,
    ) -> int:
        if not ready:
            return 0
        liquidity = max((float(market.liquidity_usd or 0.0) for market in markets), default=0.0)
        spread = min((float(market.spread_bps or 0.0) for market in markets), default=5.0)
        fee = min((float(market.fee_bps or 0.0) for market in markets), default=4.0)
        score = 55.0
        score += min(25.0, available_margin / 10.0)
        score += min(12.0, liquidity / 100_000.0)
        score -= min(10.0, spread / 2.0)
        score -= min(5.0, fee / 2.0)
        if exchange == "kucoin":
            score += 1.0
        return max(1, min(99, int(round(score))))

    def _asset_usd_price(self, asset: str) -> tuple[float, dict[str, Any] | None]:
        asset_key = str(asset or "").upper()
        market_data = current_app.extensions.get("services", {}).get("market_data")
        price = asset_usd_price(
            asset_key, lambda key: float(market_data.get_mid_price(key, "live") or 0.0) if market_data is not None else 0.0
        )
        if price <= 0:
            return 0.0, self._blocker(
                "price_unavailable",
                "Price unavailable",
                f"USD price data is unavailable for {asset_key}.",
                "warning",
                "Use USDC/USDT or retry when market data is available.",
            )
        return price, None

    def _snapshot_available_margin(self, snapshot: Any, collateral_asset: str) -> float:
        collateral = str(collateral_asset or "").upper()
        stable_assets = {"USDC", "USDT"}
        best = 0.0
        fallback = 0.0
        for row in getattr(snapshot, "balances", []) or []:
            if not isinstance(row, dict):
                continue
            asset = str(row.get("asset") or "").upper()
            amount = self._safe_float(row.get("withdrawable", row.get("available", row.get("value", 0.0))), 0.0)
            value = self._safe_float(row.get("value", amount), amount)
            free_value = amount if asset in stable_assets else value
            if asset == collateral:
                best = max(best, free_value)
            elif asset in stable_assets and collateral in stable_assets:
                fallback = max(fallback, free_value)
        return max(best, fallback, 0.0)

    def _hyperliquid_auto_funding_state(
        self,
        *,
        user: User | None,
        connection: TradingConnection | None = None,
        deposit_asset: str,
        collateral_asset: str,
        settlement_asset: str,
        amount: float,
    ) -> dict[str, Any]:
        source_asset = str(deposit_asset or "").upper().strip()
        collateral = str(collateral_asset or "").upper().strip()
        settlement = str(settlement_asset or "").upper().strip()
        requested = self._safe_float(amount, 0.0)
        if not bool(self.config.get("VAULT_CYCLE_HYPERLIQUID_AUTO_FUNDING_ENABLED", False)):
            return {
                "ready": False,
                "code": "hyperliquid_auto_funding_disabled",
                "title": "Hyperliquid auto-funding disabled",
                "description": "Hyperliquid has no USDC collateral and server-side auto-funding is disabled.",
                "fix_hint": "Enable VAULT_CYCLE_HYPERLIQUID_AUTO_FUNDING_ENABLED after configuring the conversion and transfer route.",
            }
        if collateral != "USDC" or settlement != "USDC":
            return {
                "ready": False,
                "code": "hyperliquid_usdc_settlement_unavailable",
                "title": "Hyperliquid requires USDC",
                "description": "Hyperliquid auto-funding only supports USDC collateral and settlement.",
                "fix_hint": "Select USDC settlement for Hyperliquid.",
            }
        if source_asset not in {"USDC", "USDT"}:
            return {
                "ready": False,
                "code": "hyperliquid_auto_funding_source_unsupported",
                "title": "Unsupported funding asset",
                "description": "Hyperliquid auto-funding supports USDT or USDC wallet balances only.",
                "fix_hint": "Use USDT or USDC as the funding balance.",
            }
        if requested <= 0:
            return {
                "ready": False,
                "code": "amount_required",
                "title": "Amount required",
                "description": "Enter an amount before checking Hyperliquid auto-funding.",
                "fix_hint": "Enter an allocation amount greater than zero.",
            }
        max_auto_fund = self._safe_float(self.config.get("VAULT_CYCLE_HYPERLIQUID_AUTO_FUNDING_MAX_USD"), 0.0)
        if max_auto_fund > 0 and requested > max_auto_fund + 1e-9:
            return {
                "ready": False,
                "code": "hyperliquid_auto_funding_amount_above_cap",
                "title": "Auto-funding cap exceeded",
                "description": "The requested Hyperliquid funding amount is above the configured auto-funding cap.",
                "fix_hint": "Reduce the amount or raise VAULT_CYCLE_HYPERLIQUID_AUTO_FUNDING_MAX_USD after review.",
            }
        if user is None:
            return {
                "ready": False,
                "code": "user_missing",
                "title": "User required",
                "description": "Sign in before checking Hyperliquid auto-funding.",
                "fix_hint": "Sign in and retry.",
            }
        balance = self._wallet_balance(user.id, source_asset)
        if balance is None or float(balance.available_balance or 0.0) + 1e-9 < requested:
            return {
                "ready": False,
                "code": "hyperliquid_auto_funding_balance_unavailable",
                "title": f"{source_asset} balance unavailable",
                "description": f"The wallet does not have enough available {source_asset} to fund Hyperliquid.",
                "fix_hint": f"Add {source_asset} or reduce the allocation.",
            }
        if OneInchFundingConnector.enabled(self.config):
            route = self._oneinch_auto_funding_route(
                user=user,
                connection=connection,
                source_asset=source_asset,
                amount=requested,
            )
            if not route.ready:
                code = str(route.blockers[0]) if route.blockers else "oneinch_auto_funding_unavailable"
                return {
                    "ready": False,
                    "code": code,
                    "title": "1inch funding route unavailable",
                    "description": self._oneinch_blocker_description(code, route),
                    "fix_hint": "Configure 1inch, Arbitrum token contracts, signer transaction support, and a matching Hyperliquid wallet source.",
                }
            return {
                "ready": True,
                "code": "hyperliquid_oneinch_auto_funding_ready",
                "provider": "1inch",
                "network": route.network,
                "source_wallet_address": route.source_wallet_address,
            }
        if not self._conversion_connector_configured(user.id):
            return {
                "ready": False,
                "code": "hyperliquid_auto_funding_connector_missing",
                "title": "Funding connector missing",
                "description": "Hyperliquid auto-funding requires a verified server-side connector that can convert and withdraw USDC.",
                "fix_hint": "Configure VAULT_CYCLE_CONVERSION_USER_ID and VAULT_CYCLE_CONVERSION_CONNECTION_ID, or verify the configured funding provider.",
            }
        return {"ready": True, "code": "hyperliquid_auto_funding_ready"}

    def _oneinch_auto_funding_route(
        self,
        *,
        user: User,
        connection: TradingConnection | None,
        source_asset: str,
        amount: float,
    ) -> Any:
        account_address = str(getattr(connection, "wallet_address", "") or "").strip()
        wallet_custody = None
        if has_app_context():
            wallet_custody = current_app.extensions.get("services", {}).get("wallet_custody")
        connector = OneInchFundingConnector(
            self.config,
            user_id=user.id,
            wallet_custody=wallet_custody,
            hyperliquid_account_address=account_address,
        )
        return connector.route_check(
            from_asset=source_asset,
            to_asset="USDC",
            amount=amount,
            hyperliquid_account_address=account_address,
        )

    @staticmethod
    def _oneinch_blocker_description(code: str, route: Any) -> str:
        descriptions = {
            "oneinch_api_key_missing": "The server-side 1inch API key is not configured.",
            "oneinch_signer_transactions_disabled": "Server-side wallet conversion signing is disabled.",
            "wallet_real_custody_disabled": "Real wallet custody is disabled, so 1inch cannot sign the conversion.",
            "hyperliquid_bridge_source_wallet_mismatch": "The funding wallet address does not match the Hyperliquid account that would receive Bridge2 credit.",
            "oneinch_source_wallet_network_unsupported": "The configured custody adapter does not support the 1inch funding network.",
            "oneinch_network_unsupported": "The configured 1inch funding network has no chain id.",
            "oneinch_network_not_evm": "The configured 1inch funding network is not an EVM network.",
        }
        if str(code).startswith("oneinch_source_wallet_insufficient:"):
            return "No active on-chain wallet has enough confirmed funds for the configured 1inch network."
        if str(code).endswith("_contract_missing"):
            return "The source or destination token contract is missing for the configured 1inch network."
        blockers = ", ".join(str(item) for item in list(getattr(route, "blockers", []) or [])[:3])
        return descriptions.get(code, blockers or "The 1inch funding route is not ready.")

    def _conversion_connector_configured(self, user_id: int) -> bool:
        configured_user_id = int(
            self.config.get("VAULT_CYCLE_CONVERSION_USER_ID")
            or self.config.get("PLATFORM_TREASURY_CONVERSION_USER_ID")
            or 0
        )
        configured_connection_id = int(
            self.config.get("VAULT_CYCLE_CONVERSION_CONNECTION_ID")
            or self.config.get("PLATFORM_TREASURY_CONVERSION_CONNECTION_ID")
            or 0
        )
        if configured_user_id > 0 and configured_connection_id > 0:
            connection = db.session.get(TradingConnection, configured_connection_id)
            return bool(
                connection is not None
                and int(connection.user_id) == configured_user_id
                and normalize_provider(connection.provider) != "hyperliquid"
                and connection.is_active
                and connection.verification_status == "verified"
            )
        provider = normalize_provider(self.config.get("VAULT_CYCLE_CONVERSION_PROVIDER", "kucoin"))
        if provider == "hyperliquid":
            return False
        connection = self._connection_for(user_id, provider)
        return bool(connection is not None and connection.is_active and connection.verification_status == "verified")

    def _wallet_balance(self, user_id: int | None, asset: str) -> WalletBalance | None:
        if user_id is None:
            return None
        return WalletBalance.query.filter_by(user_id=int(user_id), asset=str(asset or "").upper()).one_or_none()

    def _enabled_exchanges(self, enabled_exchanges: list[str] | tuple[str, ...] | None) -> list[str]:
        if enabled_exchanges is None:
            return list(VAULT_READINESS_EXCHANGES)
        exchanges = [
            normalize_provider(exchange) for exchange in enabled_exchanges if normalize_provider(exchange) in VAULT_READINESS_EXCHANGES
        ]
        return list(dict.fromkeys(exchanges))

    def _trading_connections(self) -> Any:
        return current_app.extensions["services"]["trading_connections"]

    def _looks_placeholder(self, value: Any) -> bool:
        normalized = str(value or "").strip().lower().replace(" ", "_")
        if not normalized:
            return True
        if normalized in PLACEHOLDER_MARKERS:
            return True
        return normalized.startswith(("your_", "replace_", "example_"))

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed == parsed else default

    def _blocker(
        self,
        code: str,
        title: str,
        description: str,
        severity: str,
        fix_hint: str,
        *,
        exchange: str | None = None,
        docs_or_action_label: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": code,
            "title": title,
            "description": description,
            "severity": severity,
            "fix_hint": fix_hint,
        }
        if exchange:
            payload["exchange"] = exchange
        if docs_or_action_label:
            payload["docs_or_action_label"] = docs_or_action_label
        return payload

    def _verified_spendable_amount(self, user_id: int, asset: str) -> float | None:
        if not has_app_context():
            return None
        network = self._default_network(asset)
        try:
            custody = current_app.extensions.get("services", {}).get("wallet_custody")
            if custody is None or not getattr(custody, "enabled", False) or not custody.supports(asset, network):
                return None
            return float(custody.verified_spendable_amount(user_id, asset, network) or 0.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vault readiness on-chain spendable check failed closed for %s/%s: %s", asset, network, exc)
            return 0.0

    @staticmethod
    def _default_network(asset: str) -> str:
        asset_key = str(asset or "").upper().strip()
        if asset_key == "BTC":
            return "Bitcoin"
        if asset_key == "SOL":
            return "Solana"
        if asset_key == "XRP":
            return "XRP Ledger"
        return "Ethereum"

    def _log_exchange_decisions(self, exchange_status: dict[str, dict[str, Any]]) -> None:
        for exchange, status in exchange_status.items():
            blocker_codes = [str(item.get("code") or "") for item in list(status.get("blockers") or []) if item.get("code")]
            logger.info(
                "Vault readiness exchange=%s enabled=%s ready=%s score=%s allocation_pct=%s blockers=%s",
                exchange,
                bool(status.get("enabled")),
                bool(status.get("ready")),
                status.get("score", 0),
                status.get("allocation_pct", 0),
                blocker_codes,
            )
