"""Global treasury solvency and withdrawal gas-liability engine."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable

from flask import current_app, has_app_context

from ..extensions import db
from ..models import (
    PlatformTreasuryReserveJob,
    TreasuryAlert,
    TreasuryGasUsage,
    TreasuryReserveForecast,
    TreasuryReserveState,
    VaultCycleSettlement,
    WalletAddress,
    WalletAuditLog,
    WalletBalance,
    WalletWithdrawal,
)
from .wallet_custody import EVM_ASSETS, EVM_NETWORKS, EvmWalletAdapter


WITHDRAWAL_LIABILITY_STATUSES = {"pending_approval", "pending_submission", "pending_gas_topup", "queued_treasury_solvency"}
OPEN_RESERVE_JOB_STATUSES = {"pending", "retryable", "submitted", "paused"}


@dataclass(frozen=True, slots=True)
class LiabilityItem:
    source: str
    asset: str
    network: str
    count: int
    amount: float = 0.0


class DexReserveProvider:
    """Future DEX conversion interface. V1 intentionally fails closed."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("TREASURY_DEX_CONVERSIONS_ENABLED", False))

    def acquire_eth(self, *, amount_eth: float, network: str, max_slippage_bps: float) -> dict[str, Any]:
        if not self.enabled:
            return {
                "status": "disabled",
                "network": network,
                "amount_eth": max(0.0, float(amount_eth or 0.0)),
                "max_slippage_bps": max_slippage_bps,
                "reason": "TREASURY_DEX_CONVERSIONS_ENABLED is disabled",
            }
        raise RuntimeError("DEX treasury conversion is not configured for live execution.")


class TreasurySolvencyEngine:
    """Calculates global EVM withdrawal gas liability and enforces reserve safety."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.dex_provider = DexReserveProvider(config)

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("TREASURY_SOLVENCY_ENABLED", True)) and bool(
            self.config.get("PLATFORM_GAS_TREASURY_ENABLED", False)
        )

    def solvency_payload(self, *, network: str = "Ethereum", recalculate: bool = False) -> dict[str, Any]:
        state = self.recalculate(network=network, persist=True, create_alerts=True) if recalculate else self.latest_or_recalculate(network=network)
        forecasts = [self._forecast_payload(row) for row in self.latest_forecasts(network=network, limit=8)]
        alerts = [self._alert_payload(row) for row in self.latest_alerts(network=network, limit=12)]
        return {
            "enabled": self.enabled,
            "state": self._state_payload(state) if state else None,
            "forecasts": forecasts,
            "alerts": alerts,
            "dex_provider": {"enabled": self.dex_provider.enabled, "status": "configured" if self.dex_provider.enabled else "disabled"},
        }

    def latest_or_recalculate(self, *, network: str = "Ethereum") -> TreasuryReserveState | None:
        state = TreasuryReserveState.query.filter_by(network=network).one_or_none()
        max_age = max(1.0, float(self.config.get("TREASURY_SOLVENCY_RECALC_INTERVAL_SECONDS", 30.0) or 30.0))
        if state is None or not state.last_recalculated_at:
            return self.recalculate(network=network, persist=True, create_alerts=True)
        age = (datetime.utcnow() - state.last_recalculated_at).total_seconds()
        if age > max_age:
            return self.recalculate(network=network, persist=True, create_alerts=True)
        return state

    def recalculate(
        self,
        *,
        network: str = "Ethereum",
        persist: bool = True,
        create_alerts: bool = True,
    ) -> TreasuryReserveState:
        network_name = str(network or "Ethereum").strip() or "Ethereum"
        treasury = self._platform_treasury()
        wallet = treasury.active_wallet(network=network_name) if treasury is not None else None
        eth_balance = 0.0
        balance_error = ""
        if wallet is not None:
            try:
                eth_balance = max(0.0, float(treasury.eth_balance(wallet.address, wallet.network) or 0.0))
            except Exception as exc:  # noqa: BLE001
                balance_error = str(exc)

        gas_payload = self._gas_price_payload(network_name)
        items = self.liability_items(network=network_name)
        estimates = self._estimate_group_fees(items, network=network_name)
        raw_liability = sum(float(row.get("total_fee_eth", 0.0) or 0.0) for row in estimates)
        volatility = self.gas_volatility_score(network=network_name)
        multiplier = self.safety_multiplier(network=network_name, gas_price_wei=float(gas_payload.get("gas_price_wei", 0.0) or 0.0), volatility=volatility)
        total_liability = raw_liability * multiplier
        ratio = eth_balance / total_liability if total_liability > 0 else math.inf
        velocity = self.withdrawal_velocity_eth_per_hour(network=network_name)
        projected_runway = eth_balance / velocity if velocity > 0 else None
        target_ratio = self.target_reserve_ratio()
        target_reserve = total_liability * target_ratio
        pending_reserve = self.pending_reserve_eth(network=network_name)
        deficit = max(0.0, target_reserve - eth_balance - pending_reserve)
        status = self.health_status(ratio=ratio, liability=total_liability, balance=eth_balance, balance_error=balance_error)
        now = datetime.utcnow()

        state = TreasuryReserveState.query.filter_by(network=network_name).one_or_none() if persist else None
        if state is None:
            state = TreasuryReserveState(network=network_name)
            if persist:
                db.session.add(state)
        state.treasury_wallet_id = wallet.id if wallet is not None else None
        state.total_eth_balance = eth_balance
        state.total_estimated_liability = total_liability
        state.raw_estimated_liability = raw_liability
        state.reserve_ratio = 9999.0 if ratio == math.inf else ratio
        state.projected_runway = projected_runway
        state.health_status = status
        state.safety_multiplier = multiplier
        state.gas_price_wei = float(gas_payload.get("gas_price_wei", 0.0) or 0.0)
        state.gas_price_source = str(gas_payload.get("fee_source") or "")
        state.active_balance_count = sum(item.count for item in items if item.source == "wallet_balance")
        state.pending_withdrawal_count = sum(item.count for item in items if item.source == "pending_withdrawal")
        state.queued_withdrawal_count = self.queued_withdrawal_count(network=network_name)
        state.active_settlement_count = sum(item.count for item in items if item.source == "vault_settlement")
        state.target_reserve_eth = target_reserve
        state.deficit_eth = deficit
        state.last_recalculated_at = now
        state.details = {
            "network": network_name,
            "balance_error": balance_error,
            "gas_oracle": gas_payload,
            "gas_volatility_score": volatility,
            "withdrawal_velocity_eth_per_hour": velocity,
            "pending_reserve_eth": pending_reserve,
            "liability_groups": estimates,
            "liability_item_count": sum(item.count for item in items),
            "thresholds": self.thresholds(),
            "target_reserve_ratio": target_ratio,
        }
        if persist:
            db.session.flush()
            self.refresh_forecasts(state)
            if create_alerts:
                self._maybe_alert(state)
        return state

    def liability_items(self, *, network: str) -> list[LiabilityItem]:
        network_name = str(network or "Ethereum").strip() or "Ethereum"
        network_key = self._network_key(network_name)
        if network_key not in self.configured_evm_network_keys():
            return []
        grouped: dict[tuple[str, str, str], LiabilityItem] = {}

        def add(source: str, asset: str, amount: float = 0.0, count: int = 1) -> None:
            asset_key = str(asset or "").upper().strip()
            if not self._supports_asset(asset_key, network_name):
                return
            key = (source, asset_key, network_name)
            previous = grouped.get(key)
            if previous is None:
                grouped[key] = LiabilityItem(source=source, asset=asset_key, network=network_name, count=count, amount=max(0.0, amount))
            else:
                grouped[key] = LiabilityItem(
                    source=source,
                    asset=asset_key,
                    network=network_name,
                    count=previous.count + count,
                    amount=previous.amount + max(0.0, amount),
                )

        for balance in WalletBalance.query.filter((WalletBalance.available_balance > 0) | (WalletBalance.locked_balance > 0)).all():
            asset = str(balance.asset or "").upper().strip()
            if asset not in EVM_ASSETS:
                continue
            addresses = (
                WalletAddress.query.filter_by(
                    user_id=balance.user_id,
                    asset=asset,
                    status="active",
                )
                .order_by(WalletAddress.rotation_index.desc(), WalletAddress.created_at.desc())
                .all()
            )
            matching = [address for address in addresses if self._network_key(address.network) == network_key]
            if matching:
                add(
                    "wallet_balance",
                    asset,
                    float(balance.available_balance or 0.0) + float(balance.locked_balance or 0.0),
                    count=len(matching),
                )
                continue
            if self._network_key(self._default_network(asset)) == network_key:
                add("wallet_balance", asset, float(balance.available_balance or 0.0) + float(balance.locked_balance or 0.0))

        for withdrawal in WalletWithdrawal.query.filter(WalletWithdrawal.status.in_(WITHDRAWAL_LIABILITY_STATUSES)).all():
            if self._network_key(withdrawal.network) != network_key:
                continue
            add("pending_withdrawal", withdrawal.asset, float(withdrawal.amount or 0.0))

        settlement_rows = VaultCycleSettlement.query.filter(VaultCycleSettlement.status != "complete").all()
        for settlement in settlement_rows:
            asset = str(settlement.settlement_asset or "").upper().strip()
            if asset not in EVM_ASSETS:
                continue
            cycle = settlement.vault_cycle
            settlement_network = self._default_network(asset)
            if self._network_key(settlement_network) != network_key:
                continue
            amount = float(settlement.final_amount or 0.0)
            if amount <= 0:
                amount = float(settlement.final_value_usd or settlement.gross_value_usd or 0.0) / max(self._asset_usd_price(asset), 1e-9)
            if amount <= 0 and cycle is not None:
                amount = float(cycle.current_estimated_value_usd or 0.0) / max(self._asset_usd_price(asset), 1e-9)
            if amount > 0:
                add("vault_settlement", asset, amount)

        return list(grouped.values())

    def evaluate_withdrawal(
        self,
        withdrawal: WalletWithdrawal,
        *,
        projected_spend_eth: float = 0.0,
        estimated_gas_eth: float | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        if not self.enabled or self._network_key(withdrawal.network) not in self.configured_evm_network_keys():
            return {"safe": True, "status": "not_required", "reason": "", "reserve_ratio": None}
        state = self.recalculate(network=withdrawal.network, persist=True, create_alerts=True)
        spend = max(0.0, float(projected_spend_eth or 0.0))
        liability = max(float(state.total_estimated_liability or 0.0), 0.0)
        projected_balance = max(0.0, float(state.total_eth_balance or 0.0) - spend)
        projected_ratio = projected_balance / liability if liability > 0 else math.inf
        minimum_ratio = max(0.0, float(self.config.get("TREASURY_SOLVENCY_WITHDRAWAL_MIN_RATIO", 1.10) or 1.10))
        status = self.health_status(ratio=projected_ratio, liability=liability, balance=projected_balance, balance_error="")
        safe = status not in {"critical", "emergency"} and projected_ratio + 1e-12 >= minimum_ratio
        reason = "" if safe else (
            f"Treasury reserve ratio would fall to {projected_ratio:.2f}x; minimum safe ratio is {minimum_ratio:.2f}x."
        )
        payload = {
            "safe": safe,
            "status": "safe" if safe else "queued_treasury_solvency",
            "reason": reason,
            "network": withdrawal.network,
            "reserve_ratio": 9999.0 if projected_ratio == math.inf else projected_ratio,
            "health_status": status,
            "projected_spend_eth": spend,
            "estimated_gas_eth": max(0.0, float(estimated_gas_eth or 0.0)),
            "state_id": state.id,
        }
        if persist:
            self.apply_withdrawal_safety(withdrawal, payload)
        return payload

    def apply_withdrawal_safety(self, withdrawal: WalletWithdrawal, payload: dict[str, Any]) -> None:
        now = datetime.utcnow()
        withdrawal.treasury_safety_status = str(payload.get("status") or "unchecked")
        withdrawal.treasury_safety_reason = str(payload.get("reason") or "")
        withdrawal.treasury_estimated_gas_eth = float(payload.get("estimated_gas_eth", 0.0) or 0.0)
        withdrawal.treasury_safety_checked_at = now
        details = withdrawal.details
        details["treasury_safety"] = {**payload, "checked_at": now.isoformat()}
        withdrawal.details = details
        if not bool(payload.get("safe", False)):
            self.queue_withdrawal(withdrawal, reason=withdrawal.treasury_safety_reason or "Treasury reserve is not solvent enough for withdrawal.")

    def queue_withdrawal(self, withdrawal: WalletWithdrawal, *, reason: str) -> None:
        if withdrawal.status != "queued_treasury_solvency":
            details = withdrawal.details
            details.setdefault("return_status_after_solvency", withdrawal.status)
            details["queued_treasury_solvency_at"] = datetime.utcnow().isoformat()
            withdrawal.details = details
        withdrawal.status = "queued_treasury_solvency"
        withdrawal.failure_reason = reason
        self._audit(
            user_id=withdrawal.user_id,
            wallet_withdrawal_id=withdrawal.id,
            action="withdrawal_queued_treasury_solvency",
            status="queued_treasury_solvency",
            message=reason,
            metadata=withdrawal.details.get("treasury_safety", {}),
        )
        self.create_alert(
            network=withdrawal.network,
            severity="critical",
            event_type="withdrawal_queued_treasury_solvency",
            reserve_ratio=float((withdrawal.details.get("treasury_safety") or {}).get("reserve_ratio") or 0.0),
            health_status=str((withdrawal.details.get("treasury_safety") or {}).get("health_status") or ""),
            message=f"Withdrawal {withdrawal.id} queued because treasury solvency would be compromised.",
            metadata={"withdrawal_id": withdrawal.id, "user_id": withdrawal.user_id, "reason": reason},
        )

    def release_queued_withdrawal_if_safe(self, withdrawal: WalletWithdrawal) -> bool:
        if withdrawal.status != "queued_treasury_solvency":
            return True
        evaluation = self.evaluate_withdrawal(withdrawal, persist=True)
        if not evaluation.get("safe"):
            return False
        details = withdrawal.details
        next_status = str(details.get("return_status_after_solvency") or ("pending_submission" if withdrawal.approved_at else "pending_approval"))
        if next_status == "queued_treasury_solvency":
            next_status = "pending_submission" if withdrawal.approved_at else "pending_approval"
        withdrawal.status = next_status
        withdrawal.failure_reason = None
        details["released_from_treasury_solvency_at"] = datetime.utcnow().isoformat()
        withdrawal.details = details
        self._audit(
            user_id=withdrawal.user_id,
            wallet_withdrawal_id=withdrawal.id,
            action="withdrawal_released_treasury_solvency",
            status=withdrawal.status,
            message="Treasury reserve recovered enough to resume the withdrawal workflow.",
            metadata=evaluation,
        )
        return True

    def rebalance_if_needed(self, *, network: str = "Ethereum", force: bool = False) -> dict[str, Any]:
        if not self.enabled:
            return {"created": False, "status": "disabled"}
        state = self.recalculate(network=network, persist=True, create_alerts=True)
        deficit = max(0.0, float(state.deficit_eth or 0.0))
        existing = self._active_rebalance_job(network=network)
        if existing is not None and not force:
            return {"created": False, "status": "existing", "job": self._job_payload(existing), "state": self._state_payload(state)}
        min_eth = max(0.0, float(self.config.get("TREASURY_REBALANCE_MIN_ETH", 0.0) or 0.0))
        if deficit <= min_eth and not force:
            return {"created": False, "status": "not_needed", "deficit_eth": deficit, "state": self._state_payload(state)}
        max_eth = max(0.0, float(self.config.get("TREASURY_REBALANCE_MAX_ETH", 0.0) or 0.0))
        target_eth = deficit if max_eth <= 0 else min(deficit, max_eth)
        if target_eth <= 0:
            return {"created": False, "status": "not_needed", "deficit_eth": deficit, "state": self._state_payload(state)}
        source_asset = str(self.config.get("TREASURY_REBALANCE_SOURCE_ASSET", "USDC") or "USDC").upper()
        conversion_amount = target_eth * self._asset_usd_price("ETH") / max(self._asset_usd_price(source_asset), 1e-9)
        treasury = self._platform_treasury()
        wallet = treasury.active_wallet(network=network) if treasury is not None else None
        job = PlatformTreasuryReserveJob(
            treasury_wallet_id=wallet.id if wallet is not None else None,
            job_type="solvency_rebalance",
            status="pending",
            asset=source_asset,
            network=network,
            source_amount=conversion_amount,
            source_amount_usd=conversion_amount * self._asset_usd_price(source_asset),
            reserve_eth_estimate=target_eth,
            reserve_multiplier=1.0,
            reserve_eth_target=target_eth,
            conversion_provider=str(self.config.get("PLATFORM_TREASURY_CONVERSION_PROVIDER", "kucoin") or "kucoin"),
            conversion_asset=source_asset,
            conversion_amount=conversion_amount,
            idempotency_key=f"treasury:solvency-rebalance:{network}:{state.id}:{int(time.time() // 60)}",
            max_attempts=max(1, int(self.config.get("PLATFORM_TREASURY_RESERVE_JOB_MAX_ATTEMPTS", 3) or 3)),
        )
        job.details = {
            "state_id": state.id,
            "deficit_eth": deficit,
            "target_eth": target_eth,
            "target_reserve_ratio": self.target_reserve_ratio(),
            "max_slippage_bps": self.rebalance_slippage_bps(),
            "conversion_mode": "cex",
            "dex_interface": self.dex_provider.acquire_eth(amount_eth=target_eth, network=network, max_slippage_bps=self.rebalance_slippage_bps()),
        }
        db.session.add(job)
        db.session.flush()
        self.create_alert(
            network=network,
            severity="warning",
            event_type="treasury_rebalance_queued",
            reserve_ratio=float(state.reserve_ratio or 0.0),
            health_status=state.health_status,
            message=f"Queued ETH reserve acquisition for {target_eth:.8f} ETH.",
            metadata={"job_id": job.id, "deficit_eth": deficit, "source_asset": source_asset, "conversion_amount": conversion_amount},
        )
        return {"created": True, "status": "queued", "job": self._job_payload(job), "state": self._state_payload(state)}

    def refresh_forecasts(self, state: TreasuryReserveState) -> list[TreasuryReserveForecast]:
        windows = self.forecast_windows()
        velocity = self.withdrawal_velocity_eth_per_hour(network=state.network)
        volatility = self.gas_volatility_score(network=state.network)
        rows: list[TreasuryReserveForecast] = []
        for label, hours in windows:
            projected_liability = float(state.total_estimated_liability or 0.0) + (velocity * hours * (1.0 + volatility))
            projected_reserve = max(0.0, float(state.total_eth_balance or 0.0) - velocity * hours)
            ratio = projected_reserve / projected_liability if projected_liability > 0 else math.inf
            risk = self.health_status(ratio=ratio, liability=projected_liability, balance=projected_reserve, balance_error="")
            risk_probability = self._risk_probability(ratio=ratio, volatility=volatility, hours=hours)
            runway = projected_reserve / velocity if velocity > 0 else None
            forecast = TreasuryReserveForecast(
                network=state.network,
                projected_liability=projected_liability,
                projected_reserve=projected_reserve,
                forecast_window=label,
                risk_level=risk,
                reserve_runway_hours=runway,
                withdrawal_velocity_eth_per_hour=velocity,
                gas_volatility_score=volatility,
                risk_probability=risk_probability,
            )
            forecast.details = {"state_id": state.id, "hours": hours, "reserve_ratio": 9999.0 if ratio == math.inf else ratio}
            db.session.add(forecast)
            rows.append(forecast)
        db.session.flush()
        self._prune_forecasts(state.network)
        return rows

    def record_withdrawal_gas_usage(self, withdrawal: WalletWithdrawal, raw: dict[str, Any] | None = None, *, receipt: dict[str, Any] | None = None) -> None:
        raw = raw or {}
        receipt = receipt or {}
        tx_hash = str(raw.get("tx_hash") or withdrawal.provider_reference or "")
        gas_used = self._hex_or_float(receipt.get("gasUsed")) or float(raw.get("gas_limit", 0.0) or 0.0)
        gas_price = self._hex_or_float(receipt.get("effectiveGasPrice")) or float(raw.get("gas_price_wei", 0.0) or 0.0)
        gas_fee_eth = gas_used * gas_price / 10**18 if gas_used > 0 and gas_price > 0 else float(withdrawal.treasury_estimated_gas_eth or 0.0)
        existing = TreasuryGasUsage.query.filter_by(withdrawal_id=withdrawal.id, tx_hash=tx_hash or None).one_or_none()
        if existing is None:
            existing = TreasuryGasUsage(withdrawal_id=withdrawal.id, user_id=withdrawal.user_id, tx_hash=tx_hash or None)
            db.session.add(existing)
        existing.asset = str(withdrawal.asset or "ETH").upper()
        existing.network = str(withdrawal.network or "Ethereum")
        existing.gas_used = gas_used
        existing.gas_price = gas_price
        existing.gas_fee_eth = gas_fee_eth
        existing.status = "complete" if receipt else str(withdrawal.status or "submitted")
        existing.details = {"raw": raw, "receipt": receipt}

    def latest_forecasts(self, *, network: str = "Ethereum", limit: int = 8) -> list[TreasuryReserveForecast]:
        return (
            TreasuryReserveForecast.query.filter_by(network=network)
            .order_by(TreasuryReserveForecast.created_at.desc(), TreasuryReserveForecast.id.desc())
            .limit(max(1, int(limit or 8)))
            .all()
        )

    def latest_alerts(self, *, network: str = "Ethereum", limit: int = 12) -> list[TreasuryAlert]:
        return (
            TreasuryAlert.query.filter_by(network=network)
            .order_by(TreasuryAlert.created_at.desc(), TreasuryAlert.id.desc())
            .limit(max(1, int(limit or 12)))
            .all()
        )

    def create_alert(
        self,
        *,
        network: str,
        severity: str,
        event_type: str,
        reserve_ratio: float,
        health_status: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> TreasuryAlert:
        window_start = datetime.utcnow() - timedelta(minutes=5)
        existing = (
            TreasuryAlert.query.filter(
                TreasuryAlert.network == network,
                TreasuryAlert.event_type == event_type,
                TreasuryAlert.created_at >= window_start,
            )
            .order_by(TreasuryAlert.created_at.desc())
            .first()
        )
        if existing is not None:
            return existing
        alert = TreasuryAlert(
            network=network,
            severity=severity,
            event_type=event_type,
            reserve_ratio=max(0.0, float(reserve_ratio or 0.0)),
            health_status=health_status,
            message=message,
        )
        alert.details = metadata or {}
        db.session.add(alert)
        db.session.flush()
        return alert

    def pending_reserve_eth(self, *, network: str) -> float:
        return sum(
            float(job.reserve_eth_target or 0.0)
            for job in PlatformTreasuryReserveJob.query.filter(
                PlatformTreasuryReserveJob.network == network,
                PlatformTreasuryReserveJob.status.in_(OPEN_RESERVE_JOB_STATUSES),
            ).all()
        )

    def withdrawal_velocity_eth_per_hour(self, *, network: str) -> float:
        since = datetime.utcnow() - timedelta(hours=max(1.0, float(self.config.get("TREASURY_FORECAST_HISTORY_HOURS", 168.0) or 168.0)))
        usage = sum(
            float(row.gas_fee_eth or 0.0)
            for row in TreasuryGasUsage.query.filter(
                TreasuryGasUsage.network == network,
                TreasuryGasUsage.created_at >= since,
            ).all()
        )
        topups = sum(
            float(event.amount or 0.0)
            for event in self._topup_events(network=network, since=since)
        )
        elapsed_hours = max((datetime.utcnow() - since).total_seconds() / 3600.0, 1.0)
        return (usage + topups) / elapsed_hours

    def gas_volatility_score(self, *, network: str) -> float:
        since = datetime.utcnow() - timedelta(hours=max(1.0, float(self.config.get("TREASURY_GAS_VOLATILITY_WINDOW_HOURS", 24.0) or 24.0)))
        values = [
            float(row.gas_price or 0.0)
            for row in TreasuryGasUsage.query.filter(
                TreasuryGasUsage.network == network,
                TreasuryGasUsage.created_at >= since,
                TreasuryGasUsage.gas_price > 0,
            ).all()
        ]
        if len(values) < 2:
            return 0.0
        mean = statistics.fmean(values)
        if mean <= 0:
            return 0.0
        return max(0.0, min(statistics.pstdev(values) / mean, float(self.config.get("TREASURY_GAS_VOLATILITY_CAP", 0.75) or 0.75)))

    def safety_multiplier(self, *, network: str, gas_price_wei: float, volatility: float) -> float:
        base = max(1.0, float(self.config.get("TREASURY_SOLVENCY_BASE_SAFETY_MULTIPLIER", 1.20) or 1.20))
        min_gwei = max(0.0, float(self.config.get("WALLET_EVM_MIN_GAS_PRICE_GWEI", 2.0) or 0.0))
        min_wei = min_gwei * 10**9
        congestion = gas_price_wei / min_wei if min_wei > 0 and gas_price_wei > 0 else 1.0
        congestion = max(1.0, min(congestion, float(self.config.get("TREASURY_GAS_CONGESTION_MULTIPLIER_CAP", 3.0) or 3.0)))
        return base * congestion * (1.0 + max(0.0, volatility))

    def configured_evm_network_keys(self) -> set[str]:
        keys = {"ETHEREUM"}
        configured = self.config.get("WALLET_EVM_NETWORKS") or {}
        if isinstance(configured, dict):
            keys.update(self._network_key(key) for key in configured)
        for row in WalletAddress.query.with_entities(WalletAddress.network).distinct().all():
            key = self._network_key(row[0])
            if key in EVM_NETWORKS:
                keys.add(key)
        return keys & set(EVM_NETWORKS)

    def target_reserve_ratio(self) -> float:
        return max(1.10, float(self.config.get("TREASURY_REBALANCE_TARGET_RATIO", 3.0) or 3.0))

    def thresholds(self) -> dict[str, float]:
        return {
            "healthy": max(0.0, float(self.config.get("TREASURY_HEALTHY_RATIO", 3.0) or 3.0)),
            "warning": max(0.0, float(self.config.get("TREASURY_WARNING_RATIO", 1.5) or 1.5)),
            "low": max(0.0, float(self.config.get("TREASURY_LOW_RATIO", 1.10) or 1.10)),
        }

    def health_status(self, *, ratio: float, liability: float, balance: float, balance_error: str) -> str:
        if balance_error:
            return "emergency"
        if liability <= 0:
            return "healthy" if balance >= 0 else "emergency"
        thresholds = self.thresholds()
        if balance <= 0:
            return "emergency"
        if ratio >= thresholds["healthy"]:
            return "healthy"
        if ratio >= thresholds["warning"]:
            return "warning"
        if ratio >= thresholds["low"]:
            return "low"
        return "critical"

    def queued_withdrawal_count(self, *, network: str) -> int:
        return WalletWithdrawal.query.filter_by(network=network, status="queued_treasury_solvency").count()

    def rebalance_slippage_bps(self) -> float:
        return max(0.0, float(self.config.get("TREASURY_REBALANCE_MAX_SLIPPAGE_BPS", 10.0) or 10.0))

    def forecast_windows(self) -> list[tuple[str, float]]:
        raw = str(self.config.get("TREASURY_FORECAST_WINDOWS_HOURS", "1,6,24,168") or "1,6,24,168")
        windows: list[tuple[str, float]] = []
        for part in raw.split(","):
            try:
                hours = max(0.25, float(part.strip()))
            except ValueError:
                continue
            label = f"{int(hours)}h" if hours >= 1 and float(hours).is_integer() else f"{hours:g}h"
            windows.append((label, hours))
        return windows or [("1h", 1.0), ("24h", 24.0), ("168h", 168.0)]

    def _estimate_group_fees(self, items: list[LiabilityItem], *, network: str) -> list[dict[str, Any]]:
        adapter = EvmWalletAdapter(self.config)
        rows: list[dict[str, Any]] = []
        destination = self._estimate_destination(network)
        for item in items:
            try:
                sample_amount = max(item.amount / max(item.count, 1), 1e-9)
                fee = max(0.0, float(adapter.estimate_fee(item.asset, item.network, destination, sample_amount) or 0.0))
                error = ""
            except Exception as exc:  # noqa: BLE001
                fee = 0.0
                error = str(exc)
            rows.append(
                {
                    "source": item.source,
                    "asset": item.asset,
                    "network": item.network,
                    "count": item.count,
                    "sample_amount": item.amount / max(item.count, 1),
                    "fee_per_withdrawal_eth": fee,
                    "total_fee_eth": fee * item.count,
                    "estimate_error": error,
                }
            )
        return rows

    def _network_for_balance(self, balance: WalletBalance) -> str:
        address = (
            WalletAddress.query.filter_by(
                user_id=balance.user_id,
                asset=str(balance.asset or "").upper(),
                status="active",
            )
            .order_by(WalletAddress.rotation_index.desc(), WalletAddress.created_at.desc())
            .first()
        )
        if address is not None:
            return str(address.network or self._default_network(balance.asset))
        return self._default_network(balance.asset)

    def _supports_asset(self, asset: str, network: str) -> bool:
        asset_key = str(asset or "").upper().strip()
        if self._network_key(network) not in self.configured_evm_network_keys():
            return False
        if asset_key == "ETH":
            return True
        if asset_key not in EVM_ASSETS:
            return False
        return bool(EvmWalletAdapter(self.config)._token_contract(asset_key, network))  # noqa: SLF001

    def _estimate_destination(self, network: str) -> str:
        treasury = self._platform_treasury()
        wallet = treasury.active_wallet(network=network) if treasury is not None else None
        if wallet is not None:
            return wallet.address
        configured = str(self.config.get("WALLET_EVM_ESTIMATE_FROM_ADDRESS", "") or "").strip()
        if configured:
            return configured
        return "0x0000000000000000000000000000000000000001"

    def _default_network(self, asset: str) -> str:
        mapping = self.config.get("VAULT_CYCLE_DEFAULT_NETWORK_BY_ASSET") or {}
        asset_key = str(asset or "").upper().strip()
        if isinstance(mapping, dict):
            value = mapping.get(asset_key) or mapping.get(asset_key.lower())
            if value:
                return str(value)
        return "Ethereum" if asset_key in EVM_ASSETS else "native"

    def _gas_price_payload(self, network: str) -> dict[str, Any]:
        try:
            return dict(EvmWalletAdapter(self.config)._gas_price_payload(network))  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            return {"gas_price_wei": 0.0, "fee_source": "unavailable", "error": str(exc)}

    def _active_rebalance_job(self, *, network: str) -> PlatformTreasuryReserveJob | None:
        return (
            PlatformTreasuryReserveJob.query.filter(
                PlatformTreasuryReserveJob.network == network,
                PlatformTreasuryReserveJob.job_type == "solvency_rebalance",
                PlatformTreasuryReserveJob.status.in_(OPEN_RESERVE_JOB_STATUSES),
            )
            .order_by(PlatformTreasuryReserveJob.created_at.desc())
            .first()
        )

    def _maybe_alert(self, state: TreasuryReserveState) -> None:
        if state.health_status == "healthy":
            return
        severity = "critical" if state.health_status in {"critical", "emergency"} else "warning"
        self.create_alert(
            network=state.network,
            severity=severity,
            event_type=f"reserve_{state.health_status}",
            reserve_ratio=float(state.reserve_ratio or 0.0),
            health_status=state.health_status,
            message=f"Treasury reserve is {state.health_status}; reserve ratio is {float(state.reserve_ratio or 0.0):.2f}x.",
            metadata={"state_id": state.id, "deficit_eth": state.deficit_eth, "liability_eth": state.total_estimated_liability},
        )

    def _topup_events(self, *, network: str, since: datetime) -> Iterable[Any]:
        from ..models import PlatformTreasuryEvent

        return PlatformTreasuryEvent.query.filter(
            PlatformTreasuryEvent.network == network,
            PlatformTreasuryEvent.event_type == "withdrawal_gas_topup",
            PlatformTreasuryEvent.status.in_(("submitted", "complete")),
            PlatformTreasuryEvent.created_at >= since,
        ).all()

    def _asset_usd_price(self, asset: str) -> float:
        asset_key = str(asset or "").upper().strip()
        if asset_key in {"USDC", "USDT", "USD"}:
            return 1.0
        fallback = max(0.0, float(self.config.get("PLATFORM_TREASURY_ETH_USD_FALLBACK", 3000.0) or 0.0)) if asset_key == "ETH" else 1.0
        if has_app_context():
            try:
                price = float(current_app.extensions["services"]["market_data"].get_mid_price(asset_key, "live") or 0.0)
                if price > 0:
                    return price
            except Exception:
                pass
        return fallback

    def _platform_treasury(self) -> Any | None:
        if not has_app_context():
            return None
        return current_app.extensions.get("services", {}).get("platform_treasury")

    def _audit(
        self,
        *,
        action: str,
        status: str,
        message: str,
        user_id: int | None = None,
        wallet_withdrawal_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        audit = WalletAuditLog(
            user_id=user_id,
            wallet_withdrawal_id=wallet_withdrawal_id,
            category="treasury_solvency",
            action=action,
            status=status,
            message=message,
        )
        audit.details = metadata or {}
        db.session.add(audit)

    def _prune_forecasts(self, network: str) -> None:
        keep = max(24, int(self.config.get("TREASURY_FORECAST_RETENTION_ROWS", 96) or 96))
        rows = (
            TreasuryReserveForecast.query.filter_by(network=network)
            .order_by(TreasuryReserveForecast.created_at.desc(), TreasuryReserveForecast.id.desc())
            .offset(keep)
            .limit(500)
            .all()
        )
        for row in rows:
            db.session.delete(row)

    @staticmethod
    def _risk_probability(*, ratio: float, volatility: float, hours: float) -> float:
        if ratio == math.inf:
            return 0.0
        ratio_risk = max(0.0, min(1.0, (3.0 - ratio) / 3.0))
        time_risk = min(1.0, hours / 168.0)
        return max(0.0, min(1.0, ratio_risk * 0.75 + volatility * 0.2 + time_risk * 0.05))

    @staticmethod
    def _hex_or_float(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            if isinstance(value, str) and value.startswith("0x"):
                return float(int(value, 16))
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _network_key(network: str) -> str:
        return "".join(ch for ch in str(network or "").upper() if ch.isalnum())

    @staticmethod
    def _state_payload(state: TreasuryReserveState) -> dict[str, Any]:
        return {
            "id": state.id,
            "network": state.network,
            "treasury_wallet_id": state.treasury_wallet_id,
            "total_eth_balance": float(state.total_eth_balance or 0.0),
            "total_estimated_liability": float(state.total_estimated_liability or 0.0),
            "raw_estimated_liability": float(state.raw_estimated_liability or 0.0),
            "reserve_ratio": float(state.reserve_ratio or 0.0),
            "projected_runway": state.projected_runway,
            "health_status": state.health_status,
            "safety_multiplier": float(state.safety_multiplier or 0.0),
            "gas_price_wei": float(state.gas_price_wei or 0.0),
            "gas_price_source": state.gas_price_source,
            "active_balance_count": int(state.active_balance_count or 0),
            "pending_withdrawal_count": int(state.pending_withdrawal_count or 0),
            "queued_withdrawal_count": int(state.queued_withdrawal_count or 0),
            "active_settlement_count": int(state.active_settlement_count or 0),
            "target_reserve_eth": float(state.target_reserve_eth or 0.0),
            "deficit_eth": float(state.deficit_eth or 0.0),
            "details": state.details,
            "last_recalculated_at": state.last_recalculated_at.isoformat() if state.last_recalculated_at else None,
        }

    @staticmethod
    def _forecast_payload(row: TreasuryReserveForecast) -> dict[str, Any]:
        return {
            "id": row.id,
            "network": row.network,
            "projected_liability": float(row.projected_liability or 0.0),
            "projected_reserve": float(row.projected_reserve or 0.0),
            "forecast_window": row.forecast_window,
            "risk_level": row.risk_level,
            "reserve_runway_hours": row.reserve_runway_hours,
            "withdrawal_velocity_eth_per_hour": float(row.withdrawal_velocity_eth_per_hour or 0.0),
            "gas_volatility_score": float(row.gas_volatility_score or 0.0),
            "risk_probability": float(row.risk_probability or 0.0),
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    @staticmethod
    def _alert_payload(row: TreasuryAlert) -> dict[str, Any]:
        return {
            "id": row.id,
            "network": row.network,
            "severity": row.severity,
            "event_type": row.event_type,
            "reserve_ratio": float(row.reserve_ratio or 0.0),
            "health_status": row.health_status,
            "message": row.message,
            "details": row.details,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "acknowledged_at": row.acknowledged_at.isoformat() if row.acknowledged_at else None,
        }

    @staticmethod
    def _job_payload(job: PlatformTreasuryReserveJob) -> dict[str, Any]:
        return {
            "id": job.id,
            "job_type": job.job_type,
            "status": job.status,
            "network": job.network,
            "conversion_asset": job.conversion_asset,
            "conversion_amount": float(job.conversion_amount or 0.0),
            "reserve_eth_target": float(job.reserve_eth_target or 0.0),
            "created_at": job.created_at.isoformat() if job.created_at else None,
        }

    @staticmethod
    def event_stream(payload_factory: Any, *, once: bool = False, interval: float = 10.0) -> Iterable[str]:
        def event(name: str, payload: dict[str, Any]) -> str:
            return f"event: {name}\ndata: {json.dumps(payload, default=str)}\n\n"

        yield event("heartbeat", {"at": datetime.utcnow().isoformat(), "status": "connected"})
        yield event("solvency", payload_factory())
        if once:
            return
        while True:
            time.sleep(max(2.0, interval))
            yield event("solvency", payload_factory())
            yield event("heartbeat", {"at": datetime.utcnow().isoformat(), "status": "ok"})
