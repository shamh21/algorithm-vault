"""Platform gas treasury support for EVM withdrawal funding."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from eth_account import Account
from flask import current_app, has_app_context

from ..extensions import db
from ..models import (
    PlatformTreasuryEvent,
    PlatformTreasuryReserveJob,
    PlatformTreasuryWallet,
    ReferralInviteCode,
    Setting,
    User,
    VaultCycle,
    VaultCycleSettlement,
    WalletAddress,
    WalletAuditLog,
    WalletBalance,
    WalletLedgerEvent,
    WalletWithdrawal,
)
from .wallet_custody import EvmWalletAdapter


class PlatformTreasuryService:
    """Manages a platform ETH treasury used only for withdrawal gas top-ups."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("PLATFORM_GAS_TREASURY_ENABLED", False))

    def status(self, *, network: str = "Ethereum", include_events: bool = True) -> dict[str, Any]:
        wallet = self.active_wallet(network=network)
        balance = None
        balance_error = ""
        if wallet is not None:
            try:
                balance = self.eth_balance(wallet.address, wallet.network)
            except Exception as exc:  # noqa: BLE001
                balance_error = str(exc)
        events: list[dict[str, Any]] = []
        reserve_jobs: list[dict[str, Any]] = []
        if include_events:
            for event in PlatformTreasuryEvent.query.order_by(PlatformTreasuryEvent.created_at.desc()).limit(20).all():
                events.append(self._event_payload(event))
            for job in PlatformTreasuryReserveJob.query.order_by(PlatformTreasuryReserveJob.created_at.desc()).limit(50).all():
                reserve_jobs.append(self._job_payload(job))
        blockers = []
        if not self.enabled:
            blockers.append("PLATFORM_GAS_TREASURY_ENABLED is disabled")
        if not self._has_valid_key():
            blockers.append("TREASURY_ENCRYPTION_KEY must be a valid Fernet key")
        if wallet is None:
            blockers.append("active platform treasury wallet is not configured")
        if self.paused:
            blockers.append("platform treasury emergency pause is active")
        analytics = self.analytics(network=network, eth_balance=balance)
        solvency = self._solvency_payload(network=network)
        reserve_health = self._reserve_health(analytics=analytics, eth_balance=balance, blockers=blockers)
        if solvency.get("state"):
            state = solvency["state"]
            reserve_health = {
                **reserve_health,
                "state": "emergency" if blockers and state.get("health_status") == "healthy" else state.get("health_status", reserve_health["state"]),
                "balance_eth": float(state.get("total_eth_balance", reserve_health["balance_eth"]) or 0.0),
                "total_estimated_liability": float(state.get("total_estimated_liability", 0.0) or 0.0),
                "reserve_ratio": float(state.get("reserve_ratio", 0.0) or 0.0),
                "projected_runway": state.get("projected_runway"),
                "deficit_eth": float(state.get("deficit_eth", 0.0) or 0.0),
            }
        return {
            "enabled": self.enabled,
            "ready": self.enabled and self._has_valid_key() and wallet is not None and not self.paused,
            "paused": self.paused,
            "reserve_health": reserve_health,
            "blockers": blockers,
            "active_wallet": self._wallet_payload(wallet) if wallet is not None else None,
            "eth_balance": balance,
            "balance_error": balance_error,
            "analytics": analytics,
            "solvency": solvency,
            "reserve_jobs": reserve_jobs,
            "events": events,
        }

    @property
    def paused(self) -> bool:
        return bool(Setting.get_json("platform_treasury_paused", False))

    def set_paused(self, paused: bool, *, user_id: int | None = None) -> None:
        Setting.set_json("platform_treasury_paused", bool(paused))
        self._audit(
            user_id=user_id,
            action="platform_treasury_paused" if paused else "platform_treasury_resumed",
            status="paused" if paused else "active",
            message="Platform treasury emergency pause updated.",
            metadata={"paused": bool(paused)},
        )

    def analytics(self, *, network: str = "Ethereum", eth_balance: float | None = None) -> dict[str, Any]:
        since = datetime.utcnow() - timedelta(days=30)
        events = (
            PlatformTreasuryEvent.query.filter(
                PlatformTreasuryEvent.created_at >= since,
                PlatformTreasuryEvent.network == network,
            )
            .order_by(PlatformTreasuryEvent.created_at.desc())
            .limit(1000)
            .all()
        )
        jobs = (
            PlatformTreasuryReserveJob.query.filter_by(network=network)
            .order_by(PlatformTreasuryReserveJob.created_at.desc())
            .limit(500)
            .all()
        )
        reserve_committed = sum(
            float(event.gas_reserve_contribution or 0.0)
            for event in events
            if event.event_type in {"deposit_gas_reserve_deducted", "vault_gas_reserve_deducted"} and event.status in {"complete"}
        )
        reserve_in = sum(
            float(event.gas_reserve_contribution or 0.0)
            for event in events
            if (
                event.event_type.endswith("_converted_to_eth")
                or event.event_type.endswith("_eth_reserved")
            )
            and event.status in {"submitted", "complete"}
        )
        withdrawal_topups = sum(
            float(event.amount or 0.0)
            for event in events
            if event.event_type == "withdrawal_gas_topup" and event.status in {"submitted", "complete"}
        )
        completed_reserve_jobs = sum(float(job.converted_eth_amount or job.reserve_eth_target or 0.0) for job in jobs if job.status == "complete")
        pending_jobs = sum(1 for job in jobs if job.status in {"pending", "retryable", "submitted", "paused"})
        completed_withdrawal_count = sum(1 for event in events if event.event_type == "withdrawal_gas_topup" and event.status in {"submitted", "complete"})
        avg_topup = withdrawal_topups / max(completed_withdrawal_count, 1)
        balance = max(0.0, float(eth_balance or 0.0))
        runway = balance / avg_topup if avg_topup > 0 else None
        pending_reserve_eth = sum(float(job.reserve_eth_target or 0.0) for job in jobs if job.status in {"pending", "retryable", "submitted", "paused"})
        reserve_utilization = withdrawal_topups / reserve_in if reserve_in > 0 else 0.0
        referral_total = sum(
            float(job.conversion_amount or 0.0)
            for job in jobs
            if job.job_type == "vault_profit_share" and job.status in {"pending", "retryable", "submitted", "complete"}
        )
        recent_depletion = sum(
            float(event.amount or 0.0)
            for event in events
            if event.event_type == "withdrawal_gas_topup"
            and event.status in {"submitted", "complete"}
            and event.created_at >= datetime.utcnow() - timedelta(days=7)
        )
        return {
            "reserve_committed_eth_30d": reserve_committed,
            "reserve_in_eth_30d": reserve_in,
            "reserve_jobs_eth": completed_reserve_jobs,
            "pending_reserve_eth": pending_reserve_eth,
            "net_reserve_eth_30d": reserve_in - withdrawal_topups,
            "reserve_utilization_pct_30d": reserve_utilization * 100.0,
            "reserve_depletion_eth_7d": recent_depletion,
            "withdrawal_topups_eth_30d": withdrawal_topups,
            "pending_jobs": pending_jobs,
            "retryable_jobs": sum(1 for job in jobs if job.status == "retryable"),
            "failed_jobs": sum(1 for job in jobs if job.status == "failed"),
            "queue_waiting": WalletWithdrawal.query.filter(
                WalletWithdrawal.network == network,
                WalletWithdrawal.status.in_(["pending_gas_topup", "queued_treasury_solvency"]),
            ).count(),
            "avg_topup_eth": avg_topup,
            "gas_runway_withdrawals": runway,
            "referral_allocation_source_total": referral_total,
            "network": network,
        }

    def active_wallet(self, *, network: str = "Ethereum") -> PlatformTreasuryWallet | None:
        return (
            PlatformTreasuryWallet.query.filter_by(network=network, is_active=True)
            .order_by(PlatformTreasuryWallet.rotation_index.desc(), PlatformTreasuryWallet.id.desc())
            .first()
        )

    def estimate_withdrawal_gas(self, *, asset: str, network: str, amount: float, destination: str | None = None) -> dict[str, Any]:
        asset_key = str(asset or "").upper().strip()
        network_name = str(network or "Ethereum").strip() or "Ethereum"
        treasury = self.active_wallet(network=network_name)
        destination_address = destination or (treasury.address if treasury is not None else "0x0000000000000000000000000000000000000001")
        adapter = EvmWalletAdapter(self.config)
        fee = max(0.0, float(adapter.estimate_fee(asset_key, network_name, destination_address, max(0.0, float(amount or 0.0))) or 0.0))
        multiplier = float(self.config.get("PLATFORM_GAS_TOPUP_MULTIPLIER", 2.0) or 2.0)
        return {
            "asset": asset_key,
            "network": network_name,
            "estimated_fee_eth": fee,
            "reserve_multiplier": multiplier,
            "reserve_eth_target": fee * multiplier,
            "destination": destination_address,
        }

    def reserve_for_deposit(self, wallet_address: WalletAddress, ledger_event: WalletLedgerEvent, *, amount: float) -> PlatformTreasuryReserveJob | None:
        """Deduct and queue an ETH reserve after a confirmed EVM deposit credit."""

        if not self.enabled:
            return None
        asset = str(wallet_address.asset or ledger_event.asset or "").upper().strip()
        network = str(wallet_address.network or ledger_event.network or "Ethereum").strip() or "Ethereum"
        if self._network_key(network) not in {"ETHEREUM", "ARBITRUM", "OPTIMISM", "BASE", "POLYGON", "AVALANCHE", "BSC"}:
            return None
        if asset not in {"ETH", "USDC", "USDT"}:
            return None
        existing_job = PlatformTreasuryReserveJob.query.filter_by(idempotency_key=f"treasury:deposit-reserve:{ledger_event.id}").one_or_none()
        if existing_job is not None:
            return existing_job
        treasury = self.active_wallet(network=network)
        estimate = self.estimate_withdrawal_gas(asset=asset, network=network, amount=amount)
        reserve_eth = float(estimate["reserve_eth_target"])
        if reserve_eth <= 0:
            return None
        asset_price = self._asset_usd_price(asset)
        eth_price = self._asset_usd_price("ETH")
        conversion_amount = reserve_eth if asset == "ETH" else reserve_eth * eth_price / max(asset_price, 1e-9)
        conversion_amount = max(0.0, conversion_amount)
        balance = WalletBalance.query.filter_by(user_id=wallet_address.user_id, asset=asset).one_or_none()
        if balance is None:
            return None
        if float(balance.available_balance or 0.0) + 1e-12 < conversion_amount:
            job = self._get_or_create_job(
                job_type="deposit_gas_reserve",
                status="failed",
                idempotency_key=f"treasury:deposit-reserve:{ledger_event.id}",
                user_id=wallet_address.user_id,
                wallet_ledger_event_id=ledger_event.id,
                treasury_wallet_id=treasury.id if treasury else None,
                asset=asset,
                network=network,
                source_amount=float(amount or 0.0),
                source_amount_usd=float(amount or 0.0) * asset_price,
                reserve_eth_estimate=float(estimate["estimated_fee_eth"]),
                reserve_multiplier=float(estimate["reserve_multiplier"]),
                reserve_eth_target=reserve_eth,
                conversion_asset=asset,
                conversion_amount=conversion_amount,
                failure_reason="deposit amount is below required treasury gas reserve",
                metadata={"estimate": estimate, "wallet_address_id": wallet_address.id},
            )
            self._record_event(
                platform_treasury_job_id=job.id,
                treasury_wallet_id=treasury.id if treasury else None,
                user_id=wallet_address.user_id,
                wallet_ledger_event_id=ledger_event.id,
                event_type="deposit_gas_reserve_failed",
                status="failed",
                network=network,
                asset=asset,
                amount=conversion_amount,
                idempotency_key=f"treasury:event:deposit-reserve-failed:{ledger_event.id}",
                gas_reserve_contribution=0.0,
                metadata={
                    "job_id": job.id,
                    "conversion_amount": conversion_amount,
                    "reserve_eth_target": reserve_eth,
                    "reason": job.failure_reason,
                },
            )
            return job

        before_available = float(balance.available_balance or 0.0)
        balance.available_balance = max(0.0, before_available - conversion_amount)
        if asset in {"USDC", "USDT", "USD"}:
            balance.estimated_usd_value = float(balance.available_balance or 0.0) + float(balance.locked_balance or 0.0)
        job = self._get_or_create_job(
            job_type="deposit_gas_reserve",
            status="pending",
            idempotency_key=f"treasury:deposit-reserve:{ledger_event.id}",
            user_id=wallet_address.user_id,
            wallet_ledger_event_id=ledger_event.id,
            treasury_wallet_id=treasury.id if treasury else None,
            asset=asset,
            network=network,
            source_amount=float(amount or 0.0),
            source_amount_usd=float(amount or 0.0) * asset_price,
            reserve_eth_estimate=float(estimate["estimated_fee_eth"]),
            reserve_multiplier=float(estimate["reserve_multiplier"]),
            reserve_eth_target=reserve_eth,
            conversion_provider=str(self.config.get("PLATFORM_TREASURY_CONVERSION_PROVIDER", "kucoin") or "kucoin"),
            conversion_asset=asset,
            conversion_amount=conversion_amount,
            metadata={
                "estimate": estimate,
                "wallet_address_id": wallet_address.id,
                "available_before": before_available,
                "available_after": balance.available_balance,
                "eth_usd": eth_price,
                "asset_usd": asset_price,
            },
        )
        self._record_event(
            platform_treasury_job_id=job.id,
            treasury_wallet_id=treasury.id if treasury else None,
            user_id=wallet_address.user_id,
            wallet_ledger_event_id=ledger_event.id,
            event_type="deposit_gas_reserve_deducted",
            status="complete",
            network=network,
            asset=asset,
            amount=conversion_amount,
            idempotency_key=f"treasury:event:deposit-reserve-deducted:{ledger_event.id}",
            gas_reserve_contribution=reserve_eth,
            metadata={"job_id": job.id, "conversion_amount": conversion_amount, "reserve_eth_target": reserve_eth},
        )
        self._audit(
            user_id=wallet_address.user_id,
            action="deposit_gas_reserve_queued",
            status="pending",
            message="Queued platform treasury gas reserve conversion for confirmed deposit.",
            metadata={"job_id": job.id, "reserve_eth_target": reserve_eth, "conversion_amount": conversion_amount},
        )
        return job

    def apply_vault_settlement_deductions(self, cycle: VaultCycle, settlement: VaultCycleSettlement, confirmed_total: float) -> dict[str, Any]:
        """Create gas/profit-share treasury jobs and return the user-credit amount."""

        asset = str(cycle.settlement_asset or settlement.settlement_asset or "").upper().strip()
        network = self._default_network(asset)
        confirmed = max(0.0, float(confirmed_total or 0.0))
        existing = settlement.details.get("treasury_deductions") if isinstance(settlement.details, dict) else None
        if isinstance(existing, dict) and existing.get("applied"):
            return existing
        estimate = self.estimate_withdrawal_gas(asset=asset, network=network, amount=confirmed)
        eth_price = self._asset_usd_price("ETH")
        asset_price = self._asset_usd_price(asset)
        gas_reserve_eth = float(estimate["reserve_eth_target"])
        gas_reserve_asset = gas_reserve_eth if asset == "ETH" else gas_reserve_eth * eth_price / max(asset_price, 1e-9)
        gas_reserve_asset = min(max(0.0, gas_reserve_asset), confirmed)
        gross_profit = max(0.0, confirmed - float(cycle.deposit_amount or 0.0))
        remaining_profit = max(0.0, gross_profit - gas_reserve_asset)
        referral_code = self._user_referral_code(cycle.user_id)
        referral_percent = self._bounded_percent(
            referral_code.percent_profit if referral_code is not None else self.config.get("PLATFORM_TREASURY_DEFAULT_PROFIT_SHARE_PCT", 50.0)
        )
        profit_share_asset = min(confirmed - gas_reserve_asset, remaining_profit * (referral_percent / 100.0))
        profit_share_asset = max(0.0, profit_share_asset)
        user_credit = max(0.0, confirmed - gas_reserve_asset - profit_share_asset)
        treasury = self.active_wallet(network=network)
        gas_job = self._get_or_create_job(
            job_type="vault_gas_reserve",
            status="pending",
            idempotency_key=f"treasury:vault-gas-reserve:{cycle.id}",
            user_id=cycle.user_id,
            vault_cycle_id=cycle.id,
            vault_cycle_settlement_id=settlement.id,
            treasury_wallet_id=treasury.id if treasury else None,
            asset=asset,
            network=network,
            source_amount=confirmed,
            source_amount_usd=confirmed * asset_price,
            reserve_eth_estimate=float(estimate["estimated_fee_eth"]),
            reserve_multiplier=float(estimate["reserve_multiplier"]),
            reserve_eth_target=gas_reserve_eth,
            conversion_provider=str(self.config.get("PLATFORM_TREASURY_CONVERSION_PROVIDER", "kucoin") or "kucoin"),
            conversion_asset=asset,
            conversion_amount=gas_reserve_asset,
            metadata={"estimate": estimate, "settlement_id": settlement.id},
        )
        profit_job = None
        if profit_share_asset > 0:
            profit_job = self._get_or_create_job(
                job_type="vault_profit_share",
                status="pending",
                idempotency_key=f"treasury:vault-profit-share:{cycle.id}",
                user_id=cycle.user_id,
                vault_cycle_id=cycle.id,
                vault_cycle_settlement_id=settlement.id,
                referral_invite_code_id=referral_code.id if referral_code is not None else None,
                treasury_wallet_id=treasury.id if treasury else None,
                asset=asset,
                network=network,
                source_amount=confirmed,
                source_amount_usd=confirmed * asset_price,
                reserve_eth_estimate=0.0,
                reserve_multiplier=1.0,
                reserve_eth_target=profit_share_asset * asset_price / max(eth_price, 1e-9),
                conversion_provider=str(self.config.get("PLATFORM_TREASURY_CONVERSION_PROVIDER", "kucoin") or "kucoin"),
                conversion_asset=asset,
                conversion_amount=profit_share_asset,
                metadata={
                    "settlement_id": settlement.id,
                    "gross_profit": gross_profit,
                    "remaining_profit_after_gas": remaining_profit,
                    "referral_percent": referral_percent,
                    "referral_invite_code": referral_code.code if referral_code is not None else "",
                },
            )
        payload = {
            "applied": True,
            "asset": asset,
            "network": network,
            "confirmed_total": confirmed,
            "gas_reserve_asset": gas_reserve_asset,
            "gas_reserve_eth": gas_reserve_eth,
            "profit_share_asset": profit_share_asset,
            "referral_percent": referral_percent,
            "referral_invite_code": referral_code.code if referral_code is not None else "",
            "user_credit_amount": user_credit,
            "gas_job_id": gas_job.id,
            "profit_job_id": profit_job.id if profit_job is not None else None,
            "gas_job_status": gas_job.status,
            "profit_job_status": profit_job.status if profit_job is not None else None,
        }
        settlement.details = {**settlement.details, "treasury_deductions": payload}
        self._record_event(
            platform_treasury_job_id=gas_job.id,
            treasury_wallet_id=treasury.id if treasury else None,
            user_id=cycle.user_id,
            vault_cycle_id=cycle.id,
            event_type="vault_gas_reserve_deducted",
            status="complete",
            network=network,
            asset=asset,
            amount=gas_reserve_asset,
            idempotency_key=f"treasury:event:vault-gas-reserve:{cycle.id}",
            gas_reserve_contribution=gas_reserve_eth,
            vault_cycle_fee_reserve=gas_reserve_asset,
            metadata=payload,
        )
        if profit_job is not None:
            self._record_event(
                platform_treasury_job_id=profit_job.id,
                treasury_wallet_id=treasury.id if treasury else None,
                user_id=cycle.user_id,
                vault_cycle_id=cycle.id,
                referral_invite_code_id=referral_code.id if referral_code is not None else None,
                referral_owner_user_id=referral_code.created_by_user_id if referral_code is not None else None,
                referral_percent=referral_percent,
                event_type="vault_profit_share_deducted",
                status="complete",
                network=network,
                asset=asset,
                amount=profit_share_asset,
                idempotency_key=f"treasury:event:vault-profit-share:{cycle.id}",
                metadata=payload,
            )
        return payload

    def create_wallet(self, *, network: str = "Ethereum", created_by_user_id: int | None = None) -> PlatformTreasuryWallet:
        self._require_enabled()
        if self.active_wallet(network=network) is not None:
            raise RuntimeError("An active platform treasury wallet already exists for this network.")
        account = Account.create()
        wallet = PlatformTreasuryWallet(
            network=network,
            address=account.address,
            encrypted_private_key=self._encrypt(account.key.hex()),
            is_active=True,
            rotation_index=1,
            created_by_user_id=created_by_user_id,
        )
        db.session.add(wallet)
        db.session.flush()
        self._record_event(
            treasury_wallet_id=wallet.id,
            user_id=created_by_user_id,
            event_type="treasury_created",
            status="complete",
            network=network,
            source_address=None,
            destination_address=wallet.address,
            idempotency_key=f"treasury:create:{network}:{wallet.address}",
            metadata={"address": wallet.address},
        )
        self._audit(
            user_id=created_by_user_id,
            action="platform_treasury_created",
            status="complete",
            message=f"Created active platform treasury wallet for {network}.",
            metadata={"treasury_wallet_id": wallet.id, "network": network, "address": wallet.address},
        )
        return wallet

    def rotate_wallet(self, *, network: str = "Ethereum", created_by_user_id: int | None = None) -> PlatformTreasuryWallet:
        self._require_enabled()
        old = self.active_wallet(network=network)
        if old is None:
            return self.create_wallet(network=network, created_by_user_id=created_by_user_id)
        account = Account.create()
        new_wallet = PlatformTreasuryWallet(
            network=network,
            address=account.address,
            encrypted_private_key=self._encrypt(account.key.hex()),
            is_active=True,
            rotation_index=int(old.rotation_index or 1) + 1,
            rotated_from_wallet_id=old.id,
            created_by_user_id=created_by_user_id,
        )
        old.is_active = False
        old.rotated_at = datetime.utcnow()
        db.session.add(new_wallet)
        db.session.flush()
        migration: dict[str, Any] = {"attempted": False}
        try:
            balance = self.eth_balance(old.address, old.network)
            gas_price = self._gas_price_wei(old.network)
            fee_eth = 21_000 * gas_price / 10**18
            amount = max(0.0, balance - fee_eth)
            if amount > 0:
                migration = self._send_eth(
                    from_wallet=old,
                    destination=new_wallet.address,
                    amount_eth=amount,
                    network=old.network,
                )
                migration["attempted"] = True
        except Exception as exc:  # noqa: BLE001
            migration = {"attempted": True, "error": str(exc)}
        self._record_event(
            treasury_wallet_id=new_wallet.id,
            user_id=created_by_user_id,
            event_type="treasury_rotated",
            status="submitted" if migration.get("provider_reference") else "complete",
            network=network,
            amount=float(migration.get("amount_eth", 0.0) or 0.0),
            fee_amount=float(migration.get("fee_eth", 0.0) or 0.0),
            provider_reference=str(migration.get("provider_reference") or ""),
            source_address=old.address,
            destination_address=new_wallet.address,
            idempotency_key=f"treasury:rotate:{network}:{old.id}:{new_wallet.id}",
            metadata={"old_wallet_id": old.id, "new_wallet_id": new_wallet.id, "migration": migration},
        )
        self._audit(
            user_id=created_by_user_id,
            action="platform_treasury_rotated",
            status="submitted" if migration.get("provider_reference") else "complete",
            message=f"Rotated platform treasury wallet for {network}.",
            metadata={"old_wallet_id": old.id, "new_wallet_id": new_wallet.id, "migration": migration},
        )
        return new_wallet

    def retry_reserve_job(self, job_id: int, *, user_id: int | None = None) -> dict[str, Any]:
        job = db.session.get(PlatformTreasuryReserveJob, int(job_id))
        if job is None:
            raise RuntimeError("Treasury reserve job was not found.")
        if job.status not in {"failed", "retryable", "paused", "submitted", "pending"}:
            raise RuntimeError("Only pending, submitted, paused, retryable, or failed treasury jobs can be retried.")
        job.status = "pending"
        job.failure_reason = None
        job.next_retry_at = None
        details = job.details
        details["retried_by_user_id"] = user_id
        details["retried_at"] = datetime.utcnow().isoformat()
        job.details = details
        return self._job_payload(self.process_reserve_job(job))

    def process_reserve_jobs(self, *, limit: int = 25) -> list[dict[str, Any]]:
        if self.paused:
            return []
        now = datetime.utcnow()
        jobs = (
            PlatformTreasuryReserveJob.query.filter(
                PlatformTreasuryReserveJob.status.in_(["pending", "retryable"]),
            )
            .filter((PlatformTreasuryReserveJob.next_retry_at.is_(None)) | (PlatformTreasuryReserveJob.next_retry_at <= now))
            .order_by(PlatformTreasuryReserveJob.created_at.asc())
            .limit(max(1, int(limit or 25)))
            .all()
        )
        return [self._job_payload(self.process_reserve_job(job)) for job in jobs]

    def process_solvency_cycle(self, *, reserve_limit: int = 25, withdrawal_limit: int = 25, network: str = "Ethereum") -> dict[str, Any]:
        rebalance: dict[str, Any] = {"created": False, "status": "unavailable"}
        solvency = self._solvency_service()
        if solvency is not None:
            rebalance = solvency.rebalance_if_needed(network=network)
        reserve_jobs = self.process_reserve_jobs(limit=reserve_limit)
        withdrawals = self.process_withdrawal_queue(limit=withdrawal_limit)
        return {
            "processed": True,
            "rebalance": rebalance,
            "reserve_jobs": reserve_jobs,
            "withdrawals": withdrawals,
            "reserve_job_count": len(reserve_jobs),
            "withdrawal_count": len(withdrawals),
        }

    def process_reserve_job(self, job: PlatformTreasuryReserveJob) -> PlatformTreasuryReserveJob:
        if job.status == "complete":
            return job
        if self.paused:
            job.status = "paused"
            job.failure_reason = "platform treasury emergency pause is active"
            return job
        treasury = self.active_wallet(network=job.network)
        if treasury is None:
            return self._mark_job_retryable(job, "active platform treasury wallet is not configured")
        job.treasury_wallet_id = treasury.id
        if float(job.conversion_amount or 0.0) <= 0:
            job.status = "complete"
            job.completed_at = datetime.utcnow()
            return job
        if str(job.conversion_asset or job.asset or "").upper() == "ETH":
            job.converted_eth_amount = float(job.conversion_amount or job.reserve_eth_target or 0.0)
            job.status = "complete"
            job.completed_at = datetime.utcnow()
            self._record_event(
                platform_treasury_job_id=job.id,
                treasury_wallet_id=treasury.id,
                user_id=job.user_id,
                wallet_ledger_event_id=job.wallet_ledger_event_id,
                vault_cycle_id=job.vault_cycle_id,
                referral_invite_code_id=job.referral_invite_code_id,
                event_type=f"{job.job_type}_eth_reserved",
                status="complete",
                network=job.network,
                asset="ETH",
                amount=job.converted_eth_amount,
                idempotency_key=f"treasury:event:{job.job_type}:eth-reserved:{job.id}",
                gas_reserve_contribution=job.converted_eth_amount if "gas" in job.job_type else 0.0,
                metadata={"job_id": job.id, "note": "ETH reserve accounted directly"},
            )
            return job
        connector = self._conversion_connector()
        if connector is None:
            return self._mark_job_retryable(job, "PLATFORM_TREASURY_CONVERSION_USER_ID and PLATFORM_TREASURY_CONVERSION_CONNECTION_ID must identify the platform conversion connection")
        try:
            max_slippage_bps = float(
                job.details.get("max_slippage_bps")
                or (
                    self.config.get("TREASURY_REBALANCE_MAX_SLIPPAGE_BPS", 10.0)
                    if job.job_type == "solvency_rebalance"
                    else self.config.get("VAULT_CYCLE_STABLECOIN_MAX_SLIPPAGE_BPS", 10.0)
                )
                or 10.0
            )
            conversion = connector.convert_stablecoin(
                "live",
                str(job.conversion_asset or job.asset).upper(),
                "ETH",
                float(job.conversion_amount or 0.0),
                max_slippage_bps,
                client_reference=f"{job.idempotency_key}:convert",
            )
            job.provider_reference = str(conversion.get("provider_reference") or "")
            converted_eth = max(0.0, float(conversion.get("confirmed_amount", 0.0) or 0.0))
            job.converted_eth_amount = converted_eth
            details = job.details
            details["conversion"] = conversion
            job.details = details
            if converted_eth <= 0:
                job.status = "submitted"
                return job
            withdrawal = connector.withdraw_to_address(
                "live",
                "ETH",
                converted_eth,
                treasury.address,
                network=job.network,
                client_reference=f"{job.idempotency_key}:withdraw-eth",
            )
            job.treasury_tx_reference = str(withdrawal.get("provider_reference") or "")
            details = job.details
            details["treasury_withdrawal"] = withdrawal
            job.details = details
            job.status = "complete" if str(withdrawal.get("status") or "").lower() in {"confirmed", "complete", "submitted"} else "submitted"
            if job.status == "complete":
                job.completed_at = datetime.utcnow()
            self._record_event(
                platform_treasury_job_id=job.id,
                treasury_wallet_id=treasury.id,
                user_id=job.user_id,
                wallet_ledger_event_id=job.wallet_ledger_event_id,
                vault_cycle_id=job.vault_cycle_id,
                referral_invite_code_id=job.referral_invite_code_id,
                event_type=f"{job.job_type}_converted_to_eth",
                status=job.status,
                network=job.network,
                asset="ETH",
                amount=converted_eth,
                provider_reference=job.treasury_tx_reference or job.provider_reference,
                destination_address=treasury.address,
                idempotency_key=f"treasury:event:{job.job_type}:converted:{job.id}",
                gas_reserve_contribution=converted_eth if "gas" in job.job_type else 0.0,
                metadata={"job_id": job.id, "conversion": conversion, "treasury_withdrawal": withdrawal},
            )
            return job
        except Exception as exc:  # noqa: BLE001
            return self._mark_job_retryable(job, str(exc))

    def process_withdrawal_queue(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if self.paused or bool(Setting.get_json("panic_lock", False)):
            return []
        queue_limit = int(limit or self.config.get("PLATFORM_TREASURY_WITHDRAWAL_QUEUE_LIMIT", 25) or 25)
        withdrawals = (
            WalletWithdrawal.query.filter(
                WalletWithdrawal.status.in_(["pending_gas_topup", "queued_treasury_solvency"]),
            )
            .order_by(WalletWithdrawal.created_at.asc())
            .limit(max(1, queue_limit))
            .all()
        )
        results: list[dict[str, Any]] = []
        wallet_service = current_app.extensions["services"].get("self_custody_wallet") if has_app_context() else None
        for withdrawal in withdrawals:
            item: dict[str, Any] = {"withdrawal_id": withdrawal.id, "status": withdrawal.status}
            try:
                solvency = self._solvency_service()
                if solvency is not None and withdrawal.status == "queued_treasury_solvency":
                    if not solvency.release_queued_withdrawal_if_safe(withdrawal):
                        item["status"] = withdrawal.status
                        item["queued"] = True
                        results.append(item)
                        continue
                    item["released"] = True
                    if withdrawal.status == "pending_approval":
                        item["status"] = withdrawal.status
                        item["awaiting_approval"] = True
                        results.append(item)
                        continue
                topup = self.refresh_withdrawal_topup(withdrawal)
                if topup.get("status") in {"missing", "failed"}:
                    topup = self.top_up_withdrawal_gas(withdrawal)
                item["topup"] = topup
                if topup.get("status") == "queued_treasury_solvency":
                    item["status"] = withdrawal.status
                elif topup.get("status") in {"complete", "not_required"} and wallet_service is not None:
                    withdrawal.status = "pending_submission"
                    submitted = wallet_service.submit_withdrawal(withdrawal, mode="live")
                    item["status"] = submitted.status
                    item["provider_reference"] = submitted.provider_reference
                else:
                    item["status"] = withdrawal.status
            except Exception as exc:  # noqa: BLE001
                item["error"] = str(exc)
            results.append(item)
        return results

    def top_up_withdrawal_gas(self, withdrawal: WalletWithdrawal) -> dict[str, Any]:
        """Fund a token source wallet with ETH gas from the active treasury."""

        self._require_enabled()
        if bool(Setting.get_json("panic_lock", False)):
            raise RuntimeError("Panic lock is active; withdrawal gas top-ups are blocked.")
        solvency = self._solvency_service()
        if solvency is not None and withdrawal.status == "queued_treasury_solvency":
            if not solvency.release_queued_withdrawal_if_safe(withdrawal):
                return {"status": "queued_treasury_solvency", "amount_eth": 0.0, "safety": withdrawal.details.get("treasury_safety", {})}
            if withdrawal.status == "pending_approval":
                return {"status": "pending_approval", "amount_eth": 0.0, "reason": "withdrawal is awaiting admin approval"}
        if str(withdrawal.network or "") != "Ethereum" or str(withdrawal.asset or "").upper() == "ETH":
            raise RuntimeError("Treasury gas top-up currently supports Ethereum ERC-20 withdrawals only.")
        if not withdrawal.source_wallet_address_id:
            raise RuntimeError("Withdrawal source wallet must be selected before gas top-up.")
        existing = self._topup_event(withdrawal)
        if existing is not None and existing.status in {"pending", "submitted", "complete"}:
            return {
                "status": existing.status,
                "event_id": existing.id,
                "provider_reference": existing.provider_reference,
                "amount_eth": existing.amount,
                "reused": True,
            }
        from ..models import WalletAddress

        source = db.session.get(WalletAddress, int(withdrawal.source_wallet_address_id))
        if source is None:
            raise RuntimeError("Withdrawal source wallet was not found for treasury top-up.")
        treasury = self.active_wallet(network=withdrawal.network)
        if treasury is None:
            raise RuntimeError("No active platform treasury wallet is available for gas top-up.")
        adapter = EvmWalletAdapter(self.config)
        estimated_fee = max(0.0, float(adapter.estimate_fee(withdrawal.asset, withdrawal.network, withdrawal.destination_address, withdrawal.amount) or 0.0))
        target_eth = estimated_fee * float(self.config.get("PLATFORM_GAS_TOPUP_MULTIPLIER", 2.0) or 2.0)
        current_eth = max(0.0, self.eth_balance(source.address, withdrawal.network))
        amount = max(0.0, target_eth - current_eth)
        max_topup = float(self.config.get("PLATFORM_GAS_TOPUP_MAX_ETH", 0.0) or 0.0)
        if max_topup > 0 and amount > max_topup:
            raise RuntimeError("Required gas top-up exceeds PLATFORM_GAS_TOPUP_MAX_ETH.")
        if amount <= 0:
            return {"status": "not_required", "amount_eth": 0.0, "current_eth": current_eth, "estimated_fee_eth": estimated_fee}
        if solvency is not None:
            safety = solvency.evaluate_withdrawal(withdrawal, projected_spend_eth=amount, estimated_gas_eth=estimated_fee, persist=True)
            if not safety.get("safe", True):
                return {
                    "status": "queued_treasury_solvency",
                    "amount_eth": 0.0,
                    "estimated_fee_eth": estimated_fee,
                    "required_topup_eth": amount,
                    "safety": safety,
                }
        self._enforce_daily_limit(amount)
        treasury_balance = self.eth_balance(treasury.address, treasury.network)
        if treasury_balance <= amount:
            raise RuntimeError("Platform treasury has insufficient ETH for gas top-up.")
        result = self._send_eth(
            from_wallet=treasury,
            destination=source.address,
            amount_eth=amount,
            network=withdrawal.network,
        )
        event = self._record_event(
            treasury_wallet_id=treasury.id,
            user_id=withdrawal.user_id,
            wallet_withdrawal_id=withdrawal.id,
            event_type="withdrawal_gas_topup",
            status="submitted",
            network=withdrawal.network,
            amount=amount,
            fee_amount=float(result.get("fee_eth", 0.0) or 0.0),
            provider_reference=str(result.get("provider_reference") or ""),
            source_address=treasury.address,
            destination_address=source.address,
            idempotency_key=self._topup_idempotency_key(withdrawal),
            gas_reserve_contribution=amount,
            metadata={
                "withdrawal_id": withdrawal.id,
                "withdrawal_asset": withdrawal.asset,
                "estimated_fee_eth": estimated_fee,
                "target_eth": target_eth,
                "source_eth_before": current_eth,
                "raw": result,
            },
        )
        details = withdrawal.details
        details["gas_topup"] = {
            "event_id": event.id,
            "status": event.status,
            "provider_reference": event.provider_reference,
            "amount_eth": amount,
            "source_address": source.address,
            "treasury_address": treasury.address,
        }
        withdrawal.details = details
        withdrawal.status = "pending_gas_topup"
        self._audit(
            user_id=withdrawal.user_id,
            wallet_withdrawal_id=withdrawal.id,
            action="withdrawal_gas_topup_submitted",
            status="submitted",
            message="Submitted platform treasury ETH top-up for withdrawal gas.",
            metadata={"event_id": event.id, "amount_eth": amount, "provider_reference": event.provider_reference},
        )
        return {
            "status": event.status,
            "event_id": event.id,
            "provider_reference": event.provider_reference,
            "amount_eth": amount,
        }

    def refresh_withdrawal_topup(self, withdrawal: WalletWithdrawal) -> dict[str, Any]:
        event = self._topup_event(withdrawal)
        if event is None:
            return {"status": "missing"}
        if event.status in {"complete", "failed"}:
            return self._event_payload(event)
        if not event.provider_reference:
            return self._event_payload(event)
        receipt = self._rpc("eth_getTransactionReceipt", [event.provider_reference], network=event.network)
        if not receipt:
            return self._event_payload(event)
        details = event.details
        details["receipt"] = receipt
        event.details = details
        if receipt.get("status") == "0x1":
            event.status = "complete"
            self._audit(
                user_id=event.user_id,
                wallet_withdrawal_id=event.wallet_withdrawal_id,
                action="withdrawal_gas_topup_confirmed",
                status="complete",
                message="Confirmed platform treasury gas top-up on-chain.",
                metadata={"event_id": event.id, "provider_reference": event.provider_reference},
            )
        elif receipt.get("status") == "0x0":
            event.status = "failed"
            self._audit(
                user_id=event.user_id,
                wallet_withdrawal_id=event.wallet_withdrawal_id,
                action="withdrawal_gas_topup_failed",
                status="failed",
                message="Platform treasury gas top-up failed on-chain.",
                metadata={"event_id": event.id, "provider_reference": event.provider_reference, "receipt": receipt},
            )
        return self._event_payload(event)

    def eth_balance(self, address: str, network: str = "Ethereum") -> float:
        raw = self._rpc("eth_getBalance", [address, "latest"], network=network)
        return int(str(raw or "0x0"), 16) / 10**18

    def _send_eth(
        self,
        *,
        from_wallet: PlatformTreasuryWallet,
        destination: str,
        amount_eth: float,
        network: str,
    ) -> dict[str, Any]:
        private_key = self._decrypt(from_wallet.encrypted_private_key)
        from_address = Account.from_key(private_key).address
        nonce = int(str(self._rpc("eth_getTransactionCount", [from_address, "pending"], network=network) or "0x0"), 16)
        gas_price = self._gas_price_wei(network)
        value = int(float(amount_eth or 0.0) * 10**18)
        tx = {
            "chainId": self._chain_id(network),
            "nonce": nonce,
            "to": destination,
            "value": value,
            "gas": 21_000,
            "gasPrice": gas_price,
            "data": b"",
        }
        signed = Account.sign_transaction(tx, private_key)
        raw_tx = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction")
        raw_hex = raw_tx.hex()
        if not raw_hex.startswith("0x"):
            raw_hex = "0x" + raw_hex
        tx_hash = str(self._rpc("eth_sendRawTransaction", [raw_hex], network=network) or "")
        return {
            "provider_reference": tx_hash,
            "amount_eth": amount_eth,
            "fee_eth": 21_000 * gas_price / 10**18,
            "source": from_address,
            "destination": destination,
            "nonce": nonce,
            "gas_price_wei": gas_price,
        }

    def _record_event(
        self,
        *,
        event_type: str,
        status: str,
        network: str,
        idempotency_key: str,
        platform_treasury_job_id: int | None = None,
        treasury_wallet_id: int | None = None,
        user_id: int | None = None,
        wallet_ledger_event_id: int | None = None,
        wallet_withdrawal_id: int | None = None,
        vault_cycle_id: int | None = None,
        referral_invite_code_id: int | None = None,
        asset: str = "ETH",
        amount: float = 0.0,
        fee_amount: float = 0.0,
        provider_reference: str = "",
        source_address: str | None = None,
        destination_address: str | None = None,
        referral_owner_user_id: int | None = None,
        referral_percent: float = 0.0,
        gas_reserve_contribution: float = 0.0,
        vault_cycle_fee_reserve: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> PlatformTreasuryEvent:
        existing = PlatformTreasuryEvent.query.filter_by(idempotency_key=idempotency_key).one_or_none()
        if existing is not None:
            return existing
        event = PlatformTreasuryEvent(
            platform_treasury_job_id=platform_treasury_job_id,
            treasury_wallet_id=treasury_wallet_id,
            user_id=user_id,
            wallet_ledger_event_id=wallet_ledger_event_id,
            wallet_withdrawal_id=wallet_withdrawal_id,
            vault_cycle_id=vault_cycle_id,
            referral_invite_code_id=referral_invite_code_id,
            event_type=event_type,
            status=status,
            asset=asset,
            network=network,
            amount=float(amount or 0.0),
            fee_amount=float(fee_amount or 0.0),
            provider_reference=provider_reference or None,
            source_address=source_address,
            destination_address=destination_address,
            idempotency_key=idempotency_key,
            referral_owner_user_id=referral_owner_user_id,
            referral_percent=float(referral_percent or 0.0),
            gas_reserve_contribution=float(gas_reserve_contribution or 0.0),
            vault_cycle_fee_reserve=float(vault_cycle_fee_reserve or 0.0),
        )
        event.details = metadata or {}
        db.session.add(event)
        db.session.flush()
        return event

    def _get_or_create_job(
        self,
        *,
        job_type: str,
        status: str,
        idempotency_key: str,
        user_id: int | None = None,
        wallet_ledger_event_id: int | None = None,
        wallet_withdrawal_id: int | None = None,
        vault_cycle_id: int | None = None,
        vault_cycle_settlement_id: int | None = None,
        referral_invite_code_id: int | None = None,
        treasury_wallet_id: int | None = None,
        asset: str = "ETH",
        network: str = "Ethereum",
        source_amount: float = 0.0,
        source_amount_usd: float = 0.0,
        reserve_eth_estimate: float = 0.0,
        reserve_multiplier: float = 2.0,
        reserve_eth_target: float = 0.0,
        conversion_provider: str = "",
        conversion_asset: str = "",
        conversion_amount: float = 0.0,
        failure_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PlatformTreasuryReserveJob:
        existing = PlatformTreasuryReserveJob.query.filter_by(idempotency_key=idempotency_key).one_or_none()
        if existing is not None:
            return existing
        job = PlatformTreasuryReserveJob(
            treasury_wallet_id=treasury_wallet_id,
            user_id=user_id,
            wallet_ledger_event_id=wallet_ledger_event_id,
            wallet_withdrawal_id=wallet_withdrawal_id,
            vault_cycle_id=vault_cycle_id,
            vault_cycle_settlement_id=vault_cycle_settlement_id,
            referral_invite_code_id=referral_invite_code_id,
            job_type=job_type,
            status=status,
            asset=str(asset or "ETH").upper(),
            network=str(network or "Ethereum"),
            source_amount=float(source_amount or 0.0),
            source_amount_usd=float(source_amount_usd or 0.0),
            reserve_eth_estimate=float(reserve_eth_estimate or 0.0),
            reserve_multiplier=float(reserve_multiplier or 1.0),
            reserve_eth_target=float(reserve_eth_target or 0.0),
            conversion_provider=conversion_provider or "",
            conversion_asset=str(conversion_asset or asset or "").upper(),
            conversion_amount=float(conversion_amount or 0.0),
            failure_reason=failure_reason,
            idempotency_key=idempotency_key,
            max_attempts=int(self.config.get("PLATFORM_TREASURY_RESERVE_JOB_MAX_ATTEMPTS", 3) or 3),
        )
        job.details = metadata or {}
        db.session.add(job)
        db.session.flush()
        return job

    def _mark_job_retryable(self, job: PlatformTreasuryReserveJob, reason: str) -> PlatformTreasuryReserveJob:
        job.retry_count = int(job.retry_count or 0) + 1
        job.failure_reason = reason
        max_attempts = max(1, int(job.max_attempts or self.config.get("PLATFORM_TREASURY_RESERVE_JOB_MAX_ATTEMPTS", 3) or 3))
        job.status = "failed" if job.retry_count >= max_attempts else "retryable"
        delay_minutes = min(60, 2 ** max(job.retry_count - 1, 0))
        job.next_retry_at = datetime.utcnow() + timedelta(minutes=delay_minutes) if job.status == "retryable" else None
        details = job.details
        details["last_error"] = reason
        details["last_error_at"] = datetime.utcnow().isoformat()
        job.details = details
        self._record_event(
            platform_treasury_job_id=job.id,
            treasury_wallet_id=job.treasury_wallet_id,
            user_id=job.user_id,
            wallet_ledger_event_id=job.wallet_ledger_event_id,
            wallet_withdrawal_id=job.wallet_withdrawal_id,
            vault_cycle_id=job.vault_cycle_id,
            referral_invite_code_id=job.referral_invite_code_id,
            event_type=f"{job.job_type}_failed",
            status=job.status,
            network=job.network,
            asset=job.asset,
            amount=job.conversion_amount,
            idempotency_key=f"treasury:event:{job.job_type}:failure:{job.id}:{job.retry_count}",
            metadata={"job_id": job.id, "reason": reason, "retry_count": job.retry_count},
        )
        return job

    def _topup_event(self, withdrawal: WalletWithdrawal) -> PlatformTreasuryEvent | None:
        return (
            PlatformTreasuryEvent.query.filter_by(
                wallet_withdrawal_id=withdrawal.id,
                event_type="withdrawal_gas_topup",
            )
            .order_by(PlatformTreasuryEvent.created_at.desc(), PlatformTreasuryEvent.id.desc())
            .first()
        )

    def _topup_idempotency_key(self, withdrawal: WalletWithdrawal) -> str:
        count = PlatformTreasuryEvent.query.filter_by(
            wallet_withdrawal_id=withdrawal.id,
            event_type="withdrawal_gas_topup",
        ).count()
        if count <= 0:
            return f"treasury:withdrawal-gas:{withdrawal.id}"
        return f"treasury:withdrawal-gas:{withdrawal.id}:retry:{count + 1}"

    def _enforce_daily_limit(self, amount: float) -> None:
        limit = float(self.config.get("PLATFORM_GAS_TREASURY_DAILY_LIMIT_ETH", 0.0) or 0.0)
        if limit <= 0:
            return
        since = datetime.utcnow() - timedelta(hours=24)
        spent = sum(
            float(event.amount or 0.0)
            for event in PlatformTreasuryEvent.query.filter(
                PlatformTreasuryEvent.event_type == "withdrawal_gas_topup",
                PlatformTreasuryEvent.status.in_(("submitted", "complete")),
                PlatformTreasuryEvent.created_at >= since,
            ).all()
        )
        if spent + float(amount or 0.0) > limit + 1e-18:
            raise RuntimeError("Platform treasury daily gas top-up limit would be exceeded.")

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise RuntimeError("PLATFORM_GAS_TREASURY_ENABLED is disabled.")
        if not self._has_valid_key():
            raise RuntimeError("TREASURY_ENCRYPTION_KEY must be a valid Fernet key.")
        if self.paused:
            raise RuntimeError("Platform treasury emergency pause is active.")

    def _conversion_connector(self):
        if not has_app_context():
            return None
        user_id = int(self.config.get("PLATFORM_TREASURY_CONVERSION_USER_ID", 0) or 0)
        connection_id = int(self.config.get("PLATFORM_TREASURY_CONVERSION_CONNECTION_ID", 0) or 0)
        if user_id <= 0 or connection_id <= 0:
            return None
        try:
            return current_app.extensions["services"]["trading_connections"].connector_for_user(user_id, connection_id)
        except Exception:
            return None

    def _asset_usd_price(self, asset: str) -> float:
        asset_key = str(asset or "").upper().strip()
        if asset_key in {"USDC", "USDT", "USD"}:
            return 1.0
        if asset_key == "ETH":
            fallback = max(0.0, float(self.config.get("PLATFORM_TREASURY_ETH_USD_FALLBACK", 3000.0) or 0.0))
        else:
            fallback = 1.0
        if has_app_context():
            try:
                price = float(current_app.extensions["services"]["market_data"].get_mid_price(asset_key, "live") or 0.0)
                if price > 0:
                    return price
            except Exception:
                pass
        return fallback

    @staticmethod
    def _bounded_percent(value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 0.0
        return max(0.0, min(parsed, 100.0))

    @staticmethod
    def _network_key(network: str) -> str:
        return "".join(ch for ch in str(network or "").upper() if ch.isalnum())

    def _default_network(self, asset: str) -> str:
        configured = self.config.get("VAULT_CYCLE_DEFAULT_NETWORK_BY_ASSET") or {}
        asset_key = str(asset or "").upper().strip()
        if isinstance(configured, dict):
            value = configured.get(asset_key) or configured.get(asset_key.lower())
            if value:
                return str(value)
        return "Ethereum" if asset_key in {"ETH", "USDC", "USDT"} else "native"

    @staticmethod
    def _user_referral_code(user_id: int | None) -> ReferralInviteCode | None:
        if not user_id:
            return None
        user = db.session.get(User, int(user_id))
        if user is None or not user.referral_invite_code_id:
            return None
        return db.session.get(ReferralInviteCode, int(user.referral_invite_code_id))

    def _has_valid_key(self) -> bool:
        try:
            Fernet(str(self.config.get("TREASURY_ENCRYPTION_KEY", "") or "").encode("utf-8"))
            return True
        except Exception:
            return False

    def _encrypt(self, value: str) -> str:
        key = str(self.config.get("TREASURY_ENCRYPTION_KEY", "") or "").encode("utf-8")
        return Fernet(key).encrypt(str(value).encode("utf-8")).decode("utf-8")

    def _decrypt(self, value: str) -> str:
        key = str(self.config.get("TREASURY_ENCRYPTION_KEY", "") or "").encode("utf-8")
        try:
            return Fernet(key).decrypt(str(value).encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:
            raise RuntimeError("Unable to decrypt platform treasury private key.") from exc

    def _rpc(self, method: str, params: list[Any], *, network: str) -> Any:
        return EvmWalletAdapter(self.config)._rpc(method, params, network=network)  # noqa: SLF001

    def _gas_price_wei(self, network: str) -> int:
        return EvmWalletAdapter(self.config)._gas_price_wei(network)  # noqa: SLF001

    @staticmethod
    def _reserve_health(*, analytics: dict[str, Any], eth_balance: float | None, blockers: list[str]) -> dict[str, Any]:
        balance = max(0.0, float(eth_balance or 0.0))
        runway = analytics.get("gas_runway_withdrawals")
        failed = int(analytics.get("failed_jobs", 0) or 0)
        retryable = int(analytics.get("retryable_jobs", 0) or 0)
        queue = int(analytics.get("queue_waiting", 0) or 0)
        if blockers:
            state = "blocked"
        elif failed > 0 or balance <= 0:
            state = "critical"
        elif retryable > 0 or queue > 0 or (runway is not None and float(runway or 0.0) < 10.0):
            state = "watch"
        else:
            state = "healthy"
        return {
            "state": state,
            "balance_eth": balance,
            "pending_reserve_eth": float(analytics.get("pending_reserve_eth", 0.0) or 0.0),
            "queue_waiting": queue,
            "gas_runway_withdrawals": runway,
        }

    def _chain_id(self, network: str) -> int:
        chain_config = self.config.get("WALLET_EVM_NETWORKS") or {}
        key = "".join(ch for ch in str(network or "").upper() if ch.isalnum())
        if isinstance(chain_config, dict):
            value = chain_config.get(key) or chain_config.get(key.lower()) or {}
            if isinstance(value, dict):
                return int(value.get("chain_id", 1) or 1)
        return 1

    def _solvency_service(self):
        if not has_app_context():
            return None
        return current_app.extensions.get("services", {}).get("treasury_solvency")

    def _solvency_payload(self, *, network: str) -> dict[str, Any]:
        solvency = self._solvency_service()
        if solvency is None:
            return {}
        try:
            return solvency.solvency_payload(network=network)
        except Exception as exc:  # noqa: BLE001
            return {"enabled": getattr(solvency, "enabled", False), "error": str(exc), "state": None, "forecasts": [], "alerts": []}

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
            category="platform_treasury",
            action=action,
            status=status,
            message=message,
        )
        audit.details = metadata or {}
        db.session.add(audit)

    @staticmethod
    def _wallet_payload(wallet: PlatformTreasuryWallet | None) -> dict[str, Any] | None:
        if wallet is None:
            return None
        return {
            "id": wallet.id,
            "network": wallet.network,
            "address": wallet.address,
            "is_active": bool(wallet.is_active),
            "rotation_index": wallet.rotation_index,
            "rotated_from_wallet_id": wallet.rotated_from_wallet_id,
            "created_at": wallet.created_at.isoformat() if wallet.created_at else None,
            "rotated_at": wallet.rotated_at.isoformat() if wallet.rotated_at else None,
        }

    @staticmethod
    def _event_payload(event: PlatformTreasuryEvent) -> dict[str, Any]:
        return {
            "id": event.id,
            "platform_treasury_job_id": event.platform_treasury_job_id,
            "event_type": event.event_type,
            "status": event.status,
            "asset": event.asset,
            "network": event.network,
            "amount": float(event.amount or 0.0),
            "fee_amount": float(event.fee_amount or 0.0),
            "provider_reference": event.provider_reference,
            "source_address": event.source_address,
            "destination_address": event.destination_address,
            "wallet_withdrawal_id": event.wallet_withdrawal_id,
            "referral_invite_code_id": event.referral_invite_code_id,
            "referral_percent": float(event.referral_percent or 0.0),
            "gas_reserve_contribution": float(event.gas_reserve_contribution or 0.0),
            "vault_cycle_fee_reserve": float(event.vault_cycle_fee_reserve or 0.0),
            "created_at": event.created_at.isoformat() if event.created_at else None,
        }

    @staticmethod
    def _job_payload(job: PlatformTreasuryReserveJob) -> dict[str, Any]:
        return {
            "id": job.id,
            "job_type": job.job_type,
            "status": job.status,
            "asset": job.asset,
            "network": job.network,
            "source_amount": float(job.source_amount or 0.0),
            "source_amount_usd": float(job.source_amount_usd or 0.0),
            "reserve_eth_estimate": float(job.reserve_eth_estimate or 0.0),
            "reserve_multiplier": float(job.reserve_multiplier or 0.0),
            "reserve_eth_target": float(job.reserve_eth_target or 0.0),
            "conversion_provider": job.conversion_provider,
            "conversion_asset": job.conversion_asset,
            "conversion_amount": float(job.conversion_amount or 0.0),
            "converted_eth_amount": float(job.converted_eth_amount or 0.0),
            "provider_reference": job.provider_reference,
            "treasury_tx_reference": job.treasury_tx_reference,
            "retry_count": int(job.retry_count or 0),
            "failure_reason": job.failure_reason,
            "user_id": job.user_id,
            "wallet_withdrawal_id": job.wallet_withdrawal_id,
            "wallet_ledger_event_id": job.wallet_ledger_event_id,
            "vault_cycle_id": job.vault_cycle_id,
            "referral_invite_code_id": job.referral_invite_code_id,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "next_retry_at": job.next_retry_at.isoformat() if job.next_retry_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }


def is_close_amount(left: float, right: float) -> bool:
    return math.isclose(float(left or 0.0), float(right or 0.0), rel_tol=1e-12, abs_tol=1e-12)
