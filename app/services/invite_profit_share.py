"""Invite-code profit-share payout calculation and ledger writes."""

from __future__ import annotations

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from ..extensions import db
from ..models import (
    AuditLog,
    ProfitSharePayout,
    ReferralInviteCode,
    User,
    VaultCycle,
    VaultCycleSettlement,
    WalletBalance,
    WalletTransaction,
)

MONEY_QUANT = Decimal("0.000001")
PERCENT_QUANT = Decimal("0.0001")
DEFAULT_PROFIT_SHARE_WALLET = "sufyanh"


class InviteProfitShareError(RuntimeError):
    """Raised when a profit-share payout cannot be completed safely."""


class InviteProfitShareService:
    """Processes invite-code profit-share payouts for completed Vault Cycles."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def process_cycle(
        self,
        cycle: VaultCycle,
        settlement: VaultCycleSettlement,
        *,
        available_credit_amount: float | Decimal | None = None,
        debit_invitee_wallet: bool = False,
    ) -> dict[str, Any]:
        """Apply a completed cycle's invite-code profit share and return accounting details."""

        if cycle is None or settlement is None:
            return self._skipped("missing_cycle_or_settlement", cycle, settlement)
        if cycle.user_id is None:
            return self._skipped("missing_invitee", cycle, settlement)

        invite = self._user_invite_code(cycle.user_id)
        if invite is None:
            return self._skipped("no_invite_code", cycle, settlement)

        idempotency_key = self._idempotency_key(cycle, invite)
        existing = ProfitSharePayout.query.filter_by(idempotency_key=idempotency_key).one_or_none()
        if existing is not None:
            if existing.status in {"failed", "retryable"}:
                raise InviteProfitShareError(existing.failed_reason or "previous invite profit-share payout attempt failed")
            return self._payload_from_payout(existing, applied=existing.status == "completed")

        source_profit = self._source_profit(cycle, settlement)
        if source_profit <= 0:
            return self._skipped("non_positive_profit", cycle, settlement, invite, source_profit=source_profit)

        if not bool(invite.profit_share_active):
            return self._skipped("profit_share_inactive", cycle, settlement, invite, source_profit=source_profit)

        now = cycle.settled_at or settlement.completed_at or datetime.utcnow()
        if invite.profit_share_starts_at is not None and now < invite.profit_share_starts_at:
            return self._skipped("profit_share_not_started", cycle, settlement, invite, source_profit=source_profit)
        if invite.profit_share_ends_at is not None and now > invite.profit_share_ends_at:
            return self._skipped("profit_share_ended", cycle, settlement, invite, source_profit=source_profit)
        if not self._applies_to_vault_type(invite, cycle):
            return self._skipped("vault_type_not_in_scope", cycle, settlement, invite, source_profit=source_profit)

        percent = self._decimal(invite.effective_profit_share_percent).quantize(PERCENT_QUANT)
        if percent <= 0:
            return self._skipped("zero_profit_share_percent", cycle, settlement, invite, source_profit=source_profit)

        destination_wallet = str(invite.profit_share_wallet or DEFAULT_PROFIT_SHARE_WALLET).strip().lower()
        if not destination_wallet:
            return self._fail_payout(
                cycle,
                settlement,
                invite,
                idempotency_key,
                source_profit,
                percent,
                Decimal("0"),
                destination_wallet,
                "profit-share wallet is required",
            )

        requested_payout = (source_profit * percent / Decimal("100")).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)
        credit_cap = self._credit_cap(cycle, available_credit_amount)
        payout_amount = min(requested_payout, credit_cap).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)
        if payout_amount <= 0:
            return self._skipped("profit_share_would_touch_principal", cycle, settlement, invite, source_profit=source_profit)

        destination_user = User.query.filter_by(username=destination_wallet).one_or_none()
        if destination_user is None:
            return self._fail_payout(
                cycle,
                settlement,
                invite,
                idempotency_key,
                source_profit,
                percent,
                payout_amount,
                destination_wallet,
                f"destination wallet user '{destination_wallet}' was not found",
            )

        asset = str(cycle.settlement_asset or settlement.settlement_asset or "USDC").upper()
        payout = ProfitSharePayout(
            invite_code_id=invite.id,
            invitee_user_id=cycle.user_id,
            destination_user_id=destination_user.id,
            vault_cycle_id=cycle.id,
            vault_cycle_settlement_id=settlement.id,
            asset=asset,
            source_profit_amount=source_profit,
            profit_share_percent=percent,
            payout_amount=payout_amount,
            destination_wallet=destination_wallet,
            status="completed",
            idempotency_key=idempotency_key,
            completed_at=datetime.utcnow(),
        )
        payout.details = {
            "cycle_public_id": cycle.public_id,
            "invite_code_public_id": invite.public_id,
            "requested_payout_amount": str(requested_payout),
            "available_credit_cap": str(credit_cap),
            "source_profit_amount": str(source_profit),
            "debit_invitee_wallet": debit_invitee_wallet,
        }
        db.session.add(payout)
        if debit_invitee_wallet:
            self._debit_invitee_wallet(cycle.user_id, asset, payout_amount, cycle, invite, percent, destination_wallet)
        self._credit_destination_wallet(destination_user.id, asset, payout_amount, cycle, invite, percent, destination_wallet)
        if not debit_invitee_wallet:
            self._record_invitee_history(cycle, asset, payout_amount, invite, percent, destination_wallet)
        self._audit(
            cycle,
            invite,
            "invite_profit_share_completed",
            f"Invite code profit share paid {payout_amount} {asset} to {destination_wallet}.",
            {
                "payout_public_id": payout.public_id,
                "destination_wallet": destination_wallet,
                "profit_share_percent": float(percent),
                "payout_amount": float(payout_amount),
                "source_profit_amount": float(source_profit),
            },
        )
        db.session.flush()
        return self._payload_from_payout(payout, applied=True)

    def _credit_destination_wallet(
        self,
        user_id: int,
        asset: str,
        payout_amount: Decimal,
        cycle: VaultCycle,
        invite: ReferralInviteCode,
        percent: Decimal,
        destination_wallet: str,
    ) -> None:
        balance = WalletBalance.query.filter_by(user_id=user_id, asset=asset).one_or_none()
        if balance is None:
            balance = WalletBalance(user_id=user_id, asset=asset)
            db.session.add(balance)
            db.session.flush()
        balance.available_balance = float(self._decimal(balance.available_balance) + payout_amount)
        if asset in {"USDC", "USDT", "USD"}:
            balance.estimated_usd_value = float(self._decimal(balance.available_balance) + self._decimal(balance.locked_balance))
        db.session.add(
            WalletTransaction(
                vault_cycle_id=cycle.id,
                user_id=user_id,
                asset=asset,
                amount=float(payout_amount),
                transaction_type="invite_profit_share_credit",
                status="complete",
                note=f"Invite code profit share: {float(percent):g}% of positive Vault Cycle profit paid to {destination_wallet}.",
            )
        )

    def _debit_invitee_wallet(
        self,
        user_id: int | None,
        asset: str,
        payout_amount: Decimal,
        cycle: VaultCycle,
        invite: ReferralInviteCode,
        percent: Decimal,
        destination_wallet: str,
    ) -> None:
        balance = WalletBalance.query.filter_by(user_id=user_id, asset=asset).one_or_none()
        if balance is None or self._decimal(balance.available_balance) < payout_amount:
            raise InviteProfitShareError("invitee wallet balance is insufficient for post-settlement profit-share debit")
        balance.available_balance = float(self._decimal(balance.available_balance) - payout_amount)
        if asset in {"USDC", "USDT", "USD"}:
            balance.estimated_usd_value = float(self._decimal(balance.available_balance) + self._decimal(balance.locked_balance))
        self._record_invitee_history(cycle, asset, payout_amount, invite, percent, destination_wallet)

    def _record_invitee_history(
        self,
        cycle: VaultCycle,
        asset: str,
        payout_amount: Decimal,
        invite: ReferralInviteCode,
        percent: Decimal,
        destination_wallet: str,
    ) -> None:
        db.session.add(
            WalletTransaction(
                vault_cycle_id=cycle.id,
                user_id=cycle.user_id,
                asset=asset,
                amount=-float(payout_amount),
                transaction_type="invite_profit_share_deduct",
                status="complete",
                note=f"Invite code profit share: {float(percent):g}% of positive Vault Cycle profit paid to {destination_wallet}.",
            )
        )

    def _fail_payout(
        self,
        cycle: VaultCycle,
        settlement: VaultCycleSettlement,
        invite: ReferralInviteCode,
        idempotency_key: str,
        source_profit: Decimal,
        percent: Decimal,
        payout_amount: Decimal,
        destination_wallet: str,
        reason: str,
    ) -> dict[str, Any]:
        payout = ProfitSharePayout(
            invite_code_id=invite.id,
            invitee_user_id=cycle.user_id,
            vault_cycle_id=cycle.id,
            vault_cycle_settlement_id=settlement.id,
            asset=str(cycle.settlement_asset or settlement.settlement_asset or "USDC").upper(),
            source_profit_amount=source_profit,
            profit_share_percent=percent,
            payout_amount=payout_amount,
            destination_wallet=destination_wallet or DEFAULT_PROFIT_SHARE_WALLET,
            status="failed",
            idempotency_key=idempotency_key,
            failed_reason=reason,
        )
        payout.details = {
            "cycle_public_id": cycle.public_id,
            "invite_code_public_id": invite.public_id,
            "reason": reason,
        }
        db.session.add(payout)
        self._audit(
            cycle,
            invite,
            "invite_profit_share_failed",
            f"Invite code profit-share payout failed: {reason}.",
            {"reason": reason, "destination_wallet": destination_wallet, "payout_amount": float(payout_amount)},
        )
        db.session.flush()
        raise InviteProfitShareError(reason)

    def _skipped(
        self,
        reason: str,
        cycle: VaultCycle | None,
        settlement: VaultCycleSettlement | None,
        invite: ReferralInviteCode | None = None,
        *,
        source_profit: Decimal | None = None,
    ) -> dict[str, Any]:
        payload = {
            "applied": False,
            "status": "skipped",
            "reason": reason,
            "payout_amount": 0.0,
            "source_profit_amount": float(source_profit or Decimal("0")),
            "profit_share_percent": float(invite.effective_profit_share_percent) if invite is not None else 0.0,
            "destination_wallet": invite.profit_share_wallet if invite is not None else DEFAULT_PROFIT_SHARE_WALLET,
            "invite_code": invite.code if invite is not None else "",
            "invite_code_public_id": invite.public_id if invite is not None else "",
        }
        if cycle is not None:
            self._audit(
                cycle,
                invite,
                "invite_profit_share_skipped",
                f"Invite code profit share skipped: {reason}.",
                payload,
            )
        return payload

    def _payload_from_payout(self, payout: ProfitSharePayout, *, applied: bool) -> dict[str, Any]:
        return {
            "applied": applied,
            "status": payout.status,
            "payout_public_id": payout.public_id,
            "payout_amount": float(payout.payout_amount or 0),
            "source_profit_amount": float(payout.source_profit_amount or 0),
            "profit_share_percent": float(payout.profit_share_percent or 0),
            "destination_wallet": payout.destination_wallet,
            "invite_code_public_id": payout.invite_code.public_id if payout.invite_code is not None else "",
            "invite_code": payout.invite_code.code if payout.invite_code is not None else "",
            "idempotency_key": payout.idempotency_key,
        }

    def _audit(
        self,
        cycle: VaultCycle,
        invite: ReferralInviteCode | None,
        action: str,
        message: str,
        metadata: dict[str, Any],
    ) -> None:
        audit = AuditLog(
            category="invite_profit_share",
            action=action,
            message=message,
            user_id=cycle.user_id,
        )
        audit.details = {
            **metadata,
            "vault_cycle_public_id": cycle.public_id,
            "invite_code_public_id": invite.public_id if invite is not None else "",
            "invite_code": invite.code if invite is not None else "",
        }
        db.session.add(audit)

    @staticmethod
    def _decimal(value: float | Decimal | int | str | None) -> Decimal:
        try:
            return Decimal(str(value or "0"))
        except Exception:  # noqa: BLE001
            return Decimal("0")

    def _source_profit(self, cycle: VaultCycle, settlement: VaultCycleSettlement) -> Decimal:
        profit = self._decimal(settlement.net_pnl_usd)
        if profit <= 0:
            profit = self._decimal(settlement.final_amount or settlement.final_value_usd) - self._decimal(cycle.deposit_amount)
        return max(Decimal("0"), profit).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)

    def _credit_cap(self, cycle: VaultCycle, available_credit_amount: float | Decimal | None) -> Decimal:
        if available_credit_amount is None:
            return Decimal("0")
        cap = self._decimal(available_credit_amount) - self._decimal(cycle.deposit_amount)
        return max(Decimal("0"), cap).quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)

    @staticmethod
    def _applies_to_vault_type(invite: ReferralInviteCode, cycle: VaultCycle) -> bool:
        allowed = {item.lower() for item in invite.applies_to_vault_types}
        if not allowed:
            return True
        candidates = {
            str(cycle.algorithm_profile or "").lower(),
            str(cycle.selected_strategy_name or "").lower(),
        }
        metadata = cycle.selection_metadata
        for key in ("vault_type", "vault_cycle_name", "strategy_type"):
            if metadata.get(key):
                candidates.add(str(metadata[key]).lower())
        return bool(allowed & candidates)

    @staticmethod
    def _idempotency_key(cycle: VaultCycle, invite: ReferralInviteCode) -> str:
        return f"invite-profit-share:vault-cycle:{cycle.public_id}:invite-code:{invite.public_id}"

    @staticmethod
    def _user_invite_code(user_id: int | None) -> ReferralInviteCode | None:
        if user_id is None:
            return None
        user = db.session.get(User, int(user_id))
        if user is None or not user.referral_invite_code_id:
            return None
        return db.session.get(ReferralInviteCode, int(user.referral_invite_code_id))
