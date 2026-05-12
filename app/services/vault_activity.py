"""Vault cycle retrieval and retention helpers."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import or_

from ..extensions import db
from ..models import Order, VaultAllocationLeg, VaultCycle, WalletTransaction


DEFAULT_VAULT_CYCLE_RETENTION_LIMIT = 50
DEFAULT_VAULT_CYCLE_PAGE_SIZE = 5
ACTIVE_CYCLE_STATUSES = {"active", "settling"}


@dataclass(frozen=True)
class VaultCyclePage:
    """Template-friendly paginated vault cycle result."""

    items: list[VaultCycle]
    page: int
    per_page: int
    total: int
    pages: int

    @property
    def has_prev(self) -> bool:
        return self.page > 1

    @property
    def has_next(self) -> bool:
        return self.page < self.pages

    @property
    def prev_page(self) -> int:
        return max(1, self.page - 1)

    @property
    def next_page(self) -> int:
        return min(self.pages, self.page + 1)

    @property
    def start_index(self) -> int:
        if self.total <= 0:
            return 0
        return ((self.page - 1) * self.per_page) + 1

    @property
    def end_index(self) -> int:
        return min(self.total, self.page * self.per_page)


class VaultCycleActivityService:
    """Manage user-scoped vault cycle display and retention."""

    def prune_user_cycles(self, user_id: int, *, limit: int = DEFAULT_VAULT_CYCLE_RETENTION_LIMIT) -> int:
        user_id = int(user_id)
        limit = max(1, int(limit))
        keep_ids = [
            row.id
            for row in VaultCycle.query.with_entities(VaultCycle.id)
            .filter(VaultCycle.user_id == user_id)
            .order_by(VaultCycle.started_at.desc(), VaultCycle.id.desc())
            .limit(limit)
            .all()
        ]
        if len(keep_ids) < limit:
            return 0

        delete_ids = [
            row.id
            for row in VaultCycle.query.with_entities(VaultCycle.id)
            .filter(
                VaultCycle.user_id == user_id,
                ~VaultCycle.id.in_(keep_ids),
                ~VaultCycle.status.in_(ACTIVE_CYCLE_STATUSES),
            )
            .all()
        ]
        if not delete_ids:
            return 0

        delete_leg_ids = [
            row.id
            for row in VaultAllocationLeg.query.with_entities(VaultAllocationLeg.id)
            .filter(VaultAllocationLeg.vault_cycle_id.in_(delete_ids))
            .all()
        ]
        WalletTransaction.query.filter(WalletTransaction.vault_cycle_id.in_(delete_ids)).update(
            {"vault_cycle_id": None},
            synchronize_session=False,
        )
        order_filter = Order.vault_cycle_id.in_(delete_ids)
        if delete_leg_ids:
            order_filter = or_(order_filter, Order.vault_leg_id.in_(delete_leg_ids))
        Order.query.filter(order_filter).update({"vault_cycle_id": None, "vault_leg_id": None}, synchronize_session=False)
        VaultAllocationLeg.query.filter(VaultAllocationLeg.vault_cycle_id.in_(delete_ids)).delete(
            synchronize_session=False
        )
        deleted = VaultCycle.query.filter(VaultCycle.id.in_(delete_ids)).delete(synchronize_session=False)
        db.session.flush()
        return deleted

    def recent_for_user(self, user_id: int, *, limit: int = DEFAULT_VAULT_CYCLE_RETENTION_LIMIT) -> list[VaultCycle]:
        return (
            VaultCycle.query.filter(VaultCycle.user_id == int(user_id))
            .order_by(VaultCycle.started_at.desc(), VaultCycle.id.desc())
            .limit(max(1, int(limit)))
            .all()
        )

    def page_for_user(
        self,
        user_id: int,
        *,
        page: int = 1,
        per_page: int = DEFAULT_VAULT_CYCLE_PAGE_SIZE,
        retention_limit: int = DEFAULT_VAULT_CYCLE_RETENTION_LIMIT,
    ) -> VaultCyclePage:
        self.prune_user_cycles(user_id, limit=retention_limit)
        per_page = max(1, int(per_page))
        requested_page = max(1, int(page))
        query = VaultCycle.query.filter(VaultCycle.user_id == int(user_id))
        total = query.count()
        pages = max(1, (total + per_page - 1) // per_page)
        current_page = min(requested_page, pages)
        items = (
            query.order_by(VaultCycle.started_at.desc(), VaultCycle.id.desc())
            .offset((current_page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        return VaultCyclePage(
            items=items,
            page=current_page,
            per_page=per_page,
            total=total,
            pages=pages,
        )
