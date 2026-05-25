"""Wallet activity retrieval and retention helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import WalletTransaction

DEFAULT_ACTIVITY_RETENTION_LIMIT = 50
DEFAULT_ACTIVITY_PAGE_SIZE = 5


@dataclass(frozen=True)
class WalletActivityPage:
    """Template-friendly paginated wallet activity result."""

    items: list[WalletTransaction]
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


class WalletActivityService:
    """Manage user-scoped wallet transaction display and retention."""

    def prune_user_activity(self, user_id: int, *, limit: int = DEFAULT_ACTIVITY_RETENTION_LIMIT) -> int:
        user_id = int(user_id)
        limit = max(1, int(limit))
        keep_ids = [
            row.id
            for row in WalletTransaction.query.with_entities(WalletTransaction.id)
            .filter(WalletTransaction.user_id == user_id)
            .order_by(WalletTransaction.created_at.desc(), WalletTransaction.id.desc())
            .limit(limit)
            .all()
        ]
        if len(keep_ids) < limit:
            return 0
        return WalletTransaction.query.filter(
            WalletTransaction.user_id == user_id,
            ~WalletTransaction.id.in_(keep_ids),
        ).delete(synchronize_session=False)

    def recent_for_user(self, user_id: int, *, limit: int = DEFAULT_ACTIVITY_RETENTION_LIMIT) -> list[WalletTransaction]:
        return (
            WalletTransaction.query.filter(WalletTransaction.user_id == int(user_id))
            .order_by(WalletTransaction.created_at.desc(), WalletTransaction.id.desc())
            .limit(max(1, int(limit)))
            .all()
        )

    def page_for_user(
        self,
        user_id: int,
        *,
        page: int = 1,
        per_page: int = DEFAULT_ACTIVITY_PAGE_SIZE,
        retention_limit: int = DEFAULT_ACTIVITY_RETENTION_LIMIT,
    ) -> WalletActivityPage:
        _ = retention_limit
        per_page = max(1, int(per_page))
        requested_page = max(1, int(page))
        query = WalletTransaction.query.filter(WalletTransaction.user_id == int(user_id))
        total = query.count()
        pages = max(1, (total + per_page - 1) // per_page)
        current_page = min(requested_page, pages)
        items = (
            query.order_by(WalletTransaction.created_at.desc(), WalletTransaction.id.desc())
            .offset((current_page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        return WalletActivityPage(
            items=items,
            page=current_page,
            per_page=per_page,
            total=total,
            pages=pages,
        )
