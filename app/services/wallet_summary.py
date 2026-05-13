"""Read-only wallet/profile summaries and reconciliation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import joinedload

from ..extensions import db
from ..models import Order, Setting, TradingConnection, User, VaultCycle, WalletAddress, WalletBalance, WalletTransaction


DEFAULT_WALLET_ASSETS = ("BTC", "ETH", "SOL", "USDC", "USDT", "XRP")
ACTIVE_ORDER_STATUSES = ("submitted", "open", "partially_filled", "pending")


@dataclass(frozen=True)
class WalletBalanceView:
    """Template-friendly read model for a wallet balance row."""

    asset: str
    available_balance: float = 0.0
    locked_balance: float = 0.0
    estimated_usd_value: float = 0.0
    active_deposit_address: Any | None = None
    onchain_balance: float = 0.0
    onchain_checked_at: Any | None = None
    onchain_status: str = "unavailable"
    onchain_reason: str = ""
    onchain_delta: float = 0.0
    onchain_mismatch_status: str = "unavailable"

    @property
    def total_balance(self) -> float:
        return float(self.available_balance or 0.0) + float(self.locked_balance or 0.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "available_balance": float(self.available_balance or 0.0),
            "locked_balance": float(self.locked_balance or 0.0),
            "total_balance": self.total_balance,
            "estimated_usd_value": float(self.estimated_usd_value or 0.0),
            "onchain_balance": float(self.onchain_balance or 0.0),
            "onchain_checked_at": _iso(self.onchain_checked_at),
            "onchain_status": self.onchain_status,
            "onchain_reason": self.onchain_reason,
            "onchain_delta": float(self.onchain_delta or 0.0),
            "onchain_mismatch_status": self.onchain_mismatch_status,
        }


@dataclass(frozen=True)
class ProfileWalletSummary:
    """User-scoped wallet status used by CLI and page renderers."""

    user: User
    balances: list[WalletBalanceView]
    portfolio_total_usd: float
    locked_total: float
    active_cycles_count: int
    order_count: int
    active_order_count: int
    trading_connections: list[dict[str, Any]]
    reconciliation_warnings: list[str]
    cached_exchange_snapshot: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "exists": True,
            "user": {
                "id": self.user.id,
                "username": self.user.username,
                "role": self.user.role,
                "two_factor_enabled": bool(self.user.two_factor_enabled),
                "created_at": _iso(self.user.created_at),
                "updated_at": _iso(self.user.updated_at),
            },
            "wallet": {
                "source": "local_app_wallet",
                "portfolio_total_usd": self.portfolio_total_usd,
                "locked_total": self.locked_total,
                "balances": [balance.as_dict() for balance in self.balances],
            },
            "activity": {
                "active_cycles_count": self.active_cycles_count,
                "order_count": self.order_count,
                "active_order_count": self.active_order_count,
            },
            "trading_connections": self.trading_connections,
            "cached_exchange_snapshot": self.cached_exchange_snapshot,
            "reconciliation_warnings": self.reconciliation_warnings,
            "notes": [
                "This command is read-only and does not modify balances, transactions, credentials, or orders.",
                "Local app wallet balances are separate from exchange margin balances.",
            ],
        }


class WalletSummaryService:
    """Build local wallet summaries without live exchange side effects."""

    def profile_wallet_check(self, *, username: str | None = None, user_id: int | None = None) -> dict[str, Any]:
        user = self._find_user(username=username, user_id=user_id)
        if user is None:
            return {
                "exists": False,
                "username": username,
                "user_id": user_id,
                "blockers": ["profile_not_found"],
            }
        return self.summary_for_user(user).as_dict()

    def summary_for_user(self, user: User, *, balances: list[WalletBalance] | None = None) -> ProfileWalletSummary:
        balance_views = self._balance_views(user.id, balances=balances)
        return ProfileWalletSummary(
            user=user,
            balances=balance_views,
            portfolio_total_usd=sum(float(balance.estimated_usd_value or 0.0) for balance in balance_views),
            locked_total=sum(float(balance.locked_balance or 0.0) for balance in balance_views),
            active_cycles_count=self._active_cycles_count(user.id),
            order_count=Order.query.filter_by(user_id=user.id).count(),
            active_order_count=Order.query.filter(Order.user_id == user.id, Order.status.in_(ACTIVE_ORDER_STATUSES)).count(),
            trading_connections=self._connection_summaries(user.id),
            reconciliation_warnings=self.reconciliation_warnings(user.id),
            cached_exchange_snapshot=self.cached_exchange_snapshot(user.id),
        )

    def cached_exchange_snapshot(self, user_id: int) -> dict[str, Any]:
        value = Setting.get_json(_exchange_balance_snapshot_key(user_id), {})
        return value if isinstance(value, dict) else {}

    def refresh_exchange_snapshot(
        self,
        user: User,
        trading_connections: Any,
        *,
        mode: str = "live",
        connection_id: int | None = None,
        snapshot: Any | None = None,
    ) -> dict[str, Any]:
        connection = (
            TradingConnection.query.filter_by(user_id=user.id, id=connection_id).one_or_none()
            if connection_id is not None
            else trading_connections.active_tradable_connection(user.id)
        )
        if connection is None:
            return self.cached_exchange_snapshot(user.id)
        if snapshot is None:
            snapshot = trading_connections.account_snapshot(user.id, mode, connection.id)
        balances: list[dict[str, Any]] = []
        for item in snapshot.balances:
            asset = str(item.get("asset", "") or "").upper()
            if not asset:
                continue
            value = _safe_float(item.get("value"))
            withdrawable = max(_safe_float(item.get("withdrawable", value)), 0.0)
            balances.append(
                {
                    "asset": asset,
                    "type": str(item.get("type", "margin") or "margin"),
                    "value": value,
                    "withdrawable": withdrawable,
                    "estimated_usd_value": _estimated_usd_value(asset, value, item),
                }
            )
        payload = {
            "mode": mode,
            "connection_id": connection.id,
            "provider": connection.provider,
            "balances": balances,
            "positions_count": len(snapshot.positions or []),
            "open_orders_count": len(snapshot.open_orders or []),
            "alerts": snapshot.alerts or [],
            "synced_at": datetime.utcnow().isoformat() + "Z",
        }
        Setting.set_json(_exchange_balance_snapshot_key(user.id), payload)
        db.session.commit()
        return payload

    def reconciliation_warnings(self, user_id: int) -> list[str]:
        warnings: list[str] = []
        duplicate_settlements = (
            db.session.query(
                WalletTransaction.vault_cycle_id,
                WalletTransaction.asset,
                func.count(WalletTransaction.id).label("transaction_count"),
                func.sum(WalletTransaction.amount).label("total_amount"),
            )
            .filter(
                WalletTransaction.user_id == user_id,
                WalletTransaction.transaction_type == "settlement",
                WalletTransaction.status == "complete",
                WalletTransaction.vault_cycle_id.isnot(None),
            )
            .group_by(WalletTransaction.vault_cycle_id, WalletTransaction.asset)
            .having(func.count(WalletTransaction.id) > 1)
            .all()
        )
        for row in duplicate_settlements:
            warnings.append(
                "duplicate_complete_settlement_transactions:"
                f" vault_cycle_id={row.vault_cycle_id}, asset={row.asset},"
                f" count={int(row.transaction_count or 0)}, total_amount={float(row.total_amount or 0.0):.10f}"
            )
        negative_balances = (
            WalletBalance.query.filter(
                WalletBalance.user_id == user_id,
                (WalletBalance.available_balance < 0) | (WalletBalance.locked_balance < 0),
            )
            .order_by(WalletBalance.asset.asc())
            .all()
        )
        for balance in negative_balances:
            warnings.append(f"negative_wallet_balance: asset={balance.asset}")
        return warnings

    def _find_user(self, *, username: str | None, user_id: int | None) -> User | None:
        query = User.query
        if user_id is not None:
            query = query.filter(User.id == int(user_id))
        if username:
            query = query.filter(func.lower(User.username) == username.strip().lower())
        return query.one_or_none()

    def _balance_views(self, user_id: int, *, balances: list[WalletBalance] | None = None) -> list[WalletBalanceView]:
        records = balances
        if records is None:
            records = (
                WalletBalance.query.options(joinedload(WalletBalance.active_deposit_address))
                .filter_by(user_id=user_id)
                .order_by(WalletBalance.asset.asc())
                .all()
            )
        by_asset = {record.asset.upper(): record for record in records}
        onchain_by_asset = self._onchain_snapshots(user_id)
        assets = sorted(set(DEFAULT_WALLET_ASSETS) | set(by_asset))
        views: list[WalletBalanceView] = []
        for asset in assets:
            record = by_asset.get(asset)
            onchain = onchain_by_asset.get(asset, {})
            available = float(record.available_balance or 0.0) if record is not None else 0.0
            locked = float(record.locked_balance or 0.0) if record is not None else 0.0
            total = available + locked
            onchain_status = str(onchain.get("status") or "unavailable")
            onchain_balance = float(onchain.get("balance", 0.0) or 0.0) if onchain_status == "checked" else 0.0
            onchain_delta = onchain_balance - total if onchain_status == "checked" else 0.0
            views.append(
                WalletBalanceView(
                    asset=asset,
                    available_balance=available,
                    locked_balance=locked,
                    estimated_usd_value=float(record.estimated_usd_value or 0.0) if record is not None else 0.0,
                    active_deposit_address=record.active_deposit_address if record is not None else None,
                    onchain_balance=onchain_balance,
                    onchain_checked_at=onchain.get("checked_at"),
                    onchain_status=onchain_status,
                    onchain_reason=str(onchain.get("reason") or ""),
                    onchain_delta=onchain_delta,
                    onchain_mismatch_status=_onchain_mismatch_status(onchain_status, onchain_delta),
                )
            )
        return views

    def _onchain_snapshots(self, user_id: int) -> dict[str, dict[str, Any]]:
        records = (
            WalletAddress.query.filter_by(user_id=user_id, status="active")
            .order_by(WalletAddress.asset.asc(), WalletAddress.rotation_index.desc(), WalletAddress.created_at.desc())
            .all()
        )
        by_asset: dict[str, dict[str, Any]] = {}
        for record in records:
            asset = str(record.asset or "").upper()
            if not asset:
                continue
            row = by_asset.setdefault(
                asset,
                {
                    "balance": 0.0,
                    "status": "unavailable",
                    "checked_at": None,
                    "reason": "",
                    "checked_count": 0,
                    "reasons": [],
                },
            )
            if str(record.onchain_status or "") == "checked":
                row["balance"] = float(row["balance"] or 0.0) + max(0.0, float(record.onchain_balance or 0.0))
                row["status"] = "checked"
                row["checked_count"] = int(row["checked_count"] or 0) + 1
                if record.onchain_checked_at is not None and (
                    row["checked_at"] is None or record.onchain_checked_at > row["checked_at"]
                ):
                    row["checked_at"] = record.onchain_checked_at
            else:
                reason = str(record.onchain_reason or record.onchain_status or "unavailable")
                if reason:
                    row["reasons"].append(reason)
        for row in by_asset.values():
            if row["status"] != "checked":
                row["reason"] = "; ".join(dict.fromkeys(row["reasons"]))
            row.pop("reasons", None)
            row.pop("checked_count", None)
        return by_asset

    def _active_cycles_count(self, user_id: int) -> int:
        return VaultCycle.query.filter(
            VaultCycle.user_id == user_id,
            VaultCycle.status.in_(("active", "settling")),
        ).count()

    def _connection_summaries(self, user_id: int) -> list[dict[str, Any]]:
        records = TradingConnection.query.filter_by(user_id=user_id).order_by(TradingConnection.id.asc()).all()
        return [
            {
                "connection_id": record.id,
                "provider": record.provider,
                "connection_type": record.connection_type,
                "is_active": bool(record.is_active),
                "verification_status": record.verification_status,
                "last_verified_at": _iso(record.last_verified_at),
                "last_verification_error": _redact_error(record.last_verification_error),
                "created_at": _iso(record.created_at),
                "updated_at": _iso(record.updated_at),
            }
            for record in records
        ]


def _exchange_balance_snapshot_key(user_id: int) -> str:
    return f"exchange_balance_snapshot:{int(user_id)}"


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _estimated_usd_value(asset: str, value: float, item: dict[str, Any]) -> float:
    explicit = item.get("estimated_usd_value")
    if explicit is not None:
        return _safe_float(explicit)
    return value if asset in {"USDC", "USDT", "USD"} else _safe_float(item.get("usd_value", value))


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None and hasattr(value, "isoformat") else None


def _onchain_mismatch_status(onchain_status: str, delta: float) -> str:
    if onchain_status != "checked":
        return "unavailable"
    if math_isclose(float(delta or 0.0), 0.0):
        return "matched"
    return "surplus_onchain" if delta > 0 else "deficit_onchain"


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-8


def _redact_error(value: str | None) -> str:
    if not value:
        return ""
    text = str(value)
    for marker in ("api_key", "api_secret", "passphrase", "secret"):
        text = text.replace(marker, "[redacted]")
    return text
