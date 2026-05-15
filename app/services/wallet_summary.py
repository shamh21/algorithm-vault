"""Read-only wallet/profile summaries and reconciliation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from flask import current_app, has_app_context
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
    sync_status: str = "not_configured"
    sync_reason: str = ""
    sync_checked_at: str | None = None
    verified_on_chain_balance: float | None = None

    @property
    def total_balance(self) -> float:
        return float(self.available_balance or 0.0) + float(self.locked_balance or 0.0)

    @property
    def sync_stale(self) -> bool:
        return self.sync_status in {"sync_failed", "partial_sync_failed", "unconfirmed", "not_synced"}

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
            "sync_status": self.sync_status,
            "sync_reason": self.sync_reason,
            "sync_checked_at": self.sync_checked_at,
            "sync_stale": self.sync_stale,
            "verified_on_chain_balance": self.verified_on_chain_balance,
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

    def account_funds_readiness(
        self,
        *,
        username: str,
        user_id: int | None = None,
        expected_snapshot: dict[str, Any] | None = None,
        require_expected_snapshot: bool = True,
        tolerance: float = 0.00000001,
    ) -> dict[str, Any]:
        """Check that a protected account exists and retains expected local wallet funds."""

        profile = self.profile_wallet_check(username=username, user_id=user_id)
        blockers: list[str] = []
        comparisons: list[dict[str, Any]] = []
        warnings: list[str] = []
        if not bool(profile.get("exists", False)):
            blockers.append("profile_not_found")
            return {
                "ready": False,
                "username": username,
                "user_id": user_id,
                "blockers": blockers,
                "warnings": warnings,
                "profile": profile,
                "comparisons": comparisons,
            }

        wallet = profile.get("wallet") if isinstance(profile.get("wallet"), dict) else {}
        current_balances = _balance_index(wallet.get("balances", []))
        funded_assets = [asset for asset, balance in current_balances.items() if balance["total_balance"] > tolerance]
        if not funded_assets:
            blockers.append("wallet_has_no_funds")

        if expected_snapshot is None:
            if require_expected_snapshot:
                blockers.append("expected_wallet_snapshot_required")
        else:
            expected_wallet = expected_snapshot.get("wallet") if isinstance(expected_snapshot.get("wallet"), dict) else expected_snapshot
            expected_balances = _balance_index(expected_wallet.get("balances", []) if isinstance(expected_wallet, dict) else [])
            if not expected_balances:
                blockers.append("expected_wallet_snapshot_has_no_balances")
            for asset, expected in sorted(expected_balances.items()):
                current = current_balances.get(asset, _empty_balance(asset))
                total_ok = current["total_balance"] + tolerance >= expected["total_balance"]
                available_ok = current["available_balance"] + tolerance >= expected["available_balance"]
                locked_ok = current["locked_balance"] + tolerance >= expected["locked_balance"]
                comparison = {
                    "asset": asset,
                    "current": current,
                    "expected_minimum": expected,
                    "total_ok": total_ok,
                    "available_ok": available_ok,
                    "locked_ok": locked_ok,
                }
                comparisons.append(comparison)
                if not total_ok:
                    blockers.append(f"balance_below_expected:{asset}")
                elif not available_ok or not locked_ok:
                    warnings.append(f"balance_distribution_changed:{asset}")
            expected_portfolio = _safe_float(expected_wallet.get("portfolio_total_usd") if isinstance(expected_wallet, dict) else None)
            current_portfolio = _safe_float(wallet.get("portfolio_total_usd"))
            if expected_portfolio > tolerance and current_portfolio + tolerance < expected_portfolio:
                blockers.append("portfolio_total_usd_below_expected")

        return {
            "ready": not blockers,
            "username": username,
            "user_id": profile.get("user", {}).get("id") if isinstance(profile.get("user"), dict) else user_id,
            "blockers": blockers,
            "warnings": warnings,
            "funded_assets": sorted(funded_assets),
            "profile": profile,
            "comparisons": comparisons,
        }

    def summary_for_user(self, user: User, *, balances: list[WalletBalance] | None = None) -> ProfileWalletSummary:
        sync_report = self._sync_and_reconcile_custody(user.id)
        balance_views = self._balance_views(user.id, balances=None if sync_report.get("touched") else balances)
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
        for item in _snapshot_rows(snapshot.balances):
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
            "positions": _snapshot_rows(snapshot.positions),
            "open_orders": _snapshot_rows(snapshot.open_orders),
            "recent_fills": _snapshot_rows(snapshot.recent_fills),
            "positions_count": len(snapshot.positions or []),
            "open_orders_count": len(snapshot.open_orders or []),
            "recent_fills_count": len(snapshot.recent_fills or []),
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
        sync_status = self._sync_status_by_asset(user_id)
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
            status = sync_status.get(asset, {"status": "not_configured", "reason": "", "checked_at": None, "on_chain_total": None})
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
                    sync_status=str(status.get("status") or "not_configured"),
                    sync_reason=str(status.get("reason") or ""),
                    sync_checked_at=status.get("checked_at"),
                    verified_on_chain_balance=status.get("on_chain_total"),
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

    def _sync_and_reconcile_custody(self, user_id: int) -> dict[str, Any]:
        if not has_app_context():
            return {"touched": False}
        custody = current_app.extensions.get("services", {}).get("wallet_custody")
        if custody is None or not getattr(custody, "enabled", False):
            return {"touched": False}
        touched = False
        try:
            custody.sync_user(user_id)
            touched = True
        except Exception as exc:  # noqa: BLE001
            current_app.logger.warning("Wallet summary custody sync failed for user %s: %s", user_id, exc)
        assets = {
            row.asset
            for row in WalletBalance.query.filter_by(user_id=user_id).all()
        } | {
            row.asset
            for row in WalletAddress.query.filter_by(user_id=user_id, status="active").all()
        }
        for asset in assets:
            try:
                custody.reconcile_custody_balance(user_id, asset)
                touched = True
            except Exception as exc:  # noqa: BLE001
                current_app.logger.warning("Wallet summary custody reconciliation failed for user %s asset %s: %s", user_id, asset, exc)
        if touched:
            db.session.flush()
        return {"touched": touched}

    def _sync_status_by_asset(self, user_id: int) -> dict[str, dict[str, Any]]:
        rows = WalletAddress.query.filter_by(user_id=user_id, status="active").order_by(WalletAddress.asset.asc()).all()
        status_by_asset: dict[str, dict[str, Any]] = {}
        grouped: dict[str, list[WalletAddress]] = {}
        for row in rows:
            grouped.setdefault(row.asset.upper(), []).append(row)
        for asset, addresses in grouped.items():
            checked_total = 0.0
            checked_count = 0
            failed_reasons: list[str] = []
            unconfirmed_reasons: list[str] = []
            unchecked_count = 0
            latest_checked_at: str | None = None
            for address in addresses:
                metadata = address.encrypted_metadata or {}
                status = str(metadata.get("last_sync_status") or "not_synced")
                reason = str(metadata.get("last_sync_reason") or "")
                checked_at = metadata.get("last_sync_checked_at")
                if isinstance(checked_at, str) and (latest_checked_at is None or checked_at > latest_checked_at):
                    latest_checked_at = checked_at
                if status == "checked":
                    checked_count += 1
                    checked_total += _safe_float(metadata.get("last_checked_balance"))
                elif status == "failed":
                    failed_reasons.append(reason or "chain balance check failed")
                elif status == "unconfirmed":
                    unconfirmed_reasons.append(reason or "waiting for confirmations")
                else:
                    unchecked_count += 1
            if failed_reasons and checked_count == 0:
                sync_status = "sync_failed"
                sync_reason = "; ".join(dict.fromkeys(failed_reasons))
            elif failed_reasons:
                sync_status = "partial_sync_failed"
                sync_reason = "; ".join(dict.fromkeys(failed_reasons))
            elif unconfirmed_reasons and checked_count == 0:
                sync_status = "unconfirmed"
                sync_reason = "; ".join(dict.fromkeys(unconfirmed_reasons))
            elif checked_count > 0 and unchecked_count == 0:
                sync_status = "verified"
                sync_reason = "Verified against active on-chain wallet address."
            elif checked_count > 0:
                sync_status = "partial"
                sync_reason = "Some active wallet addresses have not synced yet."
            else:
                sync_status = "not_synced"
                sync_reason = "Active wallet address has not synced yet."
            status_by_asset[asset] = {
                "status": sync_status,
                "reason": sync_reason,
                "checked_at": latest_checked_at,
                "on_chain_total": checked_total if checked_count > 0 else None,
            }
        return status_by_asset

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


def _snapshot_rows(rows: Any, *, limit: int = 150) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return payload
    for row in rows[: max(0, int(limit or 0))]:
        if not isinstance(row, dict):
            continue
        payload.append({str(key): _json_safe_value(value) for key, value in row.items()})
    return payload


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    return str(value)


def _balance_index(rows: Any) -> dict[str, dict[str, float | str]]:
    if not isinstance(rows, list):
        return {}
    indexed: dict[str, dict[str, float | str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        asset = str(row.get("asset") or "").strip().upper()
        if not asset:
            continue
        available = _safe_float(row.get("available_balance"))
        locked = _safe_float(row.get("locked_balance"))
        explicit_total = row.get("total_balance")
        total = _safe_float(explicit_total) if explicit_total is not None else available + locked
        indexed[asset] = {
            "asset": asset,
            "available_balance": available,
            "locked_balance": locked,
            "total_balance": total,
            "estimated_usd_value": _safe_float(row.get("estimated_usd_value")),
        }
    return indexed


def _empty_balance(asset: str) -> dict[str, float | str]:
    return {
        "asset": asset,
        "available_balance": 0.0,
        "locked_balance": 0.0,
        "total_balance": 0.0,
        "estimated_usd_value": 0.0,
    }


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
