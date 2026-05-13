"""Database models used by the dashboard."""

from __future__ import annotations

import json
import secrets
from datetime import datetime
from typing import Any

from sqlalchemy.orm import synonym

from .extensions import db


def utcnow() -> datetime:
    """Return a UTC timestamp for SQLAlchemy defaults."""

    return datetime.utcnow()


def public_token(prefix: str) -> str:
    """Return an opaque, URL-safe public identifier."""

    return f"{prefix}_{secrets.token_urlsafe(16).replace('-', '').replace('_', '')[:22]}"


class Setting(db.Model):
    """Simple JSON-backed key-value store for runtime control state."""

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(120), unique=True, nullable=False, index=True)
    value = db.Column(db.Text, nullable=False)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    @classmethod
    def get_json(cls, key: str, default: Any = None) -> Any:
        record = cls.query.filter_by(key=key).one_or_none()
        if record is None:
            return default
        try:
            return json.loads(record.value)
        except json.JSONDecodeError:
            return default

    @classmethod
    def set_json(cls, key: str, value: Any) -> Setting:
        record = cls.query.filter_by(key=key).one_or_none()
        serialized = json.dumps(value)
        if record is None:
            record = cls(key=key, value=serialized)
            db.session.add(record)
        else:
            record.value = serialized
        return record

    @classmethod
    def ensure_json(cls, key: str, default: Any) -> Any:
        record = cls.query.filter_by(key=key).one_or_none()
        if record is None:
            cls.set_json(key, default)
            return default
        return cls.get_json(key, default)


class WorkerLease(db.Model):
    """Database-backed lease for singleton worker jobs."""

    id = db.Column(db.Integer, primary_key=True)
    lease_name = db.Column(db.String(160), nullable=False, unique=True, index=True)
    owner_id = db.Column(db.String(160), nullable=False, default="", index=True)
    acquired_at = db.Column(db.DateTime, nullable=True, index=True)
    heartbeat_at = db.Column(db.DateTime, nullable=True, index=True)
    expires_at = db.Column(db.DateTime, nullable=True, index=True)
    status = db.Column(db.String(32), nullable=False, default="released", index=True)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {})


class WorkerJobRun(db.Model):
    """Idempotency and audit state for recurring worker executions."""

    id = db.Column(db.Integer, primary_key=True)
    job_name = db.Column(db.String(160), nullable=False, index=True)
    idempotency_key = db.Column(db.String(220), nullable=False, unique=True, index=True)
    lease_name = db.Column(db.String(160), nullable=False, default="", index=True)
    owner_id = db.Column(db.String(160), nullable=False, default="", index=True)
    status = db.Column(db.String(32), nullable=False, default="running", index=True)
    started_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    completed_at = db.Column(db.DateTime, nullable=True, index=True)
    failure_reason = db.Column(db.Text, nullable=True)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {})


class User(db.Model):
    """Application user with role-based access and optional authenticator 2FA."""

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(32), nullable=False, default="user", index=True)
    referral_invite_code_id = db.Column(db.Integer, db.ForeignKey("referral_invite_code.id"), nullable=True, index=True)
    totp_secret_encrypted = db.Column(db.Text, nullable=True)
    two_factor_enabled_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    referral_invite_code = db.relationship("ReferralInviteCode", foreign_keys=[referral_invite_code_id], backref="users")

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def two_factor_enabled(self) -> bool:
        return bool(self.two_factor_enabled_at and self.totp_secret_encrypted)


class ReferralInviteCode(db.Model):
    """Admin-managed signup code that snapshots treasury profit-share policy."""

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(40), unique=True, nullable=False, default=lambda: public_token("inv"), index=True)
    code = db.Column(db.String(80), unique=True, nullable=False, index=True)
    label = db.Column(db.String(120), nullable=False, default="")
    percent_profit = db.Column(db.Float, nullable=False, default=50.0)
    profit_share_percent = db.Column(db.Float, nullable=False, default=0.0)
    profit_share_wallet = db.Column(db.String(120), nullable=False, default="sufyanh", index=True)
    profit_share_starts_at = db.Column(db.DateTime, nullable=True, index=True)
    profit_share_ends_at = db.Column(db.DateTime, nullable=True, index=True)
    profit_share_active = db.Column(db.Boolean, nullable=False, default=True, index=True)
    applies_to_vault_types_json = db.Column(db.Text, nullable=False, default="[]")
    expires_at = db.Column(db.DateTime, nullable=True, index=True)
    assigned_role = db.Column(db.String(32), nullable=False, default="user", index=True)
    is_active = db.Column(db.Boolean, nullable=False, default=True, index=True)
    max_uses = db.Column(db.Integer, nullable=False, default=0)
    usage_count = db.Column(db.Integer, nullable=False, default=0)
    created_by_user_id = db.Column(db.Integer, nullable=True, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)
    disabled_at = db.Column(db.DateTime, nullable=True)
    deleted_at = db.Column(db.DateTime, nullable=True, index=True)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {}, default=str)

    @property
    def applies_to_vault_types(self) -> list[str]:
        try:
            value = json.loads(self.applies_to_vault_types_json or "[]")
        except json.JSONDecodeError:
            return []
        return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []

    @applies_to_vault_types.setter
    def applies_to_vault_types(self, value: list[str]) -> None:
        self.applies_to_vault_types_json = json.dumps([str(item).strip() for item in value or [] if str(item).strip()])

    @property
    def effective_profit_share_percent(self) -> float:
        configured = self.profit_share_percent
        if configured is None or (float(configured or 0.0) == 0.0 and float(self.percent_profit or 0.0) > 0.0):
            configured = self.percent_profit
        return max(0.0, min(float(configured or 0.0), 100.0))

    @property
    def lifecycle_status(self) -> str:
        now = utcnow()
        if self.deleted_at is not None:
            return "deleted"
        if not self.is_active:
            return "disabled"
        if self.expires_at is not None and self.expires_at <= now:
            return "expired"
        if int(self.max_uses or 0) > 0 and int(self.usage_count or 0) >= int(self.max_uses or 0):
            return "fully_used"
        return "active"

    @property
    def available(self) -> bool:
        if self.lifecycle_status != "active":
            return False
        return int(self.max_uses or 0) <= 0 or int(self.usage_count or 0) < int(self.max_uses or 0)


class InviteCodeUsage(db.Model):
    """Immutable record of a user accepting an invite code."""

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(40), unique=True, nullable=False, default=lambda: public_token("icu"), index=True)
    invite_code_id = db.Column(db.Integer, db.ForeignKey("referral_invite_code.id"), nullable=False, index=True)
    invitee_user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    used_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    status = db.Column(db.String(32), nullable=False, default="accepted", index=True)
    accepted_disclosure_version = db.Column(db.String(64), nullable=False, default="invite-profit-share-v1")
    metadata_json = db.Column(db.Text, nullable=False, default="{}")

    invite_code = db.relationship("ReferralInviteCode", backref="usage_records")
    invitee_user = db.relationship("User", backref="invite_code_usages")
    __table_args__ = (
        db.UniqueConstraint("invitee_user_id", name="uq_invite_code_usage_invitee"),
        db.Index("ix_invite_code_usage_code_status_used", "invite_code_id", "status", "used_at"),
    )

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {}, default=str)


class StrategyRun(db.Model):
    """Tracks strategy execution state for UI control and worker recovery."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    trading_connection_id = db.Column(db.Integer, db.ForeignKey("trading_connection.id"), nullable=True, index=True)
    strategy_name = db.Column(db.String(120), nullable=False, index=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    timeframe = db.Column(db.String(16), nullable=False)
    mode = db.Column(db.String(16), nullable=False, default="live")
    status = db.Column(db.String(32), nullable=False, default="stopped")
    parameters_json = db.Column(db.Text, nullable=False, default="{}")
    last_signal_json = db.Column(db.Text, nullable=False, default="{}")
    lock_duration_seconds = db.Column(db.Integer, nullable=False, default=3600)
    manual_enabled = db.Column(db.Boolean, nullable=False, default=True)
    last_heartbeat_at = db.Column(db.DateTime, nullable=True)
    last_error = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    user = db.relationship("User", backref="strategy_runs")
    trading_connection = db.relationship("TradingConnection", backref="strategy_runs")
    __table_args__ = (
        db.Index("ix_strategy_run_user_status_created", "user_id", "status", "created_at"),
        db.Index("ix_strategy_run_status_updated", "status", "updated_at"),
        db.Index("ix_strategy_run_connection_status_created", "trading_connection_id", "status", "created_at"),
    )

    @property
    def parameters(self) -> dict[str, Any]:
        try:
            return json.loads(self.parameters_json or "{}")
        except json.JSONDecodeError:
            return {}

    @parameters.setter
    def parameters(self, value: dict[str, Any]) -> None:
        self.parameters_json = json.dumps(value or {})

    @property
    def last_signal(self) -> dict[str, Any]:
        try:
            return json.loads(self.last_signal_json or "{}")
        except json.JSONDecodeError:
            return {}

    @last_signal.setter
    def last_signal(self, value: dict[str, Any]) -> None:
        self.last_signal_json = json.dumps(value or {})


class WalletBalance(db.Model):
    """Consumer-facing wallet balance tracked by asset."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    active_deposit_address_id = db.Column(db.Integer, db.ForeignKey("deposit_address.id"), nullable=True)
    asset = db.Column(db.String(32), nullable=False, index=True)
    available_balance = db.Column(db.Float, nullable=False, default=0.0)
    locked_balance = db.Column(db.Float, nullable=False, default=0.0)
    estimated_usd_value = db.Column(db.Float, nullable=False, default=0.0)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    __table_args__ = (db.UniqueConstraint("user_id", "asset", name="uq_wallet_balance_user_asset"),)

    user = db.relationship("User", foreign_keys=[user_id], backref="wallet_balances")
    active_deposit_address = db.relationship("DepositAddress", foreign_keys=[active_deposit_address_id], post_update=True)

    @property
    def total_balance(self) -> float:
        return float(self.available_balance or 0.0) + float(self.locked_balance or 0.0)


class VaultCycle(db.Model):
    """Consumer allocation cycle that owns a linked strategy run."""

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(40), unique=True, nullable=False, default=lambda: public_token("vc"), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    trading_connection_id = db.Column(db.Integer, db.ForeignKey("trading_connection.id"), nullable=True, index=True)
    strategy_run_id = db.Column(db.Integer, db.ForeignKey("strategy_run.id"), nullable=True, index=True)
    deposit_asset = db.Column(db.String(32), nullable=False, index=True)
    deposit_amount = db.Column(db.Float, nullable=False)
    settlement_asset = db.Column(db.String(32), nullable=False, default="USDC")
    lock_duration_hours = db.Column(db.Integer, nullable=False)
    lock_duration_seconds = db.Column(db.Integer, nullable=False, default=3600)
    status = db.Column(db.String(32), nullable=False, default="active", index=True)
    execution_substatus = db.Column(db.String(32), nullable=False, default="initializing", index=True)
    execution_mode = db.Column(db.String(32), nullable=False, default="live", index=True)
    live_validation_status = db.Column(db.String(32), nullable=False, default="not_required", index=True)
    validation_started_at = db.Column(db.DateTime, nullable=True)
    validation_completed_at = db.Column(db.DateTime, nullable=True)
    validation_failure_reason = db.Column(db.Text, nullable=True)
    algorithm_profile = db.Column(db.String(32), nullable=False, default="Balanced")
    selected_strategy_name = db.Column(db.String(120), nullable=True)
    selected_timeframe = db.Column(db.String(16), nullable=True)
    selection_metadata_json = db.Column(db.Text, nullable=False, default="{}")
    started_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    unlocks_at = db.Column(db.DateTime, nullable=False, index=True)
    settled_at = db.Column(db.DateTime, nullable=True)
    starting_value_usd = db.Column(db.Float, nullable=False, default=0.0)
    current_estimated_value_usd = db.Column(db.Float, nullable=False, default=0.0)
    final_settlement_amount = db.Column(db.Float, nullable=True)
    cycle_summary_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    strategy_run = db.relationship("StrategyRun", backref="vault_cycles")
    user = db.relationship("User", backref="vault_cycles")
    trading_connection = db.relationship("TradingConnection", backref="vault_cycles")
    __table_args__ = (
        db.Index("ix_vault_cycle_user_status_started", "user_id", "status", "started_at"),
        db.Index("ix_vault_cycle_user_unlocks", "user_id", "unlocks_at"),
        db.Index("ix_vault_cycle_connection_status_started", "trading_connection_id", "status", "started_at"),
    )

    @property
    def expires_at(self) -> datetime:
        return self.unlocks_at

    @expires_at.setter
    def expires_at(self, value: datetime) -> None:
        self.unlocks_at = value

    @property
    def selection_metadata(self) -> dict[str, Any]:
        try:
            value = json.loads(self.selection_metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @selection_metadata.setter
    def selection_metadata(self, value: dict[str, Any]) -> None:
        self.selection_metadata_json = json.dumps(value or {})

    @property
    def cycle_summary(self) -> dict[str, Any]:
        try:
            value = json.loads(self.cycle_summary_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @cycle_summary.setter
    def cycle_summary(self, value: dict[str, Any]) -> None:
        self.cycle_summary_json = json.dumps(value or {}, default=str)


class VaultAllocationLeg(db.Model):
    """Per-symbol execution leg owned by a consumer vault cycle."""

    id = db.Column(db.Integer, primary_key=True)
    vault_cycle_id = db.Column(db.Integer, db.ForeignKey("vault_cycle.id"), nullable=False, index=True)
    strategy_run_id = db.Column(db.Integer, db.ForeignKey("strategy_run.id"), nullable=True, index=True)
    optimizer_ranking_id = db.Column(db.Integer, db.ForeignKey("strategy_ranking.id"), nullable=True, index=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    timeframe = db.Column(db.String(16), nullable=False)
    provider = db.Column(db.String(64), nullable=False, default="global", index=True)
    trading_connection_id = db.Column(db.Integer, db.ForeignKey("trading_connection.id"), nullable=True, index=True)
    allocation_cap_usd = db.Column(db.Float, nullable=False, default=0.0)
    leverage = db.Column(db.Float, nullable=False, default=1.0)
    status = db.Column(db.String(32), nullable=False, default="active", index=True)
    realized_pnl_usd = db.Column(db.Float, nullable=False, default=0.0)
    unrealized_pnl_usd = db.Column(db.Float, nullable=False, default=0.0)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    vault_cycle = db.relationship("VaultCycle", backref="allocation_legs")
    strategy_run = db.relationship("StrategyRun", backref="vault_allocation_legs")
    optimizer_ranking = db.relationship("StrategyRanking", backref="vault_allocation_legs")
    trading_connection = db.relationship("TradingConnection", backref="vault_allocation_legs")
    __table_args__ = (
        db.Index("ix_vault_leg_cycle_status", "vault_cycle_id", "status"),
        db.Index("ix_vault_leg_run_status", "strategy_run_id", "status"),
        db.Index("ix_vault_leg_connection_symbol_status", "trading_connection_id", "symbol", "status"),
    )

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {})


class VaultCycleAllocation(db.Model):
    """Exchange-level capital allocation owned by a broader Vault Cycle."""

    id = db.Column(db.Integer, primary_key=True)
    vault_cycle_id = db.Column(db.Integer, db.ForeignKey("vault_cycle.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    trading_connection_id = db.Column(db.Integer, db.ForeignKey("trading_connection.id"), nullable=False, index=True)
    provider = db.Column(db.String(64), nullable=False, index=True)
    settlement_asset = db.Column(db.String(32), nullable=False, index=True)
    collateral_asset = db.Column(db.String(32), nullable=False, index=True)
    target_amount = db.Column(db.Float, nullable=False, default=0.0)
    allocated_amount = db.Column(db.Float, nullable=False, default=0.0)
    allocation_weight = db.Column(db.Float, nullable=False, default=0.0)
    status = db.Column(db.String(32), nullable=False, default="pending", index=True)
    risk_adjusted_score = db.Column(db.Float, nullable=False, default=0.0)
    opportunity_score = db.Column(db.Float, nullable=False, default=0.0)
    liquidity_score = db.Column(db.Float, nullable=False, default=0.0)
    slippage_bps = db.Column(db.Float, nullable=False, default=0.0)
    max_leverage = db.Column(db.Float, nullable=False, default=1.0)
    max_symbol_exposure_usd = db.Column(db.Float, nullable=False, default=0.0)
    max_concurrent_positions = db.Column(db.Integer, nullable=False, default=1)
    score_json = db.Column(db.Text, nullable=False, default="{}")
    constraints_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)
    funded_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)

    vault_cycle = db.relationship("VaultCycle", backref="exchange_allocations")
    user = db.relationship("User", backref="vault_cycle_allocations")
    trading_connection = db.relationship("TradingConnection", backref="vault_cycle_allocations")

    __table_args__ = (
        db.Index("ix_vault_cycle_allocation_cycle_status", "vault_cycle_id", "status"),
        db.Index("ix_vault_cycle_allocation_user_provider_status", "user_id", "provider", "status"),
        db.UniqueConstraint("vault_cycle_id", "trading_connection_id", name="uq_vault_cycle_allocation_connection"),
    )

    @property
    def scores(self) -> dict[str, Any]:
        try:
            value = json.loads(self.score_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @scores.setter
    def scores(self, value: dict[str, Any]) -> None:
        self.score_json = json.dumps(value or {}, default=str)

    @property
    def constraints(self) -> dict[str, Any]:
        try:
            value = json.loads(self.constraints_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @constraints.setter
    def constraints(self, value: dict[str, Any]) -> None:
        self.constraints_json = json.dumps(value or {}, default=str)


class VaultCycleTransfer(db.Model):
    """Idempotent funding, reserve, conversion, and withdrawal state for a cycle."""

    id = db.Column(db.Integer, primary_key=True)
    vault_cycle_id = db.Column(db.Integer, db.ForeignKey("vault_cycle.id"), nullable=False, index=True)
    allocation_id = db.Column(db.Integer, db.ForeignKey("vault_cycle_allocation.id"), nullable=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    trading_connection_id = db.Column(db.Integer, db.ForeignKey("trading_connection.id"), nullable=True, index=True)
    provider = db.Column(db.String(64), nullable=False, default="", index=True)
    direction = db.Column(db.String(32), nullable=False, index=True)
    transfer_type = db.Column(db.String(32), nullable=False, default="exchange_reserve", index=True)
    asset = db.Column(db.String(32), nullable=False, index=True)
    network = db.Column(db.String(64), nullable=True)
    source = db.Column(db.Text, nullable=True)
    destination = db.Column(db.Text, nullable=True)
    requested_amount = db.Column(db.Float, nullable=False, default=0.0)
    confirmed_amount = db.Column(db.Float, nullable=False, default=0.0)
    fee_amount = db.Column(db.Float, nullable=False, default=0.0)
    fee_asset = db.Column(db.String(32), nullable=True)
    idempotency_key = db.Column(db.String(180), nullable=False, unique=True, index=True)
    provider_reference = db.Column(db.String(180), nullable=True, index=True)
    status = db.Column(db.String(32), nullable=False, default="pending", index=True)
    attempts = db.Column(db.Integer, nullable=False, default=0)
    failure_reason = db.Column(db.Text, nullable=True)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    requested_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    submitted_at = db.Column(db.DateTime, nullable=True)
    confirmed_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    vault_cycle = db.relationship("VaultCycle", backref="vault_transfers")
    allocation = db.relationship("VaultCycleAllocation", backref="transfers")
    user = db.relationship("User", backref="vault_cycle_transfers")
    trading_connection = db.relationship("TradingConnection", backref="vault_cycle_transfers")

    __table_args__ = (
        db.Index("ix_vault_cycle_transfer_cycle_status", "vault_cycle_id", "status"),
        db.Index("ix_vault_cycle_transfer_allocation_direction", "allocation_id", "direction"),
        db.Index("ix_vault_cycle_transfer_user_direction_created", "user_id", "direction", "created_at"),
    )

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {}, default=str)


class VaultCycleTrade(db.Model):
    """Cycle-local trade snapshot linked back to canonical orders and fills."""

    id = db.Column(db.Integer, primary_key=True)
    vault_cycle_id = db.Column(db.Integer, db.ForeignKey("vault_cycle.id"), nullable=False, index=True)
    allocation_id = db.Column(db.Integer, db.ForeignKey("vault_cycle_allocation.id"), nullable=True, index=True)
    order_id = db.Column(db.Integer, db.ForeignKey("order.id"), nullable=True, index=True)
    fill_id = db.Column(db.Integer, db.ForeignKey("fill.id"), nullable=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    trading_connection_id = db.Column(db.Integer, db.ForeignKey("trading_connection.id"), nullable=True, index=True)
    provider = db.Column(db.String(64), nullable=False, default="", index=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    side = db.Column(db.String(8), nullable=False, default="")
    status = db.Column(db.String(32), nullable=False, default="recorded", index=True)
    quantity = db.Column(db.Float, nullable=False, default=0.0)
    notional_usd = db.Column(db.Float, nullable=False, default=0.0)
    fee_usd = db.Column(db.Float, nullable=False, default=0.0)
    realized_pnl_usd = db.Column(db.Float, nullable=False, default=0.0)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    vault_cycle = db.relationship("VaultCycle", backref="vault_trades")
    allocation = db.relationship("VaultCycleAllocation", backref="trades")
    order = db.relationship("Order", backref="vault_trade_links")
    fill = db.relationship("Fill", backref="vault_trade_links")
    user = db.relationship("User", backref="vault_cycle_trades")
    trading_connection = db.relationship("TradingConnection", backref="vault_cycle_trades")

    __table_args__ = (
        db.Index("ix_vault_cycle_trade_cycle_created", "vault_cycle_id", "created_at"),
        db.UniqueConstraint("vault_cycle_id", "order_id", "fill_id", name="uq_vault_cycle_trade_order_fill"),
    )

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {}, default=str)


class VaultCycleSettlement(db.Model):
    """Final settlement and reconciliation result for a Vault Cycle."""

    id = db.Column(db.Integer, primary_key=True)
    vault_cycle_id = db.Column(db.Integer, db.ForeignKey("vault_cycle.id"), nullable=False, unique=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    settlement_asset = db.Column(db.String(32), nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, default="pending", index=True)
    starting_value_usd = db.Column(db.Float, nullable=False, default=0.0)
    gross_value_usd = db.Column(db.Float, nullable=False, default=0.0)
    final_amount = db.Column(db.Float, nullable=False, default=0.0)
    final_value_usd = db.Column(db.Float, nullable=False, default=0.0)
    gross_pnl_usd = db.Column(db.Float, nullable=False, default=0.0)
    fees_usd = db.Column(db.Float, nullable=False, default=0.0)
    net_pnl_usd = db.Column(db.Float, nullable=False, default=0.0)
    roi_pct = db.Column(db.Float, nullable=False, default=0.0)
    conversion_status = db.Column(db.String(32), nullable=False, default="not_required")
    withdrawal_status = db.Column(db.String(32), nullable=False, default="not_started")
    provider_reference = db.Column(db.String(180), nullable=True, index=True)
    failure_reason = db.Column(db.Text, nullable=True)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    started_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    vault_cycle = db.relationship("VaultCycle", backref=db.backref("vault_settlement", uselist=False))
    user = db.relationship("User", backref="vault_cycle_settlements")

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {}, default=str)


class VaultCycleRiskEvent(db.Model):
    """Vault Cycle risk, transfer, and settlement event mirror for reporting."""

    id = db.Column(db.Integer, primary_key=True)
    vault_cycle_id = db.Column(db.Integer, db.ForeignKey("vault_cycle.id"), nullable=False, index=True)
    allocation_id = db.Column(db.Integer, db.ForeignKey("vault_cycle_allocation.id"), nullable=True, index=True)
    transfer_id = db.Column(db.Integer, db.ForeignKey("vault_cycle_transfer.id"), nullable=True, index=True)
    risk_event_id = db.Column(db.Integer, db.ForeignKey("risk_event.id"), nullable=True, index=True)
    order_id = db.Column(db.Integer, db.ForeignKey("order.id"), nullable=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    trading_connection_id = db.Column(db.Integer, db.ForeignKey("trading_connection.id"), nullable=True, index=True)
    provider = db.Column(db.String(64), nullable=False, default="", index=True)
    category = db.Column(db.String(64), nullable=False, default="risk", index=True)
    severity = db.Column(db.String(32), nullable=False, default="info", index=True)
    rule_name = db.Column(db.String(120), nullable=False, default="", index=True)
    reason = db.Column(db.Text, nullable=False, default="")
    status = db.Column(db.String(32), nullable=False, default="recorded", index=True)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    vault_cycle = db.relationship("VaultCycle", backref="vault_risk_events")
    allocation = db.relationship("VaultCycleAllocation", backref="risk_events")
    transfer = db.relationship("VaultCycleTransfer", backref="risk_events")
    risk_event = db.relationship("RiskEvent", backref="vault_cycle_events")
    order = db.relationship("Order", backref="vault_risk_event_links")
    user = db.relationship("User", backref="vault_cycle_risk_events")
    trading_connection = db.relationship("TradingConnection", backref="vault_cycle_risk_events")

    __table_args__ = (
        db.Index("ix_vault_cycle_risk_cycle_created", "vault_cycle_id", "created_at"),
        db.Index("ix_vault_cycle_risk_user_severity_created", "user_id", "severity", "created_at"),
    )

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {}, default=str)


class LeveragedMarket(db.Model):
    """Provider futures/perpetual market metadata used by 1H10 discovery."""

    id = db.Column(db.Integer, primary_key=True)
    provider = db.Column(db.String(64), nullable=False, index=True)
    trading_connection_id = db.Column(db.Integer, db.ForeignKey("trading_connection.id"), nullable=True, index=True)
    venue_symbol = db.Column(db.String(64), nullable=False, index=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, default="active", index=True)
    settlement_asset = db.Column(db.String(32), nullable=False, default="USDC", index=True)
    max_leverage = db.Column(db.Float, nullable=False, default=1.0)
    tick_size = db.Column(db.Float, nullable=False, default=0.0)
    lot_size = db.Column(db.Float, nullable=False, default=0.0)
    contract_size = db.Column(db.Float, nullable=False, default=0.0)
    min_size = db.Column(db.Float, nullable=False, default=0.0)
    funding_rate = db.Column(db.Float, nullable=False, default=0.0)
    liquidity_usd = db.Column(db.Float, nullable=False, default=0.0)
    spread_bps = db.Column(db.Float, nullable=False, default=0.0)
    fee_bps = db.Column(db.Float, nullable=False, default=0.0)
    raw_json = db.Column(db.Text, nullable=False, default="{}")
    last_seen_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    trading_connection = db.relationship("TradingConnection", backref="leveraged_markets")

    __table_args__ = (
        db.UniqueConstraint("provider", "venue_symbol", name="uq_leveraged_market_provider_symbol"),
    )

    @property
    def raw(self) -> dict[str, Any]:
        try:
            value = json.loads(self.raw_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @raw.setter
    def raw(self, value: dict[str, Any]) -> None:
        self.raw_json = json.dumps(value or {})


class LeveragedMarketFeature(db.Model):
    """Persisted higher-timeframe 1H10 feature payloads per provider market."""

    id = db.Column(db.Integer, primary_key=True)
    leveraged_market_id = db.Column(db.Integer, db.ForeignKey("leveraged_market.id"), nullable=False, index=True)
    provider = db.Column(db.String(64), nullable=False, index=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    timeframe = db.Column(db.String(16), nullable=False, index=True)
    feature_schema_version = db.Column(db.String(64), nullable=False, default="1h10_feature_v1")
    features_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    leveraged_market = db.relationship("LeveragedMarket", backref="feature_rows")

    __table_args__ = (
        db.UniqueConstraint(
            "leveraged_market_id",
            "timeframe",
            name="uq_leveraged_market_feature_timeframe",
        ),
    )

    @property
    def features(self) -> dict[str, Any]:
        try:
            value = json.loads(self.features_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @features.setter
    def features(self, value: dict[str, Any]) -> None:
        self.features_json = json.dumps(value or {})


class MarketForecast(db.Model):
    """Persisted dashboard forecast snapshots for ranked market opportunities."""

    id = db.Column(db.Integer, primary_key=True)
    provider = db.Column(db.String(64), nullable=False, default="global", index=True)
    venue_symbol = db.Column(db.String(64), nullable=False, default="", index=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    timeframe = db.Column(db.String(16), nullable=False, default="1m", index=True)
    horizon = db.Column(db.String(32), nullable=False, default="1h10", index=True)
    side = db.Column(db.String(8), nullable=False, default="hold", index=True)
    confidence = db.Column(db.Float, nullable=False, default=0.0)
    expected_return_bps = db.Column(db.Float, nullable=False, default=0.0)
    entry_price = db.Column(db.Float, nullable=False, default=0.0)
    exit_price = db.Column(db.Float, nullable=False, default=0.0)
    stop_loss_price = db.Column(db.Float, nullable=False, default=0.0)
    risk_reward = db.Column(db.Float, nullable=False, default=0.0)
    liquidity_score = db.Column(db.Float, nullable=False, default=0.0)
    slippage_bps = db.Column(db.Float, nullable=False, default=0.0)
    model_agreement = db.Column(db.Float, nullable=False, default=0.0)
    fibonacci_alignment = db.Column(db.Float, nullable=False, default=0.0)
    source = db.Column(db.String(80), nullable=False, default="dashboard")
    forecast_path_json = db.Column(db.Text, nullable=False, default="[]")
    zones_json = db.Column(db.Text, nullable=False, default="{}")
    score_breakdown_json = db.Column(db.Text, nullable=False, default="{}")
    payload_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    expires_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    __table_args__ = (
        db.Index("ix_market_forecast_provider_symbol_tf_created", "provider", "symbol", "timeframe", "created_at"),
        db.Index("ix_market_forecast_provider_expires", "provider", "expires_at"),
    )

    @property
    def forecast_path(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.forecast_path_json or "[]")
        except json.JSONDecodeError:
            return []
        return value if isinstance(value, list) else []

    @forecast_path.setter
    def forecast_path(self, value: list[dict[str, Any]]) -> None:
        self.forecast_path_json = json.dumps(value or [])

    @property
    def zones(self) -> dict[str, Any]:
        try:
            value = json.loads(self.zones_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @zones.setter
    def zones(self, value: dict[str, Any]) -> None:
        self.zones_json = json.dumps(value or {})

    @property
    def score_breakdown(self) -> dict[str, Any]:
        try:
            value = json.loads(self.score_breakdown_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @score_breakdown.setter
    def score_breakdown(self, value: dict[str, Any]) -> None:
        self.score_breakdown_json = json.dumps(value or {})

    @property
    def payload(self) -> dict[str, Any]:
        try:
            value = json.loads(self.payload_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @payload.setter
    def payload(self, value: dict[str, Any]) -> None:
        self.payload_json = json.dumps(value or {})


class WalletTransaction(db.Model):
    """Append-only wallet activity for consumer-facing history."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    vault_cycle_id = db.Column(db.Integer, db.ForeignKey("vault_cycle.id"), nullable=True, index=True)
    asset = db.Column(db.String(32), nullable=False, index=True)
    amount = db.Column(db.Float, nullable=False)
    transaction_type = db.Column(db.String(32), nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, default="complete", index=True)
    network = db.Column(db.String(64), nullable=True)
    withdraw_address = db.Column(db.Text, nullable=True)
    note = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    vault_cycle = db.relationship("VaultCycle", backref="transactions")
    user = db.relationship("User", backref="wallet_transactions")
    __table_args__ = (
        db.Index("ix_wallet_transaction_user_type_created", "user_id", "transaction_type", "created_at"),
        db.Index("ix_wallet_transaction_cycle_type_status", "vault_cycle_id", "transaction_type", "status"),
    )


class WalletLedgerEvent(db.Model):
    """Idempotent external-chain wallet event used for deposit crediting."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    deposit_address_id = db.Column(db.Integer, db.ForeignKey("deposit_address.id"), nullable=True, index=True)
    wallet_address_id = db.Column(db.Integer, db.ForeignKey("wallet_address.id"), nullable=True, index=True)
    asset = db.Column(db.String(32), nullable=False, index=True)
    network = db.Column(db.String(64), nullable=False, index=True)
    address = db.Column(db.Text, nullable=False, index=True)
    event_type = db.Column(db.String(32), nullable=False, default="deposit", index=True)
    provider_reference = db.Column(db.String(180), nullable=False, index=True)
    idempotency_key = db.Column(db.String(220), nullable=False, unique=True, index=True)
    amount = db.Column(db.Float, nullable=False, default=0.0)
    confirmations = db.Column(db.Integer, nullable=False, default=0)
    status = db.Column(db.String(32), nullable=False, default="pending", index=True)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    user = db.relationship("User", backref="wallet_ledger_events")
    deposit_address = db.relationship("DepositAddress", backref="ledger_events")
    wallet_address = db.relationship("WalletAddress", backref="ledger_events")

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {})


class ProfitSharePayout(db.Model):
    """Append-only invite-code profit-share payout ledger."""

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(40), unique=True, nullable=False, default=lambda: public_token("psp"), index=True)
    invite_code_id = db.Column(db.Integer, db.ForeignKey("referral_invite_code.id"), nullable=False, index=True)
    invitee_user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    destination_user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    vault_cycle_id = db.Column(db.Integer, db.ForeignKey("vault_cycle.id"), nullable=False, index=True)
    vault_cycle_settlement_id = db.Column(db.Integer, db.ForeignKey("vault_cycle_settlement.id"), nullable=True, index=True)
    asset = db.Column(db.String(32), nullable=False, default="USDC", index=True)
    source_profit_amount = db.Column(db.Numeric(18, 8), nullable=False, default=0)
    profit_share_percent = db.Column(db.Numeric(8, 4), nullable=False, default=0)
    payout_amount = db.Column(db.Numeric(18, 8), nullable=False, default=0)
    destination_wallet = db.Column(db.String(120), nullable=False, default="sufyanh", index=True)
    status = db.Column(db.String(32), nullable=False, default="pending", index=True)
    idempotency_key = db.Column(db.String(240), nullable=False, unique=True, index=True)
    failed_reason = db.Column(db.Text, nullable=True)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    completed_at = db.Column(db.DateTime, nullable=True, index=True)

    invite_code = db.relationship("ReferralInviteCode", backref="profit_share_payouts")
    invitee_user = db.relationship("User", foreign_keys=[invitee_user_id], backref="invite_profit_share_payouts")
    destination_user = db.relationship("User", foreign_keys=[destination_user_id], backref="received_invite_profit_share_payouts")
    vault_cycle = db.relationship("VaultCycle", backref="invite_profit_share_payouts")
    vault_cycle_settlement = db.relationship("VaultCycleSettlement", backref="invite_profit_share_payouts")
    __table_args__ = (
        db.UniqueConstraint("vault_cycle_id", "invite_code_id", name="uq_profit_share_payout_cycle_invite"),
        db.Index("ix_profit_share_payout_invite_status_created", "invite_code_id", "status", "created_at"),
    )

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {}, default=str)


class TradingConnection(db.Model):
    """User-scoped exchange or wallet connection with encrypted credentials."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    provider = db.Column(db.String(64), nullable=False, index=True)
    connection_type = db.Column(db.String(64), nullable=False, index=True)
    encrypted_api_key = db.Column(db.Text, nullable=True)
    encrypted_api_secret = db.Column(db.Text, nullable=True)
    encrypted_passphrase = db.Column(db.Text, nullable=True)
    wallet_address = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, nullable=False, default=True, index=True)
    verification_status = db.Column(db.String(32), nullable=False, default="needs_verification", index=True)
    last_verified_at = db.Column(db.DateTime, nullable=True)
    last_verification_error = db.Column(db.Text, nullable=True)
    provider_metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        db.UniqueConstraint("user_id", "provider", "connection_type", name="uq_trading_connection_scope"),
    )

    user = db.relationship("User", backref="trading_connections")

    @property
    def provider_metadata(self) -> dict[str, Any]:
        try:
            value = json.loads(self.provider_metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @provider_metadata.setter
    def provider_metadata(self, value: dict[str, Any]) -> None:
        self.provider_metadata_json = json.dumps(value or {})


class DepositAddress(db.Model):
    """Versioned deposit address history by user, asset, and network."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    asset = db.Column(db.String(32), nullable=False, index=True)
    network = db.Column(db.String(64), nullable=False, default="native", index=True)
    address = db.Column(db.Text, nullable=False, unique=True)
    version = db.Column(db.Integer, nullable=False, default=1)
    is_active = db.Column(db.Boolean, nullable=False, default=True, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    expired_at = db.Column(db.DateTime, nullable=True)
    rotated_from_id = db.Column(db.Integer, db.ForeignKey("deposit_address.id"), nullable=True)

    user = db.relationship("User", foreign_keys=[user_id], backref="deposit_addresses")
    rotated_from = db.relationship("DepositAddress", remote_side=[id])


class WalletAccount(db.Model):
    """Public self-custody wallet account record with no custody secrets."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    provider = db.Column(db.String(64), nullable=False, default="self_custody", index=True)
    asset = db.Column(db.String(32), nullable=False, index=True)
    network = db.Column(db.String(64), nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, default="active", index=True)
    encrypted_metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        db.UniqueConstraint("user_id", "provider", "asset", "network", name="uq_wallet_account_scope"),
    )

    user = db.relationship("User", backref="wallet_accounts")

    @property
    def encrypted_metadata(self) -> dict[str, Any]:
        try:
            value = json.loads(self.encrypted_metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @encrypted_metadata.setter
    def encrypted_metadata(self, value: dict[str, Any]) -> None:
        self.encrypted_metadata_json = json.dumps(value or {})


class WalletAddress(db.Model):
    """Public wallet address state used by fail-closed custody workflows."""

    id = db.Column(db.Integer, primary_key=True)
    wallet_account_id = db.Column(db.Integer, db.ForeignKey("wallet_account.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    deposit_address_id = db.Column(db.Integer, db.ForeignKey("deposit_address.id"), nullable=True, index=True)
    asset = db.Column(db.String(32), nullable=False, index=True)
    network = db.Column(db.String(64), nullable=False, index=True)
    address = db.Column(db.Text, nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, default="active", index=True)
    rotation_index = db.Column(db.Integer, nullable=False, default=1)
    rotated_from_id = db.Column(db.Integer, db.ForeignKey("wallet_address.id"), nullable=True)
    deactivated_at = db.Column(db.DateTime, nullable=True)
    encrypted_metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)
    asset_symbol = synonym("asset")

    wallet_account = db.relationship("WalletAccount", backref="addresses")
    user = db.relationship("User", backref="wallet_addresses")
    deposit_address = db.relationship("DepositAddress", backref="wallet_address_records")
    rotated_from = db.relationship("WalletAddress", remote_side=[id])

    @property
    def encrypted_metadata(self) -> dict[str, Any]:
        try:
            value = json.loads(self.encrypted_metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @encrypted_metadata.setter
    def encrypted_metadata(self, value: dict[str, Any]) -> None:
        self.encrypted_metadata_json = json.dumps(value or {})


class WalletWithdrawal(db.Model):
    """Withdrawal or rotated-address sweep workflow record."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    trading_connection_id = db.Column(db.Integer, db.ForeignKey("trading_connection.id"), nullable=True, index=True)
    wallet_account_id = db.Column(db.Integer, db.ForeignKey("wallet_account.id"), nullable=True, index=True)
    source_wallet_address_id = db.Column(db.Integer, db.ForeignKey("wallet_address.id"), nullable=True, index=True)
    source_deposit_address_id = db.Column(db.Integer, db.ForeignKey("deposit_address.id"), nullable=True, index=True)
    asset = db.Column(db.String(32), nullable=False, index=True)
    network = db.Column(db.String(64), nullable=False, index=True)
    destination_address = db.Column(db.Text, nullable=False)
    amount = db.Column(db.Float, nullable=False, default=0.0)
    amount_eth = db.Column(db.Float, nullable=False, default=0.0)
    fee_eth = db.Column(db.Float, nullable=False, default=0.0)
    status = db.Column(db.String(32), nullable=False, default="pending_approval", index=True)
    workflow_type = db.Column(db.String(32), nullable=False, default="manual_withdrawal", index=True)
    idempotency_token = db.Column(db.String(160), nullable=False, unique=True, index=True)
    provider_reference = db.Column(db.String(160), nullable=True, index=True)
    failure_reason = db.Column(db.Text, nullable=True)
    treasury_safety_status = db.Column(db.String(32), nullable=False, default="unchecked", index=True)
    treasury_safety_reason = db.Column(db.Text, nullable=True)
    treasury_estimated_gas_eth = db.Column(db.Float, nullable=False, default=0.0)
    treasury_safety_checked_at = db.Column(db.DateTime, nullable=True)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)
    approved_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship("User", backref="wallet_withdrawals")
    trading_connection = db.relationship("TradingConnection", backref="wallet_withdrawals")
    wallet_account = db.relationship("WalletAccount", backref="withdrawals")
    source_wallet_address = db.relationship("WalletAddress", backref="withdrawals")
    source_deposit_address = db.relationship("DepositAddress", backref="wallet_withdrawals")

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {})


Withdrawal = WalletWithdrawal


class PlatformTreasuryWallet(db.Model):
    """Platform-managed EVM treasury wallet used for withdrawal gas top-ups."""

    id = db.Column(db.Integer, primary_key=True)
    network = db.Column(db.String(64), nullable=False, default="Ethereum", index=True)
    address = db.Column(db.Text, nullable=False, unique=True)
    encrypted_private_key = db.Column(db.Text, nullable=False)
    is_active = db.Column(db.Boolean, nullable=False, default=True, index=True)
    rotation_index = db.Column(db.Integer, nullable=False, default=1)
    rotated_from_wallet_id = db.Column(db.Integer, db.ForeignKey("platform_treasury_wallet.id"), nullable=True)
    created_by_user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)
    rotated_at = db.Column(db.DateTime, nullable=True)

    rotated_from_wallet = db.relationship("PlatformTreasuryWallet", remote_side=[id])
    created_by_user = db.relationship("User", backref="created_treasury_wallets")


class PlatformTreasuryReserveJob(db.Model):
    """Idempotent treasury work item for gas reserves and profit-share conversion."""

    id = db.Column(db.Integer, primary_key=True)
    treasury_wallet_id = db.Column(db.Integer, db.ForeignKey("platform_treasury_wallet.id"), nullable=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    wallet_ledger_event_id = db.Column(db.Integer, db.ForeignKey("wallet_ledger_event.id"), nullable=True, index=True)
    wallet_withdrawal_id = db.Column(db.Integer, db.ForeignKey("wallet_withdrawal.id"), nullable=True, index=True)
    vault_cycle_id = db.Column(db.Integer, db.ForeignKey("vault_cycle.id"), nullable=True, index=True)
    vault_cycle_settlement_id = db.Column(db.Integer, db.ForeignKey("vault_cycle_settlement.id"), nullable=True, index=True)
    referral_invite_code_id = db.Column(db.Integer, db.ForeignKey("referral_invite_code.id"), nullable=True, index=True)
    job_type = db.Column(db.String(64), nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, default="pending", index=True)
    asset = db.Column(db.String(32), nullable=False, default="ETH", index=True)
    network = db.Column(db.String(64), nullable=False, default="Ethereum", index=True)
    source_amount = db.Column(db.Float, nullable=False, default=0.0)
    source_amount_usd = db.Column(db.Float, nullable=False, default=0.0)
    reserve_eth_estimate = db.Column(db.Float, nullable=False, default=0.0)
    reserve_multiplier = db.Column(db.Float, nullable=False, default=2.0)
    reserve_eth_target = db.Column(db.Float, nullable=False, default=0.0)
    conversion_provider = db.Column(db.String(64), nullable=False, default="")
    conversion_asset = db.Column(db.String(32), nullable=False, default="")
    conversion_amount = db.Column(db.Float, nullable=False, default=0.0)
    converted_eth_amount = db.Column(db.Float, nullable=False, default=0.0)
    provider_reference = db.Column(db.String(180), nullable=True, index=True)
    treasury_tx_reference = db.Column(db.String(180), nullable=True, index=True)
    retry_count = db.Column(db.Integer, nullable=False, default=0)
    max_attempts = db.Column(db.Integer, nullable=False, default=3)
    next_retry_at = db.Column(db.DateTime, nullable=True, index=True)
    failure_reason = db.Column(db.Text, nullable=True)
    idempotency_key = db.Column(db.String(240), nullable=False, unique=True, index=True)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

    treasury_wallet = db.relationship("PlatformTreasuryWallet", backref="reserve_jobs")
    user = db.relationship("User", backref="platform_treasury_reserve_jobs")
    wallet_ledger_event = db.relationship("WalletLedgerEvent", backref="treasury_reserve_jobs")
    wallet_withdrawal = db.relationship("WalletWithdrawal", backref="treasury_reserve_jobs")
    vault_cycle = db.relationship("VaultCycle", backref="treasury_reserve_jobs")
    vault_cycle_settlement = db.relationship("VaultCycleSettlement", backref="treasury_reserve_jobs")
    referral_invite_code = db.relationship("ReferralInviteCode", backref="treasury_reserve_jobs")

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {}, default=str)


class PlatformTreasuryEvent(db.Model):
    """Append-only treasury ledger for gas funding and future reserve accounting."""

    id = db.Column(db.Integer, primary_key=True)
    platform_treasury_job_id = db.Column(db.Integer, db.ForeignKey("platform_treasury_reserve_job.id"), nullable=True, index=True)
    treasury_wallet_id = db.Column(db.Integer, db.ForeignKey("platform_treasury_wallet.id"), nullable=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    wallet_ledger_event_id = db.Column(db.Integer, db.ForeignKey("wallet_ledger_event.id"), nullable=True, index=True)
    wallet_withdrawal_id = db.Column(db.Integer, db.ForeignKey("wallet_withdrawal.id"), nullable=True, index=True)
    vault_cycle_id = db.Column(db.Integer, db.ForeignKey("vault_cycle.id"), nullable=True, index=True)
    referral_invite_code_id = db.Column(db.Integer, db.ForeignKey("referral_invite_code.id"), nullable=True, index=True)
    event_type = db.Column(db.String(64), nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, default="pending", index=True)
    asset = db.Column(db.String(32), nullable=False, default="ETH", index=True)
    network = db.Column(db.String(64), nullable=False, default="Ethereum", index=True)
    amount = db.Column(db.Float, nullable=False, default=0.0)
    fee_amount = db.Column(db.Float, nullable=False, default=0.0)
    provider_reference = db.Column(db.String(180), nullable=True, index=True)
    source_address = db.Column(db.Text, nullable=True)
    destination_address = db.Column(db.Text, nullable=True)
    idempotency_key = db.Column(db.String(220), nullable=False, unique=True, index=True)
    referral_owner_user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    referral_percent = db.Column(db.Float, nullable=False, default=0.0)
    gas_reserve_contribution = db.Column(db.Float, nullable=False, default=0.0)
    vault_cycle_fee_reserve = db.Column(db.Float, nullable=False, default=0.0)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    platform_treasury_job = db.relationship("PlatformTreasuryReserveJob", backref="events")
    treasury_wallet = db.relationship("PlatformTreasuryWallet", backref="events")
    user = db.relationship("User", foreign_keys=[user_id], backref="platform_treasury_events")
    referral_owner = db.relationship("User", foreign_keys=[referral_owner_user_id], backref="platform_referral_treasury_events")
    referral_invite_code = db.relationship("ReferralInviteCode", foreign_keys=[referral_invite_code_id], backref="platform_treasury_events")
    wallet_ledger_event = db.relationship("WalletLedgerEvent", backref="platform_treasury_events")
    wallet_withdrawal = db.relationship("WalletWithdrawal", backref="treasury_events")
    vault_cycle = db.relationship("VaultCycle", backref="platform_treasury_events")

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {}, default=str)


class TreasuryReserveState(db.Model):
    """Latest solvency snapshot for one treasury network."""

    id = db.Column(db.Integer, primary_key=True)
    network = db.Column(db.String(64), nullable=False, index=True)
    treasury_wallet_id = db.Column(db.Integer, db.ForeignKey("platform_treasury_wallet.id"), nullable=True, index=True)
    total_eth_balance = db.Column(db.Float, nullable=False, default=0.0)
    total_estimated_liability = db.Column(db.Float, nullable=False, default=0.0)
    raw_estimated_liability = db.Column(db.Float, nullable=False, default=0.0)
    reserve_ratio = db.Column(db.Float, nullable=False, default=0.0)
    projected_runway = db.Column(db.Float, nullable=True)
    health_status = db.Column(db.String(32), nullable=False, default="emergency", index=True)
    safety_multiplier = db.Column(db.Float, nullable=False, default=1.0)
    gas_price_wei = db.Column(db.Float, nullable=False, default=0.0)
    gas_price_source = db.Column(db.String(64), nullable=False, default="")
    active_balance_count = db.Column(db.Integer, nullable=False, default=0)
    pending_withdrawal_count = db.Column(db.Integer, nullable=False, default=0)
    queued_withdrawal_count = db.Column(db.Integer, nullable=False, default=0)
    active_settlement_count = db.Column(db.Integer, nullable=False, default=0)
    target_reserve_eth = db.Column(db.Float, nullable=False, default=0.0)
    deficit_eth = db.Column(db.Float, nullable=False, default=0.0)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    last_recalculated_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    treasury_wallet = db.relationship("PlatformTreasuryWallet", backref="reserve_states")

    __table_args__ = (
        db.UniqueConstraint("network", name="uq_treasury_reserve_state_network"),
        db.Index("ix_treasury_reserve_state_network_health", "network", "health_status"),
    )

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {}, default=str)


class TreasuryGasUsage(db.Model):
    """Observed or submitted gas usage tied to a withdrawal."""

    id = db.Column(db.Integer, primary_key=True)
    withdrawal_id = db.Column(db.Integer, db.ForeignKey("wallet_withdrawal.id"), nullable=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    asset = db.Column(db.String(32), nullable=False, default="ETH", index=True)
    network = db.Column(db.String(64), nullable=False, default="Ethereum", index=True)
    gas_used = db.Column(db.Float, nullable=False, default=0.0)
    gas_price = db.Column(db.Float, nullable=False, default=0.0)
    gas_fee_eth = db.Column(db.Float, nullable=False, default=0.0)
    tx_hash = db.Column(db.String(180), nullable=True, index=True)
    status = db.Column(db.String(32), nullable=False, default="submitted", index=True)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    withdrawal = db.relationship("WalletWithdrawal", backref="treasury_gas_usage")
    user = db.relationship("User", backref="treasury_gas_usage")

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {}, default=str)


class TreasuryReserveForecast(db.Model):
    """Projected reserve runway and liability for one forecast window."""

    id = db.Column(db.Integer, primary_key=True)
    network = db.Column(db.String(64), nullable=False, default="Ethereum", index=True)
    projected_liability = db.Column(db.Float, nullable=False, default=0.0)
    projected_reserve = db.Column(db.Float, nullable=False, default=0.0)
    forecast_window = db.Column(db.String(32), nullable=False, index=True)
    risk_level = db.Column(db.String(32), nullable=False, default="low", index=True)
    reserve_runway_hours = db.Column(db.Float, nullable=True)
    withdrawal_velocity_eth_per_hour = db.Column(db.Float, nullable=False, default=0.0)
    gas_volatility_score = db.Column(db.Float, nullable=False, default=0.0)
    risk_probability = db.Column(db.Float, nullable=False, default=0.0)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {}, default=str)


class TreasuryAlert(db.Model):
    """Operator-facing treasury solvency alert."""

    id = db.Column(db.Integer, primary_key=True)
    network = db.Column(db.String(64), nullable=False, default="Ethereum", index=True)
    severity = db.Column(db.String(32), nullable=False, default="info", index=True)
    event_type = db.Column(db.String(80), nullable=False, index=True)
    reserve_ratio = db.Column(db.Float, nullable=False, default=0.0)
    health_status = db.Column(db.String(32), nullable=False, default="", index=True)
    message = db.Column(db.Text, nullable=False, default="")
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    acknowledged_at = db.Column(db.DateTime, nullable=True)

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {}, default=str)


class WalletAuditLog(db.Model):
    """Append-only audit trail for self-custody wallet workflows."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    wallet_account_id = db.Column(db.Integer, db.ForeignKey("wallet_account.id"), nullable=True, index=True)
    wallet_withdrawal_id = db.Column(db.Integer, db.ForeignKey("wallet_withdrawal.id"), nullable=True, index=True)
    category = db.Column(db.String(64), nullable=False, default="wallet", index=True)
    action = db.Column(db.String(120), nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, default="recorded", index=True)
    message = db.Column(db.Text, nullable=False)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    user = db.relationship("User", backref="wallet_audit_logs")
    wallet_account = db.relationship("WalletAccount", backref="audit_logs")
    wallet_withdrawal = db.relationship("WalletWithdrawal", backref="audit_logs")

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {})


class Order(db.Model):
    """Persisted local order record for exchange-backed orders."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    trading_connection_id = db.Column(db.Integer, db.ForeignKey("trading_connection.id"), nullable=True, index=True)
    vault_cycle_id = db.Column(db.Integer, db.ForeignKey("vault_cycle.id"), nullable=True, index=True)
    vault_leg_id = db.Column(db.Integer, db.ForeignKey("vault_allocation_leg.id"), nullable=True, index=True)
    client_order_id = db.Column(db.String(120), unique=True, nullable=False, index=True)
    exchange_order_id = db.Column(db.String(120), nullable=True, index=True)
    mode = db.Column(db.String(16), nullable=False, default="live", index=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    side = db.Column(db.String(8), nullable=False)
    order_type = db.Column(db.String(32), nullable=False, default="market")
    status = db.Column(db.String(32), nullable=False, default="pending", index=True)
    strategy_name = db.Column(db.String(120), nullable=True)
    quantity = db.Column(db.Float, nullable=False)
    filled_quantity = db.Column(db.Float, nullable=False, default=0.0)
    limit_price = db.Column(db.Float, nullable=True)
    average_fill_price = db.Column(db.Float, nullable=True)
    reduce_only = db.Column(db.Boolean, nullable=False, default=False)
    leverage = db.Column(db.Float, nullable=False, default=1.0)
    stop_loss = db.Column(db.Float, nullable=True)
    take_profit = db.Column(db.Float, nullable=True)
    risk_status = db.Column(db.String(32), nullable=False, default="pending")
    rejection_reason = db.Column(db.Text, nullable=True)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    fills = db.relationship("Fill", backref="order", lazy=True, cascade="all, delete-orphan", foreign_keys="Fill.order_id")
    user = db.relationship("User", backref="orders")
    trading_connection = db.relationship("TradingConnection", backref="orders")
    vault_cycle = db.relationship("VaultCycle", backref="orders")
    vault_leg = db.relationship("VaultAllocationLeg", backref="orders")
    __table_args__ = (
        db.Index("ix_order_user_mode_created", "user_id", "mode", "created_at"),
        db.Index("ix_order_user_mode_status_created", "user_id", "mode", "status", "created_at"),
        db.Index("ix_order_cycle_mode_created", "vault_cycle_id", "mode", "created_at"),
        db.Index("ix_order_leg_status_created", "vault_leg_id", "status", "created_at"),
        db.Index("ix_order_connection_status_created", "trading_connection_id", "status", "created_at"),
    )

    @property
    def details(self) -> dict[str, Any]:
        try:
            return json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {})


class Fill(db.Model):
    """Trade fills persisted for analytics and risk checks."""

    id = db.Column(db.Integer, primary_key=True)
    order_id = db.Column(db.Integer, db.ForeignKey("order.id"), nullable=False, index=True)
    source_order_id = db.Column(db.Integer, db.ForeignKey("order.id"), nullable=True, index=True)
    exchange_order_id = db.Column(db.String(120), nullable=True, index=True)
    exchange_fill_id = db.Column(db.String(180), nullable=True, index=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    side = db.Column(db.String(8), nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    price = db.Column(db.Float, nullable=False)
    fee = db.Column(db.Float, nullable=False, default=0.0)
    pnl = db.Column(db.Float, nullable=False, default=0.0)
    funding_fee = db.Column(db.Float, nullable=False, default=0.0)
    fee_known = db.Column(db.Boolean, nullable=False, default=True)
    realized_pnl_known = db.Column(db.Boolean, nullable=False, default=True)
    simulated = db.Column(db.Boolean, nullable=False, default=True)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    fill_time = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    source_order = db.relationship("Order", foreign_keys=[source_order_id])

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {}, default=str)


class PositionSnapshot(db.Model):
    """Periodic position snapshots for analytics and dashboard views."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    trading_connection_id = db.Column(db.Integer, db.ForeignKey("trading_connection.id"), nullable=True, index=True)
    mode = db.Column(db.String(16), nullable=False, index=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    quantity = db.Column(db.Float, nullable=False, default=0.0)
    average_entry_price = db.Column(db.Float, nullable=False, default=0.0)
    mark_price = db.Column(db.Float, nullable=False, default=0.0)
    unrealized_pnl = db.Column(db.Float, nullable=False, default=0.0)
    leverage = db.Column(db.Float, nullable=False, default=1.0)
    notional = db.Column(db.Float, nullable=False, default=0.0)
    snapshot_time = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    user = db.relationship("User", backref="position_snapshots")
    trading_connection = db.relationship("TradingConnection", backref="position_snapshots")


class BacktestRun(db.Model):
    """Stored backtest configuration and result payloads."""

    id = db.Column(db.Integer, primary_key=True)
    strategy_name = db.Column(db.String(120), nullable=False, index=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    timeframe = db.Column(db.String(16), nullable=False)
    parameters_json = db.Column(db.Text, nullable=False, default="{}")
    result_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    @property
    def parameters(self) -> dict[str, Any]:
        try:
            return json.loads(self.parameters_json or "{}")
        except json.JSONDecodeError:
            return {}

    @parameters.setter
    def parameters(self, value: dict[str, Any]) -> None:
        self.parameters_json = json.dumps(value or {})

    @property
    def result(self) -> dict[str, Any]:
        try:
            return json.loads(self.result_json or "{}")
        except json.JSONDecodeError:
            return {}

    @result.setter
    def result(self, value: dict[str, Any]) -> None:
        self.result_json = json.dumps(value or {})


class MLMarketHistory(db.Model):
    """Bounded market-data windows collected for ML training and diagnostics."""

    id = db.Column(db.Integer, primary_key=True)
    provider = db.Column(db.String(64), nullable=False, index=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    timeframe = db.Column(db.String(16), nullable=False, index=True)
    mode = db.Column(db.String(16), nullable=False, default="live", index=True)
    source = db.Column(db.String(64), nullable=False, default="ml_history_backfill", index=True)
    status = db.Column(db.String(32), nullable=False, default="ok", index=True)
    error = db.Column(db.Text, nullable=False, default="")
    candle_count = db.Column(db.Integer, nullable=False, default=0)
    liquidity_usd = db.Column(db.Float, nullable=False, default=0.0)
    spread_bps = db.Column(db.Float, nullable=False, default=0.0)
    funding_rate = db.Column(db.Float, nullable=False, default=0.0)
    candles_json = db.Column(db.Text, nullable=False, default="[]")
    order_book_json = db.Column(db.Text, nullable=False, default="{}")
    funding_json = db.Column(db.Text, nullable=False, default="{}")
    diagnostics_json = db.Column(db.Text, nullable=False, default="{}")
    window_start = db.Column(db.DateTime, nullable=True, index=True)
    window_end = db.Column(db.DateTime, nullable=True, index=True)
    fetched_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    __table_args__ = (
        db.Index("ix_ml_market_history_provider_symbol_tf_window", "provider", "symbol", "timeframe", "window_end"),
        db.Index("ix_ml_market_history_provider_status_fetched", "provider", "status", "fetched_at"),
    )

    @property
    def candles(self) -> list[dict[str, Any]]:
        try:
            value = json.loads(self.candles_json or "[]")
        except json.JSONDecodeError:
            return []
        return value if isinstance(value, list) else []

    @candles.setter
    def candles(self, value: list[dict[str, Any]]) -> None:
        self.candles_json = json.dumps(value or [])
        self.candle_count = len(value or [])

    @property
    def order_book(self) -> dict[str, Any]:
        try:
            value = json.loads(self.order_book_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @order_book.setter
    def order_book(self, value: dict[str, Any]) -> None:
        self.order_book_json = json.dumps(value or {})

    @property
    def funding(self) -> dict[str, Any]:
        try:
            value = json.loads(self.funding_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @funding.setter
    def funding(self, value: dict[str, Any]) -> None:
        self.funding_json = json.dumps(value or {})

    @property
    def diagnostics(self) -> dict[str, Any]:
        try:
            value = json.loads(self.diagnostics_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @diagnostics.setter
    def diagnostics(self, value: dict[str, Any]) -> None:
        self.diagnostics_json = json.dumps(value or {})


class OptimizerRun(db.Model):
    """Persisted batch optimization run and aggregate result payload."""

    id = db.Column(db.Integer, primary_key=True)
    profile = db.Column(db.String(32), nullable=False, default="short_term", index=True)
    status = db.Column(db.String(32), nullable=False, default="pending", index=True)
    symbols_json = db.Column(db.Text, nullable=False, default="[]")
    timeframes_json = db.Column(db.Text, nullable=False, default="[]")
    config_json = db.Column(db.Text, nullable=False, default="{}")
    result_json = db.Column(db.Text, nullable=False, default="{}")
    warnings_json = db.Column(db.Text, nullable=False, default="[]")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    completed_at = db.Column(db.DateTime, nullable=True)

    rankings = db.relationship("StrategyRanking", backref="optimizer_run", lazy=True, cascade="all, delete-orphan")
    __table_args__ = (
        db.Index("ix_optimizer_run_profile_status_created", "profile", "status", "created_at"),
    )

    @property
    def symbols(self) -> list[str]:
        return self._json_list(self.symbols_json)

    @symbols.setter
    def symbols(self, value: list[str]) -> None:
        self.symbols_json = json.dumps(value or [])

    @property
    def timeframes(self) -> list[str]:
        return self._json_list(self.timeframes_json)

    @timeframes.setter
    def timeframes(self, value: list[str]) -> None:
        self.timeframes_json = json.dumps(value or [])

    @property
    def config_payload(self) -> dict[str, Any]:
        return self._json_dict(self.config_json)

    @config_payload.setter
    def config_payload(self, value: dict[str, Any]) -> None:
        self.config_json = json.dumps(value or {})

    @property
    def result(self) -> dict[str, Any]:
        return self._json_dict(self.result_json)

    @result.setter
    def result(self, value: dict[str, Any]) -> None:
        self.result_json = json.dumps(value or {})

    @property
    def warnings(self) -> list[str]:
        return self._json_list(self.warnings_json)

    @warnings.setter
    def warnings(self, value: list[str]) -> None:
        self.warnings_json = json.dumps(value or [])

    @staticmethod
    def _json_dict(payload: str) -> dict[str, Any]:
        try:
            value = json.loads(payload or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _json_list(payload: str) -> list[Any]:
        try:
            value = json.loads(payload or "[]")
        except json.JSONDecodeError:
            return []
        return value if isinstance(value, list) else []


class StrategyRanking(db.Model):
    """Risk-adjusted strategy ranking emitted by the optimizer."""

    id = db.Column(db.Integer, primary_key=True)
    optimizer_run_id = db.Column(db.Integer, db.ForeignKey("optimizer_run.id"), nullable=False, index=True)
    provider = db.Column(db.String(64), nullable=False, default="global", index=True)
    strategy_name = db.Column(db.String(120), nullable=False, index=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    timeframe = db.Column(db.String(16), nullable=False)
    profile = db.Column(db.String(64), nullable=False, default="short_term", index=True)
    experimental = db.Column(db.Boolean, nullable=False, default=False, index=True)
    risk_label = db.Column(db.String(80), nullable=False, default="")
    parameters_json = db.Column(db.Text, nullable=False, default="{}")
    score = db.Column(db.Float, nullable=False, default=0.0, index=True)
    total_return = db.Column(db.Float, nullable=False, default=0.0)
    net_return_after_costs = db.Column(db.Float, nullable=False, default=0.0)
    recent_performance_score = db.Column(db.Float, nullable=False, default=0.0)
    recent_1h_return = db.Column(db.Float, nullable=False, default=0.0)
    estimated_fees = db.Column(db.Float, nullable=False, default=0.0)
    edge_score = db.Column(db.Float, nullable=False, default=0.0)
    expectancy = db.Column(db.Float, nullable=False, default=0.0)
    avg_win = db.Column(db.Float, nullable=False, default=0.0)
    avg_loss = db.Column(db.Float, nullable=False, default=0.0)
    win_loss_ratio = db.Column(db.Float, nullable=False, default=0.0)
    cost_drag_bps = db.Column(db.Float, nullable=False, default=0.0)
    convex_edge_score = db.Column(db.Float, nullable=False, default=0.0)
    mfe_mae_ratio = db.Column(db.Float, nullable=False, default=0.0)
    capacity_multiple = db.Column(db.Float, nullable=False, default=0.0)
    cost_adjusted_recent_1h_return = db.Column(db.Float, nullable=False, default=0.0)
    decay_penalty = db.Column(db.Float, nullable=False, default=0.0)
    max_adverse_excursion = db.Column(db.Float, nullable=False, default=0.0)
    max_favorable_excursion = db.Column(db.Float, nullable=False, default=0.0)
    no_trade_reason = db.Column(db.Text, nullable=True)
    allocation_amount_usd = db.Column(db.Float, nullable=False, default=0.0)
    lock_duration_hours = db.Column(db.Integer, nullable=False, default=0)
    leverage = db.Column(db.Float, nullable=False, default=1.0)
    liquidation_buffer_pct = db.Column(db.Float, nullable=False, default=1.0)
    capacity_usd = db.Column(db.Float, nullable=False, default=0.0)
    universe_source = db.Column(db.String(64), nullable=False, default="configured")
    vault_leg_count = db.Column(db.Integer, nullable=False, default=1)
    execution_style = db.Column(db.String(32), nullable=False, default="market")
    funding_cost_estimate = db.Column(db.Float, nullable=False, default=0.0)
    max_drawdown = db.Column(db.Float, nullable=False, default=0.0)
    profit_factor = db.Column(db.Float, nullable=False, default=0.0)
    sharpe_like = db.Column(db.Float, nullable=False, default=0.0)
    sortino_like = db.Column(db.Float, nullable=False, default=0.0)
    trades_per_day = db.Column(db.Float, nullable=False, default=0.0)
    avg_trade_return = db.Column(db.Float, nullable=False, default=0.0)
    turnover_rate = db.Column(db.Float, nullable=False, default=0.0)
    turnover_after_fees = db.Column(db.Float, nullable=False, default=0.0)
    consistency = db.Column(db.Float, nullable=False, default=0.0)
    window_stability = db.Column(db.Float, nullable=False, default=0.0)
    accepted_window_ratio = db.Column(db.Float, nullable=False, default=0.0)
    win_rate = db.Column(db.Float, nullable=False, default=0.0)
    trade_count = db.Column(db.Integer, nullable=False, default=0)
    ml_score = db.Column(db.Float, nullable=False, default=0.0)
    ml_adjusted_score = db.Column(db.Float, nullable=False, default=0.0)
    ml_warmup = db.Column(db.Boolean, nullable=False, default=True, index=True)
    ml_explanation_json = db.Column(db.Text, nullable=False, default="{}")
    rejected = db.Column(db.Boolean, nullable=False, default=False, index=True)
    rejection_reason = db.Column(db.Text, nullable=True)
    warnings_json = db.Column(db.Text, nullable=False, default="[]")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    __table_args__ = (
        db.Index("ix_strategy_ranking_provider_profile_rejected_score_created", "provider", "profile", "rejected", "score", "created_at"),
        db.Index("ix_strategy_ranking_symbol_profile_rejected_score", "symbol", "profile", "rejected", "score"),
        db.Index("ix_strategy_ranking_run_rejected_score", "optimizer_run_id", "rejected", "score"),
    )

    @property
    def parameters(self) -> dict[str, Any]:
        try:
            return json.loads(self.parameters_json or "{}")
        except json.JSONDecodeError:
            return {}

    @parameters.setter
    def parameters(self, value: dict[str, Any]) -> None:
        self.parameters_json = json.dumps(value or {})

    @property
    def warnings(self) -> list[str]:
        try:
            value = json.loads(self.warnings_json or "[]")
        except json.JSONDecodeError:
            return []
        return value if isinstance(value, list) else []

    @warnings.setter
    def warnings(self, value: list[str]) -> None:
        self.warnings_json = json.dumps(value or [])

    @property
    def ml_explanation(self) -> dict[str, Any]:
        try:
            value = json.loads(self.ml_explanation_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @ml_explanation.setter
    def ml_explanation(self, value: dict[str, Any]) -> None:
        self.ml_explanation_json = json.dumps(value or {})


class MLModelState(db.Model):
    """Persisted online ranker state for one prediction horizon."""

    id = db.Column(db.Integer, primary_key=True)
    model_key = db.Column(db.String(120), unique=True, nullable=False, index=True)
    provider = db.Column(db.String(64), nullable=False, default="global", index=True)
    horizon = db.Column(db.String(32), nullable=False, default="global", index=True)
    weights_json = db.Column(db.Text, nullable=False, default="{}")
    bias = db.Column(db.Float, nullable=False, default=0.0)
    learning_rate = db.Column(db.Float, nullable=False, default=0.03)
    l2 = db.Column(db.Float, nullable=False, default=0.001)
    prediction_cap = db.Column(db.Float, nullable=False, default=1.0)
    weight_cap = db.Column(db.Float, nullable=False, default=3.0)
    update_count = db.Column(db.Integer, nullable=False, default=0)
    total_loss = db.Column(db.Float, nullable=False, default=0.0)
    last_loss = db.Column(db.Float, nullable=False, default=0.0)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    training_events = db.relationship("MLTrainingEvent", backref="model_state", lazy=True)

    @property
    def weights(self) -> dict[str, float]:
        try:
            value = json.loads(self.weights_json or "{}")
        except json.JSONDecodeError:
            return {}
        if not isinstance(value, dict):
            return {}
        parsed: dict[str, float] = {}
        for key, raw in value.items():
            try:
                parsed[str(key)] = float(raw)
            except (TypeError, ValueError):
                continue
        return parsed

    @weights.setter
    def weights(self, value: dict[str, float]) -> None:
        self.weights_json = json.dumps({str(key): float(raw) for key, raw in (value or {}).items()})


class MLOfflineModel(db.Model):
    """Stored metadata for explicitly trained offline ranker artifacts."""

    id = db.Column(db.Integer, primary_key=True)
    model_key = db.Column(db.String(160), unique=True, nullable=False, index=True)
    provider = db.Column(db.String(64), nullable=False, default="global", index=True)
    horizon = db.Column(db.String(32), nullable=False, default="global", index=True)
    model_type = db.Column(db.String(32), nullable=False, default="sklearn", index=True)
    status = db.Column(db.String(32), nullable=False, default="candidate", index=True)
    artifact_path = db.Column(db.Text, nullable=False, default="")
    feature_schema_version = db.Column(db.String(64), nullable=False, default="offline_ranker_v2")
    feature_names_json = db.Column(db.Text, nullable=False, default="[]")
    metrics_json = db.Column(db.Text, nullable=False, default="{}")
    training_rows = db.Column(db.Integer, nullable=False, default=0)
    validation_rows = db.Column(db.Integer, nullable=False, default=0)
    validation_loss = db.Column(db.Float, nullable=False, default=0.0)
    negative_error_rate = db.Column(db.Float, nullable=False, default=0.0)
    drift = db.Column(db.Float, nullable=False, default=0.0)
    feature_schema_hash = db.Column(db.String(128), nullable=False, default="")
    dataset_version = db.Column(db.String(128), nullable=False, default="")
    dataset_hash = db.Column(db.String(128), nullable=False, default="")
    training_started_at = db.Column(db.DateTime, nullable=True)
    training_completed_at = db.Column(db.DateTime, nullable=True)
    promoted_by = db.Column(db.String(120), nullable=False, default="")
    promotion_source = db.Column(db.String(120), nullable=False, default="")
    rollback_target_model_id = db.Column(db.Integer, db.ForeignKey("ml_offline_model.id"), nullable=True, index=True)
    live_mode = db.Column(db.String(32), nullable=False, default="shadow", index=True)
    drift_status = db.Column(db.String(32), nullable=False, default="unknown", index=True)
    governance_metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    promoted_at = db.Column(db.DateTime, nullable=True, index=True)

    rollback_target = db.relationship("MLOfflineModel", remote_side=[id], uselist=False)

    @property
    def feature_names(self) -> list[str]:
        try:
            value = json.loads(self.feature_names_json or "[]")
        except json.JSONDecodeError:
            return []
        return [str(item) for item in value] if isinstance(value, list) else []

    @feature_names.setter
    def feature_names(self, value: list[str]) -> None:
        self.feature_names_json = json.dumps([str(item) for item in (value or [])])

    @property
    def metrics(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metrics_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @metrics.setter
    def metrics(self, value: dict[str, Any]) -> None:
        self.metrics_json = json.dumps(value or {})

    @property
    def governance_metadata(self) -> dict[str, Any]:
        try:
            value = json.loads(self.governance_metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @governance_metadata.setter
    def governance_metadata(self, value: dict[str, Any]) -> None:
        self.governance_metadata_json = json.dumps(value or {})


class MLModelRegistry(db.Model):
    """Auditable registry row for promoted, shadow, and rollback-capable ML models."""

    id = db.Column(db.Integer, primary_key=True)
    model_key = db.Column(db.String(180), nullable=False, unique=True, index=True)
    offline_model_id = db.Column(db.Integer, db.ForeignKey("ml_offline_model.id"), nullable=True, index=True)
    model_family = db.Column(db.String(80), nullable=False, index=True)
    model_version = db.Column(db.String(80), nullable=False, default="", index=True)
    provider = db.Column(db.String(64), nullable=False, default="global", index=True)
    horizon = db.Column(db.String(32), nullable=False, default="global", index=True)
    feature_schema_hash = db.Column(db.String(128), nullable=False, default="")
    dataset_version = db.Column(db.String(128), nullable=False, default="")
    dataset_hash = db.Column(db.String(128), nullable=False, default="")
    trained_at = db.Column(db.DateTime, nullable=True, index=True)
    promoted_at = db.Column(db.DateTime, nullable=True, index=True)
    promoted_by = db.Column(db.String(120), nullable=False, default="")
    promotion_source = db.Column(db.String(120), nullable=False, default="")
    rollback_target_model_id = db.Column(db.Integer, db.ForeignKey("ml_offline_model.id"), nullable=True, index=True)
    mode = db.Column(db.String(32), nullable=False, default="shadow", index=True)
    drift_status = db.Column(db.String(32), nullable=False, default="unknown", index=True)
    status = db.Column(db.String(32), nullable=False, default="registered", index=True)
    metrics_json = db.Column(db.Text, nullable=False, default="{}")
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    offline_model = db.relationship("MLOfflineModel", foreign_keys=[offline_model_id])
    rollback_target = db.relationship("MLOfflineModel", foreign_keys=[rollback_target_model_id])

    @property
    def metrics(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metrics_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @metrics.setter
    def metrics(self, value: dict[str, Any]) -> None:
        self.metrics_json = json.dumps(value or {})

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {})


class MLTrainingEvent(db.Model):
    """Audit trail for online ranker updates."""

    id = db.Column(db.Integer, primary_key=True)
    model_state_id = db.Column(db.Integer, db.ForeignKey("ml_model_state.id"), nullable=False, index=True)
    provider = db.Column(db.String(64), nullable=False, default="global", index=True)
    source = db.Column(db.String(64), nullable=False, default="unknown", index=True)
    source_id = db.Column(db.String(120), nullable=True, index=True)
    mode = db.Column(db.String(32), nullable=False, default="live", index=True)
    horizon = db.Column(db.String(32), nullable=False, default="global", index=True)
    features_json = db.Column(db.Text, nullable=False, default="{}")
    outcome = db.Column(db.Float, nullable=False, default=0.0)
    prediction_before = db.Column(db.Float, nullable=False, default=0.0)
    prediction_after = db.Column(db.Float, nullable=False, default=0.0)
    error = db.Column(db.Float, nullable=False, default=0.0)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    @property
    def features(self) -> dict[str, Any]:
        try:
            value = json.loads(self.features_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @features.setter
    def features(self, value: dict[str, Any]) -> None:
        self.features_json = json.dumps(value or {})

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {})


class RapidMLSession(db.Model):
    """Operator-scoped rapid ML trading session and account-level caps."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    provider_scope = db.Column(db.String(32), nullable=False, default="both", index=True)
    capital_usd = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(32), nullable=False, default="running", index=True)
    submit_requested = db.Column(db.Boolean, nullable=False, default=False)
    real_submit_enabled = db.Column(db.Boolean, nullable=False, default=False)
    allocated_capital_json = db.Column(db.Text, nullable=False, default="{}")
    blockers_json = db.Column(db.Text, nullable=False, default="[]")
    summary_json = db.Column(db.Text, nullable=False, default="{}")
    started_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    completed_at = db.Column(db.DateTime, nullable=True, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    user = db.relationship("User", backref="rapid_ml_sessions")

    @property
    def allocated_capital(self) -> dict[str, Any]:
        try:
            value = json.loads(self.allocated_capital_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @allocated_capital.setter
    def allocated_capital(self, value: dict[str, Any]) -> None:
        self.allocated_capital_json = json.dumps(value or {})

    @property
    def blockers(self) -> list[str]:
        try:
            value = json.loads(self.blockers_json or "[]")
        except json.JSONDecodeError:
            return []
        return [str(item) for item in value] if isinstance(value, list) else []

    @blockers.setter
    def blockers(self, value: list[str]) -> None:
        self.blockers_json = json.dumps([str(item) for item in (value or [])])

    @property
    def summary(self) -> dict[str, Any]:
        try:
            value = json.loads(self.summary_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @summary.setter
    def summary(self, value: dict[str, Any]) -> None:
        self.summary_json = json.dumps(value or {})


class RapidMLDecision(db.Model):
    """Per-provider ML decision, risk decision, and reconciliation record."""

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey("rapid_ml_session.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    trading_connection_id = db.Column(db.Integer, db.ForeignKey("trading_connection.id"), nullable=True, index=True)
    order_id = db.Column(db.Integer, db.ForeignKey("order.id"), nullable=True, index=True)
    provider = db.Column(db.String(64), nullable=False, index=True)
    symbol = db.Column(db.String(32), nullable=False, default="", index=True)
    side = db.Column(db.String(8), nullable=False, default="")
    action = db.Column(db.String(32), nullable=False, default="hold", index=True)
    status = db.Column(db.String(32), nullable=False, default="preview", index=True)
    confidence = db.Column(db.Float, nullable=False, default=0.0)
    expected_return = db.Column(db.Float, nullable=False, default=0.0)
    opportunity_score = db.Column(db.Float, nullable=False, default=0.0)
    allocation_usd = db.Column(db.Float, nullable=False, default=0.0)
    notional_usd = db.Column(db.Float, nullable=False, default=0.0)
    order_intent_json = db.Column(db.Text, nullable=False, default="{}")
    provider_state_json = db.Column(db.Text, nullable=False, default="{}")
    ml_decisions_json = db.Column(db.Text, nullable=False, default="{}")
    risk_json = db.Column(db.Text, nullable=False, default="{}")
    blockers_json = db.Column(db.Text, nullable=False, default="[]")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    session = db.relationship("RapidMLSession", backref="decisions")
    user = db.relationship("User", backref="rapid_ml_decisions")
    trading_connection = db.relationship("TradingConnection", backref="rapid_ml_decisions")
    order = db.relationship("Order", backref="rapid_ml_decisions")

    @property
    def order_intent(self) -> dict[str, Any]:
        return self._json_dict(self.order_intent_json)

    @order_intent.setter
    def order_intent(self, value: dict[str, Any]) -> None:
        self.order_intent_json = json.dumps(value or {})

    @property
    def provider_state(self) -> dict[str, Any]:
        return self._json_dict(self.provider_state_json)

    @provider_state.setter
    def provider_state(self, value: dict[str, Any]) -> None:
        self.provider_state_json = json.dumps(value or {})

    @property
    def ml_decisions(self) -> dict[str, Any]:
        return self._json_dict(self.ml_decisions_json)

    @ml_decisions.setter
    def ml_decisions(self, value: dict[str, Any]) -> None:
        self.ml_decisions_json = json.dumps(value or {})

    @property
    def risk(self) -> dict[str, Any]:
        return self._json_dict(self.risk_json)

    @risk.setter
    def risk(self, value: dict[str, Any]) -> None:
        self.risk_json = json.dumps(value or {})

    @property
    def blockers(self) -> list[str]:
        try:
            value = json.loads(self.blockers_json or "[]")
        except json.JSONDecodeError:
            return []
        return [str(item) for item in value] if isinstance(value, list) else []

    @blockers.setter
    def blockers(self, value: list[str]) -> None:
        self.blockers_json = json.dumps([str(item) for item in (value or [])])

    @staticmethod
    def _json_dict(raw: str) -> dict[str, Any]:
        try:
            value = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


class StrategyValidation(db.Model):
    """Validation checkpoints required before live strategy execution."""

    id = db.Column(db.Integer, primary_key=True)
    strategy_name = db.Column(db.String(120), nullable=False, index=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    timeframe = db.Column(db.String(16), nullable=False)
    stage = db.Column(db.String(32), nullable=False, index=True)
    status = db.Column(db.String(32), nullable=False, default="pending", index=True)
    parameters_json = db.Column(db.Text, nullable=False, default="{}")
    metrics_json = db.Column(db.Text, nullable=False, default="{}")
    started_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

    @property
    def parameters(self) -> dict[str, Any]:
        try:
            return json.loads(self.parameters_json or "{}")
        except json.JSONDecodeError:
            return {}

    @parameters.setter
    def parameters(self, value: dict[str, Any]) -> None:
        self.parameters_json = json.dumps(value or {})

    @property
    def metrics(self) -> dict[str, Any]:
        try:
            return json.loads(self.metrics_json or "{}")
        except json.JSONDecodeError:
            return {}

    @metrics.setter
    def metrics(self, value: dict[str, Any]) -> None:
        self.metrics_json = json.dumps(value or {})


class ShadowLiveObservation(db.Model):
    """Signal-only live-market observation used for probation before live trading."""

    id = db.Column(db.Integer, primary_key=True)
    validation_id = db.Column(db.Integer, db.ForeignKey("strategy_validation.id"), nullable=True, index=True)
    strategy_name = db.Column(db.String(120), nullable=False, index=True)
    symbol = db.Column(db.String(32), nullable=False, index=True)
    timeframe = db.Column(db.String(16), nullable=False)
    signal_action = db.Column(db.String(32), nullable=False, default="hold")
    expected_price = db.Column(db.Float, nullable=False, default=0.0)
    live_mid = db.Column(db.Float, nullable=False, default=0.0)
    expected_slippage_bps = db.Column(db.Float, nullable=False, default=0.0)
    observed_spread_bps = db.Column(db.Float, nullable=False, default=0.0)
    latency_ms = db.Column(db.Float, nullable=False, default=0.0)
    missed_fill = db.Column(db.Boolean, nullable=False, default=False)
    drawdown = db.Column(db.Float, nullable=False, default=0.0)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    validation = db.relationship("StrategyValidation", backref="shadow_observations")

    @property
    def details(self) -> dict[str, Any]:
        try:
            return json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {})


class AuditLog(db.Model):
    """Retained audit log for high-risk actions and safety controls."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    trading_connection_id = db.Column(db.Integer, db.ForeignKey("trading_connection.id"), nullable=True, index=True)
    category = db.Column(db.String(64), nullable=False, index=True)
    action = db.Column(db.String(120), nullable=False)
    message = db.Column(db.Text, nullable=False)
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    user = db.relationship("User", backref="audit_logs")
    trading_connection = db.relationship("TradingConnection", backref="audit_logs")
    __table_args__ = (
        db.Index("ix_audit_log_created_id", "created_at", "id"),
        db.Index("ix_audit_log_category_action_created", "category", "action", "created_at"),
        db.Index("ix_audit_log_user_category_created", "user_id", "category", "created_at"),
        db.Index("ix_audit_log_connection_category_created", "trading_connection_id", "category", "created_at"),
    )

    @property
    def details(self) -> dict[str, Any]:
        try:
            return json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {})


class AdminAuditLog(db.Model):
    """Append-only admin action log for financial admin workflows."""

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(40), unique=True, nullable=False, default=lambda: public_token("aal"), index=True)
    admin_user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    action = db.Column(db.String(120), nullable=False, index=True)
    entity_type = db.Column(db.String(80), nullable=False, index=True)
    entity_public_id = db.Column(db.String(80), nullable=False, default="", index=True)
    old_value_json = db.Column(db.Text, nullable=False, default="{}")
    new_value_json = db.Column(db.Text, nullable=False, default="{}")
    ip_address = db.Column(db.String(120), nullable=False, default="")
    user_agent = db.Column(db.Text, nullable=False, default="")
    metadata_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    admin_user = db.relationship("User", backref="admin_audit_logs")
    __table_args__ = (
        db.Index("ix_admin_audit_entity_created", "entity_type", "entity_public_id", "created_at"),
        db.Index("ix_admin_audit_user_action_created", "admin_user_id", "action", "created_at"),
    )

    @property
    def old_value(self) -> dict[str, Any]:
        try:
            value = json.loads(self.old_value_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @old_value.setter
    def old_value(self, value: dict[str, Any]) -> None:
        self.old_value_json = json.dumps(value or {}, default=str)

    @property
    def new_value(self) -> dict[str, Any]:
        try:
            value = json.loads(self.new_value_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @new_value.setter
    def new_value(self, value: dict[str, Any]) -> None:
        self.new_value_json = json.dumps(value or {}, default=str)

    @property
    def details(self) -> dict[str, Any]:
        try:
            value = json.loads(self.metadata_json or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @details.setter
    def details(self, value: dict[str, Any]) -> None:
        self.metadata_json = json.dumps(value or {}, default=str)


class RiskEvent(db.Model):
    """Detailed record of every rejected order and risk-rule trip."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    trading_connection_id = db.Column(db.Integer, db.ForeignKey("trading_connection.id"), nullable=True, index=True)
    order_id = db.Column(db.Integer, db.ForeignKey("order.id"), nullable=True, index=True)
    rule_name = db.Column(db.String(120), nullable=False, index=True)
    reason = db.Column(db.Text, nullable=False)
    payload_json = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    user = db.relationship("User", backref="risk_events")
    trading_connection = db.relationship("TradingConnection", backref="risk_events")

    @property
    def payload(self) -> dict[str, Any]:
        try:
            return json.loads(self.payload_json or "{}")
        except json.JSONDecodeError:
            return {}

    @payload.setter
    def payload(self, value: dict[str, Any]) -> None:
        self.payload_json = json.dumps(value or {})
