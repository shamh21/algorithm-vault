"""invite profit share admin pwa

Revision ID: c7f3a2b9d8e1
Revises: 5bb58e96ca8a
Create Date: 2026-05-13 12:00:00.000000

"""
from __future__ import annotations

import secrets

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "c7f3a2b9d8e1"
down_revision = "5bb58e96ca8a"
branch_labels = None
depends_on = None


def _public_token(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(16).replace('-', '').replace('_', '')[:22]}"


def upgrade():
    with op.batch_alter_table("referral_invite_code", schema=None) as batch_op:
        batch_op.add_column(sa.Column("public_id", sa.String(length=40), nullable=True))
        batch_op.add_column(sa.Column("profit_share_percent", sa.Float(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("profit_share_wallet", sa.String(length=120), nullable=False, server_default="sufyanh"))
        batch_op.add_column(sa.Column("profit_share_starts_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("profit_share_ends_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("profit_share_active", sa.Boolean(), nullable=False, server_default=sa.true()))
        batch_op.add_column(sa.Column("applies_to_vault_types_json", sa.Text(), nullable=False, server_default="[]"))
        batch_op.add_column(sa.Column("expires_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("assigned_role", sa.String(length=32), nullable=False, server_default="user"))
        batch_op.add_column(sa.Column("deleted_at", sa.DateTime(), nullable=True))

    with op.batch_alter_table("vault_cycle", schema=None) as batch_op:
        batch_op.add_column(sa.Column("public_id", sa.String(length=40), nullable=True))

    bind = op.get_bind()
    invite_table = sa.table(
        "referral_invite_code",
        sa.column("id", sa.Integer),
        sa.column("public_id", sa.String),
        sa.column("percent_profit", sa.Float),
        sa.column("profit_share_percent", sa.Float),
        sa.column("profit_share_wallet", sa.String),
    )
    cycle_table = sa.table("vault_cycle", sa.column("id", sa.Integer), sa.column("public_id", sa.String))
    for row in bind.execute(sa.select(invite_table.c.id)).mappings():
        bind.execute(
            invite_table.update()
            .where(invite_table.c.id == row["id"])
            .values(public_id=_public_token("inv"))
        )
    bind.execute(
        invite_table.update()
        .where(invite_table.c.profit_share_percent == 0)
        .values(profit_share_percent=invite_table.c.percent_profit)
    )
    bind.execute(
        invite_table.update()
        .where((invite_table.c.profit_share_wallet.is_(None)) | (invite_table.c.profit_share_wallet == ""))
        .values(profit_share_wallet="sufyanh")
    )
    for row in bind.execute(sa.select(cycle_table.c.id)).mappings():
        bind.execute(cycle_table.update().where(cycle_table.c.id == row["id"]).values(public_id=_public_token("vc")))

    with op.batch_alter_table("referral_invite_code", schema=None) as batch_op:
        batch_op.alter_column("public_id", existing_type=sa.String(length=40), nullable=False)
        batch_op.create_index(batch_op.f("ix_referral_invite_code_public_id"), ["public_id"], unique=True)
        batch_op.create_index(batch_op.f("ix_referral_invite_code_profit_share_wallet"), ["profit_share_wallet"], unique=False)
        batch_op.create_index(batch_op.f("ix_referral_invite_code_profit_share_active"), ["profit_share_active"], unique=False)
        batch_op.create_index(batch_op.f("ix_referral_invite_code_expires_at"), ["expires_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_referral_invite_code_deleted_at"), ["deleted_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_referral_invite_code_assigned_role"), ["assigned_role"], unique=False)

    with op.batch_alter_table("vault_cycle", schema=None) as batch_op:
        batch_op.alter_column("public_id", existing_type=sa.String(length=40), nullable=False)
        batch_op.create_index(batch_op.f("ix_vault_cycle_public_id"), ["public_id"], unique=True)

    op.create_table(
        "admin_audit_log",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("public_id", sa.String(length=40), nullable=False),
        sa.Column("admin_user_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(length=120), nullable=False),
        sa.Column("entity_type", sa.String(length=80), nullable=False),
        sa.Column("entity_public_id", sa.String(length=80), nullable=False),
        sa.Column("old_value_json", sa.Text(), nullable=False),
        sa.Column("new_value_json", sa.Text(), nullable=False),
        sa.Column("ip_address", sa.String(length=120), nullable=False),
        sa.Column("user_agent", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["admin_user_id"], ["user.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("admin_audit_log", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_admin_audit_log_public_id"), ["public_id"], unique=True)
        batch_op.create_index(batch_op.f("ix_admin_audit_log_admin_user_id"), ["admin_user_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_admin_audit_log_action"), ["action"], unique=False)
        batch_op.create_index(batch_op.f("ix_admin_audit_log_entity_type"), ["entity_type"], unique=False)
        batch_op.create_index(batch_op.f("ix_admin_audit_log_entity_public_id"), ["entity_public_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_admin_audit_log_created_at"), ["created_at"], unique=False)
        batch_op.create_index("ix_admin_audit_entity_created", ["entity_type", "entity_public_id", "created_at"], unique=False)
        batch_op.create_index("ix_admin_audit_user_action_created", ["admin_user_id", "action", "created_at"], unique=False)

    op.create_table(
        "invite_code_usage",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("public_id", sa.String(length=40), nullable=False),
        sa.Column("invite_code_id", sa.Integer(), nullable=False),
        sa.Column("invitee_user_id", sa.Integer(), nullable=False),
        sa.Column("used_at", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("accepted_disclosure_version", sa.String(length=64), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["invite_code_id"], ["referral_invite_code.id"]),
        sa.ForeignKeyConstraint(["invitee_user_id"], ["user.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("invitee_user_id", name="uq_invite_code_usage_invitee"),
    )
    with op.batch_alter_table("invite_code_usage", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_invite_code_usage_public_id"), ["public_id"], unique=True)
        batch_op.create_index(batch_op.f("ix_invite_code_usage_invite_code_id"), ["invite_code_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_invite_code_usage_invitee_user_id"), ["invitee_user_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_invite_code_usage_used_at"), ["used_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_invite_code_usage_status"), ["status"], unique=False)
        batch_op.create_index("ix_invite_code_usage_code_status_used", ["invite_code_id", "status", "used_at"], unique=False)

    op.create_table(
        "profit_share_payout",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("public_id", sa.String(length=40), nullable=False),
        sa.Column("invite_code_id", sa.Integer(), nullable=False),
        sa.Column("invitee_user_id", sa.Integer(), nullable=False),
        sa.Column("destination_user_id", sa.Integer(), nullable=True),
        sa.Column("vault_cycle_id", sa.Integer(), nullable=False),
        sa.Column("vault_cycle_settlement_id", sa.Integer(), nullable=True),
        sa.Column("asset", sa.String(length=32), nullable=False),
        sa.Column("source_profit_amount", sa.Numeric(18, 8), nullable=False),
        sa.Column("profit_share_percent", sa.Numeric(8, 4), nullable=False),
        sa.Column("payout_amount", sa.Numeric(18, 8), nullable=False),
        sa.Column("destination_wallet", sa.String(length=120), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("idempotency_key", sa.String(length=240), nullable=False),
        sa.Column("failed_reason", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["destination_user_id"], ["user.id"]),
        sa.ForeignKeyConstraint(["invite_code_id"], ["referral_invite_code.id"]),
        sa.ForeignKeyConstraint(["invitee_user_id"], ["user.id"]),
        sa.ForeignKeyConstraint(["vault_cycle_id"], ["vault_cycle.id"]),
        sa.ForeignKeyConstraint(["vault_cycle_settlement_id"], ["vault_cycle_settlement.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("vault_cycle_id", "invite_code_id", name="uq_profit_share_payout_cycle_invite"),
    )
    with op.batch_alter_table("profit_share_payout", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_profit_share_payout_public_id"), ["public_id"], unique=True)
        batch_op.create_index(batch_op.f("ix_profit_share_payout_invite_code_id"), ["invite_code_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_profit_share_payout_invitee_user_id"), ["invitee_user_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_profit_share_payout_destination_user_id"), ["destination_user_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_profit_share_payout_vault_cycle_id"), ["vault_cycle_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_profit_share_payout_vault_cycle_settlement_id"), ["vault_cycle_settlement_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_profit_share_payout_asset"), ["asset"], unique=False)
        batch_op.create_index(batch_op.f("ix_profit_share_payout_destination_wallet"), ["destination_wallet"], unique=False)
        batch_op.create_index(batch_op.f("ix_profit_share_payout_status"), ["status"], unique=False)
        batch_op.create_index(batch_op.f("ix_profit_share_payout_idempotency_key"), ["idempotency_key"], unique=True)
        batch_op.create_index(batch_op.f("ix_profit_share_payout_created_at"), ["created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_profit_share_payout_completed_at"), ["completed_at"], unique=False)
        batch_op.create_index("ix_profit_share_payout_invite_status_created", ["invite_code_id", "status", "created_at"], unique=False)


def downgrade():
    op.drop_table("profit_share_payout")
    op.drop_table("invite_code_usage")
    op.drop_table("admin_audit_log")

    with op.batch_alter_table("vault_cycle", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_vault_cycle_public_id"))
        batch_op.drop_column("public_id")

    with op.batch_alter_table("referral_invite_code", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_referral_invite_code_assigned_role"))
        batch_op.drop_index(batch_op.f("ix_referral_invite_code_deleted_at"))
        batch_op.drop_index(batch_op.f("ix_referral_invite_code_expires_at"))
        batch_op.drop_index(batch_op.f("ix_referral_invite_code_profit_share_active"))
        batch_op.drop_index(batch_op.f("ix_referral_invite_code_profit_share_wallet"))
        batch_op.drop_index(batch_op.f("ix_referral_invite_code_public_id"))
        batch_op.drop_column("deleted_at")
        batch_op.drop_column("assigned_role")
        batch_op.drop_column("expires_at")
        batch_op.drop_column("applies_to_vault_types_json")
        batch_op.drop_column("profit_share_active")
        batch_op.drop_column("profit_share_ends_at")
        batch_op.drop_column("profit_share_starts_at")
        batch_op.drop_column("profit_share_wallet")
        batch_op.drop_column("profit_share_percent")
        batch_op.drop_column("public_id")
