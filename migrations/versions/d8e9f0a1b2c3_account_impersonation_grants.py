"""add account impersonation grants

Revision ID: d8e9f0a1b2c3
Revises: a7b8c9d0e1f2
Create Date: 2026-05-25 00:00:00.000000

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "d8e9f0a1b2c3"
down_revision = "a7b8c9d0e1f2"
branch_labels = None
depends_on = None


def upgrade():
    if sa.inspect(op.get_bind()).has_table("account_impersonation_grant"):
        return
    op.create_table(
        "account_impersonation_grant",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("public_id", sa.String(length=40), nullable=False),
        sa.Column("token_hash", sa.String(length=128), nullable=False),
        sa.Column("operator_user_id", sa.Integer(), nullable=False),
        sa.Column("target_user_id", sa.Integer(), nullable=False),
        sa.Column("created_ip_address", sa.String(length=120), nullable=False, server_default=""),
        sa.Column("created_user_agent", sa.Text(), nullable=False, server_default=""),
        sa.Column("consumed_ip_address", sa.String(length=120), nullable=False, server_default=""),
        sa.Column("consumed_user_agent", sa.Text(), nullable=False, server_default=""),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("consumed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["operator_user_id"], ["user.id"]),
        sa.ForeignKeyConstraint(["target_user_id"], ["user.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
        sa.UniqueConstraint("token_hash"),
    )
    op.create_index(op.f("ix_account_impersonation_grant_consumed_at"), "account_impersonation_grant", ["consumed_at"], unique=False)
    op.create_index(op.f("ix_account_impersonation_grant_created_at"), "account_impersonation_grant", ["created_at"], unique=False)
    op.create_index(op.f("ix_account_impersonation_grant_expires_at"), "account_impersonation_grant", ["expires_at"], unique=False)
    op.create_index(
        "ix_account_impersonation_grant_operator_created",
        "account_impersonation_grant",
        ["operator_user_id", "created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_account_impersonation_grant_operator_user_id"), "account_impersonation_grant", ["operator_user_id"], unique=False
    )
    op.create_index(op.f("ix_account_impersonation_grant_public_id"), "account_impersonation_grant", ["public_id"], unique=True)
    op.create_index(
        "ix_account_impersonation_grant_target_created",
        "account_impersonation_grant",
        ["target_user_id", "created_at"],
        unique=False,
    )
    op.create_index(op.f("ix_account_impersonation_grant_target_user_id"), "account_impersonation_grant", ["target_user_id"], unique=False)
    op.create_index(op.f("ix_account_impersonation_grant_token_hash"), "account_impersonation_grant", ["token_hash"], unique=True)


def downgrade():
    op.drop_index(op.f("ix_account_impersonation_grant_token_hash"), table_name="account_impersonation_grant")
    op.drop_index(op.f("ix_account_impersonation_grant_target_user_id"), table_name="account_impersonation_grant")
    op.drop_index("ix_account_impersonation_grant_target_created", table_name="account_impersonation_grant")
    op.drop_index(op.f("ix_account_impersonation_grant_public_id"), table_name="account_impersonation_grant")
    op.drop_index(op.f("ix_account_impersonation_grant_operator_user_id"), table_name="account_impersonation_grant")
    op.drop_index("ix_account_impersonation_grant_operator_created", table_name="account_impersonation_grant")
    op.drop_index(op.f("ix_account_impersonation_grant_expires_at"), table_name="account_impersonation_grant")
    op.drop_index(op.f("ix_account_impersonation_grant_created_at"), table_name="account_impersonation_grant")
    op.drop_index(op.f("ix_account_impersonation_grant_consumed_at"), table_name="account_impersonation_grant")
    op.drop_table("account_impersonation_grant")
