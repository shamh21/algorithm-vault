"""add wallet onramp orders

Revision ID: f6a7b8c9d0e1
Revises: c7f3a2b9d8e1
Create Date: 2026-05-23 00:00:00.000000

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "f6a7b8c9d0e1"
down_revision = "c7f3a2b9d8e1"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "wallet_onramp_order",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("public_id", sa.String(length=40), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("deposit_address_id", sa.Integer(), nullable=True),
        sa.Column("asset", sa.String(length=32), nullable=False),
        sa.Column("network", sa.String(length=64), nullable=False),
        sa.Column("destination_address", sa.Text(), nullable=False),
        sa.Column("fiat_currency", sa.String(length=8), nullable=False, server_default="USD"),
        sa.Column("fiat_amount", sa.Float(), nullable=False, server_default="0"),
        sa.Column("payment_method", sa.String(length=32), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False, server_default="custom_hosted"),
        sa.Column("provider_order_id", sa.String(length=180), nullable=True),
        sa.Column("checkout_url", sa.Text(), nullable=False, server_default=""),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="created"),
        sa.Column("idempotency_key", sa.String(length=220), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=False, server_default=""),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["deposit_address_id"], ["deposit_address.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key"),
        sa.UniqueConstraint("public_id"),
    )
    op.create_index(op.f("ix_wallet_onramp_order_asset"), "wallet_onramp_order", ["asset"], unique=False)
    op.create_index(op.f("ix_wallet_onramp_order_completed_at"), "wallet_onramp_order", ["completed_at"], unique=False)
    op.create_index(op.f("ix_wallet_onramp_order_created_at"), "wallet_onramp_order", ["created_at"], unique=False)
    op.create_index(op.f("ix_wallet_onramp_order_deposit_address_id"), "wallet_onramp_order", ["deposit_address_id"], unique=False)
    op.create_index(op.f("ix_wallet_onramp_order_expires_at"), "wallet_onramp_order", ["expires_at"], unique=False)
    op.create_index(op.f("ix_wallet_onramp_order_fiat_currency"), "wallet_onramp_order", ["fiat_currency"], unique=False)
    op.create_index(op.f("ix_wallet_onramp_order_idempotency_key"), "wallet_onramp_order", ["idempotency_key"], unique=True)
    op.create_index(op.f("ix_wallet_onramp_order_network"), "wallet_onramp_order", ["network"], unique=False)
    op.create_index(op.f("ix_wallet_onramp_order_payment_method"), "wallet_onramp_order", ["payment_method"], unique=False)
    op.create_index(op.f("ix_wallet_onramp_order_provider"), "wallet_onramp_order", ["provider"], unique=False)
    op.create_index(op.f("ix_wallet_onramp_order_provider_order_id"), "wallet_onramp_order", ["provider_order_id"], unique=False)
    op.create_index("ix_wallet_onramp_order_provider_status", "wallet_onramp_order", ["provider", "status"], unique=False)
    op.create_index(op.f("ix_wallet_onramp_order_public_id"), "wallet_onramp_order", ["public_id"], unique=True)
    op.create_index(op.f("ix_wallet_onramp_order_status"), "wallet_onramp_order", ["status"], unique=False)
    op.create_index(op.f("ix_wallet_onramp_order_user_id"), "wallet_onramp_order", ["user_id"], unique=False)
    op.create_index("ix_wallet_onramp_order_user_status_created", "wallet_onramp_order", ["user_id", "status", "created_at"], unique=False)


def downgrade():
    op.drop_index("ix_wallet_onramp_order_user_status_created", table_name="wallet_onramp_order")
    op.drop_index(op.f("ix_wallet_onramp_order_user_id"), table_name="wallet_onramp_order")
    op.drop_index(op.f("ix_wallet_onramp_order_status"), table_name="wallet_onramp_order")
    op.drop_index(op.f("ix_wallet_onramp_order_public_id"), table_name="wallet_onramp_order")
    op.drop_index("ix_wallet_onramp_order_provider_status", table_name="wallet_onramp_order")
    op.drop_index(op.f("ix_wallet_onramp_order_provider_order_id"), table_name="wallet_onramp_order")
    op.drop_index(op.f("ix_wallet_onramp_order_provider"), table_name="wallet_onramp_order")
    op.drop_index(op.f("ix_wallet_onramp_order_payment_method"), table_name="wallet_onramp_order")
    op.drop_index(op.f("ix_wallet_onramp_order_network"), table_name="wallet_onramp_order")
    op.drop_index(op.f("ix_wallet_onramp_order_idempotency_key"), table_name="wallet_onramp_order")
    op.drop_index(op.f("ix_wallet_onramp_order_fiat_currency"), table_name="wallet_onramp_order")
    op.drop_index(op.f("ix_wallet_onramp_order_expires_at"), table_name="wallet_onramp_order")
    op.drop_index(op.f("ix_wallet_onramp_order_deposit_address_id"), table_name="wallet_onramp_order")
    op.drop_index(op.f("ix_wallet_onramp_order_created_at"), table_name="wallet_onramp_order")
    op.drop_index(op.f("ix_wallet_onramp_order_completed_at"), table_name="wallet_onramp_order")
    op.drop_index(op.f("ix_wallet_onramp_order_asset"), table_name="wallet_onramp_order")
    op.drop_table("wallet_onramp_order")
