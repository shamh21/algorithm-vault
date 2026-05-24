"""add wallet apple pay purchase orders

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-05-24 00:00:00.000000

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "a7b8c9d0e1f2"
down_revision = "f6a7b8c9d0e1"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "wallet_apple_pay_purchase_order",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("public_id", sa.String(length=40), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("deposit_address_id", sa.Integer(), nullable=True),
        sa.Column("asset", sa.String(length=32), nullable=False),
        sa.Column("network", sa.String(length=64), nullable=False),
        sa.Column("destination_address", sa.Text(), nullable=False),
        sa.Column("fiat_currency", sa.String(length=8), nullable=False, server_default="USD"),
        sa.Column("fiat_gross_amount", sa.Float(), nullable=False, server_default="0"),
        sa.Column("treasury_fee_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("execution_fee_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("net_asset_amount", sa.Float(), nullable=False, server_default="0"),
        sa.Column("payment_method", sa.String(length=32), nullable=False, server_default="apple_pay"),
        sa.Column("gateway_payment_id", sa.String(length=180), nullable=True),
        sa.Column("gateway_capture_id", sa.String(length=180), nullable=True),
        sa.Column("gateway_refund_id", sa.String(length=180), nullable=True),
        sa.Column("oneinch_quote_id", sa.String(length=180), nullable=True),
        sa.Column("oneinch_swap_id", sa.String(length=180), nullable=True),
        sa.Column("fulfillment_tx_hash", sa.String(length=180), nullable=True),
        sa.Column("treasury_tx_hash", sa.String(length=180), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="quoted"),
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
    table = "wallet_apple_pay_purchase_order"
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_asset"), table, ["asset"], unique=False)
    op.create_index("ix_wallet_apple_pay_purchase_order_asset_network", table, ["asset", "network"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_completed_at"), table, ["completed_at"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_created_at"), table, ["created_at"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_deposit_address_id"), table, ["deposit_address_id"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_execution_fee_usd"), table, ["execution_fee_usd"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_expires_at"), table, ["expires_at"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_fiat_currency"), table, ["fiat_currency"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_fulfillment_tx_hash"), table, ["fulfillment_tx_hash"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_gateway_capture_id"), table, ["gateway_capture_id"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_gateway_payment_id"), table, ["gateway_payment_id"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_gateway_refund_id"), table, ["gateway_refund_id"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_idempotency_key"), table, ["idempotency_key"], unique=True)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_network"), table, ["network"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_oneinch_quote_id"), table, ["oneinch_quote_id"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_oneinch_swap_id"), table, ["oneinch_swap_id"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_payment_method"), table, ["payment_method"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_public_id"), table, ["public_id"], unique=True)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_status"), table, ["status"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_treasury_tx_hash"), table, ["treasury_tx_hash"], unique=False)
    op.create_index(op.f("ix_wallet_apple_pay_purchase_order_user_id"), table, ["user_id"], unique=False)
    op.create_index("ix_wallet_apple_pay_purchase_order_user_status_created", table, ["user_id", "status", "created_at"], unique=False)


def downgrade():
    table = "wallet_apple_pay_purchase_order"
    op.drop_index("ix_wallet_apple_pay_purchase_order_user_status_created", table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_user_id"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_treasury_tx_hash"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_status"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_public_id"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_payment_method"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_oneinch_swap_id"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_oneinch_quote_id"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_network"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_idempotency_key"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_gateway_refund_id"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_gateway_payment_id"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_gateway_capture_id"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_fulfillment_tx_hash"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_fiat_currency"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_expires_at"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_execution_fee_usd"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_deposit_address_id"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_created_at"), table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_completed_at"), table_name=table)
    op.drop_index("ix_wallet_apple_pay_purchase_order_asset_network", table_name=table)
    op.drop_index(op.f("ix_wallet_apple_pay_purchase_order_asset"), table_name=table)
    op.drop_table(table)
