"""wallet onchain snapshots

Revision ID: af3b2c1d4e5f
Revises: 5bb58e96ca8a
Create Date: 2026-05-13 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "af3b2c1d4e5f"
down_revision = "5bb58e96ca8a"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("wallet_address", schema=None) as batch_op:
        batch_op.add_column(sa.Column("onchain_balance", sa.Float(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("onchain_checked_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("onchain_status", sa.String(length=32), nullable=False, server_default="unknown"))
        batch_op.add_column(sa.Column("onchain_reason", sa.Text(), nullable=False, server_default=""))
        batch_op.add_column(sa.Column("onchain_confirmations", sa.Integer(), nullable=False, server_default="0"))
        batch_op.add_column(sa.Column("onchain_provider_reference", sa.String(length=220), nullable=False, server_default=""))
        batch_op.create_index(batch_op.f("ix_wallet_address_onchain_status"), ["onchain_status"], unique=False)


def downgrade():
    with op.batch_alter_table("wallet_address", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_wallet_address_onchain_status"))
        batch_op.drop_column("onchain_provider_reference")
        batch_op.drop_column("onchain_confirmations")
        batch_op.drop_column("onchain_reason")
        batch_op.drop_column("onchain_status")
        batch_op.drop_column("onchain_checked_at")
        batch_op.drop_column("onchain_balance")
