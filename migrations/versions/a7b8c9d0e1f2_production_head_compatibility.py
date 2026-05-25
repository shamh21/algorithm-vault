"""production head compatibility for clean impersonation deploy

Revision ID: a7b8c9d0e1f2
Revises: c7f3a2b9d8e1
Create Date: 2026-05-25 00:00:00.000000

"""

from __future__ import annotations

# revision identifiers, used by Alembic.
revision = "a7b8c9d0e1f2"
down_revision = "c7f3a2b9d8e1"
branch_labels = None
depends_on = None


def upgrade():
    """Preserve the production Alembic head already present on algvault.app."""


def downgrade():
    """No schema changes are owned by this compatibility revision."""
