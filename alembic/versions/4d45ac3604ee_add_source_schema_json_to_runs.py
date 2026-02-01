"""add_source_schema_json_to_runs

Revision ID: 4d45ac3604ee
Revises: f33c7062a4db
Create Date: 2026-01-25 22:04:16.610457

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4d45ac3604ee"
down_revision: str | Sequence[str] | None = "f33c7062a4db"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add source_schema_json column to runs table for resume type restoration.

    Stores serialized source schema to enable proper type coercion when resuming
    from payloads. Without this, resumed rows have degraded types (all strings).

    Nullable for backward compatibility with existing runs.
    """
    # Use batch_alter_table for SQLite compatibility (recreates table)
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("source_schema_json", sa.Text(), nullable=True))


def downgrade() -> None:
    """Remove source_schema_json column from runs table."""
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.drop_column("source_schema_json")
