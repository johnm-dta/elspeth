"""add_source_field_resolution_to_runs

Revision ID: b8a2f1c9d5e3
Revises: 1fd7bd1a762c
Create Date: 2026-01-29 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b8a2f1c9d5e3"
down_revision: str | Sequence[str] | None = "1fd7bd1a762c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add source_field_resolution_json column to runs table.

    Stores field resolution mapping computed during source.load():
    - resolution_mapping: dict mapping original header names to final field names
    - normalization_version: algorithm version used for normalization (null if none)

    This is necessary because field resolution depends on actual file headers which
    are only known after load() runs, but node config is registered before load().

    Nullable for backward compatibility with existing runs.
    """
    # Use batch_alter_table for SQLite compatibility (recreates table)
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("source_field_resolution_json", sa.Text(), nullable=True))


def downgrade() -> None:
    """Remove source_field_resolution_json column from runs table."""
    with op.batch_alter_table("runs", schema=None) as batch_op:
        batch_op.drop_column("source_field_resolution_json")
