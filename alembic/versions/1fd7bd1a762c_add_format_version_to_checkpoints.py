"""add_format_version_to_checkpoints

Revision ID: 1fd7bd1a762c
Revises: 4d45ac3604ee
Create Date: 2026-01-27 21:45:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1fd7bd1a762c"
down_revision: str | Sequence[str] | None = "4d45ac3604ee"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add format_version column to checkpoints table.

    Enables version-based checkpoint compatibility checking instead of
    hardcoded date checks. Future format changes increment the version
    rather than adding more date checks.

    Version history:
    - Version 1: Pre-deterministic node IDs (legacy, rejected on resume)
    - Version 2: Deterministic node IDs (2026-01-24+, current)

    Nullable for backward compatibility with existing checkpoints.
    Legacy checkpoints (NULL) fall back to date-based compatibility check.
    """
    # Use batch_alter_table for SQLite compatibility (recreates table)
    with op.batch_alter_table("checkpoints", schema=None) as batch_op:
        batch_op.add_column(sa.Column("format_version", sa.Integer(), nullable=True))


def downgrade() -> None:
    """Remove format_version column from checkpoints table."""
    with op.batch_alter_table("checkpoints", schema=None) as batch_op:
        batch_op.drop_column("format_version")
