"""add_expected_branches_to_token_outcomes

Stores branch contract at fork/expand time for recovery validation.
Enables detection of "fork promised N branches but only M exist" scenarios.

Revision ID: c9d4e2f1a7b6
Revises: b8a2f1c9d5e3
Create Date: 2026-01-29 14:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c9d4e2f1a7b6"
down_revision: str | Sequence[str] | None = "b8a2f1c9d5e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "token_outcomes",
        sa.Column("expected_branches_json", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("token_outcomes", "expected_branches_json")
