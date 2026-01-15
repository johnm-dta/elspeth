# src/elspeth/core/landscape/reproducibility.py
"""Reproducibility grade computation for completed pipeline runs.

This module computes and manages the reproducibility_grade field on the runs
table, which indicates how reliably a run can be reproduced or replayed.

Grades:
- FULL_REPRODUCIBLE: All nodes are deterministic or seeded. The run can be
  fully re-executed with identical results (given the same seed).
- REPLAY_REPRODUCIBLE: At least one node is nondeterministic (e.g., LLM calls).
  Results can only be replayed using recorded external call responses.
- ATTRIBUTABLE_ONLY: Payloads have been purged. We can verify what happened
  via hashes, but cannot replay the run.
"""

from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import select

from elspeth.core.landscape.schema import nodes_table, runs_table
from elspeth.plugins.enums import Determinism

if TYPE_CHECKING:
    from elspeth.core.landscape.database import LandscapeDB


class ReproducibilityGrade(str, Enum):
    """Reproducibility levels for a completed run.

    Using str as base allows direct JSON serialization and comparison.
    """

    FULL_REPRODUCIBLE = "full_reproducible"
    REPLAY_REPRODUCIBLE = "replay_reproducible"
    ATTRIBUTABLE_ONLY = "attributable_only"


def compute_grade(db: "LandscapeDB", run_id: str) -> ReproducibilityGrade:
    """Compute reproducibility grade from node determinism values.

    Logic:
    - If any node has determinism='nondeterministic', return REPLAY_REPRODUCIBLE
    - Otherwise return FULL_REPRODUCIBLE
    - 'seeded' counts as reproducible (same seed = same output)
    - Empty pipeline (no nodes) is trivially FULL_REPRODUCIBLE

    Args:
        db: LandscapeDB instance
        run_id: Run ID to compute grade for

    Returns:
        ReproducibilityGrade enum value
    """
    # Query for any nondeterministic nodes in this run
    query = (
        select(nodes_table.c.determinism)
        .where(nodes_table.c.run_id == run_id)
        .where(nodes_table.c.determinism == Determinism.NONDETERMINISTIC.value)
        .limit(1)  # We only need to know if ANY exist
    )

    with db.connection() as conn:
        result = conn.execute(query)
        has_nondeterministic = result.fetchone() is not None

    if has_nondeterministic:
        return ReproducibilityGrade.REPLAY_REPRODUCIBLE
    else:
        return ReproducibilityGrade.FULL_REPRODUCIBLE


def set_run_grade(db: "LandscapeDB", run_id: str, grade: ReproducibilityGrade) -> None:
    """Set reproducibility grade on the runs table.

    Args:
        db: LandscapeDB instance
        run_id: Run ID to update
        grade: ReproducibilityGrade to set
    """
    with db.connection() as conn:
        conn.execute(
            runs_table.update()
            .where(runs_table.c.run_id == run_id)
            .values(reproducibility_grade=grade.value)
        )


def update_grade_after_purge(db: "LandscapeDB", run_id: str) -> None:
    """Degrade reproducibility grade after payload purge.

    After payloads are purged, nondeterministic runs can no longer be
    replayed (we don't have the recorded responses). The grade degrades:
    - REPLAY_REPRODUCIBLE -> ATTRIBUTABLE_ONLY
    - FULL_REPRODUCIBLE -> unchanged (doesn't depend on payloads)
    - ATTRIBUTABLE_ONLY -> unchanged (already at lowest grade)

    Args:
        db: LandscapeDB instance
        run_id: Run ID to potentially degrade
    """
    # Use single connection for transactional safety (read-modify-write)
    query = select(runs_table.c.reproducibility_grade).where(
        runs_table.c.run_id == run_id
    )

    with db.connection() as conn:
        result = conn.execute(query)
        row = result.fetchone()

        if row is None:
            return  # Run doesn't exist

        current_grade = row[0]

        # Only REPLAY_REPRODUCIBLE needs to degrade
        # Also guard against NULL grades (shouldn't happen, but defensive)
        if current_grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE.value:
            conn.execute(
                runs_table.update()
                .where(runs_table.c.run_id == run_id)
                .values(
                    reproducibility_grade=ReproducibilityGrade.ATTRIBUTABLE_ONLY.value
                )
            )
