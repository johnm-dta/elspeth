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

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from sqlalchemy import ColumnElement, CompoundSelect, select, union

from elspeth.contracts import Determinism, ReproducibilityGrade
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.schema import (
    calls_table,
    node_states_table,
    nodes_table,
    operations_table,
    rows_table,
    runs_table,
    tokens_table,
)

__all__ = [
    "ReproducibilityGrade",
    "compute_grade",
    "update_grade_after_purge",
]

if TYPE_CHECKING:
    from elspeth.core.landscape.database import LandscapeDB


_REPLAY_ONLY_DETERMINISM = (
    Determinism.EXTERNAL_CALL,
    Determinism.NON_DETERMINISTIC,
    Determinism.IO_READ,
    Determinism.IO_WRITE,
)


def _validate_node_determinism_values(raw_values: Sequence[object], *, run_id: str) -> list[Determinism]:
    """Validate raw node determinism values loaded from the audit database."""
    determinism_values: list[Determinism] = []
    for det_value in raw_values:
        if det_value is None:
            raise AuditIntegrityError(f"NULL determinism value in nodes table for run {run_id} — audit data corruption")
        if not isinstance(det_value, str):
            raise AuditIntegrityError(
                f"Invalid determinism value {det_value!r} in nodes table for run {run_id} — "
                f"expected one of {[d.value for d in Determinism]}"
            )
        try:
            determinism_values.append(Determinism(det_value))
        except ValueError as exc:
            raise AuditIntegrityError(
                f"Invalid determinism value '{det_value}' in nodes table for run {run_id} — "
                f"expected one of {[d.value for d in Determinism]}"
            ) from exc
    return determinism_values


def compute_grade(db: "LandscapeDB", run_id: str) -> ReproducibilityGrade:
    """Compute reproducibility grade from node determinism values.

    Logic:
    - If any node has non-reproducible determinism (EXTERNAL_CALL, NON_DETERMINISTIC,
      IO_READ, IO_WRITE), return REPLAY_REPRODUCIBLE
    - Otherwise return FULL_REPRODUCIBLE
    - Only DETERMINISTIC and SEEDED count as fully reproducible
    - Empty pipeline (no nodes) is trivially FULL_REPRODUCIBLE

    Args:
        db: LandscapeDB instance
        run_id: Run ID to compute grade for

    Returns:
        ReproducibilityGrade enum value

    Raises:
        AuditIntegrityError: If run does not exist or any node has invalid determinism value.
    """
    # Single connection for both queries — avoids TOCTOU window between
    # run existence check and node determinism fetch.
    with db.connection() as conn:
        # Verify run exists before computing grade — a nonexistent run_id
        # must not return FULL_REPRODUCIBLE (which is what "no nodes" implies).
        run_check = conn.execute(select(runs_table.c.run_id).where(runs_table.c.run_id == run_id))
        if run_check.fetchone() is None:
            raise AuditIntegrityError(f"Cannot compute reproducibility grade: run '{run_id}' does not exist")

        # Tier-1 audit data validation: Fetch ALL distinct determinism values
        # and validate each is a valid Determinism enum member.
        # Per Data Manifesto: "Bad data in the audit trail = crash immediately"
        query_all = select(nodes_table.c.determinism).where(nodes_table.c.run_id == run_id).distinct()
        result = conn.execute(query_all)
        raw_values = [row[0] for row in result.fetchall()]

    determinism_values = _validate_node_determinism_values(raw_values, run_id=run_id)

    # Check if any non-reproducible determinism values exist — both sides
    # are now Determinism enum members, no implicit StrEnum comparison.
    has_non_reproducible = any(det in _REPLAY_ONLY_DETERMINISM for det in determinism_values)

    if has_non_reproducible:
        return ReproducibilityGrade.REPLAY_REPRODUCIBLE
    else:
        return ReproducibilityGrade.FULL_REPRODUCIBLE


_PURGE_REF_CHUNK_SIZE = 100


def _replay_critical_response_selects(
    *,
    run_id: str,
    non_reproducible_values: Sequence[Determinism],
    response_ref_condition: ColumnElement[bool],
) -> tuple[Any, Any]:
    """Find replay-critical state-call and operation-call responses."""
    determinism_values = [d.value for d in non_reproducible_values]
    state_call_query = (
        select(calls_table.c.call_id)
        .select_from(
            calls_table.join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id).join(
                nodes_table,
                (node_states_table.c.node_id == nodes_table.c.node_id) & (node_states_table.c.run_id == nodes_table.c.run_id),
            )
        )
        .where(node_states_table.c.run_id == run_id)
        .where(nodes_table.c.determinism.in_(determinism_values))
        .where(calls_table.c.response_hash.isnot(None))
        .where(response_ref_condition)
    )
    operation_call_query = (
        select(calls_table.c.call_id)
        .select_from(
            calls_table.join(operations_table, calls_table.c.operation_id == operations_table.c.operation_id).join(
                nodes_table,
                (operations_table.c.node_id == nodes_table.c.node_id) & (operations_table.c.run_id == nodes_table.c.run_id),
            )
        )
        .where(operations_table.c.run_id == run_id)
        .where(nodes_table.c.determinism.in_(determinism_values))
        .where(calls_table.c.response_hash.isnot(None))
        .where(response_ref_condition)
    )
    return state_call_query, operation_call_query


def _replay_critical_response_query(
    *,
    run_id: str,
    non_reproducible_values: Sequence[Determinism],
    response_ref_condition: ColumnElement[bool],
) -> CompoundSelect[Any]:
    return union(
        *_replay_critical_response_selects(
            run_id=run_id,
            non_reproducible_values=non_reproducible_values,
            response_ref_condition=response_ref_condition,
        )
    )


def _replay_critical_deleted_ref_query(
    *,
    run_id: str,
    non_reproducible_values: Sequence[Determinism],
    deleted_refs: Sequence[str],
) -> CompoundSelect[Any]:
    """Find replay-critical payload records whose refs were deleted."""
    determinism_values = [d.value for d in non_reproducible_values]
    source_rows_query = (
        select(rows_table.c.row_id)
        .select_from(
            rows_table.join(
                nodes_table,
                (rows_table.c.source_node_id == nodes_table.c.node_id) & (rows_table.c.run_id == nodes_table.c.run_id),
            )
        )
        .where(rows_table.c.run_id == run_id)
        .where(nodes_table.c.determinism.in_(determinism_values))
        .where(rows_table.c.source_data_hash.isnot(None))
        .where(rows_table.c.source_data_ref.in_(deleted_refs))
    )
    token_payload_query = (
        select(tokens_table.c.token_id)
        .select_from(
            tokens_table.join(rows_table, tokens_table.c.row_id == rows_table.c.row_id).join(
                nodes_table,
                (rows_table.c.source_node_id == nodes_table.c.node_id) & (rows_table.c.run_id == nodes_table.c.run_id),
            )
        )
        .where(tokens_table.c.run_id == run_id)
        .where(nodes_table.c.determinism.in_(determinism_values))
        .where(tokens_table.c.token_data_ref.in_(deleted_refs))
    )
    operation_input_query = (
        select(operations_table.c.operation_id)
        .select_from(
            operations_table.join(
                nodes_table,
                (operations_table.c.node_id == nodes_table.c.node_id) & (operations_table.c.run_id == nodes_table.c.run_id),
            )
        )
        .where(operations_table.c.run_id == run_id)
        .where(nodes_table.c.determinism.in_(determinism_values))
        .where(operations_table.c.input_data_hash.isnot(None))
        .where(operations_table.c.input_data_ref.in_(deleted_refs))
    )
    operation_output_query = (
        select(operations_table.c.operation_id)
        .select_from(
            operations_table.join(
                nodes_table,
                (operations_table.c.node_id == nodes_table.c.node_id) & (operations_table.c.run_id == nodes_table.c.run_id),
            )
        )
        .where(operations_table.c.run_id == run_id)
        .where(nodes_table.c.determinism.in_(determinism_values))
        .where(operations_table.c.output_data_hash.isnot(None))
        .where(operations_table.c.output_data_ref.in_(deleted_refs))
    )
    return union(
        *_replay_critical_response_selects(
            run_id=run_id,
            non_reproducible_values=non_reproducible_values,
            response_ref_condition=calls_table.c.response_ref.in_(deleted_refs),
        ),
        source_rows_query,
        token_payload_query,
        operation_input_query,
        operation_output_query,
    )


def update_grade_after_purge(db: "LandscapeDB", run_id: str, deleted_refs: Sequence[str] | None = None) -> None:
    """Degrade reproducibility grade after payload purge.

    After payloads are purged, replay-only runs can no longer be replayed IF
    replay-critical payloads have been purged. The grade degrades:
    - REPLAY_REPRODUCIBLE -> ATTRIBUTABLE_ONLY (only if replay-critical payloads purged)
    - FULL_REPRODUCIBLE -> unchanged (doesn't depend on payloads)
    - ATTRIBUTABLE_ONLY -> unchanged (already at lowest grade)

    A payload is replay-critical when it belongs to replay-only evidence:
    call responses, source row payloads, operation inputs/outputs, and token
    payload refs attached to replay-only determinism.

    When ``deleted_refs`` is omitted, this keeps the legacy fallback check for
    rows where ``response_ref`` has already been nulled.

    Args:
        db: LandscapeDB instance
        run_id: Run ID to potentially degrade
        deleted_refs: Payload refs actually removed by the purge operation.
    """
    # Read-then-write in one transaction: carry write intent so the WAL write
    # lock is taken at BEGIN (no BUSY_SNAPSHOT upgrade hazard under peers).
    with db.write_connection() as conn:
        # Tier 1 validation: verify audit data integrity before mutation
        query = select(runs_table.c.reproducibility_grade).where(runs_table.c.run_id == run_id)
        result = conn.execute(query)
        row = result.fetchone()

        if row is None:
            raise AuditIntegrityError(f"Cannot update reproducibility grade after purge: run '{run_id}' does not exist")

        current_grade = row[0]

        # Per Data Manifesto: "Bad data in the audit trail = crash immediately"
        if current_grade is None:
            raise AuditIntegrityError(f"NULL reproducibility_grade for run {run_id} — audit data corruption")

        try:
            grade = ReproducibilityGrade(current_grade)
        except ValueError as exc:
            raise AuditIntegrityError(
                f"Invalid reproducibility_grade '{current_grade}' for run {run_id} — "
                f"expected one of {[g.value for g in ReproducibilityGrade]}"
            ) from exc

        query_all_determinism = select(nodes_table.c.determinism).where(nodes_table.c.run_id == run_id).distinct()
        raw_determinism_values = [determinism_row[0] for determinism_row in conn.execute(query_all_determinism).fetchall()]
        _validate_node_determinism_values(raw_determinism_values, run_id=run_id)

        # Only REPLAY_REPRODUCIBLE can be downgraded (other grades are unaffected)
        if grade != ReproducibilityGrade.REPLAY_REPRODUCIBLE:
            return

        # In the real purge path, refs are intentionally retained and the
        # payload-store blobs are deleted. The deleted_refs argument is the
        # authoritative evidence of that deletion. The response_ref IS NULL
        # fallback preserves compatibility with older callers/tests that mark
        # purged call responses by nulling refs.

        purged_critical = None
        if deleted_refs is None:
            purged_critical = conn.execute(
                _replay_critical_response_query(
                    run_id=run_id,
                    non_reproducible_values=_REPLAY_ONLY_DETERMINISM,
                    response_ref_condition=calls_table.c.response_ref.is_(None),
                ).limit(1)
            ).fetchone()
        else:
            unique_deleted_refs = tuple(dict.fromkeys(deleted_refs))
            for offset in range(0, len(unique_deleted_refs), _PURGE_REF_CHUNK_SIZE):
                chunk = unique_deleted_refs[offset : offset + _PURGE_REF_CHUNK_SIZE]
                purged_critical = conn.execute(
                    _replay_critical_deleted_ref_query(
                        run_id=run_id,
                        non_reproducible_values=_REPLAY_ONLY_DETERMINISM,
                        deleted_refs=chunk,
                    ).limit(1)
                ).fetchone()
                if purged_critical is not None:
                    break

        if purged_critical is not None:
            # Atomic conditional update (same compare-and-swap pattern)
            conn.execute(
                runs_table.update()
                .where(runs_table.c.run_id == run_id)
                .where(runs_table.c.reproducibility_grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE)
                .values(reproducibility_grade=ReproducibilityGrade.ATTRIBUTABLE_ONLY)
            )
