"""Audit-derived run-status counter projections."""

from __future__ import annotations

import json

from sqlalchemy import func, select

from elspeth.contracts import NodeStateStatus, NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape._database_ops import ReadOnlyDatabaseOps
from elspeth.core.landscape.schema import (
    node_states_table,
    nodes_table,
    token_outcomes_table,
    tokens_table,
)


class AuditRunStatusProjection:
    """Reconstruct terminal run counters from durable audit tables."""

    def __init__(self, ops: ReadOnlyDatabaseOps) -> None:
        self._ops = ops

    def count_distinct_source_rows_with_terminal_outcome(self, run_id: str) -> int:
        """Count the distinct source rows that reached a terminal outcome.

        ``rows_processed`` semantics — the canonical definition (F2,
        elspeth-resume-fork-reemit) — is "one per *source row*", NOT one per
        terminal token.  The live processing loops increment ``rows_processed``
        exactly once per source row pulled from the source iterator
        (``resume.py`` ``run_resume_processing_loop`` and the main
        ``_run_main_processing_loop``), and structural fan-out
        (fork / expand) or fan-in (aggregation / coalesce) never moves that
        counter.  Reconstructing it from the audit trail therefore CANNOT be a
        per-token tally: a 1-source-row fork emits two leaf tokens, a
        3-source-row aggregation emits one result token, a 1-source-row expand
        emits N children — yet each contributes exactly its *source rows* to
        ``rows_processed`` (1, 3, 1 respectively).

        ``row_id`` is the stable source-row identity (CLAUDE.md DAG model):
        fork and expand children inherit their parent's ``row_id``
        (``tokens.expand_token`` / ``fork_token`` pass ``row_id=parent.row_id``),
        and aggregation's ``BATCH_CONSUMED`` tokens retain their own source
        ``row_id`` while the synthetic result token reuses one of them.  So the
        faithful reconstruction is the count of DISTINCT ``row_id`` among tokens
        that reached a *terminal* outcome (``completed = 1``) — this counts each
        source row once regardless of how many tokens it spawned, and matches an
        uninterrupted run field-for-field (verified across fork / aggregation /
        expand archetypes).

        ``completed = 1`` is the terminal boundary: it includes structural
        TRANSIENT parents (``FORK_PARENT`` / ``EXPAND_PARENT``) and
        ``BATCH_CONSUMED`` tokens (all terminal), and excludes non-terminal
        ``BUFFERED`` rows (``completed = 0``) — a row whose only audit record is
        ``BUFFERED`` has not yet been processed to a terminal state, so it must
        not inflate ``rows_processed``.

        Args:
            run_id: Run ID

        Returns:
            Distinct source-row count among terminal token outcomes.

        Raises:
            AuditIntegrityError: If the count query returns no row — a
                ``COUNT`` aggregate always returns exactly one row, so a NULL
                result indicates Tier-1 audit-database corruption.
        """
        query = (
            select(func.count(func.distinct(tokens_table.c.row_id)))
            .select_from(
                token_outcomes_table.join(
                    tokens_table,
                    (token_outcomes_table.c.token_id == tokens_table.c.token_id) & (token_outcomes_table.c.run_id == tokens_table.c.run_id),
                )
            )
            .where(token_outcomes_table.c.run_id == run_id)
            .where(token_outcomes_table.c.completed == 1)
        )
        r = self._ops.execute_fetchone(query)
        if r is None:
            raise AuditIntegrityError(
                f"count_distinct_source_rows_with_terminal_outcome returned no row for run {run_id!r} — "
                f"a COUNT aggregate must always return exactly one row; a NULL result is a Tier-1 "
                f"audit-database integrity violation."
            )
        return int(r[0])

    def count_failed_coalesce_barrier_rows(self, run_id: str) -> int:
        """Count distinct (coalesce node, source row) join barriers that FAILED.

        ``rows_coalesce_failed`` semantics (elspeth-7294de558e): the counter is
        per failed *barrier* — one pending key ``(coalesce_name, row_id)`` that
        failed to merge — NOT per branch token.  The durable evidence is the
        family of FAILED ``node_states`` that
        ``CoalesceExecutor._fail_pending`` writes at the coalesce node: one
        FAILED state per *arrived branch token*, all sharing the same
        ``(node_id, row_id)``.  A naive count of FAILED states (or of the
        per-branch ``(FAILURE, UNROUTED)`` ``token_outcomes``, which carry no
        node attribution at all) over-reports a 2-branch barrier failure as 2;
        the faithful reconstruction is the count of DISTINCT
        ``(node_id, row_id)`` pairs.

        ANCHOR CHOICE (pinned by
        ``tests/unit/core/landscape/test_query_methods.py::TestAuditRunStatusProjection``):
        the query anchors on ``node_states.status = 'failed'`` joined to
        ``nodes.node_type = 'coalesce'`` — both indexed, structural columns —
        rather than on the ``failure_reason`` strings inside ``error_json``
        (stringly, unindexed, and ambiguous: ``all_branches_lost`` is written
        by two different resolution paths).  ``row_id`` comes from the
        ``tokens`` join (branch tokens of one barrier inherit the same source
        ``row_id`` — the pending key IS ``(coalesce_name, row_id)``).

        ONE exclusion, applied Python-side on the parsed error payload: a
        ``late_arrival_after_merge`` state is a straggler token rejected AFTER
        the barrier already resolved — it is not itself a barrier failure.
        After a *failed* merge the pair is already counted via the barrier's
        own ``_fail_pending`` states (the DISTINCT collapse absorbs the
        straggler); after a *successful* merge the pair must not be counted at
        all, which only the reason exclusion guarantees.

        DELIBERATE breadth: arrival-time barrier failures (branch-lost
        cascades via ``_evaluate_after_loss``, immediate merge failures such
        as ``select_branch_not_arrived``) ARE counted here even though the
        live accumulator only increments ``rows_coalesce_failed`` for barriers
        resolved by the timeout/flush sweeps (``outcomes.py``) — those
        arrival-time failures are real failed barriers and the durable record
        is the broader truth.  Conversely zero-arrival timeout failures
        (``best_effort_timeout_no_arrivals`` or ``first_timeout_no_arrivals``)
        consume no tokens and leave no node_states, so they are invisible here
        by construction.  Reconciling the live accumulator with this durable
        breadth is tracked:
        elspeth-ff6d48c180.

        Cumulativity: resume re-drives record under the SAME ``run_id``
        (resume provenance lives in ``resume_checkpoint_id``), so a single
        run-scoped query covers run-1 failures AND resumed-run failures, and
        the DISTINCT collapse dedupes a barrier that recorded states in both.

        Args:
            run_id: Run ID

        Returns:
            Distinct failed-barrier count for the run (run-1 + all resumes).

        Raises:
            AuditIntegrityError: If a FAILED coalesce node_state carries no
                parseable ``error_json`` — the write side requires an error
                payload for FAILED states, so its absence is Tier-1
                audit-database corruption.
        """
        query = (
            select(
                node_states_table.c.node_id,
                tokens_table.c.row_id,
                node_states_table.c.error_json,
            )
            .select_from(
                node_states_table.join(
                    nodes_table,
                    (node_states_table.c.node_id == nodes_table.c.node_id) & (node_states_table.c.run_id == nodes_table.c.run_id),
                ).join(
                    tokens_table,
                    (node_states_table.c.token_id == tokens_table.c.token_id) & (node_states_table.c.run_id == tokens_table.c.run_id),
                )
            )
            .where(node_states_table.c.run_id == run_id)
            .where(node_states_table.c.status == NodeStateStatus.FAILED.value)
            .where(nodes_table.c.node_type == NodeType.COALESCE.value)
        )
        failed_barriers: set[tuple[str, str]] = set()
        for db_row in self._ops.execute_fetchall(query):
            if db_row.error_json is None:
                raise AuditIntegrityError(
                    f"FAILED coalesce node_state for node {db_row.node_id!r} / row {db_row.row_id!r} in run "
                    f"{run_id!r} has no error_json — the write side requires an error payload for FAILED "
                    f"states, so this is a Tier-1 audit-database integrity violation."
                )
            try:
                error_payload = json.loads(db_row.error_json)
            except json.JSONDecodeError as exc:
                raise AuditIntegrityError(
                    f"FAILED coalesce node_state for node {db_row.node_id!r} / row {db_row.row_id!r} in run "
                    f"{run_id!r} has unparseable error_json — Tier-1 audit-database integrity violation: {exc}"
                ) from exc
            # error_json for a FAILED coalesce node_state is polymorphic across
            # two legitimate writers: coalesce_executor records a
            # CoalesceFailureReason (a dict WITH a required failure_reason) for
            # accept/merge failure outcomes, and the merge-cleanup handler
            # records an ExecutionError (a dict WITHOUT failure_reason, keys
            # {exception,type,phase}) for merge-time exceptions. Both are valid
            # FAILED barrier rows. A non-dict parsed payload is producible by
            # NEITHER writer (both .to_dict() to objects), so it is Tier-1 audit
            # corruption — crash with provenance, consistent with the null /
            # unparseable guards above.
            if not isinstance(error_payload, dict):
                raise AuditIntegrityError(
                    f"FAILED coalesce node_state for node {db_row.node_id!r} / row {db_row.row_id!r} in run "
                    f"{run_id!r} has a non-object error_json payload (got {type(error_payload).__name__}) — the "
                    f"write side serializes a CoalesceFailureReason or ExecutionError object, so this is a "
                    f"Tier-1 audit-database integrity violation."
                )
            # Exclude only the benign late-arrival-after-merge case (a
            # CoalesceFailureReason discriminator). Every other FAILED payload —
            # including ExecutionError shapes with no failure_reason — is a real
            # failed barrier. .get() reads the OPTIONAL discriminator across the
            # two valid payload shapes without crashing on the keyless one.
            if error_payload.get("failure_reason") == "late_arrival_after_merge":
                continue
            failed_barriers.add((db_row.node_id, db_row.row_id))
        return len(failed_barriers)
