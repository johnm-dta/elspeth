"""Token outcome persistence and ADR-019 policy (split from ``DataFlowRepository``).

Owns the ``token_outcomes`` audit aggregate: recording (outcome, path)
terminals with the ADR-019 discriminator-field and cross-table invariant
validation, the deferred I1a/I1b sweeps, and the outcome read models used by
resume and explain.
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.engine import Row as SQLAlchemyRow

from elspeth.contracts import (
    NodeType,
    TokenOutcome,
)
from elspeth.contracts.audit import _TERMINAL_PAIR_FIELD_CONSTRAINTS, DISCARD_SINK_NAME, TokenRef
from elspeth.contracts.enums import BatchStatus, NodeStateStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.canonical import canonical_json
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._helpers import generate_id, now
from elspeth.core.landscape.data_flow.ownership import RowTokenOwnership
from elspeth.core.landscape.model_loaders import TokenOutcomeLoader
from elspeth.core.landscape.schema import (
    artifacts_table,
    batches_table,
    node_states_table,
    nodes_table,
    token_outcomes_table,
    token_parents_table,
    tokens_table,
)

__all__ = ["TokenOutcomeRepository"]


class TokenOutcomeRepository:
    """Token (outcome, path) terminal recording and outcome read models."""

    def __init__(
        self,
        ops: DatabaseOps,
        *,
        token_outcome_loader: TokenOutcomeLoader,
        ownership: RowTokenOwnership,
    ) -> None:
        self._ops = ops
        self._token_outcome_loader = token_outcome_loader
        self._ownership = ownership

    def _validate_outcome_fields(
        self,
        outcome: TerminalOutcome | None,
        path: TerminalPath,
        *,
        sink_name: str | None,
        batch_id: str | None,
        fork_group_id: str | None,
        join_group_id: str | None,
        expand_group_id: str | None,
        error_hash: str | None,
    ) -> None:
        """Validate discriminator fields for the (outcome, path) pair.

        Per ADR-019, producers declare both axes; the recorder must crash before
        writing an ambiguous audit row if the pair is illegal or if required,
        exact, or forbidden discriminator fields are violated.
        """
        pair = (outcome, path)
        if pair not in _TERMINAL_PAIR_FIELD_CONSTRAINTS:
            raise ValueError(
                f"Unhandled (outcome, path) pair in validation: {pair!r}. "
                "See ADR-019 mapping table and update _TERMINAL_PAIR_FIELD_CONSTRAINTS."
            )
        constraints = _TERMINAL_PAIR_FIELD_CONSTRAINTS[pair]
        field_values = {
            "sink_name": sink_name,
            "batch_id": batch_id,
            "fork_group_id": fork_group_id,
            "join_group_id": join_group_id,
            "expand_group_id": expand_group_id,
            "error_hash": error_hash,
        }
        pair_label = f"({outcome.name if outcome else 'NULL'}, {path.name})"
        for field_name in constraints.required:
            if field_values[field_name] is None:
                raise ValueError(
                    f"{pair_label} outcome requires {field_name} but got None. Contract violation — see ADR-019 Implementation Notes."
                )
        for field_name, expected in constraints.exact.items():
            if field_values[field_name] != expected:
                raise ValueError(
                    f"{pair_label} outcome requires {field_name}={expected!r}, "
                    f"got {field_values[field_name]!r}. "
                    "Contract violation — see ADR-019 Implementation Notes."
                )
        for field_name in constraints.forbidden:
            if field_values[field_name] is not None:
                raise ValueError(
                    f"{pair_label} outcome forbids {field_name}, got {field_values[field_name]!r}. "
                    "Contract violation — see ADR-019 Implementation Notes."
                )

    def _validate_cross_table_invariants(
        self,
        ref: TokenRef,
        outcome: TerminalOutcome | None,
        path: TerminalPath,
        *,
        sink_name: str | None,
        sink_node_id: str | None,
        artifact_id: str | None,
    ) -> None:
        """Validate ADR-019 real-time cross-table invariants.

        I1c validates exact failsink node-state and artifact witnesses for
        failsink fallback. I3 validates that discard records do not coexist
        with a completed sink node-state for the same token.
        """
        pair = (outcome, path)

        if pair == (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK):
            if sink_node_id is None:
                raise AuditIntegrityError(
                    f"ADR-019 I1c violation for token {ref.token_id}: "
                    "(TRANSIENT, SINK_FALLBACK_TO_FAILSINK) requires an exact "
                    "failsink node_id witness."
                )
            if artifact_id is None:
                raise AuditIntegrityError(
                    f"ADR-019 I1c violation for token {ref.token_id}: "
                    "(TRANSIENT, SINK_FALLBACK_TO_FAILSINK) requires an exact "
                    "failsink artifact_id witness."
                )

            completed_sink_state = self._ops.execute_fetchone(
                select(node_states_table.c.state_id, node_states_table.c.node_id)
                .select_from(
                    node_states_table.join(
                        nodes_table,
                        and_(
                            node_states_table.c.node_id == nodes_table.c.node_id,
                            node_states_table.c.run_id == nodes_table.c.run_id,
                        ),
                    )
                )
                .where(node_states_table.c.token_id == ref.token_id)
                .where(node_states_table.c.run_id == ref.run_id)
                .where(node_states_table.c.node_id == sink_node_id)
                .where(node_states_table.c.status == NodeStateStatus.COMPLETED.value)
                .where(nodes_table.c.node_type == NodeType.SINK.value)
            )
            if completed_sink_state is None:
                raise AuditIntegrityError(
                    f"ADR-019 I1c violation for token {ref.token_id}: "
                    "failsink fallback requires a paired COMPLETED sink "
                    f"node_state at sink_node_id={sink_node_id!r}."
                )

            artifact_row = self._ops.execute_fetchone(
                select(artifacts_table.c.artifact_id)
                .select_from(
                    artifacts_table.join(
                        node_states_table,
                        and_(
                            artifacts_table.c.produced_by_state_id == node_states_table.c.state_id,
                            artifacts_table.c.run_id == node_states_table.c.run_id,
                        ),
                    ).join(
                        nodes_table,
                        and_(
                            node_states_table.c.node_id == nodes_table.c.node_id,
                            node_states_table.c.run_id == nodes_table.c.run_id,
                        ),
                    )
                )
                .where(artifacts_table.c.artifact_id == artifact_id)
                .where(artifacts_table.c.run_id == ref.run_id)
                .where(artifacts_table.c.sink_node_id == sink_node_id)
                .where(node_states_table.c.node_id == sink_node_id)
                .where(node_states_table.c.status == NodeStateStatus.COMPLETED.value)
                .where(nodes_table.c.node_type == NodeType.SINK.value)
            )
            if artifact_row is None:
                raise AuditIntegrityError(
                    f"ADR-019 I1c violation for token {ref.token_id}: "
                    f"failsink node {completed_sink_state.node_id!r} has no "
                    f"artifact_id={artifact_id!r} witness produced by a "
                    "COMPLETED sink node_state at this sink."
                )

        if pair == (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED):
            if sink_name != DISCARD_SINK_NAME:
                raise AuditIntegrityError(
                    f"ADR-019 I3 violation for token {ref.token_id}: "
                    f"SINK_DISCARDED requires sink_name={DISCARD_SINK_NAME!r}, "
                    f"got {sink_name!r}."
                )

            completed_sink_state = self._ops.execute_fetchone(
                select(node_states_table.c.state_id)
                .select_from(
                    node_states_table.join(
                        nodes_table,
                        and_(
                            node_states_table.c.node_id == nodes_table.c.node_id,
                            node_states_table.c.run_id == nodes_table.c.run_id,
                        ),
                    )
                )
                .where(node_states_table.c.token_id == ref.token_id)
                .where(node_states_table.c.run_id == ref.run_id)
                .where(node_states_table.c.status == NodeStateStatus.COMPLETED.value)
                .where(nodes_table.c.node_type == NodeType.SINK.value)
            )
            if completed_sink_state is not None:
                raise AuditIntegrityError(
                    f"ADR-019 I3 violation for token {ref.token_id}: discard "
                    "recording contradicts an existing COMPLETED sink "
                    f"node_state ({completed_sink_state.state_id})."
                )

    def record_token_outcome(
        self,
        ref: TokenRef,
        outcome: TerminalOutcome | None,
        path: TerminalPath,
        *,
        sink_name: str | None = None,
        sink_node_id: str | None = None,
        artifact_id: str | None = None,
        batch_id: str | None = None,
        fork_group_id: str | None = None,
        join_group_id: str | None = None,
        expand_group_id: str | None = None,
        error_hash: str | None = None,
        context: Mapping[str, object] | None = None,
    ) -> str:
        """Record a token's (outcome, path) audit terminal in the audit trail.

        Called at the moment the producer determines the terminal pair. For
        BUFFERED tokens (outcome=None, path=BUFFERED), a second call records
        the actual lifecycle terminal when the batch flushes.

        Validates that the token belongs to the specified run_id before recording.
        Cross-run contamination crashes immediately per Tier 1 trust model.

        Args:
            ref: TokenRef bundling token_id and run_id
            outcome: TerminalOutcome lifecycle answer, or None for BUFFERED
            path: TerminalPath provenance answer (always required)
            sink_name: For paths that reach a sink (REQUIRED for those)
            sink_node_id: Forward-compatible Phase 4 witness keyword for
                failsink-paired outcomes. Accepted but not written in Phase 1.
            artifact_id: Forward-compatible Phase 4 witness keyword for
                failsink-paired outcomes. Accepted but not written in Phase 1.
            batch_id: For BATCH_CONSUMED / BUFFERED (REQUIRED)
            fork_group_id: For FORK_PARENT (REQUIRED)
            join_group_id: For COALESCED (REQUIRED)
            expand_group_id: For EXPAND_PARENT (REQUIRED)
            error_hash: Error witness for failure/transient error paths
            context: Optional additional context (stored as JSON)

        Returns:
            outcome_id for tracking

        Raises:
            ValueError: If required fields for outcome type are missing
            AuditIntegrityError: If token does not belong to the specified run
            IntegrityError: If terminal outcome already exists for token
        """
        self._validate_outcome_fields(
            outcome,
            path,
            sink_name=sink_name,
            batch_id=batch_id,
            fork_group_id=fork_group_id,
            join_group_id=join_group_id,
            expand_group_id=expand_group_id,
            error_hash=error_hash,
        )

        # Validate token belongs to the specified run (Tier 1 invariant)
        self._ownership.validate_token_run_ownership(ref)
        self._validate_cross_table_invariants(
            ref,
            outcome,
            path,
            sink_name=sink_name,
            sink_node_id=sink_node_id,
            artifact_id=artifact_id,
        )

        outcome_id = f"out_{generate_id()[:12]}"
        completed = outcome is not None
        context_json = canonical_json(context) if context is not None else None

        self._ops.execute_insert(
            token_outcomes_table.insert().values(
                outcome_id=outcome_id,
                run_id=ref.run_id,
                token_id=ref.token_id,
                outcome=outcome.value if outcome is not None else None,
                path=path.value,
                completed=1 if completed else 0,
                recorded_at=now(),
                sink_name=sink_name,
                batch_id=batch_id,
                fork_group_id=fork_group_id,
                join_group_id=join_group_id,
                expand_group_id=expand_group_id,
                error_hash=error_hash,
                context_json=context_json,
            )
        )

        return outcome_id

    def find_orphaned_transient_parents(self, run_id: str) -> list[SQLAlchemyRow[Any]]:
        """Find I1a parent tokens with no child token outcome witnesses."""
        parent_paths = (
            TerminalPath.FORK_PARENT.value,
            TerminalPath.EXPAND_PARENT.value,
        )
        child_outcomes = token_outcomes_table.alias("child_outcomes")
        child_witness = (
            select(child_outcomes.c.outcome_id)
            .select_from(
                token_parents_table.join(
                    child_outcomes,
                    and_(
                        child_outcomes.c.token_id == token_parents_table.c.token_id,
                        child_outcomes.c.run_id == run_id,
                    ),
                )
            )
            .where(token_parents_table.c.parent_token_id == token_outcomes_table.c.token_id)
        )
        query = (
            select(token_outcomes_table.c.token_id, token_outcomes_table.c.path)
            .where(token_outcomes_table.c.run_id == run_id)
            .where(token_outcomes_table.c.path.in_(parent_paths))
            .where(token_outcomes_table.c.outcome == TerminalOutcome.TRANSIENT.value)
            .where(~child_witness.exists())
        )
        return list(self._ops.execute_fetchall(query))

    def find_orphaned_batch_consumptions(self, run_id: str) -> list[str]:
        """Find I1b batch IDs consumed by tokens whose batch did not complete."""
        completed_batch_witness = (
            select(batches_table.c.batch_id)
            .where(batches_table.c.batch_id == token_outcomes_table.c.batch_id)
            .where(batches_table.c.run_id == run_id)
            .where(batches_table.c.status == BatchStatus.COMPLETED.value)
        )
        query = (
            select(token_outcomes_table.c.batch_id)
            .distinct()
            .where(token_outcomes_table.c.run_id == run_id)
            .where(token_outcomes_table.c.path == TerminalPath.BATCH_CONSUMED.value)
            .where(token_outcomes_table.c.outcome == TerminalOutcome.TRANSIENT.value)
            .where(~completed_batch_witness.exists())
        )
        return [row.batch_id for row in self._ops.execute_fetchall(query)]

    def sweep_deferred_invariants_or_crash(self, run_id: str) -> None:
        """Sweep ADR-019 deferred I1a/I1b invariants at a stable run boundary."""
        orphan_parents = self.find_orphaned_transient_parents(run_id)
        if orphan_parents:
            examples = ", ".join(f"{row.token_id} (path={row.path})" for row in orphan_parents[:10])
            raise AuditIntegrityError(
                f"ADR-019 I1a violation: {len(orphan_parents)} fork/expand "
                "parent token(s) have no child token_outcomes rows at run-end. "
                f"Examples: {examples}."
            )

        orphan_batches = self.find_orphaned_batch_consumptions(run_id)
        if orphan_batches:
            examples = ", ".join(orphan_batches[:10])
            raise AuditIntegrityError(
                f"ADR-019 I1b violation: {len(orphan_batches)} batch_id(s) had "
                "BATCH_CONSUMED tokens but the batch never reached "
                f"BatchStatus.COMPLETED. Examples: {examples}."
            )

    def get_token_outcome(self, token_id: str) -> TokenOutcome | None:
        """Get the terminal outcome for a token.

        Returns the terminal outcome if one exists, otherwise the most
        recent non-terminal outcome (BUFFERED).

        Args:
            token_id: Token to look up

        Returns:
            TokenOutcome dataclass or None if no outcome recorded
        """
        # Get most recent outcome (terminal preferred)
        query = (
            select(token_outcomes_table)
            .where(token_outcomes_table.c.token_id == token_id)
            .order_by(
                token_outcomes_table.c.completed.desc(),  # Terminal first
                token_outcomes_table.c.recorded_at.desc(),  # Then by time
            )
            .limit(1)
        )
        result = self._ops.execute_fetchone(query)
        if result is None:
            return None
        return self._token_outcome_loader.load(result)

    def get_live_buffered_outcomes(self, ref: TokenRef) -> list[TokenOutcome]:
        """All LIVE BUFFERED outcomes for a token (ADR-030 §E.4 restore read).

        "Live" = the token has no ``completed=1`` outcome; a flushed token's
        BUFFERED row is dead history and exempt. ``token_outcomes`` has NO
        non-terminal uniqueness (the only unique index is partial on
        ``completed=1``), so more than one live BUFFERED row means a deposed
        leader's unfenced intake wrote a second acceptance — the restore path
        (``_derive_restored_batch_id``) refuses loudly with Tier-1 instead of
        the historical silent latest-wins of :meth:`get_token_outcome`. At
        epoch 21 this is the BACKSTOP behind the adoption CAS, which is the
        structural guarantee.

        Ordered by ``(recorded_at, outcome_id)`` for deterministic reporting.
        """
        terminal = token_outcomes_table.alias("terminal_outcomes")
        terminal_witness = (
            select(terminal.c.outcome_id)
            .where(terminal.c.token_id == ref.token_id)
            .where(terminal.c.run_id == ref.run_id)
            .where(terminal.c.completed == 1)
            .exists()
        )
        query = (
            select(token_outcomes_table)
            .where(token_outcomes_table.c.token_id == ref.token_id)
            .where(token_outcomes_table.c.run_id == ref.run_id)
            .where(token_outcomes_table.c.completed == 0)
            .where(token_outcomes_table.c.path == TerminalPath.BUFFERED.value)
            .where(~terminal_witness)
            .order_by(token_outcomes_table.c.recorded_at, token_outcomes_table.c.outcome_id)
        )
        return [self._token_outcome_loader.load(row) for row in self._ops.execute_fetchall(query)]

    def get_failed_unrouted_terminal_token_ids(self, run_id: str, token_ids: Sequence[str]) -> frozenset[str]:
        """Token ids (from ``token_ids``) holding a terminal FAILURE/UNROUTED outcome.

        The restore-side aggregation reconcile signature (ADR-030 §E.3a
        aggregation mirror, elspeth-55546a6fd6). A FAILED out-of-claim
        aggregation flush records a completed FAILURE/UNROUTED token_outcome for
        every buffered token (``_handle_flush_error``) and THEN releases their
        BLOCKED scheduler rows in a SEPARATE transaction
        (``_mark_buffered_scheduler_work_terminal``). A crash between the two
        strands durable BLOCKED rows whose tokens are already terminally failed:
        they hold no live BUFFERED outcome, so ``_derive_restored_batch_id``
        cannot proceed, yet they are genuinely done. The barrier restore
        reconcile journal-releases exactly these tokens instead of bricking the
        resume.

        Scoped to the ``(FAILURE, UNROUTED)`` pair so the success-path
        BATCH_CONSUMED crash residual (elspeth-3977d8ab60, which still owes a
        sink output and must NOT be silently released) is excluded — any other
        terminal-while-BLOCKED state keeps hitting the loud restore refusal.
        """
        if not token_ids:
            return frozenset()
        query = (
            select(token_outcomes_table.c.token_id)
            .where(token_outcomes_table.c.run_id == run_id)
            .where(token_outcomes_table.c.token_id.in_(tuple(token_ids)))
            .where(token_outcomes_table.c.completed == 1)
            .where(token_outcomes_table.c.outcome == TerminalOutcome.FAILURE.value)
            .where(token_outcomes_table.c.path == TerminalPath.UNROUTED.value)
        )
        return frozenset(row.token_id for row in self._ops.execute_fetchall(query))

    def find_duplicate_live_buffered_outcomes(self, run_id: str) -> list[tuple[str, int]]:
        """Run-wide sweep: token_ids holding >1 live BUFFERED outcome.

        The cheap belt to :meth:`get_live_buffered_outcomes`' per-token check,
        intended to run once at barrier-restore entry for a single loud
        report. Returns ``(token_id, live_buffered_count)`` pairs ordered by
        token_id; empty means the adoption-CAS invariant held.
        """
        terminal = token_outcomes_table.alias("terminal_outcomes")
        terminal_witness = (
            select(terminal.c.outcome_id)
            .where(terminal.c.token_id == token_outcomes_table.c.token_id)
            .where(terminal.c.run_id == run_id)
            .where(terminal.c.completed == 1)
            .exists()
        )
        query = (
            select(token_outcomes_table.c.token_id, func.count())
            .where(token_outcomes_table.c.run_id == run_id)
            .where(token_outcomes_table.c.completed == 0)
            .where(token_outcomes_table.c.path == TerminalPath.BUFFERED.value)
            .where(~terminal_witness)
            .group_by(token_outcomes_table.c.token_id)
            .having(func.count() > 1)
            .order_by(token_outcomes_table.c.token_id)
        )
        return [(str(row[0]), int(row[1])) for row in self._ops.execute_fetchall(query)]

    def get_token_outcomes_for_row(self, run_id: str, row_id: str) -> list[TokenOutcome]:
        """Get all token outcomes for a row in a single query.

        Uses JOIN to avoid N+1 query pattern when resolving row_id to tokens.
        Critical for explain() disambiguation with forks/expands.

        Args:
            run_id: Run ID to filter by (prevents cross-run contamination)
            row_id: Row ID

        Returns:
            List of TokenOutcome objects, empty if no outcomes recorded.
            Ordered by recorded_at for deterministic behavior.
        """
        # Single JOIN query: tokens + outcomes
        query = (
            select(
                token_outcomes_table.c.outcome_id,
                token_outcomes_table.c.run_id,
                token_outcomes_table.c.token_id,
                token_outcomes_table.c.outcome,
                token_outcomes_table.c.path,
                token_outcomes_table.c.completed,
                token_outcomes_table.c.recorded_at,
                token_outcomes_table.c.sink_name,
                token_outcomes_table.c.batch_id,
                token_outcomes_table.c.fork_group_id,
                token_outcomes_table.c.join_group_id,
                token_outcomes_table.c.expand_group_id,
                token_outcomes_table.c.error_hash,
                token_outcomes_table.c.context_json,
                token_outcomes_table.c.expected_branches_json,
            )
            .join(
                tokens_table,
                token_outcomes_table.c.token_id == tokens_table.c.token_id,
            )
            .where(tokens_table.c.row_id == row_id)
            .where(token_outcomes_table.c.run_id == run_id)
            .order_by(token_outcomes_table.c.recorded_at)
        )
        rows = self._ops.execute_fetchall(query)
        return [self._token_outcome_loader.load(r) for r in rows]
