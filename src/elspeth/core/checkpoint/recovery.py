"""Recovery protocol for resuming failed runs.

Provides the API for determining if and how a failed run can be resumed:
- can_resume(run_id) - Check if run can be resumed (failed status + checkpoint exists)
- get_resume_point(run_id) - Get checkpoint info for resuming

The actual resume logic (Orchestrator.resume()) is implemented separately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.engine import Row

from elspeth.contracts import (
    Checkpoint,
    PayloadNotFoundError,
    PayloadStore,
    PipelineRow,
    PluginSchema,
    ResumeCheck,
    ResumePoint,
    RunStatus,
    SchemaContract,
    TerminalPath,
)
from elspeth.contracts.aggregation_checkpoint import AggregationCheckpointState
from elspeth.contracts.coalesce_checkpoint import CoalesceCheckpointState
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.checkpoint.compatibility import CheckpointCompatibilityValidator
from elspeth.core.checkpoint.manager import CheckpointCorruptionError, CheckpointManager, IncompatibleCheckpointError
from elspeth.core.checkpoint.serialization import checkpoint_loads
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import (
    node_states_table,
    rows_table,
    runs_table,
    token_outcomes_table,
    tokens_table,
)

if TYPE_CHECKING:
    from elspeth.core.dag import ExecutionGraph

# SQLite's SQLITE_MAX_VARIABLE_NUMBER defaults to 999. We chunk IN clauses
# at 500 to leave headroom for other query parameters in the same statement.
_METADATA_CHUNK_SIZE = 500
_CHECKPOINT_STATE_CACHE_MAX = 16
_DELEGATION_PATHS = (TerminalPath.FORK_PARENT.value, TerminalPath.EXPAND_PARENT.value)
_RESUMABLE_RUN_STATUSES = frozenset({RunStatus.FAILED, RunStatus.INTERRUPTED})
_CheckpointStateCacheKey = tuple[str, str | None, str | None]

__all__ = [
    "IncompleteTokenSpec",
    "RecoveryManager",
    "ResumeCheck",  # Re-exported from contracts for convenience
    "ResumePoint",  # Re-exported from contracts for convenience
]


@dataclass(frozen=True, slots=True)
class IncompleteTokenSpec:
    """A non-delegation child token that lacks a terminal outcome on a resumed run.

    Identity fields read directly from persisted columns (Tier-1: no defaults, no
    coercion). ``token_data_ref`` is NULL for fork children (they share the
    parent/source payload, retrievable by ``row_id``) and set for expand children
    and post-coalesce merged tokens. ``max_attempt`` is the highest ``attempt``
    already recorded for this token in ``node_states`` (-1 if none); the re-drive
    uses ``max_attempt + 1``.

    Validates its own identity at construction (``__post_init__``) rather than
    deferring to the downstream ``TokenInfo`` guard: this spec reaches
    ``reconstruct_token_row`` — which uses ``token_id`` in diagnostics and
    ``token_data_ref`` as a payload-store key — BEFORE any ``TokenInfo`` is built.
    NOT NULL (the DB constraint) is not the same as non-empty, so an empty-string
    identity read from a corrupt/tampered Tier-1 row must crash here, mirroring the
    sibling identity types (``TokenInfo``, ``ValueSourceFinding``).

    All fields are scalars or None — no deep_freeze guard needed (frozen=True suffices).
    """

    token_id: str
    row_id: str
    branch_name: str | None
    fork_group_id: str | None
    join_group_id: str | None
    expand_group_id: str | None
    token_data_ref: str | None
    step_in_pipeline: int | None
    max_attempt: int

    def __post_init__(self) -> None:
        """Validate identity invariants at construction time (Tier-1 boundary).

        ``token_id`` and ``row_id`` are the fundamental identity fields — every
        audit record references them; empty strings would produce valid-looking
        but meaningless audit entries. The optional lineage/payload fields are
        either NULL (legitimately not applicable) or non-empty strings — an empty
        string is anomalous, and ``token_data_ref`` in particular is dereferenced
        as a payload-store key before any ``TokenInfo`` guard exists.
        """
        for _field_name in ("token_id", "row_id"):
            _value = getattr(self, _field_name)
            if not isinstance(_value, str):
                raise TypeError(f"IncompleteTokenSpec.{_field_name} must be str, got {type(_value).__name__}: {_value!r}")
            if not _value:
                raise ValueError(f"IncompleteTokenSpec.{_field_name} must not be empty")
        for _field_name in ("branch_name", "fork_group_id", "join_group_id", "expand_group_id", "token_data_ref"):
            _value = getattr(self, _field_name)
            if _value is not None:
                if not isinstance(_value, str):
                    raise TypeError(f"IncompleteTokenSpec.{_field_name} must be str or None, got {type(_value).__name__}: {_value!r}")
                if not _value:
                    raise ValueError(f"IncompleteTokenSpec.{_field_name} must be None or non-empty string, got {_value!r}")


@dataclass(frozen=True, slots=True)
class _RestoredCheckpointStates:
    aggregation_state: AggregationCheckpointState | None
    coalesce_state: CoalesceCheckpointState | None


class RecoveryManager:
    """Manages recovery of failed runs from checkpoints.

    Recovery protocol:
    1. Check if run can be resumed (failed status + checkpoint exists)
    2. Load checkpoint and aggregation state
    3. Identify unprocessed rows (sequence > checkpoint.sequence)
    4. Resume processing from checkpoint position

    Usage:
        recovery = RecoveryManager(db, checkpoint_manager)

        check = recovery.can_resume(run_id)
        if check.can_resume:
            resume_point = recovery.get_resume_point(run_id)
            # Pass resume_point to Orchestrator.resume()
    """

    def __init__(self, db: LandscapeDB, checkpoint_manager: CheckpointManager) -> None:
        """Initialize with Landscape database and checkpoint manager.

        Args:
            db: LandscapeDB instance for querying run status
            checkpoint_manager: CheckpointManager for loading checkpoints
        """
        self._db = db
        self._checkpoint_manager = checkpoint_manager
        self._checkpoint_state_cache: dict[_CheckpointStateCacheKey, _RestoredCheckpointStates] = {}

    def can_resume(self, run_id: str, graph: ExecutionGraph) -> ResumeCheck:
        """Check if a run can be resumed.

        A run can be resumed if:
        - It exists in the database
        - Its status is "failed" (not "completed" or "running")
        - At least one checkpoint exists for recovery
        - The checkpoint's upstream topology is compatible with current graph
        - The stored schema contract passes integrity verification (if present)

        Args:
            run_id: The run to check
            graph: The current execution graph to validate against

        Returns:
            ResumeCheck with can_resume=True if resumable,
            or can_resume=False with reason explaining why not.

        Raises:
            CheckpointCorruptionError: If schema contract integrity check fails.
                This is a Tier 1 failure - corruption cannot be silently ignored.
        """
        run = self._get_run(run_id)
        if run is None:
            return ResumeCheck(can_resume=False, reason=f"Run {run_id} not found")

        try:
            run_status = RunStatus(run.status)
        except ValueError as exc:
            raise CheckpointCorruptionError(f"Run {run_id} has invalid status {run.status!r}; audit trail is corrupt") from exc

        if run_status == RunStatus.COMPLETED:
            return ResumeCheck(can_resume=False, reason="Run already completed successfully")

        if run_status == RunStatus.RUNNING:
            return ResumeCheck(can_resume=False, reason="Run is still in progress")

        if run_status not in _RESUMABLE_RUN_STATUSES:
            return ResumeCheck(can_resume=False, reason=f"Run status {run_status.value!r} is not resumable")

        try:
            checkpoint = self._checkpoint_manager.get_latest_checkpoint(run_id)
        except IncompatibleCheckpointError as e:
            # Return ResumeCheck instead of propagating exception (API contract)
            return ResumeCheck(can_resume=False, reason=str(e))
        if checkpoint is None:
            return ResumeCheck(can_resume=False, reason="No checkpoint found for recovery")

        # Validate topological compatibility
        validator = CheckpointCompatibilityValidator()
        topology_check = validator.validate(checkpoint, graph)
        if not topology_check.can_resume:
            return topology_check

        # Verify schema contract integrity (Tier 1 - raises on corruption)
        # This must happen AFTER topology validation passes, as contract
        # corruption is a more serious failure than config mismatch.
        # Note: Returns None if no contract stored (valid for legacy runs)
        self.verify_contract_integrity(run_id)

        return ResumeCheck(can_resume=True)

    def get_resume_point(self, run_id: str, graph: ExecutionGraph) -> ResumePoint | None:
        """Get the resume point for a failed run.

        Returns all information needed to resume processing:
        - The checkpoint itself (for audit trail)
        - Token ID to resume from
        - Node ID where processing stopped
        - Sequence number for ordering
        - Deserialized aggregation state (if any)

        Args:
            run_id: The run to get resume point for
            graph: The current execution graph to validate against

        Returns:
            ResumePoint if run can be resumed, None otherwise
        """
        check = self.can_resume(run_id, graph)
        if not check.can_resume:
            return None

        try:
            checkpoint = self._checkpoint_manager.get_latest_checkpoint(run_id)
        except IncompatibleCheckpointError:
            return None
        if checkpoint is None:
            return None

        topology_check = CheckpointCompatibilityValidator().validate(checkpoint, graph)
        if not topology_check.can_resume:
            return None

        self.verify_contract_integrity(run_id)
        restored_states = self._restore_checkpoint_states(checkpoint)

        return ResumePoint(
            checkpoint=checkpoint,
            token_id=checkpoint.token_id,
            node_id=checkpoint.node_id,
            sequence_number=checkpoint.sequence_number,
            aggregation_state=restored_states.aggregation_state,
            coalesce_state=restored_states.coalesce_state,
        )

    def get_unprocessed_row_data(
        self,
        run_id: str,
        payload_store: PayloadStore,
        *,
        source_schema_class: type[PluginSchema],
    ) -> list[tuple[str, int, dict[str, Any]]]:
        """Get row data for unprocessed rows with type fidelity preservation.

        Retrieves actual row data (not just IDs) for rows that need
        processing during resume. Returns tuples of (row_id, row_index, row_data)
        ordered by row_index for deterministic processing.

        IMPORTANT: Type Fidelity Preservation (REQUIRED)
        -------------------------------------------------
        Payloads are stored via canonical_json(), which normalizes non-JSON types:
        - datetime → ISO string ("2024-01-01T00:00:00+00:00")
        - Decimal → string ("42.50")
        - pandas/numpy scalars → primitives

        On resume, json.loads() returns degraded types (all strings). To restore
        type fidelity, this method REQUIRES source_schema_class to re-validate rows
        through the source's Pydantic schema, which re-coerces strings back to typed values.

        Without schema validation, transforms would receive wrong types (str instead of
        datetime/Decimal), violating the Tier 2 pipeline data trust model from CLAUDE.md.

        Args:
            run_id: The run to get unprocessed rows for
            payload_store: PayloadStore for retrieving row data
            source_schema_class: Pydantic schema class for type restoration (REQUIRED).
                Resume cannot guarantee type fidelity without schema validation.
                The schema must have allow_coercion=True to handle string→typed conversions.

        Returns:
            List of (row_id, row_index, row_data) tuples, ordered by row_index.
            Empty list if run cannot be resumed or all rows were processed.

        Raises:
            AuditIntegrityError: If row not found in database, payload is corrupt,
                or decoded payload is not a dict (Tier 1 violations).
            ValueError: If payload has been purged or schema validation fails
                (operational errors that prevent resume but aren't data corruption)
        """
        row_ids = self.get_unprocessed_rows(run_id)
        if not row_ids:
            return []

        result: list[tuple[str, int, dict[str, Any]]] = []

        # Batch query: Fetch row metadata in chunks to respect SQLite bind limit.
        row_metadata: dict[str, tuple[int, str | None]] = {}
        with self._db.engine.connect() as conn:
            for i in range(0, len(row_ids), _METADATA_CHUNK_SIZE):
                chunk = row_ids[i : i + _METADATA_CHUNK_SIZE]
                rows_result = conn.execute(
                    select(
                        rows_table.c.row_id,
                        rows_table.c.row_index,
                        rows_table.c.source_data_ref,
                    ).where(rows_table.c.row_id.in_(chunk))
                ).fetchall()
                for r in rows_result:
                    row_metadata[r.row_id] = (r.row_index, r.source_data_ref)

        for row_id in row_ids:
            if row_id not in row_metadata:
                raise AuditIntegrityError(f"Row {row_id} not found in database — audit data corruption (Tier 1 violation)")

            row_index, source_data_ref = row_metadata[row_id]

            if source_data_ref is None:
                raise ValueError(
                    f"Row {row_id} has no source_data_ref — row was recorded without "
                    f"payload storage, so recovery cannot reconstruct its data. "
                    f"Re-run the pipeline from scratch instead of resuming."
                )

            # Retrieve from payload store
            try:
                payload_bytes = payload_store.retrieve(source_data_ref)
            except PayloadNotFoundError as exc:
                raise ValueError(f"Row {row_id} payload has been purged (hash={exc.content_hash}) - cannot resume") from exc

            try:
                degraded_data = json.loads(payload_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AuditIntegrityError(
                    f"Corrupt payload for row {row_id} (ref={source_data_ref}) — "
                    f"cannot decode persisted row data (Tier 1 violation). "
                    f"Error: {exc}"
                ) from exc

            if not isinstance(degraded_data, dict):
                raise AuditIntegrityError(
                    f"Corrupt payload for row {row_id} (ref={source_data_ref}) — "
                    f"expected dict, got {type(degraded_data).__name__} (Tier 1 violation)"
                )

            # TYPE FIDELITY RESTORATION:
            # Re-validate through source schema to restore types.
            # This is critical for datetime, Decimal, and other coerced types
            # that canonical_json normalizes to strings.
            # Schema is now REQUIRED - no fallback to degraded types.
            validated = source_schema_class.model_validate(degraded_data)
            row_data = validated.to_row()

            # DEFENSE-IN-DEPTH: Detect silent data loss from empty schemas
            # If source data has fields but restored data is empty, the schema is losing data.
            # This catches bugs like NullSourceSchema (no fields) being used for resume.
            if degraded_data and not row_data:
                raise ValueError(
                    f"Resume failed for row {row_id}: Schema validation returned empty data "
                    f"but source had {len(degraded_data)} fields. "
                    f"Schema class '{source_schema_class.__name__}' appears to have no fields defined. "
                    f"Cannot resume - this would silently discard all row data. "
                    f"The source plugin's schema must declare fields matching the stored row structure."
                )

            result.append((row_id, row_index, row_data))

        return result

    def get_unprocessed_rows(self, run_id: str) -> list[str]:
        """Get row IDs that were not processed before the run failed.

        Uses token outcomes to determine which rows need processing:
        - Rows with non-delegation terminal outcomes are done
        - Rows whose tokens lack terminal outcomes need reprocessing
        - Rows already buffered in checkpoint aggregation state are excluded
          (they will be restored from checkpoint, not reprocessed)

        This correctly handles multi-sink scenarios where rows are routed to
        different sinks in interleaved order. The previous row_index boundary
        approach would skip rows routed to a failed sink if a later row
        succeeded on a different sink.

        Args:
            run_id: The run to get unprocessed rows for

        Returns:
            List of row_id strings for rows that need processing.
            Empty list if run cannot be resumed or all rows were processed.
        """
        checkpoint = self._checkpoint_manager.get_latest_checkpoint(run_id)
        if checkpoint is None:
            return []

        # Extract buffered token IDs from checkpoint aggregation state.
        # These buffered tokens will be restored from checkpoint state and must not
        # trigger duplicate reprocessing, but row-level exclusion is unsafe when a row
        # has mixed buffered and non-buffered incomplete tokens.
        buffered_token_ids = self._get_buffered_checkpoint_token_ids(checkpoint)

        with self._db.engine.connect() as conn:
            # CORRECT SEMANTICS FOR FORK/AGGREGATION/COALESCE RECOVERY:
            #
            # A row is "complete" when ALL its "leaf" tokens have terminal outcomes.
            # "Leaf" tokens = tokens that are NOT delegation markers.
            #
            # Delegation markers (excluded from completion check):
            # - FORK_PARENT: Fork parent, children carry completion status
            # - EXPAND_PARENT: Deaggregation parent, expanded children carry status
            #
            # Terminal outcomes (indicate row processing is done):
            # - completed=1 marks rows with an outcome decision.
            # - FORK_PARENT/EXPAND_PARENT paths are excluded because those
            #   parent outcomes delegate completion to child tokens.
            #
            # A row is "incomplete" (needs reprocessing) if ANY of:
            # 1. No tokens at all (never started processing)
            # 2. Any non-delegation token lacks terminal outcome
            # 3. Has tokens but NONE have terminal outcomes (delegation marker only)
            #
            # BUG FIX (P2-recovery-skips-forked-rows):
            # Previous approach: "row has ANY terminal token → complete"
            # Failed: If child A completed but child B crashed, row marked done.
            # Fix: "ALL non-delegation tokens must have terminal outcomes"

            # Subquery: Tokens that are delegation markers (FORK_PARENT or EXPAND_PARENT)
            # These delegate completion to their children, so exclude from completion check
            delegation_tokens = (
                select(token_outcomes_table.c.token_id)
                .where(token_outcomes_table.c.run_id == run_id)
                .where(token_outcomes_table.c.path.in_(_DELEGATION_PATHS))
            ).scalar_subquery()

            # Subquery: Tokens with terminal outcomes
            terminal_tokens = (
                select(token_outcomes_table.c.token_id)
                .where(token_outcomes_table.c.run_id == run_id)
                .where(token_outcomes_table.c.completed == 1)
                .where(~token_outcomes_table.c.path.in_(_DELEGATION_PATHS))
            ).scalar_subquery()

            # Subquery: Rows that have at least one terminal outcome
            rows_with_terminal = (
                select(tokens_table.c.row_id)
                .distinct()
                .select_from(
                    tokens_table.join(
                        token_outcomes_table,
                        tokens_table.c.token_id == token_outcomes_table.c.token_id,
                    )
                )
                .where(token_outcomes_table.c.run_id == run_id)
                .where(token_outcomes_table.c.completed == 1)
                .where(~token_outcomes_table.c.path.in_(_DELEGATION_PATHS))
            ).scalar_subquery()

            # Main query: Find incomplete rows
            # Row is incomplete if:
            # - Case 1: No tokens at all
            # - Case 2: Has non-delegation token without terminal outcome
            # - Case 3: Has tokens but none have terminal outcomes (delegation only)
            #
            # NOTE: PostgreSQL requires ORDER BY columns to be in SELECT when using DISTINCT.
            # We select both row_id and row_index, then extract just row_id from results.
            query = (
                select(rows_table.c.row_id, rows_table.c.row_index)
                .select_from(rows_table)
                .outerjoin(
                    tokens_table,
                    rows_table.c.row_id == tokens_table.c.row_id,
                )
                .where(rows_table.c.run_id == run_id)
                .where(
                    # Case 1: No tokens at all
                    (tokens_table.c.token_id.is_(None))
                    |
                    # Case 2: Non-delegation token without terminal outcome
                    ((~tokens_table.c.token_id.in_(delegation_tokens)) & (~tokens_table.c.token_id.in_(terminal_tokens)))
                    |
                    # Case 3: Has tokens but no terminal outcomes (fork parent only)
                    (~rows_table.c.row_id.in_(rows_with_terminal))
                )
                .order_by(rows_table.c.row_index)
                .distinct()
            )

            unprocessed = [row.row_id for row in conn.execute(query).fetchall()]

        # Exclude rows only when ALL their incomplete leaf tokens are buffered.
        # This avoids silently dropping rows with mixed-state tokens where one
        # token is buffered and another incomplete token still needs processing.
        if buffered_token_ids and unprocessed:
            incomplete_tokens: list[Row[Any]] = []
            with self._db.engine.connect() as conn:
                for i in range(0, len(unprocessed), _METADATA_CHUNK_SIZE):
                    chunk = unprocessed[i : i + _METADATA_CHUNK_SIZE]
                    incomplete_tokens.extend(
                        conn.execute(
                            select(tokens_table.c.row_id, tokens_table.c.token_id)
                            .where(tokens_table.c.row_id.in_(chunk))
                            .where(~tokens_table.c.token_id.in_(delegation_tokens))
                            .where(~tokens_table.c.token_id.in_(terminal_tokens))
                        ).fetchall()
                    )

            row_to_incomplete_tokens: dict[str, set[str]] = {row_id: set() for row_id in unprocessed}
            for row_id, token_id in incomplete_tokens:
                row_to_incomplete_tokens[row_id].add(token_id)

            filtered_rows: list[str] = []
            for row_id in unprocessed:
                row_incomplete = row_to_incomplete_tokens[row_id]
                if row_incomplete and row_incomplete.issubset(buffered_token_ids):
                    continue
                filtered_rows.append(row_id)
            unprocessed = filtered_rows

        return unprocessed

    def get_incomplete_tokens_by_row(self, run_id: str) -> dict[str, list[IncompleteTokenSpec]]:
        """Return incomplete non-delegation child tokens, grouped by row_id.

        A token is incomplete when it is NOT a delegation marker (FORK_PARENT /
        EXPAND_PARENT) and has NO completed terminal outcome. Mirrors get_unprocessed_rows'
        completion semantics (shared _DELEGATION_PATHS) so recovery selection and resume
        reconstruction cannot drift apart. Returns fork, expand, AND post-coalesce tokens —
        each is dispatched in resume_incomplete_token.

        BUFFERED-TOKEN EXCLUSION (mirrors get_unprocessed_rows' buffered exclusion):
        tokens that are buffered in restored aggregation state OR held at a restored
        coalesce barrier (collected by _get_buffered_checkpoint_token_ids — see that method
        for the two arms) are EXCLUDED at the token level here. Such tokens are NOT re-driven
        on resume: they are restored into the executor's in-memory state (aggregation buffer /
        coalesce _pending) by restore_from_checkpoint and flushed/merged FROM that state.
        Re-driving them double-emits or crashes the barrier:
        - A coalesce-held branch re-driven re-arrives at the barrier where it already waits
          (restored into _pending) → CoalesceExecutor.accept's duplicate-arrival guard fires
          (OrchestrationInvariantError).
        - An aggregation-buffered branch re-driven re-enters processing while it is ALSO
          flushed from the restored buffer at end-of-source → double terminal outcome /
          duplicate physical sink write.

        Exclusion is at the TOKEN level (filter before grouping), so a row with mixed state
        (one buffered token + one genuinely-incomplete sibling) still returns the
        genuinely-incomplete token, while a row whose only incomplete tokens are all buffered
        does not appear in the returned dict at all. With the same buffered exclusion applied,
        get_incomplete_tokens_by_row's rows are a subset of get_unprocessed_rows ON THE RESUME
        PATH — i.e. when a checkpoint exists. This holds for the only caller (resume, gated by
        can_resume, which requires a checkpoint).

        The subset relation does NOT hold unconditionally: unlike get_unprocessed_rows, this
        method has no `checkpoint is None -> return []` early-return. With no checkpoint it
        applies an empty buffered set (a no-op exclusion) and still returns every incomplete
        token, whereas get_unprocessed_rows returns []. In that (non-resume) case this method's
        rows are a SUPERSET, not a subset. Safe because the run is unresumable then anyway.
        """
        checkpoint = self._checkpoint_manager.get_latest_checkpoint(run_id)
        buffered_token_ids = self._get_buffered_checkpoint_token_ids(checkpoint) if checkpoint is not None else set()

        with self._db.engine.connect() as conn:
            delegation_tokens = (
                select(token_outcomes_table.c.token_id)
                .where(token_outcomes_table.c.run_id == run_id)
                .where(token_outcomes_table.c.path.in_(_DELEGATION_PATHS))
            ).scalar_subquery()
            terminal_tokens = (
                select(token_outcomes_table.c.token_id)
                .where(token_outcomes_table.c.run_id == run_id)
                .where(token_outcomes_table.c.completed == 1)
                .where(~token_outcomes_table.c.path.in_(_DELEGATION_PATHS))
            ).scalar_subquery()
            max_attempt_sq = (
                select(func.max(node_states_table.c.attempt))
                .where(node_states_table.c.token_id == tokens_table.c.token_id)
                .where(node_states_table.c.run_id == run_id)
                .correlate(tokens_table)
                .scalar_subquery()
            )
            query = (
                select(
                    tokens_table.c.token_id,
                    tokens_table.c.row_id,
                    tokens_table.c.branch_name,
                    tokens_table.c.fork_group_id,
                    tokens_table.c.join_group_id,
                    tokens_table.c.expand_group_id,
                    tokens_table.c.token_data_ref,
                    tokens_table.c.step_in_pipeline,
                    max_attempt_sq.label("max_attempt"),
                )
                .where(tokens_table.c.run_id == run_id)
                .where(~tokens_table.c.token_id.in_(delegation_tokens))
                .where(~tokens_table.c.token_id.in_(terminal_tokens))
                .order_by(tokens_table.c.step_in_pipeline, tokens_table.c.token_id)
            )
            rows = conn.execute(query).fetchall()

        by_row: dict[str, list[IncompleteTokenSpec]] = {}
        for r in rows:
            # Buffered/held tokens are restored-and-flushed from checkpoint state, not
            # re-driven (see docstring). Exclude at the token level so mixed-state rows
            # still return their genuinely-incomplete tokens.
            if r.token_id in buffered_token_ids:
                continue
            by_row.setdefault(r.row_id, []).append(
                IncompleteTokenSpec(
                    token_id=r.token_id,
                    row_id=r.row_id,
                    branch_name=r.branch_name,
                    fork_group_id=r.fork_group_id,
                    join_group_id=r.join_group_id,
                    expand_group_id=r.expand_group_id,
                    token_data_ref=r.token_data_ref,
                    step_in_pipeline=r.step_in_pipeline,
                    max_attempt=-1 if r.max_attempt is None else int(r.max_attempt),
                )
            )
        return by_row

    def reconstruct_token_row(
        self,
        spec: IncompleteTokenSpec,
        run_id: str,
        source_row: PipelineRow,
        payload_store: PayloadStore,
    ) -> PipelineRow:
        """Build the PipelineRow to re-drive an incomplete token with.

        Fork children share the source payload → return source_row unchanged. Expand
        children / post-coalesce tokens carry a self-contained {data, contract} envelope
        in token_data_ref → restore both type-faithfully (checkpoint_loads +
        SchemaContract.from_checkpoint, which hash-validates the contract). No nodes-table
        lookup: the contract the token was produced under is persisted with its payload
        (ADDENDUM 3 — nodes.output_contract_json is NULL for non-source nodes in prod,
        making the nodes-table approach unsalvageable).

        Args:
            spec: IncompleteTokenSpec from get_incomplete_tokens_by_row.
            run_id: The run being resumed (for diagnostic messages).
            source_row: The source PipelineRow for this row_id (used for fork children
                that share the parent/source payload).
            payload_store: PayloadStore to retrieve token_data_ref bytes from.

        Returns:
            PipelineRow to re-drive the incomplete token with.

        Raises:
            ValueError: If token_data_ref is set but the payload has been purged.
            AuditIntegrityError: If the retrieved payload is not a valid {data, contract}
                envelope (Tier-1 corruption guard), or if the contract hash does not match
                (via SchemaContract.from_checkpoint — Tier-1 integrity).
        """
        if spec.token_data_ref is None:
            return source_row  # fork child — shares the source payload

        try:
            payload_bytes = payload_store.retrieve(spec.token_data_ref)
        except PayloadNotFoundError as exc:
            raise ValueError(
                f"Incomplete token {spec.token_id} (run {run_id}) payload purged "
                f"(token_data_ref={spec.token_data_ref!r}) — cannot resume; re-run instead."
            ) from exc

        envelope = checkpoint_loads(payload_bytes.decode("utf-8"))
        if not isinstance(envelope, dict) or "data" not in envelope or "contract" not in envelope:
            raise AuditIntegrityError(
                f"token_data_ref payload for token {spec.token_id} (run {run_id}) is not a "
                f"valid {{data, contract}} envelope — audit data corruption (Tier-1 violation). "
                f"Got type={type(envelope).__name__!r}."
            )

        # SchemaContract.from_checkpoint validates the version_hash (Tier-1 integrity).
        # It raises AuditIntegrityError if the hash does not match — crash is correct,
        # the contract stored in the audit trail is our data (Tier-1) and must be pristine.
        contract = SchemaContract.from_checkpoint(envelope["contract"])
        return PipelineRow(envelope["data"], contract)

    def _get_buffered_checkpoint_token_ids(self, checkpoint: Checkpoint) -> set[str]:
        """Collect token IDs restored from checkpoint state."""
        buffered_token_ids: set[str] = set()

        restored_states = self._restore_checkpoint_states(checkpoint)
        if restored_states.aggregation_state is not None:
            for node_checkpoint in restored_states.aggregation_state.nodes.values():
                for token in node_checkpoint.tokens:
                    buffered_token_ids.add(token.token_id)

        if restored_states.coalesce_state is not None:
            for pending in restored_states.coalesce_state.pending:
                for coalesce_token in pending.branches.values():
                    buffered_token_ids.add(coalesce_token.token_id)

        return buffered_token_ids

    def _restore_checkpoint_states(self, checkpoint: Checkpoint) -> _RestoredCheckpointStates:
        """Deserialize typed checkpoint states once per observed checkpoint payload."""
        key = (
            checkpoint.checkpoint_id,
            checkpoint.aggregation_state_json,
            checkpoint.coalesce_state_json,
        )
        if key in self._checkpoint_state_cache:
            cached = self._checkpoint_state_cache[key]
            return cached

        agg_state = None
        if checkpoint.aggregation_state_json:
            # Use checkpoint_loads for type restoration (datetime -> datetime, not string)
            raw = checkpoint_loads(checkpoint.aggregation_state_json)
            agg_state = AggregationCheckpointState.from_dict(raw)

        coalesce_state = None
        if checkpoint.coalesce_state_json:
            raw = checkpoint_loads(checkpoint.coalesce_state_json)
            coalesce_state = CoalesceCheckpointState.from_dict(raw)

        restored = _RestoredCheckpointStates(
            aggregation_state=agg_state,
            coalesce_state=coalesce_state,
        )
        if len(self._checkpoint_state_cache) >= _CHECKPOINT_STATE_CACHE_MAX:
            oldest_key = next(iter(self._checkpoint_state_cache))
            del self._checkpoint_state_cache[oldest_key]
        self._checkpoint_state_cache[key] = restored
        return restored

    def _get_run(self, run_id: str) -> Row[Any] | None:
        """Get run metadata from the database.

        Args:
            run_id: The run to fetch

        Returns:
            Row result with run data, or None if not found
        """
        with self._db.engine.connect() as conn:
            result = conn.execute(select(runs_table).where(runs_table.c.run_id == run_id)).fetchone()

        return result

    def verify_contract_integrity(self, run_id: str) -> SchemaContract:
        """Verify schema contract integrity for a run.

        Retrieves the stored schema contract and verifies its integrity
        via hash comparison. This is a Tier 1 check - missing or corrupt
        contracts indicate audit trail tampering or database corruption.

        Args:
            run_id: Run to verify

        Returns:
            SchemaContract - always returns a valid contract

        Raises:
            CheckpointCorruptionError: If contract is missing OR hash mismatch detected.
                Per CLAUDE.md Tier-1 trust model: "Bad data in the audit trail = crash immediately"
                Missing contract is treated as corruption - NO backward compatibility.
        """
        factory = RecorderFactory(self._db)

        try:
            contract = factory.run_lifecycle.get_run_contract(run_id)
        except AuditIntegrityError as e:
            # get_run_contract raises AuditIntegrityError for hash verification failures
            # and run-not-found. Convert to CheckpointCorruptionError for checkpoint-specific context.
            raise CheckpointCorruptionError(
                f"Contract integrity verification failed for run '{run_id}': {e}. "
                f"Resume aborted - audit trail may be corrupted or tampered with."
            ) from e
        except (ValueError, KeyError) as e:
            # get_run_contract deserializes via ContractAuditRecord.from_json, which raises
            # json.JSONDecodeError (subclass of ValueError) for malformed JSON, or KeyError
            # for missing required fields. Both indicate Tier 1 data corruption — stored
            # contract JSON is garbage.
            raise CheckpointCorruptionError(
                f"Contract integrity verification failed for run '{run_id}': {e}. "
                f"Resume aborted - stored contract JSON is malformed (database corruption)."
            ) from e

        if contract is None:
            # TIER-1 AUDIT INTEGRITY: Missing contract = audit trail corruption
            # Per CLAUDE.md: "Bad data in the audit trail = crash immediately"
            # Per NO LEGACY CODE POLICY: No backward compatibility for pre-contract runs
            raise CheckpointCorruptionError(
                f"Schema contract is missing from audit trail for run '{run_id}'. "
                f"This indicates either:\n"
                f"  1. The audit database is corrupt or incomplete\n"
                f"  2. The run was started with a version that didn't record contracts\n"
                f"Resume cannot proceed safely without the schema contract. "
                f"The audit trail must be complete and trustworthy."
            )

        return contract
