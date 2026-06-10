"""Recovery protocol for resuming failed runs.

Provides the API for determining if and how a failed run can be resumed:
- can_resume(run_id) - Check if run can be resumed (failed status + checkpoint exists)
- get_resume_point(run_id) - Get checkpoint info for resuming

The actual resume logic (Orchestrator.resume()) is implemented separately.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
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
    ResumedRow,
    ResumePoint,
    RunStatus,
    SchemaContract,
    TerminalPath,
)
from elspeth.contracts.aggregation_checkpoint import AggregationCheckpointState
from elspeth.contracts.coalesce_checkpoint import CoalesceCheckpointState
from elspeth.contracts.errors import AuditIntegrityError, EmptyResumeStateError
from elspeth.contracts.types import NodeID
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
    "NonResumableRunError",
    "RecoveryManager",
    "ResumeCheck",  # Re-exported from contracts for convenience
    "ResumePoint",  # Re-exported from contracts for convenience
    "check_run_status_resumable",
]


# TIER-2: Operator-interpretable refuse signal — the audit DB is intact and
# truthful; the run's CURRENT status (e.g. RUNNING: another worker holds the
# run mid-flight) means resuming now would be incorrect. Same register as
# elspeth.contracts.errors.IncompleteSourceResumeError: an operator-facing
# precondition failure carrying run_id + reason, NOT audit corruption
# (contrast the run-immutability guard's AuditIntegrityError in
# RunLifecycleRepository.update_run_status).
class NonResumableRunError(Exception):
    """Raised by ``ResumeCoordinator.resume()`` when run status precludes resume.

    ``RecoveryManager.can_resume()`` is ADVISORY — callers may skip it — so
    ``resume()`` re-checks the run status at entry via the same shared
    implementation (:func:`check_run_status_resumable`) and raises this
    error before any mutation (elspeth-2f23292372, operator option b).

    Carries ``run_id`` and the human-readable ``reason`` from the shared
    check so CLI/API callers can surface a precise "not resumable" outcome
    without parsing the exception text.
    """

    def __init__(self, run_id: str, reason: str) -> None:
        self.run_id = run_id
        self.reason = reason
        super().__init__(f"Cannot resume run {run_id!r}: {reason}")


def _fetch_run(db: LandscapeDB, run_id: str) -> Row[Any] | None:
    """Fetch the ``runs`` row for ``run_id``, or None if absent."""
    with db.engine.connect() as conn:
        return conn.execute(select(runs_table).where(runs_table.c.run_id == run_id)).fetchone()


def check_run_status_resumable(db: LandscapeDB, run_id: str) -> tuple[RunStatus | None, ResumeCheck]:
    """Existence + run-status portion of :meth:`RecoveryManager.can_resume`.

    SINGLE shared implementation for the advisory ``can_resume`` surface and
    the enforcing entry guard in ``ResumeCoordinator.resume()`` — the two
    must never drift (elspeth-2f23292372).

    Returns:
        ``(run_status, check)``: ``run_status`` is ``None`` when the run does
        not exist. ``check`` carries ``can_resume=True`` when the status alone
        does not preclude resume; checkpoint existence, topology, and contract
        integrity remain ``can_resume``'s remit, not this function's.

    Raises:
        CheckpointCorruptionError: If the persisted status is not a valid
            ``RunStatus`` — audit corruption, never a clean refuse.
    """
    run = _fetch_run(db, run_id)
    if run is None:
        return None, ResumeCheck(can_resume=False, reason=f"Run {run_id} not found")

    try:
        run_status = RunStatus(run.status)
    except ValueError as exc:
        raise CheckpointCorruptionError(f"Run {run_id} has invalid status {run.status!r}; audit trail is corrupt") from exc

    if run_status == RunStatus.COMPLETED:
        return run_status, ResumeCheck(can_resume=False, reason="Run already completed successfully")

    if run_status == RunStatus.RUNNING:
        return run_status, ResumeCheck(can_resume=False, reason="Run is still in progress")

    if run_status not in _RESUMABLE_RUN_STATUSES:
        return run_status, ResumeCheck(can_resume=False, reason=f"Run status {run_status.value!r} is not resumable")

    return run_status, ResumeCheck(can_resume=True)


@dataclass(frozen=True, slots=True)
class IncompleteTokenSpec:
    """A non-delegation child token that lacks a terminal outcome on a resumed run.

    Identity fields read directly from persisted columns (Tier-1: no defaults,
    no coercion). ``token_data_ref`` is NULL for fork children and set for
    expand children and post-coalesce merged tokens.
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
        """Validate Tier-1 identity invariants at construction time."""
        for field_name in ("token_id", "row_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"IncompleteTokenSpec.{field_name} must be str, got {type(value).__name__}: {value!r}")
            if not value:
                raise ValueError(f"IncompleteTokenSpec.{field_name} must not be empty")
        for field_name in ("branch_name", "fork_group_id", "join_group_id", "expand_group_id", "token_data_ref"):
            value = getattr(self, field_name)
            if value is not None:
                if not isinstance(value, str):
                    raise TypeError(f"IncompleteTokenSpec.{field_name} must be str or None, got {type(value).__name__}: {value!r}")
                if not value:
                    raise ValueError(f"IncompleteTokenSpec.{field_name} must be None or non-empty string, got {value!r}")


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
        # Existence + status checks live in check_run_status_resumable so
        # this advisory surface and the enforcing entry guard in
        # ResumeCoordinator.resume() share ONE implementation.
        _run_status, status_check = check_run_status_resumable(self._db, run_id)
        if not status_check.can_resume:
            return status_check

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
    ) -> list[ResumedRow]:
        """Get row data for unprocessed rows with type fidelity preservation.

        Retrieves actual row data (not just IDs) for rows that need
        processing during resume. Returns ``ResumedRow`` instances
        ordered by row_index for deterministic processing.

        Used on the pre-RC6 single-source resume path where ``run_sources``
        is empty; every persisted row still carries its originating
        ``source_node_id`` (NOT NULL per schema), which is preserved on
        the ResumedRow so downstream consumers can look up the row's
        schema contract by source node identity (ADR-025 §3).

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
            List of ResumedRow records, ordered by row_index.
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

        result: list[ResumedRow] = []

        # Batch query: Fetch row metadata in chunks to respect SQLite bind limit.
        # ADR-025 §4: source_node_id is now load-bearing on every row, not
        # just multi-source pipelines. Pre-RC6 audit DBs without
        # ``run_sources`` records still carry source_node_id on rows
        # (NOT NULL per schema) and resume must propagate it so downstream
        # consumers look up the per-row schema contract by source identity.
        row_metadata: dict[str, tuple[int, NodeID, str | None]] = {}
        with self._db.engine.connect() as conn:
            for i in range(0, len(row_ids), _METADATA_CHUNK_SIZE):
                chunk = row_ids[i : i + _METADATA_CHUNK_SIZE]
                rows_result = conn.execute(
                    select(
                        rows_table.c.row_id,
                        rows_table.c.row_index,
                        rows_table.c.source_node_id,
                        rows_table.c.source_data_ref,
                    ).where(rows_table.c.row_id.in_(chunk))
                ).fetchall()
                for r in rows_result:
                    row_metadata[r.row_id] = (r.row_index, NodeID(r.source_node_id), r.source_data_ref)

        for row_id in row_ids:
            if row_id not in row_metadata:
                raise AuditIntegrityError(f"Row {row_id} not found in database — audit data corruption (Tier 1 violation)")

            row_index, source_node_id, source_data_ref = row_metadata[row_id]

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

            if type(degraded_data) is not dict:
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

            result.append(
                ResumedRow(
                    row_id=row_id,
                    row_index=row_index,
                    source_node_id=source_node_id,
                    row_data=row_data,
                )
            )

        return result

    def get_unprocessed_row_data_by_source(
        self,
        run_id: str,
        payload_store: PayloadStore,
        *,
        source_schema_classes: Mapping[NodeID, type[PluginSchema]],
    ) -> list[ResumedRow]:
        """Get unprocessed row data with source-scoped type restoration.

        Multi-source resume cannot validate every persisted payload through a
        single source schema. Rows carry ``source_node_id`` in Landscape, and
        this method uses that node identity to select the schema class that
        originally ingested the row. Returns ``ResumedRow`` instances
        ordered by ``ingest_sequence`` (ADR-025 §4).
        """
        row_ids = self.get_unprocessed_rows(run_id)
        if not row_ids:
            return []

        row_metadata: dict[str, tuple[int, int, NodeID, str | None]] = {}
        with self._db.engine.connect() as conn:
            for i in range(0, len(row_ids), _METADATA_CHUNK_SIZE):
                chunk = row_ids[i : i + _METADATA_CHUNK_SIZE]
                rows_result = conn.execute(
                    select(
                        rows_table.c.row_id,
                        rows_table.c.row_index,
                        rows_table.c.ingest_sequence,
                        rows_table.c.source_node_id,
                        rows_table.c.source_data_ref,
                    ).where(rows_table.c.row_id.in_(chunk))
                ).fetchall()
                for r in rows_result:
                    row_metadata[r.row_id] = (r.row_index, r.ingest_sequence, NodeID(r.source_node_id), r.source_data_ref)

        # Per Three-Tier Trust Model: the audit DB is Tier 1; ``row_ids`` comes
        # from ``get_unprocessed_rows()`` and ``row_metadata`` is built from the
        # same DB in the lookup above. A ``row_id`` missing from ``row_metadata``
        # is internal audit corruption. Let the KeyError raise from the sort
        # key — no defensive ``else -1`` arm that would silently mis-order
        # corrupt data before any explicit check could fire.
        ordered_row_ids = sorted(
            row_ids,
            key=lambda row_id: row_metadata[row_id][1],
        )
        result: list[ResumedRow] = []
        for row_id in ordered_row_ids:
            row_index, _ingest_sequence, source_node_id, source_data_ref = row_metadata[row_id]
            if source_node_id not in source_schema_classes:
                raise AuditIntegrityError(
                    f"Row {row_id} references source_node_id={source_node_id!r}, but resume has no schema class for that source. "
                    "Per-source run_sources metadata is incomplete or corrupt."
                )

            if source_data_ref is None:
                raise ValueError(
                    f"Row {row_id} has no source_data_ref — row was recorded without "
                    f"payload storage, so recovery cannot reconstruct its data. "
                    f"Re-run the pipeline from scratch instead of resuming."
                )

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

            if type(degraded_data) is not dict:
                raise AuditIntegrityError(
                    f"Corrupt payload for row {row_id} (ref={source_data_ref}) — "
                    f"expected dict, got {type(degraded_data).__name__} (Tier 1 violation)"
                )

            source_schema_class = source_schema_classes[source_node_id]
            validated = source_schema_class.model_validate(degraded_data)
            row_data = validated.to_row()
            if degraded_data and not row_data:
                raise ValueError(
                    f"Resume failed for row {row_id}: Schema validation returned empty data "
                    f"but source had {len(degraded_data)} fields. "
                    f"Schema class '{source_schema_class.__name__}' appears to have no fields defined. "
                    f"Cannot resume - this would silently discard all row data. "
                    f"The source plugin's schema must declare fields matching the stored row structure."
                )

            result.append(
                ResumedRow(
                    row_id=row_id,
                    row_index=row_index,
                    source_node_id=source_node_id,
                    row_data=row_data,
                )
            )

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
            # We select both row_id and ingest_sequence, then extract just row_id from results.
            query = (
                select(rows_table.c.row_id, rows_table.c.ingest_sequence)
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
                .order_by(rows_table.c.ingest_sequence)
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

        A token is incomplete when it is not a delegation marker and has no
        completed terminal outcome. Tokens restored from checkpoint aggregation
        or coalesce state are excluded because those are flushed from restored
        executor state rather than re-driven from source.
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
        for row in rows:
            if row.token_id in buffered_token_ids:
                continue
            by_row.setdefault(row.row_id, []).append(
                IncompleteTokenSpec(
                    token_id=row.token_id,
                    row_id=row.row_id,
                    branch_name=row.branch_name,
                    fork_group_id=row.fork_group_id,
                    join_group_id=row.join_group_id,
                    expand_group_id=row.expand_group_id,
                    token_data_ref=row.token_data_ref,
                    step_in_pipeline=row.step_in_pipeline,
                    max_attempt=-1 if row.max_attempt is None else int(row.max_attempt),
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
        """Build the PipelineRow to re-drive an incomplete token with."""
        if spec.token_data_ref is None:
            return source_row

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
        return _fetch_run(self._db, run_id)

    def verify_contract_integrity(self, run_id: str) -> SchemaContract:
        """Verify schema contract integrity for a run.

        Per ADR-025 §3 Decision 5, ``run_sources.schema_contract_json`` is the
        single authoritative writer/reader for per-source schema contracts;
        ``runs.schema_contract_json`` is no longer consulted. Every declared
        source in a run must have a recorded contract before any row from
        that source enters the pipeline (Fix 2 in the multi-source-token-
        scheduler change set), so missing rows here mean the audit trail
        was never populated — Tier-1 corruption.

        Verifies hash integrity on every source's contract. Returns the
        contract of the lowest-ordered ``source_node_id`` (deterministic)
        for the legacy single-source consumer surface — multi-source
        callers must reach into ``RunLifecycleRepository.get_run_source_resume_records``
        for per-source contracts.

        Args:
            run_id: Run to verify

        Returns:
            SchemaContract - the first source's contract, ordered by ``source_node_id``.

        Raises:
            CheckpointCorruptionError: If no ``run_sources`` rows exist, if any
                stored contract is missing, malformed, or has mismatched hash,
                or if the run itself doesn't exist.
                Per CLAUDE.md Tier-1 trust model: "Bad data in the audit trail = crash immediately"
        """
        factory = RecorderFactory(self._db)

        # Verify the run exists (Tier-1: missing run = corruption surfaced to caller).
        if factory.run_lifecycle.get_run(run_id) is None:
            raise CheckpointCorruptionError(f"Run '{run_id}' not found in audit trail. Resume cannot proceed against an unrecorded run.")

        try:
            source_records = factory.run_lifecycle.get_run_source_resume_records(run_id)
        except AuditIntegrityError as e:
            # get_run_source_resume_records raises AuditIntegrityError on every per-source
            # corruption mode: missing schema JSON, missing contract JSON, missing or
            # mismatched contract hash. Convert to CheckpointCorruptionError so the
            # checkpoint-resume call surface stays a single exception type.
            raise CheckpointCorruptionError(
                f"Contract integrity verification failed for run '{run_id}': {e}. "
                f"Resume aborted - per-source contract metadata is corrupt or missing."
            ) from e
        except (ValueError, KeyError) as e:
            # ContractAuditRecord.from_json() raises json.JSONDecodeError (subclass of
            # ValueError) for malformed JSON, or KeyError for missing required fields.
            # Both indicate Tier-1 data corruption — stored contract JSON is garbage.
            raise CheckpointCorruptionError(
                f"Contract integrity verification failed for run '{run_id}': {e}. "
                f"Resume aborted - stored per-source contract JSON is malformed (database corruption)."
            ) from e

        if not source_records:
            # ADR-025 §3: a run with no ``run_sources`` rows reflects the
            # "nothing to resume, start fresh" outcome — typically an
            # ``on_start`` failure, a source-level abort before the first
            # ingest, or an infrastructure crash before any row was
            # persisted. The audit DB is intact and truthfully records
            # that the run did no work; it is NOT Tier-1 corruption.
            #
            # Raising ``EmptyResumeStateError`` here lets the CLI present
            # a clean "this run is not resumable; start a fresh run"
            # message rather than the audit-corruption traceback the
            # legacy ``CheckpointCorruptionError`` produced — that was
            # the reachability gap reported by elspeth-241608388f, where
            # the CLI's outer ``try`` lacked any handler for the
            # corruption-typed bubble and the operator saw a misleading
            # invariant traceback for a benign outcome.
            #
            # ``EmptyResumeStateError`` is a subclass of
            # ``OrchestrationInvariantError`` so any existing
            # ``except OrchestrationInvariantError`` catch still
            # matches it by type; the CLI catches the typed exception
            # explicitly before the broader Tier-1 handler.
            raise EmptyResumeStateError(run_id=run_id)

        # Deterministic single-source return for the legacy caller surface.
        # Multi-source callers should reach into ``get_run_source_resume_records``
        # for per-source contracts; this method only confirms integrity and
        # exposes the canonical first-source view for the unit-test contract.
        first_source_node_id = sorted(source_records)[0]
        return source_records[first_source_node_id].schema_contract
